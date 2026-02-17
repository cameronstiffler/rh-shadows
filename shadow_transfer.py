"""Shadow transfer pipeline using Gemini/Vertex.

Scans recipient images, finds matching donors by product code and view code,
requests Gemini to apply donor shadows to the recipient, and writes outputs
into a per-product generated folder. Only processes recipients that do not yet
have a corresponding output file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import base64
from io import BytesIO
import tempfile
import time
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image, ImageChops, ImageOps, ImageFilter, ImageEnhance
import numpy as np

# Allow large source images; default behavior is to send 4K long-edge images to the API.
Image.MAX_IMAGE_PIXELS = None
SCRIPT_DIR = Path(__file__).resolve().parent
FOUR_K_LONG_EDGE = 4096
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PROMPT_ID = 1
DEFAULT_PRODUCTS_DIR = SCRIPT_DIR / "products"
IMAGE_TIER_PREFERENCE = ("high_res", "low_res")
LOW_RES_DIRNAME = "low_res"
HIGH_RES_DIRNAME = "high_res"
DEBUG_DIR = SCRIPT_DIR / "debug"
DEBUG_LONG_EDGE = 2048

# Donor contrast boost before submission (1.0 = no change).
DONOR_CONTRAST_BOOST = float(os.getenv("DONOR_CONTRAST_BOOST", "1.0"))

ImageKey = Tuple[str, str]
DEFAULT_SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_HARASSMENT",
    ]
]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".webp",
    }


def extract_product_code(path: Path) -> Optional[str]:
    match = re.search(r"(E\d+)", path.stem)
    return match.group(1) if match else None


def extract_view_code(path: Path) -> Optional[str]:
    parts = path.stem.split("_")
    if not parts:
        return None
    if parts[-1].lower() == "cc" and len(parts) >= 2:
        parts = parts[:-1]
    return parts[-1] if parts else None

def normalize_inline_data(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise TypeError(f"Unsupported inline data type: {type(data)}")


def extract_image_from_response(response) -> bytes:
    # Prefer the newer SDK shape (response.parts), then fall back.
    parts = getattr(response, "parts", None)
    if parts:
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return normalize_inline_data(inline.data)
    # Fallback to candidates->content->parts
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return normalize_inline_data(inline.data)
    raise RuntimeError("No image returned by Gemini response.")


def enforce_output_dimensions(image_bytes: bytes, recipient_size: Tuple[int, int]) -> bytes:
    """Resize an image to match the recipient dimensions."""
    recipient_w, recipient_h = recipient_size
    with Image.open(BytesIO(image_bytes)) as img:
        if img.size == (recipient_w, recipient_h):
            return image_bytes
        resized = img.resize((recipient_w, recipient_h), resample=Image.LANCZOS).convert("RGBA")
        buffer = BytesIO()
        resized.save(buffer, format="PNG")
        return buffer.getvalue()


def flatten_to_white_background(image_bytes: bytes) -> bytes:
    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            out = white.convert("RGB")
        else:
            out = img.convert("RGB")
        buffer = BytesIO()
        out.save(buffer, format="PNG")
        return buffer.getvalue()


def should_flatten_output() -> bool:
    flag = _env_flag("OUTPUT_WHITE_BG")
    return bool(flag)


def should_clear_recipient_alpha_for_submit() -> bool:
    flag = _env_flag("CLEAR_RECIPIENT_ALPHA")
    return bool(flag)


def should_align_donor() -> bool:
    flag = _env_flag("ALIGN_DONOR_TO_RECIPIENT")
    return True if flag is None else bool(flag)


def summarize_response_for_debug(response) -> str:
    summaries = []
    parts = getattr(response, "parts", None)
    if parts:
        ptypes = []
        for part in parts:
            if getattr(part, "inline_data", None):
                ptypes.append("inline_data")
            elif getattr(part, "text", None):
                ptypes.append("text")
            else:
                ptypes.append(type(part).__name__)
        summaries.append(f"response.parts={','.join(ptypes) or 'none'}")
    for idx, candidate in enumerate(getattr(response, "candidates", [])):
        reason = getattr(candidate, "finish_reason", None) or "-"
        part_types = []
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) if content else []:
            if getattr(part, "inline_data", None):
                part_types.append("inline_data")
            elif getattr(part, "text", None):
                part_types.append("text")
            else:
                part_types.append(type(part).__name__)
        summaries.append(f"cand{idx}: reason={reason}, parts={','.join(part_types) or 'none'}")
    return "; ".join(summaries) if summaries else "no candidates"


def _pick_prompt_match(paths: list[Path], prompt_id: int) -> Path:
    if len(paths) == 1:
        return paths[0]
    filtered = [
        p for p in paths if not re.search(r"(?:\\bcopy\\b|\\bbackup\\b|\\bold\\b)", p.stem, re.IGNORECASE)
    ]
    if not filtered:
        filtered = paths
    filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = filtered[0]
    matches = ", ".join(p.name for p in sorted(paths))
    print(f"Warning: multiple prompt files match PID{prompt_id}: {matches}. Using {chosen.name}.")
    return chosen


def _select_prompt_by_id(prompt_id: int) -> Path:
    if not PROMPTS_DIR.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    strict: list[Path] = []
    fallback: list[Path] = []
    pid = re.escape(str(prompt_id))
    strict_pat = re.compile(rf"PID{pid}\b", re.IGNORECASE)
    fallback_pat = re.compile(rf"PID[_-]?{pid}\b", re.IGNORECASE)

    for path in PROMPTS_DIR.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if strict_pat.search(name):
            strict.append(path)
        elif fallback_pat.search(name):
            fallback.append(path)

    if len(strict) >= 1:
        return _pick_prompt_match(strict, prompt_id)
    if len(fallback) >= 1:
        return _pick_prompt_match(fallback, prompt_id)

    raise FileNotFoundError(
        f"No prompt file found containing PID{prompt_id} in the filename."
    )


def _extract_pid_from_name(name: str) -> Optional[str]:
    match = re.search(r"PID[_-]?(\d+)", name, re.IGNORECASE)
    return match.group(1) if match else None


def load_prompt_text(cli_prompt_id: Optional[str]) -> Tuple[str, str, Path]:
    prompt_path_env = os.getenv("PROMPT_PATH")
    prompt_id_env = cli_prompt_id or os.getenv("PROMPT_ID", str(DEFAULT_PROMPT_ID))
    try:
        prompt_id = int(prompt_id_env)
    except ValueError:
        prompt_id = DEFAULT_PROMPT_ID
    if prompt_path_env and not cli_prompt_id:
        prompt_path = Path(prompt_path_env)
        if not prompt_path.is_absolute():
            prompt_path = SCRIPT_DIR / prompt_path
        prompt_id_label = _extract_pid_from_name(prompt_path.name) or str(prompt_id)
    else:
        prompt_path = _select_prompt_by_id(prompt_id)
        prompt_id_label = str(prompt_id)

    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    print(f"Using prompt: {prompt_path}")
    return prompt_text, prompt_id_label, prompt_path


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_rgb_env(name: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    raw = os.getenv(name)
    if not raw:
        return default
    parts = raw.replace("#", "").split(",")
    if len(parts) == 1 and len(parts[0]) in {6, 8}:
        hex_val = parts[0][:6]
        return tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
    if len(parts) != 3:
        return default
    try:
        return tuple(max(0, min(255, int(p.strip()))) for p in parts)  # type: ignore
    except Exception:
        return default


def is_shadow_only(prompt_path: Path) -> bool:
    env = _env_flag("SHADOW_ONLY")
    if env is not None:
        return env
    name = prompt_path.name.lower()
    return "shadow_only" in name or "shadow-only" in name


def use_vertex_backend() -> bool:
    explicit = _env_flag("USE_VERTEX")
    if explicit is not None:
        return explicit
    fallback = _env_flag("GOOGLE_GENAI_USE_VERTEXAI")
    return fallback if fallback is not None else False


def _ensure_vertex_credentials() -> str:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("VERTEX_CREDENTIALS")
    if not creds:
        default_path = SCRIPT_DIR / "vertex.json"
        if default_path.exists():
            creds = str(default_path)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    if not creds or not Path(creds).is_file():
        raise FileNotFoundError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to '{creds}', which does not exist."
        )
    return creds


def create_client() -> genai.Client:
    if use_vertex_backend():
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT for Vertex AI usage.")
        _ensure_vertex_credentials()
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        return genai.Client(vertexai=True, project=project, location=location)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY for Gemini API usage.")
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))


_THREAD_LOCAL = threading.local()


def get_thread_client() -> genai.Client:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = create_client()
        _THREAD_LOCAL.client = client
    return client


def normalize_model_id(model_name: str, use_vertex: bool) -> str:
    if use_vertex:
        return model_name[7:] if model_name.startswith("models/") else model_name
    return model_name if model_name.startswith("models/") else f"models/{model_name}"


def image_part_from_bytes(data: bytes, mime_type: str = "image/png") -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    return {"inline_data": {"mime_type": mime_type, "data": encoded}}


def resize_to_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    """Resize (up or down) to the requested long-edge size."""
    if long_edge <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest == long_edge:
        return img
    scale = long_edge / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return img.resize(new_size, resample=Image.LANCZOS)


def resize_to_long_edge_cap(img: Image.Image, long_edge: int) -> Image.Image:
    """Resize down to the requested long-edge size; never upsample."""
    if long_edge <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= long_edge:
        return img
    scale = long_edge / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return img.resize(new_size, resample=Image.LANCZOS)


def resize_to_height(img: Image.Image, desired_height: int) -> Image.Image:
    width, height = img.size
    if height == desired_height:
        return img
    scale = desired_height / float(height)
    new_size = (max(1, int(round(width * scale))), desired_height)
    return img.resize(new_size, resample=Image.LANCZOS)


def write_debug_side_by_side(
    donor_path: Path,
    result_bytes: bytes,
    debug_path: Path,
    long_edge: int = DEBUG_LONG_EDGE,
) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(donor_path) as donor_img:
        donor = donor_img.convert("RGBA")
    with Image.open(BytesIO(result_bytes)) as out_img:
        result = out_img.convert("RGBA")

    composite_h = max(donor.size[1], result.size[1])
    donor = resize_to_height(donor, composite_h)
    result = resize_to_height(result, composite_h)

    composite = Image.new(
        "RGBA",
        (donor.size[0] + result.size[0], composite_h),
        (255, 255, 255, 255),
    )
    composite.paste(donor, (0, 0), donor)
    composite.paste(result, (donor.size[0], 0), result)

    composite = resize_to_long_edge(composite, long_edge)
    composite.convert("RGB").save(debug_path, format="PNG")


def sample_background_color(img: Image.Image) -> Tuple[int, int, int]:
    sample = img.convert("RGBA")
    max_dim = 256
    if max(sample.size) > max_dim:
        sample.thumbnail((max_dim, max_dim), Image.LANCZOS)
    width, height = sample.size
    pixels = sample.load()

    bg_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a < 10:
                bg_pixels.append((r, g, b))

    if not bg_pixels:
        border = max(1, min(width, height) // 10)
        for y in range(height):
            for x in range(width):
                if x < border or x >= width - border or y < border or y >= height - border:
                    r, g, b, _ = pixels[x, y]
                    bg_pixels.append((r, g, b))

    if not bg_pixels:
        return (255, 255, 255)

    r = int(sum(p[0] for p in bg_pixels) / len(bg_pixels))
    g = int(sum(p[1] for p in bg_pixels) / len(bg_pixels))
    b = int(sum(p[2] for p in bg_pixels) / len(bg_pixels))
    return (r, g, b)


def _alpha_bbox(img: Image.Image, threshold: int = 10) -> Optional[Tuple[int, int, int, int]]:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[3]
    mask = alpha.point(lambda a: 255 if a > threshold else 0)
    return mask.getbbox()


def _donor_bbox_from_bg(img: Image.Image, threshold: int = 24) -> Optional[Tuple[int, int, int, int]]:
    src = img.convert("RGB")
    w, h = src.size
    scale = 1.0
    max_dim = 512
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        src = src.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    bg = sample_background_color(src)
    pixels = src.load()
    min_x = min_y = None
    max_x = max_y = None
    for y in range(src.size[1]):
        for x in range(src.size[0]):
            r, g, b = pixels[x, y]
            dist = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
            if dist > threshold:
                if min_x is None:
                    min_x = max_x = x
                    min_y = max_y = y
                else:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
    if min_x is None:
        return None
    inv_scale = 1.0 / scale
    return (
        int(min_x * inv_scale),
        int(min_y * inv_scale),
        int((max_x + 1) * inv_scale),
        int((max_y + 1) * inv_scale),
    )


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    margin: float,
    size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if margin <= 0:
        return bbox
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return bbox
    mx = int(round(w * margin))
    my = int(round(h * margin))
    img_w, img_h = size
    return (
        max(0, x0 - mx),
        max(0, y0 - my),
        min(img_w, x1 + mx),
        min(img_h, y1 + my),
    )


def align_donor_to_recipient(
    donor_path: Path,
    recipient_path: Path,
    output_path: Path,
    long_edge: int,
    bg_threshold: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(donor_path) as donor_img:
        donor_rgb = donor_img.convert("RGB")
    with Image.open(recipient_path) as rec_img:
        recipient_rgba = rec_img.convert("RGBA")

    rec_bbox = _alpha_bbox(recipient_rgba)
    donor_bbox = _donor_bbox_from_bg(donor_rgb, threshold=bg_threshold)
    if not rec_bbox or not donor_bbox:
        convert_to_low_res_png(donor_path, output_path, long_edge)
        return output_path

    rec_w = rec_bbox[2] - rec_bbox[0]
    rec_h = rec_bbox[3] - rec_bbox[1]
    donor_w = donor_bbox[2] - donor_bbox[0]
    donor_h = donor_bbox[3] - donor_bbox[1]
    if donor_w <= 0 or donor_h <= 0 or rec_w <= 0 or rec_h <= 0:
        convert_to_low_res_png(donor_path, output_path, long_edge)
        return output_path

    crop_enabled = _env_flag("DONOR_CROP_TO_SUBJECT") is not False
    margin = float(os.getenv("DONOR_CROP_MARGIN", "0.2")) if crop_enabled else 0.0
    expanded = _expand_bbox(donor_bbox, margin, donor_rgb.size) if crop_enabled else donor_bbox
    donor_crop = donor_rgb.crop(expanded) if crop_enabled else donor_rgb
    donor_offset_x = donor_bbox[0] - expanded[0]
    donor_offset_y = donor_bbox[1] - expanded[1]

    scale = min(rec_w / float(donor_w), rec_h / float(donor_h))
    scaled_size = (
        max(1, int(round(donor_crop.size[0] * scale))),
        max(1, int(round(donor_crop.size[1] * scale))),
    )
    donor_scaled = donor_crop.resize(scaled_size, Image.LANCZOS)
    donor_bbox_scaled = (
        int(round(donor_offset_x * scale)),
        int(round(donor_offset_y * scale)),
        int(round((donor_offset_x + donor_w) * scale)),
        int(round((donor_offset_y + donor_h) * scale)),
    )

    rec_center = (
        (rec_bbox[0] + rec_bbox[2]) / 2.0,
        (rec_bbox[1] + rec_bbox[3]) / 2.0,
    )
    donor_center = (
        (donor_bbox_scaled[0] + donor_bbox_scaled[2]) / 2.0,
        (donor_bbox_scaled[1] + donor_bbox_scaled[3]) / 2.0,
    )
    offset_x = int(round(rec_center[0] - donor_center[0]))
    offset_y = int(round(rec_center[1] - donor_center[1]))

    bg_color = sample_background_color(donor_rgb)
    canvas = Image.new("RGB", recipient_rgba.size, bg_color)
    canvas.paste(donor_scaled, (offset_x, offset_y))
    canvas = resize_to_long_edge_cap(canvas, long_edge)
    canvas.save(output_path, format="PNG")
    return output_path


def _mask_bbox(mask: Image.Image, threshold: int = 10) -> Optional[Tuple[int, int, int, int]]:
    if mask.mode != "L":
        mask = mask.convert("L")
    binary = mask.point(lambda a: 255 if a > threshold else 0)
    return binary.getbbox()


def _bottom_profile(
    mask: Image.Image, threshold: int, percentile: float, band: int
) -> Optional[Tuple[float, float, float]]:
    arr = np.asarray(mask)
    ys, xs = np.where(arr > threshold)
    if xs.size == 0:
        return None
    # For each column, keep the lowest (max y) pixel.
    max_y = {}
    for x, y in zip(xs, ys):
        prev = max_y.get(x)
        if prev is None or y > prev:
            max_y[x] = y
    xs_arr = np.fromiter(max_y.keys(), dtype=np.float32)
    ys_arr = np.fromiter(max_y.values(), dtype=np.float32)
    if xs_arr.size == 0:
        return None
    bottom_y = float(np.percentile(ys_arr, percentile))
    sel = ys_arr >= (bottom_y - max(0, band))
    if not np.any(sel):
        sel = slice(None)
    xs_sel = xs_arr[sel]
    min_x = float(np.percentile(xs_sel, 5))
    max_x = float(np.percentile(xs_sel, 95))
    return bottom_y, min_x, max_x


def align_shadow_alpha_to_feet(
    shadow_alpha: Image.Image,
    recipient_alpha: Image.Image,
    width_factor: float,
    threshold: int,
    offset_x: int,
    offset_y: int,
) -> Image.Image:
    if recipient_alpha.mode != "L":
        recipient_alpha = recipient_alpha.convert("L")
    if shadow_alpha.mode != "L":
        shadow_alpha = shadow_alpha.convert("L")

    foot_percentile = float(os.getenv("SHADOW_ALIGN_FOOT_PERCENTILE", "90"))
    foot_band = int(os.getenv("SHADOW_ALIGN_FOOT_BAND", "12"))

    rec_profile = _bottom_profile(recipient_alpha, threshold, foot_percentile, foot_band)
    sh_profile = _bottom_profile(shadow_alpha, threshold, foot_percentile, foot_band)
    if not rec_profile or not sh_profile:
        return shadow_alpha

    rec_bottom, rec_min_x, rec_max_x = rec_profile
    sh_bottom, sh_min_x, sh_max_x = sh_profile
    rec_w = rec_max_x - rec_min_x
    sh_w = sh_max_x - sh_min_x
    if rec_w <= 0 or sh_w <= 0:
        return shadow_alpha

    scale = (rec_w * width_factor) / float(sh_w)
    new_w = max(1, int(round(shadow_alpha.size[0] * scale)))
    new_h = max(1, int(round(shadow_alpha.size[1] * scale)))
    scaled = shadow_alpha.resize((new_w, new_h), Image.LANCZOS)

    sh_center_x = ((sh_min_x + sh_max_x) / 2.0) * scale
    target_center_x = (rec_min_x + rec_max_x) / 2.0 + offset_x
    paste_x = int(round(target_center_x - sh_center_x))
    sh_bottom_scaled = sh_bottom * scale
    paste_y = int(round(rec_bottom - sh_bottom_scaled + offset_y))

    aligned = Image.new("L", shadow_alpha.size, 0)
    aligned.paste(scaled, (paste_x, paste_y))
    return aligned


def align_shadow_alpha(
    shadow_alpha: Image.Image,
    recipient_alpha: Image.Image,
    width_factor: float,
    threshold: int,
    offset_x: int,
    offset_y: int,
) -> Image.Image:
    rec_bbox = _alpha_bbox(
        Image.merge("RGBA", (recipient_alpha, recipient_alpha, recipient_alpha, recipient_alpha))
    )
    sh_bbox = _mask_bbox(shadow_alpha, threshold=threshold)
    if not rec_bbox or not sh_bbox:
        return shadow_alpha

    rec_w = rec_bbox[2] - rec_bbox[0]
    rec_h = rec_bbox[3] - rec_bbox[1]
    sh_w = sh_bbox[2] - sh_bbox[0]
    sh_h = sh_bbox[3] - sh_bbox[1]
    if rec_w <= 0 or rec_h <= 0 or sh_w <= 0 or sh_h <= 0:
        return shadow_alpha

    scale = (rec_w * width_factor) / float(sh_w)
    new_w = max(1, int(round(sh_w * scale)))
    new_h = max(1, int(round(sh_h * scale)))

    shadow_crop = shadow_alpha.crop(sh_bbox).resize((new_w, new_h), Image.LANCZOS)
    target_cx = (rec_bbox[0] + rec_bbox[2]) / 2.0 + offset_x
    target_bottom = rec_bbox[3] + offset_y
    paste_x = int(round(target_cx - new_w / 2.0))
    paste_y = int(round(target_bottom - new_h))

    aligned = Image.new("L", shadow_alpha.size, 0)
    aligned.paste(shadow_crop, (paste_x, paste_y))
    return aligned


def apply_shadow_mask_to_recipient(
    recipient_path: Path, mask_bytes: bytes
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    with Image.open(recipient_path) as rec:
        recipient_rgba = rec.convert("RGBA")

    with Image.open(BytesIO(mask_bytes)) as mask_img:
        mask = mask_img.convert("L")
    if mask.size != recipient_rgba.size:
        mask = mask.resize(recipient_rgba.size, Image.LANCZOS)

    white_is_shadow = _env_flag("SHADOW_MASK_WHITE_IS_SHADOW")
    if white_is_shadow is None:
        white_is_shadow = False
    shadow_alpha = mask if white_is_shadow else ImageOps.invert(mask)

    strength = float(os.getenv("SHADOW_MASK_STRENGTH", "1.0"))
    if strength != 1.0:
        shadow_alpha = shadow_alpha.point(lambda a: int(max(0, min(255, a * strength))))

    min_alpha = int(os.getenv("SHADOW_MASK_MIN_ALPHA", "0"))
    if min_alpha > 0:
        shadow_alpha = shadow_alpha.point(lambda a: 0 if a < min_alpha else a)

    def _mask_max(img: Image.Image) -> int:
        extrema = img.getextrema()
        if isinstance(extrema, tuple):
            return extrema[1]
        return 0

    remove_enabled = _env_flag("SHADOW_MASK_REMOVE_RECIPIENT")
    if remove_enabled is None:
        remove_enabled = True
    if remove_enabled:
        pre_remove = shadow_alpha
        alpha_thresh = int(os.getenv("SHADOW_MASK_REMOVE_THRESHOLD", "8"))
        remove = recipient_rgba.split()[3].point(lambda a: 255 if a > alpha_thresh else 0)
        dilate = int(os.getenv("SHADOW_MASK_REMOVE_DILATE", "4"))
        for _ in range(max(0, dilate)):
            remove = remove.filter(ImageFilter.MaxFilter(3))
        shadow_alpha = ImageChops.subtract(shadow_alpha, remove)
        if _mask_max(shadow_alpha) == 0:
            shadow_alpha = pre_remove

    align_enabled = _env_flag("SHADOW_MASK_ALIGN")
    if align_enabled is None:
        align_enabled = True
    if align_enabled:
        pre_align = shadow_alpha
        width_factor = float(os.getenv("SHADOW_ALIGN_WIDTH_FACTOR", "1.0"))
        threshold = int(os.getenv("SHADOW_ALIGN_THRESHOLD", "12"))
        if threshold <= 0:
            arr = np.asarray(shadow_alpha).astype(np.float32)
            pct = float(os.getenv("SHADOW_ALIGN_PERCENTILE", "90"))
            threshold = int(np.percentile(arr, pct))
        offset_x = int(os.getenv("SHADOW_ALIGN_OFFSET_X", "0"))
        offset_y = int(os.getenv("SHADOW_ALIGN_OFFSET_Y", "0"))
        align_mode = str(os.getenv("SHADOW_ALIGN_MODE", "feet")).strip().lower()
        if align_mode == "feet":
            shadow_alpha = align_shadow_alpha_to_feet(
                shadow_alpha,
                recipient_rgba.split()[3],
                width_factor,
                threshold,
                offset_x,
                offset_y,
            )
        else:
            shadow_alpha = align_shadow_alpha(
                shadow_alpha,
                recipient_rgba.split()[3],
                width_factor,
                threshold,
                offset_x,
                offset_y,
            )
        if remove_enabled:
            alpha_thresh = int(os.getenv("SHADOW_MASK_REMOVE_THRESHOLD", "8"))
            remove = recipient_rgba.split()[3].point(lambda a: 255 if a > alpha_thresh else 0)
            dilate = int(os.getenv("SHADOW_MASK_REMOVE_DILATE", "4"))
            for _ in range(max(0, dilate)):
                remove = remove.filter(ImageFilter.MaxFilter(3))
            shadow_alpha = ImageChops.subtract(shadow_alpha, remove)
        if _mask_max(shadow_alpha) == 0:
            shadow_alpha = pre_align

    shadow_layer = Image.new("RGBA", recipient_rgba.size, (0, 0, 0, 0))
    shadow_layer.putalpha(shadow_alpha)

    combined = ImageChops.lighter(recipient_rgba.split()[3], shadow_alpha)
    output_rgba = recipient_rgba.copy()
    output_rgba.putalpha(combined)

    preview_rgba = Image.alpha_composite(shadow_layer, recipient_rgba)
    return output_rgba, preview_rgba, shadow_alpha


def composite_on_background(img: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
    base = Image.new("RGB", img.size, bg_color)
    base.paste(img.convert("RGBA"), (0, 0), img.convert("RGBA"))
    return base


def load_image_bytes_for_api(
    path: Path,
    long_edge: int,
    force_opaque: bool = False,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    temp_name: Optional[str] = None,
) -> Tuple[str, bytes, Tuple[int, int]]:
    """Always returns PNG bytes for Gemini to avoid TIFF ingestion issues."""
    with Image.open(path) as img:
        mode = "RGBA" if img.mode in ("RGBA", "LA") else "RGB"
        converted = img.convert(mode)
        if force_opaque and converted.mode == "RGBA":
            bg = Image.new("RGBA", converted.size, (*bg_color, 255))
            bg.alpha_composite(converted)
            converted = bg.convert("RGB")
        converted = resize_to_long_edge(converted, long_edge)
        final_size = converted.size
        # Saving to a real temp file avoids PIL _idat.fileno issues on some builds.
        if temp_name:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / temp_name
                converted.save(tmp_path, format="PNG")
                data = tmp_path.read_bytes()
                return "image/png", data, final_size
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            converted.save(tmp_path, format="PNG")
            data = tmp_path.read_bytes()
            return "image/png", data, final_size
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def _image_meta(path: Path) -> str:
    try:
        with Image.open(path) as img:
            fmt = (img.format or path.suffix.lstrip(".")).upper()
            return f"{path} ({img.size[0]}x{img.size[1]}, {img.mode}, {fmt})"
    except Exception as exc:  # noqa: BLE001
        return f"{path} (unreadable: {exc})"


def _image_alpha_status(path: Path) -> str:
    try:
        with Image.open(path) as img:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                return "alpha=yes"
            return "alpha=no"
    except Exception:
        return "alpha=unknown"


def get_role_dir(path: Path, role: str) -> Optional[Path]:
    for idx, part in enumerate(path.parts):
        if part.lower() == role.lower():
            return Path(*path.parts[: idx + 1])
    return None


def find_matching_stem(root: Path, stem: str) -> Optional[Path]:
    if not root or not root.is_dir():
        return None
    preferred_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    for ext in preferred_exts:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for path in root.iterdir():
        if path.is_file() and path.stem == stem and is_image_file(path):
            return path
    return None


def convert_to_low_res_png(
    src_path: Path,
    dest_path: Path,
    long_edge: int,
    role: str,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        mode = "RGBA" if img.mode in ("RGBA", "LA") else "RGB"
        converted = img.convert(mode)
        if role.lower() == "donor":
            crop_enabled = _env_flag("DONOR_CROP_TO_SUBJECT") is not False
            if crop_enabled:
                bg_thresh = int(os.getenv("DONOR_BG_THRESHOLD", "24"))
                bbox = _donor_bbox_from_bg(converted.convert("RGB"), threshold=bg_thresh)
                if bbox:
                    margin = float(os.getenv("DONOR_CROP_MARGIN", "0.2"))
                    expanded = _expand_bbox(bbox, margin, converted.size)
                    converted = converted.crop(expanded)
        if role.lower() == "recipient" and converted.mode == "RGBA":
            # Remove background alpha shadows: binarize alpha (opaque vs transparent).
            if _env_flag("CLEAR_RECIPIENT_BG_ALPHA") is not False:
                thresh = int(os.getenv("RECIPIENT_ALPHA_BG_THRESHOLD", "220"))
                r, g, b, a = converted.split()
                a = a.point(lambda v: 255 if v >= thresh else 0)
                converted = Image.merge("RGBA", (r, g, b, a))
        resized = resize_to_long_edge_cap(converted, long_edge)
        resized.save(dest_path, format="PNG")
    print(f"[low_res] { _image_meta(src_path) } -> { _image_meta(dest_path) }")


def ensure_low_res_png(src_path: Path, role: str, long_edge: int) -> Path:
    role_dir = get_role_dir(src_path, role)
    low_dir = role_dir / LOW_RES_DIRNAME if role_dir else None
    if low_dir:
        low_path = low_dir / f"{src_path.stem}.png"
        if not low_path.exists():
            print(f"Converting to low_res: {src_path} -> {low_path}")
            convert_to_low_res_png(src_path, low_path, long_edge, role)
        else:
            print(f"[low_res] exists: { _image_meta(low_path) } (source { _image_meta(src_path) })")
        return low_path
    if src_path.suffix.lower() == ".png":
        return src_path
    fallback = src_path.with_suffix(".png")
    if not fallback.exists():
        print(f"Converting to PNG: {src_path} -> {fallback}")
        convert_to_low_res_png(src_path, fallback, long_edge, role)
    else:
        print(f"[low_res] exists: { _image_meta(fallback) } (source { _image_meta(src_path) })")
    return fallback


def expand_tier_roots(roots: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        expanded.append(root)
        if root.name == HIGH_RES_DIRNAME:
            sibling = root.parent / LOW_RES_DIRNAME
        elif root.name == LOW_RES_DIRNAME:
            sibling = root.parent / HIGH_RES_DIRNAME
        else:
            sibling = None
        if sibling and sibling.is_dir() and sibling not in seen:
            seen.add(sibling)
            expanded.append(sibling)
    return expanded


def request_shadowed_image(
    client: genai.Client,
    model_id: str,
    recipient_path: Path,
    donor_path: Path,
    max_side: int,
    prompt_text: str,
    flatten_output: Optional[bool] = None,
    attempts: int = 3,
    backoff: int = 10,
) -> Tuple[bytes, bytes]:
    force_opaque = should_clear_recipient_alpha_for_submit()
    bg_color = _parse_rgb_env("RECIPIENT_BG_COLOR", (255, 255, 255))
    recipient_mime, recipient_bytes, recipient_sent_size = load_image_bytes_for_api(
        recipient_path,
        max_side,
        force_opaque=force_opaque,
        bg_color=bg_color,
        temp_name="recipiant.png",
    )
    donor_mime, donor_bytes, _ = load_image_bytes_for_api(
        donor_path,
        max_side,
        temp_name="donor.png",
    )
    if DONOR_CONTRAST_BOOST != 1.0:
        with Image.open(BytesIO(donor_bytes)) as donor_img:
            boosted = ImageEnhance.Contrast(donor_img).enhance(DONOR_CONTRAST_BOOST)
            buffer = BytesIO()
            boosted.save(buffer, format="PNG")
            donor_bytes = buffer.getvalue()
    recipient_output_size = recipient_sent_size
    prompt = prompt_text
    print(
        f"[submit] recipient={recipient_path.name} size={recipient_sent_size[0]}x{recipient_sent_size[1]} "
        f"bytes={len(recipient_bytes)} mime={recipient_mime}"
    )
    print(
        f"[submit] donor={donor_path.name} bytes={len(donor_bytes)} mime={donor_mime}"
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    image_part_from_bytes(recipient_bytes, recipient_mime),
                    image_part_from_bytes(donor_bytes, donor_mime),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_modalities=["IMAGE"],
                    safety_settings=DEFAULT_SAFETY_SETTINGS,
                ),
            )
            raw_bytes = extract_image_from_response(response)
            final_bytes = enforce_output_dimensions(raw_bytes, recipient_output_size)
            if flatten_output is None:
                flatten_output = should_flatten_output()
            if flatten_output:
                final_bytes = flatten_to_white_background(final_bytes)
            return raw_bytes, final_bytes
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err_text = str(exc)
            should_retry = any(token in err_text for token in ("Deadline", "timeout", "504"))
            if attempt >= attempts or not should_retry:
                break
            delay = backoff * attempt
            print(f"Retrying ({attempt}/{attempts}) after {delay}s due to: {exc}")
            time.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown failure requesting shadowed image.")


def _dir_has_images(root: Path) -> bool:
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if is_image_file(path):
            return True
    return False


def _select_tier_dir(root: Path) -> Path:
    for tier in IMAGE_TIER_PREFERENCE:
        candidate = root / tier
        if _dir_has_images(candidate):
            return candidate
    return root


def _collect_role_dirs(base: Path, role: str) -> list[Path]:
    role_dirs: list[Path] = []
    if base.is_dir():
        if base.name.lower() == role.lower():
            role_dirs.append(base)
        else:
            for path in base.rglob(role):
                if path.is_dir() and path.name.lower() == role.lower():
                    role_dirs.append(path)
    return role_dirs


def resolve_image_roots(base: Path, role: str) -> list[Path]:
    role_dirs = _collect_role_dirs(base, role)
    if not role_dirs and base.is_dir() and base.name.lower() != role.lower():
        # If base itself might contain images, allow it directly.
        if _dir_has_images(base):
            role_dirs = [base]

    resolved = [_select_tier_dir(d) for d in role_dirs]
    return resolved


def path_has_role(path: Path, role: str) -> bool:
    role_lower = role.lower()
    parts = [p.lower() for p in path.parts]
    return role_lower in parts or f"{role_lower}s" in parts


def iter_images(roots: Iterable[Path], role_filter: Optional[str] = None) -> Iterable[Path]:
    for root in roots:
        for path in root.rglob("*"):
            if is_image_file(path):
                if role_filter and not path_has_role(path, role_filter):
                    continue
                yield path


def build_donor_index(roots: Iterable[Path]) -> Dict[ImageKey, Dict[str, Path]]:
    index: Dict[ImageKey, Dict[str, Path]] = {}
    for root in roots:
        for file_path in root.rglob("*"):
            if not is_image_file(file_path):
                continue
            if not path_has_role(file_path, "donor"):
                continue
            product = extract_product_code(file_path)
            view = extract_view_code(file_path)
            if not product or not view:
                continue
            key = (product, view)
            entry = index.setdefault(key, {})
            if HIGH_RES_DIRNAME in file_path.parts:
                entry.setdefault("high", file_path)
            elif LOW_RES_DIRNAME in file_path.parts:
                entry.setdefault("low", file_path)
            else:
                entry.setdefault("low", file_path)
    return index


def main() -> None:
    # Load environment variables from the script directory so we pick up .env reliably.
    load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

    parser = argparse.ArgumentParser(description="Apply donor shadows to recipient images.")
    parser.add_argument("--recipients", default="recipients", help="Recipient images root directory.")
    parser.add_argument("--donors", default="donors", help="Donor (shadow source) images root directory.")
    parser.add_argument("--product", help="Only process recipients matching this product code (e.g., E210723531).")
    parser.add_argument("--view", help="Only process recipients matching this view code (e.g., TQ).")
    parser.add_argument("--file", help="Only process a single recipient file (match by filename or stem).")
    parser.add_argument(
        "--pid",
        default=str(DEFAULT_PROMPT_ID),
        help=f"Prompt ID to use (default PID{DEFAULT_PROMPT_ID}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (and debug images).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated artifacts (low_res, generated outputs, debug) and exit.",
    )
    parser.add_argument(
        "--async",
        dest="max_async",
        type=int,
        default=int(os.getenv("MAX_ASYNC", "1")),
        help="Max number of concurrent submissions (default 1 or MAX_ASYNC).",
    )
    parser.add_argument(
        "--output",
        default="products",
        help="Base directory for outputs (per-product outputs go to <base>/<product>/generated).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=int(os.getenv("MAX_SIDE", str(FOUR_K_LONG_EDGE))),
        help="Requested long-edge resolution to send to Gemini (0 for uncapped request; actual send is clamped by API_MAX_SIDE).",
    )
    args = parser.parse_args()

    if args.clean:
        removed = 0
        targets = []
        if DEFAULT_PRODUCTS_DIR.is_dir():
            for path in DEFAULT_PRODUCTS_DIR.rglob("*"):
                if not path.is_file():
                    continue
                parts = [p.lower() for p in path.parts]
                if HIGH_RES_DIRNAME in parts:
                    continue
                if path.suffix.lower() in {".tif", ".tiff"}:
                    continue
                if LOW_RES_DIRNAME in parts or "generated" in parts:
                    targets.append(path)
        if DEBUG_DIR.is_dir():
            targets.extend([p for p in DEBUG_DIR.rglob("*") if p.is_file()])
        output_dir = SCRIPT_DIR / "output"
        if output_dir.is_dir():
            targets.extend([p for p in output_dir.rglob("*") if p.is_file()])

        for path in targets:
            try:
                path.unlink()
                removed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to delete {path}: {exc}")

        # Remove empty directories under output/debug (e.g., legacy subfolders).
        for root in (SCRIPT_DIR / "output", DEBUG_DIR):
            if not root.is_dir():
                continue
            for dirpath in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if dirpath.is_dir():
                    try:
                        if not any(dirpath.iterdir()):
                            dirpath.rmdir()
                    except Exception:
                        pass
        print(f"Cleaned {removed} files.")
        return

    try:
        _THREAD_LOCAL.client = create_client()
    except Exception as exc:  # noqa: BLE001
        raise EnvironmentError(str(exc)) from exc

    using_vertex = use_vertex_backend()
    model_id = normalize_model_id(args.model, using_vertex)

    api_max_side = int(os.getenv("API_MAX_SIDE", str(FOUR_K_LONG_EDGE)))
    requested_max_side = args.max_side
    effective_max_side = api_max_side if requested_max_side <= 0 else min(requested_max_side, api_max_side)

    prompt_text, prompt_id_label, prompt_path = load_prompt_text(args.pid)
    prompt_lower = prompt_text.lower()
    shadow_only_mode = (
        is_shadow_only(prompt_path)
        or "shadow mask" in prompt_lower
        or "shadow-only" in prompt_lower
        or "shadow_only" in prompt_lower
    )

    recipient_root = Path(args.recipients)
    donor_root = Path(args.donors)
    explicit_recipients = args.recipients != "recipients"
    explicit_donors = args.donors != "donors"

    recipient_roots = resolve_image_roots(recipient_root, "recipient")
    donor_roots = resolve_image_roots(donor_root, "donor") or resolve_image_roots(
        donor_root, "donors"
    )

    if DEFAULT_PRODUCTS_DIR.is_dir():
        product_recipients = resolve_image_roots(DEFAULT_PRODUCTS_DIR, "recipient")
        product_donors = resolve_image_roots(DEFAULT_PRODUCTS_DIR, "donor")
        if (not explicit_recipients) and product_recipients:
            recipient_roots = product_recipients
        if (not explicit_donors) and product_donors:
            donor_roots = product_donors
        if not recipient_roots and product_recipients:
            recipient_roots = product_recipients
        if not donor_roots and product_donors:
            donor_roots = product_donors

    if not recipient_roots:
        raise FileNotFoundError(
            f"No recipient images found under {recipient_root} (or {DEFAULT_PRODUCTS_DIR}/**/recipient)."
        )
    if not donor_roots:
        raise FileNotFoundError(
            f"No donor images found under {donor_root} (or {DEFAULT_PRODUCTS_DIR}/**/donor)."
        )

    output_base = Path(args.output)

    backend_label = "VERTEX" if using_vertex else "GEMINI API"
    print(f"Backend: {backend_label}")
    if using_vertex:
        print(
            f"Vertex project={os.getenv('GOOGLE_CLOUD_PROJECT')} "
            f"location={os.getenv('GOOGLE_CLOUD_LOCATION', 'global')}"
        )
    print(f"Using model: {model_id}")
    print(
        f"Input sizing: requested long_edge={requested_max_side or 'uncapped'}, "
        f"API cap={api_max_side}, effective long_edge={effective_max_side}"
    )
    print("Recipient roots:")
    for root in recipient_roots:
        print(f"  - {root}")
    print("Donor roots:")
    for root in donor_roots:
        print(f"  - {root}")

    donor_scan_roots = expand_tier_roots(donor_roots)
    print("Indexing donors...")
    donor_index = build_donor_index(donor_scan_roots)
    print(f"Indexed {len(donor_index)} donor views.")

    file_filter = args.file
    if file_filter:
        file_filter = file_filter.strip()
        file_filter = Path(file_filter).name
        file_stem = Path(file_filter).stem
    else:
        file_stem = None

    work_items: list[dict] = []
    for recipient_path in iter_images(recipient_roots, role_filter="recipient"):
        if file_filter:
            if recipient_path.name != file_filter and recipient_path.stem != file_stem:
                continue
        product_name = None
        parts = list(recipient_path.parts)
        if "products" in parts:
            idx = parts.index("products")
            if idx + 1 < len(parts):
                product_name = parts[idx + 1]

        if product_name:
            output_base_str = str(output_base)
            if "{product}" in output_base_str:
                output_root = Path(output_base_str.format(product=product_name))
            elif output_base.name == "generated":
                output_root = output_base
            elif output_base.name == "products":
                output_root = output_base / product_name / "generated"
            else:
                output_root = output_base
        else:
            output_root = output_base

        output_root.mkdir(parents=True, exist_ok=True)
        stem_base = re.sub(r"(?:[_-]?PID\\d+)$", "", recipient_path.stem, flags=re.IGNORECASE)
        output_path = output_root / f"{stem_base}_PID{prompt_id_label}.png"
        if output_path.exists() and not args.force:
            print(f"Skipping existing output: {output_path.name} (use --force to overwrite)")
            continue

        product = extract_product_code(recipient_path)
        view = extract_view_code(recipient_path)
        if not product or not view:
            print(f"Could not parse product/view for recipient: {recipient_path}")
            continue
        if args.product and product != args.product:
            continue
        if args.view and view.lower() != args.view.lower():
            continue

        donor_entry = donor_index.get((product, view))
        if not donor_entry:
            print(f"No donor found for recipient {recipient_path.name} (product {product}, view {view})")
            continue

        work_items.append(
            {
                "recipient_path": recipient_path,
                "output_path": output_path,
                "stem_base": stem_base,
                "donor_entry": donor_entry,
            }
        )

    if not work_items:
        print("No recipients matched the filters.")
        return

    def process_one(item: dict) -> None:
        recipient_path = item["recipient_path"]
        output_path = item["output_path"]
        stem_base = item["stem_base"]
        donor_entry = item["donor_entry"]

        if output_path.exists() and not args.force:
            print(f"Skipping existing output: {output_path.name} (use --force to overwrite)")
            return

        donor_high = donor_entry.get("high") or donor_entry.get("low")
        if not donor_high:
            print(f"No usable donor image for recipient {recipient_path.name}")
            return

        recipient_high = recipient_path
        recipient_role = get_role_dir(recipient_path, "recipient")
        if recipient_role:
            high_dir = recipient_role / HIGH_RES_DIRNAME
            if high_dir.is_dir() and recipient_path.parent.name != HIGH_RES_DIRNAME:
                candidate = find_matching_stem(high_dir, recipient_path.stem)
                if candidate:
                    recipient_high = candidate
        recipient_low = ensure_low_res_png(recipient_high, "recipient", FOUR_K_LONG_EDGE)

        if should_align_donor():
            bg_thresh = int(os.getenv("DONOR_BG_THRESHOLD", "24"))
            donor_dir = get_role_dir(donor_high, "donor") or donor_high.parent
            donor_low_path = donor_dir / LOW_RES_DIRNAME / f"{donor_high.stem}.png"
            donor_low = align_donor_to_recipient(
                donor_high,
                recipient_high,
                donor_low_path,
                FOUR_K_LONG_EDGE,
                bg_thresh,
            )
        else:
            donor_low = ensure_low_res_png(donor_high, "donor", FOUR_K_LONG_EDGE)

        print(f"[paths] donor_high={donor_high} ({_image_alpha_status(donor_high)})")
        print(f"[paths] donor_low={donor_low} ({_image_alpha_status(donor_low)})")
        print(f"[paths] recipient_high={recipient_high} ({_image_alpha_status(recipient_high)})")
        print(f"[paths] recipient_low={recipient_low} ({_image_alpha_status(recipient_low)})")

        client_local = get_thread_client()
        print(f"Processing recipient {recipient_low.name} with donor {donor_low.name}...")
        if should_clear_recipient_alpha_for_submit():
            print("   Recipient alpha cleared for submission.")
        shadow_request_max_side = effective_max_side
        if shadow_only_mode:
            # Avoid upscaling the recipient for shadow masks; keep native low_res size.
            with Image.open(recipient_low) as rec_img:
                shadow_request_max_side = min(shadow_request_max_side, max(rec_img.size))
            print("   Shadow-only mode: requesting shadow mask from model.")
            try:
                raw_bytes, mask_bytes = request_shadowed_image(
                    client_local,
                    model_id,
                    recipient_low,
                    donor_low,
                    shadow_request_max_side,
                    prompt_text,
                    flatten_output=False,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to process {recipient_path.name}: {exc}")
                return
        else:
            print(
                f"   Sending both images to Gemini as PNG (converted in-memory, "
                f"long_edge={effective_max_side or 'original'}); enforcing responses to the same size."
            )
            try:
                raw_bytes, image_bytes = request_shadowed_image(
                    client_local,
                    model_id,
                    recipient_low,
                    donor_low,
                    effective_max_side,
                    prompt_text,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to process {recipient_path.name}: {exc}")
                return

        raw_path = DEBUG_DIR / f"{stem_base}_PID{prompt_id_label}_raw.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw_bytes)
        print(f"Raw: {raw_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if shadow_only_mode:
            out_rgba, preview_rgba, processed_mask = apply_shadow_mask_to_recipient(
                recipient_low, mask_bytes
            )

            out_rgba.save(output_path, format="PNG")
            print(f"Wrote {output_path}")

            output_mirror = SCRIPT_DIR / "output" / f"{stem_base}_PID{prompt_id_label}.png"
            output_mirror.parent.mkdir(parents=True, exist_ok=True)
            out_rgba.save(output_mirror, format="PNG")
            print(f"Wrote {output_mirror}")

            # Save original recipient alpha for safekeeping.
            recipient_alpha_path = DEBUG_DIR / f"{stem_base}_PID{prompt_id_label}_recipient_alpha.png"
            recipient_alpha_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(recipient_low) as rec_img:
                rec_img.convert("RGBA").split()[3].save(recipient_alpha_path, format="PNG")
            print(f"Recipient alpha: {recipient_alpha_path}")

            mask_path = DEBUG_DIR / f"{stem_base}_PID{prompt_id_label}_shadowmask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(BytesIO(mask_bytes)) as mask_img:
                mask_img.convert("L").save(mask_path, format="PNG")
            print(f"Mask: {mask_path}")

            processed_mask_path = DEBUG_DIR / f"{stem_base}_PID{prompt_id_label}_shadowmask_processed.png"
            processed_mask_path.parent.mkdir(parents=True, exist_ok=True)
            processed_mask.save(processed_mask_path, format="PNG")
            print(f"Mask processed: {processed_mask_path}")

            bg_source = donor_high or donor_low
            with Image.open(bg_source) as bg_img:
                bg_color = sample_background_color(bg_img)
            preview = composite_on_background(preview_rgba, bg_color)
            preview_path = DEBUG_DIR / f"{stem_base}_PID{prompt_id_label}_preview.png"
            preview.save(preview_path, format="PNG")
            print(f"Preview: {preview_path}")

            preview_output = SCRIPT_DIR / "output" / f"{stem_base}_PID{prompt_id_label}_preview.png"
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            preview.save(preview_output, format="PNG")
            print(f"Preview output: {preview_output}")

            preview_bytes = BytesIO()
            preview.save(preview_bytes, format="PNG")
            debug_name = f"{stem_base}_PID{prompt_id_label}_debug2k.png"
            debug_path = DEBUG_DIR / debug_name
            debug_donor = donor_high or donor_low
            write_debug_side_by_side(debug_donor, preview_bytes.getvalue(), debug_path)
            print(f"Debug: {debug_path}")
        else:
            output_path.write_bytes(image_bytes)
            print(f"Wrote {output_path}")

            output_mirror = SCRIPT_DIR / "output" / f"{stem_base}_PID{prompt_id_label}.png"
            output_mirror.parent.mkdir(parents=True, exist_ok=True)
            output_mirror.write_bytes(image_bytes)
            print(f"Wrote {output_mirror}")

            debug_name = f"{stem_base}_PID{prompt_id_label}_debug2k.png"
            debug_path = DEBUG_DIR / debug_name
            debug_donor = donor_high or donor_low
            write_debug_side_by_side(debug_donor, image_bytes, debug_path)
            print(f"Debug: {debug_path}")

    if args.max_async and args.max_async > 1:
        max_async = max(1, int(args.max_async))

        async def _run_all() -> None:
            sem = asyncio.Semaphore(max_async)

            async def _run(item: dict) -> None:
                async with sem:
                    await asyncio.to_thread(process_one, item)

            await asyncio.gather(*[_run(item) for item in work_items])

        asyncio.run(_run_all())
    else:
        for item in work_items:
            process_one(item)


if __name__ == "__main__":
    main()

"""Shadow transfer pipeline using Gemini 2.5 Flash.

Scans target images, finds matching donors by product code and view code,
requests Gemini to apply donor shadows to the target, and writes outputs
into an output/shadowed folder. Only processes targets that do not yet
have a corresponding output file.
"""

from __future__ import annotations

import argparse
import os
import re
import base64
from io import BytesIO
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Allow large source images; we downscale before sending to the API.
Image.MAX_IMAGE_PIXELS = None

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


def build_donor_index(donor_root: Path) -> Dict[ImageKey, Path]:
    index: Dict[ImageKey, Path] = {}
    for file_path in donor_root.rglob("*"):
        if not is_image_file(file_path):
            continue
        product = extract_product_code(file_path)
        view = extract_view_code(file_path)
        if not product or not view:
            continue
        key = (product, view)
        index[key] = file_path
    return index


def normalize_inline_data(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise TypeError(f"Unsupported inline data type: {type(data)}")


def extract_image_from_response(response) -> bytes:
    # Primary path: check candidates for inline image data
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return normalize_inline_data(inline.data)
    # Older/alternate response shape fallback
    for part in getattr(response, "parts", []):
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return normalize_inline_data(inline.data)
    raise RuntimeError("No image returned by Gemini response.")


def summarize_response_for_debug(response) -> str:
    summaries = []
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


def resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale image to max_side on the longest edge to keep payload reasonable."""
    if max_side <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, resample=Image.LANCZOS)


def load_image_bytes_for_api(path: Path, max_side: int) -> Tuple[str, bytes]:
    """Always returns PNG bytes for Gemini to avoid TIFF ingestion issues."""
    with Image.open(path) as img:
        # Preserve transparency if present; otherwise keep RGB.
        mode = "RGBA" if img.mode in ("RGBA", "LA") else "RGB"
        converted = img.convert(mode)
        converted = resize_if_needed(converted, max_side)
        # Saving to a real temp file avoids PIL _idat.fileno issues on some builds.
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            converted.save(tmp_path, format="PNG")
            data = tmp_path.read_bytes()
            return "image/png", data
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def request_shadowed_image(
    model: genai.GenerativeModel, target_path: Path, donor_path: Path, max_side: int
) -> bytes:
    target_mime, target_bytes = load_image_bytes_for_api(target_path, max_side)
    donor_mime, donor_bytes = load_image_bytes_for_api(donor_path, max_side)
    prompt = (
        "Apply the donor image's shadowing onto the target image with high fidelity. "
        "Match the donor's shadow direction, softness, edge diffusion/blur, opacity, "
        "density/darkness, length, and contact shadows. Preserve the target furniture "
        "form, materials, textures, and colors—do not recolor or alter geometry. Keep "
        "the target's canvas size, framing, and alignment unchanged. Avoid halos, glow, "
        "and added reflections. Return a single rendered image without overlays or text."
    )
    response = model.generate_content(
        [
            {"mime_type": target_mime, "data": target_bytes},
            {"mime_type": donor_mime, "data": donor_bytes},
            prompt,
        ],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        request_options={"timeout": 300},
    )
    try:
        return extract_image_from_response(response)
    except RuntimeError as exc:
        debug_summary = summarize_response_for_debug(response)
        prompt_feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"{exc} (response summary: {debug_summary}, prompt_feedback={prompt_feedback})"
        ) from exc


def iter_targets(target_root: Path) -> Iterable[Path]:
    for path in target_root.rglob("*"):
        if is_image_file(path):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply donor shadows to target images.")
    parser.add_argument("--targets", default="targets", help="Target images root directory.")
    parser.add_argument("--donors", default="donors", help="Donor images root directory.")
    parser.add_argument(
        "--output",
        default="output/shadowed",
        help="Directory to write processed images (will be created).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=int(os.getenv("MAX_SIDE", "4096")),
        help="Max long-edge pixels to send to Gemini (0 to disable).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is missing. Populate .env or the environment.")

    target_root = Path(args.targets)
    donor_root = Path(args.donors)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Indexing donors in {donor_root}...")
    donor_index = build_donor_index(donor_root)
    print(f"Indexed {len(donor_index)} donor views.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    for target_path in iter_targets(target_root):
        output_path = output_root / f"{target_path.stem}.png"
        if output_path.exists():
            print(f"Skipping existing output: {output_path.name}")
            continue

        product = extract_product_code(target_path)
        view = extract_view_code(target_path)
        if not product or not view:
            print(f"Could not parse product/view for target: {target_path}")
            continue

        donor_path = donor_index.get((product, view))
        if not donor_path:
            print(f"No donor found for {target_path.name} (product {product}, view {view})")
            continue

        print(f"Processing {target_path.name} with donor {donor_path.name}...")
        print(
            f"   Sending both images to Gemini as PNG (converted in-memory, "
            f"max_side={args.max_side or 'original'})."
        )
        try:
            image_bytes = request_shadowed_image(model, target_path, donor_path, args.max_side)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to process {target_path.name}: {exc}")
            continue

        output_path.write_bytes(image_bytes)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

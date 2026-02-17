# Shadow Factory

Shadow transfer pipeline using Gemini or Vertex AI. It matches recipient images with donor images by product code and view code, applies donor shadows to recipients, and writes generated outputs per product.

## Setup

1. Install dependencies
   - `pip install -r requirements.txt`
2. Create env files
   - Copy `.env.example` to `.env` and fill in real values.
   - Copy `vertex.json.example` to `vertex.json` if you are using Vertex AI.

## Prompts

PID means Prompt ID.

- Default prompt file: `prompts/mv_shd_donor_to_recipient_PID1.md`
- Shadow-only prompt example: `prompts/create_shadow_only_PID2.md` (returns a grayscale shadow mask). In shadow-only mode, the output PNG keeps recipient RGB but replaces its alpha with the shadow mask.
- Shadow-only masks can come from the model (`SHADOW_MASK_SOURCE="model"`) or from local extraction (`donor`/`recipient`).
- Mask polarity is controlled by `SHADOW_MASK_WHITE_IS_SHADOW` (set `false` when white = no shadow).
- Set `SHADOW_MASK_SIMPLE=true` to apply the mask with no cleanup/clip/align/removal.
- Recipient alpha can be cleared before submission with `CLEAR_RECIPIENT_ALPHA=true` (composited on `RECIPIENT_BG_COLOR`).
- Recipient low_res PNGs can have background alpha shadows removed with `CLEAR_RECIPIENT_BG_ALPHA=true` and `RECIPIENT_ALPHA_BG_THRESHOLD` (binarizes alpha).
- Shadow-only masks can be sourced from `SHADOW_MASK_SOURCE=model`, `recipient`, or `donor`.
- Shadow extraction normalization: `SHADOW_EXTRACT_NORMALIZE=true` with `SHADOW_EXTRACT_PERCENTILE`.
- Select a prompt with `PROMPT_ID` or `PROMPT_PATH` in `.env`, or pass `--pid` (defaults to PID1).
- Use `--force` to overwrite existing outputs/debug images.
- Use `--async N` to run up to N submissions concurrently, or set `MAX_ASYNC` in `.env`.
- Use `--clean` to delete generated artifacts (low_res, generated outputs, debug) and exit.
- Optional pad-before-send: set `PAD_BEFORE_SEND=true` to letterbox to the closest supported ratio before sending to Gemini, and crop the response back to the original content area. Configure with `PAD_MAX_SIDE`, `PAD_MULTIPLE`, and `PAD_COLOR`.

## Paths

- Recipients: `products/*/recipient` (prefers `high_res`, then `low_res`)
- Donors (shadow sources): `products/*/donor` (prefers `high_res`, then `low_res`)
- Outputs: `products/<product>/generated` by default (override with `--output`). Filenames include the prompt PID (e.g., `_PID1`). A copy is also written to `output/`.
- High-res TIFFs are converted to 4K PNGs in `low_res` before model submission (alpha preserved).
- Debug images: `debug/<stem>_PID#_debug2k.png` are generated (donor + result side-by-side at 2K).
- Raw model returns are saved as `debug/<stem>_PID#_raw.png`.
- Donor low_res alignment: by default donors are aligned to recipient framing using `ALIGN_DONOR_TO_RECIPIENT=true` and `DONOR_BG_THRESHOLD` (background detection threshold).
- Final generated outputs preserve recipient alpha by default. Set `OUTPUT_WHITE_BG=true` to flatten to white.

## Run

```bash
API_MAX_SIDE=4096 python3 shadow_transfer.py
```

## Notes

- Example files are sanitized: `.env.example` and `vertex.json.example` contain no real credentials.
- `vertex.json` is ignored by git.

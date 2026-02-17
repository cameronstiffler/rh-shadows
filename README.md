# Shadow Factory

Shadow transfer pipeline using Gemini or Vertex AI. It matches recipient images with donor images by product code and view code, requests a model-generated shadow mask, and applies the mask into the recipient alpha.

## Setup

1. Install dependencies
   - `pip install -r requirements.txt`
2. Create env files
   - Copy `.env.example` to `.env` and fill in real values.
   - Copy `vertex.json.example` to `vertex.json` if you are using Vertex AI.

## Prompts

PID means Prompt ID.

- Default prompt file: `prompts/mv_shd_donor_to_recipient_PID1.md`
- Shadow-only prompt example: `prompts/ask_for_shadows_PID4.md`
- Shadow-only mode keeps recipient RGB and **combines the returned grayscale shadow mask into the recipient alpha** (white = transparent, dark = shadow).
- Select a prompt with `PROMPT_ID` or `PROMPT_PATH` in `.env`, or pass `--pid` (defaults to PID1).
- If the prompt filename or body contains `shadow_only` / `shadow-only` / `shadow mask`, shadow-only mode is used. You can also force it with `SHADOW_ONLY=true`.

## Paths

- Recipients: `products/*/recipient` (prefers `high_res`, then `low_res`)
- Donors (shadow sources): `products/*/donor` (prefers `high_res`, then `low_res`)
- Outputs: `products/<product>/generated` by default (override with `--output`). Filenames include the prompt PID (e.g., `_PID4`).
- A copy is also written to `output/`, plus a `_preview.png` for quick review.
- High-res TIFFs are converted to 4K PNGs in `low_res` before model submission (alpha preserved).
- Debug images are written to `debug/` (raw model return, processed mask, preview, and 2K donor/result side-by-side).
- Donor low_res alignment: `ALIGN_DONOR_TO_RECIPIENT=true` aligns donors to the recipient framing using `DONOR_BG_THRESHOLD`.

## Run

```bash
API_MAX_SIDE=4096 python3 shadow_transfer.py
```

## Notes

- Example files are sanitized: `.env.example` and `vertex.json.example` contain no real credentials.
- `vertex.json` is ignored by git.

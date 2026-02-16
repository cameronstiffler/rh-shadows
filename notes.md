command for table
API_MAX_SIDE=4096 python3 shadow_transfer.py

Prompting
PID means Prompt ID.
Shadow transfer prompt lives at prompts/mv_shd_donor_to_recipient_PID1.md
Shadow-only prompt example: prompts/create_shadow_only_PID2.md
Shadow-only output keeps recipient RGB and replaces alpha with the shadow mask.
Original recipient alpha is saved to debug/<stem>_PID#_recipient_alpha.png in shadow-only mode.
Use PROMPT_ID or PROMPT_PATH in .env to switch prompts.
You can also pass --pid to select a prompt (defaults to PID1).
Use --force to overwrite existing outputs.
Use --async N to allow up to N concurrent submissions.
Or set MAX_ASYNC in .env for the default concurrency.
Use --clean to delete generated artifacts (low_res, generated outputs, debug).

Paths
Recipients can live under products/*/recipient (prefers high_res, then low_res).
Donors (shadow sources) can live under products/*/donor (prefers high_res, then low_res).
If you pass --recipients or --donors, those are still used when they contain images.
Default outputs go to products/<product>/generated (override with --output). Output names include the prompt PID. A copy is also written to output/.
High-res TIFFs are converted to 4K PNGs in low_res before model submission (alpha preserved).
Debug images are written to debug/ as 2K side-by-side donor + result PNGs (overwritten by filename).
Final generated outputs preserve recipient alpha by default. Set OUTPUT_WHITE_BG=true to flatten to white.
Raw model returns are saved as debug/<stem>_PID#_raw.png.
Donor low_res alignment: ALIGN_DONOR_TO_RECIPIENT=true aligns donors to recipient framing using DONOR_BG_THRESHOLD.
Shadow-only cleanup removes mask pixels that overlap the recipient alpha (controls: SHADOW_MASK_REMOVE_*).
Shadow-only alignment can scale/translate the shadow mask to the recipient bbox (controls: SHADOW_MASK_ALIGN and SHADOW_ALIGN_*).

Examples
Use .env.example and vertex.json.example as templates. Do not put real keys in the example files.

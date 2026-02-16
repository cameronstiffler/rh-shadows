Return a grayscale shadow mask only (no color). The FIRST image is the recipient; the SECOND image is the donor shadow reference.

Rules:
- Output must be a single-channel grayscale image the same size as the recipient.
- White (255) = no shadow; black (0) = full shadow. Use smooth gradients for soft edges.
- Transfer ONLY the donor's shadow shapes/softness onto the recipient geometry (aligned, no shifts).
- Do NOT include any donor background, floor, or objects.
- Do NOT alter recipient geometry, lighting, or colors. Output only the shadow mask.
- If donor has no shadow, return an all-white mask.

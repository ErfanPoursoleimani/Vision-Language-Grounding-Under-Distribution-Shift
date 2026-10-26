"""Label-preserving image degradations with a 5-level severity ladder (in the spirit of ImageNet-C, Hendrycks & Dietterich 2019).

Only operations that cannot remove or add objects live here, so existing existence labels stay valid. Operations that CAN
change the answer (crop, occlusion, object edits) belong to the counterfactual track, where the new gold answer is computed.
Every operation is deterministic given (image, op, severity, seed).
"""
from __future__ import annotations

import hashlib
import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

OPS: Tuple[str, ...] = ("blur", "noise", "jpeg", "contrast", "pixelate")
SEVERITIES: Tuple[int, ...] = (1, 2, 3, 4, 5)
BLUR_FRAC = (0.004, 0.008, 0.014, 0.022, 0.032)   # Gaussian sigma as a fraction of max(width, height)
NOISE_SIGMA = (8, 16, 28, 42, 60)                  # std in 0-255 units
JPEG_QUALITY = (60, 40, 25, 12, 6)
CONTRAST = (0.7, 0.5, 0.35, 0.22, 0.12)            # multiplier around the mean gray level
PIXELATE = (2, 3, 5, 8, 12)                        # downscale factor, then nearest-neighbour upscale


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def apply(img: Image.Image, op: str, severity: int, seed: int = 0) -> Image.Image:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; choose from {OPS}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}")
    img = img.convert("RGB")
    i = severity - 1
    w, h = img.size
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=BLUR_FRAC[i] * max(w, h)))
    if op == "noise":
        rng = np.random.default_rng(_seed(seed, op, severity, w, h))
        a = np.asarray(img, dtype=np.float32) + rng.normal(0.0, NOISE_SIGMA[i], (h, w, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if op == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY[i])
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "contrast":
        a = np.asarray(img, dtype=np.float32)
        m = a.mean()
        return Image.fromarray(np.clip(m + CONTRAST[i] * (a - m), 0, 255).astype(np.uint8))
    k = PIXELATE[i]  # pixelate
    small = img.resize((max(1, w // k), max(1, h // k)), Image.BOX)
    return small.resize((w, h), Image.NEAREST)

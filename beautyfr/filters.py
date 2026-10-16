"""OpenCV approximations of social-media beautification filters (DEVIATIONS D2).

Paper evaluates Instagram / Snapchat / TikTok filters on FFMF. We approximate
aggressivity tiers for FR impact + SSIM studies on LFW.
"""

from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np
from PIL import Image


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def filter_original(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def filter_uploaded(img: Image.Image) -> Image.Image:
    """Platform upload degradation: mild blur + JPEG recompression."""
    bgr = _pil_to_bgr(img)
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0.6)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    if not ok:
        return _bgr_to_pil(bgr)
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return _bgr_to_pil(dec)


def filter_thinner_face(img: Image.Image) -> Image.Image:
    """Instagram-like thinner face: horizontal squeeze about center."""
    bgr = _pil_to_bgr(img)
    h, w = bgr.shape[:2]
    cx = (w - 1) / 2.0
    strength = 0.12
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x = cx + (xs - cx) * (1.0 - strength)
    map_x = np.broadcast_to(map_x, (h, w)).copy()
    map_y = np.broadcast_to(ys[:, None], (h, w)).copy()
    warped = cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _bgr_to_pil(warped)


def filter_mellow_glow(img: Image.Image) -> Image.Image:
    """Snapchat-like soft glow: bilateral + lift shadows."""
    bgr = _pil_to_bgr(img)
    soft = cv2.bilateralFilter(bgr, d=9, sigmaColor=60, sigmaSpace=60)
    glow = cv2.addWeighted(soft, 0.75, bgr, 0.25, 12)
    return _bgr_to_pil(np.clip(glow, 0, 255).astype(np.uint8))


def filter_belle(img: Image.Image) -> Image.Image:
    """TikTok-like Belle: smooth skin + warm tint."""
    bgr = _pil_to_bgr(img)
    soft = cv2.bilateralFilter(bgr, d=11, sigmaColor=75, sigmaSpace=75)
    warm = soft.astype(np.float32)
    warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.08 + 8, 0, 255)  # R
    warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.03 + 4, 0, 255)  # G
    return _bgr_to_pil(warm.astype(np.uint8))


FILTERS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "original": filter_original,
    "uploaded": filter_uploaded,
    "thinner_face": filter_thinner_face,
    "mellow_glow": filter_mellow_glow,
    "belle": filter_belle,
}


def apply_filter(img: Image.Image, name: str) -> Image.Image:
    if name not in FILTERS:
        raise KeyError(f"unknown filter {name}; choose from {list(FILTERS)}")
    return FILTERS[name](img)


def ssim_gray(a: Image.Image, b: Image.Image) -> float:
    """Simple SSIM on grayscale (paper Table III style)."""
    ga = cv2.cvtColor(_pil_to_bgr(a), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(_pil_to_bgr(b), cv2.COLOR_BGR2GRAY).astype(np.float64)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = ga.mean(), gb.mean()
    sig_a = ga.var()
    sig_b = gb.var()
    sig_ab = ((ga - mu_a) * (gb - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(num / den)

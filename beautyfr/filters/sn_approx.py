"""OpenCV approximations of social-network beautification filters (deviation D2).

Paper SN filters (Instagram Thinner_face, Snapchat Mellow glow, TikTok Belle,
plus platform upload compression) are proprietary AR pipelines. We approximate
their dominant visual effects for a reproducible lab twin.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import cv2
import numpy as np

FILTER_NAMES: List[str] = [
    "original",
    "uploaded",
    "thinner_face",
    "mellow_glow",
    "belle",
]


def _as_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def filter_original(img: np.ndarray) -> np.ndarray:
    return _as_bgr_u8(img).copy()


def filter_uploaded(img: np.ndarray, quality: int = 55, blur_ksize: int = 3) -> np.ndarray:
    """JPEG recompression + slight blur (platform upload degradation)."""
    bgr = _as_bgr_u8(img)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return bgr
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if blur_ksize and blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        decoded = cv2.GaussianBlur(decoded, (k, k), 0.6)
    return decoded


def filter_thinner_face(img: np.ndarray, squeeze: float = 0.88) -> np.ndarray:
    """Horizontal mid-face squeeze warp (Instagram Thinner_face approx)."""
    bgr = _as_bgr_u8(img)
    h, w = bgr.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    cx = (w - 1) / 2.0
    for y in range(h):
        # Stronger squeeze near vertical mid-band (cheeks / jaw)
        vert = 1.0 - 0.35 * np.cos((y / max(h - 1, 1) - 0.5) * np.pi)
        local = squeeze + (1.0 - squeeze) * (1.0 - vert)
        for x in range(w):
            dx = x - cx
            map_x[y, x] = cx + dx / local
            map_y[y, x] = float(y)
    return cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def filter_mellow_glow(img: np.ndarray) -> np.ndarray:
    """Bilateral smooth + brightness lift (Snapchat Mellow glow approx)."""
    bgr = _as_bgr_u8(img)
    soft = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    soft = cv2.convertScaleAbs(soft, alpha=1.08, beta=12)
    return soft


def filter_belle(img: np.ndarray) -> np.ndarray:
    """Skin smooth + warm tint (TikTok Belle approx)."""
    bgr = _as_bgr_u8(img)
    soft = cv2.bilateralFilter(bgr, d=7, sigmaColor=60, sigmaSpace=60)
    # Warm tint: boost R/G slightly, reduce B
    f = soft.astype(np.float32)
    f[:, :, 2] = np.clip(f[:, :, 2] * 1.06 + 8, 0, 255)  # R
    f[:, :, 1] = np.clip(f[:, :, 1] * 1.03 + 4, 0, 255)  # G
    f[:, :, 0] = np.clip(f[:, :, 0] * 0.96, 0, 255)  # B
    return f.astype(np.uint8)


_FILTER_FN: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "original": filter_original,
    "uploaded": filter_uploaded,
    "thinner_face": filter_thinner_face,
    "mellow_glow": filter_mellow_glow,
    "belle": filter_belle,
}


def apply_filter(img: np.ndarray, name: str) -> np.ndarray:
    key = name.lower().strip()
    if key not in _FILTER_FN:
        raise ValueError(f"Unknown filter '{name}'. Choose from {FILTER_NAMES}")
    return _FILTER_FN[key](img)

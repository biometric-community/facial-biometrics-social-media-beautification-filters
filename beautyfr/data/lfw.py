"""LFW pairs protocol (+ optional CALFW pairs.csv)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from beautyfr.filters.sn_approx import apply_filter


def resolve_lfw_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["lfw_root"])
    if not root.is_absolute():
        proj = Path(__file__).resolve().parents[2]
        root = (proj / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"LFW root not found: {root}")
    return root


def resolve_calfw_root(cfg: dict) -> Optional[Path]:
    raw = cfg["data"].get("calfw_root")
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute():
        proj = Path(__file__).resolve().parents[2]
        root = (proj / root).resolve()
    return root if root.exists() else None


def _lfw_image_dir(root: Path) -> Path:
    for cand in (root / "lfw", root / "lfw_funneled", root / "images", root):
        if cand.is_dir() and any(cand.iterdir()):
            # Prefer directory that contains person folders
            sub = next(cand.iterdir(), None)
            if sub is not None and sub.is_dir():
                return cand
    return root / "lfw"


def _lfw_path(img_dir: Path, name: str, idx: int) -> Path:
    return img_dir / name / f"{name}_{idx:04d}.jpg"


@dataclass
class PairItem:
    path_a: Path
    path_b: Path
    same: bool
    fold: int = 0


def load_lfw_pairs(root: Path, max_pairs: Optional[int] = None) -> List[PairItem]:
    pairs_path = root / "pairs.txt"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs.txt under {root}")
    img_dir = _lfw_image_dir(root)
    lines = pairs_path.read_text().strip().splitlines()
    header = lines[0].split()
    n_folds, n_per = int(header[0]), int(header[1])
    pairs: List[PairItem] = []
    i = 1
    for fold in range(n_folds):
        for _ in range(n_per):
            parts = lines[i].split()
            i += 1
            name, a, b = parts[0], int(parts[1]), int(parts[2])
            pairs.append(
                PairItem(_lfw_path(img_dir, name, a), _lfw_path(img_dir, name, b), True, fold)
            )
        for _ in range(n_per):
            parts = lines[i].split()
            i += 1
            n1, a, n2, b = parts[0], int(parts[1]), parts[2], int(parts[3])
            pairs.append(
                PairItem(_lfw_path(img_dir, n1, a), _lfw_path(img_dir, n2, b), False, fold)
            )
    if max_pairs is not None:
        # Fold-blocked file: prefix would be all-same; keep balanced same/diff.
        same = [p for p in pairs if p.same]
        diff = [p for p in pairs if not p.same]
        n_each = max(1, int(max_pairs) // 2)
        pairs = same[:n_each] + diff[:n_each]
    return pairs


def load_calfw_pairs(root: Path, max_pairs: Optional[int] = None) -> List[PairItem]:
    """Load biometric-community CALFW pairs.csv (aligned images)."""
    # Prefer nested calfw-hf layout
    candidates = [
        root / "calfw-hf" / "pairs.csv",
        root / "pairs.csv",
        root / "extracted" / "calfw-hf" / "pairs.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"CALFW pairs.csv not found under {root}")
    base = csv_path.parent
    aligned = base / "aligned"
    pairs: List[PairItem] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pa = aligned / row["image_a_aligned_path"]
            pb = aligned / row["image_b_aligned_path"]
            same = str(row["is_same"]).strip() in ("1", "True", "true")
            fold = int(row.get("fold_id", 0))
            pairs.append(PairItem(pa, pb, same, fold))
    if max_pairs is not None:
        same = [p for p in pairs if p.same]
        diff = [p for p in pairs if not p.same]
        n_each = max(1, int(max_pairs) // 2)
        pairs = same[:n_each] + diff[:n_each]
    return pairs


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def bgr_to_tensor(bgr: np.ndarray, transform: transforms.Compose):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(rgb))


class LFWPairsDataset(Dataset):
    """Yield (tensor_a, tensor_b, label) with optional SN-approx filter on both sides."""

    def __init__(
        self,
        pairs: Sequence[PairItem],
        filter_name: str = "original",
        image_size: int = 112,
    ):
        self.pairs = list(pairs)
        self.filter_name = filter_name
        self.transform = _eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        a = apply_filter(load_bgr(item.path_a), self.filter_name)
        b = apply_filter(load_bgr(item.path_b), self.filter_name)
        ta = bgr_to_tensor(a, self.transform)
        tb = bgr_to_tensor(b, self.transform)
        label = 1 if item.same else 0
        return ta, tb, label

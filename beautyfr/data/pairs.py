"""LFW verification pairs with optional beautification on both images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from beautyfr.filters import apply_filter


def _resolve(cfg: dict, key: str, default: str) -> Path:
    root = Path(cfg["data"].get(key, default))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    return root


def resolve_lfw(cfg: dict) -> Path:
    root = _resolve(cfg, "lfw_root", "../../datasets/lfw")
    if not (root / "lfw").is_dir() or not (root / "pairs.txt").is_file():
        raise FileNotFoundError(root)
    return root


def _tf(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _subsample_pairs(
    pairs: List[Tuple[str, str, int]], max_pairs: Optional[int]
) -> List[Tuple[str, str, int]]:
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]
    n_each = max_pairs // 2
    out = pos[:n_each] + neg[:n_each]
    if len(out) < max_pairs:
        rest = [p for p in pairs if p not in out]
        out.extend(rest[: max_pairs - len(out)])
    return out[:max_pairs]


def _parse_lfw(pairs_path: Path, max_pairs: Optional[int] = None):
    out = []
    for line in pairs_path.read_text().strip().splitlines()[1:]:
        toks = line.split()
        if len(toks) == 3:
            name, n1, n2 = toks
            a = f"{name}/{name}_{int(n1):04d}.jpg"
            b = f"{name}/{name}_{int(n2):04d}.jpg"
            out.append((a, b, 1))
        elif len(toks) == 4:
            n1, i1, n2, i2 = toks
            a = f"{n1}/{n1}_{int(i1):04d}.jpg"
            b = f"{n2}/{n2}_{int(i2):04d}.jpg"
            out.append((a, b, 0))
    return _subsample_pairs(out, max_pairs)


class FilteredPairDataset(Dataset):
    def __init__(
        self,
        img_root: Path,
        pairs,
        image_size: int = 112,
        filter_name: str = "original",
    ):
        self.img_root = Path(img_root)
        self.pairs = pairs
        self.transform = _tf(image_size)
        self.filter_name = filter_name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        a, b, label = self.pairs[index]
        ia = apply_filter(Image.open(self.img_root / a).convert("RGB"), self.filter_name)
        ib = apply_filter(Image.open(self.img_root / b).convert("RGB"), self.filter_name)
        return self.transform(ia), self.transform(ib), label


def build_lfw(cfg, image_size, max_pairs, filter_name: str = "original"):
    root = resolve_lfw(cfg)
    return FilteredPairDataset(
        root / "lfw",
        _parse_lfw(root / "pairs.txt", max_pairs),
        image_size,
        filter_name=filter_name,
    )

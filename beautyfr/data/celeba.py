"""CelebA identity classification dataset for embedding training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_celeba_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["celeba_root"])
    if not root.is_absolute():
        # configs live under project root; resolve relative to project
        proj = Path(__file__).resolve().parents[2]
        root = (proj / root).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"CelebA root not found: {root}. Place identity_CelebA.txt and "
            "img_align_celeba/ under projects/datasets/celeba/extracted/celeba/"
        )
    return root


def _default_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class CelebAIdentityDataset(Dataset):
    """Load CelebA aligned faces labeled by identity_CelebA.txt."""

    def __init__(
        self,
        root: Path,
        image_size: int = 112,
        train: bool = True,
        max_n: Optional[int] = None,
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        identity_path = self.root / "identity_CelebA.txt"
        img_dir = self.root / "img_align_celeba"
        if not identity_path.exists():
            raise FileNotFoundError(f"Missing {identity_path}")
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing {img_dir}")

        pairs: List[Tuple[Path, int]] = []
        raw_ids: List[int] = []
        with open(identity_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name, iid = parts[0], int(parts[1])
                p = img_dir / name
                if p.exists():
                    pairs.append((p, iid))
                    raw_ids.append(iid)

        # Optional train/val split via list_eval_partition.txt (0=train,1=val,2=test)
        part_path = self.root / "list_eval_partition.txt"
        if part_path.exists():
            part = {}
            with open(part_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        part[parts[0]] = int(parts[1])
            want = {0} if train else {1, 2}
            pairs = [(p, i) for p, i in pairs if part.get(p.name, 0) in want]

        # Contiguous label map over this split's identities
        uniq = sorted(set(i for _, i in pairs))
        id_map = {old: new for new, old in enumerate(uniq)}
        self.samples = [(p, id_map[i]) for p, i in pairs]
        self.num_classes = len(uniq)

        if max_n is not None and max_n < len(self.samples):
            import random

            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, int(max_n))
            # Remap classes after subsample (keep contiguous for head size stability:
            # keep full num_classes from full split for checkpoint consistency when
            # max_n is used — rebuild from subsampled labels)
            labels = sorted(set(i for _, i in self.samples))
            remap = {old: new for new, old in enumerate(labels)}
            self.samples = [(p, remap[i]) for p, i in self.samples]
            self.num_classes = len(labels)

        self.transform = transform or _default_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

"""Train ResNet18 + CosFace/ArcFace embedder on CelebA identities."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from beautyfr.data.celeba import CelebAIdentityDataset, resolve_celeba_root
from beautyfr.models.embedder import FaceEmbedder
from beautyfr.models.losses import MarginHead


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> dict:
    device = _device(cfg, device_override)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    emb_dim = int(cfg["model"].get("emb_dim", 512))
    loss_name = str(cfg["model"].get("margin_loss", "cosface"))

    ds = CelebAIdentityDataset(
        root,
        image_size=image_size,
        train=True,
        max_n=max_n,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 128)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = FaceEmbedder(emb_dim=emb_dim, pretrained=bool(cfg["model"].get("pretrained", True))).to(device)
    head = MarginHead(
        loss_name,
        emb_dim,
        ds.num_classes,
        s=float(cfg["model"].get("scale", 64.0)),
        m=cfg["model"].get("margin"),
    ).to(device)

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(
        params,
        lr=float(cfg["train"].get("lr", 0.1)),
        momentum=0.9,
        weight_decay=5e-4,
    )
    epochs = int(cfg["train"].get("epochs", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    hist = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        losses = []
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"train ep{epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            emb = model(images)
            logits = head(emb, labels)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += labels.numel()
            pbar.set_postfix(loss=f"{losses[-1]:.3f}", acc=f"{correct / max(total, 1):.3f}")
        scheduler.step()
        mean_loss = float(np.mean(losses)) if losses else 0.0
        acc = correct / max(total, 1)
        hist.append({"epoch": epoch, "loss": mean_loss, "acc": acc, "lr": float(scheduler.get_last_lr()[0])})
        print(f"[train] epoch={epoch} loss={mean_loss:.4f} acc={acc:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "embedder.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "num_classes": ds.num_classes,
            "emb_dim": emb_dim,
            "margin_loss": loss_name,
            "image_size": image_size,
            "config": cfg,
        },
        ckpt_path,
    )

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "protocol": cfg["train"].get("protocol", "full"),
        "n_train": len(ds),
        "num_classes": ds.num_classes,
        "emb_dim": emb_dim,
        "margin_loss": loss_name,
        "epochs": epochs,
        "elapsed_sec": time.time() - t0,
        "history": hist,
        "checkpoint": str(ckpt_path),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[train] done n={len(ds)} classes={ds.num_classes} ckpt={ckpt_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Train face embedder on CelebA")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()

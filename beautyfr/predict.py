"""Embed a single image (optional filter) and print L2-normalized features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from beautyfr.data.lfw import bgr_to_tensor, load_bgr
from beautyfr.filters.sn_approx import apply_filter
from beautyfr.models.embedder import FaceEmbedder
from torchvision import transforms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Predict face embedding")
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True, help="Path to face image")
    ap.add_argument("--filter", default="original", help="SN-approx filter name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="Optional JSON path for embedding")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device_name = args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "embedder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FaceEmbedder(emb_dim=int(ckpt.get("emb_dim", 512)), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    image_size = int(ckpt.get("image_size", cfg["model"].get("image_size", 112)))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    bgr = apply_filter(load_bgr(Path(args.image)), args.filter)
    tensor = bgr_to_tensor(bgr, transform).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy()[0]

    payload = {
        "image": args.image,
        "filter": args.filter,
        "emb_dim": int(emb.shape[0]),
        "embedding": emb.tolist(),
    }
    out = args.out or str(Path(cfg["paths"]["log_dir"]) / "predict.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[predict] wrote {out} dim={emb.shape[0]} norm={float((emb**2).sum()**0.5):.4f}")


if __name__ == "__main__":
    main()

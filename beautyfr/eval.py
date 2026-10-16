"""LFW (optional CALFW) verification under SN-approx beautification filters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from beautyfr.data.lfw import (
    LFWPairsDataset,
    load_bgr,
    load_calfw_pairs,
    load_lfw_pairs,
    resolve_calfw_root,
    resolve_lfw_root,
)
from beautyfr.filters.sn_approx import FILTER_NAMES, apply_filter
from beautyfr.metrics import cosine_scores, ssim_gray, summarize_verification
from beautyfr.models.embedder import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_embedder(ckpt_path: Path, device: torch.device) -> FaceEmbedder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    emb_dim = int(ckpt.get("emb_dim", 512))
    model = FaceEmbedder(emb_dim=emb_dim, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed_pairs(model: FaceEmbedder, loader: DataLoader, device: torch.device):
    emb_a, emb_b, labels = [], [], []
    for ta, tb, y in tqdm(loader, desc="embed", leave=False):
        ea = model(ta.to(device)).cpu().numpy()
        eb = model(tb.to(device)).cpu().numpy()
        emb_a.append(ea)
        emb_b.append(eb)
        labels.append(y.numpy())
    return (
        np.concatenate(emb_a),
        np.concatenate(emb_b),
        np.concatenate(labels).astype(np.int32),
    )


def mean_ssim_for_filter(pairs, filter_name: str, max_images: int = 200) -> float:
    if filter_name == "original":
        return 1.0
    vals = []
    seen = set()
    for item in pairs:
        for p in (item.path_a, item.path_b):
            if p in seen:
                continue
            seen.add(p)
            if not p.exists():
                continue
            orig = load_bgr(p)
            filt = apply_filter(orig, filter_name)
            vals.append(ssim_gray(orig, filt))
            if len(vals) >= max_images:
                return float(np.mean(vals))
    return float(np.mean(vals)) if vals else 0.0


def eval_benchmark(
    model: FaceEmbedder,
    pairs,
    filters: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    ssim_max: int,
) -> Dict:
    out = {"filters": {}}
    for fname in filters:
        ds = LFWPairsDataset(pairs, filter_name=fname, image_size=image_size)
        # Drop missing images
        valid = [p for p in ds.pairs if p.path_a.exists() and p.path_b.exists()]
        if not valid:
            raise FileNotFoundError(f"No valid pairs for filter={fname}")
        ds.pairs = valid
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        ea, eb, labels = embed_pairs(model, loader, device)
        scores = cosine_scores(ea, eb)
        summary = summarize_verification(scores, labels)
        summary["mean_ssim"] = mean_ssim_for_filter(valid, fname, max_images=ssim_max)
        summary["n_valid_pairs"] = len(valid)
        out["filters"][fname] = summary
        print(
            f"[eval:{fname}] acc={summary['accuracy']:.4f} auc={summary['auc']:.4f} "
            f"tpr@1e-2={summary['tpr_at_fpr_0.01']:.4f} ssim={summary['mean_ssim']:.4f}"
        )
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "embedder.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run train first.")
    model = load_embedder(ckpt, device)

    filters = list(cfg["eval"].get("filters", FILTER_NAMES))
    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 64))
    nw = int(cfg["train"].get("num_workers", 4))
    max_pairs = cfg["eval"].get("max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    ssim_max = int(cfg["eval"].get("ssim_max_images", 200))

    lfw_root = resolve_lfw_root(cfg)
    lfw_pairs = load_lfw_pairs(lfw_root, max_pairs=max_pairs)
    results = {
        "protocol": cfg["train"].get("protocol", "full"),
        "checkpoint": str(ckpt),
        "lfw": eval_benchmark(model, lfw_pairs, filters, image_size, bs, nw, device, ssim_max),
    }

    if cfg["eval"].get("use_calfw", False):
        calfw = resolve_calfw_root(cfg)
        if calfw is None:
            results["calfw"] = {"skipped": True, "reason": "calfw_root missing"}
        else:
            try:
                calfw_pairs = load_calfw_pairs(calfw, max_pairs=max_pairs)
                results["calfw"] = eval_benchmark(
                    model, calfw_pairs, filters, image_size, bs, nw, device, ssim_max
                )
            except Exception as e:
                results["calfw"] = {"skipped": True, "reason": str(e)}

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"[eval] wrote {log_dir / 'eval.json'}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate beautification impact on LFW")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()

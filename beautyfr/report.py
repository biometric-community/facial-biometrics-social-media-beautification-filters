"""REPORT.md + SVG figures for beautification FR impact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from beautyfr.plot_style import BAR_COLOR, FIGSIZE, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_path = log_dir / "eval.json"
    if not eval_path.is_file():
        # backward-compatible name
        alt = log_dir / "eval_results.json"
        eval_path = alt if alt.is_file() else eval_path
    train_path = log_dir / "train_summary.json"
    if not eval_path.is_file():
        Path("REPORT.md").write_text("# REPORT\n\n**BLOCKED:** run eval first.\n")
        print("BLOCKED: missing eval.json")
        return

    payload = json.loads(eval_path.read_text())
    train = json.loads(train_path.read_text()) if train_path.is_file() else {}
    # Support both nested (lfw.filters) and flat (filters) layouts
    if "lfw" in payload and isinstance(payload["lfw"], dict) and "filters" in payload["lfw"]:
        filt_map = payload["lfw"]["filters"]
    else:
        filt_map = payload.get("filters", {})

    filters = list(filt_map.keys())
    if not filters:
        Path("REPORT.md").write_text("# REPORT\n\n**BLOCKED:** eval.json has no filter metrics.\n")
        return

    accs = [filt_map[f]["accuracy"] * 100 for f in filters]
    ssim = [filt_map[f].get("mean_ssim", filt_map[f].get("ssim", 0.0)) for f in filters]

    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(filters, accs, color=BAR_COLOR, edgecolor="#002080")
    ax.set_ylabel("Best-threshold accuracy (%)")
    ax.set_title("LFW verification under beautification (ours)")
    ax.tick_params(axis="x", rotation=20)
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    acc_svg = fig_dir / "lfw_accuracy.svg"
    fig.savefig(acc_svg, format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(filters, ssim, color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_ylabel("Mean SSIM vs original")
    ax.set_title("Filter aggressivity via SSIM (ours)")
    ax.tick_params(axis="x", rotation=20)
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    ssim_svg = fig_dir / "filter_ssim.svg"
    fig.savefig(ssim_svg, format="svg")
    plt.close(fig)

    def row(f: str) -> str:
        m = filt_map[f]
        tar = m.get("tpr_at_fpr_0.01", m.get("tar_at_far_1e-2", float("nan")))
        ss = m.get("mean_ssim", m.get("ssim", float("nan")))
        return f"| {f} | {m['accuracy']*100:.2f} | {tar*100:.2f} | {ss:.3f} |"

    rows = "\n".join(row(f) for f in filters)
    report = f"""# REPORT — Beautification filters vs face verification

Lab twin of Mirabet-Herranz et al., *Facial Biometrics in the Social Media Era* (TBIOM 2025), FR strand on LFW with approximated SN filters (FFMF unavailable — see DEVIATIONS).

## Train

- CelebA CosFace: n={train.get('n_train')}, classes={train.get('num_classes')}, epochs={train.get('epochs')}, loss={train.get('margin_loss')}

## LFW results (ours)

| Filter | Acc % | TPR@FPR=1e-2 % | SSIM |
|--------|------:|---------------:|-----:|
{rows}

## Figures

![LFW accuracy]({acc_svg.as_posix()})

![SSIM]({ssim_svg.as_posix()})

## Regenerate

```bash
bash scripts/report.sh
```
"""
    Path("REPORT.md").write_text(report)
    print(f"Wrote REPORT.md + {fig_dir}")


if __name__ == "__main__":
    main()

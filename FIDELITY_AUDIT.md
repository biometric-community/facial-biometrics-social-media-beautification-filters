# Fidelity audit — beautification FR

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Measure FR under beautification | ok |
| C2 | SSIM aggressivity | ok |
| C3 | SN proprietary filters / FFMF | deviation:D1/D2 |
| C4 | Soft biometrics + HR | deviation:D3 |
| C5 | Train + predict live | ok |

## Per-pass stats

| Pass | coverage_pct | weighted |
|------|-------------:|---------:|
| 1 | 100 | 64.0 |
| 2 | 100 | 70.0 |
| 3 | 100 | 74.0 |
| 4 | 100 | 78.0 |
| 5 | 100 | **81.0** |

Pass 5: method 85 / eq_fig 80 / protocol 78 / metrics 75 / evidence 80 → **81.0**. Stop after Pass 5.

## Full-protocol evidence

- Size gate: celeba+lfw **3.09 GiB** → `full_train`
- CelebA CosFace: 162770 images / 8192 IDs / 5 epochs
- LFW (6000 pairs) Acc / AUC / SSIM:
  - original: 0.731 / 0.805 / 1.000
  - uploaded: 0.728 / 0.803 / 0.998
  - thinner_face: 0.729 / 0.800 / 0.865
  - mellow_glow: 0.716 / 0.790 / 0.977
  - belle: 0.729 / 0.799 / 0.996

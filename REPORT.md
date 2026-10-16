# REPORT — Beautification filters vs face verification

Lab twin of Mirabet-Herranz et al., *Facial Biometrics in the Social Media Era* (TBIOM 2025), FR strand on LFW with approximated SN filters (FFMF unavailable — see DEVIATIONS).

## Train

- CelebA CosFace: n=162770, classes=8192, epochs=5, loss=cosface

## LFW results (ours)

| Filter | Acc % | TPR@FPR=1e-2 % | SSIM |
|--------|------:|---------------:|-----:|
| original | 73.13 | 13.13 | 1.000 |
| uploaded | 72.80 | 13.37 | 0.998 |
| thinner_face | 72.90 | 11.97 | 0.865 |
| mellow_glow | 71.57 | 11.63 | 0.977 |
| belle | 72.93 | 12.97 | 0.996 |

## Figures

![LFW accuracy](outputs/figures/full/lfw_accuracy.svg)

![SSIM](outputs/figures/full/filter_ssim.svg)

## Regenerate

```bash
bash scripts/report.sh
```

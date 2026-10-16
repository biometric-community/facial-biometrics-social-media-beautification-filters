# Implementation plan — Beautification filters vs FR

Paper: Mirabet-Herranz et al., TBIOM 2025 (DOI 10.1109/TBIOM.2024.3438928).

## Scope

Reproduce the **face verification** strand: measure how beautification affects FR accuracy and report filter aggressivity via SSIM. Soft-biometrics / HR / weight estimation are out of scope (D3).

## Figure inventory

| Paper | Ours |
|-------|------|
| Table III SSIM | `filter_ssim.svg` |
| Table V / DET FR | `lfw_accuracy.svg` (accuracy bars) |

## Data size gate

Required: `celeba` + `lfw` (~3.5 GiB) → full_train.

## GitHub

`facial-biometrics-in-the-social-media-era-an-in-depth-analysis-of-the-challenge-posed-by-beautification-filters`

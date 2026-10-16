# Facial Biometrics in the Social Media Era (Beautification Filters)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Mirabet-Herranz et al. (IEEE TBIOM 2025): impact of social-media beautification on face verification, with SSIM filter aggressivity.

**Repository:** https://github.com/biometric-community/facial-biometrics-in-the-social-media-era-an-in-depth-analysis-of-the-challenge-posed-by-beautification-filters

## Setup

```bash
bash scripts/setup_env.sh
```

## Data

- Train: `projects/datasets/celeba`
- Eval: `projects/datasets/lfw`
- FFMF author set unavailable — see `DEVIATIONS.md`

```bash
bash scripts/check_dataset_size.sh ../../datasets/celeba ../../datasets/lfw
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
bash scripts/predict.sh --image /path/to.jpg --filter mellow_glow
```

## Citation

Mirabet-Herranz, Galdi, Dugelay — *Facial Biometrics in the Social Media Era*, IEEE TBIOM 2025.

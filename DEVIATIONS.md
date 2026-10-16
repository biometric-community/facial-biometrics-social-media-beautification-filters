# Deviations

| ID | Topic | Paper | Ours |
|----|-------|-------|------|
| D1 | Primary eval set | FFMF-extended (author SN-captured filters) | LFW pairs (real public) |
| D2 | Filters | Instagram / Snapchat / TikTok proprietary filters | OpenCV approximations (`uploaded`, `thinner_face`, `mellow_glow`, `belle`) |
| D3 | Soft / hidden biometrics | Gender, age, weight, HR | Not reproduced — FR + SSIM only |
| D4 | Backbone | Commercial / SOTA FR nets in paper | CelebA-trained ResNet18 CosFace |

No fabricated metrics; REPORT from real `eval_results.json` only.

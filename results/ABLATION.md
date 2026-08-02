# Is the second learned stratum worth carrying?

Identical corpus, attacks and operating level. The only difference is whether the coarse message grid is carried alongside the fine one.

The two configurations are matched on *total residual energy*, not on per-stratum amplitude: two roughly independent residuals add in quadrature, so a pair at 0.032 each carries the same energy as one at 0.045. The embedding PSNR and SSIM below verify that the fidelity budget is comparable before the attack rates are interpreted.

| | one learned stratum | two learned strata |
|---|---:|---:|
| Embedding PSNR | 31.7 | 31.6 |
| Embedding SSIM | 0.773 | 0.807 |
| Admissible attacks fully survived | 41/44 | 45/48 |
| Mean TPR (admissible) | 96.0% | 96.5% |
| Worst TPR (admissible) | 2% | 2% |
| False positives (unmarked) | 0.00% | 0.00% |

## What the second stratum buys

| Attack | one stratum | two strata | change |
|---|---:|---:|---:|
| `jpeg_q20` | 60% | 100% | +40 pts |
| `double_jpeg_70_40` | 89% | 100% | +11 pts |
| `rotate_15` | 21% | 32% | +11 pts |
| `jpeg_q30` | 91% | 100% | +9 pts |
| `rsid_simple_q30` | 91% | 100% | +9 pts |

## What it costs

_No attack got worse. The second stratum is free in robustness terms; it is paid for in fidelity budget and detection time._

## Reading this

The second stratum improves recovery over this suite without reducing TPR. It still consumes part of the fidelity and detection-time budgets described above.

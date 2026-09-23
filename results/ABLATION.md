# Is the second learned stratum worth carrying?

Identical corpus, attacks and operating level. The only difference is whether the coarse message grid is carried alongside the fine one.

The two configurations are matched on *total residual energy*, not on per-stratum amplitude: two roughly independent residuals add in quadrature, so a pair at 0.032 each carries the same energy as one at 0.045. The embedding PSNR and SSIM below verify that the fidelity budget is comparable before the attack rates are interpreted.

| | one learned stratum | two learned strata |
|---|---:|---:|
| Embedding PSNR | 31.7 | 40.6 |
| Embedding SSIM | 0.773 | 0.980 |
| Admissible attacks fully survived | 41/44 | 2/4 |
| Mean TPR (admissible) | 96.0% | 50.0% |
| Worst TPR (admissible) | 2% | 0% |
| False positives (unmarked) | 0.00% | 0.00% |

## What the second stratum buys

_No attack improved by more than the reporting threshold._

## What it costs

| Attack | one stratum | two strata | change |
|---|---:|---:|---:|
| `crop_0.75` | 100% | 0% | -100 pts |
| `rotate_5` | 100% | 0% | -100 pts |

## Reading this

The second stratum does not pay for itself on this suite. It should be cut unless a deployment specifically expects the geometry it covers.

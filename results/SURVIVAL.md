# Attack survival

Corpus: 1 images. Operating point $\alpha = 1e-06$, identical for every system and every condition — no per-attack tuning.

Quality budget: an attack counts as *admissible* only if it leaves PSNR ≥ 24 dB and SSIM ≥ 0.70. Geometric and global photometric edits are admissible by construction, since a pixel metric cannot describe them.

## Headline

| System | PSNR | SSIM | Clean | FPR | Mean TPR (admissible) | Worst |
|---|---:|---:|---:|---:|---:|---:|
| SIGIL | 40.6 | 0.980 | 100% | 0.0% | 33.3% | 0% |
| SynthID-style | 36.6 | 0.968 | 100% | 0.0% | 33.3% | 0% |

## Every attack

Ordered by how well SIGIL survives, worst first.

| Attack | Family | PSNR | Adm. | SIGIL | SynthID-style |
|---|---|---:|:---:|---:|---:|
| `crop_0.75` | geometric | — | ✓ | **0%** ✗ | **0%** ✗ |
| `rotate_5` | geometric | — | ✓ | **0%** ✗ | **0%** ✗ |
| `clean` | none | — | ✓ | **100%** | **100%** |
| `jpeg_q50` | valuemetric | 29.8 | ✓ | **100%** | **100%** |

## False positives

| Condition | SIGIL | SynthID-style |
|---|---:|---:|
| Unmarked image | 0.0% | 0.0% |
| Marked under a different key | 0.0% | — |

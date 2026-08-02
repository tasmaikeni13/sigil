# Attack survival

Corpus: 80 images. Operating point $\alpha = 1e-06$, identical for every system and every condition — no per-attack tuning.

Quality budget: an attack counts as *admissible* only if it leaves PSNR ≥ 24 dB and SSIM ≥ 0.70. Geometric and global photometric edits are admissible by construction, since a pixel metric cannot describe them.

## Headline

| System | PSNR | SSIM | Clean | FPR | Mean TPR (admissible) | Worst |
|---|---:|---:|---:|---:|---:|---:|
| SIGIL | 31.6 | 0.807 | 100% | 0.0% | 96.5% | 2% |
| SynthID-style | 31.3 | 0.812 | 100% | 0.0% | 66.3% | 0% |
| StableSig-style | 32.0 | 0.785 | 100% | 0.0% | 95.1% | 2% |

## Every attack

Ordered by how well SIGIL survives, worst first.

| Attack | Family | PSNR | Adm. | SIGIL | SynthID-style | StableSig-style |
|---|---|---:|:---:|---:|---:|---:|
| `img2img_s0.5` | generative | 19.7 | — | **0%** ✗ | **0%** ✗ | **0%** ✗ |
| `rotate_30` | geometric | — | ✓ | **2%** ✗ | **0%** ✗ | **2%** ✗ |
| `rotate_15` | geometric | — | ✓ | **32%** ✗ | **0%** ✗ | **20%** ✗ |
| `stack_img2img_crop_jpeg` | composite | — | — | _60%_ | **0%** ✗ | **0%** ✗ |
| `purify_r2` | generative | 19.4 | — | _62%_ | **0%** ✗ | **0%** ✗ |
| `img2img_s0.3` | generative | 22.5 | — | _81%_ | **0%** ✗ | **21%** ✗ |
| `rsid_v2_maximum` | reverse synthid | 24.2 | — | 94% | **0%** ✗ | **0%** ✗ |
| `img2img_s0.15` | generative | 24.3 | — | 99% | **0%** ✗ | _65%_ |
| `rsid_v2_aggressive` | reverse synthid | 26.2 | — | 99% | **2%** ✗ | **2%** ✗ |
| `translate_7_11` | geometric | — | ✓ | 99% | **100%** | 95% |
| `vae_sdxl_n0.2` | generative | 23.1 | — | 99% | **0%** ✗ | _51%_ |
| `aspect_1.15` | geometric | 45.1 | ✓ | **100%** | **0%** ✗ | **100%** |
| `blur_1.0` | valuemetric | 31.9 | ✓ | **100%** | 99% | **100%** |
| `blur_2.0` | valuemetric | 27.6 | ✓ | **100%** | _76%_ | **100%** |
| `clean` | none | — | ✓ | **100%** | **100%** | **100%** |
| `codebook_sub_r5` | codebook | 37.6 | ✓ | **100%** | 98% | **100%** |
| `codebook_whiten_16k` | codebook | 45.1 | ✓ | **100%** | 98% | **100%** |
| `codebook_whiten_64k` | codebook | 38.3 | ✓ | **100%** | 98% | **100%** |
| `collusion_0.3` | codebook | 21.7 | — | **100%** | 99% | **100%** |
| `contrast_1.3` | valuemetric | 22.9 | ✓ | **100%** | 96% | **100%** |
| `crop_0.5` | geometric | — | ✓ | **100%** | **0%** ✗ | **100%** |
| `crop_0.75` | geometric | — | ✓ | **100%** | **0%** ✗ | **100%** |
| `crop_0.9` | geometric | — | ✓ | **100%** | **0%** ✗ | **100%** |
| `desaturate_0.5` | valuemetric | 29.2 | ✓ | **100%** | **100%** | **100%** |
| `double_jpeg_70_40` | valuemetric | 29.4 | ✓ | **100%** | _82%_ | 99% |
| `gamma_0.7` | valuemetric | 20.8 | ✓ | **100%** | **100%** | **100%** |
| `gamma_1.4` | valuemetric | 20.7 | ✓ | **100%** | **100%** | **100%** |
| `hflip` | geometric | — | ✓ | **100%** | **34%** ✗ | **100%** |
| `jpeg_q20` | valuemetric | 28.7 | ✓ | **100%** | _60%_ | 96% |
| `jpeg_q30` | valuemetric | 29.4 | ✓ | **100%** | _76%_ | **100%** |
| `jpeg_q50` | valuemetric | 30.4 | ✓ | **100%** | 91% | **100%** |
| `jpeg_q75` | valuemetric | 31.7 | ✓ | **100%** | 95% | **100%** |
| `jpeg_q90` | valuemetric | 34.5 | ✓ | **100%** | 98% | **100%** |
| `median_3` | valuemetric | 33.4 | ✓ | **100%** | 98% | **100%** |
| `median_5` | valuemetric | 29.3 | ✓ | **100%** | _79%_ | **100%** |
| `noise_s15` | valuemetric | 24.9 | — | **100%** | 95% | **100%** |
| `noise_s4` | valuemetric | 36.1 | ✓ | **100%** | 98% | **100%** |
| `noise_s8` | valuemetric | 30.2 | ✓ | **100%** | 96% | **100%** |
| `overlay_0.15` | geometric | 19.3 | ✓ | **100%** | 96% | **100%** |
| `posterise_24` | valuemetric | 38.1 | ✓ | **100%** | 99% | **100%** |
| `rescale_0.3` | geometric | 29.0 | ✓ | **100%** | **46%** ✗ | **100%** |
| `rescale_0.5` | geometric | 32.5 | ✓ | **100%** | 95% | **100%** |
| `rotate_5` | geometric | — | ✓ | **100%** | **0%** ✗ | **100%** |
| `rsid_simple_q30` | reverse synthid | 29.4 | ✓ | **100%** | _76%_ | **100%** |
| `rsid_simple_q50` | reverse synthid | 30.4 | ✓ | **100%** | 91% | **100%** |
| `rsid_v1_aggressive` | reverse synthid | 26.5 | ✓ | **100%** | _84%_ | **100%** |
| `rsid_v1_balanced` | reverse synthid | 28.7 | ✓ | **100%** | _89%_ | **100%** |
| `rsid_v1_light` | reverse synthid | 32.5 | ✓ | **100%** | 96% | **100%** |
| `rsid_v1_maximum` | reverse synthid | 24.3 | — | **100%** | **41%** ✗ | **100%** |
| `rsid_v2_moderate` | reverse synthid | 27.4 | — | **100%** | **11%** ✗ | _82%_ |
| `rsid_v3_aggressive` | reverse synthid | 44.9 | ✓ | **100%** | **100%** | _68%_ |
| `rsid_v3_gentle` | reverse synthid | 46.8 | ✓ | **100%** | **100%** | **100%** |
| `rsid_v3_maximum` | reverse synthid | 43.8 | ✓ | **100%** | **100%** | **2%** ✗ |
| `rsid_v3_moderate` | reverse synthid | 46.2 | ✓ | **100%** | **100%** | **100%** |
| `saltpepper_0.02` | valuemetric | 21.5 | — | **100%** | _82%_ | **100%** |
| `sharpen_1.5` | valuemetric | 28.1 | ✓ | **100%** | **100%** | **100%** |
| `stack_crop_jpeg_noise` | composite | — | — | **100%** | **0%** ✗ | **100%** |
| `stack_whiten_jpeg_noise` | composite | 27.1 | — | **100%** | **0%** ✗ | **100%** |
| `vae_sd_n0.0` | generative | 27.1 | — | **100%** | **1%** ✗ | **100%** |
| `vae_sd_n0.35` | generative | 21.7 | — | **100%** | **1%** ✗ | 95% |
| `warp_3.0` | geometric | 28.0 | ✓ | **100%** | _84%_ | **100%** |
| `warp_6.0` | geometric | 24.5 | ✓ | **100%** | **9%** ✗ | **100%** |
| `webp_q40` | valuemetric | 30.7 | ✓ | **100%** | _52%_ | **100%** |
| `webp_q70` | valuemetric | 32.4 | ✓ | **100%** | _75%_ | **100%** |
| `whiten_1.0` | adaptive | 30.9 | ✓ | **100%** | **5%** ✗ | **100%** |

## False positives

| Condition | SIGIL | SynthID-style | StableSig-style |
|---|---:|---:|---:|
| Unmarked image | 0.0% | 0.0% | 0.0% |
| Marked under a different key | 0.0% | — | — |

# Phase 04: Open-Source Benchmark Corpus Ingestion & Scaling Protocol

## 1. The "Chinchilla" Scaling Philosophy for Watermarking

In modern machine learning, verifying that a novel neural architecture or optimizer represents a genuine breakthrough rather than an empirical artifact requires scaling rigor: for example, training a 125M parameter model on 2.5B tokens (the Chinchilla-optimal compute frontier) proves convergence, generalization, and stability under load.

In digital watermarking and provenance verification, **the exact equivalent scaling rigor is mandatory**:
- Small evaluations ($N \le 50$ images) are statistically incapable of verifying tail false-positive guarantees ($\alpha \le 10^{-6}$).
- Overfitting to a narrow image distribution (e.g., synthetic diffusion generations with characteristic high-frequency spectral artifacts) creates an illusion of robustness that collapses in the wild.
- A true scientific validation requires a **large-scale, diverse, open-source corpus** evaluated across thousands of image-attack pairs.

Phase 04 establishes this benchmark dataset using strictly open-source, commercially unencumbered images, automated multi-resolution normalization, and perceptual baseline recording.

---

## 2. Dataset Selection & Open-Source Lineage

The evaluation corpus is constructed exclusively from open-source repositories under CC-BY 2.0/4.0, CC0, or Apache 2.0 licenses:

| Corpus Component | Source / Open License | Target Count | Primary Evaluation Purpose |
| :--- | :--- | :--- | :--- |
| **Natural Photography** | COCO Val 2017 (CC-BY 2.0) | $1,000$ images | Real-world scenes, high dynamic range, varied textures, cluttered backgrounds |
| **High-Detail Scenery** | OpenImages / Unsplash Open Sample (CC0) | $500$ images | High-frequency detail (foliage, architectural edges, water ripples) |
| **Synthesized Generative**| LAION-Aesthetics Open Subset (CC0) | $500$ images | Diffusion-generated latents, smooth gradients, generative artifacts |
| **Extreme Dynamic Range**| Wikimedia Commons Featured Natural (CC0) | $500$ images | Low-light, high-contrast, and monochrome edge cases |
| **Statistical Null Suite**| Random Natural Unwatermarked Crops | $100,000$ crops | Monte Carlo empirical false-alarm verification at $\alpha = 10^{-6}$ |

---

## 3. Automated Ingestion & Preprocessing Pipeline

The agent manages ingestion via an automated, fault-tolerant script (`scripts/build_corpus.py`):

```bash
# Automated ingestion into local data directory
.venv/bin/python3 scripts/build_corpus.py \
    --target-dir data/corpus \
    --coco-count 1000 \
    --unsplash-count 500 \
    --laion-count 500 \
    --workers 8
```

### Preprocessing Specifications
1. **Resolution Normalization**:
   - Primary standard: $512 \times 512$ bicubic center-crop (matching the canonical detector grid).
   - High-resolution secondary standard: $768 \times 768$ and $1024 \times 1024$ multi-scale pyramids to evaluate downsampling resilience.
2. **Color Space & Channel Hygiene**:
   - Discard alpha channels; convert all images to 3-channel 8-bit sRGB (`RGB`, float32 range $[0.0, 1.0]$).
   - Reject degenerate images (standard deviation across pixels $\sigma < 0.02$ or single solid color).
3. **Integrity & Checksum**:
   - Compute SHA-256 hashes for all ingested files.
   - Record metadata in `data/corpus_manifest.json` containing: `image_id`, `source`, `license`, `sha256`, `original_dimensions`, `spectral_flatness`.

---

## 4. Perceptual Quality Baseline Protocol

Before applying watermarks or attacks, the pipeline computes and records the unwatermarked baseline perceptual statistics:
- **Spatial Frequency Content**: Laplacian variance and discrete Fourier spectral slope $\beta$ where $S(f) \propto 1/f^\beta$.
- **Perceptual Metrics**: Baseline PSNR, SSIM, and LPIPS across resolution tiers.

---

## 5. Autonomous Self-Correction & Mirror Failover Playbook

If dataset ingestion or preparation encounters failures, the agent must autonomously remediate:

### Failure Mode 1: Remote HTTP 403/429 Rate-Limiting or Broken URL
- **Diagnosis**: Upstream CDN or mirror endpoint has throttled requests or shifted URLs.
- **Agent Action**:
  1. Catch HTTP error in `scripts/build_corpus.py`.
  2. Fall back automatically to backup mirror endpoints (e.g., HuggingFace Datasets API mirrors for COCO val2017: `datasets.load_dataset('detection-datasets/coco', split='val')`).
  3. Verify downloaded bytes against SHA-256 checksums.

### Failure Mode 2: Corrupted Image File or Truncated JPEG
- **Diagnosis**: Download stream terminated prematurely, resulting in partial JPEG headers.
- **Agent Action**:
  1. Open each image using PIL and OpenCV in a validation try-catch block.
  2. If `Image.verify()` raises `UnidentifiedImageError` or `IOError`, immediately delete the corrupted file.
  3. Increment counter and fetch the next available sample from the manifest.

### Failure Mode 3: Dynamic Range Degeneracy (Near-Black or Blank Images)
- **Diagnosis**: Image contains mostly solid color or watermarks fail to embed due to zero gradient energy.
- **Agent Action**:
  1. Filter out images with variance $\operatorname{Var}(I) < 1e-4$.
  2. Replace with next candidate from the dataset pool.

---

## 6. Downstream Cascading Rules

- The ingested corpus path `data/corpus` becomes the primary input for Phase 05 (`phase_05_large_scale_competitive_arena_execution.md`), Phase 06 (`phase_06_learned_stratum_and_fusion_optimization.md`), and Phase 07 (`phase_07_statistical_audit_and_empirical_validation.md`).
- If corpus dimensions or normalization rules change, update `sigil/common.py:canonical_luma` and `sigil/invariant.py:InvariantConfig`.
- Update Table 1 ("Evaluation Corpus Statistics and Diversity") in `paper/sigil.tex`.

---

## 7. Verification Command & Acceptance Criteria

```bash
# 1. Ingest corpus and build manifest
.venv/bin/python3 scripts/build_corpus.py --target-dir data/corpus --smoke-test

# 2. Verify corpus integrity and diversity
.venv/bin/python3 -c "
from sigil.common import list_images
imgs = list_images('data/corpus')
print(f'Ingested {len(imgs)} validated images.')
assert len(imgs) >= 50, 'Corpus below minimum test size'
"
```

### Acceptance Criteria
- [ ] Ingestion script executes with 0 fatal errors.
- [ ] Minimum evaluation corpus of open-source images downloaded and SHA-256 verified.
- [ ] All images verified valid 3-channel RGB without corrupt headers.
- [ ] Manifest file `data/corpus_manifest.json` written and intact.

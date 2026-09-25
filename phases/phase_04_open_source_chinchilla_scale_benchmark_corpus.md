# Phase 04: Open-Source Benchmark Corpus Ingestion & Scaling Protocol

Full ingestion requires `scripts/build_corpus.py --source-manifest PATH`, with
each entry specifying a local image path, source, approved license, and SHA-256.
All entries are checked before output is written. The `--smoke-test` mode uses
procedural and repository fixtures, labels them non-publishable, and writes its
manifest beside the chosen target directory. It is not an open-source corpus.

## 1. The "Chinchilla" Scaling Philosophy for Watermarking

In modern machine learning, verifying that a novel neural architecture or optimizer represents a genuine breakthrough rather than an empirical artifact requires scaling rigor: for example, training a 125M parameter model on 2.5B tokens (the Chinchilla-optimal compute frontier) proves convergence, generalization, and stability under load.

In digital watermarking and provenance verification, **the exact equivalent scaling rigor is mandatory**:
- Small evaluations ($N \le 50$ images) are statistically incapable of verifying tail false-positive guarantees ($\alpha \le 10^{-6}$).
- Overfitting to a narrow image distribution (e.g., synthetic diffusion generations with characteristic high-frequency spectral artifacts) creates an illusion of robustness that collapses in the wild.
- A true scientific validation requires a **large-scale, diverse, open-source corpus** evaluated across thousands of image-attack pairs.

Phase 04 specifies this benchmark dataset. The current ingester verifies local, license-declared source images and records checksums; it does not download images, build resolution pyramids, or measure LPIPS. Those remain future corpus-work items.

---

## 2. Dataset Selection & Open-Source Lineage

The evaluation corpus is constructed from individually documented images with
an allowed license and matching source checksum. Repository or collection
names alone are not evidence of a particular image's license:

| Corpus Component | Source / Open License | Target Count | Primary Evaluation Purpose |
| :--- | :--- | :--- | :--- |
| **Natural Photography** | Individually licensed photographic images | $1,000$ images | Real-world scenes, high dynamic range, varied textures, cluttered backgrounds |
| **High-Detail Scenery** | Individually licensed high-detail images | $500$ images | High-frequency detail (foliage, architectural edges, water ripples) |
| **Synthesized Generative**| Individually licensed generated images | $500$ images | Diffusion-generated latents, smooth gradients, generative artifacts |
| **Extreme Dynamic Range**| Individually licensed high-contrast images | $500$ images | Low-light, high-contrast, and monochrome edge cases |
| **Statistical Null Suite**| Random Natural Unwatermarked Crops | $100,000$ crops | Monte Carlo empirical false-alarm verification at $\alpha = 10^{-6}$ |

---

## 3. Automated Ingestion & Preprocessing Pipeline

The agent manages ingestion via an automated, fault-tolerant script (`scripts/build_corpus.py`):

```bash
# Manifest-backed ingestion into local data directory
.venv/bin/python3 scripts/build_corpus.py \
    --target-dir data/corpus \
    --source-manifest /path/to/sources.json
```

### Preprocessing Specifications
1. **Resolution Normalization**:
   - Current ingestion preserves native aspect ratio and caps the longest side at $1024$ pixels. The detector later maps images to its canonical grid.
   - Explicit $512$, $768$, and $1024$ multi-scale pyramids remain an evaluation target.
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
- **Perceptual Metrics**: PSNR and SSIM are measured in benchmark runs; LPIPS and resolution-tier baselines are not currently produced by ingestion.

---

## 5. Autonomous Self-Correction & Mirror Failover Playbook

If dataset ingestion or preparation encounters failures, the agent must autonomously remediate:

### Failure Mode 1: Missing or changed local source
- **Diagnosis**: A manifest path is absent or its bytes no longer match the declared SHA-256.
- **Action**: Stop before writing output and require a corrected source or manifest. The ingester does not fetch remote mirrors.

### Failure Mode 2: Corrupted Image File or Truncated JPEG
- **Diagnosis**: Download stream terminated prematurely, resulting in partial JPEG headers.
- **Agent Action**:
  1. Verify each source with PIL before writing output.
  2. On an invalid source, stop and report the path; do not delete source data or silently replace the sample.

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
- [ ] Full corpus sources are license-declared and SHA-256 verified; the smoke corpus is explicitly ineligible.
- [ ] All images verified valid 3-channel RGB without corrupt headers.
- [ ] Manifest beside the chosen target directory is written and intact (`data/corpus_manifest.json` for the default full target).

# Phase 07: Comprehensive Statistical Audit, Ablation Studies & Figure Generation

Zero false positives among 100,000 independent null images does not prove an
FPR of $10^{-6}$: its one-sided 95% binomial upper bound is about
$3\times10^{-5}$. The script reports that bound and requires 100,000 distinct
images for a full audit. Small smoke runs only verify execution. Figures and
tables reject smoke, legacy public-key, or otherwise ineligible arena summaries
and require a complete private-key null audit unless `--allow-demo` is supplied
for diagnostic output outside `paper/`.

## 1. Objectives & Audit Scope

Phase 07 conducts the rigorous scientific audit required for archival journal or top-tier conference publication (e.g., ICLR, CVPR, NeurIPS, IEEE S&P).

This phase validates that every empirical curve, table, and claim presented in the manuscript reflects genuine, reproducible experimental data produced on the Google Cloud TPU v4-32 pod.

### Key Audit Mandates:
1. **The $100,000$-Image Null Audit**: Evaluate detector behavior across $10^5$ distinct unwatermarked images and report exact-binomial uncertainty at nominal $\alpha = 10^{-6}$; zero observed alarms does not prove the rate is zero or below $10^{-6}$.
2. **Exhaustive Component Ablations**: Quantify the exact margin provided by each architectural component (soft-clipping, dynamic carriers, anchor descriptors, TPU scan tiling, dual-stratum fusion).
3. **Automated Vector Figure Synthesis**: Generate publication-grade vector graphics (`results/figures/*.pdf`) with strict typographic consistency and high DPI.
4. **LaTeX Table Generation**: Automatically synthesize all formal tables in `paper/tables/` directly from parsed experimental JSON logs.

---

## 2. The Exhaustive Empirical Null Audit ($N \ge 100,000$)

To measure null behavior without overstating what a finite sample can establish:
- Stream $100,000$ unwatermarked image crops from diverse categories (ImageNet, COCO, OpenImages, and synthesized noise).
- Evaluate the maximum detection statistic $T_{\max}$ across the full geometric search grid ($M_{\text{geom}} = 6,760$ candidate transforms).
- Compute observed alarm counts and a one-sided exact-binomial confidence bound at $\alpha=10^{-6}$; neither zero alarms nor any particular count is guaranteed in advance.
- Verify the QQ-plot of empirical p-values against the uniform distribution $\mathcal{U}[0, 1]$ in the sub-critical regime.

```bash
.venv/bin/python3 scripts/fpr_study.py --tpu --trials 100000 --output results/fpr_audit.json
```

---

## 3. Systematic Ablation Suite

The agent executes controlled ablation experiments, systematically disabling one subsystem at a time to isolate individual contributions:

| Ablation Condition | Hypothesized Failure / Degradation Mode | Quantitative Acceptance Criterion |
| :--- | :--- | :--- |
| **Ablation A: Disable Soft-Clipping** | Heavy low-frequency spectral spikes inflate false-alarm tails | False-alarm rate increases by $> 100\times$; establishes necessity of Theorem T3 |
| **Ablation B: Static Carriers (SynthID-style)** | Susceptible to linear collusion attack ($N=50$) | Detection rate collapses from $96\%$ to $< 25\%$ under collusion; proves Theorem T6 |
| **Ablation C: Single Stratum (Invariant Only)** | Degrades under heavy generative diffusion purification | Detection drops from $92\%$ to $68\%$ under SD img2img ($s=0.3$) |
| **Ablation D: Single Stratum (Learned Only)** | Degrades under geometric affine rotation and crop | Detection drops from $95\%$ to $45\%$ under $30^\circ$ rotation without grid |
| **Ablation E: No Anchor Descriptor** | Requires searching full translation space (excessive multiplicity $M$) | Multiplicity increases by $> 100\times$, lowering effective sensitivity by $3\text{ dB}$ |

---

## 4. Vector Figure Synthesis Pipeline (`scripts/figures.py`)

All paper figures must be programmatically generated from actual benchmark run artifacts:

```bash
.venv/bin/python3 scripts/figures.py --results-dir results --output-dir results/figures --format pdf
```

### Primary Generated Figures:
1. `fig01_separation.pdf`: Detector statistic distribution under $H_0$ (null) vs $H_1$ (watermarked), demonstrating $> 10\sigma$ separation.
2. `fig03_strength_curves.pdf`: Detection power vs embedding amplitude, showing sharp phase transitions.
3. `fig04_codebook_collapse.pdf`: Cross-image phase coherence vs number of colluding images $N$, comparing SIGIL's $1/\sqrt{N}$ floor against baseline flatlines.
4. `fig05_null_bound.pdf`: Conservative Hoeffding bound alongside the measured null sample, with its finite-sample limit stated.
5. `fig08_quality_frontier.pdf`: Pareto frontier of PSNR/SSIM vs attack survival rate across different methods.
6. `fig14_geometry_trade.pdf`: Detection rate as a function of continuous rotation angle and scale factor.
7. `fig15_strata_roles.pdf`: Venn diagram and scatter plot illustrating complementary survival of invariant and learned strata across attack families.

---

## 5. Automated LaTeX Table Generation

The agent executes table formatting scripts that compile raw JSON metrics into clean LaTeX tabular code:
- `paper/tables/headline.tex`: Core comparison table (SIGIL vs SynthID vs Stable Signature on clean, compression, rotation, crop, collusion).
- `paper/tables/arena.tex`: Complete attack-by-attack breakdown across all 30+ attacks.
- `paper/tables/arena_families.tex`: Aggregate detection rate by attack family (valuemetric, geometric, generative, composite).
- `paper/tables/theory.tex`: Numerical validation results for Theorems T1–T9.

---

## 6. Autonomous Self-Correction & Discrepancy Protocol

If an anomaly is detected during Phase 07:
1. **Statistical Outlier**: If a single unwatermarked image yields an abnormally high statistic ($T > 4.5$), isolate the image file.
2. **Analysis**: Check if the image contains synthetic repetitive periodic textures (e.g., synthetic bar codes or artificial grid patterns) that simulate carrier frequencies.
3. **Remediation**: Adjust the carrier frequency allocation mask in `sigil/invariant.py` to randomize radial angle offsets per image, preventing resonance with man-made periodic textures.
4. **Re-audit**: Re-run the null audit over the flagged image subset.

---

## 7. Downstream Cascading Rules

- All generated `.pdf` figures in `results/figures/` and `.tex` tables in `paper/tables/` are linked directly in `paper/sigil.tex`.
- Invalidation of any empirical table requires re-compiling the manuscript via `pdflatex` in Phase 08 (`phase_08_repository_polishing_paper_synthesis_and_publication.md`).

---

## 8. Verification Command & Acceptance Criteria

```bash
# 1. Run figures generator
.venv/bin/python3 scripts/figures.py

# 2. Check generated vector PDFs
ls -la results/figures/*.pdf

# 3. Check generated LaTeX tables
ls -la paper/tables/*.tex
```

### Acceptance Criteria
- [ ] Null audit completed on 100,000 distinct sources, with alarm count and one-sided 95% upper bound reported; the empirical result is not a proof of the nominal FPR.
- [ ] All ablation experiments confirm hypothesized margins.
- [ ] All 7 core vector figures (`.pdf`) successfully generated without font or layout warnings.
- [ ] All LaTeX table files updated and populated with valid numerical data.

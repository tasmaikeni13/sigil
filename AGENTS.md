# AGENTS.md

Operational manual and machine-readable instructions for AI coding agents working on the SIGIL repository.

---

## 1. Project Overview & Architectural Mental Model

SIGIL is a research codebase implementing a **stratified watermark for generated images**, combining:
1. **SIGIL-A (Analytic Stratum)**: A spectral watermark operating in the log-magnitude spectrum of the canonical luminance grid. The ideal discrete DFT gives exact cyclic-translation invariance and exact invariance to positive gain constant on each radial bin group. Finite-grid filtering, clipping, and rescaling are approximate and must be checked empirically.
2. **SIGIL-L / SIGIL-C (Learned Strata)**: Neural spatial watermarks with periodic spatial tiling (fine $32 \times 32$ and optional coarse $16 \times 16$ grids) resilient against heavy crops, high-frequency resamplings, and generative purification. Uses per-image nonces under a keyed GF(2) linear code, resolved in $O(2^k k)$ time via the Fast Walsh-Hadamard Transform (FWHT).
3. **Distribution-Free Null Calibration**: Non-asymptotic false-positive upper bounds derived from nonzero weighted Rademacher evidence via Hoeffding's inequality and exact binomial tails; zero evidence receives p-value 1.
4. **Multiplicity Accounting & Fusion**: Explicit Bonferroni correction for all hypothesis searches (spatial offsets, rotations, scales, reflections, and content anchors) fused across strata via weighted Bonferroni at target $\alpha = 10^{-6}$.
5. **Codebook Collapse Defense**: Keyed content anchors (`spectral`, `histogram`, `nonce`) prevent cross-image carrier collusion; residual averaging over marked images yields zero signal.
6. **Hardware Acceleration**: High-throughput fused resynchronisation scan kernels compiled natively for Google Cloud TPU v4 (v4-32 pod slice) via JAX/XLA, with transparent CPU (NumPy) fallback.
7. **Machine-Checked Theory**: Formalization of statistical bounds, multiplicity rules, and geometric invariances in Lean 4.

---

## 2. Directory Structure & Key Files

```text
sigil/                      Core algorithmic package
  ├── system.py             Top-level Sigil watermarking coordinator & p-value fusion
  ├── invariant.py          SIGIL-A analytic spectral stratum & 2-stage resynchronisation scan
  ├── anchors.py            Content anchors (spectral, histogram, nonce) & lattice quantisation
  ├── latent.py             Neural building blocks (Encoder, Decoder, message coding)
  ├── learned.py            SIGIL-L wrapper, linear codes over GF(2), FWHT batch correlation
  ├── stats.py              Rademacher bounds, Hoeffding nulls, binomial tails, Bonferroni fusion
  ├── backend.py            Hardware abstraction layer (Scanner), TPU v4 dispatch & CPU fallback
  ├── kernels.py            JAX/XLA fused scan kernels for TPU v4 hardware
  ├── attacks.py            Adversarial suite (valuemetric, geometric, codebook, generative, adaptive)
  ├── baselines.py          Reference implementations (SynthID-style, Stable Signature-style)
  ├── common.py             Image I/O, colour conversions, quality metrics (PSNR, SSIM), PRF keys
  └── noise.py              Differentiable augmentation layers for learned stratum training

scripts/                    Experiment entry points, benchmarks, reports, and table generators
  ├── train_latent.py       Train the learned stratum with curriculum & adaptive remover network
  ├── theory_checks.py      Numerical verification of Theorems T1–T9
  ├── benchmark.py          Full adversarial benchmark execution
  ├── arena.py              Competitive arena (SIGIL vs SynthID vs Stable Signature)
  ├── fpr_study.py          Large-sample empirical false-positive margin study
  ├── calibrate_descriptor.py  Content-anchor diversity and stability sweep
  ├── calibrate_strength.py Measure fidelity/robustness Pareto frontier to pick operating points
  ├── geometry_frontier.py  Rotation angle vs crop fraction empirical limits
  ├── figures.py            Generate paper figures (PDF and PNG) from results/
  ├── make_tables.py        Emit paper LaTeX tables from measured results
  ├── survival_report.py    Format results/SURVIVAL.md
  └── ablation_report.py    Format results/ABLATION.md

lean/                       Lean 4 formal verification of theoretical core
  ├── Sigil.lean            Entry point importing all verified theorems
  ├── Sigil/Rademacher.lean Distribution-free Rademacher null tail bound
  ├── Sigil/Multiplicity.lean Bonferroni search charge & p-value fusion under arbitrary dependence
  ├── Sigil/Detector.lean   End-to-end detector FPR guarantee at operating alpha
  └── Sigil/Invariance.lean Exact DFT translation and radial filtering invariances

paper/                      Academic manuscript and generated artifacts
  ├── sigil.tex             Primary LaTeX source document
  ├── sigil.pdf             Compiled publication PDF
  └── tables/               Generated LaTeX table macros populated by scripts/make_tables.py

results/                    Checked-in benchmarks, telemetry, metrics, and generated figures
checkpoints/                Training run metadata and logs (weights are gitignored)
phases/                     8-phase autonomous agentic research specification and DAG protocols
images/                     Corpus metadata and licensing documentation
run_all.sh                  Full end-to-end reproduction pipeline
```

---

## 3. Environment & Execution Guidelines

### Python Environment
- Python version: `3.10` or `3.11`.
- Virtual environment: `.venv` located at repository root.
- Path convention: Execute all commands and scripts from the **repository root** so `sigil` resolves as a top-level package.
- Full experimental runs require a private 32-byte hex key in `SIGIL_MASTER_KEY_HEX`; public defaults are for tests and demonstrations only.

### Lean 4 Toolchain
- Installed via `elan` under `~/.elan/bin/` (or `../.elan/bin/`).
- Set environment before running `lake`:
  ```bash
  export PATH="$HOME/.elan/bin:$PATH"
  ```

### Hardware Dispatch
- TPU hardware is automatically used when available via JAX/XLA (`libtpu`).
- On machines without TPU (standard CPU/GPU hosts), `sigil.backend.Scanner` seamlessly falls back to optimized NumPy routines.
- **TPU Pod Slice Topology**: When operating on a single host of a Google Cloud TPU v4-32 slice, always ensure the following bounds are exported to prevent JAX distributed initialization hangs:
  ```bash
  export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
  export TPU_HOST_BOUNDS="1,1,1"
  ```

---

## 4. Verification & Testing Commands

Before committing or completing any task, run the relevant verification tiers:

### Tier 1: Static Analysis & Formatting
Always ensure clean linting and formatting:
```bash
./.venv/bin/ruff check sigil scripts tests
./.venv/bin/ruff format --check sigil scripts tests
./.venv/bin/python -m compileall -q sigil scripts tests
```
Auto-format code when needed:
```bash
./.venv/bin/ruff format sigil scripts tests
```

### Tier 2: Automated Tests
Run CPU-compatible unit and smoke tests:
```bash
# End-to-end analytic watermark embedding and detection smoke test:
./.venv/bin/pytest tests/test_sigil_system.py

# CPU backend transform bank and scanner tests:
./.venv/bin/pytest tests/test_backend.py -k "not tpu"
```
*Note on TPU tests*: `tests/test_tpu_kernels.py` and `test_tpu_available_flag` in `tests/test_backend.py` strictly require a Google Cloud TPU v4 VM environment with attached TPU devices.

### Tier 2b: Script Smoke Tests (Fast Pipeline Verification)
Verify execution scripts across the pipeline without long-running compute:
```bash
# Ingestion smoke test (>=64 validated images):
./.venv/bin/python scripts/build_corpus.py --target-dir data/smoke_corpus --smoke-test

# Geometric rotation and crop sweep on TPU:
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" ./.venv/bin/python scripts/geometry_frontier.py --tpu --trials 2

# Fused TPU scan kernel benchmark vs float64 reference:
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" ./.venv/bin/python scripts/bench_kernels.py --tpu --tile-k 128

# Multi-system competitive arena smoke test (SIGIL vs SynthID):
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" ./.venv/bin/python scripts/arena.py --smoke-test

# Neural stratum training loop smoke test:
./.venv/bin/python scripts/train_latent.py --smoke-test

# Operating point strength calibration smoke test:
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" ./.venv/bin/python scripts/calibrate_strength.py --tpu --smoke-test

# Checkpoint model selection smoke test:
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" ./.venv/bin/python scripts/select_checkpoint.py --smoke-test

# Large-sample null audit smoke test:
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" ./.venv/bin/python scripts/fpr_study.py --tpu --smoke-test
```

### Tier 3: Mathematical Formalization (Lean 4)
When editing theoretical statements, proof dependencies, or statistical formulations:
```bash
export PATH="$HOME/.elan/bin:$PATH"
(cd lean && lake build Sigil)
```
Ensure build exits with code 0.

### Tier 4: Statistical & Theory Checks
Numerical validation of theorems T1–T9 against synthetic/natural images:
```bash
./.venv/bin/python scripts/theory_checks.py --limit 40
```

### Tier 5: Paper Artifacts & Manuscript Compilation
Regenerate tables, figures, and compile the manuscript:
```bash
./.venv/bin/python scripts/make_tables.py
./.venv/bin/python scripts/figures.py
(cd paper && pdflatex -interaction=nonstopmode sigil.tex)
```
The generators refuse old or smoke arena/benchmark summaries and incomplete
null audits by default.
For diagnostic output only, use `--allow-demo --min-arena-images 1` with
an output directory outside `paper/tables` or `results/figures`.

---

## 5. Coding Standards & Conventions

1. **Python Style**:
   - Target Python 3.11 features; enforce 88-character maximum line length via Ruff.
   - Use double quotes for strings, LF line endings, and sorted imports.
   - Always include `from __future__ import annotations` in all library modules.
   - Prefer frozen `@dataclass(frozen=True)` for configurations and immutability.
2. **Mathematical & Statistical Rigor**:
   - Never replace exact distribution-free bounds with empirical heuristics.
   - When searching $M$ hypotheses, the p-value must be Bonferroni-corrected: $p_{\text{corr}} = \min(1.0, M \cdot p)$.
   - Preserve numerical precision: use `np.float64` for sensitive normalizations and carrier correlations.
   - Use `_EPS = 1e-12` guards to prevent divide-by-zero or `log(0)`.
   - Log-p-values must use `sigil.stats.log10p(p)` for stable floating-point presentation.
3. **Cryptographic Keying & Anchors**:
   - All pseudo-random streams must derive through `sigil.common.derive_key` or `sigil.common.key_stream_rng` using domain-separated labels.
   - Do not create static carrier sets: carriers must be keyed to per-image content anchors (`spectral`, `histogram`) or dynamic `nonce`.
4. **TPU / JAX Optimizations**:
   - Maintain static tensor shapes for XLA compilation. Avoid dynamic shapes inside JIT-compiled scan kernels.
   - Keep carrier grids resident in device memory; avoid transferring large evidence arrays back to host CPU during candidate searches.

---

## 6. Operational Guardrails (Always / Ask First / Never)

### ALWAYS:
- Run `ruff check` and `ruff format --check` before committing.
- Maintain documentation integrity: keep all existing docstrings, theorem cross-references, and code commentary intact.
- Keep Lean 4 proofs synchronized with any changes to mathematical invariants or null distributions.
- Record experiment metadata: when reporting benchmark figures, specify image corpus, checkpoint path, device, random seed, operating point $\alpha$, and admissibility constraints.

### ASK FIRST:
- Modifying foundational theorems (Theorems T1–T9) or alter the default operating point $\alpha = 10^{-6}$.
- Introducing new external heavyweight dependencies into `pyproject.toml`.
- Initiating full production training loops (`scripts/train_latent.py`) or full arena evaluations (`scripts/arena.py`) without prior agreement on computational budget. (Note: `--smoke-test` runs are fast and encouraged for continuous verification).

### NEVER:
- **NEVER** commit private image corpora (`data/corpus/`), raw image downloads, or synthetic datasets.
- **NEVER** generate manuscript tables or figures from smoke or legacy public-key arena summaries as publication evidence.
- **NEVER** commit model weight files (`checkpoints/*.pt`) or Hugging Face cache files (`data/hf/`).
- **NEVER** weaken or bypass Bonferroni multiplicity penalties to artificially lower p-values.
- **NEVER** hardcode static codebooks or disable per-image nonce derivation.
- **NEVER** commit broken Lean 4 proofs or introduce regressions into `paper/tables/` or `paper/sigil.tex`.

---

## 7. Autonomous Self-Correction & Recovery Protocol

When an automated experiment, theorem proof, or kernel verification fails:
1. **Isolate Failure Mode**: Identify whether the defect is mathematical/statistical (e.g., FPR violation, loss of separation), algorithmic (e.g., bit error rate under warp), or hardware-specific (e.g., XLA buffer mismatch).
2. **Autonomous Literature Research**: Investigate established mathematical principles, arXiv literature, or numerical algorithms to formulate a provably correct fix.
3. **Formal Verification Gate**: If the formulation touches core theory, update and verify the Lean 4 proof in `lean/Sigil/` and check numerical epsilon bounds with `scripts/theory_checks.py`.
4. **Hardware Parity Verification**: Verify numerical equivalence between TPU kernels and CPU reference implementations.
5. **Cascading Invalidation**: When modifying upstream parameters or interfaces, update affected downstream scripts in `scripts/`, phase files in `phases/`, and LaTeX macros in `paper/`.

---

## 8. Git & Pull Request Protocol

- **Branch Hygiene**: Work on clean branches; ensure working tree is clean before submitting.
- **Commit Messages**: Write concise, descriptive commit headers in the imperative mood (e.g., `Add AGENTS.md for AI coding agent orchestration`, `Fix numerical stability in radial profile calculation`).
- **Verification Summary**: In PR descriptions, explicitly state which verification tiers were executed (e.g., Ruff linting, pytest smoke tests, Lake Lean build, LaTeX compilation).

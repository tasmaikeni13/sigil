# Phase 08: Repository Polishing, Academic Paper Synthesis & Publication Readiness

The checked-in manuscript PDF is a corrected historical pilot, not a
publication-eligible Phase 05--07 result. Full publication artifacts require
the private-key, manifest-backed, TPU arena and null audit gates. CPU smoke
tests and a clean PDF build verify execution and typesetting only.

## 1. Objectives & Final Delivery Mandate

Phase 08 is the final stage of the autonomous research lifecycle. Its mandate is to transform the entire repository, codebase, documentation, and academic manuscript into an exemplary, publication-ready open-source research artifact.

The repository must reflect the standards of top-tier academic venues (e.g., ICLR, CVPR, NeurIPS, ACM CCS, IEEE S&P) and prestigious open-source scientific software.

### Primary Deliverables:
1. **Codebase Sanitation & PEP 8 Compliance**: Enforce strict Python style guidelines, formatting, docstring standards, and zero lint errors across all files.
2. **Commentary & Documentation Humanization**: Ensure all code comments, docstrings, and guides read with the natural authority, nuance, and insight of expert human researchers. All AI conversational residues, prompts, or historical migration notes must be eradicated.
3. **Academic Paper Compilation (`paper/sigil.pdf`)**: Seamlessly compile the primary LaTeX manuscript (`paper/sigil.tex`) with all synchronized empirical tables, figures, and verified theorem references using `pdflatex`.
4. **Repository Structure & Publication Polish**: Verify `.gitignore`, directory cleanliness, test suite completeness, reproduction scripts (`run_all.sh`), and push clean commits to the remote repository.

---

## 2. Codebase Standardization & Linting Protocol

The agent must execute automated linting, formatting, and static typing passes across the entire project:

```bash
# 1. Format all Python source files according to PEP 8
.venv/bin/ruff format sigil/ scripts/ tests/

# 2. Enforce strict lint checks and automatically apply safe fixes
.venv/bin/ruff check sigil/ scripts/ tests/ --fix

# 3. Verify Python bytecode compilation across all modules
.venv/bin/python3 -m compileall -q sigil/ scripts/ tests/
```

### Style & Code Quality Rules:
- **Type Annotations**: All public functions and class methods must feature complete typing signatures (`typing.List`, `Tuple`, `Optional`, `Dict`, `Array`).
- **Docstring Standards**: Follow Google/NumPy docstring conventions with clear `Args:`, `Returns:`, and mathematical definitions.
- **Hardware-Agnostic Modularity**: Clean device-selection abstraction with explicit TPU pod execution paths and CPU testing fallbacks.

---

## 3. Humanizing Documentation & Technical Prose

All documentation must be curated to reflect natural human authorship:
- **`README.md`**:
  - Clear, compelling motivation and high-level architectural intuition.
  - Interactive mermaid diagrams depicting the stratified watermark embedding and detection flow.
  - Exact hardware specifications for Google Cloud TPU v4-32 pods.
  - Direct reproduction commands with realistic execution timings.
  - Citation block formatted for BibTeX.
- **In-Code Comments**:
  - Explain *why* a particular mathematical normalization or kernel stride is chosen (e.g., "Tiled accumulation over blocks of 128 elements prevents VPU register spilling and maximizes TPU HBM memory throughput").
  - Eliminate any phrasing resembling "this was changed because", "formerly CUDA", "as requested by user", or "AI generated".

---

## 4. Academic Paper Synthesis (`paper/sigil.tex`)

The agent updates and compiles the primary scientific paper:

### Compilation Pipeline:
```bash
cd paper
# Run pdflatex passes to resolve all cross-references, citations, and table floats
pdflatex -interaction=nonstopmode sigil.tex
pdflatex -interaction=nonstopmode sigil.tex
cd ..
```

### Manuscript Quality Checks:
- [ ] Zero missing citation warnings (`LaTeX Warning: Citation ... undefined`).
- [ ] Zero missing cross-reference warnings (`LaTeX Warning: Reference ... undefined`).
- [ ] All table numbers match the empirical results generated in Phase 05 and Phase 07.
- [ ] All 7 vector figure PDFs are properly referenced, positioned, and legible.
- [ ] Output `paper/sigil.pdf` is rendered with crisp vector typography and correct page margins.

---

## 5. End-to-End Verification Checklist

Before issuing final commits and pushes, the agent must execute the complete verification sequence:

```bash
# Set TPU environment for local worker 0
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,1"

# 1. Run full test suite
.venv/bin/pytest -v tests/

# 2. Run mathematical theory checks
.venv/bin/python3 scripts/theory_checks.py

# 3. Run TPU kernel benchmarks
.venv/bin/python3 scripts/bench_kernels.py --tpu --tile-k 128

# 4. Check git status to ensure working tree cleanliness
git status
```

---

## 6. Remote Publication & Git Release

Once all verification checks pass:
1. Stage all new phase files, documentation updates, and compiled artifacts:
   ```bash
   git add phases/ README.md paper/ scripts/ sigil/ tests/ .gitignore
   ```
2. Create a clean, conventional commit message:
   ```bash
   git commit -m "Complete autonomous research phases and publishable release"
   ```
3. Push directly to GitHub `main`:
   ```bash
   git push origin main
   ```
4. Verify remote synchronization via `git status` (must indicate `Your branch is up to date with 'origin/main'`).

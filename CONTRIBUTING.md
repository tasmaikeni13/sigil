# Contributing to SIGIL

SIGIL is a research codebase. Contributions are easiest to review when they
make the experiment easier to reproduce and keep claims tied to measured
evidence.

## Before opening a change

- Run `python -m ruff check sigil scripts`.
- Run `python -m ruff format --check sigil scripts`.
- Run `python -m compileall -q sigil scripts`.
- For changes to the statistical core, run the relevant command in
  `scripts/theory_checks.py` and the Lean build when the local toolchain is
  available.
- For changes to the benchmark, record the corpus, checkpoint, device, seed,
  operating point, quality budget, and command used to produce new results.

## Data and generated files

Do not commit private image corpora, downloaded model caches, trained weight
files, or local environment directories. The repository’s `.gitignore` excludes
the main generated paths, but check `git status` before committing. Dataset
terms remain the contributor’s responsibility; review
`images/lite/TERMS.md` before using or redistributing associated data.

## Reporting results

Keep the comparison conditions explicit. Include denominators, the operating
point, the admissibility rule, and relevant negative results. A single summary
number is not enough when an attack family contains both successes and
failures.

## Pull requests

Describe what changed, why it changed, and how it was checked. If a result or
figure changes, link the generating script and identify the input artifacts so a
reviewer can reproduce the result without guessing.

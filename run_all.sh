#!/usr/bin/env bash
# Reproduction driver. Smoke artifacts are deliberately separate from paper inputs.
set -euo pipefail

PY="${PY:-./.venv/bin/python}"
SMOKE_TEST=0
if [ "${1:-}" = "--smoke-test" ]; then
  SMOKE_TEST=1
fi
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"

say() { printf '\n== %s\n' "$*"; }

if [ "$SMOKE_TEST" = "1" ]; then
  CORPUS="data/smoke_corpus"
  RESULTS="results/smoke"
  CHECKPOINTS="data/smoke_checkpoints"
  mkdir -p "$RESULTS" "$CHECKPOINTS"
  say "1/7 smoke corpus"
  "$PY" scripts/build_corpus.py --target-dir "$CORPUS" --smoke-test
  say "2/7 descriptor calibration"
  "$PY" scripts/calibrate_descriptor.py --image-dir "$CORPUS/natural" \
    --out "$RESULTS/descriptor_calibration.json" --smoke-test
  say "3/7 one-step learned training"
  "$PY" scripts/train_latent.py --data "$CORPUS/natural" \
    --out "$CHECKPOINTS" --smoke-test
  say "4/7 numerical theory checks"
  "$PY" scripts/theory_checks.py --corpus "$CORPUS/natural" \
    --checkpoint "$CHECKPOINTS/latent.pt" --limit 2 \
    --out "$RESULTS/theory_checks.json"
  say "5/7 benchmark and arena"
  "$PY" scripts/benchmark.py --corpus "$CORPUS/natural" \
    --checkpoint "$CHECKPOINTS/latent.pt" --device cpu --smoke-test \
    --out "$RESULTS/benchmark.csv" --summary "$RESULTS/summary.json"
  "$PY" scripts/arena.py --corpus "$CORPUS/natural" \
    --checkpoint "$CHECKPOINTS/latent.pt" --devices cpu --smoke-test \
    --out "$RESULTS/arena.csv" --summary "$RESULTS/arena_summary.json"
  say "6/7 null-audit smoke"
  "$PY" scripts/fpr_study.py --corpus "$CORPUS/natural" \
    --checkpoint "$CHECKPOINTS/latent.pt" --device cpu --smoke-test \
    --out "$RESULTS/fpr_study.json"
  say "7/7 Lean proofs"
  ( cd lean && PATH="${ELAN_HOME:-$HOME/.elan}/bin:$PATH" lake build Sigil )
  say "smoke checks complete; no paper artifacts were generated"
  exit 0
fi

: "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to a licensed image manifest for full runs}"
: "${NULL_CORPUS:?Set NULL_CORPUS to at least 100000 independent null images}"
: "${SIGIL_MASTER_KEY_HEX:?Set SIGIL_MASTER_KEY_HEX to a private 32-byte hex key}"
LIMIT="${LIMIT:-200}"
STEPS="${STEPS:-12000}"
WORKERS="${WORKERS:-4}"
if [ "$LIMIT" -lt 200 ]; then
  echo "full publication arena requires LIMIT >= 200" >&2
  exit 2
fi

say "1/8 verified corpus"
"$PY" scripts/build_corpus.py --target-dir data/corpus \
  --source-manifest "$SOURCE_MANIFEST"
say "2/8 descriptor calibration"
"$PY" scripts/calibrate_descriptor.py --image-dir data/corpus/natural
say "3/8 production training"
"$PY" scripts/train_latent.py --steps "$STEPS" --batch 8 \
  --workers "$WORKERS" --n-bits 1024 --vae-from 600 \
  --adversarial-from 9000 --severity-steps 3500 --turbo \
  --target-psnr 42.0 --out checkpoints --data data/corpus/natural
say "4/8 numerical theory checks"
"$PY" scripts/theory_checks.py --corpus data/corpus/natural --limit 40
say "5/8 benchmark, arena, null audit"
"$PY" scripts/benchmark.py --corpus data/corpus/natural --limit "$LIMIT"
"$PY" scripts/arena.py --corpus data/corpus/natural --limit "$LIMIT" \
  --workers "$WORKERS" --corpus-manifest data/corpus_manifest.json
"$PY" scripts/fpr_study.py --corpus "$NULL_CORPUS" --trials 100000
say "6/8 figures and tables"
"$PY" scripts/figures.py
"$PY" scripts/make_tables.py
say "7/8 Lean proofs"
( cd lean && PATH="${ELAN_HOME:-$HOME/.elan}/bin:$PATH" lake build Sigil )
say "8/8 manuscript"
( cd paper && pdflatex -interaction=nonstopmode sigil.tex && \
  pdflatex -interaction=nonstopmode sigil.tex )
say "done"

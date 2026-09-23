#!/usr/bin/env bash
# Reproduce the experiment pipeline used for the SIGIL paper on Google Cloud TPU v4-32.
#
# Run from the repository root with the virtual environment active. Each step
# writes to a predictable location and can be rerun after it completes.
set -euo pipefail

PY="${PY:-./.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/data/hf}"
LIMIT="${LIMIT:-20}"          # images taken from each corpus
STEPS="${STEPS:-12000}"
WORKERS="${WORKERS:-4}"
PHOTO_SOURCE="${PHOTO_SOURCE:-}"

SMOKE_TEST="${SMOKE_TEST:-0}"
if [ "${1:-}" = "--smoke-test" ]; then
  SMOKE_TEST="1"
fi

# Configure Google Cloud TPU v4 environment defaults
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/8  corpora"
if [ "$SMOKE_TEST" = "1" ]; then
  [ -d data/corpus/natural ] || $PY scripts/build_corpus.py --target-dir data/corpus --smoke-test
else
  [ -d data/corpus/natural ] || $PY scripts/build_corpus.py --n-natural 100 --n-synthetic 240
fi

# The photographic corpus is downscaled from the supplied image set.
if [ ! -d data/corpus/photo ]; then
  if [ -z "$PHOTO_SOURCE" ]; then
    say "PHOTO_SOURCE is unset; skipping the photographic corpus"
  else
    $PY - "$PHOTO_SOURCE" <<'PY'
import concurrent.futures as futures
import sys
from pathlib import Path

sys.path.insert(0, ".")

from sigil.common import load_image, save_image

source = Path(sys.argv[1])
destination = Path("data/corpus/photo")
destination.mkdir(parents=True, exist_ok=True)
paths = sorted(
    path
    for path in source.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)[:300]


def convert(path: Path) -> None:
    save_image(destination / f"{path.stem}.png", load_image(path, max_size=768))


with futures.ThreadPoolExecutor(max_workers=16) as executor:
    list(executor.map(convert, paths))
print(f"photo corpus: {len(paths)} images")
PY
  fi
fi

say "2/8  anchor calibration  ->  results/descriptor_calibration.json"
if [ "$SMOKE_TEST" = "1" ]; then
  $PY scripts/calibrate_descriptor.py --smoke-test
else
  $PY scripts/calibrate_descriptor.py --n-diversity 80 --n-stability 12
fi

say "3/8  train the learned stratum  ->  checkpoints/latent.pt"
if [ ! -f checkpoints/latent.pt ]; then
  if [ "$SMOKE_TEST" = "1" ]; then
    $PY scripts/train_latent.py --smoke-test
  else
    $PY scripts/train_latent.py \
        --steps "$STEPS" --batch 8 --workers "$WORKERS" --n-bits 1024 \
        --vae-from 600 --adversarial-from 9000 --severity-steps 3500 --turbo \
        --target-psnr 37.0 --out checkpoints \
        --data data/div2k/DIV2K_train_HR data/corpus/photo data/corpus/synthetic
  fi
fi

say "4/8  numerical verification of every analytic claim  ->  results/theory_checks.json"
$PY scripts/theory_checks.py --limit "$([ "$SMOKE_TEST" = "1" ] && echo 10 || echo 40)"

say "5a/8  the full attack benchmark  ->  results/benchmark.csv"
if [ "$SMOKE_TEST" = "1" ]; then
  $PY -u scripts/benchmark.py --smoke-test
else
  $PY -u scripts/benchmark.py --limit "$LIMIT"
fi

say "5b/8  the arena: SIGIL vs SynthID-style vs Stable-Signature-style,"
say "      including the reverse-SynthID removal suite, sharded across TPU workers"
if [ "$SMOKE_TEST" = "1" ]; then
  $PY -u scripts/arena.py --smoke-test
else
  $PY -u scripts/arena.py --limit "$LIMIT" --workers "$WORKERS"
fi

say "5c/8  larger-sample false-positive margin  ->  results/fpr_study.json"
if [ "$SMOKE_TEST" = "1" ]; then
  $PY -u scripts/fpr_study.py --smoke-test
else
  $PY -u scripts/fpr_study.py --limit 150
fi

say "6/8  figures and tables"
$PY scripts/figures.py
$PY scripts/make_tables.py

say "7/8  machine-checked proofs, then the manuscript"
( cd lean && PATH="${ELAN_HOME:-$HOME/.elan}/bin:$PATH" lake build Sigil )
if command -v latexmk >/dev/null 2>&1; then
  ( cd paper && latexmk -pdf -f -interaction=nonstopmode sigil.tex )
else
  ( cd paper && pdflatex -interaction=nonstopmode sigil.tex )
fi

say "done - paper/sigil.pdf"

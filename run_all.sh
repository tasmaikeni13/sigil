#!/usr/bin/env bash
# Reproduce the experiment pipeline used for the SIGIL paper.
#
# Run from the repository root with the virtual environment active. Each step
# writes to a predictable location and can be rerun after it completes.
set -euo pipefail

PY="${PY:-./.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/data/hf}"
LIMIT="${LIMIT:-20}"          # images taken from each corpus
STEPS="${STEPS:-12000}"
GPUS="${GPUS:-2}"
PHOTO_SOURCE="${PHOTO_SOURCE:-}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/8  corpora"
[ -d data/corpus/natural ] || $PY scripts/build_corpus.py \
    --n-natural 100 --n-synthetic 240
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
$PY scripts/calibrate_descriptor.py --n-diversity 80 --n-stability 12

say "3/8  train the learned stratum  ->  checkpoints/latent.pt"
if [ ! -f checkpoints/latent.pt ]; then
  $PY -m torch.distributed.run --nproc_per_node="$GPUS" scripts/train_latent.py \
      --steps "$STEPS" --batch 8 --workers 4 --n-bits 1024 \
      --vae-from 600 --adversarial-from 9000 --severity-steps 3500 --turbo \
      --target-psnr 37.0 --out checkpoints \
      --data data/div2k/DIV2K_train_HR data/corpus/photo data/corpus/synthetic
fi

say "4/8  numerical verification of every analytic claim  ->  results/theory_checks.json"
$PY scripts/theory_checks.py --limit 40

say "5a/8  the full attack benchmark  ->  results/benchmark.csv"
$PY -u scripts/benchmark.py --limit "$LIMIT"

say "5b/8  the arena: SIGIL vs SynthID-style vs Stable-Signature-style,"
say "      including the reverse-SynthID removal suite, sharded over both GPUs"
$PY -u scripts/arena.py --limit "$LIMIT" --workers "$GPUS"

say "5c/8  larger-sample false-positive margin  ->  results/fpr_study.json"
$PY -u scripts/fpr_study.py --limit 150

say "6/8  figures and tables"
$PY scripts/figures.py
$PY scripts/make_tables.py

say "7/8  machine-checked proofs, then the manuscript"
( cd lean && ELAN_HOME="$PWD/../.elan" PATH="$PWD/../.elan/bin:$PATH" \
    lake build Sigil )
( cd paper && latexmk -pdf -f -interaction=nonstopmode sigil.tex )

say "done - paper/sigil.pdf"

#!/usr/bin/env python3
"""Build the evaluation corpora.

Two corpora, because the two halves of the claim are different.

``natural`` — DIV2K high-resolution photographs, the standard image-processing
benchmark.  These test the watermark as a signal-processing object.

``synthetic`` — images generated locally with SD-Turbo across a spread of
subjects and styles.  Generative-image provenance is the actual application, so
the system should be measured on the kind of image it is meant to mark; these
also carry no third-party rights, since they are produced here rather than
collected.

Usage:
    python scripts/build_corpus.py --n-synthetic 200 --n-natural 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil.common import list_images, load_image, save_image

SUBJECTS = [
    "a snow-covered mountain ridge at sunrise",
    "a dense tropical rainforest canopy",
    "an empty city street at night with wet asphalt",
    "a close-up of a dragonfly on a reed",
    "a wooden sailing boat on a calm lake",
    "an abandoned greenhouse full of plants",
    "a desert canyon with layered red rock",
    "a bowl of ripe fruit on a linen cloth",
    "a lighthouse on a rocky coast in fog",
    "a field of sunflowers under clouds",
    "an old library with tall wooden shelves",
    "a glacier calving into dark water",
    "a market stall of spices in bright sun",
    "a cat asleep on a windowsill",
    "a suspension bridge seen from below",
    "a potter's hands shaping clay",
    "a flock of birds over a salt marsh",
    "a vintage motorcycle in a garage",
    "a forest path covered in autumn leaves",
    "a coral reef with small fish",
    "a steam locomotive at a rural station",
    "a chef plating a dish in a kitchen",
    "an aerial view of terraced rice fields",
    "a violin resting on sheet music",
    "a storm front over open prairie",
    "a stone staircase in an old village",
]

STYLES = [
    "photograph, 50mm, natural light",
    "cinematic photograph, shallow depth of field",
    "documentary photograph, high detail",
    "wide-angle landscape photograph",
    "studio photograph, soft lighting",
    "candid photograph, golden hour",
]


def generate_synthetic(
    out_dir: Path, n: int, device: str, size: int, seed: int
) -> list[dict]:
    import torch
    from diffusers import AutoPipelineForText2Image

    torch_dev = "cpu" if str(device).startswith("tpu") else device
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float32
    )
    pipe.to(torch_dev)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        subject = SUBJECTS[i % len(SUBJECTS)]
        style = STYLES[(i // len(SUBJECTS)) % len(STYLES)]
        prompt = f"{subject}, {style}"
        gen = torch.Generator(device=torch_dev).manual_seed(seed + i)
        img = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=size,
            width=size,
            generator=gen,
        ).images[0]
        name = f"syn_{i:04d}.png"
        img.save(out_dir / name)
        manifest.append(
            {
                "file": name,
                "prompt": prompt,
                "seed": seed + i,
                "model": "stabilityai/sd-turbo",
                "steps": 4,
                "size": size,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  generated {i + 1}/{n}", flush=True)
    return manifest


def collect_natural(src: Path, out_dir: Path, n: int, max_size: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(src)[:n]
    manifest = []
    for p in paths:
        img = load_image(p, max_size=max_size)
        save_image(out_dir / p.name, img)
        manifest.append(
            {
                "file": p.name,
                "source": "DIV2K",
                "original": str(p),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
            }
        )
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--div2k", default="data/div2k/DIV2K_valid_HR")
    ap.add_argument("--n-synthetic", type=int, default=200)
    ap.add_argument("--n-natural", type=int, default=100)
    ap.add_argument("--natural-max-size", type=int, default=1024)
    ap.add_argument("--synthetic-size", type=int, default=512)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    out = Path(args.out)
    manifest = {}
    if args.n_natural > 0:
        print(f"collecting {args.n_natural} natural images ...", flush=True)
        manifest["natural"] = collect_natural(
            Path(args.div2k), out / "natural", args.n_natural, args.natural_max_size
        )
    if args.n_synthetic > 0:
        print(f"generating {args.n_synthetic} synthetic images ...", flush=True)
        manifest["synthetic"] = generate_synthetic(
            out / "synthetic",
            args.n_synthetic,
            args.device,
            args.synthetic_size,
            args.seed,
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()

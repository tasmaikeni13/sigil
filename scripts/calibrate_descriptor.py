#!/usr/bin/env python3
"""Measure content-descriptor diversity and attack stability.

Two numbers decide whether a content-derived carrier set is viable:

*Diversity* — the fraction of a corpus that lands in distinct quantiser cells.
Low diversity means an adversary can cluster images by descriptor without
knowing the key, gather many images that share a carrier set, and run the
codebook attack inside a cluster.

*Stability* — the fraction of attacked images whose true cell is still inside
the detector's candidate list.  Low stability means the detector cannot find
the carriers even though they are still in the image.

The two pull against each other: a coarse quantiser is stable and collides, a
fine one is diverse and fragile.  This script sweeps candidate descriptor
families so the choice is made on measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil.anchors import AnchorReader, quantise_with_list
from sigil.common import list_images, load_image, to_uint8
from sigil.invariant import InvariantConfig

# ---- descriptor families --------------------------------------------------
# Both read through the deployed AnchorReader, so what is calibrated here is
# exactly what the detector uses.


def desc_spectral(img, cfg, assumed_scale=1.0):
    return AnchorReader(img, cfg.canon).spectral(assumed_scale)


def desc_histogram(img, cfg, assumed_scale=1.0):
    return AnchorReader(img, cfg.canon).histogram(assumed_scale)


FAMILIES = {"spectral": desc_spectral, "histogram": desc_histogram}
SPECTRAL_SCALES = (1.0, 0.90, 1.11, 0.80, 1.25, 0.70, 1.43)


# ---- attacks --------------------------------------------------------------


def jpeg(img, q):
    buf = BytesIO()
    Image.fromarray(to_uint8(img), "RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101
    )


def crop(img, frac):
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    return np.ascontiguousarray(
        img[(h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw]
    )


ATTACKS = {
    "identity": lambda x: x,
    "jpeg50": lambda x: jpeg(x, 50),
    "jpeg30": lambda x: jpeg(x, 30),
    "noise8": lambda x: np.clip(
        x + np.random.default_rng(0).normal(0, 8 / 255, x.shape), 0, 1
    ).astype(np.float32),
    "blur2": lambda x: cv2.GaussianBlur(x, (0, 0), 2.0),
    "rot5": lambda x: rotate(x, 5.0),
    "rot15": lambda x: rotate(x, 15.0),
    "crop90": lambda x: crop(x, 0.90),
    "crop75": lambda x: crop(x, 0.75),
    "crop50": lambda x: crop(x, 0.50),
    "scale0.5": lambda x: cv2.resize(
        x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA
    ),
    "bright+12": lambda x: np.clip(x + 12 / 255, 0, 1).astype(np.float32),
    "contrast1.2": lambda x: np.clip((x - 0.5) * 1.2 + 0.5, 0, 1).astype(np.float32),
    "gamma1.3": lambda x: np.power(np.clip(x, 1e-6, 1), 1 / 1.3).astype(np.float32),
}


def cell(desc, delta):
    return tuple(np.rint(np.asarray(desc) / delta).astype(np.int64).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="data/div2k/DIV2K_valid_HR")
    ap.add_argument("--n-diversity", type=int, default=100)
    ap.add_argument("--n-stability", type=int, default=16)
    ap.add_argument("--max-size", type=int, default=768)
    ap.add_argument("--list-size", type=int, default=64)
    ap.add_argument("--out", default="results/descriptor_calibration.json")
    args = ap.parse_args()

    cfg = InvariantConfig()
    paths = list_images(args.image_dir)
    report = {}

    for fam, fn in FAMILIES.items():
        for delta in (0.45, 0.70, 1.00, 1.40):
            descs = [
                fn(load_image(p, max_size=args.max_size), cfg)
                for p in paths[: args.n_diversity]
            ]
            cells = [cell(d, delta) for d in descs]
            diversity = len(set(cells)) / len(cells)

            stab = {}
            for aname, afn in ATTACKS.items():
                hits = 0
                for p in paths[: args.n_stability]:
                    img = load_image(p, max_size=args.max_size)
                    ref = cell(fn(img, cfg), delta)
                    att = np.ascontiguousarray(afn(img))
                    best = None
                    for sc in SPECTRAL_SCALES if fam == "spectral" else (1.0,):
                        d2 = fn(att, cfg, sc)
                        cands = quantise_with_list(d2, delta, args.list_size)
                        if any(tuple(q.tolist()) == ref for q, _ in cands):
                            best = True
                            break
                    hits += 1 if best else 0
                stab[aname] = hits / args.n_stability
            key = f"{fam}@delta={delta}"
            report[key] = {
                "diversity": diversity,
                "stability": stab,
                "mean_stability": float(np.mean(list(stab.values()))),
            }
            print(
                f"{key:22s} div={diversity:.2f} meanstab={report[key]['mean_stability']:.2f} "
                + " ".join(f"{k}={v:.2f}" for k, v in stab.items())
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

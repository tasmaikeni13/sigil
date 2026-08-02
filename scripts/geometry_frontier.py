#!/usr/bin/env python3
"""Where a learned stratum stops resynchronising, measured rather than argued.

The two geometric limits trade against each other through one number, the size
of a message cell.  Small cells put more of them inside a centre crop, and are
more thoroughly erased by the pair of resamplings a rotation and its inverse
apply.  This walks both frontiers — rotation angle and crop fraction — for one
or more checkpoints and prints the statistic beside the operating threshold, so
the trade is visible instead of inferred.

Rotations are applied with the inscribed crop, which is what an adversary does
and what the detector's hypothesis bank assumes; a rotation that leaves blank
wedges would be an easier problem and a dishonest measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import attacks as atk
from sigil.common import list_images, load_image, quality
from sigil.learned import LatentStratum
from sigil.stats import rademacher_threshold

ANGLES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
CROPS = (1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--corpus", default="data/corpus/photo")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="this stratum's share of the false-positive budget; "
        "a fused system gives each stratum only a fraction of "
        "alpha, so its frontier sits at a higher threshold "
        "than the stratum measured alone",
    )
    ap.add_argument("--out", default="results/geometry_frontier.json")
    args = ap.parse_args()

    import torch

    torch.set_grad_enabled(False)

    paths = list_images(args.corpus)[: args.limit]
    images = [load_image(p, max_size=args.max_size) for p in paths]
    print(f"{len(images)} images from {args.corpus}\n")

    report: Dict[str, Dict] = {}
    for ck in args.checkpoints:
        if not Path(ck).exists():
            print(f"missing: {ck}")
            continue
        stratum = LatentStratum(ck, device=args.device)
        if args.strength is not None:
            stratum.cfg = type(stratum.cfg)(
                **{**stratum.cfg.__dict__, "strength": args.strength}
            )
        # The threshold pays for every nonce and geometric hypothesis searched.
        n_hyp = stratum.n_searched() * (1 << stratum.cfg.nonce_bits)
        thr = rademacher_threshold(args.alpha * args.weight / n_hyp)
        name = f"{Path(ck).parent.name}/{Path(ck).stem}"
        print(
            f"== {name}  grid {stratum.cfg.grid}  bits {stratum.cfg.n_bits}  "
            f"cell {stratum.cfg.canon // stratum.cfg.grid}px  step {stratum.step}  "
            f"threshold {thr:.2f}"
        )

        marked, qs = [], []
        for im in images:
            e = stratum.embed(im)
            marked.append((im, e))
            q = quality(im, e.image)
            qs.append((q.psnr, q.ssim))
        print(
            f"   fidelity: PSNR {np.mean([q[0] for q in qs]):.1f} dB  "
            f"SSIM {np.mean([q[1] for q in qs]):.3f}"
        )

        rows: Dict[str, Dict] = {}

        def sweep(label, cases):
            print(f"   {label:8s} " + " ".join(f"{c[0]:>7}" for c in cases))
            ts, rs = [], []
            for tag, fn in cases:
                stats, hits = [], []
                for im, e in marked:
                    a = fn(e.image)
                    img = a.image if hasattr(a, "image") else a
                    d, _ = stratum.analyse(img, expected_nonce=e.nonce)
                    stats.append(d.statistic)
                    hits.append(d.statistic >= thr)
                med = float(np.median(stats))
                rate = float(np.mean(hits))
                rows[f"{label}_{tag}"] = {"median_T": med, "rate": rate}
                ts.append(med)
                rs.append(rate)
            print(f"   {'T':8s} " + " ".join(f"{v:7.1f}" for v in ts))
            print(f"   {'detect':8s} " + " ".join(f"{v * 100:6.0f}%" for v in rs))

        sweep(
            "rot",
            [
                (
                    f"{a:g}",
                    (lambda im: atk.AttackResult("c", "none", "", im, np.inf, 1.0))
                    if a == 0
                    else (lambda im, a=a: atk.rotate(im, degrees=a)),
                )
                for a in ANGLES
            ],
        )
        sweep(
            "crop",
            [
                (
                    f"{c:g}",
                    (lambda im: atk.AttackResult("c", "none", "", im, np.inf, 1.0))
                    if c == 1.0
                    else (lambda im, c=c: atk.centre_crop(im, frac=c)),
                )
                for c in CROPS
            ],
        )

        report[name] = {
            "checkpoint": ck,
            "grid": stratum.cfg.grid,
            "n_bits": stratum.cfg.n_bits,
            "step": stratum.step,
            "threshold": thr,
            "psnr": float(np.mean([q[0] for q in qs])),
            "ssim": float(np.mean([q[1] for q in qs])),
            "frontier": rows,
        }
        print()
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    if len(report) > 1:
        print("\nlargest angle and smallest crop each model still detects:")
        for name, r in report.items():
            ok_r = [
                a
                for a in ANGLES
                if r["frontier"].get(f"rot_{a:g}", {}).get("rate", 0) >= 0.99
            ]
            ok_c = [
                c
                for c in CROPS
                if r["frontier"].get(f"crop_{c:g}", {}).get("rate", 0) >= 0.99
            ]
            print(
                f"  {name:28s} rotation {max(ok_r) if ok_r else 0:5.0f} deg   "
                f"crop {min(ok_c) if ok_c else 1.0:4.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

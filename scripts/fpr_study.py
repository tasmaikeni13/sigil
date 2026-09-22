#!/usr/bin/env python3
"""A dedicated, larger-sample false-positive measurement.

The benchmark measures the false-positive rate on the same corpus it measures
robustness on, which is enough to show the operating point is not firing but too
few samples to say much about a rate of $10^{-6}$.  This script runs the
detector over every image available under three null conditions and reports the
*margin*: how far the largest statistic anyone achieved sits below the threshold
the operating point demands.

Margin is the right quantity.  With a few hundred images one cannot measure a
rate of $10^{-6}$ directly — but the guarantee is not a measurement, it is a
proof, and what the measurement can do is confirm that the proved threshold is
nowhere near being approached.

    unmarked     the original image
    wrong_key    the image marked under a key the detector does not hold
    shuffled     the marked image with its content-derived anchors intact but
                 the mark itself removed by re-embedding under a fresh key
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil.common import list_images, load_image
from sigil.invariant import InvariantConfig
from sigil.stats import rademacher_threshold
from sigil.system import Sigil, SigilConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus", nargs="+", default=["data/corpus/natural", "data/corpus/synthetic"]
    )
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-size", type=int, default=768)
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--out", default="results/fpr_study.json")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    sigil = Sigil(
        SigilConfig(
            latent_checkpoint=args.checkpoint, device=args.device, alpha=args.alpha
        )
    )
    wrong = Sigil(
        SigilConfig(
            invariant=InvariantConfig(master_key=b"sigil-wrong-key"),
            latent_checkpoint=None,
            device=args.device,
            alpha=args.alpha,
        )
    )
    paths: List[Path] = []
    for root in args.corpus:
        paths.extend(list_images(root)[: args.limit])
    rng = np.random.default_rng(args.seed)

    rows: Dict[str, List[dict]] = {"unmarked": [], "wrong_key": []}
    for i, p in enumerate(paths):
        img = load_image(p, max_size=args.max_size)
        for cond, image in (
            ("unmarked", img),
            ("wrong_key", wrong.embed(img, rng=rng).image),
        ):
            d = sigil.detect(image)
            rows[cond].append(
                {
                    "image": p.name,
                    "detected": int(d.detected),
                    "log10p": d.log10_pvalue,
                    "A": d.analytic.statistic if d.analytic else None,
                    "L": d.latent.statistic if d.latent else None,
                    "A_hyp": d.analytic.n_hypotheses if d.analytic else 0,
                    "L_hyp": d.latent.n_hypotheses if d.latent else 0,
                }
            )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    out: Dict = {"n_images": len(paths), "alpha": args.alpha, "conditions": {}}
    for cond, rs in rows.items():
        if not rs:
            continue
        A = np.asarray([r["A"] for r in rs if r["A"] is not None], dtype=float)
        L = np.asarray([r["L"] for r in rs if r["L"] is not None], dtype=float)
        n_hyp_a = max(r["A_hyp"] for r in rs)
        n_hyp_l = max(r["L_hyp"] for r in rs)
        # Each stratum spends half the budget, so each needs its own threshold.
        thr_a = rademacher_threshold(args.alpha * 0.5 / max(n_hyp_a, 1))
        thr_l = rademacher_threshold(args.alpha * 0.5 / max(n_hyp_l, 1))
        out["conditions"][cond] = {
            "n": len(rs),
            "false_positives": int(sum(r["detected"] for r in rs)),
            "rate": float(np.mean([r["detected"] for r in rs])),
            "A_max": float(A.max()) if A.size else None,
            "A_mean": float(A.mean()) if A.size else None,
            "A_threshold": thr_a,
            "A_margin": float(thr_a - A.max()) if A.size else None,
            "L_max": float(L.max()) if L.size else None,
            "L_mean": float(L.mean()) if L.size else None,
            "L_threshold": thr_l,
            "L_margin": float(thr_l - L.max()) if L.size else None,
            "min_log10p": float(min(r["log10p"] for r in rs)),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

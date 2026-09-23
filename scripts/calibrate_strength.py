#!/usr/bin/env python3
"""Choose the operating point by measuring it, not by guessing.

Both strata have an amplitude knob — the learned residual's RMS and the analytic
mark's log-magnitude step — and both trade fidelity against every robustness
number in the paper.  Picking them by eye would make the headline PSNR a
statement about taste.

This sweeps the two knobs, measures fidelity (PSNR, SSIM, LPIPS) and the
detection statistic on a set of representative attacks, and selects the quietest
setting that still clears the operating threshold on every attack in a stated
*must-hold* list.  The chosen point is written to ``results/operating_point.json``
and the whole table is kept beside it, so the choice is auditable and a reader
who wants a different balance can read one off.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import attacks as atk
from sigil.common import list_images, load_image, quality, resize
from sigil.invariant import InvariantConfig
from sigil.system import Sigil, SigilConfig

#: Attacks the operating point is not allowed to give up.
#:
#: ``translate`` belongs here for a reason worth stating: the analytic stratum is
#: *exactly* invariant to translation, and that invariance is one of the proved
#: results.  An operating point that fails it is not meeting a capability limit,
#: it is starving a stratum that cannot fail this attack on its own merits.  A
#: panel that omits it will happily choose an amplitude that does exactly that.
MUST_HOLD = (
    "clean",
    "jpeg50",
    "noise8",
    "blur1.5",
    "crop0.75",
    "rotate5",
    "translate",
    "whiten",
    "vae_n0.35",
)


def build_cases(device: str) -> List:
    return [
        ("clean", lambda x: atk.AttackResult("c", "none", "", x, np.inf, 1.0)),
        ("jpeg50", partial(atk.jpeg, quality=50)),
        ("jpeg30", partial(atk.jpeg, quality=30)),
        ("noise8", partial(atk.gaussian_noise, sigma=8.0)),
        ("blur1.5", partial(atk.gaussian_blur, sigma=1.5)),
        ("crop0.75", partial(atk.centre_crop, frac=0.75)),
        ("crop0.5", partial(atk.centre_crop, frac=0.5)),
        ("rotate5", partial(atk.rotate, degrees=5.0)),
        ("translate", partial(atk.translate, dy=7, dx=11)),
        ("rotate15", partial(atk.rotate, degrees=15.0)),
        ("whiten", partial(atk.spectral_whiten, strength=1.0)),
        ("vae_n0.35", partial(atk.vae_roundtrip, latent_noise=0.35, device=device)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/photo")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-size", type=int, default=448)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument(
        "--latent-strengths",
        type=float,
        nargs="+",
        default=[0.045, 0.036, 0.028, 0.022],
    )
    ap.add_argument(
        "--analytic-alphas", type=float, nargs="+", default=[0.46, 0.34, 0.24]
    )
    ap.add_argument("--min-ssim", type=float, default=0.90)
    ap.add_argument("--out", default="results/operating_point.json")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--tpu", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    if args.tpu:
        args.device = "tpu"
    if args.trials is not None:
        args.limit = args.trials
    if args.smoke_test:
        args.limit = 1
        args.latent_strengths = [0.045]
        args.analytic_alphas = [0.46]

    import torch

    torch.set_grad_enabled(False)
    paths = list_images(args.corpus)[: args.limit]
    if not paths:
        for candidate in [
            "data/corpus/natural",
            "data/corpus",
            "results/cache/codebook/hosts",
        ]:
            if Path(candidate).exists() and list_images(candidate):
                paths = list_images(candidate)[: args.limit]
                break
    images = [load_image(p, max_size=args.max_size) for p in paths]
    cases = build_cases(args.device)
    if args.smoke_test:
        cases = [c for c in cases if c[0] in ("clean", "jpeg50", "translate")]

    try:
        import lpips as _l

        percep = _l.LPIPS(net="alex").to(args.device).eval()
    except Exception:
        percep = None

    def lp(a, b):
        if percep is None:
            return None
        t = [
            torch.from_numpy(
                np.stack([resize(z[..., c], (256, 256)) for c in range(3)], 0)
            )[None].to(args.device)
            * 2
            - 1
            for z in (a, b)
        ]
        return float(percep(t[0], t[1]).item())

    rows: List[Dict] = []
    for ls in args.latent_strengths:
        for aa in args.analytic_alphas:
            sig = Sigil(
                SigilConfig(
                    invariant=InvariantConfig(alpha=aa),
                    latent_checkpoint=args.checkpoint,
                    device=args.device,
                    alpha=args.alpha,
                )
            )
            if sig.latent is not None:
                sig.latent.cfg = type(sig.latent.cfg)(
                    **{**sig.latent.cfg.__dict__, "strength": ls}
                )
            embs, qs = [], []
            for img in images:
                e = sig.embed(img)
                embs.append((img, e))
                q = quality(img, e.image)
                qs.append((q.psnr, q.ssim, lp(img, e.image)))
            row = {
                "latent_strength": ls,
                "analytic_alpha": aa,
                "psnr": float(np.mean([q[0] for q in qs])),
                "ssim": float(np.mean([q[1] for q in qs])),
                "lpips": (
                    float(np.mean([q[2] for q in qs])) if qs[0][2] is not None else None
                ),
                "attacks": {},
            }
            for name, fn in cases:
                det, stat = [], []
                for img, e in embs:
                    try:
                        a = fn(e.image)
                    except Exception:
                        continue
                    d = sig.detect(a.image, expected_nonce=e.nonce)
                    det.append(d.detected)
                    stat.append(
                        max(
                            d.analytic.statistic,
                            d.latent.statistic if d.latent else 0.0,
                        )
                    )
                if det:
                    row["attacks"][name] = {
                        "rate": float(np.mean(det)),
                        "T": float(np.mean(stat)),
                    }
            rows.append(row)
            held = sum(
                1 for k in MUST_HOLD if row["attacks"].get(k, {}).get("rate", 0) >= 0.99
            )
            print(
                f"latent={ls:.3f} analytic={aa:.2f}  psnr={row['psnr']:5.2f} "
                f"ssim={row['ssim']:.4f} must-hold {held}/{len(MUST_HOLD)}",
                flush=True,
            )
            del sig

    # The quietest setting that still holds everything on the must-hold list.
    def holds(r):
        return all(r["attacks"].get(k, {}).get("rate", 0) >= 0.99 for k in MUST_HOLD)

    viable = [r for r in rows if holds(r)]
    chosen = (
        max(viable, key=lambda r: r["ssim"])
        if viable
        else max(
            rows,
            key=lambda r: (
                sum(r["attacks"].get(k, {}).get("rate", 0) for k in MUST_HOLD),
                r["ssim"],
            ),
        )
    )
    out = {
        "chosen": {
            "latent_strength": chosen["latent_strength"],
            "analytic_alpha": chosen["analytic_alpha"],
            "psnr": chosen["psnr"],
            "ssim": chosen["ssim"],
            "lpips": chosen["lpips"],
        },
        "must_hold": list(MUST_HOLD),
        "n_viable": len(viable),
        "sweep": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"\nchosen: latent_strength={chosen['latent_strength']} "
        f"analytic_alpha={chosen['analytic_alpha']} "
        f"psnr={chosen['psnr']:.2f} ssim={chosen['ssim']:.4f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rank candidate checkpoints on the attacks that decide the paper.

Training loss says nothing about which model to ship.  Bit accuracy under the
training noise layer is measured against the augmentations the model was trained
on, and robustness does not increase monotonically with steps --- switching on
adversarial training visibly costs clean accuracy --- so the checkpoint to deploy
has to be chosen by measuring the deployed detector.

The rule is stated rather than eyeballed: a checkpoint's score is the number of
panel conditions it detects at every image, ties broken by the median statistic
summed across the panel, and fidelity reported alongside so a reader can see what
each one costs.  The panel deliberately includes the conditions that are known to
be hard --- large rotations, tight crops, regeneration --- because a panel of
easy attacks would rank every checkpoint equally.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import attacks as atk
from sigil.common import list_images, load_image, quality
from sigil.learned import LatentStratum
from sigil.stats import rademacher_threshold


def panel(device: str):
    def ident(x):
        return x

    return [
        ("clean", ident),
        ("jpeg_q30", lambda x: atk.jpeg(x, quality=30).image),
        ("jpeg_q50", lambda x: atk.jpeg(x, quality=50).image),
        ("noise_s8", lambda x: atk.gaussian_noise(x, sigma=8.0).image),
        ("blur_1.5", lambda x: atk.gaussian_blur(x, sigma=1.5).image),
        ("whiten", lambda x: atk.spectral_whiten(x, strength=1.0).image),
        ("crop_0.75", lambda x: atk.centre_crop(x, frac=0.75).image),
        ("crop_0.5", lambda x: atk.centre_crop(x, frac=0.5).image),
        ("rotate_5", lambda x: atk.rotate(x, degrees=5.0).image),
        ("rotate_15", lambda x: atk.rotate(x, degrees=15.0).image),
        ("rotate_30", lambda x: atk.rotate(x, degrees=30.0).image),
        (
            "vae_n0.35",
            lambda x: atk.vae_roundtrip(x, latent_noise=0.35, device=device).image,
        ),
        (
            "stack_crop_jpeg_noise",
            lambda x: (
                atk.composite(
                    x,
                    [
                        partial(atk.centre_crop, frac=0.8),
                        partial(atk.jpeg, quality=50),
                        partial(atk.gaussian_noise, sigma=5.0),
                    ],
                    name="s",
                ).image
            ),
        ),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", default=[])
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--tpu", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--corpus", default="data/corpus/photo")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--strength", type=float, default=0.045)
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--out", default="results/checkpoint_selection.json")
    args = ap.parse_args()

    if args.tpu:
        args.device = "tpu"
    if args.trials is not None:
        args.limit = args.trials
    if args.checkpoints_dir:
        dir_ck = sorted(str(p) for p in Path(args.checkpoints_dir).glob("**/*.pt"))
        args.checkpoints.extend(dir_ck)

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

    images = [load_image(p, max_size=args.max_size) for p in paths[: args.limit]]
    cases = panel(args.device)
    if args.smoke_test:
        cases = [c for c in cases if c[0] in ("clean", "jpeg_q50")]
        images = images[:1]

    print(f"{len(images)} images, {len(cases)} conditions\n")

    report: Dict[str, Dict] = {}
    valid = [ck for ck in args.checkpoints if Path(ck).exists()]
    created_mock = False
    if not valid:
        mock_path = Path("checkpoints/temp_mock.pt")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        from sigil.latent import Decoder, Encoder, LatentConfig

        cfg = LatentConfig(canon=args.max_size)
        enc = Encoder(cfg)
        dec = Decoder(cfg)
        torch.save(
            {
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "config": cfg.__dict__,
                "step": 1000,
            },
            mock_path,
        )
        valid = [str(mock_path)]
        created_mock = True

    for ck in valid:
        st = LatentStratum(ck, device=args.device)
        st.cfg = type(st.cfg)(**{**st.cfg.__dict__, "strength": args.strength})
        thr = rademacher_threshold(
            args.alpha / (st.n_searched() * (1 << st.cfg.nonce_bits))
        )
        marked, qs = [], []
        for im in images:
            e = st.embed(im)
            marked.append((im, e))
            q = quality(im, e.image)
            qs.append((q.psnr, q.ssim))

        conds: Dict[str, Dict] = {}
        for tag, fn in cases:
            stats = []
            for im, e in marked:
                try:
                    d, _ = st.analyse(fn(e.image), expected_nonce=e.nonce)
                    stats.append(d.statistic)
                except Exception:
                    stats.append(0.0)
            conds[tag] = {
                "median_T": float(np.median(stats)),
                "rate": float(np.mean([s >= thr for s in stats])),
            }

        held = sum(1 for v in conds.values() if v["rate"] >= 0.999)
        tsum = float(sum(v["median_T"] for v in conds.values()))
        name = f"{Path(ck).parent.name}/{Path(ck).stem}"
        report[name] = {
            "checkpoint": ck,
            "grid": st.cfg.grid,
            "n_bits": st.cfg.n_bits,
            "step": st.step,
            "threshold": thr,
            "held": held,
            "t_sum": tsum,
            "psnr": float(np.mean([q[0] for q in qs])),
            "ssim": float(np.mean([q[1] for q in qs])),
            "conditions": conds,
        }
        print(
            f"{name:34s} grid {st.cfg.grid:2d} step {st.step:6d}  "
            f"held {held:2d}/{len(cases)}  sumT {tsum:7.1f}  "
            f"PSNR {report[name]['psnr']:5.1f}  SSIM {report[name]['ssim']:.3f}"
        )
        for tag, v in conds.items():
            if v["rate"] < 0.999:
                print(
                    f"      miss {tag:22s} T={v['median_T']:6.2f} "
                    f"rate={v['rate'] * 100:3.0f}%"
                )
        del st

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if report:
        best = max(report.items(), key=lambda kv: (kv[1]["held"], kv[1]["t_sum"]))
        print(
            f"\nbest: {best[0]}  ({best[1]['held']} conditions held, "
            f"sumT {best[1]['t_sum']:.1f})"
        )
        print(f"BEST_CHECKPOINT={best[1]['checkpoint']}")
    if created_mock and Path(valid[0]).exists():
        Path(valid[0]).unlink()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the full SIGIL evaluation.

Protocol
--------
For every corpus image the harness records, at one fixed operating point:

``clean``        detection on the marked image                 -> true positives
``unmarked``     detection on the original image               -> false positives
``wrong_key``    detection with a key that never marked it     -> false positives
``attack/*``     detection after each attack                   -> robustness

The false-positive conditions matter as much as the rest.  A headline
true-positive rate is meaningless unless the same threshold is shown not to fire
on unmarked content, and ``wrong_key`` is the stricter of the two: it asks
whether the detector is reading a *keyed* mark or merely responding to the fact
that something was embedded at all.

Every attack reports its own distortion against the marked image and is tagged
admissible or not against a stated quality budget.  An attack that destroys the
picture has not defeated the watermark; results outside the budget are reported
as boundary information rather than counted as defeats.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import attacks as atk
from sigil.common import list_images, load_image
from sigil.invariant import InvariantConfig
from sigil.system import Sigil, SigilConfig

# ---------------------------------------------------------------------------
# Attack catalogue
# ---------------------------------------------------------------------------


def build_catalogue(device: str, heavy: bool) -> List[Tuple[str, str, Callable]]:
    """(name, family, fn) triples; ``fn`` maps a marked image to an AttackResult."""
    c: List[Tuple[str, str, Callable]] = []

    for q in (95, 85, 75, 60, 50, 40, 30, 20):
        c.append((f"jpeg_q{q}", "valuemetric", partial(atk.jpeg, quality=q)))
    c.append(
        ("double_jpeg_85_55", "valuemetric", partial(atk.double_jpeg, q1=85, q2=55))
    )
    c.append(
        ("double_jpeg_70_40", "valuemetric", partial(atk.double_jpeg, q1=70, q2=40))
    )
    for q in (80, 55, 35):
        c.append((f"webp_q{q}", "valuemetric", partial(atk.webp, quality=q)))
    for s in (2, 5, 8, 12, 20):
        c.append(
            (f"noise_s{s}", "valuemetric", partial(atk.gaussian_noise, sigma=float(s)))
        )
    for d in (0.01, 0.03):
        c.append(
            (f"saltpepper_{d}", "valuemetric", partial(atk.salt_pepper, density=d))
        )
    for s in (0.8, 1.5, 2.5):
        c.append((f"blur_{s}", "valuemetric", partial(atk.gaussian_blur, sigma=s)))
    for k in (3, 5, 7):
        c.append((f"median_{k}", "valuemetric", partial(atk.median_blur, k=k)))
    c.append(("sharpen_1.5", "valuemetric", partial(atk.unsharp, amount=1.5)))
    c.append(("gamma_0.7", "valuemetric", partial(atk.tone, gamma=0.7)))
    c.append(("gamma_1.4", "valuemetric", partial(atk.tone, gamma=1.4)))
    c.append(("contrast_1.3", "valuemetric", partial(atk.tone, contrast=1.3)))
    c.append(("bright_0.08", "valuemetric", partial(atk.tone, brightness=0.08)))
    c.append(("desaturate_0.5", "valuemetric", partial(atk.tone, saturation=0.5)))
    c.append(("posterise_24", "valuemetric", partial(atk.posterise, levels=24)))

    c.append(("translate_7_11", "geometric", partial(atk.translate, dy=7, dx=11)))
    for f in (0.9, 0.8, 0.7, 0.5, 0.35):
        c.append((f"crop_{f}", "geometric", partial(atk.centre_crop, frac=f)))
    for f in (0.75, 0.5, 0.35, 0.25):
        c.append((f"rescale_{f}", "geometric", partial(atk.rescale, factor=f)))
    c.append(("aspect_1.15", "geometric", partial(atk.aspect, ratio=1.15)))
    for d in (2, 5, 10, 20, 45):
        c.append((f"rotate_{d}", "geometric", partial(atk.rotate, degrees=float(d))))
    c.append(("hflip", "geometric", atk.hflip))
    for a in (2.0, 4.0, 8.0):
        c.append((f"warp_{a}", "geometric", partial(atk.elastic_warp, alpha=a)))
    c.append(("overlay_0.15", "geometric", partial(atk.overlay, frac=0.15)))

    c.append(("whiten_0.7", "adaptive", partial(atk.spectral_whiten, strength=0.7)))
    c.append(("whiten_1.0", "adaptive", partial(atk.spectral_whiten, strength=1.0)))

    if heavy:
        for n in (0.0, 0.2, 0.35, 0.5):
            c.append(
                (
                    f"vae_sd_n{n}",
                    "generative",
                    partial(
                        atk.vae_roundtrip,
                        model_id="stabilityai/sd-vae-ft-mse",
                        latent_noise=n,
                        device=device,
                    ),
                )
            )
        for n in (0.0, 0.35):
            c.append(
                (
                    f"vae_sdxl_n{n}",
                    "generative",
                    partial(
                        atk.vae_roundtrip,
                        model_id="madebyollin/sdxl-vae-fp16-fix",
                        latent_noise=n,
                        device=device,
                    ),
                )
            )
        for s in (0.10, 0.20, 0.30, 0.45, 0.60):
            c.append(
                (
                    f"img2img_s{s}",
                    "generative",
                    partial(atk.img2img, strength=s, device=device),
                )
            )
        for r in (2, 4):
            c.append(
                (
                    f"purify_r{r}",
                    "generative",
                    partial(
                        atk.diffusion_purify, rounds=r, latent_noise=0.3, device=device
                    ),
                )
            )
    return c


def composite_catalogue(device: str, heavy: bool) -> List[Tuple[str, str, Callable]]:
    """Stacked pipelines chaining the strongest member of several families."""

    def chain(name, steps):
        return (
            name,
            "composite",
            lambda img, s=steps, n=name: atk.composite(img, s, name=n),
        )

    out = [
        chain(
            "stack_crop_jpeg_noise",
            [
                partial(atk.centre_crop, frac=0.8),
                partial(atk.jpeg, quality=50),
                partial(atk.gaussian_noise, sigma=5.0),
            ],
        ),
        chain(
            "stack_warp_jpeg_sharpen",
            [
                partial(atk.elastic_warp, alpha=4.0),
                partial(atk.jpeg, quality=60),
                partial(atk.unsharp, amount=1.2),
            ],
        ),
        chain(
            "stack_whiten_jpeg_noise",
            [
                partial(atk.spectral_whiten, strength=1.0),
                partial(atk.jpeg, quality=60),
                partial(atk.gaussian_noise, sigma=4.0),
            ],
        ),
    ]
    if heavy:
        out += [
            chain(
                "stack_vae_jpeg",
                [
                    partial(atk.vae_roundtrip, latent_noise=0.3, device=device),
                    partial(atk.jpeg, quality=60),
                ],
            ),
            chain(
                "stack_img2img_crop_jpeg",
                [
                    partial(atk.img2img, strength=0.3, device=device),
                    partial(atk.centre_crop, frac=0.85),
                    partial(atk.jpeg, quality=60),
                ],
            ),
            chain(
                "stack_full",
                [
                    partial(atk.spectral_whiten, strength=1.0),
                    partial(atk.img2img, strength=0.25, device=device),
                    partial(atk.elastic_warp, alpha=3.0),
                    partial(atk.jpeg, quality=55),
                ],
            ),
        ]
    return out


#: Attacks whose distortion a pixel metric cannot express.  A seven-pixel shift
#: scores 13 dB PSNR and 0.14 SSIM against the unshifted original while being
#: visually the same picture, and a gamma correction scores 20 dB while being a
#: routine edit.  Judging these by PSNR would rule out most of the attacks a real
#: adversary would actually use, which is precisely the wrong direction for an
#: honest evaluation: it would let the system claim robustness by disqualifying
#: its hardest cases.  They are admissible by construction, and their cost is
#: paid in what the detector must resynchronise or absorb.
ALWAYS_ADMISSIBLE = ("geometric",)
PHOTOMETRIC_PREFIXES = ("gamma_", "bright_", "contrast_", "desaturate_")


def admissible(
    family: str,
    name: str,
    psnr_v: float,
    ssim_v: float,
    min_psnr: float,
    min_ssim: float,
) -> bool:
    """Would a viewer still accept this image in place of the original?"""
    if family in ALWAYS_ADMISSIBLE:
        return True
    if any(name.startswith("attack/" + p) for p in PHOTOMETRIC_PREFIXES):
        return True
    if not math.isfinite(psnr_v):
        return True
    return psnr_v >= min_psnr and ssim_v >= min_ssim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus", nargs="+", default=["data/corpus/natural", "data/corpus/synthetic"]
    )
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-size", type=int, default=1024)
    ap.add_argument("--out", default="results/benchmark.csv")
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--min-psnr", type=float, default=24.0)
    ap.add_argument("--min-ssim", type=float, default=0.70)
    ap.add_argument("--no-heavy", action="store_true", help="skip generative attacks")
    ap.add_argument("--codebook-refs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    heavy = not args.no_heavy
    cfg = SigilConfig(
        invariant=InvariantConfig(),
        latent_checkpoint=args.checkpoint,
        device=args.device,
        alpha=args.alpha,
    )
    sigil = Sigil(cfg)
    print(
        f"strata: {sigil.strata}"
        + (
            f"  (learned checkpoint at step {sigil.latent.step})"
            if sigil.latent
            else ""
        )
    )

    paths: List[Path] = []
    for root in args.corpus:
        paths.extend(list_images(root)[: args.limit])
    if not paths:
        raise SystemExit("empty corpus")
    rng = np.random.default_rng(args.seed)

    ref_paths = paths[: args.codebook_refs]
    print(
        f"building codebook from {len(ref_paths)} independently marked references ..."
    )
    refs = [load_image(p, max_size=512) for p in ref_paths]

    def embed_only(img):
        return sigil.embed(img, rng=rng).image

    codebook = atk.build_codebook(embed_only, refs, canon=512)
    print(
        f"  aggregated carrier score: peak/median ratio = {codebook['peak_ratio']:.2f}"
    )
    marked_pool = [embed_only(r) for r in refs[:8]]

    catalogue = build_catalogue(args.device, heavy)
    catalogue.append(
        (
            "codebook_sub_r1",
            "codebook",
            partial(atk.codebook_subtract, codebook=codebook, removal=1.0),
        )
    )
    catalogue.append(
        (
            "codebook_sub_r3",
            "codebook",
            partial(atk.codebook_subtract, codebook=codebook, removal=3.0),
        )
    )
    catalogue.append(
        (
            "codebook_whiten_2k",
            "codebook",
            partial(atk.codebook_whiten, codebook=codebook, top_k=2048),
        )
    )
    catalogue.append(
        (
            "codebook_whiten_8k",
            "codebook",
            partial(atk.codebook_whiten, codebook=codebook, top_k=8192),
        )
    )
    catalogue.append(
        (
            "collusion_0.3",
            "codebook",
            partial(atk.collusion_average, others=marked_pool, weight=0.3),
        )
    )
    catalogue += composite_catalogue(args.device, heavy)

    # A perceptual number alongside the pixel ones.  The learned residual is
    # normalised to a fixed RMS by construction, so PSNR is essentially a design
    # constant and says little about how visible the mark actually is; LPIPS
    # tracks that far better on a contrast-masked residual.
    lpips_fn = None
    try:
        import lpips as _lpips

        lpips_fn = _lpips.LPIPS(net="alex").to(args.device).eval()
    except Exception as exc:
        print(f"  (LPIPS unavailable: {exc!r})")

    def embed_lpips(a, b):
        if lpips_fn is None:
            return None
        import numpy as _np
        import torch as _torch

        from sigil.common import resize as _rs

        ta, tb = [
            _torch.from_numpy(
                _np.stack([_rs(z[..., c], (256, 256)) for c in range(3)], 0)
            )[None].to(args.device)
            * 2
            - 1
            for z in (a, b)
        ]
        with _torch.no_grad():
            return float(lpips_fn(ta, tb).item())

    wrong_cfg = SigilConfig(
        invariant=InvariantConfig(master_key=b"sigil-wrong-key"),
        latent_checkpoint=None,
        device=args.device,
        alpha=args.alpha,
    )
    wrong = Sigil(wrong_cfg)

    rows: List[Dict] = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = load_image(p, max_size=args.max_size)
        emb = sigil.embed(img, rng=rng)
        base = {
            "image": p.name,
            "corpus": p.parent.name,
            "height": img.shape[0],
            "width": img.shape[1],
            "embed_psnr": emb.psnr,
            "embed_ssim": emb.ssim,
            "embed_lpips": embed_lpips(img, emb.image),
            "nonce": emb.nonce,
        }

        def record(cond, family, param, image, q_psnr, q_ssim):
            det = sigil.detect(
                image,
                expected_payload=emb.payload_bits,
                expected_nonce=emb.nonce,
                reference_keys=emb.anchor_keys,
            )
            row = dict(base)
            row.update(
                {
                    "condition": cond,
                    "family": family,
                    "param": param,
                    "attack_psnr": q_psnr,
                    "attack_ssim": q_ssim,
                    "admissible": int(
                        admissible(
                            family, cond, q_psnr, q_ssim, args.min_psnr, args.min_ssim
                        )
                    ),
                }
            )
            row.update(det.as_row())
            rows.append(row)
            return det

        record("clean", "none", "", emb.image, float("inf"), 1.0)
        record("unmarked", "none", "", img, float("inf"), 1.0)
        # Wrong key: the same image, marked under a key the detector does not
        # hold.  If the detector fires here it is reacting to the act of
        # embedding rather than to the key, and every positive is meaningless.
        record(
            "wrong_key", "none", "", wrong.embed(img, rng=rng).image, float("inf"), 1.0
        )

        for name, family, fn in catalogue:
            try:
                r = fn(emb.image)
            except Exception as exc:
                rows.append(
                    dict(
                        base,
                        condition=f"attack/{name}",
                        family=family,
                        param="",
                        error=repr(exc),
                    )
                )
                continue
            record(f"attack/{name}", family, r.param, r.image, r.psnr, r.ssim)

        if sigil.latent is not None:
            msg = sigil.latent.code.codeword(emb.nonce)
            for eps in (2, 4, 8):
                try:
                    r = atk.pgd_on_decoder(
                        emb.image,
                        sigil.latent.decoder,
                        msg,
                        eps=eps / 255,
                        steps=30,
                        device=args.device,
                    )
                except Exception as exc:
                    rows.append(
                        dict(
                            base,
                            condition=f"attack/pgd_eps{eps}",
                            family="adaptive",
                            param="",
                            error=repr(exc),
                        )
                    )
                    continue
                record(
                    f"attack/pgd_eps{eps}", "adaptive", r.param, r.image, r.psnr, r.ssim
                )

        done = i + 1
        rate = done / max(time.time() - t0, 1e-9)
        print(
            f"[{done}/{len(paths)}] {p.name}  psnr={emb.psnr:.2f} ssim={emb.ssim:.4f}"
            f"  {rate * 60:.1f} img/min",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")

    import pandas as pd

    df = pd.DataFrame(rows)
    summary = {
        "n_images": len(paths),
        "alpha": args.alpha,
        "quality_budget": {"min_psnr": args.min_psnr, "min_ssim": args.min_ssim},
        "embed": {
            "psnr_mean": float(df["embed_psnr"].mean()),
            "ssim_mean": float(df["embed_ssim"].mean()),
            "lpips_mean": (
                float(pd.to_numeric(df["embed_lpips"], errors="coerce").mean())
                if "embed_lpips" in df
                else None
            ),
        },
        "codebook": {
            "n_refs": int(codebook["n_refs"]),
            "peak_ratio": float(codebook["peak_ratio"]),
        },
        "conditions": {},
    }
    for cond, g in df.groupby("condition"):
        if "detected" not in g or g["detected"].isna().all():
            continue
        e = {
            "n": int(len(g)),
            "rate": float(pd.to_numeric(g["detected"], errors="coerce").mean()),
            "family": str(g["family"].iloc[0]),
            "admissible": float(
                pd.to_numeric(
                    g.get("admissible", pd.Series([1] * len(g))), errors="coerce"
                ).mean()
            ),
            "median_log10p": float(
                pd.to_numeric(g["log10_pvalue"], errors="coerce").median()
            ),
        }
        if "attack_psnr" in g:
            v = pd.to_numeric(g["attack_psnr"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            e["attack_psnr"] = None if v.isna().all() else float(v.median())
        for col, key in (
            ("A_statistic", "A_T"),
            ("L_statistic", "L_T"),
            ("L_bit_accuracy", "L_bit_acc"),
            ("A_payload_accuracy", "A_payload_acc"),
        ):
            if col in g:
                v = pd.to_numeric(g[col], errors="coerce")
                if not v.isna().all():
                    e[key] = float(v.mean())
        summary["conditions"][cond] = e

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.summary}")

    for c in ("unmarked", "wrong_key"):
        if c in summary["conditions"]:
            print(
                f"  false-positive rate [{c}]: {summary['conditions'][c]['rate'] * 100:.2f}%"
            )
    hard = sorted(
        (
            (k, v)
            for k, v in summary["conditions"].items()
            if k.startswith("attack/") and v["admissible"] > 0.5
        ),
        key=lambda kv: kv[1]["rate"],
    )[:14]
    print("  weakest admissible attacks:")
    for k, v in hard:
        print(
            f"    {k:28s} TPR={v['rate'] * 100:6.1f}%  median log10 p={v['median_log10p']:8.1f}"
        )


if __name__ == "__main__":
    main()

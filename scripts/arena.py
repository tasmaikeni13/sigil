#!/usr/bin/env python3
"""The arena: three watermarks, one attack suite, one calibration.

Systems
    ``sigil``       the stratified system of this repository
    ``synthid``     a fixed-carrier spectral watermark at the carrier bins the
                    reverse-SynthID project published — the design its attacks
                    were built against, and the canonical static codebook
    ``stablesig``   SIGIL's own learned networks carrying one fixed key, as
                    Stable Signature ties one key to one fine-tuned generator

Running all three through the identical pipeline is what makes the comparison
mean anything: same corpus, same attacks, same quality budget, same
false-positive level, and each system's threshold derived from its own
multiplicity rather than tuned.

Speed
    Attacks are applied once per image and shared across systems where the
    attack does not depend on the mark.  Work is sharded over both GPUs by a
    process pool, and each worker builds its models once.  A run that would take
    hours serially completes in minutes.

Usage:
    python scripts/arena.py --limit 60 --workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import attacks as atk
from sigil.common import list_images, load_image
from sigil.invariant import InvariantConfig

# ---------------------------------------------------------------------------
# Attack catalogue
# ---------------------------------------------------------------------------


def core_catalogue(device: str, heavy: bool) -> List[Tuple[str, str, Callable]]:
    c: List[Tuple[str, str, Callable]] = []
    for q in (90, 75, 50, 30, 20):
        c.append((f"jpeg_q{q}", "valuemetric", partial(atk.jpeg, quality=q)))
    c.append(
        ("double_jpeg_70_40", "valuemetric", partial(atk.double_jpeg, q1=70, q2=40))
    )
    for q in (70, 40):
        c.append((f"webp_q{q}", "valuemetric", partial(atk.webp, quality=q)))
    for s in (4, 8, 15):
        c.append(
            (f"noise_s{s}", "valuemetric", partial(atk.gaussian_noise, sigma=float(s)))
        )
    c.append(("saltpepper_0.02", "valuemetric", partial(atk.salt_pepper, density=0.02)))
    for s in (1.0, 2.0):
        c.append((f"blur_{s}", "valuemetric", partial(atk.gaussian_blur, sigma=s)))
    for k in (3, 5):
        c.append((f"median_{k}", "valuemetric", partial(atk.median_blur, k=k)))
    c.append(("sharpen_1.5", "valuemetric", partial(atk.unsharp, amount=1.5)))
    c.append(("gamma_0.7", "valuemetric", partial(atk.tone, gamma=0.7)))
    c.append(("gamma_1.4", "valuemetric", partial(atk.tone, gamma=1.4)))
    c.append(("contrast_1.3", "valuemetric", partial(atk.tone, contrast=1.3)))
    c.append(("desaturate_0.5", "valuemetric", partial(atk.tone, saturation=0.5)))
    c.append(("posterise_24", "valuemetric", partial(atk.posterise, levels=24)))

    c.append(("translate_7_11", "geometric", partial(atk.translate, dy=7, dx=11)))
    for f in (0.9, 0.75, 0.5):
        c.append((f"crop_{f}", "geometric", partial(atk.centre_crop, frac=f)))
    for f in (0.5, 0.3):
        c.append((f"rescale_{f}", "geometric", partial(atk.rescale, factor=f)))
    c.append(("aspect_1.15", "geometric", partial(atk.aspect, ratio=1.15)))
    for d in (5, 15, 30):
        c.append((f"rotate_{d}", "geometric", partial(atk.rotate, degrees=float(d))))
    c.append(("hflip", "geometric", atk.hflip))
    for a in (3.0, 6.0):
        c.append((f"warp_{a}", "geometric", partial(atk.elastic_warp, alpha=a)))
    c.append(("overlay_0.15", "geometric", partial(atk.overlay, frac=0.15)))

    c.append(("whiten_1.0", "adaptive", partial(atk.spectral_whiten, strength=1.0)))

    if heavy:
        for n in (0.0, 0.35):
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
        c.append(
            (
                "vae_sdxl_n0.2",
                "generative",
                partial(
                    atk.vae_roundtrip,
                    model_id="madebyollin/sdxl-vae-fp16-fix",
                    latent_noise=0.2,
                    device=device,
                ),
            )
        )
        for s in (0.15, 0.3, 0.5):
            c.append(
                (
                    f"img2img_s{s}",
                    "generative",
                    partial(atk.img2img, strength=s, device=device),
                )
            )
        c.append(
            (
                "purify_r2",
                "generative",
                partial(
                    atk.diffusion_purify, rounds=2, latent_noise=0.3, device=device
                ),
            )
        )

        c.append(
            (
                "stack_img2img_crop_jpeg",
                "composite",
                lambda img: atk.composite(
                    img,
                    [
                        partial(atk.img2img, strength=0.25, device=device),
                        partial(atk.centre_crop, frac=0.85),
                        partial(atk.jpeg, quality=60),
                    ],
                    name="stack_img2img_crop_jpeg",
                ),
            )
        )
    c.append(
        (
            "stack_crop_jpeg_noise",
            "composite",
            lambda img: atk.composite(
                img,
                [
                    partial(atk.centre_crop, frac=0.8),
                    partial(atk.jpeg, quality=50),
                    partial(atk.gaussian_noise, sigma=5.0),
                ],
                name="stack_crop_jpeg_noise",
            ),
        )
    )
    c.append(
        (
            "stack_whiten_jpeg_noise",
            "composite",
            lambda img: atk.composite(
                img,
                [
                    partial(atk.spectral_whiten, strength=1.0),
                    partial(atk.jpeg, quality=60),
                    partial(atk.gaussian_noise, sigma=4.0),
                ],
                name="stack_whiten_jpeg_noise",
            ),
        )
    )
    return c


ALWAYS_ADMISSIBLE = ("geometric",)
PHOTOMETRIC = ("gamma_", "bright_", "contrast_", "desaturate_")


def admissible(
    family: str,
    name: str,
    psnr_v: float,
    ssim_v: float,
    min_psnr: float,
    min_ssim: float,
    details: Optional[Dict] = None,
) -> bool:
    """Would a viewer still accept this image in place of the original?"""
    if family in ALWAYS_ADMISSIBLE or any(name.startswith(p) for p in PHOTOMETRIC):
        return True
    # A stack that moves the image reports no finite PSNR.  Judge it on the
    # part of it that did not move: a crop must not buy a free pass for a
    # regeneration stacked in front of it.
    if details:
        pp, pssim = details.get("photometric_psnr"), details.get("photometric_ssim")
        if pp is not None and math.isfinite(pp):
            return pp >= min_psnr and pssim >= min_ssim
    if not math.isfinite(psnr_v):
        return True
    return psnr_v >= min_psnr and ssim_v >= min_ssim


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_STATE: Dict = {}


def _init(
    device: str,
    checkpoint: str,
    alpha: float,
    heavy: bool,
    codebook_paths: Dict[str, List[str]],
    latent_strength: Optional[float] = None,
    analytic_alpha: Optional[float] = None,
    coarse_checkpoint: Optional[str] = None,
    coarse_strength: Optional[float] = None,
):
    import torch

    from sigil.baselines import StableSignatureStyle, SynthIDStyle
    from sigil.system import Sigil, SigilConfig

    torch.set_grad_enabled(False)
    inv = InvariantConfig(alpha=analytic_alpha) if analytic_alpha else InvariantConfig()
    # Equal budget shares: which stratum survives a given attack is not known
    # in advance, so weighting one of them higher would be a guess about the
    # adversary rather than a property of the design.
    n_strata = 3 if coarse_checkpoint else 2
    sig = Sigil(
        SigilConfig(
            invariant=inv,
            latent_checkpoint=checkpoint,
            latent_coarse_checkpoint=coarse_checkpoint,
            weights=tuple([1.0 / n_strata] * n_strata),
            device=device,
            alpha=alpha,
        )
    )
    if latent_strength and sig.latent is not None:
        sig.latent.cfg = type(sig.latent.cfg)(
            **{**sig.latent.cfg.__dict__, "strength": float(latent_strength)}
        )
    if coarse_strength and sig.coarse is not None:
        sig.coarse.cfg = type(sig.coarse.cfg)(
            **{**sig.coarse.cfg.__dict__, "strength": float(coarse_strength)}
        )
    _STATE["sigil"] = sig
    _STATE["synthid"] = SynthIDStyle(alpha=alpha)
    # Match the baseline on total residual energy, not on the amplitude of one
    # stratum.  SIGIL carries two learned residuals; two roughly independent
    # residuals add in quadrature, so the single-key baseline gets their
    # combined amplitude and lands at comparable fidelity.  Otherwise the
    # baseline is embedded more quietly than the system it is compared to, and
    # the robustness gap partly measures loudness.
    _ls = (
        float(latent_strength)
        if latent_strength
        else (sig.latent.cfg.strength if sig.latent is not None else 0.0)
    )
    _cs = float(coarse_strength) if (coarse_checkpoint and coarse_strength) else 0.0
    _ss = math.sqrt(_ls**2 + _cs**2) if _cs else (_ls or None)
    _STATE["stablesig"] = (
        StableSignatureStyle(sig.latent, alpha=alpha, strength=_ss)
        if sig.latent is not None
        else None
    )
    _STATE["wrong"] = Sigil(
        SigilConfig(
            invariant=InvariantConfig(master_key=b"sigil-wrong-key"),
            latent_checkpoint=None,
            device=device,
            alpha=alpha,
        )
    )
    _STATE["device"] = device
    _STATE["alpha"] = alpha
    _STATE["catalogue"] = core_catalogue(device, heavy)

    # Codebook and collusion attacks are the ones that should separate a
    # per-image nonce from a fixed key, so each system faces a codebook built
    # from *its own* marked output — the adversary who collected a corpus of
    # that particular watermark.
    _STATE["per_system"] = {}
    from sigil import attacks as _atk
    from sigil import reverse_synthid as rsid

    for name, spec in (codebook_paths or {}).items():
        paths, host_paths = spec["marked"], spec["hosts"]
        if not paths:
            continue
        marked = [load_image(q) for q in paths]
        hosts = [load_image(q) for q in host_paths]
        entry: Dict = {}
        try:
            # The codebook must come from (host, marked) *pairs*: an adversary
            # who has both is the strongest realistic one, and differencing a
            # marked image against itself — the easy mistake — yields a codebook
            # of exactly zero and an attack that does nothing.
            entry["codebook"] = _atk.build_codebook_pairs(hosts, marked, canon=512)
        except Exception:
            pass
        entry["pool"] = marked[:8]
        if rsid.available():
            r = rsid.ReverseSynthID()
            try:
                r.build_codebook(marked)
                entry["rsid"] = r
            except Exception:
                entry["rsid"] = None
        _STATE["per_system"][name] = entry

    if rsid.available():
        _STATE["rsid_plain"] = rsid.ReverseSynthID()
        _STATE["catalogue"] += _STATE["rsid_plain"].catalogue(with_codebook=False)


def _detect(system: str, image, nonce=None):
    s = _STATE[system]
    if system == "sigil":
        d = s.detect(image, expected_nonce=nonce)
        from sigil.stats import log10p as _lp

        return {
            "detected": int(d.detected),
            "log10p": d.log10_pvalue,
            "A": d.analytic.statistic if d.analytic else None,
            "L": d.latent.statistic if d.latent else None,
            # Per-stratum evidence, so the complementarity and
            # strength-curve figures can be built from this one run.
            "A_log10p": _lp(d.analytic.pvalue) if d.analytic else None,
            "L_log10p": _lp(d.latent.pvalue) if d.latent else None,
            "A_anchor": d.analytic.best_anchor if d.analytic else None,
            "L_bit_accuracy": d.latent.bit_accuracy if d.latent else None,
            "C": d.coarse.statistic if d.coarse else None,
            "C_log10p": _lp(d.coarse.pvalue) if d.coarse else None,
            "C_bit_accuracy": d.coarse.bit_accuracy if d.coarse else None,
        }
    d = s.detect(image)
    return {
        "detected": int(d.detected),
        "log10p": d.log10_pvalue,
        "A": None,
        "L": d.statistic,
        "bit_acc": d.bit_accuracy,
    }


def _embed(system: str, image, rng):
    s = _STATE[system]
    if system == "sigil":
        e = s.embed(image, rng=rng)
        return e.image, e.nonce, e.psnr, e.ssim
    e = s.embed(image)
    return e.image, None, e.psnr, e.ssim


def _job(
    path: str,
    systems: List[str],
    min_psnr: float,
    min_ssim: float,
    seed: int,
    max_size: int,
) -> List[dict]:
    rng = np.random.default_rng(abs(hash(path)) % (2**31) + seed)
    img = load_image(path, max_size=max_size)
    rows: List[dict] = []
    name = Path(path).name

    for system in systems:
        if _STATE.get(system) is None:
            continue
        marked, nonce, ps, ss = _embed(system, img, rng)
        base = {"image": name, "system": system, "embed_psnr": ps, "embed_ssim": ss}

        def rec(cond, family, param, image, q_psnr, q_ssim, details=None):
            r = dict(base)
            r.update(
                {
                    "condition": cond,
                    "family": family,
                    "param": param,
                    "attack_psnr": q_psnr,
                    "attack_ssim": q_ssim,
                    "admissible": int(
                        admissible(
                            family, cond, q_psnr, q_ssim, min_psnr, min_ssim, details
                        )
                    ),
                }
            )
            if details:
                for k in ("photometric_psnr", "photometric_ssim"):
                    if details.get(k) is not None:
                        r[k] = float(details[k])
            r.update(_detect(system, image, nonce))
            rows.append(r)

        rec("clean", "none", "", marked, float("inf"), 1.0)
        rec("unmarked", "none", "", img, float("inf"), 1.0)
        wrong_marked, _, _, _ = (
            _embed("wrong", img, rng) if system == "sigil" else (None,) * 4
        )
        if wrong_marked is not None:
            rec("wrong_key", "none", "", wrong_marked, float("inf"), 1.0)

        cat = list(_STATE["catalogue"])
        # Per-system codebook / collusion, built from this system's own output.
        ent = _STATE.get("per_system", {}).get(system, {})
        if ent.get("codebook") is not None:
            # A bin-level codebook has to nominate enough bins to matter: 2k
            # out of 262k leaves the image essentially untouched and proves
            # nothing about either system.
            cat.append(
                (
                    "codebook_whiten_16k",
                    "codebook",
                    partial(atk.codebook_whiten, codebook=ent["codebook"], top_k=16384),
                )
            )
            cat.append(
                (
                    "codebook_whiten_64k",
                    "codebook",
                    partial(atk.codebook_whiten, codebook=ent["codebook"], top_k=65536),
                )
            )
            cat.append(
                (
                    "codebook_sub_r5",
                    "codebook",
                    partial(
                        atk.codebook_subtract,
                        codebook=ent["codebook"],
                        removal=5.0,
                        top_k=32768,
                    ),
                )
            )
        if ent.get("pool"):
            cat.append(
                (
                    "collusion_0.3",
                    "codebook",
                    partial(atk.collusion_average, others=ent["pool"], weight=0.3),
                )
            )
        if ent.get("rsid") is not None:
            for strength in ("gentle", "moderate", "aggressive", "maximum"):
                cat.append(
                    (
                        f"rsid_codebook_{strength}",
                        "reverse_synthid",
                        (
                            lambda img, r=ent["rsid"], strength=strength: (
                                r.codebook_subtraction(img, strength=strength)
                            )
                        ),
                    )
                )

        for cond, family, fn in cat:
            try:
                a = fn(marked)
            except Exception as exc:
                rows.append(dict(base, condition=cond, family=family, error=repr(exc)))
                continue
            rec(
                cond,
                family,
                a.param,
                a.image,
                a.psnr,
                a.ssim,
                getattr(a, "details", None),
            )
    return rows


def _worker(args):
    path, systems, min_psnr, min_ssim, seed, max_size = args
    try:
        return _job(path, systems, min_psnr, min_ssim, seed, max_size)
    except Exception as exc:
        return [{"image": Path(path).name, "error": repr(exc)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["data/corpus/photo"])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--systems", nargs="+", default=["sigil", "synthid", "stablesig"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--min-psnr", type=float, default=24.0)
    ap.add_argument("--min-ssim", type=float, default=0.70)
    ap.add_argument("--no-heavy", action="store_true")
    ap.add_argument("--codebook-refs", type=int, default=12)
    ap.add_argument("--out", default="results/arena.csv")
    ap.add_argument("--summary", default="results/arena_summary.json")
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--latent-strength", type=float, default=None)
    ap.add_argument("--analytic-alpha", type=float, default=None)
    ap.add_argument(
        "--coarse-checkpoint",
        default=None,
        help="second learned stratum at a coarser message grid; "
        "carries the rotations the fine grid gives up",
    )
    ap.add_argument("--coarse-strength", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()

    paths: List[str] = []
    for root in args.corpus:
        paths.extend(str(p) for p in list_images(root)[: args.limit])
    if not paths:
        raise SystemExit("empty corpus")
    print(f"{len(paths)} images x {len(args.systems)} systems", flush=True)

    # The codebook attack needs marked references; build them once in the parent.
    cb_paths: Dict[str, Dict[str, List[str]]] = {}
    if args.codebook_refs > 0:
        from sigil.baselines import StableSignatureStyle, SynthIDStyle
        from sigil.common import resize, save_image
        from sigil.system import Sigil, SigilConfig

        sig = Sigil(
            SigilConfig(
                latent_checkpoint=args.checkpoint,
                device=args.devices[0],
                alpha=args.alpha,
            )
        )
        makers = {
            "sigil": lambda a, r: sig.embed(a, rng=r).image,
            "synthid": (lambda mk: lambda a, r: mk.embed(a).image)(SynthIDStyle()),
            "stablesig": (
                (lambda mk: lambda a, r: mk.embed(a).image)(
                    StableSignatureStyle(sig.latent)
                )
                if sig.latent
                else None
            ),
        }
        rng = np.random.default_rng(args.seed)
        for name, mk in makers.items():
            if mk is None or name not in args.systems:
                continue
            d = Path("results/cache/codebook") / name
            dh = Path("results/cache/codebook") / "hosts"
            d.mkdir(parents=True, exist_ok=True)
            dh.mkdir(parents=True, exist_ok=True)
            out_paths, host_paths = [], []
            for i, p in enumerate(paths[: args.codebook_refs]):
                a = load_image(p, max_size=512)
                a = np.stack([resize(a[..., c], (512, 512)) for c in range(3)], -1)
                if not (dh / f"ref_{i:03d}.png").exists():
                    save_image(dh / f"ref_{i:03d}.png", a)
                save_image(d / f"ref_{i:03d}.png", mk(a, rng))
                out_paths.append(str(d / f"ref_{i:03d}.png"))
                host_paths.append(str(dh / f"ref_{i:03d}.png"))
            cb_paths[name] = {"marked": out_paths, "hosts": host_paths}
            print(f"codebook references [{name}]: {len(out_paths)}", flush=True)
        del sig

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    jobs = [
        (p, args.systems, args.min_psnr, args.min_ssim, args.seed, args.max_size)
        for p in paths
    ]
    rows: List[dict] = []
    t0 = time.time()

    if args.workers <= 1:
        _init(
            args.devices[0],
            args.checkpoint,
            args.alpha,
            not args.no_heavy,
            cb_paths,
            args.latent_strength,
            args.analytic_alpha,
            args.coarse_checkpoint,
            args.coarse_strength,
        )
        for i, j in enumerate(jobs):
            rows.extend(_worker(j))
            print(
                f"[{i + 1}/{len(jobs)}] {(i + 1) / (time.time() - t0) * 60:.1f} img/min",
                flush=True,
            )
    else:
        pools = []
        for w in range(args.workers):
            dev = args.devices[w % len(args.devices)]
            pools.append(
                ctx.Pool(
                    1,
                    initializer=_init,
                    initargs=(
                        dev,
                        args.checkpoint,
                        args.alpha,
                        not args.no_heavy,
                        cb_paths,
                        args.latent_strength,
                        args.analytic_alpha,
                        args.coarse_checkpoint,
                        args.coarse_strength,
                    ),
                )
            )
        results = []
        for i, j in enumerate(jobs):
            results.append(pools[i % len(pools)].apply_async(_worker, (j,)))
        for i, r in enumerate(results):
            rows.extend(r.get())
            if (i + 1) % 5 == 0 or i + 1 == len(results):
                print(
                    f"[{i + 1}/{len(jobs)}] {(i + 1) / (time.time() - t0) * 60:.1f} img/min",
                    flush=True,
                )
        for p in pools:
            p.close()
            p.join()

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
    print(f"wrote {out} ({len(rows)} rows) in {time.time() - t0:.0f}s")

    import pandas as pd

    df = pd.DataFrame(rows)
    summary: Dict = {
        "n_images": len(paths),
        "alpha": args.alpha,
        "quality_budget": {"min_psnr": args.min_psnr, "min_ssim": args.min_ssim},
        "systems": {},
    }
    for system, gs in df.groupby("system"):
        s: Dict = {
            "embed_psnr": float(
                pd.to_numeric(gs["embed_psnr"], errors="coerce").mean()
            ),
            "embed_ssim": float(
                pd.to_numeric(gs["embed_ssim"], errors="coerce").mean()
            ),
            "conditions": {},
        }
        for cond, g in gs.groupby("condition"):
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
                    pd.to_numeric(g["log10p"], errors="coerce").median()
                ),
            }
            v = pd.to_numeric(
                g.get("attack_psnr", pd.Series(dtype=float)), errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            e["attack_psnr"] = None if v.isna().all() else float(v.median())
            s["conditions"][cond] = e
        adm = [
            v
            for k, v in s["conditions"].items()
            if k not in ("clean", "unmarked", "wrong_key") and v["admissible"] > 0.5
        ]
        s["admissible_attacks"] = len(adm)
        s["mean_tpr_admissible"] = (
            float(np.mean([v["rate"] for v in adm])) if adm else None
        )
        s["worst_tpr_admissible"] = float(min(v["rate"] for v in adm)) if adm else None
        s["fpr_unmarked"] = s["conditions"].get("unmarked", {}).get("rate")
        s["fpr_wrong_key"] = s["conditions"].get("wrong_key", {}).get("rate")
        summary["systems"][system] = s

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.summary}\n")
    for system, s in summary["systems"].items():
        print(
            f"{system:10s} psnr={s['embed_psnr']:.2f} "
            f"clean={s['conditions'].get('clean', {}).get('rate', 0) * 100:5.1f}% "
            f"fpr={(s['fpr_unmarked'] or 0) * 100:4.1f}% "
            f"mean-TPR(adm)={(s['mean_tpr_admissible'] or 0) * 100:5.1f}% "
            f"worst={(s['worst_tpr_admissible'] or 0) * 100:5.1f}%"
        )
    print("\nweakest admissible attacks for sigil:")
    ss = summary["systems"].get("sigil", {}).get("conditions", {})
    hard = sorted(
        (
            (k, v)
            for k, v in ss.items()
            if k not in ("clean", "unmarked", "wrong_key") and v["admissible"] > 0.5
        ),
        key=lambda kv: kv[1]["rate"],
    )[:15]
    for k, v in hard:
        print(
            f"  {k:26s} TPR={v['rate'] * 100:6.1f}%  log10p={v['median_log10p']:8.1f}"
        )


if __name__ == "__main__":
    main()

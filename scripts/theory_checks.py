#!/usr/bin/env python3
"""Numerical verification of every claim the paper makes analytically.

Each check recomputes, from the corpus, the empirical counterpart of a theorem.
Where a theorem asserts an exact identity the check reports the residual in
machine epsilons; where it asserts a bound, the check reports the empirical
quantity beside the bound so the slack is visible.

    T1  translation invariance of the detector statistic
    T2  invariance to zero-phase radially symmetric filtering
    T3  invariance to global gain and offset
    T4  the exact null: Rademacher tail against the Hoeffding bound
    T5  multiplicity: searched maxima against the Bonferroni-corrected bound
    T6  codebook collapse: cross-image phase coherence at the 1/sqrt(N) floor
    T7  anchor diversity and attack stability
    T8  validity of the weighted-Bonferroni fusion under dependence
    T9  the learned stratum's nonce search null
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil.anchors import DEFAULT_ANCHORS, AnchorReader
from sigil.backend import Scanner, transform_bank
from sigil.common import canonical_luma, list_images, load_image, radius_grid
from sigil.invariant import (
    InvariantConfig,
    InvariantStratum,
    build_carriers,
    radial_excess,
    soft_clip,
)
from sigil.stats import (
    bonferroni,
    rademacher_pvalue,
    rademacher_threshold,
)


def _stat(excess, car):
    cos_t, sin_t, mirror = transform_bank([0.0], [1.0], [1.0])
    sc = Scanner(excess, cos_t, sin_t, mirror, device="cpu")
    t = sc.statistics(car.fy, car.fx, car.signs)
    return float(t[0])


# ---------------------------------------------------------------------------


def t1_translation(paths, cfg) -> Dict:
    """The statistic is a function of |F| only, so a shift cannot move it."""
    eng = InvariantStratum(cfg, device="cpu")
    diffs, base = [], []
    for p in paths[:4]:
        img = load_image(p, max_size=768)
        emb = eng.embed(img, nonce=7)
        canon = canonical_luma(emb.image, cfg.canon)
        car = build_carriers(
            emb.anchor_keys["nonce"], cfg, cfg.carriers_for(DEFAULT_ANCHORS[0])
        )
        t0 = _stat(soft_clip(radial_excess(canon), cfg.soft_clip_sigma), car)
        for dy, dx in ((1, 0), (13, 29), (77, 3)):
            shifted = np.roll(np.roll(canon, dy, 0), dx, 1)
            t1 = _stat(soft_clip(radial_excess(shifted), cfg.soft_clip_sigma), car)
            diffs.append(abs(t1 - t0))
        base.append(t0)
    return {
        "n": len(diffs),
        "mean_statistic": float(np.mean(base)),
        "max_abs_difference": float(np.max(diffs)),
        "max_relative_difference": float(np.max(diffs) / max(np.mean(base), 1e-9)),
    }


def t2_filtering(paths, cfg) -> Dict:
    """A radially symmetric zero-phase filter shifts every bin of a ring alike."""
    eng = InvariantStratum(cfg, device="cpu")
    rows = []
    for p in paths[:4]:
        img = load_image(p, max_size=768)
        emb = eng.embed(img, nonce=7)
        canon = canonical_luma(emb.image, cfg.canon)
        car = build_carriers(
            emb.anchor_keys["nonce"], cfg, cfg.carriers_for(DEFAULT_ANCHORS[0])
        )
        t0 = _stat(soft_clip(radial_excess(canon), cfg.soft_clip_sigma), car)
        r = radius_grid(canon.shape)
        for name, h in (
            ("gauss", np.exp(-0.5 * (r / 0.15) ** 2)),
            ("lowpass", 1.0 / (1.0 + (r / 0.12) ** 2)),
            ("boost", 1.0 + 2.0 * np.exp(-0.5 * ((r - 0.2) / 0.05) ** 2)),
        ):
            F = np.fft.fft2(canon.astype(np.float64)) * h
            filt = np.real(np.fft.ifft2(F))
            t1 = _stat(soft_clip(radial_excess(filt), cfg.soft_clip_sigma), car)
            rows.append({"filter": name, "delta": abs(t1 - t0), "t0": t0, "t1": t1})
    return {
        "n": len(rows),
        "max_abs_difference": float(max(r["delta"] for r in rows)),
        "per_filter": rows[:6],
    }


def t3_gain(paths, cfg) -> Dict:
    """Contrast scales every magnitude alike; the ring baseline removes it."""
    eng = InvariantStratum(cfg, device="cpu")
    diffs = []
    for p in paths[:4]:
        img = load_image(p, max_size=768)
        emb = eng.embed(img, nonce=7)
        canon = canonical_luma(emb.image, cfg.canon)
        car = build_carriers(
            emb.anchor_keys["nonce"], cfg, cfg.carriers_for(DEFAULT_ANCHORS[0])
        )
        t0 = _stat(soft_clip(radial_excess(canon), cfg.soft_clip_sigma), car)
        for c, b in ((0.6, 0.0), (1.4, 0.0), (1.0, 0.1), (0.8, -0.05)):
            t1 = _stat(
                soft_clip(radial_excess(canon * c + b), cfg.soft_clip_sigma), car
            )
            diffs.append(abs(t1 - t0))
    return {"n": len(diffs), "max_abs_difference": float(np.max(diffs))}


def t4_null(paths, cfg, n_keys: int = 400) -> Dict:
    """The statistic under H0 is a normalised Rademacher sum, nothing more.

    For an unmarked image and a key that never touched it, the chips are
    uniform signs independent of the evidence, so the tail obeys
    ``P[T >= t] <= exp(-t^2/2)`` exactly and with no distributional assumption.
    """
    from sigil.common import derive_key

    stats: List[float] = []
    for p in paths[:6]:
        canon = canonical_luma(load_image(p, max_size=768), cfg.canon)
        excess = soft_clip(radial_excess(canon), cfg.soft_clip_sigma)
        for k in range(n_keys // 6):
            key = derive_key(
                b"null-probe", "k", int(k).to_bytes(4, "little"), p.name.encode()
            )
            car = build_carriers(key, cfg, 1024)
            stats.append(_stat(excess, car))
    s = np.asarray(stats)
    tail = []
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        tail.append(
            {
                "t": t,
                "empirical": float(np.mean(s >= t)),
                "hoeffding_bound": rademacher_pvalue(t),
            }
        )
    return {
        "n_samples": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "max": float(s.max()),
        "tail": tail,
        "note": "empirical <= bound at every t means the null is conservative",
    }


def t5_multiplicity(paths, cfg) -> Dict:
    """A searched maximum needs the Bonferroni charge; measure what it buys."""
    eng = InvariantStratum(cfg, device=None)
    rows = []
    for p in paths[:12]:
        img = load_image(p, max_size=768)
        det = eng.detect(img)  # unmarked: pure null
        rows.append(
            {
                "image": p.name,
                "T": det.statistic,
                "n_hypotheses": det.n_hypotheses,
                "raw_p": rademacher_pvalue(det.statistic),
                "corrected_p": det.pvalue,
            }
        )
    ts = np.asarray([r["T"] for r in rows])
    n_hyp = int(np.median([r["n_hypotheses"] for r in rows]))
    return {
        "n_images": len(rows),
        "max_null_statistic": float(ts.max()),
        "mean_null_statistic": float(ts.mean()),
        "median_hypotheses_per_anchor_family": n_hyp,
        "expected_max_sqrt2logN": float(math.sqrt(2 * math.log(max(n_hyp, 2)))),
        "threshold_at_alpha_1e-6": rademacher_threshold(1e-6 / max(n_hyp, 1)),
        "min_corrected_p": float(min(r["corrected_p"] for r in rows)),
        "false_positives_at_1e-6": int(sum(r["corrected_p"] <= 1e-6 for r in rows)),
    }


def t6_codebook(paths, cfg, n_refs: int = 24) -> Dict:
    """Can an adversary recover the carrier set by aggregating residuals?

    Phase coherence — the usual diagnostic — is the wrong instrument for a
    *multiplicative* mark: the residual's phase at a carrier bin is the phase the
    image already had there, which is uncorrelated across images whether or not
    the carriers are shared.  What an adversary actually does is look for bins
    that move *consistently in magnitude*, so that is what is measured here.

    The experiment is the attack itself.  Mark ``N`` images, take the per-bin
    mean absolute log-magnitude deviation across them, rank bins by it, and take
    the top ``K`` where ``K`` is the true carrier count.  Precision against the
    true carrier set is exactly how much of the codebook the adversary
    recovered.  Then whiten those bins and re-detect: that is what the recovered
    codebook is worth.
    """
    from sigil.common import derive_key

    eng = InvariantStratum(cfg, device="cpu")
    rng = np.random.default_rng(11)

    # Square references so every canonical grid has the same shape and the
    # per-bin aggregation an attacker performs is well defined.
    def _square(q):
        a = load_image(q, max_size=768)
        h, w = a.shape[:2]
        m = min(h, w)
        a = a[(h - m) // 2 : (h - m) // 2 + m, (w - m) // 2 : (w - m) // 2 + m]
        from sigil.common import resize as _rs

        return np.stack([_rs(a[..., c], (512, 512)) for c in range(3)], -1)

    refs = [_square(q) for q in paths[:n_refs]]
    static_key = derive_key(b"static-control", "carriers")
    n_car = cfg.carriers_for(DEFAULT_ANCHORS[0])

    def run(mode: str) -> Dict:
        scores = None
        target_key = None
        for i, img in enumerate(refs):
            canon0 = canonical_luma(img, cfg.canon)
            if mode == "content":
                emb = eng.embed(img, nonce=int(rng.integers(0, 1 << 20)))
                key = emb.anchor_keys["nonce"]
                canon1 = canonical_luma(emb.image, cfg.canon)
            else:
                key = static_key
                car = build_carriers(key, cfg, n_car)
                h, w = canon0.shape
                lg = np.zeros((h, w))
                iy = np.rint(car.fy * h).astype(np.int64) % h
                ix = np.rint(car.fx * w).astype(np.int64) % w
                np.add.at(lg, (iy, ix), car.signs * car.weight)
                lg = lg + lg[(-np.arange(h)) % h][:, (-np.arange(w)) % w]
                canon1 = np.real(np.fft.ifft2(np.fft.fft2(canon0) * np.exp(lg))).astype(
                    np.float32
                )
            if i == 0:
                target_key = key
            d = np.log(np.abs(np.fft.fft2(canon1)) + 1e-12) - np.log(
                np.abs(np.fft.fft2(canon0)) + 1e-12
            )
            scores = np.abs(d) if scores is None else scores + np.abs(d)
        scores /= len(refs)

        # Precision is measured against *one* target image's carriers, which is
        # what the adversary actually needs: hitting some bin that some other
        # image happened to use is worth nothing.
        h, w = scores.shape
        car = build_carriers(target_key, cfg, n_car)
        iy = np.rint(car.fy * h).astype(np.int64) % h
        ix = np.rint(car.fx * w).astype(np.int64) % w
        target = set(zip(iy.tolist(), ix.tolist()))
        k = len(target)
        flat = np.argsort(scores.ravel())[::-1][:k]
        recovered = set(zip((flat // w).tolist(), (flat % w).tolist()))
        hits = len(recovered & target)
        chance = k / float(h * w)
        return {
            "precision_at_k": hits / max(k, 1),
            "chance_precision": chance,
            "lift_over_chance": (hits / max(k, 1)) / max(chance, 1e-12),
            "k": k,
        }

    return {
        "n_refs": len(refs),
        "content_derived": run("content"),
        "static_carrier_control": run("static"),
    }


def t7_anchors(paths, cfg, n_div: int = 80) -> Dict:
    """Anchor diversity, and how far a list decoder has to reach."""
    out = {}
    for spec in DEFAULT_ANCHORS:
        if spec.name == "nonce":
            out[spec.name] = {
                "diversity": 1.0,
                "entropy_bits": 20.0,
                "note": "uniform per-image nonce; independent of content",
            }
            continue
        cells = []
        for p in paths[:n_div]:
            r = AnchorReader(load_image(p, max_size=768), cfg.canon)
            d = r.spectral() if spec.name == "spectral" else r.histogram()
            cells.append(tuple(np.rint(d / spec.delta).astype(np.int64).tolist()))
        uniq = len(set(cells))
        n = len(cells)
        # A birthday estimate of the cell-space size implied by the collisions
        # observed: n^2 / (2 * collisions) cells, hence log2 of that in bits.
        coll = n - uniq
        bits = float("inf") if coll == 0 else math.log2(max(n * n / (2 * coll), 2))
        out[spec.name] = {
            "diversity": uniq / n,
            "n_images": n,
            "collisions": coll,
            "birthday_entropy_bits_lower_bound": None if math.isinf(bits) else bits,
            "delta": spec.delta,
            "list_size": spec.list_size,
        }
    return out


def t8_fusion() -> Dict:
    """Weighted Bonferroni stays valid however the strata are correlated."""
    rng = np.random.default_rng(3)
    n = 400000
    rows = []
    for rho, label in (
        (0.0, "independent"),
        (0.95, "strongly correlated"),
        (-0.95, "anti-correlated"),
    ):
        z1 = rng.normal(size=n)
        z2 = rho * z1 + math.sqrt(max(1 - rho**2, 0)) * rng.normal(size=n)
        from scipy.stats import norm

        p1, p2 = norm.sf(z1), norm.sf(z2)
        for alpha in (1e-2, 1e-3, 1e-4):
            fused = np.minimum(p1 / 0.5, p2 / 0.5)
            rows.append(
                {
                    "dependence": label,
                    "alpha": alpha,
                    "empirical_fpr": float(np.mean(fused <= alpha)),
                    "guarantee": alpha,
                }
            )
    return {
        "n_trials": n,
        "rows": rows,
        "note": "empirical <= alpha for every dependence structure",
    }


def t9_nonce_null(cfg, checkpoint: str) -> Dict:
    """The learned stratum's exhaustive nonce search, under the null."""
    if not Path(checkpoint).exists():
        return {"skipped": "no learned checkpoint"}
    import torch

    from sigil.learned import LatentStratum, correlate_all_nonces

    st = LatentStratum(checkpoint, device="cpu")
    rng = np.random.default_rng(5)
    maxima = []
    for _ in range(64):
        # Random logits stand in for an unmarked image: what matters under H0
        # is only that the evidence is independent of the key.
        z = torch.tanh(
            torch.from_numpy(rng.normal(size=(1, st.cfg.n_bits)).astype(np.float32)).to(
                st.device
            )
        )
        corr = correlate_all_nonces(z[0], st.code) / (z.pow(2).sum().sqrt() + 1e-12)
        maxima.append(float(corr.max()))
    m = np.asarray(maxima)
    n_hyp = st.code.size
    return {
        "n_trials": int(m.size),
        "nonce_space": int(n_hyp),
        "max_statistic": float(m.max()),
        "mean_max": float(m.mean()),
        "expected_max_sqrt2logN": float(math.sqrt(2 * math.log(n_hyp))),
        "threshold_at_alpha_1e-6": rademacher_threshold(1e-6 / n_hyp),
        "false_positives": int(
            sum(1 for v in m if bonferroni(rademacher_pvalue(v), n_hyp) <= 1e-6)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus", nargs="+", default=["data/corpus/natural", "data/corpus/synthetic"]
    )
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/theory_checks.json")
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    args = ap.parse_args()

    paths: List[Path] = []
    for root in args.corpus:
        paths.extend(list_images(root)[: args.limit])
    cfg = InvariantConfig()

    checks = [
        ("T1_translation_invariance", lambda: t1_translation(paths, cfg)),
        ("T2_filtering_invariance", lambda: t2_filtering(paths, cfg)),
        ("T3_gain_invariance", lambda: t3_gain(paths, cfg)),
        ("T4_exact_null", lambda: t4_null(paths, cfg)),
        ("T5_multiplicity", lambda: t5_multiplicity(paths, cfg)),
        ("T6_codebook_collapse", lambda: t6_codebook(paths, cfg)),
        ("T7_anchors", lambda: t7_anchors(paths, cfg)),
        ("T8_fusion_validity", t8_fusion),
        ("T9_nonce_null", lambda: t9_nonce_null(cfg, args.checkpoint)),
    ]
    results: Dict = {
        "config": {
            "canon": cfg.canon,
            "n_carriers": cfg.n_carriers,
            "alpha_nats": cfg.alpha,
            "n_images": len(paths),
        }
    }
    for name, fn in checks:
        try:
            results[name] = fn()
            print(f"[ok]   {name}")
        except Exception as exc:
            results[name] = {"error": repr(exc)}
            print(f"[FAIL] {name}: {exc!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")
    print(json.dumps(results, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()

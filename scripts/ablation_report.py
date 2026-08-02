#!/usr/bin/env python3
"""Does carrying two learned strata actually pay for itself?

Two grids cost real things: a second residual in the image, a second network at
detection time, and a third of the false-positive budget instead of a half.  A
design that costs those and wins nothing should be cut, so the question gets its
own measurement rather than an assertion.

Both arenas run the identical corpus, attacks and operating level; the only
difference is whether the coarse grid is carried.  This lines them up attack by
attack and reports where the second stratum changes the outcome --- in both
directions, because a per-attack loss matters as much as a per-attack gain and
averaging would hide both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load(path: str) -> Tuple[Dict, Dict]:
    d = json.loads(Path(path).read_text())
    s = d.get("systems", {}).get("sigil", {})
    return d, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="results/arena_summary.json")
    ap.add_argument("--single", default="results/arena_summary_single.json")
    ap.add_argument("--out", default="results/ABLATION.md")
    ap.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="rate change below this counts as unchanged",
    )
    args = ap.parse_args()

    for p in (args.fused, args.single):
        if not Path(p).exists():
            print(f"missing: {p}")
            return 1
    af, sf = load(args.fused)
    asg, ss = load(args.single)

    cf = sf.get("conditions", {})
    cs = ss.get("conditions", {})
    shared = sorted(set(cf) & set(cs))

    gains: List[Tuple[str, float, float, float]] = []
    losses: List[Tuple[str, float, float, float]] = []
    for k in shared:
        if k in ("unmarked", "wrong_key"):
            continue
        if cf[k].get("admissible", 1) <= 0.5:
            continue
        a, b = cf[k]["rate"], cs[k]["rate"]
        d = a - b
        if d > args.delta:
            gains.append((k, b, a, d))
        elif d < -args.delta:
            losses.append((k, b, a, d))
    gains.sort(key=lambda r: -r[3])
    losses.sort(key=lambda r: r[3])

    def adm_stats(c):
        v = [
            e["rate"]
            for k, e in c.items()
            if e.get("admissible", 1) > 0.5 and k not in ("unmarked", "wrong_key")
        ]
        n_full = sum(1 for x in v if x >= 0.995)
        return (sum(v) / len(v) if v else 0.0, min(v) if v else 0.0, n_full, len(v))

    mf, wf, nf, tf = adm_stats(cf)
    ms, ws, ns, ts = adm_stats(cs)

    L: List[str] = []
    L.append("# Is the second learned stratum worth carrying?\n")
    L.append(
        "Identical corpus, attacks and operating level. The only difference "
        "is whether the coarse message grid is carried alongside the fine "
        "one.\n"
    )
    L.append(
        "The two configurations are matched on *total residual energy*, not "
        "on per-stratum amplitude: two roughly independent residuals add in "
        "quadrature, so a pair at 0.032 each carries the same energy as one "
        "at 0.045. The embedding PSNR and SSIM below verify that the fidelity "
        "budget is comparable before the attack rates are interpreted.\n"
    )
    L.append("| | one learned stratum | two learned strata |")
    L.append("|---|---:|---:|")
    L.append(
        f"| Embedding PSNR | {ss.get('embed_psnr', 0):.1f} | {sf.get('embed_psnr', 0):.1f} |"
    )
    L.append(
        f"| Embedding SSIM | {ss.get('embed_ssim', 0):.3f} | {sf.get('embed_ssim', 0):.3f} |"
    )
    L.append(f"| Admissible attacks fully survived | {ns}/{ts} | {nf}/{tf} |")
    L.append(f"| Mean TPR (admissible) | {ms * 100:.1f}% | {mf * 100:.1f}% |")
    L.append(f"| Worst TPR (admissible) | {ws * 100:.0f}% | {wf * 100:.0f}% |")
    L.append(
        f"| False positives (unmarked) | {(ss.get('fpr_unmarked') or 0) * 100:.2f}% "
        f"| {(sf.get('fpr_unmarked') or 0) * 100:.2f}% |"
    )

    def block(title: str, rows, note: str):
        L.append(f"\n## {title}\n")
        if not rows:
            L.append(f"_{note}_")
            return
        L.append("| Attack | one stratum | two strata | change |")
        L.append("|---|---:|---:|---:|")
        for k, b, a, d in rows:
            L.append(
                f"| `{k}` | {b * 100:.0f}% | {a * 100:.0f}% | {d * 100:+.0f} pts |"
            )

    block(
        "What the second stratum buys",
        gains,
        "No attack improved by more than the reporting threshold.",
    )
    block(
        "What it costs",
        losses,
        "No attack got worse. The second stratum is free in robustness terms; "
        "it is paid for in fidelity budget and detection time.",
    )

    L.append("\n## Reading this\n")
    if gains and not losses:
        L.append(
            "The second stratum improves recovery over this suite without "
            "reducing TPR. It still consumes part of the fidelity and "
            "detection-time budgets described above."
        )
    elif gains and losses:
        L.append(
            "The second stratum trades. Whether it is worth carrying depends "
            "on which column of attacks a deployment expects, and both "
            "columns are printed above rather than averaged into one number."
        )
    elif not gains:
        L.append(
            "The second stratum does not pay for itself on this suite. It "
            "should be cut unless a deployment specifically expects the "
            "geometry it covers."
        )
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    print(f"  gains: {len(gains)}   losses: {len(losses)}")
    for k, b, a, d in gains[:8]:
        print(f"    + {k:26s} {b * 100:5.1f}% -> {a * 100:5.1f}%")
    for k, b, a, d in losses[:8]:
        print(f"    - {k:26s} {b * 100:5.1f}% -> {a * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

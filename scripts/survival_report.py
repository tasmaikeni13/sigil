#!/usr/bin/env python3
"""The survival table: every attack, every system, one page.

The paper's tables are LaTeX and the CSV is 20 000 rows; neither answers the
question a reader actually arrives with, which is "what survives what".  This
writes that as Markdown, sorted so the failures are impossible to miss rather
than buried under the successes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

SYSTEMS = [
    ("sigil", "SIGIL"),
    ("synthid", "SynthID-style"),
    ("stablesig", "StableSig-style"),
]
FAMILY_ORDER = [
    "none",
    "valuemetric",
    "geometric",
    "codebook",
    "adaptive",
    "generative",
    "reverse_synthid",
    "composite",
]


def mark(rate: float) -> str:
    if rate >= 0.995:
        return "**100%**"
    if rate >= 0.90:
        return f"{rate * 100:.0f}%"
    if rate >= 0.5:
        return f"_{rate * 100:.0f}%_"
    return f"**{rate * 100:.0f}%** ✗"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/arena_summary.json")
    ap.add_argument("--out", default="results/SURVIVAL.md")
    args = ap.parse_args()

    a = json.loads(Path(args.summary).read_text())
    sysd = a.get("systems", {})
    present = [(k, n) for k, n in SYSTEMS if k in sysd]

    conds: Dict[str, Dict] = {}
    for k, _ in present:
        for cond, e in sysd[k]["conditions"].items():
            row = conds.setdefault(
                cond,
                {
                    "family": e["family"],
                    "admissible": e["admissible"],
                    "psnr": e.get("attack_psnr"),
                },
            )
            row[k] = e["rate"]

    L: List[str] = []
    L.append("# Attack survival\n")
    L.append(
        f"Corpus: {a['n_images']} images. Operating point "
        f"$\\alpha = {a['alpha']:g}$, identical for every system and every "
        f"condition — no per-attack tuning.\n"
    )
    L.append(
        "Quality budget: an attack counts as *admissible* only if it leaves "
        f"PSNR ≥ {a['quality_budget']['min_psnr']:.0f} dB and SSIM ≥ "
        f"{a['quality_budget']['min_ssim']:.2f}. Geometric and global "
        "photometric edits are admissible by construction, since a pixel "
        "metric cannot describe them.\n"
    )

    L.append("## Headline\n")
    L.append("| System | PSNR | SSIM | Clean | FPR | Mean TPR (admissible) | Worst |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, name in present:
        v = sysd[k]
        L.append(
            f"| {name} | {v['embed_psnr']:.1f} | {v['embed_ssim']:.3f} | "
            f"{v['conditions'].get('clean', {}).get('rate', 0) * 100:.0f}% | "
            f"{(v.get('fpr_unmarked') or 0) * 100:.1f}% | "
            f"{(v.get('mean_tpr_admissible') or 0) * 100:.1f}% | "
            f"{(v.get('worst_tpr_admissible') or 0) * 100:.0f}% |"
        )

    L.append("\n## Every attack\n")
    L.append("Ordered by how well SIGIL survives, worst first.\n")
    L.append(
        "| Attack | Family | PSNR | Adm. | " + " | ".join(n for _, n in present) + " |"
    )
    L.append("|---|---|---:|:---:|" + "---:|" * len(present))

    def sortkey(item):
        cond, row = item
        return (row.get("sigil", 1.0), cond)

    for cond, row in sorted(conds.items(), key=sortkey):
        if cond in ("unmarked", "wrong_key"):
            continue
        ps = row.get("psnr")
        ps_s = "—" if ps is None else f"{ps:.1f}"
        adm = "✓" if row["admissible"] > 0.5 else "—"
        cells = " | ".join(mark(row.get(k, 0.0)) for k, _ in present)
        L.append(
            f"| `{cond}` | {row['family'].replace('_', ' ')} | {ps_s} | "
            f"{adm} | {cells} |"
        )

    L.append("\n## False positives\n")
    L.append("| Condition | " + " | ".join(n for _, n in present) + " |")
    L.append("|---|" + "---:|" * len(present))
    for cond, label in (
        ("unmarked", "Unmarked image"),
        ("wrong_key", "Marked under a different key"),
    ):
        cells = []
        for k, _ in present:
            e = sysd[k]["conditions"].get(cond)
            cells.append("—" if e is None else f"{e['rate'] * 100:.1f}%")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    fails = [
        (c, r)
        for c, r in conds.items()
        if r["admissible"] > 0.5
        and r.get("sigil", 1.0) < 0.9
        and c not in ("unmarked", "wrong_key")
    ]
    if fails:
        print(f"admissible attacks SIGIL does not fully survive ({len(fails)}):")
        for c, r in sorted(fails, key=lambda x: x[1].get("sigil", 0)):
            print(f"  {c:28s} {r.get('sigil', 0) * 100:5.1f}%")
    else:
        print("SIGIL survives every admissible attack in the suite.")


if __name__ == "__main__":
    main()

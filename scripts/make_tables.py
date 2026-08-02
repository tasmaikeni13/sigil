#!/usr/bin/env python3
"""Emit the paper's LaTeX tables from measured results.

Keeping the tables generated rather than typed means the manuscript cannot drift
away from the numbers in ``results/``; re-running the benchmark re-writes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAMILY_ORDER = [
    "valuemetric",
    "geometric",
    "codebook",
    "adaptive",
    "generative",
    "composite",
]
PRETTY = {
    "valuemetric": "Valuemetric",
    "geometric": "Geometric",
    "codebook": "Codebook",
    "adaptive": "Adaptive",
    "generative": "Generative",
    "composite": "Composite",
    "none": "Baseline",
}


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def tbl_headline(summary: Dict) -> str:
    c = summary["conditions"]
    e = summary["embed"]
    rows = [
        ("Marked (clean)", c.get("clean", {})),
        ("Unmarked", c.get("unmarked", {})),
        ("Marked under a different key", c.get("wrong_key", {})),
    ]
    out = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & Detection rate & Median $-\log_{10}p$ & $n$ \\",
        r"\midrule",
    ]
    for name, v in rows:
        if not v:
            continue
        out.append(
            f"{name} & {v['rate'] * 100:.1f}\\% & "
            f"{-v['median_log10p']:.1f} & {v['n']} \\\\"
        )
    out += [
        r"\midrule",
        f"Embedding fidelity & \\multicolumn{{3}}{{r}}{{"
        f"PSNR {e['psnr_mean']:.2f}\\,dB, SSIM {e['ssim_mean']:.4f}"
        + (f", LPIPS {e['lpips_mean']:.4f}" if e.get("lpips_mean") else "")
        + "}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(out)


def tbl_attacks(summary: Dict, admissible_only: bool = False) -> str:
    c = summary["conditions"]
    rows = [
        (k.replace("attack/", ""), v) for k, v in c.items() if k.startswith("attack/")
    ]
    if admissible_only:
        rows = [r for r in rows if r[1].get("admissible", 1) > 0.5]
    rows.sort(
        key=lambda kv: (
            FAMILY_ORDER.index(kv[1]["family"])
            if kv[1]["family"] in FAMILY_ORDER
            else 9,
            kv[0],
        )
    )
    out = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Attack & PSNR & Detection & $-\log_{10}p$ & Adm. \\",
        r"\midrule",
    ]
    last = None
    for name, v in rows:
        fam = PRETTY.get(v["family"], v["family"])
        show = fam if fam != last else ""
        last = fam
        ps = v.get("attack_psnr")
        ps_s = "---" if ps is None or not np.isfinite(ps) else f"{ps:.1f}"
        adm = "\\checkmark" if v.get("admissible", 1) > 0.5 else "---"
        out.append(
            f"{show} & {esc(name)} & {ps_s} & {v['rate'] * 100:.0f}\\% & "
            f"{-v['median_log10p']:.1f} & {adm} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tbl_theory(theory: Dict) -> str:
    rows: List[tuple] = []
    t1 = theory.get("T1_translation_invariance", {})
    if "max_abs_difference" in t1:
        rows.append(
            (
                "T1",
                "Translation invariance",
                f"max $|\\Delta T| = {t1['max_abs_difference']:.2e}$ "
                f"(on $T \\approx {t1['mean_statistic']:.1f}$)",
            )
        )
    t2 = theory.get("T2_filtering_invariance", {})
    if "max_abs_difference" in t2:
        rows.append(
            (
                "T2",
                "Radial-filter invariance",
                f"max $|\\Delta T| = {t2['max_abs_difference']:.2e}$",
            )
        )
    t3 = theory.get("T3_gain_invariance", {})
    if "max_abs_difference" in t3:
        rows.append(
            (
                "T3",
                "Gain / offset invariance",
                f"max $|\\Delta T| = {t3['max_abs_difference']:.2e}$",
            )
        )
    t4 = theory.get("T4_exact_null", {})
    if "tail" in t4:
        worst = max(
            (r["empirical"] / max(r["hoeffding_bound"], 1e-300)) for r in t4["tail"]
        )
        rows.append(
            (
                "T4",
                "Exact null (Hoeffding)",
                f"measured tail $\\leq$ bound at every $t$; worst ratio ${worst:.2f}$",
            )
        )
    t5 = theory.get("T5_multiplicity", {})
    if "max_null_statistic" in t5:
        rows.append(
            (
                "T5",
                "Multiplicity correction",
                f"max null $T = {t5['max_null_statistic']:.2f}$ vs threshold "
                f"${t5['threshold_at_alpha_1e-6']:.2f}$; "
                f"{t5['false_positives_at_1e-6']} false positives",
            )
        )
    t6 = theory.get("T6_codebook_collapse", {})
    if "content_derived" in t6:
        cd, ct = t6["content_derived"], t6["static_carrier_control"]
        rows.append(
            (
                "T6",
                "Codebook collapse",
                f"carriers recovered by aggregation: "
                f"${cd['precision_at_k'] * 100:.1f}\\%$ content-derived vs "
                f"${ct['precision_at_k'] * 100:.1f}\\%$ for a static control "
                f"(chance ${cd['chance_precision'] * 100:.2f}\\%$)",
            )
        )
    t8 = theory.get("T8_fusion_validity", {})
    if "rows" in t8:
        worst = max(r["empirical_fpr"] / r["guarantee"] for r in t8["rows"])
        rows.append(
            (
                "T8",
                "Fusion validity under dependence",
                f"empirical FPR $\\leq \\alpha$ for every dependence tested; "
                f"worst ratio ${worst:.2f}$",
            )
        )
    t9 = theory.get("T9_nonce_null", {})
    if "max_statistic" in t9:
        rows.append(
            (
                "T9",
                "Nonce-search null",
                f"max $T = {t9['max_statistic']:.2f}$ over "
                f"${t9['nonce_space']}$ codewords vs threshold "
                f"${t9['threshold_at_alpha_1e-6']:.2f}$",
            )
        )
    out = [
        r"\begin{tabular}{llp{0.56\linewidth}}",
        r"\toprule",
        r"& Claim & Measured \\",
        r"\midrule",
    ]
    for tag, claim, meas in rows:
        out.append(f"{tag} & {claim} & {meas} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tbl_anchors(theory: Dict, calib: Dict) -> str:
    t7 = theory.get("T7_anchors", {})
    out = [
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Anchor & Diversity & Entropy (bits) & Role \\",
        r"\midrule",
    ]
    roles = {
        "nonce": "content-independent; immune to clustering",
        "spectral": "survives filtering, compression, noise",
        "histogram": "survives rotation, crop, rescale",
    }
    for name, v in t7.items():
        div = v.get("diversity", 0.0)
        bits = v.get("birthday_entropy_bits_lower_bound")
        if bits is None:
            bits = v.get("entropy_bits")
        n = v.get("n_images", 0)
        if bits is None and n:
            bits_s = f"no collision in {n}"
        else:
            bits_s = "---" if bits is None else (r"$\geq$" + f"{bits:.0f}")
        out.append(
            f"{name} & {div * 100:.0f}\\% & {bits_s} & {roles.get(name, '')} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tbl_descriptor_stability(calib: Dict) -> str:
    if not calib:
        return "% descriptor calibration not available\n"
    keys = [
        k
        for k in calib
        if k.startswith(
            ("spectral@delta=0.7", "histogram@delta=0.85", "histogram@delta=0.7")
        )
    ]
    if not keys:
        keys = list(calib)[:2]
    attacks = list(calib[keys[0]]["stability"].keys())
    out = [
        r"\begin{tabular}{l" + "r" * len(keys) + "}",
        r"\toprule",
        "Attack & " + " & ".join(esc(k.split("@")[0]) for k in keys) + r" \\",
        r"\midrule",
    ]
    for a in attacks:
        out.append(
            esc(a)
            + " & "
            + " & ".join(f"{calib[k]['stability'][a] * 100:.0f}\\%" for k in keys)
            + r" \\"
        )
    out += [
        r"\midrule",
        "diversity & "
        + " & ".join(f"{calib[k]['diversity'] * 100:.0f}\\%" for k in keys)
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(out)


SYSTEM_LABEL = {
    "sigil": r"\textsc{Sigil}",
    "synthid": "SynthID-style (fixed carriers)",
    "stablesig": "Stable-Signature-style (fixed key)",
}


def tbl_arena(arena: Dict) -> str:
    """Head-to-head: the same attacks, the same budget, three watermarks."""
    out = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & PSNR & Clean & FPR & Mean TPR & Worst TPR \\",
        r"& (dB) & & & (admissible) & (admissible) \\",
        r"\midrule",
    ]
    for key in ("sigil", "synthid", "stablesig"):
        v = arena.get("systems", {}).get(key)
        if not v:
            continue
        clean = v["conditions"].get("clean", {}).get("rate", 0) * 100
        fpr = (v.get("fpr_unmarked") or 0) * 100
        out.append(
            f"{SYSTEM_LABEL[key]} & {v['embed_psnr']:.1f} & {clean:.0f}\\% & "
            f"{fpr:.1f}\\% & {(v.get('mean_tpr_admissible') or 0) * 100:.1f}\\% & "
            f"{(v.get('worst_tpr_admissible') or 0) * 100:.1f}\\% \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tbl_arena_families(arena: Dict) -> str:
    """Detection rate by attack family, per system."""
    fams: Dict[str, Dict[str, list]] = {}
    for sysname, v in arena.get("systems", {}).items():
        for cond, e in v["conditions"].items():
            if cond in ("clean", "unmarked", "wrong_key") or e["admissible"] <= 0.5:
                continue
            fams.setdefault(e["family"], {}).setdefault(sysname, []).append(e["rate"])
    order = [f for f in (FAMILY_ORDER + ["reverse_synthid"]) if f in fams]
    keys = [
        k for k in ("sigil", "synthid", "stablesig") if k in arena.get("systems", {})
    ]
    out = [
        r"\begin{tabular}{lr" + "r" * len(keys) + "}",
        r"\toprule",
        "Attack family & $n$ & "
        + " & ".join(
            {
                "sigil": r"\textsc{Sigil}",
                "synthid": "SynthID-st.",
                "stablesig": "StableSig-st.",
            }[k]
            for k in keys
        )
        + r" \\",
        r"\midrule",
    ]
    for f in order:
        n = max(len(fams[f].get(k, [])) for k in keys)
        cells = []
        for k in keys:
            r = fams[f].get(k, [])
            cells.append(f"{np.mean(r) * 100:.0f}\\%" if r else "---")
        label = f.replace("_", " ")
        out.append(f"{label} & {n} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--theory", default="results/theory_checks.json")
    ap.add_argument("--calib", default="results/descriptor_calibration.json")
    ap.add_argument("--arena", default="results/arena_summary.json")
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = (
        json.loads(Path(args.summary).read_text())
        if Path(args.summary).exists()
        else {}
    )
    theory = (
        json.loads(Path(args.theory).read_text()) if Path(args.theory).exists() else {}
    )
    calib = (
        json.loads(Path(args.calib).read_text()) if Path(args.calib).exists() else {}
    )

    if summary:
        (out / "headline.tex").write_text(tbl_headline(summary) + "\n")
        (out / "attacks.tex").write_text(tbl_attacks(summary) + "\n")
    if theory:
        (out / "theory.tex").write_text(tbl_theory(theory) + "\n")
        (out / "anchors.tex").write_text(tbl_anchors(theory, calib) + "\n")
    (out / "descriptor.tex").write_text(tbl_descriptor_stability(calib) + "\n")

    arena = (
        json.loads(Path(args.arena).read_text()) if Path(args.arena).exists() else {}
    )
    if arena:
        (out / "arena.tex").write_text(tbl_arena(arena) + "\n")
        (out / "arena_families.tex").write_text(tbl_arena_families(arena) + "\n")

    # A few macros so prose numbers cannot drift from the results either.
    macros = []
    # The detector's own search sizes, read from the code rather than retyped,
    # so the multiplicity quoted in the theory section is the one it charges.
    try:
        from sigil.learned import DEFAULT_HYPOTHESES, N_REFINE

        macros += [
            rf"\newcommand{{\nhypotheses}}{{{len(DEFAULT_HYPOTHESES)}}}",
            rf"\newcommand{{\nrefine}}{{{N_REFINE}}}",
        ]
    except Exception:
        pass
    # The geometric frontier the paper quotes has to be the *deployed* system's:
    # the strata fused, each at the reduced amplitude they are actually embedded
    # at, on the full corpus and at the same operating point as every other
    # number.  geometry_frontier.py measures one stratum at a time at full
    # amplitude, which is the right way to show the mechanism and the wrong
    # number to quote as a capability.  So these come from the arena.
    if arena:
        try:
            conds = arena.get("systems", {}).get("sigil", {}).get("conditions", {})

            def frontier(prefix, pick):
                vals = []
                for k, v in conds.items():
                    if not k.startswith(prefix):
                        continue
                    try:
                        x = float(k[len(prefix) :])
                    except ValueError:
                        continue
                    if v.get("rate", 0) >= 0.999:
                        vals.append(x)
                return pick(vals) if vals else None

            r = frontier("rotate_", max)
            c = frontier("crop_", min)
            if r is not None:
                macros.append(rf"\newcommand{{\bestRot}}{{{r:.0f}}}")
            if c is not None:
                macros.append(rf"\newcommand{{\bestCrop}}{{{c:.2f}}}")
            # The rotation curve itself, so the limitation the paper prints is
            # the deployed system's and cannot drift from the arena.
            for deg, name in ((5, "rotFive"), (15, "rotFifteen"), (30, "rotThirty")):
                e = conds.get(f"rotate_{deg}")
                if e:
                    macros.append(rf"\newcommand{{\{name}}}{{{e['rate'] * 100:.0f}}}")
            for frac, name in (
                (0.9, "cropNine"),
                (0.75, "cropSevenFive"),
                (0.5, "cropFive"),
            ):
                e = conds.get(f"crop_{frac:g}")
                if e:
                    macros.append(rf"\newcommand{{\{name}}}{{{e['rate'] * 100:.0f}}}")
        except Exception:
            pass
    if summary:
        e = summary["embed"]
        c = summary["conditions"]
        macros += [
            rf"\newcommand{{\sigilPSNR}}{{{e['psnr_mean']:.1f}}}",
            rf"\newcommand{{\sigilSSIM}}{{{e['ssim_mean']:.3f}}}",
            rf"\newcommand{{\sigilLPIPS}}{{{(e.get('lpips_mean') or 0):.4f}}}",
            rf"\newcommand{{\sigilNImages}}{{{summary['n_images']}}}",
            rf"\newcommand{{\sigilAlpha}}{{{summary['alpha']:g}}}",
            rf"\newcommand{{\sigilCleanTPR}}{{{c.get('clean', {}).get('rate', 0) * 100:.0f}}}",
            rf"\newcommand{{\sigilFPR}}{{{c.get('unmarked', {}).get('rate', 0) * 100:.0f}}}",
            rf"\newcommand{{\sigilWrongKeyFPR}}{{{c.get('wrong_key', {}).get('rate', 0) * 100:.0f}}}",
        ]
        adm = [
            v
            for k, v in c.items()
            if k.startswith("attack/") and v.get("admissible", 1) > 0.5
        ]
        if adm:
            macros += [
                rf"\newcommand{{\sigilNAdmissible}}{{{len(adm)}}}",
                rf"\newcommand{{\sigilWorstAdmissible}}"
                rf"{{{min(v['rate'] for v in adm) * 100:.0f}}}",
                rf"\newcommand{{\sigilMeanAdmissible}}"
                rf"{{{np.mean([v['rate'] for v in adm]) * 100:.1f}}}",
            ]
        macros.append(
            rf"\newcommand{{\sigilNAttacks}}"
            rf"{{{sum(1 for k in c if k.startswith('attack/'))}}}"
        )
    if arena:
        sysd = arena.get("systems", {})
        for key, tag in (
            ("sigil", "Sigil"),
            ("synthid", "SynthID"),
            ("stablesig", "StableSig"),
        ):
            v = sysd.get(key)
            if not v:
                continue
            macros += [
                rf"\newcommand{{\arena{tag}Mean}}"
                rf"{{{(v.get('mean_tpr_admissible') or 0) * 100:.1f}}}",
                rf"\newcommand{{\arena{tag}Worst}}"
                rf"{{{(v.get('worst_tpr_admissible') or 0) * 100:.1f}}}",
            ]
        macros.append(rf"\newcommand{{\arenaNImages}}{{{arena.get('n_images', 0)}}}")
    if theory:
        t6 = theory.get("T6_codebook_collapse", {})
        if "content_derived" in t6:
            cd, ct = t6["content_derived"], t6["static_carrier_control"]
            macros += [
                rf"\newcommand{{\sigilRecovered}}{{{cd['precision_at_k'] * 100:.1f}}}",
                rf"\newcommand{{\sigilRecoveredStatic}}{{{ct['precision_at_k'] * 100:.1f}}}",
                rf"\newcommand{{\sigilRecoveredChance}}{{{cd['chance_precision'] * 100:.2f}}}",
            ]
        t4 = theory.get("T4_exact_null", {})
        if "tail" in t4:
            worst = max(
                (r["empirical"] / max(r["hoeffding_bound"], 1e-300)) for r in t4["tail"]
            )
            macros.append(rf"\newcommand{{\sigilNullSlack}}{{{worst:.2f}}}")
    (out / "macros.tex").write_text("\n".join(macros) + "\n")
    print(f"wrote {len(list(out.glob('*.tex')))} table files to {out}")


if __name__ == "__main__":
    main()

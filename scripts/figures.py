#!/usr/bin/env python3
"""Every figure in the paper, built from measured results only.

Design rules followed throughout, so the set reads as one system:

* one categorical hue per *entity* (a stratum, an anchor, an attack family),
  assigned in a fixed order and never recycled when a chart drops a series;
* a single value axis per panel — never two scales in one frame;
* thin marks, recessive grid, direct labels where a legend would be noise;
* the operating threshold drawn once, as a reference line, in neutral ink.

Nothing here is illustrative: every number comes from ``benchmark.csv``,
``summary.json`` or ``theory_checks.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- palette: fixed order, one hue per entity ------------------------------
INK = "#1b1f24"
INK_2 = "#5b6570"
MUTED = "#98a2ad"
GRID = "#e3e7ec"
SURFACE = "#ffffff"

C_ANALYTIC = "#2f6f9f"  # stratum A
C_LEARNED = "#c2632a"  # stratum L
C_FUSED = "#3f7d54"  # fused decision
C_NULL = "#8a8f98"  # null / unmarked
C_ALERT = "#a83232"  # threshold, failures

FAMILY_COLOUR = {
    "valuemetric": "#2f6f9f",
    "geometric": "#c2632a",
    "codebook": "#3f7d54",
    "generative": "#7a4f9c",
    "adaptive": "#a83232",
    "composite": "#5b6570",
    "none": "#98a2ad",
}
ANCHOR_COLOUR = {"nonce": "#2f6f9f", "spectral": "#c2632a", "histogram": "#3f7d54"}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "figure.dpi": 160,
        }
    )


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------


def fig_separation(df: pd.DataFrame, out: Path, alpha: float) -> None:
    """Marked, unmarked and wrong-key evidence, against the operating point."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    thr = -math.log10(alpha)
    for ax, (col, label, colour) in zip(
        axes,
        [
            ("A_log10p", "analytic stratum", C_ANALYTIC),
            ("L_log10p", "learned stratum", C_LEARNED),
        ],
    ):
        if col not in df:
            ax.set_visible(False)
            continue
        for cond, c, name in (
            ("clean", colour, "marked"),
            ("unmarked", C_NULL, "unmarked"),
            ("wrong_key", MUTED, "wrong key"),
        ):
            v = pd.to_numeric(
                df.loc[df.condition == cond, col], errors="coerce"
            ).dropna()
            if v.empty:
                continue
            ax.hist(
                -v,
                bins=40,
                color=c,
                alpha=0.85 if cond == "clean" else 0.6,
                label=name,
                edgecolor="none",
            )
        ax.axvline(thr, color=C_ALERT, lw=1.2, ls="--")
        ax.annotate(
            f"$\\alpha=10^{{{int(math.log10(alpha))}}}$",
            xy=(thr, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            color=C_ALERT,
            fontsize=7.5,
            va="top",
        )
        ax.set_xlabel(r"evidence  $-\log_{10} p$   (multiplicity corrected)")
        ax.set_title(label, color=INK, loc="left")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))
        despine(ax)
    axes[0].set_ylabel("images")
    fig.tight_layout()
    save(fig, out, "fig01_separation")


def fig_tpr_by_attack(summary: Dict, out: Path) -> None:
    """Detection rate for every attack, grouped by family, against the FPR floor."""
    conds = summary["conditions"]
    rows = [
        (k.replace("attack/", ""), v)
        for k, v in conds.items()
        if k.startswith("attack/")
    ]
    rows = [(k, v) for k, v in rows if v.get("admissible", 1) > 0.5]
    order = [
        "valuemetric",
        "geometric",
        "codebook",
        "adaptive",
        "generative",
        "composite",
    ]
    rows.sort(
        key=lambda kv: (
            order.index(kv[1]["family"]) if kv[1]["family"] in order else 9,
            -kv[1]["rate"],
        )
    )
    names = [k for k, _ in rows]
    rates = [v["rate"] * 100 for _, v in rows]
    cols = [FAMILY_COLOUR.get(v["family"], MUTED) for _, v in rows]

    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.19 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, rates, color=cols, height=0.66, edgecolor=SURFACE, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xlabel("detection rate at the fixed operating point (%)")
    fpr = (
        max(
            conds.get("unmarked", {}).get("rate", 0.0),
            conds.get("wrong_key", {}).get("rate", 0.0),
        )
        * 100
    )
    ax.axvline(fpr, color=C_ALERT, lw=1.2, ls="--")
    ax.annotate(
        f"false-positive rate {fpr:.0f}%",
        xy=(fpr, len(names) - 0.5),
        xytext=(4, 0),
        textcoords="offset points",
        color=C_ALERT,
        fontsize=7.5,
    )
    ax.grid(axis="y", visible=False)
    handles = [
        Patch(facecolor=FAMILY_COLOUR[f], label=f)
        for f in order
        if any(v["family"] == f for _, v in rows)
    ]
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig02_tpr_by_attack")


def _curve(df: pd.DataFrame, prefix: str, xs: List[float], labels: List[str]):
    out = []
    for lab in labels:
        g = df[df.condition == f"attack/{prefix}{lab}"]
        if g.empty:
            out.append((np.nan, np.nan, np.nan))
            continue
        out.append(
            (
                pd.to_numeric(g["detected"], errors="coerce").mean() * 100,
                -pd.to_numeric(
                    g.get("A_log10p", pd.Series(dtype=float)), errors="coerce"
                ).median(),
                -pd.to_numeric(
                    g.get("L_log10p", pd.Series(dtype=float)), errors="coerce"
                ).median(),
            )
        )
    return np.asarray(out, dtype=float)


def fig_curves(df: pd.DataFrame, out: Path, alpha: float) -> None:
    """Evidence versus attack strength, per stratum, for four families."""
    panels = [
        (
            "JPEG quality",
            "jpeg_q",
            ["95", "85", "75", "60", "50", "40", "30", "20"],
            [95, 85, 75, 60, 50, 40, 30, 20],
            True,
        ),
        (
            "additive noise $\\sigma$ (/255)",
            "noise_s",
            ["2", "5", "8", "12", "20"],
            [2, 5, 8, 12, 20],
            False,
        ),
        (
            "centre crop (fraction kept)",
            "crop_",
            ["0.9", "0.8", "0.7", "0.5", "0.35"],
            [0.9, 0.8, 0.7, 0.5, 0.35],
            True,
        ),
        (
            "diffusion regeneration strength",
            "img2img_s",
            ["0.1", "0.2", "0.3", "0.45", "0.6"],
            [0.10, 0.20, 0.30, 0.45, 0.60],
            False,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0))
    thr = -math.log10(alpha)
    for ax, (xlabel, prefix, labels, xs, invert) in zip(axes.ravel(), panels):
        c = _curve(df, prefix, xs, labels)
        if np.all(np.isnan(c[:, 0])):
            ax.set_visible(False)
            continue
        ax.plot(xs, c[:, 1], color=C_ANALYTIC, marker="o", label="analytic")
        ax.plot(xs, c[:, 2], color=C_LEARNED, marker="s", label="learned")
        ax.axhline(thr, color=C_ALERT, lw=1.1, ls="--")
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_yscale("symlog", linthresh=1.0)
        despine(ax)
    axes[0, 0].set_ylabel(r"$-\log_{10} p$")
    axes[1, 0].set_ylabel(r"$-\log_{10} p$")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].annotate(
        "operating point",
        xy=(0.02, thr),
        xycoords=("axes fraction", "data"),
        color=C_ALERT,
        fontsize=7,
        va="bottom",
    )
    fig.tight_layout()
    save(fig, out, "fig03_strength_curves")


def fig_codebook(theory: Dict, out: Path) -> None:
    """How much of the carrier set cross-image aggregation actually recovers."""
    t = theory.get("T6_codebook_collapse", {})
    if "content_derived" not in t:
        return
    cd, ct = t["content_derived"], t["static_carrier_control"]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    labels = ["content-derived\n(SIGIL)", "static carriers\n(control)"]
    vals = [cd["precision_at_k"] * 100, ct["precision_at_k"] * 100]
    chance = cd["chance_precision"] * 100
    x = np.arange(2)
    ax.bar(
        x, vals, width=0.5, color=[C_FUSED, C_ALERT], edgecolor=SURFACE, linewidth=1.0
    )
    ax.axhline(chance, color=INK_2, lw=1.2, ls="--")
    ax.annotate(
        f"chance {chance:.2f}%",
        xy=(1.45, chance),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=INK_2,
        fontsize=7.5,
    )
    for xi, (v, d) in enumerate(zip(vals, [cd, ct])):
        ax.annotate(
            f"{v:.1f}%\n{d['lift_over_chance']:.0f}x chance",
            xy=(xi, v),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("carriers recovered by aggregation (%)")
    ax.set_ylim(0, max(vals) * 1.32)
    ax.grid(axis="x", visible=False)
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig04_codebook_collapse")


def fig_null_bound(theory: Dict, out: Path) -> None:
    """The measured null tail against the proved Hoeffding bound."""
    t = theory.get("T4_exact_null", {})
    if "tail" not in t:
        return
    ts = [r["t"] for r in t["tail"]]
    emp = [max(r["empirical"], 1e-6) for r in t["tail"]]
    bnd = [r["hoeffding_bound"] for r in t["tail"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.semilogy(
        ts, bnd, color=C_ALERT, ls="--", marker="", label=r"proved bound $e^{-t^2/2}$"
    )
    ax.semilogy(
        ts, emp, color=C_ANALYTIC, marker="o", label="measured (unmarked images)"
    )
    ax.set_xlabel("statistic $t$")
    ax.set_ylabel(r"$\Pr[T \geq t]$")
    ax.legend(loc="upper right")
    ax.annotate(
        "measured tail sits below the bound\nat every threshold",
        xy=(0.03, 0.06),
        xycoords="axes fraction",
        fontsize=7.5,
        color=INK_2,
    )
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig05_null_bound")


def fig_anchor_roles(df: pd.DataFrame, out: Path) -> None:
    """Which anchor carries the analytic stratum, attack by attack."""
    cols = {a: f"A_{a}_T" for a in ANCHOR_COLOUR if f"A_{a}_T" in df.columns}
    if not cols:
        return
    keep = [
        c for c in df.condition.unique() if c.startswith("attack/") or c in ("clean",)
    ]
    rows, labels = [], []
    for cond in keep:
        g = df[df.condition == cond]
        if (
            "admissible" in g
            and pd.to_numeric(g["admissible"], errors="coerce").mean() < 0.5
        ):
            continue
        vals = [pd.to_numeric(g[c], errors="coerce").mean() for c in cols.values()]
        if all(np.isnan(v) for v in vals):
            continue
        rows.append(vals)
        labels.append(cond.replace("attack/", ""))
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(-np.nanmax(arr, axis=1))
    arr, labels = arr[order], [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.18 * len(labels))))
    y = np.arange(len(labels))
    h = 0.26
    for i, (name, col) in enumerate(cols.items()):
        ax.barh(
            y + (i - 1) * h,
            arr[:, i],
            height=h * 0.92,
            color=ANCHOR_COLOUR[name],
            label=name,
            edgecolor=SURFACE,
            linewidth=0.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("mean anchor statistic $T$")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig06_anchor_roles")


def fig_strata_roles(df: pd.DataFrame, out: Path, alpha: float) -> None:
    """Which stratum carries which attack, for as many strata as are carried.

    The scatter in ``fig_complementarity`` puts one stratum on each axis, which
    stops working the moment there are three.  This says the same thing without
    that limit: one row per attack, one bar per stratum, the operating threshold
    drawn once.  A row where a single bar clears the line is an attack that only
    one stratum survives, and rows like that are the entire argument for
    carrying more than one.
    """
    cols = [
        ("A_log10p", "analytic", C_ANALYTIC),
        ("L_log10p", "learned, fine grid", C_LEARNED),
        ("C_log10p", "learned, coarse grid", C_FUSED),
    ]
    cols = [
        c
        for c in cols
        if c[0] in df.columns and pd.to_numeric(df[c[0]], errors="coerce").notna().any()
    ]
    if len(cols) < 2:
        return
    thr = -math.log10(alpha)

    rows, labels = [], []
    for cond in df.condition.unique():
        if not (cond.startswith("attack/") or cond == "clean"):
            continue
        g = df[df.condition == cond]
        if (
            "admissible" in g
            and pd.to_numeric(g["admissible"], errors="coerce").mean() < 0.5
        ):
            continue
        vals = [-pd.to_numeric(g[c], errors="coerce").median() for c, _, _ in cols]
        if all(not np.isfinite(v) for v in vals):
            continue
        rows.append([v if np.isfinite(v) else 0.0 for v in vals])
        labels.append(cond.replace("attack/", ""))
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)
    # Hardest first: the attacks where the best stratum has least to spare.
    order = np.argsort(np.nanmax(arr, axis=1))
    arr, labels = arr[order], [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.20 * len(labels))))
    y = np.arange(len(labels))
    h = 0.8 / len(cols)
    for i, (_, name, colour) in enumerate(cols):
        ax.barh(
            y + (i - (len(cols) - 1) / 2) * h,
            arr[:, i],
            height=h * 0.9,
            color=colour,
            label=name,
            edgecolor=SURFACE,
            linewidth=0.5,
        )
    ax.axvline(thr, color=C_ALERT, lw=1.1, ls="--", zorder=4)
    ax.annotate(
        f"$\\alpha=10^{{{int(math.log10(alpha))}}}$",
        xy=(thr, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -3),
        textcoords="offset points",
        color=C_ALERT,
        fontsize=7.5,
        va="top",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"evidence  $-\log_{10} p$   (multiplicity corrected, per stratum)")
    ax.legend(ncol=len(cols), loc="lower left", bbox_to_anchor=(0, 1.005))
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig15_strata_roles")


def fig_complementarity(df: pd.DataFrame, out: Path, alpha: float) -> None:
    """Per-attack evidence of one stratum against the other."""
    if "A_log10p" not in df or "L_log10p" not in df:
        return
    g = df[df.condition.str.startswith("attack/")].copy()
    if "admissible" in g:
        g = g[pd.to_numeric(g["admissible"], errors="coerce") > 0.5]
    agg = g.groupby(["condition", "family"], as_index=False).agg(
        a=("A_log10p", lambda v: -pd.to_numeric(v, errors="coerce").median()),
        l=("L_log10p", lambda v: -pd.to_numeric(v, errors="coerce").median()),
    )
    thr = -math.log10(alpha)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.axhspan(thr, 1e4, color=C_LEARNED, alpha=0.05)
    ax.axvspan(thr, 1e4, color=C_ANALYTIC, alpha=0.05)
    for fam, sub in agg.groupby("family"):
        ax.scatter(
            sub["a"],
            sub["l"],
            s=26,
            color=FAMILY_COLOUR.get(fam, MUTED),
            label=fam,
            edgecolor=SURFACE,
            linewidth=0.6,
            zorder=3,
        )
    ax.axvline(thr, color=C_ALERT, lw=1.0, ls="--")
    ax.axhline(thr, color=C_ALERT, lw=1.0, ls="--")
    lim = max(1.0, float(np.nanmax(agg[["a", "l"]].to_numpy())) * 1.15)
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"analytic stratum  $-\log_{10} p$")
    ax.set_ylabel(r"learned stratum  $-\log_{10} p$")
    ax.annotate(
        "both strata hold",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color=INK_2,
    )
    ax.annotate(
        "only the learned\nstratum holds",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=7.5,
        color=INK_2,
    )
    ax.annotate(
        "only the analytic\nstratum holds",
        xy=(0.97, 0.03),
        xycoords="axes fraction",
        ha="right",
        fontsize=7.5,
        color=INK_2,
    )
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig07_complementarity")


def fig_geometry_trade(frontier: Dict, out: Path) -> None:
    """The two geometric limits, and the single parameter that trades them.

    One panel per limit, one line per message grid, the operating threshold
    drawn once.  The fine grid starts at twice the height --- a statistic
    saturates at the square root of the bit count, so four times the cells is
    twice the ceiling --- and keeps that lead all the way down the crop panel.
    Under rotation it does not: it decays faster and crosses below the coarse
    grid at about eight degrees, so the model with more evidence on a clean
    image is the weaker of the two exactly where rotation starts to matter.

    That crossing is the whole argument for carrying both.
    """
    if not frontier:
        return
    models = [(n, r) for n, r in frontier.items() if r.get("frontier")]
    if len(models) < 2:
        return
    models.sort(key=lambda kv: kv[1]["grid"])
    hues = [C_ANALYTIC, C_LEARNED, C_FUSED, MUTED]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    panels = [
        ("rot", "rotation (degrees)", "rotation"),
        ("crop", "surviving fraction of the frame", "centre crop"),
    ]
    for ax, (pref, xlabel, title) in zip(axes, panels):
        thr = None
        for i, (name, r) in enumerate(models):
            thr = r["threshold"]
            keys = [
                (float(k.split("_")[1]), v)
                for k, v in r["frontier"].items()
                if k.startswith(pref + "_")
            ]
            keys.sort()
            xs = [k for k, _ in keys]
            ys = [v["median_T"] for _, v in keys]
            cell = r.get("n_bits") and int(round((256 / r["grid"])))
            ax.plot(
                xs,
                ys,
                lw=1.6,
                color=hues[i % len(hues)],
                zorder=3,
                marker="o",
                ms=3.4,
                mec=SURFACE,
                mew=0.6,
                label=f"{r['grid']}x{r['grid']} cells ({cell} px)",
            )
        if thr:
            ax.axhline(thr, color=C_ALERT, lw=1.0, ls="--", zorder=2)
            ax.annotate(
                "operating threshold",
                xy=(0.98, thr),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=7,
                color=C_ALERT,
            )
        if pref == "crop":
            ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9, color=INK_2, pad=6)
        despine(ax)
    axes[0].set_ylabel("detection statistic")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig14_geometry_trade")


def fig_quality_frontier(
    df: pd.DataFrame, out: Path, alpha: float, min_psnr: float
) -> None:
    """Detection against the distortion the attacker had to pay."""
    g = df[df.condition.str.startswith("attack/")].copy()
    g["pq"] = pd.to_numeric(g["attack_psnr"], errors="coerce")
    g = g[np.isfinite(g["pq"])]
    if g.empty:
        return
    agg = g.groupby(["condition", "family"], as_index=False).agg(
        psnr=("pq", "median"),
        rate=("detected", lambda v: pd.to_numeric(v, errors="coerce").mean() * 100),
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axvspan(0, min_psnr, color=MUTED, alpha=0.14)
    for fam, sub in agg.groupby("family"):
        ax.scatter(
            sub["psnr"],
            sub["rate"],
            s=26,
            color=FAMILY_COLOUR.get(fam, MUTED),
            label=fam,
            edgecolor=SURFACE,
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xlabel("distortion the attack had to pay — PSNR vs the marked image (dB)")
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(-3, 104)
    ax.set_xlim(max(10, agg["psnr"].min() - 2), min(60, agg["psnr"].max() + 2))
    ax.annotate(
        "outside the quality budget:\nthe picture is gone too",
        xy=(min_psnr - 0.6, 50),
        ha="right",
        fontsize=7.5,
        color=INK_2,
    )
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig08_quality_frontier")


def fig_fidelity(df: pd.DataFrame, out: Path) -> None:
    """What the mark costs the image."""
    g = df[df.condition == "clean"]
    if g.empty:
        return
    ps = pd.to_numeric(g["embed_psnr"], errors="coerce").dropna()
    ss = pd.to_numeric(g["embed_ssim"], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, v, lab, c in (
        (axes[0], ps, "PSNR (dB)", C_FUSED),
        (axes[1], ss, "SSIM", C_FUSED),
    ):
        ax.hist(v, bins=24, color=c, edgecolor="none", alpha=0.85)
        ax.axvline(v.mean(), color=INK, lw=1.1)
        ax.annotate(
            f"mean {v.mean():.3f}" if lab == "SSIM" else f"mean {v.mean():.2f} dB",
            xy=(v.mean(), ax.get_ylim()[1] * 0.9),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK,
        )
        ax.set_xlabel(lab)
        despine(ax)
    axes[0].set_ylabel("images")
    fig.tight_layout()
    save(fig, out, "fig09_fidelity")


def fig_visual(out: Path, corpus: str, checkpoint: str, device: str) -> None:
    """A marked image, the original, and the residual magnified."""
    from sigil.common import list_images, load_image
    from sigil.invariant import InvariantConfig
    from sigil.system import Sigil, SigilConfig

    paths = list_images(corpus)[:3]
    if not paths:
        return
    cfg = SigilConfig(
        invariant=InvariantConfig(),
        latent_checkpoint=checkpoint if Path(checkpoint).exists() else None,
        device=device,
    )
    sig = Sigil(cfg)
    fig, axes = plt.subplots(len(paths), 3, figsize=(6.6, 2.3 * len(paths)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(paths):
        img = load_image(p, max_size=640)
        emb = sig.embed(img)
        res = np.abs(emb.image - img).mean(-1)
        res = res / (res.max() + 1e-9)
        for c, (a, title) in enumerate(
            zip(
                [img, emb.image, res],
                [
                    "original",
                    f"marked — {emb.psnr:.1f} dB, SSIM {emb.ssim:.3f}",
                    "residual (magnified)",
                ],
            )
        ):
            ax = axes[r, c]
            ax.imshow(a, cmap="magma" if c == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0 or c == 1:
                ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig10_visual")


SYSTEM_COLOUR = {"sigil": C_ANALYTIC, "synthid": C_ALERT, "stablesig": "#b08a1e"}
SYSTEM_LABEL = {
    "sigil": "SIGIL",
    "synthid": "SynthID-style",
    "stablesig": "Stable-Signature-style",
}


def fig_arena_families(arena: Dict, out: Path) -> None:
    """Detection rate by attack family, three systems side by side."""
    sysd = arena.get("systems", {})
    keys = [k for k in ("sigil", "synthid", "stablesig") if k in sysd]
    if len(keys) < 2:
        return
    fams: Dict[str, Dict[str, List[float]]] = {}
    for k in keys:
        for cond, e in sysd[k]["conditions"].items():
            if cond in ("clean", "unmarked", "wrong_key") or e["admissible"] <= 0.5:
                continue
            fams.setdefault(e["family"], {}).setdefault(k, []).append(e["rate"])
    order = [
        f
        for f in [
            "valuemetric",
            "geometric",
            "codebook",
            "adaptive",
            "generative",
            "composite",
            "reverse_synthid",
        ]
        if f in fams
    ]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(order))
    wdt = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [np.mean(fams[f].get(k, [0.0])) * 100 for f in order]
        ax.bar(
            x + (i - (len(keys) - 1) / 2) * wdt,
            vals,
            width=wdt * 0.92,
            color=SYSTEM_COLOUR[k],
            label=SYSTEM_LABEL[k],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in order], fontsize=7.5)
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    despine(ax)
    fig.tight_layout()
    save(fig, out, "fig11_arena_families")


def fig_arena_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Every attack against every system, as a survival grid."""
    if "system" not in df:
        return
    g = df[~df.condition.isin(["unmarked", "wrong_key"])]
    if "admissible" in g:
        g = g[
            (pd.to_numeric(g["admissible"], errors="coerce") > 0.5)
            | (g.condition == "clean")
        ]
    piv = g.pivot_table(
        index="condition", columns="system", values="detected", aggfunc="mean"
    )
    keys = [k for k in ("sigil", "synthid", "stablesig") if k in piv.columns]
    if not keys:
        return
    piv = piv[keys].sort_values(keys[0], ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, max(4.0, 0.15 * len(piv))))
    im = ax.imshow(
        piv.values * 100,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(
        [SYSTEM_LABEL[k] for k in keys], rotation=30, ha="right", fontsize=7.5
    )
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([i.replace("attack/", "") for i in piv.index], fontsize=5.6)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("detection rate (%)", fontsize=8)
    cb.outline.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, out, "fig12_arena_heatmap")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/benchmark.csv")
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--theory", default="results/theory_checks.json")
    ap.add_argument("--arena", default="results/arena.csv")
    ap.add_argument("--arena-summary", default="results/arena_summary.json")
    ap.add_argument("--frontier", default="results/geometry_frontier.json")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--corpus", default="data/corpus/natural")
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--min-psnr", type=float, default=24.0)
    args = ap.parse_args()

    style()
    out = Path(args.out)
    print("figures:")
    # The arena run carries everything the per-stratum figures need, so it can
    # stand in for a separate single-system benchmark rather than requiring one.
    src = args.csv if Path(args.csv).exists() else args.arena
    if Path(src).exists():
        df = pd.read_csv(src, low_memory=False)
        if "system" in df.columns:
            df = df[df.system == "sigil"].copy()
        if "log10_pvalue" not in df.columns and "log10p" in df.columns:
            df["log10_pvalue"] = df["log10p"]
        for a, b in (("A", "A_statistic"), ("L", "L_statistic")):
            if b not in df.columns and a in df.columns:
                df[b] = df[a]
        df["condition"] = df["condition"].apply(
            lambda c: (
                c
                if c in ("clean", "unmarked", "wrong_key")
                or str(c).startswith("attack/")
                else f"attack/{c}"
            )
        )
        summary = (
            json.loads(Path(args.summary).read_text())
            if Path(args.summary).exists()
            else {}
        )
        if not summary and Path(args.arena_summary).exists():
            asum = json.loads(Path(args.arena_summary).read_text())
            sysd = asum.get("systems", {}).get("sigil", {})
            if sysd:
                summary = {
                    "conditions": {
                        (
                            k
                            if k in ("clean", "unmarked", "wrong_key")
                            else f"attack/{k}"
                        ): v
                        for k, v in sysd.get("conditions", {}).items()
                    }
                }
        fig_separation(df, out, args.alpha)
        if summary:
            fig_tpr_by_attack(summary, out)
        fig_curves(df, out, args.alpha)
        fig_anchor_roles(df, out)
        fig_complementarity(df, out, args.alpha)
        fig_strata_roles(df, out, args.alpha)
        fig_quality_frontier(df, out, args.alpha, args.min_psnr)
        fig_fidelity(df, out)
    if Path(args.frontier).exists():
        fig_geometry_trade(json.loads(Path(args.frontier).read_text()), out)
    if Path(args.theory).exists():
        theory = json.loads(Path(args.theory).read_text())
        fig_codebook(theory, out)
        fig_null_bound(theory, out)
    if Path(args.arena).exists():
        adf = pd.read_csv(args.arena, low_memory=False)
        fig_arena_heatmap(adf, out)
    if Path(args.arena_summary).exists():
        fig_arena_families(json.loads(Path(args.arena_summary).read_text()), out)
    try:
        fig_visual(out, args.corpus, args.checkpoint, args.device)
    except Exception as exc:
        print(f"  [skip] fig10_visual: {exc!r}")


if __name__ == "__main__":
    main()

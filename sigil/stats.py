"""Exact, distribution-free null calibration for SIGIL detectors.

The whole detector is built so that the false-positive rate can be *computed*
rather than *fitted*.  Two devices make that possible.

Rademacher keying
    Every detector statistic has the form ``T = sum_k b_k z_k`` where the signs
    ``b_k`` come from a keyed pseudo-random function and the evidence ``z_k`` is
    read off the image.  Under the null hypothesis "this image was never marked
    with this key", the image — and therefore ``z`` — is independent of the key,
    so conditionally on ``z`` the statistic is a weighted Rademacher sum with
    *known* weights.  Its tail is bounded exactly by Hoeffding's inequality, with
    no appeal to asymptotics and no assumption whatever about image statistics.

Explicit multiplicity
    A detector that searches over grid offsets, rotations, scales and
    content-seed candidates is testing many hypotheses.  Reporting the best one
    without correction inflates the false-positive rate by the size of the
    search.  :func:`bonferroni` charges for it, and :func:`fuse_pvalues` charges
    for combining strata.  Every p-value that leaves this module is valid for
    arbitrary dependence between the things being combined.

The only cryptographic assumption is that the key-derivation function behaves
like a random oracle, so that sign streams derived under different domain-
separation labels are independent of each other and of the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Exact binomial tail (sign tests)
# ---------------------------------------------------------------------------


def binom_sf_half(n: int, s: int) -> float:
    """``P[Binomial(n, 1/2) >= s]``, evaluated in exact integer arithmetic.

    Used for sign-test statistics, where the evidence is reduced to agreement
    counts.  Exact for every ``n`` a detector will ever use.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    s = int(math.ceil(s))
    if s <= 0:
        return 1.0
    if s > n:
        return 0.0
    total = 0
    c = 1  # C(n, n) == 1, iterate downward
    for k in range(n, s - 1, -1):
        total += c
        c = c * k // (n - k + 1)
    # Convert the integer ratio without first converting 2**n to float:
    # float(2**1024) overflows even though the probability is finite.
    shift = max(0, total.bit_length() - 1000)
    return math.ldexp(float(total >> shift), shift - n)


def binom_threshold_half(n: int, alpha: float) -> int:
    """Smallest ``s`` with ``P[Binomial(n, 1/2) >= s] <= alpha``."""
    lo, hi = 0, n + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if binom_sf_half(n, mid) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# Weighted Rademacher sums (soft statistics)
# ---------------------------------------------------------------------------


def rademacher_statistic(signs: Array, evidence: Array) -> float:
    """Normalised weighted Rademacher statistic ``sum b_k z_k / ||z||_2``.

    ``signs`` are the keyed +-1 chips; ``evidence`` the per-chip real-valued
    measurement.  The normalisation makes the null tail independent of the
    scale of the evidence.
    """
    b = np.asarray(signs, dtype=np.float64).ravel()
    z = np.asarray(evidence, dtype=np.float64).ravel()
    if b.size != z.size or b.size == 0:
        raise ValueError("signs and evidence must be non-empty and equal length")
    denom = float(np.sqrt(np.sum(z * z)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(b, z) / denom)


def rademacher_pvalue(t: float) -> float:
    """Hoeffding tail for a normalised Rademacher sum: ``P[T >= t] <= e^{-t^2/2}``.

    Valid for every finite number of chips and every fixed weight vector.  This
    is the p-value the analytic stratum reports.
    """
    if not np.isfinite(t) or t <= 0.0:
        return 1.0
    return float(min(1.0, math.exp(-0.5 * t * t)))


def rademacher_threshold(alpha: float) -> float:
    """Statistic value whose Hoeffding p-value equals ``alpha``."""
    alpha = float(min(max(alpha, 1e-300), 1.0))
    return math.sqrt(2.0 * math.log(1.0 / alpha))


# ---------------------------------------------------------------------------
# Multiplicity and fusion
# ---------------------------------------------------------------------------


def bonferroni(p_min: float, n_hypotheses: int) -> float:
    """Family-wise valid p-value for the best of ``n_hypotheses`` searched tests."""
    if not math.isfinite(p_min) or not 0.0 <= p_min <= 1.0:
        raise ValueError("p_min must be a finite probability")
    if n_hypotheses < 1 or int(n_hypotheses) != n_hypotheses:
        raise ValueError("n_hypotheses must be a positive integer")
    return float(min(1.0, p_min * int(n_hypotheses)))


def fuse_pvalues(
    pvalues: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    """Weighted-Bonferroni fusion, valid under arbitrary dependence.

    With weights ``w_i > 0`` summing to one, ``min_i p_i / w_i`` is a valid
    p-value for the intersection null.  Equal weights recover plain Bonferroni.
    Because no independence is assumed, the guarantee survives an adversary who
    deliberately correlates the strata.
    """
    ps = [float(p) for p in pvalues]
    if not ps:
        return 1.0
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in ps):
        raise ValueError("p-values must be finite probabilities")
    if weights is None:
        ws = [1.0 / len(ps)] * len(ps)
    else:
        ws = [float(w) for w in weights]
        if len(ws) != len(ps) or any(not math.isfinite(w) or w <= 0 for w in ws):
            raise ValueError("weights must be positive, finite, and match p-values")
        total = sum(ws)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("weight sum must be finite and positive")
        ws = [w / total for w in ws]
    return float(min(1.0, min(p / w for p, w in zip(ps, ws) if w > 0)))


def simes(pvalues: Sequence[float]) -> float:
    """Simes combination: sharper than Bonferroni, valid under PRDS dependence.

    Reported alongside the Bonferroni fusion as a diagnostic; the operating
    decision always uses :func:`fuse_pvalues` so that the guarantee needs no
    dependence assumption.
    """
    ps = np.sort(np.asarray([float(p) for p in pvalues], dtype=np.float64))
    if ps.size == 0:
        return 1.0
    m = ps.size
    return float(min(1.0, np.min(m * ps / np.arange(1, m + 1))))


# ---------------------------------------------------------------------------
# Evidence bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """A single stratum's contribution to the fused decision."""

    name: str
    statistic: float
    pvalue: float  # already multiplicity-corrected within the stratum
    n_hypotheses: int
    detail: dict

    def as_dict(self) -> dict:
        d = {
            "name": self.name,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "n_hypotheses": self.n_hypotheses,
        }
        d.update({f"{self.name}_{k}": v for k, v in self.detail.items()})
        return d


def log10p(p: float) -> float:
    """``log10`` of a p-value, floored so plots and CSVs stay finite."""
    return float(math.log10(max(float(p), 1e-300)))


__all__ = [
    "Evidence",
    "binom_sf_half",
    "binom_threshold_half",
    "bonferroni",
    "fuse_pvalues",
    "log10p",
    "rademacher_pvalue",
    "rademacher_statistic",
    "rademacher_threshold",
    "simes",
]

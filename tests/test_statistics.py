"""Regression tests for finite-sample calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from sigil.stats import (
    binom_sf_half,
    bonferroni,
    fuse_pvalues,
    rademacher_pvalue,
    rademacher_statistic,
)


def test_large_exact_binomial_tail() -> None:
    assert binom_sf_half(2048, 1024) == pytest.approx(binom.sf(1023, 2048, 0.5))
    assert binom_sf_half(2048, 2048) == pytest.approx(2.0**-2048)


def test_zero_evidence_is_not_positive_evidence() -> None:
    statistic = rademacher_statistic(np.ones(8), np.zeros(8))
    assert statistic == 0.0
    assert rademacher_pvalue(statistic) == 1.0


@pytest.mark.parametrize("p,count", [(-0.1, 2), (float("nan"), 2), (0.1, 0)])
def test_invalid_multiplicity_input_rejected(p: float, count: int) -> None:
    with pytest.raises(ValueError):
        bonferroni(p, count)


def test_invalid_fusion_weights_rejected() -> None:
    with pytest.raises(ValueError):
        fuse_pvalues([0.1, 0.2], [1.0])
    with pytest.raises(ValueError):
        fuse_pvalues([0.1, 0.2], [1.0, -0.1])

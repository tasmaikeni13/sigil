"""Nonce code and Walsh transform checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sigil.latent import LatentConfig
from sigil.learned import (
    GeoHypothesis,
    LatentStratum,
    build_code,
    correlate_all_nonces_batch,
    hypothesis_validity,
)


def test_every_code_bit_is_balanced_over_nonces() -> None:
    code = build_code(b"private-test-key", 32, 5)
    assert (code.rows != 0).all()
    words = np.stack([code.codeword(n) for n in range(code.size)])
    np.testing.assert_array_equal(words.sum(axis=0), np.full(32, code.size // 2))


def test_fwht_matches_explicit_correlation() -> None:
    code = build_code(b"private-test-key", 12, 4)
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10
    got = correlate_all_nonces_batch(logits, code)
    words = np.stack([code.codeword(n) * 2 - 1 for n in range(code.size)])
    expected = logits @ torch.from_numpy(words.T).float()
    torch.testing.assert_close(got, expected)


def _bare_stratum() -> LatentStratum:
    st = object.__new__(LatentStratum)
    st.cfg = LatentConfig(canon=64, grid=4, n_bits=16, nonce_bits=3)
    st.code = build_code(b"private-test-key", 16, 3)
    st.device = "cpu"
    st.torch_device = "cpu"
    st.hypotheses = (GeoHypothesis("identity"),)
    st.search_top = 1
    st.early_exit = 0.0
    st.refine_deg = 2.0
    return st


def test_refined_hypothesis_and_valid_bit_accuracy() -> None:
    st = _bare_stratum()
    refined = GeoHypothesis("refined", zoom=0.5)
    raw = torch.ones(16)
    corr = torch.zeros(1, st.code.size)
    det, _ = st._package(1.0, 0, 0, raw, corr, 0, 1, hypothesis=refined)
    valid = hypothesis_validity(refined, 4, "cpu")
    expected = float(np.mean(st.code.codeword(0)[valid.numpy()] == 1))
    assert det.hypothesis == "refined"
    assert det.bit_accuracy == pytest.approx(expected)


def test_refinement_failure_is_not_swallowed() -> None:
    st = _bare_stratum()

    def logits(_image, want_raw=False, only=None):
        if len(st.hypotheses) > 1:
            raise RuntimeError("refinement failed")
        values = torch.ones(1, st.cfg.n_bits)
        return (values, values) if want_raw else values

    st._hypothesis_logits = logits
    with pytest.raises(RuntimeError, match="refinement failed"):
        st.analyse(np.zeros((64, 64, 3), dtype=np.float32))

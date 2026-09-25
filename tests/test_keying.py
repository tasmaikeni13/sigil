"""Key separation and nonce derivation."""

from __future__ import annotations

import numpy as np
import pytest

from sigil.common import deployment_key_from_env, experiment_nonce, key_stream_rng
from sigil.system import Sigil, SigilConfig


def test_hmac_stream_is_reproducible_and_domain_separated() -> None:
    key = bytes(range(32))
    a = key_stream_rng(key, "carrier").integers(0, 256, size=64)
    b = key_stream_rng(key, "carrier").integers(0, 256, size=64)
    c = key_stream_rng(key, "mask").integers(0, 256, size=64)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    sample = key_stream_rng(key, "selection").choice(100, size=30)
    assert len(set(sample)) == 30


def test_private_root_separates_strata_and_nonce() -> None:
    key = bytes(range(32))
    sigil = Sigil(SigilConfig(latent_checkpoint=None, device="cpu", master_key=key))
    assert sigil.analytic.cfg.master_key != key
    assert sigil.analytic.cfg.master_key != SigilConfig().invariant.master_key
    assert experiment_nonce(key, "image-a", 4, 20) == experiment_nonce(
        key, "image-a", 4, 20
    )
    assert experiment_nonce(key, "image-a", 4, 20) != experiment_nonce(
        key, "image-b", 4, 20
    )


def test_deployment_key_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGIL_MASTER_KEY_HEX", raising=False)
    with pytest.raises(ValueError):
        deployment_key_from_env()
    monkeypatch.setenv("SIGIL_MASTER_KEY_HEX", "ab" * 32)
    assert deployment_key_from_env() == bytes.fromhex("ab" * 32)

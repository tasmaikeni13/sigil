"""Historical and smoke summaries cannot silently populate the manuscript."""

from __future__ import annotations

import json

import pytest

from scripts.publication import (
    require_publishable_arena,
    require_publishable_benchmark,
    require_publishable_null_audit,
)


def test_historical_summary_rejected() -> None:
    with pytest.raises(ValueError):
        require_publishable_arena("results/arena_summary_single.json")
    assert (
        require_publishable_arena(
            "results/arena_summary_single.json", allow_demo=True, min_images=1
        )["n_images"]
        == 80
    )


def test_benchmark_must_match_eligible_arena(tmp_path) -> None:
    arena = {
        "n_images": 200,
        "corpus": ["data/corpus/natural"],
        "checkpoint": "checkpoints/latent.pt",
        "alpha": 1e-6,
    }
    path = tmp_path / "summary.json"
    path.write_text('{"n_images": 200, "eligible_for_paper": false}')
    with pytest.raises(ValueError):
        require_publishable_benchmark(path, arena)


def test_small_historical_null_audit_rejected() -> None:
    with pytest.raises(ValueError):
        require_publishable_null_audit(
            "results/fpr_study.json",
            {"alpha": 1e-6, "checkpoint": "checkpoints/latent.pt"},
        )


def test_arena_requires_common_admissible_conditions(tmp_path) -> None:
    systems = {
        name: {"embed_psnr": 43.0, "embed_ssim": 0.99, "admissible_attacks": 30}
        for name in ("sigil", "synthid", "stablesig")
    }
    arena = {
        "eligible_for_paper": True,
        "key_mode": "private",
        "n_images": 200,
        "alpha": 1e-6,
        "quality_budget": {"min_psnr": 42.0, "min_ssim": 0.985},
        "scanner_backend": "tpu",
        "systems": systems,
        "attack_conditions": dict.fromkeys(systems, 30),
        "common_admissible_conditions": 0,
    }
    path = tmp_path / "arena.json"
    path.write_text(json.dumps(arena))
    with pytest.raises(ValueError, match="not eligible"):
        require_publishable_arena(path)
    arena["common_admissible_conditions"] = 30
    path.write_text(json.dumps(arena))
    assert require_publishable_arena(path) == arena

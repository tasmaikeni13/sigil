"""Eligibility gate for manuscript tables and figures."""

from __future__ import annotations

import json
from pathlib import Path


def require_publishable_arena(
    path: str | Path, *, allow_demo: bool = False, min_images: int = 200
) -> dict:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"missing arena summary: {source}")
    arena = json.loads(source.read_text())
    if allow_demo:
        if arena.get("n_images", 0) < min_images:
            raise ValueError("arena has fewer images than requested")
        return arena
    systems = arena.get("systems", {})
    quality = arena.get("quality_budget", {})
    attack_counts = arena.get("attack_conditions", {})
    if (
        arena.get("eligible_for_paper") is not True
        or arena.get("key_mode") != "private"
        or arena.get("n_images", 0) < min_images
        or arena.get("alpha") != 1e-6
        or quality.get("min_psnr", 0) < 42.0
        or quality.get("min_ssim", 0) < 0.985
        or arena.get("scanner_backend") != "tpu"
        or set(systems) != {"sigil", "synthid", "stablesig"}
        or arena.get("common_admissible_conditions", 0) < 30
        or any(attack_counts.get(name, 0) < 30 for name in systems)
        or any(systems[name].get("admissible_attacks", 0) < 30 for name in systems)
        or any(systems[name].get("embed_psnr", 0) < 42.0 for name in systems)
        or any(systems[name].get("embed_ssim", 0) < 0.985 for name in systems)
    ):
        raise ValueError(
            "arena is not eligible for publication; use --allow-demo only for diagnostics"
        )
    return arena


def require_publishable_benchmark(
    path: str | Path, arena: dict, *, allow_demo: bool = False, min_images: int = 200
) -> dict:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"missing benchmark summary: {source}")
    summary = json.loads(source.read_text())
    if allow_demo:
        return summary
    embed = summary.get("embed", {})
    quality = summary.get("quality_budget", {})
    if (
        summary.get("eligible_for_paper") is not True
        or summary.get("key_mode") != "private"
        or summary.get("n_images", 0) < min_images
        or summary.get("n_images") != arena.get("n_images")
        or summary.get("corpus") != arena.get("corpus")
        or summary.get("checkpoint") != arena.get("checkpoint")
        or summary.get("alpha") != arena.get("alpha")
        or summary.get("scanner_backend") != "tpu"
        or quality.get("min_psnr", 0) < 42.0
        or quality.get("min_ssim", 0) < 0.985
        or embed.get("psnr_mean", 0) < 42.0
        or embed.get("ssim_mean", 0) < 0.985
        or summary.get("attack_conditions", 0) < 30
    ):
        raise ValueError("benchmark is inconsistent with the publication arena")
    return summary


def require_publishable_null_audit(
    path: str | Path, arena: dict, *, allow_demo: bool = False
) -> dict:
    source = Path(path)
    if not source.is_file():
        if allow_demo:
            return {}
        raise ValueError(f"missing null audit: {source}")
    audit = json.loads(source.read_text())
    if allow_demo:
        return audit
    conditions = audit.get("conditions", {})
    if (
        audit.get("n_images", 0) < 100000
        or audit.get("key_mode") != "private"
        or audit.get("alpha") != arena.get("alpha")
        or audit.get("checkpoint") != arena.get("checkpoint")
        or audit.get("scanner_backend") != "tpu"
        or any(
            conditions.get(name, {}).get("n", 0) < 100000
            or conditions.get(name, {}).get("false_positives") != 0
            or conditions.get(name, {}).get("one_sided_upper_95") is None
            for name in ("unmarked", "wrong_key")
        )
    ):
        raise ValueError("the 100000-image private-key null audit is incomplete")
    return audit

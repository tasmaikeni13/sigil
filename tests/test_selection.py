"""Only a qualified, private checkpoint may be promoted."""

from __future__ import annotations

import json

from scripts.promote_selection import best


def test_mock_or_legacy_selection_is_not_promotable(tmp_path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"legacy": {"held": 2, "t_sum": 50}}))
    assert best(path) is None
    path.write_text(
        json.dumps(
            {
                "mock": {
                    "eligible": False,
                    "mock": True,
                    "key_mode": "public-smoke",
                    "score": 1,
                    "psnr": 50,
                }
            }
        )
    )
    assert best(path) is None

"""Publication corpus provenance is checked before output is written."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts.build_corpus import build_verified_corpus, verify_source_manifest


def test_verified_ingestion_and_preflight(tmp_path) -> None:
    source = tmp_path / "source.png"
    noise = np.random.default_rng(4).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    Image.fromarray(noise).save(source)
    entry = {
        "path": source.name,
        "source": "test fixture",
        "license": "CC0",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / "sources.json"
    target = tmp_path / "corpus"
    manifest.write_text(json.dumps([entry]))
    records = build_verified_corpus(manifest, target)
    assert len(records) == 1
    assert records[0]["benchmark_eligible"]
    assert (target / records[0]["file"]).is_file()


def test_bad_source_never_creates_corpus(tmp_path) -> None:
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": "missing.png",
                    "source": "x",
                    "license": "CC0",
                    "sha256": "0" * 64,
                }
            ]
        )
    )
    target = tmp_path / "corpus"
    with pytest.raises(ValueError):
        verify_source_manifest(manifest, target)
    assert not target.exists()

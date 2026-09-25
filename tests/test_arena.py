"""Arena inputs must match the licensed corpus manifest."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts.arena import verify_arena_corpus


def test_manifest_hash_and_eligibility_gate(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    natural = corpus / "natural"
    natural.mkdir(parents=True)
    path = natural / "image.png"
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "corpus_manifest.json"
    data = {
        "target_dir": str(corpus),
        "benchmark_eligible": True,
        "manifest": [
            {
                "file": "natural/image.png",
                "benchmark_eligible": True,
                "sha256": digest,
            }
        ],
    }
    manifest.write_text(json.dumps(data))
    verify_arena_corpus([str(path)], manifest)
    data["benchmark_eligible"] = False
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not benchmark-eligible"):
        verify_arena_corpus([str(path)], manifest)
    data["benchmark_eligible"] = True
    manifest.write_text(json.dumps(data))
    Image.fromarray(np.full((64, 64, 3), 50, dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="checksum changed"):
        verify_arena_corpus([str(path)], manifest)

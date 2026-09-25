#!/usr/bin/env python3
"""Build and ingest the open-source evaluation corpora.

Full ingestion uses an explicit, checksummed source manifest. Smoke mode
creates procedural execution fixtures that are never publication eligible.

Usage:
    python scripts/build_corpus.py --target-dir data/smoke_corpus --smoke-test
    python scripts/build_corpus.py --target-dir data/corpus --source-manifest sources.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil.common import load_image, resize, save_image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def spectral_flatness(img: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy) of image luminance."""
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    F = np.abs(np.fft.rfft2(luma)) ** 2
    F = F[F > 1e-12]
    if F.size == 0:
        return 1.0
    geom_mean = float(np.exp(np.mean(np.log(F))))
    arith_mean = float(np.mean(F))
    return float(geom_mean / max(arith_mean, 1e-12))


def generate_naturalistic_procedural(size: int = 512, seed: int = 0) -> np.ndarray:
    """Generate rich, continuous naturalistic spectral patterns without external dependencies."""
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:size, :size]
    cy, cx = size / 2.0, size / 2.0
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / (size / 2.0)
    theta = np.arctan2(y - cy, x - cx)

    # Multi-frequency harmonic basis with random phase
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        plane = np.zeros((size, size), dtype=np.float32)
        for k in range(1, 9):
            freq = 2.0**k + rng.uniform(-0.5, 0.5)
            phase = rng.uniform(0, 2 * math.pi)
            harm = np.sin(freq * math.pi * r + phase + rng.uniform(-1, 1) * theta)
            plane += (harm / (k**1.2)).astype(np.float32)
        # Add smooth spatial gradients
        grad = (
            np.sin(rng.uniform(1, 4) * x / size * math.pi)
            * np.cos(rng.uniform(1, 4) * y / size * math.pi)
        ).astype(np.float32)
        plane = 0.7 * plane + 0.3 * grad
        plane = (plane - plane.min()) / max(float(plane.max() - plane.min()), 1e-6)
        img[..., c] = np.clip(plane, 0.0, 1.0)
    return img


def run_smoke_test(target_dir: Path) -> List[Dict]:
    """Build procedural and repository fixtures for execution tests only."""
    target_dir.mkdir(parents=True, exist_ok=True)
    natural_dir = target_dir / "natural"
    synthetic_dir = target_dir / "synthetic"
    photo_dir = target_dir / "photo"
    for d in (natural_dir, synthetic_dir, photo_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict] = []
    idx = 0

    # 1. Ingest reference cache images from repository if available
    cache_hosts = Path("results/cache/codebook/hosts")
    if cache_hosts.exists():
        for p in sorted(cache_hosts.glob("*.png")):
            img = load_image(p, max_size=512)
            if img.shape[0] != 512 or img.shape[1] != 512:
                img = np.stack(
                    [resize(img[..., c], (512, 512)) for c in range(img.shape[-1])],
                    axis=-1,
                )
            dest_nat = natural_dir / f"img_{idx:04d}.png"
            dest_photo = photo_dir / f"img_{idx:04d}.png"
            save_image(dest_nat, img)
            save_image(dest_photo, img)

            entry = {
                "image_id": f"img_{idx:04d}",
                "file": f"natural/img_{idx:04d}.png",
                "source": "repository-fixture-unverified",
                "license": "unverified",
                "benchmark_eligible": False,
                "sha256": sha256_file(dest_nat),
                "original_dimensions": [512, 512, 3],
                "spectral_flatness": spectral_flatness(img),
            }
            manifest_entries.append(entry)
            idx += 1

    # 2. Augment with diverse naturalistic and generative procedural images to reach >= 64 images
    needed = max(64 - idx, 24)
    for i in range(needed):
        img = generate_naturalistic_procedural(size=512, seed=20260900 + i)
        dest_syn = synthetic_dir / f"syn_{i:04d}.png"
        dest_nat = natural_dir / f"img_{idx:04d}.png"
        dest_photo = photo_dir / f"img_{idx:04d}.png"
        save_image(dest_syn, img)
        save_image(dest_nat, img)
        save_image(dest_photo, img)

        entry = {
            "image_id": f"img_{idx:04d}",
            "file": f"synthetic/syn_{i:04d}.png",
            "source": "procedural-smoke",
            "license": "not-applicable",
            "benchmark_eligible": False,
            "sha256": sha256_file(dest_syn),
            "original_dimensions": [512, 512, 3],
            "spectral_flatness": spectral_flatness(img),
        }
        manifest_entries.append(entry)
        idx += 1

    return manifest_entries


ALLOWED_LICENSES = {"CC0", "CC-BY-2.0", "CC-BY-4.0", "Apache-2.0"}


def verify_source_manifest(manifest_path: Path, target: Path) -> List[Dict]:
    """Preflight every licensed source before writing any corpus output."""
    data = json.loads(manifest_path.read_text())
    entries = data["manifest"] if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest must contain a nonempty image list")
    if any(target.rglob("*.png")) or any(target.rglob("*.jpg")):
        raise ValueError("target already contains images; use an empty directory")
    verified: List[Dict] = []
    seen_hashes: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"source entry {i} must be an object")
        for field in ("path", "source", "license", "sha256"):
            if not entry.get(field):
                raise ValueError(f"source entry {i} lacks {field}")
        if entry["license"] not in ALLOWED_LICENSES:
            raise ValueError(f"source entry {i} has unapproved license")
        source = (manifest_path.parent / entry["path"]).resolve()
        if not source.is_file():
            raise ValueError(f"missing source image: {source}")
        digest = sha256_file(source)
        if digest != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {source}")
        if digest in seen_hashes:
            raise ValueError(f"duplicate source image: {source}")
        seen_hashes.add(digest)
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            width, height = image.size
        img = load_image(source, max_size=1024)
        if min(img.shape[:2]) < 64 or float(img.std()) < 0.02:
            raise ValueError(f"degenerate or undersized source image: {source}")
        verified.append(
            {
                "image_id": f"img_{i:05d}",
                "source_path": str(source),
                "source_sha256": digest,
                "source": entry["source"],
                "license": entry["license"],
                "original_dimensions": [height, width, 3],
                "benchmark_eligible": True,
            }
        )
    return verified


def build_verified_corpus(manifest_path: Path, target: Path) -> List[Dict]:
    verified = verify_source_manifest(manifest_path, target)
    natural = target / "natural"
    natural.mkdir(parents=True, exist_ok=True)
    for entry in verified:
        source = Path(entry["source_path"])
        image = load_image(source, max_size=1024)
        dest = natural / f"{entry['image_id']}.png"
        save_image(dest, image)
        entry["file"] = f"natural/{dest.name}"
        entry["sha256"] = sha256_file(dest)
        entry["spectral_flatness"] = spectral_flatness(image)
    return verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", "--out", default="data/corpus", dest="target_dir")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--source-manifest", type=Path)
    args = ap.parse_args()

    target = Path(args.target_dir)
    if args.smoke_test:
        print(f"Building non-publishable smoke fixtures under {target}...")
        manifest_entries = run_smoke_test(target)
    else:
        if args.source_manifest is None:
            ap.error(
                "full ingestion requires --source-manifest with licensed image hashes"
            )
        manifest_entries = build_verified_corpus(args.source_manifest, target)

    corpus_manifest_path = target.with_name(f"{target.name}_manifest.json")
    corpus_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "count": len(manifest_entries),
        "target_dir": str(target),
        "benchmark_eligible": not args.smoke_test,
        "manifest": manifest_entries,
    }
    corpus_manifest_path.write_text(json.dumps(data, indent=2))
    (target / "manifest.json").write_text(json.dumps(data, indent=2))
    print(f"Ingested and verified {len(manifest_entries)} images.")
    print(f"Wrote manifest: {corpus_manifest_path}")


if __name__ == "__main__":
    main()

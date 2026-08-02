"""Adapters for the published reverse-SynthID removal attacks.

The external project supplies three attack shapes: a signal-processing
pipeline, a stacked removal pass, and a spectral codebook attack. This module
keeps those implementations intact and presents them through the local
``AttackResult`` interface. The codebook is learned from marked images so the
benchmark represents an adversary with access to a collection of outputs.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .attacks import AttackResult, _finish
from .common import save_image, to_float01, to_uint8

_SRC = Path(__file__).resolve().parents[2] / "reverse-SynthID" / "src" / "extraction"


def available() -> bool:
    return (_SRC / "synthid_bypass.py").exists()


def _load():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    import synthid_bypass  # noqa: E402

    return synthid_bypass


def _quiet(fn, *a, **kw):
    """The reference implementation prints progress; keep the benchmark clean."""
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return fn(*a, **kw)


class ReverseSynthID:
    """Wrapper exposing every published bypass mode as an ``AttackResult``."""

    def __init__(self):
        mod = _load()
        self.mod = mod
        self.bypass = mod.SynthIDBypass(iterations=1, extractor=None)
        self.codebook = None

    # -- removal pipelines -------------------------------------------------

    def _run(self, name: str, param: str, image: np.ndarray, fn) -> AttackResult:
        src = to_float01(image)
        u8 = to_uint8(src)
        res = _quiet(fn, u8)
        out = np.asarray(res.cleaned_image, dtype=np.float32)
        if out.max() > 1.5:
            out = out / 255.0
        return _finish(
            name,
            "reverse_synthid",
            param,
            src,
            out,
            stages=list(getattr(res, "stages_applied", [])),
        )

    def pipeline(self, image: np.ndarray, mode: str = "balanced") -> AttackResult:
        """Run the reference signal-processing removal pipeline."""
        return self._run(
            "rsid_pipeline",
            f"mode={mode}",
            image,
            lambda u8: self.bypass.bypass(u8, mode=mode, verify=False),
        )

    def stacked(
        self, image: np.ndarray, strength: str = "aggressive", iterations: int = 2
    ) -> AttackResult:
        """Run the reference stacked removal pipeline."""
        return self._run(
            "rsid_stacked",
            f"{strength}x{iterations}",
            image,
            lambda u8: self.bypass.bypass_v2(
                u8, strength=strength, iterations=iterations, verify=False
            ),
        )

    def simple(self, image: np.ndarray, jpeg_quality: int = 50) -> AttackResult:
        return self._run(
            "rsid_simple",
            f"q={jpeg_quality}",
            image,
            lambda u8: self.bypass.bypass_simple(
                u8, jpeg_quality=jpeg_quality, verify=False
            ),
        )

    def build_codebook(
        self, marked_images: Sequence[np.ndarray], max_images: Optional[int] = None
    ) -> None:
        """Aggregate a spectral codebook from images we marked ourselves.

        This is the attack in its strongest form: the adversary is handed a
        corpus of watermarked output and runs the reference extractor on it.
        """
        mod = self.mod
        cb = mod.SpectralCodebook()
        with tempfile.TemporaryDirectory() as td:
            shapes: Dict[tuple, list] = {}
            for i, im in enumerate(marked_images):
                a = to_float01(im)
                shapes.setdefault(a.shape[:2], []).append((i, a))
            # The reference builder needs one directory of equal-sized images.
            biggest = max(shapes.items(), key=lambda kv: len(kv[1]))[1]
            for i, a in biggest:
                save_image(Path(td) / f"wm_{i:04d}.png", a)
            _quiet(cb.build_from_watermarked, td, max_images)
        self.codebook = cb

    def codebook_subtraction(
        self, image: np.ndarray, strength: str = "moderate", passes: int = 0
    ) -> AttackResult:
        """Remove a mark with a codebook learned from marked images."""
        if self.codebook is None:
            raise RuntimeError("build_codebook() must run before codebook_subtraction")
        src = to_float01(image)
        u8 = to_uint8(src)
        res = _quiet(
            self.bypass.bypass_v3,
            u8,
            self.codebook,
            strength=strength,
            passes=passes,
            verify=False,
        )
        out = np.asarray(res.cleaned_image, dtype=np.float32)
        if out.max() > 1.5:
            out = out / 255.0
        return _finish(
            "rsid_codebook",
            "reverse_synthid",
            f"{strength}",
            src,
            out,
            stages=list(getattr(res, "stages_applied", [])),
        )

    # -- catalogue ---------------------------------------------------------

    def catalogue(self, with_codebook: bool = True) -> List[tuple]:
        """Return the configured reference attacks for benchmark generation."""
        c: List[tuple] = [
            (
                f"rsid_pipeline_{m}",
                "reverse_synthid",
                (lambda img, m=m: self.pipeline(img, mode=m)),
            )
            for m in ("light", "balanced", "aggressive", "maximum")
        ]
        c += [
            (
                f"rsid_stacked_{s}",
                "reverse_synthid",
                (lambda img, s=s: self.stacked(img, strength=s, iterations=2)),
            )
            for s in ("moderate", "aggressive", "maximum")
        ]
        c.append(
            (
                "rsid_simple_q50",
                "reverse_synthid",
                lambda img: self.simple(img, jpeg_quality=50),
            )
        )
        c.append(
            (
                "rsid_simple_q30",
                "reverse_synthid",
                lambda img: self.simple(img, jpeg_quality=30),
            )
        )
        if with_codebook:
            c += [
                (
                    f"rsid_codebook_{s}",
                    "reverse_synthid",
                    (lambda img, s=s: self.codebook_subtraction(img, strength=s)),
                )
                for s in ("gentle", "moderate", "aggressive", "maximum")
            ]
        return c


__all__ = ["ReverseSynthID", "available"]

"""Reference watermarks, so SIGIL is measured against something rather than alone.

Both baselines are implemented here rather than merely cited, because the point
of the comparison is to run *the same attacks* against all three systems under
*the same detector calibration*. A number quoted from another paper, measured on
another corpus with another threshold, would not support any of the claims this
repository makes.

``SynthIDStyle``
    A fixed-carrier spectral watermark placed at the carrier bins that
    ``reverse-SynthID`` published after extracting them from Gemini output, with
    a differential-phase detector. This is the design that repository's attacks
    were built to break, so it is the fair target for them — and it is also the
    canonical example of a *static codebook*, which is exactly what SIGIL's
    content-derived carriers are meant to avoid.

``StableSignatureStyle``
    Meta's Stable Signature ties one fixed key to one fine-tuned generator: every
    image that model produces carries the *same* message. Fine-tuning a latent
    decoder is out of scope here, but the property that matters for robustness
    and for the codebook attack is the fixed key, so this baseline reuses SIGIL's
    own trained encoder and decoder with the nonce held constant. That isolates
    a single variable — per-image nonce versus fixed key — and makes the
    collusion result attributable rather than confounded with architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .common import (
    apply_canonical_residual,
    canonical_luma,
    key_stream_rng,
    quality,
    to_float01,
)
from .stats import bonferroni, log10p, rademacher_pvalue

Array = np.ndarray
_EPS = 1e-12


# ---------------------------------------------------------------------------
# SynthID-style: a fixed carrier set
# ---------------------------------------------------------------------------

#: Carrier offsets from the DFT centre at a 512x512 reference, as published by
#: the reverse-SynthID project after aggregating 288 Gemini images.
SYNTHID_CARRIERS: Tuple[Tuple[int, int], ...] = (
    (48, 0),
    (-48, 0),
    (0, 48),
    (0, -48),
    (88, 0),
    (-88, 0),
    (0, 88),
    (0, -88),
    (48, 48),
    (48, -48),
    (-48, 48),
    (-48, -48),
    (88, 88),
    (88, -88),
    (-88, 88),
    (-88, -88),
    (48, 88),
    (48, -88),
    (-48, 88),
    (-48, -88),
    (88, 48),
    (88, -48),
    (-88, 48),
    (-88, -48),
    (24, 0),
    (-24, 0),
    (0, 24),
    (0, -24),
    (64, 24),
    (64, -24),
    (-64, 24),
    (-64, -24),
    (24, 64),
    (24, -64),
    (-24, 64),
    (-24, -64),
    (112, 0),
    (-112, 0),
    (0, 112),
    (0, -112),
)


@dataclass
class BaselineEmbed:
    image: Array
    payload: np.ndarray
    psnr: float
    ssim: float


@dataclass
class BaselineDetection:
    statistic: float
    pvalue: float
    detected: bool
    bit_accuracy: Optional[float] = None

    @property
    def log10_pvalue(self) -> float:
        return log10p(self.pvalue)


class SynthIDStyle:
    """Fixed-carrier differential-phase watermark at the published SynthID bins."""

    def __init__(
        self,
        canon: int = 512,
        strength: float = 0.55,
        master_key: bytes = b"synthid-style",
        alpha: float = 1e-6,
        repeats: int = 24,
    ):
        # Tuned so the baseline starts from a *fair* position: clean detection
        # far above threshold at fidelity comparable to SIGIL's.  A baseline
        # that fails on clean images proves nothing when it later fails under
        # attack.
        self.canon = canon
        self.strength = strength
        self.master_key = master_key
        self.alpha = alpha
        self.repeats = repeats
        rng = key_stream_rng(master_key, "chips")
        base = list(SYNTHID_CARRIERS)
        # Repeat the pattern at several radial scales so the carrier count is
        # comparable to SIGIL's; the defining property stays the same, namely
        # that every image uses the identical set.
        self.carriers: List[Tuple[float, float]] = []
        for m in np.linspace(0.55, 1.45, repeats):
            for u, v in base:
                self.carriers.append((u * m / 512.0, v * m / 512.0))
        self.signs = rng.integers(0, 2, size=len(self.carriers)) * 2.0 - 1.0

    def _bins(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = shape
        iy = np.rint([c[0] * h for c in self.carriers]).astype(np.int64) % h
        ix = np.rint([c[1] * w for c in self.carriers]).astype(np.int64) % w
        return iy, ix

    def embed(self, image: Array, payload=None) -> BaselineEmbed:
        rgb = to_float01(image)
        canon = canonical_luma(rgb, self.canon)
        h, w = canon.shape
        iy, ix = self._bins((h, w))
        log_gain = np.zeros((h, w))
        np.add.at(log_gain, (iy, ix), self.signs * self.strength)
        log_gain = log_gain + log_gain[(-np.arange(h)) % h][:, (-np.arange(w)) % w]
        F = np.fft.fft2(canon.astype(np.float64)) * np.exp(log_gain)
        wm = np.real(np.fft.ifft2(F)).astype(np.float32)
        out = apply_canonical_residual(rgb, (wm - canon).astype(np.float32))
        q = quality(rgb, out)
        return BaselineEmbed(image=out, payload=self.signs, psnr=q.psnr, ssim=q.ssim)

    def detect(self, image: Array) -> BaselineDetection:
        from .invariant import radial_excess, soft_clip

        canon = canonical_luma(to_float01(image), self.canon)
        z = soft_clip(radial_excess(canon), 2.5)
        h, w = canon.shape
        iy, ix = self._bins((h, w))
        vals = np.fft.ifftshift(z, axes=(0, 1))[iy, ix]
        denom = float(np.sqrt(np.sum(vals**2))) + _EPS
        t = float(np.dot(self.signs, vals) / denom)
        p = bonferroni(rademacher_pvalue(t), 1)
        acc = float(np.mean(np.sign(vals) == self.signs))
        return BaselineDetection(
            statistic=t, pvalue=p, detected=p <= self.alpha, bit_accuracy=acc
        )


# ---------------------------------------------------------------------------
# Stable-Signature-style: one fixed key for every image
# ---------------------------------------------------------------------------


class StableSignatureStyle:
    """A learned watermark carrying one fixed key, as Stable Signature does.

    The encoder and decoder are SIGIL's own, so the only difference from the
    learned stratum is that the message never changes. Everything the comparison
    attributes to per-image nonces is therefore attributable to nonces and not
    to a different network, a different training budget or a different capacity.
    """

    def __init__(
        self,
        latent,
        fixed_nonce: int = 0,
        alpha: float = 1e-6,
        strength: Optional[float] = None,
    ):
        self.latent = latent
        self.nonce = int(fixed_nonce)
        self.alpha = alpha
        #: Amplitude for this baseline's residual, independent of whatever the
        #: stratum object it borrows. The explicit value keeps the comparison at
        #: the same residual energy as the multi-stratum system.
        self.strength = strength
        self.message = latent.code.codeword(self.nonce)

    def embed(self, image: Array, payload=None) -> BaselineEmbed:
        rgb = to_float01(image)
        e = self.latent.embed(rgb, nonce=self.nonce, strength=self.strength)
        q = quality(rgb, e.image)
        return BaselineEmbed(
            image=e.image, payload=self.message, psnr=q.psnr, ssim=q.ssim
        )

    def detect(self, image: Array) -> BaselineDetection:
        import torch

        z = self.latent._hypothesis_logits(image)
        chips = torch.from_numpy((self.message * 2 - 1).astype(np.float32)).to(z.device)
        num = z @ chips
        den = z.pow(2).sum(dim=1).sqrt() + 1e-12
        t = float((num / den).max())
        # A fixed key means no nonce search: the only multiplicity is geometric.
        p = bonferroni(rademacher_pvalue(t), self.latent.n_searched())
        best = int((num / den).argmax())
        acc = float(((z[best] > 0).cpu().numpy() == self.message).mean())
        return BaselineDetection(
            statistic=t, pvalue=p, detected=p <= self.alpha, bit_accuracy=acc
        )


__all__ = [
    "BaselineDetection",
    "BaselineEmbed",
    "SYNTHID_CARRIERS",
    "StableSignatureStyle",
    "SynthIDStyle",
]

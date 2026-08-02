"""Content anchors: how a carrier set is bound to the image it marks.

A watermark whose carriers come from the key alone is a static codebook.  An
adversary who aggregates the residuals of many marked images sees the same bins
light up every time, learns the carrier set without ever learning the key, and
erases the mark by whitening exactly those bins — at a distortion cost no larger
than the embedder paid.  Binding the carriers to the image is therefore not an
optimisation; it is the whole of the codebook defence.

The binding has to be *robust*, because the detector re-derives it from an
attacked image, and *high-entropy*, because an adversary who can cluster images
into a few hundred equivalence classes can run the codebook attack inside a
class.  No single descriptor achieves both across all attacks, so SIGIL carries
several anchors at once on disjoint carrier subsets and detects on whichever
survives.

    spectral   ripple of the low-band radial log-power profile, detrended
               against a low-order polynomial in log-radius.  Invariant to
               smooth filtering, compression and noise; slides bodily under a
               crop and does not survive rotation.

    histogram  order statistics of the luminance and chroma marginals,
               normalised by their own interquartile range.  Nearly unmoved by
               rotation, crop and rescale — they resample the same scene — but
               softer under heavy blur.

    nonce      a per-image uniform random value, recovered from the learned
               stratum rather than from the pixels.  Its entropy is independent
               of content, so clustering images by appearance tells an adversary
               nothing about it; this is the anchor that closes the clustering
               loophole the two content descriptors leave open.

Measured diversity and per-attack stability for each family are produced by
``scripts/calibrate_descriptor.py`` and reported in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .common import canonical_luma, derive_key, resize, rgb_to_ycbcr, to_float01

Array = np.ndarray
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Lattice quantisation with list decoding
# ---------------------------------------------------------------------------


def quantise_with_list(
    desc: Array, delta: float, list_size: int
) -> List[Tuple[np.ndarray, float]]:
    """Lattice-quantise ``desc`` and enumerate its most plausible neighbours.

    A quantiser cell boundary is the only place a content anchor is fragile, so
    the detector never commits to one cell: it walks outward along the
    coordinates whose residual sits closest to a boundary.  Besides absorbing
    the drift that any attack induces, this disarms the adversary who
    deliberately nudges an image across a boundary — the cell they push it into
    is already on the list.
    """
    scaled = np.asarray(desc, dtype=np.float64) / float(delta)
    base = np.rint(scaled)
    resid = scaled - base
    margin = 0.5 - np.abs(resid)
    order = np.argsort(margin)
    n_flip = max(0, int(math.floor(math.log2(max(list_size, 1)))))
    flip_idx = order[:n_flip]
    direction = np.sign(resid)
    direction[direction == 0] = 1.0
    out: List[Tuple[np.ndarray, float]] = []
    for mask in range(1 << n_flip):
        cand = base.copy()
        cost = 0.0
        for j, idx in enumerate(flip_idx):
            if mask & (1 << j):
                cand[idx] += direction[idx]
                cost += float(margin[idx])
        out.append((cand.astype(np.int64), cost))
    out.sort(key=lambda t: t[1])
    return out[:list_size]


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def fine_radial_profile(
    canon: Array, n_fine: int = 320, f_lo: float = 0.0015, f_hi: float = 0.34
) -> Tuple[Array, Array]:
    """Mean log-power on a dense log-spaced radial grid, computed once per image."""
    from .common import radius_grid

    F = np.fft.fft2(canon.astype(np.float64))
    log_power = np.log(np.abs(F) ** 2 + _EPS)
    r = radius_grid(canon.shape).astype(np.float64)
    edges = np.geomspace(f_lo, f_hi, n_fine + 1)
    idx = np.searchsorted(edges, r.ravel(), side="right") - 1
    valid = (idx >= 0) & (idx < n_fine)
    sums = np.bincount(idx[valid], weights=log_power.ravel()[valid], minlength=n_fine)
    counts = np.bincount(idx[valid], minlength=n_fine).astype(np.float64)
    centres = np.sqrt(edges[:-1] * edges[1:])
    occ = counts > 0
    return sums[occ] / counts[occ], centres[occ]


def detrend(prof: Array, centres: Array, order: int = 2) -> Array:
    """Project out a low-order polynomial in log-radius.

    Every smooth radial filter — Gaussian blur, unsharp masking, resampling
    roll-off, the mean JPEG quantisation envelope, a gamma-induced contrast
    change — adds a smooth function of log-radius to the log-power profile.
    Removing that subspace leaves only the content-specific ripple, so the
    anchor inherits the same filtering invariance the detector statistic has.
    """
    x = np.log(centres)
    V = np.vander(x, order + 1)
    coef, *_ = np.linalg.lstsq(V, prof, rcond=None)
    resid = prof - V @ coef
    return resid / (float(np.sqrt(np.mean(resid**2))) + _EPS)


def _quantiles(v: Array, n: int) -> Array:
    return np.quantile(v, np.linspace(1.0 / (n + 1), n / (n + 1), n))


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    delta: float
    list_size: int
    share: float  # fraction of the carrier budget
    scales: Tuple[float, ...] = (1.0,)


DEFAULT_ANCHORS: Tuple[AnchorSpec, ...] = (
    AnchorSpec("nonce", delta=1.0, list_size=1, share=0.50),
    AnchorSpec(
        "spectral",
        delta=0.70,
        list_size=64,
        share=0.25,
        scales=(1.0, 0.90, 1.11, 0.80, 1.25, 0.70, 1.43),
    ),
    AnchorSpec("histogram", delta=0.70, list_size=64, share=0.25),
)


class AnchorReader:
    """Compute descriptors once per image and hand out candidate anchor keys."""

    def __init__(
        self,
        image: Array,
        canon: int = 512,
        f_min: float = 0.004,
        f_max: float = 0.052,
        bands: int = 18,
    ):
        rgb = to_float01(image)
        self.canon_img = canonical_luma(rgb, canon)
        self.prof, self.centres = fine_radial_profile(self.canon_img)
        self.f_min, self.f_max, self.bands = f_min, f_max, bands
        y, cb, cr = rgb_to_ycbcr(rgb)
        self._tone = np.asarray(
            [
                (float(np.median(cb)) - 0.5) * 22.0,
                (float(np.median(cr)) - 0.5) * 22.0,
            ],
            dtype=np.float64,
        )
        self._y, self._cb, self._cr = y, cb, cr

    def spectral(self, assumed_scale: float = 1.0) -> Array:
        edges = np.geomspace(self.f_min, self.f_max, self.bands + 1) * float(
            assumed_scale
        )
        band_c = np.sqrt(edges[:-1] * edges[1:])
        sampled = np.interp(np.log(band_c), np.log(self.centres), self.prof)
        return np.concatenate([detrend(sampled, band_c), self._tone])

    def histogram(self, assumed_scale: float = 1.0) -> Array:
        # More order statistics means more entropy at the same quantiser step,
        # which is what an anchor needs: an adversary who can cluster the corpus
        # into few descriptor cells can run the codebook attack inside a cell.
        small = resize(self._y, (256, 256))
        q = _quantiles(small.ravel(), 26)
        med = float(np.median(q))
        iqr = float(np.quantile(q, 0.75) - np.quantile(q, 0.25)) + 1e-6
        shape = (q - med) / iqr
        cbq = (_quantiles(resize(self._cb, (128, 128)).ravel(), 11) - 0.5) / iqr
        crq = (_quantiles(resize(self._cr, (128, 128)).ravel(), 11) - 0.5) / iqr
        return np.concatenate([shape, cbq, crq]) * 1.6

    def keys(
        self,
        spec: AnchorSpec,
        master_key: bytes,
        nonces: Optional[Sequence[int]] = None,
    ) -> List[bytes]:
        """Candidate carrier keys for one anchor, most plausible first."""
        if spec.name == "nonce":
            return [
                derive_key(master_key, "anchor-nonce", int(v).to_bytes(8, "little"))
                for v in (nonces or ())
            ]
        fn = self.spectral if spec.name == "spectral" else self.histogram
        out: List[bytes] = []
        seen: set[bytes] = set()
        per_scale = max(1, spec.list_size // max(len(spec.scales), 1))
        for sc in spec.scales:
            desc = fn(float(sc))
            for q, _ in quantise_with_list(desc, spec.delta, per_scale):
                k = derive_key(master_key, f"anchor-{spec.name}", q.tobytes())
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out

    def embed_key(
        self, spec: AnchorSpec, master_key: bytes, nonce: Optional[int] = None
    ) -> Optional[bytes]:
        """The single key the embedder commits to for this anchor."""
        if spec.name == "nonce":
            if nonce is None:
                return None
            return derive_key(
                master_key, "anchor-nonce", int(nonce).to_bytes(8, "little")
            )
        fn = self.spectral if spec.name == "spectral" else self.histogram
        q, _ = quantise_with_list(fn(1.0), spec.delta, 1)[0]
        return derive_key(master_key, f"anchor-{spec.name}", q.tobytes())


__all__ = [
    "DEFAULT_ANCHORS",
    "AnchorReader",
    "AnchorSpec",
    "detrend",
    "fine_radial_profile",
    "quantise_with_list",
]

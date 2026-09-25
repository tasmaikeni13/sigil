"""Shared primitives for SIGIL: image I/O, colour, canonicalisation, metrics, keying.

Everything downstream of this module agrees on three conventions.

Canonical grid
    Both strata embed on a fixed ``N x N`` canonical luminance grid and push the
    resulting residual back to native resolution.  A uniform rescale of the
    delivered image therefore maps back onto the *same* canonical grid, so scale
    invariance is a property of the representation rather than something the
    detector has to search for.

Residual transport
    A stratum produces a canonical-grid residual.  ``apply_canonical_residual``
    resamples it to native resolution and adds it to the luminance channel,
    leaving chrominance untouched.  Because every residual this codebase
    produces is band-limited well below the canonical Nyquist rate, the
    resampling is faithful.

Keying
    All secret material is derived through :func:`derive_key`, an HMAC-SHA256
    based key-derivation function.  Carrier positions, chip signs and codeword
    masks are all outputs of a keyed PRF, which is what makes the exact null
    distributions in :mod:`sigil.stats` valid.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional
    cv2 = None

try:
    from skimage.metrics import structural_similarity as _sk_ssim
except Exception:  # pragma: no cover - optional
    _sk_ssim = None


Array = np.ndarray

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

#: Canonical working resolution for both strata.
CANON = 512


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------


def derive_key(master: bytes, label: str, *chunks: bytes) -> bytes:
    """HMAC-SHA256 key derivation.  ``label`` separates independent uses."""
    mac = hmac.new(master, label.encode("utf-8"), hashlib.sha256)
    for c in chunks:
        mac.update(len(c).to_bytes(4, "little"))
        mac.update(c)
    return mac.digest()


def deployment_key_from_env(name: str = "SIGIL_MASTER_KEY_HEX") -> bytes:
    """Read the private 256-bit root required for publication-scale runs."""
    value = os.environ.get(name, "")
    try:
        key = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be 64 hexadecimal characters") from exc
    if len(value) != 64 or len(key) != 32:
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return key


class KeyStream:
    """HMAC-SHA256 counter stream for cryptographic carrier draws.

    A NumPy PRNG seeded by HMAC does not itself provide a secret-key PRF.
    Each block here is an HMAC output under a domain-separated stream key.
    """

    def __init__(self, key: bytes):
        self.key = key
        self.counter = 0
        self.buffer = b""

    def _bytes(self, n: int) -> bytes:
        while len(self.buffer) < n:
            block = hmac.new(
                self.key, self.counter.to_bytes(16, "little"), hashlib.sha256
            ).digest()
            self.buffer += block
            self.counter += 1
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def _below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        nbytes = max(1, (bound.bit_length() + 7) // 8)
        ceiling = 1 << (8 * nbytes)
        cutoff = ceiling - ceiling % bound
        while True:
            value = int.from_bytes(self._bytes(nbytes), "little")
            if value < cutoff:
                return value % bound

    def integers(self, low, high=None, size=None, dtype=np.int64):
        if high is None:
            low, high = 0, low
        low, high = int(low), int(high)
        if high <= low:
            raise ValueError("high must exceed low")
        shape = (
            () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        )
        values = [low + self._below(high - low) for _ in range(math.prod(shape))]
        array = np.asarray(values, dtype=dtype).reshape(shape)
        return array.item() if size is None else array

    def uniform(self, low=0.0, high=1.0, size=None):
        shape = (
            () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        )
        values = [
            low
            + (high - low) * (int.from_bytes(self._bytes(8), "little") >> 11) / 2**53
            for _ in range(math.prod(shape))
        ]
        array = np.asarray(values, dtype=np.float64).reshape(shape)
        return array.item() if size is None else array

    def choice(self, n: int, size: int, replace: bool = False):
        if replace:
            return self.integers(0, n, size=size)
        if size < 0 or size > n:
            raise ValueError("sample size exceeds population")
        pool = np.arange(n, dtype=np.int64)
        for i in range(size):
            j = i + self._below(n - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:size]


def key_stream_rng(master: bytes, label: str, *chunks: bytes) -> KeyStream:
    """Return a domain-separated cryptographic random stream."""
    return KeyStream(derive_key(master, label, *chunks))


def experiment_nonce(master: bytes, image_id: str, seed: int, bits: int) -> int:
    """Reproducible private nonce for a named experimental sample."""
    if not 1 <= bits <= 63:
        raise ValueError("nonce bit width must be in [1, 63]")
    digest = derive_key(
        master,
        "experiment-nonce",
        image_id.encode("utf-8"),
        int(seed).to_bytes(8, "little", signed=True),
    )
    return int.from_bytes(digest[:8], "little") % (1 << bits)


# ---------------------------------------------------------------------------
# I/O and colour
# ---------------------------------------------------------------------------


def list_images(directory) -> list[Path]:
    root = Path(directory)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_image(path, max_size: Optional[int] = None) -> Array:
    """Load as float32 RGB in [0, 1], optionally capping the longest side."""
    img = Image.open(path).convert("RGB")
    if max_size is not None and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(path, image: Array) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), "RGB").save(p)


def to_uint8(image: Array) -> Array:
    a = np.asarray(image)
    if a.dtype == np.uint8:
        return a
    if a.size and float(np.nanmax(a)) > 1.5:
        return np.clip(a, 0, 255).astype(np.uint8)
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_float01(image: Array) -> Array:
    a = np.asarray(image)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    if a.size and float(np.nanmax(a)) > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def rgb_to_ycbcr(image: Array) -> Tuple[Array, Array, Array]:
    a = to_float01(image)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (b - y) * 0.564
    cr = 0.5 + (r - y) * 0.713
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def ycbcr_to_rgb(y: Array, cb: Array, cr: Array) -> Array:
    r = y + 1.403 * (cr - 0.5)
    b = y + 1.773 * (cb - 0.5)
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling and canonicalisation
# ---------------------------------------------------------------------------


def resize(a: Array, size: Tuple[int, int], *, antialias: bool = True) -> Array:
    """Resize a 2-D float array to ``(h, w)``.

    Downscaling uses an area filter (correct anti-aliasing, no ringing on the
    watermark band); upscaling uses cubic interpolation.
    """
    h, w = int(size[0]), int(size[1])
    a = np.asarray(a, dtype=np.float32)
    if a.shape[:2] == (h, w):
        return a.copy()
    if cv2 is not None:
        down = h * w < a.shape[0] * a.shape[1]
        interp = cv2.INTER_AREA if (down and antialias) else cv2.INTER_CUBIC
        return cv2.resize(a, (w, h), interpolation=interp).astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    span = max(hi - lo, 1e-8)
    pil = Image.fromarray(((a - lo) / span * 255.0).astype(np.uint8), "L")
    pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    return (np.asarray(pil, dtype=np.float32) / 255.0 * span + lo).astype(np.float32)


#: Aspect ratios outside this range are clamped so the canonical grid stays sane.
MAX_ASPECT = 4.0


def canonical_shape(h: int, w: int, n: int = CANON) -> Tuple[int, int]:
    """Aspect-preserving canonical grid of area ``n^2``.

    Preserving the aspect ratio is what keeps canonical pixels square, and
    square pixels are what make an image-plane rotation act as a rigid rotation
    of the canonical spectrum.  A fixed ``n x n`` grid would turn a rotation
    into a shear for every non-square image and destroy the invariance.
    """
    ar = float(h) / float(max(w, 1))
    ar = float(np.clip(ar, 1.0 / MAX_ASPECT, MAX_ASPECT))
    hc = int(round(n * math.sqrt(ar)))
    wc = int(round(n / math.sqrt(ar)))
    hc = max(64, hc - (hc % 2))
    wc = max(64, wc - (wc % 2))
    return hc, wc


def canonical_luma(image: Array, n: int = CANON) -> Array:
    """Luminance of ``image`` resampled onto its aspect-preserving canonical grid."""
    y, _, _ = rgb_to_ycbcr(image)
    return resize(y, canonical_shape(y.shape[0], y.shape[1], n))


def apply_canonical_residual(
    image: Array, residual: Array, *, clip: bool = True
) -> Array:
    """Add a canonical-grid luminance residual to a native-resolution RGB image."""
    rgb = to_float01(image)
    y, cb, cr = rgb_to_ycbcr(rgb)
    r_native = resize(residual, y.shape[:2], antialias=False)
    y2 = y + r_native
    if clip:
        y2 = np.clip(y2, 0.0, 1.0)
    return ycbcr_to_rgb(y2.astype(np.float32), cb, cr)


def apply_canonical_residual_rgb(
    image: Array, residual: Array, *, clip: bool = True
) -> Array:
    """Add a canonical-grid RGB residual to a native-resolution RGB image.

    The learned stratum needs all three channels.  Projecting its residual onto
    luminance, as the analytic stratum's transport does, is not a small loss:
    measured on a trained encoder, bit accuracy falls from 99.9% to chance,
    because the network puts a large part of the code in chroma where it is both
    less visible and better protected from the luminance-only processing most
    attacks apply.
    """
    rgb = to_float01(image)
    r_native = np.stack(
        [resize(residual[..., c], rgb.shape[:2], antialias=False) for c in range(3)],
        axis=-1,
    )
    out = rgb + r_native
    return np.clip(out, 0.0, 1.0).astype(np.float32) if clip else out.astype(np.float32)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def psnr(reference: Array, test: Array) -> float:
    a = to_float01(reference).astype(np.float64)
    b = to_float01(test).astype(np.float64)
    if a.shape != b.shape:
        b = np.stack([resize(b[..., c], a.shape[:2]) for c in range(b.shape[-1])], -1)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse <= 0 else float(10.0 * math.log10(1.0 / mse))


def ssim(reference: Array, test: Array) -> float:
    a = to_float01(reference)
    b = to_float01(test)
    if a.shape != b.shape:
        b = np.stack([resize(b[..., c], a.shape[:2]) for c in range(b.shape[-1])], -1)
    if _sk_ssim is not None:
        ch = -1 if a.ndim == 3 else None
        return float(_sk_ssim(a, b, data_range=1.0, channel_axis=ch))
    ga = a.mean(-1) if a.ndim == 3 else a
    gb = b.mean(-1) if b.ndim == 3 else b
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = float(ga.mean()), float(gb.mean())
    va, vb = float(ga.var()), float(gb.var())
    cov = float(((ga - mu_a) * (gb - mu_b)).mean())
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    )


# ---------------------------------------------------------------------------
# Radial helpers used by the analytic stratum
# ---------------------------------------------------------------------------


def frequency_grid(shape: Tuple[int, int]) -> Tuple[Array, Array]:
    """Per-bin frequencies in cycles per pixel, in FFT (unshifted) layout."""
    h, w = int(shape[0]), int(shape[1])
    return np.fft.fftfreq(h)[:, None].astype(np.float32), np.fft.fftfreq(w)[
        None, :
    ].astype(np.float32)


def radius_grid(shape) -> Array:
    """Radial frequency in cycles per pixel.

    Because canonical pixels are square, this radius is rotation-covariant: an
    image-plane rotation permutes bins within a ring and leaves the ring itself
    fixed.
    """
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape), int(shape))
    fy, fx = frequency_grid(shape)
    return np.hypot(fy, fx).astype(np.float32)


def ring_index(shape, n_rings: int = 512) -> Array:
    """Integer ring label per bin, uniformly spaced in radial frequency."""
    r = radius_grid(shape)
    return np.clip((r / 0.7072 * n_rings).astype(np.int32), 0, n_rings - 1)


def radial_log_baseline(log_mag: Array, rings: Array, n_rings: int) -> Array:
    """Per-bin baseline: the mean log-magnitude of the ring the bin belongs to.

    Because any zero-phase linear filter (blur, sharpening, JPEG's smooth
    quantisation envelope, global gain) acts almost identically on every bin of
    a ring, subtracting this baseline removes the attack's effect to first order
    while leaving a per-bin watermark perturbation intact.
    """
    flat = rings.ravel()
    sums = np.bincount(flat, weights=log_mag.ravel(), minlength=n_rings)
    counts = np.bincount(flat, minlength=n_rings).astype(np.float64)
    means = sums / np.maximum(counts, 1.0)
    return means[flat].reshape(log_mag.shape)


@dataclass(frozen=True)
class Quality:
    psnr: float
    ssim: float

    def as_dict(self) -> dict:
        return {"psnr": self.psnr, "ssim": self.ssim}


def quality(reference: Array, test: Array) -> Quality:
    return Quality(psnr=psnr(reference, test), ssim=ssim(reference, test))


__all__ = [
    "CANON",
    "IMAGE_EXTS",
    "MAX_ASPECT",
    "Quality",
    "apply_canonical_residual",
    "apply_canonical_residual_rgb",
    "canonical_luma",
    "canonical_shape",
    "frequency_grid",
    "derive_key",
    "deployment_key_from_env",
    "experiment_nonce",
    "KeyStream",
    "key_stream_rng",
    "list_images",
    "load_image",
    "psnr",
    "quality",
    "radial_log_baseline",
    "radius_grid",
    "resize",
    "rgb_to_ycbcr",
    "ring_index",
    "save_image",
    "ssim",
    "to_float01",
    "to_uint8",
    "ycbcr_to_rgb",
]

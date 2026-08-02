"""The adversary: every attack SIGIL is evaluated against.

An attack is only an attack if it leaves an image someone would still want.  A
transform that drives PSNR to 15 dB has not removed the watermark; it has
removed the picture, and an adversary who is willing to do that could have
posted noise in the first place.  Every attack here therefore reports its own
distortion against the marked image, and the benchmark declares an attack
*admissible* only inside a stated quality budget.  Results outside the budget
are still reported — they mark where the architecture ends — but they are not
counted as defeats.

Families
--------
``valuemetric``   compression, noise, filtering, tone and colour manipulation
``geometric``     translation, crop, rescale, rotation, warp, reflection
``codebook``      cross-image aggregation and subtraction, plus collusion
                  averaging: the attacks a static carrier set cannot survive
``generative``    VAE round-trips, latent-noise regeneration, real SD-Turbo
                  img2img, and iterated diffusion purification
``adaptive``      white-box PGD against the actual learned decoder, black-box
                  removal by a surrogate network, and spectral whitening that
                  assumes the carrier band is known
``composite``     stacked pipelines that chain the strongest members of each
                  family
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .common import (
    canonical_luma,
    psnr,
    resize,
    rgb_to_ycbcr,
    ring_index,
    ssim,
    to_float01,
    to_uint8,
    ycbcr_to_rgb,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

Array = np.ndarray


@dataclass
class AttackResult:
    name: str
    family: str
    param: str
    image: Array
    psnr: float
    ssim: float
    lpips: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


def _finish(
    name: str, family: str, param: str, src: Array, out: Array, **details
) -> AttackResult:
    out = np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32))
    return AttackResult(
        name=name,
        family=family,
        param=param,
        image=out,
        psnr=psnr(src, out),
        ssim=ssim(src, out),
        details=details,
    )


# ---------------------------------------------------------------------------
# Valuemetric
# ---------------------------------------------------------------------------


def jpeg(image: Array, quality: int = 75) -> AttackResult:
    src = to_float01(image)
    buf = BytesIO()
    Image.fromarray(to_uint8(src), "RGB").save(
        buf, format="JPEG", quality=int(quality), subsampling=2
    )
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0
    return _finish("jpeg", "valuemetric", f"q={quality}", src, out, quality=quality)


def double_jpeg(image: Array, q1: int = 85, q2: int = 55) -> AttackResult:
    src = to_float01(image)
    mid = jpeg(src, q1).image
    out = jpeg(mid, q2).image
    return _finish("double_jpeg", "valuemetric", f"q={q1}/{q2}", src, out, q1=q1, q2=q2)


def webp(image: Array, quality: int = 70) -> AttackResult:
    src = to_float01(image)
    buf = BytesIO()
    Image.fromarray(to_uint8(src), "RGB").save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0
    return _finish("webp", "valuemetric", f"q={quality}", src, out, quality=quality)


def gaussian_noise(image: Array, sigma: float = 5.0, seed: int = 0) -> AttackResult:
    src = to_float01(image)
    rng = np.random.default_rng(seed)
    out = src + rng.normal(0.0, sigma / 255.0, src.shape)
    return _finish("noise", "valuemetric", f"sigma={sigma}", src, out, sigma=sigma)


def salt_pepper(image: Array, density: float = 0.02, seed: int = 0) -> AttackResult:
    src = to_float01(image)
    rng = np.random.default_rng(seed)
    out = src.copy()
    m = rng.random(src.shape[:2])
    out[m < density / 2] = 0.0
    out[m > 1 - density / 2] = 1.0
    return _finish(
        "salt_pepper", "valuemetric", f"d={density}", src, out, density=density
    )


def gaussian_blur(image: Array, sigma: float = 1.0) -> AttackResult:
    src = to_float01(image)
    if cv2 is None:
        return _finish("blur", "valuemetric", f"sigma={sigma}", src, src)
    out = cv2.GaussianBlur(src, (0, 0), sigma)
    return _finish("blur", "valuemetric", f"sigma={sigma}", src, out, sigma=sigma)


def median_blur(image: Array, k: int = 3) -> AttackResult:
    src = to_float01(image)
    out = cv2.medianBlur(to_uint8(src), int(k)).astype(np.float32) / 255.0
    return _finish("median", "valuemetric", f"k={k}", src, out, k=k)


def unsharp(image: Array, amount: float = 1.0, sigma: float = 1.2) -> AttackResult:
    src = to_float01(image)
    blur = cv2.GaussianBlur(src, (0, 0), sigma)
    out = src + amount * (src - blur)
    return _finish("sharpen", "valuemetric", f"a={amount}", src, out, amount=amount)


def tone(
    image: Array,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    saturation: float = 1.0,
) -> AttackResult:
    src = to_float01(image)
    out = np.clip(src, 1e-6, 1.0) ** (1.0 / gamma)
    out = (out - 0.5) * contrast + 0.5 + brightness
    if saturation != 1.0:
        y, cb, cr = rgb_to_ycbcr(np.clip(out, 0, 1))
        out = ycbcr_to_rgb(
            y, 0.5 + (cb - 0.5) * saturation, 0.5 + (cr - 0.5) * saturation
        )
    tag = f"b={brightness},c={contrast},g={gamma},s={saturation}"
    return _finish(
        "tone",
        "valuemetric",
        tag,
        src,
        out,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        saturation=saturation,
    )


def posterise(image: Array, levels: int = 32) -> AttackResult:
    src = to_float01(image)
    out = np.round(src * (levels - 1)) / (levels - 1)
    return _finish("posterise", "valuemetric", f"L={levels}", src, out, levels=levels)


# ---------------------------------------------------------------------------
# Geometric
# ---------------------------------------------------------------------------


def translate(image: Array, dy: int = 7, dx: int = 11) -> AttackResult:
    """Cyclic shift.  Distortion is reported as zero: the pixels are identical,
    only their addresses moved, and a pixelwise metric comparing shifted to
    unshifted measures the shift rather than any loss of quality."""
    src = to_float01(image)
    out = np.roll(np.roll(src, dy, axis=0), dx, axis=1)
    return AttackResult(
        name="translate",
        family="geometric",
        param=f"{dy},{dx}",
        image=np.ascontiguousarray(out.astype(np.float32)),
        psnr=float("inf"),
        ssim=1.0,
        details={"dy": dy, "dx": dx},
    )


def centre_crop(image: Array, frac: float = 0.8) -> AttackResult:
    src = to_float01(image)
    h, w = src.shape[:2]
    ch, cw = max(16, int(h * frac)), max(16, int(w * frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    out = src[y0 : y0 + ch, x0 : x0 + cw]
    # Distortion is measured against the corresponding region of the source, so
    # the number reports how much the crop changed the pixels it kept rather
    # than how much of the frame it discarded.
    return AttackResult(
        name="crop",
        family="geometric",
        param=f"{frac}",
        image=np.ascontiguousarray(out),
        psnr=float("inf"),
        ssim=1.0,
        details={"frac": frac},
    )


def rescale(image: Array, factor: float = 0.5) -> AttackResult:
    src = to_float01(image)
    h, w = src.shape[:2]
    small = np.stack(
        [
            resize(src[..., c], (max(16, int(h * factor)), max(16, int(w * factor))))
            for c in range(3)
        ],
        -1,
    )
    back = np.stack([resize(small[..., c], (h, w)) for c in range(3)], -1)
    res = _finish("rescale", "geometric", f"x{factor}", src, back, factor=factor)
    res.image = np.ascontiguousarray(small.astype(np.float32))
    return res


def aspect(image: Array, ratio: float = 1.15) -> AttackResult:
    src = to_float01(image)
    h, w = src.shape[:2]
    out = np.stack([resize(src[..., c], (h, int(w * ratio))) for c in range(3)], -1)
    back = np.stack([resize(out[..., c], (h, w)) for c in range(3)], -1)
    res = _finish("aspect", "geometric", f"r={ratio}", src, back, ratio=ratio)
    res.image = np.ascontiguousarray(out.astype(np.float32))
    return res


def rotate(
    image: Array, degrees: float = 5.0, crop_inscribed: bool = True
) -> AttackResult:
    src = to_float01(image)
    h, w = src.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    out = cv2.warpAffine(
        src, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101
    )
    if crop_inscribed:
        # A real adversary crops away the wedges a rotation leaves behind; the
        # detector then faces a rotation *and* a crop, which is strictly harder.
        t = math.radians(abs(degrees))
        f = 1.0 / (abs(math.cos(t)) + abs(math.sin(t)) * max(h, w) / min(h, w))
        ch, cw = int(h * f), int(w * f)
        out = out[
            (h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw
        ]
    return AttackResult(
        name="rotate",
        family="geometric",
        param=f"{degrees}deg",
        image=np.ascontiguousarray(out.astype(np.float32)),
        psnr=float("inf"),
        ssim=1.0,
        details={"degrees": degrees},
    )


def hflip(image: Array) -> AttackResult:
    src = to_float01(image)
    return AttackResult(
        name="hflip",
        family="geometric",
        param="",
        image=np.ascontiguousarray(src[:, ::-1].copy()),
        psnr=float("inf"),
        ssim=1.0,
        details={},
    )


def elastic_warp(
    image: Array, alpha: float = 3.0, sigma: float = 48.0, seed: int = 0
) -> AttackResult:
    src = to_float01(image)
    h, w = src.shape[:2]
    rng = np.random.default_rng(seed)
    dx = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), sigma)
    n = float(np.sqrt(dx * dx + dy * dy).max()) + 1e-9
    dx *= alpha / n
    dy *= alpha / n
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = cv2.remap(
        src,
        (xx + dx).astype(np.float32),
        (yy + dy).astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return _finish(
        "warp", "geometric", f"a={alpha}", src, out, alpha=alpha, sigma=sigma
    )


def overlay(image: Array, frac: float = 0.15, seed: int = 0) -> AttackResult:
    """Paste a flat patch over part of the frame, as a caption or logo would."""
    src = to_float01(image)
    h, w = src.shape[:2]
    rng = np.random.default_rng(seed)
    ph, pw = int(h * math.sqrt(frac)), int(w * math.sqrt(frac))
    y0 = int(rng.integers(0, max(1, h - ph)))
    x0 = int(rng.integers(0, max(1, w - pw)))
    out = src.copy()
    out[y0 : y0 + ph, x0 : x0 + pw] = rng.uniform(0.1, 0.9, size=(1, 1, 3))
    return _finish("overlay", "geometric", f"f={frac}", src, out, frac=frac)


# ---------------------------------------------------------------------------
# Codebook / collusion
# ---------------------------------------------------------------------------


def build_codebook_pairs(
    hosts: Sequence[Array], marked: Sequence[Array], canon: int = 512
) -> Dict[str, Any]:
    """Aggregate a codebook from (host, marked) pairs the adversary has collected."""
    return build_codebook(None, None, canon=canon, pairs=list(zip(hosts, marked)))


def build_codebook(
    embed_fn: Optional[Callable[[Array], Array]],
    references: Optional[Sequence[Array]],
    canon: int = 512,
    pairs: Optional[Sequence[Tuple[Array, Array]]] = None,
) -> Dict[str, Any]:
    """Aggregate what many marked images reveal about the carrier set.

    This is the attack that breaks any static-carrier watermark, and it needs no
    key: mark (or collect) many images, subtract the hosts, and look for bins
    that move consistently.

    The statistic is the mean *absolute log-magnitude deviation* per bin, not the
    phase coherence usually quoted.  Phase is the wrong instrument for a
    multiplicative mark — the residual's phase at a carrier is whatever phase the
    image already had there, which is uncorrelated across images whether or not
    the carriers are shared, so phase coherence sits at the noise floor for both
    a static and a content-derived design and tells the adversary nothing.
    Magnitude deviation does discriminate: with a static carrier set the top-``K``
    bins recover roughly half of it, against a chance rate of well under one
    percent.
    """
    score = None
    n = 0
    supply = (
        list(pairs)
        if pairs is not None
        else [(r, embed_fn(r)) for r in (references or ())]
    )
    for ref, mk in supply:
        host = canonical_luma(ref, canon)
        marked = canonical_luma(mk, canon)
        if host.shape != marked.shape:
            continue
        d = np.log(np.abs(np.fft.fft2(marked.astype(np.float64))) + 1e-12) - np.log(
            np.abs(np.fft.fft2(host.astype(np.float64))) + 1e-12
        )
        sq = np.abs(d)
        if score is None or score.shape != sq.shape:
            score = sq if score is None else score
        score = sq if n == 0 else score + sq
        n += 1
    score = score / max(n, 1)
    ranked = np.argsort(score.ravel())[::-1]
    return {
        "score": score,
        "ranked": ranked,
        "n_refs": n,
        "canon": canon,
        "peak_ratio": float(np.max(score) / (np.median(score) + 1e-12)),
    }


def codebook_whiten(
    image: Array, codebook: Dict[str, Any], top_k: int = 2048, strength: float = 1.0
) -> AttackResult:
    """Whiten the bins the aggregation nominated as carriers.

    The sharpest form of the codebook attack: rather than subtracting an
    averaged residual, push every nominated bin back onto its ring mean, which
    erases any per-bin deviation there regardless of sign or phase.  Against a
    static carrier set this is close to a complete removal at a distortion cost
    no larger than the embedder paid.
    """
    src = to_float01(image)
    canon = int(codebook["canon"])
    y, cb, cr = rgb_to_ycbcr(src)
    small = resize(y, (canon, canon))
    F = np.fft.fft2(small.astype(np.float64))
    log_mag = np.log(np.abs(F) + 1e-12)
    rings = ring_index((canon, canon), 512)
    flat = rings.ravel()
    counts = np.bincount(flat, minlength=512).astype(np.float64)
    sums = np.bincount(flat, weights=log_mag.ravel(), minlength=512)
    mean = (sums / np.maximum(counts, 1.0))[flat].reshape(log_mag.shape)
    mask = np.zeros(log_mag.size, dtype=bool)
    mask[codebook["ranked"][: int(top_k)]] = True
    mask = mask.reshape(log_mag.shape)
    new_log = np.where(mask, log_mag + strength * (mean - log_mag), log_mag)
    cleaned = np.fft.ifft2(np.exp(new_log) * np.exp(1j * np.angle(F))).real
    residual = resize(
        (cleaned - small).astype(np.float32), y.shape[:2], antialias=False
    )
    out = ycbcr_to_rgb(np.clip(y + residual, 0, 1).astype(np.float32), cb, cr)
    return _finish(
        "codebook_whiten",
        "codebook",
        f"k={top_k}",
        src,
        out,
        top_k=top_k,
        strength=strength,
        n_refs=codebook["n_refs"],
    )


def codebook_subtract(
    image: Array, codebook: Dict[str, Any], removal: float = 1.0, top_k: int = 2048
) -> AttackResult:
    """Attenuate the nominated bins in proportion to how strongly they moved."""
    src = to_float01(image)
    canon = int(codebook["canon"])
    y, cb, cr = rgb_to_ycbcr(src)
    small = resize(y, (canon, canon))
    F = np.fft.fft2(small.astype(np.float64))
    # Attenuate each nominated bin in proportion to how strongly the aggregate
    # says it moved, which is the best an adversary can do without the key.
    gain = np.zeros(F.size)
    sel = codebook["ranked"][: int(top_k)]
    gain[sel] = codebook["score"].ravel()[sel]
    cleaned = np.fft.ifft2(F * np.exp(-removal * gain.reshape(F.shape))).real
    residual = resize(
        (cleaned - small).astype(np.float32), y.shape[:2], antialias=False
    )
    out = ycbcr_to_rgb(np.clip(y + residual, 0, 1).astype(np.float32), cb, cr)
    return _finish(
        "codebook",
        "codebook",
        f"r={removal}",
        src,
        out,
        removal=removal,
        top_k=top_k,
        n_refs=codebook["n_refs"],
        peak_ratio=codebook["peak_ratio"],
    )


def collusion_average(
    image: Array, others: Sequence[Array], weight: float = 0.25
) -> AttackResult:
    """Blend a marked image toward the mean of other marked images.

    Classical collusion: if every copy carries the same mark, averaging copies
    keeps the mark and cancels the content.  With per-image carriers the
    arithmetic runs the other way — averaging cancels the marks.
    """
    src = to_float01(image)
    h, w = src.shape[:2]
    acc = np.zeros_like(src, dtype=np.float64)
    for o in others:
        acc += np.stack([resize(to_float01(o)[..., c], (h, w)) for c in range(3)], -1)
    mean = acc / max(len(others), 1)
    out = (1 - weight) * src + weight * mean
    return _finish(
        "collusion",
        "codebook",
        f"w={weight}",
        src,
        out,
        weight=weight,
        n_others=len(others),
    )


def spectral_whiten(
    image: Array,
    f_lo: float = 0.05,
    f_hi: float = 0.25,
    canon: int = 512,
    strength: float = 1.0,
) -> AttackResult:
    """Flatten the log-magnitude ripple inside the carrier band.

    This is the strongest *architecture-aware* attack that does not need the
    key: the adversary is told which band the analytic mark occupies and pushes
    every bin there back onto its ring mean, erasing any per-bin deviation.  It
    bounds what the analytic stratum can promise on its own, and is exactly the
    attack that a static carrier set would also fall to — the difference being
    that here the adversary must whiten the whole band rather than a few
    thousand known bins, and pay the corresponding distortion.
    """
    src = to_float01(image)
    y, cb, cr = rgb_to_ycbcr(src)
    small = resize(y, (canon, canon))
    F = np.fft.fft2(small.astype(np.float64))
    log_mag = np.log(np.abs(F) + 1e-12)
    rings = ring_index((canon, canon), 512)
    flat = rings.ravel()
    counts = np.bincount(flat, minlength=512).astype(np.float64)
    sums = np.bincount(flat, weights=log_mag.ravel(), minlength=512)
    mean = (sums / np.maximum(counts, 1.0))[flat].reshape(log_mag.shape)
    from .common import radius_grid

    r = radius_grid((canon, canon))
    band = (r >= f_lo) & (r <= f_hi)
    new_log = np.where(band, log_mag + strength * (mean - log_mag), log_mag)
    F2 = np.exp(new_log) * np.exp(1j * np.angle(F))
    cleaned = np.fft.ifft2(F2).real
    residual = resize(
        (cleaned - small).astype(np.float32), y.shape[:2], antialias=False
    )
    out = ycbcr_to_rgb(np.clip(y + residual, 0, 1).astype(np.float32), cb, cr)
    return _finish(
        "whiten",
        "adaptive",
        f"s={strength}",
        src,
        out,
        f_lo=f_lo,
        f_hi=f_hi,
        strength=strength,
    )


# ---------------------------------------------------------------------------
# Generative
# ---------------------------------------------------------------------------

_VAE_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_vae(model_id: str, device: str):
    key = (model_id, device)
    if key in _VAE_CACHE:
        return _VAE_CACHE[key]
    import torch
    from diffusers import AutoencoderKL

    try:
        vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float16)
    except Exception:
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float16
        )
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    _VAE_CACHE[key] = vae
    return vae


def vae_roundtrip(
    image: Array,
    model_id: str = "stabilityai/sd-vae-ft-mse",
    latent_noise: float = 0.0,
    device: str = "cuda:0",
    seed: int = 0,
    tile: int = 512,
) -> AttackResult:
    """Encode to a diffusion latent, optionally perturb it, and decode.

    Perturbing the latent is what a low-strength img2img sampler does before it
    denoises, and it is the step that rewrites the mid-band spectrum an
    imperceptible linear watermark lives in.
    """
    import torch

    src = to_float01(image)
    h, w = src.shape[:2]
    vae = _get_vae(model_id, device)
    hh, ww = min(tile, (h // 8) * 8), min(tile, (w // 8) * 8)
    work = np.stack([resize(src[..., c], (hh, ww)) for c in range(3)], -1)
    x = torch.from_numpy(work).permute(2, 0, 1)[None].to(device).float()
    with torch.no_grad():
        z = vae.encode((x * 2 - 1).half()).latent_dist.mode()
        if latent_noise > 0:
            g = torch.Generator(device=device).manual_seed(seed)
            eps = torch.randn(z.shape, device=device, dtype=z.dtype, generator=g)
            z = (
                math.sqrt(max(1 - latent_noise**2, 0.0)) * z
                + latent_noise * eps * z.std()
            )
        y = vae.decode(z).sample
    rec = (y.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
    out = np.stack([resize(rec[..., c], (h, w)) for c in range(3)], -1)
    return _finish(
        "vae",
        "generative",
        f"{model_id.split('/')[-1]},n={latent_noise}",
        src,
        out,
        model=model_id,
        latent_noise=latent_noise,
    )


_PIPE_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_img2img(model_id: str, device: str):
    key = (model_id, device)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    import torch
    from diffusers import AutoPipelineForImage2Image

    torch.backends.cudnn.enabled = False
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    _PIPE_CACHE[key] = pipe
    return pipe


def img2img(
    image: Array,
    strength: float = 0.25,
    device: str = "cuda:0",
    model_id: str = "stabilityai/sd-turbo",
    seed: int = 0,
    prompt: str = "high quality photograph",
    size: int = 384,
) -> AttackResult:
    """Real diffusion regeneration through SD-Turbo."""
    import torch

    src = to_float01(image)
    h, w = src.shape[:2]
    pipe = _get_img2img(model_id, device)
    pil = Image.fromarray(to_uint8(src), "RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )
    steps = max(2, int(math.ceil(2.0 / max(strength, 1e-3))))
    gen = torch.Generator(device=device).manual_seed(seed)
    res = pipe(
        prompt=prompt,
        image=pil,
        strength=float(strength),
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=gen,
    ).images[0]
    rec = np.asarray(res.resize((w, h), Image.Resampling.LANCZOS), np.float32) / 255.0
    return _finish(
        "img2img",
        "generative",
        f"s={strength}",
        src,
        rec,
        strength=strength,
        model=model_id,
        steps=steps,
    )


def diffusion_purify(
    image: Array,
    rounds: int = 2,
    latent_noise: float = 0.25,
    device: str = "cuda:0",
    seed: int = 0,
) -> AttackResult:
    """Iterated VAE regeneration, alternating two different autoencoders.

    Repeating the round-trip is the standard way to strengthen a regeneration
    attack, and alternating models stops the image settling into one decoder's
    fixed point.
    """
    src = to_float01(image)
    cur = src
    ids = ["stabilityai/sd-vae-ft-mse", "madebyollin/sdxl-vae-fp16-fix"]
    for i in range(rounds):
        cur = vae_roundtrip(
            cur,
            model_id=ids[i % len(ids)],
            latent_noise=latent_noise,
            device=device,
            seed=seed + i,
        ).image
    return _finish(
        "purify",
        "generative",
        f"r={rounds},n={latent_noise}",
        src,
        cur,
        rounds=rounds,
        latent_noise=latent_noise,
    )


# ---------------------------------------------------------------------------
# Adaptive (white-box and surrogate)
# ---------------------------------------------------------------------------


def pgd_on_decoder(
    image: Array,
    decoder,
    message: np.ndarray,
    eps: float = 6 / 255,
    steps: int = 25,
    device: str = "cuda:0",
    canon: int = 256,
) -> AttackResult:
    """White-box removal: gradient ascent on the real decoder's loss.

    The strongest attack in the suite by construction — the adversary has the
    decoder weights, the transmitted message and an unlimited number of
    gradient steps, and is limited only by a distortion budget.
    """
    import torch
    import torch.nn.functional as F

    src = to_float01(image)
    h, w = src.shape[:2]
    work = np.stack([resize(src[..., c], (canon, canon)) for c in range(3)], -1)
    x = torch.from_numpy(work).permute(2, 0, 1)[None].to(device).float()
    m = torch.from_numpy(np.asarray(message, dtype=np.float32))[None].to(device)
    delta = torch.zeros_like(x, requires_grad=True)
    step = 2.5 * eps / steps
    for _ in range(steps):
        logits = decoder((x + delta).clamp(0, 1))
        loss = F.binary_cross_entropy_with_logits(logits, m)
        (g,) = torch.autograd.grad(loss, delta)
        delta = (delta + step * g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    adv = (x + delta.detach()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    residual = np.stack(
        [resize((adv - work)[..., c], (h, w), antialias=False) for c in range(3)], -1
    )
    out = src + residual
    return _finish(
        "pgd",
        "adaptive",
        f"eps={eps * 255:.0f}/255",
        src,
        out,
        eps=float(eps),
        steps=steps,
    )


def surrogate_removal(
    image: Array,
    remover,
    device: str = "cuda:0",
    canon: int = 256,
    budget: float = 0.06,
) -> AttackResult:
    """Black-box removal by a network trained to strip marks it has never seen."""
    import torch

    src = to_float01(image)
    h, w = src.shape[:2]
    work = np.stack([resize(src[..., c], (canon, canon)) for c in range(3)], -1)
    x = torch.from_numpy(work).permute(2, 0, 1)[None].to(device).float()
    with torch.no_grad():
        y = remover(x, budget=budget)
    rec = y[0].permute(1, 2, 0).cpu().numpy()
    residual = np.stack(
        [resize((rec - work)[..., c], (h, w), antialias=False) for c in range(3)], -1
    )
    return _finish(
        "surrogate", "adaptive", f"b={budget}", src, src + residual, budget=budget
    )


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def composite(
    image: Array,
    steps: Sequence[Callable[[Array], AttackResult]],
    name: str = "stacked",
    param: str = "",
) -> AttackResult:
    src = to_float01(image)
    cur = src
    applied = []
    families = []
    for fn in steps:
        r = fn(cur)
        cur = r.image
        applied.append(f"{r.name}({r.param})")
        families.append(r.family)
    res = _finish(
        name, "composite", param or "->".join(applied), src, cur, chain=applied
    )
    res.image = np.ascontiguousarray(cur.astype(np.float32))
    if cur.shape != src.shape:
        res.psnr, res.ssim = float("inf"), 1.0

    # A stack that moves the image cannot be scored by a pixel metric, because
    # the metric would be comparing misaligned content and would report damage
    # the viewer never sees.  That exemption is meant to cover the *geometry*,
    # and on its own it covers far too much: a stack whose first step is a
    # regeneration that visibly rewrites the picture inherits a free pass the
    # moment a crop is appended to it.
    #
    # So score the rest of the stack on its own.  Dropping the geometric steps
    # leaves an alignment-preserving chain, PSNR and SSIM mean what they usually
    # mean on it, and a stack is within budget only if that part is.  Geometry
    # stays exempt; nothing else does.
    keep = [fn for fn, fam in zip(steps, families) if fam != "geometric"]
    if keep and len(keep) < len(steps):
        cur2 = src
        for fn in keep:
            cur2 = fn(cur2).image
        if cur2.shape == src.shape:
            q = _finish(name, "composite", "", src, cur2)
            res.details["photometric_psnr"] = float(q.psnr)
            res.details["photometric_ssim"] = float(q.ssim)
    return res


__all__ = [
    "AttackResult",
    "aspect",
    "build_codebook",
    "build_codebook_pairs",
    "centre_crop",
    "codebook_subtract",
    "codebook_whiten",
    "collusion_average",
    "composite",
    "diffusion_purify",
    "double_jpeg",
    "elastic_warp",
    "gaussian_blur",
    "gaussian_noise",
    "hflip",
    "img2img",
    "jpeg",
    "median_blur",
    "overlay",
    "pgd_on_decoder",
    "posterise",
    "rescale",
    "rotate",
    "salt_pepper",
    "spectral_whiten",
    "surrogate_removal",
    "tone",
    "translate",
    "unsharp",
    "vae_roundtrip",
    "webp",
]

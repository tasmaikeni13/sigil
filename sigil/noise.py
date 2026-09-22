"""Attack simulation used to train and to stress the learned stratum.

Every transform here is written so a gradient can reach the encoder.  Where an
operation is genuinely non-differentiable — rounding to 8 bits, JPEG's
quantiser, a diffusion sampler — the forward pass runs the *real* operation and
the backward pass uses a straight-through surrogate.  Training against an
approximation of JPEG and then testing against real JPEG is a classic way to
overstate robustness; running the real thing forward removes that gap.

The pieces that matter most are the generative ones.  A VAE encode/decode
round-trip rewrites precisely the band an imperceptible linear watermark
occupies, which is why regeneration defeats spectral marks.  Putting a real
VAE — with latent noise, which is what a low-strength img2img sampler actually
does to the code — inside the training loop is what buys robustness to it, and
carrying two different VAEs stops the encoder from overfitting one decoder's
reconstruction quirks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Straight-through helpers
# ---------------------------------------------------------------------------


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass, identity in the backward pass."""
    return x + (torch.round(x) - x).detach()


def bpda(x: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Use ``exact`` in the forward pass and ``x``'s gradient in the backward."""
    return x + (exact - x).detach()


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------

_JPEG_LUMA_Q = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

_JPEG_CHROMA_Q = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def _dct_matrix(n: int = 8) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float32).view(-1, 1)
    i = torch.arange(n, dtype=torch.float32).view(1, -1)
    m = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= math.sqrt(1.0 / n)
    m[1:] *= math.sqrt(2.0 / n)
    return m


_DCT8 = _dct_matrix(8)


def _quality_scale(quality: float) -> float:
    q = float(min(max(quality, 1.0), 99.0))
    return (5000.0 / q) / 100.0 if q < 50 else (200.0 - 2 * q) / 100.0


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (b - y) * 0.564
    cr = 0.5 + (r - y) * 0.713
    return torch.stack([y, cb, cr], dim=1)


def _ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y, cb, cr = x[:, 0], x[:, 1], x[:, 2]
    r = y + 1.403 * (cr - 0.5)
    b = y + 1.773 * (cb - 0.5)
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.stack([r, g, b], dim=1)


def jpeg(x: torch.Tensor, quality: float = 50.0) -> torch.Tensor:
    """Differentiable JPEG: exact quantiser forward, straight-through backward.

    Only the quantiser is lossy — entropy coding is bit-exact — so running the
    real quantiser forward makes this pixel-identical to a JPEG round-trip at
    the same tables, up to chroma subsampling.
    """
    b, c, h, w = x.shape
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    hh, ww = x.shape[-2:]
    ycc = _rgb_to_ycbcr(x.clamp(0, 1)) * 255.0 - 128.0
    d = _DCT8.to(x.device, x.dtype)
    scale = _quality_scale(quality)
    ql = (_JPEG_LUMA_Q.to(x.device, x.dtype) * scale).clamp(1, 255)
    qc = (_JPEG_CHROMA_Q.to(x.device, x.dtype) * scale).clamp(1, 255)

    tiles = ycc.reshape(b, 3, hh // 8, 8, ww // 8, 8).permute(0, 1, 2, 4, 3, 5)
    coef = d @ tiles @ d.transpose(0, 1)
    q = torch.stack([ql, qc, qc]).view(1, 3, 1, 1, 8, 8)
    coef = ste_round(coef / q) * q
    out = d.transpose(0, 1) @ coef @ d
    out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, 3, hh, ww)
    rgb = _ycbcr_to_rgb((out + 128.0) / 255.0)
    return rgb[:, :, :h, :w].clamp(0, 1)


# ---------------------------------------------------------------------------
# Geometry and photometry
# ---------------------------------------------------------------------------


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(
        F.pad(x, (r, r, 0, 0), mode="reflect"),
        k.view(1, 1, 1, -1).expand(c, 1, 1, -1),
        groups=c,
    )
    return F.conv2d(
        F.pad(x, (0, 0, r, r), mode="reflect"),
        k.view(1, 1, -1, 1).expand(c, 1, -1, 1),
        groups=c,
    )


def resize_roundtrip(x: torch.Tensor, factor: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    small = F.interpolate(
        x,
        size=(max(8, int(h * factor)), max(8, int(w * factor))),
        mode="area" if factor < 1 else "bilinear",
        align_corners=None if factor < 1 else False,
    )
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def random_crop_resize(x: torch.Tensor, frac: float, generator=None) -> torch.Tensor:
    h, w = x.shape[-2:]
    ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
    y0 = int(torch.randint(0, h - ch + 1, (1,), generator=generator).item())
    x0 = int(torch.randint(0, w - cw + 1, (1,), generator=generator).item())
    return F.interpolate(
        x[:, :, y0 : y0 + ch, x0 : x0 + cw],
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )


def rotate(x: torch.Tensor, degrees: float, zoom: float = 1.0) -> torch.Tensor:
    th = math.radians(degrees)
    c, s = math.cos(th) / zoom, math.sin(th) / zoom
    m = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=x.device, dtype=x.dtype)
    grid = F.affine_grid(
        m.unsqueeze(0).expand(x.shape[0], -1, -1), x.shape, align_corners=False
    )
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def elastic(
    x: torch.Tensor, amplitude: float, sigma: float = 12.0, generator=None
) -> torch.Tensor:
    b, _, h, w = x.shape
    f = torch.randn(b, 2, h, w, device=x.device, dtype=x.dtype, generator=generator)
    f = gaussian_blur(f, sigma)
    f = f / (f.flatten(1).abs().amax(dim=1).view(-1, 1, 1, 1) + 1e-6) * amplitude
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx + f[:, 0] * 2.0 / w, yy + f[:, 1] * 2.0 / h], dim=-1)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def colour_jitter(
    x: torch.Tensor, brightness=0.0, contrast=1.0, saturation=1.0, gamma=1.0
) -> torch.Tensor:
    y = x
    if gamma != 1.0:
        y = y.clamp_min(1e-5) ** (1.0 / gamma)
    if contrast != 1.0 or brightness != 0.0:
        y = (y - 0.5) * contrast + 0.5 + brightness
    if saturation != 1.0:
        g = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = g + (y - g) * saturation
    return y.clamp(0, 1)


def median_filter(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    p = k // 2
    patches = F.unfold(F.pad(x, (p, p, p, p), mode="reflect"), k)
    b, _, locations = patches.shape
    patches = patches.view(b, x.shape[1], k * k, locations)
    return patches.median(dim=2).values.view_as(x)


def resample_roundtrip(
    x: torch.Tensor, degrees: float = 0.0, zoom: float = 1.0
) -> torch.Tensor:
    """Rotate/zoom and immediately undo it, keeping only the resampling damage.

    This is what the detector actually faces.  A positional code cannot be read
    from a rotated image at all, so training on rotations alone teaches the
    decoder nothing: those batches are unlearnable and the gradient is noise.
    What the detector does instead is *resynchronise* — it applies the inverse
    transform and then decodes — so the damage it has to survive is two bilinear
    resamplings, not a misalignment.  Training on exactly that closes the gap:
    without it, a fifteen-degree rotation is unreadable even when the detector
    guesses the angle exactly right.
    """
    if degrees == 0.0 and zoom == 1.0:
        return x
    y = rotate(x, degrees, zoom=zoom)
    return rotate(y, -degrees, zoom=1.0 / max(zoom, 1e-6))


def quantise8(x: torch.Tensor) -> torch.Tensor:
    return ste_round(x.clamp(0, 1) * 255.0) / 255.0


def band_whiten(
    x: torch.Tensor,
    f_lo: float = 0.04,
    f_hi: float = 0.28,
    strength: float = 1.0,
    smooth: int = 9,
) -> torch.Tensor:
    """Flatten the log-magnitude ripple inside a frequency band.

    This is the strongest attack an adversary can mount knowing only the
    *architecture* — no key, no model.  It is exactly what defeats the analytic
    stratum, so the learned stratum is trained through it: whatever the learned
    mark ends up looking like, it must not be a per-bin deviation inside that
    band.
    """
    b, c, h, w = x.shape
    F = torch.fft.fft2(x.to(torch.float32))
    mag = F.abs().clamp_min(1e-8)
    logm = mag.log()
    # A local average over the frequency plane stands in for the ring mean and
    # keeps the whole operation differentiable.
    sm = F_torch_avg(torch.fft.fftshift(logm, dim=(-2, -1)), smooth)
    sm = torch.fft.ifftshift(sm, dim=(-2, -1))
    fy = torch.fft.fftfreq(h, device=x.device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, -1)
    r = torch.sqrt(fy * fy + fx * fx)
    band = ((r >= f_lo) & (r <= f_hi)).to(x.dtype)
    new_log = logm + strength * band * (sm - logm)
    out = torch.fft.ifft2(torch.polar(new_log.exp(), F.angle())).real
    return out.clamp(0, 1).to(x.dtype)


def F_torch_avg(t: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    return F.avg_pool2d(F.pad(t, (p, p, p, p), mode="replicate"), k, 1)


# ---------------------------------------------------------------------------
# Generative round-trips
# ---------------------------------------------------------------------------


class VAEBank(nn.Module):
    """One or more autoencoders used as regeneration surrogates.

    ``latent_noise`` reproduces what a low-strength img2img sampler does before
    it denoises: it perturbs the code, so the decoder reconstructs a *plausible*
    image rather than a faithful one.  That, and not the encode/decode itself,
    is what destroys a spectral watermark, so it is the part worth training
    against.
    """

    def __init__(
        self,
        model_ids: Sequence[str],
        device: str = "tpu",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        self.models = nn.ModuleList()
        self.ids: List[str] = []
        for mid in model_ids:
            try:
                vae = AutoencoderKL.from_pretrained(mid, torch_dtype=dtype)
            except Exception:
                vae = AutoencoderKL.from_pretrained(
                    mid, subfolder="vae", torch_dtype=dtype
                )
            vae.requires_grad_(False)
            vae.eval()
            self.models.append(vae.to(device))
            self.ids.append(mid)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def _scale(self, idx: int) -> float:
        return float(getattr(self.models[idx].config, "scaling_factor", 0.18215))

    def forward(
        self, x: torch.Tensor, index: int = 0, latent_noise: float = 0.0, generator=None
    ) -> torch.Tensor:
        vae = self.models[index]
        z = vae.encode((x * 2 - 1).to(self.dtype)).latent_dist.mode()
        if latent_noise > 0:
            eps = torch.randn(
                z.shape, device=z.device, dtype=z.dtype, generator=generator
            )
            z = (
                math.sqrt(max(1.0 - latent_noise**2, 0.0)) * z
                + latent_noise * eps * z.std()
            )
        out = vae.decode(z).sample
        out = ((out.float() + 1) / 2).clamp(0, 1)
        # Half-precision autoencoders occasionally overflow on saturated
        # content.  A poisoned batch would propagate NaN into the weights, so
        # fall back to the input wherever the reconstruction is not finite.
        return torch.where(torch.isfinite(out), out, x)


# ---------------------------------------------------------------------------
# Noise-layer scheduler
# ---------------------------------------------------------------------------


@dataclass
class NoiseSpec:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    weight: float = 1.0


class TurboRegen:
    """Real SD-Turbo img2img inside the training loop, via BPDA.

    A VAE round-trip with latent noise is a good proxy for regeneration but not
    the thing itself: the sampler's denoising step moves the image
    *semantically*, and a model trained only against the proxy reads a real
    img2img output at barely above chance.  The pipeline is not differentiable,
    so the forward pass runs it for real and the backward pass treats it as the
    identity — standard BPDA, and enough to teach the encoder to put the mark
    where regeneration preserves it.
    """

    def __init__(
        self,
        device: str = "tpu",
        model_id: str = "stabilityai/sd-turbo",
        size: int = 256,
    ):
        import torch as _t
        from diffusers import AutoPipelineForImage2Image

        torch_dev = "cpu" if str(device).startswith("tpu") else device
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=_t.float32
        ).to(torch_dev)
        self.pipe.set_progress_bar_config(disable=True)
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None
        self.device = device
        self.torch_device = torch_dev
        self.size = size

    def __call__(self, x: torch.Tensor, strength: float = 0.25) -> torch.Tensor:
        import numpy as _np
        from PIL import Image as _I

        with torch.no_grad():
            arr = (
                (x.detach().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            )
            steps = max(2, int(math.ceil(2.0 / max(strength, 1e-3))))
            # One batched call, not one call per image: the sampler is the
            # dominant cost in the whole training step, and looping causes underutilisation.
            pil = [_I.fromarray(a, "RGB").resize((self.size, self.size)) for a in arr]
            res = self.pipe(
                prompt=[""] * len(pil),
                image=pil,
                strength=float(strength),
                num_inference_steps=steps,
                guidance_scale=0.0,
            ).images
            outs = [
                _np.asarray(r.resize((x.shape[-1], x.shape[-2])), dtype=_np.float32)
                / 255.0
                for r in res
            ]
        y = torch.from_numpy(_np.stack(outs)).permute(0, 3, 1, 2).to(x.device, x.dtype)
        return bpda(x, y)


class NoiseLayer(nn.Module):
    """Sample one attack per batch from a weighted menu.

    ``severity`` in [0, 1] interpolates every parameter range from barely-there
    to full strength, and biases the menu toward the identity while it is low.
    A cold encoder emits a residual of exactly zero, so if the first batches are
    cropped to a third and rotated 25 degrees there is no signal to grow the
    mark from and the message loss sits at ln 2 indefinitely.  Ramping the
    opponent is what lets a readable code establish itself before it has to
    survive anything.
    """

    def __init__(
        self,
        vaes: Optional[VAEBank] = None,
        seed: int = 0,
        severity: float = 1.0,
        turbo: "Optional[TurboRegen]" = None,
    ):
        super().__init__()
        self.vaes = vaes
        self.turbo = turbo
        self.rng = np.random.default_rng(seed)
        self.severity = float(severity)
        self.specs: List[NoiseSpec] = []
        self._build()

    def _lerp(self, mild: float, hard: float) -> float:
        return float(mild + (hard - mild) * self.severity)

    def _u(self, lo_mild, lo_hard, hi_mild, hi_hard) -> float:
        return float(
            self.rng.uniform(self._lerp(lo_mild, lo_hard), self._lerp(hi_mild, hi_hard))
        )

    def _resync(self) -> Tuple[float, float]:
        """An angle and the zoom that goes with it.

        Sampling the two independently spends most of the resync budget on pairs
        no attacker produces.  Rotating an image leaves blank wedges, so an
        adversary who rotates also crops them away, and the crop is then fixed by
        the angle: the inscribed rectangle of a theta-rotated frame is magnified
        by exactly ``1/(|cos| + |sin|)``.  That is also the pair the detector's
        own hypothesis bank is built from, so training on it is training on the
        path the detector will actually walk.

        Half the draws stay independent, because plain rotation with padding and
        rotation followed by an unrelated resize both happen too.
        """
        deg = self._u(-3, -45, 3, 45)
        if self.rng.random() < 0.5:
            t = math.radians(abs(deg))
            return deg, 1.0 / (math.cos(t) + math.sin(t))
        return deg, self._u(0.95, 0.35, 1.0, 1.0)

    def _build(self) -> None:
        r = self.rng
        add = self.specs.append
        add(NoiseSpec("identity", lambda x: x, 1.0))
        add(NoiseSpec("jpeg", lambda x: jpeg(x, self._u(88, 25, 98, 95)), 2.5))
        add(
            NoiseSpec(
                "noise",
                lambda x: (
                    x + torch.randn_like(x) * self._u(0.5, 2, 2, 14) / 255
                ).clamp(0, 1),
                1.5,
            )
        )
        add(
            NoiseSpec(
                "blur", lambda x: gaussian_blur(x, self._u(0.2, 0.4, 0.5, 2.2)), 1.2
            )
        )
        add(
            NoiseSpec(
                "resize",
                lambda x: resize_roundtrip(x, self._u(0.85, 0.25, 0.95, 0.8)),
                1.2,
            )
        )
        add(
            NoiseSpec(
                "crop",
                lambda x: random_crop_resize(x, self._u(0.9, 0.35, 0.99, 0.95)),
                1.8,
            )
        )
        add(NoiseSpec("rotate", lambda x: rotate(x, self._u(-3, -25, 3, 25)), 1.5))
        add(
            NoiseSpec("elastic", lambda x: elastic(x, self._u(0.5, 1.0, 1.5, 9.0)), 1.0)
        )
        add(
            NoiseSpec(
                "median",
                lambda x: median_filter(
                    x, 3 if self.severity < 0.5 else int(r.choice([3, 5]))
                ),
                0.8,
            )
        )
        add(
            NoiseSpec(
                "colour",
                lambda x: colour_jitter(
                    x,
                    brightness=self._u(-0.02, -0.08, 0.02, 0.08),
                    contrast=self._u(0.97, 0.8, 1.03, 1.25),
                    saturation=self._u(0.9, 0.6, 1.1, 1.4),
                    gamma=self._u(0.95, 0.75, 1.05, 1.35),
                ),
                1.2,
            )
        )
        add(NoiseSpec("quantise", quantise8, 0.5))
        add(NoiseSpec("hflip", lambda x: torch.flip(x, dims=[-1]), 0.4))
        add(NoiseSpec("resync", lambda x: resample_roundtrip(x, *self._resync()), 2.2))
        add(
            NoiseSpec(
                "whiten", lambda x: band_whiten(x, strength=self._lerp(0.2, 1.0)), 1.6
            )
        )
        if self.vaes is not None:
            n = len(self.vaes.models)
            add(
                NoiseSpec(
                    "vae",
                    lambda x: self.vaes(
                        x,
                        index=int(r.integers(0, n)),
                        latent_noise=self._u(0.0, 0.0, 0.1, 0.45),
                    ),
                    4.0,
                )
            )
            add(
                NoiseSpec(
                    "vae_jpeg",
                    lambda x: jpeg(
                        self.vaes(
                            x,
                            index=int(r.integers(0, n)),
                            latent_noise=self._u(0.0, 0.0, 0.08, 0.35),
                        ),
                        self._u(85, 40, 96, 90),
                    ),
                    1.5,
                )
            )
            add(
                NoiseSpec(
                    "vae_crop",
                    lambda x: random_crop_resize(
                        self.vaes(
                            x,
                            index=int(r.integers(0, n)),
                            latent_noise=self._u(0.0, 0.0, 0.1, 0.4),
                        ),
                        self._u(0.9, 0.45, 0.99, 0.95),
                    ),
                    1.2,
                )
            )
        if self.turbo is not None:
            add(
                NoiseSpec(
                    "img2img",
                    lambda x: self.turbo(x, strength=self._u(0.08, 0.10, 0.15, 0.45)),
                    3.0,
                )
            )
            add(
                NoiseSpec(
                    "img2img_jpeg",
                    lambda x: jpeg(
                        self.turbo(x, strength=self._u(0.08, 0.10, 0.15, 0.35)),
                        self._u(85, 45, 96, 90),
                    ),
                    1.2,
                )
            )

    def menu(self) -> List[str]:
        return [s.name for s in self.specs]

    def forward(
        self, x: torch.Tensor, name: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        if name is not None:
            spec = next(s for s in self.specs if s.name == name)
        else:
            w = np.asarray([s.weight for s in self.specs], dtype=np.float64)
            # While severity is low most of the mass sits on the identity, so
            # the encoder can find a readable code before it has to defend one.
            w[0] = 1.0 + 12.0 * (1.0 - self.severity) ** 2
            spec = self.specs[int(self.rng.choice(len(self.specs), p=w / w.sum()))]
        try:
            y = spec.fn(x)
        except Exception:
            return x, "identity"
        if not torch.isfinite(y).all():
            return x, "identity"
        return y, spec.name


def pgd_attack(
    decoder: nn.Module,
    x: torch.Tensor,
    target_bits: torch.Tensor,
    eps: float = 6 / 255,
    steps: int = 3,
) -> torch.Tensor:
    """Small white-box removal attempt against the decoder, for adversarial training.

    Training against a searching opponent rather than a fixed menu is what keeps
    the mark from living in whatever narrow feature the fixed menu happens to
    leave alone.
    """
    delta = torch.zeros_like(x).uniform_(-eps, eps).requires_grad_(True)
    for _ in range(steps):
        logits = decoder((x + delta).clamp(0, 1))
        loss = F.binary_cross_entropy_with_logits(logits, target_bits)
        (g,) = torch.autograd.grad(loss, delta)
        delta = (
            (delta + (eps / max(steps - 1, 1)) * 1.5 * g.sign())
            .clamp(-eps, eps)
            .detach()
        )
        delta.requires_grad_(True)
    return (x + delta.detach()).clamp(0, 1)


__all__ = [
    "NoiseLayer",
    "NoiseSpec",
    "TurboRegen",
    "VAEBank",
    "band_whiten",
    "bpda",
    "colour_jitter",
    "elastic",
    "gaussian_blur",
    "jpeg",
    "median_filter",
    "pgd_attack",
    "quantise8",
    "random_crop_resize",
    "resample_roundtrip",
    "resize_roundtrip",
    "rotate",
    "ste_round",
]

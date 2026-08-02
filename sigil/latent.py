"""Neural building blocks for the learned SIGIL stratum.

The learned stratum embeds a message as a spatially repeated residual and
trains its decoder against the transformations used in the benchmark. This
gives the decoder a direct signal for crops, resampling, compression, and
generative reconstruction.

Two design commitments make the stratum's null distribution exact rather than
fitted, and its robustness structural rather than incidental.

Periodic spatial coding
    The message is rendered as a spatially periodic field and the decoder ends
    in a global average pool.  Every sufficiently large window of the image
    therefore carries the whole message, which is what makes cropping and
    translation survivable instead of catastrophic.

Nonce-keyed messages
    The transmitted message is ``PRF(key, nonce)`` for a per-image random nonce,
    never a fixed codeword.  A fixed codeword would be a static codebook for the
    learned stratum — an adversary could average the residuals of many marked
    images and subtract the mean.  With per-image nonces that average is noise,
    exactly as in the analytic stratum.  Detection searches the nonce space
    exhaustively (one matrix product) and pays Bonferroni for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import derive_key

Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentConfig:
    canon: int = 256
    # A finer grid is not about payload, it is about how much of the message a
    # crop leaves behind.  With 16 cells per side a 50% centre crop keeps 64
    # cells, and the statistic's ceiling — sqrt(valid cells) even at perfect bit
    # accuracy — falls below the threshold the multiplicity demands.  Doubling
    # the grid quadruples the surviving evidence and puts that ceiling far above
    # the line.
    grid: int = 32  # message cells per side; also the decoder's stride
    n_bits: int = 1024  # <= grid * grid
    nonce_bits: int = 20
    width: int = 64  # encoder/decoder base width
    strength: float = 0.045  # residual amplitude before perceptual masking
    # Stable domain-separation namespace for learned message derivation.
    master_key: bytes = b"sigil-v1-latent"


# ---------------------------------------------------------------------------
# Message coding
# ---------------------------------------------------------------------------


def message_for_nonce(nonce: int, cfg: LatentConfig) -> np.ndarray:
    """The ``n_bits`` codeword a nonce maps to, via the keyed PRF."""
    digest = derive_key(cfg.master_key, "message", int(nonce).to_bytes(8, "little"))
    need = (cfg.n_bits + 7) // 8
    raw = b""
    counter = 0
    while len(raw) < need:
        raw += derive_key(
            cfg.master_key,
            f"message-{counter}",
            int(nonce).to_bytes(8, "little"),
            digest,
        )
        counter += 1
    bits = np.unpackbits(np.frombuffer(raw[:need], dtype=np.uint8))[: cfg.n_bits]
    return bits.astype(np.int64)


def message_table(cfg: LatentConfig, device=None) -> torch.Tensor:
    """All ``2^nonce_bits`` codewords as a +-1 matrix, for exhaustive search.

    Building it costs one pass; searching it afterwards is a single matrix
    product, which is why an exhaustive nonce search is cheaper than any
    cleverness.  The table is cached on the module by :class:`LatentDetector`.
    """
    n = 1 << cfg.nonce_bits
    out = np.empty((n, cfg.n_bits), dtype=np.float32)
    for v in range(n):
        out[v] = message_for_nonce(v, cfg) * 2.0 - 1.0
    t = torch.from_numpy(out)
    return t.to(device) if device is not None else t


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(min(8, c), c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1)
        self.norm = _norm(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.b1 = ConvBlock(c, c)
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.norm = _norm(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm(self.conv(self.b1(x))))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Render a message as an additive residual on the canonical grid.

    The message is laid out *positionally*: bit ``k`` occupies cell ``k`` of a
    ``grid x grid`` array, which is then upsampled with convolutions to canonical
    resolution and shaped by a U-Net.  The decoder's readout is the exact inverse
    — one logit per cell — and that pairing is what makes the stratum trainable
    at all.

    The positional field and per-cell readout are deliberate. A message has a
    stable spatial address during embedding, and the decoder reads the same
    address at every location. That structure gives the network a short path
    from message bits to pixels while keeping the message redundant across the
    image, which is the combination needed for crop tolerance.
    """

    def __init__(self, cfg: LatentConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        ups = int(round(math.log2(cfg.canon // cfg.grid)))
        layers = [nn.Conv2d(1, w, 3, 1, 1), _norm(w), nn.SiLU(inplace=True)]
        for _ in range(ups):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(w, w, 3, 1, 1),
                _norm(w),
                nn.SiLU(inplace=True),
            ]
        layers += [nn.Conv2d(w, w, 3, 1, 1)]
        self.msg_up = nn.Sequential(*layers)

        self.stem = ConvBlock(3, w)
        self.d1 = nn.Sequential(ConvBlock(2 * w, 2 * w, stride=2), ResBlock(2 * w))
        self.d2 = nn.Sequential(ConvBlock(2 * w, 4 * w, stride=2), ResBlock(4 * w))
        self.mid = nn.Sequential(ResBlock(4 * w), ResBlock(4 * w))
        self.u2 = ConvBlock(4 * w + 4 * w, 2 * w)
        self.u1 = ConvBlock(2 * w + 2 * w, w)
        self.out = nn.Conv2d(w + w + 3, 3, 3, 1, 1)

    def message_grid(self, m: torch.Tensor) -> torch.Tensor:
        """Bits laid out one per cell, as a (B, 1, grid, grid) signed array."""
        g = self.cfg.grid
        flat = torch.zeros(m.shape[0], g * g, device=m.device, dtype=m.dtype)
        flat[:, : self.cfg.n_bits] = m * 2.0 - 1.0
        return flat.view(-1, 1, g, g)

    def message_field(self, m: torch.Tensor, h: int, w: int) -> torch.Tensor:
        z = self.msg_up(self.message_grid(m))
        if z.shape[-2:] != (h, w):
            z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return z

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f = self.message_field(m, h, w)
        s0 = torch.cat([self.stem(x), f], dim=1)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        mid = self.mid(s2)
        u2 = self.u2(
            torch.cat(
                [
                    F.interpolate(mid, size=s1.shape[-2:], mode="nearest"),
                    F.interpolate(s2, size=s1.shape[-2:], mode="nearest"),
                ],
                1,
            )
        )
        u1 = self.u1(
            torch.cat(
                [
                    F.interpolate(u2, size=s0.shape[-2:], mode="nearest"),
                    s0[:, : 2 * self.cfg.width],
                ],
                1,
            )
        )
        raw = self.out(torch.cat([u1, f, x], dim=1))
        # Normalise the residual to unit RMS rather than squashing it.  A tanh
        # here is fatal: the message loss always prefers a louder mark, the
        # activation saturates within a few hundred steps, its derivative goes to
        # zero, and the encoder freezes at a constant pattern.  Normalising fixes
        # the amplitude the caller asked for, keeps the gradient path linear, and
        # makes the fidelity budget an explicit scalar.
        rms = raw.flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1, 1)
        return (raw / (rms + 1e-6)).clamp(-4.0, 4.0)


def perceptual_mask(x: torch.Tensor, floor: float = 0.35) -> torch.Tensor:
    """Local-activity mask in [floor, 1]: more mark where the image is busy.

    Contrast masking is the strongest perceptual lever available — the same
    residual is far less visible over texture than over a clear sky.  The floor
    keeps a usable amount of mark in flat regions, which matters because flat
    regions are also where a regeneration attack reconstructs most faithfully.
    """
    g = x.mean(dim=1, keepdim=True)
    mu = F.avg_pool2d(g, 9, 1, 4)
    var = F.avg_pool2d((g - mu) ** 2, 9, 1, 4).clamp_min(0)
    act = var.sqrt()
    act = act / (act.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return floor + (1.0 - floor) * act.clamp(0, 1).pow(0.5)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Read the message back: one logit per message cell."""

    def __init__(self, cfg: LatentConfig):
        super().__init__()
        w = cfg.width
        self.cfg = cfg
        n_stride = int(round(math.log2(cfg.canon // cfg.grid)))
        # Full-resolution blocks preserve the small residual until the decoder
        # has extracted it. Downsampling at the input would alias away the
        # signal before later layers could learn where to place it.
        blocks = [ConvBlock(3, w), ResBlock(w), ResBlock(w)]
        c = w
        for i in range(n_stride):
            co = min(4 * w, c * 2)
            blocks += [ConvBlock(c, co, stride=2), ResBlock(co)]
            c = co
        blocks += [ResBlock(c)]
        self.trunk = nn.Sequential(*blocks)
        # One logit per cell from a shared 1x1 filter. Sharing the readout across
        # cells mirrors the repeated structure of the encoder and keeps the
        # detector translation tolerant.
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(self.trunk(x), self.cfg.grid)
        return self.head(z).flatten(1)[:, : self.cfg.n_bits]


class Remover(nn.Module):
    """Adversary network: strip the mark while keeping the image intact.

    Trained jointly against the decoder.  Without it the learned stratum is only
    robust to the fixed attack list in the noise layer; with it, the encoder has
    to survive an opponent that is *searching* for whatever regularity the mark
    leaves behind, which is the closest tractable stand-in for an adaptive
    real-world attacker.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            ConvBlock(3, w),
            ResBlock(w),
            ConvBlock(w, 2 * w, stride=2),
            ResBlock(2 * w),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(2 * w, w),
            ResBlock(w),
            nn.Conv2d(w, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, budget: float = 0.06) -> torch.Tensor:
        return (x + budget * torch.tanh(self.net(x))).clamp(0, 1)


__all__ = [
    "ConvBlock",
    "Decoder",
    "Encoder",
    "LatentConfig",
    "Remover",
    "ResBlock",
    "message_for_nonce",
    "message_table",
    "perceptual_mask",
]

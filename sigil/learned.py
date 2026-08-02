"""SIGIL-L: deployment wrapper for the learned stratum.

The transmitted message is never a fixed codeword — that would be a static
codebook, and averaging the residuals of enough marked images would recover it.
Instead each image carries ``m(nu)``, the codeword of a per-image random nonce
under a keyed linear code over GF(2).  Averaging across images then cancels, in
exactly the way the analytic stratum's content-derived carriers cancel.

Blind detection has to find the nonce, which naively means correlating the
decoder's logits against every one of ``2^nonce_bits`` codewords.  Choosing a
*linear* code makes that a Walsh-Hadamard transform: scatter the logits into
bins indexed by the code's generator rows, transform, and every codeword
correlation appears at once in ``O(2^k k)`` time.  A 24-bit nonce space is then
a few tens of milliseconds rather than hours, and the p-value simply pays
Bonferroni for the ``2^k`` hypotheses it searched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    apply_canonical_residual_rgb,
    key_stream_rng,
    resize,
    to_float01,
)
from .latent import Decoder, Encoder, LatentConfig, perceptual_mask
from .stats import Evidence, bonferroni, rademacher_pvalue

Array = np.ndarray


# ---------------------------------------------------------------------------
# Keyed linear code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearCode:
    """``m_i(nu) = <nu, rows_i> XOR mask_i`` over GF(2)."""

    rows: np.ndarray  # (n_bits,) packed generator rows, values < 2^k
    mask: np.ndarray  # (n_bits,) in {0, 1}
    k: int

    @property
    def size(self) -> int:
        return 1 << self.k

    def codeword(self, nonce: int) -> np.ndarray:
        par = np.array(
            [bin(int(r) & int(nonce)).count("1") & 1 for r in self.rows], dtype=np.int64
        )
        return par ^ self.mask


def build_code(master_key: bytes, n_bits: int, k: int) -> LinearCode:
    rng = key_stream_rng(master_key, "linear-code", k.to_bytes(2, "little"))
    rows = rng.integers(0, 1 << k, size=n_bits, dtype=np.int64)
    mask = rng.integers(0, 2, size=n_bits).astype(np.int64)
    return LinearCode(rows=rows, mask=mask, k=k)


def fwht(a: torch.Tensor) -> torch.Tensor:
    """In-place-style fast Walsh-Hadamard transform along the last axis."""
    n = a.shape[-1]
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    lead = a.shape[:-1]
    x = a.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        u = x[:, :, 0, :]
        v = x[:, :, 1, :]
        x = torch.stack([u + v, u - v], dim=2).reshape(-1, n)
        h *= 2
    return x.reshape(*lead, n)


def correlate_all_nonces(logits: torch.Tensor, code: LinearCode) -> torch.Tensor:
    """Correlation of ``logits`` with every codeword, via one transform."""
    return correlate_all_nonces_batch(logits.unsqueeze(0), code)[0]


def correlate_all_nonces_batch(
    logits: torch.Tensor, code: LinearCode, chunk: int = 16
) -> torch.Tensor:
    """Every codeword correlation for a *batch* of logit vectors.

    The detector evaluates a hundred-odd geometric hypotheses, and running one
    transform per hypothesis leaves the GPU launching tiny kernels: measured, it
    is the single largest cost in the whole detector, larger than the analytic
    stratum's entire resynchronisation search.  Batching the scatter and the
    transform turns it into a handful of large operations.
    """
    dev = logits.device
    h, n = logits.shape
    rows = torch.as_tensor(code.rows, device=dev, dtype=torch.long)
    sign = torch.as_tensor(code.mask * 2 - 1, device=dev, dtype=logits.dtype)
    outs = []
    for i in range(0, h, chunk):
        blk = logits[i : i + chunk] * sign
        g = torch.zeros(blk.shape[0], code.size, device=dev, dtype=logits.dtype)
        g.index_add_(1, rows, blk)
        outs.append(fwht(g))
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Geometric test-time hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoHypothesis:
    name: str
    rotation: float = 0.0
    zoom: float = 1.0
    flip: bool = False


def build_hypotheses(
    rotations=(0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0),
    # Five-percentage-point zoom rungs keep the residual scale error below one
    # message cell over the detector's working radius. The ladder also includes
    # the crop fractions used by the attack suite.
    crop_zooms=(
        1.0,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
    ),
    mixed_rotations=(5.0, 10.0, 20.0),
    mixed_zooms=(0.85, 0.70, 0.55),
) -> Tuple[GeoHypothesis, ...]:
    """The resynchronisation bank, built from what attacks actually do.

    A positional code moves with the image, so the detector has to undo the
    geometry before it can read anything.  The naive bank is the full cross
    product of rotations and zooms, but that is mostly waste: a real adversary
    who rotates also crops away the wedges the rotation leaves, and the crop is
    then *determined* by the angle --- an image rotated by theta and cropped to
    its inscribed rectangle is magnified by exactly ``1/(|cos| + |sin|)``.
    Pairing each rotation with its implied zoom covers that whole family in one
    hypothesis per angle instead of one per angle and zoom.

    What remains is a crop ladder at zero rotation, a small mixed block for
    adversaries who rotate and crop independently, and reflections.  The result
    is a third the size of the cross product and covers more of what is actually
    seen.
    """
    seen = set()
    out: List[GeoHypothesis] = []

    def add(r: float, z: float, f: bool = False) -> None:
        key = (round(r, 2), round(z, 3), f)
        if key in seen:
            return
        seen.add(key)
        out.append(
            GeoHypothesis(
                f"r{r:+.0f}z{z:.2f}" + ("f" if f else ""), rotation=r, zoom=z, flip=f
            )
        )

    for r in rotations:
        t = math.radians(abs(r))
        z = 1.0 / (math.cos(t) + math.sin(t))  # inscribed-crop magnification
        for sgn in (1.0, -1.0) if r else (1.0,):
            add(sgn * r, z)
            add(sgn * r, 1.0)
    for z in crop_zooms:
        add(0.0, z)
    for r in mixed_rotations:
        for z in mixed_zooms:
            add(r, z)
            add(-r, z)
    for z in (1.0, 0.80, 0.60):
        add(0.0, z, True)
    return tuple(out)


DEFAULT_HYPOTHESES: Tuple[GeoHypothesis, ...] = build_hypotheses()

#: The local grid the detector searches around whichever coarse hypothesis wins.
REFINE_DEGREES: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
REFINE_ZOOMS: Tuple[float, ...] = (0.94, 0.97, 1.0, 1.03, 1.06)
#: How many hypotheses that refinement adds to the family being searched.  The
#: detector pays Bonferroni for this whether or not the refinement wins, because
#: the choice of *which* coarse hypothesis to refine around is made from the
#: data — so the honest multiplicity is the whole grid it could have landed on,
#: not the one it did.
N_REFINE: int = len(REFINE_DEGREES) * len(REFINE_ZOOMS)


def _apply_hypothesis(x: torch.Tensor, h: GeoHypothesis) -> torch.Tensor:
    y = torch.flip(x, dims=[-1]) if h.flip else x
    if h.rotation == 0.0 and h.zoom == 1.0:
        return y
    th = math.radians(h.rotation)
    c, s = math.cos(th) / h.zoom, math.sin(th) / h.zoom
    m = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=x.device, dtype=x.dtype)
    grid = F.affine_grid(
        m.unsqueeze(0).expand(y.shape[0], -1, -1), y.shape, align_corners=False
    )
    return F.grid_sample(
        y, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def hypothesis_validity(h: GeoHypothesis, grid: int, device) -> torch.Tensor:
    """Which message cells a geometric hypothesis actually recovers.

    Undoing a crop means zooming *out*, and everything outside the surviving
    frame is reflection padding — content the encoder never marked.  Reading
    message cells there does not merely add noise, it adds *wrong* bits, and on
    a 50% centre crop that is three quarters of the grid: enough to hold the
    statistic near chance no matter how well the mark survived in the part that
    is still there.

    The mask is a function of the hypothesis alone, never of the key, so
    restricting the statistic to valid cells leaves the exact null untouched —
    it just stops the detector reading padding.
    """
    th = math.radians(h.rotation)
    c, s = math.cos(th) / h.zoom, math.sin(th) / h.zoom
    lin = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid * 2 - 1
    vy, vx = torch.meshgrid(lin, lin, indexing="ij")
    sx = c * vx - s * vy
    sy = s * vx + c * vy
    return ((sx.abs() <= 1.0) & (sy.abs() <= 1.0)).reshape(-1)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class LatentEmbedResult:
    image: Array
    residual_canon: Array
    nonce: int
    message: np.ndarray


@dataclass
class LatentDetection:
    statistic: float
    pvalue: float
    n_hypotheses: int
    nonce: int
    hypothesis: str
    bit_accuracy: Optional[float]

    def as_evidence(self) -> Evidence:
        return Evidence(
            name="L",
            statistic=self.statistic,
            pvalue=self.pvalue,
            n_hypotheses=self.n_hypotheses,
            detail={
                "nonce": self.nonce,
                "hypothesis": self.hypothesis,
                "bit_accuracy": self.bit_accuracy,
            },
        )


# ---------------------------------------------------------------------------
# Stratum
# ---------------------------------------------------------------------------


class LatentStratum:
    """Embed and blind-detect the learned mark."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda:0",
        hypotheses: Sequence[GeoHypothesis] = DEFAULT_HYPOTHESES,
        squash: float = 2.0,
        search_top: int = 12,
        early_exit: float = 12.0,
        refine_deg: float = 2.0,
    ):
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        raw = dict(ck.get("config", {}))
        cfg_fields = LatentConfig.__dataclass_fields__.keys()
        self.cfg = LatentConfig(**{k: v for k, v in raw.items() if k in cfg_fields})
        self.device = device
        self.encoder = Encoder(self.cfg).to(device).eval()
        self.decoder = Decoder(self.cfg).to(device).eval()
        self.encoder.load_state_dict(ck["encoder"])
        self.decoder.load_state_dict(ck["decoder"])
        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
            p.requires_grad_(False)
        self.code = build_code(
            self.cfg.master_key, self.cfg.n_bits, self.cfg.nonce_bits
        )
        self.hypotheses = tuple(hypotheses)
        self.squash = float(squash)
        self.search_top = int(search_top)
        self.early_exit = float(early_exit)
        self.refine_deg = float(refine_deg)
        self.step = int(ck.get("step", 0))

    # -- helpers -----------------------------------------------------------

    def n_searched(self) -> int:
        """Size of the geometric family the detector is allowed to search.

        Every hypothesis in the bank, plus the local grid the refinement stage
        may open around whichever one of them wins.  Counting only the bank
        would understate the multiplicity and quietly inflate the evidence.
        """
        n = len(self.hypotheses)
        return n * (1 + N_REFINE) if self.refine_deg > 0 else n

    def _canon_tensor(self, image: Array) -> torch.Tensor:
        rgb = to_float01(image)
        n = self.cfg.canon
        small = np.stack([resize(rgb[..., c], (n, n)) for c in range(3)], axis=-1)
        t = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1)[None]
        return t.to(self.device).float().clamp(0, 1)

    def random_nonce(self, rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.code.size))

    # -- embedding ---------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        image: Array,
        nonce: Optional[int] = None,
        strength: Optional[float] = None,
    ) -> LatentEmbedResult:
        rgb = to_float01(image)
        nonce = self.random_nonce() if nonce is None else int(nonce) % self.code.size
        msg = self.code.codeword(nonce)
        x = self._canon_tensor(rgb)
        m = torch.from_numpy(msg.astype(np.float32))[None].to(self.device)
        residual = self.encoder(x, m)
        amp = self.cfg.strength if strength is None else float(strength)
        delta = (amp * residual * perceptual_mask(x))[0]
        # All three channels are carried.  Projecting onto luminance — which is
        # how the analytic stratum transports its residual — costs the learned
        # stratum everything: bit accuracy drops from 99.9% to chance, because
        # the encoder places much of the code in chroma.
        rgb_delta = delta.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        out = apply_canonical_residual_rgb(rgb, rgb_delta)
        return LatentEmbedResult(
            image=out, residual_canon=rgb_delta, nonce=nonce, message=msg
        )

    # -- detection ---------------------------------------------------------

    @torch.no_grad()
    def _hypothesis_logits(
        self, image: Array, want_raw: bool = False, chunk: int = 24, only=None
    ):
        """Masked, squashed logits for the geometric hypotheses, batched."""
        x = self._canon_tensor(image)
        idx = list(range(len(self.hypotheses))) if only is None else list(only)
        outs = []
        for i in range(0, len(idx), chunk):
            grp = [self.hypotheses[j] for j in idx[i : i + chunk]]
            batch = torch.cat([_apply_hypothesis(x, h) for h in grp], dim=0)
            outs.append(self.decoder(batch))
        logits = torch.cat(outs, dim=0)
        masks = torch.stack([self._masked_logits()[j] for j in idx])
        z = torch.tanh(logits / self.squash) * masks
        return (z, logits) if want_raw else z

    def _mask_counts(self) -> torch.Tensor:
        if getattr(self, "_mcounts", None) is None:
            self._mcounts = torch.stack([m.sum() for m in self._masked_logits()])
        return self._mcounts

    def _masked_logits(self) -> List[torch.Tensor]:
        """Per-hypothesis validity masks over the message cells, cached."""
        if getattr(self, "_masks", None) is None:
            self._masks = [
                hypothesis_validity(h, self.cfg.grid, self.device)[
                    : self.cfg.n_bits
                ].to(torch.float32)
                for h in self.hypotheses
            ]
        return self._masks

    @torch.no_grad()
    def analyse(
        self, image: Array, top_m: int = 8, expected_nonce: Optional[int] = None
    ) -> Tuple["LatentDetection", List[int]]:
        """Detection and nonce candidates from a single pass.

        The detection statistic and the analytic stratum's nonce anchors are the
        same computation; running it twice doubled the detector's cost for
        nothing.
        """
        # Most attacks do not move the image.  Evaluating the identity
        # hypothesis first and stopping when it is already decisive skips the
        # whole resynchronisation bank for every valuemetric attack, which is
        # most of them, and costs nothing in rigour: an early exit only shrinks
        # the search, while the p-value still pays Bonferroni for the full
        # nominal grid.
        if self.early_exit > 0:
            z0, lg0 = self._hypothesis_logits(image, want_raw=True, only=(0,))
            n0 = z0.pow(2).sum(dim=1).sqrt() + 1e-12
            c0 = correlate_all_nonces_batch(z0, self.code) / n0[:, None]
            v0, i0 = c0.max(dim=1)
            if float(v0[0]) >= self.early_exit:
                return self._package(
                    float(v0[0]), int(i0[0]), 0, lg0[0], c0, expected_nonce, top_m
                )

        z, logits = self._hypothesis_logits(image, want_raw=True)
        norms = z.pow(2).sum(dim=1).sqrt() + 1e-12

        # Two-stage search.  Running the nonce transform for all hundred-odd
        # geometric hypotheses is the detector's dominant cost, and almost all of
        # it is spent on hypotheses that are plainly wrong.  Decoder confidence
        # ranks them for free: a hypothesis that has resynchronised produces
        # saturated logits, a misaligned one produces mush.  The ranking is a
        # function of the image alone, so it is a search heuristic and the
        # p-value still pays Bonferroni for the entire nominal grid.
        keep = min(self.search_top, z.shape[0])
        conf = z.abs().sum(dim=1) / (self._mask_counts() + 1e-9)
        sel = torch.topk(conf, k=keep).indices
        corr_sel = correlate_all_nonces_batch(z[sel], self.code) / norms[sel][:, None]
        corr = torch.full(
            (z.shape[0], corr_sel.shape[1]), -1e9, device=z.device, dtype=corr_sel.dtype
        )
        corr[sel] = corr_sel
        vals, idx = corr.max(dim=1)
        best_h = int(vals.argmax())
        best_t = float(vals[best_h])
        best_nonce = int(idx[best_h])

        # Refine around the winner.  The bank is spaced three to five degrees
        # apart, and on a 32-cell grid a two-degree error already displaces the
        # outer cells by most of a cell, so the coarse maximum is usually a
        # near-miss rather than the answer.  A handful of extra decoder passes
        # around it costs little and recovers the alignment.
        if self.refine_deg > 0:
            base = self.hypotheses[best_h]
            # Zero has to be among the angle offsets.  Without it the refinement
            # can only propose a *different* angle, so a plain centre crop —
            # rotation genuinely zero, zoom slightly off the ladder — has no
            # candidate that keeps the angle and fixes the scale, which is
            # exactly the correction it needs.
            fine = [
                GeoHypothesis(
                    f"{base.name}+{d:+.1f}x{z:.2f}",
                    rotation=base.rotation + d,
                    zoom=base.zoom * z,
                    flip=base.flip,
                )
                for d in REFINE_DEGREES
                for z in REFINE_ZOOMS
                if not (d == 0.0 and z == 1.0)
            ]
            saved_h, saved_m = self.hypotheses, getattr(self, "_masks", None)
            try:
                self.hypotheses = tuple(fine)
                self._masks = None
                zf, lf = self._hypothesis_logits(image, want_raw=True)
                nf = zf.pow(2).sum(dim=1).sqrt() + 1e-12
                cf = correlate_all_nonces_batch(zf, self.code) / nf[:, None]
                vf, jf = cf.max(dim=1)
                k = int(vf.argmax())
                if float(vf[k]) > best_t:
                    best_t, best_nonce = float(vf[k]), int(jf[k])
                    self.hypotheses, self._masks = saved_h, saved_m
                    return self._package(
                        best_t, best_nonce, best_h, lf[k], cf, expected_nonce, top_m
                    )
            except Exception:
                pass
            finally:
                self.hypotheses, self._masks = saved_h, saved_m

        return self._package(
            best_t, best_nonce, best_h, logits[best_h], corr, expected_nonce, top_m
        )

    def _package(
        self, best_t, best_nonce, best_h, raw_logits, corr, expected_nonce, top_m
    ):
        n_hyp = self.code.size * self.n_searched()
        p = bonferroni(rademacher_pvalue(best_t), n_hyp)
        acc = None
        if expected_nonce is not None:
            want = torch.from_numpy(
                self.code.codeword(int(expected_nonce)).astype(np.float32)
            ).to(self.device)
            acc = float(((raw_logits > 0).float() == want).float().mean())
        det = LatentDetection(
            statistic=float(best_t),
            pvalue=float(p),
            n_hypotheses=n_hyp,
            nonce=int(best_nonce),
            hypothesis=self.hypotheses[int(best_h)].name,
            bit_accuracy=acc,
        )
        k = min(top_m, corr.shape[1])
        tv, ti = torch.topk(corr, k=k, dim=1)
        order = torch.argsort(tv.flatten(), descending=True)
        cands: List[int] = []
        for nv in ti.flatten()[order].tolist():
            if nv not in cands:
                cands.append(int(nv))
            if len(cands) >= top_m:
                break
        return det, cands

    @torch.no_grad()
    def nonce_candidates(self, image: Array, top_m: int = 8) -> List[int]:
        """The most likely nonces, for the analytic stratum to anchor on.

        This is the cross-stratum link: the analytic carriers are keyed by the
        nonce, so whenever the learned decoder still reads the message the
        analytic stratum inherits a synchronisation source that owes nothing to
        the pixels' own statistics — and therefore nothing an adversary can
        cluster on.
        """
        return self.analyse(image, top_m=top_m)[1]

    @torch.no_grad()
    def detect(
        self, image: Array, expected_nonce: Optional[int] = None
    ) -> LatentDetection:
        return self.analyse(image, top_m=1, expected_nonce=expected_nonce)[0]


__all__ = [
    "DEFAULT_HYPOTHESES",
    "GeoHypothesis",
    "hypothesis_validity",
    "LatentDetection",
    "LatentEmbedResult",
    "LatentStratum",
    "LinearCode",
    "build_code",
    "build_hypotheses",
    "correlate_all_nonces",
    "correlate_all_nonces_batch",
    "fwht",
]

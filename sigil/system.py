"""The complete SIGIL system and its fused decision rule.

SIGIL combines an analytic spectral stratum with one or more learned spatial
strata. The analytic component supplies exact invariances and a
machine-checkable null; the learned components supply robustness to
regeneration and other transformations that rewrite the spectrum.

Each stratum returns a p-value that already accounts for its own search. The
system combines those values with weighted Bonferroni, so the false-positive
budget is spent once across all active strata and no independence assumption is
needed.

The strata also cooperate rather than merely voting.  The learned stratum
carries a per-image nonce; the analytic stratum keys one of its carrier anchors
to that nonce.  When the learned decoder survives an attack, it hands the
analytic detector a synchronisation source with content-independent entropy —
closing the one loophole a purely content-derived anchor leaves open, namely an
adversary who clusters images by appearance to find others that share a
carrier set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import quality, to_float01
from .invariant import InvariantConfig, InvariantDetection, InvariantStratum
from .learned import LatentDetection, LatentStratum
from .stats import fuse_pvalues, log10p

Array = np.ndarray


@dataclass
class SigilConfig:
    invariant: InvariantConfig = field(default_factory=InvariantConfig)
    latent_checkpoint: Optional[str] = "checkpoints/latent.pt"
    #: An optional second learned stratum at a coarser message grid.
    #:
    #: The fine grid is what lets a 50% centre crop leave enough surviving
    #: message cells to clear the threshold, and it is also what puts the cells
    #: at eight pixels — small enough that a large rotation and its inverse
    #: resample the code away.  A coarser grid should trade those the other way.
    #: Carrying both and fusing costs one extra Bonferroni factor, which the
    #: threshold absorbs as the square root of a logarithm.
    latent_coarse_checkpoint: Optional[str] = None
    device: str = "tpu"
    alpha: float = 1e-6
    #: Budget shares in the order (analytic, learned, coarse). If fewer weights
    #: are supplied than active strata, detection uses equal shares.
    weights: Tuple[float, ...] = (0.5, 0.5)
    nonce_candidates: int = 8
    #: Per-stratum embedding amplitudes; two learned marks at full strength
    #: would double the residual energy for no gain.
    latent_strength: Optional[float] = None
    latent_coarse_strength: Optional[float] = None


@dataclass
class SigilEmbedResult:
    image: Array
    nonce: int
    payload_bits: Tuple[int, ...]
    anchor_keys: Dict[str, bytes]
    psnr: float
    ssim: float

    def as_dict(self) -> dict:
        return {
            "nonce": self.nonce,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "payload_bits": "".join(str(b) for b in self.payload_bits),
        }


@dataclass
class SigilDetection:
    detected: bool
    pvalue: float
    alpha: float
    analytic: Optional[InvariantDetection]
    latent: Optional[LatentDetection]
    coarse: Optional[LatentDetection] = None
    nonce_candidates: List[int] = field(default_factory=list)

    @property
    def log10_pvalue(self) -> float:
        return log10p(self.pvalue)

    def as_row(self) -> dict:
        analytic, learned = self.analytic, self.latent
        row = {
            "detected": int(self.detected),
            "pvalue": self.pvalue,
            "log10_pvalue": self.log10_pvalue,
            "alpha": self.alpha,
        }
        if analytic is not None:
            row.update(
                {
                    "A_statistic": analytic.statistic,
                    "A_pvalue": analytic.pvalue,
                    "A_log10p": log10p(analytic.pvalue),
                    "A_anchor": analytic.best_anchor,
                    "A_hypotheses": analytic.n_hypotheses,
                    "A_payload_accuracy": analytic.payload_accuracy,
                }
            )
            for an in analytic.per_anchor:
                row[f"A_{an.anchor}_T"] = an.statistic
                row[f"A_{an.anchor}_log10p"] = log10p(an.pvalue)
                row[f"A_{an.anchor}_scale"] = an.scale
                if an.key_correct is not None:
                    row[f"A_{an.anchor}_key_ok"] = int(an.key_correct)
        if learned is not None:
            row.update(
                {
                    "L_statistic": learned.statistic,
                    "L_pvalue": learned.pvalue,
                    "L_log10p": log10p(learned.pvalue),
                    "L_nonce": learned.nonce,
                    "L_hypothesis": learned.hypothesis,
                    "L_bit_accuracy": learned.bit_accuracy,
                    "L_hypotheses": learned.n_hypotheses,
                }
            )
        if self.coarse is not None:
            row.update(
                {
                    "C_statistic": self.coarse.statistic,
                    "C_log10p": log10p(self.coarse.pvalue),
                    "C_bit_accuracy": self.coarse.bit_accuracy,
                }
            )
        return row


class Sigil:
    """Embed and detect the full stratified mark."""

    def __init__(self, config: Optional[SigilConfig] = None):
        self.cfg = config or SigilConfig()
        self.analytic = InvariantStratum(self.cfg.invariant, device=self.cfg.device)
        self.latent: Optional[LatentStratum] = None
        ck = self.cfg.latent_checkpoint
        if ck and Path(ck).exists():
            self.latent = LatentStratum(ck, device=self.cfg.device)
            if self.cfg.latent_strength:
                self.latent.cfg = type(self.latent.cfg)(
                    **{
                        **self.latent.cfg.__dict__,
                        "strength": float(self.cfg.latent_strength),
                    }
                )
        self.coarse: Optional[LatentStratum] = None
        ckc = self.cfg.latent_coarse_checkpoint
        if ckc and Path(ckc).exists():
            self.coarse = LatentStratum(ckc, device=self.cfg.device)
            if self.cfg.latent_coarse_strength:
                self.coarse.cfg = type(self.coarse.cfg)(
                    **{
                        **self.coarse.cfg.__dict__,
                        "strength": float(self.cfg.latent_coarse_strength),
                    }
                )

    @property
    def strata(self) -> List[str]:
        return (
            ["A"]
            + (["L"] if self.latent is not None else [])
            + (["C"] if self.coarse is not None else [])
        )

    # -- embedding ---------------------------------------------------------

    def embed(
        self,
        image: Array,
        payload: Optional[Sequence[int]] = None,
        nonce: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> SigilEmbedResult:
        src = to_float01(image)
        if nonce is None:
            rng = rng or np.random.default_rng()
            span = (
                (1 << self.latent.cfg.nonce_bits)
                if self.latent is not None
                else (1 << 24)
            )
            nonce = int(rng.integers(0, span))

        cur = src
        if self.latent is not None:
            cur = self.latent.embed(cur, nonce=nonce).image
        if self.coarse is not None:
            # Both learned strata carry the same nonce, so the analytic
            # stratum's nonce anchor is recoverable from whichever survives.
            cur = self.coarse.embed(
                cur, nonce=nonce % (1 << self.coarse.cfg.nonce_bits)
            ).image
        emb = self.analytic.embed(cur, nonce=nonce, payload=payload)
        q = quality(src, emb.image)
        return SigilEmbedResult(
            image=emb.image,
            nonce=nonce,
            payload_bits=emb.payload_bits,
            anchor_keys=emb.anchor_keys,
            psnr=q.psnr,
            ssim=q.ssim,
        )

    # -- detection ---------------------------------------------------------

    def detect(
        self,
        image: Array,
        *,
        alpha: Optional[float] = None,
        expected_payload: Optional[Sequence[int]] = None,
        expected_nonce: Optional[int] = None,
        reference_keys: Optional[Dict[str, bytes]] = None,
    ) -> SigilDetection:
        alpha = self.cfg.alpha if alpha is None else float(alpha)
        lat: Optional[LatentDetection] = None
        cands: List[int] = []
        if self.latent is not None:
            lat, cands = self.latent.analyse(
                image, top_m=self.cfg.nonce_candidates, expected_nonce=expected_nonce
            )

        crs: Optional[LatentDetection] = None
        if self.coarse is not None:
            crs, ccands = self.coarse.analyse(
                image,
                top_m=self.cfg.nonce_candidates,
                expected_nonce=(
                    expected_nonce % (1 << self.coarse.cfg.nonce_bits)
                    if expected_nonce is not None
                    else None
                ),
            )
            # Only pass these on when the two learned strata share a nonce
            # width.  A narrower coarse stratum recovers the nonce modulo its
            # own space, and handing that truncated value to the analytic
            # stratum would key its carriers to a number the embedder never
            # used --- the anchors would simply miss, quietly.
            if (
                self.latent is None
                or self.coarse.cfg.nonce_bits == self.latent.cfg.nonce_bits
            ):
                for c in ccands:
                    if c not in cands:
                        cands.append(c)

        if expected_nonce is not None and int(expected_nonce) not in cands:
            cands.append(int(expected_nonce))

        ana = self.analytic.detect(
            image,
            nonces=cands,
            expected_payload=expected_payload,
            reference_keys=reference_keys,
        )

        pvals = [ana.pvalue]
        if lat is not None:
            pvals.append(lat.pvalue)
        if crs is not None:
            pvals.append(crs.pvalue)
        weights = list(self.cfg.weights[: len(pvals)])
        if len(weights) < len(pvals):
            weights = [1.0 / len(pvals)] * len(pvals)
        p = fuse_pvalues(pvals, weights)
        return SigilDetection(
            detected=bool(p <= alpha),
            pvalue=float(p),
            alpha=alpha,
            analytic=ana,
            latent=lat,
            coarse=crs,
            nonce_candidates=cands,
        )


__all__ = ["Sigil", "SigilConfig", "SigilDetection", "SigilEmbedResult"]

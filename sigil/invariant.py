"""SIGIL-A: the analytic invariant stratum.

The mark lives in the *log-magnitude spectrum* of the canonical luminance grid,
normalised against its own radial profile.  Several exact invariances follow
from that choice alone, before any training or tuning.

* **Translation.**  ``|F|`` is unchanged by a shift of the image, so the
  detector statistic is literally the same number.  There is no grid search.
* **Zero-phase radially-symmetric filtering** — blur, sharpening, the smooth
  envelope of a JPEG quantisation table, any circular convolution — multiplies
  every bin of a ring by one common factor, which the radial baseline removes.
* **Global gain.**  A contrast change scales all magnitudes identically and is
  absorbed by the same baseline.  DC is never used, so brightness is free.
* **Uniform rescaling.**  The canonical grid fixes the sampling, so an image
  delivered at any resolution lands on the same canonical representation.

Cropping and reflection act on the canonical spectrum as a radial dilation and
an axis flip.  The detector does not re-render the image for these: it resamples
one already-computed normalised spectrum at transformed carrier coordinates, so
a hypothesis costs a few thousand bilinear taps.  The grid has to be fine — a
carrier at radius ``r`` moves by ``r * ds`` bins under a scale error, so
half-bin accuracy at the outer radius needs a few hundred scales — which is why
the search runs in two stages.

Carrier positions and chip signs come from a keyed PRF applied to the *anchors*
of :mod:`sigil.anchors`, so no two images share a carrier set and an adversary
who averages residuals over an arbitrarily large corpus recovers noise.  Each
anchor owns a disjoint slice of the carrier budget and is detected
independently, so the stratum survives whenever *any* anchor does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anchors import DEFAULT_ANCHORS, AnchorReader, AnchorSpec
from .backend import Scanner, transform_bank
from .common import (
    CANON,
    apply_canonical_residual,
    canonical_luma,
    key_stream_rng,
    ring_index,
    to_float01,
)
from .stats import Evidence, bonferroni, rademacher_pvalue

Array = np.ndarray
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantConfig:
    canon: int = CANON

    # Carrier annulus in cycles per canonical pixel (Nyquist = 0.5).
    freq_min: float = 0.055
    freq_max: float = 0.240
    n_carriers: int = 4096

    # Strength in nats of log-magnitude.  The modulation is *proportional*:
    # every carrier bin is multiplied by exp(+-alpha), never forced to an
    # absolute level.  Proportionality matters twice over — it follows Weber's
    # law, so the perturbation hides in whatever energy the image already has
    # there, and it is exactly sign-symmetric, so it leaves band-averaged
    # spectral statistics unbiased.  An absolute rule would systematically lift
    # low-magnitude bins, and the mark would then perturb the very anchor that
    # selects it.
    alpha: float = 0.46
    csf_exponent: float = 0.35  # >0 pushes energy toward higher radii

    # Payload carried on a disjoint carrier subset of each anchor (0 disables).
    n_payload_bits: int = 32
    payload_reps: int = 12

    anchors: Tuple[AnchorSpec, ...] = DEFAULT_ANCHORS

    # Detector resynchronisation search.
    scale_min: float = 0.45
    scale_max: float = 1.35
    n_scales: int = 260
    rotations_deg: Tuple[float, ...] = tuple(np.round(np.arange(-3.0, 3.01, 0.5), 3))
    search_reflection: bool = True

    # Two-stage economy.  Stage one ranks (key, scale, reflection) with a
    # carrier subset; stage two evaluates the reported statistic with every
    # carrier on a local grid around each surviving proposal.  Purely a compute
    # split: the p-value always pays for the full nominal grid.
    proposal_carriers: int = 512
    n_proposals: int = 8
    refine_scale_span: float = 0.012
    refine_scale_steps: int = 7

    # Robustifying transform applied to per-bin evidence before the statistic.
    soft_clip_sigma: float = 2.5

    # Stable domain-separation namespace for analytic carrier derivation.
    master_key: bytes = b"sigil-v1-analytic"

    def carriers_for(self, spec: AnchorSpec) -> int:
        return max(64, int(round(self.n_carriers * spec.share)))


@dataclass
class InvariantEmbedResult:
    image: Array
    residual_canon: Array
    anchor_keys: Dict[str, bytes]
    payload_bits: Tuple[int, ...]
    n_carriers: int


@dataclass
class AnchorDetection:
    anchor: str
    statistic: float
    pvalue: float
    n_hypotheses: int
    scale: float
    rotation_deg: float
    reflected: bool
    key_rank: int
    key_correct: Optional[bool]
    payload_bits: Tuple[int, ...] = ()
    payload_accuracy: Optional[float] = None


@dataclass
class InvariantDetection:
    statistic: float
    pvalue: float
    n_hypotheses: int
    best_anchor: str
    per_anchor: List[AnchorDetection] = field(default_factory=list)
    payload_bits: Tuple[int, ...] = ()
    payload_accuracy: Optional[float] = None

    def as_evidence(self) -> Evidence:
        best = next((a for a in self.per_anchor if a.anchor == self.best_anchor), None)
        return Evidence(
            name="A",
            statistic=self.statistic,
            pvalue=self.pvalue,
            n_hypotheses=self.n_hypotheses,
            detail={
                "anchor": self.best_anchor,
                "scale": best.scale if best else 1.0,
                "rotation_deg": best.rotation_deg if best else 0.0,
                "reflected": bool(best.reflected) if best else False,
                "key_correct": best.key_correct if best else None,
                "payload_accuracy": self.payload_accuracy,
            },
        )


# ---------------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierSet:
    fy: np.ndarray  # normalised frequency, cycles per canonical pixel
    fx: np.ndarray
    signs: np.ndarray  # keyed Rademacher chips
    weight: np.ndarray  # per-bin embedding strength, nats
    payload_slot: np.ndarray


def build_carriers(anchor_key: bytes, cfg: InvariantConfig, n: int) -> CarrierSet:
    """Draw one anchor's carriers in continuous normalised frequency.

    Positions come from one PRF stream and signs from another, under different
    domain-separation labels.  Conditioning on the positions — that is, on
    everything the observed image can reveal — leaves the signs uniform and
    independent, which is precisely the condition Hoeffding's inequality needs
    for the exact null in :mod:`sigil.stats`.

    Frequencies are continuous rather than tied to a bin lattice so that the
    same carrier set is reproducible on any canonical grid shape; that is what
    lets the detector survive a change of aspect ratio or a crop.
    """
    rng_pos = key_stream_rng(anchor_key, "positions")
    r = np.sqrt(rng_pos.uniform(cfg.freq_min**2, cfg.freq_max**2, size=n))
    t = rng_pos.uniform(0.0, math.pi, size=n)
    fy, fx = r * np.sin(t), r * np.cos(t)

    signs = (
        key_stream_rng(anchor_key, "signs").integers(0, 2, size=n).astype(np.float64)
        * 2
        - 1
    )
    weight = cfg.alpha * np.power(
        np.clip(r / cfg.freq_max, 1e-3, None), cfg.csf_exponent
    )

    slot = np.full(n, -1, dtype=np.int64)
    if cfg.n_payload_bits > 0:
        n_pay = min(cfg.n_payload_bits * cfg.payload_reps, n // 3)
        chosen = key_stream_rng(anchor_key, "payload-slots").choice(
            n, size=n_pay, replace=False
        )
        slot[chosen] = np.arange(n_pay) % cfg.n_payload_bits
    return CarrierSet(fy=fy, fx=fx, signs=signs, weight=weight, payload_slot=slot)


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def radial_excess(canon: Array, n_rings: int = 512) -> Array:
    """Log-magnitude standardised against its own ring, fftshifted.

    Subtracting the ring mean is what delivers the filtering and gain
    invariances.  Dividing by the ring standard deviation costs nothing extra
    and buys real detection power: the natural spread of log-magnitude is much
    larger at low radii than at high ones, so an unstandardised statistic is
    dominated by its noisiest bins.  Both moments are functions of the observed
    image alone, never of the keyed signs, so the exact null is untouched.
    """
    F = np.fft.fft2(canon.astype(np.float64))
    log_mag = np.log(np.abs(F) + _EPS)
    rings = ring_index(canon.shape, n_rings)
    flat = rings.ravel()
    counts = np.bincount(flat, minlength=n_rings).astype(np.float64)
    sums = np.bincount(flat, weights=log_mag.ravel(), minlength=n_rings)
    sq = np.bincount(flat, weights=(log_mag**2).ravel(), minlength=n_rings)
    denom = np.maximum(counts, 1.0)
    mean = sums / denom
    var = np.maximum(sq / denom - mean**2, 0.0)
    std = np.sqrt(var) + 1e-3
    z = (log_mag - mean[flat].reshape(log_mag.shape)) / std[flat].reshape(log_mag.shape)
    return np.fft.fftshift(z)


def soft_clip(z: Array, sigma_mult: float) -> Array:
    """Bound the influence of a few extreme bins without discarding them.

    The statistic is normalised by ``||z||``, so a handful of huge outliers —
    spectral leakage from a specular highlight, a JPEG blocking harmonic —
    would inflate the denominator and cost real detection power.  The clip is a
    function of the observed image only, never of the keyed signs, so the exact
    null is untouched.
    """
    if sigma_mult <= 0:
        return z
    scale = 1.4826 * float(np.median(np.abs(z - np.median(z)))) + _EPS
    lim = sigma_mult * scale
    return lim * np.tanh(z / lim)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InvariantStratum:
    """Embed and detect the analytic invariant mark."""

    def __init__(
        self, config: Optional[InvariantConfig] = None, device: Optional[str] = None
    ):
        self.cfg = config or InvariantConfig()
        self.device = device

    # -- embedding ---------------------------------------------------------

    def embed(
        self,
        image: Array,
        nonce: Optional[int] = None,
        payload: Optional[Sequence[int]] = None,
    ) -> InvariantEmbedResult:
        cfg = self.cfg
        rgb = to_float01(image)
        canon = canonical_luma(rgb, cfg.canon)
        h, w = canon.shape
        reader = AnchorReader(rgb, cfg.canon)

        bits = self._payload_bits(payload)
        log_gain = np.zeros((h, w), dtype=np.float64)
        keys: Dict[str, bytes] = {}
        total = 0
        for spec in cfg.anchors:
            key = reader.embed_key(spec, cfg.master_key, nonce=nonce)
            if key is None:
                continue
            keys[spec.name] = key
            car = build_carriers(key, cfg, cfg.carriers_for(spec))
            chips = car.signs.copy()
            has = car.payload_slot >= 0
            if cfg.n_payload_bits > 0 and has.any():
                chips[has] *= np.where(bits[car.payload_slot[has]] > 0, 1.0, -1.0)
            iy = np.rint(car.fy * h).astype(np.int64) % h
            ix = np.rint(car.fx * w).astype(np.int64) % w
            # Gains compose multiplicatively, so two anchors that land on the
            # same bin simply add their log-domain contributions.  Nothing has
            # to be de-duplicated and no anchor is silently overwritten.
            np.add.at(log_gain, (iy, ix), chips * car.weight)
            total += int(car.fy.size)

        # Carriers are drawn from a half-plane, so a bin and its conjugate are
        # distinct; copying the log-gain onto the conjugate keeps the modified
        # spectrum Hermitian — and therefore the inverse transform exactly real —
        # without halving the modulation depth the way an averaging rule would.
        log_gain = log_gain + log_gain[(-np.arange(h)) % h][:, (-np.arange(w)) % w]
        F = np.fft.fft2(canon.astype(np.float64)) * np.exp(log_gain)
        wm_canon = np.real(np.fft.ifft2(F)).astype(np.float32)
        residual = (wm_canon - canon).astype(np.float32)
        out = apply_canonical_residual(rgb, residual)
        return InvariantEmbedResult(
            image=out,
            residual_canon=residual,
            anchor_keys=keys,
            payload_bits=tuple(int(b) for b in bits),
            n_carriers=total,
        )

    def _payload_bits(self, payload: Optional[Sequence[int]]) -> np.ndarray:
        cfg = self.cfg
        if cfg.n_payload_bits <= 0:
            return np.zeros(0, dtype=np.int64)
        if payload is None:
            return (
                key_stream_rng(cfg.master_key, "default-payload")
                .integers(0, 2, size=cfg.n_payload_bits)
                .astype(np.int64)
            )
        bits = np.asarray([int(b) & 1 for b in payload], dtype=np.int64)
        if bits.size != cfg.n_payload_bits:
            raise ValueError(f"payload must hold {cfg.n_payload_bits} bits")
        return bits

    # -- detection ---------------------------------------------------------

    def detect(
        self,
        image: Array,
        *,
        nonces: Optional[Sequence[int]] = None,
        expected_payload: Optional[Sequence[int]] = None,
        reference_keys: Optional[Dict[str, bytes]] = None,
    ) -> InvariantDetection:
        cfg = self.cfg
        rgb = to_float01(image)
        canon = canonical_luma(rgb, cfg.canon)
        excess = soft_clip(radial_excess(canon), cfg.soft_clip_sigma)
        reader = AnchorReader(rgb, cfg.canon)

        rots = np.asarray(cfg.rotations_deg, dtype=np.float64)
        scls = np.geomspace(cfg.scale_min, cfg.scale_max, cfg.n_scales)
        refls = np.asarray((1.0, -1.0) if cfg.search_reflection else (1.0,))
        cos_p, sin_p, mir_p = transform_bank([0.0], scls, refls)
        proposer = Scanner(excess, cos_p, sin_p, mir_p, device=self.device)
        span = np.linspace(
            1.0 - cfg.refine_scale_span,
            1.0 + cfg.refine_scale_span,
            cfg.refine_scale_steps,
        )

        results: List[AnchorDetection] = []
        for spec in cfg.anchors:
            keys = reader.keys(spec, cfg.master_key, nonces=nonces)
            if not keys:
                continue
            n_car = cfg.carriers_for(spec)
            n_hyp = len(keys) * rots.size * scls.size * refls.size
            sub = slice(None, min(cfg.proposal_carriers, n_car))

            cache: Dict[int, CarrierSet] = {
                rank: build_carriers(key, cfg, n_car) for rank, key in enumerate(keys)
            }
            stack = np.stack([cache[r].fy[sub] for r in range(len(keys))])
            stack_x = np.stack([cache[r].fx[sub] for r in range(len(keys))])
            stack_s = np.stack([cache[r].signs[sub] for r in range(len(keys))])
            tm = proposer.statistics_multi(stack, stack_x, stack_s)  # (L, M)
            ranked: List[Tuple[float, int, int]] = []
            for rank in range(len(keys)):
                row = tm[rank]
                for j in np.argsort(row)[-cfg.n_proposals :]:
                    ranked.append((float(row[j]), rank, int(j)))
            ranked.sort(key=lambda r: -r[0])

            # Refine every surviving proposal in one pass: build the union of
            # their local (rotation x scale x reflection) neighbourhoods as a
            # single bank, then evaluate each distinct key against it once.
            picks: List[Tuple[int, int, int]] = []
            seen: set = set()
            for _, rank, j in ranked[: cfg.n_proposals]:
                si, fi = j // refls.size, j % refls.size
                if (rank, si, fi) in seen:
                    continue
                seen.add((rank, si, fi))
                picks.append((rank, si, fi))
            cos_a, sin_a, mir_a, meta = [], [], [], []
            for rank, si, fi in picks:
                local_s = scls[si] * span
                ct, st, mr = transform_bank(rots, local_s, [refls[fi]])
                cos_a.append(ct)
                sin_a.append(st)
                mir_a.append(mr)
                for k in range(ct.size):
                    meta.append(
                        (
                            float(rots[k // local_s.size]),
                            float(local_s[k % local_s.size]),
                            float(refls[fi]),
                        )
                    )
            sc = Scanner(
                excess,
                np.concatenate(cos_a),
                np.concatenate(sin_a),
                np.concatenate(mir_a),
                device=self.device,
            )
            ranks = sorted({r for r, _, _ in picks})
            tm = sc.statistics_multi(
                np.stack([cache[r].fy for r in ranks]),
                np.stack([cache[r].fx for r in ranks]),
                np.stack([cache[r].signs for r in ranks]),
            )
            ri, ki = np.unravel_index(int(np.argmax(tm)), tm.shape)
            rot_b, scale_b, refl_b = meta[int(ki)]
            best = {
                "t": float(tm[ri, ki]),
                "rot": rot_b,
                "scale": scale_b,
                "refl": refl_b,
                "rank": ranks[int(ri)],
                "car": cache[ranks[int(ri)]],
                "z": sc.evidence(
                    cache[ranks[int(ri)]].fy, cache[ranks[int(ri)]].fx, int(ki)
                ),
            }
            if best["car"] is None:
                continue
            bits, acc = self._read_payload(best["car"], best["z"], expected_payload)
            correct = None
            if reference_keys is not None and spec.name in reference_keys:
                correct = bool(keys[best["rank"]] == reference_keys[spec.name])
            results.append(
                AnchorDetection(
                    anchor=spec.name,
                    statistic=float(best["t"]),
                    pvalue=float(
                        bonferroni(rademacher_pvalue(float(best["t"])), n_hyp)
                    ),
                    n_hypotheses=n_hyp,
                    scale=float(best["scale"]),
                    rotation_deg=float(best["rot"]),
                    reflected=bool(float(best["refl"]) < 0),
                    key_rank=int(best["rank"]),
                    key_correct=correct,
                    payload_bits=bits,
                    payload_accuracy=acc,
                )
            )

        if not results:
            return InvariantDetection(
                statistic=0.0, pvalue=1.0, n_hypotheses=1, best_anchor="none"
            )
        # Bonferroni across anchors: the stratum reports the best of several
        # tests, so it pays for having looked at all of them.
        best_a = min(results, key=lambda a: a.pvalue)
        return InvariantDetection(
            statistic=best_a.statistic,
            pvalue=float(min(1.0, best_a.pvalue * len(results))),
            n_hypotheses=sum(a.n_hypotheses for a in results),
            best_anchor=best_a.anchor,
            per_anchor=results,
            payload_bits=best_a.payload_bits,
            payload_accuracy=best_a.payload_accuracy,
        )

    def _read_payload(
        self,
        car: Optional[CarrierSet],
        z: Optional[Array],
        expected: Optional[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], Optional[float]]:
        cfg = self.cfg
        if cfg.n_payload_bits <= 0 or car is None or z is None:
            return tuple(), None
        acc = np.zeros(cfg.n_payload_bits, dtype=np.float64)
        has = car.payload_slot >= 0
        np.add.at(acc, car.payload_slot[has], (car.signs * z)[has])
        bits = tuple(int(v > 0) for v in acc)
        if expected is None:
            return bits, None
        exp = np.asarray([int(b) & 1 for b in expected], dtype=np.int64)
        if exp.size != cfg.n_payload_bits:
            return bits, None
        return bits, float(np.mean(np.asarray(bits) == exp))


__all__ = [
    "AnchorDetection",
    "CarrierSet",
    "InvariantConfig",
    "InvariantDetection",
    "InvariantEmbedResult",
    "InvariantStratum",
    "build_carriers",
    "radial_excess",
    "soft_clip",
]

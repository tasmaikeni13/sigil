# Phase 01: Formal Mathematical Foundations, Statistical Nulls & Multiplicity Theory

## 1. Objectives & Theoretical Scope

The objective of Phase 01 is to establish the rigorous mathematical, statistical, and formal foundations of SIGIL, proving that its detection statistic and false-positive rate (FPR) guarantees match or strictly exceed all peer watermarking systems. 

SIGIL uses a conservative, non-asymptotic Hoeffding bound for keyed Rademacher sums. This phase establishes the conditional null calculation, validates numerical checks, verifies theorems in Lean 4, and defines the self-correction loop. Competitive superiority remains an empirical target, not a consequence of the null theorem.

---

## 2. Comparative Formal Analysis: SIGIL vs. Peers

| Watermarking System | Null Hypothesis Model | Multiplicity Control | Detector Statistic | False-Alarm Guarantee | Collusion Vulnerability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SIGIL (Ours)** | **Hoeffding upper bound** for $\sum_{i=1}^K s_i z_i$ ($s_i \in \{-1, +1\}$) | **Bonferroni** over all searched hypotheses | Normalized keyed projection $T = \frac{\langle \mathbf{s}, \mathbf{z} \rangle}{\|\mathbf{z}\|_2}$ | Conditional private-key bound at $\alpha = 10^{-6}$; empirical rate requires separate measurement | Per-image content carriers; collusion performance requires measurement |
| **SynthID (Google DeepMind)** | Asymptotic Gaussian approximation $\mathcal{N}(\mu, \sigma^2)$ | Heuristic post-hoc thresholding | Differential phase correlation at fixed spectral bins | Empirical only; susceptible to tail breakdown under heavy-tailed image spectra | **Critical ($\mathcal{O}(1)$ static codebook collapse)** across multiple images |
| **Stable Signature (Meta)** | Empirical cosine similarity threshold | None (single key per generator) | Cosine similarity in latent extractor space | Unbounded tail risk under out-of-distribution latents | **Vulnerable to key leakage** via multi-image averaging |
| **TrustMark / RivaGAN** | Softmax classification margin | Uncorrected peak detection | Cross-entropy logit margin | High false-positive rate under adversarial perturbations | Vulnerable to model extraction and feature subtraction |
| **Tree-Ring / AquaLoRA** | Gaussian circular symmetric test | Fixed radial ring thresholds | Fourier ring energy shift | Degrades severely under rotation and anisotropic resizing | Vulnerable to ring phase cancellation attacks |

### Mathematical Superiority Criterion
SIGIL must demonstrate:
1. **Zero Empirical False Alarms**: At nominal $\alpha = 10^{-6}$, empirical false-positive rate across $\ge 10^7$ Monte Carlo trials must be $\le 10^{-6}$ (no anti-conservative tail inflation).
2. **Strictly Superior Detection Power**: At equal embedding distortion (PSNR $\ge 42\text{ dB}$, SSIM $\ge 0.985$), SIGIL's True Positive Rate (TPR) at $\alpha = 10^{-6}$ under standard compression (JPEG QF=50) must strictly exceed SynthID and Stable Signature by $\ge 5\%$ absolute margin.

---

## 3. Exact Statistical Derivations

### 3.1 The Exact Rademacher Null Distribution
Let $\mathbf{s} = (s_1, \dots, s_K) \in \{-1, +1\}^K$ be the pseudo-random carrier signs, independent of the image content feature vector $\mathbf{z} \in \mathbb{R}^K$. Under the null hypothesis $H_0$ that the image does not contain the watermark keyed by $\mathbf{s}$:
$$T = \frac{\sum_{i=1}^K s_i z_i}{\sqrt{\sum_{i=1}^K z_i^2}}$$

For nonzero $\mathbf{z}$, conditioned on $\mathbf{z}$, $T$ is the sum of independent, zero-mean bounded random variables $X_i = s_i w_i$, where $w_i = z_i / \|\mathbf{z}\|_2$ and $\sum_{i=1}^K w_i^2 = 1$. The combinatorial expression for its exact tail is:
$$p(t) = \mathbb{P}_{H_0}(T \ge t) = 2^{-K} \sum_{\mathbf{s} \in \{-1, +1\}^K} \mathbf{1}\left(\sum_{i=1}^K s_i w_i \ge t\right)$$

The implementation reports the conservative Hoeffding bound, not that exact
combinatorial tail. When $\mathbf{z}=0$, it assigns $p=1$; the unit-norm theorem
cannot be applied to zero evidence. The guarantee is over an independently
chosen private key under the PRF assumption, not a deterministic promise for a
fixed leaked key.

### 3.2 Non-Asymptotic Hoeffding & Chernoff Dominance
By Hoeffding's inequality, for any unit vector $\mathbf{w}$ with $\|\mathbf{w}\|_2 = 1$:
$$\mathbb{P}_{H_0}(T \ge t) \le \exp\left(-\frac{2 t^2}{\sum_{i=1}^K (2 w_i)^2}\right) = \exp\left(-\frac{t^2}{2}\right)$$
The bound is conservative. A finite Rademacher tail need not lie below the
Gaussian tail at every threshold; no such ordering is assumed.

### 3.3 Multiplicity Control under Continuous Transformation Grids
When the detector evaluates $M$ candidate geometric hypotheses (rotations $\theta \in \Theta$, scales $s \in \mathcal{S}$, translations, nonces $n \in \mathcal{N}$):
$$T_{\max} = \max_{m \in \{1, \dots, M\}} T_m$$
By Boole's inequality (Bonferroni bound), without assuming independence among geometric transforms:
$$\mathbb{P}_{H_0}(T_{\max} \ge t) \le \min\left(1, \; M \cdot \mathbb{P}_{H_0}(T \ge t)\right)$$
To ensure an overall family-wise error rate (FWER) $\le \alpha$, choose a threshold $\tau_\alpha$ satisfying the conservative bound:
$$M\exp(-\tau_\alpha^2/2) \le \alpha.$$

---

## 4. High-Precision Monte Carlo Verification Protocol

To audit the implementation and estimate tail behavior without claiming a finite-sample proof:
1. **Trial Volume**: Run $N = 10^7$ independent trials under $H_0$ using synthetic noise and unwatermarked natural image spectra.
2. **Extreme Quantile Validation**:
   - Compute empirical quantiles at $q = 10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}, 10^{-6}$.
   - Compare empirical quantiles against the analytical Rademacher threshold $\tau(q)$ and the asymptotic normal threshold $\Phi^{-1}(1 - q)$.
3. **Multiplicity Search Verification**:
   - For grid sizes $M \in \{1, 64, 512, 4096, 6760\}$, evaluate $T_{\max}$ over unwatermarked images.
   - Confirm that the empirical survival function $\mathbb{P}(T_{\max} \ge \tau_{\alpha/M}) \le \alpha$, verifying that the Bonferroni correction is universally valid and non-leaking.

---

## 5. Formal Proof Verification Gate (Lean 4)

All mathematical claims must be formally checked against the Lean 4 proof suite located in `lean/Sigil/`:
- `Invariance.lean`: Formal proof of translation invariance (Theorem T1) and radial filter invariance (Theorem T2).
- `Rademacher.lean`: Formal proof of the Rademacher tail bound (Theorem T4).
- `Multiplicity.lean`: Formal proof of the Bonferroni union bound under arbitrary hypothesis dependencies (Theorems T5, T8).
- `Detector.lean`: Formal verification of detector monotonicity and p-value validity.

**Verification Command**:
```bash
cd lean && lake build && cd ..
```
The build must succeed with zero errors, zero warnings, and zero `sorry` axioms.

---

## 6. Autonomous Self-Correction & Adaptation Playbook

When an automated check fails during Phase 01, the agent must execute the following playbook:

### Scenario A: Empirical FPR Exceeds Nominal $\alpha$
1. **Diagnosis**: Spectrum whitening is insufficient, or spectral energy exhibits low-frequency auto-correlation violating sign exchangeability.
2. **Literature Search**: Research robust spatial whitening filters (e.g., adaptive Wiener deconvolution, radial excess normalization).
3. **Mathematical Modification**: Update `sigil/invariant.py:radial_excess` with a steeper high-pass transition band or modify soft-clipping parameter $\sigma_{\text{clip}}$ to suppress heavy spectral outliers.
4. **Lean & Monte Carlo Update**: Re-verify Theorem T3 and re-run Monte Carlo verification ($10^7$ trials) until empirical FPR $\le \alpha$.

### Scenario B: Detection Power Lower Than Peer Baselines
1. **Diagnosis**: Carrier frequency allocation overlaps with high natural image variance, degrading signal-to-noise ratio (SNR).
2. **Mathematical Modification**: Recalibrate carrier ring allocation in `sigil/invariant.py:build_carriers` to shift energy toward mid-frequency bands (spatial frequencies where natural images have low power, but JPEG compression does not quantize heavily).
3. **Iterate**: Re-run detection sensitivity benchmarks against SynthID and Stable Signature until SIGIL's TPR @ $\alpha=10^{-6}$ exceeds both baselines.

### Scenario C: Multiplicity Bound Too Conservative (Excessive Slack)
1. **Diagnosis**: Nearby geometric grid angles exhibit near-perfect correlation ($\rho \approx 1.0$), making raw Bonferroni overly conservative.
2. **Mathematical Modification**: Retain the full Bonferroni charge for every evaluated hypothesis unless a replacement bound is separately proved for the actual dependent search and reflected in Lean. Correlation alone does not justify $M_{\text{eff}} < M$.

---

## 7. Downstream Cascading Rules

If Phase 01 results in changes to:
- **Carrier Sign Generation or Weighting**: Trigger immediate updates to Phase 02 (collusion math), Phase 03 (TPU scan kernels in `sigil/kernels.py`), and Phase 05 (Arena evaluation).
- **Threshold Equation or Multiplicity Formula**: Update `sigil/stats.py`, `scripts/theory_checks.py`, and the mathematical exposition in `paper/sigil.tex` Section 3 ("Statistical Guarantees").

---

## 8. Verification Command & Acceptance Criteria

```bash
# 1. Run Lean 4 formal proof build
(cd lean && lake build)

# 2. Run numerical theorem verification
.venv/bin/python3 scripts/theory_checks.py

# 3. Run high-resolution unit tests
.venv/bin/pytest -v tests/test_backend.py tests/test_tpu_kernels.py
```

### Acceptance Criteria
- [ ] All 9 theoretical theorems (T1–T9) pass with `[ok]` in `scripts/theory_checks.py`.
- [ ] Lean 4 compilation completes with exit code 0.
- [ ] Statistical test shows empirical false-alarm rate $\le 10^{-6}$ across Monte Carlo null runs.
- [ ] Statistical power strictly matches or exceeds SynthID and Stable Signature.

# Phase 02: Adversarial Geometry, Spectral Invariance & Codebook Collapse Resistance

The finite-grid detector only inherits exact translation and ring-gain
identities in the idealized DFT model. Interpolated rotation, crop, and
rescaling survival are empirical properties. A $1/\sqrt{N}$ average-residual
heuristic additionally requires independent, balanced per-image carriers; it
does not follow for repeated content descriptors without the nonce defense.

## 1. Objectives & Theoretical Scope

Phase 02 addresses the core physical vulnerability of existing watermarking architectures: **codebook vulnerability under collusion** and **loss of synchronization under continuous geometric distortions**.

The primary theoretical objectives of this phase are:
1. Prove information-theoretic capacity and rate-distortion-robustness bounds under geometric deformations (affine group $\text{Aff}(2, \mathbb{R})$, projective transforms, non-linear elastic warping) and photometric attacks.
2. Formally derive the codebook collapse floor: proving that SIGIL's content-derived carrier structure limits cross-image phase coherence to the theoretical minimum of $\mathcal{O}(1/\sqrt{N})$, whereas static codebook baselines (SynthID, Stable Signature) suffer catastrophic $\mathcal{O}(1)$ collapse.
3. Validate through continuous Monte Carlo parameter sweeps that SIGIL strictly exceeds peer baselines in surviving geometric and collusion attacks without perceptual degradation.

---

## 2. Theoretical Analysis of Codebook Collapse

### 2.1 The Collusion Threat Model
Let an adversary gather $N$ independently generated watermarked images $I_1, I_2, \dots, I_N$. The adversary computes the linear collusion average:
$$\bar{I}_N = \frac{1}{N} \sum_{k=1}^N I_k$$

In a watermarking scheme with watermark residual $W_k = I_k - I_k^{\text{clean}}$, the colluded residual is $\bar{W}_N = \frac{1}{N} \sum_{k=1}^N W_k$.

### 2.2 Static Codebook Collapse in Baselines (SynthID & Stable Signature)
In static codebook architectures, the watermark is anchored to fixed carrier locations or a single latent key $\mathbf{k}^*$:
$$W_k = \mathbf{w}_{\text{static}} + \epsilon_k$$
where $\mathbf{w}_{\text{static}}$ is identical across all images produced by the model. Consequently:
$$\bar{W}_N = \mathbf{w}_{\text{static}} + \frac{1}{N} \sum_{k=1}^N \epsilon_k \xrightarrow{N \to \infty} \mathbf{w}_{\text{static}}$$
The adversary can extract the exact static watermark pattern $\mathbf{w}_{\text{static}}$ with signal-to-noise ratio increasing linearly with $\sqrt{N}$, allowing complete watermark subtraction from any protected image. Furthermore, across $N$ images, the cross-image correlation of residual spectra exhibits an $\mathcal{O}(1)$ non-vanishing component:
$$\mathbb{E}\left[ \left|\frac{1}{N} \sum_{k=1}^N e^{j \angle \mathcal{F}(W_k)(u, v)}\right| \right] = \Theta(1)$$

### 2.3 Conditional Codebook-Averaging Bound
SIGIL has separate content-derived and nonce-derived anchor families. Content
anchors can repeat across related images; only independently sampled nonces
guarantee distinct nonce-anchor inputs. Under a private PRF and independent
nonces, the associated signs are modeled as independent:
$$\mathbb{E}\left[ \langle \mathbf{s}_k, \mathbf{s}_l \rangle \right] = 0, \quad \operatorname{Var}\left( \langle \mathbf{s}_k, \mathbf{s}_l \rangle \right) = K$$
If the residual phasors at a frequency are additionally independent, unit
magnitude, and mean zero, Jensen's inequality gives:
$$\mathbb{E}\left[ \left|\frac{1}{N} \sum_{k=1}^N e^{j \angle \mathcal{F}(W_k)(u, v)}\right| \right] \le \frac{C}{\sqrt{N}}$$
with $C=1$ under those assumptions. This does not prove that arbitrary image
residuals, repeated content anchors, or an adaptive attacker satisfy the
assumptions. Codebook subtraction must therefore be evaluated empirically.

---

## 3. Geometric Invariance & Resynchronization Theory

### 3.1 Translation and Phase Invariance
Let $\mathcal{F}\{I\}(u, v) = R(u, v) e^{j \phi(u, v)}$. Under spatial translation $I'(x, y) = I(x - x_0, y - y_0)$:
$$\mathcal{F}\{I'\}(u, v) = R(u, v) e^{j (\phi(u, v) - 2\pi (u x_0 / W + v y_0 / H))}$$
The magnitude $|F(u, v)| = R(u, v)$ is strictly translation invariant. By embedding the invariant stratum entirely in the magnitude excess over radial expectation:
$$Z(u, v) = \frac{\log|F(u, v)| - \mu_R(\sqrt{u^2 + v^2})}{\sigma_R(\sqrt{u^2 + v^2})}$$
the ideal discrete-grid detector statistic is identical under cyclic integer
translations (Theorem T1). Interpolated subpixel translation is not covered by
that theorem.

### 3.2 Affine Lie Group Action on Fourier Space
For the continuous pullback $I'(x)=I(A^{-1}(x-b))$, the Fourier magnitude
transforms as:
$$|\mathcal{F}\{I'\}(\boldsymbol{\omega})| = |\det \mathbf{A}|\, |\mathcal{F}\{I\}(\mathbf{A}^{\top} \boldsymbol{\omega})|$$
By decomposing $\mathbf{A}$ via singular value decomposition into rotation $R(\theta)$, uniform scaling $s$, aspect ratio deformation $\lambda$, and shear, the detector evaluates the carrier correlation over a compact discrete grid $\mathcal{G}_{\text{geom}} \subset \text{SE}(2) \times \mathbb{R}^+$. The grid spacing $\Delta \theta, \Delta s$ is formally bounded by the Nyquist rate of the spectral excess correlation function to ensure worst-case correlation drop $\Delta \rho \le 0.05$.

---

## 4. Empirical Superiority Imperative

The agent must run comparative Monte Carlo sweeps validating that SIGIL outperforms SynthID and Stable Signature across all geometric and adversarial dimensions:

| Attack Dimension | Evaluation Parameter Range | Success Target for SIGIL | Competitor Baseline Target |
| :--- | :--- | :--- | :--- |
| **Collusion Resistance** | $N \in \{5, 10, 25, 50, 100\}$ colluding images | **TPR $\ge 98\%$ @ $\alpha=10^{-6}$** after collusion averaging | Baselines must suffer $> 50\%$ detection failure or key extraction |
| **Rotation** | $\theta \in [-45^\circ, +45^\circ]$ continuous | **TPR $\ge 95\%$** across all angles | SynthID fails ($< 20\%$ TPR beyond $5^\circ$ without grid search) |
| **Aspect Ratio Shift**| $\lambda \in [0.8, 1.25]$ | **TPR $\ge 96\%$** | Baselines degrade below $60\%$ |
| **Combined Crop + JPEG**| Crop fraction $0.7$, JPEG QF=50 | **TPR $\ge 94\%$** | Baselines achieve $< 75\%$ |
| **Perceptual Distortion**| Clean unattacked image | **PSNR $\ge 42.0\text{ dB}$, SSIM $\ge 0.985$** | Must match or exceed baseline fidelity |

---

## 5. Autonomous Self-Correction & Adaptation Playbook

### Failure Mode 1: Phase Coherence Exceeds Theoretical $1/\sqrt{N}$ Bound
- **Root Cause**: Carrier frequency hashing has non-uniform marginal distribution or anchor collisions across different images.
- **Agent Action**:
  1. Inspect `sigil/anchors.py` and `sigil/invariant.py:build_carriers`.
  2. Check the HMAC-SHA256 counter stream (`sigil/common.py:key_stream_rng`) and image-dependent nonce salts.
  3. Re-simulate cross-image phase coherence across 200 random images until empirical coherence converges to $\frac{1}{\sqrt{N}} \pm 2\sigma$.

### Failure Mode 2: Geometric Sensitivity Gap Under Large Rotations ($> 15^\circ$)
- **Root Cause**: Angular interpolation blur in discrete grid sampling or insufficient angular search resolution.
- **Agent Action**:
  1. Query literature on high-order polar-log Fourier transforms and Fourier-Mellin invariant representations.
  2. Refine the rotation search grid in `sigil/backend.py:transform_bank` or increase angular sampling density while updating Bonferroni multiplicity $M$.
  3. Verify that the increase in $M$ does not degrade detection sensitivity by more than $0.2\text{ dB}$ equivalent SNR.

---

## 6. Downstream Cascading Rules

- If the rotation/scale grid resolution is modified in `sigil/backend.py`, update the multiplicity formula in Phase 01 (`phase_01_formal_mathematical_foundations.md`) and re-compile TPU kernel tile sizes in Phase 03 (`phase_03_pod_kernel_architecture_and_competitor_acceleration.md`).
- If carrier generation logic changes, update `scripts/theory_checks.py:t6_codebook_collapse` and `scripts/geometry_frontier.py`.
- Synchronize Theorem T6 and Section 4 ("Robustness under Affine Deformations and Collusion") in `paper/sigil.tex`.

---

## 7. Verification Command & Acceptance Criteria

```bash
# 1. Verify codebook collapse and geometric invariance theorems
.venv/bin/python3 scripts/theory_checks.py

# 2. Run geometry frontier trade-off evaluation
.venv/bin/python3 scripts/geometry_frontier.py --tpu --trials 50
```

### Acceptance Criteria
- [ ] Cross-image phase coherence matches the $\mathcal{O}(1/\sqrt{N})$ curve with correlation coefficient $R^2 \ge 0.95$.
- [ ] Theorem T6 passes with empirical residual at theoretical floor.
- [ ] Detection rate under continuous rotation $[-30^\circ, +30^\circ]$ is $\ge 95\%$ at nominal $\alpha = 10^{-6}$.
- [ ] Detection rate strictly outperforms SynthID on all non-zero rotations and collusion attacks.

/-
SIGIL — the two invariances the analytic stratum is built on.

The analytic mark is read from the log-magnitude of the canonical spectrum,
normalised against the mean over its own ring.  That single choice is what
delivers robustness to whole families of attacks *by construction* rather than
by training, and both mechanisms are elementary enough to check formally.

`dft_translate_norm`   Shifting the image multiplies each Fourier coefficient by
                       a unit character, so the magnitude — and therefore the
                       statistic — is unchanged on the ideal cyclic grid.

`excess_radial_gain`   Multiplying every coefficient of a ring by one common
                       factor shifts the ring's log-magnitudes by one common
                       constant, which the ring mean removes exactly.  Gaussian
                       blur, unsharp masking, resampling roll-off, a global
                       contrast change can approximate this form; finite-grid
                       image operators need not preserve it exactly.
-/
import Mathlib

namespace Sigil

open Finset

/-! ### Translation invariance of the magnitude spectrum -/

section Translation

open ZMod

variable {N : ℕ} [NeZero N]

/-- Translating the input multiplies each Fourier coefficient by a character of
modulus one. -/
lemma dft_translate (Φ : ZMod N → ℂ) (d k : ZMod N) :
    ZMod.dft (fun j => Φ (j - d)) k = stdAddChar (-(d * k)) • ZMod.dft Φ k := by
  classical
  simp only [ZMod.dft_apply]
  rw [Finset.smul_sum]
  rw [← Equiv.sum_comp (Equiv.addRight d) (fun j => stdAddChar (-(j * k)) • Φ (j - d))]
  refine Finset.sum_congr rfl fun m _ => ?_
  have h1 : (Equiv.addRight d) m = m + d := rfl
  rw [h1, add_sub_cancel_right]
  have h2 : -((m + d) * k) = -(m * k) + -(d * k) := by ring
  rw [h2, AddChar.map_add_eq_mul, mul_comm, mul_smul]

/-- **Translation invariance.**  The magnitude spectrum does not move when the
image does, so the detector statistic — a function of magnitudes only — is
exactly invariant to translation.  No search, no approximation. -/
theorem dft_translate_norm (Φ : ZMod N → ℂ) (d k : ZMod N) :
    ‖ZMod.dft (fun j => Φ (j - d)) k‖ = ‖ZMod.dft Φ k‖ := by
  simp [dft_translate Φ d k]

end Translation

/-! ### Ring normalisation absorbs any radially symmetric gain -/

section RadialGain

variable {ι : Type*}

/-- The ring-normalised log-magnitude: a bin's log-magnitude minus the mean
log-magnitude of the ring it belongs to.  This is the evidence the analytic
detector correlates against its keyed chips. -/
noncomputable def excess (R : Finset ι) (lm : ι → ℝ) (i : ι) : ℝ :=
  lm i - (∑ j ∈ R, lm j) / R.card

/-- **Filter invariance.**  Adding a constant to every log-magnitude in a ring —
which is what multiplying that ring by a common gain does — leaves the
normalised evidence unchanged.

Every zero-phase, radially symmetric linear filter acts this way, so the
analytic statistic is invariant to the whole family at once. -/
theorem excess_radial_gain (R : Finset ι) (lm : ι → ℝ) (c : ℝ)
    {i : ι} (hi : i ∈ R) :
    excess R (fun j => lm j + c) i = excess R lm i := by
  have hcard : (0:ℝ) < R.card := by
    exact_mod_cast Finset.card_pos.mpr ⟨i, hi⟩
  simp only [excess, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
  field_simp
  ring

/-- The same statement for a genuine multiplicative gain on the magnitudes:
`log (c * m) = log c + log m`, so a positive gain shifts the ring's
log-magnitudes by a constant and is likewise absorbed. -/
theorem excess_mul_gain (R : Finset ι) (m : ι → ℝ)
    (hm : ∀ j ∈ R, 0 < m j) (c : ℝ) (hc : 0 < c) {i : ι} (hi : i ∈ R) :
    excess R (fun j => Real.log (c * m j)) i = excess R (fun j => Real.log (m j)) i := by
  have hrw : ∀ j ∈ R, Real.log (c * m j) = Real.log (m j) + Real.log c := by
    intro j hj
    rw [Real.log_mul (ne_of_gt hc) (ne_of_gt (hm j hj))]
    ring
  have hsum : ∑ j ∈ R, Real.log (c * m j)
      = (∑ j ∈ R, Real.log (m j)) + R.card * Real.log c := by
    rw [Finset.sum_congr rfl hrw, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
  have hcard : (0:ℝ) < R.card := by exact_mod_cast Finset.card_pos.mpr ⟨i, hi⟩
  simp only [excess, hsum, hrw i hi]
  field_simp
  ring

end RadialGain

end Sigil

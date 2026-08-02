/-
SIGIL — the end-to-end false-positive guarantee.

This is the theorem the whole design exists to support.  The deployed detector
maximises a keyed statistic over a large family of hypotheses in each of two
strata, then fuses the two.  Put together, the fraction of keys on which an
image that was never marked is declared marked is at most `α` — the number the
operator chose — for every image, every attack and every adversary.

Nothing here is calibrated against data.  There is no held-out set, no fitted
null, and no Gaussian approximation: the guarantee is a counting fact about the
key space.
-/
import Sigil.Rademacher
import Sigil.Multiplicity

namespace Sigil

open Finset Real
open scoped Classical

variable {n : ℕ} {ι ιA ιL : Type*}

/-- The threshold each stratum must clear, given its share of the budget and the
size of the search it performs. -/
noncomputable def stratumThreshold (α w : ℝ) (N : ℕ) : ℝ := threshold (w * α / N)

/-- One stratum, calibrated: searching `H` at its own threshold spends exactly
its share `w * α` of the budget. -/
theorem stratum_spend (H : Finset ι) (a : ι → Fin n → ℝ)
    (hnorm : ∀ h ∈ H, ∑ i, a h i ^ 2 = 1) (α w : ℝ)
    (hα : 0 < α) (hw : 0 < w) (hne : H.Nonempty) (hle : w * α / H.card ≤ 1) :
    (((univ : Finset (Fin n → Bool)).filter fun σ =>
        ∃ h ∈ H, stratumThreshold α w H.card ≤ stat (a h) σ).card : ℝ) / 2 ^ n
      ≤ w * α := by
  have hcard : (0:ℝ) < H.card := by exact_mod_cast Finset.card_pos.mpr hne
  have hq0 : 0 < w * α / H.card := by positivity
  have ht : 0 ≤ stratumThreshold α w H.card := Real.sqrt_nonneg _
  have hsearch := searched_tail H a hnorm ht
  rw [stratumThreshold] at hsearch ⊢
  rw [exp_neg_threshold_sq (w * α / H.card) hq0 hle] at hsearch
  calc (((univ : Finset (Fin n → Bool)).filter fun σ =>
          ∃ h ∈ H, threshold (w * α / H.card) ≤ stat (a h) σ).card : ℝ) / 2 ^ n
      ≤ H.card * (w * α / H.card) := hsearch
    _ = w * α := by field_simp

/-- **End-to-end calibration.**

`HA` and `HL` are the hypothesis families the two strata search — anchors,
quantiser cells, scales, rotations, reflections, nonces.  `aA` and `aL` are the
evidence vectors the detector reads for each hypothesis, whatever they happen to
be.  The detector fires when some hypothesis in either family clears that
stratum's threshold.

The conclusion: over a uniformly drawn key, that happens on at most an `α`
fraction of keys.  The two strata may be arbitrarily dependent — they read the
same image, and an adversary picks it. -/
theorem detector_fpr
    (HA : Finset ιA) (aA : ιA → Fin n → ℝ) (hA : ∀ h ∈ HA, ∑ i, aA h i ^ 2 = 1)
    (HL : Finset ιL) (aL : ιL → Fin n → ℝ) (hL : ∀ h ∈ HL, ∑ i, aL h i ^ 2 = 1)
    (α wA wL : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (hwA : 0 < wA) (hwL : 0 < wL) (hw : wA + wL ≤ 1)
    (hAne : HA.Nonempty) (hLne : HL.Nonempty)
    (hAle : wA * α / HA.card ≤ 1) (hLle : wL * α / HL.card ≤ 1) :
    (((univ : Finset (Fin n → Bool)).filter fun σ =>
        (∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ) ∨
        (∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ)).card : ℝ) / 2 ^ n
      ≤ α := by
  have hpos : (0:ℝ) < 2 ^ n := by positivity
  have hAcard : (0:ℝ) < HA.card := by exact_mod_cast Finset.card_pos.mpr hAne
  have hLcard : (0:ℝ) < HL.card := by exact_mod_cast Finset.card_pos.mpr hLne

  have bA := stratum_spend HA aA hA α wA hα0 hwA hAne hAle
  have bL := stratum_spend HL aL hL α wL hα0 hwL hLne hLle

  -- The fired set is the union of the two stratum events.
  have hsub :
      ((univ : Finset (Fin n → Bool)).filter fun σ =>
        (∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ) ∨
        (∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ))
      ⊆ ((univ : Finset (Fin n → Bool)).filter fun σ =>
            ∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ)
        ∪ ((univ : Finset (Fin n → Bool)).filter fun σ =>
            ∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ) := by
    intro σ hσ
    obtain ⟨hu, hor⟩ := Finset.mem_filter.mp hσ
    rcases hor with h | h
    · exact Finset.mem_union_left _ (Finset.mem_filter.mpr ⟨hu, h⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨hu, h⟩)

  have hcard := Finset.card_le_card hsub
  have hunion := Finset.card_union_le
    ((univ : Finset (Fin n → Bool)).filter fun σ =>
        ∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ)
    ((univ : Finset (Fin n → Bool)).filter fun σ =>
        ∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ)

  rw [div_le_iff₀ hpos] at bA bL ⊢
  calc (((univ : Finset (Fin n → Bool)).filter fun σ =>
          (∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ) ∨
          (∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ)).card : ℝ)
      ≤ ((((univ : Finset (Fin n → Bool)).filter fun σ =>
            ∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ).card
          + ((univ : Finset (Fin n → Bool)).filter fun σ =>
            ∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ).card : ℕ) : ℝ) := by
        exact_mod_cast le_trans hcard hunion
    _ = (((univ : Finset (Fin n → Bool)).filter fun σ =>
            ∃ h ∈ HA, stratumThreshold α wA HA.card ≤ stat (aA h) σ).card : ℝ)
        + (((univ : Finset (Fin n → Bool)).filter fun σ =>
            ∃ h ∈ HL, stratumThreshold α wL HL.card ≤ stat (aL h) σ).card : ℝ) := by
        push_cast; ring
    _ ≤ wA * α * 2 ^ n + wL * α * 2 ^ n := add_le_add bA bL
    _ = (wA + wL) * α * 2 ^ n := by ring
    _ ≤ 1 * α * 2 ^ n := by
        have : (wA + wL) * (α * 2 ^ n) ≤ 1 * (α * 2 ^ n) :=
          mul_le_mul_of_nonneg_right hw (by positivity)
        nlinarith [this]
    _ = α * 2 ^ n := by ring

end Sigil

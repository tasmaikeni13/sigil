/-
SIGIL — the price of searching.

A detector that reports the best of many hypotheses is testing many hypotheses.
SIGIL's detector maximises over content anchors, list-decoded quantiser cells,
scales, rotations, reflections and nonces — hundreds of thousands of them — and
then fuses two strata.  Reporting the best raw p-value would inflate the
false-positive rate by the size of that search, so every one of those maxima is
charged for here.

Both results below hold for *arbitrary dependence* between the things being
combined.  That is not a technicality: an adversary chooses the image, and can
therefore deliberately correlate what the two strata see.
-/
import Sigil.Rademacher

namespace Sigil

open Finset Real
open scoped Classical

variable {n : ℕ} {ι : Type*}

/-- **Multiplicity.**  The fraction of keys on which *some* searched hypothesis
reaches `t` is at most the number of hypotheses times the single-hypothesis
bound.

This is what licenses SIGIL's large resynchronisation grid.  The charge is
linear in the number of hypotheses but the threshold only grows like
`sqrt (2 log N)`, so breadth of search is cheap in detection power — which is
why the detector can afford to sweep hundreds of scales rather than hope the
attacker left the geometry alone. -/
theorem searched_tail (H : Finset ι) (a : ι → Fin n → ℝ)
    (hnorm : ∀ h ∈ H, ∑ i, a h i ^ 2 = 1) {t : ℝ} (ht : 0 ≤ t) :
    (((univ : Finset (Fin n → Bool)).filter
        fun σ => ∃ h ∈ H, t ≤ stat (a h) σ).card : ℝ) / 2 ^ n
      ≤ H.card * exp (-(t ^ 2) / 2) := by
  have hpos : (0:ℝ) < 2 ^ n := by positivity
  have hsub :
      ((univ : Finset (Fin n → Bool)).filter fun σ => ∃ h ∈ H, t ≤ stat (a h) σ)
        ⊆ H.biUnion (fun h => (univ : Finset (Fin n → Bool)).filter
            fun σ => t ≤ stat (a h) σ) := by
    intro σ hσ
    rw [Finset.mem_filter] at hσ
    obtain ⟨h, hH, hle⟩ := hσ.2
    exact Finset.mem_biUnion.mpr ⟨h, hH, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hle⟩⟩
  have hcard :
      (((univ : Finset (Fin n → Bool)).filter
          fun σ => ∃ h ∈ H, t ≤ stat (a h) σ).card : ℝ)
        ≤ ∑ h ∈ H, (((univ : Finset (Fin n → Bool)).filter
            fun σ => t ≤ stat (a h) σ).card : ℝ) := by
    have h1 := Finset.card_le_card hsub
    have h2 := Finset.card_biUnion_le (s := H)
      (t := fun h => (univ : Finset (Fin n → Bool)).filter fun σ => t ≤ stat (a h) σ)
    exact_mod_cast le_trans h1 h2
  rw [div_le_iff₀ hpos]
  calc (((univ : Finset (Fin n → Bool)).filter
          fun σ => ∃ h ∈ H, t ≤ stat (a h) σ).card : ℝ)
      ≤ ∑ h ∈ H, (((univ : Finset (Fin n → Bool)).filter
            fun σ => t ≤ stat (a h) σ).card : ℝ) := hcard
    _ ≤ ∑ _h ∈ H, exp (-(t ^ 2) / 2) * 2 ^ n := by
        refine Finset.sum_le_sum fun h hH => ?_
        have := rademacher_tail (a h) (hnorm h hH) ht
        rw [div_le_iff₀ hpos] at this
        exact this
    _ = H.card * exp (-(t ^ 2) / 2) * 2 ^ n := by
        rw [Finset.sum_const, nsmul_eq_mul]; ring

/-- **Adaptive refinement is charged for everywhere it could have looked.**

SIGIL's detector does not search its bank and stop.  It takes whichever
hypothesis scored best and opens a second, finer grid around *that* one — so
which hypotheses get evaluated is chosen after seeing the image.  Charging only
for the bank plus the one refinement actually performed would be wrong, because
a different image would have refined somewhere else, and the adversary picks the
image.

The honest charge is the union over every refinement the detector could have
opened.  If the bank is `B` and each refinement offers at most `k` hypotheses,
that union has at most `B.card * (1 + k)` members, whatever the data does — and
the tail bound then follows from `searched_tail` with no adaptivity left in it.

The cost is mild: the threshold grows like `sqrt (2 log N)`, so multiplying the
family by `1 + k` adds only `sqrt (2 log (1 + k))`-ish to the bar. -/
theorem refined_tail (B : Finset ι) (r : ι → Finset ι) (k : ℕ)
    (hr : ∀ h ∈ B, (r h).card ≤ k) (a : ι → Fin n → ℝ)
    (hnorm : ∀ h ∈ B ∪ B.biUnion r, ∑ i, a h i ^ 2 = 1) {t : ℝ} (ht : 0 ≤ t) :
    (((univ : Finset (Fin n → Bool)).filter
        fun σ => ∃ h ∈ B ∪ B.biUnion r, t ≤ stat (a h) σ).card : ℝ) / 2 ^ n
      ≤ (B.card * (1 + k) : ℕ) * exp (-(t ^ 2) / 2) := by
  have hbase := searched_tail (B ∪ B.biUnion r) a hnorm ht
  refine hbase.trans (mul_le_mul_of_nonneg_right ?_ (le_of_lt (exp_pos _)))
  have hbi : (B.biUnion r).card ≤ B.card * k := by
    refine le_trans (Finset.card_biUnion_le) ?_
    calc ∑ h ∈ B, (r h).card ≤ ∑ _h ∈ B, k := Finset.sum_le_sum hr
      _ = B.card * k := by rw [Finset.sum_const, smul_eq_mul]
  have : (B ∪ B.biUnion r).card ≤ B.card * (1 + k) := by
    refine le_trans (Finset.card_union_le _ _) ?_
    calc B.card + (B.biUnion r).card ≤ B.card + B.card * k := by omega
      _ = B.card * (1 + k) := by ring
  exact_mod_cast this

/-- **Weighted Bonferroni fusion.**  If each stratum's p-value is valid on its
own, then `min pᵢ / wᵢ` is a valid p-value for the intersection null whenever
the weights sum to at most one.

No independence is assumed anywhere, which is what makes the guarantee survive
an adversary who deliberately couples the strata — for instance by choosing an
attack that damages both in the same way. -/
theorem fuse_le {Ω : Type*} (S : Finset Ω) (p₁ p₂ : Ω → ℝ) (w₁ w₂ α : ℝ)
    (hw₁ : 0 < w₁) (hw₂ : 0 < w₂) (hsum : w₁ + w₂ ≤ 1)
    (h₁ : ((S.filter fun ω => p₁ ω ≤ w₁ * α).card : ℝ) ≤ w₁ * α * S.card)
    (h₂ : ((S.filter fun ω => p₂ ω ≤ w₂ * α).card : ℝ) ≤ w₂ * α * S.card)
    (hα : 0 ≤ α) :
    ((S.filter fun ω => min (p₁ ω / w₁) (p₂ ω / w₂) ≤ α).card : ℝ) ≤ α * S.card := by
  have hsub : (S.filter fun ω => min (p₁ ω / w₁) (p₂ ω / w₂) ≤ α)
      ⊆ (S.filter fun ω => p₁ ω ≤ w₁ * α) ∪ (S.filter fun ω => p₂ ω ≤ w₂ * α) := by
    intro ω hω
    obtain ⟨hS, hmin⟩ := Finset.mem_filter.mp hω
    rcases min_le_iff.mp hmin with h | h
    · exact Finset.mem_union_left _ (Finset.mem_filter.mpr ⟨hS,
        by rw [div_le_iff₀ hw₁] at h; linarith [h]⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨hS,
        by rw [div_le_iff₀ hw₂] at h; linarith [h]⟩)
  have hcard := Finset.card_le_card hsub
  have hunion := Finset.card_union_le
    (S.filter fun ω => p₁ ω ≤ w₁ * α) (S.filter fun ω => p₂ ω ≤ w₂ * α)
  have hSnn : (0:ℝ) ≤ (S.card : ℝ) := Nat.cast_nonneg _
  calc ((S.filter fun ω => min (p₁ ω / w₁) (p₂ ω / w₂) ≤ α).card : ℝ)
      ≤ (((S.filter fun ω => p₁ ω ≤ w₁ * α).card
          + (S.filter fun ω => p₂ ω ≤ w₂ * α).card : ℕ) : ℝ) := by
        exact_mod_cast le_trans hcard hunion
    _ = ((S.filter fun ω => p₁ ω ≤ w₁ * α).card : ℝ)
          + ((S.filter fun ω => p₂ ω ≤ w₂ * α).card : ℝ) := by push_cast; ring
    _ ≤ w₁ * α * S.card + w₂ * α * S.card := add_le_add h₁ h₂
    _ = (w₁ + w₂) * α * S.card := by ring
    _ ≤ 1 * α * S.card := by
        have : (w₁ + w₂) * (α * S.card) ≤ 1 * (α * S.card) :=
          mul_le_mul_of_nonneg_right hsum (by positivity)
        nlinarith [this]
    _ = α * S.card := by ring

end Sigil

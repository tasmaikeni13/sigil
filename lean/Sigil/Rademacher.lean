/-
SIGIL — machine-checked false-positive calibration.

The detector's null hypothesis is "this image was never marked with this key".
Under it the chip signs are a uniform random element of `Fin n → Bool`, drawn by
the key-derivation function and independent of the image, so the statistic is a
weighted Rademacher sum with weights the detector reads off the picture.

Everything here is finite and counting-based: a "probability" is the fraction of
sign patterns on which an event happens.  That is exactly the operational
quantity — the false-positive rate over the choice of key — and it avoids
needing a measure space at all.
-/
import Mathlib

namespace Sigil

open Finset Real

/-- The `±1` value of a Boolean chip sign. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

@[simp] lemma sgn_true : sgn true = 1 := rfl
@[simp] lemma sgn_false : sgn false = -1 := rfl

variable {n : ℕ}

/-- The detector statistic: the keyed sign pattern correlated with the evidence.

`a` is the per-carrier evidence the detector measures on the image, already
normalised so that `∑ aᵢ² = 1`; `σ` is the keyed sign pattern. -/
def stat (a : Fin n → ℝ) (σ : Fin n → Bool) : ℝ := ∑ i, a i * sgn (σ i)

/-- Summing `exp (t · stat)` over every sign pattern factors across carriers. -/
lemma sum_exp_stat (a : Fin n → ℝ) (t : ℝ) :
    ∑ σ : Fin n → Bool, exp (t * stat a σ)
      = ∏ i, (exp (t * a i) + exp (-(t * a i))) := by
  have hpt : ∀ σ : Fin n → Bool,
      exp (t * stat a σ) = ∏ i, exp (t * (a i * sgn (σ i))) := by
    intro σ
    rw [stat, Finset.mul_sum, Real.exp_sum]
  calc ∑ σ : Fin n → Bool, exp (t * stat a σ)
      = ∑ σ : Fin n → Bool, ∏ i, exp (t * (a i * sgn (σ i))) := by
        exact Finset.sum_congr rfl fun σ _ => hpt σ
    _ = ∏ i, ∑ b : Bool, exp (t * (a i * sgn b)) := by
        rw [Fintype.prod_sum (fun i (b : Bool) => exp (t * (a i * sgn b)))]
    _ = ∏ i, (exp (t * a i) + exp (-(t * a i))) := by
        refine Finset.prod_congr rfl fun i _ => ?_
        rw [Fintype.sum_bool]
        simp [sgn, mul_neg]

/-- Hoeffding's bound on the moment generating function, carrier by carrier. -/
lemma prod_exp_le (a : Fin n → ℝ) (hnorm : ∑ i, a i ^ 2 = 1) (t : ℝ) :
    ∏ i, (exp (t * a i) + exp (-(t * a i))) ≤ 2 ^ n * exp (t ^ 2 / 2) := by
  have hcosh : ∀ i : Fin n,
      exp (t * a i) + exp (-(t * a i)) = 2 * Real.cosh (t * a i) := by
    intro i; rw [Real.cosh_eq]; ring
  have hstep : ∀ i : Fin n,
      exp (t * a i) + exp (-(t * a i)) ≤ 2 * exp ((t * a i) ^ 2 / 2) := by
    intro i
    rw [hcosh i]
    have := Real.cosh_le_exp_half_sq (t * a i)
    linarith
  have hnonneg : ∀ i : Fin n, (0:ℝ) ≤ exp (t * a i) + exp (-(t * a i)) := by
    intro i; positivity
  calc ∏ i, (exp (t * a i) + exp (-(t * a i)))
      ≤ ∏ i, (2 * exp ((t * a i) ^ 2 / 2)) :=
        Finset.prod_le_prod (fun i _ => hnonneg i) (fun i _ => hstep i)
    _ = 2 ^ n * ∏ i, exp ((t * a i) ^ 2 / 2) := by
        rw [Finset.prod_mul_distrib]; simp
    _ = 2 ^ n * exp (∑ i, (t * a i) ^ 2 / 2) := by rw [Real.exp_sum]
    _ = 2 ^ n * exp (t ^ 2 / 2) := by
        congr 1
        congr 1
        have : ∑ i, (t * a i) ^ 2 / 2 = t ^ 2 / 2 * ∑ i, a i ^ 2 := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by ring
        rw [this, hnorm, mul_one]

/-- **Distribution-free null bound.** For a unit-norm evidence vector,
the fraction of keys on which the statistic reaches `t` is at most
`exp (-t²/2)`.

Nothing is assumed about the image: the bound holds for every weight vector, at
every finite `n`, with no asymptotics and no distributional model.  This is what
makes SIGIL's false-positive rate a design parameter rather than a measurement. -/
theorem rademacher_tail (a : Fin n → ℝ) (hnorm : ∑ i, a i ^ 2 = 1)
    {t : ℝ} (ht : 0 ≤ t) :
    (((univ : Finset (Fin n → Bool)).filter fun σ => t ≤ stat a σ).card : ℝ) / 2 ^ n
      ≤ exp (-(t ^ 2) / 2) := by
  classical
  set S := (univ : Finset (Fin n → Bool)).filter fun σ => t ≤ stat a σ with hS
  have hpos : (0:ℝ) < 2 ^ n := by positivity
  have hmark : (S.card : ℝ) * exp (t * t) ≤ ∑ σ ∈ S, exp (t * stat a σ) := by
    have : ∀ σ ∈ S, exp (t * t) ≤ exp (t * stat a σ) := by
      intro σ hσ
      have hσ' : t ≤ stat a σ := by
        have := Finset.mem_filter.mp hσ
        exact this.2
      exact Real.exp_le_exp.mpr (by nlinarith)
    calc (S.card : ℝ) * exp (t * t) = ∑ _σ ∈ S, exp (t * t) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ σ ∈ S, exp (t * stat a σ) := Finset.sum_le_sum this
  have hsub : ∑ σ ∈ S, exp (t * stat a σ)
      ≤ ∑ σ : Fin n → Bool, exp (t * stat a σ) :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
      (fun σ _ _ => (Real.exp_pos _).le)
  have hall : ∑ σ : Fin n → Bool, exp (t * stat a σ) ≤ 2 ^ n * exp (t ^ 2 / 2) := by
    rw [sum_exp_stat]; exact prod_exp_le a hnorm t
  have hchain : (S.card : ℝ) * exp (t * t) ≤ 2 ^ n * exp (t ^ 2 / 2) :=
    le_trans hmark (le_trans hsub hall)
  have hexp : exp (t ^ 2 / 2) / exp (t * t) = exp (-(t ^ 2) / 2) := by
    rw [← Real.exp_sub]; congr 1; ring
  rw [div_le_iff₀ hpos]
  have hgt : (0:ℝ) < exp (t * t) := Real.exp_pos _
  rw [← le_div_iff₀ hgt] at hchain
  calc (S.card : ℝ) ≤ 2 ^ n * exp (t ^ 2 / 2) / exp (t * t) := hchain
    _ = exp (-(t ^ 2) / 2) * 2 ^ n := by
        rw [mul_div_assoc, hexp]; ring

/-- The statistic value whose Hoeffding p-value is exactly `α`. -/
noncomputable def threshold (α : ℝ) : ℝ := Real.sqrt (2 * Real.log (1 / α))

lemma exp_neg_threshold_sq (α : ℝ) (h0 : 0 < α) (h1 : α ≤ 1) :
    exp (-(threshold α ^ 2) / 2) = α := by
  have hlog : 0 ≤ 2 * Real.log (1 / α) := by
    have hge : 1 ≤ 1 / α := by rw [le_div_iff₀ h0]; linarith
    have := Real.log_nonneg hge
    linarith
  rw [threshold, Real.sq_sqrt hlog]
  rw [show -(2 * Real.log (1 / α)) / 2 = -Real.log (1 / α) by ring]
  rw [Real.log_div one_ne_zero (ne_of_gt h0)]
  simp [Real.exp_log h0]

/-- Calibration: at the Hoeffding threshold, the false-positive fraction is at
most `α`, for every evidence vector. -/
theorem tail_at_threshold (a : Fin n → ℝ) (hnorm : ∑ i, a i ^ 2 = 1)
    {α : ℝ} (h0 : 0 < α) (h1 : α ≤ 1) :
    (((univ : Finset (Fin n → Bool)).filter
        fun σ => threshold α ≤ stat a σ).card : ℝ) / 2 ^ n ≤ α := by
  have ht : 0 ≤ threshold α := Real.sqrt_nonneg _
  have := rademacher_tail a hnorm ht
  rwa [exp_neg_threshold_sq α h0 h1] at this

end Sigil

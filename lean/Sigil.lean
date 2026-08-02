/-
SIGIL — the machine-checked core of the paper's theory.

Four things are proved here, and they are precisely the four the security
argument rests on.

* `Sigil.rademacher_tail` — the exact null.  For *any* evidence a detector could
  read off *any* image, the fraction of keys on which the statistic reaches `t`
  is at most `exp (-t²/2)`.  Finite, distribution-free, no asymptotics.
* `Sigil.searched_tail`, `Sigil.fuse_le` — the price of searching.  A detector
  that maximises over rotations, scales, content anchors and nonces is testing
  many hypotheses, and a fused detector tests several families; both are charged
  for, under arbitrary dependence.
* `Sigil.detector_fpr` — the two combined: the end-to-end false-positive
  guarantee at the operating point the implementation actually uses.
* `Sigil.dft_translate_norm`, `Sigil.excess_radial_gain` — the two invariances
  the analytic stratum is built on: translation leaves the magnitude spectrum
  alone, and a radially symmetric gain leaves the ring-normalised log-magnitude
  alone.
-/
import Sigil.Rademacher
import Sigil.Multiplicity
import Sigil.Invariance
import Sigil.Detector

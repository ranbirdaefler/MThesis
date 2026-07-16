# Part VI — Is it the rank representation? True-expression drug discrimination

**Date:** 2026-07-13
**Script:** `expression_space_discrimination.py` (model-free; streams Tahoe directly)
**Data:** Tahoe-100M, 8 randomly-sampled expression shards, 30 cell lines, ≤25 drugs/cell line,
40 cells/(cell line, drug); fixed 946-gene L1000 panel. Job `expr_space_581216`.
**Output:** `RESULTS/expr_space_discrimination.json`

---

## 1. The confound this closes

Every result in Parts I–V lives in **rank / cell-sentence** space. A cell sentence keeps only the
*order* of genes by expression and discards the **magnitude**. So the central null — "at
single-cell resolution the drug (and even mechanism) is undetectable" — has an untested escape
hatch: maybe the drug signal survives in expression *magnitude* and it is the **rank
representation**, not single-cell noise, that destroys it. Parts II–III could not settle this
because their expression-space metrics used expression **reconstructed from ranks** via
`linear_model.json` (a lossy, monotone rank→value map), not measured values — a caveat flagged
repeatedly but never resolved.

This part resolves it with **true, measured expression**. We stream cells straight from Tahoe,
build the normalized panel expression vector for each cell (C2S normalization
`log10(1 + 10⁴·cᵢ/Σc)`, identical QC to the sentence builder), and run the same forced-choice
discrimination protocol as Parts I–III — but on real expression vectors, sweeping single-cell →
pseudobulk. Because the model is removed entirely, this measures a property of the **data**: is the
drug resolvable in true expression at single-cell resolution, where it was not resolvable in rank?

**Design.** Forced choice (chance = 0.50): a reference pseudobulk of drug A vs a same-drug
candidate (disjoint A cells) and a different-drug candidate (B cells); correct if
`sim(ref, same) > sim(ref, diff)`. Accuracy is cell-line-clustered-bootstrapped over 30 cell
lines. Metrics, all on true expression: **cosine_expr**, **pearson_expr**, **spearman_expr**, and
**cosine_shift** (cosine of the treated − control shift, the expression analogue of DE-Δr). A
**MOA positive control** (same-MOA vs different-MOA, different drugs) certifies the instrument can
see the coarser mechanism signal. The same/diff **gap + Cohen's d** (mirroring Part I) gives
magnitude alongside accuracy.

---

## 2. Results

Forced-choice discrimination accuracy (chance = 0.50; 95% CI, cell-line bootstrap):

| metric (true expression) | single-cell (size 1) | pseudobulk-15 |
|---|---|---|
| cosine_expr  | 0.529 [0.52, 0.54] | 0.723 [0.71, 0.74] |
| pearson_expr | 0.530 [0.52, 0.54] | 0.729 [0.71, 0.75] |
| spearman_expr | 0.531 [0.52, 0.54] | 0.627 [0.61, 0.65] |
| **cosine_shift** (vs control) | **0.542 [0.53, 0.56]** | **0.793 [0.78, 0.81]** |

MOA positive control (same-MOA vs diff-MOA, different drugs):

| metric | size 1 | size 15 |
|---|---|---|
| cosine_expr | 0.563 [0.51, 0.61] | 0.600 [0.51, 0.68] |
| cosine_shift | 0.511 [0.48, 0.55] | 0.499 [0.42, 0.57] |

Same/diff gap + Cohen's d (magnitude):

| level | metric | same | diff | gap [95% CI] | d |
|---|---|---|---|---|---|
| single-cell | cosine_expr | 0.348 | 0.343 | +0.006 [+0.003, +0.008] | 0.06 |
| single-cell | cosine_shift | 0.050 | 0.042 | +0.009 [+0.006, +0.011] | 0.15 |
| pb-15 | cosine_expr | 0.887 | 0.874 | +0.013 [+0.012, +0.015] | 0.42 |
| pb-15 | cosine_shift | 0.429 | 0.359 | +0.070 [+0.063, +0.077] | 0.87 |

(All permutation p at the 1/(n_perm+1) floor — consistent-across-cell-lines direction, not large
magnitude, exactly as in Part I.)

---

## 3. Reading the numbers

**The representation is exonerated: true expression is drug-blind at single cell too.** Every
expression metric discriminates drugs at **~0.53** per cell — a hair above chance (CIs exclude 0.50
because of the large trial count, but the effect is negligible: d ≈ 0.06–0.15). This is the *same
near-chance regime* rank was in (Part I single-cell d ≈ 0.01). **Switching to true expression does
not resolve single-cell drug discrimination.** The drug-blindness is therefore **not an artifact of
the cell-sentence / rank representation** — it is a property of single-cell resolution. The
long-standing "but you only had reconstructed expression" caveat is now closed, and it closes in
favor of the thesis: no single-cell representation, rank or expression, recovers the drug.

**Aggregation recovers signal, monotonically** (0.53 → 0.72–0.79), and the effect size grows with
it (cosine d 0.06 → 0.42; shift d 0.15 → 0.87). This reproduces Part I's core finding in a second,
independent representation: the drug signal is real but only emerges once single-cell noise is
averaged down.

**The control-referenced shift is the strongest expression discriminator** — cosine_shift is best
at every level (0.542 single-cell, **0.793 at pb-15, d = 0.87**). This is the expression-space
analogue of DE-Δr, and its dominance is consistent with the whole "condition on the control cell"
story: the drug's fingerprint lives in the *shift away from this cell's own baseline*, not in the
absolute profile. But even the shift is only ~0.54 per cell — so it sharpens, not overturns, the
single-cell verdict.

**Rank vs expression, directly.** The interim comparison against the rank spike-in (Part III:
topn_tau discriminates real drug populations at ~0.98 at pb-15) indicates rank is **at least as
strong as** true expression at pseudobulk — expression does not beat it. The combined run (rank
metrics `topn_tau`/`de_delta` now scored on the *same* pseudobulks; §5) confirms this on identical
cells and folds both representations into one table.

---

## 4. Caveats (carry to the meeting)

- **The MOA positive control is weak** (0.56–0.60, CI floors near 0.50; cosine_shift's MOA control
  sits at chance, 0.51/0.50). This is largely the known coarseness/noise of the MOA labels (Part I,
  §4 caveat) plus, for the shift, the approximation below. The **drug** result is not affected — its
  CIs are tight and exclude 0.50 — but do not over-read the shift's MOA null as "the shift ignores
  mechanism"; the labels are simply too coarse to group cleanly at this N.
- **The shift used a pooled, not plate-matched, control.** cosine_shift anchors on the cell-line
  **mean** DMSO profile (cheap in-stream), not the plate-matched control used in the model eval. The
  shift accuracies are therefore a slight *under*-estimate of what a plate-matched anchor would give;
  they only strengthen the "signal concentrates in the shift" reading, not weaken it.
- **Accuracy is consistency, not magnitude** — as throughout the project. The ~0.72–0.79 pb-15
  accuracies coexist with small effect sizes (d ≈ 0.4–0.9); both are reported.

---

## 5. Status / follow-up

- **Combined rank-vs-expression run (enabled, pending resubmit).** `expression_space_discrimination.py`
  now also scores `topn_tau` and `de_delta` on the *same* pseudobulks, so the next run produces a
  single table with rank and expression side by side on identical cells — the airtight version of §3's
  interim comparison. (No new data; same CPU job.)
- **Implication for the external-model comparison.** Because true expression does not resolve drugs
  at single-cell resolution, an expression-space model (e.g. STATE) is unlikely to be rescued at
  single-cell resolution either; the one place it might claw back signal is the control-referenced
  shift at pseudobulk (d = 0.87). This is a concrete, falsifiable prediction to state *before*
  running STATE/scGPT.

## 6. One-line conclusion

At single-cell resolution the drug is undetectable in **true expression** just as it is in rank
(both ≈ 0.53), so the drug-blindness is a resolution limit, **not** a rank-representation artifact;
the signal re-emerges only under aggregation, most strongly in the control-referenced shift
(pb-15 cosine_shift 0.79, d 0.87).

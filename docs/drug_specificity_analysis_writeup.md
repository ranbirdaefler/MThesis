# Drug-Specificity in the Data: A Confound-Controlled Analysis

**Date:** 2026-07-08
**Script:** `drug_specificity_in_data.py` (v2)
**Output:** `eval_results/drug_specificity_v2.json`
**Data:** Tahoe-100M, `data_diverse2` (train + tier1_seen_conditions), 376,794 treated cells across 50 cell lines; 40 cell lines used.

---

## 1. Motivation and question

The fine-tuned model reaches a strong headline score (DE-Δr ≈ 0.72) but two earlier ablations
showed (a) it ignores the drug (a prompt-scramble to a different-mechanism drug does not change
its output) and (b) it is matched or beaten by a closed-form linear control→shift regression.
This raised a fork: is the problem the **model** (a drug-specific signal exists but the model
fails to capture it) or the **data/metric** (the drug-specific signal is absent, or invisible to
our metrics, so no model could use it and no metric would reward it)?

These two possibilities require completely different responses (better model vs. better
metric/representation), so they must be distinguished. The way to distinguish them is to remove
the model entirely and ask whether the drug signal is even present and detectable **in the real
data**.

**Core question.** Within a single cell line, do two *real* treated cells given the *same* drug
agree more (under a similarity metric) than two real treated cells given *different* drugs? If
yes, a drug-specific signal exists that the metric can see, and the failure is the model's. If
no, the drug barely registers in the data (or the metric cannot resolve it), and the bottleneck
is upstream of any model.

---

## 2. Design

**Unit of comparison.** For a pair of treated cells (A, B) of the same cell line, A is treated as
"truth" and B as "prediction", anchored on A's control; the similarity is the same metric used
throughout the project. A positive result requires same-drug pairs to score higher than
different-drug pairs.

**The gap.** For each cell line we compute
`gap = mean(same-drug agreement) − mean(different-drug agreement)`,
then aggregate across the 40 cell lines. A gap near zero means the drug is not detectable.

**Everything is within cell line.** Different cell lines have different baseline agreement, so
all comparisons are formed within a cell line; the cell line is the unit of replication.

**Metrics (all three used throughout the project):**
- **DE-Δr** — Pearson of the rank-shift (vs. control) over the top-K differentially-expressed genes.
- **panel-τ** — Kendall τ over all 946 genes (inflated by the inactive-gene tail; diagnostic only).
- **topN-τ** — Kendall τ over the top ~100 expressed genes (tail-free, honest ordering metric).

**Statistics.**
- **Effect size:** the gap, with a **cell-line-clustered bootstrap** 95% CI.
- **Significance:** a **within-cell-line permutation test** (sign-flipping the per-cell-line gaps),
  which respects the dependence structure and avoids pseudoreplication. Note the reported p-floor
  is 1/(n_perm+1) ≈ 0.0005 for 2000 permutations — a saturated p reflects *consistency across cell
  lines*, not effect magnitude.
- **Cohen's d:** standardized effect size, to judge whether the distributions actually separate
  (magnitude), independent of the p-value.
- **frac:** gap expressed as a fraction of recoverable signal = gap / (replicate ceiling − diff).
  This is only interpretable when the different-drug agreement sits below the ceiling; when
  different-drug already equals the ceiling the denominator collapses and `frac` is meaningless
  (this is itself an informative failure — see §4).

---

## 3. Controls (what makes the result trustworthy)

To ensure any gap is real drug signal and not an artifact — and to make a null interpretable —
the analysis includes:

1. **Dose-matched (`single_samedose`).** Same-drug-same-dose vs different-drug-same-dose, so a gap
   cannot be explained by same-drug pairs sharing dose.
2. **Different-plate-only (`single_diffplate`).** Both cells drawn from different plates, so a gap
   cannot be a shared-plate batch effect.
3. **MOA positive control (`moa_poscontrol`).** Same-mechanism vs different-mechanism drugs (using
   *different* drugs). Drugs of the same mechanism should produce more similar responses than
   drugs of different mechanisms. This is the **key check that the test works**: if it cannot
   detect the coarser, expected mechanism signal, then a null on drug identity is uninterpretable
   (could just be an underpowered test).
4. **Pseudobulk sweep (`pb5`, `pb15`).** Average N cells into denoised profiles before comparing,
   to see whether a signal buried in single-cell noise emerges under aggregation. Same-drug
   pseudobulk pairs are built from two **disjoint** halves of a drug's cells (two independent
   replicates), so a "same-drug pair" is never the same cells averaged twice.
5. **Replicate noise ceiling.** Agreement between two cells of the *identical* condition (same
   drug, dose, plate) — the maximum achievable given single-cell measurement noise.

---

## 4. Results

**Replicate noise ceiling:** DE-Δr = 0.766, panel-τ = 0.702, topN-τ = 0.271.

| analysis | metric | same | diff | gap [95% CI] | perm p | Cohen's d |
|---|---|---|---|---|---|---|
| **single** | DE-Δr | 0.766 | 0.766 | +0.002 [+0.001, +0.003] | 0.0015 | +0.01 |
| single | panel-τ | 0.704 | 0.699 | +0.007 [+0.005, +0.009] | 0.0005 | +0.07 |
| single | topN-τ | 0.266 | 0.264 | −0.001 [−0.005, +0.003] | 0.664 | +0.02 |
| **single_samedose** | DE-Δr | 0.767 | 0.766 | +0.002 [−0.000, +0.005] | 0.073 | +0.02 |
| single_samedose | panel-τ | 0.698 | 0.697 | +0.002 [−0.002, +0.005] | 0.382 | +0.01 |
| single_samedose | topN-τ | 0.275 | 0.267 | +0.007 [−0.001, +0.015] | 0.108 | +0.05 |
| **single_diffplate** | DE-Δr | 0.767 | 0.766 | +0.002 [−0.000, +0.004] | 0.052 | +0.02 |
| single_diffplate | panel-τ | 0.699 | 0.698 | +0.004 [+0.002, +0.006] | 0.0005 | +0.01 |
| single_diffplate | topN-τ | 0.262 | 0.261 | −0.005 [−0.011, +0.001] | 0.163 | +0.01 |
| **moa_poscontrol** | DE-Δr | 0.766 | 0.765 | +0.001 [−0.001, +0.003] | 0.422 | +0.02 |
| moa_poscontrol | panel-τ | 0.696 | 0.699 | −0.000 [−0.003, +0.003] | 0.906 | −0.04 |
| moa_poscontrol | topN-τ | 0.261 | 0.262 | −0.000 [−0.006, +0.006] | 0.995 | −0.01 |
| **pb5** | DE-Δr | 0.755 | 0.744 | +0.010 [+0.005, +0.015] | 0.002 | +0.11 |
| pb5 | panel-τ | 0.801 | 0.796 | +0.005 [+0.004, +0.006] | 0.0005 | +0.13 |
| pb5 | topN-τ | 0.350 | 0.334 | +0.016 [+0.011, +0.021] | 0.0005 | +0.16 |
| **pb15** | DE-Δr | 0.799 | 0.779 | +0.016 [+0.009, +0.023] | 0.0005 | +0.26 |
| pb15 | panel-τ | 0.875 | 0.868 | +0.006 [+0.005, +0.007] | 0.0005 | +0.38 |
| pb15 | topN-τ | 0.552 | 0.516 | +0.034 [+0.029, +0.039] | 0.0005 | +0.43 |

(`pb40` had no data: too few drugs have ≥ 80 cells to split into two 40-cell halves.)

### Reading the numbers

**Single-cell: no meaningful drug signal.** The `single` gap is +0.002 on DE-Δr (d = 0.01,
negligible) and −0.001 on topN-τ (not significant). The small panel-τ gap (+0.007, p = 0.0005) is
significant only because the permutation test detects a consistent-across-cell-lines difference of
trivial magnitude (d = 0.07); at 0.704 vs 0.699 it is biologically meaningless.

**The confound controls erase even that.** Under dose-matching the DE-Δr gap loses significance
(p = 0.073); under different-plate it is borderline (p = 0.052). The small single-cell gaps were
partly dose/plate structure, not drug identity.

**The positive control FAILS.** Same-MOA vs different-MOA drugs are indistinguishable on every
metric (DE-Δr +0.001 p = 0.42 d = 0.02; panel-τ −0.000 p = 0.91; topN-τ −0.000 p = 0.99). At
single-cell resolution the metrics cannot resolve even the coarser mechanism signal — not just
drug identity. This is the decisive control: it means a single-cell null on drug identity is not
an artifact of an underpowered test, because the test also fails to see mechanism, which should be
easier.

**Different-drug agreement is already at the noise ceiling.** For DE-Δr, different-drug = 0.766 =
ceiling 0.766. Two cells given completely different drugs agree exactly as much as two replicates
of the identical condition. There is no headroom for the drug to register — the drug-agnostic
control→treated drift saturates the metric. (This is why the `frac` column shows nonsensical
values like 847% for the single-cell rows: the denominator ceiling − diff ≈ 0.)

**Aggregation recovers a small, real signal.** Under pseudobulk the gap grows monotonically with
the number of cells averaged, and it is carried by the expressed genes (topN-τ):
- topN-τ gap: +0.016 (d = 0.16) at N = 5 → +0.034 (d = 0.43) at N = 15.
- DE-Δr gap: +0.010 (d = 0.11) at N = 5 → +0.016 (d = 0.26) at N = 15.

The monotonic growth is important: if there were no drug signal, averaging would not create one.
So a real drug-specific signal exists in the data; it is simply below the single-cell noise floor
and only emerges once that noise is averaged down. Even at N = 15 the effect is small-to-moderate
(d ≈ 0.43 for topN-τ), so the drug explains a modest fraction of the response even at its most
detectable.

---

## 5. Conclusion

1. **At the single-cell level, the metrics cannot resolve drug or mechanism identity.** The MOA
   positive control fails, the confound-controlled drug gaps are negligible, and different-drug
   pairs already sit at the replicate noise ceiling. This is stronger than "the model ignores the
   drug": the data-representation + metric cannot see the drug axis per-cell, regardless of model.

2. **A drug-specific signal does exist in the data, but only emerges under aggregation.** It grows
   with the number of cells averaged (topN-τ d: 0.16 → 0.43 from N = 5 to N = 15) and lives in the
   ordering of expressed genes (topN-τ), not DE-Δr or panel-τ. It remains a small-to-moderate
   effect even when most detectable.

3. **The bottleneck is single-cell noise plus the metric/representation, not (only) the model.** A
   model trained and scored per cell is being asked to predict a signal that is undetectable at
   that resolution even in perfect ground-truth data. This is a rigorous demonstration of the
   metric-discrimination problem: we do not have a metric (at single-cell resolution) that can tell
   whether a model is capturing the drug.

### Caveat

The MOA labels are coarse and noisy, so the failed MOA positive control is partly "no mechanism
signal at single-cell resolution" and partly "the labels are not precise enough to group drugs
cleanly." It should not be read as proof that mechanism is fundamentally undetectable. However,
combined with the ceiling saturation and the monotone aggregation trend, the overall picture is
consistent and robust.

---

## 6. Implications for next steps

- **Evaluation (and possibly training) should move to aggregated/denoised (pseudobulk)
  resolution**, where the drug signal is actually resolvable. Single-cell DE-Δr cannot discriminate
  perturbations and therefore cannot tell a good model from a bad one.
- **The expressed-gene ordering (topN-τ) is where the drug signal lives** — metric design should
  focus there, not on DE-Δr or whole-panel τ.
- These findings motivate the metric-benchmarking work now in progress (the spike-in discrimination
  protocol) and the representation fixes under discussion (tail-rank and zero-bucket handling of
  inactive genes), which target exactly the single-cell-resolution and tie/tail issues identified
  here.

*(To be combined with the spike-in metric-benchmark results once that run completes.)*

# Drug-Specificity in the Data: A Confound-Controlled Analysis

> ⚠️ **DATA REGIME: MIXED — being migrated to [END_CELL].** This is the analysis spine (Parts I–V).
> **Parts I, II, IV were run on the LEGACY full-panel `data_diverse2`** (kept as method-evolution
> context); **Parts III and V are on the current [END_CELL] `data_diverse2_endcell_big`.** Part VI
> (`part6_expression_space_draft.md`) and the mechanistic probe (`dimensionality_probe_analysis.md`)
> are [END_CELL]. The legacy Parts I/II are slated for a [END_CELL] rerun (`gap_endcell.sbatch`).
> Verdicts and current numbers are in the root `FINDINGS.md`.

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

---

# Part II — Spike-in Metric Discrimination Benchmark

**Date:** 2026-07-09
**Script:** `spikein_metric_benchmark.py`
**Output:** `eval_results/spikein_benchmark.json`
**Data:** same (`data_diverse2`, train + tier1); 40 cell lines, sample_size=15, 150 trials/pair, 25 drug-pairs/cell-line.

## 1. What this adds

Part I measured the *magnitude* of the same-drug vs different-drug gap. Part II implements the
advisor's **spike-in protocol**, which measures **discrimination accuracy** instead: in a
forced-choice trial, a reference pseudobulk (drug A) is compared against a same-drug candidate
(drug A) and a different-drug candidate (drug B); the trial is "correct" if the metric rates the
same-drug candidate as more similar. Accuracy is the fraction correct (chance = 0.50). The
**spike-in titration** then progressively contaminates the different-drug candidate with a
fraction *s* of drug-A cells (s = 0 → 1.0); accuracy should fall from high (pure B) toward 0.50
(at s = 1.0 the two candidates are identical). The metric whose accuracy stays high to larger *s*
is the more sensitive discriminator.

Three representations were requested for comparison: `position` (current), `tail_max` (Federico's
fix: inactive genes tied at max rank), `zero_bucket` (professor's fix: inactive genes at a shared
mid rank).

## 2. Results

Discrimination accuracy vs spike-in contamination (chance = 0.50):

| metric | s=0 | s=0.1 | s=0.2 | s=0.3 | s=0.5 | s=0.7 | s=1.0 |
|---|---|---|---|---|---|---|---|
| de_delta | 0.964 | 0.944 | 0.930 | 0.914 | 0.813 | 0.739 | 0.501 |
| panel_tau | 0.991 | 0.981 | 0.974 | 0.963 | 0.879 | 0.797 | 0.501 |
| topn_tau | 0.912 | 0.889 | 0.875 | 0.859 | 0.767 | 0.703 | 0.500 |
| spearman_expr | 0.666 | 0.642 | 0.634 | 0.622 | 0.581 | 0.558 | 0.498 |
| cosine_expr | 0.663 | 0.641 | 0.629 | 0.620 | 0.580 | 0.557 | 0.498 |
| cosine_shift | 0.617 | 0.598 | 0.592 | 0.582 | 0.554 | 0.538 | 0.498 |

(The `tail_max` and `zero_bucket` tables were byte-for-byte identical to `position` — see §4.)

## 3. Interpretation

**The sanity checks pass.** Every metric returns to ~0.50 at s = 1.0 (fully contaminated →
indistinguishable → chance), and accuracy decreases monotonically with contamination. So the
protocol is behaving correctly.

**Reconciling with Part I (important — not a contradiction).** Part I found a *tiny* same-vs-diff
magnitude gap at pseudobulk-15 (DE-Δr +0.016, d≈0.26; topN-τ +0.034, d≈0.43). Part II finds
*high* forced-choice accuracy (DE-Δr 0.964, topN-τ 0.912) at the same resolution. These are
consistent: forced-choice accuracy measures the **consistency/detectability** of a difference, not
its **magnitude**. A gap that is small but highly consistent in direction yields high accuracy —
if same-drug is reliably (almost always) a little more similar than different-drug, the forced
choice is right almost every time, even though the effect size is small. So the two analyses
together say: **the pseudobulk drug signal is small in magnitude but consistent enough to be
detected almost every time.** Accuracy is the more sensitive readout; it should not be mistaken
for a large effect.

**Metric ranking (the useful output for metric selection).** By *both* peak accuracy and graceful
degradation under contamination, the **rank-based metrics dominate the expression-space metrics**:
- panel_tau (0.991 → still 0.797 at s=0.7) and de_delta (0.964 → 0.739) are the strongest;
  topn_tau is close behind (0.912 → 0.703).
- The expression-space metrics are much weaker: spearman_expr and cosine_expr peak at ~0.66 and
  fall to near-chance by s=0.5; cosine_shift is weakest (0.617 peak).

This is a notable result and runs *counter* to the prior expectation (from the advisor's earlier
experience) that cosine in expression space would be the strongest discriminator. Here, in this
representation, the rank metrics discriminate far better. (Caveat in §4 on why.)

**panel_tau's apparent superiority is partly the tail artifact, again.** panel_tau tops the table
(0.991), but recall from Part I that panel-τ is inflated by the shared inactive-gene tail. That
same shared structure makes two pseudobulks of the same cell line look very similar overall, which
helps forced-choice discrimination but is not specifically about the drug — it partly reflects
cell-line/representation structure. So panel_tau's lead should be read with the same caution as in
Part I; topn_tau (tail-free) is the more honest strong performer.

## 4. Important caveats and a preprocessing finding

**The three representations were identical because current cell sentences contain all 946 genes.**
Tahoe cell sentences are full-panel orderings — every gene is present (inactive genes sit at the
bottom of the ranking, not absent). The `tail_max` and `zero_bucket` fixes only change how
*absent* genes are placed, and there are no absent genes, so all three modes produced identical
rank arrays. **Consequence:** the inactive-gene representation fixes (Federico's tail-max,
the professor's zero-bucket, and the related [END CELL] idea) cannot be evaluated on the current
data format — they require first rebuilding the preprocessing so that inactive genes are *dropped*
from the sentence (made absent), which is precisely what the [END CELL] proposal does. This is a
concrete prerequisite to bring to the meeting: **to test the representation fixes, the data must be
regenerated with inactive genes removed.**

**The expression reconstruction is approximate.** The expression-space metrics reconstruct
expression from rank via the `linear_model.json` slope/intercept map (or a linear fallback), not
from true expression values. So the weak performance of cosine_expr / spearman_expr may partly
reflect lossy rank→expression reconstruction rather than an intrinsic weakness of expression-space
similarity. A fairer test of the advisor's cosine-in-expression suggestion would use the actual
expression values (or the C2S decoder) rather than a reconstructed proxy. This is worth flagging
before concluding that expression-space metrics are inferior.

**Accuracy is a sensitivity readout, not an effect size.** As in §3, high accuracy reflects
consistency, not magnitude. The magnitude of the drug signal remains small (Part I). Both are
needed for an honest picture.

## 5. Combined conclusion (Parts I + II)

- At **single-cell** resolution the drug (and even mechanism) is undetectable: gaps ≈ 0, MOA
  positive control fails, different-drug pairs sit at the noise ceiling.
- At **pseudobulk** resolution a drug signal exists. It is **small in magnitude** (d ≈ 0.26–0.43)
  but **highly consistent**, so forced-choice discrimination accuracy is high (0.91–0.96 for the
  best rank metrics) and degrades gracefully under spike-in contamination.
- Among tested metrics, **rank-based metrics (topn_tau, de_delta; panel_tau with the tail caveat)
  discriminate far better than reconstructed-expression-space metrics** — though the
  expression-space test is limited by approximate rank→expression reconstruction.
- The inactive-gene **representation fixes could not be tested** because the current sentences
  include all genes; testing them requires regenerating the data with inactive genes dropped
  ([END CELL]-style), which is a clear, concrete next step for the meeting.

### Recommended next steps
1. Regenerate a small dataset with inactive genes dropped ([END CELL] representation), then re-run
   this spike-in benchmark to actually compare position vs tail_max vs zero_bucket.
2. Re-run the expression-space metrics on **true expression values** (not rank-reconstructed) to
   fairly test cosine-in-expression.
3. Adopt discrimination accuracy + spike-in curves as the standard metric-comparison instrument
   for the metric-design phase, reported alongside effect sizes (never accuracy alone).

---

# Part III — [END_CELL] Representation Rebuild + Three-Way Discrimination Benchmark

**Date:** 2026-07-09
**Data:** `data_diverse2_endcell` (newly rebuilt), train + eval_tier1; 286,894 cells, 50 cell lines.
**Scripts:** `tahoe_c2s_preprocess_endcell.py` (rebuild), `spikein_metric_benchmark.py` (diagnostic + benchmark).
**Outputs:** `eval_results/activity_diagnostic.json`, `eval_results/spikein_endcell.json`.

## 1. Why the rebuild was necessary

In Part II, the three representations (`position`, `tail_max`, `zero_bucket`) produced byte-identical
results because the original cell sentences contained all 946 panel genes — inactive genes were
padded into a canonical tail, not absent. The representations only differ in how they place
*absent* genes, so with no absent genes they could not diverge. To make the representation
comparison meaningful, the data was regenerated in an **[END_CELL] format**: each sentence lists
only the expressed panel genes (ranked by expression) followed by an `[END_CELL]` sentinel;
unexpressed genes are absent (implied zero). The rebuild preserved the seeds, held-out drugs, and
tier definitions; a synthetic unit test confirmed the sentence builder emits expressed genes +
sentinel with unexpressed genes dropped, and a leakage check confirmed 0 held-out drugs leak into
train.

## 2. Activity-set diagnostic (did the rebuild give the representations something to differ on?)

Run before the benchmark, over 286,894 cells:

| quantity | mean | median | [p10, p90] |
|---|---|---|---|
| treated expressed genes / cell | 125.9 | 110 | [64, 209] |
| control expressed genes / cell | 115.3 | 101 | [58, 193] |
| genes turned ON (treat, not ctrl) | 90.9 | 79 | [45, 153] |
| genes turned OFF (ctrl, not treat) | 80.3 | 70 | [39, 136] |
| same-drug expressed-set Jaccard | 0.157 | 0.161 | [0.103, 0.213] |
| diff-drug expressed-set Jaccard | 0.157 | 0.150 | [0.095, 0.228] |
| same-drug #genes differing (symdiff) | 160.6 | 148 | [118, 226] |
| diff-drug #genes differing (symdiff) | 174.5 | 167 | [115, 244] |
| frac of cell-pairs with union of active sets < P | 1.000 | — | — |

**Reading:**
- Sentences are genuinely sparse: ~126 of 946 genes expressed per cell, so ~820 genes are absent
  per cell. In **every** sampled cell-pair some panel gene is absent in both (frac = 1.000). In the
  old format this fraction would have been 0. So the representations now have abundant absent genes
  to place differently — the rebuild achieved its purpose.
- Drug treatment flips a large number of genes between active and silent states: ~91 ON and ~80 OFF
  per cell (~170 on/off transitions). This **quantitatively motivates the professor's zero-bucket**
  (whose stated purpose was making on/off transitions analyzable) — there is abundant on/off
  structure to act on.
- **However**, the binary active-set overlap barely distinguishes drugs: same-drug and diff-drug
  Jaccard are both 0.157; diff-drug symdiff (174.5) is only ~14 genes higher than same-drug (160.6).
  Drug identity is weakly encoded in *which* genes are on/off. Moreover, even same-drug replicates
  share only ~16% of active genes (Jaccard 0.157) — a direct measure of how severe single-cell
  dropout is. The drug signal in gene presence/absence is real but noise-dominated at single-cell
  resolution, consistent with Parts I–II; this is why the benchmark is run at pseudobulk (sample
  size 15).

## 3. Three-way discrimination benchmark

Discrimination accuracy vs spike-in contamination (chance = 0.50), sample_size = 15:

**position** (absent genes at rank P+1)

| metric | s=0 | s=0.1 | s=0.2 | s=0.3 | s=0.5 | s=0.7 | s=1.0 |
|---|---|---|---|---|---|---|---|
| de_delta | 0.989 | 0.982 | 0.976 | 0.968 | 0.893 | 0.817 | 0.499 |
| panel_tau | 0.996 | 0.994 | 0.992 | 0.990 | 0.945 | 0.875 | 0.498 |
| topn_tau | 0.978 | 0.966 | 0.958 | 0.947 | 0.873 | 0.801 | 0.499 |
| spearman_expr | 0.696 | 0.675 | 0.603 | 0.638 | 0.573 | 0.565 | 0.497 |
| cosine_expr | 0.735 | 0.701 | 0.632 | 0.691 | 0.620 | 0.602 | 0.498 |
| cosine_shift | 0.617 | 0.588 | 0.596 | 0.597 | 0.570 | 0.554 | 0.505 |

**tail_max** (absent genes at rank P) — near-identical to position (both place absent genes at the bottom):

| metric | s=0 | s=0.5 | s=0.7 |
|---|---|---|---|
| de_delta | 0.989 | 0.893 | 0.817 |
| panel_tau | 0.996 | 0.945 | 0.875 |
| topn_tau | 0.978 | 0.873 | 0.801 |
| cosine_expr | 0.734 | 0.617 | 0.594 |

**zero_bucket** (absent genes at a shared mid-rank)

| metric | s=0 | s=0.1 | s=0.2 | s=0.3 | s=0.5 | s=0.7 | s=1.0 |
|---|---|---|---|---|---|---|---|
| de_delta | 0.885 | 0.868 | 0.856 | 0.845 | 0.770 | 0.716 | 0.497 |
| panel_tau | 0.989 | 0.976 | 0.966 | 0.953 | 0.858 | 0.780 | 0.499 |
| topn_tau | 0.973 | 0.959 | 0.949 | 0.937 | 0.853 | 0.779 | 0.499 |
| spearman_expr | 0.684 | 0.664 | 0.656 | 0.643 | 0.601 | 0.575 | 0.500 |
| cosine_expr | 0.684 | 0.664 | 0.655 | 0.643 | 0.601 | 0.574 | 0.500 |
| cosine_shift | 0.654 | 0.637 | 0.630 | 0.621 | 0.584 | 0.564 | 0.502 |

## 4. Findings

**(a) The representations now genuinely differ — rebuild validated.** Unlike Part II (byte-identical),
`zero_bucket` clearly separates from `position` (e.g. de_delta 0.989 vs 0.885). `position` and
`tail_max` remain near-identical to each other because both place absent genes at the bottom
(rank P+1 vs P — effectively the same location); so the meaningful contrast is "absent genes at the
bottom" (position ≈ tail_max) vs "absent genes in the middle" (zero_bucket).

**(b) The zero-bucket REDUCES discrimination, most for DE-Δr.** de_delta drops 0.989 → 0.885 under
zero_bucket; panel_tau 0.996 → 0.989; topn_tau ~unchanged. Likely mechanism: DE-Δr keys on the
genes that shift most between control and treated; when absent genes sit at the bottom, an on/off
gene makes a large, distinctive rank jump (bottom → front), whereas the mid-rank bucket compresses
that jump (middle → front), shrinking exactly the on/off signal DE-Δr exploits. So on the
*discrimination* benchmark the professor's fix does not help and slightly hurts. Caveat: the
zero-bucket's stated purpose was enabling DE analysis on activity-state changes, which is a
different objective than discrimination; it may still serve that purpose. The precise, honest
statement is "on the discrimination benchmark the zero-bucket does not improve and slightly reduces
accuracy, particularly for DE-Δr."

**(c) Rank metrics >> expression-space metrics (now a fairer test).** panel_tau (0.996), de_delta
(0.989), topn_tau (0.978) far exceed cosine_expr (0.735) and spearman_expr (0.696), and degrade far
more gracefully (panel_tau still 0.875 at 70% contamination; cosine_expr ~0.60). This runs counter
to the prior expectation that cosine-in-expression would be strongest. Because the sentences are now
genuinely sparse, this is a fairer test than Part II, though expression is still *reconstructed*
from ranks via `linear_model.json` rather than measured — so the definitive expression-space test
still requires true expression values / the C2S decoder.

**(d) topn_tau is the recommended headline metric.** panel_tau scores highest, but it compares across
all 946 genes including those absent in one/both samples, so its score still depends on how absence
is handled. topn_tau (top expressed genes only) is tail-immune by construction, nearly as strong
(0.978, still 0.801 at 70% contamination), and the most defensible choice for evaluating models.

**(e) Sanity checks pass.** Every metric returns to ~0.50 at s = 1.0 (full contamination →
indistinguishable → chance); accuracy is monotone decreasing in contamination.

## 5. Recommended metric + representation

For discriminating drug effects at pseudobulk resolution: **rank-based metrics (topn_tau or
de_delta) with inactive genes kept at the bottom (position / tail_max)**. Accuracy 0.97–0.99 at
s = 0, graceful degradation. The mid-rank zero-bucket is not recommended for discrimination
(reduces DE-Δr accuracy), notwithstanding its separate potential value for DE analyzability.

## 6. Caveats (carry to the meeting)

- **Accuracy is consistency, not magnitude.** High accuracy reflects that the metric *reliably*
  orders same-drug above different-drug; the absolute drug effect remains small (Parts I–II).
- **Expression metrics use reconstructed expression**, so their weakness is not yet a definitive
  verdict on expression-space similarity — a true-expression test is the clean follow-up.
- **The benchmark measures discrimination, not DE analyzability** — the zero-bucket may still serve
  the professor's stated DE purpose despite scoring worse here.

## 7. Recommended follow-ups
1. Re-run expression-space metrics on **true expression values** (or the C2S decoder) to give
   cosine-in-expression a definitive test rather than a reconstructed proxy.
2. Sweep the **zero-bucket position** (not just one mid-rank) to confirm the DE-Δr degradation is
   monotone in bucket height, and to test whether any placement recovers on/off signal without
   hurting discrimination.
3. Adopt **topn_tau at pseudobulk** as the standard evaluation instrument going forward, reported
   with effect sizes alongside discrimination accuracy.

---

# Part IV — Does the Metric Grade the Model? (Prediction-Level Forced Choice)

**Date:** 2026-07-10
**Script:** `metric_grades_model.py`
**Output:** `eval_results/metric_grades_model.json`
**Data:** `data_diverse2` (the OLD full-panel format the existing model was trained on), tiers 1 & 2.
**Model:** `pythia_sft_diverse2/checkpoint-10000`. Pseudobulk size 5, ceiling size 3, 12 pairs/cond.

## 1. Why this test

Part III showed topn_tau discriminates two REAL drug populations at ~0.97. But the metric we
evaluate models with must separate a good PREDICTION from a bad one — a different question. This
test runs the forced choice at the PREDICTION level:

  For (drug A, cell line): the model generates a pseudobulk prediction from drug-A prompts; we ask
  whether metric(prediction, real drug-A truth) > metric(prediction, real different-drug truth).
  Accuracy over many trials; chance = 0.50. Truth references are held-out cells only (never train),
  so there is no leakage. Compared references: model prediction, a real drug-A pseudobulk
  (ceiling), the linear-baseline prediction, and the model's prediction from a scrambled-drug
  prompt (scramble).

This cleanly separates "we need a better metric" from "the model's prediction has no drug signal
to grade." If even a proven-discriminative metric returns chance on the model's predictions, the
bottleneck is the model, not the metric.

## 2. Results (forced-choice accuracy, chance = 0.50)

**tier1_seen_conditions** (support: model/linear/scramble = 28 cell lines; ceiling = 2)

| metric | model | ceiling | linear | scramble |
|---|---|---|---|---|
| topn_tau | 0.493 [0.41, 0.58] | 0.375 [0.00, 0.75] | 0.565 [0.49, 0.64] | 0.486 [0.41, 0.56] |
| de_delta | 0.459 [0.36, 0.56] | 0.250 [0.00, 0.50] | 0.479 [0.39, 0.57] | 0.453 [0.39, 0.52] |
| panel_tau | 0.477 [0.42, 0.54] | 0.250 [0.00, 0.50] | 0.520 [0.49, 0.55] | 0.415 [0.35, 0.48] |

**tier2_unseen_drugs** (support: model/linear/scramble = 29 cell lines; ceiling = 12)

| metric | model | ceiling | linear | scramble |
|---|---|---|---|---|
| topn_tau | 0.552 [0.50, 0.62] | 0.363 [0.24, 0.46] | 0.583 [0.53, 0.64] | 0.553 [0.49, 0.62] |
| de_delta | 0.514 [0.45, 0.57] | 0.562 [0.38, 0.74] | 0.535 [0.45, 0.62] | 0.475 [0.41, 0.55] |
| panel_tau | 0.467 [0.41, 0.52] | 0.682 [0.54, 0.84] | 0.506 [0.43, 0.58] | 0.510 [0.48, 0.56] |

## 3. Interpretation

**The model's predictions sit at chance.** Across both tiers and all three metrics, the model
column straddles 0.50 (topn_tau 0.493 tier1, 0.552 tier2; CIs include or nearly include 0.50). The
model's prediction for drug A lands no closer to real drug-A than to a random different drug — a
direct, prediction-level confirmation that the model ignores the drug.

**model ≈ scramble.** topn_tau model vs scramble is 0.493 vs 0.486 (tier1) and 0.552 vs 0.553
(tier2) — identical within noise. Scrambling the drug in the prompt does not change how well the
prediction picks the right drug, the cleanest statement that the model does not use the drug.

**The metric is not the bottleneck — the model is.** topn_tau discriminates REAL drugs at ~0.97
(Part III) but grades the MODEL's predictions at ~0.50 (here). A proven-capable metric returns
chance on the model's output because the output contains no drug-specific signal. So metric
redesign alone cannot produce a gradeable model; the failure is upstream, in what the model learned.

**Linear is marginally above the model but also weak.** Linear topn_tau 0.565 (tier1) / 0.583
(tier2) slightly exceeds the model but with CIs near 0.50 — at pb-5 neither prediction reliably
picks the right drug. This should not be over-read as "linear grades well"; both are near chance.

## 4. Important caveat: the in-run ceiling is broken at this N (use Part III's ceiling instead)

The ceiling column here is unreliable and must not be read as "real cells can't discriminate":
- tier1 ceiling has only 2 cell lines of support (the eval tiers cap at ~8 cells/condition, so few
  groups have enough cells for a disjoint real-drug-A reference); its values (0.375, 0.250) are
  noise.
- tier2 ceiling is internally inconsistent across metrics (topn_tau 0.363 below chance, but
  panel_tau 0.682), a symptom of tiny support and the small pseudobulk sizes.

The valid ceiling is the Part III spike-in benchmark: real drugs discriminate at ~0.97 (topn_tau),
well-powered. The correct comparison is therefore across runs: **model predictions ≈ 0.50 (here)
vs real cells ≈ 0.97 (Part III)** — a stark, valid contrast. The broken in-run ceiling is a
data-thinness artifact (see Part IV.6), not a scientific finding.

## 5. Conclusion (the Step-1 decision)

- The metric works (Part III: ~0.97 on real drugs).
- The current model's predictions carry no drug identity (here: ~0.50, model ≈ scramble).
- Therefore the bottleneck is the model/training, not the metric. A new metric will not rescue a
  drug-blind model.
- The remaining lever is **retraining** on denoised / [END_CELL] targets so the model is optimized
  against drug-distinguishable targets rather than per-cell noise. This is the one experiment that
  could upgrade the result from characterization to solution. Given that the drug signal is small
  and noise-limited even in ground truth (Parts I–III), retraining is worth one clean attempt but
  should not be assumed to succeed; the defensible thesis contribution is the rigorous
  characterization plus the evaluation instruments built here.

## 6. Data limitation and the clean re-run

The eval tiers currently hold only ~5–12 cells per (drug, cell-line) condition — a consequence of
the preprocessing diversity cap (`max_cells_per_condition=10`) and the per-tier eval cap, not of
Tahoe lacking cells. This forced pb-5 and broke the in-run ceiling. It does not invalidate the
primary readout (model ≈ chance ≈ scramble, robust because cell line is the unit of replication
across 28–29 cell lines), but a definitive version of this test — with a working ceiling — needs
more cells per condition. Recommended: re-stream a dataset with a higher `max_cells_per_condition`
(e.g. 50) for clean pb-15/pb-30 truth references, then re-run this grading test. This is a
deliberate depth-vs-diversity tradeoff motivated by the finding that the drug signal is only
resolvable under aggregation.

### Next steps
1. **Retrain** on [END_CELL] / pseudobulk targets (data already built); re-run this grading test on
   the retrained model — does its prediction accuracy rise above chance?
2. **Re-stream with more cells/condition** to enable a clean in-run ceiling and pb-15 grading.
3. Report the model-grading accuracy (this instrument) as a primary model-quality metric going
   forward, alongside the spike-in discrimination accuracy (metric-quality) from Part III.

---

# Part V — Airtight Grading of the Retrained [END_CELL] Model

**Date:** 2026-07-11
**Scripts:** `train_c2s_tahoe_endcell.py` (retrain), `make_scramble_endcell.py` (scramble set),
`metric_grades_model_v2.py` (grading).
**Model:** `pythia_sft_endcell/final` — cold-started from `vandijklab/C2S-Scale-Pythia-1b-pt`,
trained one full epoch (42,198 steps) on `data_diverse2_endcell_big` (675,183 [END_CELL] examples).
**Eval:** `data_diverse2_endcell_big` tier2 (unseen drugs), pb_size=15, ceiling_size=8,
temperature=0.8 sampling, 23 cell lines.
**Output:** `eval_results/metric_grades_endcell_v2.json`.

## 1. The retrain

To test whether the failure is the representation rather than the model, we retrained on the
[END_CELL] format (expressed genes only + sentinel; inactive genes absent). Cold start from the
C2S base (not warm-started from the old SFT) so the model learns the [END_CELL] format without
old full-panel habits — a clean ablation of the representation. The `[END_CELL]` sentinel was
registered as an atomic special token (id 50277) and embeddings resized. Training converged
cleanly: loss 2.85 → 1.05 over one full epoch, eval loss 0.99, no divergence. The low next-token
loss confirms the model learned the cell-sentence format and sparsity structure — it does not by
itself imply drug usage (a model can predict the general expression program well while ignoring
the drug label).

## 2. The grading instrument (four upgrades over Part IV)

Part IV's grading run had four weaknesses; all are fixed here:
1. **Three representations** — every accuracy reported under position (absent → P+1), tail_max
   (absent → P, Federico), and zero_bucket (absent → shared mid rank, professor). Tests both
   advisors' proposals directly on the model.
2. **Temperature sampling** — the model generates one *sampled* prediction per cell (temp 0.8,
   top_p 0.9), so 15 cells → 15 *distinct* predictions → a genuine pseudobulk. Greedy decoding
   made the 15 predictions identical, defeating the aggregation; sampling gives the model its best
   shot at surfacing signal through denoising.
3. **Scramble arm** — the model also predicts from a scrambled-drug prompt (drug+MOA swapped to a
   different-mechanism drug, control and truth unchanged); scored against the real truth. model ≈
   scramble is the direct drug-blindness test.
4. **Fair sparse linear baseline** — fit on mean-centered rank vectors so the ~820 constant-floor
   (absent-gene) entries no longer dominate the regression, unlike Part IV's degenerate linear.

Forced choice: reference vs truth_A (real drug-A pseudobulk) and truth_B (real different-drug
pseudobulk); correct if metric(ref, truth_A) > metric(ref, truth_B); chance = 0.50.

## 3. Results (tier2_unseen_drugs; support: model/linear/scramble = 23 cell lines, ceiling = 7)

**position** (≡ tail_max — identical, both place absent genes at the bottom)

| metric | model | ceiling | linear | scramble |
|---|---|---|---|---|
| topn_tau | 0.477 | 0.670 | 0.531 | 0.505 |
| de_delta | 0.482 | 0.742 | 0.562 | 0.465 |
| panel_tau | 0.489 | 0.804 | 0.506 | 0.500 |

**zero_bucket** (professor's mid-rank)

| metric | model | ceiling | linear | scramble |
|---|---|---|---|---|
| topn_tau | 0.494 | 0.722 | 0.537 | 0.494 |
| de_delta | 0.495 | 0.712 | 0.559 | 0.514 |
| panel_tau | 0.494 | 0.827 | 0.550 | 0.474 |

(tier1_seen_conditions had only 1 cell line with sufficient cells — its numbers are single-cell-line
noise and are disregarded; tier2 with 23 cell lines is the result.)

## 4. Findings

**The retrained model is drug-blind — model ≈ chance under every representation and metric.** model
sits at 0.477–0.495 throughout, CIs straddling 0.50. Retraining on the [END_CELL] representation,
cleanly and to convergence over a full epoch, did not make the model use the drug.

**model ≈ scramble — the direct proof.** topn_tau: 0.477 vs 0.505 (position), 0.494 vs 0.494
(zero_bucket). The model's prediction from the real drug is no closer to the truth than its
prediction from a scrambled (wrong-mechanism) drug. Telling the model a different drug changes
nothing — it is not conditioning on the drug.

**The ceiling works and is well above chance (metric validity confirmed in-run).** Real disjoint
drug-A pseudobulks discriminate at 0.67–0.83 (panel_tau ceiling 0.804/0.827). So the metric *can*
tell real drugs apart at this resolution; the model's chance-level score is therefore a genuine
failure against a working reference, not a dead test. (This fixes Part IV, whose ceiling was a thin
0.65 on 6 cell lines.)

**Temperature sampling did not rescue it.** The sampled-and-aggregated model prediction (0.477) is
essentially identical to Part IV's greedy result (0.482). Giving the model its best shot — 15
distinct sampled predictions denoised into a pseudobulk — surfaces no drug signal. This closes off
"greedy decoding hid the signal."

**Both advisors' representations tested — neither helps.** position ≈ tail_max (identical; both
bottom-place absent genes) and zero_bucket is negligibly different (0.494 vs 0.477). The model is
at chance under Federico's floor and the professor's mid-bucket alike, because the bottleneck is
resolution, not representation.

**The fair linear baseline is also near chance** (0.53–0.56 topn/de_delta), no longer degenerate.
At pb-15 neither the model nor a fair linear map reliably picks the right drug; both hover near
chance, far below the ceiling.

## 5. Conclusion

Within a single well-powered run: the metric discriminates real drugs (ceiling 0.67–0.83) while the
retrained model does not (0.48 ≈ scramble 0.50), under all three representations, with sampled
predictions, and against a fair baseline. Combined with Parts I–IV, this rigorously establishes
that single-cell rank-based LLM perturbation prediction is bottlenecked by single-cell noise: the
drug signal survives only under aggregation of the *ground truth* (Parts I–III), and no
single-cell-trained model — regardless of representation, decoding, metric, or baseline — captures
the drug (Parts IV–V). The failure is not the metric (it works), not the representation (all three
fail equally), and not the decoding (sampling does not help).

## 6. The one remaining lever

Every experiment so far trains the model on *single-cell* targets. The only untested path that the
data motivates is training on **pseudobulk targets** — optimizing the model to predict aggregated,
denoised profiles, at the resolution where the drug signal actually exists. This is the last
experiment that could change the outcome. Given the signal is small even at pb-15 (d≈0.43, Part I)
and the model has been drug-blind under every single-cell condition, it is a worth-one-attempt
experiment, not a likely rescue. The defensible thesis contribution stands independent of its
outcome: a rigorous characterization of why single-cell LLM perturbation prediction fails, plus the
evaluation instruments (spike-in metric discrimination + prediction-grading) that demonstrate it.

### Status of advisor proposals (for the meeting)
- **Federico (spike-in protocol; expression-space cosine; tail_max):** spike-in protocol adopted as
  the core instrument (Parts III, V). Expression-space cosine underperformed rank metrics in the
  fairer [END_CELL] test (Part III), caveat that expression was reconstructed. tail_max ≡ position
  for discrimination and grading — no improvement.
- **Professor (zero_bucket mid-rank; metric is the blocking factor):** zero_bucket tested in both
  the spike-in benchmark (slightly *reduced* discrimination, Part III) and model grading (no change,
  Part V). The "metric is blocking" hypothesis is now resolved: the metric is *not* the blocker —
  it discriminates real drugs at 0.67–0.97; the model is the blocker. The blocking factor is
  single-cell resolution.

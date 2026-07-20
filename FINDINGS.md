# FINDINGS — C2S-Scale × Tahoe-100M drug-perturbation thesis

**This is the source of truth for results.** Every other MD is a detailed writeup or a draft; if they
disagree with this file, this file wins. Append here as results land — do not scatter numbers across
new documents.

**How to use it.** One entry per scientific *question*. Each entry is: **Q** (the question) →
**Why** (why it matters) → **How** (script/data) → **Answer** (the result, with numbers) →
**Status**. Status tags: ✅ done · ⏳ pending · 🔁 superseded. Newest evidence per question replaces
old; superseded numbers are struck, not deleted, so we never re-litigate.

---

## The one-paragraph thesis (the spine)

On single-cell cell-sentence perturbation prediction, (1) the standard **DE-Δr metric is saturated
by a control regression-to-the-mean artifact** — a zero-information baseline (every gene at the
middle rank) scores DE-Δr = 1.0, above the noise ceiling; on the un-confounded panel-τ the 1B-param
model, a linear map, and a no-fit baseline are all ~0.26, so the model has **no measurable
prediction advantage** over trivial baselines; and (2) the model is **drug-blind** — on instruments
that isolate the drug (discrimination, grading, scramble, output-invariance) it sits at chance,
equal to a scrambled-drug prompt. Mechanistically the drug **is** read into the representation
(decodable 76–82%) but the generation **ignores** it. The drug signal is real but noise-limited,
only resolvable under aggregation. Net: a deep single-cell LLM neither beats simple baselines on an
honest metric nor captures the drug — the single-cell, transformer extension of the DrEval critique.

---

## Canonical datasets / models (use these; everything else is historical)

- **Data:** `data_diverse2_endcell_big` (675k, [END_CELL] format, expressed genes + sentinel id 50277).
  Scramble set: `data_diverse2_endcell_big_scram` (tier1+tier2; `drug`=original, `scrambled_to_drug`=swapped, truth preserved).
- **Model:** `checkpoints/pythia_sft_endcell/final` (cold-started, full epoch). Base = `vandijklab/C2S-Scale-Pythia-1b-pt`.
- **Panel:** 946 genes (L1000 ∩ Tahoe). Absent-gene conventions: `worst` = rank P; `francesca` = fixed mid-rank P//2 = 473.
- 🔁 Historical: `data_diverse2` (full-panel) + `pythia_sft_diverse2/checkpoint-10000`. Do not report.

---

## Methods note — plate/batch confound control (read before quoting any discrimination number)

Drug and **plate** are partially confounded by Tahoe's design (each drug is assayed on its own
plate(s); ~4–12 drugs/plate, ~8 plates/cell line; controls are plate-matched). Our earlier
discrimination runs grouped by **cell line only**, so a "same drug" reference shared a plate with its
truth while "different drug" candidates sat on other plates — letting the **batch signature** stand
in for drug identity. We found this (a zero-drug-info control-copy scored NIR 0.766 cross-plate; a
pure-batch synthetic reproduced spike-in ≈0.93), added a **`--same_plate_only`** mode to every
discrimination instrument, and **re-ran everything within-plate**. Full audit, scripts, and A/B
tables: [`endcell/plate_control/README.md`](endcell/plate_control/README.md).

**All discrimination numbers below are within-plate unless marked 🔁.** The conclusions are unchanged
under plate control; some magnitudes shrink (leakage inflated magnitudes, never flipped a conclusion).

---

## Findings

### Q1. Is there drug-specific signal in the *data* at single-cell resolution?
- **Why:** separates "model can't capture the drug" from "no detectable drug signal exists" — different fixes.
- **How:** within-cell-line same-drug vs different-drug agreement + MOA positive control + dose/plate controls + pseudobulk sweep (`drug_specificity_in_data.py`).
- **Answer:** **No at single cell** — gap ≈ +0.002 (d≈0.01), different-drug agreement already sits at the replicate ceiling, and the MOA positive control fails. Signal **emerges only under aggregation** (topN-τ effect d 0.16→0.43 from pb5→pb15).
- **Status:** ✅ on old data · ⏳ [END_CELL] rerun (`gap_endcell.sbatch`) — confirm it ran; numbers expected to match.

### Q2. Does the metric actually discriminate real drugs (is the metric the bottleneck)?
- **Why:** if the metric can't tell real drugs apart, no model result is interpretable.
- **How:** spike-in forced-choice discrimination of two real drug populations (`spikein_metric_benchmark.py`).
- **Answer:** **The metric works** — rank metrics discriminate real drug populations at **~0.95–0.99** at pb15 (topn_tau/de_delta/panel_tau). The metric is not the bottleneck; the model is.
- **Status:** ✅

### Q3. Does the absent-gene convention matter (Federico worst-rank vs Francesca mid-rank)?
- **Why:** both advisors proposed a scheme; the thesis must show the choice doesn't change conclusions.
- **How:** spike-in with `zero_bucket_fixed` (P//2); and every CPU-eval number reported under both conventions.
- **Answer:** **No.** Spike-in DE-Δr 0.954 (francesca) vs 0.952 (position/tail_max). Noise ceiling & baselines differ by ≤0.01 between conventions everywhere (see Key Numbers). The earlier "zero_bucket hurts" claim used a buggy *variable* mid-rank (~127) and is 🔁 superseded.
- **Status:** ✅

### Q4. Is the retrained [END_CELL] model drug-blind?
- **Why:** the central claim.
- **How:** prediction-grading forced choice (`metric_grades_model_v2.py`); output-invariance (`output_invariance.py`); scramble-DE-Δr (`evaluate_endcell.py --mode scramble`).
- **Answer:** **Yes.** Grading ≈ **0.48 ≈ scramble** (ceiling 0.67–0.83). Output-invariance gap **0.000** [−0.019, +0.016] (job 581230). **Scramble-DE-Δr (job 583489, [END_CELL]): DE-Δr(real) ≈ DE-Δr(scramble)** — 0.740 vs 0.739 (tier1), 0.733 vs 0.729 (tier2). The near-ceiling DE-Δr is achieved *identically* with a scrambled wrong-mechanism drug → the score is drug-independent. This is "competent but blind" in the headline metric.
- **Status:** ✅ (grading, output-invariance, scramble-DE-Δr all agree)

### Q5. Is the drug-blindness a *rank-representation* artifact (does true expression carry it)?
- **Why:** cell sentences discard magnitude; maybe the drug lives there. Also de-risks the STATE comparison.
- **How:** same/diff discrimination on **true** normalized expression streamed from Tahoe, single-cell → pb15 (`expression_space_discrimination.py`).
- **Answer:** **No — representation exonerated.** True expression discriminates drugs at **~0.53 per cell**, the same near-chance regime as rank. Signal recovers under aggregation, strongest in the control-referenced shift (**cosine_shift 0.79 at pb15, d=0.87**). Combined one-table rank-vs-expression comparison on identical cells ⏳ pending (rerun after xet 403; `HF_HUB_DISABLE_XET=1`).
- **Status:** ✅ (core) · ⏳ (combined rank-vs-expr table)

### Q6. Does the drug enter the model's internal representation at all?
- **Why:** distinguishes "can't represent the drug" from "represents but doesn't use it."
- **How:** per-layer linear probe on residual-stream activations (`mechanistic_drug_probe.py`; writeup `dimensionality_probe_analysis.md`).
- **Answer:** **Yes.** Drug identity is linearly decodable at **82% (layer 9), 76% (layer 16)** in the [END_CELL] model (chance ~8%); original SFT peaks 74%, decays to 52%. Between/within variance ratio collapses ~3× in the final layers — the drug is present but geometrically de-emphasized toward the output.
- **Status:** ✅

### Q7. Is the drug information *causally used* in generation?
- **Why:** decodability ≠ functional use; turns "read but not used" from inference into proof.
- **How:** activation steering — inject the drug-A−drug-B direction into a drug-B forward pass; random-direction + sampling-noise controls (`causal_drug_probe.py`).
- **Answer:** ⚠️ **INCONCLUSIVE (job 583489).** `effect_toward_A ≈ 0` (−0.02 to +0.03 across layers/scales) — *but* the hook-sanity control **failed its bar**: `output_change_rand` (0.80–0.85) does **not** clearly exceed `noise_floor` (0.809), so even a random-direction steer barely moved the output beyond sampling noise. The intervention was too weak — the steering vector (`act_A − act_B`, a small drug-difference) is tiny vs the residual-stream norm, so scaling it ×1–8 barely perturbs generation. The `effect_toward_A ≈ 0` therefore can't be read as "drug inert." **Fix:** re-run scaling steering relative to the residual norm (e.g. `scale × ‖residual‖ × unit(dir)`) and higher scales, so `rand` clearly exceeds `noise` first.
- **Status:** ⚠️ re-run needed (methodology, not result)

### Q8. Is the model *competent* at perturbation prediction? → NO, and DE-Δr is a broken metric
- **Why:** "competent" was the nuance in "competent but blind" — it does not survive.
- **How:** DE-Δr + panel-τ vs the ceiling AND vs drug-agnostic baselines incl. two trivial no-fit ones (`evaluate_endcell.py --mode model,linear`).
- **Answer:** **The DE-Δr "competence" is mostly a control regression-to-mean ARTIFACT.** A zero-information baseline — every gene at rank P/2 (`revert_center`) — scores **DE-Δr(K50) = 1.000**, above the ceiling (0.76) and the model (0.73); inverted ordering (less info → higher DE-Δr) proves the artifact. Fix = **partial-DE** (control regressed out): the raw 0.95 drops to **~0.26** of *real but drug-AGNOSTIC generic-response skill*, matched by a no-fit baseline (revert_mean partial 0.25). partial-DE ≈ panel-τ ≈ 0.26 = the honest signal level. **The model's honest number (panel-τ 0.26) equals the drug-agnostic baselines → the LLM adds no measurable skill over a trivial predictor**, and the drug-specific component is ~0 (Q4). Response = control-reversion (artifact) + generic program (~0.26, trivially matched) + drug-specific (~0, nobody). Model partial-DE pending a GPU re-run (predicted ~0.26).
- **Status:** ✅ (competence RETRACTED; DE-Δr shown exploitable; honest metrics = panel-τ / partial-DE; LLM ≈ trivial baseline)

### Q9. What do the baselines say (drug-specificity vs control-conditioning)?
- **Why:** the earlier results.md over-claimed the per-MOA×cellline margin as "drug-specificity."
- **How:** mean-shift ladder (control/global/per-cellline/per-MOA/per-MOA×cellline), both conventions (`evaluate_endcell.py --mode baselines`).
- **Answer:** **Cell-line identity is the only informative grouping; MOA adds nothing over global.** (tier2/worst: global 0.056 ≈ moa 0.052 ≪ cellline 0.142 ≈ moa_cellline 0.120.) The model's margin over the toughest baseline reflects **control-conditioning (per-cell tailoring), not drug knowledge** — confirmed by Q4/Q5. Baselines are low in absolute terms because [END_CELL] DE genes are on/off-dominated and cell-specific; **anchor the model claim to the model/ceiling ratio, not absolute DE-Δr.**
- **Status:** ✅ (baselines) · ⏳ (model-vs-baseline margin needs the model number)

### Q10. Which metrics are even calibrated for this task? (Miller et al. DRF port)
- **Why:** the strongest steelman (Miller et al. 2025) argues single-cell nulls are metric miscalibration, not model failure. We must show our metric critique survives their calibration test — and use the metrics *their* framework endorses.
- **How:** Dynamic Range Fraction per metric at pseudobulk, stratified by cell line, on true expression: `DRF = [m(pos)−m(neg)]/[m(perfect)−m(neg)]`, pos = interpolated-duplicate noise ceiling, neg = mean baseline (and a stringent zero-info control) (`calibration_eval.py`, 25 cell lines).
- 🔁 **Answer (CROSS-PLATE — historical; superseded by the within-plate table below. Kept to show the method's evolution and the cell-count sensitivity):**

  | metric | DRF | m(neg, LOO mean) | m(pos, interp-dup ceiling) |
  |---|---|---|---|
  | weighted_r2 | −0.16 | 0.808 | 0.777 |
  | spearman_expr | −0.53 | 0.849 | 0.769 |
  | de_delta | **−0.92** | 0.864 | 0.740 |
  | panel_tau | −0.28 | 0.693 | 0.606 |
  | **nir** | **+0.64** | 0.458 | 0.805 |

  The leave-one-out fix barely moved the numbers (real cell lines have 20–204 drugs, so including the test drug was a <1% leak) — so **the inversion is a real property of the data, not the leak I first suspected.**
- **CORRECTED by the sensitivity run (~121 cells/drug, clean ceiling, DEG-pool capped at 400):**

  | metric | DRF @~50 cells | DRF @~121 cells | verdict |
  |---|---|---|---|
  | nir | +0.64 | **+0.80** | clearly calibrated (stronger with more cells) |
  | weighted_r2 | −0.16 | **+0.03** | flipped → inversion was ceiling noise; now ~neutral |
  | panel_tau | −0.28 | −0.23 | still inverted |
  | spearman_expr | −0.53 | −0.47 | still inverted |
  | de_delta | −0.92 | −0.52 | still inverted (real, some noise) |

- **WITHIN-PLATE re-run (`calibration_eval.py --same_plate_only`, groups by (cell_line, plate), 61 groups, 655 drugs, ~53 cells/drug):**

  | metric | DRF (within-plate) | m(neg) | m(pos) | verdict |
  |---|---|---|---|---|
  | **nir** | **+0.446** | 0.274 | 0.597 | only calibrated metric |
  | weighted_r2 | −0.081 | 0.793 | 0.776 | inverted |
  | panel_tau | −0.231 | 0.694 | 0.623 | inverted |
  | spearman_expr | −0.415 | 0.851 | 0.790 | inverted |
  | de_delta | −0.448 | 0.886 | 0.834 | inverted |

- **ROBUST claim (holds within-plate):** **NIR is the *only* calibrated metric** (sole positive DRF); all rank/correlation prediction metrics are inverted (they reward the generic gene ordering, not the drug) — even with the plate held constant. Model drug-blind on NIR (grading ≈ 0.48; within-plate model−scramble +0.014, CI spans 0).
- 🔁 **Superseded magnitude:** the cross-plate **+0.80 (@121 cells)** is replaced by the within-plate **+0.446 (@53 cells)**. Two effects: (a) fewer cells — NIR-DRF rises with cell count (cross-plate was +0.64 @~50 → +0.80 @~121), and (b) removed batch (~0.15–0.20 at matched ~50 cells). Still clearly positive/calibrated.
- **RETRACTED:** "even WMSE fails" — weighted_r2 is ~neutral (−0.08 within-plate, +0.03 cross-plate at high cells); not strongly inverted. Far weaker than Miller's genetic data, but not the clear failure the rank metrics are.
- **Status:** ✅ **SETTLED.** NIR is the only calibrated metric within-plate (+0.446, sole positive DRF); all rank prediction metrics uncalibrated. The +0.446 is a *conservative* value (53 cells; NIR-DRF rises with cell count, and the ceiling sweep in `drug_biology_atlas.py` confirms the noise ceiling climbs) — but the sign and the NIR-vs-prediction-metric contrast are unambiguous and do not depend on cell count, so no higher-n re-run is needed. `--same_plate_only` reproduction in `endcell/plate_control/`.

### Q11. Which drugs are even identifiable, and does correcting for drug difficulty rescue the model? (advisor's question)
- **Why:** every discrimination result assumes each drug induces a substantial, distinct change. The advisor flagged that this may not hold — some drugs may be inert, others near-duplicates — so the aggregate "model is at chance" could be an artifact of averaging over unwinnable drugs. This must be characterized before proposing new models.
- **How (methodology):** `drug_biology_atlas.py` — streams TRUE expression from Tahoe (24 shards × 250k rows → **median 44 cells/drug**, p10 32 / p90 85), grouped by **(cell_line, plate)** so every comparison is **within-plate** (batch/plate identity held constant; see the plate-control methods note). For each of **6,628 (drug × cell_line × plate)** conditions:
  - **Potency vs plate-matched DMSO** — a label-**permutation test**: is the real ‖pseudobulk(drug) − pseudobulk(DMSO)‖ larger than the null from shuffling drug/DMSO labels (p<0.05 = statistically active)? Plus **#DEG** = per-gene Welch t-test vs DMSO at **Benjamini-Hochberg FDR q<0.05**. **SNR** = effect ÷ replicate-noise (half-A vs half-B).
  - **Identifiability** — **same-plate ceiling NIR**: a real held-out replicate (half-B) ranked by Euclidean distance against every same-plate drug's truth (half-A); ≥0.8 = identifiable-in-principle.
  - **Redundancy** — **isolation** = nearest-other-drug distance ÷ replicate-noise; <1 = a plate-mate is closer than the drug's own replicate.
  - **MoA structure** — within- vs between-MoA pseudobulk distance. **Drug×cell-line interaction** — identifiability swing (max−min ceiling) for drugs seen in ≥3 lines. **Dose** — doses-per-drug distribution. **Cell-count sweep** — ceiling NIR vs sub-sampled cells/drug. Selftest validates recovery of planted inert / redundant-twin / distinct drugs.
- **Answer — the advisor's confound is REAL and now fully characterized:**

  | question | finding |
  |---|---|
  | do all drugs do something? | **No — 27.3% statistically inert** (indistinguishable from DMSO by permutation). Of the 72.7% "active", effects are **subtle**: median **0** DEGs (p90 5), SNR **0.75** (effect < replicate noise) → a faint *diffuse* shift, not strong gene changes. |
  | are drugs redundant? | **Yes — 78.7%** have a plate-mate closer than their own replicate; only **46.2% identifiable** (ceiling≥0.8), 34.1% at chance. |
  | does mechanism explain response? | **Barely** — within-MoA 2.322 vs between-MoA 2.377 (ratio **0.977 ≈ 1**). Same-MoA drugs are only marginally more similar than random pairs. |
  | is identifiability context-dependent? | **Strongly** — median cell-line **swing 0.83** (same drug: ceiling 0.0 in one line → 1.0 in another; e.g. Rapamycin vs Everolimus, both mTOR). Real target-dependency. |
  | dose design | **Real series** — 351 drugs, **62% have 2–3 doses** (hist {1:133, 2:122, 3:96}). Dose is a genuine variable we pool over; some "subtle" drugs may be low-dose. |
  | is the task winnable? | **Yes** — ceiling NIR rises with cells: 0.61 (n10) → 0.76 (n40) → **0.85 (n120)**. Signal is real, noise-limited. |

  Biological ranking is coherent (instrument validity): most identifiable = Paclitaxel, Encorafenib/Dabrafenib (RAF), Lapatinib (EGFR), Everolimus (mTOR) — potent targeted/cytotoxic agents; least = inert/nutrient-like conditions.
- **Does this rescue the model? No.** Stratifying by ceiling and grading only the identifiable subset (Q-stratify, within-plate): on identifiable drugs the model *appears* to beat the linear (0.768 vs 0.531, +0.237) **but this is not drug knowledge** — a zero-drug-info control-copy matches it (0.766) and the leak-immune scramble is null (model−scramble +0.014, CI [−0.016,+0.042]). The model is drug-blind **even where the task is provably winnable**. Correcting for drug difficulty sharpens the drug-blindness claim rather than overturning it.

### Q12. Is the drug signal learnable if the TARGET carries it? (Arm 1a — denoised/consensus targets)
- **Why:** the leading mechanistic hypothesis for drug-blindness was that the **single-cell target is information-starved about the drug** — at SNR ≈ 0.75 the drug moves a cell less than noise does, so next-token cross-entropy gets almost no gradient from drug-specific genes. If so, giving the model a *denoised* target should unlock it. This is the cheapest decisive test of the objective/target hypothesis.
- **How (methodology):** `build_consensus_targets.py` transforms `train.jsonl` into **consensus targets** — cells grouped by **(drug, cell_line, dose)** (dose kept separate: 62% of drugs are multi-dose), each group's cells averaged into one pseudobulk-derived cell sentence (genes ordered by mean expression, truncated to the group's median cell length, `[END_CELL]` appended). `--emit per_cell` keeps **every original prompt** (so the input is still an *individual* control cell) and swaps **only the response** → example count preserved exactly (hard-checked). Retrained **cold-start from the same base model with identical hyperparameters**, so the training target is the sole variable. Stopped at **checkpoint-25500 (~60% of an epoch)** on a flat loss (1.329). Evaluated with the leak-immune instruments: within-plate NIR + scramble + control-copy, and `output_invariance.py`.
- **Answer — NO. Denoising the target did not teach the model to use the drug.**

  | test | consensus model | single-cell model |
  |---|---|---|
  | **output-invariance gap** (topn_τ) | **+0.005** [−0.005, +0.016] | 0.000 [−0.019, +0.016] |
  | **output-invariance gap** (jaccard) | **+0.003** [−0.007, +0.012] | — |
  | NIR scramble (tier2, identifiable) | **−0.010** [−0.022, +0.003] | +0.011 [−0.018, +0.041] |

  Swapping the drug token changes the output **no more than resampling**. Both CIs span zero on both metrics; statistically indistinguishable from the single-cell model.
- **Behavioural changes that DID occur (training was not inert):** the consensus model **stops echoing the control** — on the identifiable subset it scores 0.653 vs the single-cell model's 0.771, where control-copy is 0.766 (the single-cell model sits *on* the control; the consensus model deviates from it). And its prediction geometry is **not** mode-collapsed: drug-drug distance spread CV ratio **1.054** (real 0.072 vs model 0.076) versus the single-cell model's **0.505**. So predictions vary realistically across drugs — but the variation is **uncorrelated with real drug identity** (Mantel ≈ +0.03). Not collapse; wrong structure.
- **Interpretation (hypothesis for the next arm):** denoising removed the *noise* but not the *weighting*. Because drug effects are subtle, consensus targets for different drugs **in the same cell line remain nearly identical** — they differ in a handful of genes out of hundreds in the sentence. Under next-token cross-entropy the loss is still dominated by the generic gene ordering, so the drug-specific difference stays a rounding error in the gradient **regardless of target quality**. If correct, no target-side fix can work; the objective must explicitly reward discrimination.
- **Caveats:** trained to ~60% of an epoch (loss plateaued, but not step-matched to the single-cell run — note that a *win* would have been strengthened by undertraining, whereas this null is weakened by it); **tier1 could not be scored by NIR at all** (n=0) because the tier1 eval set holds only ~2.4 cells per (cell_line, plate, drug) — far too sparse for per-condition pseudobulk, so the seen-drug NIR test remains unrun and the output-invariance test carried the learnability question.
- **Status:** ✅ target-side hypothesis **refuted**; drug-blindness is not attributable to single-cell target noise. ➡️ next lever is **objective-side** (contrastive / auxiliary discrimination loss, or GRPO with an NIR-shaped reward). Knowledge injection (Arm 2) deprioritised: adding drug information is unlikely to help a model that ignores the drug information it already has.

---

## Synthesis — the unifying principle (evaluate by discrimination, not absolute prediction)

The whole project collapses to one distinction:

- **Discrimination metrics** (Federico's spike-in, our forced-choice grading, NIR) ask *"is this profile
  closer to the RIGHT drug than to WRONG drugs?"* The generic response (the stress/cell-cycle program
  every drug triggers) is present in all drugs' truths equally, so it **cancels**, and only the
  drug-specific difference decides. These **isolate the drug** — and they are the calibrated metrics
  (NIR DRF +0.64; spike-in discriminates real drugs at 0.95–0.99).
- **Absolute prediction-quality metrics** (DE-Δr, WMSE, panel-τ) ask *"how close is the prediction to
  its OWN truth?"* The generic response is *most* of the truth, so a drug-agnostic mean scores high
  (DE-Δr 0.86; a zero-info mid-rank predictor scores 1.0). These are **dominated by the generic
  response** — exploitable/uncalibrated.

**These three instruments are one family** (discrimination) in three dialects — and they agree
(within-plate): **real drugs are discriminable at pseudobulk** (spike-in panel-τ ≈1.0/de_delta 0.99;
NIR DRF +0.446), **and the model is at chance** (grading 0.48; model−scramble +0.014). The apparent
tension "spike-in says DE-Δr works (0.99) but DRF says it
fails (−0.92)" is not a contradiction: DE-Δr has a *small but consistent* drug signal (a forced choice
reliably picks it → high accuracy) that is *tiny relative to the generic response* (so its absolute
score is saturated by the mean → fails calibration). The missing-gene representation debate
(worst/tail_max/fixed-mid-rank) is orthogonal — it only touches the rank-based absolute metrics, barely
changes them (≤0.004), and NIR lives in expression space so it sidesteps ranks entirely.

**Take-home:** the field-standard fix is to evaluate perturbation prediction by *discrimination*, and
on that axis this model captures nothing drug-specific.

## Key numbers (measured — do not re-derive)

**Noise ceiling, DE-Δr K50** (job `endcell_cpu`, 2026-07-13):

| tier | conv | cell-vs-cell | cell-vs-consensus | n_cond |
|---|---|---|---|---|
| tier1 seen | worst / francesca | 0.761 / 0.769 | 0.871 / 0.875 | 4325 |
| tier2 unseen drugs | worst / francesca | 0.758 / 0.764 | 0.910 / 0.913 | 2188 |
| tier3 unseen combos | worst / francesca | 0.757 / 0.768 | 0.938 / 0.939 | 99 |

**Baseline ladder, DE-Δr K50** (same job; `worst` convention; `francesca` within ≤0.01):

| tier | control | global | cellline | moa | moa×cellline |
|---|---|---|---|---|---|
| tier1 seen | NA* | 0.098 | 0.161 | 0.098 | 0.164 |
| tier2 unseen drugs | NA* | 0.056 | 0.142 | 0.052 | 0.120 |
| tier3 unseen combos | NA* | 0.051 | 0.107 | 0.063 | 0.097 |

\*control-as-prediction has an identically-zero shift → correlation undefined (expected).

**Model DE-Δr (K50) vs ceiling — [END_CELL], job 583489:**

| tier | model (worst) | model (francesca) | scramble (worst) | ceiling (cvc) | model/ceiling |
|---|---|---|---|---|---|
| tier1 seen | 0.740 | 0.751 | 0.739 | 0.761 | 0.97 |
| tier2 unseen drugs | 0.733 | 0.744 | 0.729 | 0.758 | 0.97 |
| tier3 unseen combos | 0.739 | 0.749 | — | 0.757 | 0.98 |

panel-τ ≈ 0.26 (convention-invariant by construction); DE-Δr(Spearman K50) ≈ Pearson (0.73–0.75). Validity (clean re-run): emits_end_cell 0.99, recall ~0.45, precision ~0.33, len_ratio ~1.7, hallucination 0.000.

**Drug-agnostic baselines vs model — DE-Δr is a control artifact; honest skill ~0.26 is generic (`--mode linear`):**

| predictor | info | DE-Δr K50 | partial-DE (ctrl removed) | panel-τ |
|---|---|---|---|---|
| revert_center (all genes → P/2) | none | **1.000** | NA (pure control) | NA |
| revert_mean (predict = mean control) | none (no fit) | 0.961 | 0.25 | 0.27 |
| ridge linear (control→shift) | control | 0.947 | 0.28 | 0.27 |
| noise ceiling (2 real cells) | — | 0.76 | — | — |
| **model (1B LLM)** | control+drug | **0.73** | *pending GPU (~0.26 exp.)* | 0.26 |

Reading: raw DE-Δr is inflated by control regression-to-mean (revert_center = 1.0, NA partial). Removing
control leaves ~0.26 of **real but drug-AGNOSTIC generic-response skill** — matched by a no-fit baseline
(revert_mean partial 0.25). partial-DE ≈ panel-τ ≈ 0.26 = the honest signal level. **Model panel-τ (0.26) =
these baselines → the LLM adds no measurable skill over a trivial drug-agnostic predictor.** Response
decomposes: control-reversion (artifact) + generic program (~0.26, drug-agnostic, trivially matched) +
drug-specific (~0, nobody). Reproduces tier1/2/3, both conventions. partial-DE is the fix for DE-Δr's exploit.

**NIR benchmark — model vs baselines on the calibrated metric (within-plate comparison sets, `nir_benchmark.py --same_plate_only`, tier2 unseen drugs, expr-NIR / Euclidean). Every comparison set is restricted to drugs on the same (cell_line, plate), so batch identity carries no drug information.**

Aggregate (n=606 drug×cell-line, ≥8 cells/drug):

| predictor | NIR |
|---|---|
| **ceiling** (real replicate) | 0.576 |
| model | 0.498 |
| linear (drug-agnostic) | 0.500 |
| control-copy (zero drug info) | 0.504 |
| mean | 0.180 |

Chance = 0.50. The model sits **at chance — equal to a drug-agnostic linear map and to a zero-information control-copy** → drug-blind on the calibrated metric. Real drugs are discriminable (ceiling > chance) and identifiability **rises steeply with aggregation** (per-drug ceiling headroom +0.046 → +0.170 as cells/drug go 4 → 20) → the task is real and noise-limited; the model captures none of it.

**Per-drug difficulty stratification (Q: which drugs are even identifiable?):**

| stratum | n | ceiling | model | linear |
|---|---|---|---|---|
| unwinnable (ceiling < 0.6) | 296 | 0.287 | 0.286 | 0.475 |
| marginal (0.6–0.8) | 106 | 0.689 | 0.569 | 0.508 |
| identifiable (ceiling ≥ 0.8) | 204 | 0.938 | 0.768 | 0.531 |

~⅓ of drug×cell-line pairs are identifiable-in-principle; the rest are inert/redundant and unwinnable by *any* predictor. Drug ranking is biologically coherent: potent = pemetrexed, crizotinib, irinotecan (ceiling 1.0); inert = adenine, folic acid, allantoin, vitamin K4 (ceiling 0.0). Batch/nutrient compounds correctly land at the bottom.

**Causal test — does the model USE the drug? (identifiable subset, n=204 drugs, 40 cell lines, clustered CIs):**

| comparison | value | 95% CI (cluster by cell line) | reading |
|---|---|---|---|
| model − linear | +0.237 | [+0.190, +0.286] | confounded — control-conditioning, NOT isolated drug use |
| control-copy (zero drug info) | 0.766 | ≈ model (0.768) | a predictor with no drug information matches the model |
| **model − scramble** (same control, wrong drug token) | **+0.014** | **[−0.016, +0.042]** | **NULL** |

The scramble arm is the decisive control-matched manipulation: identical control cell, only the drug token in the prompt changes. It is **null** → **swapping the drug changes the model's output by nothing → the model does not use the drug**, even on drugs that are provably identifiable. `model − linear` looks positive only because the model retains control/batch structure that the drug-agnostic ridge smooths away — a zero-drug-info control-copy (0.766) matches the model. Consistent with grading 0.48 and output-invariance 0.000. **Anchor the drug-use claim to `model − scramble`, never to `model − linear`.**

**Other established numbers (leak-immune tests):** grading model ≈0.48 ≈ scramble, ceiling 0.67–0.83; output-invariance gap 0.000 [−0.019,+0.016]; true-expression per-cell discrimination ~0.53, cosine_shift 0.79/d0.87 (pb15); probe decodability 82%/76% (layers 9/16); drug-drug geometry Mantel ≈ 0.05, model CV/real CV ≈ 0.50 (predictions collapse toward one profile).

**Spike-in — the metric separates real drug populations (within-plate comparison sets, `spikein_metric_benchmark.py`, pb15, tail_max, 60 (cell_line×plate) groups, spike=0):** panel-τ **1.000**, de_delta **0.995**, topn-τ **0.987** (CI on panel-τ [0.99994, 0.99998]). Titrates cleanly to 0.50 at spike=1.0. Discrimination is essentially perfect with the plate held constant across candidates → **real drugs are separable and the metric works; this capability is genuine drug signal, not batch.** (Saturated near ceiling, so it certifies the metric *can* discriminate real drug populations, not the finer model-vs-baseline question — that is NIR/DRF's job.)

**DRF within-plate — ✅ settled:** NIR is the only calibrated metric (**+0.446** @53 cells; all prediction metrics inverted); the sign/contrast is cell-count-independent, so no higher-n re-run needed. See Q10.

---

## Pending queue (updated 2026-07-14)

1. ✅ **GPU job** (583489) done — Q8 competent ✅, Q4 scramble-DE-Δr ✅, Q7 causal ⚠️ inconclusive.
2. ⚠️ **Causal probe re-run (Q7)** — scale steering relative to the residual norm so `rand` clears `noise` first.
3. ⚠️ **Eval validity hygiene re-run (Q8)** — eos=`[END_CELL]` (or decode-without-skip + truncate) for trustworthy recall/precision. Does not affect DE-Δr.
4. ⏳ **Combined rank-vs-expression** (Q5) — `expr_rerun.sbatch` (currently PENDING, QOSMaxJobsPerUserLimit).
5. ⏳ **[END_CELL] gap analysis** (Q1) — confirm `gap_endcell.sbatch` ran.
6. **Docs rewrite + advisor message** — reframe around "read but not used"; the scramble-DE-Δr line is the headline.
7. **Next phase:** task ceiling → PubChem injection → GRPO/aux-loss (does forcing the objective make it use the drug?) → STATE/scGPT generality.

## Detailed writeups (in `docs/`, organized by regime — see `docs/README.md`)
- **`docs/endcell/`** (current): `drug_specificity_analysis_writeup.md` (spine, Parts I–V) · `part6_expression_space_draft.md` (rank-vs-expr) · `dimensionality_probe_analysis.md` (probe)
- **`docs/legacy_l1000/`** (historical/background): `results.md` (full-panel eval — superseded)
- **`docs/methods/`**: `dataset_construction.md` (pipeline, panel, tiers)
- **`docs/proposals/`**: `pubchem_drug_injection_spec.md` (next phase)

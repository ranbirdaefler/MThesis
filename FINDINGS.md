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
- **Answer (corrected run, leave-one-out baseline, 25 cell lines):**

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

- **ROBUST claims:** **NIR is the calibrated metric (+0.80)** — mean baseline at chance (0.41) *by construction* (same profile for every drug → cannot discriminate, robust to denoising), ceiling 0.88. **Rank-based prediction metrics (DE-Δr, spearman, panel-τ) are genuinely uncalibrated** (inverted even with a clean ceiling → they reward the generic gene ordering). Model drug-blind on NIR (grading ≈ 0.48).
- **RETRACTED:** "even WMSE fails" — weighted_r2 flipped to +0.03 with the clean ceiling, so its inversion was the denoising artifact. WMSE is ~neutral/marginally calibrated here (far weaker than Miller's genetic data, but not inverted).
- **Minor caveat:** DEG test still flags ~116/946 genes even capped (plausibly real, mildly over-powered); rank-metric inversions retain a possible residual denoising component but are robust to a 2.4× cleaner ceiling. Pseudobulk de_delta is the systematic-variation critique, DISTINCT from the single-cell `revert_center`=1.0 exploit (Q8).
- **Status:** ✅ NIR calibrated (+0.80); rank prediction metrics uncalibrated; WMSE neutral. Half-2 NIR model benchmark next.

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

**These three instruments are one family** (discrimination) in three dialects — and they agree: **real
drugs are discriminable at pseudobulk** (spike-in 0.95–0.99; NIR ceiling 0.80), **and the model is
at chance** (grading 0.48). The apparent tension "spike-in says DE-Δr works (0.95) but DRF says it
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

**NIR benchmark — model vs baselines on the calibrated metric (`nir_benchmark.py`, held-out tiers, expr-NIR / Euclidean):**

| predictor | tier2 unseen | tier3 combos | tier4 dose |
|---|---|---|---|
| **ceiling** (real replicate) | 0.69 | 0.80 | 0.63 |
| model | 0.52 | 0.51 | 0.51 |
| linear (drug-agnostic) | 0.50 | 0.50 | 0.50 |
| mean | 0.32 | 0.09 | 0.29 |

Chance = 0.50. Real drugs are identifiable at pseudobulk (ceiling 0.63–0.80); the model is at chance, **equal to a drug-agnostic linear map** → drug-blind on the calibrated metric, on a WINNABLE task. rank-NIR agrees on the ordering (ceiling > model ≈ linear), noisier on 7-cell halves. Ceiling understated by thin-tier halves (calibration ceiling was 0.88). (Fixed a scoring bug: truths must be at consistent denoising — all half-A, ceiling = disjoint half-B — else the ceiling inverts.)

**Other established numbers:** spike-in real-drug discrimination ~0.95–0.99 (pb15); grading model ≈0.48 ≈ scramble, ceiling 0.67–0.83; output-invariance gap 0.000 [−0.019,+0.016]; true-expression per-cell discrimination ~0.53, cosine_shift 0.79/d0.87 (pb15); probe decodability 82%/76% (layers 9/16).

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

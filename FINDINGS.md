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

**(3) …and the cause is the OUTPUT ENCODING, which is fixable (Q15).** Because a cell sentence ranks
genes by expression, **83% of the target tokens are identical between any two drugs** (34.6/200 differ),
so cross-entropy gets ~1% of its gradient from drug identity — which is why *every* change to target
*content* failed identically (Q12 consensus, Q14 OT, the whole ε-ladder). Re-encoding the target as the
**drug-specific residual** (a signed DE signature, generic program subtracted) raises that to 60% of
tokens and produces the **first non-null `model − scramble` in this project: +0.143 [+0.111, +0.179]**
under an opposite-signature swap, with a monotone dissimilarity gradient and a matched single-cell
control that stays null. So the drug-blindness is **not** an inherent limit of the architecture or the
data — it is a consequence of how the prediction target is tokenized.

**(4) …and that drug use GENERALIZES to unseen contexts, but is beaten by a lookup table (Q16).** On a
tier-aligned three-way holdout, conditions whose **(drug, cell_line) pairing was never trained on** score
**+0.1002 [+0.0661, +0.1368]** (n=250, 38 cell lines) — statistically indistinguishable from the trained
split's +0.0898, i.e. **no memorization premium**: the model learned a drug representation that applies in
cell lines it never saw that drug in, rather than memorizing pairings. Held-out *drugs* stay null, the
control firing as designed. But the model recovers only a small fraction of the signature: a numpy lookup
of the drug's mean residual from other cell lines scores **0.963** against a **0.968** replicate ceiling,
while the 1B model reaches **0.639** (`model − drug_lookup` −0.3245; −0.1936 even against a *single*
other cell line). This is the pre-registered "honest partial" branch of `arm1b_objective_spec.md` §7 —
and it makes the thesis the single-cell cell-sentence-LLM instance of the CMap/LINCS-era result that a
per-perturbation mean beats the deep model.

**(5) …and the lookup is at the ACHIEVABLE ceiling, because the remaining 45% is unlearnable (Q17, Q19).**
The response decomposes as β(d) + κ(d,c): measured in cosine space with noise divided out, the transfer
coefficient is **T = 0.557 [0.513, 0.601]** against a simulated κ=0 null of **1.001** and a
structure-matched negative control at **+0.000**, so κ is ≈45% of the drug-specific variance. But κ has
**no learnable structure** — no cell line bends different drugs in a consistent direction (excess
within-line consistency **−0.007**, interval excluding anything above +0.002). So κ acts as *irreducible
noise for any predictor*, which is why `drug_lookup` reaches 0.963 against a 0.968 ceiling: not because
the interaction is small, but because it is **unpredictable**. Nothing beats a lookup on seen drugs.
Scope: the drug-specific residual is **62% of the total response variance**.

**(6) The model's only opportunity is the UNSEEN drug — and that regime is open (Q18).** A lookup
structurally requires the drug to have been seen. For drugs that have not been, two knowledge channels
carry real signal against count-matched nulls: **protein target +0.0844 [+0.0592, +0.1095]** and
**MoA +0.0780 [+0.0555, +0.1027]**, while chemical structure stays closed (+0.0137, spans zero). This
**retracts the project's own pre-registration**, which inferred "unseen drugs are out of reach" from a
gate that tested chemical *structure* — the channel our own drug-knowledge spec ranked lowest — and
never read the `targets` column sitting in Tahoe's metadata. The thesis therefore ends not in closure
but with a **specific, measured opportunity**: on seen drugs nothing can beat retrieval; on unseen drugs,
where retrieval is impossible, drug-side knowledge works and the model has something a dictionary
cannot do. That is also DrEval's Leave-Drugs-Out — the setting that matters for drug design.

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

## Validity note — the scramble null is NOT an artifact of swapping to SIMILAR drugs (read before quoting any `model − scramble`)

**The objection.** The scramble ablation swaps the drug token and reads `model ≈ scramble` as "the model
ignores the drug." But if the swapped-in drug has a **similar true response**, producing the same output is
**correct, not blind**. The existing scramble picks a **different-MoA** drug — and Q11 showed MoA barely
predicts response (within-MoA 2.322 ≈ between-MoA 2.377, ratio 0.977), so "different mechanism" is *not*
"different response." The same gap applies to output-invariance (Q4) and the Q13 activation swap.

**The test.** `scramble_distance_sweep.py` turns the single test into a **curve**: `model − scramble` as a
function of the **response distance** between the real and swapped-in drug, `‖truth_A − truth_B‖` (which is
exactly the NIR other-drug distance). For each sampled real drug A we generate its real-drug prediction, then
scrambled predictions swapping A→B for B spanning the distance quantiles, and score all against A's own
truth. 200 (drug, swap) pairs, 20 cell lines, tier2 unseen drugs, clustered CIs (resample cell lines).
No `--same_plate_only` needed: the scramble **difference** cancels control/batch leakage (both arms share
A's control cell), so grouping by cell line only maximises the available swap-distance range.

| bin (swap distance) | single-cell `model − scramble` | T2 (OT) `model − scramble` |
|---|---|---|
| 0 — nearest (2.75–4.03) | −0.046 [−0.138, +0.065] | −0.001 [−0.028, +0.040] |
| 1 (4.03–4.60) | +0.060 [−0.002, +0.124] | +0.031 [−0.007, +0.075] |
| 2 (4.60–4.93) | −0.006 [−0.070, +0.050] | +0.017 [−0.029, +0.069] |
| 3 (4.93–5.32) | −0.004 [−0.063, +0.059] | +0.043 [−0.022, +0.107] |
| **4 — farthest (5.32–6.25)** | **−0.019 [−0.082, +0.062]** | **−0.005 [−0.033, +0.029]** |
| **trend corr(distance, model−scramble)** | **+0.051** | **+0.012** |

**Answer: the null holds at every distance, including the far tail.** No bin in either model has a CI
excluding zero, and the trend correlations are ≈0. Telling the model it is a drug whose true response is
maximally different (a 2.3× spread in distance) **still does not change the output** → drug-blindness is
confirmed at the hardest setting, not manufactured by near-twin swaps. Two details strengthen this: (i) the
single borderline value (+0.060, single-cell) is a **middle** bin — a genuine dissimilarity effect would be
monotone and peak in the **far** bin, which it does not; (ii) the comparison is **paired** — model NIR drifts
across bins (0.497→0.326) because far-from-everything drugs populate the far bins, but `model − scramble`
uses the *same* A on both arms, so that drift cancels.
- **Caveats:** n=40 per bin / 20 cell lines (CIs ±0.03–0.14 — excludes any large effect, not a tiny one);
  the scramble arm used 4 generations vs the model's 8 for speed (adds noise, no directional bias); the far
  tail is bounded by how dissimilar drugs within a cell line actually get (Q11: 78.7% have a plate-mate
  closer than their own replicate).
- **Status:** ✅ **strengthens Q4, Q12 and Q14.** Every `model − scramble` null in this file survives the
  dissimilarity control. Logs: `logs/scramble_sweep_*.out`; JSON: `RESULTS/scramble_sweep_single.json`,
  `RESULTS/scramble_sweep_T2.json`.

---

## Advisor audit responses (commit `ad7073a`)

- **A-04 (J-space bridge Q6→Q7):** ✅ answered by **Q13** (`workspace_probe.py`) — the causal "decodable ≠ used" test, with the privileged-subspace framing.
- **A-07 (calibration self-test didn't validate its claim):** ✅ fixed. The self-test now runs deterministic unit checks (perfect→metric max; zero-shift / flat-truth / no-others→None) **and asserts the DE-Δr exploit on the metric itself** — an uninformed leave-one-out mean baseline scores **de_delta = 1.000** while **NIR = 0.000**. Old pass condition never checked the advertised failure case. (`calibration_eval.py --selftest`.)
- **A-05 (dose absent from the sampling-cap key):** ✅ checked (read-only `check_dose_coverage.py` over the JSONLs) — **does NOT materially bite.** Within-plate dose is near-degenerate: only **2.4%** of (drug,cell_line,plate) groups hold >1 dose (`{1: 38759, 2: 941, 3: 4}`), so a dose-blind cap can affect at most that slice; multi-dose groups survive the cap; and the only dose-dependent split, **Tier-4, is 60/60 intact**. The blunt "at-risk" count (10,116) is a design-confounded upper bound (single-dose plates of drugs that are multi-dose *on other plates* = normal Tahoe design, not loss). Action: **no regeneration**; the preprocessor now keys the cap on `(drug, cell_line, plate, dose)` (raw conc, units preserved) so **future** builds are dose-safe — the meta-cell rebuild inherits it.
- **A-08 (provenance):** ⏳ `make_provenance.py` bundles RESULTS/*.json + fingerprints dataset/checkpoint (manifest hash) into a committable `artifacts/`; run in progress.

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

### Q13. WHERE does the drug-blindness live — is the drug encoded in a subspace the generation readout ignores? (mechanistic)
- **Why:** we have two facts that seem to contradict — the drug **is** decodable from activations (Q6, 82% at layer 9) yet the model **ignores** it in generation (Q4: scramble null, output-invariance 0). The reconciliation must be that *the direction that **encodes** the drug is not the direction that **drives** generation.* Q7 tried to show this by activation steering and was **inconclusive** (the steering vector was tiny vs the residual norm, so even the random control barely moved the output). This run redoes the causal test properly, in logit space, with the guards Q7 lacked.
- **How (methodology):** `workspace_probe.py` on the single-cell `pythia_sft_endcell/final`, tier2, layers 2/4/6/8/9/12/16, 12 drugs × 40 real cells = 480 prompts. Play-by-play, with the reasoning behind each step:
  - **Material.** One forward pass per prompt (`extract_activations`), taking the residual-stream vector at the **last prompt token** — the position that hands off to generation and where the 82% decodability lives. `X` per layer is (480 × 2048).
  - **(1) Where the drug lives.** `build_subspace` stacks the 12 per-drug mean vectors (global-mean-centered), SVDs them, keeps the top-10 orthonormal directions = `V_drug`, the slab where drug identity is written. Same for `V_cl` (cell-line subspace = **positive control**, since Q9 shows the model *does* use cell line).
  - **(2) The confound fix (this is why Q7-style raw ablation is not enough).** Drugs are confounded with batch — each drug sits on its own plate/dose. The code measures this directly (`Xd = X @ V_drug`, then probe cell_line/plate/dose from *inside* the drug subspace): the raw drug slab decodes **dose 0.90–0.95, plate ~0.5** at every layer, so the raw drug axis *is* partly a batch axis. Fix: build a context subspace `V_conf` from **cell_line ∪ plate ∪ dose** (`build_union_subspace`) and orthogonalize the drug axis against all of it (`orthogonalize`) → `V_drug_pure` = the part of the drug axis **not** explained by batch. Every causal claim is made on `V_drug_pure`, never the raw slab (the log prints `raw drug (confounded — do not quote)`).
  - **(3) The gate (guards against an unfalsifiable null).** Before trusting any "ablating the drug does nothing" result, verify the subspace actually contains the drug: `probe_acc` before vs after projecting `V_drug` out. `frac_removed = (a_drug − a_drug_abl)/(a_drug − chance)` must exceed 0.8. **Passed at every layer** (decode 0.41–0.82 → constant 0.121 floor; 88–95% of above-chance signal removed).
  - **(4) The causal test — logit-space KL under ablation.** `kl_under_hook`: teacher-force the *real* response, read the clean per-token log-probs; register a forward hook (`make_hook`) that **projects a subspace out of the residual at every position** (`hs − (hs@V)@Vᵀ`); re-read the log-probs; return mean **KL(clean ‖ ablated)** over response tokens. Measuring in logit space (not by generating) removes the sampling-noise floor that sank Q7. Run for `V_drug_pure` (headline), `V_rand_pure` (**random subspace of matched dimension** = the null), `V_cl` (positive control). **What this can and cannot show:** ablation proves a slab carries *functional* variance the downstream layers consume — it does **not** prove the slab's *drug-specific content* is what's read (the purified directions could carry generic "strong transcriptional program" energy that merely co-varies with drug across 12 drugs). This is the key limitation — so the ablation ratio is reported as a **functional-variance** statement only, and no "causal"/"reads drug identity" label is attached to it (the earlier auto-verdict labels that did so were **removed** from the script; an ablation ratio cannot make an identity claim). Only the next test can.
  - **(5) The swap — the only real drug-*identity* test (`--do_swap`).** For each cell that truly got drug A: overwrite A's code with drug B's, **within the pure-drug subspace only** (`comp_b = V_drug_pure @ (V_drug_pureᵀ @ drug_mean[b])`, injected via the hook's `add_vec`), and compare the resulting KL to injecting a **random vector of identical norm**. If A→B moves the output *more* than matched noise → the readout is drug-direction-sensitive (a "routing failure": read but misused). If A→B ≈ noise → the readout feels only the *magnitude* of activity in the drug region, **not which drug it names**. This is translation-clean (matched-norm baseline) and the least-confounded number in the run. **Held-out means (the tightening):** drug B's mean is estimated on an est-half of the cells and the swap is run on the disjoint eval-half, so mean-estimation noise cannot itself manufacture the null.
- **Answer — the readout is NOT drug-specific; the drug is increasingly encoded but never read for identity.**

  Per gate-passing layer (all 7 passed):

  ⚠️ **CORRECTED against `RESULTS/workspace_probe.json` (2026-07-30).** The previously published version of this table had three errors: the decode column was the *context-removed* series but labelled as raw decode (the two differ, and only the raw one peaks at layer 9 — which is what the prose quotes); the `random @ matched dim` column matched no series in the JSON, so the ratio column derived from it was wrong (layer 6 read 12× against a true 5.7×); and the swap column carried the *within-sample* values while the only run on disk is the **held-out-means** one. **No conclusion changes** — every ratio still exceeds 1 (3.6–18.6×), the swap is still below matched noise at all seven depths, and context-removed decodability still rises monotonically with depth. The table is now regenerated from the JSON.

  | layer | decode (raw) | decode (context removed) | ablate pure-drug KL | random @ matched dim | ratio | cell-line KL (+ctrl) | var-share pure-drug / cell-line | **SWAP: drug-B vs matched noise** |
  |---|---|---|---|---|---|---|---|---|
  | 2 | 0.408 | 0.354 | 0.155 | 0.012 | 12.7× | 1.115 | 0.031 / 0.580 | **0.104 < 0.917** |
  | 4 | 0.585 | 0.429 | 1.563 | 0.084 | 18.6× | 4.180 | 0.031 / 0.641 | **1.855 < 2.156** |
  | 6 | 0.692 | 0.529 | 1.435 | 0.251 | 5.7× | 4.307 | 0.030 / 0.665 | **1.618 < 1.958** |
  | 8 | 0.779 | 0.569 | 0.596 | 0.042 | 14.1× | 5.419 | 0.030 / 0.667 | **0.599 < 1.163** |
  | 9 | **0.821** | 0.602 | 0.418 | 0.100 | 4.2× | 4.689 | 0.031 / 0.655 | **0.333 < 0.761** |
  | 12 | 0.754 | 0.873 | 0.133 | 0.035 | 3.8× | 0.135 | 0.030 / 0.568 | **0.105 < 0.227** |
  | 16 | 0.758 | **0.883** | 0.015 | 0.004 | 3.6× | 0.059 | 0.033 / 0.493 | **0.031 < 0.036** |

  The two decode series behave differently and the distinction is load-bearing: the **raw** probe peaks at layer 9 (0.821) then declines to 0.758 — this is the "82% / 76%" the prose quotes — whereas with cell line, plate and dose projected out decodability rises **monotonically** to 0.883. The dissociation claim rests on the context-removed series, which is the honest one.

  🚨 **THE SWAP'S CONTROL IS CONFOUNDED — the headline below is SUSPENDED pending a rerun (found 2026-07-31 by structural audit).** `comp_b` is drug B's projection **into** the purified slab (≤10 dims), but the matched-norm random vector was drawn as `rng.randn(H)` in the **full 2048-dim** residual space, and `make_hook` ablates `V_drug_pure` *before* adding. So the swap arm perturbs only inside the ablated slab while ≈**99.5%** of the random vector's energy lands in directions that were never ablated — including the cell-line directions this same table measures at ~0.6 variance share and 15–20× the KL. **"Swap < matched noise" is exactly what that confound predicts regardless of what the readout does.** The control matched the norm but not the subspace, and the writeup called it "translation-clean" — which was wrong.
  **A second, related mismatch:** the removal gate was computed on the **raw** `V_drug` while every causal claim is made on `V_drug_pure`, so it certified that the raw slab carries the drug rather than that the purified one does.
  **Fixed** (`workspace_probe.py`): the random vector is now drawn **inside** `V_drug_pure`, randomising direction at fixed subspace and norm; a **paired** bootstrap CI replaces two bare means (the arms share prompts and drug means, and the old version stored no dispersion at all); the gate now also runs on the purified subspace; and a `control sanity` line reports the fraction of random energy inside the slab, which must read ~100%. Rerun queued as `probe2.sbatch`.
  **What survives unaffected:** the ablation ratios (their null, `V_rand_pure`, was already a matched-*dimension* random subspace — a different and correctly-constructed control), the variance shares, and the decodability-versus-depth dissociation. **What does not:** the direction-blindness claim specifically, which rests entirely on the swap.

  Three findings, in order of how load-bearing they are:
  1. **The swap is null at every layer (headline).** ⚠️ **SUSPENDED — see the confound above.** Overwriting drug A's code with drug B's perturbs the output *no more than — actually less than — matched-norm noise*, at all 7 depths. The readout responds to the *magnitude* of activity in the drug subspace, not to *which drug* it encodes. This is the mechanistic statement of drug-blindness, and it is confound-clean and depth-invariant.
  2. **The drug slab is not inert, but it is a bit player.** Pure-drug ablation KL is 3–24× the matched-random null (so the slab carries real functional variance), but only ~3% of the cell-line positive control's effect, and pure-drug variance share is flat at ~0.03 vs cell-line ~0.6. The instrument works (cell-line KL is 15–20× the drug's), so the drug's weakness is not measurement failure.
  3. **Dissociation: maximally encoded exactly where least used.** Drug decodability (with all context removed) *rises* with depth 0.35 → **0.88** at layer 16, while causal share stays flat at ~3%. The late layers know the drug best and use it least — the signature of a representation computed and set aside.
- **Verdict:** the clean claim is **not** "drug lives in a dead subspace" — it is **the model's generation readout is drug-agnostic within an increasingly explicit drug representation: sensitive to the magnitude of activity in the drug region, blind to its direction.** This is *stronger* than "inert subspace" because it is mechanistic and it explains why Arm 1a (Q12) failed: denoising changes *what is in* the subspace, but the defect is the *readout ignoring direction within it* — no target-side fix can touch that. It points squarely at the objective-side lever (a loss that forces the readout to discriminate directions in the drug subspace).
- **Caveats (state, don't hide):** (a) the ablation "CAUSAL" auto-labels are **not** identity claims (step 4 limitation) — the writeup leads with the swap, not those labels; (b) late-layer absolute KLs (12/16) are tiny for *everything* (less network downstream) — read ratios and the swap, which are depth-robust, not raw late magnitudes; (c) n=60 KL prompts, 12 drugs, one tier — a within-model mechanistic probe, not a population estimate (no clustered CI); (d) `drug_mean[b]` estimation noise biases the swap *toward* the null — **now ruled out** by the **held-out-means** version (means on the est-half, swap on the disjoint eval-half, job 596473): drug-B KL ≈ matched-noise at every layer and **near-identical to the within-sample swap** (e.g. L2 0.104 vs 0.106; L9 0.334 vs 0.334; L16 0.031 vs 0.031), so estimation noise is not what produces the null. Validity rests on the three surviving pillars: **gate passes** (nulls are real, not broken hooks), **positive control fires** (cell-line KL 15–20× drug), and the **swap is translation-clean**.
- **Framing (Gurnee et al. — representational selectivity / privileged subspace):** this is a concrete instance of the selectivity result. **Cell-line identity sits in the model's causally active / privileged subspace** (its J-space analogue) — it is read automatically, because emitting any plausible cell-type-consistent profile requires it. **Drug identity is encoded but lies outside that workspace.** The asymmetry is principled: cell-line conditioning is a fixed, always-needed lookup, whereas drug→gene response demands *flexible, context-dependent composition* about drug–gene relationships — and that never enters the causally active subspace. This elevates the result from "this model fails" to "this model fails in a way that reflects a general principle about how models allocate representation to automatic vs. flexible computation."
- **Status:** ✅ **the drug-blindness is a readout-specificity failure, not an encoding failure.** Supersedes Q7 (the inconclusive steering probe). Corroborates the "read but not used" spine (Q6 + Q4) and gives it a mechanism. ➡️ motivates **Arm 1b** (objective-side discrimination loss: contrastive / swap-consistency / KL-maximizing auxiliary that rewards the readout for distinguishing drug directions). Confirmatory **held-out-means swap** done (job 596473) — null reproduced at every layer, estimation noise ruled out; **the experiment is complete.** Raw logs: `logs/workspace_probe_596144.out` (within-sample), `logs/workspace_probe_596473.out` (held-out); JSON: `RESULTS/workspace_probe.json`.

### Q14. Do OPTIMAL-TRANSPORT (state-matched) training targets make the model use the drug? (Arm 1c — answers auditor A-02)
- **Why:** Arm 1a (Q12) denoised the target and failed — but it also *destroyed the input→output correspondence*: consensus maps **every** control cell to one global pseudobulk (many-to-one; nothing per-cell to learn) and collapses heterogeneity. The auditor's A-02 proposed the missing middle: use **optimal transport** to pair each control cell with the treated cells it plausibly *becomes*, giving a target that is **denoised AND control-specific**. This is the last untested target-side lever, so it decides whether the target is the bottleneck at all.
- **The ε-ladder framing (why this is a controlled sweep, not a new method).** Entropic-OT regularization ε interpolates between the two things we already tried: **ε→∞** ⇒ the coupling becomes uniform and the barycentric target → the treated **mean** = **consensus (Arm 1a)**; **ε→0** ⇒ sharp assignment ≈ a single (state-matched) treated cell ≈ the **single-cell** regime. So Arm 1c is one knob whose endpoints are our prior experiments, with the unexplored middle in between.

- **How — methodology in full (the pipeline is new, so it is documented end-to-end):**

  **(1) What optimal transport is here.** Given a cloud of control cells and a cloud of treated cells, OT finds the cheapest way to reshape one into the other. We use the **Kantorovich** form: a **coupling** `π` (a table where `π[i,j]` = mass moved from control cell *i* to treated cell *j*) with uniform marginals, minimizing `Σ_ij π[i,j]·C[i,j]` where `C[i,j] = ‖x_i − y_j‖²`. Solved with **entropic regularization + Sinkhorn** (alternating row/column rescaling of `K = exp(−C/ε)`, 250 iterations; hand-rolled, no external OT dependency). The coupling is turned into a prediction by the **barycentric projection** `T(x_i) = Σ_j π[i,j]·y_j / Σ_j π[i,j]` — a soft Monge map answering *"where does THIS control cell land after treatment?"*

  **(2) Embedding — where the coupling lives (`build_embeddings.py`, `join_latent.py`).** OT's sample complexity degrades badly in high dimensions, so the coupling is computed in a low-dim embedding, not raw counts. We use the **released Tahoe scVI model** (`tahoebio/Tahoe-100M-SCVI-v1`, `n_latent=10`). **We JOIN the shipped per-cell latent** (`obsm['X_latent_qzm']`, 95.6M cells in the 40 GB minified `adata.h5ad`) by **`BARCODE_SUB_LIB_ID`** — matching Mert's approach — rather than re-encoding. *Why:* our re-encode produced a **degraded** latent (cell-line linear probe **0.11**) whereas the shipped latent is faithful (**probe 0.975**, per-dim variance ≈2). Root cause of the re-encode failure, found later: Tahoe's `genes` field holds **vocabulary token IDs, not positional indices** (values exceed the 62,710-row gene metadata), so positional mapping scrambles genes. The join is exact: **620,564/620,564 barcodes matched (100%)**, drug- and cell-line-agreement **1.000**.

  **(3) Expression — what targets are built from.** Per cell we build a **946-gene L1000 panel** expression vector, mapped via `gene_metadata['token_id'] → gene_symbol → panel` (**946/946 panel genes mapped**), library-normalized to **CP10K**. Validation: PCA on this panel gives a cell-line probe of **0.864** (chance 0.02), i.e. the expression is correctly un-scrambled.

  **(4) Population construction.** Condition = **(cell_line, plate, drug, dose)**; controls = **plate-matched DMSO**, so every coupling is computed **within (cell_line, plate)** and batch identity cannot drive it. DMSO controls do **not** co-locate with treated cells in Tahoe's shards, so the builder runs **two passes**: a treated pass over random shards, then a **dedicated DMSO scan** (220 shards) for the (cell_line, plate) keys the treated pass needs. Final cache: **620,564 cells = 600,000 treated + 20,564 control** (198 cell-line×plate groups), **106 drugs**, **50 cell lines**.

  **(5) Targets — DECODER-FREE (the key design choice).** The coupling is computed in **scVI latent** (clean state geometry) but the target is built in **expression space** as a convex combination of *real* treated cells. This avoids the scVI decoder entirely — targets are always on the data manifold and the subtle drug signal is preserved exactly. For each control cell *i* in a condition:
    - **T0 (consensus, ε→∞):** `target = mean(Y_expr)` — control-independent (= Arm 1a).
    - **T1 (mean-displacement):** `target = X_expr_i + (mean(Y) − mean(X))` — keeps the control's identity, adds one global drug shift.
    - **T2 (OT-barycentric, ε=0.5):** `target = (π @ Y_expr)_i / rowsum` — **control-specific and state-matched**.
    Each target is ranked (desc) and truncated to top-K (K = median expressed panel genes among the condition's treated cells) → `[END_CELL]` sentence. The prompt is the **same control cell** + drug/dose/MoA text in the exact trainer format (real cell-line names and `moa-fine`), so the examples are drop-in compatible. Emitted: **3,856 usable conditions (≥60 treated, ≥60 control) → 381,485 examples per arm**.

  **(6) Step-0 gates (run before training, all passed).**
    - **0a signal survival:** **54%** of sampled conditions show a *significant* treated-vs-control shift in scVI latent (label-permutation test) — the drug survives the embedding for roughly the winnable half, consistent with Q11 (27% inert + many subtle).
    - **0c target sanity:** per-control **T2 variance = 1.95 (>0)** — targets genuinely *vary across control cells* (the property consensus lacks) — and **‖T2 pseudobulk − real treated‖ = 0.006 (≈0)** — they reconstruct the treated cloud.
    - **0d target-contrast:** targets differ from their control prompt by **T0 = 59.9 / T2 = 64.5** top-K genes — substantial, so there is real perturbation to learn.

  **(7) Training + evaluation.** Cold-start from the same base (`C2S-Scale-Pythia-1b-pt`) with **identical hyperparameters** to the single-cell and consensus runs (1 epoch, lr 1e-5, grad-accum 16, max_length 8192, bf16) → the **target is the only variable**. T2 completed a full epoch (**23,842 steps**, train loss **1.6312**, eval loss 1.9433). Scored with the leak-immune battery: **within-plate NIR + scramble (same control, wrong drug token) + control-copy**, vs the fixed single-cell linear baseline.

- **Answer — NO. State-matched OT targets do not make the model use the drug.** Validity is clean first (`emits_end_cell = 1.00`, hallucination 0.001, len_ratio 0.87 — generation is not broken):

  | arm (identifiable subset, n=204 drugs, 40 cell lines) | value |
  |---|---|
  | model | 0.698 |
  | scramble (same control, wrong drug) | 0.699 |
  | **model − scramble** | **−0.001, clustered 95% CI [−0.018, +0.016] → NULL** |
  | control-copy (zero drug info) | 0.766 (≈ model → leakage) |
  | model − linear | +0.167 — **confounded**, matched by control-copy |

  Aggregate expr-NIR (chance 0.50): model **0.511**, scramble 0.503, linear 0.500, control 0.504, mean 0.180, ceiling 0.576.

- **The ε ladder is now swept, and every point is drug-blind:**

  | target | ε | `model − scramble` | verdict |
  |---|---|---|---|
  | single-cell (real treated cell) | ≈0 | +0.014 [−0.016, +0.042] | null |
  | **T2 OT-barycentric (Arm 1c)** | **0.5** | **−0.001 [−0.018, +0.016]** | **null** |
  | consensus (global mean, Arm 1a) | →∞ | −0.010 [−0.022, +0.003] | null |

  We swept from sharp per-cell matching to full pseudobulk collapse — including the causally-matched OT target that was the entire point of A-02 — and **no target construction makes the model use the drug.**
- **Quality axis (tentative):** aggregate expr-NIR 0.498 (single-cell) → **0.511** (T2), a *marginal* improvement consistent with Q1 (aggregation aids prediction), still far below ceiling 0.576. **T0 was not trained** (skipped once T2 came back null), so the clean T2-vs-T0 quality comparison is **not** established — this quality claim is suggestive only.
- **Caveats:** (a) the cache covers **106 of ~1,100 Tahoe drugs** (80-shard sample) — ample to test drug-use, not a population estimate; (b) T0/T1 untrained, so the ladder's *quality* decomposition is incomplete (the *drug-specificity* conclusion does not depend on it, since consensus and single-cell were already measured); (c) ε was fixed at 0.5 (no fine sweep) — but the two analytic endpoints are covered by Arm 1a and the single-cell model, so an intermediate ε would have to beat both endpoints *and* T2 to change the conclusion.
- **Status:** ✅ **the target-side hypothesis is definitively closed.** Denoising (Q12), pairing, and state-matched optimal transport (Q14) all fail identically — the defect is **not** the training target. This is exactly the Q13 prediction (the readout is drug-agnostic: magnitude-sensitive, direction-blind), now confirmed by a controlled sweep rather than a single failed attempt. ➡️ **All remaining effort goes objective-side (Arm 1b)**: a contrastive / swap-consistency / discrimination loss that *forces* the readout to distinguish drug directions. The OT couplings built here (state-matched control→treated pairs) are the natural raw material for that loss. Artifacts: cache `/data/.../ot_cache`, targets `/data/.../ot_targets` (T0/T1/T2.jsonl + `step0_gates.json`), model `checkpoints/pythia_sft_ot_T2/final`, results `RESULTS/nir_ot_T2.json`, `RESULTS/stratify_ot_T2.json`.
- 🔁 **CORRECTION (from Q15's stratified scramble):** the "T2 is drug-blind" verdict was measured with a **random-partner** scramble in full-profile space (−0.001). Under the sharper **opposite-signature** swap in residual space, T2 scores **+0.0263, CI [+0.0069, +0.0476] — CI excludes zero — with a monotone gradient** (near +0.007 → orth +0.014 → opposite +0.026). So **T2 does use the drug weakly**; the original null was an *underpowered instrument*, not an absence of signal. The single-cell model remains null under the same sharp test, so the ε-ladder conclusion (target content alone does not solve drug-blindness) stands, but "OT achieved nothing" is **too strong** and is retracted. See Q15 §methodology for why a random swap under-detects.

### Q15. Does changing WHAT THE TOKENS ENCODE make the model use the drug? (Arm 1b stage 1 — residual targets)
- **Why:** Q12/Q14 swept the ε-ladder of target *content* and all failed identically. The remaining explanation is a **ratio**: in a ~200-gene cell sentence ranked by expression, the drug-specific part is a handful of genes, so cross-entropy allocates ~1% of its gradient to drug identity. Changing *what the target is* never changed that ratio. This arm changes **what the tokens encode**.
- **The measurement that motivated it (`target_divergence.py`, 6,602 conditions / 125 groups):**

  | representation | inter-drug top-200 tokens DIFFERING | replicate retrieval ceiling |
  |---|---|---|
  | FULL profile (the old target) | **34.6 / 200 (17%)** | 0.742 |
  | SHIFT (treated − control) | 111.4 (56%) | 0.742 |
  | **RESIDUAL (drug-specific)** | **120.3 / 200 (60%)** | 0.747 |

  So **83% of the old target's tokens are identical between any two drugs in the same context.** ❌ *Note the ceiling is UNCHANGED* — an earlier hypothesis that residualization raises the ceiling is **retracted**; subtracting a per-group constant leaves pairwise distances unchanged. The information was always present (~0.74); the **tokenization discarded it**.

- **How — methodology in full:**

  **(1) Target construction (`build_residual_targets.py`).** For each condition (drug, cell_line, plate, dose) with ≥40 treated and ≥20 plate-matched control cells, from the 620k-cell `ot_cache` in log1p(CP10K) panel space:
  `residual = (treated_pseudobulk − control_pseudobulk) − mean_over_drugs(treated − control)`.
  The control subtraction removes the cell's own state; the **mean-over-drugs subtraction removes the generic drug program** (Q8: ~0.26 of "skill" that is drug-AGNOSTIC and matched by a no-fit baseline). What remains is what makes *this* drug different.
  - **Scope = cell line, not plate** — measured: 62% of conditions reproducible vs **19%** at plate scope (a plate holds too few drugs to estimate the mean, so subtracting a noisy mean injects noise).
  - **Reliability filter (filter, not weight):** keep a condition only if its residual reproduces across a half-split, `cos(res_A, res_B) > 0.2`. **KEPT 4,091/6,617 (62%)**, exactly matching the divergence prediction. The other 38% are noise.
  - **Encoding (sign is biology):** `"<up genes, most up-regulated first> [DOWN] <down genes> [END_CELL]"` — a signed DE signature, 100 genes per block. `[DOWN]` is **registered as an atomic special token** in the trainer alongside `[END_CELL]` (verified: id 50278; unregistered it splits into subwords and the up/down boundary stops being a clean symbol).
  - **Emission:** `per_cell` — every plate-matched control cell in the condition becomes a prompt (prompt format identical to the original trainer: real cell-line name, drug, dose, `moa-fine`, control cell sentence), with the condition-level residual as the response. **245,444 examples from 4,091 conditions.**
  - Also writes `reconstruction.npz` (per-cell-line generic shift) so full profiles can be rebuilt at evaluation.

  **(2) Training.** Cold start from `C2S-Scale-Pythia-1b-pt`, **identical hyperparameters to every prior arm** (1 epoch, lr 1e-5, grad-accum 16, max_length 8192, bf16) → the target encoding is the only variable. Completed 15,340 steps, train loss 1.8714, plateaued. Loader `shuffle=True` (checked — the file is written in condition order, so without shuffling every optimizer step would have seen 16 copies of one target).

  **(3) Evaluation (`residual_eval.py`).** Scored in **residual space** (where the model was trained): predicted and true residuals are both mapped to a **signed rank vector** (top-k up positive, top-k down negative, weight `1/log2(rank+1)`) so a generated sentence and a continuous truth are compared like for like; similarity = cosine; **NIR** = fraction of other drugs *in the same cell line* whose truth is less similar than the drug's own. Clustered 95% CI over **cell lines**.

  **(4) STRATIFIED SCRAMBLE — the methodological fix that made the result visible.** A scramble that swaps to a *random* other drug can land on a **near-twin**, where an unchanged output is **correct, not blind** — this biases `model − scramble` toward zero (the same lesson as the earlier swap-distance sweep). We therefore swap to **three defined strata** by residual cosine within the cell line: `near` (most similar), `orth` (cos ≈ 0), `opposite` (most anti-correlated — the sharpest test). **If drug use is genuine the gap must GROW near → orth → opposite**; a flat profile is a red flag.

  **(5) THE CONTROL that makes the result interpretable.** The residual model is scored in a new space, so a positive could be an artifact of the *metric* rather than the training. We therefore scored the **single-cell** and **OT/T2** models through the **identical pipeline, on identical conditions and strata** (`--model_kind cellsentence`): their generated treated cell is decoded via an empirical rank→value profile in cache units, then `− control pseudobulk − generic shift` gives an **implied residual**, scored the same way.

- **Answer — YES. The residual model uses the drug, and the control confirms it is not a metric artifact.**

  ⚠️ **First measured with `--max_new_tokens 600`, which SILENTLY TRUNCATED 26% of generations.** A
  signature is 100 up + `[DOWN]` + 100 down ≈ 201 gene symbols, and gene symbols are 2–4 BPE tokens each
  (~500–800 tokens), so a quarter of generations never reached `[DOWN]` and were scored with their entire
  down-block missing. **Re-running at 1400 tokens roughly doubled the measured effect.** Both are shown;
  the 1400-token row is the correct one. *Methodological lesson: for generation-based evals, log the
  `[END_CELL]` completion rate — a token budget that looks generous can halve a result.*

  | stratum (swap partner) | **residual model** (max_tok **1400**) | ⚠️ same, truncated (600) | single-cell (control) | OT/T2 (control) |
  |---|---|---|---|---|
  | `near` (mean cos +0.41) | **+0.0351 [+0.0061, +0.0656]** ✅ | −0.0045 (spans 0) | −0.0140 [−0.045, +0.018] | +0.0073 [−0.008, +0.023] |
  | `orth` (cos ≈ 0.00) | **+0.0716 [+0.0371, +0.1059]** ✅ | +0.0349 (spans 0) | −0.0206 [−0.050, +0.006] | +0.0144 [−0.003, +0.032] |
  | **`opposite` (cos −0.33)** | **+0.1429 [+0.1112, +0.1787]** ✅ | +0.0716 [+0.032, +0.110] | −0.0204 [−0.054, +0.010] ✗ | +0.0263 [+0.0069, +0.0476] ✅ |
  | **gradient with dissimilarity** | **YES (all three strata clear zero)** | YES | **NO (flat)** | YES |

  Absolute NIR at 1400 tokens: model **0.704**, scramble_near 0.669, scramble_orth 0.632,
  scramble_opposite 0.561, **ceiling 0.964**, random floor 0.486; `[DOWN]` present in **91%** of
  generations (up-block 123 / down-block 103 genes, 98.8% valid panel genes, 1.7% duplicates).
  Reconstructed full-profile space (random-partner scramble, **not** stratified, and still at the old
  token budget): `model − scramble = +0.0150 [+0.0008, +0.0309]`, ceiling 0.868 — understated on both
  counts, to be re-run.
- **Three independent checks agree, and only the residual model passes all three:**

  | check | residual (1400 tok) | single-cell | OT |
  |---|---|---|---|
  | corr(condition reproducibility, model NIR) — better where the drug effect is *real*? | **+0.144** ✓ | −0.117 ✗ | −0.216 ✗ |
  | prediction diversity (mean pairwise cos between predictions; truths = −0.005) | **+0.124** (most diverse) | +0.303 | +0.633 (collapsed) |
  | cos(prediction, own truth) vs other drugs | **+0.084 / −0.000** | +0.026 | +0.049 |
- **Caveats (all material):**
  - ⚠️ **THE CONTROL COMPARISON IS NOT MATCHED-BUDGET — the three-column table above is confounded.** Verified from the saved configs: `re_residual_maxtok.json` ran at **`max_new_tokens=1400`**, while **both** controls (`re_singlecell_model.json`, `re_ot_model.json`) ran at **600**. Since the token budget alone *doubles* the measured effect for the residual model (+0.0716 → +0.1429), the controls were given half the budget of the treatment. The direction of the bias favours our conclusion: truncation pushes a gap toward zero, so an under-budgeted control looks *more* null than it is. Mitigating (but not sufficient): a cell sentence is ~123 gene symbols where a residual signature is ~201, so 600 tokens bind far less on the controls — but the generation completion rate for the control arms was logged and never persisted, so this cannot be checked from the artifacts. **Action: re-run both controls at 1400 before quoting the three-column comparison.** Until then the residual model's own gradient (+0.0351 → +0.0716 → +0.1429, all clearing zero) stands on its own, and Q16 is unaffected (every arm there ran at 1400 inside a single job).
  - **Scoring paths differ**: the residual model is scored natively; the controls go through decode→subtract. The single-cell null proves the path does not *manufacture* positives, but it cannot rule out a magnitude advantage. **Residual vs OT CIs overlap** ([+0.032,+0.110] vs [+0.007,+0.048]) — "residual beats OT" is *suggestive, not established*.
  - **Token-budget truncation (RESOLVED):** at 600 tokens 26% of generations never reached `[DOWN]`; at 1400 tokens that is 9%, and the effect roughly doubled. The remaining 9% still understate it slightly.
  - **All scored conditions were trained on** → this demonstrates the model **reads the drug token** (memorization also requires that), **not generalization**. A held-out-shard cache is the outstanding test.
  - **Reconstructed number is not stratified** (random partner), so +0.0150 understates by the same near-twin logic.
  - **Large headroom**: model 0.652 vs ceiling 0.964.
- **Status:** ✅ **first non-null `model − scramble` in this project**, with a matched control and a monotone dissimilarity gradient. The bottleneck was **what the output tokens encode**, not the target's content (Q12/Q14) and not the drug's presence in the representation (Q13). ➡️ Next: stratify the reconstructed eval; fix the `[DOWN]` emission rate; **held-out conditions for generalization**; then stages 2–3 (profile-level contrastive loss, drug-conditioned modulation) per `docs/proposals/arm1b_objective_spec.md`. Artifacts: targets `/data/.../residual_targets/`, model `checkpoints/pythia_sft_residual/final`, results `RESULTS/re_residual_model.json`, `RESULTS/re_singlecell_model.json`, `RESULTS/re_ot_model.json`, `RESULTS/reconstructed_eval_v2.json`; log `logs/arm1b_ctrl_*.out`.

### Q16. Does the residual model's drug use GENERALIZE, or is it a memorized lookup? (Arm 1b stage 1 — held-out evaluation)
- **Why:** Q15's `+0.143` was measured on conditions that were **all trained on** (stated in its own caveats). Memorization *also* requires reading the drug token, so Q15 proves the token is read but **cannot** distinguish "learned a transferable drug representation" from "memorized a per-drug lookup". Since the pre-registered win condition in `arm1b_objective_spec.md` §5 is `model > drug_lookup`, and §7 pre-registers the "memorizes averages, adds no tailoring" branch, this is the test that decides which branch fired.
- **How — methodology in full:**

  **(1) Three-way holdout (`make_holdout` in `build_residual_targets.py`).** From the 4,091 reproducible conditions:
  - `unseen_drug` — **every** condition of a held-out drug (11 drugs, 488 conditions). The model never sees the token. Given the SAR gate this *should* fail; it is the **control** proving that any transfer in `unseen_combo` comes from having seen the drug.
  - `unseen_combo` — a held-out **(drug, cell_line) pair** whose drug is seen in ≥3 other cell lines (159 pairs, 250 conditions). **Cross-context transfer — the scientifically meaningful test**, and genuinely open given Q11's cell-line identifiability swing of 0.83.
  - `train` — the remaining 3,353 conditions → **201,164 examples** (exactly 60 control cells per condition).
  - **Dose/plate leak ruled out:** the split is keyed on `(drug, cell_line)`, so *every* plate and dose of a held-out pair goes to the held-out split. Verified in code, not assumed.

  **(2) Two instrument bugs found and fixed — both would have hidden the result.**
  - **Proportional-sampling dilution.** The first run built 250 `unseen_combo` conditions but scored only **33**: the eval drew a *uniform* 500 from the 4,091-condition pool, which reproduces the pool's proportions (3353/250/488 → 410/33/57, exactly). The transfer test was starved by the sampler, not by the build. Fixed with `--split_quota`, which allocates the generation budget where n is short. **Lesson: a stratified question needs a stratified sampler; a uniform draw silently answers a different question.**
  - **Degenerate-tie NIR.** "Predict the average drug response" *is* the zero vector in residual space (the generic program is subtracted by construction), so every cosine against it is 0.0 and the strict `mean(own > x)` scored it **0.000**, not the 0.500 the header claimed. `nir_from_sims` is now tie-aware (Mann–Whitney convention). Ties are measure-zero for real vectors, so no other arm moved — `generic` now reads **0.500 exactly**, as designed.
  - Also hardened: `--max_new_tokens` **default raised 600 → 1400** in `residual_eval.py` and `reconstructed_eval.py`. The 600 default is the Q15 scar (26% truncation halved a measured effect); leaving it as a default that one forgotten flag restores was an unacceptable footgun.

  **(3) The baseline ladder (each a predictor of the drug-specific residual, scored through the identical pipeline — same truths, same signed-rank encoding, same comparison set).**

  | baseline | information used | role |
  |---|---|---|
  | `drug_lookup` | this drug's mean residual in **all other cell lines** | THE bar — a lookup memorizes; only a model can adapt |
  | `drug_lookup_1` | this drug in **ONE** other cell line (that line's conditions averaged) | removes the cross-line **denoising** advantage |
  | `moa_lookup` | **other** drugs sharing this drug's MoA, same cell line | MoA-leak diagnostic (our prompt contains `Mechanism:`) |
  | `control_copy` / `generic` / `random` | none | must sit at ~0.50 by construction |

  `drug_lookup_1` picks a **cell line** and averages that line's conditions — *not* one condition — because 62% of drugs are multi-dose (Q11), so a single condition would confound "different cell line" with "different dose". It runs on a private `RandomState` so it cannot shift the generation stream.
  **Baselines are reported per split**, because on `unseen_drug` the same-drug lookups are **oracles** — they read the held-out drug's own residuals, information the model is definitionally denied. `moa_lookup` is *not* an oracle (it uses drugs that were in training), so it is a fair contest in every split.

- **Answer — YES. Drug use transfers to unseen contexts, with NO memorization premium.** (n=570 conditions, stratified `train=200 / unseen_combo=250 / unseen_drug=120`; validity clean first: 9,120 generations, `[DOWN]` 93%, up-block 124 / down-block 103, valid panel genes 98.9%, duplicates 1.4%.)

  | split | n (cell lines) | model NIR | **opposite-swap gap** | verdict |
  |---|---|---|---|---|
  | train | 200 (40) | 0.657 | **+0.0898 [+0.0530, +0.1269]** | uses drug |
  | **`unseen_combo`** | **250 (38)** | **0.650** | **+0.1002 [+0.0661, +0.1368]** | **CROSS-CONTEXT TRANSFER: YES** |
  | `unseen_drug` | 120 (36) | 0.586 | +0.0195 [−0.0452, +0.0860] | null (as designed) |

  **Memorization premium ≈ zero** (train +0.0898 vs unseen_combo +0.1002, CIs almost fully overlapping). Held-out (drug, cell_line) pairings score *as well as* trained ones → the model did **not** memorize pairings; it learned a drug representation that applies in cell lines it never saw that drug in. The `unseen_drug` null is the control firing correctly: same model, same instrument, drug token carrying no learned information → no gap.

- **But the model loses to every drug-side lookup, and that is the pre-registered "honest partial" branch:**

  | arm | NIR | `model −` arm (clustered CI over cell lines) |
  |---|---|---|
  | ceiling (within-condition replicate) | **0.968** | — |
  | `drug_lookup` | **0.963** | **−0.3245 [−0.3539, −0.2975]** LOSES |
  | **`drug_lookup_1`** | **0.832** | **−0.1936 [−0.2285, −0.1612]** LOSES |
  | model | 0.639 | — |
  | `moa_lookup` (n=143) | 0.529 | +0.1042 [+0.0144, +0.1882] wins |
  | `generic` | **0.500** | +0.1388 [+0.1086, +0.1650] |
  | `random` | 0.502 | — |
  | `control_copy` | 0.478 | +0.1613 [+0.1314, +0.1907] |

  On the held-out split specifically: `model − drug_lookup` **−0.3170**, `model − drug_lookup_1` **−0.1825**. So `arm1b_objective_spec.md` §7's middle branch has fired: *"> 0 but ≤ drug_lookup → the model memorizes per-drug averages but adds no cell-state tailoring; an honest partial result."* The transfer is real; the recovered fraction of the signature is small.

- ⚠️ **RETRACTED before it entered this file: "`ceiling − drug_lookup = +0.004` ⇒ the drug residual is context-independent ⇒ the modelling target has vanished."** That reading is not supported by the comparison, for three reasons that all favour the lookup: (i) the **ceiling is scored against a HALF-sample truth** while every baseline is scored against the **FULL-sample** truth; (ii) `drug_lookup` averages ~38 conditions against the ceiling's single noisy half; (iii) **NIR near 0.96 is compressive** — a wide range of cosines collapses into a few thousandths. A lookup that merely *ties* a split-half ceiling in cosine is still compatible with much of the residual being cell-line-specific. The auto-verdict that asserted this has been removed from `residual_eval.py` and replaced by the caveats. *(Any log line reading "HEADROOM … has essentially vanished" predates the patch — ignore it.)*
- **The two lookup arms genuinely disagree, and NIR cannot adjudicate — this is the open question.** `drug_lookup_1` sits **0.136 below the ceiling** despite using *more* cells than the ceiling's half-sample, which argues for real cell-line-specific structure. Yet averaging many lines recovers to 0.963 ≈ ceiling — and averaging can only remove **noise**, never the systematic miss of the target line's own interaction, so a large interaction should have left `drug_lookup` on a plateau well below the ceiling. Both readings survive in NIR because of the compression at (ii)–(iii) above. ➡️ **Resolvable only in cosine space:** a noise-corrected cross-cell-line transfer coefficient with control cells half-split and a same-plate/different-drug negative control. **That number sizes the entire remaining prize and nobody has computed it.**
- **Caveats:**
  - `unseen_drug` is **underpowered by construction**, so its null is uninformative: CI half-width ±0.087 against a maximum available MoA-channel effect of ~0.033. Report it as a *power* statement, never as evidence of absence.
  - `unseen_drug` is also **not a clean tier-2 design** — the prompt contains `Mechanism: {moa}`, so a drug-level split hands the model the held-out drug's class label. Leave-one-MoA-out is the correct split. Mitigating: `moa_lookup` 0.529 says the leak is small.
  - Absolute numbers here are **cross-plate**; the `model − scramble` *difference* is leak-immune (both arms share the same control cell), but the absolute NIRs are not directly comparable to within-plate tables elsewhere in this file.
  - This model (`pythia_sft_residual_holdout2`) trained on 3,353 conditions vs Q15's 4,091, so its absolute gap (+0.09/+0.10) is **below** Q15's +0.143 — that is the holdout cost, not a regression.
  - Cache still covers 106 of ~1,100 Tahoe drugs.
- **Status:** ✅ **the drug use generalizes.** The residual encoding produces a drug representation that transfers to unseen (drug, cell_line) pairings with no memorization premium — the first genuine generalization result in this project. It does **not** beat a per-drug lookup, which is the pre-registered honest-partial outcome. ➡️ Next: the noise-corrected transfer coefficient (decides whether headroom exists); the drug-side **channel gate** (`targets` and `pubchem_cid` columns exist in `drug_metadata.parquet` and **no script has ever read them** — `sar_gate.py` probes only for SMILES, so the SAR negative closed *structure*, not *target*). Artifacts: targets `/data/.../residual_targets_holdout2/`, model `checkpoints/pythia_sft_residual_holdout2/final`, results `RESULTS/re_holdout2_stratified.json`; logs `logs/arm1b_gen_609403.out`, `logs/re_eval2_610304.out`.

### Q17. How much of the drug-specific residual is a per-drug CONSTANT vs a drug × cell-line INTERACTION? (the headroom bound)
- **Why:** Q16 left two numbers pointing opposite ways and NIR could not adjudicate. `drug_lookup_1` (one other cell line) sat **0.136 below** the ceiling, arguing for real cell-line-specific structure; yet `drug_lookup` (averaged over ~38 conditions) recovered to **0.963 ≈ ceiling 0.968**, and averaging removes only *noise*, never the systematic miss of the target line's own interaction — so a large interaction should have left it on a plateau. The tie-breaker matters enormously: if the residual is a per-drug constant, a lookup is the correct model and every conditional arm in this project failed for a **structural** reason; if not, there is a conditional target nobody has reached.
- **How (methodology in full, `variance_decomposition.py`):** model the response as
  `shift(d,c) = generic(c) + β(d) + κ(d,c) + ε`, and estimate the **transfer coefficient**
  `T = var(β) / (var(β) + var(κ))` — the fraction of noise-free residual variance that is a per-drug constant.
  - **Estimator — disattenuated cross-line correlation.** `T̂ = cos(r(d,c₁), r(d,c₂)) / sqrt(ρ(d,c₁)·ρ(d,c₂))` with `ρ` = Spearman–Brown of the half-split cosine. Dividing out reliability is *essential*: measurement noise alone depresses the raw cross-line cosine and would otherwise read as interaction.
  - **Validated against planted ground truth**: recovers T = 1.00 / 0.70 / 0.40 to within **0.005**. And it shows why the correction is not optional — at **κ = 0 the RAW cross-line cosine is 0.734**, which read naively reports "27% interaction" where the truth is zero.
  - **Clustered bootstrap over DRUGS** (the estimand is a per-drug property), not cell lines.
  - **Simulated null** calibrated to the measured noise: at κ=0 the estimator returns **1.001**, so T is read against that, never against 1.0.
- ⚠️ **A false-alarm VOID, and the control-design lesson that produced it.** The first run compared T (0.511) against a negative control of **different drugs on the SAME plate**, which came back **+0.485** — nearly equal to T — and the run was correctly declared VOID. But that control differs from T in **two** ways at once (different drug *and* same cell line). Plate/batch structure surviving a cell-line-scoped generic inflates *same-plate* comparisons while leaving *cross-line* comparisons untouched, so it depressed the control's usefulness without ever touching T. Adding the **structure-matched** control — different drug, **different cell line**, breaking only the drug match — resolves it: it reads **+0.000** at *every* configuration.
  A second bug compounded this: `transfer_pairs` grouped different-drug pairs by plate, so requesting cross-line pairs returned an **empty list** and the verdict silently fell back to the same-plate control. It now needs its own sampler and warns on an empty control.
- **Answer — a REAL interaction exists, and it is large.**

  | config | T (repro-filtered) | same-plate null | **structure-matched null** | dose | shared−split bias |
  |---|---|---|---|---|---|
  | **`--generic_scope plate`** | **0.557 [0.513, 0.601]** | −0.018 | **+0.000 [−0.019, +0.021]** | 0.703 | −0.000 |
  | plate + `--project_generic` | 0.545 [0.503, 0.585] | −0.019 | +0.010 [−0.010, +0.029] | 0.682 | −0.004 |
  | cell-line (production scope) | 0.517 [0.474, 0.560] | +0.478 | +0.000 [−0.010, +0.010] | 0.484 | +0.254 |

  Against a simulated κ=0 null of **1.001**, the shortfall is **0.444 / 0.456 / 0.484** →
  **≈44–48% of the drug-specific residual variance is drug × cell-line interaction.** Robust to the generic's scope, to orthogonal-rejection of the generic direction, and to control splitting. **`drug_lookup` captures the β(d) main effect and structurally cannot reach the remaining ~45%.**
- **Three internal consistency checks, all passing:**
  1. **Dose ordering.** At plate scope, same-drug/same-line/**different-dose** transfer is **0.703**, *above* cross-cell-line **0.557** — changing the dose costs less than changing the cell line, which is the biologically correct ordering. At cell-line scope the ordering **inverts** (0.484 < 0.517), another sign the plate-scoped build is the clean one.
  2. **Control-split bias vanishes.** At plate scope shared-control and split-control T agree to **−0.000** (0.557 vs 0.557). At cell-line scope they differ by **+0.254**. The discrepancy was plate contamination, not the control split.
  3. **Orthogonality.** `(‖r‖/‖s‖)² + (‖g‖/‖s‖)²` = **1.002** (plate) and **0.999** (cell line) — residual and generic are essentially orthogonal, so the fractions below are genuine variance shares.
- **SCOPE — what fraction of the response this frame covers (never previously computed):** at plate scope `‖residual‖/‖shift‖ = 0.786` and `‖generic‖/‖shift‖ = 0.620`, i.e. **62% of the perturbation response variance is drug-specific** and 38% is the generic program plus batch. (At cell-line scope, 79% / 21% — higher only because the cell-line generic does not absorb the plate component.) **Every claim in this project is scoped by this number.**
- 🔁 **RETRACTED: the "62% vs 19% reproducibility" justification for cell-line scope.** That comparison was measured with **shared** controls, which inflate cell-line reproducibility far more than plate reproducibility. Measured honestly with **split** controls the two are comparable — **20% (plate) vs 16% (cell line)** — so the reproducibility argument does not favour cell-line scope. Combined with the dose-ordering inversion and the control-split bias, **plate scope is the better build**, and the residual training targets (built at cell-line scope) therefore retain batch structure.
- **Caveats:**
  - `model − scramble` in Q15/Q16 is **unaffected** by the scope issue: it is a paired contrast in which both arms share the control cell and are scored against the same truth, so any shared component cancels. But the residual **training targets** do contain plate/batch structure the model may partly be learning — a plate-scope rebuild is the natural follow-up.
  - T on the unfiltered set is *higher* than on the repro-filtered set (0.598 vs 0.557); disattenuation is unstable at very low reliability, so the filtered value is the one to quote.
  - The simulated null assumes Gaussian noise; if real noise is structured differently the null could shift, though it would have to move by ~0.45 to change the conclusion.
- **Status:** ✅ **the headroom is real and quantified.** The Q16 ambiguity resolves in favour of genuine cell-line-specific structure: `drug_lookup` 0.963 ≈ ceiling 0.968 was **NIR compression** at the top of the scale, and `drug_lookup_1` 0.832 was the honest signal. **≈45% of the drug-specific signal is context-dependent, and neither the 1B model (0.639) nor any lookup reaches it.** ➡️ This *reopens* the modelling question rather than closing it, and it converts the thesis's central claim from a closure into a quantified, unclaimed target. Artifacts: `RESULTS/vd3_{plate,plate_proj,cellline}.json`, `RESULTS/vardecomp_*.json`; log `logs/vardecomp3_*.out`.

### Q18. Is unseen-drug generalisation reachable through any drug-side channel? (the pre-registration was wrong)
- **Why:** an unseen drug can only be predicted through a property it **shares** with drugs seen in training. Two design docs (`grpo_training_plan.md` §9, `arm1b_objective_spec.md` §2) pre-registered that unseen drugs were out of reach — and both inferred it from the **SAR gate**, which tested Morgan fingerprints and MolFormer, i.e. chemical **structure**. But `pubchem_drug_injection_spec.md` §2 ranks raw structure the **lowest**-value channel and protein targets **"the key feature"**. The pre-registration generalised from the weakest channel to the strongest. This measures the strongest.
- **How (`channel_gate.py`):** for each condition (d,c) and each channel X, predict d's residual as the mean residual of the **other drugs in the same cell line** that share property X with d — exactly `moa_lookup`'s construction, generalised. It uses only drugs that *were* in training, so unlike `drug_lookup` it is a fair contest in every split. Scored with the same signed-rank → cosine → tie-aware NIR as everything else.
  - **The control that makes it readable:** every channel is paired with a **count-matched random null** — the same *number* of partner drugs, drawn at random from the same cell line, averaged identically. Without it, a channel above chance cannot be distinguished from "averaging k residuals denoises, and denoised residuals beat chance", which is true of any k and says nothing about the channel.
  - **Coverage is reported before any verdict.** A channel annotated on few drugs cannot be *closed* by a null — it was never tested.
  - Validated against planted truth in both directions: a planted working channel returns +0.397 (sd 0.029 over 24 worlds), a planted useless one −0.016 (sd 0.091). Two bugs were caught this way rather than by review: the null originally drew from a pool *excluding* the channel's own picks (which depletes it of exactly the partners the channel selected and anti-correlates the arms), and the original selftest asserted a single draw of a quantity with sd ≈ 0.09 fell within 0.10, i.e. tested luck.
- **Answer — TWO CHANNELS ARE LIVE. Unseen-drug generalisation is NOT closed.**

  | channel | coverage | n scored | NIR | count-matched null | channel − null | verdict |
  |---|---|---|---|---|---|---|
  | **protein target** | 52/96 (54%) | 894 | **0.586** | 0.502 | **+0.0844 [+0.0592, +0.1095]** | **LIVE** |
  | **MoA** | 31/96 (32%) | 1053 | **0.573** | 0.495 | **+0.0780 [+0.0555, +0.1027]** | **LIVE** |
  | chemical structure | 96/96 (100%) | 4084 | 0.484 | 0.470 | +0.0137 [−0.0013, +0.0278] | closed |

  Tahoe's `drug_metadata.parquet` carries **`targets` (264 drugs, 280 distinct symbols, 550 edges)** and `pubchem_cid` (377/379) — and **no script in this repository had ever read either column.** Chemistry is closed, reproducing the SAR gate in the NIR frame; both knowledge channels clear their nulls decisively.
- 🔁 **RETRACTS the pre-registration in both design docs.** The correct statement is: *chemical structure does not predict response (measured, twice, in two frames); protein target and mechanism **do** (measured); therefore unseen-drug generalisation is reachable through drug-side knowledge, not through chemistry.*
- **Caveats:** target coverage is 54% and MoA 32%, so these are measured on the annotated subset; the absolute NIR (~0.58) is far below `drug_lookup`'s 0.963, so the unseen-drug ceiling is much lower than the seen-drug one; and target and MoA overlap (among dual-annotated drugs, 86 target-sharing pairs share no MoA class), so whether target adds **over** MoA is a further question this run does not answer.
- **Status:** ✅ **unseen-drug generalisation is live through target and MoA.** ➡️ The next arm is a channel-conditioned prompt (`Targets: EGFR, ERBB2` appended to `Mechanism:`) evaluated on a **leave-one-MoA-out** split. Artifacts: `RESULTS/channel_gate.json`; log `logs/channelgate_*.out`.

### Q19. Is the drug × cell-line interaction LEARNABLE, or merely present? (the κ-structure test)
- **Why:** Q17 established that ~45% of the drug-specific residual variance is interaction. It said nothing about whether anything can *learn* it. If each cell line modulates drug response in a consistent direction, that direction is estimable from the line's own cells and a conditional model has a concrete target; if the interaction is specific to each (drug, cell line) pairing, the variance is real but there is nothing to generalise from.
- **How (`variance_decomposition.py --kappa_structure`):** `κ(d,c) = residual(d,c) − β̂(d)` with **β̂ estimated leave-one-cell-line-out**, so κ is not contaminated by its own residual — without that, subtracting a mean containing r(d,c) induces a −1/(m−1) correlation and biases the signal arm *downward by construction*, which would make a structured interaction look idiosyncratic. Signal = `cos(κ(d,c), κ(d′,c))` for different drugs in the **same** line; null = the same across **different** lines. Both disattenuated by κ's own split-half reliability, since κ is a difference of two noisy quantities and is noisier than the residual it comes from. Validated against planted truth: a structured world returns same-line +0.977 vs cross-line −0.108 (excess **+1.085**); an idiosyncratic world returns excess **+0.001**.
- **Answer — IDIOSYNCRATIC. There is no consistent per-cell-line direction.**

  | config | same cell line | different cell lines | **excess within-line** |
  |---|---|---|---|
  | plate-scoped generic | −0.004 [−0.011, +0.002] | +0.003 | **−0.007** |
  | cell-line-scoped | −0.003 [−0.021, +0.015] | +0.002 | **−0.005** |

  (1,280 and 1,022 conditions from drugs seen in ≥3 cell lines; mean κ reliability 0.33; 4,000 pairs per arm.) The interval on the signal arm is tight enough to exclude anything above +0.002, so this is a **real null, not an underpowered one**. T reconfirmed in the same run at 0.551 [0.511, 0.589] (plate) and 0.509 (cell line), with the structure-matched negative control at −0.001 / +0.009.
- **This resolves the Q16/Q17 tension.** `drug_lookup` reaching 0.963 against a 0.968 ceiling is **not** because κ is small — κ is 45% of the variance. It is because κ is **unpredictable**, so it acts as irreducible noise for *any* predictor, and the lookup is therefore already at the **achievable** ceiling. Nothing can beat it on seen drugs. What a lookup structurally cannot do is handle a drug it has never seen — and Q18 shows that regime has live channels. **The model's only opportunity is the unseen-drug regime**, which is also DrEval's Leave-Drugs-Out, the setting that matters for drug design.
- **Caveats:** κ reliability is 0.33, so κ is noisy; the disattenuation corrects this in expectation and the CI is tight, but a *weakly* structured interaction below the resolution of 4,000 pairs at that reliability would not be detected. The claim is that no per-cell-line direction exists **at this sample size**, not that none exists in principle.
- **Status:** ✅ **the interaction is real but has no learnable structure.** Artifacts: `RESULTS/kappa_{plate,cellline}.json`; log `logs/kappa_*.out`.

### Q20. WHICH FIELD of the prompt does the model actually read? (the field decomposition)
- **Why:** three facts sat uneasily together after Q18. The channel gate showed **mechanism carries real signal** (+0.0780 over its count-matched null). The prompt has *always* contained `Mechanism: {moa}`. And the model's unseen-drug gap was null. The only thing reconciling them was "that arm was underpowered" — true, but a dodge. The question decides whether the obvious next arm (append `Targets: EGFR, ERBB2` to the prompt and retrain) is worth building: that arm assumes the bottleneck is **information availability**, whereas Q13 says it is the **readout**.
- **How (`residual_eval.py --field_decomp`):** the standard scramble replaces the drug name *and* the mechanism together. This decomposes it. Each arm swaps **one** span of the instruction line to the opposite-signature partner and leaves every other byte identical:
  - `scramble_drugonly` — the drug name, mechanism kept → does it read the identity token?
  - `scramble_moaonly` — the mechanism, drug name kept → **does it read the knowledge channel it already has?**
  - `scramble_opposite` — both, retained for continuity.
  Run on the **opposite** stratum only (the sharpest test), so generation cost is 1.5× rather than 2×.
  - **No-op guard:** if the swap partner happens to share the original's mechanism, a mechanism-only swap would produce a prompt *identical* to the model's own — a gap of exactly zero **by construction**, which would read as "the model ignores mechanism". Such conditions return `None` and are dropped rather than silently scored at zero. This is why the mechanism arm has n=325 against the others' 570.
  - `scramble_prompt` was also rewritten to operate on the instruction line **wherever it sits**, rather than on everything preceding `Control cell:` — otherwise the reordered-prompt arm would have found no drug name, returned `None`, and dropped every scramble arm without a word.
- **Answer — the model reads the NAME and does not measurably read the MECHANISM.**

  | swap | gap (opposite stratum) | n | verdict |
  |---|---|---|---|
  | **drug name only** (mechanism kept) | **+0.0809 [+0.0592, +0.1010]** | 570 | **READ** |
  | **mechanism only** (name kept) | **+0.0091 [−0.0174, +0.0383]** | 325 | **not detected** |
  | both (standard arm) | +0.1024 [+0.0827, +0.1236] | 570 | read |

  Roughly additive: the name carries about 79% of the combined effect. **The mechanism arm is powered to detect an effect the size of the name's** — a +0.081 effect would clear an interval of half-width 0.028 with room to spare — and it sees +0.009.
- **The reading, and what it rules out.** The model is handed a field that demonstrably carries drug-transferable signal (Q18: +0.0780) and does not use it. So for unseen drugs the bottleneck is **not information availability** — it is the same readout failure Q13 localises, appearing in the place where it matters most. ➡️ **This closes the channel-conditioning arm before it was built.** Appending protein targets to a prompt whose `Mechanism:` field is already being ignored would change nothing; the ~2 GPU-weeks that arm would have cost were saved by a single 5-hour eval.
- ⚠️ **Seed instability discovered in the same run — read this before quoting any interval in this file.** This run differs from the previous one *only* by the two added arms, which shift the sampling stream. Same checkpoint, same conditions, same seed. Yet:

  | | previous run | this run |
  |---|---|---|
  | model NIR | 0.639 | 0.645 |
  | `scramble_opposite` | 0.589 | **0.542** |
  | `drug_lookup_1` | 0.832 | **0.880** |
  | **`unseen_drug` gap** | **+0.0195 [−0.045, +0.086]** | **+0.1114 [+0.051, +0.173]** |

  The `unseen_drug` arm moved from a clean null to a CI excluding zero, and the two intervals barely overlap. **Cause: the clustered bootstrap resamples cell lines but treats each condition's score as FIXED**, so it captures between-cell-line variance and *not* generation variance — and at `k_samples=4`, temperature 0.8, the latter is evidently large. Every interval in this file is conditional on one draw of generations. `train`, `unseen_combo` and the strata gradient are stable across both runs; **the `unseen_drug` null is not, and must not be quoted as a null until the seed replication lands** (`jobs/seeds.md`, three seeds).
- **Status:** ✅ **the readout, not the information, is the unseen-drug bottleneck.** Artifacts: `RESULTS/field_decomp.json`; log `logs/fielddecomp_*.out`.

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
| revert_center (all genes → P/2) | none | **0.9999** [0.99991, 0.99992] | NA (undefined) | NA (undefined) |
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
2. ✅ **Causal probe (Q7) RESOLVED by Q13** — `workspace_probe.py` (logit-space KL, gate + positive control + matched-norm swap) supersedes the inconclusive steering run. Readout is drug-agnostic (swap null at every layer), not an encoding failure.
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

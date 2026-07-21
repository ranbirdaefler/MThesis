# Arm 1c — Optimal-transport / meta-cell training targets (Option A: round-trip)

**Status:** plan (pre-implementation). Answers auditor **A-02** (meta-cells + OT as less-noisy learning
targets). Builds on Mert Kaan's OT methodology (scVI latent space, Bures/MW₂, HiRef couplings) and on
our own **Arm 1a** (consensus targets, refuted) and **Q13** (readout is drug-agnostic).

---

## 0. Framing — why this is not just Arm 1a again

Arm 1a replaced each cell's target with the **global pseudobulk** of its (drug, cell-line, dose) group.
It removed noise but also removed the *input→output correspondence*: every control cell in a condition
mapped to the **same** mean target. The model saw "any control → one mean," so the drug stayed a
rounding error **and** the per-cell signal was destroyed. It did not help (drug-blind, output-invariance
≈ 0).

OT changes exactly the thing Arm 1a broke: it gives a **paired, control-specific, heterogeneity-preserving**
target. Instead of one mean, we learn the *movement* from each control cell to where it plausibly lands
under treatment. This is the auditor's phrase — "plausible movements from the control distribution toward
the treated one, rather than an arbitrary pairing of individual cells."

**The key conceptual link (and a knob we exploit):** entropic OT with regularization ε interpolates between
the two regimes. As **ε → ∞** the transport plan → the independent (uniform) coupling and the barycentric
target → the treated **mean** = *Arm 1a consensus*. As **ε → 0** it → a sharp, control-specific assignment.
So Arm 1a is OT at ε = ∞; Arm 1c asks whether a **finite ε** (sharper pairing) is what was missing. This
makes the comparison a clean, monotone family rather than two unrelated methods.

**Honest prior (from Q13).** Q13 showed the failure is on the **objective/readout** side (generation is
magnitude-sensitive, direction-blind), not obviously the target. A better *target* may still not overcome
a cross-entropy that is dominated by the generic gene ordering. Therefore Arm 1c is a **necessary,
cheap-to-falsify** experiment whose main deliverable is either (a) a target-side win, or (b) definitive
evidence that even a causally-matched target cannot help — which hands the objective-side arm (Option B /
Arm 1b) its motivation. Crucially, the OT machinery built here is the **shared substrate** for both.

---

## 1. Representation and the round-trip (Option A)

The model reads and writes **[END_CELL] cell sentences** (expressed genes ranked + sentinel over the fixed
946-gene L1000 panel). All OT lives in a **low-dimensional embedding**. Option A = build the OT target in
embedding space, **decode back to expression → ranks → a cell sentence**, and swap only the *response*
(prompt/control unchanged), reusing `build_consensus_targets.py --emit per_cell`.

Two candidate embeddings, tested head-to-head in Step 0:

| Embedding | Pro | Con | Decode |
|---|---|---|---|
| **scVI 10-dim** (Tahoe-released model + embeddings) | denoises dropout; clean cell-line manifold; Mert-validated | non-linear bottleneck may **discard subtle drug signal** (SNR≈0.75) | scVI decoder (expected expression) |
| **PCA-d on panel** (log1p-normalized 946 genes) | **exact linear inverse** (lossless round-trip on top-d); we control it | keeps technical noise; subtle signal may sit in dropped low-variance PCs | inverse transform (exact) |

Decision is **data-driven** (Step 0a/0b), not a priori. PCA is the safer round-trip (exact decode); scVI is
the stronger denoiser. We may end up using **scVI for the coupling geometry and PCA/expression for the
decoded target**, or scVI for both — Step 0 decides.

**Gene-space check:** scVI decoder output must cover the 946-panel genes (L1000 landmarks are common HVGs;
verify overlap; restrict/reorder to the panel before ranking).

---

## 2. Step 0 — three make-or-break gates BEFORE any training (cheap, CPU/1-GPU)

These are the senior-bioinformatician kill-shots: each is hours, not days, and any red light re-routes the
plan before we spend a training run.

- **0a. Does the drug signal survive the embedding?** For a stratified sample of conditions (identifiable /
  marginal / inert per our drug atlas Q11), compute the **MW₂ location shift** (‖μ_treated − μ_control‖ in
  embedding space) with a label-permutation null, plate-matched control. *Gate:* identifiable drugs must
  show a significant latent shift; if even they are ≈ 0, the embedding erased the drug → do not build OT
  there (switch embedding, or raise scVI dim, or abandon Option A for that embedding). Cross-check the
  ranking against the atlas potency (instrument validity).

- **0b. Round-trip decoder fidelity.** Encode→decode real cells; measure (i) panel **rank correlation /
  Jaccard** between decoded and true cell sentence, and (ii) **DE-recovery**: do decoded-treated vs
  decoded-control DE genes overlap the *real* DE genes? *Gate:* a decoded target must be a faithful cell
  sentence and must retain the real drug DEGs. (PCA passes (i) by construction; the real test is scVI.)

- **0c. Target sanity (the whole pipeline, no training).** For a few conditions, build OT-barycentric
  targets and check the **decoded target pseudobulk ≈ real treated pseudobulk** (MW₂ small) while the
  **per-control-cell target variance > 0** (targets differ across control cells — the property Arm 1a
  lacked). *Gate:* if targets collapse to the mean (variance ≈ 0), ε is too high / OT adds nothing.

- **0d. (pre-registration) Target-contrast magnitude.** Purely diagnostic, no training: for each OT target,
  count how many of the top-K ranked genes **differ from its control prompt's** top-K. Compare to the same
  count for Arm 1a consensus targets. *Prediction:* if OT targets differ from control by only a handful of
  genes (like consensus), Q13 predicts the objective still under-weights the drug and Arm 1c will **not**
  rescue drug-blindness — recorded in advance so the outcome is interpretable either way.

---

## 3. Population construction (matches A-05/A-09 and Mert)

- **Condition = (cell_line, drug, dose)** — dose kept **separate** (Mert does this; our A-05 fix does too).
- **Control pool = plate-matched DMSO** for each (cell_line, plate) (as the current preprocessor).
- **Compute OT strictly within (cell_line, plate)** so batch identity cannot drive the coupling.
- **≥ 100 cells per condition** (Mert's robustness threshold for OT); log what is dropped.
- **Train cells only** — eval tiers are held out by construction (no OT/embedding statistic sees an eval cell).
- Embeddings streamed once and cached per cell (join released scVI embeddings, or encode raw counts).

---

## 4. OT coupling (a monotone ladder of target constructions)

Cost = squared Euclidean in embedding space, `C_ij = ‖x_i − y_j‖²` (control x, treated y), as in Mert/HiRef.

Target constructions, from least to most structure — this **is** the experiment (isolates where any benefit
comes from, per the auditor):

1. **T0 consensus (= Arm 1a):** target = decode(ȳ) [treated mean]; control-independent. *Baseline / ε=∞.*
2. **T1 mean-displacement:** target = decode(x_i + (ȳ − x̄)); keeps the control's identity, adds one drug
   direction. No transport plan needed.
3. **T2 OT-barycentric:** entropic Sinkhorn plan π (uniform marginals); target = decode(Σ_j π_ij y_j / Σ_j π_ij)
   — a control-specific OT image; ε tuned so 0c passes (targets vary, don't collapse).
4. **T3 HiRef sub-population displacement:** hierarchical low-rank OT (Alvarez-Melis 2025) co-clusters
   control↔treated into matched sub-populations (S_k, T_k); target = decode(x_i + v_{k(i)}),
   v_k = mean(T_k) − mean(S_k). Restores **heterogeneity in the displacement** (different sub-states move
   differently). Rank schedules from Mert: [5,25]→K=125 for N≥250; [5,5]→K=25 for small conditions.

Orthogonal **meta-cell** axis (the other half of A-02): optionally replace targets by the average of their
k-NN in the treated set (extra denoising). Tested as an on/off ablation on T2/T3 — flagged because
over-smoothing can erase rare responses (auditor's caveat); we report retained heterogeneity (Bures term).

**Solver choice by size** (Mert Table 7): Sinkhorn `O(N²)` for typical conditions (≤ few hundred cells);
HiRef `O(N·r)` only where N is large. ε is the central hyperparameter (§0c gate).

---

## 5. Target generation (reuse the consensus skeleton)

Reuse `build_consensus_targets.py --emit per_cell` verbatim except the target source:
- keep every **prompt/control** (input unchanged) → example count preserved exactly (hard-checked, as Arm 1a);
- swap only the **response** to the decoded OT target: rank the decoded panel expression, take the top-K
  (K = median expressed-genes-per-cell for the condition, matching consensus), append `[END_CELL]`.
- Emit one JSONL per construction (T0..T3, ± meta-cell) so all arms are drop-in for the existing trainer.

---

## 6. Retraining (target is the only variable)

- **Cold-start from the same base** (`C2S-Scale-Pythia-1b-pt`), **identical hyperparameters** to Arm 1a and
  the single-cell run, so the *training target* is the sole independent variable.
- **Match training length** to the single-cell / Arm 1a run (step-matched head-to-head; Arm 1a's undertraining
  caveat is avoided by matching, or by running a full epoch for all arms — decide once, apply to all).
- Validity gate before trusting any score: `emits_end_cell ≈ 0.99`, sane length ratio, hallucination ≈ 0.

---

## 7. Evaluation (leak-immune battery + the new latent metric)

Every number **within-plate** (the methods-note protocol). Score all arms identically:

- **Existing leak-immune instruments:** within-plate NIR + **scramble** (same control, wrong drug token) +
  **control-copy** (zero-drug-info) + **output-invariance**. The decisive causal number stays
  `model − scramble` with clustered CIs.
- **New — MW₂ / Bures population metric (answers A-01):** in embedding space, score a prediction by its
  **MW₂ distance to its own drug's treated Gaussian vs to other drugs'** (a latent-space NIR), with
  **matched population depth** across model / truth / ceiling / baselines (the A-01 fix). Report the Bures
  (shape) term separately — does the model capture response *heterogeneity*, not just the mean shift?
- **Ablation table (the A-02 deliverable):** T0(consensus) vs T1 vs T2 vs T3, ± meta-cell. "Paired vs purely
  distributional" = T2/T3 vs T0; "unsmoothed vs smoothed" = meta-cell on/off. This isolates *where* any
  benefit originates instead of reporting one number.

---

## 8. Instrument validity (does the OT see real biology?)

- Displacement vectors `v_k` should correlate with the condition's **known DEGs**; MW₂ potency ranking should
  match the drug atlas (Q11) — potent targeted agents high, nutrient/inert compounds ≈ 0.
- Sub-population structure (T3) should recover meaningful axes (e.g. cell-cycle / EMT, per Mert) rather than
  batch — checked by coloring co-clusters by plate (must **not** separate by plate within a condition).

---

## 9. Risks, failure modes, and what each outcome means

| Risk / outcome | Meaning | Response |
|---|---|---|
| 0a red (no latent drug shift) | embedding erased the drug | switch embedding (PCA / higher-dim scVI); if all fail, Option A can't work in latent space → go Option B |
| 0b red (decode infidelity) | round-trip loses the sentence/DEGs | use PCA (exact) or keep coupling latent but decode from **expression-space barycentric** target |
| 0c collapse (targets ≈ mean) | ε too high → reduces to Arm 1a | lower ε; if only high-ε is stable, OT ≡ consensus (informative null) |
| 0d tiny target-contrast | Q13 prediction: objective will ignore it | pre-registered; a training null then *confirms* the objective diagnosis → Option B/Arm 1b |
| Trains, still `model−scramble`≈0 | target-side is exhausted (even causally-matched, heterogeneity-preserving targets don't help) | **strong result**: definitively redirects the thesis to the objective; OT infra feeds it |
| Trains, `model−scramble` > 0 with CI excluding 0 | **the pairing mattered** — first genuine drug use | characterize which construction (T1/T2/T3) and whether the Bures term is captured |

Either training outcome is publishable: a win is a positive result; a null with the T0→T3 ladder + the 0d
pre-registration is the cleanest possible proof that the defect is the objective, not the target.

---

## 10. Order of operations (kill-shots first)

1. Acquire/cache embeddings (scVI join or encode; PCA fit on train panel). Verify gene-space overlap.
2. **Step 0a/0b/0c/0d gates.** Stop or re-route on any red. *(These decide scVI vs PCA and whether to train.)*
3. Build T0..T3 (+meta-cell) target JSONLs on train cells (eval-disjoint).
4. Cold-start retrain each arm, matched hyperparameters/length; validity-gate generations.
5. Evaluate: leak-immune battery + MW₂/Bures; ablation table with clustered CIs.
6. Log to FINDINGS as Arm 1c; provenance-bundle the run.

## 11. What this sets up (beyond Arm 1c)

- **Option B (latent predictor):** the embeddings + MW₂ loss are already built here → a head that predicts in
  latent space, trained/evaluated by MW₂, is a small step from this infrastructure.
- **Arm 1b (objective-side):** the OT displacements/co-clusters are natural material for a contrastive /
  discrimination auxiliary loss that rewards drug-direction sensitivity (the Q13-motivated fix).
- **A-03 drug-specific baseline (Carlo):** the per-(drug,cell-line) displacement vectors computed here are
  exactly the train-only drug-lookup baseline; reuse them there.

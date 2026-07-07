# Results: Fine-Tuning C2S-Scale-Pythia-1b on Tahoe-100M

*Working results document. Numbers from the `checkpoint-10000` evaluation on the
`data_diverse2` dataset (K-sweep run), with the per-MOA baseline ladder and the
replicate noise ceiling. Living document — further ablations appended as run.*

---

## 1. Evaluation setup

The fine-tuned model (`C2S-Scale-Pythia-1b` SFT, checkpoint at 10,000 optimizer steps,
~43% of one epoch over 371,794 training pairs) was evaluated through four generalisation
tiers of `data_diverse2`:

- **Tier 1 — seen conditions** (in-distribution)
- **Tier 2 — unseen drugs** (50 drugs held entirely out of training; per-drug balanced)
- **Tier 3 — unseen drug × cell-line combos**
- **Tier 4 — dose interpolation** (held-out middle dose)

Each tier was scored on 300 sampled cells (subsample seed 42). The **headline metric is
DE-Δr**: Pearson correlation between predicted and true rank-shift (treated − control) over
the top-K differentially-expressed genes. A control-as-prediction baseline scores 0 by
construction, so DE-Δr isolates perturbation-prediction skill from baseline expression.
Uncertainty is a drug-clustered bootstrap (1,000 resamples; effective n = held-out drugs
per tier). DE-Δr is reported across a **K-sweep** (K = 20, 50, 100, 200). A guard nulls
DE-Δr (excluded from aggregates) for predictions below `min_coverage = 0.2`.

The **comparison ladder** of generic baselines, each scored on identical cells, ordered by
how much structure they encode:

- **control** — predict treated = control (DE-Δr ≡ 0).
- **global mean-shift** — control + global per-gene mean rank-shift (drug- and
  cell-line-independent).
- **per-cell-line mean-shift** — + per-cell-line mean shift (captures cell-line identity).
- **per-MOA mean-shift** — + per-mechanism mean shift (captures "what this drug class does").
- **per-MOA × cell-line mean-shift** — + per-(mechanism, cell-line) mean shift (the toughest
  generic comparator: knows both mechanism and cell line). Beating *this* isolates
  drug-specific signal above both mechanism and cell-line context.

---

## 2. Fine-tuned model performance

DE-Δr ≈ 0.72 at the headline K = 50, **flat across all four tiers**:

| Tier | DE-Δr (K=50) | 95% CI | n drugs | topN-expr τ | panel τ | coverage | halluc. |
|---|---|---|---|---|---|---|---|
| Tier 1 — seen conditions | 0.724 | [0.716, 0.732] | 69 | 0.256 | 0.549 | 0.88 | 0.037 |
| Tier 2 — unseen drugs | 0.725 | [0.719, 0.730] | 50 | 0.249 | 0.530 | 0.87 | 0.047 |
| Tier 3 — unseen combos | 0.724 | [0.717, 0.731] | 73 | 0.255 | 0.543 | 0.87 | 0.043 |
| Tier 4 — dose interpolation | 0.714 | [0.707, 0.722] | 18 | 0.245 | 0.540 | 0.87 | 0.042 |

Unseen-drug DE-Δr (0.725) is statistically indistinguishable from seen conditions (0.724);
unseen combos and dose interpolation are barely lower. Outputs are well-formed (coverage
≈ 0.87, ≈ 865 of 946 panel genes per response, hallucination ≈ 0.04, `n_degenerate` ≈ 0).

### 2.1 K-sweep (robustness of DE-Δr)

| Tier | K=20 | K=50 | K=100 | K=200 |
|---|---|---|---|---|
| Tier 1 — seen conditions | 0.754 | 0.724 | 0.691 | 0.644 |
| Tier 2 — unseen drugs | 0.756 | 0.725 | 0.687 | 0.641 |
| Tier 3 — unseen combos | 0.754 | 0.724 | 0.691 | 0.645 |
| Tier 4 — dose interpolation | 0.739 | 0.714 | 0.684 | 0.638 |

DE-Δr decays smoothly and identically across tiers (≈ 0.75 → 0.64): the model predicts the
strongest movers best, getting noisier on weaker movers out to K=200. The headline K=50 is a
fair mid-range choice; the *shape* is the same on unseen drugs as on seen conditions, so the
generalisation gap does not widen with K. (§4 shows this decay tracks the noise ceiling.)

---

## 3. Drug-specificity: the baseline ladder

Paired Δ in DE-Δr space (model − comparator) at K = 50; all Wilcoxon p ≪ 1e-40, all CIs
exclude 0:

| Tier | vs control | vs global MS | vs per-cell-line MS | vs per-MOA MS | vs per-MOA×CL MS *(toughest)* |
|---|---|---|---|---|---|
| Tier 1 — seen | +0.724 | +0.630 | +0.363 | +0.632 | **+0.389 [0.361, 0.422]** |
| Tier 2 — unseen drugs | +0.725 | +0.662 | +0.526 | +0.663 | **+0.545 [0.504, 0.586]** |
| Tier 3 — unseen combos | +0.721 | +0.660 | +0.545 | +0.661 | **+0.558 [0.539, 0.577]** |
| Tier 4 — dose interp. | +0.714 | +0.685 | +0.596 | +0.689 | **+0.588 [0.558, 0.616]** |

The ladder is monotonic and complete: each baseline that captures more structure leaves a
smaller — but still large and significant — residual for the model to beat.

**The model is drug-specific, not mechanism-specific.** Beating per-MOA mean-shift by
+0.63 to +0.69 shows the model carries signal *above* the mechanism-class average — it is
not merely learning "MEK inhibitors upregulate these genes." The decisive test is **per-MOA
× cell-line** (knows both mechanism and cell line): the model still beats it by **+0.39
(seen) rising to +0.59 (dose)**, all CIs excluding zero. This is the strongest
drug-specificity statement available — even against a baseline conditioned on both
mechanism and cell line, the model adds substantial drug-specific signal, and the margin
*grows on the harder tiers* (unseen drugs/combos/dose), indicating transferable drug-level
response learning rather than reliance on mechanism or cell-line priors.

(Note: per-MOA (+0.66 on Tier 1) is a *weaker* baseline than per-cell-line (+0.36), i.e.
cell-line identity is more informative at the rank level than mechanism is — which is why
per-MOA×cell-line, combining both, is the comparator to lead with.)

---

## 4. Noise ceiling (replicate-to-replicate, per-tier, condition-matched)

Two real treated cells from the same condition do not agree perfectly — biological
stochasticity (transcriptional bursting, cell-cycle) and technical noise (capture,
amplification) impose an **irreducible** gap below a perfect score. The ceiling is measured
**per tier on exactly the conditions the model was evaluated on** (computed from the eval
files themselves, so the population matches), with two variants:

- **cell-vs-cell** — one real treated cell scored against another from the same condition.
  This is the ceiling *most directly comparable to how the model is scored* (single-cell
  truth, DE genes selected from that truth cell), and is the one cited below.
- **cell-vs-consensus** — a cell scored against the leave-one-out rank-consensus of the
  other cells in the condition (a denoised, condition-representative truth in rank space).

DE genes are selected from the truth, and the same plate-matched controls are used, exactly
as in the real eval. Replicate coverage is healthy: Tier 1 = 1,448 conditions (median 2
cells, max 7), Tier 2 = 1,306 (median 3, max 12), Tier 3 = 211 (median 13, max 20),
Tier 4 = 720 (median 6, max 20). Medians [IQR] over thousands of pairs per tier.

**DE-Δr vs the cell-vs-cell ceiling (K = 50):**

| Tier | Model | Ceiling (cell-vs-cell) | Model / ceiling |
|---|---|---|---|
| Tier 1 — seen conditions | 0.724 | 0.764 | ~95% |
| Tier 2 — unseen drugs | 0.725 | 0.768 | ~94% |
| Tier 3 — unseen combos | 0.724 | 0.769 | ~94% |
| Tier 4 — dose interpolation | 0.714 | 0.785 | ~91% |

**The model sits at ~91–95% of the single-cell replicate ceiling on every tier, on matched
conditions.** The residual to a perfect score is overwhelmingly irreducible single-cell
noise, not model error — near-optimal perturbation prediction. Across the full K-sweep the
model is **below the cell-vs-cell ceiling at every K on every tier** (e.g. Tier 2 K=200:
model 0.641 vs ceiling 0.705), and the model's K-decay tracks the ceiling's own decay,
confirming the large-K falloff is a property of the data (weak movers are intrinsically
noisy), not a model weakness.

**topN-τ matches the replicate ceiling — the tail-noise interpretation, now a measured,
per-tier fact.** At N=100 the cell-vs-cell ceiling is 0.27 (T1), 0.26 (T2), 0.25 (T3), 0.23
(T4); the model is ≈0.25 throughout. Two real cells of the same condition agree only at
τ≈0.25–0.27 on their top-expressed genes, so the model's 0.25 is the intrinsic noise floor
of saturated housekeeping genes, not model failure. The earlier "housekeeping genes are
near-noise" interpretation is confirmed by measurement.

**Headroom against a denoised truth (the honest "room to improve").** The cell-vs-consensus
ceiling *rises* with K and N where cell-vs-cell falls — averaging cells denoises the broadly-
expressed and weakly-moved genes. At large K the model is well below this denoised ceiling
(e.g. Tier 3 K=200: model 0.645 vs consensus 0.771; Tier 4 topN-τ N=200: model ≈0.25 vs
consensus 0.50). Interpretation: against *single-cell* truth (how it is scored) the model is
near-optimal; against a *condition-level denoised* truth there is visible headroom,
concentrated in the weaker movers and the expressed tail. This is the natural target for the
knowledge-injection stage — DE-Δr on strong movers is near-saturated, so gains should be
sought in denoised/condition-level response rather than in the headline DE-Δr K=50.

*Caveat on precision:* Tier 1's median is 2 cells/condition, so its cell-vs-consensus column
is barely denoised (leave-one-out of 2 = 1 cell) — the cited **cell-vs-cell** column is
unaffected and well-sampled (8,688 pairs). IQRs are wide (single-cell noise), so report the
qualitative "~91–95% of ceiling / at the topN-τ floor," not third-decimal fractions.

*(panel τ is not used as a ceiling comparison: two real cells share the identical
deterministic ordering of the unexpressed tail, inflating their panel τ to ≈0.71 by a
formatting convention the model does not reproduce token-for-token; this is why DE-Δr, not
panel τ, is the headline metric. See dataset methods §4.)*

---

## 5. Ablation: base C2S-Scale (no fine-tuning)

The pretrained base model (`vandijklab/C2S-Scale-Pythia-1b-pt`) **does not perform the
perturbation task in this format**: coverage = 0.00, hallucination ≈ 1.00, zero valid panel
genes. Inspection shows it emits **cell-type annotations** ("Cell 1: T cell. …"), not gene
sentences — the `-pt` checkpoint is the C2S foundation model (cell-type prediction,
conditioned generation), and C2S-Scale performs perturbation prediction only after
task-specific fine-tuning (their `PerturbationPromptFormatter`, repo tutorial 10).

A zero-shot base DE-Δr is therefore **undefined by design** and the fine-tuned − base DE-Δr
difference is not reported as a comparison. (Unguarded, the base records a spurious DE-Δr
≈ 0.98: emitting no valid genes sends every gene to the worst rank, making the predicted
shift a near-constant function of the control rank that correlates with truth — the coverage
guard nulls this.) The legitimate base comparison is a format/validity floor:

| Metric | Fine-tuned | Base (no FT) |
|---|---|---|
| Coverage | 0.87 | 0.00 |
| Hallucination | 0.04 | ≈ 1.00 |
| panel τ | ~0.53 | ~0.05 |
| top-expressed τ | ~0.25 | ~0.03 |
| DE-Δr | 0.72 | *undefined (degenerate)* |

The fine-tuning contribution is stated as the lift from an unusable-format baseline
(cell-type annotations, coverage 0) to coverage 0.87 / DE-Δr 0.72. The Pythia-base (no-C2S)
SFT ablation — isolating the value of C2S pretraining under an identical recipe — is in
progress (§7).

---

## 6. Summary so far

- **DE-Δr ≈ 0.72 at K=50, flat across all four tiers**, decaying smoothly over the K-sweep —
  and **at ~91–95% of the single-cell replicate ceiling on every tier** (matched conditions,
  below ceiling at all K). The residual is overwhelmingly irreducible single-cell noise.
- **Drug-specific above mechanism and cell line**: the model beats a per-MOA × cell-line
  baseline by +0.39 (seen) to +0.59 (dose), all CIs excluding zero, with the margin growing
  on harder tiers — drug-level resolution, not mechanism- or cell-line-level.
- **topN-τ (0.25) matches the replicate ceiling (0.23–0.27 per tier)**: the modest absolute
  value is the intrinsic noise floor of saturated genes, now confirmed by measurement.
- **Headroom is against the *denoised* (consensus) truth, not the single-cell truth**: at
  large K/N the model trails the consensus ceiling — the natural target for knowledge
  injection, since DE-Δr on strong movers is already near-saturated.
- The **base model cannot perform the task in this format** (cell-type annotation); its DE-Δr
  is undefined by design.
- All from `checkpoint-10000` (~43% of one epoch).

---

## 7. Open items / to iterate

- **Prompt-scramble ablation (queued, GPU):** swap the drug/MOA token in each prompt
  (different-MOA and random-drug modes), keeping control/cell-line/dose/truth fixed. A drop
  in DE-Δr confirms the model reads and uses the drug token; no drop would mean it ignores
  drug identity. The decisive direct test of drug-specificity, complementing the per-MOA
  ladder.
- **Pythia-base (no-C2S) SFT ablation (in progress):** identical recipe on raw
  `EleutherAI/pythia-1b` — isolates the value of C2S pretraining. Compare DE-Δr at
  checkpoint-10000.
- **C2S-Scale native perturbation baseline:** reproduce the authors'
  `PerturbationPromptFormatter` recipe on the same data/eval — isolates the contribution of
  the leak-free fixed-panel design vs the authors' native recipe.
- **Per-drug DE-Δr breakdown:** per-drug performance vs MOA class and training
  representation — turns the aggregate into a finding (which drug classes are well/poorly
  predicted; is Tier-2 generalisation MOA-mediated?).
- **Resume ablation:** later checkpoint / full epoch — is 0.72 a plateau or still climbing?
  (Given the ceiling, headroom is small — worth confirming.)
- **Spearman K-sweep:** already computed (`de_delta_spearman_k*`); surface it as a robustness
  check (Spearman ≈ 0.68 vs Pearson ≈ 0.72 — close, no single-gene-outlier artifact).
- **valid_rate definition:** the strict ~0.18 reflects the ≥95%-coverage / ≤5%-halluc bar;
  coverage 0.87 is the figure to report.

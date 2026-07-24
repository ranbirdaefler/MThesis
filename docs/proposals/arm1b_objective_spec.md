# Arm 1b — Objective/representation fix for drug-specific perturbation prediction

**Status:** spec (pre-implementation). Every design choice below is tied to a measured number from
Q1–Q14, the scramble-distance sweep, the SAR gate, or the target-divergence check.

---

## 1. The diagnosis (measured, corrected)

**The information is present. The output encoding throws it away.**

`target_divergence.py` on the 620k-cell cache (6,602 conditions, 125 groups):

| representation | inter-drug top-200 tokens DIFFERING | replicate retrieval ceiling |
|---|---|---|
| **FULL profile (current target)** | **34.6 / 200 (17%)** | 0.742 |
| SHIFT (treated − control) | 111.4 (56%) | 0.742 |
| **RESIDUAL (drug-specific)** | **120.3 (60%)** | 0.747 |

Two things follow, and the second is a correction to an earlier hypothesis:

1. **The bottleneck is tokenization.** A cell sentence ranks genes by expression, so the emitted tokens
   are dominated by housekeeping genes that are identical across drugs. **83% of the target tokens are
   the same for any two drugs in the same context** → cross-entropy has almost nothing to learn drug
   identity from. This explains why *every* target-content change failed identically (Q12 consensus,
   Q14 OT, the whole ε-ladder): they all changed *what* the target was, never *what the tokens encode*.
2. ❌ **RETRACTED: "residualization raises the ceiling."** It does not (0.742 → 0.747), and it cannot:
   subtracting a per-group constant leaves pairwise distances unchanged (`r_A − r_B = s_A − s_B`). The
   earlier 0.576-vs-0.805 comparison was a cross-setup artifact. The ceiling is a property of the data
   (~0.74), not of the representation.

**Corollary that shapes the design:** since the *continuous* profile carries drug identity at 0.742
under every representation, a loss computed on the **continuous predicted profile** bypasses the
tokenization bottleneck entirely. Residual targets fix what the tokens encode; a profile-level
contrastive loss sidesteps tokens altogether. **Both attack the same bottleneck from opposite sides.**

Mechanistic corroboration (Q13): the readout is magnitude-sensitive and direction-blind — drug
decodability rises to 0.88 by layer 16 while causal variance share stays ~3%. A model that receives ~1%
of its gradient from drug identity learning to ignore drug direction is the expected outcome.

---

## 2. What is NOT the fix (each closed by measurement)

| hypothesis | closed by | number |
|---|---|---|
| target is too noisy → denoise | Q12 consensus | `model−scramble` −0.010 |
| target pairing is wrong → OT state-matching | Q14 T2 | −0.001 |
| any point on the denoising ladder | ε sweep | all null |
| the scramble swaps similar drugs (artifact) | scramble-distance sweep | null in every distance bin, far tail −0.019 |
| chemistry can supply drug identity | SAR gate | retrieval 0.474/0.500 vs ceiling 0.805 |

The SAR negative also **scopes** this arm: molecular structure does not predict response here, so
**unseen-drug generalization is out of reach**. Arm 1b is a *seen-drug* experiment (tier1/tier3), and
that limitation is a finding, not an omission.

---

## 3. The design — three components

### ① Residual targets (fixes what the tokens encode)
Target = the **drug-specific residual**: `(treated − control) − mean_over_drugs(treated − control)`,
genes ranked by **|residual|**, top-K → `[END_CELL]` sentence. 60% of tokens now drug-specific vs 17%.
- **Scope = cell line** (not plate): reproducibility 62% vs 19% — plate has too few drugs to estimate
  the mean, so subtracting a noisy mean injects noise.
- **Reliability weighting:** weight each condition by `cos(res_halfA, res_halfB)` (or filter at >0.2).
  38% of conditions have an irreproducible residual; training on them is training on noise.
- Prompt unchanged (control cell + drug/dose/MoA), so examples stay drop-in for the trainer.

### ② Profile-level contrastive loss (bypasses tokenization) — *now the most important component*
From the logits, build a **differentiable expected profile**: `score(g) = Σ_i P(token_i = g) · w(i)`
with a rank weight `w(i)`. Apply **InfoNCE**: the predicted profile for drug A must be closer to A's true
residual than to any other drug's.
- **This is literally training the NIR metric**, and it is differentiable — no RL, no reward model.
- **Batch construction is load-bearing:** batches must contain **multiple drugs from the same
  (cell_line, plate)**, so negatives are *hard* and the generic program cancels. With random batching the
  negatives are separable by cell line and the loss learns cell line, not drug — the exact failure we are
  trying to fix.

### ③ Drug-conditioned modulation (makes the drug causal)
A **learned per-drug embedding** (not chemistry — SAR closed that) generating a low-rank/FiLM modulation
of the transformer's mid-layer projections (Q13 localizes the decodable-but-unused drug signal around
layers 4–9). The drug parameterizes the *transformation* control→treated rather than sitting in the
residual stream as an input the readout can round away.

---

## 4. Staging (stop as soon as it works, so we know which piece did it)

| stage | components | isolates |
|---|---|---|
| 1 | ① alone | does fixing the token encoding suffice? |
| 2 | ① + ② | does the profile-level objective add on top? |
| 3 | ① + ② + ③ | does causal conditioning add on top? |

Cold start from `C2S-Scale-Pythia-1b-pt`, identical hyperparameters to every prior arm, 1 epoch,
checkpoints every 1000 steps. The target/loss/architecture is the only variable.

---

## 5. Evaluation (unchanged instruments — no moving the goalposts)

- **Primary:** within-plate NIR + **scramble** (`model − scramble`, clustered CI) — null in *every* arm so
  far, and validated against the swap-distance confound.
- **Win condition:** **`model > drug_lookup`** (A-03 train-only per-drug displacement). A lookup can
  reproduce a memorized average; only a model can tailor the drug response to the control cell's state.
  Beating chance is not enough — residual targets may be trivially memorizable.
- **Reconstruction for full-profile metrics:** `control_cell + generic_shift(cell line) + predicted_residual`.
  ⚠️ This hands the model the generic component, so **absolute metrics (DE-Δr, panel-τ) are inflated and
  NOT comparable across arms**. NIR/scramble are unaffected (the generic cancels in discrimination).
- Validity gate first (`emits_end_cell ≈ 0.99`, hallucination ≈ 0) before trusting any score.
- Report on the **identifiable subset** (Q11) — 27% of drugs are inert; aggregate dilutes the signal.

---

## 6. Risks, stated up front

| risk | mitigation / how we'd know |
|---|---|
| **Memorization** — residual targets may be a per-drug lookup | that is exactly why the bar is `model > drug_lookup`, not `> chance` |
| **38% of residuals are noise** | reliability weighting/filtering (§①) |
| **Only 106 drugs** in the cache | widen shards before any final claim; fine for the go/no-go |
| **Reconstruction inflates absolute metrics** | primary metrics are discrimination-based; stated explicitly |
| **Ceiling ~0.74** — nothing can exceed it | success = closing the gap to it, not reaching 1.0 |
| **Leak-proofing of residual eval is UNVERIFIED** | the divergence-check baselines were degenerate; verify with per-drug controls in `nir_benchmark` |

---

## 7. Pre-registered interpretation

- **`model − scramble` > 0 (CI excludes 0) and `> drug_lookup`** → the objective/representation was the
  bottleneck; the first genuine drug-specific prediction in this project.
- **> 0 but ≤ drug_lookup** → the model memorizes per-drug averages but adds no cell-state tailoring; an
  honest partial result.
- **Still ≈ 0 after all three stages** → the failure is not target, not pairing, not chemistry, not
  encoding, and not conditioning. Combined with Q13, that is a strong claim about the limits of
  autoregressive cell-sentence models for this task — and the ceiling/decodability numbers make it
  quantitative rather than anecdotal.

# GRPO training plan — drug-specific perturbation prediction

**Status:** design draft, for iteration. Every choice below is either tied to a measured number from
`FINDINGS.md` or flagged as an open decision.

---

## 1. Objective

One model, evaluated exactly the way every prior arm was evaluated (standard `[END_CELL]` output,
standard tiers, within-plate NIR), that satisfies **both**:

| criterion | target | current best |
|---|---|---|
| **Predictive** — beats the simple baselines on the calibrated metric | NIR **> 0.55** vs linear/control-copy ~0.50 | 0.498 (single-cell), 0.511 (OT) |
| **Drug-specific** — provably uses the drug | `model − scramble` **> 0**, clustered CI excludes 0, and the gap **grows** with swap dissimilarity | +0.014 [−0.016,+0.042] (single-cell, null) |
| **Beats a drug lookup** | `model > drug_lookup` | model 0.199 < lookup 0.232 (tier1) |

Either criterion alone is insufficient: NIR without drug specificity is the control-copy leak; drug
specificity without NIR is a model that reacts to the token but predicts poorly.

---

## 2. Why GRPO, specifically

Q15 established the mechanism blocking SFT: a cell sentence ranked by expression has only **34.6 of
200 tokens differing between two drugs (17%)**, so token-level cross-entropy gives drug identity ~1% of
the gradient. Every target-content change (Q12 consensus, Q14 OT, the full ε-ladder) failed identically
because none of them changed that ratio.

**GRPO does not have this problem.** It computes a single reward for the *whole generated sequence* and
multiplies every token's log-probability by that advantage. A more drug-specific generation gets its
entire sequence up-weighted. The dilution ratio never enters the gradient.

That is the core argument for this arm: **the reward is a sequence-level object, so we can score exactly
the thing we care about (drug discrimination) instead of a token-level proxy that buries it.**

It also keeps the output format unchanged → all prior tier-1/2/3/4 numbers remain directly comparable,
and no reconstruction step is needed.

---

## 3. Setup

- **Policy:** `C2S-Scale-Pythia-1b` SFT checkpoint (warm start decided in §6). Output = ordinary
  `[END_CELL]` cell sentence. **Format never changes.**
- **Reference model:** frozen copy of the warm start, for the KL penalty.
- **Prompt:** unchanged (`cell line, drug, dose, MoA, control cell`).
- **Group:** K samples of the *same* prompt (standard GRPO); advantage = `(R_i − mean_K) / std_K`.
  Per-prompt normalization is a good fit here because achievable reward varies hugely by drug (Q11: 27%
  of drugs are inert) — group-relative advantage cancels that automatically.

---

## 4. The reward

### 4.1 Turning a generated cell into a drug-specific quantity

Precomputed per condition (from the `ot_cache`, all in log1p CP10K panel space):
`ctrl_pb` (plate-matched control pseudobulk), `generic[cell_line]` (mean-over-drugs shift),
`residual_true` (the drug-specific residual), and `residual_other[]` for the other drugs in that cell line.

For a generated sentence *g*:
```
expr(g)      = decode(g) via the empirical rank→value profile   (already implemented)
shift(g)     = expr(g) − ctrl_pb
residual(g)  = shift(g) − generic[cell_line]
```

### 4.2 Contrastive discrimination term (primary)

```
R_disc = sim(residual(g), residual_true) − max_over_j sim(residual(g), residual_other_j)
```

Why this form and not plain similarity to the truth:
- **It cancels the generic program by construction.** Any component shared across drugs raises both
  terms and nets to zero — the model cannot farm reward by predicting the average response. This is the
  same reason NIR is the only calibrated metric (Q10) and why `model − scramble` is leak-immune.
- **It is dense.** Unlike rank-based NIR, it produces gradient even when every sample in the group is
  mediocre — which matters at cold start.
- **Hard negatives.** Use the **opposite-signature** drug (most anti-correlated residual) as the primary
  negative, since the stratified scramble showed that is where discrimination is most detectable
  (+0.143 vs +0.035 for a similar-drug swap). Sampling `max over j` approximates this; restricting `j`
  to the bottom-quartile-cosine drugs makes it explicit and cheaper.

**Open decision — the similarity function.** A single generated cell is noisy against a 40-cell
pseudobulk truth. Two candidates, to be settled offline (§5):
- `cosine` on the residual vectors — smooth, but noisy for a single sparse cell.
- **rank-weighted set overlap** between the generation's top-N shifted genes and the truth's top-N
  up/down sets — more robust to single-cell noise, slightly less smooth.

### 4.3 Validity term (anti-hacking)

```
R_valid = −λ_dup · dup_rate − λ_oov · out_of_panel_rate − λ_len · |len − target_len| / target_len
```
Cheap, and it closes the obvious degenerate strategies (repeat one high-value gene, emit a stub, drift
out of the panel vocabulary). The measured baseline for a healthy model is ~99% valid panel genes,
<2% duplicates.

### 4.4 KL anchor

```
R_total = R_disc + R_valid − β · KL(π ‖ π_ref)
```
Standard, and non-optional here: the SFT model's fluency is the only thing keeping generations
biologically plausible. Without the anchor, RL will happily trade realism for reward.

---

## 5. STEP 0 — offline reward calibration (gate; do this before any GPU-hours)

**The single most important step in this plan.** If the reward cannot distinguish a *real* drug-A cell
from a *real* drug-B cell, no amount of RL will help — we would be optimizing noise. This is the same
discipline as the SAR gate and the Step-0 gates, and it is cheap (no model, no training).

Using held-out real cells:
1. **Discrimination:** compute `R_disc` for real cells of drug A scored against A's truth vs against
   other drugs' truths. **Gate: reward for own-drug cells must exceed other-drug cells with a clear
   margin.**
2. **Single-cell noise:** how far does `R_disc` degrade going from a 40-cell pseudobulk to a single
   cell? This directly sets **K** (how many samples we need for a usable advantage estimate).
3. **Variant selection:** run (1) and (2) for cosine vs rank-overlap; pick the winner.
4. **Hackability:** compute `R_disc` for adversarial inputs — the control copied verbatim, the generic
   response, a duplicate-heavy sentence, a truncated stub. **Gate: all must score below real cells.**
5. **Difficulty stratification:** report the margin separately for identifiable / marginal / inert drugs
   (Q11). This tells us which conditions are trainable and feeds the curriculum (§7).

*If step 5 shows the margin is near zero for most drugs, we restrict GRPO to the identifiable subset and
say so — training on inert drugs is training on noise.*

---

## 6. Warm start (open decision — three candidates)

GRPO amplifies variance that already exists; it is bad at creating signal from nothing. The starting
checkpoint therefore matters more than usual.

| candidate | drug signal at start | risk |
|---|---|---|
| single-cell SFT | **null** (−0.020) | cold start: low reward variance → weak advantage |
| **OT/T2 SFT** | **+0.0263** [+0.007,+0.048] | mild; currently the best *format-preserving* checkpoint |
| **DE-weighted SFT (new, ~3h)** | untested, plausibly best | one extra training run |

**The DE-weighted warm start is the clever option** and I would build it: a short SFT on *ordinary cell
sentences* (format preserved) with the token-level loss **up-weighted on genes in that drug's residual
signature**. This applies the Q15 insight — fix the gradient ratio — *without* changing the output
format, so it is a legal GRPO starting point. It is the missing bridge between "residual SFT proved the
point" and "GRPO needs a warm start."

The residual-SFT model itself is **not** a valid warm start: different output format.

---

## 7. Training loop

- **Curriculum by difficulty.** Start on conditions with reproducible residuals (`cos(res_A,res_B) > 0.2`,
  62% of conditions) and identifiable drugs (Q11 ceiling ≥ 0.8), then widen. Rationale: 27% of drugs are
  statistically inert — their reward is pure noise and would only add gradient variance.
- **Reliability-weighted prompt sampling**: sample conditions with probability ∝ `cos(res_A,res_B)`.
- **Hard-negative refresh**: recompute the opposite-signature negative per condition once, cache it.
- **Held-out conditions never sampled** (tier-aligned holdout manifest already built).
- Starting hyperparameters (to tune): K = 8–16, temperature 0.8–1.0, lr ~1e-6 (an order below SFT),
  β_KL ~0.02–0.1, batch = 8–16 prompts/step.
- **Generation budget:** ~200 gene tokens ≈ 500–800 BPE tokens. *This bit us once already — a 600-token
  cap silently truncated 26% of generations and halved a measured effect.* Set ≥1400 and log the
  `[END_CELL]` completion rate every eval.

---

## 8. Failure modes and mitigations

| # | failure | why it would happen here | mitigation |
|---|---|---|---|
| 1 | **Cold start** — all K samples equally drug-agnostic, advantage ≈ 0 | the SFT model is drug-blind | DE-weighted warm start (§6); dense reward; higher temperature; larger K |
| 2 | **Reward hacking** — degenerate outputs | RL optimizes the letter of the reward | validity term (§4.3); KL anchor; adversarial gate in §5.4 |
| 3 | **Generic-program farming** — reward without drug specificity | shared response is most of the signal | contrastive form cancels it by construction |
| 4 | **Noisy reward** — single generated cell vs pseudobulk truth | resolution mismatch | §5.2 sets K empirically; rank-overlap variant if cosine is too noisy |
| 5 | **Training on noise** — inert/irreproducible conditions | 27% inert, 38% irreproducible residuals | reliability filter + curriculum |
| 6 | **Fluency collapse** | RL trades realism for reward | KL anchor; validity monitored every N steps with a hard stop |
| 7 | **Cell-line shortcut** | model learns context, not drug | negatives are drugs in the *same* cell line, so context cancels |
| 8 | **Overfitting to trained conditions** | reward needs the true residual | tier-aligned holdout; evaluate on unseen combos/drugs |
| 9 | **Silent truncation** | long sentences | ≥1400 token budget + completion-rate logging |

---

## 9. Monitoring (every N steps, on a fixed held-out slice)

- mean reward, and its decomposition (`R_disc` vs `R_valid` vs KL)
- **generation validity**: `[END_CELL]` rate, duplicate rate, out-of-panel rate, length
- **`model − scramble` (stratified)** on held-out conditions — the actual objective
- **prediction diversity** across drugs (mode-collapse guard; truths sit at cos ≈ −0.005)
- KL from reference (drift guard)

**Hard stop** if validity drops below ~0.95 or diversity collapses (pairwise cos → 1).

---

## 10. Evaluation (unchanged, so results stay comparable)

Standard tiers, within-plate, `nir_benchmark` + `drug_stratify_geometry`, reporting:
`model`, `scramble` (stratified near/orth/opposite), `control-copy`, `linear`, **`drug_lookup`**,
`mean`, `ceiling` — and **broken out by drug-difficulty stratum**, not aggregated over inert drugs.

---

## 11. Staging

| stage | what | gate to proceed |
|---|---|---|
| **0** | Offline reward calibration (§5) | reward separates real drugs; adversarial inputs score low |
| **1** | DE-weighted warm-start SFT (§6) | format valid; ideally non-zero `model − scramble` |
| **2** | GRPO, identifiable drugs only, small scale | reward rises, validity holds, no collapse |
| **3** | GRPO, widened curriculum, full run | held-out `model − scramble` > 0 |
| **4** | Full standard-tier evaluation | the §1 table |

---

## 12. Open questions for iteration

1. **Similarity function** — cosine vs rank-overlap (settled empirically in §5).
2. **Warm start** — build the DE-weighted SFT, or start from OT/T2 to save a run?
3. **Reward on single generations vs a pooled pseudobulk of the K samples.** Pooling reduces noise but
   gives every sample in the group the *same* reward → zero advantage variance → no learning. A hybrid
   (per-sample reward, but truth pooled) is what is written above; worth stating explicitly as a choice.
4. **Should the KL anchor be to the SFT model or to the DE-weighted warm start?** (Probably the latter.)
5. **Token-level credit assignment** — GRPO credits all tokens equally. A per-token bonus for emitting
   genes in the true signature would sharpen credit but adds complexity; park unless stage 2 stalls.

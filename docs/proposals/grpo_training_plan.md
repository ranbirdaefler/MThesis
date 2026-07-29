# GRPO training plan — drug-specific perturbation prediction (v2)

**Status:** design, hardened against a literature review (10 parallel searches + adversarial
verification). **Source-confidence caveat:** the verification pass ran during a network outage, so most
primary sources could **not** be re-opened; findings below are marked ⚠ where the citation is
*unconfirmed*. Anything load-bearing is either (a) verified locally in this repo, (b) derived from the
estimator algebra, or (c) demoted to "check before relying on it."

---

## 1. Objective

One model, evaluated exactly as every prior arm was (standard `[END_CELL]` output, standard tiers,
within-plate NIR), satisfying **both**:

| criterion | target | current best |
|---|---|---|
| **Predictive** — beats simple baselines on the calibrated metric | NIR **> 0.55** vs linear/control-copy ~0.50 | 0.498 single-cell, 0.511 OT |
| **Drug-specific** | `model − scramble` > 0, clustered CI excludes 0, gap **grows** with swap dissimilarity | +0.014 [−0.016,+0.042] (null) |
| **Beats a drug lookup** | `model > drug_lookup` | model 0.199 < lookup 0.232 (tier1) |

---

## 2. Why GRPO

Q15: only **34.6/200 target tokens differ between two drugs**, so token-level CE gives drug identity ~1%
of the gradient; every target-*content* change failed identically. GRPO computes **one reward per
sequence** and multiplies every token's log-prob by that advantage — **the dilution ratio never enters
the gradient.** It also preserves the output format, so all prior tier numbers stay comparable.

⚠ **No published result shows RL beating SFT on drug-identity discrimination in perturbation
prediction.** This mechanism argument is simultaneously the contribution and the risk.

---

## 3. Precedent and honest prior

⚠ *unconfirmed citation* — **C2S-Scale (bioRxiv 2025.04.14.648850v4) reportedly ran GRPO on this exact
model family/format/scale** with a programmatic reward (Kendall τ over 40 apoptosis genes; negative MSE
in scGPT space), reporting τ **+9.2% at 410M, +4.9% at 1B**. Three discounts:

1. **Their metrics are the family our own instrument says is inverted.** Q10 within-plate: `panel_tau`
   DRF −0.231, and a drug-agnostic leave-one-out mean scores **0.694** vs a real replicate of the drug
   itself at **0.623**; Q4: DE-Δr 0.740 real vs 0.739 scramble. **A +4.9% τ gain is fully compatible with
   zero drug-specificity gain.** Evidence for profile fidelity, not discrimination.
2. The gain roughly **halves from 410M → 1B**; we are at 1B.
3. Their reward is *absolute* similarity-to-own-truth — structurally our **control** arm, not our
   treatment arm.

**Calibrated priors (mine, not literature-derived):** P(reward rises, validity holds) ≈ 0.8 ·
P(held-out `model − scramble` > 0 with CI excluding zero) ≈ **0.25–0.35** · P(the full §1 table) ≈ **0.15**.
**The warm start dominates every GRPO knob** — GRPO amplifies variance that already exists.

---

## 4. Setup

- **Policy:** `C2S-Scale-Pythia-1b`, warm-started per §7. Output = ordinary `[END_CELL]` cell sentence.
- **Reference (KL anchor):** the **DE-weighted warm start** (the actual init) — *not* the base SFT, or the
  anchor fights stage 1.
- **Group:** K samples of the same prompt.

---

## 5. Reward

### 5.1 Generated cell → drug-specific quantity
Precomputed per condition from `ot_cache` (log1p CP10K panel space): `ctrl_pb`, `generic[cell_line]`,
`residual_true`, `residual_other[]`.
```
expr(g)     = decode(g) via the empirical rank→value profile
residual(g) = expr(g) − ctrl_pb − generic[cell_line]
```

### 5.2 Contrastive discrimination term
```
R_disc = sim(residual(g), residual_true) − max_over_j sim(residual(g), residual_other_j)
```
- Cancels the generic program **by construction** — the model cannot farm reward by predicting the
  average response (same logic that makes NIR calibrated and `model − scramble` leak-immune).
- Dense → gradient even when all K samples are mediocre.
- **Negatives: frozen per condition, cached, identical across rollouts and steps.** This is load-bearing,
  not hygiene: under mean-only centering (§6.1) *any per-prompt constant cancels exactly* — including the
  per-drug baseline offset **and** the upward bias of `max_over_j` (E[max of J noisy sims] ≈ σ√(2 ln J),
  and J varies 20–204 per cell line). That cancellation requires J and the negative set to be constant
  within a group.
- **Similarity function (cosine vs rank-weighted set overlap): no literature guidance exists.** Settle in
  Step 0.

### 5.3 Validity term — with a dead-band, never a cliff
```
R_valid = −λ_dup·dup_rate − λ_oov·oov_rate + R_len
R_len   = 0                                          if len ∈ [0.8, 1.2]·target_len
        = −λ_len·(excess beyond band)/target_len      otherwise
```
**Why a dead-band.** ⚠ A reward term is amplifier-safe only if its within-group variance vanishes once
mastered. `dup_rate` and `oov_rate` do (measured: 98.8% valid genes, 1.7% dups). **`|len − target|` never
vanishes under temperature sampling** — so once `R_disc` plateaus, residual length variance becomes the
dominant within-group spread and the policy is optimized to hit a length number. A hard cliff is
separately wrong under group-relative advantages: two samples one gene apart get opposite full-magnitude
advantages.

**Length is mechanically coupled to the reward**, not just cosmetic: `expr(g)` is built from the rank→value
profile, so sentence length moves `residual(g)` directly. (This repo already has the scar — a 600-token cap
truncated 26% of generations and *halved* a measured effect, Q15.)

### 5.4 Invalid generations get a floor, they are **not** dropped
`R = R_floor` (constant below any valid reward) for: no `[END_CELL]`, dup > 0.2, oov > 0.1. Dropping
shrinks effective K and shifts the group mean under mean-only centering, silently inflating survivors'
advantages. *(Follows from the estimator; no source needed.)*

### 5.5 Do **not** per-component-normalize (reject group-z-scoring each component)
Our validity terms start near-saturated, so 1/std makes validity's weight *escalate precisely as it is
solved* (at 15/16 valid, weight ≈ 4×, and the one invalid rollout gets ≈ −3.9 z from validity alone). Use
fixed weights calibrated in Step 0.

### 5.6 Scale
Divide `R_disc` by a **single global constant** `c = std(R_disc)` measured once in Step 0. Under AdamW a
global gradient rescale is ~a no-op for the effective lr; scale matters only for three *ratios*:
`R_disc↔λ_*`, `R_disc↔β_KL`, and `R_disc↔grad-clip 1.0`. **Do not use it to justify importing anyone's lr.**

### 5.7 KL goes in the **loss**, not the reward
`R_total = R_disc + R_valid`; KL is a separate loss term. In-reward KL gets divided by any group
normalizer, making β data-dependent, and only k1 would be correct there.

---

## 6. GRPO configuration

**6.1 — `advantage = R_i − mean_K(R)`. NO std division.** (TRL `scale_rewards=False`; verl
`norm_adv_by_std_in_grpo=False`.) ⚠ Dr.GRPO. **The mechanism is worse for us than for them:** with binary
rewards std is bounded below (~0.33 at G=8), capping the 1/std amplifier at ~3×; with a *continuous*
reward std can approach zero, making 1/std an **unbounded amplifier that fires hardest on the ~27%
statistically inert drugs** whose within-group spread is pure noise.

> **This corrects v1**, which specified `(R_i − mean)/std` and justified it as "cancels per-drug
> difficulty." That is an argument for **mean-centering** only. Dividing by group std additionally
> equalizes per-drug reward *scale* — erasing exactly the distinction between a drug with signal and a
> drug with none.

**6.2 — Strictly on-policy: one gradient step per rollout batch** (`num_iterations=1`). This closes four
questions at once: ratio ρ ≡ 1 → clipping never binds → the clip-higher / GSPO debate is moot; the KL
estimator needs no importance correction; and it removes the mechanism behind ⚠ the "spurious reward"
result (random rewards producing RLVR-like gains via clipping bias). Cost: less sample efficiency, and
generation is our dominant cost. If throughput later forces reuse, move to `num_iterations=2` **and
simultaneously** switch the KL estimator to `ρ·k3` or `sg(ρ)·k2`.

**6.3 — KL: β = 0.02–0.05, loss term, estimator k2 differentiated directly (valid on-policy).**
- **Never k3-as-loss:** differentiating k3 yields the **forward**-KL gradient (mass-covering, coefficient
  (1−ρ) instead of −log ρ).
- **Never k1-as-loss:** expected gradient is exactly zero. ✅ **Verified in this repo** —
  `legacy_whole_panel/train/grpo_c2s_tahoe.py:330` computes `kl = policy_lp − ref_lp`; that anchor is
  **inert** and contributes only variance.
- **Do not set β=0** (as DAPO/Dr.GRPO do): their justification is a *rule-based, unhackable* verifier. Our
  reward is a decode→pseudobulk→cosine proxy that is demonstrably hackable, and the reference is the only
  thing encoding what a plausible cell sentence looks like.

**6.4 — Loss aggregation:** `seq-mean-token-sum-norm` with a **constant** normalizer = generation budget
(Dr.GRPO replaces 1/|o_i| with a constant, it does not delete it). Low stakes here (near-fixed length), but
**check first**: our targets are truncated at each condition's median cell length, so length varies
systematically by condition and `token-mean` would re-weight the curriculum toward longer-sentence
conditions. ⚠ No source addresses this trade-off.

**6.5 — K: start at 16, then set by the Step-0 rule** (smallest K such that the drug-signal component of
within-group reward variance exceeds the decode-noise component). ⚠ No source supports a universal "G≥8";
C2S-Scale reportedly used 24/prompt but on **bulk** profiles — a far cleaner reward than ours.

**6.6 — Rest:** lr 1e-6, AdamW(0.9, 0.95), wd 0.0, constant schedule, grad-norm clip 1.0, temperature 1.0,
top-p 1.0. `max_new_tokens ≥ 1400` (**non-negotiable**, §5.3).

**6.7 — Explicitly rejected knobs** (each would be cargo-culted from a different regime):
| knob | why rejected |
|---|---|
| clip-higher (ε_hi=0.28) | inert under 6.2; measured at β=0; ⚠ reported minor-to-harmful on base models |
| dynamic sampling | criterion is `0 < |correct| < G` on a **binary** verifier; with continuous reward exact ties have measure zero → never fires |
| entropy bonus / Clip-Cov / KL-Cov | ⚠ validated only ≥7B; reported to stabilize entropy with **zero** performance gain. Log entropy as a diagnostic only |
| unlikeliness reward | designed for R∈{0,1}; ours is continuous **and signed**, so for R<0 the multiplier *protects* likely-but-wrong samples |

---

## 7. Warm start — and it is also a competing hypothesis

**Build the DE-weighted SFT:** a short SFT on *ordinary cell sentences* (format preserved) with the
token-level loss **up-weighted on genes in that drug's residual signature**. This applies the Q15 fix —
repair the gradient ratio — *without* changing the output format, so it is a legal GRPO init.

**It is not merely a warm start: it is the cheaper competing hypothesis.** If it alone produces
drug-specificity in the standard format, we may not need GRPO at all. Run it as **its own arm first**.

| candidate | drug signal at start | note |
|---|---|---|
| single-cell SFT | **null** (−0.020) | cold start, worst option |
| OT/T2 SFT | +0.0263 [+0.007,+0.048] | best *format-legal* checkpoint today |
| **DE-weighted SFT** | untested | preferred; ~3 GPU-h |

The residual-SFT model (+0.143) is **format-illegal** as an init — different output space.

**Stopping rule, pre-registered:** if the DE-weighted warm start does not clear stage 1 with a non-null
`model − scramble`, the honest read is that GRPO will not clear stage 3 either.

---

## 8. STEP 0 — offline gates (pre-GPU, near-zero cost). Do not skip.

If the reward cannot distinguish a *real* drug-A cell from a *real* drug-B cell, RL optimizes noise.

1. **Discrimination** — `R_disc` for real drug-A cells against A's truth vs others'. *Gate: clear margin.*
2. **Single-cell noise** — how far `R_disc` degrades from a 40-cell pseudobulk to one cell. **This sets K.**
3. **Variant selection** — cosine vs rank-overlap; pick empirically (§5.2).
4. **Hackability** — the control copied verbatim, the generic response, duplicate-spam, a truncated stub,
   **plus three the v1 plan missed:** (a) a residual **orthogonal to all** profiles (drives `max_other`→0
   with no drug knowledge); (b) one canonical "maximally contrastive" output reused for every drug;
   (c) a generation whose **length shift alone** moves `residual(g)`. *Gate: all below real cells.*
5. **Difficulty stratification** — margin for identifiable / marginal / inert drugs separately.
6. **Within-group reward variance at the real warm start**, at the real K and temperature; decompose into
   drug-signal vs decode-noise. **Kill if indistinguishable.**
7. **Cross-drug rank-invariance — the sharpest single kill condition.** Rank the K rollouts by reward
   against drug A's truth, then re-rank *the same rollouts* against drug B's truth. **Spearman ≈ 1 ⇒ the
   reward is sorting by generic quality, not by drug ⇒ stop.** (The contrastive algebra cancels the generic
   *component*; it does not automatically cancel the generic *ordering* — and Q10 showed every
   rank/correlation metric on this data is inverted, i.e. they reward generic ordering.)
8. **Reward reliability** — re-score identical rollouts against truths built from **disjoint halves** of the
   treated cells; report σ_signal/σ_noise.
9. **Shuffled-reward null arm**, spec'd at matched compute/steps/validity. Read one-sided: a shuffled arm
   that *improves* is strong evidence of a placebo; one that fails is only weak evidence the real arm works.

---

## 9. Data, curriculum, and split discipline

- **⚠ CRITICAL — unseen-drug evaluation must be leave-one-MoA-out, not leave-one-drug-out.** ✅ **Verified
  locally:** our prompt contains `Mechanism: {moa}`, so a drug-level split **hands the model the held-out
  drug's class label**. *This affects the generalization job currently running* — its `unseen_drug` arm is
  not a clean tier-2 test. (Mitigating context: Q11 found MoA barely predicts response, ratio 0.977, so the
  leak is probably small — but it must be stated, and the MoA-out split is the correct design.)
- **Pre-register that leave-one-MoA-out likely returns chance** — Morgan + MolFormer are at chance vs a
  0.805 ceiling (SAR gate). Treat unseen-drug evaluation as reviewer-proofing, not a GPU-hours experiment.
  Our result lives on **seen drugs**.
- **Keep** the reliability filter (cos(res_A,res_B) > 0.2 → 62%) and the difficulty curriculum
  (identifiable ≥ 0.8 → 46.2%). With 1/std removed these are our *honest* substitute for what the
  advantage denominator was silently doing.
- **Do NOT restrict the reward to a phenotype gene subset.** Selecting "genes that vary between drugs" from
  the same data that defines `residual_true` makes the reward partly a function of the answer key.
- **Check** whether generated sentence length correlates with condition biology before finalizing §6.4.

---

## 10. Failure modes

| # | failure | mitigation |
|---|---|---|
| 1 | **Cold start** — advantage ≈ 0 | DE-weighted warm start; dense reward; **not** "raise temperature" (see 5b) |
| 2 | Reward hacking | validity term; KL anchor; §8.4 adversarial gate |
| 3 | Generic-program farming | contrastive form cancels it by construction |
| 4 | Noisy reward (1 cell vs 40-cell truth) | §8.2 sets K; rank-overlap variant if cosine too noisy |
| 5 | Training on noise (27% inert, 38% irreproducible) | reliability filter + curriculum |
| 5b | **⚠ "Higher temperature, larger K" is half wrong** | higher temperature adds *drug-agnostic* diversity: healthy-looking within-group variance that is pure measurement noise, which GRPO will confidently reinforce. **A run that looks well-conditioned and learns nothing is the most expensive outcome available.** Replace with §8.6/8.7 |
| 6 | Fluency collapse | KL anchor; validity monitor with hard stop |
| 7 | Cell-line shortcut | negatives are drugs in the *same* cell line |
| 8 | Overfitting to trained conditions | tier-aligned holdout; MoA-out split (§9) |
| 9 | Silent truncation | ≥1400 tokens + `[END_CELL]` rate logging |
| 10 | **Persistent-variance takeover** — length term dominates once `R_disc` plateaus | dead-band (§5.3); log **per-component within-group std**; alert if std(R_len)/std(R_disc) > 1 |
| 11 | Validity-weight escalation | fixed weights + floor (§5.4, §5.5) |
| 12 | **KL anchor silently inert or optimizing forward KL** | §6.3 + a **unit test**: perturb π from π_ref, assert the KL-term gradient matches a finite-difference *reverse*-KL gradient to <1e-5 |
| 13 | Spurious-gradient artifact | §6.2 kills the mechanism; run the shuffled-reward null (§8.9) |
| 14 | **Proxy-gold overoptimization** — reward is computed from the same `residual_true` that defines NIR | alarm = held-out gold falls **below the reference policy's gold**, not merely "proxy up / gold flat". Monitor gold with a *different* distance (Spearman on logFC) and truths from disjoint halves. **Never use gold for checkpoint selection** — that burns the holdout |
| 15 | Reward-noise dominance | §8.8 |
| 16 | **Reward ranks by generic quality, not drug** | §8.7 — the sharpest kill condition |
| 17 | Effective-K shrinkage from filtering | floor, don't drop (§5.4) |
| 18 | Length↔reward mechanical coupling | §5.3 + §8.4(c) + ≥1400 tokens |

---

## 11. Implementation

**Stack: TRL `GRPOTrainer`.** Single-GPU friendly; takes our reward as a plain Python callable over
decoded text + numpy; integrates vLLM (`use_vllm=True, vllm_mode="colocate"`). Set explicitly:
`scale_rewards=False`, `beta=0.03` (TRL's default is now 0.0), `num_iterations=1`, `loss_type` per §6.4 —
and **verify against the installed `trl/trainer/grpo_trainer.py`**, since `loss_type="dr_grpo"` bundles the
constant normalizer with the no-std change. *(verl exposes the same flags but its payoff is multi-node
scaling we don't need, and it is Ray-based/Windows-hostile.)*

**Do NOT extend `grpo_c2s_tahoe.py`** — ✅ verified locally: no importance ratio, no clipping, generation in
a batch-1 `for` loop, log-probs computed during generation and never used, and an **inert KL** (line 330).

**(a) 4070 Ti Super (16GB) — debugging only.** Full-FT of 1B + frozen reference + K-way KV cache does not
fit. **Preferred: run the whole loop on `C2S-Scale-Pythia-410m`** — same tokenizer, same format, exercises
every code path (~0.8GB weights + ~5GB optimizer + 0.8GB ref, K=8, 1400 tokens, grad checkpointing).
Fallback: 1B + LoRA + 8-bit optimizer + K=4 for plumbing only — **do not port hyperparameters out of it**
(LoRA is the wrong tool for a change that requires the readout to start conditioning on a token it
currently ignores). Use batched `generate(num_return_sequences=K)`, not a for-loop.
*Exit criteria:* reward plumbing end-to-end; decode path; `[END_CELL]` rate; per-component reward logging;
the KL gradient unit test (#12); `assert |mean(advantage)| < 1e-6` and no std division; a 20-step
overfit-one-condition run where reward must rise.

**(b) One H200 — the real run.** 1B full-FT bf16 ≈ 16GB + 2GB reference; >100GB left for generation.
**Colocate vLLM** (`gpu_memory_utilization ≈ 0.35`) — generation dominates (8–16 prompts × K=16 = 128–256
sequences × ~1400 tokens/step) and is the single biggest throughput lever. Compute rewards in
float32/float64 numpy over the 946-gene panel — **not bf16**, since differences of order 1e-2 are below
bf16 resolution. ⚠ Budget: plan a 2–4 day run; **measure tok/s in stage 2 before committing.**

---

## 12. Monitoring (fixed held-out slice, every N steps)

mean reward + **per-component decomposition and per-component within-group std** (#10) · `[END_CELL]` rate,
dup, oov, length · **stratified `model − scramble`** on held-out conditions · prediction diversity
(mode-collapse guard; truths sit at cos ≈ −0.005) · KL from reference · clip-activation rate (should be 0
under §6.2) · per-token entropy (diagnostic only).
**Hard stop** if validity < 0.95, diversity collapses, or held-out gold falls below the reference policy's.

---

## 13. Staging

| stage | what | gate |
|---|---|---|
| **0** | Offline reward calibration (§8) | margin exists; adversarial inputs low; **rank-invariance ≠ 1** |
| **1** | DE-weighted warm-start SFT (§7) — *its own arm* | format valid; **non-null `model − scramble`** (else stop) |
| **2** | GRPO on 410M locally, then 1B small-scale, identifiable drugs | reward rises, validity holds, no collapse |
| **3** | GRPO full run, widened curriculum | held-out `model − scramble` > 0 |
| **4** | Full standard-tier evaluation | the §1 table |

---

## 14. Open questions where the literature offered nothing

Stated explicitly rather than invented: the **similarity function** for one sparse generated cell vs a
pseudobulk · **K under biological-measurement (not decoding) noise** · **relative weights** of `R_disc` vs
λ_dup/λ_oov/λ_len · whether a **contrastive reward works at all** in perturbation prediction (no precedent
anywhere; the one near precedent uses absolute similarity-to-own-truth) · any **1B-scale** validation of
clip-higher, Clip-Cov/KL-Cov, or GSPO.

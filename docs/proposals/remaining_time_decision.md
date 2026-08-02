# Decision memo: what to do with the remaining time

**Purpose.** A second opinion is wanted on one judgement: given the experiment described below, which
modelling directions are still worth running before submission, and which should be closed. The
recommendation is at the end; the places where it might be wrong are listed after it. Attack those.

---

## 1. The thesis in one paragraph

We fine-tune `C2S-Scale-Pythia-1b-pt` — a decoder-only transformer that consumes a cell as a *cell
sentence*, its gene symbols written in descending order of expression — on Tahoe-100M, a single-cell
atlas of ~1,100 compounds across 50 cancer cell lines. The task: given an untreated cell and a drug,
predict the treated transcriptome. The model does not use the drug. Established, with numbers:

| result | value |
|---|---|
| model vs a zero-drug-information baseline (copies the control) | 0.768 vs **0.766** — a tie |
| a per-drug average measured in *other* cell lines (`drug_lookup`) | **0.963**, against a replicate ceiling of 0.968 |
| the same model, residual frame | **0.639** |
| target tokens identical between any two drugs in one context | **83%** → ~1% of the gradient reaches drug identity |
| re-encoding the target as the drug-specific residual | **+0.143** [+0.111, +0.179] |
| …and it generalises to unseen (drug, cell line) pairings | **+0.1002** [+0.0661, +0.1368], no memorisation premium |
| transfer coefficient — share of drug signal that is a per-drug constant | **T = 0.557** [0.513, 0.601]; matched null **+0.000** |

So ~45% of the drug-specific variance is a drug×cell-line **interaction** (κ) that a per-drug average
cannot structurally reach, and that the model does not reach either. The thesis currently ends by
calling that "a quantified, unclaimed target."

**Two facts about the data that constrain everything.** The median condition has **zero**
differentially expressed genes at q<0.05, and a median SNR (effect ÷ replicate noise) of **0.75**.
At the level of an individual condition, the drug is smaller than the noise. `drug_lookup` works by
pooling across cell lines until the main effect emerges.

---

## 2. The experiment just run

**Question.** Is κ a property of the drug's **mechanism** or of the individual **molecule**? If
mechanism-structured, the 45% is reachable and a mechanism-conditioned model is motivated. If
idiosyncratic, the closing claim sharpens to "real, but unreachable from available covariates."

**Why an existing result did not answer it.** A prior "channel gate" found mechanism worth +0.078 —
but it predicted the *whole* residual, which is ~55% main effect. Two EGFR inhibitors having similar
*average* effects is nearly tautological. This experiment runs the same logic on κ alone.

**Design.** For drugs A≠B in cell line c, compare `cos(κ(A,c), κ(B,c))` between mechanism-matched and
mismatched pairs. κ(d,c) = r(d,c) − mean over that drug's *other* cell lines. Controls:

- both arms **within cell line** by construction;
- headline arm restricted to **different-plate** pairs, so class-structured plating cannot inflate it;
- every matched pair given a mismatched partner from the same line at the **nearest reliability
  product** — cosine rewards well-resolved κ, so potent mechanism classes would otherwise win on SNR;
- similarity **disattenuated** by each κ's split-half reliability;
- **ceiling** = κ's own split-half reliability, so the scale is one where 1.0 means "as similar as κ
  is to its own replicate";
- decision by **label-permutation p-value**, not by the bootstrap interval (see below);
- run at two reliability floors, pre-registered: disagreement means neither number is quotable.

An adversarial audit of the estimator found nine defects before it ran, including one fatal (a
non-existent function call) and one conceptual: the leave-one-out construction of κ is algebraically
`(m/(m−1))·(r − mean)`, i.e. a positive scalar times plain centring, and since every statistic is a
cosine the leave-one-out does nothing. That does not affect the headline (different drugs, same line)
but it dominates the same-drug-across-lines arm, whose null is −1/(m−1), not zero. On synthetic data
the artifact reproduces to three decimals: observed −0.0907 against predicted −0.0909.

**Result.**

| min_rel | channel | pairs | bootstrap 95% CI | permutation p |
|---|---|---|---|---|
| 0.05 | target | 238 | [+0.157, +0.324] — excludes 0 | **0.150** |
| 0.05 | moa | 665 | [+0.047, +0.143] — excludes 0 | **0.165** |
| 0.20 | target | 70 | [+0.328, +0.536] — excludes 0 | **0.228** |
| 0.20 | moa | 202 | [+0.031, +0.164] — excludes 0 | **0.308** |

Three things, all decisive:

1. **Every bootstrap interval excludes zero; no permutation p is near 0.05.** The audit predicted the
   clustered bootstrap would be too narrow (~0.92 of the true sampling spread against the
   permutation's ~1.01). The pre-fix version decided on the bootstrap CI and would have reported
   MECHANISM-STRUCTURED on all four rows.
2. **κ's split-half reliability — the ceiling — is 0.195** (0.324 at the stricter floor). In synthetic
   worlds it was 0.75. κ is barely estimable in this atlas.
3. **The plate control collapses.** Only 3–7% of mechanism-matched pairs survive the different-plate
   requirement (1% at the stricter floor): same-mechanism compounds are co-plated in Tahoe, so the
   confound cannot be removed. The script refused to issue a headline verdict for this reason.

Also: `target` is unstable across reliability floors (+0.241 → +0.433, 1.80×) so by pre-registration
it is not quotable; `moa` is stable (1.00×) and non-significant.

**Verdict: not determinable from this atlas.** Not "idiosyncratic" — unresolvable.

---

## 3. What I conclude

**(a) The interaction line is closed.** A direct statistical estimator — 946-dim continuous residual
space, explicit β removal, split-half reliability — resolves κ at 0.195. A language model trained on
the same atlas gets the same conditions at the same ~44 cells apiece, through a lossier
representation and a noisier objective. If the clean estimator cannot see the structure, the model
will not. This argument does not require knowing whether the structure exists. Any method aimed at
the 45% — RL, reasoning chains, larger backbones — is aimed at something this data cannot resolve.

**(b) A reframe that I think we had backwards.** We have been reading "the model loses to the lookup"
as a statement about the *hard* part. It is not. `drug_lookup` scores 0.963 against a ceiling of
0.968 — essentially perfect at the per-drug main effect. The model scores 0.639. That gap of 0.324 is
nothing to do with the interaction. **The model is failing at the easy part**: a pooled, high-SNR,
well-conditioned quantity that is provably learnable, because a simple average learns it. The live
question is therefore not "can it get the interaction" but "why can it not learn a per-drug constant."

**(c) Reasoning tokens survive, retargeted.** The original idea — let the model emit intermediate
tokens, converting an implicit low-gain internal signal into explicit tokens adjacent to generation —
was aimed at composing drug identity with cell state to recover the interaction. That target is now
unpromising. The same mechanism aimed at the *main effect* is not: emit the drug's signature, then
condition on it. Same intervention, reachable target.

Supporting evidence that the readout is the constraint: an activation-swap probe (replicated across
three seeds) detects drug-identity reading in the residual-trained model at layer 12 (+0.0197,
+0.0118, +0.0085, all CIs excluding zero) and detects **none** anywhere in the single-cell model. The
readout is low-gain, not structurally blind.

---

## 4. Recommendation

Run two experiments, then stop and write. Both are cheap, and neither depends on the other's outcome.

**1. Frame reconciliation** (one eval pass, no retraining). The residual arm's +0.143 and the
single-cell arm's 0.768 currently live in incomparable frames, connected by nothing. The machinery to
join them already exists (`reconstruction.npz`: predicted_treated = control + generic_shift(cell_line)
+ predicted_residual). Reconstruct residual predictions into full profiles and score them in the
expression frame. This answers "does the repair help at the original task, or only at the re-encoded
one" — a question an examiner will certainly ask, and which we currently cannot answer.

**2. Signature-in-prompt** (one eval pass, no retraining). Put the drug's training-set mean signature
into the prompt as text, immediately before generation. Does the model's score move toward 0.963?
- **Yes** → the failure is retrieval/routing, reasoning tokens are justified, and there is a concrete
  proposal for future work.
- **No** → the readout cannot use drug information *however explicitly it is delivered*. That is the
  strongest available close for the mechanism chapter, and it is a cleaner negative than the one we
  have.

Either outcome is publishable, which is what makes it worth running.

**Leave alone:** GRPO (its best case is convergence to the lookup, which is already computable in
closed form); the two-stage reasoning *training* (only if signature-in-prompt is positive, and there
is likely not time); Gemma-2 with LoRA (the argument for it is prior pharmacological knowledge, not
capacity — a real hypothesis, but future work); anything targeting the 45%.

**On whether the thesis is already complete:** the argument closes without further experiments —
metric broken → model drug-blind → task winnable → bottleneck is the tokenisation → re-encoding
repairs it → repair generalises → a per-drug average still wins → ~45% is an interaction this atlas
cannot resolve. It ends on a measured quantity and a named obstacle. More experiments make it longer,
not more defensible, and the writing is further behind than the science.

---

## 5. Where this could be wrong — please attack these

1. **The "model can't beat the clean estimator" argument may be too strong.** A model pools across
   conditions in ways a per-pair estimator does not; it might extract structure from κ that
   pairwise cosine similarity cannot see. Counter-argument wanted.
2. **"Not determinable" may be over-cautious.** The `moa` point estimate is positive and stable across
   reliability floors (+0.0924 / +0.0927) with 665 pairs over 40 cell lines. Is treating p≈0.17 as
   "no evidence" the right call, or is this an underpowered true positive worth pursuing?
3. **The plate confound may be fatal in the other direction.** With only 3–7% of pairs surviving the
   plate control, is the plate=any number so contaminated that even its *sign* is uninformative?
4. **The reframe may be wrong.** Is "the model fails at the main effect" actually the right reading,
   or is 0.639 vs 0.963 partly a frame/metric artifact rather than a real capability gap? (This is
   precisely what experiment 1 would settle — so if you think the reframe is load-bearing, say so.)
5. **Two experiments may be one too many, or one too few.** Given ~6 weeks and a thesis whose writing
   is behind, is even this much experimentation the wrong call?

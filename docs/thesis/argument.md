# Thesis argument and citation map

**Purpose.** The written argument, chapter by chapter: what each chapter claims, which measured
number carries it, and what must be cited so the claim lands as a contribution rather than as a
rediscovery. `FINDINGS.md` remains the source of truth for numbers; this file is the *argument*.

⚠️ **Citation confidence is flagged throughout.** `[VERIFY]` means I am confident the work exists and
have characterized it correctly in substance, but the year/venue/author-order must be checked against
the actual paper before submission. Nothing here should be cited from this file without opening the
source.

---

## The claim, in one sentence

A 1B-parameter single-cell language model fine-tuned on the largest available drug-perturbation
atlas does **not** capture drug-specific transcriptional response; the failure is **not** an
architecture or data limit but a property of how the prediction target is **tokenized**; repairing
the tokenization makes drug use real and **transferable to unseen cell-line contexts**; and what
remains after the repair is bounded by the finding that a drug's response is largely a **per-drug
constant** that a lookup table reproduces at the replicate ceiling.

**Why that last clause matters.** It converts three chapters of negative results from *"our model
did not work"* into *"here is the structural reason no conditional model could have"*. That is the
difference between a null result and a contribution.

---

## Four contributions

| # | Contribution | Carried by | Status |
|---|---|---|---|
| 1 | **A metric-calibration audit** showing the field-standard evaluation is exploitable: a zero-information baseline scores above the noise ceiling | Q3, Q8, Q10 | ✅ measured |
| 2 | **A mechanistic localization** of drug-blindness: the drug is increasingly encoded with depth yet the generation readout is magnitude-sensitive and direction-blind | Q6, Q13 | ✅ measured |
| 3 | **A demonstrated repair** — re-encoding the target as the drug-specific residual — that produces drug use which **generalizes** to unseen (drug, cell line) pairings with no memorization premium | Q15, Q16 | ✅ measured |
| 4 | **A quantitative bound on the remaining headroom**: the transfer coefficient T, separating the per-drug main effect from the drug×cell-line interaction | `variance_decomposition.py` | ⏳ **running** |

Contribution 1 is the most portable — it applies to any paper using DE-Δr-family metrics.
Contribution 2 is the most novel. Contribution 3 is the positive result. Contribution 4 decides
whether the thesis ends in closure or in one more arm.

---

## Chapter argument

### Ch. 1 — Setup
Task, data (Tahoe-100M), model (C2S-Scale-Pythia-1b), the 946-gene L1000∩Tahoe panel, the tier
structure, and the `[END_CELL]` representation.

**Cite:** Tahoe-100M `[VERIFY — Vevo Therapeutics, 2025]` · Cell2Sentence `[VERIFY — Levine et al.,
ICML 2024]` and C2S-Scale `[VERIFY — bioRxiv 2025.04.14.648850]` · L1000 landmark genes
`[VERIFY — Subramanian et al., Cell 2017]` · foundation-model context: scGPT `[VERIFY — Cui et al.,
Nature Methods 2024]`, Geneformer `[VERIFY — Theodoris et al., Nature 2023]`.

### Ch. 2 — The metric is the first result
**Claim.** The standard evaluation cannot support the conclusions drawn from it. A zero-information
baseline (every gene at mid-rank) scores **DE-Δr = 1.000**, *above* the noise ceiling (0.76) and
above the model (0.73) — an inverted ordering that proves an artifact. Under Dynamic Range Fraction
within-plate, **NIR is the only calibrated metric** (+0.446); every rank/correlation prediction
metric is inverted.

**Why it must come first.** Every later negative result would otherwise be attributable to a broken
instrument. This chapter earns the right to make claims with the rest of the thesis.

**Cite:** the DRF framework `[VERIFY — Miller et al., 2025]`, which we port and whose calibration
test our critique must survive · Ahlmann-Eltze, Huber & Anders `[VERIFY — "Deep learning-based
predictions of gene perturbation effects do not yet outperform simple linear baselines", bioRxiv
2024 / Nature Methods 2025]` · Kernfeld et al. `[VERIFY — systematic comparison of expression
forecasting methods]` · PerturBench `[VERIFY — 2024 benchmark]`.

**Positioning.** These works establish that simple baselines match deep models on perturbation
prediction. Our addition is *why* — the metric they are all scored on is saturated by a control
regression-to-the-mean artifact, which we demonstrate with a baseline that scores 1.0 while knowing
nothing.

### Ch. 3 — The model is drug-blind
**Claim.** On every instrument that isolates the drug — forced-choice grading, output-invariance,
scramble, NIR — the model sits at chance and is indistinguishable from a scrambled-drug prompt.
Grading ≈ 0.48 ≈ scramble; output-invariance gap 0.000 [−0.019, +0.016]; `model − scramble` +0.014
[−0.016, +0.042]. A zero-drug-information control-copy (0.766) *matches* the model (0.768).

**Two defences that must be in the chapter**, because both are the obvious referee objections:
- **Plate confound.** Drug and plate are partially confounded by Tahoe's design. Every discrimination
  number was re-run `--same_plate_only`. Conclusions unchanged; some magnitudes shrink.
- **Similar-drug swaps.** If the scramble swaps in a drug with a similar true response, an unchanged
  output is *correct*, not blind. The swap-distance sweep turns the single test into a curve across
  response-distance quantiles: **null in every bin including the far tail** (−0.019), trend ≈ 0.

### Ch. 4 — Where the blindness lives
**Claim.** Drug identity **is** linearly decodable from the residual stream (82% at layer 9, rising
to 0.88 by layer 16 with context removed) while its causal variance share stays flat at ~3%. The
decisive test is the **matched-norm swap**: overwriting drug A's code with drug B's, inside the
confound-purified drug subspace, perturbs the output *no more than matched-norm noise* at all seven
depths. The readout responds to the **magnitude** of activity in the drug region, not to **which
drug** it encodes.

**Why this is the novel chapter.** It converts "the model ignores the drug" from a behavioural
observation into a mechanism, and it *predicts* Ch. 5 — if the defect is the readout ignoring
direction, no target-side fix can touch it.

**Cite:** representational selectivity / privileged subspaces `[VERIFY — Gurnee et al.]`. The framing:
cell-line identity sits in the causally active subspace because emitting any plausible profile
requires it; drug identity is encoded but lies outside that workspace. The asymmetry is principled —
cell-line conditioning is a fixed lookup, drug→gene response demands flexible composition.

### Ch. 5 — What does not fix it (the ε-ladder)
**Claim.** Entropic-OT regularization ε interpolates between the two target-side fixes we tried:
ε→∞ is the treated mean (consensus), ε→0 is a single state-matched treated cell. We swept the ladder
and **every point is drug-blind**: consensus −0.010, OT/T2 −0.001, single-cell +0.014.

**Framing note.** This is the chapter that earns Ch. 6. It is a *controlled sweep*, not a sequence of
failed attempts — the endpoints are our own prior experiments, which is what makes the negative
conclusive.

**Cite:** optimal transport in single-cell trajectory/perturbation modelling `[VERIFY — Schiebinger
et al., Cell 2019 for Waddington-OT]`; entropic regularization and Sinkhorn `[Cuturi, NeurIPS 2013]`;
scVI `[VERIFY — Lopez et al., Nature Methods 2018]`.

### Ch. 6 — The repair: it was the tokenization
**Claim.** A cell sentence ranks genes by *expression*, so **83% of the target tokens are identical
between any two drugs** in the same context (34.6/200 differ). Cross-entropy therefore allocates
~1% of its gradient to drug identity — which is why every change to target *content* failed
identically. Re-encoding the target as the drug-specific **residual** raises the differing fraction
to 60% and produces the first non-null `model − scramble` in the project: **+0.143 [+0.111, +0.179]**
under an opposite-signature swap, with a monotone dissimilarity gradient and a matched single-cell
control that stays null.

**The methodological lesson worth a box:** a 600-token generation cap silently truncated 26% of
generations and **halved** the measured effect. For any generation-based evaluation, log the
completion rate.

### Ch. 7 — And it generalizes
**Claim.** On a three-way holdout keyed at the (drug, cell line) level, conditions whose *pairing*
was never trained on score **+0.1002 [+0.0661, +0.1368]** (n=250, 38 cell lines) —
indistinguishable from the trained split's +0.0898. **No memorization premium.** The model learned a
drug representation that applies in cell lines it never saw that drug in. Held-out *drugs* stay null,
the control firing as designed.

**Two instrument bugs belong in the methods**, both of which would have hidden this:
proportional sampling (250 built combos → 33 scored, because a uniform draw reproduces the pool's
proportions) and a degenerate-tie NIR (the zero vector scoring 0.000 instead of 0.500).

### Ch. 8 — But a lookup wins, and that is the point
**Claim.** A numpy dictionary of the drug's mean residual from other cell lines scores **0.963**
against a **0.968** replicate ceiling; the 1B model reaches **0.639** (`model − drug_lookup` −0.3245,
and −0.1936 even against a *single* other cell line). The pre-registered "honest partial" branch
fired: the model reads and transfers the drug, but recovers a small fraction of its signature.

**The synthesis** — and the sentence the thesis is built around:

> If a drug's transcriptional response is largely a **per-drug constant**, then a conditional
> generative model has almost nothing to condition on. The failure of the consensus arm, the OT arm,
> and the entire ε-ladder is then not an engineering failure but a **structural** one, and the
> correct model for this task is retrieval, not generation.

**⏳ This chapter is gated on contribution 4.** T quantifies "largely". Without it the claim is
qualitative and a referee will ask exactly the question we cannot answer.

**Cite — this is the load-bearing citation block, currently empty in the repo:**
- **Connectivity Map** `[VERIFY — Lamb et al., Science 2006]` and **L1000/CMap 2.0**
  `[VERIFY — Subramanian et al., Cell 2017]`. The entire CMap programme *presupposes* that a drug's
  signature transfers across cell lines — that is what makes connectivity scoring work. Our T is a
  direct, noise-corrected measurement of that presupposition at single-cell resolution.
- **Ahlmann-Eltze, Huber & Anders** `[VERIFY]` — the modern statement that simple baselines match
  deep models. Ours is the single-cell, autoregressive-LLM instance, with a mechanism attached.
- **Identity/mean-baseline critiques of GEARS** `[VERIFY — Roohani et al., Nature Biotechnology 2023]`,
  **CPA** `[VERIFY — Lotfollahi et al., MSB 2023]`, **scGen** `[VERIFY — Lotfollahi et al., Nature
  Methods 2019]`.
- **scPerturb** `[VERIFY — Peidli et al., Nature Methods 2024]` for the harmonized-benchmark context.

**Do not skip this.** *"A per-perturbation mean beats the deep model"* is the established headline of
that literature. Cited, our work is its cell-sentence-LLM instance measured with a calibrated metric
and a proper replicate ceiling — a contribution. Uncited, an examiner supplies the framing.

### Ch. 9 — Scope and what is closed by measurement
State the boundary honestly and quantitatively. Closed *by measurement*, not by assumption:

| channel | measurement | verdict |
|---|---|---|
| chemical structure → response | SAR gate: Morgan 0.500, MolFormer 0.474 vs ceiling 0.805 | closed |
| MoA → response | `moa_lookup` 0.529; Q11 within/between ratio 0.977 | closed |
| **protein target → response** | **never measured** — `drug_metadata.parquet` carries `targets` and `pubchem_cid` and no script has read either | **open** |

⚠️ **Correct the over-generalization in the design docs.** `grpo_training_plan.md` §9 and
`arm1b_objective_spec.md` §2 both infer "unseen-drug generalization is out of reach" from the SAR
gate. But `sar_gate.py` probes `drug_metadata.parquet` for SMILES only, and
`pubchem_drug_injection_spec.md` §2 itself ranks raw structure as the **lowest**-value channel and
protein targets as **"the key feature"**. The docs generalized from the weakest channel to the
strongest. The honest pre-registration is: *structure does not predict response (measured);
mechanism at MoA granularity does not either (measured); target-level knowledge is untested.*

Also state: `unseen_drug` as currently built is **not a clean tier-2 design** — the prompt contains
`Mechanism: {moa}`, so a drug-level split hands the model the held-out drug's class label.
Leave-one-MoA-out is the correct split. And its null is **underpowered by construction** (CI
half-width ±0.087 against a maximum available effect of ~0.033), so it must be written as a power
statement, never as evidence of absence.

---

## Methods chapter — the discipline that makes the results quotable

Worth its own section, because collectively it is a contribution:

1. **Plate control.** Drug/plate confounded by design; `--same_plate_only` on every discrimination
   instrument; a zero-drug-info control-copy scores 0.766 cross-plate, which is what leakage looks
   like.
2. **Clustered bootstrap over cell lines**, not drugs — drugs within a cell line are pseudoreplicates.
3. **Leak-immune contrasts.** `model − scramble` shares the control cell across arms, so control and
   batch structure cancel. Anchor every drug-use claim to it, never to `model − linear`.
4. **Stratified scramble.** A random swap can land on a near-twin where an unchanged output is
   correct; swapping to `near`/`orth`/`opposite` strata turns one test into a gradient, and the
   gradient is itself the evidence.
5. **Reliability filtering**, not weighting: 38% of residuals do not reproduce across a half-split.
6. **Pre-registered interpretation** (`arm1b_objective_spec.md` §7) written *before* the result — and
   the branch that fired was the middle one. Say so; it is a credibility asset.
7. **Retractions kept in place, struck not deleted** — DE-Δr "competence", "residualization raises the
   ceiling", "OT achieved nothing", "the headroom has vanished". A visible retraction trail is
   evidence of method, not of error.

---

## Open before submission

| # | Item | Blocks |
|---|---|---|
| 1 | **T** — the transfer coefficient | Ch. 8's central claim; the difference between qualitative and quantitative |
| 2 | **‖residual‖/‖shift‖** — what fraction of the response this frame covers | Ch. 9 scope; an examiner *will* ask "you closed what fraction of the phenomenon?" |
| 3 | **`target_lookup`** — the one untested channel | Ch. 9's "open" row; possibly one more arm |
| 4 | Verify every `[VERIFY]` citation against the source | all |
| 5 | Leave-one-MoA-out split if any unseen-drug claim is made | Ch. 9 |

Items 1 and 2 both come out of `variance_decomposition.py`. Item 3 is `channel_gate.py`, unwritten.

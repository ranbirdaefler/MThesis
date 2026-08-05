# Master’s Thesis Committee Assessment

**Document reviewed:** current compiled `thesis.pdf` and active LaTeX sources  
**Field:** computational biology, perturbational transcriptomics, and machine learning  
**Recommendation:** **major revision**  
**Grade for the current PDF:** **21/31**

The scientific core is strong enough for a Master’s degree, but the current artifact should not be defended or deposited. The most valuable contribution is a controlled forensic audit of whether a cell-sentence model actually uses drug information. The likely grading range is **18–26/31**, depending on whether a committee grades the unfinished residual analysis or the narrower validated core. With the mandatory corrections below, the same project could plausibly reach **27–28/31**.

## Bottom line

Keep the metric exploit and expression-frame drug-blindness result at the centre of the thesis. Treat the residual sensitivity and lookup ordering as provisional until target identity and residual-metric calibration are frozen. Demote the activation-mechanism story and the 44.7% coefficient to exploratory analyses.

The defensible contribution is a rigorous benchmarking and failure-analysis thesis. It is not yet a state-of-the-art perturbation predictor, an identified biological discovery, or a general mechanistic account of autoregressive cell models.

## Grade breakdown

| Criterion | Score | Maximum | Assessment |
|---|---:|---:|---|
| Research question and significance | 4.0 | 4 | Important, precise, and well motivated |
| Novelty and contribution | 3.5 | 5 | Strong diagnostic audit; not a new predictive model |
| Literature and biological positioning | 2.5 | 4 | Focused, but misses direct chemical-perturbation comparators |
| Methods and controls | 4.0 | 6 | Unusually thoughtful controls; residual-frame estimands remain mixed |
| Statistical correctness | 2.5 | 5 | Strong expression-frame audit; residual metric and uncertainty need reruns |
| Reproducibility and completeness | 1.5 | 4 | Useful code safeguards, but target provenance, seeds, and submission remain incomplete |
| Writing and argument | 3.0 | 3 | Exceptional clarity, candour, and argumentative structure |
| **Total** | **21.0** | **31** | **Pass-quality scientific core; major revision required** |

## What is genuinely strong

### 1. Question design

The thesis asks the right falsifiable question: does changing only the named drug change the answer correctly, rather than merely causing the model to generate a plausible treated profile? This distinction is sharper and more scientifically useful than reporting one aggregate benchmark score.

### 2. Metric audit

The zero-information DE-Δr exploit is concrete, memorable, and potentially useful beyond this model. A predictor carrying no drug, cell, or biological information reaches approximately 0.9999, showing that the metric can reward loss of information rather than accurate perturbation prediction.

### 3. Experimental-unit reasoning

The thesis treats cells, cell lines, doses, plates, treatment wells, and split construction more carefully than much published work. Its rejection of ordinary cell-level or pair-level holdouts in favour of whole-treatment-well withholding is a substantial methodological strength.

### 4. Negative controls

Prompt scrambling, within-plate comparisons, forced choice, output invariance, removal gates, matched interventions, and explicit retractions show strong scientific instincts. The thesis repeatedly asks what a measurement fails to establish instead of presenting every non-zero quantity as a mechanism.

### 5. Hard baseline

The training-only per-drug lookup gives the model an appropriate challenge. It tests whether a billion-parameter generator adds anything beyond estimating a drug’s average residual response from training conditions. The ordering is directionally decisive, although its exact numerical gap needs a target-matched rerun.

### 6. Writing and scientific candour

The argument is unusually clear, quantitative, and candid about failed designs. Retaining discarded designs and explaining the confound that invalidated each one strengthens the scientific record.

## What is actually novel

The following contributions are sufficiently novel for a strong Master’s thesis:

- Adversarial calibration of perturbation metrics on chemical single-cell data.
- A drug-header scramble and physical-well leakage audit for a Cell2Sentence fine-tune.
- An integrated diagnostic chain connecting behavioural tests, representation probes, interventions, target reformulation, and hard baselines.
- A residual-target package that increases prompt sensitivity on trained conditions.
- A strong provisional demonstration of the gap between LLM generation and training-only retrieval.
- A Tahoe-specific demonstration that the tested 1B expression-frame model shows no detectable use of drug identity under calibrated controls.

The following are **not** established as novel contributions:

- A state-of-the-art perturbation predictor.
- A general solution for unseen-drug prediction.
- A clean drug-by-cell-line variance decomposition.
- A biological discovery about pathways, lineage dependence, resistance, or mechanism.
- A general mechanism explaining autoregressive cell models.
- The general observation that perturbation responses vary across cellular contexts.

The primary novelty is methodological and diagnostic—not architectural or biological.

## Claim-by-claim adjudication

| Thesis claim | Committee verdict | Defensible reading |
|---|---|---|
| DE-Δr can be saturated by a zero-information predictor | **Accepted** | This is the thesis’s strongest and most portable result; see pp. 22–26. |
| The original expression-frame checkpoint uses the drug | **No use detected in the tested tier-2 setting** | Scope the conclusion to the tested Pythia-1B checkpoint, target representation, fine-tuning recipe, and fine-tuning-held-out prompts. |
| Residual re-encoding creates prompt sensitivity | **Moderately supported on training conditions** | The model’s NIR exceeds chance, but the scramble partner is selected using observed response geometry and training/generation seeds are missing. |
| A training-only drug lookup beats the model | **Direction likely; exact magnitude provisional** | The large ordering survives held-out wells, but residual-target provenance and residual-metric calibration must be resolved before quoting −0.3631 as final. |
| Residual-model transfer to held-out wells | **Not established** | The drug-clustered interval crosses zero, and the split mainly recombines drugs and cell lines already present in training. |
| The 44.7% remainder is a drug×cell-line interaction or variance share | **Rejected** | It mixes cell line, dose, well, generic estimation, and target-estimation effects. |
| Protein-target metadata enables unseen-drug prediction | **Promising but not established** | The analysis is small-n, the claim-relevant drug interval is weak, and the target population is not restricted to held-out drugs. |

## Major concerns

### 1. The PDF is not submission-ready

**Location:** front matter; pp. 66–70; p. 82

There is no completed title page or abstract, the acknowledgements remain a TODO, two analyses remain TODOs in Limitations, and a planted-world validation remains marked pending. These are submission blockers rather than stylistic details. The final PDF must contain no unresolved analysis status markers.

### 2. The residual evaluation’s target provenance is internally contradictory

**Location:** §3.11, p. 50; Table 17

One passage states that evaluation refits the generic on reliability-surviving training conditions rather than the full inventory used to construct checkpoint targets. Elsewhere, captions state that the build’s fit digest was verified.

The current code can reconstruct the generic from an exact build report, compare its digest, and reject a mismatched frame. That safeguard is good. However, the locally available headline result artifact does not include the target report or execution log required to prove which reconstruction path produced the quoted numbers.

Evaluation truths, lookup residuals, and model targets must be defined in exactly the same frame. Rebuild the truths from the immutable checkpoint target artifact, retain the digest-verification log, and rerun every residual table before treating **0.913 versus 0.549** or the paired **−0.3631 [−0.3982, −0.3280]** gap as final.

### 3. The residual-frame metric was not calibrated in the form actually used

**Location:** §§3.2, 3.8, and 4.1

The DRF audit validates expression-space NIR using Euclidean distance. The residual analysis instead uses cosine similarity over signed, discounted rank vectors, with comparison sets spanning plates.

The discrimination principle remains reasonable, but calibration does not transfer automatically after changing:

- the representation;
- the similarity function;
- the comparison set;
- plate restrictions;
- the truth-construction procedure.

Run positive- and negative-control DRF experiments directly for the residual signed-rank cosine NIR. Complete the promised within-plate rescoring for generation-free arms. Until then, the lookup/model ordering is suggestive, while absolute values and chance-to-reference coverage figures are provisional.

### 4. The residual task is not the individual-cell counterfactual posed in the Introduction

**Location:** §3.8, pp. 39–42

The residual target is a condition-level pseudobulk signature repeated across many control-cell prompts. The model is therefore not trained to predict an individual cell’s counterfactual response or treatment heterogeneity.

The current experiment establishes condition-level signature generation from single-cell data. It does not establish single-cell counterfactual prediction. The thesis also has not shown that the model uses the supplied control cell rather than solving the task from drug and cell-line metadata.

Required controls include:

- scrambling the control cell while retaining drug and cell line;
- replacing the control with another cell from the same line;
- a drug-only prompt;
- a cell-line-label-only prompt;
- a no-control prompt;
- auditing whether exact DMSO control sentences or pools occur in both training and held-out prompts.

### 5. The mechanistic identity test contains a mathematical error

**Location:** §3.6, pp. 35–38

The log-sum-exp normalizer does not cancel algebraically between two different teacher-forced response sequences once their prefixes diverge. It cancels only when logits are compared under the same hidden state and prefix.

The observed small layer-specific intervention effect is therefore suggestive, not a clean proof that the model reads a drug-identity representation. A corrected design should use same-prefix token contrasts, natural activation patching, or another intervention whose normalizer and prefix are genuinely shared.

### 6. The transfer coefficient remains descriptive

**Location:** §3.9, pp. 42–45

The prose correctly retracts the original interaction interpretation, but a boxed claim still describes the 44.7% remainder as “variance.” The estimator is a disattenuated angular-similarity coefficient rather than an identified variance component.

The calculation also mixes:

- cell-line context;
- dose;
- treatment well;
- plate;
- generic estimation;
- target-estimation noise;
- reliability correction assumptions.

The supported statement is that one selected residual construction gives a cross-context coefficient of approximately **0.553 [0.514, 0.593]**. Consequently, 44.7% of that measured similarity scale remains unshared for unidentified reasons. It is not an estimated fraction of drug-specific variance, an interaction share, or demonstrated conditional headroom.

### 7. The unseen-drug channel gate evaluates the wrong target population

**Location:** §3.13, pp. 53–55

Restricting partner residuals to training drugs repairs an earlier leakage problem. However, the channel-gate targets still span all retained conditions rather than only held-out drugs.

The result therefore concerns general channel availability, not unseen-drug predictive performance. Protein-target evidence is based on only 18 scored drugs, and its claim-relevant drug-clustered lower bound does not clearly exceed the stated relevance margin. The correct conclusion is “promising exploratory channel,” not “useful unseen-drug prediction is established.”

### 8. Several inference procedures still understate dependence

**Location:** principally §§3.2–3.3 and the residual/channel analyses

The main remaining dependence problems are:

- DRF intervals cluster only by cell line despite crossed treatment wells.
- Output invariance resamples dependent pairwise similarities as if they were independent.
- Separate cell-line×well and drug×well intervals do not equal joint drug×cell-line×well inference.
- The orthogonal scramble partner is selected using observed response geometry.
- Training-seed and generation-seed variability is absent.
- Small-cluster channel analyses are interpreted through asymptotic intervals.

These weaknesses are unlikely to reverse the large expression-frame negative result. They matter substantially for the small residual, transfer, mechanistic, and channel effects.

### 9. The lookup score mixes clean held-out-well support with potentially same-well training support

**Location:** §3.12, pp. 52–53

For a trained target, the other-cell-line lookup can draw the same drug from another cell line in the same physical treatment well. That leaks well-specific preparation and batch information into what is described as cross-context transfer.

The central ordering is nevertheless robust in direction: on the held-out-well split, where the target well cannot enter a training-only lookup, the model-minus-lookup gap remains approximately **−0.357 over 893 conditions**.

The thesis should:

- report lookup performance separately by split;
- make the held-out-well comparison primary;
- add a leave-one-well-out lookup for training targets;
- avoid describing the pooled 0.913 value as a uniformly independent-transfer score.

### 10. Tahoe contains an independent replication plate

**Location:** §3.1 and the Tahoe-100M source paper, Figure 3D

The statement that a biological replicate does not exist in the atlas is false without a subset qualifier. Tahoe Plate 14 replicates Plate 6 over **4,796 shared cell-line×treatment conditions** and was reserved for validation and model criticism.

If this thesis’s cache or subset excludes Plate 14, it must say so explicitly. If the overlap is available, it is the natural independent reproducibility analysis and should be used to distinguish within-well sampling precision from treatment-level reproducibility.

### 11. The NIR implementation is correct, but the mathematical specification is incomplete

**Location:** §3.2, pp. 23–24

Equation 9 defines NIR using a higher-is-better similarity, \(\sigma\). The text then describes `nir_expr` as Euclidean distance, where lower is better. The implementation correctly reverses the inequality for distances, but the thesis does not state this.

Define either:

\[
\sigma(\hat y,y)=-\lVert\hat y-y\rVert_2,
\]

or give separate similarity and distance formulas. The statement that chance equals exactly 0.5 also requires an exchangeability null for the target label within the declared comparison set; it is not true without assumptions about how the target is selected.

### 12. Two preprocessing validations are mathematically incorrect

**Location:** §3.1, pp. 18–19

First, CP10K multiplication and a monotone logarithmic transformation cannot change within-cell gene ordering. They may affect pseudobulk magnitudes and residuals, but they cannot make a rank sentence meaningful by changing its order.

Second, a PCA cell-line classifier cannot validate gene identities. A fixed permutation of gene columns preserves pairwise geometry and therefore preserves PCA and classification performance. Direct gene-token mapping checks, known-marker checks, and round-trip tests are required.

### 13. Literature coverage and biological validation are too thin

**Location:** §2 and throughout Results

The literature review needs direct positioning against chemical-perturbation methods and benchmarks including:

- CPA and chemCPA;
- scGen;
- CellOT and CondOT;
- scVIDR and BioLord;
- PerturBench and scPerturBench;
- Tahoe-x1;
- simple dose-aware additive, nearest-neighbour, matrix-factorisation, and linear baselines.

The biological interpretation is also largely asserted. The purported generic stress and cell-cycle programme requires pathway enrichment, representative genes, named drug–cell-line cases, dose dependence, and replication across plates. Without this evidence, call it a plate-centering or generic component rather than a validated biological programme.

### 14. Figures, tables, and prose mix canonical and stale states

**Location:** pp. 17, 26, 35, and 58–68

Examples include:

- Figure 3 values disagreeing with its corrected caption.
- Figure 6 panel and caption describing different interventions.
- “Tier-2 drugs” being used for a count of conditions.
- 46.2% later being described as “a third.”
- A within-well split-half being called a real replicate.
- Duplicated or corrupted prose.
- Tables deferred many pages after their first citation.

A final artifact-level consistency pass must regenerate all figures, tables, captions, claim boxes, Introduction numbers, Limitations, and Conclusions from one canonical manifest.

## Required revision order

### Priority 1 — Prove target identity and rerun the residual evaluation

Rebuild evaluation truths from the checkpoint’s exact target report. Verify every generic-fit and frame-setting digest, preserve the execution log, and regenerate all residual numbers from that run.

**Acceptance criterion:** a reader can identify one checkpoint, target artifact, fit inventory, generic configuration, split manifest, and evaluation run behind every residual number.

### Priority 2 — Calibrate the residual metric

Run positive- and negative-control DRF analyses for signed-rank cosine NIR in its actual comparison sets. Complete the within-plate rescoring of generation-free arms.

**Acceptance criterion:** the exact residual metric is shown to rank a valid positive control above a drug-agnostic negative control under the same representation and plate structure used for headline evaluation.

### Priority 3 — Finish and freeze the artifact

Add the title page, abstract, acknowledgements, code/data-availability statement, environment, software versions, hardware, training configuration, decoding configuration, and immutable artifact identifiers. Remove all TODO and PENDING text.

**Acceptance criterion:** one manifest generates every result, and the compiled PDF contains no unresolved analysis status.

### Priority 4 — Replicate the stochastic stages

Use independent fine-tuning seeds and repeated decoding seeds. Report hierarchical uncertainty and completion, validity, truncation, and duplicate rates.

**Acceptance criterion:** conclusions do not depend on one checkpoint or one set of four generations per arm.

### Priority 5 — Narrow or repair the mechanistic claims

Remove the normalizer-cancellation argument. Use same-prefix token contrasts or natural activation patching and dependence-aware uncertainty for drug-pair interventions.

**Acceptance criterion:** the intervention isolates drug identity without conflating prefix divergence, off-manifold damage, or generic perturbation.

### Priority 6 — Correct the remaining statistical units

Use inference that respects drug, cell line, and well; resample output-invariance statistics at the context level; and select scramble partners using independent training data or an independent truth half.

**Acceptance criterion:** the resampling unit corresponds to the unit over which the associated claim generalises.

### Priority 7 — Use the independent replication structure

Evaluate the Plate 6/Plate 14 overlap. Report lookup performance by split and add a leave-one-well-out training estimate.

**Acceptance criterion:** within-well precision, treatment-level reproducibility, and cross-well generalisation are reported as distinct quantities.

### Priority 8 — Test what the residual model conditions on

Audit control-cell IDs and control pools across splits. Add control-cell, cell-line-label, drug-only, and no-control prompt ablations.

**Acceptance criterion:** the thesis can say whether the model uses the individual control cell, cell-line metadata, drug identity, or some combination.

### Priority 9 — Repair unseen-drug claims and deepen the biology

Restrict channel-gate targets and partners to the intended held-out regime, use drug-level uncertainty and multiplicity control, add leave-one-mechanism-out evaluation, and include pathway and comparator analyses.

**Acceptance criterion:** unseen-drug conclusions apply to held-out drugs and are supported across drugs rather than conditions alone.

## Recommended final narrative

### 1. The standard metric can fail catastrophically

Keep the zero-information exploit as the lead methodological result. It is the cleanest, most general contribution.

### 2. The tested original Cell2Sentence fine-tune showed no detectable drug-header use

State this only for the tested Pythia-1B checkpoint, expression-frame target, fine-tuning recipe, and calibrated tier-2 evaluation. Do not generalise it to all Cell2Sentence models or foundation models.

### 3. Residual re-encoding creates sensitivity, not established fidelity

The residual-target package makes the model respond on trained conditions. The model remains far below direct retrieval, but the exact gap must remain provisional until target identity and residual-metric calibration are frozen.

### 4. Context-specific and unseen-drug opportunities remain unresolved

Present the transfer coefficient, channel gate, and activation analyses as hypothesis-generating measurements. They identify possible next experiments, not settled mechanisms or useful unseen-drug prediction.

## Likely oral-defense questions

1. What is the independent treatment-assignment unit in Tahoe-100M, and why is a cell split not a biological replicate?
2. Why was the Plate 6/Plate 14 biological replicate not used as an external reproducibility check?
3. Did the headline residual run reconstruct the exact generic used to train the checkpoint, and where is the matching digest and execution log?
4. What validates signed-rank cosine NIR after changing the representation, similarity function, and comparison set?
5. Do CP10K scaling and a monotone log transform alter within-cell rank, and why?
6. How can a PCA classifier detect a fixed permutation of gene identities?
7. Does the 0.913 drug lookup ever draw another cell line from the target’s physical treatment well, and what is its leave-one-well-out score?
8. What exactly does “unseen drug” mean after Pythia natural-language pretraining and supplying the mechanism label?
9. Does the residual model use the control cell, or only drug and cell-line metadata?
10. Are any exact DMSO control-cell sentences or control pools shared between training and held-out prompts?
11. Why should a log normalizer cancel after two teacher-forced response prefixes diverge?
12. Which assumptions make a disattenuated cosine equal a variance-component ratio, and which assumptions are tested here?
13. Why is the orthogonal scramble a prespecified null if it is selected using observed response geometry?
14. Why does the channel-gate target set include trained drugs when the claim concerns unseen drugs?
15. What is genuinely novel here: the predictor, the biology, the metric audit, or the diagnostic protocol?

## Final committee recommendation

**Grade: 21/31. Major revision before defense or deposit.**

The thesis contains a pass-worthy and potentially distinction-level scientific core. Its strongest conclusions—the DE-Δr failure and the scoped expression-frame drug-blindness result—are credible. The current document nevertheless gives final status to residual-frame quantities whose target provenance, metric calibration, stochastic uncertainty, and dependence structure remain unresolved.

The thesis should be accepted only after the residual evaluation is rebuilt from one immutable target artifact, the exact residual NIR is calibrated, stochastic stages are replicated, the 44.7% and mechanistic claims are narrowed, and the document is completed.

## Evidence base for this assessment

This review used:

- the complete compiled PDF and active LaTeX sources;
- targeted inspection of metric-calibration, NIR, residual-evaluation, output-invariance, mechanistic-probe, channel-gate, split, and target-builder code;
- canonical numerical artifacts;
- verification that 73 split/design tests pass when unrelated external pytest plugins are disabled;
- the Tahoe-100M source paper and current Cell2Sentence, PerturBench/scPerturBench, Tahoe-x1, and perturbational-modelling literature.

This is an independent committee-style assessment rather than an institutional grading rubric.

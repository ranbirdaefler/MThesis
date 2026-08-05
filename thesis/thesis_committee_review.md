# Independent Committee Review of the Master’s Thesis

**Thesis reviewed:** [`thesis.pdf`](./thesis.pdf)  
**Review date:** 4 August 2026  
**Perspective:** Master’s-thesis committee member in computational biology, single-cell genomics, perturbation modelling, statistical evaluation, and machine learning  
**Grade assigned to the current PDF:** **21/31**  
**Recommendation:** **Major revision; do not deposit the current PDF**

---

## Executive judgment

This thesis contains a strong and potentially distinction-level scientific core: it asks an important question, discovers a serious evaluation failure, constructs unusually thoughtful negative controls, and shows that a simple per-drug lookup can outperform a billion-parameter generative model. Its best contribution is not a new perturbation-prediction model or a new biological discovery. It is a controlled forensic audit of whether an autoregressive cell-sentence model actually uses the drug information supplied to it.

The current PDF is nevertheless not a coherent final thesis. It combines incompatible analysis runs, retains unfinished work and TODOs, and promotes several descriptive or exploratory results into claims that their estimators do not identify. At least one headline confidence interval is based on pseudoreplication in the analysis code. The claimed “roughly 45% drug × cell-line interaction” is not identified by the design. The unseen-drug channel analysis does not enforce the training-only partner pool described in the text. Several mechanistic claims also exceed what the probes and interventions establish.

My committee decision would therefore be:

- **The underlying research is sufficient for a Master’s thesis.**
- **The current document is not ready for submission or deposit.**
- **A defense should proceed only after the numerical record is frozen and the main claims are narrowed or re-estimated.**
- **The grade of 21/31 applies to the submitted artifact, not to the project’s potential after correction.**

The shortest accurate summary of the surviving result is:

> In the evaluated C2S-Scale-Pythia-1B fine-tuning configuration, the standard full-profile target produced no detectable drug-header use on the calibrated tier-2 evaluation. Re-encoding the target as a condition-level signed residual produced measurable sensitivity to the combined drug-and-mechanism prompt on trained conditions, but this sensitivity did not approach the fidelity of a training-only per-drug lookup and did not establish context-specific generalisation.

That statement is important, defensible, and enough to support a strong thesis. Several broader statements in the current PDF are not.

---

## Scope and basis of this review

This review examined:

1. The complete compiled PDF, including the Introduction, Literature Review, Investigation, Limitations, Conclusions, references, and Appendix.
2. The active LaTeX sources:
   - [`Sections/Introduction.tex`](./Sections/Introduction.tex)
   - [`Sections/Literature-review.tex`](./Sections/Literature-review.tex)
   - [`Sections/Investigation-v4.tex`](./Sections/Investigation-v4.tex)
   - [`Sections/Limitations-and-Future-Research-Directions.tex`](./Sections/Limitations-and-Future-Research-Directions.tex)
   - [`Sections/Conclusions.tex`](./Sections/Conclusions.tex)
   - [`Sections/Appendix.tex`](./Sections/Appendix.tex)
3. Targeted result artifacts and implementation paths supporting central claims, especially:
   - [`../RESULTS_cluster/calibration_v3.json`](../RESULTS_cluster/calibration_v3.json)
   - [`../endcell/analysis/calibration_eval.py`](../endcell/analysis/calibration_eval.py)
   - [`../endcell/analysis/scramble_stratum_audit.py`](../endcell/analysis/scramble_stratum_audit.py)
   - [`../endcell/analysis/channel_gate.py`](../endcell/analysis/channel_gate.py)
   - [`../endcell/analysis/output_invariance.py`](../endcell/analysis/output_invariance.py)
   - [`../endcell/ot/build_residual_targets.py`](../endcell/ot/build_residual_targets.py)
4. Current literature relevant to Tahoe-100M, C2S-Scale, perturbation benchmarks, simple baselines, and direct Tahoe-100M competitors.

This was a static audit. I did not rerun the training or generation pipelines. Implementation findings therefore identify concrete defects or unresolved risks in the current code and artifacts; a corrected rerun may change the numerical result.

---

## Grade breakdown

| Criterion | Score | Maximum | Committee assessment |
|---|---:|---:|---|
| Research question and significance | 4.0 | 4 | Important, precise, and well motivated |
| Novelty and contribution | 3.5 | 5 | Strong forensic audit; not a new predictive model or biological discovery |
| Literature and biological positioning | 2.5 | 4 | Good treatment of the metric dispute; major methods and direct competitors are missing |
| Methods and controls | 4.0 | 6 | Creative and often excellent controls, but several estimands are not isolated |
| Statistical correctness | 2.5 | 5 | Important experimental-unit insights, offset by invalid or unidentified headline inference |
| Reproducibility and completeness | 1.5 | 4 | Mixed result generations, TODOs, incomplete provenance, and limited seed replication |
| Writing and argument | 3.0 | 3 | Exceptionally clear, candid, and intellectually engaging |
| **Total** | **21.0** | **31** | **Pass-quality scientific core; major revision required** |

---

# Part I — What the thesis does well

## 1. The research question is important and unusually well posed

The thesis does not ask only whether the model generates plausible treated cells. It asks whether changing the named compound changes the prediction in a drug-specific and biologically correct way. That distinction is central to perturbation modelling and is often obscured by aggregate fit metrics.

The four questions in the Introduction create a strong investigation:

1. Is the evaluation trustworthy?
2. Does the model use the drug?
3. If not, where does the failure occur?
4. How much drug-specific signal is recoverable?

The sequence is scientifically productive because each question invalidates a different class of explanation. This is much stronger than reporting one benchmark score.

## 2. The DE-Δr exploit is a substantial methodological contribution

Section 3.2 shows that a predictor carrying no drug, cell, or fitted biological information can obtain approximately `DE-Δr = 0.9999`, above both the model and the within-well split-half reference. This is the thesis’s most convincing and portable result.

The result matters because it demonstrates a failure rather than merely asserting that a metric is imperfect:

- The gene subset is selected using the truth.
- Control reversion creates a shared direction.
- A zero-information predictor can therefore align with the selected truth-referenced change.
- Less information receives a higher score.

This is exactly the kind of adversarial metric test that the perturbation-prediction literature needs. Even if every later mechanistic claim were removed, this analysis would remain thesis-worthy.

## 3. The thesis takes experimental units and plate confounding seriously

The physical-layout audit in Section 3.1 is excellent. It recognizes that:

- cells are not independent experimental assignments;
- many cell-line conditions share one treated well;
- drug and plate are partially confounded;
- a `(drug, cell line)` holdout can leave the same treated well in training through other cell lines;
- plate labels are not globally unique and must be paired with cell line;
- dose units cannot be discarded safely.

This is a level of design awareness that is often absent in machine-learning studies on large biological atlases. The decision to move from pair-level to whole-well holdout is methodologically correct, even though the resulting estimand is narrower than the thesis initially wanted.

## 4. The negative-control philosophy is unusually strong

The thesis repeatedly asks what an apparent effect could mean under a broken instrument. Strong examples include:

- comparing the model against a drug-agnostic control-copy;
- holding plate fixed in discrimination tests;
- scrambling only the drug-related spans of the prompt;
- measuring output invariance without using a biological truth;
- checking whether the intervention removes the decodable subspace before interpreting an ablation null;
- using matched-dimension or matched-norm controls;
- retaining failed substitution designs and documenting why they failed;
- identifying the opposite-signature scramble as an active wrong answer rather than a neutral comparator;
- using a training-only per-drug lookup as the real baseline.

The retractions are a strength. They show that the author is willing to invalidate an attractive result when its control fails.

## 5. The training-only lookup is the correct hard baseline

The per-drug residual lookup asks whether the generative model learns anything beyond an average response already observed for the same drug in other contexts. This is the correct benchmark for a seen-drug task.

The qualitative result is important:

- a direct data lookup is near the split-half precision reference;
- the model is far below it;
- averaging one other cell line is already strong;
- the model therefore fails to recover even the easy per-drug component available to a dictionary.

This result should remain central after the numerical versions are reconciled.

## 6. The residual target is a useful intervention

The residual frame removes the untreated baseline and an estimated generic drug programme, then encodes the top up- and down-regulated genes. This makes the target more discriminative across drugs and creates a measurable prompt effect.

The strongest defensible interpretation is:

> Changing the target representation can expose drug-specific variation to the autoregressive loss and can make the fine-tuned model respond to the drug-related prompt.

That is a valuable engineering and scientific finding. The concern is not whether the intervention did something; it did. The concern is whether it isolates tokenisation as the unique causal bottleneck. It does not yet do that.

## 7. The writing is excellent

The prose is clear, claim-first, quantitative, and unusually honest. The investigation is narrated as it developed, including failed designs and withdrawn claims. Most sections state what a result does not show. Figures and captions are argumentative rather than decorative.

This is substantially better than the average Master’s thesis. The main writing problem is not style. It is that stale numerical and inferential claims survived into different parts of an otherwise carefully written document.

---

# Part II — Major findings requiring correction

## Finding 1 — The thesis does not contain one canonical analysis

**Severity:** Submission blocker

The Introduction, Investigation, boxed claims, Limitations, tables, and Conclusions contain incompatible numerical versions.

Examples include:

- Section 3.11 Table 13 reports:
  - `n = 300 / 895 / 199`
  - neutral gaps `+0.0898 / +0.0332 / −0.0315`
- The prose immediately following reports an older evaluation:
  - `n = 200 / 250 / 120`
  - gaps `+0.1457 / +0.0449 / +0.0097`
- The same section gives more than one set of slope estimates:
  - approximately `+0.390 / +0.374 / +0.238`
  - later approximately `+0.373 / +0.337 / +0.342`
- Section 3.12 Table 14 gives ceiling/lookup/model as:
  - `0.958 / 0.941 / 0.569`
- Its boxed claim instead gives:
  - lookup `0.913`
  - model `0.549`
  - common support `n = 1192`
- The Introduction and Conclusions use:
  - ceiling `0.968`
  - lookup `0.963`
  - model `0.639`
- Section 3.9 gives `T = 0.553 [0.514, 0.593]`; the Conclusion gives `0.557 [0.513, 0.601]`.
- Section 3.8 reports a data-resolved reliability threshold of `0.109` retaining `50.5%` of training conditions; the Limitations retain the older threshold `0.2` and `62%`.
- Figure 3 displays, captions, and discusses different DRF values: approximately `0.446`, `0.519`, and `0.635`.
- Section 3.14 first withdraws name-versus-mechanism attribution because the corrected run was not done, then Table 16 and its caption state that the name is read and contributes roughly nine tenths of the effect.

The PDF also says:

- a corrected model-evaluation rerun is “in flight”;
- three-seed replication remains TODO;
- within-plate rescoring remains TODO;
- the corrected field decomposition was not rerun;
- the model’s partial-DE-Δr was never computed.

### Why this matters

A committee cannot determine:

- which checkpoint produced the headline result;
- which target build was used;
- which split manifest was used;
- whether held-out conditions were filtered;
- which population each interval describes;
- which numerical statement is final.

This is not a cosmetic consistency issue. It breaks the chain from estimator to conclusion.

### Required action

Create one immutable run manifest containing at least:

- repository commit;
- dataset/cache digest;
- checkpoint path and hash;
- target-build digest;
- generic-fit digest;
- split-manifest digest;
- exact training inventory;
- exact evaluation inventory;
- training and generation seeds;
- decoding configuration;
- result JSON for every figure and table.

Regenerate all numerical prose, tables, figures, Introduction claims, Limitations, and Conclusions from that manifest. Delete or move every superseded result to an explicitly labelled historical appendix.

---

## Finding 2 — The DRF calibration interval pseudoreplicates drug-level rows

**Severity:** Submission blocker

The PDF’s principal calibration result is:

> NIR is the only calibrated metric, with DRF approximately `+0.635 [+0.618, +0.652]` and Holm-adjusted significance.

The implementation does not support that interval.

In [`calibration_eval.py`](../endcell/analysis/calibration_eval.py):

1. `calibrate_cellline` returns one row for each drug in a cell-line or cell-line×plate group.
2. The main loop extends one pooled list with every drug-level row.
3. `aggregate_drf` assigns each pooled row a unique cluster, with the comment that there is “one row per cell line.”

The saved artifact makes the contradiction explicit:

- actual cell lines: `25`;
- pooled drug-level rows: `1820`;
- reported `n_cell_lines` for each metric: `1820`.

The interval therefore treats drug-by-line rows as independent cell-line clusters.

The one-sided bootstrap p-value is also problematic. It resamples the observed distribution and counts bootstrap draws at or below zero without constructing a null-centred distribution. The reported NIR p-value is exactly the Monte Carlo floor, `1/2001`, and the Holm value is `5/2001`. That is not, by itself, a valid null test.

### What probably survives

The qualitative ordering may be real:

- NIR has positive observed DRF;
- the other tested metrics have negative observed DRF.

The current interval, p-value, and “only calibrated metric” inferential claim do not survive the implementation audit.

### Required action

- Preserve the real cell-line, plate, drug, and well identifiers in each calibration row.
- Define the intended generalisation unit.
- Resample complete clusters, not pooled drug rows.
- Recompute the entire DRF ratio inside each resample.
- Use a null-valid cluster test or permutation scheme.
- Reapply Holm correction to the corrected p-values.
- Report both all-plate and within-plate calibration under the same inferential protocol.

Until this is done, phrase the result descriptively:

> In the observed calibration sample, NIR was the only tested metric with positive DRF; uncertainty is pending cluster-correct re-estimation.

---

## Finding 3 — The “45% drug × cell-line interaction” is not identified

**Severity:** Submission blocker

The thesis writes:

`r(d,c) = β(d) + κ(d,c) + ε`

and interprets a disattenuated cross-context cosine as:

`T = var(β) / [var(β) + var(κ)]`.

This equality is not generally true. It requires assumptions including:

- `β`, `κ`, and measurement error are centred appropriately;
- main and interaction components are orthogonal;
- interaction terms are independent across contexts;
- norms and reliabilities are comparable across contexts;
- dose does not alter the latent drug effect;
- treatment-well effects are absent or independent;
- averaging pairwise cosine ratios is equivalent to a global variance ratio;
- the split-half reliability correction is valid for these high-dimensional cosine objects.

The thesis does not establish these assumptions.

Section 3.9 correctly acknowledges that the actual calculation:

- allows cross-line pairs with different doses;
- can compare different treatment wells;
- uses a particular generic estimator;
- mixes cell line, dose, well, and target-estimation differences.

That acknowledgement directly contradicts the Introduction and Conclusion, which repeatedly call `1−T` a drug × cell-line interaction share.

### Bias-direction error

The thesis states that a within-well split-half reliability is optimistic relative to biological reproducibility and that this optimism makes `1−T` understated.

Given:

`T_hat = observed similarity / sqrt(reliability_1 × reliability_2)`,

an optimistically high reliability makes the denominator too large, which makes `T_hat` too small and `1−T_hat` too large. The unshared fraction is therefore **overstated**, not understated.

### Why the planted-truth simulation is insufficient

The simulation shows that the code recovers truth in worlds generated under its assumptions. It does not establish that Tahoe obeys those assumptions. Estimator implementation validation and estimand identification are different questions.

### Defensible replacement

Call the quantity:

> A disattenuated cross-context residual-similarity coefficient.

The corresponding result is:

> Approximately 45% of the estimated drug-specific residual was not shared across the compared contexts under this estimator; the unshared component combines cell line, dose, well, and residual-estimation differences and cannot be interpreted as a cell-line interaction variance share.

### Required action

Choose one:

1. **Narrow the claim** to descriptive cross-context non-sharing and remove every “drug × cell-line interaction” statement; or
2. **Redesign the analysis** around matched doses and independently replicated wells, then fit a measurement-error-aware multilevel variance model.

The first option is sufficient for this thesis.

---

## Finding 4 — “Replicate ceiling” is the wrong name

**Severity:** High

Most “replicates” in the thesis are disjoint cell splits from the same treated well. They measure within-well sampling precision. They do not measure:

- independent treatment reproducibility;
- plate-to-plate reproducibility;
- biological replicate reproducibility;
- performance on a new treatment assignment.

Calling them “replicate ceilings” invites an upper-bound interpretation they do not support. The lookup can exceed the bar because:

- the lookup averages many conditions or cell lines;
- the reference uses one noisy half-well;
- both are compared with a full-sample truth in some analyses;
- NIR compresses near its upper range.

The thesis partly recognizes this, but the Introduction and Conclusion continue to use “proper replicate ceiling” and reason from distance to that bar.

The Tahoe paper also reports Plate 14 as a biological replicate of Plate 6. Wherever treatment overlap is sufficient, that independent plate should be used as a genuine external reproducibility check.

### Required action

Rename these quantities throughout:

> within-well split-half precision reference

Reserve “biological replicate” for independently treated wells or plates.

---

## Finding 5 — The held-out regimes do not support the advertised generalisation claims

**Severity:** High

### Whole-well holdout

Holding out whole treatment wells is the correct assignment-respecting design. It does not, however, test clean fixed-dose transfer to an unseen cell-line context.

It tests:

> A new physical treatment well of a seen drug, often involving cell lines already represented elsewhere.

The thesis itself reports that:

- 30 of 34 drugs in the `unseen_combo` evaluation occur in training;
- 39 of 42 cell lines occur in training;
- 82% of conditions recombine a seen drug with a seen cell line.

The label `unseen_combo` therefore overstates the split. `unseen_well` is more accurate.

### Outcome filtering

Table 13 says it evaluates the full held-out population, while later prose states that the compatibility path filters held-out conditions by their own split-half reproducibility. That is outcome-dependent selection on the evaluation set.

Even if the selection makes the task easier and therefore does not explain a failure, it changes the population to which the interval applies.

### Required action

- Rename the split `unseen_well`.
- Evaluate the complete held-out inventory without using held-out outcomes to select conditions.
- If a reproducible-only secondary analysis is useful, report it separately as post-stratification.
- Stop describing the split as unseen drug–cell-line pair transfer.

---

## Finding 6 — “Unseen by the whole pipeline” is factually incorrect

**Severity:** High

The thesis describes `C2S-Scale-Pythia-1b-pt` as atlas-pretrained and concludes that a held-out drug was never seen at any stage.

The C2S checkpoint is based on `EleutherAI/pythia-1b`. Pythia was pretrained on natural-language data before the C2S single-cell adaptation. Drug names, mechanisms, targets, and biomedical relationships may therefore have been present before Tahoe fine-tuning.

The prompt also includes:

`Mechanism: {moa}`

so a held-out drug is supplied with a class label that can link it to trained drugs.

### Accurate terminology

Use:

> held out from Tahoe fine-tuning, with mechanism supplied.

Do not use:

> unseen by the whole pipeline.

### Required controls

- Evaluate the base checkpoint before Tahoe fine-tuning.
- Compare opaque drug identifiers with natural drug names.
- Compare name-only, mechanism-only, both, and shuffled name–mechanism mappings.
- Use leave-one-mechanism-out if the intended claim is unseen drug knowledge rather than class interpolation.

---

## Finding 7 — The reported unseen-drug slope does not evaluate the real held-out-drug prompt

**Severity:** High

The “output tracks named drug” slope in [`scramble_stratum_audit.py`](../endcell/analysis/scramble_stratum_audit.py) is constructed from the three scrambled partner prompts:

- near partner;
- orthogonal partner;
- opposite partner.

For each target condition, the analysis relates:

- the true similarity between the target drug and the substituted partner drug;
- the score of the generation produced after naming that partner.

The real target-drug prompt is not a slope point.

### What the slope establishes

The model responds to substituted partner identities in contexts whose scoring target belongs to the unseen-drug split.

### What it does not establish

It does not show that:

- the held-out target drug maps to its own response signature;
- the model has learned a correct zero-shot map for unseen compounds;
- the positive slope derives from held-out drugs rather than trained partner drugs.

### Required action

- Include the real target-drug prompt in the geometry.
- Restrict partner prompts to held-out drugs for the zero-shot claim.
- Test whether each held-out drug is closer to its own truth than to other held-out-drug truths.
- Separate “prompt responsiveness in an unseen-drug context” from “correct unseen-drug prediction.”

---

## Finding 8 — The channel gate does not use a training-only retrieval pool

**Severity:** Submission blocker for Sections 3.13–3.14

The text says that target-, MoA-, and chemistry-based predictions average only drugs present in training.

In [`channel_gate.py`](../endcell/analysis/channel_gate.py), the candidate pool is constructed from every retained condition in a cell line:

- all retained keys are grouped by cell line;
- every other drug becomes a candidate;
- channel partners are selected from that full pool;
- no training-split condition filters the candidate partners.

The holdout manifest controls how the generic is fitted, but it does not restrict retrieval partners to training conditions.

This allows held-out residuals to contribute to a prediction that is described as training-only.

### Additional verdict problem

The implementation declares a channel `closed` whenever the lower confidence bound does not exceed a relevance margin. That is not equivalence testing. The possibilities are:

1. positive above the relevance margin;
2. equivalent to zero within a prespecified margin;
3. inconclusive.

Failure to establish the first is not proof of the second.

This matters for chemistry and MoA:

- chemistry is positive against one plate-matched null and inconclusive against a stricter different-plate null;
- MoA has a positive point estimate with an interval that may not clear the relevance margin;
- neither result licenses a universal “closed” statement.

### Required action

- Restrict targets to the intended held-out-drug set.
- Restrict all partner residuals to training conditions.
- Select partners only within the training pool.
- Cluster additionally by target drug or annotation family where appropriate.
- Use three verdicts: `live`, `equivalent`, `inconclusive`.
- Apply multiplicity correction across channels and null constructions.
- Rerun every table and conclusion that depends on Sections 3.13–3.14.

Until then, remove the claim that unseen-drug generalisation is reachable through target or MoA retrieval.

---

## Finding 9 — The rank-normalisation rationale is mathematically wrong

**Severity:** High

Section 3.1 says that:

- library normalisation makes the rank representation meaningful;
- otherwise rank tracks sequencing depth;
- the log transform prevents ultra-high genes from occupying the head.

Within a single cell:

- multiplying all counts by the same positive CP10K factor preserves every gene ordering;
- applying the monotone transform `log(1+x)` also preserves every ordering.

Neither operation can change the cell sentence’s within-cell rank.

These transformations can matter for:

- pseudobulk averaging;
- Euclidean distances;
- residual magnitudes;
- differential-expression calculations;
- optimal-transport targets in expression space.

They do not matter for the order of genes in one cell sentence.

### Required action

Correct the rationale and specify separately:

1. how ranks are constructed;
2. how expression-space statistics are constructed;
3. whether pseudobulks average raw counts, CP10K values, or log-transformed values;
4. why averaging transformed values is preferred to aggregating counts before normalisation.

---

## Finding 10 — The PCA probe cannot validate gene-identity mapping

**Severity:** High

The thesis uses a PCA cell-line classifier as validation that Tahoe token identifiers were mapped to the correct gene symbols.

A fixed permutation of gene columns preserves:

- pairwise Euclidean distances;
- PCA geometry up to a corresponding permutation;
- cell-line classification performance.

The proposed validation can therefore remain excellent even when every gene label is wrong.

### Required validation

- sample token IDs and compare them directly with `gene_metadata["token_id"]`;
- verify known housekeeping and lineage markers;
- compare reconstructed per-gene totals with trusted Tahoe summaries;
- test several cells against an independently decoded source;
- assert exact round-trip token ID → gene symbol → panel index mappings.

---

## Finding 11 — Tahoe preprocessing and QC are underreported

**Severity:** High

The thesis gives panel size, broad cell counts, and minimum cells per condition, but it does not fully document:

- `pass_filter` use;
- UMI thresholds;
- gene-count thresholds;
- mitochondrial-read thresholds;
- singlet or demultiplexing criteria;
- doublet handling;
- species filtering;
- barcode uniqueness and joins;
- shard selection;
- random seeds for cell sampling;
- condition balancing;
- duplicate sample handling;
- the relationship between the 50 raw lines and 47 high-quality lines in the Tahoe paper.

The public Tahoe paper distinguishes:

- approximately 1,100 perturbations at varied doses;
- 379 distinct drugs;
- 1,135 drug-dose combinations after QC;
- 47 high-quality cell lines for downstream analyses.

The thesis says “some 1,100 small molecules across 50 cancer cell lines,” which reads as 1,100 distinct compounds. The populations used in the thesis—351 drugs in broad characterisation and 106 in the residual cache—also need reconciliation with the released atlas and its QC.

### Required action

Add a preprocessing and cohort-flow table:

`raw atlas → cell QC → condition QC → panel intersection → sampled cache → split inventory → reliability-retained training set → evaluated support`.

Every exclusion should have a count and reason.

---

## Finding 12 — The L1000-derived 946-gene panel is defensible but not biologically neutral

**Severity:** Moderate

The context-window argument is reasonable: a full transcriptome does not fit comfortably when the prompt carries an instruction and multiple cell sentences. The L1000 landmark panel is also a recognised perturbation-signature panel.

However, the panel may omit:

- low-expression regulators;
- drug targets;
- lineage-specific markers;
- adaptive-resistance genes;
- context-specific response genes;
- genes selected as variable or responsive in Tahoe itself.

No sensitivity analysis establishes that the main conclusions are stable under another panel.

### Required action

At minimum, compare the central retrieval and lookup results on:

- the 946-gene L1000 intersection;
- Tahoe highly variable genes of similar size;
- Tahoe perturbation-responsive genes of similar size.

If retraining is too expensive, run the generation-free measurement and baseline analyses on all three panels and scope the model claim accordingly.

---

## Finding 13 — Dose is insufficiently separated from drug identity

**Severity:** High

The thesis reports that 62% of drugs have a real dose series. Several “per-drug” quantities average across dose distributions:

- drug lookup;
- the proposed `β(d)` main effect;
- cross-context transfer;
- one-cell-line lookup;
- some channel comparisons.

This means that a “drug main effect” can partly be a dose-mixture effect. Different cell lines or wells may be represented at different concentrations.

### Required action

- Use molar-resolved, dose-matched comparisons where possible.
- Report a same-dose drug lookup.
- Model log-dose or dose strata explicitly.
- Separate:
  - within-dose cross-line transfer;
  - within-line cross-dose transfer;
  - pooled-dose transfer.
- Do not call a pooled quantity a pure drug main effect.

---

## Finding 14 — The residual task is condition-level signature generation, not individual-cell counterfactual prediction

**Severity:** High

The residual model assigns a condition-level pseudobulk signature to many control-cell prompts. Every control cell in a condition can therefore share the same target.

This removes most single-cell response heterogeneity and creates an easy shortcut:

> drug name → condition-average residual signature

The model need not use the input control cell to perform that task.

This matters because the Introduction frames the task as predicting what an untreated cell would have expressed after treatment. The residual arm does not evaluate that individual counterfactual.

### Required controls

- scramble the control cell while holding the drug fixed;
- scramble the cell-line label while holding the drug fixed;
- remove the control sentence entirely;
- compare a drug-only decoder with the full prompt;
- quantify whether the output changes with baseline cellular state;
- report condition-level signature generation as a separate task from single-cell transition prediction.

---

## Finding 15 — The NIR definition needs tighter mathematical specification

**Severity:** Moderate

The NIR equation treats `σ` as a higher-is-better similarity. The prose later calls Euclidean distance in expression space a similarity. If Euclidean distance is used, the formula must use:

- `σ = −distance`; or
- a reversed inequality.

The equation should also state explicitly whether the target condition is excluded from the comparison set.

The claim that chance is exactly 0.5 is strongest when:

- the target is excluded;
- ties receive half credit;
- the predictor is independent of target identity;
- comparison members are exchangeable under the null.

### Required action

Write the implemented definition exactly, including:

- target exclusion;
- distance sign;
- tie handling;
- comparison-set construction;
- minimum comparison-set size.

Add deterministic unit examples for perfect, random, tied-zero, and reversed predictions.

---

## Finding 16 — The “model is drug-blind” conclusion needs tighter scope

**Severity:** High

The strongest calibrated NIR drug-blindness evidence is on tier-2 prompts. The tier-1 scramble evidence in Table 5 uses DE-Δr, which Section 3.2 has already ruled inadmissible for drug-use claims.

The output-invariance experiment also uses only eight independent contexts, but its interval resamples hundreds of pairwise similarities as if they were independent. Pairs share:

- contexts;
- drugs;
- generations;
- control cells.

The interval is therefore anti-conservative.

### Defensible claim

> On the evaluated fine-tuning-held-out tier-2 prompts, the original single-cell-target checkpoint showed no detectable drug-header effect under the calibrated NIR/scramble evaluation.

### Required action

- Repeat calibrated NIR and scramble tests on seen drugs.
- Resample output invariance at the context level, with drug and generation nesting.
- Use multiple generation seeds.
- Avoid a model-wide drug-blindness claim until both seen and held-out conditions are covered.

---

## Finding 17 — Linear decodability does not establish a biological drug representation

**Severity:** High

The probe reads the final prompt position after the prompt literally states:

- drug name;
- mechanism.

High drug classification accuracy therefore demonstrates that lexical drug-label information remains in the residual stream. That is expected. It does not establish that the representation contains:

- the drug’s transcriptional effect;
- its mechanism in a biologically usable form;
- a context-specific response;
- information learned from Tahoe rather than inherited from the text model.

The folds are also stratified by prompt rather than grouped by treatment well.

### Required controls

- probe the base checkpoint;
- probe opaque drug IDs;
- decode response-signature clusters rather than drug labels;
- group cross-validation by well;
- test held-out drugs or mechanisms;
- compare name-only and mechanism-only prompts;
- decode from positions that do not contain a direct lexical copy.

### Correct interpretation

> Drug-label information is linearly recoverable from the residual stream.

Do not automatically translate that into:

> The model learned a biological drug representation.

---

## Finding 18 — The ablation result is not a “causal variance share”

**Severity:** High

The thesis projects a purified drug-correlated subspace out of activations and measures KL change. It then compares this effect with random and cell-line subspaces and describes a causal variance share near 3%.

Problems include:

- the subspace is estimated at the final prompt position but removed at every sequence position;
- projection can remove any function correlated with the drug classes, not specifically drug identity;
- the ratio of KL effects is not a statistical variance decomposition;
- the interventions are non-additive;
- the n=60 logit-space measurement has no interval;
- the cell-line and drug subspaces differ in geometry and semantic breadth.

### Defensible interpretation

> Removing a drug-correlated low-rank subspace changes the model’s logits more than a matched random subspace and less than a cell-line-correlated positive control.

That is a causal intervention result. Calling it a causal variance share is too strong.

---

## Finding 19 — The identity-substitution normaliser argument is algebraically incorrect

**Severity:** High

Section 3.6 claims that the full log-sum-exp normaliser cancels in:

`log P(g_B | prompt_A) − log P(g_A | prompt_A)`.

The normaliser cancels for a token-level logit contrast evaluated under the same hidden state and prefix. Here, `g_A` and `g_B` are different teacher-forced response sequences.

After the first differing token:

- the prefixes differ;
- the hidden states differ;
- the vocabulary normalisers differ.

The later normalisers therefore do not cancel algebraically.

Additional concerns:

- target strings may differ in length;
- mean token log probability weights sequences differently;
- class-mean displacements are injected broadly although the subspace was estimated at one position;
- ordered-pair clustering does not capture dependence among pairs that share drug A or drug B;
- the detected effect is small, appears at one layer, and is absent in a sibling residual arm.

### Required action

- use same-prefix token-level logit contrasts;
- match response lengths or use aligned positions;
- perform natural activation patching between alternate prompts;
- add free-generation validation;
- use a dyadic or multi-membership variance estimator for pairs sharing either drug;
- state the result as low-gain sensitivity, not proof that natural generation reads a biological drug code.

---

## Finding 20 — Residual re-encoding does not isolate tokenisation as the binding constraint

**Severity:** High

The residual intervention changes several factors simultaneously:

- single-cell target → condition-level pseudobulk target;
- full expression → control-referenced shift;
- shift → generic-subtracted residual;
- unsigned expression ordering → signed up/down ordering;
- long target → shorter target;
- common-token frequency;
- response entropy;
- sequence length;
- number of response tokens seen per optimizer update;
- decoding difficulty;
- effective training budget.

The thesis’s token-overlap measurement is useful:

- approximately 83% of top-200 full-profile gene positions are shared across drugs;
- residual encoding increases cross-drug differences.

It does not show that only a corresponding small fraction of the gradient carries drug identity. BPE tokenisation further separates gene positions from loss-token positions.

### Required factorial experiment

Train replicated arms with matched optimizer updates and matched response-token budgets:

1. original full-profile target;
2. length-matched full-profile target;
3. control-referenced shift only;
4. generic-subtracted residual;
5. signed versus unsigned residual;
6. pseudobulk full profile;
7. pseudobulk shift;
8. pseudobulk residual.

This would separate:

- aggregation;
- control subtraction;
- generic subtraction;
- sign encoding;
- length;
- token divergence.

### Current defensible claim

> The residual target package improved prompt sensitivity.

### Current unsupported claim

> Target tokenisation, rather than data, objective, optimization, or target aggregation, is the uniquely identified binding constraint.

---

## Finding 21 — The optimal-transport experiment is overinterpreted

**Severity:** Moderate to high

The OT analysis is technically interesting but does not support the strength of the “entire ε-ladder” conclusion.

Concerns:

1. A barycentric combination of observed cells lies in their convex hull, not necessarily on the biological data manifold.
2. Balanced OT with uniform marginals assumes conserved mass, while drugs can induce:
   - death;
   - arrest;
   - proliferation changes;
   - subpopulation depletion.
3. A strong cell-line probe in the released 10-dimensional scVI latent does not validate local geometry for state matching.
4. The empirical ladder contains:
   - a near-single-cell endpoint;
   - one finite OT value (`ε = 0.5`);
   - a consensus endpoint.
5. The consensus arm stopped at roughly 60% of one epoch and had an unscoreable tier-1 NIR.

### Required action

- Replace “entire ε-ladder” with the exact tested target constructions.
- Do not claim that barycentric targets are on-manifold.
- Justify balanced versus unbalanced OT.
- Report a real ε sweep or call the result a three-point target comparison.
- Separate OT target quality from training adequacy.

---

## Finding 22 — Training adequacy is insufficient for a broad negative verdict

**Severity:** High

Most conclusions rest on:

- one 1B checkpoint;
- one learning rate;
- approximately one epoch;
- generally one training seed;
- stochastic decoding at temperature 0.8;
- one target panel;
- limited hyperparameter exploration.

The consensus arm did not complete an epoch. Learning curves, held-out losses, convergence diagnostics, and model-size comparisons are absent.

The negative result therefore applies to:

> This specific checkpoint, target construction, fine-tuning recipe, and evaluation.

It does not establish:

- failure of C2S generally;
- failure of larger language models;
- failure of autoregressive perturbation modelling;
- failure of foundation models;
- failure of perturbation-specialised checkpoints.

### Required action

At minimum:

- run multiple training seeds for the decisive original and residual arms;
- show training and validation loss curves;
- report generation validity for every arm;
- compare greedy or logit-space evaluation with stochastic sampling;
- test one additional model scale if feasible;
- otherwise narrow every general statement to the 1B configuration.

---

## Finding 23 — Generation uncertainty and validity are incompletely reported

**Severity:** High

The thesis correctly notes that an earlier 600-token cap truncated 26% of generations and materially changed the measured effect. It promises to report:

- completion rate;
- valid panel-gene fraction;
- duplicate rate.

Those statistics are not consistently present before each generation result.

Most intervals resample conditions while treating the generated outputs as fixed. They omit:

- generation-seed variability;
- training-seed variability;
- checkpoint variability;
- decoding stochasticity.

A fixed per-condition seed makes contrasts reproducible. It does not estimate these uncertainty components.

### Required action

For each generation-based table, report:

- checkpoint seed;
- generation seeds;
- number of generations per condition;
- temperature, top-p, top-k, repetition penalty, and maximum tokens;
- `[END CELL]` and `[DOWN]` completion;
- valid-gene fraction;
- duplicate-gene fraction;
- truncation rate;
- within-checkpoint generation spread;
- between-checkpoint spread.

Use a hierarchical analysis or present seed-level results directly.

---

## Finding 24 — The uncertainty framework is inconsistent across analyses

**Severity:** High

Section 3.1 says intervals are generally two-way cluster-robust over cell line and treatment well. The Limitations instead say every interval is a cell-line clustered bootstrap. Individual analyses use still other units:

- ordered drug pairs;
- cell line;
- drug;
- cell line and well;
- drug and well;
- unclustered permutations;
- pairwise similarity rows.

This is not necessarily wrong—different estimands can require different dependence structures—but the thesis does not provide a reliable analysis-by-analysis map.

Specific risks include:

- output invariance resampling pairwise similarities;
- identity tests clustering ordered `(A,B)` pairs while pairs sharing A or B remain dependent;
- channel analyses omitting target-drug or annotation-family dependence;
- repeated drugs across lines creating a third crossed axis;
- post-selected comparators receiving ordinary intervals.

### Required action

Add an inference table with one row per reported result:

| Result | Observation | Independent assignment | Generalisation axis | Cluster variables | Reference distribution | Multiplicity family |

Use the estimator appropriate to each claim and state when a fully crossed three-way estimator is not feasible.

---

## Finding 25 — The orthogonal scramble is outcome-selected

**Severity:** High

The `orth` comparator is called neutral because its observed score is near 0.5. The partner stratum is selected using true response geometry, and neutrality is then evaluated on the same response data.

This is useful diagnostically, but it is not a fully independent prespecified null.

### Required action

Cross-fit the comparator:

- use one truth half to select near/orthogonal/opposite partners;
- use a disjoint truth half to score them;
- or predefine similarity bands using training data only.

Report the opposite arm as an active negative treatment, not as a null.

---

## Finding 26 — The baseline suite is still incomplete

**Severity:** High for broad comparative claims

The thesis includes important simple baselines, especially:

- control-copy;
- generic;
- linear map;
- per-drug lookup;
- one-line lookup;
- MoA lookup.

It does not compare the final task on the same split against major perturbation-prediction methods, including:

- scGen;
- CPA;
- chemCPA;
- CellOT;
- CondOT;
- GEARS where applicable;
- BioLord and related compositional models;
- PerturBench reference baselines;
- Tahoe-x1 or its released state-transition evaluation.

By August 2026, Tahoe-x1 is a direct perturbation-trained foundation-model comparator involving Tahoe-100M. A literature-wide statement that no method reaches the target cannot stand without discussing it.

### Required action

Choose one:

1. Run at least one recognised non-LLM perturbation model and one strong linear/kNN baseline on the exact split; or
2. Remove claims about what “the literature” reaches and scope the comparison to methods evaluated in this thesis.

The second is acceptable for a Master’s thesis.

---

## Finding 27 — Biological interpretation is too thin

**Severity:** Moderate

The thesis repeatedly interprets the generic component as a stress and cell-cycle programme. It does not show:

- gene-set enrichment;
- representative genes;
- pathway activity;
- dose dependence;
- toxicity association;
- cell-cycle phase shifts;
- examples of specific drug–cell-line responses.

The interaction or non-shared component is also not related to:

- lineage;
- driver mutations;
- target expression;
- pathway state;
- resistance markers;
- tissue of origin.

### Required action

Add a compact biological validation section:

- enrichment of the generic component;
- two or three representative drugs;
- one transferable and one context-specific response;
- relation to known targets or pathways;
- sensitivity to dose;
- cautious discussion of annotation quality.

This would materially improve the computational-biology depth without requiring a new model.

---

## Finding 28 — External validity is narrow

**Severity:** Moderate

The residual arm covers:

- 106 drugs;
- 50 cancer cell lines;
- a sampled subset of Tahoe;
- a 946-gene panel;
- one treatment-time regime;
- condition-level pseudobulks;
- no primary cells or donors;
- no independent dataset.

The thesis should not generalise to:

- primary human tissues;
- unseen donors;
- non-cancer contexts;
- other treatment durations;
- other sequencing technologies;
- full-transcriptome prediction;
- individual-cell counterfactuals.

The current Limitations acknowledge some but not all of this scope.

---

## Finding 29 — The literature review is focused but incomplete

**Severity:** Moderate to high

The review handles the metric-calibration dispute well and appropriately discusses:

- C2S/C2S-Scale;
- scGPT and Geneformer;
- simple-baseline critiques;
- DrEval;
- Miller et al.;
- Connectivity Map;
- interpretability interventions;
- optimal transport.

It gives too little attention to the methods most directly related to chemical perturbation prediction:

- CPA and chemCPA;
- scGen;
- CellOT and CondOT;
- compositional and latent-state perturbation models;
- modern benchmark suites such as PerturBench and scPerturBench;
- direct Tahoe-100M prediction work such as Tahoe-x1.

Some of these entries appear in the bibliography database but not in the printed argument.

### Required action

Expand the review or narrow the claimed literature contribution. A useful comparison should distinguish:

- genetic versus chemical perturbation;
- seen-drug versus unseen-drug tasks;
- seen-context versus unseen-context tasks;
- single-cell distribution prediction versus pseudobulk signature prediction;
- autoregressive token generation versus latent-state transition;
- simple retrieval versus compositional prediction.

---

## Finding 30 — The PDF is visibly unfinished

**Severity:** Submission blocker

The document lacks or contains unfinished standard front matter:

- no title page;
- no author/degree/institution information;
- no abstract;
- Acknowledgements contain `[TODO: write at the end]`.

It also lacks a sufficient reproducibility statement:

- no code-availability declaration;
- no immutable repository commit;
- no dataset snapshot;
- no exact environment;
- no software versions;
- no hardware description;
- no complete optimizer/scheduler settings in the main methods;
- no exact decoding configuration;
- no released split lists or run manifest.

There are figure and table inconsistencies, including the Figure 6 panel/content mismatch and several contradictory captions.

### Required action

Complete the document as a finished academic artifact before further grading.

---

# Part III — Claim-by-claim adjudication

| Thesis claim | Committee verdict | Defensible replacement |
|---|---|---|
| DE-Δr can be saturated by a zero-information predictor | **Accepted** | Keep; rerun uncertainty only if an interval is needed |
| NIR is the only calibrated metric | **Descriptively plausible; inferentially unverified** | NIR is the only tested metric with positive observed DRF; rerun cluster-correct inference |
| The original model is drug-blind | **Accepted only on the evaluated tier-2 configuration** | No detectable drug-header use was found for the tested checkpoint and tier-2 prompts |
| Drug identity is represented | **Accepted only as lexical-label decodability** | Drug-label information remains linearly decodable from the residual stream |
| Drug identity is represented but not read | **Too strong** | Drug-correlated information is decodable; interventions suggest low-gain downstream sensitivity |
| The causal variance share of drug is approximately 3% | **Rejected** | Drug-subspace ablation changes logits less than the cell-line positive control |
| The identity substitution proves the readout barely consults drug identity | **Suggestive, not established** | A small layer-specific intervention effect was observed under an imperfect contrast |
| No content-preserving target fix works | **Narrow** | The tested single-cell, one-ε OT, and incomplete consensus arms remained near null |
| Tokenisation is the binding constraint | **Not isolated** | The residual target package increased prompt sensitivity |
| Re-encoding raises drug-specific target exposure | **Accepted descriptively** | Residual/shift encodings produce more cross-drug target divergence |
| Re-encoding creates prompt sensitivity | **Accepted** | The residual model responds to the combined drug-and-mechanism prompt on trained conditions |
| Re-encoding generalises to unseen combinations | **Not established** | Whole-well held-out performance is compatible with zero and with a modest positive effect |
| The unseen-drug slope proves zero-shot drug knowledge | **Rejected** | Scrambled partner prompts modulate outputs in unseen-drug scoring contexts |
| A lookup table wins | **Accepted qualitatively** | A training-only per-drug average strongly outperforms the model on common support |
| The lookup nearly reaches the biological ceiling | **Rejected wording** | The lookup approaches or exceeds a within-well split-half precision reference |
| Approximately 45% is drug × cell-line interaction | **Rejected** | Approximately 45% is unshared across compared contexts under a descriptive estimator that mixes axes |
| No learnable structure exists in the interaction | **Rejected** | No structure was detected by the tested low-reliability analyses |
| Protein target and MoA make unseen-drug prediction reachable | **Pending corrected rerun** | Metadata channels may carry signal; the current retrieval pool is not training-only |
| Chemical structure is closed | **Rejected** | The tested chemistry retrieval is weak and control-dependent; nonlinear structure models remain untested |
| The model reads the name but not mechanism | **Inconsistent and not final** | Sensitivity to the combined header is established; field attribution requires one canonical corrected run |
| No method in the literature reaches the interaction | **Unsupported** | No method evaluated in this thesis reaches the unshared component |

---

# Part IV — Required revision plan

## Priority 0 — Mandatory before submission

### P0.1 Freeze one canonical run

**Acceptance criterion:** Every headline number can be traced to one immutable artifact and appears identically in the Introduction, body, tables, figures, Limitations, and Conclusions.

### P0.2 Correct the DRF inference

**Acceptance criterion:** The artifact reports the actual number of independent cell-line or line×plate clusters, uses a null-valid test, and recomputes Holm correction.

### P0.3 Remove or redesign the 45% interaction claim

**Acceptance criterion:** No part of the thesis calls the current quantity a cell-line interaction variance share unless the estimand is reidentified under a valid design.

### P0.4 Repair or remove Sections 3.13–3.14 claims

**Acceptance criterion:** All channel predictions use only training conditions as partners, target only the intended held-out population, and use three-state inference.

### P0.5 Reconcile all held-out evaluations

**Acceptance criterion:** One whole-well split, one unfiltered evaluation inventory, one set of sample sizes, and one set of intervals.

### P0.6 Correct factual and mathematical errors

At minimum:

- rank-normalisation rationale;
- PCA mapping validation;
- Pythia pretraining provenance;
- transfer-coefficient bias direction;
- log-normaliser cancellation;
- OT “on-manifold” language;
- split-half “ceiling” terminology.

### P0.7 Finish the document

Add:

- title page;
- abstract;
- completed acknowledgements;
- code/data availability;
- run-manifest appendix;
- exact training and decoding settings.

Remove:

- TODOs;
- “in flight” statements;
- superseded results;
- unresolved contradictory captions.

---

## Priority 1 — Strongly recommended for a defensible high grade

### P1.1 Replicate decisive model comparisons

Run at least three training seeds for:

- original single-cell target;
- residual target.

Use multiple generation seeds per checkpoint.

### P1.2 Add a target-factorial ablation

The minimum useful set is:

- original full profile;
- length-matched full profile;
- pseudobulk full profile;
- control-referenced shift;
- signed residual.

Match optimizer updates and response-token budgets.

### P1.3 Test whether the residual model uses the control cell

Add:

- control-cell scramble;
- cell-line-label scramble;
- drug-only prompt;
- no-control prompt.

### P1.4 Add one recognised perturbation-model baseline

Prefer:

- chemCPA or CPA for drug conditioning;
- CellOT/CondOT for state transition;
- Tahoe-x1 state-transition baseline if its exact task can be aligned.

### P1.5 Add biological validation

Include:

- enrichment of the generic component;
- representative drug–cell-line examples;
- dose effects;
- known mechanism or target concordance.

---

## Priority 2 — Valuable extensions, not required for degree completion

- Test a larger C2S checkpoint.
- Evaluate a full-transcriptome or larger-panel generation-free benchmark.
- Use independent Plate 6/Plate 14 treatment overlap.
- Evaluate unseen cell lines on a dataset with independent treatment assignments.
- Fit a proper multilevel variance model on a crossed, replicated design.
- Test leave-one-mechanism-out and opaque drug identifiers.
- Develop a learned drug-conditioned modulation architecture.

---

# Part V — Recommended final thesis narrative

The revised thesis should not try to preserve every current headline. A stronger and more defensible narrative is:

## Claim 1 — The standard metric can fail catastrophically

> In this Tahoe-100M evaluation, DE-Δr can assign a near-perfect score to a zero-information predictor because truth-selected genes and control reversion create an exploitable direction.

## Claim 2 — The tested original C2S fine-tune does not measurably use the drug header on the calibrated held-out evaluation

> Under within-plate NIR and a drug-only prompt scramble, the tested Pythia-1B C2S fine-tune showed no detectable drug-specific effect on fine-tuning-held-out tier-2 prompts.

## Claim 3 — A residual target changes model behaviour

> Replacing the full-profile target with a signed condition-level residual increased cross-drug target divergence and produced measurable sensitivity to the combined drug-and-mechanism prompt on trained conditions.

## Claim 4 — Sensitivity is not fidelity

> The residual model’s predictions remained far below a training-only per-drug lookup, and whole-well held-out context-specific transfer was not established.

## Claim 5 — The remaining context-specific target is unresolved

> A descriptive cross-context analysis indicates substantial non-sharing of drug-specific residuals, but the present design cannot separate cell line, dose, well, and estimator effects.

## Claim 6 — Unseen-drug channels remain open

> Protein target, mechanism, and chemistry channels require a corrected training-only retrieval analysis before claims about unseen-drug reachability can be made.

This version is still novel, interesting, and scientifically consequential. It is also much harder to attack in a defense.

---

# Part VI — Oral-defense questions

## Experimental design and data

1. What is the independent treatment-assignment unit in Tahoe-100M?
2. Why is a split of cells from one well not a biological replicate?
3. Which conditions overlap between Tahoe Plate 6 and Plate 14, and why were they not used as an independent validation?
4. How were `pass_filter`, mitochondrial content, UMI count, doublets, singlets, and low-cell conditions handled?
5. Why does the thesis use 50 cell lines when the Tahoe paper reports 47 high-quality lines for downstream analysis?
6. Are the approximately 1,100 items distinct compounds, perturbations, or drug-dose combinations?
7. How were the 600,000 treated cells sampled across drugs, doses, cell lines, plates, and shards?
8. What biological population is represented by the 106-drug residual cache?

## Preprocessing and representation

9. Why should CP10K scaling or a monotone log transform alter within-cell gene rank?
10. How can a PCA classifier detect a fixed permutation of gene labels?
11. Why is the L1000 landmark panel the right panel for context-specific drug response?
12. How stable are the findings under Tahoe HVGs or Tahoe-responsive genes?
13. Why average per-cell log-normalised expression instead of aggregating raw counts before normalisation?

## Metrics and inference

14. Why does the DRF artifact contain 25 cell lines but report 1,820 cell-line clusters?
15. What null distribution supports the bootstrap p-value?
16. Is the target condition excluded from NIR’s comparison set?
17. Is Euclidean distance negated before using the higher-is-better NIR formula?
18. Which cluster variables correspond to each headline claim?
19. How are training-seed and generation-seed variability represented?
20. Why is the orthogonal comparator a valid null if its neutrality was selected on observed outcomes?

## Transfer and interaction

21. Derive the equality between disattenuated cosine and the claimed variance-component ratio.
22. Which assumptions of that derivation are tested in Tahoe?
23. How are dose, treatment well, and residual-estimation effects separated from cell-line interaction?
24. Why does optimistic reliability make the unshared fraction smaller rather than larger?
25. What experimental design would actually identify drug × cell-line interaction?

## Mechanistic interpretation

26. Does the probe decode biological response information or the literal drug name in the prompt?
27. What happens in the base Pythia/C2S checkpoint before Tahoe fine-tuning?
28. Why is a ratio of KL changes a causal variance share?
29. Why should a subspace estimated at the final prompt position be removed at every sequence position?
30. How can the sequence log-normaliser cancel after the teacher-forced prefixes diverge?
31. Why does the small identity effect appear at one layer and fail in the sibling residual arm?

## Re-encoding

32. Which experiment separates target length from residual biology?
33. Which experiment separates pseudobulk denoising from token divergence?
34. Which experiment separates control subtraction from generic subtraction?
35. Does the residual model use the control cell at all?
36. Why does top-gene overlap directly measure how much gradient carries drug identity under BPE tokenisation?

## Generalisation and novelty

37. What exactly does `unseen_drug` mean after natural-language Pythia pretraining?
38. Why does the prompt contain the held-out drug’s MoA?
39. Where does the unseen-drug slope score the real held-out-drug prompt?
40. How does the channel-gate code prevent held-out residuals from entering the retrieval pool?
41. Why are CPA, chemCPA, CellOT/CondOT, and Tahoe-x1 not compared on the same split?
42. What is genuinely novel: the model, the biology, the metric audit, or the diagnostic protocol?

## Provenance

43. Which exact checkpoint, target digest, and result JSON produced Table 13?
44. Which numerical version of the lookup/model/reference comparison is final?
45. Why do the Introduction, Table 14, boxed claim, and Conclusion report different values?
46. Which analyses were still pending when the PDF was compiled?

---

# Part VII — Final committee comments

## On correctness

The thesis contains both excellent methodological reasoning and several serious correctness failures. The strongest design insights—experimental-unit recognition, plate-aware comparison, the DE-Δr exploit, and the hard lookup baseline—are credible. The current DRF uncertainty, interaction interpretation, channel gate, and parts of the mechanistic narrative are not.

## On novelty

The work is novel primarily as a diagnostic and evaluation study:

- adversarial audit of a cell-sentence drug-response model;
- demonstration of an exploitable standard metric;
- prompt-level drug scramble;
- physical-well split audit;
- residual target intervention;
- strong per-drug retrieval comparison.

It is not yet:

- a novel state-of-the-art predictive model;
- a general solution to unseen-drug prediction;
- a clean measurement of drug × cell-line interaction;
- a biological discovery about response pathways.

## On methods

The methods are ambitious and often unusually careful, but the thesis attempts too many inferential layers at once:

- metric validity;
- behavioural sensitivity;
- mechanistic representation;
- causal use;
- target encoding;
- context transfer;
- variance decomposition;
- unseen-drug channels.

The result is a rich investigation whose central spine is obscured by analyses that are not equally mature. Removing or demoting the weakest claims would improve the thesis more than adding another speculative experiment.

## On biological contribution

The computational evaluation is much stronger than the biological interpretation. Adding pathway enrichment and a few grounded drug–cell-line examples would make the work read as computational biology rather than primarily machine-learning evaluation.

## On presentation

The writing quality is high enough for an excellent thesis. The present score is reduced by document-state failures, not prose quality. Once the artifact is internally consistent, the narrative style will become one of the thesis’s main strengths.

## Final grade

**21/31 for the current PDF.**

The scientific core is a clear pass. The document requires major correction before submission. If the canonical run is frozen, invalid inference is repaired or removed, the central claims are narrowed, and the thesis is completed as an academic document, the work could reasonably move into a substantially higher grade band.

---

# Selected external works that should be addressed

- Zhang et al. **Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas for Context-Dependent Gene Function and Cellular Modeling.** bioRxiv, 2025. <https://doi.org/10.1101/2025.02.20.639398>
- Rizvi et al. **Scaling Large Language Models for Next-Generation Single-Cell Analysis.** bioRxiv, 2025–2026. <https://doi.org/10.1101/2025.04.14.648850>
- C2S-Scale-Pythia-1B model card. <https://huggingface.co/vandijklab/C2S-Scale-Pythia-1b-pt>
- Lotfollahi et al. **Predicting cellular responses to complex perturbations in high-throughput screens (CPA).** Molecular Systems Biology, 2023. <https://doi.org/10.15252/msb.202211517>
- Hetzel et al. **Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution (chemCPA).** <https://arxiv.org/abs/2204.13545>
- Bunne et al. **Learning single-cell perturbation responses using neural optimal transport (CellOT).** Nature Methods, 2023. <https://doi.org/10.1038/s41592-023-01969-x>
- Wu et al. **PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis.** NeurIPS Datasets and Benchmarks, 2025. <https://proceedings.neurips.cc/paper_files/paper/2025/file/8aee537279a66ced96319dfca3c00002-Paper-Datasets_and_Benchmarks_Track.pdf>
- Wei et al. **Benchmarking algorithms for generalizable single-cell perturbation response prediction (scPerturBench).** Nature Methods, 2025. <https://doi.org/10.1038/s41592-025-02980-0>
- Gandhi et al. **Tahoe-x1: Scaling Perturbation-Trained Single-Cell Foundation Models to 3 Billion Parameters.** bioRxiv, 2025. <https://doi.org/10.1101/2025.10.23.683759>


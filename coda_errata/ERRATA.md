# Errata: what three audits found, what was changed, and what the runs returned

**Part I** is the register written before any job ran. **Part II** is what the first two jobs
returned -- including two results that overturn Part I, one of which overturns a fix Part I
presented as complete. **Part III** is the channel gate and the rebuild, and contains the first
result that touches a printed claim. **Part IV** is the retrain and its evaluation, and contains
two retractions. **Part V** is the transfer coefficient and the metric audit, and contains none --
both claims under repair came back confirmed. **Part VI** is the fourth audit: one blocker that
reaches the printed numbers, and a second retraction that has to be softened.

This document is appended to as work proceeds rather than written at the end, so the order of
discovery is visible and nothing is reconstructed after the fact.

**For the auditors.** This is the register of defects found between `823f447` and `294b890`, what
each one was, why it mattered, and exactly what was done about it. It is written to be checked, not
believed. Every fix names a file and a test.

Three audits are folded in. Where they overlap it is noted, because agreement between independent
readers is stronger evidence than either alone. Where an audit was wrong, that is noted too.

**Current state.** All five queued jobs have now run and are reported in Parts II-IV.

| | status | where |
|---|---|---|
| artifact scan | done | Part II |
| experimental-unit audit | done | Part II |
| plate-matched channel gate | done -- Q18 survives with smaller numbers | Part III |
| target rebuild | done -- plate scope, threshold resolved from the data | Part III |
| retrain + evaluation | done -- **two printed claims retracted** | Part IV |
| transfer coefficient (scope 7) | done -- T = 0.553, dose arm real, controls pass | Part V |
| metric calibration (scope 8) | done -- NIR the only calibrated metric under Holm | Part V |

**Parts I-III were written before later parts existed and are left as written**, so the order of
discovery stays visible. Where a later part supersedes a statement, a forward pointer says so
inline. Nothing has been reconstructed after the fact.

| audit | read the tree at | net |
|---|---|---|
| A — the original defensibility review | pre-`762951a` | transductive holdout, experimental unit, oracle baselines |
| B — Kimi K3 | ~`823f447` | plate-confounded channel gate, broken eval stack, missing artifacts |
| C — GPT 5.6 SOL | `823f447` + my mid-edit tree | fail-open plate scope, outcome-selected holdout, wrong validation format |

---

## Index of defects

Numbered by order of DISCOVERY, so the sequence is not always the document order. Line numbers are
for this file.

| # | part | line | defect |
|---|---|---|---|
| 1 | I | 97 | The fit saw the holdout |
| 2 | I | 127 | The generic was cell-line-scoped, in a frame already measured as contaminated |
| 3 | I | 153 | The reliability measurement reproduced a flaw the repository had already documented |
| 4 | I | 180 | The generic was a mean over conditions, and contained the condition's own drug |
| 5 | I | 202 | "Plate scope" silently became cell-line scope |
| 6 | I | 227 | The held-out set was selected on its own outcome |
| 7 | I | 251 | `--split_unit` was bypassed whenever tier files supplied the split |
| 8 | I | 276 | The experimental unit was never recorded, and neither was the dose |
| 9 | I | 312 | A second, independent dose defect, in a different pipeline |
| 10 | I | 338 | Molar doses collapsed to a single group |
| 11 | I | 357 | The channel gate measured plates as well as channels |
| 12 | I | 393 | We deleted the evaluation stack's API and did not notice |
| 13 | I | 421 | Five defects in the evaluator itself (Step 6) |
| 14 | I | 458 | The retrain validated against the wrong output format |
| 15 | I | 477 | Six quoted result sets have no committed artifact |
| 16 | I | 502 | Thesis language asserted more than the evidence carries |
| 17 | II | 624 | the auditing tool manufactured the gap it reported |
| 18 | II | 707 | our own verdict was a count where it needed a rate |
| 19 | II | 730 | our clustering fix was wrong, and wrong in the anti-conservative direction |
| 20 | II | 766 | the well-level split would have collapsed the training set |
| 21 | III | 1065 | the reliability threshold was inherited across a change of measurement scale |
| 22 | III | 930 | the continuity column is a re-measurement, not a reproduction |
| 23 | III | 1138 | the auto threshold crashed the job it was written to improve |
| 24 | IV | 1287 | the evaluation scored against a different quantity than the model learned |
| 25 | IV | 1349 | the scramble comparator was not a null |
| 26 | V | 1688 | the dyadic estimator omitted the drug, and the omission announced itself |
| 27 | V | 1720 | the dose arm was reading another arm's variable |
| 28 | V | 1736 | the structure-matched control had no multiway interval |
| 29 | V | 1754 | the by-split table measured against a comparator that is not a null |
| 30 | V | 1781 | a broken commit, caught by the check that follows every edit |
| 31 | VI | 1923 | the memorisation premium is carried by drugs that appear in only one arm |
| 32 | VI | 1962 | `max_abs_cos` is a maximum of two means |
| 33 | VI | 1990 | "hedges rather than confabulates" is not a measured quantity |
| 34 | VI | 2005 | the coverage headline compares two different populations |
| 35 | VI | 2026 | the channel gate was never brought into the repaired frame |
| 36 | VI | 2045 | the reference distribution ignores the cluster count |
| 37 | VI | 2077 | the run that carries the argument had the field decomposition switched off |
| 38 | VI | 2095 | `crossed_bootstrap` is a subsampler |
| 39 | VII | 2235 | reconstructing a tuple from the manifest key would have reintroduced the defect |
| 40 | VII | 2246 | the channel gate could not import `inference` at all |
| 41 | VII | 2256 | a `fit_digest` mismatch only logged, and my own tests were failing it silently |
| 42 | VII | 2276 | the rebuild job would have used the wrong threshold, and the fix for that would have crashed |

---

# Part I -- the register, written before any job ran

## The defects, one at a time

### 1. The fit saw the holdout

**Why did this matter?**

**Answer:** `build_residual_targets.run()` computed the generic shift and applied the reliability
filter over *every* condition, then assigned the train/holdout split afterwards. Each held-out target
was therefore defined partly by the other held-out conditions, so the cross-context transfer number
(+0.1002) was measured against targets built with knowledge of the holdout.

**What changed?** `run()` is four stages that each see only what the stage before may expose:
inventory (metadata and cell counts only) → split (metadata only, before any pseudobulk exists) → fit
(`Generic.__init__` is handed the train keys explicitly) → transform (held-out conditions are
projected through the fitted generic and never enter it).

**How can you check it?** `tests/test_split_before_fit.py` is a poison test: corrupt every held-out
expression vector, rebuild, require the training JSONL to be byte-identical and `fit.sha` — a hash of
every fitted quantity — unchanged. Reintroduce the defect with

```bash
sed -i 's/gen = Generic(shifts, conds, train_keys, shrink_k=args.shrink_k,/gen = Generic(shifts, conds, list(conds), shrink_k=args.shrink_k,/' endcell/ot/build_residual_targets.py
```

and it fails on the digest. `git checkout` to restore. The negative control —
`test_poisoning_a_training_condition_does_change_things` — requires both to move when a *training*
condition is poisoned, which is what stops the gate passing vacuously.

*Found by audit A. Commit `41c6716`.*

---

### 2. The generic was cell-line-scoped, in a frame already measured as contaminated

**Why did this matter?**

**Answer:** `FINDINGS.md:519-523` had already measured that cell-line scope leaves plate structure in
the target — the same-plate null reads **+0.478** against **−0.018** at plate scope, and the
shared-minus-split control bias **+0.254** against **−0.000**. `variance_decomposition.py:727` has had
a `--generic_scope` flag since that analysis. `build_residual_targets.py` did not: it was hardcoded to
cell-line scope. So the contamination was measured and then trained on anyway.

**What changed?** `--generic_scope` now exists in the builder and defaults to `plate`, meaning
`(cell_line, plate)`.

**What about the retention cost?** This is worth stating because an earlier draft of our own
verification report got it wrong in the direction that flattered the old choice. That draft cited "62%
of conditions reproducible at cell-line scope against 19% at plate scope" as the honest price of the
change. Those figures are **retracted at `FINDINGS.md:532`**: they were measured with *shared*
controls, which inflate cell-line reproducibility far more than plate. With split controls the two are
comparable — **20% plate against 16% cell line** — and the same entry concludes plate scope is the
better build. Retention is not an argument for cell-line scope.

*Found by audit A; the retracted-number error was found by re-auditing our own report. Commits
`41c6716`, `6c76a7d`.*

---

### 3. The reliability measurement reproduced a flaw the repository had already documented

**Why did this matter?**

**Answer:** `repro_cos` is `cos(res_A, res_B)`, meant to be the agreement of two independent
measurements of one condition. `compute_shifts` subtracted the *same* control pseudobulk from both
halves, so the two shared their entire control-noise term and were correlated for a reason unrelated
to the drug. `variance_decomposition.py:55` had already written this mechanism down, and
`FINDINGS.md:532` retracts a scope comparison because of it — yet the new builder reproduced it.

It matters specifically because the inflation is **not neutral between scopes**: it favours cell-line
scope. So `--scope_sensitivity`, the table that decides whether the rebuild proceeds at plate scope,
was biased against the rebuild it exists to adjudicate.

**What changed?** Half A is measured against control half A and half B against control half B, so the
two residuals share nothing. The *full* shift still uses every control cell, because that is the
training target rather than a reliability estimate. `--shared_control_reliability` reproduces the old
behaviour for a side-by-side.

**How can you check it?** `test_reliability_halves_do_not_share_a_control` asserts the full shift is
unaffected, the halves are not, and the shared-control version reports strictly higher reliability.

*Found while re-auditing our own verification report; corroborated by the repo's own prior note.
Commit `b74a171`.*

---

### 4. The generic was a mean over conditions, and contained the condition's own drug

**Why did this matter?**

**Answer:** Two separate arithmetic errors. A drug measured at five doses contributed five times the
weight of a single-dose drug. And once the generic became train-only, an asymmetry appeared that had
not existed before: a *train* condition's generic contains its own drug while a *held-out* condition's
does not, so the two splits would have been scored against differently defined targets and the
generalisation gap would partly measure that difference.

**What changed?** The generic is a mean over **drugs** — each drug's conditions averaged first — and
`Generic.value()` excludes the condition's own drug from its own generic, so the definition is
identical on both sides: "the mean over training drugs other than this one, within scope".

**How can you check it?** `test_dose_weighting_does_not_let_one_drug_count_twice` (two doses of X at
6.0 and one of Y at 0.0 must give 3.0, not the condition-weighted 4.0) and the
`leave-one-drug-out removes the condition's own drug` line in `--selftest`.

*Leave-one-drug-out was not reported by any audit; it surfaced while fixing defect 1. Commit `41c6716`.*

---

### 5. "Plate scope" silently became cell-line scope

**Why did this matter?**

**Answer:** `Generic.value()` returned the cell-line generic whenever a plate group had no other
training drug. So `--generic_scope plate --shrink_k 0` was not fail-closed plate scope: an unreported
subset of conditions received cell-line-scoped targets — the contaminated frame the entire rebuild
exists to leave — and nothing in the output said which ones.

**What changed?** A plate group with fewer than `--min_plate_drugs` (default 3) other training drugs
now yields `None`, the condition is dropped, and the count is reported. With `--shrink_k > 0` the
blend toward the cell-line generic is a **declared estimator** rather than a hidden fallback, and
`report.json` names the frame accordingly: `"plate"` at `shrink_k=0`, otherwise
`"hierarchical(shrink_k=…)"`.

`Generic.export()` also ignored shrinkage, so `reconstruction.npz` would have inverted the targets
with a different definition of the same quantity than had built them. Fixed in the same pass.

**How can you check it?** `test_plate_scope_fails_closed_instead_of_falling_back_to_cell_line` and
`test_export_uses_the_same_frame_as_the_targets`.

*Found by audit C. Commit `48528e0`.*

---

### 6. The held-out set was selected on its own outcome

**Why did this matter?**

**Answer:** The reliability filter dropped held-out conditions for failing their own reproducibility
threshold. That is selection on the outcome: the evaluation set stops being representative of the
conditions the split defined, and the reported performance is conditioned on the targets happening to
be measurable.

Our verification report had flagged this as an open question we were not confident about. Audit C
ruled on it, and we agree with the ruling.

**What changed?** The primary result now uses the complete metadata-eligible holdout.
`--eval_repro_filter` opts into the filtered version as a **labelled sensitivity**, and `repro_cos` is
written on every example so the filtered arm can be computed afterwards without another build.

**How can you check it?** `test_the_held_out_set_is_not_selected_on_its_own_outcome` asserts the
guarantee directly: moving the reliability threshold must change the *training* count and must leave
the held-out count untouched.

*Found by audit C. Commit `48528e0`.*

---

### 7. `--split_unit` was bypassed whenever tier files supplied the split

**Why did this matter?**

**Answer:** `--split_unit sample` was only honoured on the random-split path. Whenever the tier files
produced a split it was silently ignored, so the estimand quietly reverted to condition-level
leave-pairs-out with treated wells shared across the boundary. Which of two scientifically different
claims the thesis makes depended on which holdout source happened to fire.

**What changed?** The flag applies to every split. Promotion takes the strictest label present
(`unseen_drug` > `unseen_combo` > `train`), so it never moves a condition *into* training. §9 of the
run plan now builds **both** units, making the estimand a reported comparison rather than an accident.

**The residue we cannot engineer away:** a `(drug, cell_line)` holdout *cannot* stop a held-out
condition sharing a treated well with a training one, because one well carries many cell lines. That
is the experimental design, not our code. `sample_crossing_report()` counts it into `holdout.json` as
`frac_heldout_conditions_sharing_a_well_with_train`, so it is a number in Methods rather than a hedge.

**How can you check it?** `test_sample_split_unit_is_honoured_even_when_tier_files_supply_the_split`
and `test_well_crossing_is_counted_rather_than_assumed`.

*Found by audit C, and the scientific framing is theirs. Commit `48528e0`.*

---

### 8. The experimental unit was never recorded, and neither was the dose

**Why did this matter?**

**Answer:** Tahoe assigns a treatment to a **sample/well** holding a mixture of cell lines; the
per-line profiles are deconvolved observations nested inside one assignment. Everything keyed on
`(drug, cell_line, plate, dose)`, which identifies no physical experiment — and whose fourth element
was a sample identifier anyway.

The origin is one line. `build_embeddings.py:134`, before the fix:

```python
drug, cl, plate, dose = row.get("drug"), row.get("cell_line_id"), row.get("plate"), row.get("sample")
```

`row["sample"]` — the treated well — assigned to a variable named `dose`, then written to a column
named `dose`. That is the entire origin of `metadata["dose_float"] == "smp_1841"`, and it means the
well identity was **mislabelled, not lost**. The cache is enriched in place rather than rebuilt, which
was the largest cost uncertainty in the whole remediation.

**What changed?** Three places. `build_embeddings.py` writes `sample_id`, so a reader doing
`meta["dose"]` now gets a `KeyError` instead of silently receiving well identifiers.
`tahoe_design.sample_column()` recovers the identifier from existing caches and reports which branch it
took. The concentration is resolved from `drugname_drugconc` through `parse_treatment` → `Dose.molar`,
which is `None` with a stated reason when unconvertible and never `0.0`. Emitted metadata is
`sample_id` / `dose_molar` / `dose_raw`; `dose_float` is gone. Combination samples are dropped rather
than analysed as their first component.

Both builders write to a `.partial` path and rename only after checking the emitted dose column with
`looks_like_sample_id`, so `REFUSING TO WRITE` is literally true. (It was not: the check originally ran
after the file was closed.)

*Found by audit A. Commits `762951a`, `41c6716`, `b74a171`.*

---

### 9. A second, independent dose defect, in a different pipeline

**Why did this matter?**

**Answer:** `tahoe_c2s_preprocess_endcell.py:972` computes `dose_float = float(dose_str.split()[0])`,
discarding the unit. Two opposite failure modes: **collision** (1 µM and 1 nM both become 1.0) and
**split** (0.05 µM and 50 nM are one concentration but become 0.05 and 50.0). Tier 4 — the held-out
dose test — is decided at lines 1006-1008 by exactly that comparison, so a collision means tier 4 held
out a dose it also trained on.

Note this is a *different* defect in a *different* pipeline from defect 8: the original preprocessing
writes the well identifier correctly, under `metadata["sample"]`. Only the caches got that wrong.

**What changed?** Nothing is asserted. Whether either mode actually bit is a property of the data, so
`experimental_unit_audit.py` question 5b counts collisions and splits per drug from
`sample_metadata.parquet`. **That job has not run**, so at present this is a demonstrated possibility,
not a demonstrated fact.

> **-> Resolved in Part II.** It ran: **0 collisions, 0 splits, 289/289 samples yield a molar dose.**
> Every drug states its concentrations in a single unit, so unit-stripping never bit and tier 4 is
> sound. The possibility was real and did not materialise.

*Found by us while patching readers for defect 8. Commit `41c6716`.*

---

### 10. Molar doses collapsed to a single group

**Why did this matter?**

**Answer:** `dose_response_analysis.py` groups on `round(float(dose), 6)`. `round(5e-8, 6)` and
`round(5e-7, 6)` are **both 0.0**, so every molar concentration falls into one group and a
dose-response curve reads a single point.

This is a regression we introduced: it became live the moment our fix for defect 8 made `dose_molar`
the preferred field. Before that, the field held either a sample-ID string (rejected) or a µM-scale
float, where rounding to six places was harmless.

**What changed?** Grouping uses `tahoe_design.molar_key()`, which normalises on significant figures —
so 1000 nM and 1 µM are one group, and 0.05 µM and 5 µM are two.

*Found by audit C. Commit `48528e0`.*

---

### 11. The channel gate measured plates as well as channels

**Why did this matter?**

**Answer:** `Conclusions.tex:123-129` states two live channels — MoA at **+0.078** and target at
**+0.084** over a count-matched random null. The gate's estimand requires the arm and its null to
differ in exactly one respect. They differ in two: mechanism- and target-paired drugs are largely
**co-plated** (a mechanism series is laid out together), drugs drawn at random are co-plated at the
background rate, and the residuals being averaged come from the cell-line-scoped build that
`FINDINGS.md:523` measures as plate-retaining. Some unknown share of +0.078 is plate, not mechanism.

This is the same class of defect as the two probe retractions this project has already published — a
control matched on one axis and silently unmatched on another. We did not catch it; audit B did, and
the instrument for fixing it (`kappa_channel.py`'s different-plate pair policy) was already in the
repository.

**What changed?** `channel_gate.py` now reports three constructions side by side — `count_matched`
(the published null, kept for continuity), `plate_matched` (same partner count **and** the same
same-plate/different-plate split), and `different_plate` (both arms restricted to non-shared plates) —
plus a **co-plating diagnostic printed before any verdict**. If the channel and the null are co-plated
at the same rate, there is no confound and the published number stands as measured.

The verdict is now the plate-matched gap. A matched null spanning fewer than `--min_lines_for_verdict`
cell lines yields **UNTESTABLE** rather than a verdict: "we could not build the control" must never be
written as "closed".

**How can you check it?** `--selftest` plants worlds with a plate effect and **no channel biology at
all**, partners 90% co-plated. Over 12 worlds the count-matched null reads **+0.384** and the
plate-matched null **+0.016**. The parameters are deliberately adverse, so that is not a claim about
Tahoe's true magnitude — it establishes that the construction can manufacture an effect several times
the size of the one being defended.

*Found by audit B. Commit `b7a2827`.*

---

### 12. We deleted the evaluation stack's API and did not notice

**Why did this matter?**

**Answer:** Restructuring `run()` into stages (defect 1) deleted `build_residuals`, and five scripts
call it: `residual_eval`, `channel_gate`, `reconstructed_eval`, `reward_calibration`,
`build_de_weights`. That is the entire evaluation stack. Nothing caught it — the poison test exercises
`run()`, and none of the five has a test — so it would have surfaced on the cluster at the eval step of
a job whose training had already finished.

Audit B reported two callers; audit C reported two. There were five.

**What changed?** Restored as a shim over the new stages, plus a regression test asserting both that
the API exists and that the returned contract still matches what the consumers unpack.

**The shim's defaults reproduce the old, defective semantics on purpose** — cell-line scope, no
leave-one-out, shared controls. The published numbers came from that build, and an audit asks for every
quoted number to be regenerable; silently repairing them here would make the published figures
unreproducible and hide the size of each repair. Flags opt into the corrected frame, and a test asserts
the two frames actually differ.

**How can you check it?** `test_the_five_evaluation_consumers_still_have_their_api` and
`test_the_shim_reproduces_the_published_semantics_by_default`.

*Found by audits B and C independently. Commit `b74a171`.*

---

### 13. Five defects in the evaluator itself (Step 6)

**Why did this matter?**

**Answer:** Restoring the API would have produced a runnable evaluator computing invalid numbers. Audit
C's phrasing is right: this had to land *before* the retrain, not after, or the GPU slot produces
results that have to be discarded.

**What changed, one by one:**

- **Scramble partners.** `b != a` allowed the *same drug at another dose or plate* to be the partner.
  Substituting a drug for itself is not a scramble — an unchanged output is correct rather than blind —
  and the `near` stratum was the most contaminated, because a drug's own other doses are usually its
  nearest neighbours. That stratum is exactly what the monotone-gradient argument leans on. Partners
  are now a **different drug** and, by default, **on the same plate**, so the swap differs from the
  target in the drug alone. `--partner_policy any_plate` reproduces the published pool.
- **Gallery negatives.** The comparison set included the drug's own other doses, so every arm was
  scored on "can you tell this drug's 0.5 µM from its own 5 µM" rather than "is this closer to its own
  truth than to other drugs'". Same-drug siblings excluded.
- **Lookup baselines were oracles.** `drug_lookup`, `drug_lookup_1` and `moa_lookup` were fitted from
  the full cache, so on a held-out condition they read residuals the model was never shown. They now
  fit from **training conditions only**, and the full-cache versions are reported alongside as
  `drug_lookup_oracle` / `moa_lookup_oracle` — bounding how much the published comparison was inflated
  rather than hiding it.
- **Generation was unseeded.** `do_sample=True` advanced from global torch state, so no reported number
  could be reproduced exactly, and the two arms of a `model − scramble` contrast were drawn from
  different points of one stream. Each call now seeds a `torch.Generator` from (condition, arm, batch).
- **Intervals assumed cell lines are independent.** One well carries many cell lines, so lines within a
  well are one assignment observed several times. Every head-to-head and every scramble stratum now
  reports both a line-clustered and a well-clustered interval with the widening ratio, and **the well
  interval is what the verdict reads**.

*Scramble partners, negatives, oracles and seeding were audit A; well clustering was audits B and C.
Commit `04fd303`.*

---

### 14. The retrain validated against the wrong output format

**Why did this matter?**

**Answer:** The job trained on residual signed-DE targets (`<up> [DOWN] <down> [END_CELL]`) and
validated against `eval_tier1_seen_conditions.jsonl`, which is ordinary cell sentences. The validation
loss was measuring a different output distribution than the one being learned, so it could not detect
overfitting on the actual objective and was not comparable across arms.

**What changed?** `--val_frac` carves training **wells** into `residual_val.jsonl`, in the same format
as training, held out by well so a validation condition never shares a treated well with a training
one. §10 points `--eval_file` at it.

**How can you check it?** `test_a_residual_format_validation_shard_is_written`.

*Found by audit C. Commit `48528e0`.*

---

### 15. Six quoted result sets have no committed artifact

**Why did this matter?**

**Answer:** In a viva "show me this number" is a fair request for any number in the thesis. A local
scan finds **8 of 17** manifest entries with no file in `RESULTS_cluster/`: the channel gate, the probe
arms and their replication, the three κ runs, the field decomposition, and the variance decomposition
carrying the matched-null / dose / shared-split arms.

Audit B calls these "not established". That is right for what it could check, but probably not what
happened: those jobs ran and wrote into `~/tahoe/RESULTS`, and what is missing is the copy back.

**What changed?** `artifact_manifest.py` maps every quoted number to its artifact, its producing
script, and what it backs. It runs where the artifacts are, distinguishes "never computed" from "never
copied", and tars up what it finds. Whatever is still missing afterwards gets one of two honest
outcomes and no third: re-run, or withdraw the number.

This also reverses a cut we made unilaterally. `remediation_execution_scope.md` cut the artifact
infrastructure as "weeks of work that changes no number". That was right about the full
claim-registry-and-CI version and wrong about the minimum.

*Found by audit B; audit C independently required a minimal manifest. Commit `b7a2827`.*

---

### 16. Thesis language asserted more than the evidence carries

**Why did this matter?**

**Answer:** Audit C listed four phrases still claiming causally what is under re-measurement. All four
were present.

**What changed?**

| was | now |
|---|---|
| "The repair: it was the tokenisation" (`Results:417`) | "Changing what the tokens encode" |
| "That distinction turns out to be the whole story" (`Results:421`) | dropped; the set-vs-order limitation of `target_divergence` stated instead |
| "There is no memorisation premium" (`Results:544`, `Conclusions:60`) | restated as a reading, with the leaked generic and the shared-well caveat named |
| "And the repaired drug use generalises" (`Conclusions:58`) | "And the drug use appears to generalise", with the same two caveats |
| the unseen-drug channel numbers asserted (`Conclusions:120-129`) | the count-matched-but-not-plate-matched caveat stated inline, pending §11 |

*Found by audit C. Commit `48528e0`.*

---

## Where the audits disagreed with us, and we think they were wrong

**Audit C reported "39 passed, 4 failed, 9 errors" and two runtime blockers** — a `compute_shifts`
return-arity mismatch and a deleted `build_residuals`. That failure signature is our **mid-edit working
tree**, not any commit: the arity mismatch and its call site were fixed together in `b74a171`, which
was committed before the audit arrived. The `build_residuals` deletion was real (defect 12) and had
already been fixed from audit B's report. Current HEAD: **60 passed**.

The finding was still useful — it is why defect 12 has a regression test rather than just a fix.

**Audit B's magnitude claim for the channel gate (+0.19 to +0.28 spurious)** comes from a simulation we
cannot inspect, and its citation of the "one-difference rule at `channel_gate.py:8-9`" does not match
those lines. Neither affects the finding: the objection stands on the null construction alone, and our
own selftest reproduces the mechanism independently.

**Audit B described the rename as `build_conditions`.** It is `inventory`.

---

## What is still open

1. **No corrected number exists.** Every job below is queued; none has run. The corrected numbers may
   move in **either** direction — the leak and the plate structure inflate the shipped residual result,
   while the oracle lookups and the same-drug scramble partners deflate it. "Everything shrinks" is not
   the safe prediction.
   > **-> Resolved in Parts III and IV.** They ran, and the prediction held: the channel gate SHRANK
   > (+0.078 to +0.0749 plate-matched) while the scramble gap ROSE (+0.1002 to +0.1281) because the
   > comparator got sharper — and then collapsed to +0.0449 once the comparator was made neutral.
   > Movement in both directions, for different reasons.
2. **Step 2's job has not run**, so none of the experimental-unit questions is answered yet — including
   the pseudoreplication factor that sets the clustering unit, and whether defect 9 actually bit.
   > **-> Resolved in Part II.** 48.6 cell lines per treated well across 289 wells; 286 of 287
   > treatments occur in exactly one well, so there is effectively no independent-well replication;
   > defect 9 did not bite.
3. **The residual arm remains a per-condition retrieval task.** Every control cell in a condition gets
   the same target string. A clean retrain does not change that, and it is not a repair of the original
   perturbation-prediction estimand.
4. **Shrinkage selection.** Audit C rules that `k` must not be chosen from retention alone but must also
   pass a train-only plate-contamination null. Not implemented. If `--scope_sensitivity` sends us to
   `shrink_k > 0`, that null is required before the frame can be called anything but hierarchical.
5. **Generation seeds.** One seed makes the eval exploratory; three make it inferential. Which we get
   depends on the queue, and the wording will follow whichever happens.
6. **Commit `41c6716` is not clean.** Its `train_c2s_tahoe_endcell.py` changes are 22 insertions of ours
   (the seeding) and 146 insertions plus all 17 deletions of pre-existing uncommitted DE-weighting work
   swept in by `git add`. That code rewrites the loss path via `forward_loss`, is exercised by no test,
   and the retrain will run through it. Scope any review of that commit to the two seeding hunks.

---
---

# Part II — what the first two jobs actually returned

Written after Steps 1 and 2 ran on the cluster (2026-08-03, jobs at 02:41 and 02:48). Part I was
about machinery; this is the first real evidence. Two of these results overturn something Part I
said, and one of them overturns a fix Part I was proud of.

Steps 3–5 have not run. Nothing here touches a thesis number yet.

---

## Step 1 — the artifact scan

**Why did we run this?**

**Answer:** Audit B found six thesis-quoted result sets with no committed artifact and called them
"not established". We argued that was the right word for what it could check but probably not what
happened — that the jobs ran and the copy back was what went missing. That was a prediction, and it
needed testing rather than asserting.

**What came back** (the real scan; the first 95 log lines are the selftest running in a temp dir):

```
99:   MISS  variance_decomposition       [quoted] 0 file(s)
112:  ok    probe_arms                   [quoted] 5 file(s)
113:  ok    probe_replication            [quoted] 6 file(s)
114:  ok    channel_gate                 [quoted] 1 file(s)
116:  ok    kappa_plate                  [quoted] 1 file(s)
117:  ok    kappa_cellline               [quoted] 1 file(s)
118:  ok    kappa_channel                [quoted] 2 file(s)
119:  ok    field_decomp                 [quoted] 1 file(s)
120:  ok    residual_eval_1400           [quoted] 4 file(s)
122:  ok    residual_holdout2            [quoted] 1 file(s)
123:  ok    calibration                  [quoted] 2 file(s)
124:  ok    nir_benchmark                [quoted] 2 file(s)
125:  ok    dose_coverage                [quoted] 1 file(s)
126:  ok    leak_audit                   [quoted] 3 file(s)
129:  14/17 entries complete; 1 THESIS-QUOTED entries have no artifact here
133:  bundled 31 artifacts -> /home/3180408/tahoe/artifacts_bundle.tar.gz
```

**What it establishes:** the prediction holds. Every result audit B listed — the probe arms, the
replication, the channel gate, all three κ runs, the field decomposition — is present on the cluster.
They were computed and never copied back. F3 is a provenance failure, not a missing-computation one,
and it closes with an scp.

The three incomplete entries are `variance_decomposition` (below), plus
`channel_gate_platematched` and `residual_targets_repaired`, which are outputs of jobs that have not
run yet and are correctly marked not-quoted.

---

### Defect 17 — the auditing tool manufactured the gap it reported

**Why does this matter?**

**Answer:** `variance_decomposition` reported `0 file(s)`. That is not a real gap. `scan()` built its
found-list and missing-list in one pass with an index splice:

```python
(found if hits else missing).append(pat)
found[len(found) - 1:] = hits if hits else []
```

When a later pattern in a multi-file entry missed, the else-branch spliced `found[-1:] = []` and
**deleted a hit an earlier pattern had already recorded**. `bundle()` reads the same list, so those
files were also **left out of the tarball**. A script whose only purpose is finding missing artifacts
was fabricating them, and quietly dropping the evidence from the rescue bundle.

`variance_decomposition` was the only entry that could trip it — three exact filenames of which at
most one exists — which is exactly why it was the single remaining "gap" in a scan that otherwise
found 14 of 17. Its patterns were wrong too: the runs wrote `vardecomp_baseline`, `_plate`,
`_project`, `_proj_ctrl60` and `variance_decomp.json`, not the `_matched`/`_cellline` names the
manifest had invented.

**What changed:** patterns matched independently, every hit kept, globbed filenames, and a `PART`
status so "some of it is here" can no longer collapse into either `ok` or `MISS`.

**How can you check it?** `--selftest`, and re-running the scan: the first bundle of 31 artifacts is
superseded by the re-run's.

*Found by running our own tool on real data. Commit `c243e72`.*

---

## Step 2 — the experimental-unit audit

**Why did we run this?**

**Answer:** Audit A said the analysis keys on `(drug, cell_line, plate, dose)`, which identifies no
physical experiment. Audits B and C both said the clustering unit behind every confidence interval
was unresolved. This is the discovery step that settles what the design actually is.

**What came back:**

```
182:  verdict: YES   examples: ['smp_1783', 'smp_1784', 'smp_1785', 'smp_1786', 'smp_1787']
183:  -> sample identity is MISLABELLED, not lost. The cache can be enriched in place
187:  289 treatment samples, 600000 treated rows
188:  cell lines per sample: mean 48.6  median 49  range 43-50
193:  distinct (drug, dose) treatments: 287
194:  of which replicated in >1 sample: 1 (0%)
195:  samples per treatment: {1: 286, 3: 1}
205:  with a usable molar dose  : 289 (100%)
209:  COLLISIONS (one float, several concentrations): 0
210:  SPLITS     (one concentration, several floats): 0
211:  -> every drug states its doses in a single unit, so unit-stripping happened to be harmless
216:  samples CROSSING the split: 124
222:  -> the split is broken at the ASSIGNMENT level.
```

Four things are settled by this, two of them in the thesis's favour.

**The dose defect never bit.** 289 of 289 samples yield a usable molar dose, and unit-stripping
produced **zero** collisions and **zero** splits. Every drug states its doses in one unit, so tier 4
is sound and defect 9 is closed as a possibility that did not materialise. This is the outcome we
could not assert without measuring, and it is worth noting that the measurement was cheap and the
assertion would have been wrong in the cautious direction.

**Sample identity is recoverable**, confirming the Part I prediction and the enrich-not-rebuild path.

**The pseudoreplication factor is 48.6** cell lines per treated well, tightly distributed (median 49,
range 43–50), across 289 wells.

**There is effectively no independent-well replication.** 286 of 287 distinct (drug, dose) treatments
occur in exactly one well; the histogram is `{1: 286, 3: 1}`.

**124 wells carry conditions on both sides of the existing holdout.** Against 289 wells total, the
crossing is severe rather than marginal.

The combination-sample count and the exact denominator for the crossing figure were not in our grep
pattern; both are in `RESULTS/experimental_unit_audit.json`, which is in the bundle.

---

### Defect 18 — our own verdict was a count where it needed a rate

**Why does this matter?**

**Answer:** The audit printed `-> independent-well replication EXISTS` on the strength of one
replicated treatment out of 287. The condition was `n_replicated > 0`, which is true and useless: one
treatment cannot estimate reproducibility. Had we read the verdict rather than the numbers, we would
have concluded that Workstream H's replicate arm was available when it is not.

**What changed:** the verdict now requires a count (≥20) and a rate (≥5%), and the negative branch
states both consequences explicitly. `repro_cos` is split-half sampling precision **within one well**
and must not be called biological reproducibility anywhere in the thesis.

**The consequence cuts in the thesis's favour, which is why it needs saying plainly.** The
disattenuation divisor in the transfer coefficient is therefore a within-well reliability. A
within-well estimate is optimistic relative to true biological reliability, and it sits in the
denominator — so the ~45% drug × cell-line interaction share is, if anything, **understated**. Audit
B made the same observation independently.

*Found by reading the real output against our own verdict logic. Commit `294b890`.*

---

### Defect 19 — our clustering fix was wrong, and wrong in the anti-conservative direction

**Why does this matter?**

**Answer:** Part I reported a fix for the independence unit: every interval also reported a
well-clustered version, and the well interval decided, on the reasoning that the well is the
assignment unit. The measured numbers show that reasoning was incomplete.

There are **289 wells and roughly 50 cell lines**. Clustering on wells therefore uses *more* clusters
than clustering on cell lines and yields a **narrower** interval. "Use the well, it is the assignment
unit" would have made every headline interval tighter while sounding more rigorous.

The design is **crossed, not nested**: a well contains ~49 cell lines, and each cell line appears in
many wells. Two conditions sharing a well share the drug assignment and the batch; two conditions
sharing a cell line share the biology. Neither grouping alone captures both dependencies.

**What changed:** `_two_way_ci` implements the Cameron–Gelbach–Miller two-way cluster-robust variance,

```
V_2way = V_line + V_well − V_intersection
```

where the intersection of the two groupings is the condition itself, so `V_intersection` is the
ordinary independent-sampling variance. It falls back to the wider one-way variance when the
estimator goes negative in finite samples and records which branch it took. Every head-to-head, every
scramble stratum and the headline verdict now read it; the line-clustered interval is still printed
so the change is visible.

**We flag this as the most instructive error of the round.** It was introduced *while fixing* a
finding two audits agreed on, it passed its tests, and it would have tightened exactly the intervals
under scrutiny. Only the measured cluster counts exposed it.

*Found by reading 48.6 lines/well and 289 wells against our own fix. Commit `294b890`.*

---

### Defect 20 — the well-level split would have collapsed the training set

**Why does this matter?**

**Answer:** Audit C required `--split_unit sample` to be honoured whatever produced the split, and we
implemented that by promoting a condition-level split to the well level. With the measured design
that is catastrophic. A tier split holds out (drug, cell_line) **pairs**; a well spans 48.6 cell
lines; so a well is held out if **any** of its pairs is. At a 15% pair holdout that is
`1 − 0.85^48.6`, which is indistinguishable from 1. Essentially every well would have been held out,
the training set would have gone to nothing, and the rebuild would have consumed its queue slot
producing an unusable build.

**What changed:** when the estimand is the well, the holdout is drawn **at the well level from the
start** and the tier files define the unseen-drug set only. `make_holdout_by_sample` never takes a
drug's last training well — without that guard a two-well drug could lose both, silently becoming an
unseen *drug* while sitting in the unseen_combo arm, which would make the two arms measure the same
thing. And `--min_train_frac` (default 0.30) refuses to build rather than emit a near-empty training
set, so this class of error surfaces in a one-hour CPU job instead of after a twenty-hour GPU one.

**How can you check it?** `test_promoting_a_tier_split_to_wells_would_collapse_training_and_is_refused`
and `test_sample_holdout_never_takes_a_drugs_last_training_well`.

*Found by reasoning about `--split_unit sample` against the measured 48.6, before Step 4 ran. Commit
`d33bea1`.*

---

## The structural finding, which we think is the most important thing in Part II

**Why does this change what the thesis can claim?**

**Answer:** A drug-dose in Tahoe is applied to **one well containing ~49 cell lines simultaneously**.
It follows directly that there is no way for a drug to be present in cell line A and absent in cell
line B at the same dose — they are physically the same well. So the two available holdouts are:

- **Condition-level** — hold out (drug, cell_line) pairs. The held-out condition's well is in
  training through its other ~48 cell lines. This is unavoidable, and 124 crossing wells is the
  measured size of it.
- **Well-level** — hold out whole wells. This removes the drug-dose from *every* cell line, so it is
  not cross-context transfer at all; it is unseen-drug-dose generalisation.

**Cross-context transfer, as the thesis currently frames it, is therefore not cleanly measurable in
this atlas.** That is a property of Tahoe's design, not of our code, and no rebuild fixes it.

Audit B put this in almost these words: the condition-level arm supports "internal, potentially
same-well matrix completion", not independent generalisation. The measured 48.6 turns that from a
plausible objection into a structural fact.

**What this costs:** a wording change in Results and Conclusions, not a retraction. The +0.1002 gap
remains a real measurement of something; what it measures is matrix completion within shared wells,
and it must be labelled that way. The rebuild will report both estimands side by side so the reader
can see the difference rather than take our word for which one was intended.

**What we have not done:** decided which estimand leads the chapter. We would value a ruling.

---

## Revised state of the open questions

| Part I said | Part II says |
|---|---|
| six result sets may be unrecoverable | all present on the cluster; provenance failure, closes with an scp |
| the tier-4 dose collision is an untested possibility | tested: 0 collisions, 0 splits. Closed. |
| the clustering unit is unresolved | resolved: crossed design, 48.6 lines/well, two-way cluster-robust |
| `repro_cos` may not be biological reproducibility | confirmed: 286/287 treatments are single-well. It is not. |
| well crossing is "a number for Methods" | 124 of 289 wells. Large enough to change the claim, not just footnote it. |
| cross-context transfer needs a caveat | it may not be measurable in this atlas at all |

Still open and unchanged from Part I: no corrected number exists; the residual arm remains a
per-condition retrieval task; shrinkage selection needs a train-only plate-contamination null if
`--scope_sensitivity` sends us there; one generation seed makes the eval exploratory; and commit
`41c6716` carries unrelated DE-weighting work on the training path.

**Commits in this round:** `c243e72` (manifest), `d33bea1` (well-level split, min_train_frac),
`294b890` (two-way clustering, replication verdict). 62 tests pass; six selftests pass.

---
---

# Part III — the channel gate and the rebuild

Steps 3 and 4, run 2026-08-03 at 03:01 and 03:20. This part contains the first result that touches a
printed claim, and one defect that only became visible once real numbers existed.

Step 5, the retrain, had not run when this part was written. **-> It is reported in Part IV, and it
carries two retractions.**

---

## Step 3 — the channel gate, re-measured against a plate-matched null

**Why did we run this?**

**Answer:** `Conclusions.tex:123-129` states two live channels — protein target at +0.084 and
mechanism at +0.078 over a count-matched random null. Audit B found that the null is matched on the
number of partners but not on plate, that mechanism-paired drugs are largely co-plated, and that the
residuals being averaged retain plate structure. Two differences where the estimand allows one. The
re-run reports three constructions side by side and prints the co-plating rates before any verdict,
so the confound is measured rather than argued about.

**Co-plating — how often a partner shares the target's plate:**

```
434:  target   channel 0.348   count-matched null 0.323   excess +0.025
435:  moa      channel 0.420   count-matched null 0.340   excess +0.080
436:  chem     channel 0.344   count-matched null 0.345   excess -0.001
```

**The three constructions:**

```
439:  target    count_matched         877  0.600   0.473   +0.1269 [+0.1046, +0.1482]
440:  target    plate_matched         877  0.600   0.503   +0.0973 [+0.0741, +0.1183]
441:  target    different_plate       790  0.360   0.296   +0.0643 [+0.0439, +0.0839]
442:  target    -> LIVE
443:  moa       count_matched        1033  0.586   0.479   +0.1067 [+0.0814, +0.1331]
444:  moa       plate_matched        1033  0.586   0.511   +0.0749 [+0.0538, +0.0948]
445:  moa       different_plate       874  0.296   0.262   +0.0344 [+0.0148, +0.0539]
446:  moa       -> LIVE
447:  chem      count_matched        4074  0.483   0.462   +0.0206 [+0.0085, +0.0320]
448:  chem      plate_matched        4074  0.483   0.467   +0.0162 [+0.0051, +0.0264]
449:  chem      different_plate      4072  0.235   0.229   +0.0056 [-0.0012, +0.0120]
450:  chem      -> closed
452:  >>> target, moa beats its PLATE-MATCHED null by more than 0.03.
```

**The verdict survives.** Both channels clear the plate-matched null with intervals excluding the
0.03 margin. Q18 stands, and so does the sentence in Conclusions — with smaller numbers and a
better-specified control.

**The confound was real but far smaller than audit B predicted.** Their κ memo implied ~90%
co-plating for mechanism partners against a ~20–25% background, and their simulation put the
spurious gap at +0.19 to +0.28. Measured: 42.0% against 34.0%, an excess of eight points, and the
correction costs about 0.03. The objection was valid; the magnitude estimate was not.

**Four things make the correction credible rather than merely applied.**

*The channel arm is identical across the first two rows* — 0.600, 0.586, 0.483 in both — because
only the null construction changed. Anything else would have meant the plate-matching leaked into the
arm.

*`chem` is a clean internal control.* Its co-plating excess is −0.001, and its null moves by +0.005
against +0.030 and +0.032 for the other two. The channel with no confound receives no correction.

*`chem` closes* (+0.0056, CI spanning zero at different-plate), which is the pre-registered
expectation — chemical structure was always the weak channel, and the SAR gate said so.

*The different-plate arm is not underpowered.* 790, 874 and 4072 conditions, nowhere near the 3–7%
survival the κ memo predicted, and both real channels stay positive under it.

**One thing we cannot fully explain, stated rather than smoothed over.** The correction is not
proportional to the co-plating excess: `moa` has three times `target`'s excess (+0.080 vs +0.025) but
both nulls move by about the same amount (+0.032 vs +0.030). `chem` at zero excess moving by +0.005
gives the right qualitative ordering at the extremes, but the middle is not a clean dose-response.
Co-plating rate is evidently not the only thing that differs between the pools.

**A number worth knowing before a viva.** In the different-plate arm the absolute NIRs are *below
chance* — 0.360 and 0.296 against a chance level of 0.5, with nulls at 0.296 and 0.262. Cross-plate,
every prediction in this frame is worse than random in absolute terms. Both arms degrade together so
the gap remains interpretable, but the level says the cell-line-scoped residual frame is dominated by
plate. That is independent support for the plate-scoped rebuild.

---

### Defect 22 — the continuity column is a re-measurement, not a reproduction

**Why does this matter?**

**Answer:** The `count_matched` column was retained to reproduce the published number. It does not:
it reads +0.1269 and +0.1067 where the thesis says +0.084 and +0.078.

The cause is in the compatibility shim. The original `build_residuals` shuffled every condition's
half-split from **one shared RNG stream**; the rewrite seeds **per condition** (`_stable_rng(seed, k)`),
which is better practice and makes the build order-independent. But different halves give different
`repro_cos`, and since only about a fifth of conditions clear the reliability threshold, small changes
near the threshold flip membership. A different set of conditions is being scored.

So the shim reproduces the published **definitions**, not the published **realisation**, and the
manifest's wording claiming otherwise needs correcting.

**The implication is larger than the bookkeeping.** If the gap moves from +0.078 to +0.107 purely
from re-randomising a half-split, then the published interval [+0.056, +0.103] never included
half-split variability and understates the real uncertainty. Any interval in this family that is
quoted from a single split assignment has the same gap in its error budget.

*Found by comparing the re-run against the thesis. Not yet fixed; the honest options are to seed-sweep
the half-split or to widen the quoted intervals accordingly.*

---

## Step 4 — the rebuild, and the decision gate

**Why did we run this?**

**Answer:** To rebuild the residual targets with the leakage, the scope and the experimental unit all
repaired, and to produce the three numbers that decide whether the retrain is worth a GPU slot: what
plate scope retains, how much training data survives, and how much well crossing each estimand
carries.

**Inventory** — identical in both arms:

```
16:  inventory: 6617 conditions over 268 treatment samples (max 44 cell lines nested in one sample);
     dropped 0 combination, 7426 too-few-cells, 0 no-control-group;
     0 conditions have no recoverable molar dose
```

Zero combination samples and zero unrecoverable doses, consistent with Part II. The 7,426 dropped for
too-few-cells is the `--min_treated 40` filter and is a pre-split, metadata-only criterion.

### The scope question is settled, and it settles in the rebuild's favour

At the inherited threshold (0.2):

```
24:  scope cell_line       mean cos=+0.115  retained 16% of 4999 train conditions
25:  scope plate           mean cos=+0.127  retained 19% of 4980 train conditions
26:  scope plate+shrink5   mean cos=+0.127  retained 19% of 4999 train conditions
27:  scope plate+shrink20  mean cos=+0.125  retained 19% of 4999 train conditions
```

and at the resolved threshold (+0.1086), from the re-run:

```
39:  scope cell_line       mean cos=+0.115  retained 47% of 4999 train conditions
40:  scope plate           mean cos=+0.127  retained 51% of 4980 train conditions
41:  scope plate+shrink5   mean cos=+0.127  retained 51% of 4999 train conditions
42:  scope plate+shrink20  mean cos=+0.125  retained 50% of 4999 train conditions
```

**The ordering is the same at both thresholds** -- plate above cell line, shrinkage flat -- so the
scope conclusion does not depend on where the reliability bar is set. That is worth more than either
table on its own: the decision is robust to the parameter that turned out to be inherited.

**Plate scope wins on both axes.** It removes the plate structure cell-line scope was measured to
retain (+0.478 against −0.018), *and* it retains more conditions — 19% against 16%. There is no
trade-off to manage. This is exactly what `FINDINGS.md:532`'s split-control retraction predicted (20%
plate, 16% cell line), now confirmed independently on the real cache by a different code path.

**Shrinkage buys nothing.** 19% at k=0, k=5 and k=20 alike. So `--shrink_k 0` is correct, the frame is
labelled `"plate"` rather than `"hierarchical"`, and audit C's requirement for a train-only
plate-contamination null before using shrinkage is moot — we are not using it.

The fail-closed plate rule from defect 5 cost 19 of 4,999 conditions (4999 → 4980), which is the
0.4% of plate groups with fewer than three other training drugs. Cheap, and now visible instead of
silently falling back to the contaminated frame.

### The two estimands, side by side

```
[A] sample split
19:  holdout (by sample): 4999 train | 904 unseen_combo (36 wells) | 714 unseen_drug (10 drugs)
20:  split promoted to the treated well: 0 conditions relabelled
21:  well crossing: 0/268 wells contribute to both sides; 0.0% of held-out conditions share a well

[B] condition split
53:  holdout (tier-aligned): 5735 train | 37 unseen_combo (tier3) | 845 unseen_drug (tier2)
55:  well crossing: 142/268 wells contribute to both sides; 23.0% of held-out conditions share a well
```

The sample split delivers exactly what it promised: **zero well crossing**. And the promotion pass
relabelled **zero** conditions, which confirms the defect-20 fix — the well-level draw was already
clean, so nothing needed promoting.

The condition split carries 23.0% crossing. That is the measured price of the cross-context estimand
and it is now a number in the manifest rather than a hedge in prose.

Both arms are well powered on the evaluation side: [A] 900 unseen_combo and 711 unseen_drug after
filtering; [B] 251 and 844. ([B]'s train count drops from 5,735 to 5,520 because the combo top-up
moved 215 conditions into `unseen_combo`, taking it from 37 to 252 — the two numbers reconcile
exactly.)

### The gate did not pass cleanly on the first run

```
31:  reliability [train]: mean cos(A,B)=+0.127; KEPT 948/4999 (19%) at cos > 0.2
39:  wrote 56564 examples from 948 conditions
```

948 training conditions against the shipped build's ~4,100. About a quarter of the data. That is not
a regression in the build -- it is the next defect.

**Resolved on the re-run.** With `--repro_thr auto` the same build yields:

```
45:  reliability [train]:       KEPT 2523/4999 (50%) at cos > 0.10856742188334463
44:  reliability [unseen_drug]: KEPT 711/714 (100%)
46:  reliability [unseen_combo]: KEPT 900/904 (100%)
61:  wrote 147710 examples from 2523 conditions
62:  wrote 3600 validation examples (4 wells)
```

2,523 training conditions and 147,710 examples -- 2.6x the first attempt, and a retention fraction
(50%) close enough to the shipped build's (62%) that the retrain differs from `holdout2` mainly in
target construction rather than in data volume. The held-out arms are retained at ~100% by design,
because the reliability filter no longer applies to them (defect 6).

---

### Defect 21 — the reliability threshold was inherited across a change of measurement scale

**Why does this matter?**

**Answer:** `--repro_thr 0.2` was chosen when reliability was measured with **shared** controls,
where it kept 62% of conditions (`FINDINGS.md:393`). Defect 3 replaced that with split controls,
which is a different measurement on a different scale — the mean drops from roughly 0.6 to +0.127.
The number 0.2 did not change, so nobody noticed that a moderate quality bar had silently become a
stringent one.

The threshold is the single largest determinant of the training set, and it had been inherited rather
than chosen.

**What we did about it:** rather than pick a new number, the builder now reports the whole retention
curve against a null. The null pairs half A of one condition with half B of a **different** condition
— same encoding, same dimensionality, same noise, no shared biology.

```
30:  observed cos(res_A, res_B): mean +0.127  median +0.110  n=4980
31:  null (half A of one condition vs half B of another): mean -0.000  95th pct +0.109  99th pct +0.175
32:    thr +0.000  keeps  92.9%  (4628 conditions)
33:    thr +0.050  keeps  77.7%  (3870 conditions)
34:    thr +0.100  keeps  54.6%  (2719 conditions)
35:    thr +0.109  keeps  50.5%  (2516 conditions)  (null 95th)
36:    thr +0.150  keeps  33.5%  (1667 conditions)
37:    thr +0.175  keeps  25.3%  (1258 conditions)  (null 99th)
38:    thr +0.200  keeps  19.0%  (948 conditions)  <- in use
39:    thr +0.250  keeps  11.1%  (552 conditions)
40:    thr +0.300  keeps   6.4%  (318 conditions)
41:  -> the fraction of conditions with a residual that reproduces at all is 51%
```

The condition-split arm gives the same picture independently: observed mean +0.132, median +0.111;
null mean +0.001, 95th +0.107; 51% above the null. The calibration does not depend on which split ran.

**A result, not just a configuration note.** The null is centred at zero, so the construction is
sound. And the observed **median (+0.110) sits essentially on the null's 95th percentile (+0.109)**.
Half the conditions in this atlas have a drug-specific residual whose two WITHIN-WELL halves agree no
better than two unrelated conditions' halves do.

**WORDING CORRECTED IN PART VI.** This paragraph originally read "about 51% reproduce at all ... a
statement about how much drug-specific signal Tahoe contains at this depth", which contradicts
defect 18 four hundred lines above -- the rule that `repro_cos` measures within-well SAMPLING
PRECISION and must never be called reproducibility. A fourth audit caught this register breaking its
own rule. Defect 18 wins. The defensible statement is:

> 50.5% of eligible training conditions had within-well split-half residual cosines exceeding the
> pooled 95th percentile of mismatched-condition half pairs, under this target estimator and this
> half-split seed.

It is NOT an estimate of how many conditions carry biological drug signal, because 286 of 287
drug-dose treatments have no independent-well replicate to measure that against. It belongs in
Results as a target-quality criterion, labelled as one.

**The implied false-discovery rate**, read off the curve as (null fraction above t) / (observed
fraction above t):

| threshold | keeps | conditions | implied FDR |
|---|---|---|---|
| +0.109 (null 95th) | 50.5% | 2,516 | ~10% |
| +0.175 (null 99th) | 25.3% | 1,258 | ~4% |
| +0.200 (inherited) | 19.0% | 948 | ~2.6% |

**What changed:** `--repro_thr auto` / `auto95` resolves to the null's 95th percentile and `auto99`
to its 99th, logged and written to `report.json`. The training set is now defined by the data rather
than by a number in a job script.

*Found by reading the rebuild's retention against the shipped build's. Commits `38ce609`, `1c26cc0`.*

---

---

### Defect 23 — the auto threshold crashed the job it was written to improve

**Why does this matter?**

**Answer:** `--repro_thr auto` stays a string until it is resolved against the run's null. The
resolution happened AFTER `scope_sensitivity`, which compares `cos > repro_thr` -- so the comparison
raised `TypeError` and the rebuild died in stage 3, producing no transform, no targets and no log
past the fit.

The local end-to-end check exercised the auto path and passed, because the test fixture defaults
`scope_sensitivity=False`. The one flag combination the cluster job actually used was the only one
nothing exercised. This is the same shape as defect 12 (the deleted evaluation API): a path with no
test, found by the cluster rather than by us.

**What changed:** resolution happens immediately after the fit, before anything can read the value,
and the sensitivity table is computed afterwards so its retention column means the same thing as the
build's. `auto` with too few conditions to build a null now exits with a message rather than a
`KeyError`.

**How can you check it?** `test_auto_threshold_resolves_before_anything_reads_it` sets BOTH flags.
Reverting the reorder makes it fail with the exact `TypeError` from the cluster log, so it is a
regression test rather than a restatement of the fix. It also asserts `auto99` is stricter than
`auto95`, that the resolved value reaches `report.json` as a float, and that the null sits near zero
-- if it does not, it is not a null.

**Verified on the re-run:** `--repro_thr auto -> +0.1086 (the null's 95th percentile)`, and
`report.json` carries `repro_thr 0.10856742188334463`.

*Found by the cluster. Commit `2bfec0a`.*

---

## Why we re-ran the rebuild rather than proceeding to the retrain

Three reasons, in order of weight.

**The comparison would have been confounded.** At 0.2 the corrected checkpoint trains on 948
conditions against `holdout2`'s ~4,100. A difference between them would confound target construction
with a fourfold reduction in training data, and there is no budget for the matched-volume control
that would separate them. At the null's 95th percentile retention is 50.5% against the shipped 62% —
close enough that target construction is the dominant difference.

**The threshold is now purely a training-data choice, so the cost of loosening it is bounded.**
Defect 6 removed the reliability filter from the held-out set, so noise in the retained training
conditions cannot contaminate the measurement. It costs learning efficiency, not validity. Roughly
10% of training targets being unlearnable is a tolerable price for 2.7× the data.

**"Auto" is defensible in one sentence and "0.2" is not.** A condition is kept when its two
independent halves agree more than two unrelated conditions' halves do. That is a validity line
derived from the data. The alternative is a number carried over from a measurement it no longer
describes.

The counter-argument, recorded because it is reasonable: `auto99` gives ~4% FDR at 1,258 conditions,
and someone who weights target purity above sample size should prefer it. We are taking the data.

**Everything else the gate had to decide is already decided:** plate scope, no shrinkage, the
sample-split estimand at zero well crossing, and an evaluator carrying the Step 6 repairs. The
re-run changes one flag.

---

## The estimand decision, recorded before the retrain returns

**Why does this need writing down now?**

**Answer:** Because it is a choice about what the thesis claims, and a choice made after seeing the
result is not the same choice. Both builds exist; one GPU slot does not stretch to two. They cannot
be mixed -- arm [B]'s held-out conditions sit mostly inside arm [A]'s training set, so training on
one and evaluating on the other is leakage.

| | [A] sample split | [B] condition split |
|---|---|---|
| train | 2,523 conditions / 147,710 examples | 1,114 / 66,284 |
| well crossing | **0.0%** (0/268) | 23.0% (142/268) |
| `unseen_combo` | 900 -- an unseen WELL of a seen drug | 251 -- the published (drug, cell line) estimand |
| `unseen_drug` | 711 | 844 |
| threshold | `auto` -> +0.1086 | inherited 0.2 |

**We trained on [A].** Three reasons, in order of weight.

It is the only estimand this atlas cleanly supports. Part II established that a drug-dose enters one
well containing ~49 cell lines, so a drug cannot be present in line A and absent in line B at the
same dose. Cross-context transfer is not measurable here, and arm [B]'s 23% crossing is the measured
size of that.

It carries 2.3x the training data, which keeps the comparison against `holdout2` about target
construction rather than about data volume.

And its two arms still form a proper pair: `unseen_combo` is an unseen well of a *seen* drug, and
`unseen_drug` is a drug never seen at all, which remains the control.

**What it costs, stated plainly.** The thesis's generalisation section stops being a *corrected*
version of the published +0.1002 and becomes a *different* claim. An examiner who asks "you claimed
generalisation, is it real?" receives a reframe rather than a direct answer, and the reframe has to
carry its own justification: the original question cannot be answered in this data.

We judge the reframe to be the stronger position, because reporting a corrected +0.1002 on a split we
have just documented as 23% well-crossed would be reporting a number we know to be confounded. But it
is a judgement, the alternative is defensible, and we would rather have it challenged than assumed.

**One wart, recorded rather than hidden.** Arm [B] was rebuilt at the inherited 0.2 while [A] used
`auto`, because the flag edit touched only [A]. Since only [A] is trained on, this affects nothing
that will be quoted -- but [B] must be rebuilt at `auto` before the two are ever set side by side.

---

## Running state after Part III

| | status |
|---|---|
| Step 1 artifacts | done — 14/17 present on cluster, bundle pending re-fetch after the manifest fix |
| Step 2 unit audit | done — 48.6 lines/well, no replication, dose clean |
| Step 3 channel gate | **done — both channels survive plate matching; Q18 stands with smaller numbers** |
| Step 4 rebuild | **done** — `auto` resolved to +0.1086, threshold now data-derived |
| Step 5 retrain | unblocked |

Open and unchanged: the residual arm is still a per-condition retrieval task; cross-context transfer
is still not cleanly measurable in this atlas (Part II); commit `41c6716` still carries unrelated
DE-weighting work on the training path; and defect 22's half-split variability is identified but not
yet costed into any published interval.

**Commits in this part:** `38ce609` (calibration reporting), `1c26cc0` (`--repro_thr auto`), `2bfec0a` (resolution order + regression test). 63 tests pass; six selftests pass.

**One inconsistency to record.** The re-run applied `auto` to arm [A] only; arm [B], the condition-split comparison, still built at the inherited 0.2. The two estimands are therefore not constructed identically, which is a wart rather than an error since only [A] will be trained on -- but it should be rebuilt at `auto` before either is quoted against the other.

---
---

# Part IV — the retrain, and a control that failed

Step 5, run 2026-08-03. This part contains **two retractions of printed claims** and one result that
replaces them. It also contains the largest single error of the round: the evaluation was run once
against the wrong target definition and had to be discarded.

---

## The retrain itself

One epoch, 9,231 optimizer steps over 147,710 examples, 1h53m on an H200. Loss 2.891 to 1.886. The
cosine schedule annealed the learning rate to 3e-10 by the end, so the run completed rather than
being cut short; the flat tail is the schedule, not a convergence claim.

Generation quality is good: 99.1% valid panel gene symbols, 95% of outputs carry the `[DOWN]`
separator, 0.8% duplicate genes, and no mode collapse -- mean pairwise cosine between predictions for
different drugs is +0.125 against +0.007 between the truths, so the model is less diverse than
reality but not degenerate.

---

### Defect 24 — the evaluation scored against a different quantity than the model learned

**Why does this matter?**

**Answer:** The first eval ran in the PUBLISHED target frame -- cell-line scope, no leave-one-drug-out,
shared controls, threshold 0.2, generic fitted over every condition -- while the checkpoint had been
trained on plate scope, leave-one-drug-out, split controls, threshold 0.1086, generic fitted on
training conditions only. The model emits one quantity and was being scored against another. Any
resulting number measures the mismatch between two definitions, not the model.

The cause is a default we chose deliberately and then failed to trace. `build_residuals` is a
compatibility shim whose defaults reproduce the published build so that old figures stay
regenerable (defect 12). `residual_eval` calls it -- and inherits those defaults. The one consumer
that most needs the NEW semantics silently got the OLD ones.

The shim's own warnings fired in the log and named the problem exactly:

```
[WARNING] reliability measured with SHARED controls...
[WARNING] build_residuals: the generic is being fitted over EVERY condition.
          Any held-out split scored against this truth is transductive.
```

**What changed:** `--truth_from <build>/report.json` reads the build's own report and reproduces its
frame -- scope, leave-one-drug-out, split controls, threshold -- and passes the holdout so the
generic is fitted on training conditions only. Pinning to the report rather than repeating flags
means the two cannot drift. Without the flag the run prints a four-line warning stating that the
published frame is in use and that this is correct only when reproducing a published number.

**Cost:** the job was killed ~3.5 h in and re-run. The checkpoint was unaffected, so no training was
lost.

*Found by reading the log rather than the result. Commit in this part.*

---

## The eval, correctly framed

`truth frame taken from ...: scope=plate loo=True split_controls=True repro_thr=0.1086 generic
fitted on TRAIN ONLY`. 570 conditions, 41 cell lines, 3,424 truth conditions retained (52%).

```
ceiling        0.958      the achievable bar
model          0.569
drug_lookup    0.941      training-only, the fair bar
drug_lookup_1  0.859      one other cell line, no averaging
control_copy   0.511      drug-agnostic by construction -- must be 0.5
generic        0.500      drug-agnostic by construction -- must be 0.5
random         0.521
```

Chance is 0.5 and the ceiling 0.958, so the achievable range is 0.458. **The model covers 15% of it.
A lookup table covers 96%.** The two drug-agnostic baselines land exactly where they must, which is
the instrument validating itself.

`model - drug_lookup = -0.3559`, two-way CI [-0.3988, -0.3130]. The oracle variant -- fitted over the
full cache, which on a held-out condition reads residuals the model was never shown -- sits a further
0.06 above the fair lookup. That difference is how much the published comparison was inflated by
letting the baseline see held-out data, and it had never been measured.

---

### Defect 25 — the scramble comparator was not a null

**Why does this matter?**

**Answer:** The evaluation measures drug use as `model - scramble`: name the true drug, name a
different one, see whether the output degrades. The test rests on an assumption nobody had checked --
that naming the WRONG drug is uninformative.

It is not. The comparator is selected as the most ANTI-correlated drug in the cell line, and the
partner pool is mostly drugs the model was TRAINED on. The scramble arm therefore does not emit
noise; it emits a learned signature chosen to point away from the truth.

The three strata make it visible, because they differ only in how correlated the partner's true
response is with the target's:

```
scramble_near      partner cos +0.26  ->  NIR 0.572   ABOVE chance -- a partly CORRECT answer
scramble_orth      partner cos  0.00  ->  NIR 0.497   AT chance    -- a neutral null
scramble_opposite  partner cos -0.24  ->  NIR 0.403   BELOW chance -- an actively WRONG answer
```

So `gap = (how much naming the TRUTH helps) + (how much naming a LIE hurts)`, and the second term is
large precisely because the model knows the partner. It exists even when the model knows nothing
about the target.

`residual_eval` hard-codes `scramble_opposite` for its split table -- the stratum that maximises the
contaminating term.

**The control proves it rather than arguing it.** `unseen_drug` conditions are drugs the model has
never seen, so the gap there must be zero:

| | under `opposite` | under `orth` |
|---|---|---|
| `unseen_drug` | **+0.0862** [+0.0017, +0.1708] -- FAILS | **+0.0097** [-0.0655, +0.0848] -- passes |

**This was visible in the repository already and was misdiagnosed.** `FINDINGS.md` Q20 records the
`unseen_drug` gap moving from +0.0195 [-0.045, +0.086] to +0.1114 [+0.051, +0.173] between two runs
differing only by sampling stream, and attributes it to seed instability. Unseeded generation was
real and is fixed, but the deeper reason the control is unstable is that under `opposite` it measures
an artefact whose size depends on which partner is drawn. The symptom was logged; the cause was not
found.

**What changed:** `scramble_stratum_audit.py` recomputes the split table for every stratum from the
saved per-condition records -- no model, no GPU, no re-generation, because all three strata were
generated in the same run. `opposite` is not discarded: it remains a good test of whether the model
reads the drug NAME at all, where a sharper comparator is a feature. It simply cannot carry a claim
about how much naming the TRUE drug helps.

---

## The corrected result

Against the neutral comparator, with two-way cluster-robust intervals over cell lines and wells:

```
split          stratum      n   model   scram       gap              two-way CI
train          orth       200   0.635   0.489   +0.1457  [+0.0836, +0.2078]  EXCLUDES 0
unseen_combo   orth       250   0.545   0.500   +0.0449  [-0.0074, +0.0972]  spans 0
unseen_drug    orth       120   0.512   0.502   +0.0097  [-0.0655, +0.0848]  control passes
```

`unseen_combo` spans zero under line-clustering as well ([-0.0050, +0.0993]), so this is the
comparator and not the crossed-interval correction.

**Memorisation premium:** train +0.1457 against unseen_combo +0.0449, difference **+0.1008,
clustered z = 2.43, p = 0.0150 (this register long printed the superseded unclustered permutation p = 0.0054; see Part VI)**.

### Two printed claims do not survive

**"There is no memorisation premium"** (`Results-and-Analysis.tex:544`, `Conclusions.tex:60`) is
false. There is one, and it is significant. The original was inferred from two overlapping intervals,
which was never a test of a difference.

**"Cross-context transfer: YES"** is not established. +0.0449 with an interval spanning zero.

Stated precisely: this is **not established**, not **shown to be zero**. The interval reaches +0.097,
so an effect up to about +0.1 remains compatible with the data at n=250. It is a limit of power, and
the thesis must say so rather than claim a null.

---

## Two comparator-free tests, and what they settle

Both came from asking what the sweep across partner drugs actually measures. Neither needs a null
arm, so the fault above cannot reach them.

**Does the output track the drug it was TOLD?** For each condition there are three generations, one
per named partner. Regress the resulting NIR on `cos(truth_target, truth_partner)`. Zero slope means
the output ignores the name; positive slope means the model emits the named drug's signature.

```
train         600 points   slope +0.390  [+0.278, +0.503]
unseen_combo  750 points   slope +0.374  [+0.250, +0.480]
unseen_drug   360 points   slope +0.238  [+0.081, +0.379]
```

**The model holds a drug to signature map**, and it does not degrade on held-out wells: +0.374
against +0.390.

**Told the TRUE drug, does it beat chance?** NIR against 0.500, no comparator at all.

```
train         n=200  NIR 0.635  [0.5773, 0.6924]  ABOVE chance
unseen_combo  n=250  NIR 0.545  [0.4951, 0.5943]  not distinguishable
unseen_drug   n=120  NIR 0.512  [0.4456, 0.5778]  not distinguishable
```

Two instruments with DIFFERENT FAILURE MODES agree that transfer to held-out wells is not
established. They are not statistically independent -- they share the same generations and the
same truths -- but the neutral-comparator gap and the direct NIR-against-chance test break in
different ways, so agreement between them is worth more than either alone.

### The reading these support

A positive slope with a chance-level NIR is not a contradiction. The map exists but is **coarse**:
the model knows roughly what a drug does, and not what it does *in this well*. NIR asks the second
question, ranking against ~40 other drugs measured in the same cell line.

That is the drug **main effect** without the drug x cell-line **interaction** -- which Q17 measures at
~45% of the residual variance, and which is exactly what a per-cell-line discrimination test
requires. It also explains the lookup's dominance: `drug_lookup` *is* the main effect, estimated
directly from data far more precisely than the model learned it.

So the sentence the evidence supports is:

> Re-encoding the target made the model read the drug and learn a drug-level average response. It
> learned the main effect and not the interaction. A lookup table computes that same main effect
> better, which is why retrieval beats it.

That is narrower than "the repaired drug use generalises", better supported, and it sits alongside
the drug-blindness spine rather than against it.

---

## Three further checks, and the one that changes the wording

### The output barely resembles anything

`NIR` is a RANK statistic, so it can move on tiny differences. The absolute similarity between what
the model emits and any real response:

```
split            n   cos(pred,own)  cos(pred,others)   difference
train          200        +0.0511           -0.0073     +0.0584  [+0.0334, +0.0834]  excl 0
unseen_combo   250        +0.0085           -0.0053     +0.0138  [-0.0007, +0.0283]  spans 0
unseen_drug    120        -0.0037           -0.0066     +0.0029  [-0.0181, +0.0239]  spans 0
```

The largest absolute cosine anywhere is **0.051** -- essentially orthogonal, even where the model
trained. So the drug effect the slope detects is a faint **tilt** that reorders a ranking, not a
signature the model reproduces.

**This constrains the wording of the whole part.** "The model learned the drug main effect" is too
generous: `drug_lookup` IS the main effect and reaches NIR 0.941 where the model reaches 0.569, so
the model holds something closer to a sixth of it. The defensible phrase is a *faint drug-dependent
tilt*, and the slope is what makes even that measurable.

### Told a drug it does not know, the model hedges rather than confabulating

A chance-level NIR is ambiguous on its own: a confidently WRONG output -- a plausible signature for
the wrong drug -- also scores about 0.5, because it is equally unrelated to every truth. The
absolute cosines separate the two cases, and on `unseen_drug` they are -0.004 and -0.007. The model
commits to nothing.

That is a reportable property in its own right. On a scientific prediction task, a generative model
that declines to invent a plausible-looking answer for an unknown input is behaving well, and it is
worth saying so alongside the negative results.

### The splits are equally hard, so the comparison is not confounded

```
train          ceiling 0.962   mean condition reproducibility +0.216
unseen_combo   ceiling 0.955   mean condition reproducibility +0.204
unseen_drug    ceiling 0.956   mean condition reproducibility +0.254
```

`ceiling` is one half of a condition's cells scored against the other half's truth -- a WITHIN-WELL
split-half bar, not a biological replicate (defect 18: 286 of 287 treatments are single-well). It
measures how discriminable the truths are independently of any model, which is what is needed
here. The true achievable bar is lower and unmeasured, so the model's 15% coverage is an upper
estimate of its share and the lookup's 96% could exceed a real replicate -- both of which
strengthen rather than weaken the retrieval argument. The ceilings differ by 0.007, and the
held-out targets are if anything CLEANER than the trained ones. The model's weaker showing on
held-out data is therefore a property of the model, not of the targets. That confound is closed.

### What the null claims would have missed

`unseen_drug` at n=120: an effect below about **+0.094** above chance would not have been detected at
80% power. So "the model cannot generalise to unseen drugs" is not a finding. The finding is **not
established, with that detection limit**, and the thesis must carry the limit wherever it carries the
claim.

### The corrected summary sentence

> Re-encoding the target made the model read the drug name and produce a faint drug-dependent tilt --
> enough to shift a ranking, not enough to resemble any real response (cosine about 0.05 even where
> it trained). That tilt does not sharpen into identification on wells it never saw. Told a drug it
> has never seen, it hedges rather than confabulating. A lookup table, which estimates the same
> drug-level effect directly from the data, is roughly six times more discriminative.

Every clause there is measured, and the claim is narrower than anything the chapter previously made.

---

## Running state after Part IV

| | status |
|---|---|
| Steps 1-4 | done (Parts II, III) |
| Step 5 retrain | **done** — checkpoint sound, 1h53m |
| Step 5 eval | **done on the second attempt**; first discarded, wrong target frame |
| scope 7 `vardecomp`, scope 8 `calib` | written, not yet run |
| scope 9 | written, runs locally once artifacts return |

**Still open at the time of writing** (superseded -- see Part IX for the current list). `residual_eval` hard-coded `scramble_opposite` in its own split table; the
audit script corrects it after the fact but the eval should report every stratum. Defect 22's
half-split variability is identified and not yet costed into any interval. Commit `41c6716` still
carries unrelated DE-weighting work on the training path, verified inert at default settings but
exercised by no test. And whether a second training epoch would help is unresolved -- the validation
curve was not inspected before the job was killed.

**Commits in this part:** `c841079` (stratum audit), `14bb010` (slope and chance tests), plus the
`--truth_from` fix to `residual_eval`.

---
---

# Part V — the transfer coefficient and the metric audit

Scope steps 7 and 8, run 2026-08-03. Both were written to repair instruments rather than to produce
new claims, and both did that. This part contains **no retraction**: the two claims under repair came
back confirmed, one of them for the first time with a valid dose axis.

It also contains three defects found by reading the logs rather than the results — including one
that could have printed another arm's interval on the dose row without a word.

---

## Scope 8 — the metric audit, hardened

**Why did we run this?**

**Answer:** "NIR is the only calibrated metric" is the thesis's strongest contribution and was the
cheapest thing to make bulletproof. It needed four repairs: average ranks for tied expression
(index-order ties leaked panel position into every rank metric), tie-aware NIR (exact ties counted as
losses, scoring a degenerate predictor 0.000 instead of 0.500 and disagreeing with `residual_eval`'s
implementation of the same metric), the DRF ratio recomputed inside every resample rather than
against a fixed estimated denominator, and Holm across the five metrics so the claim is simultaneous
rather than five separate ones.

**What came back:**

```
DRF, Holm-adjusted across 5 metrics -- calibrated: nir
  weighted_r2    -0.163 [-0.177, -0.148]   holm=1.0000   not calibrated
  spearman_expr  -0.650 [-0.660, -0.641]   holm=1.0000   not calibrated
  de_delta       -0.686 [-0.706, -0.666]   holm=1.0000   not calibrated
  panel_tau      -0.319 [-0.324, -0.316]   holm=1.0000   not calibrated
  nir            +0.635 [+0.618, +0.652]   holm=0.0025   CALIBRATED
```

**The claim survives everything.** NIR is the only metric whose discrimination-recovery fraction
clears zero, and it does so under family-wise correction with an interval. NIR's DRF also barely
moved from the previously reported +0.641 to +0.635, so the hardening did not disturb the number it
was hardening.

**One thing to fix in the code, not the result.** The script's docstring predicts that
`weighted_r2` and `spearman_expr` will come back POSITIVE ("they reward the noise ceiling"). Both are
strongly negative. The headline is unaffected -- the published claim was always that NIR alone is
calibrated -- but a docstring contradicting its own output is a trap for the next reader and should
be corrected.

---

## Scope 7 — the transfer coefficient, with a dose axis that exists

**Why did we run this?**

**Answer:** the dose arm was void. It compared WELL identifiers, and since one well carries every
cell line at one concentration, "different well" was never "different dose". The scope figures were
also norm ratios squared and compared to 1 as an orthogonality CHECK, presented as a decomposition.

**Dose resolution: 298/298 samples yield a molar concentration, 0 combinations excluded.** The arm is
real for the first time.

### The energy decomposition is now exact, and it vindicates the old number

```
residual 0.621 + generic 0.390 + cross -0.011 = 1.000
the cross term is negligible (-0.011), so the two shares read as variance components
```

Because `shift = residual + generic` exactly, the three shares sum to one by construction. The cross
term is measured at **-0.011** rather than assumed away, so the residual and generic shares really are
variance components -- which the shipped norm-ratio version had asserted without being able to show.
**62.1% of the perturbation response energy is drug-specific.** The old figure of ~62% was right; it is
now right for a stated reason.

### The transfer coefficient

```
T [repro-filtered, the training set]   0.553   multiway CI [0.514, 0.593]   1335 nodes
T [all conditions, no filter]          0.597   multiway CI [0.563, 0.632]   4309 nodes
```

T is the share of the drug-specific residual that transfers across cell lines, so **1 - T = 44.7%
[40.7%, 48.6%] is drug x cell-line interaction.** The previously quoted "~45%" is confirmed, now with
a dependence-aware interval and a dose arm that measures dose.

That number is the quantitative backbone of Part IV's reading: a model that learned the drug main
effect and none of the interaction is leaving ~44% of the drug-specific signal untouched, and the
interaction is precisely the part a per-cell-line discrimination test asks about.

### Both controls pass

```
diff_drug_same_plate   T = -0.017   drug-clustered [-0.030, -0.004]   multiway [-0.050, +0.016]
diff_drug_cross_line   T = -0.008   drug-clustered [-0.028, +0.014]   multiway [-0.036, +0.020]
```

The same-plate control read **+0.478 at cell-line scope**. At plate scope it reads -0.017 and its
interval spans zero. The plate contamination that motivated the whole scope change is gone, measured
rather than argued. The structure-matched control -- different drug, different cell line, the arm the
verdict is explicitly gated on -- reads -0.008 and spans zero.

### The dose ordering is right, and it used to be inverted

```
DOSE (same drug, same line, DIFFERENT MOLAR dose)   T = 0.707   multiway [0.632, 0.782]
cross-line T                                        T = 0.553   multiway [0.514, 0.593]
```

Changing the dose costs **less** than changing the cell line, which is the biologically correct
ordering. Under the old cell-line-scoped measurement the ordering INVERTED (0.484 against 0.517), and
that inversion was one of three signals that the frame was contaminated. With plate scope and a real
molar axis it is the right way round. The arm has repaired itself.

---

### Defect 26 — the dyadic estimator omitted the drug, and the omission announced itself

**Why does this matter?**

**Answer:** the first scope-7 run reported a dyadic interval **narrower** than one-way drug
clustering (x0.55 and x0.48). That is impossible: a dependence model capturing a superset of another
one cannot produce a narrower interval. The impossibility is how the error was found.

The nodes were the two CONDITIONS. But T is a SAME-DRUG statistic -- every pair is (drug d in line
c1, drug d in line c2) -- so two pairs of one drug across four different cell lines share `beta(drug)`
while sharing no condition. The dominant dependence channel was simply absent from the node set.

**What changed:** `multiway_cluster_ci` generalises Fafchamps-Gubert to a SET of nodes per
observation, two observations being dependent if their sets intersect. The T arm passes
`{drug_1, drug_2, condition_1, condition_2}`; for a same-drug pair the two drug entries collapse to
one, so it is the general form rather than a special case. The reporting now warns whenever a
multiway interval comes out narrower than one-way, which is the signature of a missing channel.

**Confirmed by the re-run:** the ratios moved from x0.55 and x0.48 to **x1.01 and x0.96** -- the
multiway estimator now agrees with the coarse grouping that captured the dominant channel, instead of
undercutting it.

**Two selftest lessons worth keeping.** "More nodes always widen" is FALSE: the estimator sums signed
cross terms, so an arbitrary added node can narrow it. The property that holds -- and is now tested --
is that including a channel carrying REAL shared variance widens. And a node shared by EVERY
observation is the G=1 case where a cluster-robust variance does not exist: the single cluster's
deviations sum to zero, V collapses to a floating-point residue, and the interval comes out absurdly
narrow rather than refusing. That is now caught on the graph structure rather than on the arithmetic,
because whether the residue lands at exactly zero depends on accumulation order.

---

### Defect 27 — the dose arm was reading another arm's variable

**Why does this matter?**

**Answer:** the dose block computed its point estimate and one-way interval from `dose_rows`, but read
its dyadic interval from `dy` -- a variable left behind by the preceding negative-control loop. In the
run where this was caught, `dy` happened to be `None`, so nothing wrong was printed. **The identical
code would have reported another arm's interval on the dose row without a word.**

It surfaced only because an expected log line was ABSENT, which is the argument for grepping the log
for what should be there rather than reading the result for what is.

**What changed:** the block computes `dy_dose = _dyadic(dose_rows)`.

---

### Defect 28 — the structure-matched control had no multiway interval

**Why does this matter?**

**Answer:** `cross_line_diff_drug_pairs` builds its rows in a separate function that was never given
node fields, so `_dyadic` returned `None` for it. The one arm whose own log line reads "the verdict is
gated on this" was silently falling back to drug clustering alone.

Also, different-drug pairs depend through BOTH drugs, and the node set carried only the first.

**What changed:** the control's rows carry `drug`, `drug_b`, `node_a` and `node_b`, and the node set
is `{drug_1, drug_2, condition_1, condition_2}` throughout. The control now reports
**-0.008 [-0.036, +0.020]**.

---

---

### Defect 29 — the by-split table measured against a comparator that is not a null

**Why does this matter?**

**Answer:** this is Part IV's central finding applied to the place it does the most damage. The
three-way split table -- the block the cross-context transfer verdict is read off -- hard-coded
`scramble_opposite` as its comparator. That stratum does not sit at chance: it TRACKS the partner
drug, measured at NIR 0.403 when the partner's true residual is at cosine -0.24, against 0.497 at
cosine 0.00.

So every gap in that table was "model minus a comparator that was actively pushed the wrong way",
and the difference is not small: under the neutral comparator `unseen_combo` moves from a verdict to
**+0.0449 [-0.0074, +0.0972], spanning zero.** The audit script corrected the table after the fact,
which is the wrong place -- the number an examiner reads comes out of `residual_eval`.

**What changed:** `--split_comparator` defaults to the neutral stratum, every row additionally
reports the gap against the old comparator for continuity, and the fallback path warns in the log
when the neutral arm is unpopulated rather than substituting silently. A test pins the default and
asserts the hard-coded call site is gone.

**Deliberately NOT changed:** the field-decomposition block still uses `scramble_opposite`, and
correctly. There the arms (name-only, mechanism-only, both) are compared against EACH OTHER with the
partner held fixed, so the partner's push is common to all three and cancels. Only the by-split
table used it as a null.

---

### Defect 30 — a broken commit, caught by the check that follows every edit

**Why does this matter?**

**Answer:** the fix for the calibration docstring spliced a multi-line string into a single-quoted
`logger.info(...)` call, leaving `calibration_eval.py` unparseable, and it was COMMITTED that way.
The patch script printed its own success message; the syntax error surfaced one line later when the
selftest ran and could not import the module.

It is recorded because it is the cheapest possible lesson: a patch script reporting success says
only that the text substitution matched, never that the result is valid Python. The selftest that
runs immediately after every edit is what makes that harmless, and it is the reason the window
between the broken commit and its repair was one commit wide.

**What changed:** repaired in `f912292` as four separate log lines; `ast.parse` now precedes the
selftest on every scripted edit.

---

### Not a defect in our code, but it will bite the next person: a zeroed `click`

`python -m pytest tests/` fails at COLLECTION with `ValueError: source code string cannot contain
null bytes`. Nothing in this repository is at fault and no test is failing.

`site-packages/click/__init__.py` is 4634 bytes of which 4634 are NUL -- the file is entirely zeroed,
the signature of an interrupted write in a OneDrive-synced `site-packages`. pytest autoloads the
`dash` plugin at startup, `dash` imports `flask`, `flask` imports `click`, and collection dies before
a single test runs.

Run the suite with plugin autoload disabled and it is untouched:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
64 passed
```

Left unrepaired on purpose: reinstalling `click` rewrites a global interpreter that other projects
share, which is not a change to make silently in the middle of a thesis run.

## Running state after Part V

| | status |
|---|---|
| Steps 1-5 | done (Parts II-IV) |
| scope 7 transfer coefficient | **done** -- T = 0.553 [0.514, 0.593], dose arm real, both controls pass |
| scope 8 metric calibration | **done** -- NIR the only calibrated metric under Holm |
| scope 9 figures and generated assets | written; runs locally once artifacts return |

**Every number destined for the thesis now comes from a corrected instrument.**

Closed since Part V opened: the by-split comparator (defect 29) and the calibration docstring.

Still open: defect 22's half-split variability is identified and not costed;
commit `41c6716` carries unrelated DE-weighting code on the training path, verified inert at defaults
but exercised by no test; and one training epoch with one generation seed.

**The one remaining spend that could change a printed verdict** is scoring the full 900 `unseen_combo`
conditions rather than the 250-condition quota. The truths already exist; only generation costs. It
tightens the interval by about 1.9x, moving the detection limit from +0.097 to roughly +0.05 -- which
is the difference between "not established" and a decision.

**Commits in this part:** `e17336d`, `4196fae`, `4ac83d3`, `55fb813`, `f912292`.

---
---

# Part VI — the fourth audit, and the one defect that reaches the printed numbers

GPT 5.6 SOL, reviewing commit `55fb813`, 3 August 2026. Thirty-four distinct claims were checked
against the source by an independent pass and then adversarially re-checked by a second pass
instructed to refute the first. Twenty-seven verified TRUE, five PARTLY TRUE, two of the reviewer's
sub-assertions were refuted, and several of the reviewer's own statements are stale because it read a
commit predating Part V.

This part records the outcome. It contains **one blocker that reaches numbers already written down**,
and **one statistical finding that forces a second retraction to be softened.**

---

## The blocker — the evaluation's truth is not the truth the model was trained on

**Why does this matter?**

**Answer:** Part IV's evaluation was run with `--truth_from` and `--holdout`, and those flags do what
they claim: scope, leave-one-drug-out, split controls and the reliability threshold are all pinned to
the build's own `report.json`. Defect 24 is genuinely closed. But the compatibility shim underneath
them reconstructs the training set from the WRONG FILE, and the file it reads has already been
filtered.

The chain, all four links verified in source:

```
builder     :893   train_keys = [k for k in conds if split[k] == "train"]      <- FULL pre-filter
builder     :932   kept, relstats = transform(..., eval_filter=args.eval_repro_filter)
builder     :1032  json.dump({"split": {... for k, v in split.items() if k in kept}, ...})
                                                             ^^^^^^^^^^^ the manifest is FILTERED
shim        :704   want = {k for k, v in hm.get("split", {}).items() if v == "train"}
shim        :715   split = {k: "train" for k in conds}                         <- every label erased
shim        :716   kept, _ = transform(..., eval_filter=True)                  <- HARDCODED
```

So two distinct things go wrong at evaluation time:

**(i) The generic is refitted on the survivors.** On the repository's own test fixture the builder
fits on 44 training conditions and the shim reconstructs 21 -- a strict subset. The builder's generic
reproduces `report.json`'s `fit_digest` exactly (`7d76a716f1be7d33`); the shim's digests
`83b2bdf5eb9fd66e`. Three of eight plate-scoped generic groups become entirely UNDEFINED for the
shim, which makes `transform` skip those conditions silently, and the five surviving groups drift a
mean 21% and a maximum 35% in relative L2. `fit_digest` is written but no consumer ever checks it.

**(ii) Held-out truth is selected on its own outcome.** `eval_filter=True` is hardcoded, so a held-out
condition is kept only if ITS OWN split-half cosine clears the threshold. `report.json` records
`eval_repro_filter`, and `--truth_from` reads that file and propagates four fields -- but not that one.

The second is the one that contradicts something already written down. Defect 6 states "the primary
result now uses the complete metadata-eligible holdout." **That is true of the builder and false of
the evaluation.** Defect 6 has to be split into a build-side half (closed) and an eval-side half
(open). There is a test, `test_the_held_out_set_is_not_selected_on_its_own_outcome`, and it does not
catch this: it exercises `run()`, not the shim. The only two tests that touch the shim pass
`repro_thr=-1.0`, which makes the filter inert, so the filtering branch has never once executed
under test.

**Which direction does it push?** Filtering held-out truth by reproducibility keeps the CLEANEST
conditions -- the measured ceilings, 0.962/0.955/0.956, are the evidence. Those are easier to predict.
So the headline finding, that `unseen_combo` does not clear zero, is measured on the FAVOURABLE
subset and is therefore CONSERVATIVE: fixing this cannot turn a null into a positive. What it does
change is the population the number describes and the exact value of every absolute NIR, because the
generic that defines the residual is not the one the checkpoint learned.

**What has to happen.** The builder must persist the complete pre-filter split (nothing in the repo
does -- `split_all`, `split_full`, `prefilter` return zero hits outside the reviewer's own report) and
the exact train-key list; the shim must accept `split` and `eval_filter` instead of hardcoding both;
`--truth_from` must propagate `eval_repro_filter`; and a test must assert that a replay reproduces the
build's `fit_digest`. That last assertion is the real fix -- the digest already exists and would have
caught this on the day it was introduced.

Cost: the code is an afternoon. The target rebuild is CPU. Whether generation must be re-run depends
on whether the corrected frame admits conditions that have no saved records -- it will, because
conditions currently dropped for an undefined generic come back.

---

### Defect 31 — the memorisation premium is carried by drugs that appear in only one arm

**Why does this matter?**

**Answer:** Part IV retracted "there is no memorisation premium" and replaced it with a premium of
+0.1008, clustered z = 2.43, p = 0.0150. The reviewer asked whether the two arms are comparable. They
are not.

```
train         200 observations   68 drugs
unseen_combo  250 observations   29 drugs
shared                           20 drugs
```

Only 22.5% of train observations and 64.4% of held-out observations sit on a drug present in both
arms. Restricting to those 20 shared drugs:

```
shipped            d = +0.1008   SE 0.0414   z 2.43   p = 0.0150   [+0.0196, +0.1820]
20 shared drugs    d = +0.0420   SE 0.0590   z 0.71   p = 0.4764   [-0.0736, +0.1576]
```

The components show where it goes: train-shared +0.0640 against train-only +0.1694; combo-shared
+0.0220 against combo-only +0.0864. **The premium is carried by the drugs that appear in one arm
only.** That is a difference in drug composition, not necessarily a difference in training exposure.

The retraction of "there is no memorisation premium" STANDS -- overlapping intervals were never a
test. But the replacement claim must be weakened from "a memorisation premium exists" to a
condition-weighted training-exposure advantage whose generality across drugs is unresolved. The word
"memorisation" is causal and the design does not license it while composition is confounded.

**One thing the reviewer got wrong here.** It argued that adding the two arms' separately estimated
variances is invalid because the arms share cell lines. Adding variances omits `-2Cov`; with
positively correlated arms that makes the interval TOO WIDE, i.e. conservative. The joint CGM
calculation confirms it. The covariance point is worth fixing for correctness, but it does not
inflate the significance -- the drug-composition point does all the damage.

---

### Defect 32 — `max_abs_cos` is a maximum of two means

**Why does this matter?**

**Answer:** the printed sentence is "the largest absolute cosine between any prediction and any real
response is 0.051", and it is used to argue the model commits to almost nothing.

`scramble_stratum_audit.py:256`:

```python
"max_abs_cos": float(max(abs(own.mean()), abs(oth.mean())))
```

That is a maximum over TWO SCALARS, each already a split-level mean. The artifact confirms it:
`cos_own` and `max_abs_cos` are byte-identical at 0.051107174886856226. Individual records reach
**+0.4395** (train) and **+0.3194** (unseen drug).

Correct statement: mean own-truth cosine was +0.051 on trained conditions and +0.009 on held-out
wells, **with substantial condition-level heterogeneity.** Rename the field or delete it.

The "faint tilt" wording built on this needs re-deriving, and the reviewer is right that it is partly
over-corrected in the other direction: the slope is NOT faint on the rank scale (+0.390 across the
partner range is a large NIR movement). What is weak is mean cosine fidelity, held-out
identification, and performance against retrieval. Those are three separate weaknesses and the thesis
should say which one it means each time.

---

### Defect 33 — "hedges rather than confabulates" is not a measured quantity

**Why does this matter?**

**Answer:** the sentence claims something about the model's behaviour under ignorance. Nothing in the
pipeline measures abstention, confidence, calibration, or consistency between the four sampled
generations per condition. A near-zero MEAN cosine is equally consistent with confident positive and
negative alignments cancelling, with inconsistent samples cancelling, with a coherent signature
pointing somewhere outside the measured truth library, and with confident nonsense.

Withdraw it, or measure it -- per-generation consistency is computable from saved records with
`k_samples = 4` and would make the claim real for CPU cost.

---

### Defect 34 — the coverage headline compares two different populations

**Why does this matter?**

**Answer:** the model mean is over 570 conditions and the drug-lookup mean over 449, because each arm
is averaged over its own non-null support (`residual_eval.py:710-716`, no intersection). On the
common 449 conditions:

```
                 full support (as printed)      common support (correct)
model                 15.2%                          18.6%
drug lookup           96.4%                          96.3%
```

The gap survives comfortably -- this changes nothing scientifically -- but "roughly six times more
discriminative" is a ratio of two numbers computed on different sets. **The paired contrast is
already correct and should be the headline instead:** `model - lookup = -0.3559 [-0.3988, -0.3130]`,
n = 449, which is a within-condition comparison and needs no support caveat.

---

### Defect 35 — the channel gate was never brought into the repaired frame

**Why does this matter?**

**Answer:** `channel_gate.py:242` calls the compatibility shim with its published defaults -- cell-line
scope, no leave-one-drug-out, shared controls. That is defect 24's exact failure mode surviving in a
second consumer, and the gate is load-bearing: Q18 and the Conclusions rest on it.

The same file computes its intervals with one-way cell-line clustering at `:346-357`, while a
two-way helper added in `e17336d` sits at `:86-95` and is never called. Defect 19 mandated two-way
clustering wherever conditions are crossed by well and cell line; the mandate was never applied to
the gate.

The gate also licenses less than it is used for: it shows that response information is recoverable by
metadata-neighbour retrieval, not that the checkpoint uses those channels. Those are different
statements and only the first is measured.

---

### Defect 36 — the reference distribution ignores the cluster count

**Why does this matter?**

**Answer:** every interval in the stack uses a normal critical value. `shared/inference.py:62` defines
`Z95 = 1.959963984540054` and it is the only critical value in the module; 1.96 is additionally
hardcoded at `residual_eval.py:669`, `scramble_stratum_audit.py:151` and
`evaluate_endcell.py:237-238`. There is no t quantile anywhere in the repository.

The actual cluster counts:

```
train          n=200   35 cell lines   113 wells
unseen_combo   n=250   39 cell lines    31 wells
unseen_drug    n=120   38 cell lines    25 wells
```

With G as low as 25, t(24) = 2.064 against 1.96 -- intervals are about 5% too narrow. Sweeping every
analytic interval in the register with `df = min(Ga, Gb) - 1`:

**Exactly one published verdict flips.** `unseen_drug / opposite`, +0.08623, normal lower bound
positive, t(24) lower bound **-0.00282**. Everything else is monotonically safe: the headline
`unseen_combo / orth` already spans zero and only widens, and `train / orth` survives at t-lower
+0.08127.

That single flip is on the CONTROL arm against the comparator we have already retired as non-neutral,
so it costs nothing scientifically -- but it is a verdict currently printed as positive, and it should
be corrected rather than discovered by an examiner. The fix is a t quantile with `df = G - 1`; it is
CPU-seconds from the saved records.

---

### Defect 37 — the run that carries the argument had the field decomposition switched off

**Why does this matter?**

**Answer:** the claim "the model reads the drug name" is not identified by the experiment that was
run. `re_repaired.json:13` reads `"field_decomp": false`, and enumerating all 570 records confirms
`scramble_drugonly` = 0/570 and `scramble_moaonly` = 0/570. The scramble swaps the drug NAME and the
MECHANISM STRING together, so every result is about the combined block.

The arms are implemented and working (`residual_eval.py:129-160`); they simply were not exercised.
No artifact anywhere in the checkout has field-decomposition means.

Correct wording throughout: **the model responds to the combined drug-and-mechanism prompt.** Running
the decomposition needs fresh generation -- it is a GPU cost, not a CPU one -- and it is the single
experiment that would convert the headline from a description into a mechanism.

---

### Defect 38 — `crossed_bootstrap` is a subsampler

**Why does this matter?**

**Answer:** the slope intervals -- the comparator-free evidence, the strongest thing in Part IV --
come from `shared/inference.py:202`, which resamples clusters with replacement and then converts the
draw to a SET, discarding multiplicities. That is random cluster inclusion, not a cluster bootstrap.

The slope point estimates are unaffected. The intervals are the wrong width. The reviewer's proposed
replacement is also the better design: stack the three strata per condition, fit
`NIR ~ partner_cosine * split + condition fixed effects`, cluster-robust by line and well, and test
the train-minus-held-out slope interaction DIRECTLY rather than inferring it from overlapping
intervals. CPU-only, from saved records.

---

### Two of the reviewer's claims that do NOT survive

**The IID fallback is latent, not active.** `multiway_cluster_ci` does fall back to an
independent-sampling variance when the estimator goes non-positive, and IID is indeed the wrong
fallback for dependent pairs -- the two-way path at `:192-194` correctly falls back to
`max(va, vb)` instead. But instrumenting the estimator shows the branch has never fired: 40/40
selftests with `{'calls': 5, 'fallback': 0, 'undefined': 1}`, and zero occurrences across all 89
artifacts. Worth fixing to match the two-way path; not worth re-running anything for.

**The covariance omission is conservative, not anti-conservative.** Covered under defect 31.

---

### What the reviewer says that is simply stale

It read `55fb813` and states that the corrected scope-7 and scope-8 jobs have not run, that no
corrected artifact exists, and that the ~45% interaction figure should not be used. **Part V reports
both jobs as run**: T = 0.553 multiway [0.514, 0.593], interaction 44.7% [40.7%, 48.6%], both
negative controls spanning zero, the dose arm real and correctly ordered, and NIR the only
Holm-calibrated metric. Its objection to the 45% number is answered.

It also flags `residual_eval` hardcoding `scramble_opposite` in the by-split table. That was defect 29,
fixed in `f912292` -- one commit after the tree it read.

One caveat survives inside the stale objection and should be honoured: the corrected scope-8 run did
not regenerate every configuration the thesis prints, and the local `calibration.json` is the July 29
artifact with no CI or Holm keys. The corrected numbers exist in the log, not yet in a file.

---

## Where four audits now agree

Three findings have arrived independently from three different reviewers and should stop being
reopened:

1. **The estimand.** One drug-dose per well of ~49 cell lines makes fixed-dose cross-context transfer
   structurally unidentifiable; a `(drug, cell_line)` holdout is within-atlas matrix completion. This
   belongs in Methods and again in Limitations, not in an appendix.
2. **`repro_cos` is within-well sampling precision**, not biological reproducibility. The new audit
   sharpens it by catching this register violating its own rule: defect 18 forbids the phrase, and
   defect 21 then writes "About 51% reproduce at all ... belongs in Results." **Defect 18 wins;
   defect 21's wording must change.**
3. **Comparator neutrality.** Found here, confirmed twice.

Where the new audit does not converge is where it is most valuable: the truth-replay chain, the
premium's drug composition, both channel-gate defects, the reference distribution, and
`field_decomp` being off are all first sightings, and four of the five are major or blocker.

---

## Also stale: this document

The register still prints the memorisation premium's evidence as "permutation p = 0.0054" at the Part
IV summary, although `e17336d` made the clustered test primary at z = 2.43, p = 0.0150 -- before the
reviewer even read the tree. The reviewer is right about the DOCUMENT and wrong about the CODE. Fixed
in this part.

---

---

### 43 — the transfer coefficient was transcribed wrong, by me, from a pasted log

**Why does this matter?**

**Answer:** Part V records `T = 0.558` and an interaction share of `44.2%`. The artifact says
`T = 0.5532717521206881` and therefore `1 - T = 44.7%`. The point estimate was read off a terminal
paste rather than the JSON, and it propagated into this register, into the auditors' shared-context
brief, and into five places in the thesis chapter.

The intervals were never wrong -- `[0.514, 0.593]` and `[40.7%, 48.6%]` both reproduce exactly -- so
nothing that depends on the interval moves, and no verdict changes. It is a wrong number in a table,
which is the kind of thing an examiner finds.

**Root cause worth naming:** Part V was written from the log the run printed, because
`vardecomp_matched.json` was never pulled back from the cluster. A number with no local artifact
cannot be checked, and this one was not. The file has since been retrieved, along with the corrected
`calibration.json` -- whose values (`drf 0.6345`, `ci [0.6178, 0.6521]`, `p_holm 0.0025`) do reproduce
what the thesis quotes.

**What changed:** all five thesis sites, this register, and the auditors' brief. The lesson is the
one `build_thesis_assets.py` already encodes and which was not being used: a number that cannot be
regenerated from an artifact should not be quotable.

---

### 44 — 1 minus the transfer coefficient is not an interaction share

**Why does this matter?**

**Answer:** Part V presented `T` and then `1 - T` as "the drug x cell-line interaction share", with an
interval, as a headline quantitative finding. It is not that quantity, and the reason is visible in
the run's own arguments -- which I had read for the dose fix and never read for this.

```
same_dose_only  False     cross-line pairs may differ in DOSE
loo_generic     False     the generic subtracted from a condition INCLUDES that condition's drug
repro_thr       0.2       not the build's resolved 0.1086
```

A cross-line pair may also sit in a different treatment well. So `T` combines cell-line, dose, well
and target-estimator differences. Reading `1 - T` as an interaction share requires the other three
axes to be inert, and nothing establishes that.

**Found by an independent rewrite (Opus 5.6), not by me.** Its handoff demoted the coefficient to "a
descriptive sensitivity analysis rather than a load-bearing estimate" and enumerated exactly these
flags. That judgement is correct and has been adopted.

**One thing it got wrong while being right about this.** It replaced the estimate with
`T = 0.549 [0.507, 0.590]` from `vardecomp_plate.json` -- the **29 July pre-correction** run, which
has no dose resolution, no multiway intervals and no structure-matched control. Its objection to
`T = 0.558` was sound (that figure was my transcription error, defect 43) but the fix reached for a
stale artifact because the corrected one had not yet been pulled off the cluster. The right value is
`T = 0.553`, multiway `[0.514, 0.593]`, from `vardecomp_matched.json` -- **carrying the caveat above.**

**The dose arm is the exception and stays clean.** It holds drug and cell line fixed and varies only
molar concentration, so `T_dose = 0.707 [0.632, 0.782] > T_cross-line = 0.553` remains a legitimate
ordering contrast.

**What changed:** the interaction-share table row is retired in favour of "residual not shared across
contexts", the claim block and the cross-reference are reworded, and an explicit
identification caveat now sits immediately before the dose arm.

---

### 45 — the p-value and the interval used different reference distributions

**Why does this matter?**

**Answer:** the memorisation-premium block computed its interval from a Student-t critical value and
its p-value from a normal. The same statistic cannot be bounded under one reference and tested under
another; one of the two numbers was always going to be wrong.

```
pooled          p 0.0150 -> 0.0212   (df = 30)
drug-matched    p 0.4765 -> 0.4851   (df = 19)
```

Neither verdict moves -- the pooled difference still clears 0.05 and still fails to survive drug
matching -- but both printed values were wrong. Also found by the Opus 5.6 pass, and adopted.

---

### 46 — the token-overlap result was credited to the wrong subtraction

**Why does this matter?**

**Answer:** the chapter's causal story is that the standard target is uninformative because a generic
stress programme dominates it. The chapter's own table says otherwise about which subtraction does
the work:

```
full profile (standard target)   34.6 / 200 tokens differ between drugs
shift (treated - control)       111.4        <- control subtraction: +76.8, i.e. 92% of the recovery
residual (drug-specific)        118.2        <- generic subtraction: +6.8, i.e. 8%
```

**The dominant reason two drugs' standard targets look alike is the cell's own baseline state, not a
shared drug response.** The generic is real and separable -- it carries 39% of the shift's energy --
but it is the second stage, and much the smaller one, in token terms.

A second misattribution sat in the same sentence. It credited the generic drug programme with
"essentially all of the model's apparent skill" under the standard metric, citing the metric section.
That section shows something different: what saturates DE_dr is **reversion toward a central point**.
`revert_center` (map every gene to the panel midpoint, no fitting at all) scores **0.9999**, and
`revert_mean` (predict the mean control) scores 0.961. Neither is the mean-over-drugs generic.

**Found by the cross-version comparison against the Opus 5.6 rewrite**, whose handoff attributes the
overlap "mainly to baseline cell-state ordering, with generic response adding further overlap". That
reading is correct.

**Does the thesis arc survive?** Yes, and it is sharper for the correction. The residual target is
still the repair, and it still needs both subtractions. What changes is the diagnosis of *why* the
standard target fails: the tokens a cell sentence spends are dominated by which cell it is, and the
drug -- generic and specific parts together -- is what is left over. That is a stronger claim about
the representation than "a generic stress programme dominates", and it is the one the data supports.

---

# Part VIII — the corrected evaluation, and what it does to the claims

`re_v3.json`, 1394 conditions, 4 August. The first evaluation scored against the truth the checkpoint
was actually trained on: `fit_digest` verified against the build, `eval_repro_filter=false`, split
labels from the complete pre-filter assignment. It supersedes `re_repaired.json` entirely.

### 47 — the transfer verdict depends on the clustering axis, and I reported the wrong one

**Why does this matter?**

**Answer:** the eval script printed `CROSS-CONTEXT TRANSFER: YES` and I relayed it. The interval it
used clusters on CELL LINE. The claim is that the model applies a learned DRUG signature in a context
it never saw, so drug is the axis the claim has to generalise across -- and it is the scarce one.

```
unseen_combo gap +0.0332
  line x well   [+0.0069, +0.0594]   excludes zero
  DRUG x well   [-0.0031, +0.0695]   SPANS ZERO
```

It spans zero under drug clustering in all three conditionings -- as run, reliability-matched
(+0.0542) and ceiling-banded (+0.0461). **Transfer is not established.**

Three further facts point the same way. The split is 82% recombination: 30 of its 34 drugs and 39 of
its 42 cell lines appear in training, and only 4 drugs are genuinely novel (145 records, gap +0.0454
[-0.0711, +0.1620]). The effect is a majority tilt, positive for 20 of 34 drugs, sign test p = 0.39.
And against the NEAR comparator the model is numerically worse, -0.0060 [-0.0401, +0.0282]: told a
mechanistically similar wrong drug, it does as well as when told the truth.

**A mechanism I stated wrongly.** I explained the paired/unpaired discrepancy as pairing removing
condition-level variance. It does not. The paired gap (+0.033164) and model minus chance (+0.033430)
are the SAME NUMBER, because the neutral comparator sits at 0.500266. Pairing INCREASES iid variance
here -- sd(diff) 0.335 against sd(model) 0.279, correlation +0.275. It helps only in the cluster
dimension.

### 48 — what the residual re-encoding did establish, stated precisely

The claim survives, and on firmer ground than transfer. On trained conditions, clustered on the drug:

```
gap vs the neutral comparator   +0.0898  [+0.0291, +0.1506]
NIR against chance, no comparator +0.0981  [+0.0407, +0.1554]
```

And the comparator-free slope excludes zero on **every split under both clusterings**:

```
train         +0.3728  drug x well [+0.2797, +0.4660]
unseen_combo  +0.3365  drug x well [+0.2230, +0.4499]
unseen_drug   +0.3420  drug x well [+0.2152, +0.4689]
```

**The unseen-drug slope is the interesting one.** The model tracks the named drug just as well on
compounds held out of fine-tuning entirely. So re-encoding did not teach the model about drugs --
it changed the target so that drug knowledge the model already had could reach the output. That is a
sharper claim than "the residual helped", and better supported.

Two cautions to carry: the model RESPONDS to the drug, it does not PREDICT it (mean own-truth cosine
+0.039; the lookup beats it by -0.3631). And even on train the drug-level sign test is 46/76,
p = 0.085.

### 49 — the exposure advantage is gone, and the ceilings explain why

```
pooled        +0.1008 p=0.0212  ->  +0.0567 p=0.0556
drug-matched  +0.0420 p=0.4851  ->  -0.0606 p=0.1797   (sign flipped)
```

The cause is a confound the chapter previously reported as CLOSED. Split ceilings were
0.962/0.955/0.956; they are now **0.956/0.824/0.836**, differing by 0.132. Training conditions are
reliability-filtered regardless of `eval_filter`; held-out conditions no longer are. So train keeps
clean targets while the held-out splits include the hard conditions the old path silently discarded.

**Do not fix this by filtering the held-out set.** That is the outcome selection removed in Part VI,
and it would discard 499 of 895 held-out conditions on their own outcome. Condition in the ANALYSIS
instead: matched at equal ceilings (0.956 vs 0.961) the exposure difference is +0.0356, p = 0.3131.
The confound was carrying it.

**Claim boxes rewritten:** three of twelve. The channel-gate box waits on Job 3.

## Running state after Part VI

| | status |
|---|---|
| scope 7, scope 8 | done (Part V) |
| **truth replay / held-out outcome selection** | **OPEN -- blocker, reaches printed numbers** |
| premium drug composition (31) | **DONE** -- restricted contrast shipped; the claim is weakened |
| `max_abs_cos` (32), hedging (33), coverage support (34) | **DONE** |
| channel gate frame + clustering (35) | OPEN -- load-bearing for Q18 |
| t reference distribution (36) | **DONE** -- one control verdict flipped; see below |
| field decomposition (37) | OPEN -- needs GPU; the one experiment worth adding |
| `crossed_bootstrap` (38) | **DONE** |

---

### The t correction invalidated an argument the same afternoon it was written

Worth recording because of the timing rather than the size. While the CPU tier was being implemented,
a parallel session resolved a separate contradiction about the `unseen_drug` arm and wrote into
`Results-and-Analysis.tex` and `FINDINGS.md` that the arm's gap against `opposite`, +0.0862, has an
interval **excluding zero** -- arguing that a positive result on drugs the model never saw is
impossible, so the control is failing, and that this is what exposes the comparator.

The t correction landed minutes later and moved that interval from `[+0.0017, +0.1708]` to
`[-0.0028, +0.1753]`. It spans zero. The argument as written no longer holds.

**The conclusion survives; the route to it does not.** The comparator's non-neutrality is established
directly -- the scramble arm's own NIR is 0.572 / 0.497 / 0.403 as the partner's true response moves
from aligned to anti-aligned -- and that measurement needs no control row at all. Five sites were
rewritten to rest on it. The `unseen_drug` row corroborates; it does not carry.

Two lessons. A verdict standing 0.0017 from the boundary was never load-bearing enough to hang a
section on, and nothing in the pipeline said so, because the reference distribution was wrong in a
direction that always flatters. And an argument built on "the interval excludes zero" should name the
critical value it used.

**The premium's restricted contrast, measured:**

```
DRUG COMPOSITION  train 68 drugs, unseen_combo 29 drugs, 20 shared (22.5% / 64.4% of observations)
                  drugs in ONE arm only: train +0.1694  combo +0.0864
pooled            +0.1008  SE 0.0414  z 2.43  p 0.0150
RESTRICTED to 20  +0.0420  SE 0.0590  z 0.71  p 0.4764
```

**The geometry, reported properly:**

```
train         mean +0.0511  sd 0.1141  median +0.0300  range [-0.1819, +0.4395]   |cos|>0.1: 32.0%
unseen_combo  mean +0.0085  sd 0.0849  median +0.0084  range [-0.2290, +0.2510]   |cos|>0.1: 21.6%
unseen_drug   mean -0.0037  sd 0.0852  median -0.0076  range [-0.2794, +0.3194]   |cos|>0.1: 20.0%
```

**Commits in this part:** `0532eae`.


---

# Part VII — fixing the blocker, and the four defects that fix introduced

The truth-replay blocker from Part VI is closed (`96b6daa`, `5797a37`, `92bc5b3`). What is worth
recording is not the fix but its failure rate: **an adversarial pass over the finished work found
four real defects, two of which would have wasted the 24-hour GPU run**, and two of them were in
code written specifically to prevent that class of error.

### 39 — reconstructing a tuple from the manifest key would have reintroduced the defect

The manifest stores keys as `"|".join(map(str, k))`, which is lossy. Reconstructing with
`tuple(s.split("|"))` yields strings; condition keys come from parquet, so an element can be an int.
Nothing matches, every lookup defaults to `"train"`, and `transform` filters on the `s == "train"`
disjunct exactly as before -- while logging that the labels had been applied.

It passed the suite because the fixture's `cell_line_id` is `'CVCL_0000'`, a string. Caught by
checking key types empirically rather than trusting green tests. Comparison now happens in the string
space the manifest was written in, in one place, and **refuses** below 50% resolution.

### 40 — the channel gate could not import `inference` at all

`sys.path` carried `endcell/analysis` and `endcell/ot` but never `shared`, so `two_way_ci` returned
`None` on every call and `gap_vs` fell back to the one-way bootstrap -- the estimator the fix was
written to replace. Tracing the selftest shows `two_way_ci` and `gap_vs` are both **NOT CALLED**, so
73 tests and a passing selftest proved nothing about it.

On a 39-line x 25-well design the two-way interval is **1.7x wider**, which at the observed gap flips
the `moa` verdict. Both the ImportError and the per-channel fallback now announce themselves.

### 41 — a `fit_digest` mismatch only logged, and my own tests were failing it silently

`residual_eval` loads a causal LM onto CUDA minutes after this call, so "log three errors and
continue" means a GPU day spent scoring against a truth the checkpoint never learned, ending in a
number that looks real. It now raises.

Making it raise **immediately failed two of my own replay tests**: they replayed at `seed=0` against
a fixture built at `seed=42`, so they had been printing `fit_digest MISMATCH` and passing. That is
this whole chain's failure mode, occurring inside the tests written to prove it had been removed.

**And the digest is weaker than it looks.** `Generic.digest()` hashes the fitted drug means, not the
estimator that turns them into targets, so `scope`, `shrink_k`, `loo` and `min_plate_drugs` can all
differ while it reports MATCH -- each changing the targets by up to their own norm. Measured: at
`min_plate_drugs=99` the kept set collapses to zero and the digest still says MATCH.

The hash is deliberately **not** changed. Doing so would make identical data hash differently and
fail Job 1's build-to-build comparison against an artifact written by the old code, inviting a
spurious retrain -- and all 73 tests would still pass. Those four parameters are cross-checked against
the manifest instead, and the error text no longer names causes that cannot fire.

### 42 — the rebuild job would have used the wrong threshold, and the fix for that would have crashed

The previous build used `--repro_thr auto`, resolved to ~0.1086; the argparse default is 0.2 and the
job script omitted the flag. Independent confirmation from the artifact: `re_repaired.json`'s 570
records have min `repro_cos` 0.10890 and **zero** at or below 0.10857.

Rebuilding at 0.2 writes a different `residual.jsonl` and fails the gate -- and the `fit_digest` half
would still have said SAME, because the digest is computed **before** the threshold resolves. The job
now reads the literal out of the old `report.json`, since `auto` re-resolves against a sampled null.

Then that fix would itself have crashed: `--repro_thr` has no argparse `type=`, so a command-line
number arrives as a string and hit the keyword dict -- `KeyError: '0.1086'`, an hour in, after the
fit. Numeric strings are coerced; nonsense is refused rather than treated as a keyword.

### Also closed: `--holdout` alone was doing nothing

Two GPU jobs in `jobs/next.md` passed `--holdout` without `--truth_from`. `residual_eval` reads the
holdout only *inside* the `--truth_from` branch, so those runs built the truth in the published
frame. Measured on the fixture: every target differs, median relative L2 **0.512**, median cos 0.868.

---

## The lesson this part is actually about

Three of these four survived a full green suite, and two were introduced *by* the remediation. A
passing test tells you the code does what the test says; it says nothing about whether the test
exercises the path that matters, runs in the right frame, or asserts on a quantity the estimator can
actually see. Every one of these was caught by running the code and reading the output -- key types,
a traced call list, a digest table across perturbed parameters -- rather than by reasoning about it.

**Queue state:** the four cluster jobs are in `coda_errata/HPC_QUEUE.md`, all flags verified against
the live argparse blocks, nothing on the login node (2 `srun` checks, 4 `sbatch` submits).


---

# Part IX — closing the register

Forty-nine numbered defects across eight parts. **Every correctness concern raised by the four
external audits is closed.** What follows is what is deliberately NOT closed, so the distinction
between a defect and a scope limit is on the record.

## Closed in this final pass

- **The prompt-field attribution**, which was the last empty table. Swapping the drug name moves the
  output by **+0.1026 [+0.0625, +0.1427]**; swapping the mechanism with the name left in place moves
  it by **+0.0303 [-0.0078, +0.0685]** -- an interval containing zero on the drug axis *and* on the
  cell-line axis. The name carries about nine tenths of the combined effect.

  The evaluation script printed the opposite conclusion ("the model DOES read the mechanism") from a
  ONE-WAY interval, `[+0.0021, +0.0596]`. It does not survive two-way clustering. This is the same
  error class as defect 47 and it is the second time a one-way interval produced a verdict the
  two-way estimator withdraws.

  The result closes a loop the chapter opens: the channel gate shows the mechanism field **carries**
  transferable signal (+0.0604 [+0.0094, +0.1114] under the different-plate null), and the field
  decomposition shows the model **does not read it**. The bottleneck is readout, not availability --
  and the channel-conditioning arm stays closed for the right reason.

- **The channel gate in the repaired frame**, two-way clustered, with the different-plate null
  printed beside the plate-matched one. Chemistry clears zero under plate matching and does not
  under different-plate (+0.0183 [-0.0033, +0.0399]): its apparent signal is a co-plating artefact.

- **The plate-matched calibration**, DRF +0.519 [+0.479, +0.557], quoted beside the all-plates +0.635
  as the confound-free figure.

## Open, and open on purpose

These are limitations, not defects. Each is stated in the thesis as such.

1. **One training seed, one training epoch, one target half-split, one generation seed.** Every
   interval is conditional on those choices. Defect 22's half-split variability is identified and not
   costed.
2. **The splits are no longer equally hard** (ceilings 0.956 / 0.824 / 0.836). This is a *consequence*
   of removing the outcome selection, not a regression. It is handled by conditioning in the
   analysis; it must not be handled by filtering the evaluation.
3. **`unseen_drug` sits below chance** at NIR 0.459. The control passes -- the gap against the
   neutral comparator spans zero -- but a below-chance point estimate on a control arm deserves a
   sentence in the text rather than a table cell.
4. **Commit `41c6716` carries unrelated DE-weighting code on the training path**, verified inert at
   defaults and exercised by no test.
5. **Fixed-dose cross-context transfer is structurally unidentifiable in this atlas**, which no
   amount of analysis repairs.

## What is left is writing

The numbers are settled and every one is regenerable from an artifact in `RESULTS_cluster/`. What
remains is prose: the chapter is correct and still reads in places like the defect register it grew
out of. That work is tracked in `REWRITE_PROMPT_V5.md`, not here.

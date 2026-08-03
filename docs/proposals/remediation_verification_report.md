# Remediation, steps 1–4: what changed, and how to check it

**For the auditors.** This is a verification request, not a summary. Every claim below is written so
you can falsify it: file and line, the command that exercises it, and what the command should print
if the claim is true. Where I am uncertain, or where a repair is partial, it says so in the same
voice as the rest — please attack those sections first.

**State.** `HEAD = b7a2827`, 2026-08-02. Steps 1, 3 and 4 of `remediation_execution_scope.md`
are landed. **Step 2 is script-only**: `experimental_unit_audit.py` is committed but the job has
not been run, so none of the discovery questions the scope document requires it to answer is
answered yet. Steps 5–10 are not landed. **No corrected number exists yet**: the rebuild and the retrain are written and
queued but have not run, so nothing in the thesis has moved. This document is about whether the
machinery that will produce those numbers is sound.

| commit | contents |
|---|---|
| `f5cdca7` | Step 0 — removal of known-false text (prior session) |
| `762951a` | Step 1 — `shared/tahoe_design.py`, the canonical design layer, + 36 tests |
| `533c293` | the scoped execution plan this follows |
| `8525254` | Step 2 — `experimental_unit_audit.py` (script only; job not yet run) |
| `41c6716` | **Steps 3–4** — split-before-fit, generic scope, drug weighting, experimental unit |
| `823f447` | `FINDINGS.md` Q22 — source-level record of the defects |
| `b74a171` | restores `build_residuals` for the five eval consumers; **split controls** for reliability |
| `b7a2827` | plate-matched channel-gate null; artifact manifest |

The table is in commit order.

---

## 0. The one thing that changed the plan

`build_embeddings.py`, before the fix, line 134:

```python
drug, cl, plate, dose = row.get("drug"), row.get("cell_line_id"), row.get("plate"), row.get("sample")
```

`row["sample"]` — the treated well — assigned to a variable named `dose`, then appended to a metadata
column named `dose`. That single line is the entire origin of `metadata["dose_float"] == "smp_1841"`,
and it means the well identity was **mislabelled, not lost**. The consequence for the remediation is
large: the cache is **enriched in place** rather than rebuilt. `remediation_execution_scope.md` named
this as the plan's largest cost uncertainty, and it has resolved to the cheap branch.

**Check:** `git show 41c6716 -- endcell/ot/build_embeddings.py`.

---

## 1. Verification harness

Everything below runs offline in a few seconds. No cluster, no GPU, no network.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_split_before_fit.py tests/test_tahoe_design.py -q
```

Expect `55 passed` — 19 in `test_split_before_fit.py`, 36 in `test_tahoe_design.py`. The plugin
disable is unrelated to this work; a broken `dash` pytest plugin is installed in the environment.

```bash
python shared/tahoe_design.py --selftest
```

```bash
python endcell/ot/build_residual_targets.py --selftest
```

```bash
python endcell/analysis/experimental_unit_audit.py --selftest
```

Each prints an `ok`/`FAIL` line per property and exits non-zero on failure.

---

## 2. Defect 1 — the fit saw the holdout

**The finding.** `build_residual_targets.run()` computed the generic shift and applied the
reliability filter over **every** condition, and assigned the train/holdout split **afterwards**. So
each held-out target was defined partly by the other held-out conditions, and the reported
cross-context transfer was measured against targets constructed with knowledge of the holdout. This
is the audit's transductive-holdout finding, and it is correct.

**The repair.** `run()` is now four stages, each seeing only what the stage before is allowed to
expose:

| stage | function | what it may read |
|---|---|---|
| inventory | `inventory()` | metadata and cell **counts**. No expression is aggregated. |
| split | `holdout_from_tiers()` / `make_holdout()` / `make_holdout_by_sample()` | metadata only, before a single pseudobulk exists |
| fit | `Generic.__init__(shifts, conds, train_keys)` | the train keys, passed in explicitly |
| transform | `transform()` | held-out conditions projected through the fitted generic; they never enter it |

The count filters (`--min_treated`, `--min_control`) are applied at inventory. They are
metadata-derived — row counts, not outcomes — so applying them before the split is legitimate. If you
disagree, that is a finding I want.

**The gate.** `tests/test_split_before_fit.py` is a **poison test**, chosen because it is the only
form of this check that cannot be satisfied by moving the split earlier and leaving the arithmetic
wrong. It corrupts every held-out expression vector with large noise, rebuilds, and requires:

1. the training JSONL to be **byte-identical**, and
2. `fit.sha` — a SHA-256 over every fitted quantity (`Generic.digest()`, which hashes each scope
   group's per-drug means at all three resolutions: the full aggregate and both half-splits) — to
   be **unchanged**.

**How to falsify the gate.** Reintroduce the defect and confirm the test fails. One line:

```bash
sed -i 's/gen = Generic(shifts, conds, train_keys, shrink_k=args.shrink_k)/gen = Generic(shifts, conds, list(conds), shrink_k=args.shrink_k)/' endcell/ot/build_residual_targets.py
```

Then rerun the suite. I ran exactly this. It fails on the digest with the intended message:

```
AssertionError: held-out expression reached the fitted generic -- the split is still being
assigned after the fit, or the generic is not restricted to train keys
assert 'f40d5288e101...' == '681beaf8bcbe...'
```

`git checkout endcell/ot/build_residual_targets.py` to restore.

**The negative control**, without which the gate would pass for a builder that reads nothing at all:
`test_poisoning_a_training_condition_does_change_things` poisons three *training* conditions and
requires **both** the digest and the JSONL to move.

**Two further guards in the same file:** `test_the_split_is_actually_three_way` (a poison test over
an empty holdout proves nothing) and `test_held_out_drugs_are_absent_from_the_fitted_generic`, which
inspects `Generic.fine` / `Generic.coarse` directly rather than trusting the poison.

---

## 3. Defect 2 — the generic was cell-line-scoped, and we already knew it was contaminated

**The finding.** The generic was a mean over conditions **per cell line**. The transfer analysis
(`FINDINGS.md` Q17, table at lines 519–523, produced by `variance_decomposition.py`; the
plate-scope figures are on line 521 and the cell-line ones on 523) had already measured
that this leaves plate structure in the target. Three independent signals, at cell-line scope versus
plate scope:

| diagnostic | cell-line scope | plate scope |
|---|---|---|
| same-plate null (different drugs, same plate) | **+0.478** | −0.018 |
| shared-control minus split-control bias | **+0.254** | −0.000 |
| dose ordering (different-dose vs cross-line) | **inverted** (0.484 < 0.517) | correct (0.703 > 0.557) |

The first says different drugs on the same plate look nearly as similar as true transfer. The uncomfortable
part is the timeline: `variance_decomposition.py` has had a `--generic_scope` flag since that
analysis (line 727, default `cell_line`), but `build_residual_targets.py` did **not** — it was
hardcoded to cell-line scope. So we measured that plate scope was the clean one and then trained on
targets built the contaminated way.

**Caveat on the evidence, stated because it cuts against me.** The third row — dose ordering — comes
from the arm that keyed "different dose" on **sample identifiers**, so it is void as a dose statement.
It survives only as *"changing the well costs less than changing the cell line"*, which is weaker.
The first two rows are unaffected: neither involves dose.

**The repair.** `--generic_scope` now exists in the builder and **defaults to `plate`**, meaning
`(cell_line, plate)` — the grouping the control pseudobulk already uses.

**The cost of plate scope — and a correction that cuts in my favour, which is why it needs saying
loudly.** An earlier draft of this document argued that plate scope is expensive, citing 62% of
conditions reproducible at cell-line scope against 19% at plate scope. Two things were wrong with
that. The figures are not in the builder's docstring, as claimed — they are at `FINDINGS.md:392`.
And **`FINDINGS.md:532` retracts them**: that comparison was measured with *shared* controls,
which inflate cell-line reproducibility far more than plate reproducibility. Measured with split
controls the two are comparable — **20% (plate) against 16% (cell line)** — and the same entry
concludes plate scope is the better build.

So retention is not an argument for cell-line scope; if anything it slightly favours plate. What
survives is the *absolute* concern: ~20% under either scope may not leave enough to train on.
That is a question about training-set size, not about which scope is right, and the builder
offers:

- `--shrink_k K` — blend the plate generic toward the cell-line generic with weight `n/(n+K)`,
  where `n` is the number of **other** training drugs in the plate group (leave-one-drug-out
  applies here too; `build_residual_targets.py:456`). The standard hierarchical compromise: plate scope where a plate has
  enough drugs, cell-line scope where it does not. `K=0` (default) is pure plate scope.
- `--scope_sensitivity` — report reliability retention at `cell_line`, `plate`, `plate+shrink5`,
  `plate+shrink20`, **computed on train conditions only**, so the configuration is chosen from
  numbers rather than from either argument alone.

The queued CPU job (`jobs/next.md` §9) runs with `--scope_sensitivity` and that table **is** the
decision gate. **This is an open configuration choice, not a settled one.**

One further correction, and it is the most serious thing found in reviewing this document.
**That gate was itself biased against the rebuild it was meant to adjudicate.** `compute_shifts`
subtracted the *same* control pseudobulk from both halves, so `res_A` and `res_B` shared the entire
control-noise term and were correlated for a reason unrelated to the drug — exactly the shared-control
flaw `FINDINGS.md:532` retracts, and one that `variance_decomposition.py:55` had already documented.
The new builder reproduced a defect the repository already knew about. Since the inflation favours
cell-line scope, `--scope_sensitivity` would have argued against plate scope on a measurement
artifact. Fixed in `b74a171`: halves are measured against disjoint control halves, the full shift
still uses every control cell, and `--shared_control_reliability` reproduces the old behaviour for
a side-by-side. `test_reliability_halves_do_not_share_a_control` asserts the inflation exists and
that the fix removes it.

---

## 4. Defect 3 — drug weighting, and a fourth defect found while fixing it

**Drug weighting.** The generic was a mean over **conditions**, so a drug measured at five doses
contributed five times the weight of a single-dose drug. It is now a mean over **drugs**: each drug's
conditions within a scope group are averaged first, then drugs are averaged.

Covered by `test_dose_weighting_does_not_let_one_drug_count_twice` — two doses of X at 6.0, one of Y
at 0.0, leave Z out: the generic must be 3.0, not the condition-weighted 4.0.

**Leave-one-drug-out — found while fixing the leak, not reported by the audit.** Once the generic is
train-only, an asymmetry appears that did not exist before. A **train** condition's generic contains
its own drug; a **held-out** condition's does not, because its drug may not be in the train set. The
two splits would then be scored against differently defined targets, and the generalisation gap would
partly measure that difference rather than generalisation.

`Generic.value()` excludes the condition's own drug from its own generic, so the definition is
identical on both sides: *"the mean over training drugs other than this one, within scope"*. For an
`unseen_drug` condition it is a no-op, which is correct — its drug was never in the train set.

Covered by `leave-one-drug-out removes the condition's own drug` in `--selftest`.

---

## 5. Defect 4 — the experimental unit

**The finding** (audit's, and correct): Tahoe assigns a treatment to a **sample/well** carrying a
mixture of cell lines. The per-cell-line profiles are deconvolved observations nested inside one
assignment. Everything keyed on `(drug, cell_line, plate, dose)`, which does not identify a physical
experiment — and whose fourth element was a sample identifier anyway.

**The repair, in three places:**

1. **Source.** `build_embeddings.py` writes `sample_id`, not `dose`. A reader doing `meta["dose"]`
   now gets a `KeyError` instead of silently receiving well identifiers — the failure mode we want.
2. **Recovery.** `tahoe_design.sample_column(df)` returns `(column, how)`: an explicit `sample_id` /
   `sample` column if one exists, else `dose` **only if `looks_like_sample_id` confirms its contents
   really are identifiers**, else `None`. Both builders log which branch they took
   (`build_residual_targets.py:189`, `build_ot_targets.py:156`); the residual builder records it in
   `report.json` as `inventory.sample_id_source`, and the OT builder now records it in
   `step0_gates.json`. Existing caches are read through the recovery branch; newly built ones through
   the explicit branch.
3. **Dose.** The concentration is resolved from `drugname_drugconc` through
   `tahoe_design.parse_treatment` → `Dose.molar`, which is `None` with a stated `reason` when
   unconvertible and **never `0.0`** — a missing dose and a zero dose are different experiments.
   Emitted metadata is now `sample_id`, `dose_molar`, `dose_raw`; `dose_float` is gone.

**Combinations.** `drugname_drugconc` is a list of `(drug, value, unit)` triples and the shipped
parser took `parsed[0]`, so a two-drug sample was analysed as its first component. `parse_treatment`
returns every component and `primary` is `None` for a combination, forcing callers to branch.
The builder drops them by default (`--keep_combinations` to override) because the prompt cannot
express a combination.

**A write-time refusal.** Reviewing this document caught that the check ran *after* the JSONL was
written and closed, so `REFUSING TO WRITE` was not literally true and a refused run left a
complete-looking training file on disk. Fixed in `b74a171`: both builders now write to a
`.partial` path and rename only once the check passes, deleting the partial otherwise. This
catches the specific defect, not dose absence in general — that is what
`inventory.n_without_molar_dose` is for.

**A limitation I could not engineer away, so it is counted instead.** A `(drug, cell_line)` holdout
**cannot** prevent a held-out condition sharing a treated well with a training condition, because one
well carries many cell lines. This is the experimental design, not our code. `sample_crossing_report()`
counts it and writes it into `holdout.json` as
`frac_heldout_conditions_sharing_a_well_with_train`, so it becomes a number in Methods rather than a
hedge. `--split_unit sample` drives it to zero — verified by
`test_sample_split_unit_removes_well_crossing` — but changes the estimand from cross-context transfer
to dose/replicate generalisation, which is a different claim. **Which estimand the thesis should
claim is a question for you.**

---

## 6. Defect 5 — ours, not in the audit: the original preprocessing strips the unit

Found while patching the readers, and I think it matters.

`endcell/preprocess/tahoe_c2s_preprocess_endcell.py:972`:

```python
dose_float = float(dose_str.split()[0])
```

`dose_str` is a display string like `"0.05 uM"`. The unit is discarded. Two opposite failure modes:

- **collision** — `1 uM` and `1 nM` both become `1.0`: two concentrations treated as one;
- **split** — `0.05 uM` and `50 nM` are one concentration but become `0.05` and `50.0`.

This is a **different defect in a different pipeline** from the sample-ID one. Note that the original
preprocessing writes the well identifier *correctly*, under `metadata["sample"]`; only the caches got
it wrong.

Why it is load-bearing: tier 4 — the held-out-dose evaluation — is decided at lines 1006–1008 by
`abs(dose_float - held_out_dose_per_drug[drug]) < 1e-6`, i.e. by exactly that comparison. A collision
means tier 4 held out a dose it also trained on. Other consumers of the same field:
`build_consensus_targets.py` (`--group_keys drug,cell_line_id,dose_float`),
`shared/evaluate_c2s_tahoe.py:1045`, `check_dose_coverage.py`.

**Whether either mode actually bit is a property of the data, so it is measured rather than
asserted.** `experimental_unit_audit.py` question 5b counts collisions and splits per drug from
`sample_metadata.parquet`. Both directions are covered in the selftest. **That job has not run yet**,
so at present this is a demonstrated *possibility*, not a demonstrated *fact* — please read it that
way.

---

## 7. Consumers patched

| file | change |
|---|---|
| `endcell/analysis/residual_eval.py:254` | trained-set key reads `sample_id`, falling back to `dose_float` for files built before the fix |
| `endcell/analysis/workspace_probe.py:702` | context field prefers `dose_molar`; carries `sample_id` separately |
| `endcell/analysis/dose_response_analysis.py:231-237` | prefers `dose_molar`; **refuses a string `dose_float`**, which is what a sample identifier would be. A dose-response curve read from one of the old residual/OT files was ordering wells, not doses. |

Left alone deliberately: `check_dose_coverage.py`, `build_consensus_targets.py`,
`evaluate_c2s_tahoe.py` and the preprocessing scripts all consume the **original** preprocessed
JSONL, which is a different data source with a different (§6) defect. Patching them to read
`dose_molar` would silently change what those analyses mean.

---

## 8. Reproducibility

`--seed` added to `train_c2s_tahoe_endcell.py`, seeding Python, NumPy and Torch at the top of
`train()`. This fixes the shuffle order and the dropout masks. It does **not** make a GPU run
bit-exact — cuBLAS reductions are non-deterministic — and the docstring says so. The claim is only
that every source of variation under our control is removed, so a difference between two arms is a
difference between the arms.

---

## 9. Deliberately not done

From `remediation_execution_scope.md`, with the reasoning restated so you can push back:

| cut | why |
|---|---|
| artifact lock, run manifests, claim registry, CI | weeks of infrastructure that changes no number; replaced by a claim→artifact table and generated macros |
| token/gradient diagnostic | only needed to *retain* a gradient claim; that claim was downgraded to a hypothesis, which removes the requirement |
| general power-simulation framework | replaced by targeted power statements for the two null claims that need them |
| `residual_metrics.py` unification | patching two evaluators is lower-risk than merging them in week five |
| κ and channel-gate reruns | the plan's own stop rule: coverage and power are inadequate, so a rerun cannot change the conclusion |
| survivor-selection quantitative audit | the *wording* is required and costs nothing; the audit needs uncapped metadata we may not have |
| common frozen checkpoint, arm matching (D2–D3) | supports a causal target-encoding claim we no longer make; seeding (D1) is kept, and is §8 |
| panel sensitivity | not required if every claim is panel-scoped in wording, which is cheaper and equally honest |
| OT/consensus arms rebuilt | they support the ε-ladder, a negative result unaffected by the residual repairs |

---

## 10. What is still wrong, or still unknown

Listed because these are the parts most likely to matter to you.

1. **No corrected number exists.** The rebuild (CPU) and retrain (GPU) are written and queued and
   have not run. The corrected numbers may move in **either** direction: the leak and the plate
   structure inflate the shipped residual result, while the oracle `drug_lookup` and the same-drug
   scramble partners deflate it. *"Everything shrinks"* is not the safe prediction.
2. **The reliability filter still selects the eval set on its own outcome.** By default the
   `cos(res_A, res_B) > thr` filter applies to held-out conditions too. It now uses the *train*-fitted
   generic, so no held-out condition is defined by another; but a condition is still dropped based on
   its own measured reproducibility, which makes the eval set non-representative. My reasoning is that
   scoring a prediction against a target that is not itself reproducible is meaningless, so this is a
   measurement-quality restriction rather than a performance one — and the per-split retention rates
   are reported so the selection is visible. `--no_eval_repro_filter` restricts it to training
   conditions. **I am not confident this default is right and would like a ruling.**
3. **Step 6 has not landed.** `drug_lookup` / `moa_lookup` are still fitted from the full cache and
   are therefore **oracles**; scramble partners can still be the same drug at another dose, which is
   why the `near` stratum is contaminated and the monotone-gradient argument is weaker than it was
   presented. Until that lands, the lookup baseline should be read as an upper bound rather than a
   competitor.
4. **The residual arm remains a per-condition retrieval task.** Every control cell in a condition
   gets the *same* target string. A clean retrain does not change that, and it is not a repair of the
   original perturbation-prediction estimand. Only the frame reconciliation speaks to that, and if it
   cannot be done properly the honest conclusion is exploratory evidence on a different question.
5. **`--split_unit sample` is untested at scale.** It passes its unit test but has never been run on
   the real cache, and it would materially reduce the training set.
6. **The `target_divergence` provenance note.** The builder's docstring cites "34.6 of 200 tokens
   differ" as the rationale for the residual target. `target_divergence.py` compares gene **sets**,
   ignoring order, so a rank sentence can share every gene and still differ in every position — the
   direction of that error is not established. The docstring now says so and marks the number as
   provenance only. The rebuild does not depend on it.

7. **I broke the evaluation stack and did not notice.** Restructuring `run()` into stages deleted
   `build_residuals`, which five scripts call: `residual_eval`, `channel_gate`, `reconstructed_eval`,
   `reward_calibration` and `build_de_weights`. Nothing caught it — the poison test exercises `run()`,
   and none of the five has a test — so it would have surfaced on the cluster at the eval step of a
   job whose training had already finished. Restored as a shim in `b74a171`, with a regression test
   asserting both that the API exists and that the returned contract still matches what the consumers
   unpack. **The shim's defaults reproduce the old, defective semantics on purpose** — cell-line scope,
   no leave-one-out, shared controls — because the published numbers came from that build and silently
   repairing them here would make the published figures unreproducible.
8. **A second audit found a defect of the same class in a printed claim.** The Q18 channel gate pairs
   each channel arm with a *count-matched* random null. Mechanism- and target-paired drugs are largely
   co-plated; random partners are not; and the residuals come from the plate-retaining cell-line-scoped
   build. So the arm and its null differ in two respects where the estimand allows one — the same class
   as the two probe retractions. A selftest planting a plate effect and **no** channel biology gives
   +0.384 against the count-matched null and +0.016 against a plate-matched one, so the construction can
   manufacture an effect several times the +0.078 being defended. `b7a2827` adds the plate-matched null,
   a different-plate arm, and a co-plating diagnostic; the re-run is `jobs/next.md` §11 and it may
   retract `Conclusions.tex:123-129`.

---

## 11. Disclosure: commit `41c6716` is not clean

`41c6716` touches `endcell/train/train_c2s_tahoe_endcell.py` with **168 insertions and 17
deletions** (`git diff --numstat 41c6716^ 41c6716 -- endcell/train/train_c2s_tahoe_endcell.py`).
Only **22 insertions are mine**: 17 in `train()` (hunk `@@ -158,6 +262,23 @@`, of which 10 are
executable) plus the 5-line `--seed` argparse block. The remaining **146 insertions and all 17
deletions** are pre-existing uncommitted DE-weighting work that was in the working tree and that I
swept in with `git add`. The commit message describes only the seed.

It is not purely additive, which matters more than the line count. The DE-weighting code adds
`de_weight`, `de_share`, `_token_weights`, `measure_de_token_share` **and `forward_loss`**, and it
**rewrites executed code**: both the AMP and non-AMP loss branches now route through
`forward_loss`, and `C2SDataset.__init__`, `collate_fn` and the loss accumulator all changed. The
retrain in §10 will run through that path. It is exercised by no test here, and its companion
`build_de_weights.py` is still untracked.

No other file in that commit is affected — I checked each. Scope your review of `41c6716` to the
**two** seeding hunks and treat the DE-weighting hunks as unrelated, unreviewed work that happens
to sit on the training path.

---

## 12. What I would most like challenged

1. **Are the inventory-stage count filters legitimate pre-split operations?** They are
   metadata-derived, but they do determine which conditions exist to be split.
2. **The eval-side reliability filter** (§10.2). Measurement-quality restriction, or outcome-based
   selection that biases the holdout?
3. **Plate scope at low retention.** Under honest split-control measurement the two scopes retain
   comparably (20% plate, 16% cell line, `FINDINGS.md:532`), so the question is not which scope but
   whether ~20% leaves enough to train on. Is shrinkage a principled fix here, or a hyperparameter
   that invites tuning to taste?
4. **Well crossing.** Is a `(drug, cell_line)` holdout with ~X% of held-out conditions sharing a well
   with training data defensible as "cross-context transfer", or does the estimand have to become
   `--split_unit sample`?
5. **Is the poison test sufficient**, or is there a class of leakage it cannot see? It gates on the
   training JSONL and the fitted generic. It does **not** gate the reliability threshold, the split
   assignment itself, or anything in `residual_eval.py`.

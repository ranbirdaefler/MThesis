# Remediation: what I will actually do, and what I am cutting

**Purpose.** The defensibility plan is comprehensive but not prioritised. This scopes it into work I
will perform, in order, with the cuts stated and justified. The test applied to every item is: *does
skipping this leave a claim in the thesis that a viva can break?* If no, it is cut regardless of how
good an idea it is.

**Standing constraint.** I write and test scripts locally; cluster jobs are submitted by hand. So the
critical path is set by queue latency on exactly two GPU jobs, not by engineering time.

---

## The judgement in one paragraph

Four of the plan's twenty-six sections are load-bearing: the data-integrity fix (dose/sample), the
leakage fix (split-before-fit), the evaluation fixes (partners, negatives, baselines), and the thesis
rewrite. Three more are cheap and worth doing (survivor/panel/timepoint scoping, metric-calibration
hardening, transfer recomputation with real dose). The rest is either already done, engineering
hygiene that changes no number, or explicitly ruled out by the plan's own stop rules.

**One decision dominates everything else:** whether the residual retrain happens. If it does, the
thesis has six results. If it cannot be run cleanly, it has four, and the generalisation and repair
claims become exploratory evidence on a different estimand. Both are defensible; only one is
ambitious. That decision should be made at the end of Step 4 and not revisited.

---

## Steps

### Step 0 — remove known-false text — **DONE**

Retracted swap out of Conclusions/Methods/Results/figure caption; gradient claim downgraded to a
hypothesis; "irreducible" replaced with "unresolved"; `no lookup can structurally reach` softened.
Commit `f5cdca7`.

### Step 1 — canonical design layer — **DONE**

`shared/tahoe_design.py` + 36 tests. Dose and sample separated, molar normalisation, combination
treatments no longer silently truncated, `looks_like_sample_id` so any builder can check its own
output. Commit `762951a`.

### Step 2 — experimental-unit audit *(CPU, ~30 min)*

`endcell/analysis/experimental_unit_audit.py`. This is a **discovery** step and its output changes
the cost of everything after it. It must answer:

- how many cell lines nest inside one treatment sample (sets the correct clustering unit);
- whether plate-6/plate-14 replicate treatments exist and how many (decides whether independent-well
  reliability is available at all, which Workstream H depends on);
- how many samples are combination treatments (currently analysed as single-drug);
- what fraction of conditions have a recoverable molar dose;
- whether any sample crosses the existing holdout.

**Why it comes before the rebuild:** if replicate wells are plentiful, the reliability story changes
from split-half precision to genuine biological replication and several claims get stronger. If they
are absent, I stop proposing analyses that need them.

### Step 3 — patch the builders for dose and sample *(engineering, ~half a day)*

`build_residual_targets.py`, `build_embeddings.py`, `build_ot_targets.py`: write `sample_id` to
`sample_id`, resolve `dose_molar` through `tahoe_design`, keep the raw string, and refuse to write a
dose column that fails `looks_like_sample_id`. No cache rebuild unless Step 2 says the existing cache
cannot be enriched in place.

### Step 4 — split-first, plate-scoped, drug-weighted residual targets *(engineering, ~1 day)*

The core repair. Restructure `build_residual_targets.run()` into inventory → split → fit → transform,
where the split is assigned from **metadata only** and everything outcome-derived happens after.
Three separate defects get fixed in the same pass:

1. **Leakage** — the generic and the reliability filter see training conditions only.
2. **Plate scope** — the generic becomes plate-scoped, not cell-line-scoped. We measured that the
   cell-line-scoped choice leaves same-plate structure at +0.478 against −0.018 at plate scope, and
   then trained on targets built the contaminated way.
3. **Drug weighting** — the generic is a mean over *drugs*, each drug's doses/plates averaged first.
   Currently it is a mean over conditions, so a five-dose drug carries five times the weight.

Plus `tests/test_split_before_fit.py`: poison every held-out expression vector with noise, rebuild,
and assert the training JSONL and fitted arrays are byte-identical. That test is the release gate for
the generalisation chapter and it is the single most valuable thing in this document.

**Decision gate at the end of this step.** If the rebuild is clean and the retrain can be queued,
proceed. If not, withdraw the generalisation claim and skip Steps 5–6 entirely.

### Step 5 — one clean retrain *(GPU, 5–20 h queue)*

One arm, one seed, same recipe, `--seed` wired through so it is reproducible. Not three arms, not a
seed sweep. The claim it supports is checkpoint-specific and will be worded that way.

### Step 6 — evaluation fixes and rescore *(engineering ~half a day; GPU 3–6 h)*

Patch `residual_eval.py`:

- scramble partners drawn **within plate** and required to be a **different drug** (currently a
  partner can be the same drug at another dose, which is why the `near` stratum is contaminated and
  the monotone-gradient argument is weaker than presented);
- gallery negatives exclude same-drug siblings;
- `drug_lookup` / `moa_lookup` fit from **training conditions only**, with any full-cache version
  reported separately and labelled an oracle;
- generation seeded per (condition, arm, replicate) so a rerun is byte-identical;
- three generation seeds if the queue allows, one if not.

### Step 7 — recompute what the dose fix invalidates *(CPU, ~2 h)*

The transfer coefficient's dose arm is void — its "different dose" comparison keyed on sample IDs.
Rerun with real molar dose, and replace the norm-ratio scope calculation with energy shares that sum
to one including the cross term. `shared/inference.py` gets written here because both this and Step 8
need crossed/clustered resampling and TOST.

### Step 8 — harden the metric-calibration chapter *(CPU, ~2 h)*

This is the thesis's strongest contribution and the cheapest to make bulletproof: tie-aware NIR,
average ranks for expression ties, recompute the DRF ratio inside each resample rather than treating
numerator and denominator as fixed, and Holm across the five metrics so "only NIR is calibrated"
becomes a simultaneous statement.

### Step 9 — figures and numbers from artifacts *(engineering, ~1 day)*

`aggregate_workspace_probe.py` (canonical Q21 family, one global BH, missing layers marked),
regenerate `fig_mechanism` from it, and `build_thesis_assets.py` to emit numeric macros from result
JSON so no figure is copied by hand. A claim → artifact table in the appendix rather than a full
artifact-lock system.

### Step 10 — thesis rewrite *(5–7 writing days)*

Methods becomes the most explicit chapter: treatment-well design, split-before-fit order, exact
generic definition and source set, partner and gallery rules, training-only baselines, resampling
units for every interval. Introduction reframed to the recovered post-treatment population.
Survivor selection, 946-gene scope and the 24-hour endpoint promoted to first-class limitations.
Results reordered so the metric audit leads.

---

## What I am cutting, and why

| Cut | Plan ref | Why |
|---|---|---|
| Full artifact lock, run manifests, claim registry schema, CI workflow | M1–M7 | Weeks of infrastructure that changes no number. Replaced by a claim→artifact table and generated macros, which captures the part a viva actually tests. |
| Token/gradient diagnostic | G | Only needed to *retain* a gradient claim. We downgraded it to a hypothesis, which removes the requirement. Reinstate only if a reviewer demands the number back. |
| General power-simulation framework | K5 | Replaced by targeted power statements for the two null claims that need them. A framework is not required to say what effect size a specific test could detect. |
| `residual_metrics.py` unification | E1 | Patching two evaluators is lower-risk than merging them late. Unification is good hygiene and a bad idea in week five. |
| Common frozen checkpoint, arm matching | D2–D3 | These exist to support a *causal* target-encoding claim we are no longer making. Seeding (D1) is kept because it is cheap and makes the retrain reproducible. |
| κ and channel-gate reruns | I2–I3 | The plan's own stop rule: coverage and power are inadequate, so rerunning cannot change the conclusion. |
| Panel sensitivity | J4 | Genuinely interesting, not required if every claim is panel-scoped in wording — which is cheaper and equally honest. |
| Survivor-selection quantitative audit | J3 | The *wording* is required and costs nothing. The audit needs uncapped metadata we may not have, and its absence is a stated limitation rather than a broken claim. |
| OT/consensus arms rebuilt | — | They support the ε-ladder, which is a negative result and unaffected by the residual repairs. |

---

## Honest risks

**The retrain is the whole schedule.** Steps 5–6 are the only queue-bound items and everything
downstream of them waits. If the queue is slow, cut to one generation seed and say so.

**Step 2 may reveal the cache must be rebuilt.** If `sample_id` cannot be recovered by enrichment,
Step 3 becomes a full cache rebuild and costs an extra day plus CPU. This is the largest cost
uncertainty in the plan.

**The corrected numbers may move in either direction.** The leak and plate structure inflate the
residual result; the oracle lookup and same-drug scramble partners deflate it. "Everything shrinks"
is not the safe prediction, and no wording should be drafted before the numbers land.

**Even a clean retrain does not make the residual arm a repair of the original task.** It remains a
per-condition retrieval task — the same target string for every control cell in a condition. Only the
frame reconciliation speaks to the original estimand, and if that cannot be done properly the honest
conclusion is exploratory evidence on a different question.

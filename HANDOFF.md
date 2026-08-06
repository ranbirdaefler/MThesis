# Handoff — MSc thesis, final hardening pass

**Repo:** `C:\Users\avsd8\OneDrive\Desktop\tahoe` · **Branch:** `main` · **Last commit:** `580b372`
**Build:** `thesis/thesis.pdf`, 117 pp, 0 errors, 0 undefined refs or citations.

You are picking up a thesis that is **scientifically finished and verified** but still has ~9
examiner-level writing defects open. Every number in it reconciles to a stored artifact. Nothing
below asks you to re-run an experiment — both cluster jobs are done and there is nothing queued.

---

## 1. Read this first: how work goes wrong here

Four expensive lessons. They are not general advice; each one cost real rework on this document.

**1.1 — A check whose discrimination you have not measured is worthless in either direction.**
A point-estimate reconciler was built that reported "0 problems." Testing it showed **100% of random
numbers in [0,1] "match" an artifact**, because 90 artifacts × rounding gives ~340k values. It could
not fail. It was discarded. The *interval* reconciler is sound only because bounds are pairs — its
false-match rate on fabricated intervals is **0.22%**, measured. Before you trust any checker you
write, feed it known-bad input and confirm it fails.

**1.2 — Fixing findings one at a time creates contradictions.**
88 edits, each individually correct, were applied mechanically and produced **88 new examiner-level
findings at the seams**. The document says the same claim in several places; patching one site moves
a contradiction rather than removing it. The fix that worked: cluster findings by the *claim in
dispute*, decide once from the artifacts, then fix **every site that claim touches in the same pass**.

**1.3 — Edits drafted against an older file state must be re-read before applying.**
One proposed edit would have deleted the literature review's *only* remaining scoping disclaimer, on
the (once-true) grounds that there were five redundant ones. Four had already been removed. Always
re-verify the anchor *and the rationale* against the current text.

**1.4 — Never resolve a contradiction by strengthening a claim.**
This document has been narrowed four times. When two passages disagree, the **more hedged one is
almost always correct** and the strong one is the regression. A guard pass that enforced this caught
a mass reintroduction of overclaims, including a synthetic planted-world number (`calib_selftest.json`,
`"selftest": true`) being quoted as a real measurement.

---

## 2. Hard constraints

- **HPC:** never run on the login node. Everything through `sbatch`/`srun`. (User's rule, verbatim:
  *"its the one rule they have on the HPC we never run anything on the login node"*.)
- **Never destroy cluster data.** Building on top of it is fine.
- **Do not touch `thesis/main_v4_annotated.pdf`** — hand annotations, file-locked.
- **Do not edit** `Sections/Methods*.tex` or `Sections/Results-and-Analysis*.tex`. They are
  superseded drafts **not in the build**. Confirm what is built via `thesis/main_v4.tex`.
  Flagging a problem in them is a false positive.
- `coda_errata/`, `CLAUDE.md`, `AUDIT_HANDOVER.md` and `thesis/thesis_committee_assessment.md` are
  gitignored on purpose (private working documents).

---

## 3. Ground truth — do not let these drift

Canonical summary: `RESULTS_cluster/CANONICAL_NUMBERS.json` (generated, not hand-maintained).

| Quantity | Value | Source |
|---|---|---|
| Zero-information baseline on DE-Δr | 0.999913 [0.999908, 0.999919] | `eval_endcell_linear.json` |
| DRF, expression frame across plates | +0.635 [+0.604, +0.663], 25 clusters | `calibration_v4.json` |
| DRF, within plate | +0.545 [+0.516, +0.569], 27 clusters, 44 groups | `calibration_v5_sameplate_wide.json` |
| DRF, residual frame | +0.704 [+0.667, +0.736], 42 cell lines | `residual_frame_drf.json` |
| Expression frame, tier 2 (n=606) | model 0.498, reference 0.576, control-copy 0.504 | `nir_sameplate.json` |
| Identifiable stratum (n=204) | model 0.768, control-copy 0.766, contrast +0.002 [−0.026, +0.028] | `thesis/figs/fig-difficulty.json` |
| Train gap vs neutral comparator | +0.0898 [+0.0291, +0.1506] clustered on **drug** | `re_v3.json` |
| `unseen_combo` gap | +0.0332; [+0.0069,+0.0594] line×well, **[−0.0031,+0.0695] drug×well** | `re_v3.json` |
| Lookup, common support n=1192 | 0.913 vs model 0.549, reference 0.857 | `lookup_by_split.json` |
| Lookup by split | 0.980 train / 0.890 held-out — the gap is same-well support | `lookup_by_split.json` |
| All 1394 conditions | model 0.537, reference 0.854 | `re_v3.json` |
| Transfer coefficient | T = 0.553 [0.514, 0.593] | `vardecomp_matched.json` |
| Channel gate, plate-matched | target +0.1351 [+0.0745,+0.1958]; moa +0.0805 [+0.0279,+0.1332]; chem +0.0261 [+0.0035,+0.0487] | `channel_gate_v4.json` |
| Channel gate, **drug axis** | target +0.1351 [+0.0287,+0.2416] — misses the 0.03 margin | `channel_gate_v4_byaxis.json` |
| Held-out targets only | target +0.1197 (2 drugs); moa +0.0439 **spans zero**; chem +0.0378 **spans zero** | `channel_gate_heldout_targets.json` |
| Identity test | +0.0197 [+0.0063,+0.0335], q=0.026 — **at α=0.5 only** | `probe_family_bh.json` |
| Budget-matched controls | single-cell −0.0128 (flat); **OT +0.0292 [+0.0144,+0.0441] (not flat)** | `ctrl1400_summary.json` |

**Two traps in the artifacts themselves:**
- `calibration_*.json` key `n_drugs` is actually a **row count** (one row per drug per group).
  1,820 "drugs" is impossible on an atlas of ~1,100 compounds. `n_rows` now carries the true name.
- Populations are easy to mix: `0.913`/`0.549`/`0.857` belong to n=1192; `0.537`/`0.854` to n=1394.
  Never put figures from both in one sentence.

---

## 4. Load-bearing hedges — do not smooth these away

Each was added after a review round. Removing one is a regression, not an edit.

1. **Mechanism and chemistry are INCONCLUSIVE, not closed.** Failing a relevance margin is not
   equivalence.
2. **Protein target clears the 0.03 margin on the cell-line axis only.** On the drug axis it clears
   only under the count-matched null — which the chapter itself calls the weaker control. It rests
   on **18 annotated drugs, 2 of them held out**.
3. **The activation-intervention result is the chapter's weakest element.** Its multiplicity survival
   is rung-dependent: α=0.5 survives (q=0.026), α=1 nothing survives, α=2 a *different arm* survives.
   Nothing downstream may rest on it alone.
4. **Re-encoding is first in SEQUENCE, not uniquely responsible** — the optimal-transport control also
   clears zero at matched budget.
5. **1 − T is NOT a variance share or a drug×cell-line interaction.** It mixes cell line, dose, well
   and estimator.
6. **Transfer to held-out combinations is NOT established** (spans zero on the drug axis).
7. **"Not established" ≠ "shown to be zero".** The difference is load-bearing in several places.
8. **The plate-scoped generic removes plate structure in expectation only**; what survives in an
   individual residual was never measured.
9. **`unseen_combo` is NOT Leave-Pairs-Out.** The chapter uses that name only for the estimand it
   *rejects*. Our split withholds whole treatment wells, which is strictly stronger.
10. **`weighted_r2` carries no directional claim** — its DRF sign tracks cell count. "All four rivals
    inverted" is wrong; three are.

---

## 5. The verification toolchain

All committed under `tools_verification/`. Run them from the repo root.


| Script | Checks | Expected |
|---|---|---|
| `tools_verification/audit_runC.py` | superseded values absent; retracted claims only stated to be retracted (negation-aware); canonical values present; no author notes in source | `RESULT: CLEAN` |
| `tools_verification/reconcile_ci.py` | every interval in the build matches an artifact | `OK: every interval...` |
| `tools_verification/check_figs.py` | each figure caption's numerals appear in its own `thesis/figs/*.json` | 4/4 `clean` |
| `tools_verification/check_structure.py` | environment balance, empty claim boxes, doubled words, truncated sentences | `structurally clean` |
| `tools_verification/check_dup_prose.py` | duplicated prose shingles, with line provenance | `0 in the same region` |
| `tools_verification/check_selfref.py` | a passage `\ref`-ing the section it sits in (covers `figs/*.tex` too) | `no self-references` |
| `endcell\analysis\artifact_manifest.py --scan RESULTS_cluster --strict` | every thesis-quoted result has a local artifact | `exit=0` |

**Run all of them after every batch of edits.** Build first:

```powershell
cd C:\Users\avsd8\OneDrive\Desktop\tahoe\thesis
latexmk -pdf -jobname=thesis -interaction=nonstopmode main_v4.tex
```

> Note: `thesis/build.sh` is a bash script; in PowerShell call `latexmk` directly as above.

**Applying edits safely.** `tools_verification/apply_edits.py` takes
`[{"file","old_string","new_string","why"}]`, asserts each `old_string` matches **exactly once**, and
writes a file only after its whole edit list resolves — so a file is never left half-edited. Use
`--dry-run` first. Caveat learned the hard way: a dry run *after* a real run makes applied edits look
like failures; distinguish by whether the **new** string is present.

---

## 6. What is open

`OPEN_FINDINGS.json` (repo root) has all nine with full text, quotes and suggested fixes.

| # | Where | Problem |
|---|---|---|
| 7 | `Literature-review.tex:250-255` | Miller reconciliation emphasis: the "divergence" promoted to the headline (rank-based family) is not a divergence from Miller — their only endorsed rank-based metric is NIR, which *survived*. The real divergence (weighted R²-Δ, in their calibrated set, negative here) is demoted to an aside. Broken pronoun at :252 leaves the surviving metric grammatically inside the set of failures. |
| 15 | `Investigation-v4.tex:456-458` | Wrong reason for a right conclusion: `revert_center` predicts every gene at rank *G*/2, so its shift is *G*/2 − x, **not zero** — it is `control_copy` whose shift is identically zero. The real mechanism is that the shift is an exact affine function of the control, so nothing survives partialling. |
| 19 | `Investigation-v4.tex:1049` | Claim box asserts a quantity its own section never derives (the 0.883 context-removed figure lives two subsections later, no forward pointer), and "instead" has no antecedent. |
| 20 | `Investigation-v4.tex:1876-1878` | The three stratum gaps (+0.0351/+0.0716/+0.1429) are each measured against **their own** stratum's partner; the sentence reads as if against a common one. |
| 21 | `Investigation-v4.tex:1749` (`tab:transfer`) | The −0.017 entry is `diff_drug_same_plate` from `vardecomp_matched.json` — the control this same section spends a paragraph **voiding**. |
| 23 | `Investigation-v4.tex:1353-1355` | Grouping key omits the drug: the builder iterates `(cell_line, plate, drug, dose)`, the text implies `(cell_line, plate)`. |
| 26 | `Investigation-v4.tex:~1906` | Opens claiming only one of three arms passed all three diagnostics, then reports the third as passing for **all three**. Survivor of the pre-withdrawal framing. |
| 27 | `Investigation-v4.tex:~2748` | The chapter-summary table is captioned as the complete arm ledger but omits four residual-frame arms the chapter's own ladder prints: `moa_lookup` 0.552, `control_copy` 0.506, `random` 0.501, scramble-`near` 0.550. |
| 32 | `Investigation-v4.tex:991-992` + `Appendix.tex:91-133` | The OT rehabilitation reached Claim 6 and two tables but may not have reached the appendix that disqualifies the swap-distance sweep. **Verify before acting** — Claim 2's third limb already carries its hedge, so part of this finding is stale. |

Below these: **35 "reader-would-stumble"** and **33 "cosmetic"** findings in the same source file
(`tasks\w9m5ys8w6.output`, keys `stumble` / `cosmetic`). None touches a result.

**Still outstanding and not in that list:** the thesis has **no title page and no abstract**
(`thesis/main_v4.tex` goes Acknowledgements → TOC → Introduction), and `Acknowledgements.tex` still
has `\todo{write at the end}`. Those are hard submission blockers. The abstract should be written
**last**, so it reflects the final framing.

---

## 7. The argument, in case you need to judge a claim

1. The field's standard metric (DE-Δr) is **saturated by a zero-information predictor** (0.999913).
   A meta-metric (DRF) grades metrics; of five, only NIR is positive.
2. On the repaired metric the model is **drug-blind in the expression frame** — 0.498 against a 0.576
   within-well precision reference, with a control-copy at 0.504. On the identifiable stratum, model
   0.768 vs control-copy **0.766**.
3. Drug identity **is** in the activations; the readout barely consults it. (Weakest element — see
   hedge 3.)
4. Re-encoding the target as the drug-specific residual **produces prompt sensitivity** — but not
   uniquely (hedge 4), and sensitivity is **not fidelity**.
5. A training-only per-drug **lookup beats the model** (0.913 vs 0.549). Arguably the most portable
   finding; it is in the Introduction's contribution list, and should stay there.
6. For unseen drugs, **protein target is the only channel worth pursuing**, on 18 drugs, on one
   clustering axis (hedge 2).

---

## 8. Suggested order of work

1. Fix findings **15, 21, 26, 27** first — each is a flat factual error, checkable against an artifact.
2. Then **7, 19, 20, 23** — wording and pointer defects.
3. **Verify 32 before acting**; part of it is already fixed.
4. Re-run all seven checks; rebuild.
5. Then a fresh end-to-end read for seams, since fixes create them (lesson 1.2).
6. Title page + abstract **last**.

Commit in small batches with a message that says what was wrong and what the artifact actually says.
The git history is a defensible record of every correction — keep it that way.

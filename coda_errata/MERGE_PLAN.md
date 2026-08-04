# Merge plan — v4 structure, B's grafts

**Decision, made:** keep the **merged v4 Investigation chapter**. Take three things from the Opus 5.6
split version (B) and nothing else structural.

Apply this **after** the rewrite comes back. The rewrite is working from `Investigation-v4.tex`, so
its output is the base; these are grafts onto that output, not onto the current file.

---

## What to take from B, and why

### Graft 1 — promote the comparator diagnosis to its own section, and put it BEFORE the magnitudes

This is the single biggest structural win available. v4 states three magnitudes at §3.10 and only
explains 110 lines later that the comparator they are measured against is not a null. B makes it a
section and leads with it.

Take B's **`\subsection{The active comparator is the hinge}`**
(`coda_errata/Results-and-Analysis_rewritten.tex:532–637`) essentially whole. Three sentences in it
are better than anything in v4:

> The substituted-prompt arm was intended as a null, but its own score changes with the response of
> the partner it names.

> This diagnosis changes the residual result rather than removing it.

> The last interval spans zero after the small-cluster correction, so the unseen-drug row
> corroborates but does not establish comparator non-neutrality. The direct $0.572/0.497/0.403$
> ordering establishes it.

That third one is the exact point v4 had wrong until yesterday and is worth importing verbatim.

**Also take B's `tab:generalisation`** (`:560–605`). It is better than v4's equivalent in three ways:
it reports `n (lines; wells)` rather than just lines, so the reader can see the 25-well arm that
drives the t correction; it carries `df` per row; and its caption states **why the slope intervals
are percentile intervals rather than Student-t ones** — a distinction v4 never draws and an examiner
will ask about.

### Graft 2 — the transfer-coefficient identification caveat

**Already applied to v4** (commit `a218368`), derived from B's version. Recorded here so the rewrite
does not undo it. If the returned rewrite has re-simplified this, restore it.

B's contribution was to notice that the run's own arguments — `same_dose_only=false`,
`loo_generic=false`, `repro_thr=0.2` — mean `T` combines cell line, dose, well and estimator, so
`1 − T` is **not** a drug×cell-line interaction share. Their demotion to "a descriptive sensitivity
analysis rather than a load-bearing estimate" is correct and adopted.

**Do NOT take B's number.** B quotes `T = 0.549 [0.507, 0.590]` from `vardecomp_plate.json`, the
**29 July pre-correction** run: no dose resolution, no multiway intervals, no structure-matched
control. Use **`T = 0.553`, multiway `[0.514, 0.593]`** from `vardecomp_matched.json`, with the
caveat.

### Graft 3 — a section for the prompt-field attribution

v4 has no home for this at all. Take B's **`\subsection{Prompt-field attribution remains unresolved}`**
(`:744–796`) for its framing and its discipline — it states plainly that the corrected evaluation
contains no single-span arms and therefore cannot attribute the slopes to either span, and it marks
the older estimates "descriptive only, non-comparable frame" rather than folding them into the
conclusion.

**But rebuild both its tables.**

- **Panel A (channel gate) is stale in B and correct in v4.** B has n = 894 / 1053 / 4084 and gaps
  +0.0844 / +0.0780 / +0.0137 — the superseded count-matched-only run. v4 has 877 / 1033 / 4074 and
  the plate-matched gaps +0.0973 / +0.0749 / +0.0162. **Use v4's numbers in B's table.**
- **Panel B is about to be superseded.** See below.

---

## The thing this section is actually about, and what Job 2 will do to it

The field decomposition **has been run once already**. `FINDINGS.md:598–602`:

| swap | gap | n | verdict as recorded |
|---|---|---|---|
| drug name only (mechanism kept) | +0.0809 [+0.0592, +0.1010] | 570 | read |
| mechanism only (name kept) | +0.0091 [−0.0174, +0.0383] | 325 | not detected |
| both | +0.1024 [+0.0827, +0.1236] | 570 | read |

FINDINGS concludes the model reads the **name** and not the **mechanism**, with a power argument (the
mechanism arm could have detected an effect the size of the name's and saw +0.009). It is a strong
result and it closed the channel-conditioning arm before it was built.

**Two reasons it cannot be quoted as it stands.** It is measured against the `opposite` stratum,
which is not a null — so both gaps carry the partner-push term. And it comes from the run whose
comparator swing is documented immediately below it in the same file. B was right to fence it off.

**Job 2 is regenerating exactly these arms against the neutral comparator, in the corrected truth
frame.** So write the section now with B's framing and leave the table to be filled. The interesting
possibility to be ready for: if the name/mechanism asymmetry survives the neutral comparator, the
thesis can say something considerably sharper than "responds to the combined drug-and-mechanism
prompt" — and if it does not survive, that is itself a finding about how much the old comparator was
carrying.

Note the `n = 325` on the mechanism arm is not a defect: when a swap partner happens to share the
target's mechanism, a mechanism-only swap produces a byte-identical prompt and a gap of exactly zero
*by construction*. Those conditions are dropped rather than scored. Say this in the caption — an
examiner will otherwise read the unequal n as cherry-picking.

---

## What NOT to take from B

- **The Methods/Results split.** The merged arc is better and you have already decided this.
- **`T = 0.549`.** Stale artifact. See Graft 2.
- **Panel A's channel-gate numbers.** Stale. v4 is correct.
- **B's mechanistic-probe treatment.** B's Methods covers probing in ~45 lines
  (`Methods_rewritten.tex:379–424`); v4 devotes a full arc to decodability → ablation → substitution.
  Your instinct was right: keep v4 here. (The comparison workflow is testing this specifically; if it
  disagrees I will say so.)

---

## Numbers to use, single authoritative list

| quantity | value | source |
|---|---|---|
| gap vs neutral, train | +0.1457 [+0.0813, +0.2101], df 34 | `scramble_stratum_audit_v3.json` |
| gap vs neutral, held-out well | +0.0449 [−0.0095, +0.0994], df 30 | same |
| gap vs neutral, unseen drug | +0.0097 [−0.0694, +0.0888], df 24 | same |
| gap vs opposite, unseen drug | +0.0862 [−0.0028, +0.1753] | same — spans zero, corroborates only |
| comparator strata NIR | 0.572 / 0.497 / 0.403 | same — this is what establishes non-neutrality |
| true-prompt NIR | 0.635 / 0.545 / 0.512 | same |
| slopes | +0.390 / +0.374 / +0.238 | same — percentile, not t |
| exposure difference, pooled | +0.1008 [+0.0161, +0.1854], **p = 0.0212** (t, df 30) | same |
| restricted to 20 shared drugs | +0.0420 [−0.0815, +0.1655], **p = 0.4851** (t, df 19) | same |
| model − lookup | −0.3559 [−0.3988, −0.3130], n = 449 | `re_repaired.json` (**two-way**, not the one-way `ci`) |
| coverage, common support | 18.6% vs 96.3% | same |
| transfer coefficient | **T = 0.553**, multiway [0.514, 0.593], **descriptive only** | `vardecomp_matched.json` |
| dose transfer | 0.707 [0.632, 0.782] — clean, holds drug and line fixed | same |
| energy identity | 0.621 + 0.390 − 0.011 = 1.000 | same |
| DRF | +0.635 [+0.618, +0.652], Holm p = 0.0025 | `calibration.json` (3 Aug) |
| channel gate | 877 / 1033 / 4074; +0.0973 / +0.0749 / +0.0162 | plate-matched register |

Everything derived from `re_repaired.json` is replaced when Job 2 lands.

---

## Execution risks

1. **The rewrite may have been sent an older `Investigation-v4.tex`.** Nineteen number corrections
   plus the interaction caveat and the t p-values landed today. Diff the returned file against
   `a218368` before accepting it, and re-apply anything lost. Check specifically: `0.553` not
   `0.558`; `p = 0.0212` not `0.0150`; channel-gate n `877/1033/4074`; the `\ci` bounds in
   `tab:generalisation`.
2. **Duplicate labels.** B uses `\label{sec:res-generalisation}` for its comparator section and
   `\label{tab:generalisation}`; v4 uses the same names for different objects. Renaming is required
   or the cross-references silently point at the wrong float.
3. **B's field section carries two `\label`s on one table** (`tab:channels` and `tab:fielddecomp`).
   Keep one.
4. **Both versions will define the comparator strata.** After grafting, the definition will exist
   twice. Keep the one in the new hinge section and cut the other.
5. **B writes `\resizebox{\textwidth}{!}{...}` around tables.** v4 does not. Mixing will produce
   visibly inconsistent table type sizes across the chapter.

---

# The comparison's verdict (7 agents, section by section)

## Keep the merge — and the argument is empirical, not preference

**B's own text documents the split failing five times in 1,476 lines:**

1. **NIR is defined twice, in incompatible notations.** `Methods_rewritten.tex:183–193` uses
   $(\hat y, y_i, y_j)$ over $J$; `Results-and-Analysis_rewritten.tex:103–112` uses $(p, t, o_j)$ over
   $m$ rivals. B could not trust the reader to carry the definition 250 lines, so it paid twelve
   lines to restate it — and gave one quantity two symbol sets.
2. **A forward reference inside a verdict.** B's Results:50 tabulates the DRF row and declares NIR the
   only calibrated metric; NIR is not defined in that chapter until :103.
3. **A qualification drifted from its claim.** Methods:292 calls the model-minus-`orth` gap
   "corroborating" with NIR-against-0.50 "primary"; Results:540 promotes the same gap to "the neutral
   active comparator for a magnitude contrast". Two positions, one document.
4. **A rationale stranded from its use.** The baseline-fairness argument (why full-cache lookups are
   oracles) is at Methods:588–596; the number it licenses is at Results:637–729.
5. **An undefined term inside a headline sentence.** Results:119 reports "the leak-immune gap"; the
   adjective appears nowhere else in either file, and the construction sits at Methods:278–279.

**The mitigation for "where are your methods".** A already carries **both label families at every
anchor** — `\label{sec:methods-metrics}` sits beside `\label{sec:res-metrics}`, and likewise for all
nine pairs — so every cross-reference from Introduction, Literature review, Limitations, Conclusions
and Appendix resolves unchanged. **Do not rename or renumber these.** Then add a one-page **Methods
index** to the Appendix: instrument → where defined → page, built from the nine `sec:methods-*`
labels. Half a day, and it is the whole answer to the objection.

## Must survive from B (beyond the three grafts above)

- **The `orth` arm's true status** — outcome-adaptive, empirically near chance, not a prespecified
  null. *(A said "chance by construction" in its summary table; fixed in `c1b3460`.)*
- **The two-step token attribution** (B Results:428–431): baseline cell state does most of the work,
  not the generic subtraction. A shows the rows and draws the wrong conclusion — **check this.**
- **The assumption list for the disattenuated cosine** (B Methods:632–639). A asserts an identity and
  never states what it needs.
- **Composition-sensitivity of the exposure comparison** (B Results:621–629): 68 / 29 / 20 drugs,
  22.5% and 64.4% of observations. A says the arms are unmatched; only B shows it.
- **Support bookkeeping on the lookup** (B Results:679–689): 570 vs 449, paired on the shared 449.
- **The dose-siblings exclusion** (B Methods:212–213) — exists only in B and is required for the
  residual gallery to be correct.
- **Instrument scope limits** (B Methods:279–280, 313–316, 334–338): what each control *cannot* claim.

## Must survive from A

**The mechanistic-probe arc (A 657–914) — non-negotiable, and for a sharper reason than depth.**
B deleted the two things that make it an argument:

- **The cell-line calibration of the ablation.** It is the only place in either version that tells the
  reader whether "3.6–18.6×" is large. B never reports the cell-line ablation KL at all, so its phrase
  "positive-control intervention" has no number attached anywhere in the document.
- **The third failed substitution design** — *"a random direction inside the slab is off the
  class-mean manifold: it inflates the logsumexp normaliser (measured +2.75 nats)"* — which is the
  sole justification for the contrast score **B itself adopts**.

Also from A: `fig:timeline` and the `claim` environment (B has no visual map and no verdict boxes);
the scramble design block (A 523–548) with its rejected alternatives; and the forward guard that the
residual repair *"keeps rank throughout and changes only what is being ranked"*.

**Your instinct on the probe was right.**

# Rewrite prompt — the merged Investigation chapter, revision 5

*Supersedes `REWRITE_PROMPT_METHODS.md` and `REWRITE_PROMPT_RESULTS.md`, which assumed a
Methods/Results split. The chapter is now merged. Paste this whole file and attach
`thesis/Sections/Investigation-v4.tex`.*

---

## What you are revising

Chapter 3 of an MSc thesis (Bocconi, computational biology), a single merged narrative called **The
investigation** — 1673 lines, 14 subsections, ~54 pages. It replaced a conventional Methods/Results
split and the merge was the right call. **You are not being asked to restructure it again.** You are
being asked to make it readable, self-contained, and correct.

The previous revision fixed the science. It did not fix the prose, and it left the chapter assuming
a reader who already knows this project.

## The thesis in one paragraph — keep this argument intact

> The field's standard perturbation metrics can be maximised by a prediction that ignores the drug
> entirely, so the first result is that the instrument was wrong. Under a metric whose chance value
> is fixed by construction, the model is drug-blind: it produces the same answer whether told the
> true drug or a different one. The blindness is traced to the target — the drug-specific part of a
> response is small beside a generic stress programme shared across drugs, and the tokens the model
> predicts are dominated by the generic part. Re-encoding the target as the drug-specific residual
> makes the model demonstrably sensitive to the drug information in its prompt. That sensitivity
> does not become reliable prediction on held-out treatment wells, and a simple training-only lookup
> table substantially outperforms the model. The result is a dissociation between prompt sensitivity
> and biological fidelity.

---

# Part 1 — the three things that most damage readability

These are diagnosed, not guessed. Line numbers refer to the file as given to you.

## 1.1 The chapter reports its apparatus before its findings

This is structural and it is the single largest cause of the "rambling" feeling. It also breaks a
promise the chapter makes about itself at lines 30–33: *"each part below poses a question, defines
the instrument built to answer it … reports what the instrument showed."* Four sections break it:

- **§3.3 (395–577).** Line 401 promises "four instruments that fail in different ways". The verdict —
  the model at `0.498` against a `0.576` ceiling — does not arrive until **line 517**. That is 116
  lines of apparatus first. Worse, **the reader cannot count to four**: three instruments are named
  (434, 466, 470) but eight measurements are presented, and `tab:blind` (486) lists five rows in
  three different metrics with no column saying which is which.
- **§3.8 (914–997) is 84 lines and reports no result at all.** It defines the residual frame and
  stops; the payoff is 250 lines later.
- **§3.9 opens with the word "Before"** — *"Before training anything on the residual, it is worth
  knowing how much drug-specific signal exists"* — and is placed **after** §3.8, which already built
  the residual. The section announces that it should have come first.
- **§3.10 quotes three magnitudes and then withdraws them.** `+0.0351 / +0.0716 / +0.1429` at
  1174–1176, a forward reference at 1216, and only at 1288 the explanation that the comparator they
  are measured against is not a null. A caption at 1193 adds that they are "superseded". The reader
  is asked to believe, unbelieve, then partially re-believe.

**The fix is a shape, not sentences: verdict first, then the instruments walked as objections
closed.** The chapter already contains that shape three times, and those are the best passages in it
— **524–569** (three italic-question objections, each answered by a measurement), **626–659** (four
instrument steps, each justified by the failure it prevents), **1518–1523** (the plate-matched null:
"two differences where the estimand allows one"). **Use those three passages as the target register
for the whole chapter.**

## 1.2 The definitions that matter most are inside floats the text never points at

The chapter has **22 floats and 4 in-text references**. Three load-bearing definitions live in the
unreferenced ones:

- **`fig-nir.tex`** (input at 316, never `\ref`ed) contains **the only definition in the thesis of the
  comparison set `J`** — which conditions a prediction is ranked against. The body's NIR equation
  (289–291) sums over `j ∈ J`, divides by `|J|`, and never declares it. The chapter's most repeated
  claim, *"chance is 0.50 by construction"* (296), is a statement about `J` and nothing else. The
  difference between `J` spanning plates and `J` within plate moves a baseline from `0.399` to
  `0.161` (430). **This must be in the body.**
- **`fig-frame.tex`** (input at 996, never `\ref`ed) is the clearest explanation of the residual
  construction anywhere in the thesis — **and it illustrates the rejected scope.** Its body and
  caption both say the generic is "the mean over the other drugs in the same cell line", placed
  immediately after thirteen lines (940–952) establishing that **the scope is the plate, not the cell
  line**. Fix the figure or move it.
- **`control-copy`** carries the chapter's strongest single sentence (`0.768` model vs `0.766`
  control-copy, 522). It appears as a bare table row at 417 and is glossed only at **543, inside a
  caption**, 126 lines later.

## 1.3 Four words and six symbols carry incompatible meanings

| word | distinct senses in this chapter |
|---|---|
| **ceiling** | **nine different numbers**: 0.76, 0.625/0.575, 0.67–0.83, 0.576, 0.61/0.76/0.85, ≥0.8, 0.742, 0.958, 0.962/0.955/0.956 — one noun, no taxonomy anywhere. At line 113, 46.2% of conditions are said to clear a bar *higher than either ceiling the chapter reports*. |
| **arm** | a model variant · one side of a paired contrast · a baseline predictor · a table column · a single table cell. Used ~20×, defined never. **"The five training arms" (783, 807) is never enumerated**, so the denominator of the headline multiplicity correction ("26 headline tests — five arms by up to seven depths") cannot be checked. The list exists in `Appendix.tex:98–109`; nothing points at it. |
| **gate** | removal gate (642) · headroom gate (793) · "the structure gate" (1547) — **the third is never defined anywhere**. |
| **gap** | `\gap = NIR_model − NIR_scramble` (437) · a display-math similarity difference (475) · "same-drug versus different-drug gap" (564) · "clean preference gap" (784). `tab:blind` lists two of these as rows with no cue that they are different quantities in different units. |

**Symbol collisions, verified against `utilities.tex`:**

- `P` = panel size (242) · a probability, `log P(r_B | prompt_A)` (765) · a projector, `P(μ_B − μ_A)`
  (771). **The last two are seven lines apart.**
- `\resid` is defined as the letter `r` and means the drug-specific residual (921, 963, 1011). At
  753–767, `r_A` and `r_B` mean *a generated response string*. Both render as italic *r*.
- `\shift` is the letter `s`. At 749, `2⟨c_B, s⟩ = 9.179` uses `s` for the ablated slab component.
  Neither `c_B` nor that `s` is ever defined.
- `K` = the top-K gene count fixed at 50 (217) · the Sinkhorn kernel `K = exp(−C/ε)` (872), where `C`
  is also undefined.
- `ρ` is written `ρ(d, c₁)` at 1029 and defined at 1031 as `ρ(resid_A, resid_B)` — two argument
  conventions two lines apart.

**Rename or disambiguate every one of these.** A reader who has to hold two meanings for `r` while
following an argument about whether the model reads the drug will lose the argument.

## 1.4 Sentence length and emphasis, measured

| | current | target |
|---|---|---|
| median sentence | **28 words** | 18–20 |
| over 40 words | **107 (23%)** | under 8% |
| over 55 words | **32** | 0 |
| longest | **103 words** | — |
| `\emph{}` + `\textbf{}` | **204** across 466 sentences | ~one per page |
| em-dashes | **245** | roughly halve |

The failure mode is consistent: a claim, a parenthetical qualifying it, then a clause qualifying the
parenthetical. **Every caveat that needs its own clause needs its own sentence.** An emphasis mark
every 2.3 sentences emphasises nothing.

---

# Part 2 — the definitions debt

**Add a notation-and-vocabulary block early in the chapter** — after the orientation, before the first
result. Prose the reader passes through, not an appendix glossary. Every term must be defined before
first use; a term used in a caption counts as used.

**The single highest-value fix in the chapter: a ceiling taxonomy.** Almost every negative result is
stated as a distance from a ceiling, and nine numbers share the name with no sentence saying whether
they are one estimator under different comparison sets and cell budgets, or different estimators.
Say which is which, and at 1458 say why split-half rather than a biological replicate — Tahoe has one
well per treatment, so a biological replicate does not exist (line 983 already says this). Note also
that line 1036 concedes split-half precision is "optimistic", which means both the `18.6%` and `96.3%`
figures are inflated — and that concession sits 400 lines from the numbers it distorts.

Then, in order of where they should first be defined:

- **well** — one well receives one drug at one dose and holds ~49 cell lines. This single physical
  fact drives the two-way clustering, the plate-leak result and the whole-well holdout, and it is
  currently stated 337 and 1050 lines after its first consequence.
- **the comparison set `J`** — see 1.2. Move out of the figure caption.
- **plate**, **condition**, **treatment sample**, **cell line**, **pseudobulk**, **control pseudobulk**
- **split**: `train`, `unseen_combo` (an unseen *well* of a seen drug), `unseen_drug`; **quota**
- **scope** (`cell_line` vs `plate`) and what changing it does
- **the generic** — the drug-agnostic mean subtracted to form the residual. The most important object
  in the chapter and currently under-explained.
- **residual target**, **leave-one-drug-out**, **split-before-fit**, **control-copy**
- **the five training arms** — enumerate them where the multiplicity correction is claimed
- **NIR** — spell out Normalised Inverse Rank, and state why chance is 0.5 *by construction*. That
  property is what makes "the model is at chance" a statement rather than a comparison.
- **DE_dr**, **partial-DE_dr**, **DRF** (a fraction *of what*), **repro_cos**
- **two-way cluster-robust (Cameron–Gelbach–Miller)** — why crossed cell line × well needs it
- **Student-t with df = (smaller cluster count) − 1** — why the cluster count sets the reference
- **multiway / dyadic (Fafchamps–Gubert)** — why pair-valued statistics cannot use CGM
- **Holm**, **disattenuation**, **transfer coefficient**, **main effect vs interaction**
- **the scramble strata** `near` / `orth` / `opposite`, how a partner is chosen, and why the neutral
  one is the comparator

Give the prose expansion of every macro at first use: `\NIR`, `\DEdr`, `\gap`, `\Tcoef`, `\DRF`, `\ci`.

---

# Part 3 — the motivation debt

The chapter does this **perfectly four times** — 181–184 (a tempting shortcut that would have
*tightened* the intervals), 202–204 (a 600-token cap that halved a measured effect), 642–645 (the
removal gate: "otherwise the null is unfalsifiable, a measurement of nothing"), 654–658 (logit-space
scoring as a repair of a failed attempt). **Those are the template.** Everywhere else the choice is
stated and the alternative is not.

Ranked by how much credibility rests on them:

1. **Which ceiling construction, and why that one.** See Part 2. Highest value in the chapter.
2. **The comparison set and the same-plate rule — and its apparent violation.** Line 431 imposes a
   chapter-wide same-plate rule after showing plate leakage more than doubles a drug-agnostic
   baseline. Lines 994 and 1374 then score against "other drugs in the same cell line", and a cell
   line spans plates. If the plate-scoped generic already removes plate structure, **that is the
   missing sentence** and it belongs at 994. Sharpened by 1557, where the channel arms sit at NIR
   `0.360` and `0.296` — *below chance* — under the different-plate null, disposed of as an aside.
   That looks like a frame failure and needs meeting head-on.
3. **Anchoring a model comparison to `panel-τ`** (267–270) when the chapter elsewhere rules it
   inadmissible. Defend the relative-vs-absolute distinction or the claim block at 391 is a free shot.
4. **`nir_expr` over `nir_rank`** (305–314). The evidence is strong, but the choice appears to
   contradict the chapter's own framing, which invokes Miller et al. for the finding that *rank-based*
   metrics survive calibration. One sentence on what NIR's rank character refers to — ranking
   *conditions*, not genes — closes it.
5. **The 946-gene panel and the normalisation** (70–72). No failure named. The real reason is almost
   certainly the 8192-token context limit, which also connects this choice to the truncation disaster
   at 204. Say that, and say what breaks without CP10K + log. (Also fix `log1p` → `log10(1+x)`.)
6. **The mechanistic section's parameterisation** (601–659) — four unmotivated constants in one
   instrument: twelve drugs out of 351 with no selection rule (this sets the 8% chance floor and
   bounds the SVD at rank 11); ten SVD directions from twelve centred class means, i.e. essentially
   the whole span including its noisiest directions; seven layers with 9 off the even grid, justified
   *after* the list; and 60 prompts for the ablation, down from 480, supporting a null-shaped
   conclusion with no interval anywhere in `tab:workspace`. **Also: the CV folding of the 82% probe is
   unspecified** — if folds are random over prompts rather than grouped by (cell line, plate), prompts
   sharing a well land on both sides, which is exactly the failure lines 176–199 refuse to tolerate.
7. **Estimating `Tcoef` by a disattenuated cross-line cosine rather than by fitting the
   variance-components model written two equations earlier.** The justification is strong and entirely
   absent: with 286 of 287 treatments in exactly one well there is no replicate to separate `var κ`
   from `var ε`, so the decomposition is unidentified and the cosine route buys identification through
   split-half reliability. Same paragraph: **Spearman–Brown is applied to a cosine** (1031) when it is
   derived for correlations between parallel halves, and it sits in the denominator of the headline.
8. **The `≥40 treated / ≥20 control` inclusion thresholds.** They define the entire training
   population and read as inherited constants. The chapter's own ceiling-vs-cell-count curve (531:
   0.61 at 10, 0.76 at 40, 0.85 at 120) is the motivation and is not carried over.

---

# Part 4 — corrections that must land in this revision

## 4.1 Corrections ALREADY APPLIED to the file you are given — do not undo them

A verification pass checked every numeric claim in the chapter against the artifacts. Nineteen
corrections have already been made to the `.tex` you are working from. They are listed so you do not
"restore" them:

- **Every interval was recomputed with a Student-t reference** (df = smaller cluster count − 1)
  instead of a normal 1.96. Six sites: train and `unseen_combo` gaps against both comparators, the
  `unseen_drug` gap, and the training-exposure interval.
- **T = 0.553, not 0.558**, and the interaction share is **44.7%, not 44.2%** (5 sites). The
  intervals `[0.514, 0.593]` and `[40.7%, 48.6%]` were always right.
- **The power limit is +0.099**, not +0.097.
- **87 drugs**, not 90, in the scored evaluation.
- **"27.3% of tested conditions"**, not "of drugs" — the denominator is conditions, as the chapter's
  own scale table states.
- **Conditioning on cell line raises the baseline ~2.5-fold**, not "triples" (0.142/0.056 = 2.55).
- **The single-cell discrimination gap is +0.006 [+0.003, +0.008]**, not +0.002.
- **The channel-gate table's n column** was stale: 877 / 1033 / 4074, not 894 / 1053 / 4084. The
  effect sizes in those rows were already from the current run; only the counts lagged.
- **The mechanistic-probe comparison was quoting two different analyses as one.** The claim that the
  instrument reads cell line at "15–20× the drug's" is the *variance-share* ratio; the *ablation*
  reads it at 1.0–11.2×, and the drug ablation is 9–37% of the cell-line control at layers 2–9, 98%
  at layer 12, 25% at layer 16 — not a flat 3%. Both quantities are now stated, separately.
- **The shared-control reliability mean is ≈0.24**, not ≈0.6 — 0.6 was the *fraction* reproducible.
- **The purified drug slab is ~5% of cell line's variance share**, not 3% (0.03/0.6).

## 4.2 Still open — these need your judgement, not a substitution

1. **The DE_dr baseline table (lines ~243–250) mixes tiers.** Its `DE_dr` column is tier 2 and its
   partial-`DE_dr` column is tier 1. Make the whole table one tier — tier 2 matches the model row and
   the figure caption — and adjust the caption, which currently says "the two fitted baselines" where
   only two of three predictors have a defined partial-`DE_dr`.
2. **Line ~339 attributes the four rival DRF values to "the earlier unhardened calibration".** They
   are correct, but they are the *within-plate* run (655 drugs, 61 groups); the hardened NIR estimate
   beside them is the *all-plates* run (1,820 drugs, 25 cell lines). Same issue in the figure caption
   at ~373. Say which scope each number belongs to.
3. **Lines ~897–899 quote optimal-transport gaps from a random-partner scramble** while the chapter
   elsewhere uses the stratified comparator. Say which test produced them, or move them.
4. **Line ~828 asserts a layer-12 result the cited replication never measured** — that seed scored
   only layers 2 and 4. State the gap rather than implying coverage.
5. **Several numbers have no backing artifact** (lines ~279, ~819, ~947, ~1495, ~1530). Keep them,
   but do not harmonise them to a nearby value that *does* have one — flag each with
   `% AUTHOR NOTE: no artifact` so it can be regenerated or dropped.

## 4.3 Verified correct against the artifacts — do NOT "fix" these

Each of these was checked directly against the JSON and reproduces at the printed precision. Several
look like they might be stale and are not:

- **DRF `+0.635 [+0.618, +0.652]`, Holm p = 0.0025.** The artifact reads 0.6345 / [0.6178, 0.6521] /
  0.0024988. All four rivals carry Holm p = 1.0000, exactly as stated.
- **Model minus lookup, `−0.3559 [−0.3988, −0.3130]`, n = 449.** This is the **two-way** interval.
  The same artifact also contains a *narrower* one-way interval, `[−0.3863, −0.3263]`. Do not swap to
  it — the design is crossed and the one-way interval is too narrow.
- Common-support coverage 18.6% vs 96.3%; dose transfer 0.707 `[0.632, 0.782]`; energy shares
  0.621 / 0.390 / −0.011; ceilings 0.962 / 0.955 / 0.956; drug-matched premium +0.0420, p = 0.4851;
  the comparator strata 0.572 / 0.497 / 0.403; slopes +0.390 / +0.374 / +0.238.
- The four unhardened rival DRF values (−0.081, −0.231, −0.415, −0.448) are correct; only their
  *scope label* needs fixing, per 4.2 item 2.

## 4.4 Claims that must not reappear

The previous revision removed all of these. Keep them out.

1. "cross-context transfer: **yes**" → transfer is **not established**, which is also not "absent" —
   the interval reaches +0.099, a power limit rather than evidence of absence.
2. "a **memorisation** premium exists" → the pooled advantage does not survive restriction to drugs
   present in both arms (+0.0420, p = 0.4764). Say *a condition-weighted training-exposure advantage
   whose generality across drugs is unresolved*. Avoid "memorisation": it is causal, and the design
   cannot separate exposure from drug composition.
3. "the model reads the drug **name**" → name and mechanism are swapped together. Say *responds to the
   combined drug-and-mechanism prompt*.
4. "the largest absolute cosine anywhere is 0.051" → that was a maximum over two split-level *means*;
   individual conditions reach +0.4395. Report the mean **with** the spread.
5. "the model **hedges** rather than confabulates" → nothing measures abstention or confidence.
6. "51% of conditions are **biologically reproducible**" → within-well sampling precision only.
7. "roughly **six times** more discriminative" → computed across unequal supports; use the paired gap.

## 4.5 One sentence of provisionality, and no more

A corrected generation run is in flight. Every number derived from the model evaluation will move in
magnitude; none is expected to change direction. Put **one** sentence somewhere sensible saying the
evaluation numbers come from the run identified in the artifact table. Do not hedge repeatedly — a
chapter that apologises in every section reads as unreliable rather than careful.

---

# Part 5 — what to cut

**Target: −2,000 to −2,500 words from the body**, most of it recovered as appendix material, and the
space spent on Parts 2 and 3. The chapter should get shorter even though you are adding definitions
and motivation.

**Move to an appendix — engineering record, not argument:**

| lines | what | why |
|---|---|---|
| 732–777 | the three failed substitution designs | The failures earn a place, but the reader meets `r_B`, class means, the logsumexp normaliser and the projector inside the *failure* narrative and gets the definitions only afterwards. **Reverse it:** state the surviving design and what it measures, then the three failures as the constraints it satisfies, one sentence each (~12 lines). Send the algebra and the normaliser measurement to the appendix. |
| 788–805 | "the guards … are listed in full" | Keep four (TOST + margin, the headroom gate, the cluster bootstrap, the magnitude ladder). The drug-proxy screen, the `dose_float` anecdote, the injected-norm diagnostic and the three planted worlds go to an appendix. A list announced is a list the reader skims. |
| 1040–1058 | the void first run + "five controls in all" | Keep the control-design lesson (1040–1047, which is good). The five-control sentence is a 12-line sentence with three nested parentheticals — make it a small table or an appendix paragraph. |
| 139–143 | `tab:scale` caption's source-file list | Transcribed from the run record. |
| 712–721 | "what survives from the pre-revision analyses stands with them" | Defensive, and the reader does not know what the pre-revision analyses were. |
| 249–253 | `tab:de-artifact` caption's aside on an un-run GPU pass | Footnote at most. |
| 359–377 | `fig:calibration` caption, 19 lines | Contains argument that belongs in the body (the DRF scale explanation) and defensiveness that belongs nowhere (why no intervals are drawn). Split, trim to ~8 lines. |

**Delete as redundant restatement — each of these is told three or four times:**

- **703–711 restates `fig:mechanism`'s caption (689–699), which restates `tab:workspace`'s caption
  (677–682).** Three tellings of the same dissociation within 35 lines. Keep the body; cut the
  captions to labels.
- **1161–1166 restates 1129–1159, which restates `tab:divergence`'s caption, which restates
  `fig-frame`'s caption.** Four tellings of "59% at an unchanged ceiling."
- **344–357** re-says 336–342. Cut to four sentences. Keep 356–357 — that is a good sentence.
- **1499–1506 reproduces 1468–1470 verbatim**, including both percentages. And `tab:baselines`'s Role
  column is copied cell-for-cell into `tab:lookup`'s note column, two tables 20 lines apart.
- **1395–1402** tells two instrument defects a third time; both are already at 1279–1282 and 300–303.
- **1206–1214** retells the 600-token truncation, already the flagship example at 202–204. Keep only
  the optimal-transport withdrawal, which is new.
- **1668–1672** recites `tab:allarms` row by row. Keep the last sentence.

Rule of thumb: **if a claim block, a caption and a paragraph all say the same thing, the paragraph
survives and the other two become labels.**

---

# Part 6 — section order

**The macro-order is right and should be defended, not changed.** The metric audit must precede "the
model is at chance", because the force of the null depends on chance being 0.50 by construction; and
the probe/ablation/substitution arc must follow it, because "where does the failure live" is only a
question once the failure exists. Do not restructure. Nine local moves:

1. **Add a 6–8 line "what this chapter found" paragraph after line 37**, before `fig:timeline`. The
   timeline gives eight beat *titles* and no results. An examiner should hold the four verdicts — the
   standard metric is saturated; the model is drug-blind; the drug is represented and barely read; a
   lookup beats it — before meeting any apparatus. **This alone fixes half of problem 1.1.**
2. **Move the assay-layout description (405–409) and the well arithmetic (1242–1244) into §3.1**,
   before line 176. The two-way clustering argument, the plate-leak result and the whole-well holdout
   are all consequences of one physical fact currently stated 337 and 1050 lines late. This also
   reconciles 289 samples with 6,628 conditions in one sentence.
3. **Move `fig:nir` from 316 to 292 and lift `J`'s definition into the body.** The schematic
   explaining what NIR is currently sits after the definition, the two-properties paragraph and the
   whole `nir_expr`/`nir_rank` discussion.
4. **Move the `worst`/`francesca` convention paragraph (275–281) before the `panel-τ` comparison
   (267–270).** `worst` is used at 270 and defined at 277, and the defining paragraph closes with "the
   convention is not mentioned again" — after it has already been used.
5. **In §3.3: state the verdict (517–522) first**, then walk the four instruments as objections
   closed. Enumerate the four explicitly at 401, add a metric column to `tab:blind`, and relocate the
   mean-shift ladder (504–515) — it answers "what is there to know", a different question.
6. **Move the divergence measurement and `tab:divergence` (1129–1153) plus `fig:frame` to the head of
   §3.8, and delete "Before" from 1002.** §3.8's central justification has its evidence 220 lines
   downstream, and §3.9 opens by announcing it should have come first.
7. Fix `fig-frame`'s caption and TikZ to show **plate** scope, or move the figure to where cell-line
   scope is still live.
8. Resolve the three magnitudes in §3.10 (1174–1176) at first mention rather than 110 lines later —
   either state the comparator caveat there, or move the numbers to after it is established.
9. Point at the appendix where the five training arms are enumerated, at the line that claims the
   multiplicity correction.

---

# Part 7 — constraints and deliverable

**Hard constraints**

- **Do not change any number, interval or verdict** beyond the corrections in Part 4. If one looks
  wrong, add `% AUTHOR NOTE:` at that line and continue.
- Keep every `\label{}` — other chapters cross-reference them.
- Keep the macros (`\NIR`, `\DEdr`, `\gap`, `\Tcoef`, `\DRF`, `\ci{}{}{}`). Define them in prose; do
  not redefine them in LaTeX.
- Do not introduce a citation not already in the file. Mark gaps `% AUTHOR NOTE: needs citation`.
- Tables may be merged, cut, or demoted to a sentence — say which and why. Do not delete a number
  without relocating it.
- British spelling. No serial comma.

**Tone.** Two failure modes, both bad. Do not soften into hedging mush — "not established" is a
precise finding, not an apology, and the positive results are real. Do not dramatise — no
"surprisingly", no "strikingly", no narrative of discovery.

**Deliverable**

1. The complete revised `Investigation-v4.tex`, in one LaTeX code block, compilable.
2. A note (≤ 600 words) before it: what you cut and why; where you put the orientation and the
   notation block; the three sentences you consider the worst offenders in the original and how you
   split them; and — most valuable — any place where the text asserts something the numbers do not
   obviously support.
3. All `% AUTHOR NOTE:` lines collected at the end.

**Not wanted:** a critique, a plan, or an outline. A finished chapter. If you run short of room,
revise fewer subsections *completely* rather than all of them partially and say which you did — and
if so, prioritise, in this order: the orientation paragraph (Part 6, move 1); the notation block and the
ceiling taxonomy (Part 2); §3.3 restructured verdict-first (Part 6, move 5); and the cuts in Part 5.

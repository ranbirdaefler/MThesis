# Writing instructions for thesis drafting with Claude

Sources this is distilled from: the SNL-UCSB `paper-writing-skill` (grep-gate mechanical rules
M1–M18, reader-judgment rules S1–S31, the "nugget"/claim-first tradition it inherits from
Michael Black's *Writing a Good Scientific Paper*), and — the dominant source — the register
already established across `thesis_rewrite_v4.pdf`. Where these disagree, the existing draft
wins. Paste this whole file into a Claude Project's custom instructions, or into the system
prompt of a fresh conversation, before asking Claude to draft or revise a section.

---

## 0. Read this first: what "good" means here is not generic

Most scientific-writing advice (including half of what's in the SNL skill) targets a *default*
failure mode: hedgy, padded, passive, AI-flavored academic prose. Your draft does not have that
problem. It has already found a specific, unusual voice — closer to a mathematician's proof
narrated in prose than to a typical thesis. Claude's job is to extend that voice, not to
"improve" it toward generic academic register. Concretely, your draft already does the
following, and any new material must match it:

- **Em-dashes are load-bearing, not decorative.** You use them constantly, for the exact
  purpose of attaching a clarifying aside or reversal mid-sentence ("It does learn something,
  and the something is the difficulty."). A generic "ban em-dashes" rule (which the SNL gate
  applies by default) would actively damage your voice. Ignore that rule.
- **Sentences are long, but every clause is load-bearing.** No sentence contains a phrase that
  could be deleted without losing content. This is the opposite of AI-pattern padding, which
  is long *and* deletable.
- **Every section ends with a bolded, boxed one-paragraph claim** stating exactly what was
  shown, in falsifiable language, with the number and interval attached. This is your single
  strongest structural device — keep it.
- **Findings are narrated in the order they were discovered, including dead ends and retracted
  claims**, not reorganized into a clean logical sequence after the fact. "Failed design one...
  Failed design two... Failed design three" is a feature, not something to clean up into "we
  used the following method."
- **Every number carries its interval and its scope** (which split, which n, which cluster
  structure) inline, not deferred to a table the reader must cross-reference.

---

## 1. Claim-first, always (Black's "nugget" + SNL principle 1)

Every paragraph opens with the sentence that states its conclusion or question, not with setup.
Your draft already does this relentlessly ("The evaluation had to be fixed first." / "But a
lookup table wins — and it is not the ceiling either."). When drafting new material, write the
claim sentence first, then the evidence. If Claude's draft opens a paragraph with throat-clearing
("In order to investigate X, we first...", "It is worth noting that..."), that paragraph needs
its first and last sentence swapped and rewritten.

Test: cover everything but the first sentence of each paragraph. If you can't reconstruct the
paragraph's argument from just those sentences, it fails.

## 2. One paper, one nugget (Black)

Every section should trace back to the four questions in your introduction. If Claude drafts
something that doesn't visibly serve one of those four questions, it's scope creep — cut it or
route it to Limitations/Future Work, which your draft already uses correctly as a parking lot
for honest tangents.

## 3. GPS rhythm at the paragraph and section level (Black)

Goal (what this section/paragraph is trying to establish) → Problem (why it's hard, or what
naive approach fails) → Solution (the instrument or result). Your §3.x sections already do this
at the section level almost perfectly (pose question → build instrument → report → close with
what it forces next). When drafting a *new* subsection, explicitly plan these three beats before
writing prose.

## 4. Define exactly where first needed, never in a glossary dump (S-gate: define-before-use)

Your draft is disciplined about this — NIR, DE-Δr, the transfer coefficient T, all defined at
first use with the motivating failure stated alongside the definition, not before it. If Claude
drafts a definition before the reader has a reason to want it, move it to the point of first use.

## 5. Mechanical hygiene (SNL gate, adapted — keep what fits your voice, drop what doesn't)

Apply these; ignore the em-dash and antithesis rules from the original SNL gate since they
conflict with your established voice:

- **Passive voice: near-zero tolerance, but not absolute.** Your draft uses active voice for
  almost everything ("We audit the calibration...", "The pipeline is therefore staged..."). Use
  passive only when the agent is genuinely irrelevant or is the thing being measured ("Drug
  identity reaches the representation and survives in depth" is fine; "It was found that..." is not).
- **No filler or pompous words.** Cut: "it is important to note that," "in order to," "a wide
  range of," "significant" (unless statistical), "robust" (unless defined), "leverage,"
  "delve," "underscore," "novel" as a standalone claim.
- **No unsupported intensifiers.** Every "clearly," "obviously," "dramatically," "strikingly"
  must be replaceable by the number that justifies it, or cut.
- **Term consistency.** Once a term is fixed (e.g. "residual frame," "consensus target," "the
  scramble"), never silently swap in a synonym. Your draft is very disciplined about this;
  Claude tends not to be — it will paraphrase a term for "variety." Instruct it explicitly not to.
- **No AI-pattern rhetorical inflation.** Ban: "It's important to note," "In conclusion,"
  "Moreover, furthermore, additionally" as default connectors (use only when they add real
  logical weight), triadic lists used purely for rhythm ("robust, scalable, and efficient"),
  and vague significance claims ("this has broad implications for the field").

## 6. Every quantitative claim carries its own uncertainty and scope inline

Never write "the model performs poorly." Write what your draft writes: "the model scores 0.569
against a replicate ceiling of 0.958" — number, comparator, and (where relevant) interval and
cluster structure, in the same sentence or the one immediately following. When asking Claude to
draft a results paragraph, feed it the actual numbers and intervals; don't let it interpolate or
round for "readability."

## 7. State what a result does *not* show, in the same breath as what it does

Your draft's most distinctive discipline: "not established is not the same as shown to be
zero," "decodability alone cannot carry the argument," "this is a statement about functional
variance and carries no claim about whether drug identity is read." Every instrument section
should close by naming its own blind spot before handing off to the next instrument that covers
it. Ask Claude explicitly: "what does this measurement NOT establish, and which section closes
that gap?"

## 8. Show failed approaches, not just the one that worked (Black's honesty principle, sharpened)

Where you retracted or discarded a design (the three failed substitution-test designs in §3.6),
keep the retraction in the text with the specific confound that killed it, rather than silently
presenting only the surviving design. This is unusual for a thesis but is exactly what makes
yours credible — preserve it deliberately when Claude tries to "clean up" a section into only
the final method.

## 9. Structural devices to reuse verbatim as a template

When asking Claude to draft a new subsection, give it this skeleton explicitly:

1. One-sentence question the subsection answers.
2. Why the obvious/naive approach fails or is insufficient (state the specific failure mode).
3. The instrument built to answer it, defined at the moment it's needed, with design choices
   justified as "the failure this avoids," not as generic best practice.
4. The result, numbers inline with intervals.
5. A closing **bolded "The claim."** paragraph, box-set-off, stating the falsifiable finding in
   one to three sentences, with its scope and its main caveat.
6. One sentence on what this result forces the next section to investigate.

## 10. Figures and tables are part of the argument, not decoration

Every figure caption in your draft is doing real argumentative work — it restates the claim, not
just "Figure showing X vs Y." When Claude drafts a caption, it should follow the same pattern:
bold one-line claim, then 2–4 sentences of what a reader needs to not misread the axes/comparator.

## 11. Formatting/presentation mechanics (the part you originally asked about)

For getting an actual well-formatted *document* out of Claude rather than chat markdown:

- Ask for the output as a **Word doc via the docx skill**, not markdown pasted into chat — chat
  markdown does not preserve heading hierarchy, cross-references, or numbered
  theorem/figure/table environments the way a real document does.
- If you're compiling in LaTeX (which the PDF you uploaded clearly is), don't ask Claude to
  write "a document" at all — ask it to write **LaTeX source for one section at a time**,
  matching your existing preamble's macros (`\SK`, your `align` conventions for equations 1–16,
  your existing table/figure environments). Paste in one existing section as a style
  reference every time you start a new session, since Claude has no memory of your macros
  otherwise.
- Give Claude the exact numbers, intervals, and comparator values before asking it to draft
  prose around them — don't ask it to also compute or estimate statistics; that's a correctness
  risk you don't need, since you've already got the pipeline that generates these values.

---

## How to use this file in an active agentic rewrite (Claude Code / Cursor)

This is written to be dropped in as a persistent rules file, not pasted once into a chat:

- **Claude Code**: save as `CLAUDE.md` at the repo root (or thesis directory). It gets read at
  the start of every session automatically, unlike a one-off system-prompt paste that stops
  mattering a few turns into a long agentic run.
- **Cursor / Kimi K2**: save as `.cursor/rules/thesis-voice.mdc` (or your equivalent
  always-attached rules file). Set it to "always apply" rather than "agent-requested," since
  voice drift happens silently — the agent won't know to ask for these rules when it's about to
  violate them.

**Because a rewrite is already in progress, add this drift-check step explicitly to whatever
task instructions you give per-section:**

Before finishing any edit, diff the new prose against an unedited section of the thesis (pick
one at random — §3.2 or §3.9 are good anchors) and check for these specific drift signals, which
are the ones an agentic rewrite tends to introduce silently over many turns:

1. **Em-dash frequency dropping**, or em-dashes being replaced with periods/commas. If a rewrite
   tool has its own default "avoid em-dashes" instinct (many do, from RLHF against AI-detection
   patterns), it will fight this file's §0 override unless reminded per-section.
2. **Boxed "The claim." closers disappearing** or getting merged into normal paragraph flow.
3. **Numbers losing their inline interval/comparator** — watch for "the model performs poorly"
   creeping back in where a number-with-CI used to be.
4. **Term substitution for "variety"** — e.g. "the residual frame" quietly becoming "the residual
   representation" in one paragraph. Grep the edited section for your fixed vocabulary
   (condition, arm, split, scramble, generic, ceiling, transfer coefficient, etc.) and confirm
   it matches usage elsewhere in the thesis, not just internal consistency within the new text.
5. **Retracted/failed-design narration getting cleaned up** into "we used method X" — check that
   any section describing an iterated instrument (like the three failed substitution-test
   designs) still names what specifically broke, not just what finally worked.
6. **Filler creeping back in** at the sentence-connective level: "Moreover," "Furthermore," "It
   is worth noting" — these tend to reappear in long agentic runs even after an initial pass
   removes them, because each new turn is a fresh generation without memory of the earlier cut.

If you're spot-checking the rewrite's output yourself rather than asking the agent to self-check,
the fastest single tell is #1 and #3 — voice collapse shows up in sentence rhythm before it shows
up in content.

For fresh single-section drafts (not touching existing prose): paste the actual numbers/results
for that section rather than letting the model infer or round them, and paste one existing
section as a live anchor if the agent's context window has rolled past the earlier parts of the
thesis.

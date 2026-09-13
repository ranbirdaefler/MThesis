# Thesis v7

This folder is a submission revision of `thesis_v6` at commit
`4696986b283dbbb74e4efaf126357981a3e9efb2`. Earlier thesis folders are preserved.
All figure inputs needed to compile the PDF are included in `figs/`.

## Revision summary

- Corrected unseen-drug membership using explicit held-out split labels. The
  sampled training evaluation no longer acts as a training-set inventory.
  Four drugs with training evidence are excluded from that subgroup. The
  protein-target result remains unchanged; the mechanism and chemistry
  estimates change and remain inconclusive. The computation, provenance and
  corresponding figure values are checked by a reproducible script.
- Aligned the consensus methods with the implementation: targets average
  rank-based proxies from cell sentences, not measured expression. Narrowed
  the OT-family and common-target claims, included the entropy term, and
  propagated these distinctions into the appendix and interpretation.
- Replaced the cross-estimand confidence-interval comparison with a figure of
  the two intervals for the unseen-drug model-minus-neutral gap. Removed the
  unsupported mechanism upper-bound argument and qualified the power
  approximation. Residual energy is described as an observed squared-norm
  share that can include noise and context-specific effects.
- Applied the submission format last: A4, four blank opening pages, 12 pt main
  body on a 24 pt baseline (about 27–29 lines on full prose pages), and 25 mm
  left/right outer margins around the entire 160 mm live area. The body,
  note gap and note column occupy 110, 7 and 43 mm respectively. Tables keep
  compact 10.5 pt text and their own leading; figure captions and margin
  notes retain separate readable sizes. Figure widths, long equations and
  selected heading/float transitions were adjusted to the new measure.

The formatting targets follow Bocconi's official master's thesis instructions:
[Thesis content and format, academic year 2025–2026](https://didattica.unibocconi.eu/tsg/testo.php?comando=Apri&edizione=2026&idAnt=26641&idr=26641&strperc=10.&volume=R2).

Corrected unseen-drug channel summaries (estimate and 95% interval):

| Channel | Conditions | Drugs | Gap | 95% interval |
| --- | ---: | ---: | ---: | --- |
| Protein target | 161 | 2 | +0.1197 | [+0.0499, +0.1894] |
| Mechanism | 163 | 3 | +0.0065 | [-0.1622, +0.1752] |
| Chemistry | 711 | 10 | +0.0427 | [-0.0030, +0.0885] |

One bibliographic limitation remains: the companion thesis entry `mertkaan`
retains the known metadata for Mert Kaan's 2026 Bocconi master's thesis, but its
title and full author name could not be confirmed. The consulted copy lacks a
title page; no missing metadata has been invented. This is separate from the
four main corrections implemented in this revision.

## Build

Use TeX Live with `pdflatex`, `latexmk`, `biber`, and Python 3 (including NumPy
and SciPy) available on PATH:

```sh
cd thesis_v7
bash tools/build.sh
```

The compiler writes its working output to `main.pdf`. A full successful build
runs the held-out-channel recomputation and `tools/audit.py`, then copies the
PDF to `thesis_v7.pdf`. For an intermediate
compile without submission checks, use `bash tools/build.sh --fast`; its output
remains `main.pdf`.

The v7 build was established with TeX Live 2026, pdfTeX 1.40.29, latexmk 4.88,
and Biber 2.22. A portable TinyTeX installation can supply these tools; no
system-wide TeX installation is required. In addition to TinyTeX-1's packages,
the thesis uses these packages and their dependencies:

```sh
tlmgr install latexmk biber biblatex ebgaramond ebgaramond-maths alegreya \
  inconsolata fontaxes mweights svn-prov fancyhdr tcolorbox tikzfill pdfcol \
  needspace changepage titlesec fmtcount marginnote mathtools csquotes \
  threeparttable multirow siunitx enumitem cleveref pgf ragged2e etoolbox \
  setspace microtype caption booktabs pdflscape upquote psnfss adjustbox
```

On Windows, the same commands can run inside WSL; access this folder through
its `/mnt/c/...` path. Build after installing all required packages. If a
previous attempt stopped at a missing package, use `latexmk -g -pdf
-halt-on-error -interaction=nonstopmode main.tex` once to force a fresh attempt.
Set `PYTHON` to a Python executable path to use a specific environment. This
also permits a Windows Python installation with NumPy/SciPy to perform the
checks from a WSL build shell.

## Verification

The current audit checks the recorded SHA-256 hashes of the original
`thesis_v6/` and `thesis/` files (CRLF is normalized to LF for text; PDFs remain
byte-exact), unresolved source placeholders, ordering of
numeric confidence intervals, and compiler/bibliography errors. The full build
fails on undefined references/citations, unresolved rerun requests, or overfull
boxes, oversized floats, unusable table columns, missing glyphs or broken PDF
destinations. Underfull boxes are reported for visual review. The preserved source
folders must be present to run the preservation audit, though the PDF itself
compiles using this folder alone.

The full build also runs `python3 tools/correct_heldout_channels.py --check`.
That read-only check reproduces the published subgroup before recomputing the
corrected selection, verifies the saved corrected artifact and figure ledger,
and compares the rounded values and axis range against the rendered TeX source.
It uses pinned source data from the repository and does not retrain a model.
Run this command separately to verify the numerical correction without TeX.

`tools/audit_legacy.py` preserves the historical v1-vs-v2 numeric-parity audit.
It is not the v7 acceptance test: v7 intentionally corrects numbers and claims
identified during the pre-submission review. Scientific verification and a
rendered-page review are required in addition to the automated checks.

## Final validation, 13 September 2026

The full build and its scientific/source checks passed. The final
`thesis_v7.pdf` has **255 pages** and **1,561,267 bytes**; its SHA-256 is
`5d21a76210d2f8ecc46506b46b54a1d92994632d19b201d5b8f9bbd48578edeb`.

- The original 132 files pass the preservation check. The corrected channel
  artifact and figure reproduce, and all 184 parsed numeric confidence
  intervals have ordered bounds.
- The final compiler log has no errors, overfull boxes, oversized floats,
  unusable table columns, undefined references/citations, missing glyphs,
  duplicate reference labels or missing PDF destinations. Its 24 underfull
  box advisories were covered by visual review.
- Rendered-page review covered all pages, including detailed checks of small
  labels, tables and captions. After the final local fixes, 26 changed pages
  were inspected again; the other 229 body rasters matched the reviewed
  candidate. No remaining clipping, overlaps or orphaned section headings
  were identified.
- A4 dimensions, four initial blank pages, 12 pt body text and ordinary prose
  line counts were verified. Front-matter folios run i–iv and body folios
  1–247. All 732 named PDF destinations and 711 internal links resolve.

The unconfirmed companion-thesis metadata noted above remains the known
bibliographic limitation.

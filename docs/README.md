# docs/ — writeups, organized by function and data regime

The results **verdicts** live in the root [`FINDINGS.md`](../FINDINGS.md) (the source of truth).
This folder holds the **detailed prose/method writeups** behind those verdicts, grouped so the
old L1000 full-panel work is clearly separated from the current [END_CELL] work.

## The one thing to know: two data regimes

The project pivoted mid-way. Both matter for the thesis, but for different reasons:

| Regime | What it is | Role in the thesis | Where |
|---|---|---|---|
| **Legacy L1000 (full-panel)** | Original `data_diverse2`: every cell sentence contains all 946 panel genes (expressed ranked, unexpressed appended in a fixed tail). Model `pythia_sft_diverse2/checkpoint-10000`. | **Method-evolution / background.** Shows *why* we moved to [END_CELL] (the fixed tail made the three absent-gene treatments identical and gave the model a long uninformative tail). Not cited as current results. | `legacy_l1000/` |
| **[END_CELL] (current)** | `data_diverse2_endcell_big`: expressed genes only + `[END_CELL]` sentinel — matches C2S's real `generate_sentences()`. Model `pythia_sft_endcell/final`. | **The reportable results.** Everything in the advisor message and thesis conclusions comes from here. | `endcell/` |

If a number appears in both regimes, the **[END_CELL] number is the one to report**; the legacy
number is context for the evolution story only.

## Folders

- **`methods/`** — how the dataset/pipeline is built (foundation, applies across regimes).
  - `dataset_construction.md` — Tahoe streaming, leak-free fixed L1000 panel, control matching,
    eval tiers, QC, reproducibility. *(Written in the full-panel era; the panel/tiers/control-matching
    carry over unchanged to [END_CELL]; only the sentence construction differs — expressed-only + sentinel.)*

- **`legacy_l1000/`** — historical full-panel results (background; superseded).
  - `results.md` — full-panel model eval (DE-Δr, K-sweep, baseline ladder, noise ceiling). Carries a
    LEGACY banner; superseded by the [END_CELL] eval (pending) + `FINDINGS.md`.

- **`endcell/`** — current [END_CELL] work (what we report).
  - `drug_specificity_analysis_writeup.md` — the analysis spine, Parts I–V. **Mixed regime** (see its
    banner): Parts I/II/IV legacy, III/V [END_CELL]; being migrated.
  - `part6_expression_space_draft.md` — Part VI: rank-vs-true-expression (representation is not the bottleneck).
  - `dimensionality_probe_analysis.md` — mechanistic probe: the drug is decodable from the representation
    (76–82%) but attenuated toward the output ("read but not used").

- **`proposals/`** — forward-looking specs (next phase, not yet run).
  - `pubchem_drug_injection_spec.md` — drug-knowledge injection experiment (gated on the task ceiling).

## Reading order for someone new
1. Root `FINDINGS.md` — what we know, in Q→A form.
2. `methods/dataset_construction.md` — how the data is built.
3. `endcell/` — the current results (spine → Part VI → probe).
4. `legacy_l1000/results.md` — the "why we pivoted" background.
5. `proposals/` — what's next.

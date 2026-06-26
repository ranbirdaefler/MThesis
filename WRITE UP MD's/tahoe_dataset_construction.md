# Construction of the Tahoe-100M Perturbation Dataset for C2S-Scale Fine-Tuning

*Technical methods documentation. Prepared as source material for the thesis report.*

---

## 1. Overview and objective

The data pipeline turns the **Tahoe-100M** single-cell drug-perturbation atlas into a corpus
of **(control cell, perturbed cell) sentence pairs** for supervised fine-tuning (SFT) of a
Cell2Sentence model (`C2S-Scale-Pythia-1b`). Each example asks the model to predict the
transcriptional response of a single cell to a drug, given (i) a textual description of the
perturbation (cell line, drug, dose, mechanism) and (ii) the matched untreated (DMSO) control
cell for the same biological context.

The pipeline is built around three requirements that determine its design:

1. **No information leakage** between the prompt and the target, so the task measures genuine
   response prediction rather than recovery of a leaked gene set.
2. **Condition diversity** — the corpus must span many distinct drugs, cell lines and doses,
   because the scientific question (drug-*specific* response prediction) cannot be studied on
   data dominated by a handful of conditions.
3. **Valid control matching** — every perturbed cell must be paired with an appropriate
   untreated control, since the entire learning signal is the *change* relative to control.

This document describes the source data, the cell-sentence representation, the leakage-free
panel design, the sampling and control-matching architecture, the held-out evaluation splits,
and the reproducibility engineering of the final pipeline.

---

## 2. Source data: Tahoe-100M

Tahoe-100M (`tahoebio/Tahoe-100M` on the HuggingFace Hub) is a single-cell drug-perturbation
screen of roughly **100 million cells**. Its relevant structure:

| Property | Approximate value |
|---|---|
| Total cells | ~100,000,000 |
| Drugs | ~379 |
| Cell lines | 50 |
| Cell-line × drug conditions | ~17,813 |
| Dose-resolved conditions | ~60,000 |
| Mechanism-of-action (MOA) classes | ~180 |
| Vehicle control | DMSO (`DMSO_TF` / `DMSO`) |

The data are organised into **plates** (experimental batches), each containing many cell
lines, the drug panel at several doses, and DMSO vehicle controls. On the Hub the corpus is
stored as a large set of **Parquet shards** (plus small metadata tables under a `metadata/`
prefix that are excluded from the cell stream).

Each cell row provides, among other fields: a sparse gene representation (`genes` = token
ids, `expressions` = counts), the `drug` label, `cell_line_id` (Cellosaurus `CVCL_*`),
`plate`, `sample` (used to recover the administered concentration), and MOA annotation.

**Streaming, not downloading.** Because the corpus is ~100M cells, it is never materialised to
disk. The pipeline uses the HuggingFace `datasets` library in **streaming mode**, reading
individual Parquet shards on demand through `hf://datasets/tahoebio/Tahoe-100M/<shard>` URLs.
This keeps the memory footprint bounded and avoids a multi-terabyte download.

---

## 3. The cell-sentence representation

Following the Cell2Sentence (C2S) framework, a cell's transcriptome is encoded as a **"cell
sentence"**: the gene symbols ordered by **descending expression**, so the most highly
expressed genes appear first. The model consumes and produces an ordered sequence of gene
names (a permutation), and the prediction task reduces to predicting the *rank ordering* of
genes in the perturbed cell.

A raw count vector is converted to normalised expression before ranking. For a cell with raw
counts `x_g` over expressed genes, expression is normalised as

```
norm_expr_g = log10( 1 + (x_g / sum_g x_g) * 1e4 )
```

(library-size normalisation to 10,000 counts followed by a log transform), and genes are then
sorted by `norm_expr_g`.

---

## 4. Leakage-free design: the fixed L1000 panel

### 4.1 Why the panel is fixed independently of treatment

Selecting the gene set of an example from the treated cell's expression would leak treatment
information into the prompt: the identity of the genes would itself reveal which genes the
drug switched on, allowing a model to score well by exploiting the gene vocabulary rather than
by predicting the response. Such a construction is also irreproducible at inference, where the
treated cell — and therefore the "correct" gene set — is unknown.

To prevent this, every example is built over a **single fixed gene panel**, identical across
all cells and computed independently of any treatment information:

- The panel is the intersection of the **L1000 landmark genes** with the Tahoe gene
  vocabulary, giving **946 genes** (`l1000_panel.json`). This follows the L1000 methodology
  used by C2S-Scale and keeps the panel biologically meaningful (landmark genes selected to be
  informative of the broader transcriptome).
- It is computed once and reused for every example.

### 4.2 The "P-full" sentence

Each cell sentence (control and response alike) lists **all 946 panel genes**, in two
deterministic blocks:

1. **Expressed block** — panel genes expressed in the cell, ranked by descending expression.
2. **Unexpressed tail** — the remaining panel genes (expression ≤ 0), appended in a fixed
   **canonical panel order** (their index order in `l1000_panel.json`).

Because the panel and tail order are fixed, the gene *identities* carry no treatment
information; only the *ordering of the expressed block* does. Control and response are each
exactly 946 tokens long, and gene overlap between any two sentences is 1.0 by construction (so
overlap is not an informative metric — the signal lives entirely in the ranking). At inference
the control sentence and the panel fully determine the input, so the construction is
reproducible.

One consequence is recorded here because it shapes the evaluation. The *set* of unexpressed
genes is per-cell — there is no panel gene that is unexpressed in every cell — but the
*ordering convention* for whichever genes are unexpressed (canonical panel order) is the same
for every cell. When a predicted sentence is scored against the true sentence, any gene that is
unexpressed in *both* falls into that shared canonical ordering in each, and therefore agrees by
construction rather than by prediction; the expressed-versus-unexpressed split is likewise
trivial to reproduce. A naïve rank correlation over the full 946-gene panel is thus inflated by
these freely-agreeing positions and overstates the genuine difficulty of the task (ordering the
expressed genes and capturing the drug-driven shifts). The evaluation therefore reports a
**delta metric over the top differentially-expressed genes** as its headline rather than the
raw panel correlation (documented separately in the evaluation methods).

### 4.3 Prompt / target format

Each example is a prompt → response pair:

```
PROMPT:
Predict the response of {cell_line_name} to {drug} at {dose}. Mechanism: {moa}.
Control cell: {control_cell_sentence}

Response cell:

RESPONSE:
{treated_cell_sentence}
```

The prompt contains the perturbation description plus the matched control's 946-gene sentence;
the target is the treated cell's 946-gene sentence. At ~3.25 tokens per gene a 946-gene
sentence is ~3,079 tokens, so a full example (control + response + prompt text) is ~6,200
tokens, within the model's `max_length` of 8,192.

Per-example **metadata** is stored alongside each pair: `drug`, `cell_line_id`,
`cell_line_name`, `plate`, `sample`, `dose`, `dose_float`, `moa`, and a `control_plate_matched`
flag (§6).

---

## 5. Shard-level diversity sampling

Because Tahoe is written to disk in processing order, a contiguous prefix of the stream is
deep but narrow — many cells per condition, but the *set* of distinct conditions saturates
quickly. To obtain diversity, the pipeline draws **whole Parquet shards at random from across
the corpus** for the treated cells. Shards are enumerated with `discover_expression_shards()`
(excluding the `metadata/` tables) and a reproducible subset is selected by
`select_shards(all_shards, num_shards, seed)`; because each shard spans many conditions,
sampling shards from across the corpus pulls condition diversity from the whole dataset.

Two knobs control the treated stream:

- `--num_shards` — the **diversity dial**: how many shards to sample. More shards → more unique
  drugs / combos / doses.
- `--rows_per_shard` — the per-shard row cap, bounding total work (~`num_shards ×
  rows_per_shard` rows per pass).

A fixed `--shard_seed` makes the shard sample reproducible.

---

## 6. Control matching: decoupled DMSO collection

### 6.1 Plate-matched controls

The learning signal is the *difference* between a treated cell and its untreated baseline, so
each treated cell is paired with a **DMSO vehicle control** from the same `(cell_line_id,
plate)`. Matching on plate as well as cell line is deliberate: it controls for **plate-level
batch effects**, so the predicted change reflects the drug rather than a batch artefact. The
control key is therefore `(cell_line_id, plate)`.

### 6.2 Decoupled DMSO scan with early-stop and fallback

DMSO controls are a small fraction of each plate, and a plate's controls do not reliably
co-occur in the same shards as its treated cells. Collecting controls only from the sampled
treated shards would therefore leave many treated cells unmatched, and would get worse as more
treated shards are added (more distinct `(cell_line, plate)` keys to cover). To avoid this,
**control collection is decoupled from treated-cell sampling** and performed in a dedicated
scan. The run has three phases:

1. **Observe pass** (over the `num_shards` *treated* shards). Catalogues the observed drugs,
   `(drug, cell_line)` combos, and `(drug, dose)` points — used to define the held-out tiers
   (§7) — and records the exact set of `(cell_line, plate)` **control keys that will be
   needed**.

2. **DMSO scan** (over a broad shard set — by default **all** shards — read **whole**, without
   the `rows_per_shard` cap). Only DMSO rows are retained, capped at 8 per `(cell_line, plate)`
   key. The scan **early-stops** as soon as every needed control key has at least one control,
   so it does not read the whole corpus unless coverage requires it. In parallel it pools
   controls **per cell line** (up to 16 each) to support the fallback.

3. **Build pass** (over the `num_shards` treated shards). For each treated cell the control is
   resolved as: prefer the **plate-matched** `(cell_line, plate)` DMSO; if absent and the
   fallback is enabled, use any **same-cell-line** DMSO from another plate, tagging the example
   `control_plate_matched = false`; otherwise drop the cell.

Decoupling control collection means coverage does not degrade as `num_shards` grows, so the
diversity dial can be pushed freely. The same-cell-line fallback recovers residual misses while
keeping them **auditable**: because every example records whether its control was plate-matched,
a sensitivity analysis can exclude fallback examples to confirm they do not change the
conclusions. A `--no_cellline_fallback` switch enforces a strict plate-matched-only dataset.

The two control knobs are:

- `--dmso_shards` — number of shards scanned for controls (default: all). Decoupled from
  `--num_shards`.
- `--no_cellline_fallback` — strict mode; drop unmatched cells instead of using a
  same-cell-line control.

Coverage is reported per run as `no_control_rate` (over treated cells reaching the control
lookup), `control_fallback_rate`, and `plate_matched_keys / needed_control_keys`.

---

## 7. Generalisation tiers (held-out evaluation splits)

Held-out sets are drawn from the **observed** entities catalogued in the observe pass (not from
the full external catalogue), which guarantees that every held-out drug / combo / dose actually
has treated cells available to evaluate on. Four tiers test increasingly demanding
generalisation:

- **Tier 1 — seen conditions.** In-distribution cells (~10% of non-held-out cells are routed to
  eval; the remainder become training data).
- **Tier 2 — unseen drugs.** A set of drugs (capped at ≤ 20% of observed drugs, target 50) held
  out entirely from training — tests generalisation to drugs never seen.
- **Tier 3 — unseen drug × cell-line combos.** `(drug, cell_line)` pairs held out (for drugs not
  themselves Tier-2 held out), testing transfer of a known drug to a new cell line.
- **Tier 4 — dose interpolation.** For drugs observed at ≥ 3 doses (and not otherwise held out),
  the **middle** dose is held out, testing interpolation along the dose axis.

Each eval tier file is capped (`--max_eval_per_tier`, default 5,000). Tier-1 overflow spills
back into training rather than being discarded; held-out tier overflow is dropped.

---

## 8. Auxiliary models, quality control, and stratification

**Expression-recovery linear model.** Because the model predicts a gene *ranking*, an auxiliary
model maps rank back to (normalised) expression for the expression-space evaluation metrics. A
linear fit of `log10(rank) → normalised expression` is estimated on control (DMSO) cells,
restricted to the fixed panel for consistency with the panel ranks, and saved as
`linear_model.json` (slope, intercept, R²); it falls back to a global fit if too few control
cells are available.

**Quality control.** Cells with too few expressed genes (degenerate sentences) fail QC and are
skipped; the pair builder returns `None` for such cells.

**Stratification.** A per-condition cap (`--cells_per_condition`) limits how many cells are taken
from any single `(drug, cell_line, plate)` condition, preventing high-frequency conditions from
dominating and keeping the training set stratified across conditions. The dataset is therefore
grown in **breadth** (more conditions, via more shards) rather than **depth** (more cells per
condition): additional cells of an already-covered condition are near-redundant replicates,
whereas additional conditions improve coverage and the drug-specific signal — and adding shards
preserves stratification because the per-condition cap is unchanged.

---

## 9. Robustness and reproducibility engineering

- **Incremental, timeout-safe writing.** `train.jsonl` and `train_text.jsonl` are streamed to
  disk during the build pass and flushed periodically, so a node failure or wall-clock timeout
  leaves a valid partial dataset.
- **Determinism.** Shard selection, held-out split selection, and control sampling are seeded
  (`shard_seed` and fixed NumPy seeds), making a run reproducible given the same configuration.
- **Diversity report.** Every run writes `diversity_report.json` capturing `n_train`,
  `unique_drugs_train`, `unique_combos_train`, `unique_dose_points_train`, `unique_conditions`,
  `observed_drugs`, `no_control_rate`, `control_fallback_rate`, `plate_matched_keys`,
  `needed_control_keys`, `dmso_shards_scanned`, `treated_shards`, and `rows_per_shard` — the
  audit trail for the dataset.

---

## 10. Final configuration and invocation

The dataset was produced (on the Bocconi HPC `defq` CPU partition, via HuggingFace streaming)
with:

```
python tahoe_c2s_preprocess.py --mode full \
  --num_shards 32 \           # diversity: 32 shards sampled across the corpus
  --rows_per_shard 400000 \   # whole-shard treated reads
  --cells_per_condition 20 \  # per-condition stratification cap
  --held_out_drugs 50 \       # Tier-2 unseen-drug budget
  --panel_file l1000_panel.json \
  --output_dir <data_dir>
  # --dmso_shards defaults to ALL shards (whole-shard DMSO scan, early-stop)
  # cell-line fallback enabled by default (tagged control_plate_matched)
```

The size target is set by **breadth** (number of conditions / drugs), not by raw cell count.
With control coverage fixed by the decoupled DMSO scan, `num_shards` grows the corpus while
preserving stratification (the per-condition cap is unchanged); training compute is then managed
through the number of epochs rather than by shrinking the data.

---

## 11. Output summary

Each run produces, in its output directory:

| File | Contents |
|---|---|
| `train.jsonl` | Training pairs: `{prompt, response, metadata}` per line |
| `train_text.jsonl` | Concatenated `prompt + response` text (Trainer-friendly format) |
| `eval_tier1_seen_conditions.jsonl` | Tier-1 held-out eval examples |
| `eval_tier2_unseen_drugs.jsonl` | Tier-2 held-out eval examples |
| `eval_tier3_unseen_combos.jsonl` | Tier-3 held-out eval examples |
| `eval_tier4_dose_interpolation.jsonl` | Tier-4 held-out eval examples |
| `l1000_panel.json` | The fixed 946-gene panel |
| `linear_model.json` | Rank→expression recovery fit |
| `gene_id_to_symbol.json` | Token-id → gene-symbol mapping |
| `held_out_drugs.json` | Names of the Tier-2 held-out drugs |
| `diversity_report.json` | Coverage / diversity / control-matching audit metrics |

---

## 12. Final dataset statistics

The final dataset (`data_diverse2`, 32 treated shards, all-shard treated-first DMSO scan,
preprocessing wall-clock 2 h 29 m) has the following characteristics, from its
`diversity_report.json`:

| Metric | Value |
|---|---|
| Training examples (`n_train`) | 371,794 |
| Unique drugs in train | 320 |
| Unique (drug, cell_line) combos in train | 15,085 |
| Unique (drug, dose) points in train | 580 |
| Unique conditions | 38,773 |
| Observed drugs | 370 |
| Drugs held out of training (Tier 2) | 50 |
| Mean cells per condition | ≈ 9.6 (per-condition cap of 20 non-binding) |
| `no_control_rate` | 0.000 |
| `control_fallback_rate` | 0.0066 (0.66%) |
| Plate-matched key coverage | 631 / 650 (97.1%) |
| DMSO shards scanned | 283 |
| Cell-sentence length (control / response) | 946 / 946 genes |
| Tier 1 / 2 / 3 / 4 eval sizes | 5,000 / 5,000 / 2,790 / 5,000 |

The control matching is essentially fully plate-matched: 97.1% of needed `(cell_line, plate)`
keys were covered directly, leaving only 0.66% of examples on the same-cell-line fallback (all
tagged via `control_plate_matched`). The training set spans 320 of the 370 observed drugs, with
the remaining 50 held entirely out of training for the unseen-drug tier. Growth came from
breadth (more shards), not depth: at ≈ 9.6 cells per condition the per-condition cap never
binds, so the corpus is broad and stratified rather than dominated by a few heavily-sampled
conditions.

*Tier-2 note: 50 drugs are held out of training, but the Tier-2 eval file is capped at 5,000
examples, which in this build are drawn from 10 distinct held-out drugs. For the final
evaluation the Tier-2 sampler should be stratified per drug so that all 50 unseen drugs are
represented, strengthening the effective sample size of the unseen-drug generalisation
estimate.*

# C2S-Scale Perturbation Prediction on Tahoe-100M

Fine-tuning a Cell2Sentence-Scale language model to predict single-cell drug-perturbation
responses, with a rigorous, leakage-controlled evaluation framework. MS thesis project
(Bocconi, Buffa Lab).

**One-line summary:** given a control (untreated) cell's expression profile and a drug, the
model predicts the treated cell's transcriptional response, represented as a *cell sentence*
(genes ordered by expression). We evaluate it against biologically meaningful baselines and a
replicate noise ceiling, and ablate what the model actually learns.

---

## 1. What this project does

Single-cell drug-perturbation prediction: *how does a cell's transcriptome change when exposed
to a drug?* We fine-tune `C2S-Scale-Pythia-1b` on the **Tahoe-100M** screen (~100M single cells,
~379 drugs × ~50 cancer cell lines) and evaluate whether it predicts perturbation responses that
(a) generalise to unseen drugs/contexts and (b) are *drug-specific* rather than generic.

The **cell-sentence** representation (Cell2Sentence) encodes each cell as an ordered list of gene
symbols, highest-expressed first. The model reads a prompt containing the drug, cell line, dose,
mechanism, and the control cell's sentence, and generates the treated cell's sentence.

### Core design principles
- **Leakage-free fixed panel.** All cells are represented over one fixed 946-gene panel
  (L1000 ∩ Tahoe), so the model never gets to choose the gene set from the answer.
- **Rank-based headline metric (DE-Δr).** We score prediction quality on the genes the drug
  actually *moves*, in rank space, avoiding the inflation that whole-transcriptome correlations
  suffer from (Simpson's-paradox-style artifacts; see §5).
- **A baseline ladder, not a single baseline.** Mean-shift baselines of increasing strength
  (global → per-cell-line → per-MOA → per-MOA×cell-line) let us prove the model beats "predict
  the class/context average," which is the failure mode that inflates most perturbation results.
- **A measured noise ceiling.** Two real replicate cells of the same condition don't agree
  perfectly; we measure that irreducible ceiling so absolute scores are interpretable.

---

## 2. Key results so far

*(Model = `C2S-Scale-Pythia-1b` SFT, `checkpoint-10000`, ~43% of one epoch. Full detail in
[`docs/results.md`](docs/results.md).)*

- **DE-Δr ≈ 0.72** (rank-shift correlation on top-50 differentially-expressed genes), **flat
  across all four generalisation tiers** (seen conditions / unseen drugs / unseen combos / dose
  interpolation), with a smooth, robust K-sweep.
- **At ~91–95 % of the single-cell replicate noise ceiling** (per-tier, condition-matched) — the
  residual to a perfect score is mostly irreducible biological/technical noise, not model error.
- **Drug-specific above mechanism and cell line:** the model beats a per-MOA × cell-line
  mean-shift baseline by **+0.39 (seen) to +0.59 (dose)**, all CIs excluding zero, with the margin
  *growing* on harder tiers.
- **topN-expressed τ (≈0.25) sits at the replicate floor** — the modest value reflects the
  intrinsic noise of ranking saturated housekeeping genes, confirmed by the ceiling.
- **Base model (no fine-tuning) does not perform the task in this format** — the pretrained
  checkpoint emits cell-type annotations, not perturbation responses; DE-Δr undefined by design.

### Ablation in progress
- **Value of C2S pretraining** — a vanilla `pythia-1b` fine-tuned identically reaches DE-Δr ≈ 0.75,
  i.e. C2S pretraining confers *no measurable benefit at 10k steps* (paired Δ −0.03). This
  surprising result is under a validation audit (see the ablation write-up — not yet checked into
  `docs/`, see §7 Status).
- **Prompt-scramble ablation** (does the model use the drug token, or predict a context mean?) —
  running via [`src/scramble_eval.py`](src/scramble_eval.py) and
  [`src/paired_by_position.py`](src/paired_by_position.py); the decisive test of drug-specificity.

The roadmap of planned analyses is tracked outside this repo for now (see §7 Status).

---

## 3. Repository contents

### Pipeline scripts (`src/`)
| File | Purpose |
|---|---|
| [`src/tahoe_c2s_preprocess.py`](src/tahoe_c2s_preprocess.py) | Build the dataset: stream Tahoe shards, match plate-level DMSO controls to treated cells, construct fixed-panel cell sentences, write train + 4 eval tiers. The core data pipeline. |
| [`src/tahoe_c2s_preprocess_endcell.py`](src/tahoe_c2s_preprocess_endcell.py) | Variant of the preprocessing pipeline (end-cell construction). |
| [`src/build_l1000_panel.py`](src/build_l1000_panel.py) | Build the fixed 946-gene L1000∩Tahoe panel (`l1000_panel.json`) from the LINCS landmark gene list. |
| [`src/train_c2s_tahoe.py`](src/train_c2s_tahoe.py) | Supervised fine-tuning (hand-rolled PyTorch loop): loss masked to the response, bf16 autocast, gradient checkpointing, cosine schedule, disk-safe checkpoint pruning. |
| [`src/grpo_c2s_tahoe.py`](src/grpo_c2s_tahoe.py) | GRPO (RL) training variant. |
| [`src/evaluate_c2s_tahoe.py`](src/evaluate_c2s_tahoe.py) | ⭐ Evaluation harness **and shared metric library**: DE-Δr headline + K-sweep, topN-τ, panel-τ, mean-shift baseline ladder, drug-clustered bootstrap CIs, coverage guard, paired comparisons. Every baseline/ablation below imports its scoring functions. |
| [`src/regen_tier2_eval.py`](src/regen_tier2_eval.py) | Rebuild the unseen-drug (Tier-2) eval set per-drug-balanced (all 50 held-out drugs), harvesting controls from the existing train set — no DMSO re-scan. |
| [`src/sft_pythia_l1000.sbatch`](src/sft_pythia_l1000.sbatch) | SLURM submit script for the SFT run. |

### Baselines (`src/`, all score through `evaluate_c2s_tahoe`)
| File | Purpose |
|---|---|
| [`src/control_knn_baseline.py`](src/control_knn_baseline.py) | Control-similarity kNN retrieval baseline: for each test cell, retrieve same-cell-line training cells with the most similar control profile and predict the rank-consensus of their treated responses. Drug-agnostic, leakage-guarded. |
| [`src/control_state_baseline.py`](src/control_state_baseline.py) | Control-state-conditioned mean-shift baseline (cluster each cell line's controls into baseline-state groups) plus a kNN denoising diagnostic. |
| [`src/simple_baselines_and_consensus.py`](src/simple_baselines_and_consensus.py) | Two modes: `linear` — ridge regression predicting rank shift from the control rank vector (is the task linear in the control?); `consensus` — samples the model k times and averages, for a fair mean-vs-mean comparison against kNN. |

### Analysis / audit scripts (`src/`)
| File | Purpose |
|---|---|
| [`src/noise_ceiling_matched.py`](src/noise_ceiling_matched.py) | **Per-tier, condition-matched** replicate noise ceiling, computed directly from the eval files (cell-vs-cell and cell-vs-consensus). The version cited in results. |
| [`src/noise_ceiling.py`](src/noise_ceiling.py) | Streaming noise ceiling (cell-vs-cell and cell-vs-pseudobulk) over re-streamed Tahoe cells; broader but pooled across conditions. |
| [`src/scramble_eval.py`](src/scramble_eval.py) | Drug-specificity ablation: rewrite the drug/MOA token in eval prompts (different-MOA or random-drug), keeping control/cell-line/dose/truth fixed. |
| [`src/paired_by_position.py`](src/paired_by_position.py) | Paired real-vs-scrambled Δ, matched by position (scrambling changes the content-hash example ID, so pairing is done by row position + sanity checks). |
| [`src/inspect_generation.py`](src/inspect_generation.py) | Print raw generations from one or two models on a few examples (used to diagnose the base model's cell-type-annotation behaviour). |
| [`src/drug_specificity_in_data.py`](src/drug_specificity_in_data.py) | Model-free test: do two real cells of the *same* drug agree more (in a metric) than two of *different* drugs, within a cell line? Confound-controlled (dose, batch, MOA positive control) — separates "the model can't capture drug-specificity" from "no detectable drug-specific signal in the data." |
| [`src/pseudobulk_eval.py`](src/pseudobulk_eval.py) | Re-score the single-cell-trained model at pseudobulk (denoised) resolution, sweeping aggregation size — including a scrambled-drug sensitivity test and a pseudobulk noise ceiling. |
| [`src/spikein_metric_benchmark.py`](src/spikein_metric_benchmark.py) | Benchmark which metric best discriminates two drug populations (forced-choice accuracy) and how gracefully it degrades under spike-in contamination titration. |

> **Note:** `check_base_compat.py` (confirming `C2S-Scale-Pythia-1b` and `EleutherAI/pythia-1b`
> share architecture + tokenizer for the no-C2S ablation) is referenced in project notes but is not
> currently checked into this repo.

### Offline unit tests (`src/`, no model/network — "safe to delete")
- [`src/_test_eval_baselines.py`](src/_test_eval_baselines.py)
- [`src/_test_fixed_panel.py`](src/_test_fixed_panel.py)

### Committed reference inputs (`src/`)
- [`src/l1000_panel.json`](src/l1000_panel.json) — the 946-gene L1000∩Tahoe panel, canonical order.
- [`src/l1000_landmark_genes.txt`](src/l1000_landmark_genes.txt) — LINCS L1000 landmark gene symbols (panel source).

### Documentation (`docs/`)
| File | Purpose |
|---|---|
| [`docs/dataset_construction.md`](docs/dataset_construction.md) | Thesis methods: dataset construction, fixed-panel leak-free design, control matching, tiers, QC, reproducibility. |
| [`docs/results.md`](docs/results.md) | Full results writeup (performance, K-sweep, baseline ladder, noise ceiling, base-model finding). |
| [`docs/drug_specificity_analysis_writeup.md`](docs/drug_specificity_analysis_writeup.md) | Writeup of the drug-specificity analyses (data-level signal, pseudobulk, spike-in metric benchmark). |

> Additional docs referenced in earlier drafts (a no-C2S ablation audit, a living analysis
> roadmap, and fixed-panel implementation notes) are not yet checked into `docs/` — see §7 Status.

---

## 4. Repository structure

```
MThesis/
├── README.md                 # this file
├── requirements.txt          # inferred from imports (torch, transformers, datasets, numpy, scipy, tqdm, huggingface_hub)
├── .gitignore                # excludes data/, checkpoints/, eval_results/, caches, private drafts
├── src/                      # ALL Python lives here (flat — see note below)
│   ├── tahoe_c2s_preprocess.py, tahoe_c2s_preprocess_endcell.py, build_l1000_panel.py,
│   │   train_c2s_tahoe.py, grpo_c2s_tahoe.py, evaluate_c2s_tahoe.py,
│   │   regen_tier2_eval.py, sft_pythia_l1000.sbatch                             (pipeline)
│   ├── control_knn_baseline.py, control_state_baseline.py,
│   │   simple_baselines_and_consensus.py                                        (baselines)
│   ├── scramble_eval.py, paired_by_position.py, noise_ceiling.py,
│   │   noise_ceiling_matched.py, inspect_generation.py,
│   │   drug_specificity_in_data.py, pseudobulk_eval.py, spikein_metric_benchmark.py  (ablations/analysis)
│   ├── _test_eval_baselines.py, _test_fixed_panel.py                            (offline tests)
│   └── l1000_panel.json, l1000_landmark_genes.txt                              (reference inputs)
├── docs/
│   ├── dataset_construction.md
│   ├── results.md
│   └── drug_specificity_analysis_writeup.md
└── (data/, checkpoints/, eval_results/, hf_cache/ are generated and gitignored)
```

Everything Python sits **flat in `src/`** on purpose: the baselines and ablations do
`import evaluate_c2s_tahoe` (and some `import tahoe_c2s_preprocess`) as plain module imports, so
scoring is byte-for-byte identical everywhere. Splitting them into `baselines/`/`ablations/`
subpackages would break those imports and force a `sys.path` shim into nearly every file — §3 above
does the logical grouping instead. The two committed reference files and the offline tests live
alongside the code because the tests open the panel relative to the working directory.

---

## 5. How the pieces fit together

```
src/build_l1000_panel.py  ──→  src/l1000_panel.json   (the fixed 946-gene panel; committed)
                                        │
                                        ▼
src/tahoe_c2s_preprocess.py  ──→  data/{train.jsonl, eval_tier{1..4}.jsonl,
   (stream + control-match          l1000_panel.json, linear_model.json, held_out_drugs.json, ...}
    + fixed-panel sentences)
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                                ▼                               ▼
 src/train_c2s_tahoe.py         src/evaluate_c2s_tahoe.py         src/noise_ceiling_matched.py
 (SFT → checkpoint-10000)      ⭐ DE-Δr, K-sweep, baselines,      (replicate ceiling, per tier)
        │                       paired comparisons                       │
        │                                │        ▲                       │
        └───────────► eval_results/ ◄────┘        └── import metric fns ──┤
                      (per-tier results                src/scramble_eval.py, src/control_knn_baseline.py,
                       + paired deltas)                src/simple_baselines_and_consensus.py, ...
```

`evaluate_c2s_tahoe.py` exposes the functions everything else reuses — among them
`cell_sentence_to_gene_ranks`, `select_top_de_genes`, `select_top_expressed`, `delta_correlation`,
`compute_scalar_metrics`, `cluster_bootstrap_ci`, and `control_from_prompt`. Because the model,
baselines, and ablations all call these same functions, every number in the results is on the same
footing.

---

## 6. Metrics glossary (so results make sense)

- **Cell sentence** — a cell as an ordered list of gene symbols, highest-expressed first.
- **DE-Δr (headline)** — Pearson correlation between predicted and true *rank shift*
  (`treated_rank − control_rank`) over the top-K **d**ifferentially-**e**xpressed genes (the genes
  the drug moved most). Purely rank-space; the control baseline scores 0 by construction. **K-sweep**
  reports it at K = 20/50/100/200 so the headline K is not a magic number.
- **topN-expressed τ** — Kendall τ over the N most highly-expressed genes in the true cell;
  measures ordering quality on the genes that are "on," stripping the deterministic unexpressed
  tail that inflates whole-panel correlation.
- **panel τ** — Kendall τ over the whole 946-gene panel; *inflated* by the shared tail ordering,
  hence a diagnostic, not the headline.
- **Mean-shift baselines** — predict `control + average rank-shift`, grouped by nothing (global),
  cell line, MOA, or MOA×cell-line. Beating the strongest (MOA×cell-line) is the drug-specificity
  claim.
- **Noise ceiling** — the metric computed between two *real* replicate cells of the same condition;
  the irreducible upper bound no model can exceed. Reported as cell-vs-cell (single-cell truth) and
  cell-vs-consensus/pseudobulk (denoised truth).
- **Generalisation tiers** — Tier 1 seen conditions · Tier 2 unseen drugs · Tier 3 unseen
  drug×cell-line combos · Tier 4 held-out-dose interpolation.

---

## 7. Reproducing the pipeline (outline)

> Runs on an HPC cluster with GPU (single H200 used here). Paths below are placeholders.

```bash
# 0. Environment
pip install -r requirements.txt
# (transformers 5.x, datasets, torch cu124. Set HF_HOME + HF token.)

# 1. Build the fixed gene panel
python src/build_l1000_panel.py

# 2. Build the dataset (streams Tahoe from HuggingFace; CPU, long-running)
python src/tahoe_c2s_preprocess.py --num_shards 32 --rows_per_shard 400000 \
    --cells_per_condition 20 --held_out_drugs 50 --output_dir DATA

# 3. Fine-tune (GPU; ~1 H200-day to checkpoint-10000) — or submit src/sft_pythia_l1000.sbatch on SLURM
python src/train_c2s_tahoe.py --mode full --model_name vandijklab/C2S-Scale-Pythia-1b-pt \
    --train_file DATA/train.jsonl --eval_file DATA/eval_tier1_seen_conditions.jsonl \
    --output_dir CKPTS --max_length 8192 --bf16 --gradient_checkpointing \
    --batch_size 4 --grad_accum 4 --learning_rate 1e-5 --num_epochs 1 --keep_checkpoints 1

# 4. Evaluate (GPU; generation + K-sweep + baselines)
python src/evaluate_c2s_tahoe.py --model_path CKPTS/checkpoint-10000 --eval_dir DATA \
    --output_dir RESULTS/model --max_eval 300 --bf16 --topk_de_sweep 20,50,100,200 --min_coverage 0.2
#   ... add --baseline {global,per_cellline,per_moa,per_moa_cellline}_mean_shift --train_file DATA/train.jsonl
#   ... then --paired_compare --model_results ... --baseline_results ... for drug-specificity

# 5. Baselines (each scored through the same metric functions)
python src/control_knn_baseline.py           --eval_dir DATA
python src/control_state_baseline.py         --eval_dir DATA
python src/simple_baselines_and_consensus.py --mode linear    --eval_dir DATA
python src/simple_baselines_and_consensus.py --mode consensus --eval_dir DATA

# 6. Noise ceiling (CPU; from the eval files)
python src/noise_ceiling_matched.py --eval_dir DATA --out RESULTS/noise_ceiling_matched.json

# 7. Drug-specificity ablation (build scrambled prompts, then eval as in step 4)
python src/scramble_eval.py --eval_dir DATA --out_dir DATA_scram_diffmoa \
    --train_file DATA/train.jsonl --mode diff_moa --seed 42
python src/paired_by_position.py ...
```

Run scripts from the repo root (as above) or from inside `src/`. Use `--help` on any script for its
exact flags. Exact recipe values, cluster settings, and per-step numbers are in `docs/`.

---

## 8. Status & roadmap

**Done:** dataset (leak-free, 371k pairs) · SFT · evaluation harness (DE-Δr + K-sweep + guard) ·
baseline ladder through per-MOA×cell-line · per-tier condition-matched noise ceiling ·
base-model characterisation.

**In progress:** no-C2S (base-Pythia) SFT ablation + its validation audit · prompt-scramble
drug-specificity ablation.

**Planned:** expression-space metric bridge · external model comparison (scGPT, possibly STATE) on
matched splits · simple learned baselines (linear / tree) · **orthogonal drug-knowledge injection**
(the intended novel contribution) · GRPO.

This is **active research code**. `docs/` currently holds the dataset-construction and results
writeups; the no-C2S ablation audit and the living analysis roadmap mentioned above are tracked
outside this repo and not yet checked in. Treat `docs/` as the authoritative writeup for what's
there, and the code as the source of truth for exact behavior.

---

## 9. Notes for readers

- The evaluation philosophy deliberately follows the lessons of recent critical benchmarking work
  in drug-response prediction (mean-effect baselines, normalized/within-group metrics,
  clustered/pseudoreplication-aware statistics, leakage control, ablations). Numbers are reported
  with drug-clustered bootstrap CIs; "beats baseline" always means a paired comparison on matched
  cells, not a raw score difference.
- Results are from a mid-training checkpoint (~43% of one epoch); they are stable but not a
  converged final number.
- This is active research: some documents are living (roadmap, ablation audit) and will change as
  experiments complete.


## 1. Repository layout

Scripts are separated first by **data representation** (the current `[END_CELL]` work, the
superseded full-panel work, and the representation-agnostic core they share), and within each by
**pipeline phase** (`preprocess → train → eval → analysis`). Every top-level code folder has its own
README with a per-file table.

```
FINDINGS.md                 # results source of truth (Q→A)
README.md                   # this file — layout & descriptions only
requirements.txt

shared/                     # representation-agnostic core — used by BOTH pipelines (shared/README.md)
  evaluate_c2s_tahoe.py     #   the metric library imported by ~13 scripts (DE-Δr, τ, baselines, CIs)
  l1000_panel.json          #   fixed 946-gene panel (L1000 ∩ Tahoe) — the gene order everything uses
  l1000_landmark_genes.txt  #   raw L1000 landmark symbols the panel is built from
  build_l1000_panel.py      #   rebuilds the panel
  inspect_generation.py     #   debug helper for eyeballing generations

endcell/                    # current [END_CELL] pipeline (endcell/README.md)
  preprocess/               #   tahoe_c2s_preprocess_endcell_v2.py (current) + v1
  train/                    #   train_c2s_tahoe_endcell.py
  eval/                     #   evaluate_endcell.py, metric_grades_model_{endcell,v2}.py, make_scramble_endcell.py
  analysis/                 #   calibration_eval, nir_benchmark, spikein_metric_benchmark,
                            #   expression_space_discrimination, output_invariance,
                            #   mechanistic_drug_probe, causal_drug_probe, drug_specificity_in_data
  jobs/                     #   *.sbatch SLURM submit scripts

legacy_whole_panel/         # superseded full-panel pipeline — kept for provenance (legacy_whole_panel/README.md)
  preprocess/               #   tahoe_c2s_preprocess.py
  train/                    #   train_c2s_tahoe.py, grpo_c2s_tahoe.py, sft_pythia_l1000.sbatch
  eval/                     #   metric_grades_model.py, pseudobulk_eval, scramble_eval, noise_ceiling(_matched),
                            #   paired_by_position, regen_tier2_eval, spikein_metric_benchmark
  baselines/                #   control_knn_baseline, control_state_baseline, simple_baselines_and_consensus

tests/                      # offline self-tests (_test_eval_baselines.py, _test_fixed_panel.py)

docs/                       # detailed writeups, organized by regime (docs/README.md)
  methods/ legacy_l1000/ endcell/ proposals/
```

---

## 2. The two data representations

The project uses two cell-sentence representations. This is a factual distinction that determines
which folder a script belongs to — it is not a claim about results.

| Regime | Cell sentence | Status | Code | Docs |
|---|---|---|---|---|
| **[END_CELL] (current)** | expressed genes only, ranked, + an `[END_CELL]` sentinel | active | `endcell/` | `docs/endcell/` |
| **Legacy full-panel** | all 946 panel genes (expressed ranked, then unexpressed tail) | superseded | `legacy_whole_panel/` | `docs/legacy_l1000/` |

The `[END_CELL]` format matches C2S's own `generate_sentences()` (which emits only expressed genes),
so it is the representation the base model was pretrained on. The full-panel format is retained for
provenance and method-evolution context. See each folder's README for the per-file breakdown.

---

## 3. What each pipeline phase contains

- **`preprocess/`** — streams Tahoe-100M and builds (drug + dose + cell line) → treated-cell
  sentence pairs, the train/eval tiers, the panel, and the rank↔expression linear model.
- **`train/`** — supervised fine-tuning (and, in legacy, a GRPO experiment) of
  `C2S-Scale-Pythia-1b`.
- **`eval/`** — scores model predictions: standard DE-Δr / validity / baselines / ceiling, the
  forced-choice prediction graders, and the scrambled-drug eval builder.
- **`analysis/`** (endcell) — the standalone evaluation instruments. Each answers one question and
  ships with a synthetic `--selftest` so it can be validated with no GPU or data:

  | Script | Question it answers |
  |---|---|
  | `calibration_eval.py` | Which metrics are calibrated for this task? (Miller et al. DRF port) |
  | `nir_benchmark.py` | How does the model compare to baselines on the NIR metric, per held-out tier? |
  | `spikein_metric_benchmark.py` | Can a metric separate two real drug populations, and how gracefully under contamination? |
  | `expression_space_discrimination.py` | Does the same test behave differently on true expression vs ranks? |
  | `output_invariance.py` | Does swapping the drug in the prompt change the output more than resampling? |
  | `mechanistic_drug_probe.py` | Is the drug decodable from the model's internal activations, layer by layer? |
  | `causal_drug_probe.py` | Is the drug direction causally used during generation? |
  | `drug_specificity_in_data.py` | At what resolution (single-cell vs pseudobulk) does drug signal exist in the data? |

- **`baselines/`** (legacy) — non-model reference predictors (kNN-over-controls, control-state,
  mean-shift ladder / consensus).

---

## 4. How imports work across folders

Every script begins with a small `sys.path` bootstrap that adds `shared/` plus its sibling pipeline
subfolders to the path. So `import evaluate_c2s_tahoe` (shared), `import
expression_space_discrimination` (same pipeline), and `import tahoe_c2s_preprocess_endcell_v2`
(sibling subfolder) all resolve regardless of which subfolder the caller lives in, and regardless of
the working directory the job is launched from.

`evaluate_c2s_tahoe.py` is the single shared scoring core — model, baselines, and every ablation call
the same functions, so all numbers are computed on the same footing. It imports no local modules
(it is the leaf of the dependency graph).

SLURM jobs `cd ~/tahoe` and invoke scripts by their subpath, e.g. `endcell/eval/evaluate_endcell.py`.

---

## 5. Running / reproducing (outline)

Runs on an HPC cluster: a GPU for generation, CPU nodes for scoring and data streaming. Call the
environment's Python directly (`.../envs/c2s/bin/python`); `conda` is not on PATH in batch jobs.
Paths below are relative to the repo root (`~/tahoe/`); scripts resolve their own imports regardless
of the working directory.

```bash
# Build panel + dataset ([END_CELL] format)
python shared/build_l1000_panel.py
python endcell/preprocess/tahoe_c2s_preprocess_endcell_v2.py --num_shards 80 --cells_per_condition 30 --output_dir DATA_endcell_big

# Fine-tune (cold start, one epoch)
python endcell/train/train_c2s_tahoe_endcell.py --model_name vandijklab/C2S-Scale-Pythia-1b-pt --train_file DATA_endcell_big/train.jsonl ...

# Evaluate + run the instruments
python endcell/eval/evaluate_endcell.py --mode model,scramble,baselines,ceiling --eval_dir DATA_endcell_big ...
python endcell/analysis/calibration_eval.py --out RESULTS/calibration.json ...
python endcell/analysis/nir_benchmark.py --eval_dir DATA_endcell_big --model_path CKPT/final ...
```

Every instrument supports `--selftest` (synthetic, no GPU/data) — run it before any cluster job.
SLURM submit scripts live in `endcell/jobs/`. Use `--help` on any script for exact flags; exact
recipe values are in `docs/methods/dataset_construction.md`.

---

## 6. Glossary (definitions only)

- **Cell sentence** — a cell represented as a list of gene symbols, highest-expressed first
  (`[END_CELL]` format appends a sentinel and omits unexpressed genes).
- **DE-Δr** — correlation between predicted and true *rank shift* (`treated − control`) over the
  top-K differentially-expressed genes.
- **panel-τ** — Kendall τ over all 946 panel genes (sensitive to how absent genes are placed).
- **NIR (normalized inverse rank)** — a discrimination metric: how the distance from a prediction to
  its own drug's profile ranks against its distances to other drugs' profiles.
- **DRF (dynamic range fraction)** — a meta-metric (Miller et al.) that grades whether a *metric*
  rewards real signal, using perfect / uninformed / noise-ceiling reference predictors.
- **Noise ceiling** — a metric computed between two real replicate cells of the same condition; the
  interpretable upper bound for that metric.
- **Absent-gene conventions** — `worst` (absent genes → bottom rank) vs `francesca` (fixed
  mid-rank); reported side by side where relevant.
- **Generalization tiers** — tier1 seen · tier2 unseen drugs · tier3 unseen drug×cell-line combos ·
  tier4 dose interpolation.

---

*This is active research code. `FINDINGS.md` holds the current results and marks what is still
pending; treat the code as the source of truth for exact behavior.*

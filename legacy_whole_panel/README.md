# `legacy_whole_panel/` — the original full-panel pipeline (superseded)

Cell sentences = **all 946 panel genes** (expressed genes ranked, then the unexpressed genes as a
tail). This was the first representation. It is **kept for method-evolution context and
reproducibility, not for reporting** — the thesis reports the `endcell/` results.

**Why we pivoted:** C2S's real `generate_sentences()` emits only *expressed* genes, so the
full-panel format didn't match the base model's pretraining. It also made the two advisor-proposed
absent-gene treatments identical (every gene was "present"), so they couldn't be compared. See
`docs/legacy_l1000/` for the historical results and `docs/README.md` for the pivot note.

Folders follow **preprocess → train → eval → baselines**:

### `preprocess/`
| File | Role |
|---|---|
| `tahoe_c2s_preprocess.py` | Original full-panel dataset constructor (imported by `noise_ceiling.py` and `regen_tier2_eval.py` for their exact builders). |

### `train/`
| File | Role |
|---|---|
| `train_c2s_tahoe.py` | Original SFT on full-panel pairs. |
| `grpo_c2s_tahoe.py` | GRPO / RL fine-tuning experiment (full-panel era). |
| `sft_pythia_l1000.sbatch` | SLURM submit for the original SFT + eval. |

### `eval/`
| File | Role |
|---|---|
| `metric_grades_model.py` | v1 forced-choice prediction grading (full-panel; no `[END_CELL]` handling — superseded by `endcell/eval/metric_grades_model_v2.py`). |
| `pseudobulk_eval.py` | Pseudobulk-level evaluation. |
| `scramble_eval.py` | Original scramble ablation. |
| `noise_ceiling.py` / `noise_ceiling_matched.py` | Replicate noise-ceiling measurement (matched-depth variant). |
| `paired_by_position.py` | Position-paired control/treated construction helper. |
| `regen_tier2_eval.py` | Rebuilds the Tier-2 (unseen-drug) eval file per-drug without a full preprocess rerun. |
| `spikein_metric_benchmark.py` | Full-panel version of the spike-in benchmark (the current one is in `endcell/analysis/`). |

### `baselines/`
| File | Role |
|---|---|
| `control_knn_baseline.py` | k-NN-over-controls baseline. |
| `control_state_baseline.py` | Control-state (predict-the-control) baseline. |
| `simple_baselines_and_consensus.py` | Mean-shift ladder + consensus baselines. |

**Imports.** Same bootstrap as `endcell/`: each script adds `shared/` + all sibling
`legacy_whole_panel/*` subfolders to `sys.path`, so `import evaluate_c2s_tahoe` (shared) and
`import tahoe_c2s_preprocess` (this pipeline's `preprocess/`) both resolve.

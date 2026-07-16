# `endcell/` — the current [END_CELL] pipeline (reportable results)

Cell sentences = **expressed genes only, ranked, + an `[END_CELL]` sentinel**, matching C2S's own
`generate_sentences()`. This is the representation all reported thesis results are on. Data:
`data_diverse2_endcell_big` (~675k pairs); model: `pythia_sft_endcell/final` (cold-started, one epoch).

Folders follow the pipeline order **preprocess → train → eval → analysis**:

### `preprocess/`
| File | Role |
|---|---|
| `tahoe_c2s_preprocess_endcell_v2.py` | **Current** dataset constructor: streams Tahoe-100M, builds `[END_CELL]` (drug+dose+cell-line → treated-cell) pairs, train/eval tiers, panel + linear rank↔expression model. |
| `tahoe_c2s_preprocess_endcell.py` | v1 of the above (kept for provenance). |

### `train/`
| File | Role |
|---|---|
| `train_c2s_tahoe_endcell.py` | SFT of `C2S-Scale-Pythia-1b-pt` on the `[END_CELL]` pairs. |

### `eval/` — model scoring
| File | Role |
|---|---|
| `evaluate_endcell.py` | Standard eval: DE-Δr / validity / baselines / ceiling under both absent-gene conventions; `model`/`scramble`/`baselines`/`ceiling` modes. |
| `metric_grades_model_endcell.py` | Forced-choice **prediction grading** at pseudobulk (is the model's drug-A prediction closer to real drug A than to a different drug B?). Chance 0.50; the ~0.48 result. |
| `metric_grades_model_v2.py` | Airtight grading variant: 3 absent-gene representations + temperature-sampled pseudobulk + scramble arm. |
| `make_scramble_endcell.py` | Builds the scrambled-drug eval tiers (swap drug+MOA in the prompt, keep control + truth) for the scramble ablation. |

### `analysis/` — the evaluation instruments (the methodological contribution)
Each is standalone with a synthetic `--selftest` (verify-before-GPU discipline).

| File | Question it answers |
|---|---|
| `calibration_eval.py` | **Which metrics are calibrated?** Miller et al. DRF port — grades metrics (not the model). NIR calibrated at DRF +0.80; rank prediction metrics inverted. |
| `nir_benchmark.py` | **Does the model beat baselines on the calibrated metric?** Model vs linear/mean/ceiling on NIR (rank + expr distance), held-out tiers. Model ≈ chance ≈ linear. |
| `spikein_metric_benchmark.py` | **Can a metric tell two real drug populations apart?** Federico's spike-in / titrated contamination. ~0.95–0.99 for real drugs. |
| `expression_space_discrimination.py` | **Is drug-blindness a rank artifact?** Same discrimination on *true* expression streamed from Tahoe. No — still blind per cell. |
| `output_invariance.py` | **Does swapping the drug change the output more than resampling?** Gap ≈ 0.000. |
| `mechanistic_drug_probe.py` | **Does the drug reach the representation and survive through layers?** Decodable at 76–82%. |
| `causal_drug_probe.py` | **Is the drug direction causally used in generation?** (steering probe) |
| `drug_specificity_in_data.py` | **Does the drug signal exist in the data at all, and at what resolution?** Single-cell vs pseudobulk. Representation-agnostic data-level check. |

### `jobs/` — SLURM submit scripts
`endcell_gpu.sbatch` (model + scramble + causal probe), `endcell_cpu.sbatch` (baselines/ceiling +
expression-space), `expr_space.sbatch`, `output_invariance.sbatch`. All `cd ~/tahoe` and invoke
scripts by their new subpaths (e.g. `endcell/eval/evaluate_endcell.py`).

**Imports.** Each script's path bootstrap adds `shared/` plus all sibling `endcell/*` subfolders to
`sys.path`, so `import evaluate_c2s_tahoe`, `import expression_space_discrimination`, and
`import tahoe_c2s_preprocess_endcell_v2` all resolve no matter which subfolder the caller is in.

## What this is

An MSc thesis project on **drug-perturbation prediction at single-cell resolution**. The starting
point is `C2S-Scale-Pythia-1b` — a language model that represents a cell as a "cell sentence" (its
genes listed in order of expression) — fine-tuned on **Tahoe-100M**, a ~100M-cell dataset of ~1,100
drugs across ~50 cancer cell lines. The task: given a control cell and a drug, predict that cell's
treated state.

The work has two halves. The first is **measurement** — building instruments that can tell whether a
model actually uses the drug, rather than reproducing the generic transcriptional response that every
drug triggers. That turned out to matter a great deal: several standard metrics reward drug-agnostic
predictors, and batch structure in the data can masquerade as drug signal, so most of the analysis
code here exists to separate those. The second half is **intervention** — a series of controlled arms
(denoised targets, optimal-transport targets, chemical-structure injection, and a change to how the
prediction target is encoded) that ask what, if anything, makes the model use the drug.

`FINDINGS.md` is the results record: one entry per question, with the methodology, the numbers, and
what has been retracted or superseded. `thesis/` holds the write-up itself. This file is only about
where the code lives.

---

## 1. Repository layout

Scripts are separated first by **data representation** (the current `[END_CELL]` work, the superseded
full-panel work, and the representation-agnostic core they share), and within each by **pipeline
phase** (`preprocess → train → eval → analysis`). Most top-level code folders have their own README
with a per-file table.

```
FINDINGS.md                 # results source of truth (Q→A)
README.md                   # this file — layout & descriptions only
requirements.txt

shared/                     # representation-agnostic core — used by BOTH pipelines (shared/README.md)
  evaluate_c2s_tahoe.py     #   the metric library imported by ~13 scripts (DE-Δr, τ, baselines, CIs)
  inference.py              #   the one place uncertainty is computed — clustered, two-way, multiway
  tahoe_design.py           #   what a Tahoe (drug, cell_line, plate, dose) key does and does not mean
  l1000_panel.json          #   fixed 946-gene panel (L1000 ∩ Tahoe) — the gene order everything uses
  l1000_landmark_genes.txt  #   raw L1000 landmark symbols the panel is built from
  build_l1000_panel.py      #   rebuilds the panel
  inspect_generation.py     #   debug helper for eyeballing generations

endcell/                    # current [END_CELL] pipeline (endcell/README.md)
  preprocess/               #   tahoe_c2s_preprocess_endcell_v2.py (current) + v1
                            #   build_consensus_targets.py — denoised pseudobulk targets (Arm 1a)
  train/                    #   train_c2s_tahoe_endcell.py
  eval/                     #   evaluate_endcell.py, metric_grades_model_{endcell,v2}.py,
                            #   make_scramble_endcell.py
  analysis/                 #   the standalone instruments (table in §3)
  ot/                       #   optimal-transport + residual target pipeline (§4)
  plate_control/            #   the plate/batch-leak audit: scripts + A/B tables (plate_control/README.md)
  jobs/                     #   *.sbatch SLURM submit scripts

legacy_whole_panel/         # superseded full-panel pipeline — kept for provenance
  preprocess/ train/ eval/ baselines/

tests/                      # offline self-tests, incl. the split-before-fit release gate
                            #   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q

thesis/                     # the write-up (thesis/build.sh → thesis.pdf)
  main_v4.tex               #   document root
  Sections/Investigation-v4.tex   #   the investigation, told in the order it happened
  archive/                  #   superseded drafts, kept for provenance

RESULTS_cluster/            # the artifacts every quoted number is regenerable from (JSON/CSV tracked;
                            #   .npz activation profiles are not)

docs/                       # detailed writeups, organized by regime (docs/README.md)
  methods/ legacy_l1000/ endcell/ proposals/
```

---

## 2. The two data representations

Two cell-sentence formats exist in the repo. This is a factual distinction that determines which
folder a script belongs to — it is not a claim about results.

| Regime | Cell sentence | Status | Code | Docs |
|---|---|---|---|---|
| **[END_CELL] (current)** | expressed genes only, ranked, + an `[END_CELL]` sentinel | active | `endcell/` | `docs/endcell/` |
| **Legacy full-panel** | all 946 panel genes (expressed ranked, then unexpressed tail) | superseded | `legacy_whole_panel/` | `docs/legacy_l1000/` |

A third target encoding was added later for the objective-side work: a **residual signature**
(`<up genes> [DOWN] <down genes> [END_CELL]`), built by `endcell/ot/build_residual_targets.py`. It
reuses the same panel and prompt format — only the response changes.

---

## 3. The analysis instruments

Each script in `endcell/analysis/` answers one question and ships with a synthetic `--selftest`, so it
can be validated with no GPU and no data. Run the self-test before any cluster job.

| Script | Question it answers |
|---|---|
| `calibration_eval.py` | Which metrics are calibrated for this task? (Miller et al. DRF port) |
| `nir_benchmark.py` | How does the model compare to baselines on NIR, per held-out tier? Includes the scramble arm, the control-copy leak baseline, and the train-only drug-lookup baseline |
| `spikein_metric_benchmark.py` | Can a metric separate two real drug populations, and how gracefully under contamination? |
| `expression_space_discrimination.py` | Does the same test behave differently on true expression vs ranks? |
| `output_invariance.py` | Does swapping the drug in the prompt change the output more than resampling? |
| `drug_specificity_in_data.py` | At what resolution (single-cell vs pseudobulk) does drug signal exist in the data? |
| `drug_biology_atlas.py` | Which drugs are potent / inert / redundant, and how does that vary by cell line? |
| `drug_stratify_geometry.py` | Per-drug stratified NIR + the causal verdict (scramble vs control-copy) |
| `perturbation_strength.py` | E-distance + permutation test: which perturbations are real, without circularity |
| `mechanistic_drug_probe.py` | Is the drug decodable from the model's internal activations, layer by layer? |
| `causal_drug_probe.py` | Is the drug direction causally used during generation? (superseded by `workspace_probe`) |
| `workspace_probe.py` | Is drug identity encoded in a subspace the generation readout ignores? (ablation + activation swap, in logit space) |
| `scramble_distance_sweep.py` | Is the scramble null real, or an artifact of swapping to *similar* drugs? |
| `sar_gate.py` | Does chemical structure predict drug response at all? (Morgan + MolFormer, against a noise ceiling) |
| `target_divergence.py` | How much of the training target actually differs between two drugs? |
| `residual_eval.py` | Does the residual-trained model use the drug? (stratified scramble, three-way holdout) |
| `reconstructed_eval.py` | The same question in the original full-profile space, for comparability |
| `check_dose_coverage.py` | Did the dose-blind sampling cap reduce dose diversity? (read-only) |
| `leak_audit.py` | Train/eval contamination checks |
| `make_provenance.py` | Bundles results + fingerprints dataset/checkpoint into a committable manifest |
| `scramble_stratum_audit.py` | Is the scramble comparator actually a null? Recomputes every split gap against the neutral stratum, from saved records, with no GPU |
| `channel_gate.py` | Is unseen-drug generalisation reachable through *any* drug-side channel — protein target, mechanism, chemical structure? |
| `variance_decomposition.py` | How much of the drug-specific residual is shared across contexts, and how much is not? (transfer coefficient, energy shares, dose axis) |
| `experimental_unit_audit.py` | What is the unit of independent treatment assignment in this atlas, and does the holdout respect it? |
| `dose_response_analysis.py` | Do the doses of one drug behave as a dose–response, or as unrelated conditions? |
| `drug_difficulty_atlas.py` | Which drugs are hard for reasons other than the model — target quality, cell count, redundancy |
| `kappa_channel.py` | Is there learnable structure in the interaction component, or only in the main effect? |
| `leak_magnitude.py` | How much does a transductively-fitted target inflate a score, measured rather than argued |
| `reward_calibration.py` | Could this reward distinguish two real drugs? (the precondition for any RL arm) |
| `aggregate_workspace_probe.py` | Collects the per-arm, per-layer probe jobs into one multiplicity-corrected table |
| `artifact_manifest.py` | Which quoted results have a committed artifact behind them, and which do not |
| `build_thesis_assets.py` | Emits `thesis/generated/numbers.tex` — a number with no backing artifact cannot be typeset |

---

## 4. The `endcell/ot/` pipeline

Built for the optimal-transport arm and then reused for the residual-target work. It streams Tahoe
into a cached, analysis-ready form and produces training targets from it.

| Script | Role |
|---|---|
| `scout_data.py` | Read-only: what does the Tahoe repo actually contain (embeddings? schema? panel overlap?) |
| `download_scvi_adata.sbatch`, `inspect_scvi_adata.py` | Fetch and inspect the released Tahoe scVI model/embeddings |
| `scvi_encode.py` | Encode cells with the released scVI model (superseded by the JOIN path below) |
| `build_embeddings.py` | Streams Tahoe → per-cell 946-gene panel expression (CP10K) + PCA + metadata + barcode. Two passes, because DMSO controls do not co-locate with treated cells |
| `join_latent.py` | Attaches the shipped scVI latent to the cache by barcode |
| `build_ot_targets.py` | Sinkhorn coupling control↔treated → T0/T1/T2 target ladder + Step-0 gates |
| `build_residual_targets.py` | Drug-specific residual targets (signed DE signature) + the tier-aligned holdout manifest |
| `ot_*.sbatch`, `ot_train.sbatch`, `ot_eval.sbatch` | SLURM wrappers for the above |

---

## 5. How imports work across folders

Every script begins with a small `sys.path` bootstrap that adds `shared/` plus its sibling pipeline
subfolders to the path. So `import evaluate_c2s_tahoe` (shared), `import
expression_space_discrimination` (same pipeline), and `import tahoe_c2s_preprocess_endcell_v2`
(sibling subfolder) all resolve regardless of which subfolder the caller lives in, and regardless of
the working directory the job is launched from.

`evaluate_c2s_tahoe.py` is the single shared scoring core — model, baselines, and every ablation call
the same functions, so all numbers are computed on the same footing. It imports no local modules
(it is the leaf of the dependency graph). `shared/inference.py` plays the same role for uncertainty:
every interval in the project comes from it, so an estimator improvement lands everywhere at once
rather than in whichever script was edited last.

Note that the cluster copy is a **flat** layout (`~/tahoe/<script>.py`), while this repo is nested.
Scripts handle both; SLURM files use flat paths.

---

## 6. Running / reproducing (outline)

Runs on an HPC cluster: a GPU for generation, CPU nodes for scoring and data streaming. Call the
environment's Python directly (`.../envs/c2s/bin/python`); `conda` is not on PATH in batch jobs.

```bash
# Build panel + dataset ([END_CELL] format)
python shared/build_l1000_panel.py
python endcell/preprocess/tahoe_c2s_preprocess_endcell_v2.py --num_shards 80 --cells_per_condition 30 --output_dir DATA_endcell_big

# Fine-tune (cold start, one epoch)
python endcell/train/train_c2s_tahoe_endcell.py --model_name vandijklab/C2S-Scale-Pythia-1b-pt --train_file DATA_endcell_big/train.jsonl ...

# Evaluate + run the instruments
python endcell/eval/evaluate_endcell.py --mode model,scramble,baselines,ceiling --eval_dir DATA_endcell_big ...
python endcell/analysis/nir_benchmark.py --eval_dir DATA_endcell_big --model_path CKPT/final --same_plate_only ...

# The OT / residual-target route
python endcell/ot/build_embeddings.py --panel l1000_panel.json --out_dir ot_cache ...
python endcell/ot/join_latent.py --cache_dir ot_cache --adata SCVI_MODEL/adata.h5ad
python endcell/ot/build_residual_targets.py --cache_dir ot_cache --out_dir residual_targets
python endcell/analysis/residual_eval.py --cache_dir ot_cache --model_path CKPT/final ...
```

```bash
# Build the write-up
./thesis/build.sh          # → thesis/thesis.pdf
```

Use `--help` on any script for exact flags; recipe values are in `docs/methods/dataset_construction.md`.
Two conventions worth knowing before quoting any number: discrimination results are computed
**within-plate** (`--same_plate_only`) because drug and plate are partially confounded by the
experimental design, and generation-based evals need a token budget large enough for the full
sentence (a short budget silently truncates and halves the measured effect).

---

## 7. Glossary (definitions only)

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
- **Scramble arm** — the same control cell with a different drug named in the prompt. `model −
  scramble` is the leak-immune test of whether the drug is used at all; its sensitivity depends on how
  *dissimilar* the swapped drug is, so it is reported stratified by response similarity.
- **Control-copy** — a predictor that returns the plate-matched control unchanged; it carries zero
  drug information, so anything it scores above chance is leakage.
- **Residual (drug-specific)** — `(treated − control) − mean-over-drugs(treated − control)`: what a
  drug does beyond the generic response every drug triggers.
- **Absent-gene conventions** — `worst` (absent genes → bottom rank) vs `francesca` (fixed
  mid-rank); reported side by side where relevant.
- **Generalization tiers** — tier1 seen · tier2 unseen drugs · tier3 unseen drug×cell-line combos ·
  tier4 dose interpolation.
- **Well** — one physical container. In Tahoe a well receives one drug at one dose and holds ~49 cell
  lines at once, so the well, not the (drug, cell line) pair, is the unit of treatment assignment.
- **Comparison set** — the conditions a prediction is ranked against when NIR is computed. NIR's
  chance value of 0.5 is a statement about this set and nothing else.
- **Two-way cluster-robust** — Cameron–Gelbach–Miller: conditions are crossed by cell line and well,
  so neither alone is the unit of independence. Reported with a Student-*t* reference on the smaller
  cluster count, because a cluster-robust variance is asymptotic in clusters, not observations.
- **Multiway / dyadic variance** — Fafchamps–Gubert, for statistics whose observations are *pairs*
  and therefore belong to two groups at once; the two-way estimator does not apply there.
- **Split-before-fit** — the holdout is assigned from metadata before any target is estimated, so no
  held-out condition can influence the quantity it is later scored against.

---

*Active research code. `FINDINGS.md` holds the current results and marks what is still pending;
treat the code as the source of truth for exact behavior.*

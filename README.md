# C2S-Scale Perturbation Prediction on Tahoe-100M

Fine-tuning a Cell2Sentence-Scale language model to predict single-cell drug-perturbation
responses — and a rigorous, leakage-controlled characterization of **why it captures the general
perturbation response but not the drug**. MS thesis project (Bocconi, Buffa Lab; defense Sept 2026).

**One-line summary:** given a control (untreated) cell's expression profile and a drug, the model
predicts the treated cell's transcriptional response as a *cell sentence* (genes ordered by
expression). We evaluate it against biologically meaningful baselines and a measured noise ceiling,
and build evaluation instruments to test *what the model actually learns about the drug*.

> **📊 Results live in [`FINDINGS.md`](FINDINGS.md)** — the single source of truth, in
> Question→Answer form. Detailed writeups are in [`docs/`](docs/README.md), organized by data regime
> (see the pivot note below). If any document disagrees with `FINDINGS.md`, `FINDINGS.md` wins.

---

## The headline result: "read but not used"

The fine-tuned model predicts single-cell perturbation responses **near the replicate noise
ceiling**, yet it is **drug-blind** — it cannot distinguish one drug from another. These are not
contradictory; they measure different things:

- **It predicts the *generic* perturbation response well** by conditioning on the control cell's
  state (DE-Δr near the ~0.76 ceiling).
- **It ignores the drug**: its prediction for drug A is no closer to drug A's real response than to
  a random other drug's (forced-choice grading ≈ 0.48 ≈ chance), and **swapping the drug in the
  prompt changes the output no more than resampling does** (output-invariance gap ≈ 0.000).

We show this is **not** an artifact of the metric (it discriminates real drugs at ~0.95), the
representation (true expression is equally blind per cell), or the absent-gene convention (both
advisors' schemes give identical numbers). Mechanistically, the drug **is** read into the
representation (linearly decodable at 76–82%) but the generation **ignores** it — because at
single-cell resolution the drug barely moves the target, so the training objective gives no
incentive to use it. The signal is real but only resolvable under aggregation.

This extends the **DrEval** critique (Nat Commun 2026 — bulk drug-response models fail to generalize
to unseen drugs) into the single-cell, transformer, post-treatment-profile regime, and contributes
reusable single-cell **evaluation instruments** that demonstrate the failure.

---

## 1. What this project does

Single-cell drug-perturbation prediction: *how does a cell's transcriptome change when exposed to a
drug?* We fine-tune `C2S-Scale-Pythia-1b` on the **Tahoe-100M** screen (~100M single cells, ~379
drugs × ~50 cancer cell lines) and ask whether it predicts responses that (a) generalize to unseen
drugs/contexts and (b) are **drug-specific** rather than generic.

The **cell-sentence** representation encodes each cell as an ordered list of gene symbols,
highest-expressed first. The model reads a prompt (drug, cell line, dose, mechanism, control-cell
sentence) and generates the treated cell's sentence.

### Core design principles
- **Leakage-free fixed panel.** All cells are represented over one fixed 946-gene panel (L1000 ∩
  Tahoe), so the model never chooses the gene set from the answer.
- **Rank-based headline metric (DE-Δr).** Prediction quality is scored on the genes the drug
  actually *moves*, in rank space, avoiding whole-transcriptome correlation inflation.
- **A baseline ladder, not a single baseline.** Mean-shift baselines of increasing strength (global
  → per-cell-line → per-MOA → per-MOA×cell-line) test whether the model beats "predict the
  class/context average" — the failure mode that inflates most perturbation results.
- **A measured noise ceiling.** Two real replicate cells of the same condition don't agree
  perfectly; we measure that irreducible ceiling so absolute scores are interpretable.
- **Instruments over point scores.** Because "high DE-Δr" turned out not to imply drug knowledge, we
  built forced-choice discrimination, prediction-grading, output-invariance, expression-space, and
  causal-steering instruments to interrogate *what the model uses*.

---

## 2. The [END_CELL] pivot (important for reading the results)

The project uses two data representations. **Only the current one is reported**; the older one is
kept as method-evolution context.

| Regime | Cell sentence | Role | Location |
|---|---|---|---|
| **Legacy L1000 (full-panel)** | all 946 genes (expressed ranked + unexpressed tail) | background — *why we pivoted* | `docs/legacy_l1000/` |
| **[END_CELL] (current)** | expressed genes only + `[END_CELL]` sentinel | **the reportable results** | `docs/endcell/` |

We moved to [END_CELL] after confirming C2S's real `generate_sentences()` emits only expressed
genes. The full-panel format also made the two advisor-proposed absent-gene treatments identical
(every gene was present), so they couldn't be compared. Current data = `data_diverse2_endcell_big`
(675k pairs); current model = `pythia_sft_endcell/final` (cold-started, full epoch).

---

## 3. Key results (current, [END_CELL]) — summary; full detail in [`FINDINGS.md`](FINDINGS.md)

- **Drug-blind (established):** prediction-grading ≈ 0.48 ≈ scramble (real-drug ceiling 0.67–0.83);
  output-invariance gap ≈ 0.000; true-expression per-cell discrimination ≈ 0.53 (rank is not the
  culprit); drug decodable from the representation at 76–82% but attenuated toward the output.
- **The metric works:** spike-in discrimination of real drug populations ≈ 0.95–0.99 — the
  bottleneck is the model/resolution, not the metric.
- **Absent-gene convention is a non-issue:** Federico's worst-rank and Francesca's fixed mid-rank
  give ceiling/baseline/spike-in numbers within ≤0.01 of each other.
- **Noise ceiling ≈ 0.76** (cell-vs-cell DE-Δr, [END_CELL], all tiers); baseline ladder shows
  cell-line identity is informative but **mechanism (MOA) adds nothing** over global.
- **Corrected framing:** the model beats the per-MOA×cell-line baseline via **control-conditioning
  (per-cell tailoring), not drug-specificity** — an earlier overclaim, now corrected.
- **In progress:** the [END_CELL] model DE-Δr eval (the "competent" half), the causal steering
  probe, and the combined rank-vs-expression comparison. See the pending queue in `FINDINGS.md`.

---

## 4. Evaluation instruments (the methodological contribution)

Each is a standalone script with a synthetic `--selftest` (verify-before-GPU discipline):

| Instrument | Question it answers | Script |
|---|---|---|
| Metric calibration (DRF) | Which metrics actually reward drug signal? (Miller et al. port) | `endcell/analysis/calibration_eval.py` |
| NIR benchmark | Does the model beat baselines on the *calibrated* metric? | `endcell/analysis/nir_benchmark.py` |
| Spike-in discrimination | Can a metric tell two real drug populations apart? | `endcell/analysis/spikein_metric_benchmark.py` |
| Prediction grading | Is the model's *prediction* closer to the right drug than a wrong one? | `endcell/eval/metric_grades_model_v2.py` |
| Output invariance | Does swapping the drug change the output more than resampling? | `endcell/analysis/output_invariance.py` |
| Expression-space discrimination | Is the drug-blindness a rank-representation artifact? | `endcell/analysis/expression_space_discrimination.py` |
| Mechanistic probe | Does the drug enter the representation, and survive through layers? | `endcell/analysis/mechanistic_drug_probe.py` |
| Causal steering | Is the drug direction *causally used* in generation? | `endcell/analysis/causal_drug_probe.py` |
| Standard eval ([END_CELL]) | DE-Δr / validity / baselines / ceiling under both conventions | `endcell/eval/evaluate_endcell.py` |

---

## 5. Repository structure

Scripts are separated by **data representation** — the current `[END_CELL]` work, the superseded
full-panel work, and the representation-agnostic core they share — and within each, by pipeline
phase (`preprocess → train → eval → analysis`). Each top-level folder has its own README.

```
FINDINGS.md                 # ⭐ results source of truth (Q→A)
README.md                   # this file
requirements.txt

shared/                     # representation-agnostic core (see shared/README.md)
  evaluate_c2s_tahoe.py     #   ⭐ metric library imported by BOTH pipelines (DE-Δr, τ, baselines, CIs)
  l1000_panel.json, l1000_landmark_genes.txt, build_l1000_panel.py, inspect_generation.py

endcell/                    # ⭐ current [END_CELL] pipeline — the reportable results (endcell/README.md)
  preprocess/               #   tahoe_c2s_preprocess_endcell_v2.py (+ v1)
  train/                    #   train_c2s_tahoe_endcell.py
  eval/                     #   evaluate_endcell.py, metric_grades_model_{endcell,v2}.py, make_scramble_endcell.py
  analysis/                 #   calibration_eval, nir_benchmark, spikein, expression_space, output_invariance,
                            #   mechanistic_drug_probe, causal_drug_probe, drug_specificity_in_data
  jobs/                     #   *.sbatch SLURM submit scripts (endcell_gpu, endcell_cpu, expr_space, ...)

legacy_whole_panel/         # superseded full-panel pipeline — kept for provenance (legacy_whole_panel/README.md)
  preprocess/ train/ eval/ baselines/

tests/                      # offline self-tests (_test_eval_baselines.py, _test_fixed_panel.py)

docs/                       # detailed writeups, organized by regime (see docs/README.md)
  methods/ legacy_l1000/ endcell/ proposals/
```

**How imports work across the folders.** Every script begins with a small path bootstrap that adds
`shared/` plus its sibling pipeline subfolders to `sys.path`. So `import evaluate_c2s_tahoe` (shared),
`import expression_space_discrimination` (same pipeline), and `import tahoe_c2s_preprocess_endcell_v2`
(sibling subfolder) all resolve no matter which subfolder the caller lives in, and regardless of the
working directory the job is launched from. `evaluate_c2s_tahoe.py` is the single shared scoring core
— model, baselines, and every ablation call the same functions, so all numbers are on the same
footing. SLURM jobs still `cd ~/tahoe` and invoke scripts by their subpath
(e.g. `endcell/eval/evaluate_endcell.py`).

---

## 6. Metrics glossary

- **Cell sentence** — a cell as an ordered list of gene symbols, highest-expressed first (+`[END_CELL]`).
- **DE-Δr (headline)** — Pearson (and Spearman) correlation between predicted and true *rank shift*
  (`treated − control`) over the top-K differentially-expressed genes. Control-as-prediction scores 0.
- **panel-τ** — Kendall τ over all 946 genes; sensitive to how absent genes are placed (hence
  reported under both the worst-rank and fixed-mid-rank conventions).
- **Noise ceiling** — the metric between two *real* replicate cells of the same condition
  (cell-vs-cell = single-cell truth; cell-vs-consensus = denoised). The interpretable upper bound.
- **Absent-gene conventions** — `worst` (Federico: absent → bottom) vs `francesca` (fixed mid-rank).
- **Generalization tiers** — tier1 seen · tier2 unseen drugs · tier3 unseen drug×cell-line combos ·
  (tier4 dose interpolation, legacy).

---

## 7. Reproducing (outline)

Runs on an HPC cluster with a single H200 GPU (generation) and CPU nodes (scoring/streaming). Call
the environment's Python directly (`.../envs/c2s/bin/python`); `conda` is not on PATH in batch jobs.

Paths below are relative to the repo root (`~/tahoe/` on the cluster); scripts resolve their own
imports regardless of the working directory.

```bash
# Build panel + dataset ([END_CELL] format)
python shared/build_l1000_panel.py
python endcell/preprocess/tahoe_c2s_preprocess_endcell_v2.py --num_shards 80 --cells_per_condition 30 --output_dir DATA_endcell_big

# Fine-tune (cold start, one epoch)
python endcell/train/train_c2s_tahoe_endcell.py --model_name vandijklab/C2S-Scale-Pythia-1b-pt --train_file DATA_endcell_big/train.jsonl ...

# Evaluate (both absent-gene conventions) + drug-blindness instruments
python endcell/eval/evaluate_endcell.py --mode model,scramble,baselines,ceiling --eval_dir DATA_endcell_big ...
python endcell/analysis/calibration_eval.py --out RESULTS/calibration.json ...
python endcell/analysis/nir_benchmark.py --eval_dir DATA_endcell_big --model_path CKPT/final ...
python endcell/analysis/expression_space_discrimination.py --panel_file DATA_endcell_big/l1000_panel.json ...
```

Every instrument supports `--selftest` (synthetic, no GPU/data) — run it before any cluster job.
SLURM submit scripts live in `endcell/jobs/` (`endcell_gpu.sbatch`, `endcell_cpu.sbatch`, ...) and
bundle the jobs. `--help` on any script for exact flags; exact recipe values are in
`docs/methods/dataset_construction.md`.

---

## 8. Status & roadmap

**Done:** [END_CELL] dataset + cold-start SFT · shared metric library · spike-in metric benchmark ·
prediction-grading · output-invariance · expression-space (true-value) discrimination · mechanistic
probe · noise ceiling + baseline ladder ([END_CELL], both conventions).

**In progress:** [END_CELL] standard eval (model DE-Δr — the "competent" half) · causal steering
probe · combined rank-vs-expression comparison.

**Next phase:** task-ceiling analysis (is the task winnable by any method?) → PubChem drug-knowledge
injection → GRPO / auxiliary-loss (does *forcing* the objective make the model use the drug?) →
external-model generality (STATE / scGPT) → thesis/advisor writeup.

**Legacy (superseded, kept as background):** full-panel results in `docs/legacy_l1000/`.

---

## 9. Notes for readers

- The evaluation philosophy follows recent critical benchmarking work (mean-effect baselines,
  within-group metrics, clustered/pseudoreplication-aware statistics, leakage control, ablations).
  "Beats baseline" always means a paired comparison on matched cells, not a raw score difference.
- This is active research code. `FINDINGS.md` reflects current verdicts and marks what's pending;
  treat the code as the source of truth for exact behavior.

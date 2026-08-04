# Runbook — in order, with what to watch

**Two numbering schemes exist and they collide. This one wins for anything you run.**

This runbook numbers CLUSTER JOBS: 0 scp, 1 artifacts, 2 unitaudit, 3 chgate, 4 rebuild, 5 retrain.
`docs/proposals/remediation_execution_scope.md` numbers REMEDIATION WORK ITEMS 0-10, most of which are
code rather than jobs. The numbers overlap and do not mean the same thing. Mapping:

| scope step | what it is | where it lives here |
|---|---|---|
| 0 remove false text | writing | done |
| 1 design layer | code | `shared/tahoe_design.py` |
| 2 unit audit | code + job | **job 2** |
| 3 builder patches | code | inside **job 4** |
| 4 split-before-fit | code | inside **job 4** |
| 5 retrain | job | **job 5** |
| 6 evaluator fixes | code | inside **job 5**'s eval |
| 7 recompute what the dose fix voided | code, then a job | **not written** |
| 8 harden metric calibration | code, then a job | **not written** |
| 9 figures and numbers from artifacts | code | **not written** |
| 10 thesis rewrite | writing | after everything |

Scope steps 7-9 are the remaining analysis and are now **written** — jobs 6 and 7 below; scope 9
runs locally and needs no cluster time.

| script | state |
|---|---|
| `shared/inference.py` | new — Holm/BH, statistic-inside-resample bootstrap, CGM two-way, **dyadic**, TOST, exact energy shares |
| `variance_decomposition.py` | patched — real molar dose, exact energy shares, dyadic intervals |
| `calibration_eval.py` | patched — average ranks for ties, tie-aware NIR, DRF resampled whole, Holm |
| `aggregate_workspace_probe.py` | new — one global BH, un-measured cells named |
| `build_thesis_assets.py` | new — LaTeX macros + claim to artifact table |

They absorb the clustering decision from ERRATA defect 19, which is not in the scope document.
`residual_eval` uses Cameron-Gelbach-Miller because its observations are conditions and the two
groupings partition them. The transfer coefficient cannot: its observations are PAIRS, and a pair
belongs to two nodes at once, so no pair of groupings partitions anything. That arm uses the
Fafchamps-Gubert dyadic estimator instead. Each arm names its unit rather than inheriting one.


Five jobs. Four are CPU and independent; the fifth is the GPU retrain and is gated on two of the
others. Nothing runs on the login node — every step is `sbatch`.

Paste each log-check command after the job finishes and save the output; that is what comes back to me.

---

## Step 0 — one scp, everything at once

```bash
scp shared/tahoe_design.py endcell/ot/build_residual_targets.py endcell/ot/build_ot_targets.py endcell/ot/build_embeddings.py endcell/analysis/residual_eval.py endcell/analysis/channel_gate.py endcell/analysis/artifact_manifest.py endcell/analysis/experimental_unit_audit.py endcell/analysis/dose_response_analysis.py endcell/train/train_c2s_tahoe_endcell.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

Then confirm the environment before spending queue time on a typo:

```bash
ssh 3180408@login.hpc.unibocconi.it "cd ~/tahoe && srun --account=3180408 --partition=defq --cpus-per-task=2 --mem=8G --time=00:10:00 /data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python -c 'import build_residual_targets, channel_gate, artifact_manifest, experimental_unit_audit, tahoe_design; print(\"all modules import\")'"
```

---

## Step 1 — `artifacts` (CPU, ~5 min)

Are the six missing result sets lost, or just never copied back?

```bash
cd ~/tahoe && cat > artifacts.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=artifacts
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/artifacts_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
$PY artifact_manifest.py --selftest
$PY artifact_manifest.py --scan ~/tahoe/RESULTS \
    --bundle ~/tahoe/artifacts_bundle.tar.gz --out RESULTS/artifact_scan.json
ls -la ~/tahoe/artifacts_bundle.tar.gz
echo done
EOF
sbatch artifacts.sbatch
```

View:

```bash
cd ~/tahoe && LOG=$(ls -t logs/artifacts_*.out | head -1); grep -nE "SELFTEST|^.*(ok  |MISS)|entries complete|needs a re-run|bundled|done" $LOG
```

Bring the bundle home afterwards:

```bash
scp 3180408@login.hpc.unibocconi.it:~/tahoe/artifacts_bundle.tar.gz . && tar xzf artifacts_bundle.tar.gz -C RESULTS_cluster/
```

---

## Step 2 — `unitaudit` (CPU, ~30 min)

What the physical experiment is: cell lines per well, replicate wells, combinations, recoverable
doses, split crossing, and whether unit-stripping corrupted the tier-4 dose test.

```bash
cd ~/tahoe && cat > unitaudit.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=unitaudit
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/unitaudit_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
$PY tahoe_design.py --selftest
$PY experimental_unit_audit.py --selftest
$PY experimental_unit_audit.py --cache_dir "$D/ot_cache" \
    --holdout "$D/residual_targets_holdout2/holdout.json" \
    --out RESULTS/experimental_unit_audit.json
echo done
EOF
sbatch unitaudit.sbatch
```

View:

```bash
cd ~/tahoe && LOG=$(ls -t logs/unitaudit_*.out | head -1); grep -nE "SELFTEST|verdict|cell lines per sample|treatment samples|distinct \(drug|replicated in|samples per treatment|combination|usable molar|unusable|COLLISION|SPLITS|CROSSING|->|done" $LOG
```

---

## Step 3 — `chgate` (CPU, ~2 h)

The plate-matched channel gate. **This is the one that can retract a printed claim.**

```bash
cd ~/tahoe && cat > chgate.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=chgate
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chgate_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
$PY channel_gate.py --selftest
$PY channel_gate.py --cache_dir "$D/ot_cache" --seed 42 \
    --out RESULTS/channel_gate_platematched.json
echo done
EOF
sbatch chgate.sbatch
```

View:

```bash
cd ~/tahoe && LOG=$(ls -t logs/chgate_*.out | head -1); grep -nE "SELFTEST|PLATE-ONLY|count-matched null is|plate-matched null closes|CO-PLATING|excess|count_matched|plate_matched|different_plate|-> |>>>|done" $LOG
```

---

## Step 4 — `rebuild` (CPU, ~1 h) — the decision gate

Both split units, so the estimand is a comparison rather than an accident.

```bash
cd ~/tahoe && cat > rebuild.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=rebuild
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rebuild_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
DATA=$D/data_diverse2_endcell_big
TGT=$D/residual_targets_repaired

$PY build_residual_targets.py --selftest

echo "=== [A] sample split -- independent-treatment generalisation ==="
$PY build_residual_targets.py --cache_dir "$D/ot_cache" --out_dir "$TGT" \
    --generic_scope plate --shrink_k 0 --min_plate_drugs 3 --scope_sensitivity \
    --split_unit sample --val_frac 0.02 \
    --tier2_file "$DATA/eval_tier2_unseen_drugs.jsonl" \
    --tier3_file "$DATA/eval_tier3_unseen_combos.jsonl" \
    --holdout_combos 0.15 --holdout_drugs 0.10 --min_combo_conditions 250 --seed 42 \
    --emit_fit_digest "$TGT/fit.sha"

echo "=== [B] condition split -- the published estimand, for comparison ==="
$PY build_residual_targets.py --cache_dir "$D/ot_cache" --out_dir "${TGT}_condition" \
    --generic_scope plate --shrink_k 0 --min_plate_drugs 3 \
    --split_unit condition --val_frac 0.02 \
    --tier2_file "$DATA/eval_tier2_unseen_drugs.jsonl" \
    --tier3_file "$DATA/eval_tier3_unseen_combos.jsonl" \
    --holdout_combos 0.15 --holdout_drugs 0.10 --min_combo_conditions 250 --seed 42

echo "=== reports ==="
$PY -c "import json; [print(k, json.dumps(v)[:900]) for k,v in json.load(open('$TGT/report.json')).items()]"
echo done
EOF
sbatch rebuild.sbatch
```

View:

```bash
cd ~/tahoe && LOG=$(ls -t logs/rebuild_*.out | head -1); grep -nE "SELFTEST|===|treatment identifier|inventory:|scope |reliability |well crossing|promoted to the treated well|validation shard|holdout|frame|fit_digest|wrote |REFUSING|done" $LOG
```

**Read three numbers before Step 5.** `scope plate` retention from the sensitivity table; `n_train`
after the fail-closed plate rule; and `frac_heldout_conditions_sharing_a_well_with_train` from the
condition-split build. If plate retention is too thin, the answer is `--shrink_k 5` and a rerun of
[A] — not a fall back to cell-line scope, which is the contaminated frame.

---

## Step 5 — `retrain` (GPU, ~20 h) — **only after Steps 3 and 4 are read**

Two hard preconditions: Step 4's gate read, and Step 3 finished (its outcome decides whether any
channel wording survives). Step 6's evaluator fixes are already in the scp'd `residual_eval.py`.

```bash
cd ~/tahoe && cat > retrain.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=retrain
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/retrain_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
TGT=$D/residual_targets_repaired
CKPT=$D/checkpoints/pythia_sft_residual_repaired

test -s "$TGT/residual.jsonl" || { echo "run rebuild.sbatch first"; exit 1; }
test -s "$TGT/residual_val.jsonl" || { echo "no residual-format validation shard"; exit 1; }

echo "=== [1] retrain -- seeded, validated in the training format ==="
$PY train_c2s_tahoe_endcell.py --mode full \
    --model_name vandijklab/C2S-Scale-Pythia-1b-pt \
    --train_file "$TGT/residual.jsonl" \
    --eval_file "$TGT/residual_val.jsonl" \
    --output_dir "$CKPT" \
    --num_epochs 1 --batch_size 1 --grad_accum 16 \
    --bf16 --gradient_checkpointing --max_length 8192 \
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
    --log_every 50 --save_every 1000 --keep_checkpoints 3 --seed 42

echo "=== [2] eval -- training-only baselines, within-plate different-drug partners ==="
$PY residual_eval.py --cache_dir "$D/ot_cache" \
    --model_path "$CKPT/final" --model_kind residual \
    --holdout "$TGT/holdout.json" --train_file "$TGT/residual.jsonl" \
    --partner_policy within_plate \
    --split_quota "train=200,unseen_combo=250,unseen_drug=120" \
    --k_samples 4 --max_new_tokens 1400 --bf16 --seed 42 \
    --out RESULTS/re_repaired.json
echo done
EOF
sbatch retrain.sbatch
```

View:

```bash
cd ~/tahoe && LOG=$(ls -t logs/retrain_*.out | head -1); grep -nE "===|Seed:|lookup baselines fitted|stratified scramble partners|ceiling|model - |line \[|well \[|scramble_|drug_lookup|oracle|train |unseen_combo|unseen_drug|HEADLINE|>>>|done" $LOG
```

---

## Step 6 — `vardecomp` (CPU, ~2 h) — scope 7

The transfer coefficient with a real dose axis, an exact energy decomposition, and dyadic intervals.
Its dose arm was void: it compared WELL identifiers, and one well carries every cell line at one
concentration, so "different well" was never "different dose".

```bash
scp shared/inference.py shared/tahoe_design.py endcell/analysis/variance_decomposition.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > vardecomp.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=vardecomp
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vardecomp_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
$PY inference.py --selftest
$PY variance_decomposition.py --selftest
$PY variance_decomposition.py --cache_dir "$D/ot_cache" --generic_scope plate \
    --kappa_structure --seed 42 --out RESULTS/vardecomp_matched.json
echo done
EOF
sbatch vardecomp.sbatch
```

```bash
cd ~/tahoe && LOG=$(ls -t logs/vardecomp_*.out | head -1); grep -nE "SELFTEST|dose resolution|unresolved|SCOPE|cross|T  \[|dyadic|quote this one|NEGATIVE CONTROL|DOSE|Traceback|Error|done" $LOG
```

Three things decide whether the chapter changes: whether the cross term is small enough that the
energy shares read as variance components; how much the dyadic interval widens T against the
drug-clustered one; and whether the dose arm, now actually measuring dose, still orders below
cross-line transfer the way the biology says it should.

---

## Step 7 — `calib` (CPU, ~2 h) — scope 8

The metric audit, hardened. Average ranks for tied expression, tie-aware NIR, the DRF ratio
recomputed inside every resample rather than against a fixed denominator, and Holm across the five
metrics so "only NIR is calibrated" becomes a simultaneous statement.

```bash
scp shared/inference.py endcell/analysis/calibration_eval.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > calib.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=calib
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/calib_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
$PY inference.py --selftest
$PY calibration_eval.py --selftest --out RESULTS/calib_selftest.json
$PY calibration_eval.py --seed 42 --out RESULTS/calibration.json
echo done
EOF
sbatch calib.sbatch
```

```bash
cd ~/tahoe && LOG=$(ls -t logs/calib_*.out | head -1); grep -nE "SELFTEST|unit checks|Holm-adjusted|DRF |CALIBRATED|not calibrated|Traceback|Error|done" $LOG
```

The claim to watch: under Holm, is NIR still the only metric whose interval clears zero? If a second
survives, the metric chapter's headline needs rewording. If NIR stops surviving, that is a much
larger problem and nothing downstream should be written until it is understood.

---

## Scope 9 — runs locally, no cluster time

Once the artifacts are back from the bundle:

```bash
python endcell/analysis/aggregate_workspace_probe.py --glob 'RESULTS_cluster/probe_arm_*.json' 'RESULTS_cluster/probe_rep_*.json' --out RESULTS_cluster/probe_canonical.json --csv RESULTS_cluster/probe_canonical.csv
```

```bash
python endcell/analysis/build_thesis_assets.py --results RESULTS_cluster --out_dir thesis/generated
```

The second prints a claim-by-claim resolution table and writes `numbers.tex`. A macro absent from
that file is a number with no artifact, and LaTeX will fail on it rather than keep a stale figure --
which is the point. Add the two generated files to the preamble and the appendix respectively.

---

## Watching a job while it runs

```bash
squeue -u 3180408
```

```bash
cd ~/tahoe && tail -f $(ls -t logs/*.out | head -1)
```

```bash
sacct -u 3180408 --format=JobID,JobName%14,State,Elapsed,MaxRSS --starttime today
```

---

## What to send back

For each step, the output of its **View** command, in order, in one MD. If a job fails, the last 40
lines of its log instead:

```bash
cd ~/tahoe && tail -40 $(ls -t logs/*.out | head -1)
```

The three things I most need: the co-plating `excess` and the `plate_matched` row from Step 3; the
scope-sensitivity table and `frac_heldout_conditions_sharing_a_well_with_train` from Step 4; and
whether Step 1 recovered the missing artifacts or confirmed they were never computed.

---

## Running the test suite locally

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

The env var is REQUIRED on this machine and has nothing to do with the tests. Plain
`python -m pytest` dies during collection with `ValueError: source code string cannot contain null
bytes`: pytest autoloads the `dash` plugin, `dash` imports `flask`, `flask` imports `click`, and
`site-packages/click/__init__.py` is 4634 bytes of which all 4634 are NUL -- an interrupted write in
a OneDrive-synced `site-packages`. Disabling autoload skips the dash plugin and the suite runs
untouched. Expected: **64 passed**.

Every analysis script additionally carries its own selftest, which needs no pytest at all:

```
python shared/inference.py --selftest
python -m endcell.analysis.residual_eval --selftest --out /tmp/st.json
python -m endcell.analysis.calibration_eval --selftest --out /tmp/st.json
python -m endcell.analysis.variance_decomposition --selftest --out /tmp/st.json
python -m endcell.analysis.scramble_stratum_audit --selftest --out /tmp/st.json
python -m endcell.analysis.channel_gate --selftest --out /tmp/st.json
```

Note `shared/inference.py` takes no `--out` (it writes nothing); the others require it.

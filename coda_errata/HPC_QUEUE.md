# The HPC queue — everything left to run

Written against HEAD `92bc5b3`. Five jobs. **Nothing runs on the login node**; every step is `sbatch`,
and the two inline checks are `srun`. Every job writes to a **new** directory — nothing overwrites
`ot_cache`, `residual_targets_repaired`, or any checkpoint.

**The gate:** Job 1 must print `GATE PASSED` before Job 2 is submitted. Job 2 is the expensive one
and it is pointless if the rebuild came out different. (Job 1 does not print `fit_digest MATCHES` --
that string comes from the *replay* check inside `build_residuals`, which Job 2 prints.)

---

## Step 0 — copy the changed files

Five files changed since the last upload. `build_residual_targets.py` and `residual_eval.py` are the
blocker fix; `channel_gate.py` is defect 35; `inference.py` is the t distribution and the repaired
cluster bootstrap; `scramble_stratum_audit.py` is the drug-balance arm and the geometry fix.

```bash
cd ~/OneDrive/Desktop/tahoe && scp endcell/ot/build_residual_targets.py endcell/analysis/residual_eval.py endcell/analysis/channel_gate.py endcell/analysis/scramble_stratum_audit.py endcell/analysis/calibration_eval.py shared/inference.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

Then confirm they import, on a compute node:

```bash
ssh 3180408@login.hpc.unibocconi.it "cd ~/tahoe && srun --account=3180408 --partition=defq --cpus-per-task=2 --mem=8G --time=00:10:00 /data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python -c 'import build_residual_targets, residual_eval, channel_gate, scramble_stratum_audit, inference; print(\"all modules import\")'"
```

---

## Job 1 — `rebuild3` (CPU, ~1 h). Emit a manifest a replay can actually use.

This does **not** change the training data and does **not** require a retrain. The builder always
fitted the generic correctly; only the *manifest* was written filtered. The job therefore ends by
proving that `residual.jsonl` and `fit_digest` are identical to the existing build — if they are, the
current checkpoint remains valid and Job 2 can proceed.

```bash
cd ~/tahoe && cat > rebuild3.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=rebuild3
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rebuild3_%j.out
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
OLD=$D/residual_targets_repaired
TGT=$D/residual_targets_v3

$PY build_residual_targets.py --selftest

# THE THRESHOLD IS NOT THE DEFAULT, AND GETTING IT WRONG FAILS THE GATE.
# $OLD was built with `--repro_thr auto`, which resolved to about 0.1086 against that run own
# null. The argparse default is 0.2. Rebuilding at 0.2 writes a DIFFERENT residual.jsonl, so
# the gate prints GATE FAILED -- and note the fit_digest would still say SAME, because the
# digest is computed BEFORE the threshold is resolved. If anyone pushed past that, report.json
# would carry 0.2, Job 2 would inherit it through --truth_from, and transform would drop
# training conditions the checkpoint was actually trained on.
#
# Read the literal out of the old report rather than passing `auto` again: the reliability null
# is sampled, so `auto` is only reproducible up to its RNG.
THR=$($PY -c "import json;print(json.load(open('$OLD/report.json'))['repro_thr'])")
echo "rebuilding at the threshold the previous build used: $THR"

# IDENTICAL flags to the build that produced $OLD. Anything different here invalidates the
# comparison below, which is the only evidence that no retrain is needed.
$PY build_residual_targets.py --cache_dir "$D/ot_cache" --out_dir "$TGT" \
    --repro_thr "$THR" \
    --generic_scope plate --shrink_k 0 --min_plate_drugs 3 --scope_sensitivity \
    --split_unit sample --val_frac 0.02 \
    --tier2_file "$DATA/eval_tier2_unseen_drugs.jsonl" \
    --tier3_file "$DATA/eval_tier3_unseen_combos.jsonl" \
    --holdout_combos 0.15 --holdout_drugs 0.10 --min_combo_conditions 250 --seed 42 \
    --emit_fit_digest "$TGT/fit.sha"

echo "=== THE GATE: is this the same build, with a better manifest? ==="
$PY - <<'PYEOF'
import hashlib, json, os
D = "/data/BuffaF-Projetcs/florian_c2s"
old, new = D + "/residual_targets_repaired", D + "/residual_targets_v3"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
ok = True
for name in ("residual.jsonl",):
    a, b = sha(os.path.join(old, name)), sha(os.path.join(new, name))
    print(f"{name:20s} old {a[:16]}  new {b[:16]}  {'SAME' if a == b else 'DIFFERENT'}")
    ok &= a == b
ro, rn = (json.load(open(os.path.join(p, "report.json"))) for p in (old, new))
print(f"{'fit_digest':20s} old {ro['fit_digest'][:16]}  new {rn['fit_digest'][:16]}  "
      f"{'SAME' if ro['fit_digest'] == rn['fit_digest'] else 'DIFFERENT'}")
ok &= ro["fit_digest"] == rn["fit_digest"]
hm = json.load(open(os.path.join(new, "holdout.json")))
for k in ("split_all", "train_keys_fit", "fit_digest", "eval_repro_filter"):
    print(f"  manifest carries {k:20s} {k in hm}")
    ok &= k in hm
print(f"  split_all {len(hm.get('split_all', {}))} conditions vs filtered split "
      f"{len(hm.get('split', {}))}; fit inventory {len(hm.get('train_keys_fit', []))}")
print("GATE PASSED -- no retrain needed, proceed to Job 2" if ok else
      "GATE FAILED -- do NOT submit the GPU job; report this output")
PYEOF
echo done
EOF
sbatch rebuild3.sbatch
```

**View, and this is the output to send back:**

```bash
cd ~/tahoe && LOG=$(ls -t logs/rebuild3_*.out | head -1); grep -nE "SELFTEST|===|rebuilding at the threshold|inventory:|fitting on|fit_digest|reliability \[|holdout manifest|wrote |SAME|DIFFERENT|manifest carries|split_all|GATE|REFUSING|done" $LOG
```

If the gate fails, stop. It means something other than the manifest changed and the checkpoint may no
longer match its targets.

**What the two halves of the gate can see.** The `residual.jsonl` sha is the strong check -- it moves
if the scope, the threshold, the shrinkage, leave-one-drug-out or `min_plate_drugs` move, because all
of those determine the written targets. The `fit_digest` comparison is weaker than it looks: it hashes
the fitted drug MEANS, so it moves with the cache, the seed, the inventory flags and the reliability
controls, but it is blind to scope, shrink_k, loo and min_plate_drugs, and it is computed before the
`auto` threshold resolves. Both must say SAME. The frame parameters the digest cannot see are checked
separately, against the manifest, inside `build_residuals` -- which now refuses rather than warns.

---

## Job 2 — `eval3` (GPU, the expensive one). Three outstanding things in one generation run.

Do not split this. Generation dominates the cost, and the truth-replay fix, the field decomposition
and the quota increase all need the same generated outputs.

- **correct truth frame** — `--truth_from` now propagates `eval_repro_filter`, and the manifest now
  carries `split_all`, so held-out truth is no longer selected on its own reproducibility
- **`--field_decomp`** — adds the drug-name-only and mechanism-only arms. This is the experiment that
  decides whether "the model reads the drug name" is sayable at all; right now it is not
- **a larger quota** — the detection limit is currently +0.099, which is why `unseen_combo` reads
  "not established" rather than a decision

**On the quota.** The previous run was `train=200,unseen_combo=250,unseen_drug=120` = 570 conditions
with 4 generated arms. Below is `300/500/200` = 1000 conditions with 6 arms — about **2.6×** the
previous generation cost. If the previous run finished comfortably inside 24 h this fits; if it ran
close to the wall, drop `unseen_combo` to 400 and resubmit. Raising `unseen_combo` is what tightens
the interval, so spend the budget there rather than on `train`.

```bash
cd ~/tahoe && cat > eval3.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval3
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval3_%j.out
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
TGT=$D/residual_targets_v3
CKPT=$D/checkpoints/pythia_sft_residual_repaired

$PY residual_eval.py --selftest --out /tmp/st.json

# --truth_from is MANDATORY and was missing from the old runbook. --field_decomp is the new arm.
# --split_comparator defaults to the neutral stratum now; passed explicitly so the log records it.
$PY residual_eval.py --cache_dir "$D/ot_cache" \
    --model_path "$CKPT/final" --model_kind residual \
    --truth_from "$TGT/report.json" \
    --holdout "$TGT/holdout.json" --train_file "$TGT/residual.jsonl" \
    --partner_policy within_plate \
    --split_quota "train=300,unseen_combo=500,unseen_drug=200" \
    --field_decomp \
    --split_comparator scramble_orth \
    --k_samples 4 --max_new_tokens 1400 --bf16 --seed 42 \
    --out RESULTS/re_v3.json
echo done
EOF
sbatch eval3.sbatch
```

**View:**

```bash
cd ~/tahoe && LOG=$(ls -t logs/eval3_*.out | head -1); grep -nE "SELFTEST|truth frame taken from|fit_digest MATCHES|fit_digest MISMATCH|eval_repro_filter|split labels for the truth|conditions with residuals|===|WHICH FIELD IS READ|swap |GENERALIZATION|train |unseen_combo|unseen_drug|common support|coverage|drug_lookup|ceiling|-> |>>>|done" $LOG
```

**Three lines to check before reading any number:**

1. `fit_digest MATCHES the build` — the truth is the one the checkpoint trained on
2. `eval_repro_filter=False` and **no** warning about the reliability-surviving subset
3. `split labels for the truth build: N conditions from split_all`

If any is missing the run is in the old frame and the numbers are not usable.

---

## Job 3 — `chgate3` (CPU, ~6 h). The gate in the repaired frame, two-way clustered.

Gated on Job 1 only, so it can run alongside Job 2.

```bash
cd ~/tahoe && cat > chgate3.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=chgate3
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chgate3_%j.out
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
TGT=$D/residual_targets_v3

$PY channel_gate.py --selftest --out /tmp/st.json
$PY channel_gate.py --cache_dir "$D/ot_cache" --seed 42 \
    --truth_from "$TGT/report.json" --holdout "$TGT/holdout.json" \
    --out RESULTS/channel_gate_v3.json
echo done
EOF
sbatch chgate3.sbatch
```

**View:**

```bash
cd ~/tahoe && LOG=$(ls -t logs/chgate3_*.out | head -1); grep -nE "SELFTEST|truth frame from|fit_digest|COVERAGE|CO-PLATING|PLATE-ONLY|two-way|one-way|n_wells|different_plate|moa|excess|-> |>>>|done" $LOG
```

Expect the intervals to **widen** relative to `channel_gate_platematched.json`. The `moa` row under
different-plate was barely positive on a one-way cell-line interval and is the row most likely to
lose its verdict. That is the point of running it.

---

## Job 4 — `calib3` (CPU, ~2 h). Put the scope-8 numbers in a file.

The corrected calibration exists only in a log; `RESULTS/calibration.json` on disk is the 29 July
artifact with no CI and no Holm keys, so nothing in the thesis can cite it. Runs both configurations
because the published claim is the plate-matched one and the runbook command omitted the flag.

```bash
cd ~/tahoe && cat > calib3.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=calib3
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/calib3_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs

$PY calibration_eval.py --selftest --out /tmp/st.json
echo "=== [A] as published (all plates) ==="
$PY calibration_eval.py --seed 42 --out RESULTS/calibration_v3.json
echo "=== [B] plate-matched -- the confound-free version ==="
$PY calibration_eval.py --seed 42 --same_plate_only --out RESULTS/calibration_v3_sameplate.json
echo done
EOF
sbatch calib3.sbatch
```

**View:**

```bash
cd ~/tahoe && LOG=$(ls -t logs/calib3_*.out | head -1); grep -nE "SELFTEST|===|DRF|holm|calibrated|weighted_r2|spearman|de_delta|panel_tau|nir|Read:|done" $LOG
```

---

## After the jobs — pull the artifacts back and re-run the local analysis

```bash
cd ~/OneDrive/Desktop/tahoe && scp 3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/re_v3.json 3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/channel_gate_v3.json 3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/calibration_v3.json 3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/calibration_v3_sameplate.json RESULTS_cluster/
```

Then, locally, no GPU:

```bash
cd ~/OneDrive/Desktop/tahoe && python -m endcell.analysis.scramble_stratum_audit --result RESULTS_cluster/re_v3.json --out RESULTS_cluster/scramble_stratum_audit_v3.json
```

That recomputes the comparator strata, the drug-matched premium, the slopes, the geometry
distribution and the detection limit from the new records.

---

## Not queued, and why

**Seed replication** (`jobs/seeds.md`, 3 seeds, ~15 GPU-h). Its motivating example turned out to be a
comparator artefact rather than generation noise, so it no longer blocks a claim. Across-seed
generation variance is still real and uncaptured by any interval in the thesis — run it after Job 2
if there is budget, as a precision job.

**A retrain.** Not needed unless Job 1's gate fails. The training data does not change.

**`vardecomp`.** Already correct — it computes its own shifts and never touches the compatibility
shim, so Part V's transfer coefficient is unaffected by the blocker.

---

## What is still open after all five jobs

- One training epoch, one training seed. Every interval is conditional on this checkpoint.
- One target half-split seed (defect 22) — identified, not costed.
- Commit `41c6716` carries DE-weighting code on the training path, verified inert at defaults and
  exercised by no test.
- `coda_errata/` is gitignored and `RESULTS_cluster/` untracked, so this register and the artifacts
  it describes are outside version control.

# Seed replication — how much of the interval is generation noise?

## Why

Two runs of the **same checkpoint on the same conditions with the same seed**, differing only by the
addition of two extra scramble arms, produced:

| | run A | run B |
|---|---|---|
| model NIR | 0.639 | 0.645 |
| `scramble_opposite` | 0.589 | **0.542** |
| `drug_lookup_1` | 0.832 | **0.880** |
| **`unseen_drug` gap** | **+0.0195** [−0.045, +0.086] | **+0.1114** [+0.051, +0.173] |

The extra arms shift torch's sampling stream, so every generation after the first condition differs.
That is expected. The size is not: `unseen_drug` moved from a clean null to a CI excluding zero, and
the two intervals barely overlap.

The cause is structural rather than a bug. **The clustered bootstrap resamples cell lines and treats
each condition's score as fixed**, so it captures between-cell-line variance and not generation
variance. With `k_samples=4` at temperature $0.8$, the second source is evidently large. Every
interval in this thesis is conditional on one draw of generations.

This job measures that variance instead of assuming it away. Three seeds, identical in every other
respect. The spread across them is the missing component.

## Run it

```bash
cd ~/tahoe && cat > evalseeds.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=evalseed
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/evalseed_%A_%a.out
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
S=${SLURM_ARRAY_TASK_ID}
echo "############ SEED $S ############"
$PY residual_eval.py --cache_dir "$D/ot_cache" \
    --model_path "$D/checkpoints/pythia_sft_residual_holdout2/final" --model_kind residual \
    --holdout "$D/residual_targets_holdout2/holdout.json" \
    --train_file "$D/residual_targets_holdout2/residual.jsonl" \
    --field_decomp --split_quota "train=200,unseen_combo=250,unseen_drug=120" \
    --k_samples 4 --max_new_tokens 1400 --bf16 --seed "$S" \
    --out "RESULTS/re_seed${S}.json"
echo done
EOF
sbatch evalseeds.sbatch
```

An **array job**, so the three seeds are independent: one failing does not take the others with it,
and they run in parallel if slots are free. Budget ~5 h each based on the field-decomposition run,
which is the same six arms.

`--field_decomp` is kept deliberately. The field decomposition is now a load-bearing result --- it is
what rules out the channel-conditioning arm --- so its seed stability matters as much as the
headline's.

## Read it

```bash
for s in 1 2 3; do
  echo "=== seed $s ==="
  LOG=$(ls -t logs/evalseed_*_$s.out | head -1)
  grep -E "model  |scramble_opposite|drug_lookup_1|swap |train  |unseen_combo|unseen_drug|HEADLINE" $LOG
done
```

Then the quantity that matters, across the three:

| arm | seed 1 | seed 2 | seed 3 | spread |
|---|---|---|---|---|
| headline gap (opposite) | | | | |
| `train` gap | | | | |
| `unseen_combo` gap | | | | |
| **`unseen_drug` gap** | | | | |
| swap drug-name-only | | | | |
| swap mechanism-only | | | | |

**What each outcome means.** If the spread is small relative to the stated intervals, the intervals
are approximately right and the two divergent runs were unlucky. If the spread is comparable to the
intervals --- which the `unseen_drug` swing suggests --- then every interval in the thesis needs
widening, and the honest form of each result is a point estimate with a seed spread beside it rather
than a single bootstrap CI.

Either way the fix is cheap: report the across-seed spread alongside the within-seed bootstrap, and
say plainly which sources of variation each captures.

## What this does NOT fix

The bootstrap will still be conditional on one *sample of conditions*. Three seeds vary the
generations, not the 570 conditions drawn from the pool. That is a smaller effect --- the split quota
fixes the split sizes and the pool is only 4,091 --- but it is worth a sentence in Limitations rather
than a pretence.

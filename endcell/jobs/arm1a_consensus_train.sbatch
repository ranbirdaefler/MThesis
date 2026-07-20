#!/bin/bash
#SBATCH --job-name=arm1a_consensus
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/arm1a_consensus_%j.out
# ============================================================================================
# ARM 1a — denoised-target ("consensus") training.  THE experiment: is drug signal LEARNABLE by
# this architecture when the target carries it (pseudobulk), vs the single-cell target that doesn't?
#
# Design = a clean controlled comparison. The ONLY thing that differs from the single-cell model
# (pythia_sft_endcell) is the TRAINING TARGET. Everything else is held identical:
#   - same base model + COLD START (not continued from the single-cell model -> no confound)
#   - same hyperparameters (lr, epochs, batch, grad_accum, max_length, warmup, weight_decay)
#   - same #training examples (per_cell emit: one example per original cell, only the target swapped)
#   - evaluation uses the ORIGINAL held-out eval tiers (real cells) + the ORIGINAL single-cell linear
#     baseline as a fixed drug-agnostic yardstick, WITHIN-PLATE + scramble (see the eval block below).
#
# LEAK / CONTAMINATION GUARDS (why each step is here):
#  1. Consensus is built ONLY from train.jsonl. Eval tiers (tier2/3/4 = held-out drugs/combos/doses)
#     are NEVER read or modified -> no train->eval leak. Verified: we touch only train_consensus.jsonl.
#  2. Grouping is (drug, cell_line, DOSE): 62% of drugs are multi-dose and different doses give
#     different responses, so pooling doses would fabricate a blurred target. Dose is kept separate.
#  3. per_cell emit preserves the example COUNT exactly (hard-checked below) -> the single-cell vs
#     consensus comparison differs ONLY in target content, not in dataset size / example weighting.
#  4. Consensus never crosses drug / cell line / dose -> no cross-condition information bleed.
#  5. Cold start from the base model -> the two models start identical; target is the sole variable.
#  6. Output goes to a SEPARATE checkpoint dir -> the single-cell model is not overwritten.
#  7. Build is idempotent (skips if the consensus file exists) -> a re-queue after preemption does
#     not rebuild or perturb the data.
#  8. Hard abort if the consensus line count != train.jsonl (a corrupted build must not train).
#
# NOTE: paths are FLAT (~/tahoe/<script>.py) to match the cluster layout.
# ============================================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
CONSENSUS="$DATA/train_consensus.jsonl"
CKPT_OUT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell_consensus
BASE=vandijklab/C2S-Scale-Pythia-1b-pt

# ---- (1) build consensus targets from train.jsonl (idempotent) ------------------------------
if [ ! -f "$CONSENSUS" ]; then
    echo "[build] constructing consensus targets (grouping drug x cell_line x dose) ..."
    $PY build_consensus_targets.py \
        --train_file "$DATA/train.jsonl" \
        --out "$CONSENSUS" \
        --report RESULTS/consensus_report.json \
        --group_keys drug,cell_line_id,dose_float \
        --min_cells 2 --emit per_cell
else
    echo "[build] consensus targets already exist: $CONSENSUS (skipping build)"
fi

# ---- guard #8: example count must be preserved (per_cell). Abort if not. ---------------------
IN=$(wc -l < "$DATA/train.jsonl"); OUT=$(wc -l < "$CONSENSUS")
echo "[check] train.jsonl lines=$IN  |  consensus lines=$OUT"
if [ "$IN" != "$OUT" ]; then
    echo "[FATAL] consensus line count != train.jsonl — corrupted build, refusing to train."
    exit 1
fi
# surface the denoisable fraction (if this is low, consensus is ~a no-op and the experiment is weak)
$PY - "$DATA" << 'PYEOF'
import json, sys
d = json.load(open("RESULTS/consensus_report.json"))
print(f"[check] denoisable examples: {100*d['frac_examples_denoised']:.1f}%  "
      f"(cells/condition median {d['cells_per_condition']['median']:.0f}, "
      f"max {d['cells_per_condition']['max']})")
if d["frac_examples_denoised"] < 0.5:
    print("[WARN] <50% of examples denoisable -> consensus barely differs from single-cell; "
          "results will be weak. Consider re-streaming with more cells/condition.")
PYEOF

# ---- (2) cold-start SFT on the consensus targets (identical recipe to the single-cell run) ---
echo "[train] cold-start SFT on consensus targets -> $CKPT_OUT"
$PY train_c2s_tahoe_endcell.py --mode full \
    --model_name "$BASE" \
    --train_file "$CONSENSUS" \
    --eval_file "$DATA/eval_tier1_seen_conditions.jsonl" \
    --output_dir "$CKPT_OUT" \
    --num_epochs 1 --batch_size 1 --grad_accum 16 \
    --bf16 --gradient_checkpointing --max_length 8192 \
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
    --log_every 50 --save_every 500 --keep_checkpoints 2

echo "done -> $CKPT_OUT/final"
echo ""
echo "NEXT (separate GPU job) — evaluate WITHIN-PLATE + scramble, vs the FIXED single-cell linear:"
echo "  \$PY nir_benchmark.py --same_plate_only \\"
echo "     --eval_dir $DATA --scram_dir ${DATA}_scram --model_path $CKPT_OUT/final \\"
echo "     --train_file $DATA/train.jsonl \\"
echo "     --tiers tier1_seen_conditions,tier2_unseen_drugs \\"
echo "     --k_samples 8 --temperature 0.8 --min_cells 8 --min_drugs_per_cl 3 --max_groups 80 --bf16 \\"
echo "     --out RESULTS/nir_consensus_sameplate.json --profiles RESULTS/nir_consensus_profiles.npz"

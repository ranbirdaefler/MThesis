#!/bin/bash
#SBATCH --job-name=arm1_consensus
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/arm1_consensus_%j.out
# ARM 1a — the denoised-target objective fix.
#   (1) build consensus (pseudobulk) training targets from the EXISTING train.jsonl (no re-streaming)
#   (2) cold-start SFT on those targets, identical hyperparams to the single-cell run, so the ONLY
#       difference vs pythia_sft_endcell is the target (single noisy cell -> condition consensus).
# Evaluate afterward with the usual NIR benchmark / grading against pythia_sft_endcell + linear.

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
CKPT_OUT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell_consensus

# (1) build consensus targets (idempotent: skip if already built so a re-queue after preemption
#     doesn't rebuild). Reports cells/condition + how much of the data is actually denoised.
if [ ! -f "$DATA/train_consensus.jsonl" ]; then
    $PY endcell/preprocess/build_consensus_targets.py \
        --train_file "$DATA/train.jsonl" \
        --out "$DATA/train_consensus.jsonl" \
        --report RESULTS/consensus_report.json \
        --group_keys drug,cell_line_id,dose_float \
        --min_cells 2 --emit per_cell
else
    echo "consensus targets already exist: $DATA/train_consensus.jsonl (skipping build)"
fi

# (2) cold-start SFT on the consensus targets (same recipe as the single-cell run)
$PY endcell/train/train_c2s_tahoe_endcell.py --mode full \
    --model_name vandijklab/C2S-Scale-Pythia-1b-pt \
    --train_file "$DATA/train_consensus.jsonl" \
    --eval_file "$DATA/eval_tier1_seen_conditions.jsonl" \
    --output_dir "$CKPT_OUT" \
    --num_epochs 1 --batch_size 1 --grad_accum 16 \
    --bf16 --gradient_checkpointing --max_length 8192 \
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
    --log_every 50 --save_every 500 --keep_checkpoints 2

echo "done -> $CKPT_OUT/final"

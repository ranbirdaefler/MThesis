#!/bin/bash
#SBATCH --job-name=ot_train
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ot_train_%j.out
# Arm 1c training. Cold-start SFT on an OT target arm (T0/T1/T2), IDENTICAL recipe to the single-cell
# and consensus runs -> the training TARGET is the only variable. Pick the arm with:  ARM=T2 sbatch ...
#   T0 = consensus (global mean, eps->inf) — the matched consensus baseline
#   T1 = control + mean displacement
#   T2 = OT barycentric (finite eps) — the real OT target; headline test is T2 vs T0
# All three JSONLs share the SAME prompts/cells (only the response differs), so this is the clean,
# fully-matched ladder. 24h wall-clock, 1000-step checkpoints (compare arms at the last shared step).
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p logs
ARM=${ARM:-T2}
OTDIR=/data/BuffaF-Projetcs/florian_c2s/ot_targets
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
BASE=vandijklab/C2S-Scale-Pythia-1b-pt
TRAIN="$OTDIR/${ARM}.jsonl"
CKPT_OUT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_ot_${ARM}

echo "[train] ARM=$ARM  train=$TRAIN  -> $CKPT_OUT"
test -f "$TRAIN" || { echo "[FATAL] $TRAIN not found"; exit 1; }
wc -l "$TRAIN"

$PY train_c2s_tahoe_endcell.py --mode full \
    --model_name "$BASE" \
    --train_file "$TRAIN" \
    --eval_file "$DATA/eval_tier1_seen_conditions.jsonl" \
    --output_dir "$CKPT_OUT" \
    --num_epochs 1 --batch_size 1 --grad_accum 16 \
    --bf16 --gradient_checkpointing --max_length 8192 \
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
    --log_every 50 --save_every 1000 --keep_checkpoints 4

echo "done -> $CKPT_OUT/final  (ARM=$ARM)"

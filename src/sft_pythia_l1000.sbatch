#!/bin/bash
#SBATCH --job-name=c2s_sft_l1000
#SBATCH --partition=<CONFIRM_PARTITION>      # <-- CONFIRM: the H200 partition on the new cluster
#SBATCH --gres=gpu:1                          # <-- CONFIRM syntax (maybe gpu:h200:1)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sft_l1000_%j.out
# #SBATCH --account=<CONFIRM_ACCOUNT>         # <-- uncomment + set if the cluster requires an account

# Respects the 4-GPU / 10-queued-job per-user limits: this is a single-GPU job.
# Do NOT add multi-GPU/DDP for this first 1B run.

set -euo pipefail
export PYTHONUNBUFFERED=1                      # required for live SLURM log output
mkdir -p logs checkpoints eval_results

module load miniconda3
# source activate <env>                        # <-- uncomment + set if your env isn't the base env

# Data directory containing train.jsonl, eval_tier*.jsonl, l1000_panel.json,
# linear_model.json, gene_id_to_symbol.json (all written by tahoe_c2s_preprocess.py).
DATA=/path/to/data                             # <-- CONFIRM path

# =============================================================================
# 1) Supervised fine-tuning (single H200, bf16, flash-attention-2)
# =============================================================================
srun python train_c2s_tahoe.py --mode full \
    --model_name vandijklab/C2S-Scale-Pythia-1b-pt \
    --train_file $DATA/train.jsonl \
    --eval_file  $DATA/eval_tier1_seen_conditions.jsonl \
    --output_dir ./checkpoints/pythia_l1000 \
    --max_train_examples 300000 \
    --num_epochs 2 --batch_size 1 --grad_accum 16 \
    --max_length 8192 \
    --attn_implementation flash_attention_2 \
    --bf16 --gradient_checkpointing \
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
    --log_every 50 --save_every 500

# =============================================================================
# 2) Evaluation: model + control baseline. Report Δτ = model − control per tier.
#    (The headline metric is the delta, not the absolute panel τ.)
# =============================================================================
python evaluate_c2s_tahoe.py \
    --model_path ./checkpoints/pythia_l1000/best \
    --eval_dir $DATA --output_dir ./eval_results/pythia_l1000 \
    --bf16 --max_eval 500 --max_new_tokens 3800 --gen_batch_size 16

python evaluate_c2s_tahoe.py \
    --model_path ./checkpoints/pythia_l1000/best \
    --eval_dir $DATA --output_dir ./eval_results/pythia_l1000_baseline \
    --bf16 --max_eval 500 --baseline control

# Optional should-have baselines (mean-shift); uncomment to run:
# python evaluate_c2s_tahoe.py \
#     --model_path ./checkpoints/pythia_l1000/best \
#     --eval_dir $DATA --output_dir ./eval_results/pythia_l1000_meanshift \
#     --baseline global_mean_shift --train_file $DATA/train.jsonl --max_eval 500

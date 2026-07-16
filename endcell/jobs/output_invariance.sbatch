#!/bin/bash
#SBATCH --job-name=out_inv
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/out_inv_%j.out
# GPU job: generates treated sentences under swapped-drug prompts (same fixed control)
# and measures output-vs-output invariance. Single H200.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

# Call the env's Python directly — robust whether or not `conda` is on PATH.
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final

$PY endcell/analysis/output_invariance.py \
    --eval_dir $DATA --model_path $CKPT --model_name endcell \
    --tier tier2_unseen_drugs --n_contexts 8 --n_drugs 12 --n_samples 2 \
    --temperature 0.8 --top_p 0.9 --topn 100 --max_new_tokens 3800 --bf16 \
    --gen_batch_size 48 --out RESULTS/output_invariance_endcell.json --seed 42

# Optional: also run the ORIGINAL full-panel SFT model as a comparator. It was trained on
# the old format (no [END_CELL]); point --eval_dir at its matching data dir if you run it.
# $PY endcell/analysis/output_invariance.py \
#     --eval_dir /data/BuffaF-Projetcs/florian_c2s/data_diverse2 \
#     --model_path /data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_diverse2/checkpoint-10000 \
#     --model_name original_sft --tier tier2_unseen_drugs \
#     --n_contexts 8 --n_drugs 12 --n_samples 2 --temperature 0.8 --top_p 0.9 --topn 100 \
#     --max_new_tokens 3800 --bf16 --out RESULTS/output_invariance_original.json --seed 42

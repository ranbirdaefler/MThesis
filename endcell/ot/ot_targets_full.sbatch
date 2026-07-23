#!/bin/bash
#SBATCH --job-name=ot_targets
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ot_targets_%j.out
# STAGE 3/3: Sinkhorn OT -> T0/T1/T2 target JSONLs + step-0 gates. c2s env (HF for metadata maps).
set -euo pipefail
mkdir -p logs
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
$PY build_ot_targets.py \
    --cache_dir /data/BuffaF-Projetcs/florian_c2s/ot_cache \
    --emb scvi --epsilon 0.5 --targets T0,T1,T2 \
    --min_treated 60 --min_control 60 --max_ctrl_per_cond 100 \
    --out_dir /data/BuffaF-Projetcs/florian_c2s/ot_targets
echo "STAGE 3 done -> /data/BuffaF-Projetcs/florian_c2s/ot_targets (T0/T1/T2.jsonl + step0_gates.json)"

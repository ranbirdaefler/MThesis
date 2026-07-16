#!/bin/bash
#SBATCH --job-name=endcell_cpu
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/endcell_cpu_%j.out
# CPU-bound [END_CELL] work, sequential in one allocation (runs in parallel with endcell_gpu):
#   (1) baseline ladder (control/global/per-cellline/per-MOA/per-MOA×cellline) + noise ceiling,
#       both absent-gene conventions, on tier1/2/3
#   (2) combined rank-vs-expression discrimination (Part VI, with the rank arm)
#
# NOTE: the same/different gap analysis (Finding-1 equivalent) is NOT here — it's your existing
#       gap_endcell.sbatch (drug_specificity_in_data.py --worst_mode). Check squeue/RESULTS first;
#       submit that separately if it hasn't run.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big

# (1) baselines + noise ceiling, both conventions, no model
$PY endcell/eval/evaluate_endcell.py --mode baselines,ceiling \
    --eval_dir $DATA --train_file $DATA/train.jsonl \
    --tiers tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos \
    --max_eval 400 --max_train 200000 \
    --out RESULTS/eval_endcell_cpu.json --seed 42

# (2) combined rank-vs-expression (true expression streamed from Tahoe; rank arm on same pseudobulks)
$PY endcell/analysis/expression_space_discrimination.py \
    --out RESULTS/expr_space_discrimination_combined.json \
    --panel_file $DATA/l1000_panel.json \
    --num_shards 8 --rows_per_shard 150000 --shard_seed 7 \
    --cells_per_drug 40 --min_cells_per_drug 30 \
    --n_celllines 30 --drugs_per_cellline 25 \
    --pb_sizes 1,15 --n_trials 200 --topn 100 --de_k 50 \
    --n_boot 1000 --n_perm 2000 --collect_controls --seed 42

#!/bin/bash
#SBATCH --job-name=workspace_probe
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/workspace_probe_%j.out
# Is drug identity encoded in a CAUSALLY INERT subspace? (workspace / privileged-subspace analysis)
# Bridges "drug is decodable (82%)" and "model ignores the drug" into one mechanistic claim, by
# testing whether the direction that ENCODES the drug is the direction that DRIVES generation.
#
# Guards that make the result interpretable (not an unfalsifiable null):
#   * POSITIVE CONTROL = cell-line subspace (the model DOES use it) -> its KL must be large.
#   * NULL = random subspace.
#   * ABLATION GATE: after ablating the drug subspace, drug decodability MUST collapse to chance.
#   * CONFOUND: the drug subspace is orthogonalized against the cell-line subspace before the causal
#     claim (drug and plate/cell-line are entangled — the selftest shows raw KL 1.42 vs orthogonalized
#     0.23, so the RAW number would overstate the drug's causal effect).
# Measurement is in LOGIT space (teacher-forced KL over response positions), not stochastic
# generation -> no sampling-noise floor (what made our earlier causal probe inconclusive).
#
# Runs on the SINGLE-CELL model (the drug-blind one the story is about). Flat paths for the cluster.

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
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final

$PY workspace_probe.py \
    --eval_dir "$DATA" --model_path "$CKPT" --tier tier2_unseen_drugs \
    --layers 2,4,6,8,9,12,16 --n_drugs 12 --n_per_drug 40 --n_dims 10 \
    --n_kl_prompts 60 --do_swap --bf16 --seed 42 \
    --out RESULTS/workspace_probe.json

echo "done -> RESULTS/workspace_probe.json"

#!/bin/bash
#SBATCH --job-name=dl_scvi_adata
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/dl_scvi_adata_%j.out
# Download the 42GB minified adata.h5ad from the released Tahoe scVI model repo. It holds the SHIPPED
# per-cell latent (obsm['X_latent_qzm']) that Mert used -> the JOIN source (re-encoding was degraded).
set -euo pipefail
mkdir -p logs
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
PY=/data/BuffaF-Projetcs/florian_c2s/envs/scvi_venv/bin/python
$PY -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('tahoebio/Tahoe-100M-SCVI-v1', 'adata.h5ad',
                    local_dir='/data/BuffaF-Projetcs/florian_c2s/SCVI_MODEL')
print('downloaded ->', p)
"
echo "done"

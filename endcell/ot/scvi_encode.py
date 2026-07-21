#!/usr/bin/env python
r"""
scvi_encode.py — encode Tahoe cells with the RELEASED tahoebio/Tahoe-100M-SCVI-v1 model (Arm 1c)
================================================================================================
Produces a per-cell 10-dim scVI latent cache (keyed by barcode) for our train cells, WITHOUT the
42 GB reference adata: we download only model.pt (~1 GB) and run scvi-tools query-data encoding.

Runs in a SEPARATE `scvi` env (scvi-tools==1.2.0 pins torch/lightning; keep it off the c2s env).
Output is an env-agnostic .npz that the OT/target pipeline (c2s env) reads.

FLOW
  1. gene_metadata.parquet -> var_names for the 62,710-gene model input space.
  2. stream Tahoe rows, build a sparse (cells x 62710) counts AnnData in that gene order, obs = barcode + plate.
  3. prepare_query_anndata(query, model_dir); model = SCVI.load_query_data(query, model_dir);
     latent = model.get_latent_representation()   # reference weights, no surgery training.
  4. save {barcode, latent(10-dim), plate, cell_line, drug, dose} to --out .npz.

SMOKE FIRST: `--smoke 2000` encodes ~2k streamed cells and prints latent shape + per-dim variance +
cell-line separ(silhouette) to confirm the model loads and the latent is sane before the full run.

USAGE (scvi env, cluster)
  # one-time model download (model.pt only, NOT the 42GB adata):
  #   python -c "from huggingface_hub import snapshot_download as s; \
  #       s('tahoebio/Tahoe-100M-SCVI-v1', allow_patterns=['model.pt','*.json','README*'], \
  #         local_dir='SCVI_MODEL')"
  python scvi_encode.py --model_dir SCVI_MODEL --smoke 2000 --out RESULTS/scvi_smoke.npz
  python scvi_encode.py --model_dir SCVI_MODEL --num_shards 24 --cells_per_condition 30 \
      --out /data/.../scvi_latent.npz
"""
import argparse, json, os, sys, logging
import numpy as np

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects the numpy globals stored in the
# scvi-tools 1.2.0 model.pt (raises UnpicklingError on numpy.core.multiarray._reconstruct). The model is
# the official tahoebio/Tahoe-100M-SCVI-v1 release (trusted source) -> restore the full (weights_only=False)
# load that scvi 1.2.0 expects. Applied before scvi imports torch.load so its internal calls pick it up.
try:
    import torch as _torch
    _ORIG_TORCH_LOAD = _torch.load
    def _full_load(*a, **k):
        k["weights_only"] = False
        return _ORIG_TORCH_LOAD(*a, **k)
    _torch.load = _full_load
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAHOE_REPO = "tahoebio/Tahoe-100M"


def load_var_names(repo):
    """62,710 gene identifiers in the model's expected order, from gene_metadata.parquet."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "metadata/gene_metadata.parquet", repo_type="dataset")
    gdf = pd.read_parquet(p)
    logger.info(f"gene_metadata: {len(gdf)} genes, columns={list(gdf.columns)}")
    # scVI var_names are almost always ensembl ids; fall back to symbol/token if not present.
    for col in ("ensembl_id", "gene_id", "ensembl", "gene_symbol", "gene_name", "symbol", "token_id"):
        if col in gdf.columns:
            logger.info(f"using gene_metadata['{col}'] as var_names")
            return gdf[col].astype(str).tolist(), gdf
    # last resort: the row 'genes' are positional indices -> use the frame's own index order
    return [str(i) for i in range(len(gdf))], gdf


def stream_cells(repo, var_names, n_target, num_shards, cells_per_condition, seed):
    """Stream Tahoe treated+DMSO rows; build a CSR counts matrix (cells x n_genes) in var_names order,
    plus obs. Caps per (drug, cell_line, plate, dose) like the preprocessor. Returns adata."""
    import anndata as ad
    from scipy import sparse
    from datasets import load_dataset
    G = len(var_names)
    rows_idx, cols_idx, vals = [], [], []
    obs = {"barcode": [], "plate": [], "cell_line_id": [], "drug": [], "dose": []}
    counts_per_cond = {}
    ds = load_dataset(repo, split="train", streaming=True)
    n = 0
    for row in ds:
        g = row.get("genes"); e = row.get("expressions")
        if g is None or e is None:
            continue
        drug = row.get("drug"); cl = row.get("cell_line_id"); plate = row.get("plate")
        bc = row.get("BARCODE_SUB_LIB_ID") or row.get("barcode")
        dose = row.get("sample")  # concentration proxy; kept as a separate condition axis
        key = (drug, cl, plate, dose)
        if counts_per_cond.get(key, 0) >= cells_per_condition:
            continue
        counts_per_cond[key] = counts_per_cond.get(key, 0) + 1
        # row 'genes' are positional indices into the gene vocabulary (0..G-1)
        gi = np.asarray(g, dtype=np.int64)
        ev = np.asarray(e, dtype=np.float32)
        m = (gi >= 0) & (gi < G)
        rows_idx.append(np.full(m.sum(), n, dtype=np.int64))
        cols_idx.append(gi[m]); vals.append(ev[m])
        for k, v in (("barcode", bc), ("plate", plate), ("cell_line_id", cl), ("drug", drug), ("dose", dose)):
            obs[k].append(v)
        n += 1
        if n % 5000 == 0:
            logger.info(f"  streamed {n} cells")
        if n >= n_target:
            break
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows_idx), np.concatenate(cols_idx))),
                          shape=(n, G))
    import pandas as pd
    A = ad.AnnData(X=X, obs=pd.DataFrame(obs))
    A.var_names = var_names
    A.obs_names = [str(b) for b in obs["barcode"]]
    logger.info(f"built query AnnData: {A.shape}")
    return A


def encode(adata, model_dir):
    import scvi
    logger.info(f"scvi-tools {scvi.__version__}; preparing query against {model_dir}")
    scvi.model.SCVI.prepare_query_anndata(adata, model_dir)
    model = scvi.model.SCVI.load_query_data(adata, model_dir)
    model.is_trained = True  # use reference weights directly, no surgery epochs
    latent = model.get_latent_representation()
    logger.info(f"latent: {latent.shape}")
    return np.asarray(latent, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="local dir with model.pt (snapshot_download)")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--smoke", type=int, default=0, help="encode this many cells and print sanity, then stop")
    ap.add_argument("--num_shards", type=int, default=24)
    ap.add_argument("--cells_per_condition", type=int, default=30)
    ap.add_argument("--max_cells", type=int, default=800000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/scvi_latent.npz")
    args = ap.parse_args()

    var_names, _ = load_var_names(args.repo)
    n_target = args.smoke if args.smoke else args.max_cells
    adata = stream_cells(args.repo, var_names, n_target, args.num_shards,
                         args.cells_per_condition, args.seed)
    latent = encode(adata, args.model_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, latent=latent,
        barcode=np.array(adata.obs["barcode"].tolist(), dtype=object),
        plate=np.array(adata.obs["plate"].tolist(), dtype=object),
        cell_line=np.array(adata.obs["cell_line_id"].tolist(), dtype=object),
        drug=np.array(adata.obs["drug"].tolist(), dtype=object),
        dose=np.array(adata.obs["dose"].tolist(), dtype=object))
    logger.info(f"-> {args.out}")

    if args.smoke:
        v = latent.var(0)
        logger.info(f"SMOKE latent per-dim variance: {np.round(v, 3)}")
        logger.info(f"SMOKE latent dims={latent.shape[1]} (expect 10)  cells={latent.shape[0]}")
        try:
            from sklearn.metrics import silhouette_score
            cl = np.array(adata.obs["cell_line_id"].tolist())
            keep = np.array([c is not None for c in cl])
            if keep.sum() > 50 and len(set(cl[keep])) > 1:
                s = silhouette_score(latent[keep], cl[keep])
                logger.info(f"SMOKE cell-line silhouette in latent: {s:.3f} (higher=cleaner cell-line clusters)")
        except Exception as e:
            logger.info(f"SMOKE silhouette skipped: {e}")
        logger.info("SMOKE OK if dims==10, variances are non-degenerate, silhouette clearly > 0.")


if __name__ == "__main__":
    main()

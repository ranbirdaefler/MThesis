#!/usr/bin/env python
r"""
inspect_scvi_adata.py — understand the shipped minified adata + VALIDATE its latent (Arm 1c JOIN)
=================================================================================================
Read-only. Opens the 42GB adata.h5ad (backed, no counts loaded) and reports the structure we need to
join the SHIPPED scVI latent (obsm['X_latent_qzm']) to our cells:
  * n_obs / n_vars, obs columns, obsm keys + shapes
  * the obs INDEX / barcode field and a few example values (to match parquet BARCODE_SUB_LIB_ID)
  * a cell-line PROBE on the shipped latent (a random sample) -> should be >> chance, proving the
    shipped latent is faithful (unlike our re-encode, which scored ~0.11).

USAGE (scvi env, cluster):
  python inspect_scvi_adata.py --adata /data/.../SCVI_MODEL/adata.h5ad --probe_n 5000
"""
import argparse, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--probe_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import anndata as ad
    logger.info(f"opening {args.adata} (backed)...")
    A = ad.read_h5ad(args.adata, backed="r")
    logger.info(f"n_obs={A.n_obs:,}  n_vars={A.n_vars:,}")
    logger.info(f"obs columns: {list(A.obs.columns)}")
    logger.info(f"obsm keys: { {k: A.obsm[k].shape for k in A.obsm.keys()} }")
    logger.info(f"obs index name: {A.obs.index.name}; examples: {list(map(str, A.obs.index[:5]))}")
    for c in A.obs.columns:
        try:
            ex = list(map(str, A.obs[c].iloc[:3].tolist()))
            logger.info(f"  obs['{c}'] e.g. {ex}")
        except Exception:
            pass

    # locate the latent
    latent_key = None
    for k in ("X_latent_qzm", "_scvi_latent_qzm", "X_scVI", "X_scvi"):
        if k in A.obsm:
            latent_key = k
            break
    if latent_key is None:
        logger.error(f"no latent obsm found among {list(A.obsm.keys())}")
        return
    logger.info(f"using latent obsm['{latent_key}'] shape={A.obsm[latent_key].shape}")

    # PROBE the shipped latent on a random sample
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(A.n_obs, min(args.probe_n, A.n_obs), replace=False))
    Z = np.asarray(A.obsm[latent_key][idx], dtype=np.float32)
    cl_col = "cell_line_id" if "cell_line_id" in A.obs.columns else \
             next((c for c in A.obs.columns if "cell_line" in c.lower() or "cell_name" in c.lower()), None)
    logger.info(f"latent sample: shape={Z.shape}  per-dim var={np.round(Z.var(0), 3)}")
    if cl_col is None:
        logger.warning("no cell-line column found; skipping probe")
    else:
        cls = A.obs[cl_col].values[idx].astype(str)
        uq, ct = np.unique(cls, return_counts=True)
        keep = set(uq[ct >= 10])
        m = np.array([c in keep for c in cls])
        logger.info(f"probe sample: {len(keep)} cell lines with >=10 cells, {m.sum()} cells")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            acc = float(np.mean(cross_val_score(
                LogisticRegression(max_iter=1000), Z[m], cls[m], cv=3)))
            logger.info(f"SHIPPED-LATENT cell-line PROBE accuracy: {acc:.3f} "
                        f"(chance ~ {1/max(1,len(keep)):.3f}) <<< expect >>chance (~0.7-0.95) if faithful")
        except Exception as e:
            logger.info(f"probe skipped: {e}")

    logger.info("KEY QUESTIONS ANSWERED: barcode field (obs index above), latent key, and whether the "
                "shipped latent separates cell lines. Next: join this latent to our cells by barcode.")


if __name__ == "__main__":
    main()

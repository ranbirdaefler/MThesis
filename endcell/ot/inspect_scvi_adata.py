#!/usr/bin/env python
r"""
inspect_scvi_adata.py — low-memory inspect + validate the shipped scVI latent (Arm 1c JOIN)
============================================================================================
Reads the 40GB adata.h5ad with raw h5py (NOT anndata) so we never load 100M rows into RAM. Reports:
  * HDF5 structure: root keys, obs columns (+ which are categorical), obsm keys + shapes
  * the obs INDEX / barcode field + example values (to match parquet BARCODE_SUB_LIB_ID)
  * a cell-line PROBE on the shipped latent from a random ROW SAMPLE -> should be >> chance, proving
    the shipped latent is faithful (unlike our re-encode, which scored ~0.11).

USAGE (scvi env, cluster):
  python inspect_scvi_adata.py --adata /data/.../SCVI_MODEL/adata.h5ad --probe_n 5000
"""
import argparse, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _decode(a):
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in a])


def read_categorical(obs, name, idx):
    """Read obs[name] at row indices idx, decoding an h5ad categorical (group of codes+categories)."""
    import h5py
    node = obs[name]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        cats = _decode(node["categories"][:])
        codes = node["codes"][idx]
        return np.array([cats[c] if 0 <= c < len(cats) else "NA" for c in codes])
    data = node[idx]
    return _decode(data) if data.dtype.kind in ("S", "O") else np.asarray(data).astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--probe_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import h5py
    logger.info(f"opening {args.adata} with h5py (low memory)...")
    f = h5py.File(args.adata, "r")
    logger.info(f"root keys: {list(f.keys())}")

    obs = f["obs"]
    obs_cols = [k for k in obs.keys()]
    logger.info(f"obs columns: {obs_cols}")
    idx_name = obs.attrs.get("_index", "_index")
    if isinstance(idx_name, (bytes, bytearray)):
        idx_name = idx_name.decode()
    logger.info(f"obs index dataset: '{idx_name}'")
    try:
        ex = _decode(obs[idx_name][:5])
        logger.info(f"obs index examples (barcodes?): {list(ex)}")
    except Exception as e:
        logger.info(f"could not read index examples: {e}")
    # show a few example values per obs column (helps spot the barcode/plate/drug/cell_line fields)
    for c in obs_cols:
        try:
            v = read_categorical(obs, c, np.arange(3))
            logger.info(f"  obs['{c}'] e.g. {list(v)}")
        except Exception:
            pass

    obsm = f["obsm"]
    logger.info(f"obsm keys + shapes: { {k: obsm[k].shape for k in obsm.keys()} }")
    latent_key = next((k for k in ("X_latent_qzm", "_scvi_latent_qzm", "X_scVI", "X_scvi")
                       if k in obsm), None)
    if latent_key is None:
        logger.error(f"no latent obsm found among {list(obsm.keys())}")
        return
    lat = obsm[latent_key]
    n_obs = lat.shape[0]
    logger.info(f"latent obsm['{latent_key}'] shape={lat.shape}  n_obs={n_obs:,}")

    # random ROW SAMPLE (sorted, unique -> valid h5py fancy indexing), read only those rows
    rng = np.random.RandomState(args.seed)
    idx = np.unique(rng.choice(n_obs, min(args.probe_n, n_obs), replace=False))
    Z = np.asarray(lat[idx], dtype=np.float32)
    logger.info(f"latent sample shape={Z.shape}  per-dim var={np.round(Z.var(0), 3)}")

    cl_col = next((c for c in ("Cell_ID_Cellosaur", "cell_line_id", "cell_line", "Cell_Name_Vevo",
                               "cell_name", "cellline") if c in obs_cols), None)
    if cl_col is None:
        logger.warning(f"no cell-line column found in {obs_cols}; skipping probe")
        return
    logger.info(f"probing with cell-line column '{cl_col}'")
    cls = read_categorical(obs, cl_col, idx)
    uq, ct = np.unique(cls, return_counts=True)
    keep = set(uq[ct >= 10])
    m = np.array([c in keep for c in cls])
    logger.info(f"probe sample: {len(keep)} cell lines with >=10 cells, {int(m.sum())} cells")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        acc = float(np.mean(cross_val_score(LogisticRegression(max_iter=1000), Z[m], cls[m], cv=3)))
        logger.info(f"SHIPPED-LATENT cell-line PROBE accuracy: {acc:.3f} "
                    f"(chance ~ {1/max(1,len(keep)):.3f}) <<< expect >>chance (~0.7-0.95) if faithful")
    except Exception as e:
        logger.info(f"probe skipped: {e}")
    logger.info("NEXT: use the obs index/barcode field + latent_key to join this latent to our cells.")


if __name__ == "__main__":
    main()

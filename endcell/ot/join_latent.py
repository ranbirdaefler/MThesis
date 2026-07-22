#!/usr/bin/env python
r"""
join_latent.py — attach the shipped scVI latent to our panel-expression cache by barcode (Arm 1c step 4)
========================================================================================================
The 40GB adata.h5ad holds the FAITHFUL shipped latent obsm['X_latent_qzm'] (95.6M cells, cell-line probe
0.975), indexed by BARCODE_SUB_LIB_ID -- the SAME key our build_embeddings cache carries. This joins the
two by barcode, memory-frugally: we hold only our ~1M-barcode set + one scan chunk, never the 95.6M-row
index. Output is a latent array aligned to the cache's cell order (NaN rows = barcode not found in adata).

USAGE (scvi env, cluster; needs h5py):
  python join_latent.py --cache_dir /data/.../ot_cache --adata /data/.../SCVI_MODEL/adata.h5ad \
      --out /data/.../ot_cache/scvi_latent.npz
  python join_latent.py --selftest
"""
import argparse, logging, os, sys
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _decode_block(a):
    if a.dtype.kind == "S":
        return a.astype(str)
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in a])


def join_latent_core(adata_path, barcodes, latent_key="X_latent_qzm", chunk=2_000_000,
                     sanity_cols=None):
    """barcodes: list aligned to cache order. Returns (latent [n x d] with NaN for missing, found mask,
    sanity dict of adata obs values at found rows for a few columns)."""
    import h5py
    f = h5py.File(adata_path, "r")
    idx_name = f["obs"].attrs.get("_index", "_index")
    if isinstance(idx_name, (bytes, bytearray)):
        idx_name = idx_name.decode()
    idx_ds = f["obs"][idx_name]
    lat = f["obsm"][latent_key]
    N, d = lat.shape[0], lat.shape[1]
    logger.info(f"adata: index='{idx_name}' n_obs={N:,}  latent='{latent_key}' dim={d}")

    bc2cache = {b: i for i, b in enumerate(barcodes)}
    n_cells = len(barcodes)
    found_cache, found_global = [], []
    for s in range(0, N, chunk):
        block = _decode_block(idx_ds[s:s + chunk])
        for j, b in enumerate(block):
            ci = bc2cache.get(b)
            if ci is not None:
                found_cache.append(ci); found_global.append(s + j)
        if (s // chunk) % 5 == 0:
            logger.info(f"  scanned {min(s+chunk, N):,}/{N:,}  found {len(found_cache):,}/{n_cells:,}")
    logger.info(f"barcode match: {len(found_cache):,}/{n_cells:,} ({len(found_cache)/max(1,n_cells):.1%})")

    out = np.full((n_cells, d), np.nan, dtype=np.float32)
    found_global = np.asarray(found_global); found_cache = np.asarray(found_cache)
    order = np.argsort(found_global)
    g_sorted, c_sorted = found_global[order], found_cache[order]
    for s in range(0, len(g_sorted), 100000):
        gs = g_sorted[s:s + 100000]
        out[c_sorted[s:s + 100000]] = np.asarray(lat[gs], dtype=np.float32)

    sanity = {}
    if sanity_cols and len(g_sorted):
        take = g_sorted[:min(2000, len(g_sorted))]
        tc = c_sorted[:len(take)]
        for col in sanity_cols:
            if col in f["obs"]:
                node = f["obs"][col]
                if isinstance(node, h5py.Group) and "codes" in node:
                    cats = _decode_block(node["categories"][:])
                    codes = node["codes"][take]
                    vals = np.array([cats[c] if 0 <= c < len(cats) else "NA" for c in codes])
                else:
                    vals = _decode_block(node[take])
                sanity[col] = (tc, vals)
    return out, ~np.isnan(out[:, 0]), sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--adata")
    ap.add_argument("--latent_key", default="X_latent_qzm")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not (args.cache_dir and args.adata):
        ap.error("--cache_dir and --adata required (unless --selftest)")

    import pandas as pd
    meta = pd.read_parquet(os.path.join(args.cache_dir, "meta.parquet"))
    barcodes = [str(b) for b in meta["barcode"].tolist()]
    logger.info(f"cache: {len(barcodes):,} cells")

    latent, found, sanity = join_latent_core(
        args.adata, barcodes, latent_key=args.latent_key, chunk=args.chunk,
        sanity_cols=["drug", "Cell_ID_Cellosaur", "plate"])

    # sanity: does adata's drug/cell_line at found rows match our cache metadata?
    if "drug" in sanity:
        tc, vals = sanity["drug"]
        ours = meta["drug"].values[tc].astype(str)
        agree = float(np.mean([o == v for o, v in zip(ours, vals)]))
        logger.info(f"SANITY drug agreement (adata vs cache) on {len(tc)} found cells: {agree:.3f} "
                    f"(expect ~1.0 if barcodes truly correspond)")
    if "Cell_ID_Cellosaur" in sanity:
        tc, vals = sanity["Cell_ID_Cellosaur"]
        ours = meta["cell_line_id"].values[tc].astype(str)
        agree = float(np.mean([o == v for o, v in zip(ours, vals)]))
        logger.info(f"SANITY cell-line agreement: {agree:.3f}")

    out = args.out or os.path.join(args.cache_dir, "scvi_latent.npz")
    np.savez_compressed(out, latent=latent, found=found,
                        barcode=np.array(barcodes, dtype=object))
    logger.info(f"latent: {latent.shape}, {int(found.sum())} found -> {out}")


def selftest():
    import h5py, tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, "a.h5ad")
    f = h5py.File(path, "w")
    obs = f.create_group("obs"); obs.attrs["_index"] = "BARCODE_SUB_LIB_ID"
    bcs = np.array([f"bc{i}".encode() for i in range(100)])
    obs.create_dataset("BARCODE_SUB_LIB_ID", data=bcs)
    # categorical drug
    g = obs.create_group("drug")
    g.create_dataset("categories", data=np.array([b"D0", b"D1"]))
    g.create_dataset("codes", data=(np.arange(100) % 2).astype(np.int8))
    obsm = f.create_group("obsm")
    lat = (np.arange(100 * 4).reshape(100, 4)).astype("float32")
    obsm.create_dataset("X_latent_qzm", data=lat)
    f.close()

    # cache barcodes: a subset in scrambled order, plus one missing barcode
    barcodes = ["bc10", "bc5", "bc50", "bc_missing", "bc99"]
    out, found, sanity = join_latent_core(path, barcodes, chunk=32, sanity_cols=["drug"])
    exp = {0: 10, 1: 5, 2: 50, 4: 99}
    ok = True
    for ci, gi in exp.items():
        if not np.allclose(out[ci], lat[gi]):
            logger.error(f"  FAIL row {ci}: got {out[ci]} expected {lat[gi]}"); ok = False
    if found[3] or not np.isnan(out[3, 0]):
        logger.error("  FAIL: missing barcode should be NaN/not-found"); ok = False
    if found.sum() != 4:
        logger.error(f"  FAIL: found {found.sum()} expected 4"); ok = False
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'} (scattered barcode join + missing handling)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

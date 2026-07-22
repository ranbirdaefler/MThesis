#!/usr/bin/env python
r"""
build_embeddings.py — panel-expression + PCA cache for the OT pipeline (Arm 1c, step 3)
=======================================================================================
Streams Tahoe (random shards for diversity), and for every treated + plate-matched DMSO cell builds:
  * barcode (BARCODE_SUB_LIB_ID)      -> join key for the shipped scVI latent (join_latent.py)
  * metadata: drug, cell_line_id, plate, dose (conc string), is_control
  * PANEL expression (946 L1000 genes), CP10K-normalized (per-cell library-normalized to 1e4)
    -> the source for BOTH the decoder-free OT targets (pi-weighted average of real treated cells)
       AND the PCA embedding (the parallel-to-scVI coupling space).
Then fits PCA on log1p(panel expr) and projects all cells.

This is the c2s-env, no-scVI half of "both embeddings in parallel". scVI latent is attached later by
barcode. Output is a cache DIRECTORY:
  meta.parquet            barcode, drug, cell_line_id, plate, dose, is_control, lib_size
  panel_expr.npz          scipy CSR (N x 946), CP10K linear (rank/average in this space)
  pca.npz                 coords (N x pca_dim), components, mean, explained_variance_ratio
  panel_genes.json        the 946 panel symbols in column order
  config.json             params + counts

USAGE (c2s env, cluster)
  python build_embeddings.py --selftest
  python build_embeddings.py --panel l1000_panel.json --num_shards 60 \
      --cells_per_condition 120 --control_per_group 200 --pca_dim 50 \
      --out_dir /data/.../ot_cache
"""
import argparse, json, os, sys, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAHOE_REPO = "tahoebio/Tahoe-100M"


def _is_control(drug):
    return drug is not None and "DMSO" in str(drug).upper()


def load_panel_symbols(panel_path):
    obj = json.load(open(panel_path))
    for k in ("genes", "panel", "symbols"):
        if isinstance(obj, dict) and k in obj:
            v = obj[k]
            return [g if isinstance(g, str) else g.get("symbol", g.get("gene")) for g in v]
    if isinstance(obj, list):
        return [g if isinstance(g, str) else g.get("symbol") for g in obj]
    raise ValueError(f"cannot parse panel json {panel_path}")


def build_tok2panel(repo, panel_symbols):
    """array indexed by TOKEN ID: token_id -> panel index (or -1). Tahoe row 'genes' are token IDs into
    the gene vocabulary (they exceed the 62,710-row metadata length), NOT positional indices; map via
    gene_metadata['token_id'] -> gene_symbol -> panel."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "metadata/gene_metadata.parquet", repo_type="dataset")
    gdf = pd.read_parquet(p)
    if "token_id" not in gdf.columns:
        raise ValueError(f"gene_metadata has no token_id column: {list(gdf.columns)}")
    tok = gdf["token_id"].astype(np.int64).values
    sym = gdf["gene_symbol"].astype(str).values
    panel_index = {s: i for i, s in enumerate(panel_symbols)}
    max_tok = int(tok.max())
    tok2panel = np.full(max_tok + 1, -1, dtype=np.int32)
    hit = 0
    for i in range(len(tok)):
        j = panel_index.get(sym[i], -1)
        if j >= 0 and 0 <= tok[i] <= max_tok:
            tok2panel[tok[i]] = j
            hit += 1
    logger.info(f"panel mapping via token_id: {hit}/{len(panel_symbols)} panel genes mapped; "
                f"max_token={max_tok}")
    return tok2panel


def stream_and_build(repo, tok2panel, n_panel, num_shards, cells_per_condition,
                     control_per_group, max_cells, seed, smoke=0):
    """Stream random shards; build CP10K panel expression + metadata. Returns (CSR expr, meta dict)."""
    from scipy import sparse
    from datasets import load_dataset
    from huggingface_hub import HfApi
    all_files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                 if f.startswith("data/") and f.endswith(".parquet")]
    rng = np.random.RandomState(seed)
    k = min(max(1, num_shards), len(all_files))
    picks = sorted(rng.choice(len(all_files), k, replace=False).tolist())
    urls = [f"https://huggingface.co/datasets/{repo}/resolve/main/{all_files[i]}" for i in picks]
    logger.info(f"streaming {k} random shards of {len(all_files)}")
    ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)

    rows_i, cols_i, vals = [], [], []
    meta = {"barcode": [], "drug": [], "cell_line_id": [], "plate": [], "dose": [],
            "is_control": [], "lib_size": []}
    treated_cnt, control_cnt = {}, {}
    from collections import Counter
    drug_seen = Counter()
    target = smoke if smoke else max_cells
    n = 0
    for row in ds:
        g, e = row.get("genes"), row.get("expressions")
        if g is None or e is None:
            continue
        drug, cl, plate = row.get("drug"), row.get("cell_line_id"), row.get("plate")
        dose = row.get("sample")
        drug_seen[str(drug)] += 1
        is_ctrl = _is_control(drug)
        if is_ctrl:
            key = (cl, plate)
            if control_cnt.get(key, 0) >= control_per_group:
                continue
            control_cnt[key] = control_cnt.get(key, 0) + 1
        else:
            key = (drug, cl, plate, dose)
            if treated_cnt.get(key, 0) >= cells_per_condition:
                continue
            treated_cnt[key] = treated_cnt.get(key, 0) + 1
        gi = np.asarray(g, dtype=np.int64)
        ev = np.maximum(np.asarray(e, dtype=np.float32), 0.0)   # clip rare negatives (see scvi_encode)
        lib = float(ev.sum())                                   # full-library size (all genes)
        if lib <= 0:
            continue
        # map token IDs -> panel; drop out-of-vocab tokens (genes exceed the metadata length)
        p = np.full(len(gi), -1, dtype=np.int32)
        inrange = (gi >= 0) & (gi < len(tok2panel))
        p[inrange] = tok2panel[gi[inrange]]
        keep = p >= 0
        if keep.any():
            pj = p[keep]
            cp10k = ev[keep] / lib * 1e4                        # CP10K (linear); rank/average in this space
            # collapse duplicate panel indices within the cell
            order = np.argsort(pj)
            pj_s, cv_s = pj[order], cp10k[order]
            uniq, start = np.unique(pj_s, return_index=True)
            summed = np.add.reduceat(cv_s, start)
            rows_i.append(np.full(len(uniq), n, dtype=np.int64))
            cols_i.append(uniq.astype(np.int64)); vals.append(summed.astype(np.float32))
        meta["barcode"].append(row.get("BARCODE_SUB_LIB_ID") or row.get("barcode"))
        meta["drug"].append(drug); meta["cell_line_id"].append(cl); meta["plate"].append(plate)
        meta["dose"].append(dose); meta["is_control"].append(bool(is_ctrl)); meta["lib_size"].append(lib)
        n += 1
        if n % 20000 == 0:
            logger.info(f"  built {n} cells ({sum(meta['is_control'])} control)")
        if n >= target:
            break
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))),
                          shape=(n, n_panel)) if vals else sparse.csr_matrix((n, n_panel))
    logger.info(f"built {n} cells; {sum(meta['is_control'])} control, {n - sum(meta['is_control'])} treated")
    dmso_like = [(d, c) for d, c in drug_seen.most_common() if "DMSO" in d.upper()]
    logger.info(f"distinct drugs seen: {len(drug_seen)}; DMSO-like: {dmso_like[:5]}; "
                f"top drugs: {drug_seen.most_common(8)}")
    return X, meta


def fit_pca(X_csr, dim, seed, subsample=60000):
    """Fit PCA on log1p(CP10K) using a random subsample for the components, then project all in chunks."""
    from sklearn.decomposition import PCA
    n = X_csr.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, min(subsample, n), replace=False)
    Xs = np.log1p(X_csr[idx].toarray())
    d = min(dim, Xs.shape[1], len(idx) - 1)
    pca = PCA(n_components=d, random_state=seed).fit(Xs)
    logger.info(f"PCA fit on {len(idx)} cells -> {d} comps; cum var={pca.explained_variance_ratio_.sum():.3f}")
    coords = np.empty((n, d), dtype=np.float32)
    for s in range(0, n, 20000):
        coords[s:s+20000] = pca.transform(np.log1p(X_csr[s:s+20000].toarray())).astype(np.float32)
    return coords, pca


def save_cache(out_dir, X, meta, coords, pca, panel_symbols, cfg):
    import pandas as pd
    from scipy import sparse
    os.makedirs(out_dir, exist_ok=True)
    sparse.save_npz(os.path.join(out_dir, "panel_expr.npz"), X)
    pd.DataFrame(meta).to_parquet(os.path.join(out_dir, "meta.parquet"))
    np.savez_compressed(os.path.join(out_dir, "pca.npz"), coords=coords,
                        components=pca.components_.astype(np.float32),
                        mean=pca.mean_.astype(np.float32),
                        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32))
    json.dump(list(panel_symbols), open(os.path.join(out_dir, "panel_genes.json"), "w"))
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2, default=str)
    logger.info(f"-> cache written to {out_dir}")


def selftest():
    """Synthetic: 3 cell lines x 3 latent programs; verify panel build, PCA separates cell lines."""
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_panel, n_genes = 40, 200
    pos2panel = np.full(n_genes, -1, dtype=np.int32)
    pos2panel[:n_panel] = np.arange(n_panel)                    # first 40 gene positions are the panel
    # fabricate rows
    rows_i, cols_i, vals, meta = [], [], [], {"cell_line_id": []}
    programs = {c: rng.rand(n_panel) * 5 for c in range(3)}
    N = 300
    for i in range(N):
        c = i % 3
        base = programs[c]
        expr = np.maximum(0, base + rng.randn(n_panel) * 0.5)
        lib = expr.sum() + 1e-6
        cp = expr / lib * 1e4
        nz = cp > 0
        rows_i.append(np.full(nz.sum(), i)); cols_i.append(np.where(nz)[0]); vals.append(cp[nz])
        meta["cell_line_id"].append(c)
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))),
                          shape=(N, n_panel), dtype=np.float32)
    coords, pca = fit_pca(X, 5, 0, subsample=N)
    cl = np.array(meta["cell_line_id"])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    acc = float(np.mean(cross_val_score(LogisticRegression(max_iter=500), coords, cl, cv=3)))
    logger.info(f"SELFTEST PCA cell-line probe acc={acc:.3f} (chance 0.33)")
    ok = X.shape == (N, n_panel) and acc > 0.8
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="l1000_panel.json")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--num_shards", type=int, default=60)
    ap.add_argument("--cells_per_condition", type=int, default=120)
    ap.add_argument("--control_per_group", type=int, default=200)
    ap.add_argument("--max_cells", type=int, default=1_500_000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="RESULTS/ot_cache")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    panel_symbols = load_panel_symbols(args.panel)
    tok2panel = build_tok2panel(args.repo, panel_symbols)
    X, meta = stream_and_build(args.repo, tok2panel, len(panel_symbols), args.num_shards,
                               args.cells_per_condition, args.control_per_group,
                               args.max_cells, args.seed, smoke=args.smoke)
    coords, pca = fit_pca(X, args.pca_dim, args.seed)
    cfg = {k: v for k, v in vars(args).items()}
    cfg["n_cells"] = int(X.shape[0]); cfg["n_control"] = int(sum(meta["is_control"]))
    save_cache(args.out_dir, X, meta, coords, pca, panel_symbols, cfg)

    # quick sanity: PCA cell-line probe (should be >> chance if panel expr carries structure)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        cl = np.array([str(c) for c in meta["cell_line_id"]])
        uq, ct = np.unique(cl, return_counts=True)
        keep = set(uq[ct >= 10]); m = np.array([c in keep for c in cl])
        if m.sum() > 100 and len(keep) > 2:
            acc = float(np.mean(cross_val_score(LogisticRegression(max_iter=500),
                                                coords[m], cl[m], cv=3)))
            logger.info(f"PCA cell-line probe acc={acc:.3f} (chance ~{1/len(keep):.3f}, {len(keep)} lines)")
    except Exception as e:
        logger.info(f"probe skipped: {e}")


if __name__ == "__main__":
    main()

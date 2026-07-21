#!/usr/bin/env python
r"""
scout_data.py — resolve the scVI acquisition path before building anything (Arm 1c, Step 1)
===========================================================================================
READ-ONLY. Answers three questions that decide how we get the embeddings for the OT pipeline:

  (1) Does Tahoe-100M ship PRE-COMPUTED per-cell scVI embeddings we can JOIN? (fast path)
      -> list the HF repo tree for anything embedding/scVI/latent-shaped, and peek an expression
         shard's columns for an embedding field.
  (2) If not, can we ENCODE with the released scVI model? -> is `scvi-tools` importable, and is the
      released model reachable (repo files / a path)?
  (3) What is the raw gene space? -> gene_metadata size + how many of our 946 panel genes are present
      (scVI needs the full gene vector in its own order; PCA only needs the panel).

Prints a VERDICT naming the scVI path (join / encode / none) so we can commit the builder. Also
confirms the row schema the OT builder will stream (genes / expressions / drug / cell_line_id / plate /
sample) and whether a dose/embedding column already exists.

USAGE (cluster, from ~/tahoe):
  python endcell/ot/scout_data.py --panel shared/l1000_panel.json
  python endcell/ot/scout_data.py --panel shared/l1000_panel.json --scvi_model_path /some/scvi_model
"""
import argparse, json, os, sys, re


def _load_panel_symbols(panel_path):
    if not panel_path or not os.path.exists(panel_path):
        return None
    obj = json.load(open(panel_path))
    # panel json may be {"genes": [...]} or a list or {"panel": [...]}
    for k in ("genes", "panel", "symbols"):
        if isinstance(obj, dict) and k in obj:
            v = obj[k]
            return [g if isinstance(g, str) else g.get("symbol", g.get("gene")) for g in v]
    if isinstance(obj, list):
        return [g if isinstance(g, str) else g.get("symbol") for g in obj]
    return None


def scout_repo_files(repo):
    out = {"scvi_like_files": [], "n_files": 0, "expression_shards": 0, "listing_error": None}
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        out["n_files"] = len(files)
        pat = re.compile(r"scvi|scVI|embedding|embed|latent|obsm|\.pt$|\.ckpt$|model", re.I)
        out["scvi_like_files"] = [f for f in files if pat.search(f)][:60]
        out["expression_shards"] = sum(1 for f in files if f.endswith(".parquet")
                                       and not f.startswith("metadata/"))
    except Exception as e:
        out["listing_error"] = repr(e)
    return out


def peek_row(repo):
    """Stream one expression row and report its columns + whether any embedding-like field exists."""
    out = {"columns": None, "embedding_field": None, "n_genes_in_row": None, "error": None}
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, split="train", streaming=True)
        row = next(iter(ds))
        cols = list(row.keys())
        out["columns"] = cols
        emb = [c for c in cols if re.search(r"scvi|embed|latent|obsm|x_", c, re.I)]
        out["embedding_field"] = emb or None
        if "genes" in row:
            try:
                out["n_genes_in_row"] = len(row["genes"])
            except Exception:
                pass
    except Exception as e:
        out["error"] = repr(e)
    return out


def gene_space(repo, panel_symbols):
    out = {"n_genes_metadata": None, "panel_size": None, "panel_in_metadata": None, "error": None}
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, "metadata/gene_metadata.parquet", repo_type="dataset")
        gdf = pd.read_parquet(p)
        out["gene_metadata_columns"] = list(gdf.columns)
        out["n_genes_metadata"] = int(len(gdf))
        if panel_symbols:
            sym_cols = [c for c in gdf.columns if re.search(r"symbol|gene_name|name", c, re.I)]
            present = set()
            for c in sym_cols:
                present |= set(map(str, gdf[c].tolist()))
            hit = sum(1 for g in panel_symbols if g in present)
            out["panel_size"] = len(panel_symbols)
            out["panel_in_metadata"] = hit
            out["panel_symbol_columns_used"] = sym_cols
    except Exception as e:
        out["error"] = repr(e)
    return out


def scvi_tools_available():
    try:
        import scvi  # noqa
        return getattr(scvi, "__version__", "unknown")
    except Exception as e:
        return f"NOT importable: {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="tahoebio/Tahoe-100M")
    ap.add_argument("--panel", default="shared/l1000_panel.json")
    ap.add_argument("--scvi_model_path", default=None, help="local path to a released scVI model, if known")
    ap.add_argument("--out", default="RESULTS/ot_scout.json")
    args = ap.parse_args()

    panel = _load_panel_symbols(args.panel)
    report = {
        "repo": args.repo,
        "scvi_tools": scvi_tools_available(),
        "repo_files": scout_repo_files(args.repo),
        "row": peek_row(args.repo),
        "genes": gene_space(args.repo, panel),
        "scvi_model_path": args.scvi_model_path,
        "scvi_model_path_exists": bool(args.scvi_model_path and os.path.exists(args.scvi_model_path)),
    }

    # ---- verdict: which scVI path is available? ----
    row_emb = report["row"].get("embedding_field")
    repo_hits = report["repo_files"].get("scvi_like_files") or []
    tools_ok = not str(report["scvi_tools"]).startswith("NOT")
    if row_emb:
        path = f"JOIN (per-cell embedding in the stream: {row_emb})"
    elif any(re.search(r"embed|latent|obsm", f, re.I) for f in repo_hits):
        path = "JOIN (released embedding file in repo — download + index by cell id)"
    elif (report["scvi_model_path_exists"] or any(re.search(r"scvi|model|\.pt$|\.ckpt$", f, re.I)
                                                  for f in repo_hits)) and tools_ok:
        path = "ENCODE (released scVI model + scvi-tools available — run the encoder on raw counts)"
    else:
        path = "NONE found (proceed with PCA only; revisit scVI if a model/embedding surfaces)"
    report["scvi_verdict"] = path

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, default=str)

    print("=" * 78)
    print(f"scvi-tools:        {report['scvi_tools']}")
    print(f"repo files:        {report['repo_files']['n_files']} "
          f"({report['repo_files']['expression_shards']} expression shards)")
    print(f"scvi-like files:   {repo_hits[:12]}{' ...' if len(repo_hits) > 12 else ''}")
    print(f"row columns:       {report['row'].get('columns')}")
    print(f"embedding in row:  {row_emb}")
    print(f"genes (metadata):  {report['genes'].get('n_genes_metadata')}  "
          f"panel present: {report['genes'].get('panel_in_metadata')}/{report['genes'].get('panel_size')}")
    print("-" * 78)
    print(f"SCVI PATH -> {path}")
    print("=" * 78)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

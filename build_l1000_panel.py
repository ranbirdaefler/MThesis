"""
build_l1000_panel.py — Build the fixed L1000 landmark gene panel.

This is the single source of truth for the fixed gene panel used by
`tahoe_c2s_preprocess.py` (data construction) and `evaluate_c2s_tahoe.py`
(metric scoring). Run it ONCE, commit the outputs alongside the data.

Outputs (written next to this script unless --out_dir given):
  l1000_landmark_genes.txt  — the LINCS L1000 landmark gene symbols, one per line
  l1000_panel.json          — landmark genes ∩ Tahoe vocabulary, in canonical
                              (landmark) order. This is what preprocessing/eval read.

Provenance
----------
The canonical landmark list is the LINCS L1000 gene_info table from GEO
GSE92742, keeping rows where `pr_is_lm == 1` (978 landmark genes):
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz

If `--landmark_file` already exists it is reused (no download). Run this on a
machine WITH internet (e.g. a login node if compute nodes are offline).

Usage
-----
    python build_l1000_panel.py
    python build_l1000_panel.py --out_dir ./data
    python build_l1000_panel.py --landmark_file l1000_landmark_genes.txt
"""

import argparse
import gzip
import json
import logging
import os
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_gene_info.txt.gz"
)


def download_landmark_genes(url):
    """Download the LINCS gene_info table and return (symbols, entrez_ids) for
    rows where pr_is_lm == 1, preserving file order and de-duplicating.
    """
    logger.info(f"Downloading LINCS gene_info from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    lines = text.splitlines()
    header = lines[0].split("\t")

    def col(*names):
        for n in names:
            if n in header:
                return header.index(n)
        raise KeyError(f"None of {names} in gene_info header: {header}")

    sym_idx = col("pr_gene_symbol")
    lm_idx = col("pr_is_lm")
    id_idx = col("pr_gene_id")

    symbols, entrez = [], []
    seen = set()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(sym_idx, lm_idx, id_idx):
            continue
        if parts[lm_idx].strip() == "1":
            sym = parts[sym_idx].strip()
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
                entrez.append(parts[id_idx].strip())
    logger.info(f"  Parsed {len(symbols)} landmark gene symbols (pr_is_lm == 1)")
    return symbols, entrez


def load_landmark_file(path):
    with open(path) as f:
        syms = [l.strip() for l in f if l.strip()]
    # de-dupe preserving order
    out, seen = [], set()
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    logger.info(f"  Loaded {len(out)} landmark symbols from {path}")
    return out


def load_tahoe_gene_metadata(local_path=None):
    """Return the Tahoe gene metadata DataFrame (downloads the small parquet if needed)."""
    import pandas as pd

    if local_path and os.path.exists(local_path):
        logger.info(f"Reading Tahoe gene metadata from {local_path}")
        return pd.read_parquet(local_path)

    from huggingface_hub import hf_hub_download

    logger.info("Downloading Tahoe gene_metadata.parquet (small file, not the dataset) ...")
    gpath = hf_hub_download(
        "tahoebio/Tahoe-100M", "metadata/gene_metadata.parquet", repo_type="dataset"
    )
    return pd.read_parquet(gpath)


def resolve_aliases(missing_symbols, tahoe_symbol_set, entrez_for_missing=None,
                    tahoe_df=None):
    """Best-effort recovery of landmark symbols not found in the Tahoe vocab.

    Returns dict {landmark_symbol -> tahoe_symbol}. Tries, in order:
      1. Entrez-id match against a Tahoe id column (authoritative, alias-proof)
      2. mygene symbol->current-symbol resolution (if `mygene` is installed)

    No guessed/hardcoded aliases — we only map when an authoritative source agrees.
    """
    resolved = {}

    # --- 1. Entrez-id based recovery, if Tahoe metadata exposes an Entrez/NCBI id ---
    if tahoe_df is not None and entrez_for_missing is not None:
        id_cols = [c for c in tahoe_df.columns
                   if any(k in c.lower() for k in ("entrez", "ncbi", "gene_id"))]
        sym_col = "gene_symbol" if "gene_symbol" in tahoe_df.columns else None
        for c in id_cols:
            if sym_col is None:
                break
            try:
                id_to_sym = {str(i): str(s) for i, s in zip(tahoe_df[c], tahoe_df[sym_col])}
            except Exception:
                continue
            for sym, eid in zip(missing_symbols, entrez_for_missing):
                if sym in resolved or not eid:
                    continue
                cand = id_to_sym.get(str(eid))
                if cand and cand in tahoe_symbol_set:
                    resolved[sym] = cand
            if resolved:
                logger.info(f"  Entrez-id match via Tahoe column '{c}' recovered "
                            f"{len(resolved)} symbols")

    # --- 2. mygene resolution for whatever is still missing ---
    still = [s for s in missing_symbols if s not in resolved]
    if still:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            hits = mg.querymany(still, scopes="symbol,alias", fields="symbol",
                                species="human", verbose=False)
            for h in hits:
                q, cur = h.get("query"), h.get("symbol")
                if cur and cur in tahoe_symbol_set:
                    resolved[q] = cur
            logger.info(f"  mygene recovered {sum(1 for s in still if s in resolved)} "
                        f"of {len(still)} remaining symbols")
        except ImportError:
            logger.warning("  `mygene` not installed — skipping alias resolution. "
                           "Install with `pip install mygene` to recover more genes.")
        except Exception as e:
            logger.warning(f"  mygene query failed ({e}); skipping alias resolution.")

    return resolved


def main():
    ap = argparse.ArgumentParser(description="Build the fixed L1000 landmark gene panel")
    ap.add_argument("--out_dir", default=".", help="Directory to write outputs")
    ap.add_argument("--landmark_file", default=None,
                    help="Pre-existing landmark gene list (one symbol/line). "
                         "Default: <out_dir>/l1000_landmark_genes.txt (downloaded if absent)")
    ap.add_argument("--tahoe_gene_metadata", default=None,
                    help="Local Tahoe gene_metadata.parquet (downloaded from HF if absent)")
    ap.add_argument("--gene_info_url", default=GENE_INFO_URL,
                    help="LINCS gene_info.txt.gz URL")
    ap.add_argument("--min_expected", type=int, default=900,
                    help="If the intersection is below this, attempt alias resolution")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    landmark_path = args.landmark_file or os.path.join(args.out_dir, "l1000_landmark_genes.txt")
    panel_path = os.path.join(args.out_dir, "l1000_panel.json")

    # --- 1. Landmark genes (symbols + Entrez ids) ---
    entrez = None
    if os.path.exists(landmark_path):
        landmark = load_landmark_file(landmark_path)
    else:
        landmark, entrez = download_landmark_genes(args.gene_info_url)
        with open(landmark_path, "w") as f:
            f.write("\n".join(landmark) + "\n")
        logger.info(f"  Wrote {len(landmark)} landmark symbols to {landmark_path}")

    if not landmark:
        logger.error("No landmark genes obtained — aborting.")
        sys.exit(1)

    # --- 2. Tahoe vocabulary ---
    tahoe_df = load_tahoe_gene_metadata(args.tahoe_gene_metadata)
    logger.info(f"  Tahoe gene metadata columns: {list(tahoe_df.columns)}")
    if "gene_symbol" not in tahoe_df.columns:
        logger.error("Tahoe gene metadata has no 'gene_symbol' column — aborting.")
        sys.exit(1)
    tahoe_syms = set(tahoe_df["gene_symbol"].astype(str))
    logger.info(f"  Tahoe vocabulary: {len(tahoe_syms)} gene symbols")

    # --- 3. Intersect, preserving landmark order as the canonical order ---
    panel = [g for g in landmark if g in tahoe_syms]
    logger.info(f"{len(panel)}/{len(landmark)} landmark genes found directly in Tahoe vocab")

    # --- 4. Alias resolution if the direct intersection is materially low ---
    if len(panel) < args.min_expected:
        missing = [g for g in landmark if g not in tahoe_syms]
        ent_for_missing = None
        if entrez is not None:
            emap = dict(zip(landmark, entrez))
            ent_for_missing = [emap.get(g) for g in missing]
        logger.warning(f"  Direct intersection {len(panel)} < {args.min_expected}; "
                       f"attempting alias resolution for {len(missing)} missing genes")
        resolved = resolve_aliases(missing, tahoe_syms, ent_for_missing, tahoe_df)
        if resolved:
            in_panel = set(panel)
            rebuilt = []
            for g in landmark:
                tahoe_sym = g if g in tahoe_syms else resolved.get(g)
                if tahoe_sym and tahoe_sym not in in_panel:
                    in_panel.add(tahoe_sym)
                    rebuilt.append(tahoe_sym)
            panel = rebuilt
            logger.info(f"  After alias resolution: {len(panel)}/{len(landmark)} genes")
        still_missing = [g for g in landmark if g not in tahoe_syms and g not in resolved]
        logger.warning(f"  {len(still_missing)} genes still unresolved "
                       f"(first 20): {still_missing[:20]}")

    # --- 5. Write the panel ---
    with open(panel_path, "w") as f:
        json.dump(panel, f)

    logger.info("=" * 60)
    logger.info(f"PANEL SIZE: {len(panel)} genes  ->  {panel_path}")
    logger.info(f"  (from {len(landmark)} LINCS L1000 landmark genes)")
    logger.info(f"  First 10: {panel[:10]}")
    logger.info("=" * 60)
    if len(panel) < args.min_expected:
        logger.warning("Panel is below the expected size. Review unresolved genes "
                       "above before running full preprocessing.")


if __name__ == "__main__":
    main()

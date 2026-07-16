"""
Tahoe-100M → C2S Training Data Constructor

Constructs (drug + dose + cell_line) → treated cell sentence pairs
for fine-tuning C2S-Scale on perturbation prediction.

Usage:
    # Local test (tiny subsample)
    python tahoe_c2s_preprocess.py --mode test --output_dir ./test_output

    # Full HPC run
    python tahoe_c2s_preprocess.py --mode full --output_dir /path/to/output \
        --cells_per_condition 10 --held_out_drugs 50
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: Load metadata tables (small, fits in memory)
# =============================================================================

def load_metadata():
    """Load gene, sample, drug, and cell line metadata from HuggingFace.
    
    Uses direct parquet file loading to avoid downloading the full expression dataset.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    repo_id = "tahoebio/Tahoe-100M"
    
    logger.info("Loading gene metadata...")
    path = hf_hub_download(repo_id, "metadata/gene_metadata.parquet", repo_type="dataset")
    gene_df = pd.read_parquet(path)
    logger.info(f"  Gene metadata columns: {list(gene_df.columns)}")
    logger.info(f"  Shape: {gene_df.shape}")
    
    gene_id_to_symbol = dict(zip(gene_df["token_id"], gene_df["gene_symbol"]))
    logger.info(f"  Loaded {len(gene_id_to_symbol)} gene symbols")
    logger.info(f"  Sample: {dict(list(gene_id_to_symbol.items())[:3])}")

    logger.info("Loading sample metadata...")
    path = hf_hub_download(repo_id, "metadata/sample_metadata.parquet", repo_type="dataset")
    sample_df = pd.read_parquet(path)
    logger.info(f"  Sample metadata columns: {list(sample_df.columns)}")
    logger.info(f"  Shape: {sample_df.shape}")
    logger.info(f"  First row:\n{sample_df.iloc[0].to_dict()}")
    
    sample_to_conc = {}
    for _, row in sample_df.iterrows():
        sample_id = row.get("sample", None)
        conc = row.get("drugname_drugconc", "unknown")
        if sample_id is not None:
            sample_to_conc[str(sample_id)] = str(conc)
    logger.info(f"  Loaded {len(sample_to_conc)} sample entries")

    logger.info("Loading drug metadata...")
    path = hf_hub_download(repo_id, "metadata/drug_metadata.parquet", repo_type="dataset")
    drug_df = pd.read_parquet(path)
    logger.info(f"  Drug metadata columns: {list(drug_df.columns)}")
    logger.info(f"  Shape: {drug_df.shape}")
    
    drug_info = {}
    for _, row in drug_df.iterrows():
        drug_name = row.get("drug", None)
        if drug_name:
            drug_info[drug_name] = {
                "moa": str(row.get("moa-fine", row.get("moa_fine", "unknown"))),
                "approved": str(row.get("fda-approved", row.get("fda_approved", "unknown"))),
            }
    logger.info(f"  Loaded {len(drug_info)} drug entries")

    logger.info("Loading cell line metadata...")
    path = hf_hub_download(repo_id, "metadata/cell_line_metadata.parquet", repo_type="dataset")
    cl_df = pd.read_parquet(path)
    logger.info(f"  Cell line metadata columns: {list(cl_df.columns)}")
    logger.info(f"  Shape: {cl_df.shape}")
    logger.info(f"  First row:\n{cl_df.iloc[0].to_dict()}")
    
    cvcl_to_name = {}
    for _, row in cl_df.iterrows():
        cvcl_id = row.get("Cell_ID_Cellosaur", None)
        name = row.get("cell_name", None)
        if name is None or (isinstance(name, float) and pd.isna(name)):
            name = str(cvcl_id)
        if cvcl_id is not None:
            cvcl_to_name[str(cvcl_id)] = str(name)
    logger.info(f"  Loaded {len(cvcl_to_name)} cell line entries")
    logger.info(f"  Sample mappings: {dict(list(cvcl_to_name.items())[:5])}")

    return gene_id_to_symbol, sample_to_conc, drug_info, cvcl_to_name


# =============================================================================
# Step 2: C2S transformation — raw counts to cell sentence
# =============================================================================

def raw_to_cell_sentence(gene_indices, expressions, gene_id_to_symbol, max_genes=None):
    """
    Convert sparse (gene_index, expression) pairs to a C2S cell sentence.
    
    Steps:
        1. Filter out zero/negative expression
        2. Library-size normalize to 10,000
        3. log10(1+x) transform (C2S convention)
        4. Rank-order genes by descending normalized expression
        5. Map gene indices to gene symbols
        6. Return space-separated gene name string
    
    Args:
        gene_indices: list of integer gene token IDs
        expressions: list of raw counts (matching gene_indices)
        gene_id_to_symbol: dict mapping token_id -> gene_symbol
        max_genes: optional cap on number of genes in sentence
    
    Returns:
        cell_sentence: string of space-separated gene names, or None if cell fails QC
    """
    # Convert to numpy for vectorized ops
    gene_ids = np.array(gene_indices)
    expr = np.array(expressions, dtype=np.float64)

    # Filter: keep only positive expression
    mask = expr > 0
    gene_ids = gene_ids[mask]
    expr = expr[mask]

    if len(expr) == 0:
        return None

    # QC: skip cells with fewer than 200 expressed genes
    if len(expr) < 200:
        return None

    # Check mitochondrial fraction
    mito_count = 0
    total_count = expr.sum()
    for i, gid in enumerate(gene_ids):
        symbol = gene_id_to_symbol.get(gid, "")
        if symbol.startswith("MT-"):
            mito_count += expr[i]
    
    if total_count > 0 and (mito_count / total_count) > 0.20:
        return None

    # Library-size normalize to 10,000 + log10(1+x) per C2S convention
    expr_norm = (expr / total_count) * 1e4
    expr_norm = np.log10(1 + expr_norm)

    # Rank-order by descending normalized expression
    sorted_idx = np.argsort(-expr_norm)
    
    # Apply max_genes cap if specified
    if max_genes is not None:
        sorted_idx = sorted_idx[:max_genes]

    # Map to gene symbols
    gene_names = []
    for idx in sorted_idx:
        gid = gene_ids[idx]
        symbol = gene_id_to_symbol.get(gid, None)
        if symbol is not None:
            gene_names.append(symbol)

    if len(gene_names) < 50:  # too few valid genes after mapping
        return None

    return " ".join(gene_names)


# Sentinel appended to every cell sentence to mark the end of the EXPRESSED gene
# block. Under the [END_CELL] representation, only expressed panel genes are emitted
# (ranked by expression); unexpressed genes are ABSENT (implied zero) rather than
# appended as a canonical tail. This makes on/off transitions analyzable and lets the
# three inactive-gene representations (position / tail_max / zero_bucket) actually
# differ downstream. Must be a token the tokenizer treats atomically when retraining
# (register as a special token in train_c2s_tahoe.py); as a plain string it is safe
# because it cannot collide with any gene symbol.
END_CELL_TOKEN = "[END_CELL]"


def build_panel_sentence(genes, exprs, gene_id_to_symbol, panel_symbols, panel_index,
                         min_expressed=200, min_panel_expressed=0):
    """Build ONE fixed-panel cell sentence from raw (genes, exprs).

    [END_CELL] representation: returns the space-joined EXPRESSED panel genes ordered
    by descending normalized expression, followed by the END_CELL_TOKEN sentinel.
    Unexpressed panel genes are ABSENT (implied zero) — there is no canonical tail.
    Sentences are therefore variable length (= number of expressed panel genes + 1).
    Returns None if the cell fails QC. Shared by make_paired_cell_sentences_fixed_panel
    and the eval-tier regenerator so train/eval sentence construction can never diverge.
    """
    gid = np.array(genes)
    ex = np.array(exprs, dtype=np.float64)
    m = ex > 0
    gid, ex = gid[m], ex[m]
    if len(ex) < min_expressed:          # general per-cell QC
        return None
    tot = ex.sum()
    if tot == 0:
        return None
    norm = np.log10(1 + (ex / tot) * 1e4)
    d = {}
    for g, v in zip(gid, norm):
        sym = gene_id_to_symbol.get(g, None)
        if sym is not None:
            d[sym] = float(v)
    # Expressed PANEL genes only (present in this cell with expression > 0).
    expressed_panel = [g for g in panel_symbols if d.get(g, 0.0) > 0.0]
    if min_panel_expressed and len(expressed_panel) < min_panel_expressed:
        return None
    # Order expressed panel genes by descending expression; canonical panel_index
    # breaks ties deterministically (same tie convention as before, but only among
    # EXPRESSED genes now — the unexpressed genes are dropped, not tail-appended).
    expressed_panel.sort(key=lambda g: (-d[g], panel_index[g]))
    return " ".join(expressed_panel) + " " + END_CELL_TOKEN


def make_paired_cell_sentences_fixed_panel(
    ctrl_genes, ctrl_exprs, treat_genes, treat_exprs,
    gene_id_to_symbol, panel_symbols, panel_index,
    min_panel_expressed=50,
):
    """
    Build (control, response) sentences over a FIXED panel (leak-free).

    Both sentences contain ALL panel genes, every time: genes expressed in the
    cell are ranked by descending normalized expression; genes not expressed (0)
    are appended in canonical panel order (a deterministic, leak-free worst-rank
    convention). Every sentence is therefore exactly len(panel) genes long, and
    the gene *identities* carry zero information about the treatment — the panel
    is identical for every example and reconstructable at inference from the
    control alone.

    panel_symbols : list[str]      canonical-ordered fixed panel (from l1000_panel.json)
    panel_index   : dict[str,int]  symbol -> canonical position (tie-break key)
    """
    ctrl_s = build_panel_sentence(
        ctrl_genes, ctrl_exprs, gene_id_to_symbol, panel_symbols, panel_index,
        min_expressed=200, min_panel_expressed=0,
    )
    treat_s = build_panel_sentence(
        treat_genes, treat_exprs, gene_id_to_symbol, panel_symbols, panel_index,
        min_expressed=200, min_panel_expressed=min_panel_expressed,
    )
    if ctrl_s is None or treat_s is None:
        return None, None
    return ctrl_s, treat_s


def fit_expression_linear_model_panel(control_cells_raw, gene_id_to_symbol,
                                      panel_symbols, panel_index, n_cells=500):
    """
    Fit the C2S expression-recovery model (expression = slope*log10(rank) + intercept)
    on the PANEL's rank<->expression relationship, so it is consistent with the
    fixed-panel ranks 1..len(panel) produced by make_paired_cell_sentences_fixed_panel.

    We draw (log10(rank), norm_expr) pairs from the expressed panel genes of control
    cells, ranking them within the panel exactly as the sentence builder does. The
    unexpressed tail (norm_expr == 0) is excluded from the fit; recovery clamps those
    high ranks to 0 at eval time anyway.
    """
    from scipy import stats as scipy_stats

    panel_set = set(panel_symbols)
    log_ranks, norm_exprs = [], []
    used = 0
    for genes, exprs in control_cells_raw:
        if used >= n_cells:
            break
        gid = np.array(genes)
        ex = np.array(exprs, dtype=np.float64)
        m = ex > 0
        gid, ex = gid[m], ex[m]
        if len(ex) < 50:
            continue
        tot = ex.sum()
        if tot == 0:
            continue
        norm = np.log10(1 + (ex / tot) * 1e4)
        d = {}
        for g, v in zip(gid, norm):
            sym = gene_id_to_symbol.get(g, None)
            if sym in panel_set:
                d[sym] = float(v)
        if not d:
            continue
        # rank expressed panel genes 1..k by expression desc (their position in the
        # full panel ordering, since expressed genes always precede the 0-tail).
        expressed = sorted(d.items(), key=lambda kv: (-kv[1], panel_index[kv[0]]))
        for rank, (g, v) in enumerate(expressed, 1):
            log_ranks.append(np.log10(rank))
            norm_exprs.append(v)
        used += 1

    if used == 0 or len(log_ranks) < 100:
        return None  # caller falls back to the global fit

    log_ranks = np.array(log_ranks)
    norm_exprs = np.array(norm_exprs)
    slope, intercept, r_value, _, _ = scipy_stats.linregress(log_ranks, norm_exprs)
    spearman_r, _ = scipy_stats.spearmanr(log_ranks, norm_exprs)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "pearson_r": float(r_value),
        "spearman_r": float(spearman_r),
        "fit": "panel_restricted",
        "n_control_cells": int(used),
        "panel_size": len(panel_symbols),
    }


def fit_expression_linear_model(gene_indices_list, expressions_list, gene_id_to_symbol,
                                 n_samples=10000):
    """
    Fit the C2S linear model: expression = slope * log(rank) + intercept.
    
    Samples (log_rank, normalized_expression) pairs from multiple cells
    and fits a linear regression in log-log space.
    
    Args:
        gene_indices_list: list of gene index arrays from sampled cells
        expressions_list: list of expression arrays from sampled cells
        gene_id_to_symbol: gene mapping dict
        n_samples: total gene samples to collect across cells
    
    Returns:
        dict with 'slope', 'intercept', 'r_squared', 'pearson_r', 'spearman_r'
    """
    from scipy import stats as scipy_stats
    
    log_ranks = []
    log_exprs = []
    samples_per_cell = max(1, n_samples // len(gene_indices_list))
    
    for gene_indices, expressions in zip(gene_indices_list, expressions_list):
        gene_ids = np.array(gene_indices)
        expr = np.array(expressions, dtype=np.float64)
        
        # Filter positive
        mask = expr > 0
        gene_ids = gene_ids[mask]
        expr = expr[mask]
        
        if len(expr) < 50:
            continue
        
        # Normalize
        total = expr.sum()
        expr_norm = (expr / total) * 1e4
        expr_norm = np.log10(1 + expr_norm)
        
        # Rank order
        sorted_idx = np.argsort(-expr_norm)
        ranks = np.arange(1, len(sorted_idx) + 1)
        
        # Sample genes from this cell
        if len(ranks) > samples_per_cell:
            sample_idx = np.random.choice(len(ranks), samples_per_cell, replace=False)
        else:
            sample_idx = np.arange(len(ranks))
        
        log_ranks.extend(np.log10(ranks[sample_idx]).tolist())
        log_exprs.extend(expr_norm[sorted_idx[sample_idx]].tolist())
    
    log_ranks = np.array(log_ranks)
    log_exprs = np.array(log_exprs)
    
    # Fit linear regression: expr = slope * log_rank + intercept
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_ranks, log_exprs)
    
    # Also compute Spearman
    spearman_r, _ = scipy_stats.spearmanr(log_ranks, log_exprs)
    
    model = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "pearson_r": float(r_value),
        "spearman_r": float(spearman_r),
    }
    
    return model


def cell_sentence_to_expression(cell_sentence, linear_model, all_gene_symbols=None):
    """
    Convert a cell sentence back to an expression vector using the fitted linear model.
    
    For each gene in the sentence, expression = slope * log10(rank) + intercept.
    Genes not in the sentence get expression 0.
    
    Args:
        cell_sentence: space-separated gene names
        linear_model: dict with 'slope' and 'intercept'
        all_gene_symbols: optional list of all gene symbols for full vector output
    
    Returns:
        dict of gene_name -> reconstructed expression value
    """
    genes = cell_sentence.strip().split()
    slope = linear_model["slope"]
    intercept = linear_model["intercept"]
    
    # Handle duplicates by averaging rank
    gene_positions = defaultdict(list)
    for pos, gene in enumerate(genes, 1):
        gene_positions[gene].append(pos)
    
    expr_dict = {}
    for gene, positions in gene_positions.items():
        avg_rank = np.mean(positions)
        expr_val = slope * np.log10(avg_rank) + intercept
        expr_dict[gene] = max(0.0, expr_val)  # clamp to non-negative
    
    return expr_dict


# =============================================================================
# Step 3: Parse dose from sample metadata
# =============================================================================

def parse_dose(drugname_drugconc_str):
    """
    Parse dose from the drugname_drugconc field.
    Format: "[('drug_name', concentration, 'unit')]"
    Returns: string like "0.05 uM" or "unknown"
    """
    try:
        # The field looks like: [('8-Hydroxyquinoline',0.05,'uM')]
        # Use eval safely-ish (it's controlled metadata)
        parsed = eval(drugname_drugconc_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            entry = parsed[0]
            if len(entry) >= 3:
                return f"{entry[1]} {entry[2]}"
    except:
        pass
    return "unknown"


# =============================================================================
# Step 4: Format C2S-Scale prompt
# =============================================================================

def format_prompt(cell_line_name, drug_name, dose_str, moa, control_cell_sentence=None):
    """
    Format the input prompt following C2S-Scale perturbation prediction style.
    Always includes Mechanism field — 'unclear' when unknown, so the model
    learns to use MOA when available and treat 'unclear' as a null signal.
    
    Includes plate-matched DMSO control cell sentence when available,
    following C2S-Scale Figure 8C format.
    """
    if not moa or moa == "unknown" or moa == "nan" or moa == "None":
        moa = "unclear"
    
    prompt = (
        f"Predict the response of {cell_line_name} to {drug_name} "
        f"at {dose_str}. Mechanism: {moa}."
    )
    
    if control_cell_sentence:
        prompt += f"\nControl cell: {control_cell_sentence}"
    
    prompt += "\n\nResponse cell:"
    return prompt


# =============================================================================
# Shard discovery + sampling  (DIVERSITY + CONTROL COVERAGE)
# =============================================================================
# Tahoe-100M is written in processing order, so the first N rows are "deep but
# narrow". We instead randomly sample whole parquet SHARDS for TREATED cells
# (diversity), and SEPARATELY scan a broad set of shards (default: all) for DMSO
# CONTROLS. Decoupling the two is essential: controls are sparse and a plate's
# DMSO does NOT reliably co-locate in the same shards as its treated cells, so
# tying control collection to the treated-shard sample makes no_control_rate
# climb as you add shards. The dedicated DMSO scan reads whole shards and
# early-stops once every needed (cell_line, plate) key is covered; a same-
# cell-line fallback (tagged) recovers any residual misses.

TAHOE_REPO = "tahoebio/Tahoe-100M"


def discover_expression_shards():
    """List the expression parquet shards (excludes the small metadata/* tables,
    which are loaded separately in load_metadata)."""
    api = HfApi()
    files = api.list_repo_files(TAHOE_REPO, repo_type="dataset")
    shards = sorted(
        f for f in files
        if f.endswith(".parquet") and not f.startswith("metadata/")
    )
    if not shards:
        raise RuntimeError(
            f"No expression parquet shards found in {TAHOE_REPO}. "
            f"Sample of repo files: {files[:10]}"
        )
    return shards


def select_shards(all_shards, num_shards, seed):
    """Reproducible random subset of shards spread across the dataset. Random
    selection (not the first K) is exactly what breaks the contiguous-prefix
    diversity ceiling."""
    if num_shards is None or num_shards >= len(all_shards):
        return list(all_shards)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(all_shards), size=num_shards, replace=False).tolist())
    return [all_shards[i] for i in idx]


# =============================================================================
# Step 5: Main data construction pipeline
# =============================================================================

def construct_training_data(
    gene_id_to_symbol,
    sample_to_conc,
    drug_info,
    cvcl_to_name,
    panel_file,
    output_dir,
    max_cells_per_condition=10,
    num_held_out_drugs=50,
    max_genes=None,
    test_mode=False,
    test_limit=500,
    max_train_examples=300_000,
    max_eval_per_tier=5000,
    num_shards=60,
    rows_per_shard=150_000,
    shard_seed=7,
    dmso_shards=None,
    dmso_coverage_frac=0.97,
    cellline_fallback=True,
):
    """
    Two-pass construction of training data from Tahoe-100M.
    Uses HuggingFace datasets streaming — no shard pre-download needed.

    Pass 1: Stream all data, collect DMSO control cell sentences
            keyed by (cell_line_id, plate).
    Pass 2: Stream again, for each treated cell pair with a
            plate-matched DMSO control.

    Cell sentences are built over a FIXED L1000 panel (panel_file) so the gene
    identities carry no treatment information and are reconstructable at inference.

    train.jsonl + train_text.jsonl are streamed to output_dir incrementally during
    Pass 2 (so a node failure / timeout leaves a valid partial dataset).

    Returns:
        n_train: int, number of training examples written to disk
        eval_examples: dict of tier -> list of examples (saved by caller)
        linear_model: dict, expression-recovery fit (or None)
        first_train_example: dict, first example (for the sanity print) or None
        diversity_stats: dict of coverage counts (drugs/combos/doses/conditions)
    """
    # --- Load the fixed gene panel (single source of truth, leak-free) ---
    panel_symbols = json.load(open(panel_file))
    panel_index = {g: i for i, g in enumerate(panel_symbols)}
    logger.info(f"Loaded fixed L1000 panel: {len(panel_symbols)} genes from {panel_file}")
    logger.info(f"  First 8: {panel_symbols[:8]}")

    def iterate_shards(shard_list, per_shard_cap=None, limit=None):
        """Stream rows from a list of parquet shards, up to per_shard_cap rows
        from each (None = whole shard), visiting every shard in order."""
        count = 0
        for shard in shard_list:
            url = f"hf://datasets/{TAHOE_REPO}/{shard}"
            ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
            sc = 0
            for row in ds:
                yield row
                count += 1
                sc += 1
                if per_shard_cap and sc >= per_shard_cap:
                    break
                if limit and count >= limit:
                    return

    def iterate_test(limit=None):
        ds = load_dataset(TAHOE_REPO, split="train", streaming=True)
        count = 0
        for row in ds:
            yield row
            count += 1
            if limit and count >= limit:
                return

    # --- Select shards ------------------------------------------------------
    # treated_shards: where TREATED cells are drawn (diversity dial = num_shards,
    #                 capped at rows_per_shard each).
    # dmso_shard_list: where DMSO controls are COLLECTED, in scan ORDER. A plate's
    #                 DMSO is written together with its treated cells, so we scan the
    #                 TREATED shards FIRST (co-located controls are found immediately
    #                 and the early-stop fires fast), then a BOUNDED buffer of other
    #                 shards for any (cell_line, plate) whose control lives elsewhere.
    #                 Scanning raw corpus order instead ploughs through the whole
    #                 dataset just to reach high-index treated shards (the timeout bug).
    if test_mode:
        treated_shards = None
        dmso_shard_list = None
        all_shards = []
    else:
        all_shards = discover_expression_shards()
        treated_shards = select_shards(all_shards, num_shards, shard_seed)
        extra_cap = dmso_shards if dmso_shards is not None else 256
        extra_cap = min(extra_cap, len(all_shards))
        treated_set = set(treated_shards)
        extra = [s for s in select_shards(all_shards, extra_cap, shard_seed + 1)
                 if s not in treated_set]
        dmso_shard_list = list(treated_shards) + extra   # TREATED FIRST
        logger.info(
            f"Tahoe-100M exposes {len(all_shards)} expression shards. "
            f"Treated: {len(treated_shards)} shards (seed={shard_seed}, "
            f"<= {rows_per_shard:,} rows each). "
            f"DMSO scan: up to {len(dmso_shard_list)} shards (treated-first, whole-shard, "
            f"early-stop at {int(dmso_coverage_frac * 100)}% key coverage)."
        )
        logger.info(f"  First treated shards: {treated_shards[:4]}")

    def treated_iter(limit=None):
        if test_mode:
            return iterate_test(limit=limit)
        return iterate_shards(treated_shards, per_shard_cap=rows_per_shard, limit=limit)

    def dmso_iter(limit=None):
        if test_mode:
            return iterate_test(limit=limit)
        return iterate_shards(dmso_shard_list, per_shard_cap=None, limit=limit)

    # Absolute safety ceilings (the shard budget normally bounds the run well
    # below these; they only guard against a runaway config).
    MAX_ROWS_OBSERVE = (num_shards * rows_per_shard * 2) if not test_mode else test_limit
    MAX_ROWS_PASS2 = (num_shards * rows_per_shard * 2) if not test_mode else test_limit
    # Hard cap on train size to protect disk in case of a huge shard config.
    ABS_MAX_TRAIN = max(max_train_examples * 4, 2_000_000)

    # =============================================
    # OBSERVE PASS: catalogue treated cells + the control KEYS we will need
    # =============================================
    logger.info("\n--- Observe pass: cataloguing treated cells (treated shards) ---")
    observed_drugs = set()
    observed_combos = set()                 # (drug, cell_line_id)
    observed_drug_doses = defaultdict(set)  # drug -> {dose_float, ...}
    needed_control_keys = set()             # (cell_line_id, plate) we must control
    needed_cell_lines = set()

    total_observe = 0
    for row in tqdm(treated_iter(limit=MAX_ROWS_OBSERVE),
                    desc="Observe (treated)", disable=False):
        total_observe += 1
        drug = row["drug"]
        if drug == "DMSO_TF" or drug == "DMSO":
            continue
        cl = row["cell_line_id"]
        observed_drugs.add(drug)
        observed_combos.add((drug, cl))
        needed_control_keys.add((cl, row["plate"]))
        needed_cell_lines.add(cl)
        try:
            dose_v = float(parse_dose(sample_to_conc.get(row["sample"], "unknown")).split()[0])
            observed_drug_doses[drug].add(dose_v)
        except Exception:
            pass
    logger.info(f"  Streamed {total_observe:,} treated-shard cells")
    logger.info(f"  Observed drugs: {len(observed_drugs):,} | combos: {len(observed_combos):,}")
    logger.info(f"  Control keys needed (cell_line,plate): {len(needed_control_keys):,} "
                f"across {len(needed_cell_lines):,} cell lines")

    # =============================================
    # DMSO SCAN: collect controls across the (broad) DMSO shard set
    # =============================================
    # Whole-shard reads over dmso_shard_list, early-stopping the moment every
    # needed (cell_line, plate) key has a control. Also pool controls per
    # cell_line for the optional cross-plate fallback.
    logger.info("\n--- DMSO scan: collecting controls (whole-shard, early-stop) ---")
    dmso_controls_raw = defaultdict(list)        # (cell_line_id, plate) -> [(genes,exprs)]
    dmso_by_cellline = defaultdict(list)         # cell_line_id -> [(genes,exprs)] (fallback)
    MAX_DMSO_PER_KEY = 8
    MAX_DMSO_PER_CELLLINE = 16

    total_dmso_seen = 0
    keys_covered = 0
    cover_target = max(1, int(np.ceil(dmso_coverage_frac * len(needed_control_keys)))) \
        if needed_control_keys else 0
    for row in tqdm(dmso_iter(), desc="DMSO scan", disable=False):
        drug = row["drug"]
        if drug != "DMSO_TF" and drug != "DMSO":
            continue
        total_dmso_seen += 1
        cl = row["cell_line_id"]
        key = (cl, row["plate"])
        if key in needed_control_keys and len(dmso_controls_raw[key]) < MAX_DMSO_PER_KEY:
            if len(dmso_controls_raw[key]) == 0:
                keys_covered += 1
            dmso_controls_raw[key].append((row["genes"], row["expressions"]))
        if cl in needed_cell_lines and len(dmso_by_cellline[cl]) < MAX_DMSO_PER_CELLLINE:
            dmso_by_cellline[cl].append((row["genes"], row["expressions"]))
        if total_dmso_seen % 200000 == 0:
            logger.info(f"  ... {total_dmso_seen:,} DMSO seen, "
                        f"{keys_covered:,}/{len(needed_control_keys):,} keys covered "
                        f"(target {cover_target:,}), {len(dmso_by_cellline):,} cell lines pooled")
        # Early-stop once coverage hits the target fraction. Requiring 100% would
        # chase a rare-combo coupon-collector tail; the cell-line fallback covers
        # the residual cleanly (and tags it), so we stop at dmso_coverage_frac.
        if cover_target and keys_covered >= cover_target:
            logger.info(f"  Reached {keys_covered}/{len(needed_control_keys)} control keys "
                        f"(>= {int(dmso_coverage_frac * 100)}%) — stopping DMSO scan early.")
            break

    n_keys_with_ctrl = sum(1 for k in needed_control_keys if dmso_controls_raw.get(k))
    logger.info(f"  DMSO cells scanned: {total_dmso_seen:,}")
    logger.info(f"  Plate-matched control keys: {n_keys_with_ctrl:,}/{len(needed_control_keys):,} "
                f"({100.0 * n_keys_with_ctrl / max(1, len(needed_control_keys)):.1f}%)")
    logger.info(f"  Cell lines with fallback control: {len(dmso_by_cellline):,}/"
                f"{len(needed_cell_lines):,}")
    total_controls = sum(len(v) for v in dmso_controls_raw.values())
    if total_controls == 0 and not any(dmso_by_cellline.values()):
        logger.error("  No DMSO controls found! Cannot build training pairs.")
        return 0, {}, None, None, {}
    
    # =============================================
    # Fit expression recovery linear model (PANEL-RESTRICTED)
    # =============================================
    # Fit on the panel's rank<->expression relationship using control (DMSO) cells,
    # so recovery is consistent with the fixed-panel ranks 1..len(panel). Falls back
    # to the global fit if too few control cells are available.
    control_cells = [cell for cells in dmso_controls_raw.values() for cell in cells]
    if len(control_cells) < 200:
        control_cells += [cell for cells in dmso_by_cellline.values() for cell in cells]
    np.random.seed(123)
    np.random.shuffle(control_cells)
    logger.info(f"\n--- Fitting PANEL-RESTRICTED expression linear model "
                f"from {min(len(control_cells), 500)} control cells ---")
    linear_model = fit_expression_linear_model_panel(
        control_cells, gene_id_to_symbol, panel_symbols, panel_index, n_cells=500
    )
    if linear_model is None:
        logger.warning("  Panel-restricted fit unavailable; falling back to global fit.")
        sample = control_cells[:500]
        linear_model = fit_expression_linear_model(
            [g for g, _ in sample], [e for _, e in sample], gene_id_to_symbol, n_samples=10000
        )
        linear_model["fit"] = "global_fallback"
    logger.info(f"  Fit type: {linear_model.get('fit', 'global')}")
    logger.info(f"  Slope: {linear_model['slope']:.4f}")
    logger.info(f"  Intercept: {linear_model['intercept']:.4f}")
    logger.info(f"  R²: {linear_model['r_squared']:.4f}")
    logger.info(f"  Pearson R: {linear_model['pearson_r']:.4f}")
    logger.info(f"  Spearman R: {linear_model['spearman_r']:.4f}")

    # =============================================
    # Cell sentence conversion: FIXED L1000 PANEL (leak-free)
    # =============================================
    # Pairs are now built by the module-level make_paired_cell_sentences_fixed_panel():
    # both control and response are ranked over the SAME fixed panel every time, so the
    # gene identities encode nothing about the treatment and the prompt is reconstructable
    # at inference from the control alone.
    #
    # LEAKY — replaced by fixed panel. The previous nested make_paired_cell_sentences()
    # selected the top-K genes by the TREATED cell's expression, which leaks treatment
    # information into the prompt (the gene set reveals which genes the drug turned on) and
    # cannot be reconstructed at inference. Kept below, commented, for auditability:
    #
    # def make_paired_cell_sentences(ctrl_genes, ctrl_exprs, treat_genes, treat_exprs,
    #                                gene_id_to_symbol, max_genes):
    #     # Normalize control
    #     c_ids = np.array(ctrl_genes)
    #     c_expr = np.array(ctrl_exprs, dtype=np.float64)
    #     c_mask = c_expr > 0
    #     c_ids, c_expr = c_ids[c_mask], c_expr[c_mask]
    #     if len(c_expr) < 200:
    #         return None, None
    #     c_total = c_expr.sum()
    #     if c_total == 0:
    #         return None, None
    #     c_norm = np.log10(1 + (c_expr / c_total) * 1e4)
    #     ctrl_dict = {}
    #     for gid, val in zip(c_ids, c_norm):
    #         symbol = gene_id_to_symbol.get(gid, None)
    #         if symbol is not None:
    #             ctrl_dict[symbol] = val
    #     # Normalize treated
    #     t_ids = np.array(treat_genes)
    #     t_expr = np.array(treat_exprs, dtype=np.float64)
    #     t_mask = t_expr > 0
    #     t_ids, t_expr = t_ids[t_mask], t_expr[t_mask]
    #     if len(t_expr) < 200:
    #         return None, None
    #     t_total = t_expr.sum()
    #     if t_total == 0:
    #         return None, None
    #     t_norm = np.log10(1 + (t_expr / t_total) * 1e4)
    #     treat_dict = {}
    #     for gid, val in zip(t_ids, t_norm):
    #         symbol = gene_id_to_symbol.get(gid, None)
    #         if symbol is not None:
    #             treat_dict[symbol] = val
    #     all_genes = set(ctrl_dict.keys()) | set(treat_dict.keys())
    #     if len(all_genes) < 100:
    #         return None, None
    #     # Select top-K genes by TREATED cell expression  <-- THE LEAK
    #     treat_ranked = sorted(all_genes, key=lambda g: treat_dict.get(g, 0.0), reverse=True)
    #     selected_genes = treat_ranked[:max_genes] if max_genes else treat_ranked
    #     response_sentence = " ".join(selected_genes)
    #     ctrl_ranked = sorted(selected_genes, key=lambda g: ctrl_dict.get(g, 0.0), reverse=True)
    #     control_sentence = " ".join(ctrl_ranked)
    #     return control_sentence, response_sentence

    # =============================================
    # PASS 2: Pair treated cells with controls
    # =============================================
    logger.info("\n--- Pass 2: Building training pairs ---")
    
    # --- Decide held-out splits (drawn from OBSERVED entities) ---
    # Drawing from what actually appears in the sampled shards guarantees each
    # held-out drug/combo/dose has treated cells in Pass 2. Capped at a fraction
    # of observed so we never starve train of drugs.
    np.random.seed(42)
    observed_drug_list = sorted(observed_drugs)

    # Tier 2: entirely unseen drugs (<= 20% of observed drugs)
    n_hold = min(num_held_out_drugs, max(1, len(observed_drug_list) // 5))
    if observed_drug_list and n_hold > 0:
        held_out_drugs = set(np.random.choice(observed_drug_list, size=n_hold, replace=False))
    else:
        held_out_drugs = set()
    logger.info(f"  Held-out drugs (Tier 2): {len(held_out_drugs)} "
                f"of {len(observed_drug_list)} observed")

    # Tier-2 is stratified PER held-out drug: cap each drug's eval cells at
    # (max_eval_per_tier / n_held_out_drugs) so all held-out drugs are represented
    # rather than the file filling from whichever drugs the build pass hits first.
    # This makes the effective sample size of the unseen-drug bootstrap = n drugs.
    tier2_cap_per_drug = max(1, max_eval_per_tier // max(1, len(held_out_drugs)))
    tier2_per_drug_count = defaultdict(int)
    logger.info(f"  Tier-2 per-drug cap: {tier2_cap_per_drug} "
                f"(=> up to {tier2_cap_per_drug * len(held_out_drugs):,} balanced across "
                f"{len(held_out_drugs)} drugs)")

    # Tier 3: held-out (drug x cell_line) combos, from observed combos whose drug
    # is NOT itself held out (otherwise it's already a Tier-2 case).
    candidate_combos = sorted(
        c for c in observed_combos if c[0] not in held_out_drugs
    )
    np.random.shuffle(candidate_combos)
    num_held_out_combos = min(100, len(candidate_combos) // 10)
    held_out_combos = set(candidate_combos[:num_held_out_combos])
    logger.info(f"  Held-out drug x cell_line combos (Tier 3): {len(held_out_combos)} "
                f"of {len(candidate_combos)} observed candidates")

    # Tier 4: dose interpolation — hold out the MIDDLE observed dose for drugs that
    # (a) have >= 3 observed doses and (b) are not held out / combo-held-out.
    held_out_dose_per_drug = {}
    for drug_name, doses in observed_drug_doses.items():
        if drug_name in held_out_drugs:
            continue
        if len(doses) >= 3:
            sorted_doses = sorted(doses)
            held_out_dose_per_drug[drug_name] = sorted_doses[len(sorted_doses) // 2]
    logger.info(f"  Drugs with held-out middle dose (Tier 4): {len(held_out_dose_per_drug)}")
    
    # --- Process treated cells ---
    condition_counts = defaultdict(int)
    tier1_eval = []
    tier2_eval = []
    tier3_eval = []
    tier4_eval = []

    # --- Incremental train writing (timeout/node-failure safe) --------------
    os.makedirs(output_dir, exist_ok=True)
    train_f = open(os.path.join(output_dir, "train.jsonl"), "w")
    train_text_f = open(os.path.join(output_dir, "train_text.jsonl"), "w")
    n_train = 0
    first_train_example = None
    train_drugs, train_combos, train_doses = set(), set(), set()

    def write_train(ex):
        nonlocal n_train, first_train_example
        train_f.write(json.dumps(ex) + "\n")
        train_text_f.write(json.dumps({"text": ex["prompt"] + " " + ex["response"]}) + "\n")
        n_train += 1
        if first_train_example is None:
            first_train_example = ex
        md = ex["metadata"]
        train_drugs.add(md["drug"])
        train_combos.add((md["drug"], md["cell_line_id"]))
        if md["dose_float"] is not None:
            train_doses.add((md["drug"], round(md["dose_float"], 6)))
        if n_train % 5000 == 0:
            train_f.flush()
            train_text_f.flush()

    total_pass2 = 0
    total_treated = 0
    total_no_control = 0
    total_fallback = 0
    total_qc_failed = 0

    for row in tqdm(treated_iter(limit=MAX_ROWS_PASS2),
                    desc="Pass 2 (pairs)", disable=False):
        total_pass2 += 1
        
        drug = row["drug"]
        if drug == "DMSO_TF" or drug == "DMSO":
            continue  # skip controls in pass 2
        
        cell_line_id = row["cell_line_id"]
        plate = row["plate"]
        sample = row["sample"]
        
        # Check condition cap
        condition_key = (drug, cell_line_id, plate)
        if condition_counts[condition_key] >= max_cells_per_condition:
            continue
        
        # Look up control: prefer the plate-matched DMSO; optionally fall back to
        # any same-cell-line DMSO (tagged control_plate_matched=False); else drop.
        control_key = (cell_line_id, plate)
        plate_matched = True
        pool = dmso_controls_raw.get(control_key)
        if not pool:
            if cellline_fallback and dmso_by_cellline.get(cell_line_id):
                pool = dmso_by_cellline[cell_line_id]
                plate_matched = False
                total_fallback += 1
            else:
                total_no_control += 1
                continue
        
        # Pick random control raw data
        ctrl_raw = pool[np.random.randint(len(pool))]
        ctrl_genes_raw, ctrl_exprs_raw = ctrl_raw
        
        # Convert both cells to cell sentences over the FIXED L1000 panel (leak-free)
        control_sentence, cell_sentence = make_paired_cell_sentences_fixed_panel(
            ctrl_genes_raw, ctrl_exprs_raw,
            row["genes"], row["expressions"],
            gene_id_to_symbol, panel_symbols, panel_index,
        )
        if control_sentence is None or cell_sentence is None:
            total_qc_failed += 1
            continue
        
        total_treated += 1
        
        # Build prompt
        cell_line_name = cvcl_to_name.get(cell_line_id, cell_line_id)
        dose_str = parse_dose(sample_to_conc.get(sample, "unknown"))
        moa = row.get("moa-fine", drug_info.get(drug, {}).get("moa", "unknown"))
        
        dose_float = None
        try:
            dose_float = float(dose_str.split()[0])
        except:
            pass
        
        prompt = format_prompt(cell_line_name, drug, dose_str, moa, control_sentence)
        
        example = {
            "prompt": prompt,
            "response": cell_sentence,
            "metadata": {
                "drug": drug,
                "cell_line_id": cell_line_id,
                "cell_line_name": cell_line_name,
                "plate": plate,
                "sample": sample,
                "dose": dose_str,
                "dose_float": dose_float,
                "moa": moa,
                "control_plate_matched": plate_matched,
            }
        }
        
        # Assign to split. Eval tiers are capped at max_eval_per_tier so the files
        # stay bounded. Tiers 2-4 are held-out by design, so any overflow beyond the
        # cap is simply dropped (we keep far more than the ~500 eval needs). Tier-1 is
        # in-distribution, so its overflow spills into train rather than being wasted.
        if drug in held_out_drugs:
            # per-drug cap so every held-out drug is represented (not a flat total)
            if tier2_per_drug_count[drug] < tier2_cap_per_drug:
                tier2_eval.append(example)
                tier2_per_drug_count[drug] += 1
        elif (drug, cell_line_id) in held_out_combos:
            if len(tier3_eval) < max_eval_per_tier:
                tier3_eval.append(example)
        elif (dose_float is not None
              and drug in held_out_dose_per_drug
              and abs(dose_float - held_out_dose_per_drug[drug]) < 1e-6):
            if len(tier4_eval) < max_eval_per_tier:
                tier4_eval.append(example)
        else:
            if np.random.random() < 0.1 and len(tier1_eval) < max_eval_per_tier:
                tier1_eval.append(example)
            else:
                write_train(example)

        condition_counts[condition_key] += 1
        
        if total_pass2 % 500000 == 0:
            logger.info(f"  ... streamed {total_pass2:,} cells, "
                       f"{total_treated:,} treated pairs built, "
                       f"{n_train:,} train examples, "
                       f"{len(train_drugs):,} drugs")
        
        # Safety ceiling only (the shard budget normally bounds the run). We do
        # NOT early-stop on a train target in shard mode, because stopping early
        # would starve the later shards of representation and re-narrow diversity.
        if n_train >= ABS_MAX_TRAIN:
            logger.warning(f"  Hit ABS_MAX_TRAIN={ABS_MAX_TRAIN:,}; stopping Pass 2.")
            break

    train_f.flush(); train_f.close()
    train_text_f.flush(); train_text_f.close()

    treated_considered = total_treated + total_no_control  # cells that reached the lookup
    logger.info(f"\n--- Processing Stats ---")
    logger.info(f"Observe-pass cells streamed: {total_observe:,}")
    logger.info(f"Pass 2 cells streamed: {total_pass2:,}")
    logger.info(f"Treated pairs built: {total_treated:,}  "
                f"(plate-matched: {total_treated - total_fallback:,}, "
                f"cell-line fallback: {total_fallback:,})")
    logger.info(f"No control (dropped): {total_no_control:,}  "
                f"({100.0 * total_no_control / max(1, treated_considered):.2f}% of treated cells "
                f"that reached control lookup)")
    logger.info(f"QC failed: {total_qc_failed:,}")
    logger.info(f"Train examples: {n_train:,}")
    logger.info(f"Tier 1 eval (seen conditions): {len(tier1_eval):,}")
    logger.info(f"Tier 2 eval (unseen drugs): {len(tier2_eval):,}")
    logger.info(f"Tier 3 eval (unseen combos): {len(tier3_eval):,}")
    logger.info(f"Tier 4 eval (dose interpolation): {len(tier4_eval):,}")
    logger.info(f"Unique (drug,cell_line,plate) conditions: {len(condition_counts):,}")
    logger.info(f"--- DIVERSITY (train) ---")
    logger.info(f"  Unique drugs in train: {len(train_drugs):,}")
    logger.info(f"  Unique (drug,cell_line) combos in train: {len(train_combos):,}")
    logger.info(f"  Unique (drug,dose) points in train: {len(train_doses):,}")
    if n_train:
        logger.info(f"  Mean examples per drug: {n_train / max(1, len(train_drugs)):.1f}")
    
    eval_examples = {
        "tier1_seen_conditions": tier1_eval,
        "tier2_unseen_drugs": tier2_eval,
        "tier3_unseen_combos": tier3_eval,
        "tier4_dose_interpolation": tier4_eval,
    }

    diversity_stats = {
        "n_train": n_train,
        "unique_drugs_train": len(train_drugs),
        "unique_combos_train": len(train_combos),
        "unique_dose_points_train": len(train_doses),
        "unique_conditions": len(condition_counts),
        "observed_drugs": len(observed_drugs),
        "no_control_rate": total_no_control / max(1, treated_considered),
        "control_fallback_rate": total_fallback / max(1, total_treated),
        "plate_matched_keys": int(n_keys_with_ctrl),
        "needed_control_keys": len(needed_control_keys),
        "dmso_shards_scanned": (0 if test_mode else len(dmso_shard_list)),
        "treated_shards": (0 if test_mode else len(treated_shards)),
        "rows_per_shard": rows_per_shard,
    }
    
    return (n_train, eval_examples, linear_model, first_train_example, diversity_stats,
            sorted(held_out_drugs))


# =============================================================================
# Step 6: Save to disk
# =============================================================================

def save_examples(examples, filepath):
    """Save examples as JSONL for easy loading during training."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Saved {len(examples)} examples to {filepath}")


def save_as_text_pairs(examples, filepath):
    """
    Save as simple text format compatible with HuggingFace Trainer.
    Each line: prompt + response concatenated with separator.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for ex in examples:
            # Format: prompt [SEP] response
            full_text = ex["prompt"] + " " + ex["response"]
            f.write(json.dumps({"text": full_text}) + "\n")
    logger.info(f"Saved {len(examples)} text pairs to {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Construct C2S training data from Tahoe-100M")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                        help="test = tiny local run, full = HPC run")
    parser.add_argument("--output_dir", type=str, default="./tahoe_c2s_data",
                        help="Output directory for processed data")
    parser.add_argument("--cells_per_condition", type=int, default=10,
                        help="Max cells to sample per (drug, cell_line, plate) condition")
    parser.add_argument("--held_out_drugs", type=int, default=50,
                        help="Number of drugs to hold out for Tier 2 eval")
    parser.add_argument("--panel_file", type=str, required=True,
                        help="Path to l1000_panel.json (the fixed gene panel built by "
                             "build_l1000_panel.py). Both control and response are ranked "
                             "over this fixed panel — leak-free and reconstructable at inference.")
    parser.add_argument("--max_train_examples", type=int, default=300_000,
                        help="Soft target only. In shard mode the run size is set by "
                             "num_shards x rows_per_shard x cells_per_condition; this is "
                             "used only to derive an absolute disk-safety ceiling (4x).")
    parser.add_argument("--max_eval_per_tier", type=int, default=5000,
                        help="Cap on examples kept per eval tier file (eval-time generation "
                             "is separately capped). Tier-1 overflow spills into train.")
    parser.add_argument("--max_genes", type=int, default=1500,
                        help="DEPRECATED under the fixed panel — pair construction now uses "
                             "the full panel from --panel_file and ignores this. Kept for "
                             "backward compatibility with raw_to_cell_sentence only.")
    parser.add_argument("--test_limit", type=int, default=500,
                        help="Number of rows to process in test mode")
    parser.add_argument("--num_shards", type=int, default=60,
                        help="DIVERSITY KNOB: number of parquet shards to randomly sample "
                             "from across the corpus (full mode). More shards = more unique "
                             "drugs/combos/doses. Clamped to the number available.")
    parser.add_argument("--rows_per_shard", type=int, default=150_000,
                        help="Max rows consumed per sampled shard, in each pass. Total "
                             "rows/pass ~= num_shards x rows_per_shard.")
    parser.add_argument("--shard_seed", type=int, default=7,
                        help="Seed for the shard sample (change to draw a different, "
                             "reproducible subset).")
    parser.add_argument("--dmso_shards", type=int, default=None,
                        help="CONTROL-COVERAGE KNOB: size of the EXTRA shard buffer scanned "
                             "for DMSO controls beyond the treated shards (treated shards are "
                             "always scanned first, whole, because a plate's DMSO co-locates "
                             "with its treated cells). Default 256. The scan early-stops at "
                             "--dmso_coverage_frac key coverage, so this is an upper bound.")
    parser.add_argument("--dmso_coverage_frac", type=float, default=0.97,
                        help="Stop the DMSO scan once this fraction of needed "
                             "(cell_line,plate) control keys is covered (default 0.97). The "
                             "cell-line fallback covers the residual; requiring 1.0 would chase "
                             "a rare-combo coupon-collector tail (the cause of the timeout).")
    parser.add_argument("--no_cellline_fallback", action="store_true",
                        help="STRICT mode: drop treated cells lacking a plate-matched DMSO "
                             "control instead of falling back to a same-cell-line control. "
                             "Default off (fallback enabled, examples tagged "
                             "control_plate_matched=false).")
    args = parser.parse_args()

    # Authenticate to HF Hub (helps rate limits / any gated access on compute nodes)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Authenticated to HuggingFace Hub via HF_TOKEN.")
        except Exception as e:
            logger.warning(f"HF login failed ({e}); continuing unauthenticated.")
    else:
        logger.info("No HF_TOKEN in env; streaming Tahoe-100M unauthenticated.")

    # Load metadata
    gene_id_to_symbol, sample_to_conc, drug_info, cvcl_to_name = load_metadata()

    # Print a few examples for sanity checking
    logger.info("\nSample gene mappings:")
    for k, v in list(gene_id_to_symbol.items())[:5]:
        logger.info(f"  token_id {k} -> {v}")

    logger.info("\nSample cell line mappings:")
    for k, v in list(cvcl_to_name.items())[:5]:
        logger.info(f"  {k} -> {v}")

    # Construct training data (train + train_text are streamed to disk inside)
    n_train, eval_examples, linear_model, first_train_example, diversity_stats, held_out_drugs = \
        construct_training_data(
            gene_id_to_symbol=gene_id_to_symbol,
            sample_to_conc=sample_to_conc,
            drug_info=drug_info,
            cvcl_to_name=cvcl_to_name,
            panel_file=args.panel_file,
            output_dir=args.output_dir,
            max_cells_per_condition=args.cells_per_condition,
            num_held_out_drugs=args.held_out_drugs,
            max_genes=args.max_genes,
            test_mode=(args.mode == "test"),
            test_limit=args.test_limit,
            max_train_examples=args.max_train_examples,
            max_eval_per_tier=args.max_eval_per_tier,
            num_shards=args.num_shards,
            rows_per_shard=args.rows_per_shard,
            shard_seed=args.shard_seed,
            dmso_shards=args.dmso_shards,
            dmso_coverage_frac=args.dmso_coverage_frac,
            cellline_fallback=(not args.no_cellline_fallback),
        )

    # train.jsonl + train_text.jsonl are already written incrementally during Pass 2.
    for tier_name, tier_examples in eval_examples.items():
        if tier_examples:
            save_examples(tier_examples, os.path.join(args.output_dir, f"eval_{tier_name}.jsonl"))

    # Save the fixed panel next to the data so training + eval read a consistent panel
    os.makedirs(args.output_dir, exist_ok=True)
    panel_out = os.path.join(args.output_dir, "l1000_panel.json")
    with open(panel_out, "w") as f:
        json.dump(json.load(open(args.panel_file)), f)
    logger.info(f"Saved fixed gene panel to {panel_out}")

    # Save linear model for expression recovery during evaluation
    if linear_model:
        lm_path = os.path.join(args.output_dir, "linear_model.json")
        with open(lm_path, "w") as f:
            json.dump(linear_model, f, indent=2)
        logger.info(f"Saved expression linear model to {lm_path}")
    
    # Save gene_id_to_symbol mapping for evaluation
    gene_map_path = os.path.join(args.output_dir, "gene_id_to_symbol.json")
    with open(gene_map_path, "w") as f:
        json.dump({str(k): v for k, v in gene_id_to_symbol.items()}, f)
    logger.info(f"Saved gene mapping to {gene_map_path}")

    # Save the diversity report alongside the data (so we can compare runs)
    div_path = os.path.join(args.output_dir, "diversity_report.json")
    with open(div_path, "w") as f:
        json.dump(diversity_stats, f, indent=2)
    logger.info(f"Saved diversity report to {div_path}: {json.dumps(diversity_stats)}")

    # Save a few examples for manual inspection
    logger.info("\n--- Sample Training Example ---")
    if first_train_example:
        ex = first_train_example
        panel_symbols = json.load(open(args.panel_file))
        ctrl_sentence = ex["prompt"].split("Control cell:", 1)[1].split("Response cell:", 1)[0].strip() \
            if "Control cell:" in ex["prompt"] else ""
        n_ctrl = len(ctrl_sentence.split())
        n_resp = len(ex["response"].split())
        logger.info(f"PROMPT:\n{ex['prompt'][:300]}...")
        logger.info(f"RESPONSE (first 200 chars):\n{ex['response'][:200]}...")
        logger.info(f"GENE COUNTS: control={n_ctrl}, response={n_resp}, "
                    f"panel={len(panel_symbols)} ([END_CELL] repr: sentences are "
                    f"variable length = #expressed panel genes + 1 sentinel, <= panel+1)")
        # [END_CELL] representation sanity: sentences must END with the sentinel, be
        # shorter than panel+1 (some genes unexpressed), and contain no unexpressed tail.
        ctrl_tokens = ctrl_sentence.split()
        resp_tokens = ex["response"].split()
        problems = []
        if not ctrl_tokens or ctrl_tokens[-1] != "[END_CELL]":
            problems.append("control does not end with [END_CELL]")
        if not resp_tokens or resp_tokens[-1] != "[END_CELL]":
            problems.append("response does not end with [END_CELL]")
        if n_ctrl > len(panel_symbols) + 1 or n_resp > len(panel_symbols) + 1:
            problems.append("sentence longer than panel+1 (unexpected)")
        if n_ctrl == len(panel_symbols) + 1 and n_resp == len(panel_symbols) + 1:
            problems.append("every gene expressed in both (suspicious — tail not dropped?)")
        if problems:
            logger.warning("  [END_CELL] sanity issues: " + "; ".join(problems))
        else:
            logger.info("  [END_CELL] sanity OK: variable-length, sentinel-terminated, "
                        "unexpressed genes absent.")
        logger.info(f"METADATA:\n{json.dumps(ex['metadata'], indent=2)}")

    # Save the full held-out drug list (all Tier-2 drugs, whether or not each one
    # landed cells in the capped eval file).
    held_out_drug_names = sorted(held_out_drugs)
    drugs_in_tier2_eval = len({
        ex["metadata"]["drug"]
        for ex in eval_examples.get("tier2_unseen_drugs", [])
        if "metadata" in ex and "drug" in ex["metadata"]
    })
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "held_out_drugs.json"), "w") as f:
        json.dump(held_out_drug_names, f, indent=2)
    logger.info(f"Saved {len(held_out_drug_names)} held-out drug names "
                f"({drugs_in_tier2_eval} represented in the Tier-2 eval file)")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

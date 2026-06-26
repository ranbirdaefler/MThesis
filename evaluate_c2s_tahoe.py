"""
C2S-Scale Tahoe Evaluation Suite

Comprehensive evaluation of a fine-tuned C2S-Scale model on Tahoe-100M perturbation
data, designed to be the single, reproducible scoring harness for every arm of the
thesis matrix and for external-model benchmarking.

Metrics (per tier):
    HEADLINE  DE-delta Pearson/Spearman — correlation of the PREDICTED rank-shift
              (pred − control) with the TRUE rank-shift (treated − control), scored
              over the top-K differentially-expressed genes (--topk_de). This is the
              perturbation-effect metric; the control-as-prediction baseline scores
              ~0 here by construction, so DE-delta IS the signal (no subtraction).
    τ top-N expressed — Kendall τ over the N highest-expressed true genes
              (--topn_expressed); strips the deterministic unexpressed tail.
    τ panel — Kendall τ over the whole fixed panel (worst-rank for missing). Inflated
              by the canonical tail and the control≈treated structure; SECONDARY.
    Per-pathway rank/expression metrics; dose-response (predicted pathway activity vs
    dose, + agreement with the true dose-trend); generation-format validity
    (panel coverage / hallucination / dup) so τ can be trusted; scFID (opt-in,
    within-model only — NOT comparable across arms).

Robustness:
    * Deterministic GREEDY decoding by default (reproducible); seeded sampling opt-in.
    * Drug-CLUSTERED bootstrap CIs (effective n = #held-out drugs, not #cells).
    * Per-example metrics saved (with example_id) -> paired Δ significance via
      --paired_compare (Wilcoxon + clustered bootstrap on model − baseline).
    * run_manifest.json per run (git, seeds, decoding, file hashes) for provenance.

Usage:
    # Model (greedy, deterministic)
    python evaluate_c2s_tahoe.py \
        --model_path ./checkpoints/pythia_l1000/best \
        --eval_dir ./data --output_dir ./eval_results/pythia_l1000 \
        --bf16 --max_eval 500

    # Control-as-prediction baseline (mandatory reference point)
    python evaluate_c2s_tahoe.py \
        --model_path ./checkpoints/pythia_l1000/best \
        --eval_dir ./data --output_dir ./eval_results/pythia_l1000_control \
        --bf16 --max_eval 500 --baseline control

    # Paired Δ significance of model vs control (no model load; pure post-processing)
    python evaluate_c2s_tahoe.py --paired_compare \
        --model_results ./eval_results/pythia_l1000 \
        --baseline_results ./eval_results/pythia_l1000_control \
        --output_dir ./eval_results/pythia_l1000

    # Optional mean-shift baselines (need --train_file)
    python evaluate_c2s_tahoe.py --model_path ... --eval_dir ./data \
        --output_dir ./eval_results/pythia_l1000_meanshift \
        --baseline global_mean_shift --train_file ./data/train.jsonl
"""

import argparse
import hashlib
import json
import os
import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Hallmark gene sets (MSigDB) — curated subsets for pathway evaluation
# =============================================================================

PATHWAY_GENE_SETS = {
    "apoptosis": [
        "CASP3", "CASP8", "CASP9", "BAX", "BCL2", "BCL2L1", "BID", "CYCS",
        "FADD", "FAS", "TNFRSF10B", "BIRC5", "MCL1", "DIABLO", "APAF1",
        "BAK1", "BBC3", "PMAIP1", "BCL2L11", "CASP7", "CASP6", "CASP10",
        "XIAP", "BIRC2", "BIRC3", "CFLAR", "TNFSF10", "TRADD", "RIPK1",
        "CYLD", "DFFA", "DFFB", "ENDOG", "AIFM1", "HTRA2", "LMNA",
        "PARP1", "ACIN1", "SATB1", "ROCK1", "PAK2",
    ],
    "mapk_signaling": [
        "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "BRAF", "RAF1", "KRAS",
        "HRAS", "NRAS", "SOS1", "GRB2", "EGFR", "ERBB2", "FGFR1",
        "FOS", "JUN", "JUNB", "JUND", "ELK1", "MYC", "DUSP1", "DUSP6",
        "SPRY2", "SPRY4", "ETS1", "ETS2", "MAPK8", "MAPK9", "MAPK14",
        "MAP3K1", "MAP3K5", "MAP3K7", "MKNK1", "RPS6KA1", "RPS6KA3",
    ],
    "pi3k_akt": [
        "PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "AKT3", "MTOR",
        "PTEN", "TSC1", "TSC2", "RPTOR", "RICTOR", "RPS6KB1", "EIF4EBP1",
        "GSK3B", "FOXO1", "FOXO3", "BAD", "CDKN1A", "CDKN1B", "MDM2",
        "RPS6", "EIF4E", "PDK1", "SGK1", "INPP5D", "PIK3CD",
    ],
    "cell_cycle": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1",
        "CCNE1", "RB1", "E2F1", "E2F2", "TP53", "CDKN1A", "CDKN2A",
        "CDKN1B", "MYC", "CDC20", "CDC25A", "CDC25C", "PLK1", "AURKA",
        "AURKB", "BUB1", "MAD2L1", "CHEK1", "CHEK2", "ATM", "ATR",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "ORC1",
        "PCNA", "TOP2A", "MKI67",
    ],
    "interferon_response": [
        "STAT1", "STAT2", "IRF1", "IRF7", "IRF9", "MX1", "MX2", "OAS1",
        "OAS2", "OAS3", "ISG15", "ISG20", "IFIT1", "IFIT2", "IFIT3",
        "IFI6", "IFI27", "IFI35", "IFI44", "IFI44L", "IFITM1", "IFITM2",
        "IFITM3", "BST2", "RSAD2", "DDX58", "HERC5", "USP18", "TRIM22",
        "GBP1", "GBP2", "CXCL10", "CXCL11",
    ],
}


# =============================================================================
# Generation: produce cell sentences from the model
# =============================================================================

def _build_gen_kwargs(tokenizer, max_new_tokens, do_sample, temperature, top_p):
    """Assemble generate() kwargs. In greedy mode we pass NEITHER temperature nor
    top_p (transformers warns and ignores them when do_sample=False), so the call
    is exactly argmax decoding — fully deterministic and reproducible."""
    kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        kw["temperature"] = temperature
        kw["top_p"] = top_p
    return kw


def generate_cell_sentence(model, tokenizer, prompt, max_new_tokens=3800,
                           do_sample=False, temperature=0.7, top_p=0.9, device="cuda"):
    """Generate a single cell sentence from a prompt. Greedy by default."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **_build_gen_kwargs(tokenizer, max_new_tokens, do_sample, temperature, top_p),
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_cell_sentences_batched(model, tokenizer, prompts, max_new_tokens=3800,
                                    do_sample=False, temperature=0.7, top_p=0.9,
                                    device="cuda"):
    """Generate cell sentences for a BATCH of prompts (the main eval-time speedup).

    Greedy by default (deterministic). Left-pads the batch so every row's prompt
    ends at the same column, then slices each output past the shared prompt length.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **enc,
                **_build_gen_kwargs(tokenizer, max_new_tokens, do_sample, temperature, top_p),
            )
        prompt_len = enc["input_ids"].shape[1]  # shared, thanks to left-padding
        texts = []
        for i in range(len(prompts)):
            gen_ids = outputs[i][prompt_len:]
            texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return texts
    finally:
        tokenizer.padding_side = prev_side


# =============================================================================
# Baselines (control-as-prediction is mandatory; mean-shift are should-have)
# =============================================================================

def control_from_prompt(prompt):
    """Extract the control cell sentence embedded in the prompt."""
    seg = prompt.split("Control cell:", 1)[1] if "Control cell:" in prompt else ""
    return seg.split("Response cell:", 1)[0].strip()


_BASELINE_KEYMODE = {
    "global_mean_shift": "global",
    "per_cellline_mean_shift": "cellline",
    "per_moa_mean_shift": "moa",
    "per_moa_cellline_mean_shift": "moa_cellline",
}


def _baseline_key(meta, key_mode):
    """Grouping key for a mean-shift baseline, from example metadata.
    Returns None for the global baseline (no grouping). Shared by the shift
    estimation and the eval-time resolution so they cannot diverge."""
    if key_mode == "global":
        return None
    moa = meta.get("moa") or "unclear"
    if moa in ("unknown", "nan", "None", "", None):
        moa = "unclear"
    cl = meta.get("cell_line_name")
    if key_mode == "cellline":
        return cl
    if key_mode == "moa":
        return moa
    if key_mode == "moa_cellline":
        return f"{moa}||{cl}"
    raise ValueError(f"unknown key_mode {key_mode}")


def compute_mean_shift(train_file, panel, key_mode="global", limit=20000):
    """Estimate the mean rank shift (treated_rank - control_rank) per gene over the
    fixed panel, from training pairs. Used by the mean-shift baselines.

    key_mode selects the grouping: 'global', 'cellline', 'moa', 'moa_cellline'.
    Returns (per_key_map, global_fallback) where per_key_map is {key: {gene: shift}}
    (== global_fallback for key_mode='global'); the fallback covers keys unseen at eval.
    """
    worst = len(panel) + 1
    panel_list = list(panel)
    global_sum = {g: 0.0 for g in panel_list}
    global_cnt = {g: 0 for g in panel_list}
    key_sum = defaultdict(lambda: {g: 0.0 for g in panel_list})
    key_cnt = defaultdict(lambda: {g: 0 for g in panel_list})

    n = 0
    with open(train_file) as f:
        for line in f:
            if n >= limit:
                break
            ex = json.loads(line)
            cr = cell_sentence_to_gene_ranks(control_from_prompt(ex["prompt"]))
            tr = cell_sentence_to_gene_ranks(ex["response"])
            key = _baseline_key(ex.get("metadata", {}), key_mode)
            for g in panel_list:
                d = tr.get(g, worst) - cr.get(g, worst)
                global_sum[g] += d
                global_cnt[g] += 1
                if key is not None:
                    key_sum[key][g] += d
                    key_cnt[key][g] += 1
            n += 1
    logger.info(f"  Estimated mean shift from {n} training pairs (key_mode={key_mode}, "
                f"n_groups={len(key_sum) if key_mode != 'global' else 1})")

    global_shift = {g: (global_sum[g] / global_cnt[g] if global_cnt[g] else 0.0)
                    for g in panel_list}
    if key_mode == "global":
        return global_shift, global_shift
    per_key = {}
    for k in key_sum:
        per_key[k] = {g: (key_sum[k][g] / key_cnt[k][g] if key_cnt[k][g] else 0.0)
                      for g in panel_list}
    return per_key, global_shift


def predict_mean_shift(prompt, shift_map, panel, panel_index):
    """Apply a per-gene mean rank shift to the control ranks, then re-rank the panel."""
    worst = len(panel) + 1
    cr = cell_sentence_to_gene_ranks(control_from_prompt(prompt))
    scores = {g: cr.get(g, worst) + shift_map.get(g, 0.0) for g in panel}
    # lower score = higher expression = earlier in the sentence; canonical order breaks ties
    ordered = sorted(panel, key=lambda g: (scores[g], panel_index[g]))
    return " ".join(ordered)


# =============================================================================
# Convert cell sentence to ranked gene list
# =============================================================================

def cell_sentence_to_gene_ranks(cell_sentence):
    """
    Convert a cell sentence (space-separated gene names) to a dict of gene -> rank.
    Rank 1 = highest expression.
    """
    genes = cell_sentence.strip().split()
    gene_ranks = {}
    for rank, gene in enumerate(genes, 1):
        if gene not in gene_ranks:  # keep first occurrence (highest rank)
            gene_ranks[gene] = rank
    return gene_ranks


# =============================================================================
# Metric computation
# =============================================================================

def compute_gene_overlap(pred_genes, true_genes):
    """Fraction of true genes present in predicted genes."""
    pred_set = set(pred_genes.keys())
    true_set = set(true_genes.keys())
    if len(true_set) == 0:
        return 0.0
    return len(pred_set & true_set) / len(true_set)


def compute_rank_correlation(pred_ranks, true_ranks, gene_subset=None):
    """
    Compute Kendall τ and Pearson R between predicted and true gene ranks.
    If gene_subset is provided, compute on ALL genes in that subset,
    assigning worst rank to missing genes (following C2S-Scale paper).
    If no gene_subset, compute only on overlapping genes.
    """
    if gene_subset is not None:
        # Use all pathway genes with worst-rank assignment for missing ones
        pred_worst = max(pred_ranks.values()) + 1 if pred_ranks else 1
        true_worst = max(true_ranks.values()) + 1 if true_ranks else 1
        
        eval_genes = list(gene_subset)
        if len(eval_genes) < 3:
            return {"kendall_tau": None, "pearson_r": None, "n_genes": 0,
                    "kendall_p": None, "pearson_p": None}
        
        pred_vals = [pred_ranks.get(g, pred_worst) for g in eval_genes]
        true_vals = [true_ranks.get(g, true_worst) for g in eval_genes]
    else:
        common_genes = set(pred_ranks.keys()) & set(true_ranks.keys())
        if len(common_genes) < 5:
            return {"kendall_tau": None, "pearson_r": None, "n_genes": len(common_genes),
                    "kendall_p": None, "pearson_p": None}
        
        eval_genes = list(common_genes)
        pred_vals = [pred_ranks[g] for g in eval_genes]
        true_vals = [true_ranks[g] for g in eval_genes]
    
    pred_arr = np.asarray(pred_vals, dtype=np.float64)
    true_arr = np.asarray(true_vals, dtype=np.float64)
    # Constant input -> correlation undefined (e.g. a tiny pathway subset all tied
    # at worst-rank). Return None rather than emitting a NaN + warning.
    if np.std(pred_arr) == 0 or np.std(true_arr) == 0:
        return {"kendall_tau": None, "pearson_r": None, "n_genes": len(eval_genes),
                "kendall_p": None, "pearson_p": None}

    tau, tau_p = stats.kendalltau(pred_arr, true_arr)
    r, r_p = stats.pearsonr(pred_arr, true_arr)
    tau = float(tau) if tau == tau else None
    r = float(r) if r == r else None

    return {
        "kendall_tau": tau,
        "pearson_r": r,
        "n_genes": len(eval_genes),
        "kendall_p": float(tau_p) if tau_p == tau_p else None,
        "pearson_p": float(r_p) if r_p == r_p else None,
    }


# =============================================================================
# Perturbation-effect (delta) metrics + high-signal gene subsets
# =============================================================================
# The panel-wide rank τ is inflated by the deterministic unexpressed tail and by
# the fact that the treated profile ~ the control (which is in the prompt). The
# metrics below isolate the actual signal: the CHANGE from control (delta), scored
# on the genes the drug actually moved. Gene-subset selection uses the true cell's
# ranks, which is a scoring choice applied identically to model and every baseline
# (not a model input), so it is fair and leak-free.

def _safe_corr(pred_vals, true_vals, method="pearson"):
    """Correlation that handles constant inputs sanely for DELTA metrics:
      - true is constant (no real change) -> None (degenerate; exclude from means)
      - pred is constant (model predicts no change) -> 0.0 (zero predictive skill)
    This is what makes the control-as-prediction baseline correctly score ~0 delta
    skill (its predicted shift is identically zero)."""
    pred = np.asarray(pred_vals, dtype=np.float64)
    true = np.asarray(true_vals, dtype=np.float64)
    if len(pred) < 3 or len(true) < 3:
        return None
    if np.std(true) == 0:
        return None
    if np.std(pred) == 0:
        return 0.0
    if method == "pearson":
        r, _ = stats.pearsonr(pred, true)
    elif method == "spearman":
        r, _ = stats.spearmanr(pred, true)
    else:
        raise ValueError(method)
    return float(r) if r == r else None  # guard NaN


def ranks_over(ranks_dict, genes, worst):
    """Vector of ranks for `genes`, worst-rank for genes absent from the sentence."""
    return [ranks_dict.get(g, worst) for g in genes]


def select_top_de_genes(true_ranks, control_ranks, panel, k, worst):
    """The k genes the drug moved most: largest |true_rank - control_rank|."""
    shifts = [(g, abs(true_ranks.get(g, worst) - control_ranks.get(g, worst))) for g in panel]
    shifts.sort(key=lambda gs: gs[1], reverse=True)
    return [g for g, _ in shifts[:k]]


def select_top_expressed(true_ranks, panel, n, worst):
    """The n highest-expressed genes in the true cell (smallest true rank).
    Strips the deterministic unexpressed tail from the τ computation."""
    return sorted(panel, key=lambda g: true_ranks.get(g, worst))[:n]


def delta_correlation(pred_ranks, true_ranks, control_ranks, genes, worst):
    """Correlation between predicted rank-shift and true rank-shift over `genes`.
    shift = treated_rank - control_rank. This is the perturbation-effect metric
    (the field-standard 'PCC-delta' analogue in rank space)."""
    c = ranks_over(control_ranks, genes, worst)
    p = ranks_over(pred_ranks, genes, worst)
    t = ranks_over(true_ranks, genes, worst)
    pred_shift = [pi - ci for pi, ci in zip(p, c)]
    true_shift = [ti - ci for ti, ci in zip(t, c)]
    return {
        "delta_pearson": _safe_corr(pred_shift, true_shift, "pearson"),
        "delta_spearman": _safe_corr(pred_shift, true_shift, "spearman"),
        "n_genes": len(genes),
    }


def format_validity(pred_sentence, panel_set):
    """Diagnostic on a generated sentence: is it a well-formed panel permutation?
    A model that emits few genes / many hallucinated tokens produces a degenerate
    rank vector (mass-tied at worst-rank), so τ becomes untrustworthy. Reported,
    not failed."""
    toks = pred_sentence.split()
    n_panel = len(panel_set)
    if not toks:
        return {"coverage": 0.0, "hallucination_rate": 1.0, "dup_rate": 0.0,
                "length": 0, "n_panel_unique": 0, "valid": False}
    emitted_panel = [t for t in toks if t in panel_set]
    uniq = set(emitted_panel)
    coverage = len(uniq) / n_panel
    hallucination_rate = (len(toks) - len(emitted_panel)) / len(toks)
    dup_rate = (len(emitted_panel) - len(uniq)) / max(1, len(emitted_panel))
    valid = (coverage >= 0.95) and (hallucination_rate <= 0.05)
    return {"coverage": coverage, "hallucination_rate": hallucination_rate,
            "dup_rate": dup_rate, "length": len(toks),
            "n_panel_unique": len(uniq), "valid": bool(valid)}


def pathway_mean_rank(ranks_dict, gene_set, worst):
    """Mean rank of a pathway's genes (lower = higher predicted expression).
    Used for dose-response: a real activity proxy, unlike using prediction τ."""
    vals = [ranks_dict.get(g, worst) for g in gene_set]
    return float(np.mean(vals)) if vals else None


# =============================================================================
# Cluster (drug-level) bootstrap for confidence intervals
# =============================================================================

def cluster_bootstrap_ci(values, groups, n_boot=1000, seed=0):
    """Bootstrap CI for a mean, resampling at the GROUP (drug) level so that
    correlated cells within a drug don't inflate the effective sample size — the
    effective n for an unseen-drug claim is the number of held-out drugs, not cells.
    Falls back to per-example resampling if fewer than 2 groups are available."""
    pairs = [(v, g) for v, g in zip(values, groups) if v is not None and v == v]
    if not pairs:
        return None
    vals = np.array([v for v, _ in pairs], dtype=np.float64)
    grps = [g for _, g in pairs]
    point = float(np.mean(vals))
    rng = np.random.RandomState(seed)

    by_group = defaultdict(list)
    for v, g in zip(vals, grps):
        by_group[g].append(v)
    uniq = list(by_group.keys())

    boots = []
    if len(uniq) < 2:
        for _ in range(n_boot):
            samp = rng.choice(vals, size=len(vals), replace=True)
            boots.append(samp.mean())
    else:
        keys = list(uniq)
        for _ in range(n_boot):
            chosen = rng.choice(len(keys), size=len(keys), replace=True)
            pooled = []
            for ci in chosen:
                pooled.extend(by_group[keys[ci]])
            if pooled:
                boots.append(np.mean(pooled))
    boots = np.array(boots, dtype=np.float64)
    return {
        "mean": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "boot_std": float(np.std(boots)),
        "n": int(len(vals)),
        "n_groups": int(len(uniq)),
    }


def cell_sentence_to_expression(cell_sentence, linear_model):
    """
    Convert a cell sentence back to expression values using the linear model.
    expression = slope * log10(rank) + intercept
    """
    from collections import defaultdict
    genes = cell_sentence.strip().split()
    slope = linear_model["slope"]
    intercept = linear_model["intercept"]

    gene_positions = defaultdict(list)
    for pos, gene in enumerate(genes, 1):
        gene_positions[gene].append(pos)

    expr_dict = {}
    for gene, positions in gene_positions.items():
        avg_rank = np.mean(positions)
        expr_val = slope * np.log10(avg_rank) + intercept
        expr_dict[gene] = max(0.0, expr_val)
    return expr_dict


def compute_expression_correlation(pred_sentence, true_sentence, linear_model, gene_subset=None):
    """
    Convert cell sentences to expression vectors using the linear model,
    then compute Pearson R on actual expression values.
    """
    if linear_model is None:
        return {"expr_pearson_r": None, "expr_n_genes": 0}
    
    pred_expr = cell_sentence_to_expression(pred_sentence, linear_model)
    true_expr = cell_sentence_to_expression(true_sentence, linear_model)
    
    # Find common genes
    if gene_subset is not None:
        common = set(pred_expr.keys()) & set(true_expr.keys()) & set(gene_subset)
    else:
        common = set(pred_expr.keys()) & set(true_expr.keys())
    
    if len(common) < 5:
        return {"expr_pearson_r": None, "expr_n_genes": len(common)}
    
    pred_vals = np.array([pred_expr[g] for g in common])
    true_vals = np.array([true_expr[g] for g in common])

    if np.std(pred_vals) == 0 or np.std(true_vals) == 0:
        return {"expr_pearson_r": None, "expr_n_genes": len(common)}
    r, p = stats.pearsonr(pred_vals, true_vals)
    if r != r:
        return {"expr_pearson_r": None, "expr_n_genes": len(common)}

    return {
        "expr_pearson_r": float(r),
        "expr_pearson_p": float(p),
        "expr_n_genes": len(common),
    }


def compute_scfid(pred_embeddings, true_embeddings):
    """
    Compute single-cell Fréchet Inception Distance (scFID) between
    predicted and true cell embeddings.
    
    scFID = ||μ_r - μ_g||² + tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^½)
    
    Uses model embeddings as the feature space (analogous to Inception
    features in image FID). Lower is better.
    
    Args:
        pred_embeddings: np.array of shape (n_pred, hidden_dim)
        true_embeddings: np.array of shape (n_true, hidden_dim)
    
    Returns:
        float: scFID score, or None if insufficient data
    """
    if len(pred_embeddings) < 2 or len(true_embeddings) < 2:
        return None
    
    from scipy.linalg import sqrtm
    
    mu_pred = np.mean(pred_embeddings, axis=0)
    mu_true = np.mean(true_embeddings, axis=0)
    
    sigma_pred = np.cov(pred_embeddings, rowvar=False)
    sigma_true = np.cov(true_embeddings, rowvar=False)
    
    # Mean difference
    diff = mu_pred - mu_true
    mean_term = np.dot(diff, diff)
    
    # Matrix square root term
    try:
        covmean = sqrtm(sigma_pred @ sigma_true)
        # sqrtm can return complex numbers due to numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        trace_term = np.trace(sigma_pred + sigma_true - 2 * covmean)
    except Exception:
        trace_term = None
    
    if trace_term is None or np.isnan(trace_term):
        return float(mean_term)  # fall back to just mean difference
    
    return float(mean_term + trace_term)


# =============================================================================
# Embedding extraction
# =============================================================================

def extract_embedding(model, tokenizer, text, device="cuda"):
    """
    Extract embedding by average-pooling the last hidden state.
    Follows C2S-Scale Section 4.6.3.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        
        # Average pool over sequence length (excluding padding)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)
    
    return pooled.squeeze(0).cpu().numpy()


# =============================================================================
# Evaluation routines
# =============================================================================

def compute_scalar_metrics(pred_sentence, true_sentence, true_ranks, control_ranks,
                           panel, panel_index, linear_model, worst,
                           de_genes_by_k, topn_genes, headline_k=50):
    """All per-example scalar metrics for one (prediction, truth, control) triple.
    de_genes_by_k = {K: [top-K DE genes]} precomputed from TRUTH (same for every draw),
    applied identically to model and baselines. DE-Δr is computed at every K (sweep);
    the headline de_delta_pearson is the value at headline_k."""
    pred_ranks = cell_sentence_to_gene_ranks(pred_sentence)

    panel_rank = compute_rank_correlation(pred_ranks, true_ranks, gene_subset=panel)
    topn_rank = compute_rank_correlation(pred_ranks, true_ranks, gene_subset=topn_genes)
    panel_delta = delta_correlation(pred_ranks, true_ranks, control_ranks, panel, worst)
    de_deltas = {k: delta_correlation(pred_ranks, true_ranks, control_ranks, gs, worst)
                 for k, gs in de_genes_by_k.items()}
    headline = de_deltas.get(headline_k) or next(iter(de_deltas.values()))
    expr_overall = compute_expression_correlation(pred_sentence, true_sentence, linear_model)
    overlap = compute_gene_overlap(pred_ranks, true_ranks)

    pathways = {}
    for pw_name, gene_set in PATHWAY_GENE_SETS.items():
        pw_rank = compute_rank_correlation(pred_ranks, true_ranks, gene_subset=gene_set)
        pw_expr = compute_expression_correlation(pred_sentence, true_sentence,
                                                 linear_model, gene_subset=gene_set)
        pathways[pw_name] = {
            "kendall_tau": pw_rank["kendall_tau"],
            "pearson_r": pw_rank["pearson_r"],
            "expr_pearson_r": pw_expr.get("expr_pearson_r"),
            "pred_mean_rank": pathway_mean_rank(pred_ranks, gene_set, worst),
            "true_mean_rank": pathway_mean_rank(true_ranks, gene_set, worst),
        }

    out = {
        "panel_tau": panel_rank["kendall_tau"],
        "panel_pearson": panel_rank["pearson_r"],
        "topn_expressed_tau": topn_rank["kendall_tau"],
        "de_delta_pearson": headline["delta_pearson"],     # HEADLINE (at headline_k)
        "de_delta_spearman": headline["delta_spearman"],
        "panel_delta_pearson": panel_delta["delta_pearson"],
        "panel_delta_spearman": panel_delta["delta_spearman"],
        "expr_pearson_r": expr_overall.get("expr_pearson_r"),
        "gene_overlap": overlap,
        "n_pred_genes": len(pred_ranks),
        "pathways": pathways,
    }
    # K-sweep: DE-Δr at each K, so the headline's K isn't a magic number.
    for k, dd in de_deltas.items():
        out[f"de_delta_pearson_k{k}"] = dd["delta_pearson"]
        out[f"de_delta_spearman_k{k}"] = dd["delta_spearman"]
    return out


def average_metric_dicts(dicts):
    """Mean of a list of identically-structured metric dicts, ignoring None/NaN and
    recursing into nested dicts. With one draw (greedy) this is the identity."""
    if len(dicts) == 1:
        return dicts[0]
    out = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        if isinstance(vals[0], dict):
            out[key] = average_metric_dicts(vals)
        else:
            nums = [float(v) for v in vals
                    if isinstance(v, (int, float, bool)) and float(v) == float(v)]
            out[key] = float(np.mean(nums)) if nums else None
    return out


def evaluate_tier(model, tokenizer, eval_examples, tier_name, panel, panel_index,
                  device="cuda", max_eval=None, extract_embeddings=False,
                  linear_model=None, max_new_tokens=3800, baseline="none",
                  gen_batch_size=16, shift_map=None,
                  do_sample=False, temperature=0.7, top_p=0.9, gen_samples=1,
                  gen_seed=0, n_boot=1000, topk_de=50, topn_expressed=100,
                  subsample_seed=42, min_coverage=0.2, de_k_list=(20, 50, 100, 200)):
    """
    Run prediction + metric computation on a set of eval examples.

    Predictions come from one of:
      baseline="none"                 -> batched model generation
      baseline="control"              -> the control cell sentence, unchanged
      baseline="global_mean_shift"    -> control ranks + global per-gene mean shift
      baseline="per_cellline_mean_shift" -> control ranks + per-cell-line mean shift

    The overall metric is scored over the WHOLE fixed panel (worst-rank for missing),
    matching C2S-Scale — directly comparable across model and baselines.

    Returns:
        results: dict with per-example and aggregated metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {tier_name}: {len(eval_examples)} examples "
                f"(prediction = {baseline if baseline != 'none' else 'model generation'})")
    logger.info(f"{'='*60}")

    worst = len(panel) + 1
    panel_set = set(panel)

    if max_eval and len(eval_examples) > max_eval:
        rng = np.random.RandomState(subsample_seed)
        indices = rng.choice(len(eval_examples), max_eval, replace=False)
        eval_examples = [eval_examples[i] for i in indices]
        logger.info(f"  Subsampled to {len(eval_examples)} examples (seed={subsample_seed})")

    # --- Produce predictions: per example, a list of one-or-more draws ---
    n_draws = gen_samples if (baseline == "none" and do_sample) else 1
    draws_per_example = [[] for _ in eval_examples]

    if baseline == "none":
        for d in range(n_draws):
            if do_sample:
                torch.manual_seed(gen_seed + d)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(gen_seed + d)
            for i in tqdm(range(0, len(eval_examples), gen_batch_size),
                          desc=f"Generating ({tier_name}) draw {d+1}/{n_draws}"):
                batch = eval_examples[i:i + gen_batch_size]
                prompts = [ex["prompt"] for ex in batch]
                gens = generate_cell_sentences_batched(
                    model, tokenizer, prompts, device=device,
                    max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature, top_p=top_p,
                )
                for j, g in enumerate(gens):
                    draws_per_example[i + j].append(g)
    elif baseline == "control":
        for i, ex in enumerate(eval_examples):
            draws_per_example[i].append(control_from_prompt(ex["prompt"]))
    elif baseline in _BASELINE_KEYMODE:
        # compute_mean_shift always returns (per_key_map, global_fallback).
        per_map, global_fallback = shift_map
        key_mode = _BASELINE_KEYMODE[baseline]
        for i, ex in enumerate(eval_examples):
            k = _baseline_key(ex.get("metadata", {}), key_mode)
            sm = global_fallback if k is None else per_map.get(k, global_fallback)
            draws_per_example[i].append(predict_mean_shift(ex["prompt"], sm, panel, panel_index))
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    per_example = []
    pred_embeddings_list = []
    true_embeddings_list = []
    embeddings_meta = []

    for ex, draws in zip(eval_examples, draws_per_example):
        true_ranks = cell_sentence_to_gene_ranks(ex["response"])
        control_ranks = cell_sentence_to_gene_ranks(control_from_prompt(ex["prompt"]))

        # High-signal gene subsets are chosen from TRUTH, so they are identical across
        # draws and applied identically to model and every baseline (fair, leak-free).
        # K-sweep: select the top max(K) DE genes once (rank-ordered by |shift|), then
        # each smaller K is a prefix — DE-Δr is reported at every K so the headline K
        # is a robustness axis, not a magic number.
        _max_k = max(de_k_list)
        de_ranked = select_top_de_genes(true_ranks, control_ranks, panel, _max_k, worst)
        de_genes_by_k = {k: de_ranked[:k] for k in de_k_list}
        topn_genes = select_top_expressed(true_ranks, panel, topn_expressed, worst)

        draw_metrics = [
            compute_scalar_metrics(ps, ex["response"], true_ranks, control_ranks,
                                   panel, panel_index, linear_model, worst,
                                   de_genes_by_k, topn_genes, headline_k=topk_de)
            for ps in draws
        ]
        m = average_metric_dicts(draw_metrics)
        fmt = average_metric_dicts([format_validity(ps, panel_set) for ps in draws])

        # Degenerate-output guard. When a model emits almost no valid panel genes,
        # the missing-gene worst-rank fill makes the rank/shift correlations
        # spurious — every missing gene gets the constant worst rank, so the
        # predicted shift collapses to a near-constant function of the control
        # rank and correlates with the truth for reasons unrelated to prediction
        # (a model that predicts NOTHING can score a high DE-Δr). We null those
        # metrics so they are excluded from the aggregate rather than rewarded.
        # gene_overlap / n_pred_genes / format stats are kept (they describe the
        # failure). Threshold is min_coverage (fraction of the panel emitted).
        if (fmt.get("coverage") or 0.0) < min_coverage:
            for _k in list(m.keys()):
                if (_k.startswith("de_delta_pearson") or _k.startswith("de_delta_spearman")
                        or _k.startswith("panel_delta")
                        or _k in ("panel_tau", "panel_pearson",
                                  "topn_expressed_tau", "expr_pearson_r")):
                    m[_k] = None
            m["degenerate"] = True
        else:
            m["degenerate"] = False

        per_example.append({
            "example_id": hashlib.sha1(ex["prompt"].encode("utf-8")).hexdigest()[:16],
            "drug": ex.get("metadata", {}).get("drug"),
            "cell_line_name": ex.get("metadata", {}).get("cell_line_name"),
            "metadata": ex["metadata"],
            "metrics": m,
            "format": fmt,
            "n_true_genes": len(true_ranks),
            "n_draws": len(draws),
        })

        if extract_embeddings:
            pred_emb = extract_embedding(model, tokenizer, draws[0], device=device)
            true_emb = extract_embedding(model, tokenizer, ex["response"], device=device)
            pred_embeddings_list.append(pred_emb)
            true_embeddings_list.append(true_emb)
            embeddings_meta.append(ex["metadata"])

    # Aggregate metrics (+ drug-clustered bootstrap CIs)
    agg = aggregate_metrics(per_example, n_boot=n_boot)
    agg["baseline"] = baseline

    # scFID (opt-in). WARNING: uses the evaluated model's OWN embeddings as the
    # feature space, so values are NOT comparable across different arms.
    scfid = None
    if extract_embeddings and pred_embeddings_list:
        logger.warning("  scFID uses THIS model's embeddings as the feature space — "
                       "not comparable across arms; within-model diagnostic only.")
        pred_emb_array = np.stack(pred_embeddings_list)
        true_emb_array = np.stack(true_embeddings_list)
        scfid = compute_scfid(pred_emb_array, true_emb_array)
        agg["scfid"] = scfid
    
    def fmt_ci(key):
        c = agg["metrics"].get(key)
        if not c or c.get("mean") is None:
            return "N/A"
        if "ci_low" in c:
            return f"{c['mean']:.4f} [{c['ci_low']:.4f}, {c['ci_high']:.4f}]"
        return f"{c['mean']:.4f}"

    logger.info(f"\n--- {tier_name} Results ({'baseline=' + baseline if baseline != 'none' else 'model'}) ---")
    logger.info(f"  DE-delta Pearson (top-{topk_de}):   {fmt_ci('de_delta_pearson')}   <- HEADLINE (perturbation effect; read Δ vs control)")
    logger.info(f"  DE-delta Spearman (top-{topk_de}):  {fmt_ci('de_delta_spearman')}")
    logger.info(f"  τ top-{topn_expressed} expressed:         {fmt_ci('topn_expressed_tau')}")
    logger.info(f"  τ panel (tail-inflated, 2ndary):  {fmt_ci('panel_tau')}")
    logger.info(f"  panel Δ Pearson:                  {fmt_ci('panel_delta_pearson')}")
    if agg["metrics"].get("expr_pearson_r", {}).get("mean") is not None:
        logger.info(f"  expr Pearson (derived from rank): {fmt_ci('expr_pearson_r')}")
    fm = agg.get("format_means", {})
    if fm.get("coverage") is not None:
        logger.info(f"  format: coverage={fm['coverage']:.3f} "
                    f"hallucination={fm['hallucination_rate']:.3f} "
                    f"dup={fm['dup_rate']:.3f} valid_rate={fm['valid_rate']:.3f}"
                    f"  (low coverage / high hallucination => τ untrustworthy)")
    if scfid is not None:
        logger.info(f"  scFID (within-model only): {scfid:.4f}")
    for pw_name, pw in agg["pathway_means"].items():
        if pw["kendall_tau"] is not None:
            logger.info(f"  {pw_name:25s} τ={pw['kendall_tau']:.4f}  r={pw['pearson_r']:.4f}")
    
    # Package embeddings for saving
    embeddings_output = None
    if extract_embeddings and pred_embeddings_list:
        embeddings_output = {
            "pred_embeddings": [e.tolist() for e in pred_embeddings_list],
            "true_embeddings": [e.tolist() for e in true_embeddings_list],
            "metadata": embeddings_meta,
        }
    
    return {
        "per_example": per_example,
        "aggregated": agg,
        "embeddings": embeddings_output,
    }


def aggregate_metrics(per_example, n_boot=1000):
    """Mean of each scalar metric across examples. Headline metrics get a
    drug-clustered bootstrap CI; pathway and format stats get simple means."""
    if not per_example:
        return {"metrics": {}, "pathway_means": {}, "format_means": {}, "n_examples": 0}

    groups = [e.get("drug") for e in per_example]
    metric_keys = ["de_delta_pearson", "de_delta_spearman", "topn_expressed_tau",
                   "panel_tau", "panel_pearson", "panel_delta_pearson",
                   "panel_delta_spearman", "expr_pearson_r", "gene_overlap",
                   "n_pred_genes"]
    # include the K-sweep keys (de_delta_pearson_k20, ...), sorted by K
    swept = [k for k in per_example[0]["metrics"]
             if k.startswith("de_delta_pearson_k") or k.startswith("de_delta_spearman_k")]
    swept.sort(key=lambda k: (k.split("_k")[0], int(k.split("_k")[1])))
    metric_keys = metric_keys + swept
    metrics = {}
    for k in metric_keys:
        vals = [e["metrics"].get(k) for e in per_example]
        ci = cluster_bootstrap_ci(vals, groups, n_boot=n_boot)
        metrics[k] = ci if ci is not None else {"mean": None, "n": 0}

    def _mean(xs):
        xs = [float(v) for v in xs if v is not None and float(v) == float(v)]
        return float(np.mean(xs)) if xs else None

    pathway_means = {}
    for pw in PATHWAY_GENE_SETS:
        col = lambda f: [e["metrics"]["pathways"][pw].get(f) for e in per_example]
        pathway_means[pw] = {
            "kendall_tau": _mean(col("kendall_tau")),
            "pearson_r": _mean(col("pearson_r")),
            "expr_pearson_r": _mean(col("expr_pearson_r")),
            "pred_mean_rank": _mean(col("pred_mean_rank")),
            "true_mean_rank": _mean(col("true_mean_rank")),
            "n_examples": sum(1 for v in col("kendall_tau") if v is not None and v == v),
        }

    format_means = {
        "coverage": _mean([e["format"].get("coverage") for e in per_example]),
        "hallucination_rate": _mean([e["format"].get("hallucination_rate") for e in per_example]),
        "dup_rate": _mean([e["format"].get("dup_rate") for e in per_example]),
        "length": _mean([e["format"].get("length") for e in per_example]),
        "valid_rate": _mean([e["format"].get("valid") for e in per_example]),
        "n_degenerate": sum(1 for e in per_example if e["metrics"].get("degenerate")),
        "n_scored_de_delta": sum(1 for e in per_example
                                 if e["metrics"].get("de_delta_pearson") is not None),
        "n": len(per_example),
    }

    return {
        "metrics": metrics,
        "pathway_means": pathway_means,
        "format_means": format_means,
        "n_examples": len(per_example),
    }


# =============================================================================
# Generic-output check (does the model react to the drug at all?)
# =============================================================================

def generic_output_check(model, tokenizer, eval_examples, panel, device,
                         max_new_tokens=3000, n_drugs=4):
    """Feed ONE control cell with several different drug prompts and check the
    generated responses differ. Pairwise Kendall τ over the panel well below 1.0 =
    the model is drug-sensitive; τ near 1.0 = it is ignoring the perturbation
    (flat / generic output). Logged for inspection; does not fail the run.
    """
    logger.info(f"\n--- Generic-Output Check (1 control x up to {n_drugs} drugs) ---")
    if len(eval_examples) < 2:
        logger.info("  Not enough examples for the generic-output check")
        return {}

    base = eval_examples[0]
    control = control_from_prompt(base["prompt"])
    cell_line = base["metadata"].get("cell_line_name", "the cell line")
    if not control:
        logger.info("  Could not extract a control cell from the prompt; skipping")
        return {}

    # Pick several DISTINCT drugs to apply to the same control cell.
    seen, picks = set(), []
    for ex in eval_examples:
        d = ex["metadata"].get("drug")
        if d and d not in seen:
            seen.add(d)
            picks.append(ex["metadata"])
        if len(picks) >= n_drugs:
            break
    if len(picks) < 2:
        logger.info("  Fewer than 2 distinct drugs available; skipping")
        return {}

    prompts = []
    for m in picks:
        moa = m.get("moa") or "unclear"
        if moa in ("unknown", "nan", "None"):
            moa = "unclear"
        prompts.append(
            f"Predict the response of {cell_line} to {m['drug']} at "
            f"{m.get('dose', 'unknown')}. Mechanism: {moa}."
            f"\nControl cell: {control}\n\nResponse cell:"
        )

    gens = generate_cell_sentences_batched(
        model, tokenizer, prompts, device=device, max_new_tokens=max_new_tokens
    )
    ranks = [cell_sentence_to_gene_ranks(g) for g in gens]
    pairwise = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            tau = compute_rank_correlation(ranks[i], ranks[j], gene_subset=panel)["kendall_tau"]
            if tau is not None:
                pairwise.append(tau)

    drugs = [m["drug"] for m in picks]
    summary = {
        "drugs": drugs,
        "mean_pairwise_kendall": float(np.mean(pairwise)) if pairwise else None,
        "max_pairwise_kendall": float(np.max(pairwise)) if pairwise else None,
        "n_pairs": len(pairwise),
    }
    if pairwise:
        logger.info(f"  Drugs: {drugs}")
        logger.info(f"  Mean pairwise τ across drug outputs: "
                    f"{summary['mean_pairwise_kendall']:.4f} "
                    f"(near 1.0 = generic/drug-insensitive; well below 1.0 = drug-sensitive)")
    return summary


# =============================================================================
# Dose-response analysis
# =============================================================================

def analyze_dose_response(per_example_results):
    """
    Check whether the model captures dose-response relationships.

    For each (drug, cell_line) with 3+ doses, measure how the PREDICTED pathway
    activity (mean predicted rank of the pathway's genes — lower rank = higher
    expression) tracks dose. We report:
      pred_dose_corr : |Spearman(dose, predicted pathway activity)| — does the
                       model's prediction move monotonically with dose at all?
      pred_vs_true   : Spearman between the model's dose-trend and the TRUE
                       dose-trend — does it move the RIGHT way?
    (The previous version used prediction-accuracy τ as an activity proxy, which
    did not actually measure dose-response.)
    """
    logger.info("\n--- Dose-Response Analysis ---")

    groups = defaultdict(list)
    for ex in per_example_results:
        meta = ex["metadata"]
        dose = meta.get("dose_float")
        if dose is not None:
            key = (meta.get("drug"), meta.get("cell_line_name"))
            groups[key].append((dose, ex))

    multi_dose = {k: sorted(v, key=lambda x: x[0])
                  for k, v in groups.items() if len(v) >= 3}

    if not multi_dose:
        logger.info("  No drug-cell_line pairs with 3+ doses found")
        return {}

    logger.info(f"  Analyzing {len(multi_dose)} drug-cell_line pairs with 3+ doses")

    pathway_rows = defaultdict(list)
    for (drug, cl), dose_results in multi_dose.items():
        doses = [d[0] for d in dose_results]
        for pw_name in PATHWAY_GENE_SETS:
            pred_activity, true_activity = [], []
            for _, ex_result in dose_results:
                pw = ex_result["metrics"]["pathways"][pw_name]
                pred_activity.append(pw.get("pred_mean_rank"))
                true_activity.append(pw.get("true_mean_rank"))
            if any(v is None for v in pred_activity) or len(pred_activity) < 3:
                continue
            if np.std(pred_activity) == 0 or np.std(doses) == 0:
                continue
            pred_corr, _ = stats.spearmanr(doses, pred_activity)
            pred_vs_true = np.nan
            if not any(v is None for v in true_activity) and np.std(true_activity) > 0:
                pred_vs_true, _ = stats.spearmanr(pred_activity, true_activity)
            pathway_rows[pw_name].append({
                "drug": drug, "cell_line": cl,
                "pred_dose_corr": float(pred_corr) if pred_corr == pred_corr else None,
                "pred_vs_true": float(pred_vs_true) if pred_vs_true == pred_vs_true else None,
                "n_doses": len(pred_activity),
            })

    summary = {}
    for pw_name, rows in pathway_rows.items():
        mono = [abs(r["pred_dose_corr"]) for r in rows if r["pred_dose_corr"] is not None]
        agree = [r["pred_vs_true"] for r in rows if r["pred_vs_true"] is not None]
        if mono:
            summary[pw_name] = {
                "mean_pred_dose_monotonicity": float(np.mean(mono)),
                "mean_pred_vs_true_trend": float(np.mean(agree)) if agree else None,
                "n_pairs": len(mono),
            }
            agree_str = f", pred-vs-true trend={np.mean(agree):.3f}" if agree else ""
            logger.info(f"  {pw_name:25s} |dose-monotonicity|={np.mean(mono):.3f} "
                        f"(n={len(mono)}){agree_str}")

    return summary


# =============================================================================
# Multi-step perturbation probe
# =============================================================================

def probe_multistep_perturbation(model, tokenizer, eval_examples, device="cuda",
                                 n_probes=20, max_new_tokens=3000):
    """
    Test sequential perturbation: take a generated treated cell and apply
    a second drug. Checks if the model produces coherent sequential responses.

    This is exploratory — even negative results are informative.
    """
    logger.info(f"\n--- Multi-Step Perturbation Probe (n={n_probes}) ---")

    if len(eval_examples) < 2:
        logger.info("  Not enough examples for multi-step probe")
        return {}

    results = []
    np.random.seed(42)

    for i in range(min(n_probes, len(eval_examples))):
        ex = eval_examples[i]

        # Step 1: Generate treated cell from the original prompt
        step1_response = generate_cell_sentence(
            model, tokenizer, ex["prompt"], device=device, max_new_tokens=max_new_tokens
        )
        
        # Step 2: Pick a different drug and apply to the generated cell
        other_idx = (i + 1) % len(eval_examples)
        other_ex = eval_examples[other_idx]
        other_drug = other_ex["metadata"]["drug"]
        other_dose = other_ex["metadata"]["dose"]
        other_moa = other_ex["metadata"]["moa"]
        cell_line = ex["metadata"]["cell_line_name"]
        
        # Construct a sequential prompt
        step2_prompt = (
            f"The following cell has been treated with {ex['metadata']['drug']} "
            f"at {ex['metadata']['dose']}. "
            f"Predict the additional response of {cell_line} to {other_drug} "
            f"at {other_dose}.\n\n"
            f"Current cell state: {step1_response[:2000]}\n\n"  # truncate for context
            f"Response cell:"
        )
        
        step2_response = generate_cell_sentence(
            model, tokenizer, step2_prompt, device=device, max_new_tokens=max_new_tokens
        )

        # Analyze: is step2 response different from step1?
        step1_ranks = cell_sentence_to_gene_ranks(step1_response)
        step2_ranks = cell_sentence_to_gene_ranks(step2_response)
        
        # Compute correlation between step1 and step2
        correlation = compute_rank_correlation(step1_ranks, step2_ranks)
        overlap = compute_gene_overlap(step1_ranks, step2_ranks)
        
        results.append({
            "drug1": ex["metadata"]["drug"],
            "drug2": other_drug,
            "cell_line": cell_line,
            "step1_step2_kendall": correlation["kendall_tau"],
            "step1_step2_pearson": correlation["pearson_r"],
            "step1_step2_overlap": overlap,
            "step2_n_genes": len(step2_ranks),
        })
        
        if i < 3:  # log first few for inspection
            _tau = correlation["kendall_tau"]
            _tau_s = f"{_tau:.4f}" if _tau is not None else "NA"
            logger.info(f"  Probe {i}: {ex['metadata']['drug']} → {other_drug}")
            logger.info(f"    Step1-Step2 τ={_tau_s}, "
                        f"overlap={overlap:.4f}, "
                        f"step2 genes={len(step2_ranks)}")
    
    # Summarize
    if results:
        taus = [r["step1_step2_kendall"] for r in results 
                if r["step1_step2_kendall"] is not None]
        overlaps = [r["step1_step2_overlap"] for r in results]
        
        summary = {
            "mean_step1_step2_kendall": np.mean(taus) if taus else None,
            "mean_step1_step2_overlap": np.mean(overlaps),
            "n_probes": len(results),
        }
        
        logger.info(f"\n  Multi-step summary:")
        logger.info(f"    Mean step1-step2 τ: {summary['mean_step1_step2_kendall']:.4f}")
        logger.info(f"    Mean step1-step2 overlap: {summary['mean_step1_step2_overlap']:.4f}")
        logger.info(f"    (τ close to 1.0 = model ignores drug2; "
                    f"τ < 0.9 = model changes expression)")
        
        return summary
    
    return {}


# =============================================================================
# Provenance + paired Δ comparison
# =============================================================================

def file_sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def paired_delta_analysis(model_dir, baseline_dir, output_dir,
                          metric_key="de_delta_pearson", n_boot=1000):
    """Pair the model and baseline per-example results by example_id (robust to
    ordering), then test the per-example difference model−baseline: Wilcoxon
    signed-rank p plus a drug-clustered bootstrap CI on the mean Δ. This is the
    proper significance for the headline Δ, since both predictions are scored on
    the SAME cells."""
    tiers = ["tier1_seen_conditions", "tier2_unseen_drugs",
             "tier3_unseen_combos", "tier4_dose_interpolation"]

    def find_pe(d, tier):
        cands = sorted(f for f in os.listdir(d)
                       if f.startswith(f"{tier}_per_example") and f.endswith(".json"))
        return os.path.join(d, cands[0]) if cands else None

    out = {}
    for tier in tiers:
        mp = find_pe(model_dir, tier)
        bp = find_pe(baseline_dir, tier)
        if not (mp and bp):
            continue
        with open(mp) as f:
            mrows = {r["example_id"]: r for r in json.load(f)}
        with open(bp) as f:
            brows = {r["example_id"]: r for r in json.load(f)}
        common = sorted(set(mrows) & set(brows))
        diffs, groups = [], []
        for eid in common:
            mv = mrows[eid]["metrics"].get(metric_key)
            bv = brows[eid]["metrics"].get(metric_key)
            if mv is None or bv is None or mv != mv or bv != bv:
                continue
            diffs.append(float(mv) - float(bv))
            groups.append(mrows[eid].get("drug"))
        if len(diffs) < 3:
            continue
        ci = cluster_bootstrap_ci(diffs, groups, n_boot=n_boot)
        wil_p = None
        if any(d != 0 for d in diffs):
            try:
                _, wil_p = stats.wilcoxon(diffs)
                wil_p = float(wil_p)
            except Exception:
                wil_p = None
        out[tier] = {
            "metric": metric_key,
            "n_paired": len(diffs),
            "mean_delta": ci["mean"] if ci else None,
            "ci_low": ci["ci_low"] if ci else None,
            "ci_high": ci["ci_high"] if ci else None,
            "n_drugs": ci["n_groups"] if ci else None,
            "wilcoxon_p": wil_p,
        }
        sig = ""
        if out[tier]["ci_low"] is not None:
            sig = " (CI excludes 0)" if (out[tier]["ci_low"] > 0 or out[tier]["ci_high"] < 0) else " (CI includes 0)"
        logger.info(f"  {tier:28s} Δ{metric_key}={out[tier]['mean_delta']:.4f} "
                    f"[{out[tier]['ci_low']:.4f}, {out[tier]['ci_high']:.4f}] "
                    f"p={wil_p}{sig}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"paired_delta_{metric_key}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"  Saved paired Δ analysis to {out_path}")
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate C2S-Scale on Tahoe perturbation data")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model checkpoint (not needed for --paired_compare)")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Directory containing eval JSONL files (not needed for --paired_compare)")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Where to save evaluation results")
    parser.add_argument("--max_eval", type=int, default=500,
                        help="Max examples to evaluate per tier")
    parser.add_argument("--gen_batch_size", type=int, default=16,
                        help="Batched generation size (the biggest eval-time speedup). "
                             "Tune to fit the H200.")
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "control", "global_mean_shift", "per_cellline_mean_shift",
                                 "per_moa_mean_shift", "per_moa_cellline_mean_shift"],
                        help="Prediction source. 'none' = model generation; 'control' = the "
                             "control cell unchanged (MANDATORY baseline — report Δτ = model - "
                             "control); mean-shift baselines need --train_file. per_moa / "
                             "per_moa_cellline test how much signal is mechanism-of-action "
                             "(not drug) specific.")
    parser.add_argument("--train_file", type=str, default=None,
                        help="train.jsonl, required for the mean-shift baselines")
    parser.add_argument("--baseline_train_limit", type=int, default=20000,
                        help="Training pairs used to estimate the mean shift")
    parser.add_argument("--extract_embeddings", action="store_true",
                        help="Extract and save embeddings for visualization")
    parser.add_argument("--multistep_probes", type=int, default=20,
                        help="Number of multi-step perturbation probes (model runs only)")
    parser.add_argument("--check_generic", action="store_true",
                        help="Run the generic-output (drug-sensitivity) check. Opt-in: "
                             "it generates and is a diagnostic, not part of the core metrics.")
    parser.add_argument("--probe_multistep", action="store_true",
                        help="Run the multi-step perturbation probe. Opt-in (generates).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--bf16", action="store_true",
                        help="Load model in bf16")
    parser.add_argument("--max_new_tokens", type=int, default=3800,
                        help="Max tokens to generate per example. MEASURED: a 946-gene panel "
                             "response is ~3079 tokens (3.255 tokens/gene); 3800 = panel x "
                             "tokens/gene x 1.2 headroom. 2000-3000 would truncate generation.")
    # --- decoding (deterministic by default for reproducible metrics) ---
    parser.add_argument("--decoding", choices=["greedy", "sample"], default="greedy",
                        help="greedy = deterministic argmax (default; reproducible). "
                             "sample = stochastic; use a fixed --gen_seed and optionally "
                             "average over --gen_samples draws.")
    parser.add_argument("--gen_seed", type=int, default=0,
                        help="Seed for sampling decoding (and torch), for reproducibility.")
    parser.add_argument("--gen_samples", type=int, default=1,
                        help="Number of stochastic draws per example to average (sample mode only).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    # --- statistics ---
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Bootstrap resamples for drug-clustered CIs.")
    parser.add_argument("--subsample_seed", type=int, default=42,
                        help="Seed for per-tier subsampling. Keep IDENTICAL across model and "
                             "baseline runs so per-example pairing (paired Δ) lines up.")
    parser.add_argument("--topk_de", type=int, default=50,
                        help="Top-K differentially-expressed genes (by true rank shift) for the "
                             "HEADLINE delta metric.")
    parser.add_argument("--topk_de_sweep", type=str, default="20,50,100,200",
                        help="Comma-separated K values at which to ALSO report DE-Δr "
                             "(de_delta_pearson_k{K}), so the headline K is a robustness axis "
                             "rather than a magic number. The headline --topk_de is added "
                             "automatically if absent.")
    parser.add_argument("--topn_expressed", type=int, default=100,
                        help="Top-N highest-expressed true genes for the tail-stripped τ.")
    parser.add_argument("--min_coverage", type=float, default=0.2,
                        help="Degenerate-output guard: if a prediction emits fewer than this "
                             "fraction of the panel's genes, its rank/shift metrics (incl. "
                             "DE-Δr) are nulled and excluded from aggregates, since the "
                             "worst-rank fill makes them spurious. Set 0 to disable.")
    # --- paired Δ comparison mode (no model load; pure post-processing) ---
    parser.add_argument("--paired_compare", action="store_true",
                        help="Compute paired Δ (model − baseline) significance from two prior "
                             "result dirs. Requires --model_results and --baseline_results.")
    parser.add_argument("--model_results", type=str, default=None,
                        help="Results dir of the MODEL run (with *_per_example.json).")
    parser.add_argument("--baseline_results", type=str, default=None,
                        help="Results dir of the BASELINE run (with *_per_example.json).")
    parser.add_argument("--compare_metric", type=str, default="de_delta_pearson",
                        help="Metric key for paired Δ (default: the headline DE-delta Pearson).")

    args = parser.parse_args()

    # Parse the DE K-sweep; always include the headline --topk_de.
    de_k_list = sorted({int(x) for x in str(args.topk_de_sweep).split(",") if x.strip()}
                       | {int(args.topk_de)})

    # --- Paired Δ comparison mode: no model needed, pure post-processing ---
    if args.paired_compare:
        if not (args.model_results and args.baseline_results):
            logger.error("--paired_compare needs --model_results and --baseline_results.")
            return
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"\nPaired Δ ({args.compare_metric}): "
                    f"{args.model_results} vs {args.baseline_results}")
        paired_delta_analysis(args.model_results, args.baseline_results,
                              args.output_dir, metric_key=args.compare_metric,
                              n_boot=args.n_boot)
        logger.info("Done (paired compare).")
        return

    if not args.model_path or not args.eval_dir:
        logger.error("--model_path and --eval_dir are required for an evaluation run "
                     "(only --paired_compare may omit them).")
        return
    
    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {args.device}")
    logger.info(f"Decoding: {args.decoding}"
                + (f" (greedy/deterministic)" if args.decoding == "greedy"
                   else f" temp={args.temperature} top_p={args.top_p} "
                        f"samples={args.gen_samples} seed={args.gen_seed}"))
    torch.manual_seed(args.gen_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.gen_seed)
    do_sample = (args.decoding == "sample")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    _load_dtype = torch.bfloat16 if args.bf16 else torch.float32
    try:
        # transformers >=5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=_load_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=_load_dtype)
    model = model.to(args.device)
    model.eval()
    logger.info(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load eval datasets
    eval_tiers = {}
    tier_files = {
        "tier1_seen_conditions": "eval_tier1_seen_conditions.jsonl",
        "tier2_unseen_drugs": "eval_tier2_unseen_drugs.jsonl",
        "tier3_unseen_combos": "eval_tier3_unseen_combos.jsonl",
        "tier4_dose_interpolation": "eval_tier4_dose_interpolation.jsonl",
    }
    
    for tier_name, filename in tier_files.items():
        filepath = os.path.join(args.eval_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                examples = [json.loads(line) for line in f]
            if examples:
                eval_tiers[tier_name] = examples
                logger.info(f"  Loaded {len(examples)} examples for {tier_name}")
    
    # Load linear model for expression recovery
    linear_model = None
    lm_path = os.path.join(args.eval_dir, "linear_model.json")
    if os.path.exists(lm_path):
        with open(lm_path) as f:
            linear_model = json.load(f)
        logger.info(f"  Loaded linear model: slope={linear_model['slope']:.4f}, "
                    f"intercept={linear_model['intercept']:.4f}, "
                    f"R²={linear_model['r_squared']:.4f} (fit={linear_model.get('fit', 'global')})")
    else:
        logger.warning("  No linear_model.json found — expression-space metrics will be skipped")

    # Load the FIXED panel — the overall metric is scored over the whole panel.
    panel_path = os.path.join(args.eval_dir, "l1000_panel.json")
    if not os.path.exists(panel_path):
        logger.error(f"No l1000_panel.json found in {args.eval_dir} — required for "
                     f"fixed-panel scoring. Run build_l1000_panel.py / preprocessing first.")
        return
    with open(panel_path) as f:
        panel = json.load(f)
    panel_index = {g: i for i, g in enumerate(panel)}
    logger.info(f"  Loaded fixed panel: {len(panel)} genes")

    if not eval_tiers:
        logger.error("No eval data found!")
        return

    # For mean-shift baselines, estimate the per-gene shift from train.jsonl once.
    shift_map = None
    if args.baseline in _BASELINE_KEYMODE:
        if not args.train_file or not os.path.exists(args.train_file):
            logger.error(f"--baseline {args.baseline} needs --train_file (train.jsonl).")
            return
        logger.info(f"\nEstimating mean shift for baseline '{args.baseline}' ...")
        shift_map = compute_mean_shift(args.train_file, panel,
                                       key_mode=_BASELINE_KEYMODE[args.baseline],
                                       limit=args.baseline_train_limit)

    # Tag output files so model and baseline runs don't collide in a shared dir.
    suffix = "" if args.baseline == "none" else f"_{args.baseline}"

    # Run evaluation per tier
    all_results = {}
    all_per_example = []  # collect for dose-response analysis

    for tier_name, examples in eval_tiers.items():
        tier_results = evaluate_tier(
            model, tokenizer, examples, tier_name, panel, panel_index,
            device=args.device,
            max_eval=args.max_eval,
            extract_embeddings=args.extract_embeddings,
            linear_model=linear_model,
            max_new_tokens=args.max_new_tokens,
            baseline=args.baseline,
            gen_batch_size=args.gen_batch_size,
            shift_map=shift_map,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            gen_samples=args.gen_samples,
            gen_seed=args.gen_seed,
            n_boot=args.n_boot,
            topk_de=args.topk_de,
            topn_expressed=args.topn_expressed,
            subsample_seed=args.subsample_seed,
            min_coverage=args.min_coverage,
            de_k_list=de_k_list,
        )
        all_results[tier_name] = tier_results
        all_per_example.extend(tier_results["per_example"])

        # Save aggregated tier results
        tier_output = os.path.join(args.output_dir, f"{tier_name}_results{suffix}.json")
        with open(tier_output, "w") as f:
            json.dump(tier_results["aggregated"], f, indent=2, default=str)

        # Save per-example metrics (drops embeddings; keeps id/drug/metrics/format)
        # so a later --paired_compare run can pair model vs baseline by example_id.
        pe_slim = [
            {"example_id": e["example_id"], "drug": e["drug"],
             "cell_line_name": e["cell_line_name"],
             "metrics": e["metrics"], "format": e["format"]}
            for e in tier_results["per_example"]
        ]
        pe_output = os.path.join(args.output_dir, f"{tier_name}_per_example{suffix}.json")
        with open(pe_output, "w") as f:
            json.dump(pe_slim, f, default=str)

        # Save embeddings separately if extracted
        if tier_results.get("embeddings"):
            emb_output = os.path.join(args.output_dir, f"{tier_name}_embeddings{suffix}.json")
            with open(emb_output, "w") as f:
                json.dump(tier_results["embeddings"], f)

    # Dose-response analysis (across all tiers)
    dose_results = analyze_dose_response(all_per_example)
    with open(os.path.join(args.output_dir, f"dose_response{suffix}.json"), "w") as f:
        json.dump(dose_results, f, indent=2, default=str)

    # Model-only diagnostics (these require generation, so skip for baselines).
    # They are OPT-IN and wrapped defensively: a diagnostic must never abort the
    # run after the real per-tier results have already been saved.
    if args.baseline == "none":
        if args.check_generic and "tier1_seen_conditions" in eval_tiers:
            try:
                generic_results = generic_output_check(
                    model, tokenizer, eval_tiers["tier1_seen_conditions"], panel,
                    device=args.device, max_new_tokens=args.max_new_tokens,
                )
                with open(os.path.join(args.output_dir, "generic_output_check.json"), "w") as f:
                    json.dump(generic_results, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"  generic-output check skipped (non-fatal): {e}")

        if args.probe_multistep and "tier1_seen_conditions" in eval_tiers:
            try:
                multistep_results = probe_multistep_perturbation(
                    model, tokenizer, eval_tiers["tier1_seen_conditions"],
                    device=args.device, n_probes=args.multistep_probes,
                    max_new_tokens=args.max_new_tokens,
                )
                with open(os.path.join(args.output_dir, "multistep_probe.json"), "w") as f:
                    json.dump(multistep_results, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"  multi-step probe skipped (non-fatal): {e}")
    
    # Summary table
    logger.info(f"\n{'='*92}")
    logger.info(f"SUMMARY  (prediction = {args.baseline if args.baseline != 'none' else 'model generation'})")
    logger.info(f"{'='*92}")
    logger.info(f"{'Tier':<28} {'DE-Δr(top'+str(args.topk_de)+')':>16} {'τ top'+str(args.topn_expressed):>12} "
                f"{'τ panel':>10} {'coverage':>10} {'valid':>8}")
    logger.info("-" * 92)

    def cell(metric_dict, key, fmt="{:.4f}"):
        c = metric_dict["metrics"].get(key, {})
        return fmt.format(c["mean"]) if c.get("mean") is not None else "N/A"

    for tier_name, results in all_results.items():
        agg = results["aggregated"]
        de = cell(agg, "de_delta_pearson")
        tn = cell(agg, "topn_expressed_tau")
        tp = cell(agg, "panel_tau")
        fm = agg.get("format_means", {})
        cov = f"{fm['coverage']:.3f}" if fm.get("coverage") is not None else "N/A"
        val = f"{fm['valid_rate']:.3f}" if fm.get("valid_rate") is not None else "N/A"
        logger.info(f"{tier_name:<28} {de:>16} {tn:>12} {tp:>10} {cov:>10} {val:>8}")

    logger.info("\n  HEADLINE = DE-Δr: Pearson of predicted vs true rank-shift over the top-K")
    logger.info("  differentially-expressed genes. The control baseline scores ~0 here by")
    logger.info("  construction, so DE-Δr IS the perturbation-prediction signal (no subtraction")
    logger.info("  needed). Panel τ is tail-inflated and secondary. For a significance test on")
    logger.info("  the model vs a baseline, run both then: --paired_compare --model_results <dir>")
    logger.info("  --baseline_results <dir>.")
    if args.baseline == "none":
        fm1 = all_results.get("tier1_seen_conditions", {}).get("aggregated", {}).get("format_means", {})
        if fm1.get("coverage") is not None and fm1["coverage"] < 0.9:
            logger.warning(f"  !! tier1 panel coverage is {fm1['coverage']:.3f} (<0.9): the model is "
                           f"emitting malformed sentences; τ/Δr are UNRELIABLE until this is fixed.")

    # --- Provenance manifest (reproducibility) ---
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_path": args.model_path,
        "baseline": args.baseline,
        "decoding": args.decoding,
        "gen_seed": args.gen_seed,
        "gen_samples": args.gen_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_eval": args.max_eval,
        "subsample_seed": args.subsample_seed,
        "topk_de": args.topk_de,
        "topn_expressed": args.topn_expressed,
        "n_boot": args.n_boot,
        "panel_sha256": file_sha256(panel_path),
        "panel_size": len(panel),
        "eval_file_sha256": {
            t: file_sha256(os.path.join(args.eval_dir, fn))
            for t, fn in tier_files.items()
            if os.path.exists(os.path.join(args.eval_dir, fn))
        },
        "linear_model_sha256": file_sha256(lm_path) if os.path.exists(lm_path) else None,
        "n_examples_per_tier": {t: len(ex) for t, ex in eval_tiers.items()},
        "args": vars(args),
    }
    manifest_path = os.path.join(args.output_dir, f"run_manifest{suffix}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"\n  Provenance written to {manifest_path}")

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

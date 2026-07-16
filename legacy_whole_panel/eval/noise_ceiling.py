#!/usr/bin/env python
"""
noise_ceiling.py — replicate-to-replicate ceiling for the perturbation metrics.

Two real treated cells from the same (drug, cell_line, plate, dose) condition do
not agree perfectly (transcriptional bursting, cell-cycle, capture noise). The gap
between a perfect score and this replicate agreement is IRREDUCIBLE — no model can
close it. This script measures it for DE-Δr (K-sweep), topN-τ, and panel-τ, so the
model's 0.72 can be reported as a fraction of ceiling rather than as an absolute number.

Two ceilings are computed (their difference matters):
  * cell_vs_cell      — score one real cell against ANOTHER single real cell.
                        The HARSH ceiling: DE genes are selected from one noisy
                        cell and matched against another noisy cell.
  * cell_vs_pseudobulk — score one real cell against the leave-one-out PSEUDOBULK
                        (mean expression of the other cells in the condition).
                        Closer to what the model is actually scored against (a
                        condition-representative truth), so a higher, fairer ceiling.

Reuses the EXACT sentence builder (tahoe_c2s_preprocess.build_panel_sentence) and
the EXACT metric functions (evaluate_c2s_tahoe), so the ceiling and the model eval
cannot diverge. Controls are harvested from the existing train.jsonl (plate-matched
DMSO), identical to the eval pipeline. Runs on CPU (defq); streams treated shards.

Usage:
  python noise_ceiling.py --data_dir DATA --num_shards 32 --rows_per_shard 400000 \
      --shard_seed 7 --cells_per_condition 6 --max_conditions 500 \
      --max_pairs_per_condition 6 --out DATA/../eval_results/noise_ceiling.json
"""
# --- repo path bootstrap (reorg): make shared/ + sibling pipeline dirs importable ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
for _p in [os.path.join(_ROOT, "shared"), *sorted(glob.glob(os.path.join(_PIPE, "*")))]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, os, logging
from collections import defaultdict

import numpy as np
from datasets import load_dataset

import tahoe_c2s_preprocess as prep
import evaluate_c2s_tahoe as ev   # metric functions (cell_sentence_to_gene_ranks, etc.)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAHOE_REPO = "tahoebio/Tahoe-100M"
MAX_CTRL_PER_KEY = 8
MAX_CTRL_PER_CL = 16


def harvest_controls(train_path):
    """(cell_line_id, plate) -> [control_sentence] and cell_line_id -> [control_sentence]."""
    by_key = defaultdict(list)
    by_cl = defaultdict(list)
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            p = ex["prompt"]; m = ex.get("metadata", {})
            cl, plate = m.get("cell_line_id"), m.get("plate")
            if cl is None or plate is None or "\nControl cell: " not in p:
                continue
            ctrl = p.split("\nControl cell: ", 1)[1].split("\n\nResponse cell:", 1)[0]
            if len(by_key[(cl, plate)]) < MAX_CTRL_PER_KEY:
                by_key[(cl, plate)].append(ctrl)
            if len(by_cl[cl]) < MAX_CTRL_PER_CL:
                by_cl[cl].append(ctrl)
    logger.info(f"  Harvested controls: {len(by_key):,} (cl,plate) keys, {len(by_cl):,} cell lines")
    return by_key, by_cl


def iterate_shards(shard_list, per_shard_cap=None):
    for shard in shard_list:
        url = f"hf://datasets/{TAHOE_REPO}/{shard}"
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        sc = 0
        for row in ds:
            yield row
            sc += 1
            if per_shard_cap and sc >= per_shard_cap:
                break


def pseudobulk_sentence(cells, gene_id_to_symbol, panel, pidx):
    """Leave-the-set pseudobulk: mean expression per gene across `cells`
    (each cell = (genes, exprs)), then build the panel sentence."""
    acc = defaultdict(float)
    for genes, exprs in cells:
        for g, e in zip(genes, exprs):
            acc[g] += float(e)
    n = len(cells)
    gs = list(acc.keys())
    es = [acc[g] / n for g in gs]
    return prep.build_panel_sentence(gs, es, gene_id_to_symbol, panel, pidx,
                                     min_expressed=200, min_panel_expressed=50)


def pair_metrics(pred_sent, truth_sent, ctrl_sent, panel, pidx, de_k_list, topn_list):
    """All ceiling metrics for one (pred, truth, control) triple, using the eval
    module's exact functions. DE genes selected from TRUTH, as in the real eval."""
    worst = len(panel) + 1
    pr = ev.cell_sentence_to_gene_ranks(pred_sent)
    tr = ev.cell_sentence_to_gene_ranks(truth_sent)
    cr = ev.cell_sentence_to_gene_ranks(ctrl_sent)
    out = {}
    de_ranked = ev.select_top_de_genes(tr, cr, panel, max(de_k_list), worst)
    for k in de_k_list:
        dd = ev.delta_correlation(pr, tr, cr, de_ranked[:k], worst)
        out[f"de_delta_pearson_k{k}"] = dd["delta_pearson"]
    for n in topn_list:
        tg = ev.select_top_expressed(tr, panel, n, worst)
        out[f"topn_tau_n{n}"] = ev.compute_rank_correlation(pr, tr, gene_subset=tg)["kendall_tau"]
    out["panel_tau"] = ev.compute_rank_correlation(pr, tr, gene_subset=panel)["kendall_tau"]
    return out


def summarize(vals):
    a = np.array([v for v in vals if v is not None and v == v], dtype=float)
    if a.size == 0:
        return None
    return {"median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "mean": float(a.mean()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--num_shards", type=int, default=32)
    ap.add_argument("--rows_per_shard", type=int, default=400000)
    ap.add_argument("--shard_seed", type=int, default=7)
    ap.add_argument("--cells_per_condition", type=int, default=6)
    ap.add_argument("--max_conditions", type=int, default=500)
    ap.add_argument("--max_pairs_per_condition", type=int, default=6)
    ap.add_argument("--de_k_list", type=str, default="20,50,100,200")
    ap.add_argument("--topn_list", type=str, default="50,100,200")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    de_k_list = [int(x) for x in args.de_k_list.split(",")]
    topn_list = [int(x) for x in args.topn_list.split(",")]
    rng = np.random.RandomState(args.seed)

    panel = json.load(open(os.path.join(args.data_dir, "l1000_panel.json")))
    pidx = {g: i for i, g in enumerate(panel)}
    gene_id_to_symbol, sample_to_conc, drug_info, cvcl_to_name = prep.load_metadata()
    by_key, by_cl = harvest_controls(os.path.join(args.data_dir, "train.jsonl"))

    all_shards = prep.discover_expression_shards()
    treated = prep.select_shards(all_shards, args.num_shards, args.shard_seed)
    logger.info(f"Treated shards: {len(treated)} (seed {args.shard_seed})")

    # Collect up to N treated cells per condition (store raw genes/exprs for pseudobulk).
    cond_cells = defaultdict(list)   # ckey -> [(genes, exprs)]
    cond_meta = {}                   # ckey -> (cl, plate)
    full = 0
    for row in iterate_shards(treated, per_shard_cap=args.rows_per_shard):
        drug = row["drug"]
        if drug in ("DMSO_TF", "DMSO"):
            continue
        cl, plate, sample = row["cell_line_id"], row["plate"], row["sample"]
        ckey = (drug, cl, plate, sample)
        if len(cond_cells[ckey]) >= args.cells_per_condition:
            continue
        cond_cells[ckey].append((row["genes"], row["expressions"]))
        cond_meta[ckey] = (cl, plate)
        # stop once enough conditions have >=2 cells
        if len(cond_cells[ckey]) == 2:
            full += 1
            if full >= args.max_conditions:
                # keep draining current shard row-by-row is fine; just stop early
                break

    usable = {k: v for k, v in cond_cells.items() if len(v) >= 2}
    logger.info(f"Conditions with >=2 replicate cells: {len(usable)}")

    cvc = defaultdict(list)   # cell_vs_cell metric lists
    cvp = defaultdict(list)   # cell_vs_pseudobulk metric lists
    n_cvc = n_cvp = 0
    for ckey, cells in usable.items():
        cl, plate = cond_meta[ckey]
        pool = by_key.get((cl, plate)) or by_cl.get(cl)
        if not pool:
            continue
        ctrl = pool[rng.randint(len(pool))]

        # build panel sentences for each replicate
        sents = []
        for genes, exprs in cells:
            s = prep.build_panel_sentence(genes, exprs, gene_id_to_symbol, panel, pidx,
                                          min_expressed=200, min_panel_expressed=50)
            if s is not None:
                sents.append(s)
        if len(sents) < 2:
            continue

        # cell-vs-cell: sample ordered pairs
        idx = list(range(len(sents)))
        pairs = []
        for _ in range(args.max_pairs_per_condition):
            i, j = rng.choice(idx, size=2, replace=False)
            pairs.append((i, j))
        for i, j in pairs:
            m = pair_metrics(sents[i], sents[j], ctrl, panel, pidx, de_k_list, topn_list)
            for k, v in m.items():
                cvc[k].append(v)
            n_cvc += 1

        # cell-vs-pseudobulk: each cell vs leave-one-out pseudobulk of the rest
        for i in idx:
            rest = [cells[t] for t in idx if t != i]
            if len(rest) < 1:
                continue
            pb = pseudobulk_sentence(rest, gene_id_to_symbol, panel, pidx)
            if pb is None:
                continue
            m = pair_metrics(sents[i], pb, ctrl, panel, pidx, de_k_list, topn_list)
            for k, v in m.items():
                cvp[k].append(v)
            n_cvp += 1

    result = {
        "n_conditions": len(usable),
        "n_cell_vs_cell_pairs": n_cvc,
        "n_cell_vs_pseudobulk_pairs": n_cvp,
        "de_k_list": de_k_list, "topn_list": topn_list,
        "cell_vs_cell": {k: summarize(v) for k, v in cvc.items()},
        "cell_vs_pseudobulk": {k: summarize(v) for k, v in cvp.items()},
    }
    out = args.out or os.path.join(args.data_dir, "noise_ceiling.json")
    # Safety: never write inside the (possibly read-only, precious) dataset dir.
    data_abs = os.path.abspath(args.data_dir)
    out_abs = os.path.abspath(out)
    if out_abs == data_abs or out_abs.startswith(data_abs + os.sep):
        raise SystemExit(
            f"Refusing to write output inside the dataset dir ({data_abs}). "
            f"Pass --out to a path OUTSIDE it (e.g. .../eval_results/noise_ceiling.json).")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nNoise ceiling ({n_cvc} cell-vs-cell, {n_cvp} cell-vs-pseudobulk pairs):")
    for k in [f"de_delta_pearson_k{kk}" for kk in de_k_list] + \
             [f"topn_tau_n{nn}" for nn in topn_list] + ["panel_tau"]:
        cc = result["cell_vs_cell"].get(k); pp = result["cell_vs_pseudobulk"].get(k)
        cc_s = f"{cc['median']:.3f} [{cc['q25']:.3f},{cc['q75']:.3f}]" if cc else "NA"
        pp_s = f"{pp['median']:.3f} [{pp['q25']:.3f},{pp['q75']:.3f}]" if pp else "NA"
        logger.info(f"  {k:24s}  cell-vs-cell {cc_s:24s}  cell-vs-pseudobulk {pp_s}")
    logger.info(f"  -> {out}")


if __name__ == "__main__":
    main()

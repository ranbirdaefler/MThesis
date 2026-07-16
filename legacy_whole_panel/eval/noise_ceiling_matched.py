#!/usr/bin/env python
"""
noise_ceiling_matched.py — PER-TIER, CONDITION-MATCHED replicate noise ceiling.

Unlike noise_ceiling.py (which re-streams arbitrary conditions), this computes the
ceiling directly from the eval tier files, so it is measured on EXACTLY the
conditions the model was evaluated on — same cells, per tier. This removes the
population-mismatch concern: "model vs ceiling" is now on a matched population.

The eval files store cell SENTENCES (ranked gene lists), not raw expression, so:
  * cell_vs_cell      — real treated sentence i scored against real treated
                        sentence j from the same condition (the ceiling most
                        directly comparable to how the model is scored: single-cell
                        truth, DE genes selected from that truth cell).
  * cell_vs_consensus — sentence i scored against the leave-one-out RANK CONSENSUS
                        of the other cells in the condition (mean rank per gene,
                        re-ranked). A denoised, condition-representative truth in
                        rank space (the analogue of pseudobulk for sentences).

Controls are the plate-matched DMSO already in each prompt. Reuses the eval
module's exact metric functions. Runs in seconds (no streaming, no GPU).

Usage:
  python noise_ceiling_matched.py --eval_dir DATA \
      --out /path/OUTSIDE/DATA/noise_ceiling_matched.json
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
import evaluate_c2s_tahoe as ev

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TIERS = ["tier1_seen_conditions", "tier2_unseen_drugs",
         "tier3_unseen_combos", "tier4_dose_interpolation"]


def control_from_prompt(prompt):
    if "\nControl cell: " not in prompt:
        return None
    return prompt.split("\nControl cell: ", 1)[1].split("\n\nResponse cell:", 1)[0]


def cond_key(m):
    return (m.get("drug"), m.get("cell_line_id"), m.get("plate"), m.get("sample"))


def rank_consensus_sentence(sentences, panel, worst):
    """Leave-one-out denoised truth: mean rank per gene across `sentences`, re-ranked."""
    acc = {g: 0.0 for g in panel}
    for s in sentences:
        r = ev.cell_sentence_to_gene_ranks(s)
        for g in panel:
            acc[g] += r.get(g, worst)
    n = len(sentences)
    mean_rank = {g: acc[g] / n for g in panel}
    return " ".join(sorted(panel, key=lambda g: mean_rank[g]))


def pair_metrics(pred_sent, truth_sent, ctrl_sent, panel, de_k_list, topn_list):
    """Same metrics as the real eval; DE genes selected from TRUTH."""
    worst = len(panel) + 1
    pr = ev.cell_sentence_to_gene_ranks(pred_sent)
    tr = ev.cell_sentence_to_gene_ranks(truth_sent)
    cr = ev.cell_sentence_to_gene_ranks(ctrl_sent)
    out = {}
    de_ranked = ev.select_top_de_genes(tr, cr, panel, max(de_k_list), worst)
    for k in de_k_list:
        out[f"de_delta_pearson_k{k}"] = ev.delta_correlation(
            pr, tr, cr, de_ranked[:k], worst)["delta_pearson"]
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
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True,
                    help="Output path OUTSIDE the eval_dir.")
    ap.add_argument("--de_k_list", type=str, default="20,50,100,200")
    ap.add_argument("--topn_list", type=str, default="50,100,200")
    ap.add_argument("--min_cells", type=int, default=2)
    ap.add_argument("--max_pairs_per_condition", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # write guard: never inside eval_dir
    data_abs = os.path.abspath(args.eval_dir)
    out_abs = os.path.abspath(args.out)
    if out_abs == data_abs or out_abs.startswith(data_abs + os.sep):
        raise SystemExit(f"Refusing to write inside eval_dir ({data_abs}). Pass --out elsewhere.")

    de_k_list = [int(x) for x in args.de_k_list.split(",")]
    topn_list = [int(x) for x in args.topn_list.split(",")]
    rng = np.random.RandomState(args.seed)
    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    worst = len(panel) + 1

    result = {"de_k_list": de_k_list, "topn_list": topn_list, "min_cells": args.min_cells,
              "tiers": {}}

    for tier in TIERS:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            continue
        groups = defaultdict(list)   # cond_key -> [(response_sentence, control_sentence)]
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                ctrl = control_from_prompt(ex["prompt"])
                if ctrl is None:
                    continue
                groups[cond_key(ex.get("metadata", {}))].append((ex["response"], ctrl))

        usable = {k: v for k, v in groups.items() if len(v) >= args.min_cells}
        cells_per = [len(v) for v in usable.values()]
        cvc = defaultdict(list)
        cvk = defaultdict(list)
        n_cvc = n_cvk = 0
        for k, cells in usable.items():
            sents = [s for s, _ in cells]
            ctrls = [c for _, c in cells]
            idx = list(range(len(sents)))

            # cell-vs-cell: sampled ordered pairs, control drawn from the condition
            for _ in range(args.max_pairs_per_condition):
                i, j = rng.choice(idx, size=2, replace=False)
                ctrl = ctrls[rng.randint(len(ctrls))]
                for mk, mv in pair_metrics(sents[i], sents[j], ctrl, panel, de_k_list, topn_list).items():
                    cvc[mk].append(mv)
                n_cvc += 1

            # cell-vs-consensus: each cell vs leave-one-out rank consensus of the rest
            for i in idx:
                rest = [sents[t] for t in idx if t != i]
                if not rest:
                    continue
                cons = rank_consensus_sentence(rest, panel, worst)
                ctrl = ctrls[rng.randint(len(ctrls))]
                for mk, mv in pair_metrics(sents[i], cons, ctrl, panel, de_k_list, topn_list).items():
                    cvk[mk].append(mv)
                n_cvk += 1

        result["tiers"][tier] = {
            "n_conditions": len(usable),
            "median_cells_per_condition": float(np.median(cells_per)) if cells_per else 0,
            "max_cells_per_condition": int(max(cells_per)) if cells_per else 0,
            "n_cell_vs_cell_pairs": n_cvc,
            "n_cell_vs_consensus_pairs": n_cvk,
            "cell_vs_cell": {kk: summarize(v) for kk, v in cvc.items()},
            "cell_vs_consensus": {kk: summarize(v) for kk, v in cvk.items()},
        }
        logger.info(f"\n{tier}: {len(usable)} conditions (median {result['tiers'][tier]['median_cells_per_condition']:.0f} "
                    f"cells/cond, max {result['tiers'][tier]['max_cells_per_condition']}), "
                    f"{n_cvc} cvc / {n_cvk} cvk pairs")
        for kk in [f"de_delta_pearson_k{x}" for x in de_k_list] + \
                  [f"topn_tau_n{x}" for x in topn_list] + ["panel_tau"]:
            cc = result["tiers"][tier]["cell_vs_cell"].get(kk)
            kc = result["tiers"][tier]["cell_vs_consensus"].get(kk)
            cs = f"{cc['median']:.3f}[{cc['q25']:.3f},{cc['q75']:.3f}]" if cc else "NA"
            ks = f"{kc['median']:.3f}[{kc['q25']:.3f},{kc['q75']:.3f}]" if kc else "NA"
            logger.info(f"    {kk:22s} cvc {cs:22s} cvk {ks}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

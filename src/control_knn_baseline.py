#!/usr/bin/env python
"""
control_knn_baseline.py
=======================
A control-conditioned k-NN RETRIEVAL baseline for single-cell perturbation prediction.

MOTIVATION
----------
The drug-scramble ablation showed the fine-tuned model largely ignores the drug token:
swapping the drug (even to a different MOA) leaves DE-Δr essentially unchanged. Yet the
model beats the mean-shift baselines by +0.39..+0.59. Where does that margin come from?

Hypothesis: the model reads the CONTROL cell (the untreated baseline in the prompt) and
predicts, in effect, "cells whose baseline looks like this one respond THIS way." If so, a
non-parametric baseline that ONLY does control-similarity retrieval — with no drug modeling
whatsoever — should match the model. This script is that baseline: the honest measuring
stick for "does the model add anything beyond control-based retrieval?"

WHAT IT DOES (per test cell)
----------------------------
1. Represent the test cell's CONTROL sentence as a rank vector over the fixed panel.
2. Retrieve the k TRAINING cells whose CONTROL rank vector is most similar (Spearman by
   default), restricted to the SAME CELL LINE (primary) — with strict leakage guards.
3. Predict the rank-CONSENSUS of those k neighbours' TREATED sentences (mean rank per gene,
   re-ranked) — exactly the rank-consensus used in the noise-ceiling analysis.
4. Score that predicted sentence through the IDENTICAL harness metric functions
   (DE-Δr K-sweep, topN-τ, panel-τ), against the real treated truth using the real matched
   control as reference. No metric corruption: the control is always the test cell's real one.

The prediction uses ONLY: the test cell's control + the training cells' (control, treated)
pairs. No drug identity, no MOA, no learned parameters. Pure control-based retrieval.

FAIRNESS / RIGOUR
-----------------
- Reuses evaluate_c2s_tahoe's EXACT metric functions (no metric divergence).
- Leakage guards (see build_neighbor_index / retrieve): a test cell can never retrieve
  itself, a replicate of its own (drug, cell_line, plate, dose) condition, or — critically —
  any training cell whose DRUG equals the test cell's drug (so seen-condition tiers can't
  cheat by copying same-drug responses; the baseline is drug-agnostic BY CONSTRUCTION).
- Same-cell-line retrieval is the PRIMARY, fairest, hardest-to-beat setting. Global optional.
- k is SWEPT (1,5,20,100). Sanity check: as k grows the baseline should converge toward the
  per-cell-line mean-shift baseline (retrieval → cell-line average).
- Drug-clustered bootstrap CIs (cluster by test-cell drug), identical to the harness.
- Deterministic (seeded subsample matches the model eval's --subsample_seed).

USAGE
-----
  python control_knn_baseline.py \
      --eval_dir DATA --train_file DATA/train.jsonl \
      --out RESULTS/control_knn.json \
      --k_list 1,5,20,100 --scope cellline --sim spearman \
      --max_eval 300 --subsample_seed 42 --n_boot 1000

Then compare to the model with the harness's paired step (or paired_by_position.py),
metric de_delta_pearson, to ask: does the model beat control-based retrieval?
"""
import argparse, json, os, logging
from collections import defaultdict

import numpy as np

import evaluate_c2s_tahoe as ev  # EXACT metric functions — single source of truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TIERS = ["tier1_seen_conditions", "tier2_unseen_drugs",
         "tier3_unseen_combos", "tier4_dose_interpolation"]


# ---------------------------------------------------------------------------
# Representation helpers
# ---------------------------------------------------------------------------
def sentence_to_rankvec(sentence, panel_index, worst):
    """Dense rank vector over the fixed panel: panel position -> rank of that gene in the
    sentence (missing genes -> worst rank). Deterministic, same convention as the metric."""
    ranks = ev.cell_sentence_to_gene_ranks(sentence)  # gene -> rank
    v = np.full(len(panel_index), worst, dtype=np.float64)
    for g, i in panel_index.items():
        r = ranks.get(g)
        if r is not None:
            v[i] = r
    return v


def cond_key(meta):
    return (meta.get("drug"), meta.get("cell_line_id"), meta.get("plate"), meta.get("sample"))


# ---------------------------------------------------------------------------
# Training index
# ---------------------------------------------------------------------------
def build_train_index(train_file, panel, panel_index, worst, limit=None):
    """Load training cells: control rank vector (for retrieval), treated sentence (for the
    prediction), and metadata (for leakage guards + cell-line scoping)."""
    ctrl_vecs, treated_sents, metas = [], [], []
    n = 0
    with open(train_file) as f:
        for line in f:
            if limit and n >= limit:
                break
            ex = json.loads(line)
            ctrl = ev.control_from_prompt(ex["prompt"])
            if not ctrl:
                continue
            ctrl_vecs.append(sentence_to_rankvec(ctrl, panel_index, worst))
            treated_sents.append(ex["response"])
            metas.append(ex.get("metadata", {}))
            n += 1
    logger.info(f"  Indexed {len(ctrl_vecs):,} training cells")
    ctrl_mat = np.vstack(ctrl_vecs)  # (N_train, P)
    # group row-indices by cell line for scoped retrieval
    by_cl = defaultdict(list)
    for i, m in enumerate(metas):
        by_cl[m.get("cell_line_id")].append(i)
    return {"ctrl_mat": ctrl_mat, "treated": treated_sents, "metas": metas,
            "by_cl": {k: np.array(v) for k, v in by_cl.items()}}


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def _rankdata_rows(mat):
    """Row-wise rank transform for Spearman (ties -> average ranks)."""
    order = mat.argsort(axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    r = np.arange(mat.shape[1])
    for i in range(mat.shape[0]):
        ranks[i, order[i]] = r
    return ranks


def similarity(query_vec, cand_mat, mode):
    """Similarity of a single query control vector to each candidate row. Higher = closer."""
    if mode == "spearman":
        q = query_vec.argsort().argsort().astype(np.float64)  # rank of query
        C = _rankdata_rows(cand_mat)
        qz = q - q.mean()
        Cz = C - C.mean(axis=1, keepdims=True)
        num = Cz @ qz
        den = (np.sqrt((Cz ** 2).sum(axis=1)) * np.sqrt((qz ** 2).sum()) + 1e-12)
        return num / den
    elif mode == "euclidean":
        d = np.sqrt(((cand_mat - query_vec[None, :]) ** 2).sum(axis=1))
        return -d  # higher = closer
    else:
        raise ValueError(f"unknown sim mode {mode}")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def rank_consensus(sentences, panel, worst):
    """Denoised consensus of treated sentences: mean rank per gene, re-ranked.
    Same construction as the noise-ceiling consensus."""
    acc = {g: 0.0 for g in panel}
    for s in sentences:
        r = ev.cell_sentence_to_gene_ranks(s)
        for g in panel:
            acc[g] += r.get(g, worst)
    n = len(sentences)
    mean_rank = {g: acc[g] / n for g in panel}
    return " ".join(sorted(panel, key=lambda g: mean_rank[g]))


def predict_knn(test_ex, idx, panel, panel_index, worst, k, scope, sim_mode):
    """Predict the test cell's treated sentence via control-kNN retrieval.
    Returns (pred_sentence, n_used) or (None, 0) if no valid neighbours."""
    meta = test_ex.get("metadata", {})
    tkey = cond_key(meta)
    tdrug = meta.get("drug")
    tcl = meta.get("cell_line_id")

    ctrl = ev.control_from_prompt(test_ex["prompt"])
    if not ctrl:
        return None, 0
    qvec = sentence_to_rankvec(ctrl, panel_index, worst)

    # candidate pool
    if scope == "cellline":
        cand = idx["by_cl"].get(tcl)
        if cand is None or len(cand) == 0:
            return None, 0
    else:  # global
        cand = np.arange(idx["ctrl_mat"].shape[0])

    # LEAKAGE GUARDS: drop neighbours that are (a) the same exact condition (self/replicate)
    # or (b) the SAME DRUG as the test cell. (b) is the crucial one: it forces the baseline
    # to be drug-agnostic — it can never predict by copying a same-drug response.
    metas = idx["metas"]
    keep = []
    for j in cand:
        mj = metas[j]
        if cond_key(mj) == tkey:
            continue
        if mj.get("drug") == tdrug:
            continue
        keep.append(j)
    if not keep:
        return None, 0
    keep = np.array(keep)

    sims = similarity(qvec, idx["ctrl_mat"][keep], sim_mode)
    kk = min(k, len(keep))
    top = keep[np.argpartition(-sims, kk - 1)[:kk]]
    neigh_sents = [idx["treated"][j] for j in top]
    return rank_consensus(neigh_sents, panel, worst), len(neigh_sents)


# ---------------------------------------------------------------------------
# Scoring (reuses harness metric functions exactly)
# ---------------------------------------------------------------------------
def score_example(pred_sentence, test_ex, panel, panel_index, linear_model, worst,
                  de_k_list, topn):
    """Per-example metrics, identical to the harness (DE genes from truth; real control)."""
    ctrl = ev.control_from_prompt(test_ex["prompt"])
    true_sent = test_ex["response"]
    true_ranks = ev.cell_sentence_to_gene_ranks(true_sent)
    control_ranks = ev.cell_sentence_to_gene_ranks(ctrl)
    de_ranked = ev.select_top_de_genes(true_ranks, control_ranks, panel, max(de_k_list), worst)
    de_by_k = {kk: de_ranked[:kk] for kk in de_k_list}
    topn_genes = ev.select_top_expressed(true_ranks, panel, topn, worst)
    m = ev.compute_scalar_metrics(pred_sentence, true_sent, true_ranks, control_ranks,
                                  panel, panel_index, linear_model, worst,
                                  de_by_k, topn_genes, headline_k=de_k_list[0]
                                  if 50 not in de_k_list else 50)
    return m


def summarize(vals, drugs, n_boot, seed):
    v = [(x, d) for x, d in zip(vals, drugs) if x is not None and x == x]
    if not v:
        return None
    xs = [x for x, _ in v]; gs = [d for _, d in v]
    ci = ev.cluster_bootstrap_ci(xs, gs, n_boot=n_boot, seed=seed)
    return ci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out", required=True, help="Output JSON path (OUTSIDE eval_dir).")
    ap.add_argument("--k_list", type=str, default="1,5,20,100")
    ap.add_argument("--scope", choices=["cellline", "global"], default="cellline")
    ap.add_argument("--sim", choices=["spearman", "euclidean"], default="spearman")
    ap.add_argument("--de_k_list", type=str, default="20,50,100,200")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--max_eval", type=int, default=300)
    ap.add_argument("--subsample_seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--train_limit", type=int, default=None,
                    help="Cap #training cells indexed (speed); default all.")
    ap.add_argument("--save_per_example", action="store_true",
                    help="Also write per-example predictions per (tier,k) for paired tests.")
    args = ap.parse_args()

    # write guard
    if os.path.abspath(args.out).startswith(os.path.abspath(args.eval_dir) + os.sep):
        raise SystemExit("Refusing to write inside eval_dir; choose --out elsewhere.")

    k_list = [int(x) for x in args.k_list.split(",")]
    de_k_list = [int(x) for x in args.de_k_list.split(",")]

    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    worst = len(panel) + 1
    lm_path = os.path.join(args.eval_dir, "linear_model.json")
    linear_model = json.load(open(lm_path)) if os.path.exists(lm_path) else None

    logger.info("Building training control index ...")
    idx = build_train_index(args.train_file, panel, panel_index, worst, limit=args.train_limit)

    result = {"scope": args.scope, "sim": args.sim, "k_list": k_list,
              "de_k_list": de_k_list, "headline_metric": "de_delta_pearson",
              "tiers": {}}
    per_example_dump = {}

    for tier in TIERS:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            continue
        examples = [json.loads(l) for l in open(path)]
        # deterministic subsample, matching the model eval
        rng = np.random.RandomState(args.subsample_seed)
        if args.max_eval and len(examples) > args.max_eval:
            sel = rng.choice(len(examples), size=args.max_eval, replace=False)
            examples = [examples[i] for i in sorted(sel)]

        tier_out = {}
        for k in k_list:
            de_vals = defaultdict(list)   # metric_key -> [per-example]
            drugs, n_used_list = [], []
            pe = []
            for ex in examples:
                pred, n_used = predict_knn(ex, idx, panel, panel_index, worst,
                                           k, args.scope, args.sim)
                drug = ex.get("metadata", {}).get("drug")
                if pred is None:
                    continue
                m = score_example(pred, ex, panel, panel_index, linear_model, worst,
                                  de_k_list, args.topn)
                for mk in ["de_delta_pearson", "topn_expressed_tau", "panel_tau"] + \
                          [f"de_delta_pearson_k{kk}" for kk in de_k_list]:
                    de_vals[mk].append(m.get(mk))
                drugs.append(drug); n_used_list.append(n_used)
                if args.save_per_example:
                    pe.append({"example_id": ex.get("example_id"), "drug": drug,
                               "cell_line_name": ex.get("metadata", {}).get("cell_line_name"),
                               "metrics": {mk: m.get(mk) for mk in
                                           ["de_delta_pearson"] +
                                           [f"de_delta_pearson_k{kk}" for kk in de_k_list]}})
            agg = {mk: summarize(de_vals[mk], drugs, args.n_boot, args.subsample_seed)
                   for mk in de_vals}
            tier_out[f"k{k}"] = {
                "n_scored": len(drugs),
                "mean_neighbors_used": float(np.mean(n_used_list)) if n_used_list else 0,
                "metrics": agg,
            }
            hk = agg.get("de_delta_pearson")
            logger.info(f"  {tier:26s} k={k:<4d} n={len(drugs):3d} "
                        f"DEdr={hk['mean']:.3f} [{hk['ci_low']:.3f},{hk['ci_high']:.3f}]"
                        if hk else f"  {tier:26s} k={k:<4d} no valid predictions")
            if args.save_per_example:
                per_example_dump[f"{tier}_k{k}"] = pe
        result["tiers"][tier] = tier_out

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n-> {args.out}")

    if args.save_per_example:
        pedir = os.path.splitext(args.out)[0] + "_per_example"
        os.makedirs(pedir, exist_ok=True)
        for name, arr in per_example_dump.items():
            with open(os.path.join(pedir, name + ".json"), "w") as f:
                json.dump(arr, f)
        logger.info(f"-> per-example predictions in {pedir}/")


if __name__ == "__main__":
    main()

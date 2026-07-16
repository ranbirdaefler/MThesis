#!/usr/bin/env python
"""
control_state_baseline.py
=========================
Two things, both aimed at making the "does the model beat control-based prediction?"
comparison rigorous:

(A) control-state-conditioned MEAN-SHIFT baseline  [the primary baseline]
    Predict  test_control + mean_rank_shift[cell_line, control_state_cluster].
    - Uses the real control + cell-line identity + a training-estimated average
      perturbation direction conditioned on the control's baseline STATE.
    - NO drug, NO retrieval-of-real-answers.
    - Output form is "control + a shift" — exactly like a generative model would
      produce — so the comparison to the model is apples-to-apples and, crucially,
      FREE of the denoising/averaging artifact that inflates the kNN baseline.
    This tests whether the model adds anything beyond applying the average shift for
    cells in this baseline state.

(B) kNN metric-validity DIAGNOSTIC  [is the kNN 0.91 score valid?]
    The kNN baseline predicts a rank-CONSENSUS of k real treated cells. That is an
    AVERAGE of real answers, which (i) denoises (variance reduction rewarded by a
    correlation metric) and (ii) is "made of real treated cells." Neither is available
    to a single-sample generative model, so kNN's high score may be partly artifact.
    This diagnostic decomposes it by re-scoring kNN predictions built from DIFFERENT
    numbers of neighbours AND by a "single random real neighbour" control, so we can
    separate:
        - real retrieval signal (similar controls -> similar responses), vs
        - the denoising-by-averaging artifact (score rises with k regardless).
    If the score climbs steeply with k and a single-random-neighbour scores like k=1,
    the high number is largely averaging, not retrieval signal.

Both reuse evaluate_c2s_tahoe's EXACT metric functions. CPU only.

USAGE
-----
  python control_state_baseline.py --eval_dir DATA --train_file DATA/train.jsonl \
      --out RESULTS/control_state.json \
      --n_state_clusters 8 --de_k_list 20,50,100,200 --topn 100 \
      --max_eval 300 --subsample_seed 42 --n_boot 1000 --save_per_example
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


def sentence_to_rankvec(sentence, panel_index, worst):
    ranks = ev.cell_sentence_to_gene_ranks(sentence)
    v = np.full(len(panel_index), worst, dtype=np.float64)
    for g, i in panel_index.items():
        r = ranks.get(g)
        if r is not None:
            v[i] = r
    return v


def cond_key(m):
    return (m.get("drug"), m.get("cell_line_id"), m.get("plate"), m.get("sample"))


def apply_shift_to_control(control_sentence, shift_vec, panel, panel_index, worst):
    """Predict a treated sentence = control ranks + shift, re-ranked. shift_vec is a
    dense per-panel-gene mean rank shift. Output is ONE cell's worth of structure
    (control nudged by the shift), matching a generative model's output form."""
    cr = ev.cell_sentence_to_gene_ranks(control_sentence)
    scored = []
    for g in panel:
        base = cr.get(g, worst)
        scored.append((g, base + shift_vec[panel_index[g]]))
    scored.sort(key=lambda x: x[1])
    return " ".join(g for g, _ in scored)


# ---------------------------------------------------------------------------
# (A) control-state-conditioned mean-shift
# ---------------------------------------------------------------------------
def build_state_conditioned_shifts(train_file, panel, panel_index, worst,
                                   n_clusters, limit=None, seed=42):
    """For each cell line, cluster training controls into n_clusters baseline-state groups
    (k-means on control rank vectors), and compute the mean (treated-control) rank shift
    per (cell_line, cluster). Returns structures needed to assign a test cell to a cluster
    and fetch the corresponding shift. Also returns per-cell-line and global fallbacks."""
    P = len(panel)
    # gather per-cell-line control vectors, shift vectors, and drugs (for leakage guard)
    cl_ctrl = defaultdict(list)   # cell_line -> list of control vecs
    cl_shift = defaultdict(list)  # cell_line -> list of shift vecs (treated-control)
    cl_drug = defaultdict(list)
    global_shift_sum = np.zeros(P); global_n = 0
    n = 0
    with open(train_file) as f:
        for line in f:
            if limit and n >= limit:
                break
            ex = json.loads(line); m = ex.get("metadata", {})
            cr = sentence_to_rankvec(ev.control_from_prompt(ex["prompt"]), panel_index, worst)
            trk = ev.cell_sentence_to_gene_ranks(ex["response"])
            tv = np.array([trk.get(g, worst) for g in panel], dtype=np.float64)
            shift = tv - cr
            cl = m.get("cell_line_id")
            cl_ctrl[cl].append(cr); cl_shift[cl].append(shift); cl_drug[cl].append(m.get("drug"))
            global_shift_sum += shift; global_n += 1
            n += 1
    logger.info(f"  Indexed {n:,} training cells across {len(cl_ctrl)} cell lines")
    global_shift = global_shift_sum / max(global_n, 1)

    rng = np.random.RandomState(seed)
    model = {}  # cell_line -> dict(centroids, cluster_shift, cluster_drugsets, cl_mean_shift)
    for cl, ctrls in cl_ctrl.items():
        C = np.vstack(ctrls); S = np.vstack(cl_shift[cl]); drugs = cl_drug[cl]
        cl_mean_shift = S.mean(axis=0)
        k = min(n_clusters, len(C))
        if k < 2:
            model[cl] = dict(centroids=C.mean(axis=0, keepdims=True),
                             cluster_shift=[cl_mean_shift],
                             cluster_drugs=[set(drugs)], cl_mean_shift=cl_mean_shift)
            continue
        centroids, labels = _kmeans(C, k, rng)
        cluster_shift, cluster_drugs = [], []
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                cluster_shift.append(cl_mean_shift); cluster_drugs.append(set())
            else:
                cluster_shift.append(S[mask].mean(axis=0))
                cluster_drugs.append(set(d for d, mm in zip(drugs, mask) if mm))
        model[cl] = dict(centroids=centroids, cluster_shift=cluster_shift,
                         cluster_drugs=cluster_drugs, cl_mean_shift=cl_mean_shift)
    return {"per_cl": model, "global_shift": global_shift}


def _kmeans(X, k, rng, iters=25):
    idx = rng.choice(len(X), size=k, replace=False)
    cent = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.sum() > 0:
                cent[c] = X[m].mean(axis=0)
    return cent, labels


def predict_state_shift(test_ex, shifts, panel, panel_index, worst, leak_guard=True):
    """Assign test control to its cell-line cluster (excluding same-drug clusters if the
    guard makes a cluster ill-defined) and predict control + that cluster's mean shift."""
    m = test_ex.get("metadata", {})
    cl = m.get("cell_line_id"); tdrug = m.get("drug")
    ctrl_sent = ev.control_from_prompt(test_ex["prompt"])
    if not ctrl_sent:
        return None
    mdl = shifts["per_cl"].get(cl)
    if mdl is None:
        shift = shifts["global_shift"]
        return apply_shift_to_control(ctrl_sent, shift, panel, panel_index, worst)
    qv = sentence_to_rankvec(ctrl_sent, panel_index, worst)
    d = ((mdl["centroids"] - qv[None, :]) ** 2).sum(axis=1)
    c = int(d.argmin())
    shift = mdl["cluster_shift"][c]
    # leakage note: the shift is an AVERAGE DIRECTION over many cells/drugs, not a copied
    # response; same-drug contamination of a *direction* is negligible, but if a cluster is
    # dominated by the test's own drug we fall back to the cell-line mean shift.
    if leak_guard and mdl["cluster_drugs"][c] and mdl["cluster_drugs"][c] <= {tdrug}:
        shift = mdl["cl_mean_shift"]
    return apply_shift_to_control(ctrl_sent, shift, panel, panel_index, worst)


# ---------------------------------------------------------------------------
# (B) kNN diagnostic — separate retrieval signal from denoising artifact
# ---------------------------------------------------------------------------
def build_knn_index(train_file, panel, panel_index, worst, limit=None):
    vecs, treated, metas = [], [], []
    n = 0
    with open(train_file) as f:
        for line in f:
            if limit and n >= limit:
                break
            ex = json.loads(line)
            vecs.append(sentence_to_rankvec(ev.control_from_prompt(ex["prompt"]), panel_index, worst))
            treated.append(ex["response"]); metas.append(ex.get("metadata", {}))
            n += 1
    by_cl = defaultdict(list)
    for i, m in enumerate(metas):
        by_cl[m.get("cell_line_id")].append(i)
    return {"mat": np.vstack(vecs), "treated": treated, "metas": metas,
            "by_cl": {k: np.array(v) for k, v in by_cl.items()}}


def rank_consensus(sentences, panel, worst):
    acc = {g: 0.0 for g in panel}
    for s in sentences:
        r = ev.cell_sentence_to_gene_ranks(s)
        for g in panel:
            acc[g] += r.get(g, worst)
    n = len(sentences)
    mean_rank = {g: acc[g] / n for g in panel}
    return " ".join(sorted(panel, key=lambda g: mean_rank[g]))


def knn_neighbors(test_ex, idx, panel_index, worst):
    m = test_ex.get("metadata", {})
    tkey = cond_key(m); tdrug = m.get("drug"); cl = m.get("cell_line_id")
    qv = sentence_to_rankvec(ev.control_from_prompt(test_ex["prompt"]), panel_index, worst)
    cand = idx["by_cl"].get(cl)
    if cand is None:
        return None, None
    keep = [j for j in cand if cond_key(idx["metas"][j]) != tkey and idx["metas"][j].get("drug") != tdrug]
    if not keep:
        return None, None
    keep = np.array(keep)
    # euclidean on rank vecs (order-equivalent to the spearman retrieval for ranking here)
    d = ((idx["mat"][keep] - qv[None, :]) ** 2).sum(axis=1)
    order = keep[np.argsort(d)]
    return order, qv


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(pred_sentence, test_ex, panel, panel_index, linear_model, worst, de_k_list, topn):
    ctrl = ev.control_from_prompt(test_ex["prompt"]); true_sent = test_ex["response"]
    true_ranks = ev.cell_sentence_to_gene_ranks(true_sent)
    control_ranks = ev.cell_sentence_to_gene_ranks(ctrl)
    de_ranked = ev.select_top_de_genes(true_ranks, control_ranks, panel, max(de_k_list), worst)
    de_by_k = {kk: de_ranked[:kk] for kk in de_k_list}
    topn_genes = ev.select_top_expressed(true_ranks, panel, topn, worst)
    hk = 50 if 50 in de_k_list else de_k_list[0]
    return ev.compute_scalar_metrics(pred_sentence, true_sent, true_ranks, control_ranks,
                                     panel, panel_index, linear_model, worst,
                                     de_by_k, topn_genes, headline_k=hk)


def agg(vals, drugs, n_boot, seed):
    v = [(x, d) for x, d in zip(vals, drugs) if x is not None and x == x]
    if not v:
        return None
    return ev.cluster_bootstrap_ci([x for x, _ in v], [d for _, d in v], n_boot=n_boot, seed=seed)


def subsample(examples, max_eval, seed):
    if max_eval and len(examples) > max_eval:
        rng = np.random.RandomState(seed)
        sel = sorted(rng.choice(len(examples), size=max_eval, replace=False))
        return [examples[i] for i in sel]
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_state_clusters", type=int, default=8)
    ap.add_argument("--knn_diag_k", type=str, default="1,2,5,20,100",
                    help="k values for the kNN denoising diagnostic")
    ap.add_argument("--de_k_list", type=str, default="20,50,100,200")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--max_eval", type=int, default=300)
    ap.add_argument("--subsample_seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--skip_knn_diag", action="store_true")
    ap.add_argument("--save_per_example", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.out).startswith(os.path.abspath(args.eval_dir) + os.sep):
        raise SystemExit("Refusing to write inside eval_dir.")

    de_k_list = [int(x) for x in args.de_k_list.split(",")]
    diag_k = [int(x) for x in args.knn_diag_k.split(",")]
    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    worst = len(panel) + 1
    lm_path = os.path.join(args.eval_dir, "linear_model.json")
    linear_model = json.load(open(lm_path)) if os.path.exists(lm_path) else None

    logger.info("(A) Building control-state-conditioned shifts ...")
    shifts = build_state_conditioned_shifts(args.train_file, panel, panel_index, worst,
                                            args.n_state_clusters, limit=args.train_limit,
                                            seed=args.subsample_seed)
    knn_idx = None
    if not args.skip_knn_diag:
        logger.info("(B) Building kNN index for the denoising diagnostic ...")
        knn_idx = build_knn_index(args.train_file, panel, panel_index, worst, limit=args.train_limit)

    result = {"n_state_clusters": args.n_state_clusters, "de_k_list": de_k_list,
              "headline_metric": "de_delta_pearson", "tiers": {}}
    pe_dump = {}

    for tier in TIERS:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            continue
        examples = subsample([json.loads(l) for l in open(path)], args.max_eval, args.subsample_seed)
        tout = {}

        # (A) control-state-conditioned mean-shift
        vals = defaultdict(list); drugs = []; pe = []
        for ex in examples:
            pred = predict_state_shift(ex, shifts, panel, panel_index, worst)
            if pred is None:
                continue
            m = score(pred, ex, panel, panel_index, linear_model, worst, de_k_list, args.topn)
            for mk in ["de_delta_pearson", "topn_expressed_tau", "panel_tau"] + \
                      [f"de_delta_pearson_k{kk}" for kk in de_k_list]:
                vals[mk].append(m.get(mk))
            drugs.append(ex.get("metadata", {}).get("drug"))
            if args.save_per_example:
                pe.append({"example_id": ex.get("example_id"),
                           "drug": ex.get("metadata", {}).get("drug"),
                           "cell_line_name": ex.get("metadata", {}).get("cell_line_name"),
                           "metrics": {"de_delta_pearson": m.get("de_delta_pearson")}})
        tout["control_state_meanshift"] = {
            "n_scored": len(drugs),
            "metrics": {mk: agg(vals[mk], drugs, args.n_boot, args.subsample_seed) for mk in vals},
        }
        hk = tout["control_state_meanshift"]["metrics"].get("de_delta_pearson")
        logger.info(f"  {tier:26s} STATE-SHIFT  DEdr={hk['mean']:.3f} [{hk['ci_low']:.3f},{hk['ci_high']:.3f}] n={len(drugs)}"
                    if hk else f"  {tier:26s} STATE-SHIFT no predictions")
        if args.save_per_example:
            pe_dump[f"{tier}_control_state"] = pe

        # (B) kNN denoising diagnostic
        if knn_idx is not None:
            diag = {}
            # precompute neighbour orders once per example
            orders = {}
            for i, ex in enumerate(examples):
                o, _ = knn_neighbors(ex, knn_idx, panel_index, worst)
                orders[i] = o
            # (b1) consensus of top-k (the real kNN prediction) at each k
            for k in diag_k:
                vv = []; dd = []
                for i, ex in enumerate(examples):
                    o = orders[i]
                    if o is None or len(o) == 0:
                        continue
                    kk = min(k, len(o))
                    pred = rank_consensus([knn_idx["treated"][j] for j in o[:kk]], panel, worst)
                    m = score(pred, ex, panel, panel_index, linear_model, worst, de_k_list, args.topn)
                    vv.append(m.get("de_delta_pearson")); dd.append(ex.get("metadata", {}).get("drug"))
                diag[f"consensus_topk_{k}"] = agg(vv, dd, args.n_boot, args.subsample_seed)
            # (b2) CONTROL: single RANDOM real neighbour (same pool) — no similarity, no averaging.
            # If this scores like consensus_topk_1, the k=1 score is "a real cell", not retrieval skill.
            rng = np.random.RandomState(args.subsample_seed)
            vv = []; dd = []
            for i, ex in enumerate(examples):
                o = orders[i]
                if o is None or len(o) == 0:
                    continue
                j = o[rng.randint(len(o))]
                m = score(knn_idx["treated"][j], ex, panel, panel_index, linear_model, worst, de_k_list, args.topn)
                vv.append(m.get("de_delta_pearson")); dd.append(ex.get("metadata", {}).get("drug"))
            diag["random_single_neighbor"] = agg(vv, dd, args.n_boot, args.subsample_seed)
            # (b3) CONTROL: nearest single neighbour (k=1 retrieval, no averaging) is consensus_topk_1
            tout["knn_diagnostic"] = diag
            def s(x): return f"{x['mean']:.3f}" if x else "NA"
            logger.info(f"  {tier:26s} kNN-diag: " +
                        " ".join(f"k{k}={s(diag.get(f'consensus_topk_{k}'))}" for k in diag_k) +
                        f"  rand1={s(diag.get('random_single_neighbor'))}")
        result["tiers"][tier] = tout

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n-> {args.out}")
    if args.save_per_example:
        pedir = os.path.splitext(args.out)[0] + "_per_example"
        os.makedirs(pedir, exist_ok=True)
        for name, arr in pe_dump.items():
            json.dump(arr, open(os.path.join(pedir, name + ".json"), "w"))
        logger.info(f"-> per-example in {pedir}/")


if __name__ == "__main__":
    main()

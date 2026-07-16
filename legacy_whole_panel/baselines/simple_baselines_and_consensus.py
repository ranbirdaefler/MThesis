#!/usr/bin/env python
"""
simple_baselines_and_consensus.py
=================================
Two experiments in one file (choose with --mode), both scored through the EXACT harness
metric functions (evaluate_c2s_tahoe) so numbers are directly comparable to the model and
the other baselines.

--------------------------------------------------------------------------------------------
MODE 1 (--mode linear):  a simple PARAMETRIC learned baseline  [CPU, minutes]
--------------------------------------------------------------------------------------------
Ridge regression that predicts the per-gene rank SHIFT (treated_rank - control_rank) from the
control rank vector (+ optional cell-line / MOA one-hots). Prediction = control + predicted
shift, re-ranked — same "control + a shift" output form as the model and the mean-shift ladder.

Why: the mean-shift ladder is parametric-but-trivial (group averages); kNN is non-parametric.
A ridge/linear model fills the "simple PARAMETRIC learned" rung. If it reaches ~0.7, the task
is largely linear in the control and the 1B model's capacity is not buying much. If it's well
below kNN, the control->treated map is locally smooth (kNN-friendly) but not globally linear.
Leakage-safe: fit on train only; cell-line/MOA encoders fit on train; unseen keys -> zero vec.

--------------------------------------------------------------------------------------------
MODE 2 (--mode consensus):  model-vs-kNN comparison at matched aggregation  [GPU, ~k x eval cost]
--------------------------------------------------------------------------------------------
The kNN baseline predicts the rank-CONSENSUS of k real cells (an estimate of the conditional
MEAN). The model as evaluated emits ONE sample. A correlation metric vs one noisy true cell
rewards the mean over a single sample purely by variance reduction, so kNN's edge is partly
sample-vs-mean, not "better prediction".

This mode removes that asymmetry: for each test cell, sample the MODEL k times (temperature>0),
average the k generated rank vectors into ONE consensus sentence (identical operation to kNN),
and score that consensus once. Now it's mean-vs-mean.

  * IMPORTANT: this is NOT the harness's --gen_samples path. That averages the k draws'
    per-draw METRICS (expected score of a single sample) and stays ~0.72. Here we average the
    k draws' RANKS into one prediction and score once. Different object.

Read-out:
  - consensus jumps 0.72 -> ~0.85+  => most of the kNN gap was sample-vs-mean; the model is a
    competitive conditional-mean estimator (and, unlike kNN, can later take drug knowledge).
  - consensus stays ~0.72          => averaging doesn't help => the model's samples share a
    systematic bias (correlated errors don't cancel); kNN genuinely wins.
  - Also reports per-k (1,5,20) so you see the SAME variance-reduction curve kNN showed, or its
    absence — a direct like-for-like against the kNN diagnostic.

USAGE
-----
  # linear (CPU):
  python simple_baselines_and_consensus.py --mode linear \
      --eval_dir DATA --train_file DATA/train.jsonl --out RESULTS/linear_baseline.json \
      --features control,cellline,moa --de_k_list 20,50,100,200 --topn 100 \
      --max_eval 300 --subsample_seed 42 --n_boot 1000 --train_limit 60000

  # consensus (GPU):
  python simple_baselines_and_consensus.py --mode consensus \
      --eval_dir DATA --model_path CKPT/checkpoint-10000 --out RESULTS/model_consensus.json \
      --consensus_k 20 --consensus_k_report 1,5,20 --temperature 0.9 --top_p 0.9 \
      --de_k_list 20,50,100,200 --topn 100 --max_eval 300 --subsample_seed 42 \
      --gen_batch_size 48 --max_new_tokens 3800 --n_boot 1000
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


# --------------------------------------------------------------------------- shared helpers
def rankvec(sentence, panel, worst):
    r = ev.cell_sentence_to_gene_ranks(sentence)
    return np.array([r.get(g, worst) for g in panel], dtype=np.float64)


def ranks_to_sentence(rank_arr, panel):
    """Given a per-panel-gene score vector (lower = more expressed), produce a sentence."""
    order = np.argsort(rank_arr, kind="stable")
    return " ".join(panel[i] for i in order)


def consensus_from_rank_matrix(rank_mat, panel):
    """Average rank vectors (n x P) into one consensus sentence (re-rank by mean rank)."""
    return ranks_to_sentence(rank_mat.mean(axis=0), panel)


def score_pred(pred_sentence, ex, panel, panel_index, linear_model, worst, de_k_list, topn):
    ctrl = ev.control_from_prompt(ex["prompt"]); true_sent = ex["response"]
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


def load_common(eval_dir):
    panel = json.load(open(os.path.join(eval_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    worst = len(panel) + 1
    lm_path = os.path.join(eval_dir, "linear_model.json")
    linear_model = json.load(open(lm_path)) if os.path.exists(lm_path) else None
    return panel, panel_index, worst, linear_model


# =========================================================================== MODE 1: linear
def onehot_fitter(values):
    uniq = sorted(set(v for v in values if v is not None))
    idx = {v: i for i, v in enumerate(uniq)}
    def enc(v):
        z = np.zeros(len(uniq), dtype=np.float64)
        if v in idx:
            z[idx[v]] = 1.0
        return z
    return enc, len(uniq)


def run_linear(args, panel, panel_index, worst, linear_model):
    from numpy.linalg import solve
    P = len(panel)
    feats = args.features.split(",")
    use_cl = "cellline" in feats
    use_moa = "moa" in feats
    use_ctrl = "control" in feats

    # ---- load training pairs
    Xc, Y, cls, moas = [], [], [], []
    n = 0
    with open(args.train_file) as f:
        for line in f:
            if args.train_limit and n >= args.train_limit:
                break
            ex = json.loads(line); m = ex.get("metadata", {})
            cr = rankvec(ev.control_from_prompt(ex["prompt"]), panel, worst)
            tr = rankvec(ex["response"], panel, worst)
            Xc.append(cr); Y.append(tr - cr)          # predict the SHIFT
            cls.append(m.get("cell_line_id")); moas.append(m.get("moa"))
            n += 1
    Xc = np.vstack(Xc); Y = np.vstack(Y)
    logger.info(f"  Loaded {len(Xc):,} training pairs; predicting shift (P={P})")

    cl_enc, ncl = onehot_fitter(cls) if use_cl else (None, 0)
    moa_enc, nmoa = onehot_fitter(moas) if use_moa else (None, 0)

    def featurize(control_vec, cl, moa):
        parts = []
        if use_ctrl: parts.append(control_vec)
        if use_cl:   parts.append(cl_enc(cl))
        if use_moa:  parts.append(moa_enc(moa))
        return np.concatenate(parts) if parts else control_vec

    Xtr = np.vstack([featurize(Xc[i], cls[i], moas[i]) for i in range(len(Xc))])
    D = Xtr.shape[1]
    logger.info(f"  Feature dim D={D} (control={use_ctrl} cellline={use_cl}:{ncl} moa={use_moa}:{nmoa})")

    # ---- ridge closed form: W = (X'X + lam I)^-1 X'Y   (multi-output)
    lam = args.ridge_lambda
    XtX = Xtr.T @ Xtr + lam * np.eye(D)
    XtY = Xtr.T @ Y
    W = solve(XtX, XtY)              # (D, P)
    logger.info(f"  Fit ridge (lambda={lam}); W shape {W.shape}")

    result = {"mode": "linear", "features": feats, "ridge_lambda": lam,
              "de_k_list": [int(x) for x in args.de_k_list.split(",")], "tiers": {}}
    de_k_list = result["de_k_list"]

    for tier in TIERS:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            continue
        examples = subsample([json.loads(l) for l in open(path)], args.max_eval, args.subsample_seed)
        vals = defaultdict(list); drugs = []
        for ex in examples:
            m = ex.get("metadata", {})
            cr = rankvec(ev.control_from_prompt(ex["prompt"]), panel, worst)
            x = featurize(cr, m.get("cell_line_id"), m.get("moa"))
            pred_shift = x @ W                    # (P,)
            pred_ranks = cr + pred_shift          # control + predicted shift
            pred_sent = ranks_to_sentence(pred_ranks, panel)
            mm = score_pred(pred_sent, ex, panel, panel_index, linear_model, worst, de_k_list, args.topn)
            for mk in ["de_delta_pearson", "topn_expressed_tau", "panel_tau"] + \
                      [f"de_delta_pearson_k{kk}" for kk in de_k_list]:
                vals[mk].append(mm.get(mk))
            drugs.append(m.get("drug"))
        result["tiers"][tier] = {"n_scored": len(drugs),
                                 "metrics": {mk: agg(vals[mk], drugs, args.n_boot, args.subsample_seed)
                                             for mk in vals}}
        hk = result["tiers"][tier]["metrics"].get("de_delta_pearson")
        logger.info(f"  {tier:26s} LINEAR DEdr={hk['mean']:.3f} [{hk['ci_low']:.3f},{hk['ci_high']:.3f}] n={len(drugs)}"
                    if hk else f"  {tier:26s} LINEAR no predictions")
    return result


# ======================================================================= MODE 2: consensus
def run_consensus(args, panel, panel_index, worst, linear_model):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    de_k_list = [int(x) for x in args.de_k_list.split(",")]
    k_report = sorted(int(x) for x in args.consensus_k_report.split(","))
    K = args.consensus_k
    assert max(k_report) <= K, "consensus_k_report values must be <= consensus_k"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Loading model {args.model_path} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()

    result = {"mode": "consensus", "consensus_k": K, "k_report": k_report,
              "temperature": args.temperature, "top_p": args.top_p,
              "de_k_list": de_k_list, "tiers": {}}

    for tier in TIERS:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            continue
        examples = subsample([json.loads(l) for l in open(path)], args.max_eval, args.subsample_seed)

        # generate K sampled draws for every example (store as rank vectors to save memory)
        draw_ranks = [[] for _ in examples]   # per example: list of K rank vectors
        for d in range(K):
            torch.manual_seed(args.subsample_seed + d)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.subsample_seed + d)
            for i in range(0, len(examples), args.gen_batch_size):
                batch = examples[i:i + args.gen_batch_size]
                prompts = [ex["prompt"] for ex in batch]
                gens = ev.generate_cell_sentences_batched(
                    model, tok, prompts, device=device,
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p)
                for j, g in enumerate(gens):
                    draw_ranks[i + j].append(rankvec(g, panel, worst))
            logger.info(f"    {tier}: draw {d+1}/{K} done")

        # for each report-k, build consensus from the first k draws and score
        tier_out = {}
        for kk in k_report:
            vals = defaultdict(list); drugs = []
            for ex, dr in zip(examples, draw_ranks):
                mat = np.vstack(dr[:kk])
                cons = consensus_from_rank_matrix(mat, panel)
                mm = score_pred(cons, ex, panel, panel_index, linear_model, worst, de_k_list, args.topn)
                for mk in ["de_delta_pearson", "topn_expressed_tau", "panel_tau"] + \
                          [f"de_delta_pearson_k{q}" for q in de_k_list]:
                    vals[mk].append(mm.get(mk))
                drugs.append(ex.get("metadata", {}).get("drug"))
            tier_out[f"consensus_k{kk}"] = {
                "n_scored": len(drugs),
                "metrics": {mk: agg(vals[mk], drugs, args.n_boot, args.subsample_seed) for mk in vals},
            }
        # also: mean of single-draw scores (== harness gen_samples behavior) for contrast
        vals_single = []; drugs_single = []
        for ex, dr in zip(examples, draw_ranks):
            per = []
            for v in dr:
                s = ranks_to_sentence(v, panel)
                per.append(score_pred(s, ex, panel, panel_index, linear_model, worst, de_k_list, args.topn).get("de_delta_pearson"))
            per = [x for x in per if x is not None and x == x]
            vals_single.append(float(np.mean(per)) if per else None)
            drugs_single.append(ex.get("metadata", {}).get("drug"))
        tier_out["mean_of_single_draw_scores"] = agg(vals_single, drugs_single, args.n_boot, args.subsample_seed)

        result["tiers"][tier] = tier_out
        def s(x): return f"{x['mean']:.3f}" if x else "NA"
        line = " ".join(f"cons_k{kk}={s(tier_out[f'consensus_k{kk}']['metrics'].get('de_delta_pearson'))}"
                        for kk in k_report)
        logger.info(f"  {tier:26s} {line}  | mean_single_draw={s(tier_out['mean_of_single_draw_scores'])}")
    return result


# =================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear", "consensus"], required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--de_k_list", default="20,50,100,200")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--max_eval", type=int, default=300)
    ap.add_argument("--subsample_seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    # linear
    ap.add_argument("--train_file")
    ap.add_argument("--features", default="control,cellline,moa")
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--train_limit", type=int, default=60000)
    # consensus
    ap.add_argument("--model_path")
    ap.add_argument("--consensus_k", type=int, default=20)
    ap.add_argument("--consensus_k_report", default="1,5,20")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--gen_batch_size", type=int, default=48)
    ap.add_argument("--max_new_tokens", type=int, default=3800)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.out).startswith(os.path.abspath(args.eval_dir) + os.sep):
        raise SystemExit("Refusing to write inside eval_dir.")
    panel, panel_index, worst, linear_model = load_common(args.eval_dir)

    if args.mode == "linear":
        assert args.train_file, "--train_file required for linear mode"
        result = run_linear(args, panel, panel_index, worst, linear_model)
    else:
        assert args.model_path, "--model_path required for consensus mode"
        result = run_consensus(args, panel, panel_index, worst, linear_model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

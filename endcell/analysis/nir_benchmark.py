#!/usr/bin/env python
r"""
nir_benchmark.py
================
Half 2 of the calibration story: benchmark the model on the ONE calibrated metric (NIR), against
fair baselines, at pseudobulk, on the held-out eval tiers.

NIR (Normalized Inverse Rank) = discrimination: for a predictor's pseudobulk profile, rank its
similarity to its OWN drug's truth against its similarity to ALL OTHER drugs' truths. 1.0 = own is
the single closest (perfect identifiability); ~0.5 = chance. Reported under TWO distances:
  * rank-NIR : rank correlation over ALL genes expressed in the own-drug truth (no top-N cap; the
               ~few-hundred expressed genes of [END_CELL]). Clean — the model's native output, no decode.
  * expr-NIR : Euclidean distance after decoding ranks -> expression via linear_model.json
               (matches the distance used to establish NIR as calibrated; inherits the lossy decode).

PREDICTORS scored per drug x cell line:
  * model   : K temperature-sampled predictions from the drug's held-out prompts, pseudobulk-averaged.
  * linear  : ridge control->shift fit on train, applied to the drug's control pseudobulk (drug-AGNOSTIC
              -> same output for every drug in a cell line -> chance by construction).
  * mean    : leave-one-out drug-agnostic mean profile (chance by construction).
  * ceiling : a real disjoint half of the drug's cells (the achievable discrimination bar).

USAGE (GPU)
  python nir_benchmark.py --eval_dir DATA_endcell_big --model_path CKPT/final \
     --tiers tier2_unseen_drugs,tier3_unseen_combos,tier4_dose_interpolation \
     --train_file DATA_endcell_big/train.jsonl --k_samples 8 --temperature 0.8 \
     --min_cells 20 --min_drugs_per_cl 6 --bf16 --out RESULTS/nir_benchmark.json

TEMP SWEEP diagnostic (how output diversity changes with temperature)
  python nir_benchmark.py --temp_sweep --eval_dir ... --model_path ... --tier tier2_unseen_drugs \
     --temps 0,0.5,0.8,1.0 --k_samples 8 --bf16 --out RESULTS/temp_sweep.json

SELFTEST (no model/data) — validates the NIR + similarity machinery
  python nir_benchmark.py --selftest --out /tmp/nir_selftest.json
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

# --- repo path bootstrap (reorg): make shared/ + sibling pipeline dirs importable ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
for _p in [os.path.join(_ROOT, "shared"), *sorted(glob.glob(os.path.join(_PIPE, "*")))]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"


# ----------------------------------------------------------------- representations
def genes_of(sentence):
    out = []
    for t in sentence.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def sentence_to_rankarr(sentence, panel_index, P, fill=None):
    fill = P if fill is None else fill
    arr = np.full(P, float(fill))
    seen = set()
    pos = 0
    for g in genes_of(sentence):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        pos += 1
        seen.add(gi)
        arr[gi] = float(pos)
    return arr


def sentence_to_expr(sentence, panel_index, P, lm):
    """Decode a cell sentence to an expression vector via the C2S linear model
    (expr = slope*log10(rank) + intercept, clamped >=0; absent genes -> 0)."""
    slope, intercept = lm["slope"], lm["intercept"]
    arr = np.zeros(P)
    seen = set()
    pos = 0
    for g in genes_of(sentence):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        pos += 1
        seen.add(gi)
        arr[gi] = max(0.0, slope * np.log10(pos) + intercept)
    return arr


def pb_rank(sentences, panel_index, P):
    return np.mean(np.stack([sentence_to_rankarr(s, panel_index, P) for s in sentences]), axis=0)


def pb_expr(sentences, panel_index, P, lm):
    return np.mean(np.stack([sentence_to_expr(s, panel_index, P, lm) for s in sentences]), axis=0)


# ----------------------------------------------------------------- similarities + NIR
def rank_corr(pred_rank, true_rank, expressed_idx):
    """Correlation of two rank profiles over the reference's expressed genes (all of them)."""
    if len(expressed_idx) < 3:
        return None
    a, b = pred_rank[expressed_idx], true_rank[expressed_idx]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def nir_from_sims(sim_own, sims_other):
    """NIR = fraction of other drugs LESS similar than own (higher sim = more similar)."""
    s = [x for x in sims_other if x is not None]
    if sim_own is None or not s:
        return None
    return float(np.mean([sim_own > x for x in s]))


def nir_from_dists(dist_own, dists_other):
    """NIR under a DISTANCE (lower = more similar): fraction of others FARTHER than own."""
    d = [x for x in dists_other if x is not None]
    if dist_own is None or not d:
        return None
    return float(np.mean([dist_own < x for x in d]))


# ----------------------------------------------------------------- ridge linear (control->shift)
def fit_ridge(train, panel_index, P, ev, max_fit, ridge_lambda, rng):
    Xs, Ys = [], []
    idx = rng.permutation(len(train))[:max_fit]
    for i in idx:
        ex = train[i]
        ctrl = ev.control_from_prompt(ex["prompt"])
        if not ctrl:
            continue
        c = sentence_to_rankarr(ctrl, panel_index, P)
        t = sentence_to_rankarr(ex["response"], panel_index, P)
        Xs.append(c); Ys.append(t - c)
    X, Y = np.asarray(Xs), np.asarray(Ys)
    mu_c, mu_s = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_c, Y - mu_s
    XtX = Xc.T @ Xc
    lam = ridge_lambda * (np.trace(XtX) / P + 1e-9)
    W = np.linalg.solve(XtX + lam * np.eye(P), Xc.T @ Yc)
    return mu_c, mu_s, W


# ----------------------------------------------------------------- benchmark one cell line
def score_cellline(by_drug, panel_index, P, lm, model_pb_fn, lin_fn, rng):
    """by_drug: {drug: {"resp":[sentences], "ctrl":[sentences]}}. Returns per-drug NIR per predictor.

    CONSISTENT DENOISING is essential: split each drug into two disjoint halves; use half A as EVERY
    drug's held-out truth, and half B as the ceiling's real-replicate predictor. All predictors
    (ceiling / model / linear / mean) are then scored against the same half-A truths at the same noise
    level. (The earlier version compared the ceiling's own noisy half to other drugs' clean FULL
    profiles, which inverted it.)"""
    drugs0 = list(by_drug.keys())
    A, B = {}, {}
    for d in drugs0:
        resp = by_drug[d]["resp"]
        if len(resp) < 4:
            continue
        idx = list(range(len(resp))); rng.shuffle(idx); h = len(idx) // 2
        A[d] = [resp[i] for i in idx[:h]]
        B[d] = [resp[i] for i in idx[h:]]
    drugs = list(A.keys())
    if len(drugs) < 3:
        return []
    truth_rank = {d: pb_rank(A[d], panel_index, P) for d in drugs}          # held-out truth = half A
    truth_expr = {d: pb_expr(A[d], panel_index, P, lm) for d in drugs}
    expressed = {d: np.where(truth_rank[d] < P)[0] for d in drugs}          # all expressed genes of the truth
    ceil_rank = {d: pb_rank(B[d], panel_index, P) for d in drugs}           # disjoint real replicate = half B
    ceil_expr = {d: pb_expr(B[d], panel_index, P, lm) for d in drugs}

    rows = []
    for d in drugs:
        others = [dd for dd in drugs if dd != d]
        if len(others) < 2:
            continue
        preds = {"ceiling": (ceil_rank[d], ceil_expr[d]),
                 "linear": lin_fn(by_drug[d]["ctrl"]),
                 "mean": (np.mean(np.stack([truth_rank[o] for o in others]), axis=0),
                          np.mean(np.stack([truth_expr[o] for o in others]), axis=0))}
        mr, me = model_pb_fn(d)
        if mr is not None:
            preds["model"] = (mr, me)

        row = {}
        exp_idx = expressed[d]
        for name, (pr, pe) in preds.items():
            s_own = rank_corr(pr, truth_rank[d], exp_idx)
            s_oth = [rank_corr(pr, truth_rank[o], exp_idx) for o in others]
            d_own = float(np.linalg.norm(pe - truth_expr[d]))
            d_oth = [float(np.linalg.norm(pe - truth_expr[o])) for o in others]
            row[name] = {"nir_rank": nir_from_sims(s_own, s_oth),
                         "nir_expr": nir_from_dists(d_own, d_oth)}
        rows.append(row)
    return rows


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic: drug-aware predictor should get NIR ~1 (identifies its drug); drug-agnostic
    predictor (same output for all) ~0.5 (chance), under BOTH distances."""
    rng = np.random.RandomState(0)
    P = 300
    panel = [f"G{i}" for i in range(P)]
    lm = {"slope": -0.4, "intercept": 1.6}
    pidx = {g: i for i, g in enumerate(panel)}

    # each drug: a fixed pool + a fixed per-gene expression level, so cells of the same drug emit genes
    # in a CONSISTENT rank order (real model outputs are ordered by expression; random order can't test
    # a rank metric). Different drugs -> different genes/order -> low cross-drug similarity.
    pools, levels = {}, {}
    for d in range(15):
        pool = rng.choice(P, 130, replace=False)
        pools[f"d{d}"] = pool
        levels[f"d{d}"] = {int(g): rng.rand() for g in pool}

    def make_cell(dname):
        pool, lev = pools[dname], levels[dname]
        drawn = sorted(rng.choice(pool, 90, replace=False), key=lambda g: -lev[int(g)])
        return " ".join(panel[g] for g in drawn) + " " + SENTINEL

    def make_random():
        genes = sorted(rng.choice(P, 90, replace=False))
        return " ".join(panel[g] for g in genes) + " " + SENTINEL

    by_drug = {f"d{d}": {"resp": [make_cell(f"d{d}") for _ in range(20)],
                         "ctrl": [make_random() for _ in range(20)]} for d in range(15)}

    def model_aware(d):
        s = [make_cell(d) for _ in range(8)]
        return pb_rank(s, pidx, P), pb_expr(s, pidx, P, lm)
    fixed = [make_random() for _ in range(8)]           # drug-AGNOSTIC: same output for every drug
    def lin_agnostic(ctrl_sents):
        return pb_rank(fixed, pidx, P), pb_expr(fixed, pidx, P, lm)

    rows = score_cellline(by_drug, pidx, P, lm, model_aware, lin_agnostic, rng)
    def agg(name, key):
        v = [r[name][key] for r in rows if name in r and r[name][key] is not None]
        return float(np.mean(v)) if v else None
    m_rank, m_expr = agg("model", "nir_rank"), agg("model", "nir_expr")
    l_rank, l_expr = agg("linear", "nir_rank"), agg("linear", "nir_expr")
    logger.info(f"  drug-AWARE model  NIR: rank={m_rank:.3f} expr={m_expr:.3f} (expect ~1, identifies drug)")
    logger.info(f"  drug-AGNOSTIC lin NIR: rank={l_rank:.3f} expr={l_expr:.3f} (expect ~chance, cannot discriminate)")
    # the machinery is right if the drug-aware predictor identifies its drug and the agnostic one can't
    ok = (m_rank > 0.85 and m_expr > 0.85 and l_rank < m_rank - 0.25 and l_expr < m_expr - 0.25
          and l_rank < 0.7 and l_expr < 0.7)
    out = {"selftest": True, "passed": bool(ok),
           "model": {"nir_rank": m_rank, "nir_expr": m_expr},
           "linear": {"nir_rank": l_rank, "nir_expr": l_expr}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def load_tier_by_drug(eval_dir, tier, ev):
    path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
    if not os.path.exists(path):
        logger.warning(f"  missing {path}")
        return None
    by_cl = defaultdict(lambda: defaultdict(lambda: {"resp": [], "ctrl": [], "prompts": []}))
    for line in open(path):
        ex = json.loads(line)
        m = ex.get("metadata", {})
        cl, drug = m.get("cell_line_id"), m.get("drug")
        ctrl = ev.control_from_prompt(ex["prompt"])
        if cl is None or drug is None or not ctrl:
            continue
        slot = by_cl[cl][drug]
        slot["resp"].append(ex["response"]); slot["ctrl"].append(ctrl); slot["prompts"].append(ex["prompt"])
    return by_cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--temp_sweep", action="store_true")
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--tiers", default="tier2_unseen_drugs,tier3_unseen_combos")
    ap.add_argument("--tier", default="tier2_unseen_drugs", help="single tier for --temp_sweep")
    ap.add_argument("--temps", default="0,0.5,0.8,1.0")
    ap.add_argument("--k_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--gen_batch_size", type=int, default=48)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--min_drugs_per_cl", type=int, default=6)
    ap.add_argument("--n_celllines", type=int, default=20)
    ap.add_argument("--max_fit", type=int, default=40000)
    ap.add_argument("--ridge_lambda", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    import torch
    import evaluate_c2s_tahoe as ev
    from transformers import AutoModelForCausalLM, AutoTokenizer

    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    lm_path = os.path.join(args.eval_dir, "linear_model.json")
    lm = json.load(open(lm_path))
    logger.info(f"Panel {P}; linear_model slope={lm['slope']:.3f} intercept={lm['intercept']:.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()
    ec = tok.encode(SENTINEL, add_special_tokens=False)
    end_id = ec[0] if len(ec) == 1 else tok.convert_tokens_to_ids(SENTINEL)
    eos = [end_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])

    def generate(prompts, temperature):
        prev = tok.padding_side; tok.padding_side = "left"
        outs = []
        try:
            for i in range(0, len(prompts), args.gen_batch_size):
                batch = prompts[i:i + args.gen_batch_size]
                enc = tok(batch, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                       do_sample=(temperature > 0), temperature=max(temperature, 1e-2),
                                       top_p=args.top_p)
                plen = enc["input_ids"].shape[1]
                for j in range(len(batch)):
                    ids = g[j][plen:].tolist()
                    if end_id in ids:
                        ids = ids[:ids.index(end_id)]
                    outs.append(tok.decode(ids, skip_special_tokens=True).strip() + " " + SENTINEL)
        finally:
            tok.padding_side = prev
        return outs

    if args.temp_sweep:
        run_temp_sweep(args, ev, panel_index, P, generate)
        return

    # fit the drug-agnostic ridge linear once
    logger.info("Fitting ridge control->shift on train ...")
    train = [json.loads(l) for l in open(args.train_file)]
    mu_c, mu_s, W = fit_ridge(train, panel_index, P, ev, args.max_fit, args.ridge_lambda,
                              np.random.RandomState(args.seed))

    def lin_fn(ctrl_sents):
        c = pb_rank(ctrl_sents, panel_index, P)
        pred = c + mu_s + (c - mu_c) @ W
        # decode the predicted rank profile to expression (approx: order genes by predicted rank)
        order = np.argsort(pred)
        expr = np.zeros(P)
        for r, gi in enumerate(order, 1):
            if pred[gi] < P:  # treat fill-level as absent
                expr[gi] = max(0.0, lm["slope"] * np.log10(r) + lm["intercept"])
        return pred, expr

    rng = np.random.RandomState(args.seed)
    result = {"tiers": {}, "config": {k: v for k, v in vars(args).items()}}
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        by_cl = load_tier_by_drug(args.eval_dir, tier, ev)
        if not by_cl:
            continue
        all_rows = []
        used = 0
        for cl, dd in by_cl.items():
            drugs = {d: s for d, s in dd.items() if len(s["resp"]) >= args.min_cells}
            if len(drugs) < args.min_drugs_per_cl:
                continue

            def model_pb_fn(d, _drugs=drugs):
                prompts = _drugs[d]["prompts"][:args.k_samples]
                if not prompts:
                    return None, None
                gens = generate(prompts, args.temperature)
                return pb_rank(gens, panel_index, P), pb_expr(gens, panel_index, P, lm)

            rows = score_cellline(drugs, panel_index, P, lm, model_pb_fn, lin_fn, rng)
            all_rows.extend(rows)
            used += 1
            logger.info(f"  [{tier}] {str(cl)[:22]:22s} {len(drugs)} drugs -> {len(rows)} scored")
            if used >= args.n_celllines:
                break

        agg = {}
        for name in ("model", "linear", "mean", "ceiling"):
            for key in ("nir_rank", "nir_expr"):
                vals = [r[name][key] for r in all_rows if name in r and r[name][key] is not None]
                agg[f"{name}_{key}"] = (float(np.mean(vals)), len(vals)) if vals else (None, 0)
        result["tiers"][tier] = {"n_drugs": len(all_rows), "agg": agg}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=float)

    logger.info("")
    logger.info("=" * 100)
    logger.info("  NIR BENCHMARK (chance ~0.50; ceiling = achievable) — rank-NIR | expr-NIR")
    for tier, td in result["tiers"].items():
        logger.info(f"  [{tier}] (n={td['n_drugs']})")
        for name in ("model", "linear", "mean", "ceiling"):
            r = td["agg"][f"{name}_nir_rank"][0]
            e = td["agg"][f"{name}_nir_expr"][0]
            g = lambda x: f"{x:.3f}" if x is not None else "NA"
            logger.info(f"      {name:8s}  rank-NIR={g(r)}   expr-NIR={g(e)}")
    logger.info("=" * 100)
    logger.info("  Read: ceiling >> chance (drug signal exists); model ~ linear ~ mean ~ chance => drug-blind.")
    logger.info(f"-> {args.out}")


def run_temp_sweep(args, ev, panel_index, P, generate):
    """Diagnostic: for a few (drug,control) contexts, K samples at each temperature; report the
    'core' genes (present in ALL K samples), 'variable' genes (in exactly one), and mean pairwise
    Jaccard. Shows whether higher temperature adds signal or just noise."""
    by_cl = load_tier_by_drug(args.eval_dir, args.tier, ev)
    contexts = []
    for cl, dd in by_cl.items():
        for d, s in dd.items():
            if s["prompts"]:
                contexts.append(s["prompts"][0])
        if len(contexts) >= 12:
            break
    contexts = contexts[:12]
    temps = [float(x) for x in args.temps.split(",")]
    out = {"temps": temps, "per_temp": {}}
    for T in temps:
        cores, variables, jaccs = [], [], []
        for prompt in contexts:
            gens = generate([prompt] * args.k_samples, T)
            sets = [set(genes_of(g)) for g in gens]
            allg = set().union(*sets) if sets else set()
            core = set.intersection(*sets) if sets else set()
            counts = {g: sum(g in s for s in sets) for g in allg}
            variable = [g for g, c in counts.items() if c == 1]
            cores.append(len(core)); variables.append(len(variable))
            js = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
                  for i in range(len(sets)) for j in range(i + 1, len(sets))]
            jaccs.append(float(np.mean(js)) if js else None)
        out["per_temp"][str(T)] = {
            "mean_core_genes": float(np.mean(cores)), "mean_variable_genes": float(np.mean(variables)),
            "mean_pairwise_jaccard": float(np.mean([j for j in jaccs if j is not None]))}
        logger.info(f"  T={T}: core(all K)={np.mean(cores):.0f}  variable(1 of K)={np.mean(variables):.0f}  "
                    f"pairwise-Jaccard={out['per_temp'][str(T)]['mean_pairwise_jaccard']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

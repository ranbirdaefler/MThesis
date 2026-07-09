#!/usr/bin/env python
r"""
pseudobulk_eval.py  (Version A — pseudobulk EVALUATION of the existing model)
=============================================================================
Question: when we evaluate at PSEUDOBULK (denoised) resolution instead of single-cell, does
the drug become visible to the model? The drug-specificity-in-data test showed the drug signal
is invisible per-cell but emerges under aggregation (topN-τ gap grows to d~0.43 at pb15). So we
re-score the EXISTING single-cell-trained model at pseudobulk level.

Construction (per condition = drug × cell_line × dose):
  * gather the eval cells of that condition
  * MODEL: generate the model's single-cell prediction for each, then average the predicted rank
    vectors into ONE pseudobulk prediction (re-ranked). Truth = average of the real treated cells.
    Control = average of the controls. Score pred-vs-truth. (Both sides aggregated identically.)
  * This is done at several pseudobulk sizes N (cells averaged), swept, since the data test showed
    the signal grows with N.

What we score, each with ALL metrics (DE-Δr K-sweep, panel-τ, topN-τ):
  1. MODEL  (real prompts)        — is the model good at pseudobulk?
  2. MODEL  (scrambled drug)      — does swapping the drug now HURT? (the key drug-sensitivity test)
  3. LINEAR (control→shift ridge) — does the model beat the linear baseline at pseudobulk?
  4. CONTROL-as-prediction        — sanity floor (should be ~0 on DE-Δr)
  5. Replicate NOISE CEILING at pseudobulk (two disjoint-half pseudobulks of the same condition)

Outputs the per-(source, N) table + the paired real-vs-scramble delta (drug-clustered bootstrap).

USAGE
-----
  python pseudobulk_eval.py \
     --eval_dir DATA --model_path CKPT/checkpoint-10000 \
     --scramble_dir DATA_scram_diff_moa \        # optional: pre-scrambled eval prompts
     --linear_model RESULTS/linear_control_only.json \  # optional: reuse fitted ridge W
     --train_file DATA/train.jsonl \             # if no --linear_model, fit ridge here
     --out RESULTS/pseudobulk_eval.json \
     --tiers tier1_seen_conditions,tier2_unseen_drugs \
     --pb_sizes 5,15,30 --de_k_list 20,50,100,200 --topn 100 \
     --gen_batch_size 48 --max_new_tokens 3800 --bf16 --n_boot 1000 --seed 42
"""
import argparse, json, os, logging
from collections import defaultdict
import numpy as np
import evaluate_c2s_tahoe as ev

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRIC_KEYS = ["de_delta_pearson", "panel_tau", "topn_expressed_tau"]


# ---------------------------------------------------------------- rank / aggregation helpers
def rankvec(sentence, panel, worst):
    r = ev.cell_sentence_to_gene_ranks(sentence)
    return np.array([r.get(g, worst) for g in panel], dtype=np.float64)


def ranks_to_sentence(arr, panel):
    return " ".join(panel[i] for i in np.argsort(arr, kind="stable"))


def pseudobulk_sentence(sentences, panel, worst):
    """Average a list of cell sentences into one pseudobulk sentence (mean rank, re-ranked)."""
    if not sentences:
        return None
    acc = np.zeros(len(panel))
    for s in sentences:
        acc += rankvec(s, panel, worst)
    return ranks_to_sentence(acc / len(sentences), panel)


def score(pred_sent, true_sent, ctrl_sent, panel, panel_index, linear_model, worst, de_k_list, topn):
    true_ranks = ev.cell_sentence_to_gene_ranks(true_sent)
    control_ranks = ev.cell_sentence_to_gene_ranks(ctrl_sent)
    de_ranked = ev.select_top_de_genes(true_ranks, control_ranks, panel, max(de_k_list), worst)
    de_by_k = {kk: de_ranked[:kk] for kk in de_k_list}
    topn_genes = ev.select_top_expressed(true_ranks, panel, topn, worst)
    hk = 50 if 50 in de_k_list else de_k_list[0]
    return ev.compute_scalar_metrics(pred_sent, true_sent, true_ranks, control_ranks,
                                     panel, panel_index, linear_model, worst,
                                     de_by_k, topn_genes, headline_k=hk)


def agg(vals, drugs, n_boot, seed):
    v = [(x, d) for x, d in zip(vals, drugs) if x is not None and x == x]
    if not v:
        return None
    return ev.cluster_bootstrap_ci([x for x, _ in v], [d for _, d in v], n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------- linear baseline (ridge W)
def fit_or_load_linear(args, panel, worst):
    P = len(panel)
    if args.linear_model and os.path.exists(args.linear_model):
        d = json.load(open(args.linear_model))
        if "W" in d:
            logger.info("  Loaded ridge W from --linear_model")
            return np.array(d["W"])
    # fit fresh from train (control rank vec -> shift), same as simple_baselines linear mode
    logger.info("  Fitting ridge W from train_file (control -> shift) ...")
    Xc, Y, n = [], [], 0
    with open(args.train_file) as f:
        for line in f:
            if args.train_limit and n >= args.train_limit:
                break
            ex = json.loads(line)
            cr = rankvec(ev.control_from_prompt(ex["prompt"]), panel, worst)
            tr = rankvec(ex["response"], panel, worst)
            Xc.append(cr); Y.append(tr - cr); n += 1
    Xc = np.vstack(Xc); Y = np.vstack(Y)
    W = np.linalg.solve(Xc.T @ Xc + args.ridge_lambda * np.eye(P), Xc.T @ Y)
    logger.info(f"  Fit ridge on {len(Xc):,} pairs; W {W.shape}")
    return W


def linear_predict_sentence(ctrl_sent, W, panel, worst):
    cr = rankvec(ctrl_sent, panel, worst)
    return ranks_to_sentence(cr + cr @ W, panel)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--scramble_dir", default=None,
                    help="dir with pre-scrambled eval_{tier}.jsonl (drug swapped, truth kept)")
    ap.add_argument("--linear_model", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--train_limit", type=int, default=60000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="tier1_seen_conditions,tier2_unseen_drugs")
    ap.add_argument("--pb_sizes", default="5,15,30")
    ap.add_argument("--de_k_list", default="20,50,100,200")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--min_cells_per_cond", type=int, default=6)
    ap.add_argument("--max_conds_per_tier", type=int, default=300)
    ap.add_argument("--gen_batch_size", type=int, default=48)
    ap.add_argument("--max_new_tokens", type=int, default=3800)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.abspath(args.out).startswith(os.path.abspath(args.eval_dir) + os.sep):
        raise SystemExit("Refusing to write inside eval_dir.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.RandomState(args.seed)
    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    worst = len(panel) + 1
    tiers = [t.strip() for t in args.tiers.split(",")]
    pb_sizes = [int(x) for x in args.pb_sizes.split(",")]
    de_k_list = [int(x) for x in args.de_k_list.split(",")]

    lm_path = os.path.join(args.eval_dir, "linear_model.json")
    expr_linear = json.load(open(lm_path)) if os.path.exists(lm_path) else None  # for expr metrics
    W = fit_or_load_linear(args, panel, worst) if (args.train_file or args.linear_model) else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Loading model {args.model_path} on {device}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()

    def generate_for(examples):
        gens = []
        for i in range(0, len(examples), args.gen_batch_size):
            batch = [e["prompt"] for e in examples[i:i + args.gen_batch_size]]
            gens.extend(ev.generate_cell_sentences_batched(
                model, tok, batch, device=device,
                max_new_tokens=args.max_new_tokens, do_sample=False))
        return gens

    result = {"pb_sizes": pb_sizes, "de_k_list": de_k_list, "tiers": {}}

    for tier in tiers:
        real_path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(real_path):
            logger.warning(f"  missing {real_path}"); continue
        examples = [json.loads(l) for l in open(real_path)]

        # group by condition (drug, cell_line, dose)
        by_cond = defaultdict(list)
        for e in examples:
            m = e.get("metadata", {})
            by_cond[(m.get("drug"), m.get("cell_line_id"), m.get("dose"))].append(e)
        conds = [c for c, v in by_cond.items() if len(v) >= args.min_cells_per_cond]
        rng.shuffle(conds); conds = conds[:args.max_conds_per_tier]
        logger.info(f"  {tier}: {len(conds)} conditions with >= {args.min_cells_per_cond} cells")

        # scrambled prompts (optional): map by position within condition
        scram_map = {}
        if args.scramble_dir:
            sp = os.path.join(args.scramble_dir, f"eval_{tier}.jsonl")
            if os.path.exists(sp):
                s_examples = [json.loads(l) for l in open(sp)]
                s_by_cond = defaultdict(list)
                for e in s_examples:
                    m = e.get("metadata", {})
                    s_by_cond[(m.get("drug"), m.get("cell_line_id"), m.get("dose"))].append(e)
                scram_map = s_by_cond  # note: scrambled prompt names a different drug but metadata truth kept

        # generate all needed cells once, per condition, up to max(pb_sizes)*2 (for ceiling halves)
        need = max(pb_sizes) * 2
        tier_out = {f"pb{N}": {src: [] for src in ["model", "scramble", "linear", "control", "ceiling"]}
                    for N in pb_sizes}
        drugs_out = {f"pb{N}": [] for N in pb_sizes}

        for c in conds:
            cells = by_cond[c]
            if len(cells) > need:
                cells = [cells[i] for i in rng.choice(len(cells), need, replace=False)]
            true_sents = [e["response"] for e in cells]
            ctrl_sents = [ev.control_from_prompt(e["prompt"]) for e in cells]
            model_gens = generate_for(cells)
            # linear predictions per cell
            lin_sents = [linear_predict_sentence(cs, W, panel, worst) for cs in ctrl_sents] if W is not None else None
            # scramble predictions: generate on scrambled prompts if available (match by count)
            scram_gens = None
            if scram_map:
                s_cells = scram_map.get(c, [])[:len(cells)]
                if len(s_cells) >= min(pb_sizes):
                    scram_gens = generate_for(s_cells)
                    scram_true = [e["response"] for e in s_cells]
                    scram_ctrl = [ev.control_from_prompt(e["prompt"]) for e in s_cells]

            drug = c[0]
            for N in pb_sizes:
                if len(cells) < N:
                    continue
                idx = list(range(len(cells))); rng.shuffle(idx)
                sel = idx[:N]
                tb_true = pseudobulk_sentence([true_sents[i] for i in sel], panel, worst)
                tb_ctrl = pseudobulk_sentence([ctrl_sents[i] for i in sel], panel, worst)
                # MODEL
                pb_model = pseudobulk_sentence([model_gens[i] for i in sel], panel, worst)
                mm = score(pb_model, tb_true, tb_ctrl, panel, panel_index, expr_linear, worst, de_k_list, args.topn)
                tier_out[f"pb{N}"]["model"].append(mm)
                # LINEAR
                if lin_sents is not None:
                    pb_lin = pseudobulk_sentence([lin_sents[i] for i in sel], panel, worst)
                    tier_out[f"pb{N}"]["linear"].append(
                        score(pb_lin, tb_true, tb_ctrl, panel, panel_index, expr_linear, worst, de_k_list, args.topn))
                # CONTROL-as-prediction
                tier_out[f"pb{N}"]["control"].append(
                    score(tb_ctrl, tb_true, tb_ctrl, panel, panel_index, expr_linear, worst, de_k_list, args.topn))
                # SCRAMBLE (pred from scrambled-drug prompts, scored vs the SAME true pseudobulk)
                if scram_gens is not None and len(scram_gens) >= N:
                    pb_scr = pseudobulk_sentence(scram_gens[:N], panel, worst)
                    tier_out[f"pb{N}"]["scramble"].append(
                        score(pb_scr, tb_true, tb_ctrl, panel, panel_index, expr_linear, worst, de_k_list, args.topn))
                # CEILING: two disjoint-half pseudobulks of the TRUE cells
                if len(cells) >= 2 * N:
                    h1 = pseudobulk_sentence([true_sents[i] for i in idx[:N]], panel, worst)
                    h2 = pseudobulk_sentence([true_sents[i] for i in idx[N:2 * N]], panel, worst)
                    tier_out[f"pb{N}"]["ceiling"].append(
                        score(h2, h1, tb_ctrl, panel, panel_index, expr_linear, worst, de_k_list, args.topn))
                drugs_out[f"pb{N}"].append(drug)

        # aggregate
        tier_res = {}
        for N in pb_sizes:
            key = f"pb{N}"; drugs = drugs_out[key]
            tier_res[key] = {}
            for src in ["model", "scramble", "linear", "control", "ceiling"]:
                rows = tier_out[key][src]
                if not rows:
                    tier_res[key][src] = None; continue
                mset = {}
                for mk in METRIC_KEYS + [f"de_delta_pearson_k{kk}" for kk in de_k_list]:
                    mset[mk] = agg([r.get(mk) for r in rows], drugs, args.n_boot, args.seed)
                tier_res[key][src] = {"n": len(rows), "metrics": mset}
            # paired real-vs-scramble delta on DE-Δr (same conditions, by position)
            mrows = tier_out[key]["model"]; srows = tier_out[key]["scramble"]
            if srows and len(srows) == len(mrows):
                deltas = [ (mrows[i].get("de_delta_pearson") or np.nan) - (srows[i].get("de_delta_pearson") or np.nan)
                           for i in range(len(mrows)) ]
                tier_res[key]["real_minus_scramble_de_delta"] = agg(deltas, drugs, args.n_boot, args.seed)
        result["tiers"][tier] = tier_res

        # log
        for N in pb_sizes:
            r = tier_res[f"pb{N}"]
            def g(src, mk="de_delta_pearson"):
                x = r.get(src)
                return f"{x['metrics'][mk]['mean']:.3f}" if x and x["metrics"].get(mk) else "NA"
            logger.info(f"  {tier} pb{N}: model={g('model')} scramble={g('scramble')} "
                        f"linear={g('linear')} control={g('control')} ceiling={g('ceiling')} "
                        f"| topN-τ model={g('model','topn_expressed_tau')} scr={g('scramble','topn_expressed_tau')}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

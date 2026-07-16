#!/usr/bin/env python
r"""
metric_grades_model.py  (Step 1 — does the discriminative metric grade MODEL predictions?)
==========================================================================================
The spike-in benchmark showed topn_tau discriminates two REAL drug populations at ~0.98. But the
metric we evaluate models with must separate a good PREDICTION from a bad one — a different test.
This script runs the forced choice at the PREDICTION level:

  For (drug A, cell line):
    prediction = model's pseudobulk prediction for drug A (generated from drug-A prompts)
    truth_A    = real drug-A pseudobulk (held-out cells)
    truth_B    = real DIFFERENT-drug pseudobulk (same cell line)
    CORRECT if  metric(prediction, truth_A) > metric(prediction, truth_B)
    i.e. the metric rates the model's drug-A prediction as closer to the REAL drug-A response
    than to a different drug's response. Accuracy over many (A,B) trials; chance = 0.50.

References compared (all scored with the SAME forced choice):
  1. model    : model prediction as reference  -> does the MODEL's output pick the right drug?
  2. ceiling  : a REAL held-out drug-A pseudobulk as reference (disjoint from truth_A)
                -> the metric's ceiling when the reference genuinely IS drug A (upper bound the
                model could reach). This should reproduce the ~0.98 spike-in number.
  3. linear   : linear-baseline prediction as reference (context)
  4. scramble : model prediction from a DIFFERENT-drug (B) prompt as reference, scored the same
                way. If the model ignores the drug, scramble ~= model (no movement toward B).

Metrics: topn_tau (headline), de_delta, panel_tau. Run on the OLD-format data the model was
trained on (data_diverse2); topn_tau/de_delta are tail-immune so the full-panel tail is ignored.

READOUT: model accuracy vs ceiling accuracy.
  - model near ceiling  -> model IS drug-sensitive; metric grades it; task is learnable/graded.
  - model near 0.50     -> model ignores drug; the metric cannot grade what the prediction lacks;
                           retraining (or a representation change) is required, not a new metric.

USAGE
-----
  python metric_grades_model.py \
     --eval_dir DATA --model_path CKPT/checkpoint-10000 \
     --scramble_dir DATA_scram_diff_moa --linear_model RESULTS/linear_control_only.json \
     --train_file DATA/train.jsonl \
     --out RESULTS/metric_grades_model.json \
     --tiers tier1_seen_conditions,tier2_unseen_drugs \
     --pb_size 15 --n_pairs_per_cond 8 --topn 100 --de_k 50 \
     --gen_batch_size 48 --bf16 --n_boot 1000 --seed 42
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


def rankvec(sentence, panel, worst):
    r = ev.cell_sentence_to_gene_ranks(sentence)
    return np.array([r.get(g, worst) for g in panel], dtype=np.float64)


def ranks_to_sentence(arr, panel):
    return " ".join(panel[i] for i in np.argsort(arr, kind="stable"))


def pseudobulk(sentences, panel, worst):
    if not sentences:
        return None
    acc = np.zeros(len(panel))
    for s in sentences:
        acc += rankvec(s, panel, worst)
    return acc / len(sentences)   # mean rank vector (we score on vectors directly)


# ---- metrics on rank vectors (higher = more similar) ----
def topn_tau(ref, cand, topn):
    idx = np.argsort(ref)[:topn]                 # most-expressed (lowest rank) genes of the reference
    x, y = ref[idx], cand[idx]
    if x.std() < 1e-9 or y.std() < 1e-9: return None
    return float(np.corrcoef(x, y)[0, 1])


def de_delta(ref, cand, control, de_k):
    if control is None: return None
    idx = np.argsort(-np.abs(ref - control))[:de_k]
    rs, cs = ref[idx] - control[idx], cand[idx] - control[idx]
    if rs.std() < 1e-9 or cs.std() < 1e-9: return None
    return float(np.corrcoef(rs, cs)[0, 1])


def panel_tau(ref, cand):
    if ref.std() < 1e-9 or cand.std() < 1e-9: return None
    return float(np.corrcoef(ref, cand)[0, 1])


def score_all(ref, cand, control, topn, de_k):
    return {"topn_tau": topn_tau(ref, cand, topn),
            "de_delta": de_delta(ref, cand, control, de_k),
            "panel_tau": panel_tau(ref, cand)}


def cl_bootstrap(percl, n_boot, seed):
    rng = np.random.RandomState(seed)
    cls = list(percl.keys())
    vals = np.array([percl[c][0] / percl[c][1] for c in cls if percl[c][1] > 0])
    if len(vals) == 0: return None
    boots = [np.mean(vals[rng.choice(len(vals), len(vals), replace=True)]) for _ in range(n_boot)]
    return dict(acc=float(np.mean(vals)), ci_low=float(np.percentile(boots, 2.5)),
                ci_high=float(np.percentile(boots, 97.5)), n_cl=len(vals))


def fit_linear_W(train_file, panel, worst, lam, limit):
    P = len(panel); Xc, Y, n = [], [], 0
    with open(train_file) as f:
        for line in f:
            if limit and n >= limit: break
            ex = json.loads(line)
            cr = rankvec(ev.control_from_prompt(ex["prompt"]), panel, worst)
            tr = rankvec(ex["response"], panel, worst)
            Xc.append(cr); Y.append(tr - cr); n += 1
    Xc = np.vstack(Xc); Y = np.vstack(Y)
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(P), Xc.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--scramble_dir", default=None)
    ap.add_argument("--linear_model", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--train_limit", type=int, default=60000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="tier1_seen_conditions,tier2_unseen_drugs")
    ap.add_argument("--pb_size", type=int, default=15)
    ap.add_argument("--ceiling_size", type=int, default=3,
                    help="pseudobulk size for the disjoint real-drug-A ceiling reference; smaller "
                         "so more groups qualify (needs 2*ceiling_size cells). Ceiling at small N "
                         "is a conservative (lower) estimate of the metric's achievable ceiling.")
    ap.add_argument("--n_pairs_per_cond", type=int, default=8,
                    help="number of different-drug B partners drawn per drug-A condition")
    ap.add_argument("--min_cells_per_cond", type=int, default=30, help="need >= 2*pb_size ideally")
    ap.add_argument("--max_conds_per_tier", type=int, default=200)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--de_k", type=int, default=50)
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
    worst = len(panel) + 1
    tiers = [t.strip() for t in args.tiers.split(",")]
    METRICS = ["topn_tau", "de_delta", "panel_tau"]

    W = None
    if args.train_file or args.linear_model:
        if args.linear_model and os.path.exists(args.linear_model):
            d = json.load(open(args.linear_model))
            if "W" in d: W = np.array(d["W"]); logger.info("  loaded ridge W")
        if W is None and args.train_file:
            logger.info("  fitting ridge W ..."); W = fit_linear_W(args.train_file, panel, worst,
                                                                   args.ridge_lambda, args.train_limit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()
    logger.info(f"  model on {device}")

    def generate(examples):
        gens = []
        for i in range(0, len(examples), args.gen_batch_size):
            batch = [e["prompt"] for e in examples[i:i + args.gen_batch_size]]
            gens.extend(ev.generate_cell_sentences_batched(
                model, tok, batch, device=device, max_new_tokens=args.max_new_tokens, do_sample=False))
        return gens

    N = args.pb_size
    result = {"pb_size": N, "topn": args.topn, "de_k": args.de_k, "tiers": {}}

    for tier in tiers:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  missing {path}"); continue
        examples = [json.loads(l) for l in open(path)]
        # group by (drug, cell_line) pooling across doses for enough cells
        by_cond = defaultdict(list)
        for e in examples:
            m = e.get("metadata", {})
            by_cond[(m.get("drug"), m.get("cell_line_id"))].append(e)
        # cells by cell line -> to find different-drug B partners in the same cell line
        by_cl = defaultdict(list)
        for (drug, cl), cells in by_cond.items():
            by_cl[cl].append(drug)
        conds = [c for c, v in by_cond.items() if len(v) >= args.min_cells_per_cond]
        rng.shuffle(conds); conds = conds[:args.max_conds_per_tier]
        logger.info(f"  {tier}: {len(conds)} drug-conditions with >= {args.min_cells_per_cond} cells")

        # scramble prompts by (drug,cell_line)
        scram_by_cond = {}
        if args.scramble_dir:
            sp = os.path.join(args.scramble_dir, f"eval_{tier}.jsonl")
            if os.path.exists(sp):
                s_by = defaultdict(list)
                for e in [json.loads(l) for l in open(sp)]:
                    m = e.get("metadata", {})
                    s_by[(m.get("drug"), m.get("cell_line_id"))].append(e)
                scram_by_cond = s_by

        # per-cell-line accuracy accumulators: acc[ref_kind][metric][cl] = [correct, total]
        acc = {k: {m: defaultdict(lambda: [0, 0]) for m in METRICS}
               for k in ["model", "ceiling", "linear", "scramble"]}

        for (A, cl) in conds:
            cellsA = by_cond[(A, cl)]
            if len(cellsA) < 2 * N:      # need disjoint halves for truth_A + ceiling ref
                # still usable with >= N for truth_A; ceiling needs 2N, skip ceiling if short
                pass
            # candidate different drugs B in the same cell line with enough cells
            B_options = [d for d in by_cl[cl] if d != A and len(by_cond[(d, cl)]) >= N]
            if not B_options:
                continue

            idxA = list(range(len(cellsA))); rng.shuffle(idxA)
            truthA_cells = [cellsA[i] for i in idxA[:N]]
            truthA = pseudobulk([e["response"] for e in truthA_cells], panel, worst)
            controlA = pseudobulk([ev.control_from_prompt(e["prompt"]) for e in truthA_cells], panel, worst)
            # ceiling reference: a real drug-A pseudobulk of size ceiling_size drawn from cells
            # DISJOINT from truthA (the cells after the first N). Needs N + ceiling_size cells.
            # This measures how well a genuine second draw of drug A matches truthA — the metric's
            # achievable ceiling when the reference really is drug A (at this small size, a
            # conservative/low estimate). Scored against truthA and truthB just like the model.
            C = args.ceiling_size
            ceilingA = None
            if len(cellsA) >= N + C:
                ceilingA = pseudobulk([cellsA[i]["response"] for i in idxA[N:N+C]], panel, worst)

            # model prediction reference (generate on drug-A prompts)
            model_pred = pseudobulk(generate(truthA_cells), panel, worst)
            # linear prediction reference
            linear_pred = None
            if W is not None:
                lin_sents = []
                for e in truthA_cells:
                    cr = rankvec(ev.control_from_prompt(e["prompt"]), panel, worst)
                    lin_sents.append(ranks_to_sentence(cr + cr @ W, panel))
                linear_pred = pseudobulk(lin_sents, panel, worst)
            # scramble prediction reference (model on different-drug prompts, if available)
            scram_pred = None
            s_cells = scram_by_cond.get((A, cl), [])[:N]
            if len(s_cells) >= N:
                scram_pred = pseudobulk(generate(s_cells), panel, worst)

            for _ in range(args.n_pairs_per_cond):
                B = B_options[rng.randint(len(B_options))]
                cellsB = by_cond[(B, cl)]
                jb = rng.choice(len(cellsB), min(N, len(cellsB)), replace=False)
                truthB = pseudobulk([cellsB[j]["response"] for j in jb], panel, worst)

                for kind, ref in [("model", model_pred), ("ceiling", ceilingA),
                                  ("linear", linear_pred), ("scramble", scram_pred)]:
                    if ref is None: continue
                    sA = score_all(ref, truthA, controlA, args.topn, args.de_k)
                    sB = score_all(ref, truthB, controlA, args.topn, args.de_k)
                    for m in METRICS:
                        if sA[m] is None or sB[m] is None: continue
                        correct = 1 if sA[m] > sB[m] else (0 if sA[m] < sB[m] else None)
                        if correct is None: continue
                        cell = acc[kind][m][cl]
                        cell[0] += correct; cell[1] += 1

        tier_res = {}
        for kind in ["model", "ceiling", "linear", "scramble"]:
            tier_res[kind] = {m: cl_bootstrap(acc[kind][m], args.n_boot, args.seed) for m in METRICS}
        # group support: how many cell lines contributed at least one trial for each kind
        support = {kind: len({cl for cl in acc[kind]["topn_tau"] if acc[kind]["topn_tau"][cl][1] > 0})
                   for kind in ["model", "ceiling", "linear", "scramble"]}
        tier_res["_support_celllines"] = support
        result["tiers"][tier] = tier_res

        def g(kind, m):
            r = tier_res[kind][m]
            return f"{r['acc']:.3f}[{r['ci_low']:.3f},{r['ci_high']:.3f}]" if r else "NA"
        logger.info(f"  === {tier} : forced-choice accuracy (chance=0.50) ===")
        logger.info(f"    support (cell lines): model={support['model']} ceiling={support['ceiling']} "
                    f"linear={support['linear']} scramble={support['scramble']}")
        for m in METRICS:
            logger.info(f"    {m:10s} model={g('model',m)}  ceiling={g('ceiling',m)}  "
                        f"linear={g('linear',m)}  scramble={g('scramble',m)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    logger.info("")
    logger.info("  READOUT: compare MODEL vs CEILING.")
    logger.info("   model near ceiling -> model is drug-sensitive, metric grades it.")
    logger.info("   model near 0.50    -> model ignores drug; metric can't grade what's not there.")
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

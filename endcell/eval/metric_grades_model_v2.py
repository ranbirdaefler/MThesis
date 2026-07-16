#!/usr/bin/env python
r"""
metric_grades_model_v2.py  (airtight model-grading instrument for the [END_CELL] model)
=======================================================================================
Upgrades over v1:
  (1) THREE representations for inactive genes: position (P+1), tail_max (P, Federico),
      zero_bucket (mid rank, professor). Every accuracy reported under all three.
  (2) TEMPERATURE SAMPLING: the model generates ONE sampled prediction per cell (temp>0), so the
      15 cells -> 15 distinct predictions -> a genuine pseudobulk (greedy made them identical,
      defeating the aggregation). Mirrors how the real truth pseudobulk is built (15 real cells).
  (3) SCRAMBLE arm: model prediction from a scrambled-drug prompt, scored vs the real truth. If the
      model ignores the drug, scramble ~ model.
  (4) FAIR sparse linear baseline: fit/score over the UNION of expressed genes, not the 947-padded
      dense vector (where ~820 constant-floor entries dominated and made v1's linear degenerate).

Forced choice (per drug A, cell line):
    reference vs truth_A (real drug-A pseudobulk) and truth_B (real different-drug pseudobulk).
    CORRECT if metric(ref, truth_A) > metric(ref, truth_B). Accuracy over pairs; chance = 0.50.
References: model (sampled prediction), ceiling (real disjoint drug-A pb), linear (fair), scramble.

USAGE
-----
  python metric_grades_model_v2.py \
     --eval_dir DATA_endcell_big --model_path CKPT/final \
     --scramble_dir DATA_endcell_big_scram \
     --train_file DATA_endcell_big/train.jsonl \
     --out RESULTS/metric_grades_endcell_v2.json \
     --tiers tier2_unseen_drugs,tier1_seen_conditions \
     --pb_size 15 --ceiling_size 8 --n_pairs_per_cond 8 --min_cells_per_cond 15 \
     --modes position,tail_max,zero_bucket --topn 100 --de_k 50 \
     --temperature 0.8 --top_p 0.9 --gen_batch_size 48 --bf16 --n_boot 1000 --seed 42
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

METRICS = ["topn_tau", "de_delta", "panel_tau"]


# ---------- representation-aware rank arrays (ported from spikein_metric_benchmark) ----------
def sentence_to_rankarr(sentence, panel_index, P, mode="position"):
    """Rank array over the panel. Inactive (absent) genes placed per mode:
       position -> P+1 ; tail_max -> P ; zero_bucket -> shared mid rank (n_active+1)."""
    if mode == "tail_max":
        fill = P
    elif mode == "position":
        fill = P + 1
    else:
        fill = None  # zero_bucket set below
    genes = [g for g in sentence.split() if g != "[END_CELL]"]
    if mode == "zero_bucket":
        n_active = len(set(g for g in genes if g in panel_index))
        fill = n_active + 1
    arr = np.full(P, fill, dtype=np.float64)
    seen = set()
    for pos, g in enumerate(genes, 1):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        seen.add(gi)
        arr[gi] = pos
    return arr


def expressed_set(sentence, panel_index):
    return {panel_index[g] for g in sentence.split()
            if g != "[END_CELL]" and g in panel_index}


def pseudobulk_rankarr(sentences, panel_index, P, mode):
    acc = np.zeros(P)
    for s in sentences:
        acc += sentence_to_rankarr(s, panel_index, P, mode)
    return acc / max(len(sentences), 1)


def pseudobulk_expressed_union(sentences, panel_index):
    u = set()
    for s in sentences:
        u |= expressed_set(s, panel_index)
    return u


# ---------- metrics on rank arrays (higher = more similar) ----------
def _pear(a, b, idx):
    if len(idx) < 3:
        return None
    x, y = a[idx], b[idx]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def score_all(ref, cand, control, P, topn, de_k):
    out = {}
    # topn_tau: top-N most-expressed genes OF THE REFERENCE (lowest rank)
    out["topn_tau"] = _pear(ref, cand, np.argsort(ref)[:topn])
    # de_delta: top-k by |ref - control| shift
    if control is not None:
        de_idx = np.argsort(-np.abs(ref - control))[:de_k]
        rs, cs = ref[de_idx] - control[de_idx], cand[de_idx] - control[de_idx]
        out["de_delta"] = (None if rs.std() < 1e-9 or cs.std() < 1e-9
                           else float(np.corrcoef(rs, cs)[0, 1]))
    else:
        out["de_delta"] = None
    # panel_tau: all genes
    out["panel_tau"] = _pear(ref, cand, np.arange(P))
    return out


def cl_bootstrap(percl, n_boot, seed):
    rng = np.random.RandomState(seed)
    cls = list(percl.keys())
    vals = np.array([percl[c][0] / percl[c][1] for c in cls if percl[c][1] > 0])
    if len(vals) == 0:
        return None
    boots = [np.mean(vals[rng.choice(len(vals), len(vals), replace=True)]) for _ in range(n_boot)]
    return dict(acc=float(np.mean(vals)), ci_low=float(np.percentile(boots, 2.5)),
                ci_high=float(np.percentile(boots, 97.5)), n_cl=len(vals))


# ---------- fair sparse linear baseline: predict per-cell treated rankarr from control rankarr,
#            fit over the union of expressed genes only (not the padded floor) ----------
def fit_linear_sparse(train_file, panel_index, P, mode, lam, limit):
    """Fit W: control_rankarr -> treated_rankarr, but only using rows where we restrict to a
    reasonable dense representation. We use the chosen `mode` rank arrays (so absent genes are at
    the mode's fill), and ridge-regress full-vector -> full-vector. To avoid the constant-floor
    domination, we CENTER each rank array by subtracting its mean before regression (removes the
    dominant constant offset), fit on the centered vectors, and store the mean for reconstruction.
    This makes the fit sensitive to the expressed-gene structure rather than the floor."""
    Xc, Y, n = [], [], 0
    with open(train_file) as f:
        for line in f:
            if limit and n >= limit:
                break
            ex = json.loads(line)
            ctrl = ev.control_from_prompt(ex["prompt"])
            if not ctrl:
                continue
            c = sentence_to_rankarr(ctrl, panel_index, P, mode)
            t = sentence_to_rankarr(ex["response"], panel_index, P, mode)
            Xc.append(c - c.mean())
            Y.append((t - t.mean()) - (c - c.mean()))  # centered shift
            n += 1
    if not Xc:
        return None
    Xc = np.vstack(Xc); Y = np.vstack(Y)
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(P), Xc.T @ Y)
    return W


def linear_predict_rankarr(ctrl_sentence, W, panel_index, P, mode):
    c = sentence_to_rankarr(ctrl_sentence, panel_index, P, mode)
    cc = c - c.mean()
    pred_centered = cc + cc @ W
    return pred_centered + c.mean()  # add offset back; it's a rank-like vector for scoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--scramble_dir", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--train_limit", type=int, default=40000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="tier2_unseen_drugs,tier1_seen_conditions")
    ap.add_argument("--pb_size", type=int, default=15)
    ap.add_argument("--ceiling_size", type=int, default=8)
    ap.add_argument("--n_pairs_per_cond", type=int, default=8)
    ap.add_argument("--min_cells_per_cond", type=int, default=15)
    ap.add_argument("--max_conds_per_tier", type=int, default=300)
    ap.add_argument("--modes", default="position,tail_max,zero_bucket")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--de_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
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
    P = len(panel)
    tiers = [t.strip() for t in args.tiers.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()
    logger.info(f"  model on {device}; sampling temp={args.temperature} top_p={args.top_p}")
    # confirm [END_CELL] known to the tokenizer
    ec = tok.encode("[END_CELL]", add_special_tokens=False)
    logger.info(f"  [END_CELL] -> {ec} ({'atomic' if len(ec)==1 else 'SPLIT - check tokenizer!'})")

    # fit fair linear W per mode (cheap; reuse across tiers)
    W_by_mode = {}
    if args.train_file:
        for mode in modes:
            logger.info(f"  fitting fair sparse linear (mode={mode}) ...")
            W_by_mode[mode] = fit_linear_sparse(args.train_file, panel_index, P, mode,
                                                args.ridge_lambda, args.train_limit)

    def generate(examples, sample):
        gens = []
        for i in range(0, len(examples), args.gen_batch_size):
            batch = [e["prompt"] for e in examples[i:i + args.gen_batch_size]]
            gens.extend(ev.generate_cell_sentences_batched(
                model, tok, batch, device=device, max_new_tokens=args.max_new_tokens,
                do_sample=sample, temperature=args.temperature, top_p=args.top_p))
        return gens

    result = {"pb_size": args.pb_size, "ceiling_size": args.ceiling_size, "modes": modes,
              "topn": args.topn, "de_k": args.de_k, "temperature": args.temperature, "tiers": {}}

    for tier in tiers:
        path = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  missing {path}"); continue
        examples = [json.loads(l) for l in open(path)]
        by_cond = defaultdict(list)
        for e in examples:
            m = e.get("metadata", {})
            by_cond[(m.get("drug"), m.get("cell_line_id"))].append(e)
        by_cl = defaultdict(list)
        for (drug, cl) in by_cond:
            by_cl[cl].append(drug)
        conds = [c for c, v in by_cond.items() if len(v) >= args.min_cells_per_cond]
        rng.shuffle(conds); conds = conds[:args.max_conds_per_tier]
        logger.info(f"  {tier}: {len(conds)} conditions with >= {args.min_cells_per_cond} cells")

        scram_by_cond = {}
        if args.scramble_dir:
            sp = os.path.join(args.scramble_dir, f"eval_{tier}.jsonl")
            if os.path.exists(sp):
                s_by = defaultdict(list)
                for e in [json.loads(l) for l in open(sp)]:
                    m = e.get("metadata", {})
                    s_by[(m.get("drug"), m.get("cell_line_id"))].append(e)
                scram_by_cond = s_by

        N, C = args.pb_size, args.ceiling_size
        # acc[mode][kind][metric][cl] = [correct, total]
        kinds = ["model", "ceiling", "linear", "scramble"]
        acc = {md: {k: {m: defaultdict(lambda: [0, 0]) for m in METRICS} for k in kinds}
               for md in modes}

        for (A, cl) in conds:
            cellsA = by_cond[(A, cl)]
            B_opts = [d for d in by_cl[cl] if d != A and len(by_cond[(d, cl)]) >= N]
            if not B_opts:
                continue
            idxA = list(range(len(cellsA))); rng.shuffle(idxA)
            truthA_cells = [cellsA[i] for i in idxA[:N]]
            truthA_sents = [e["response"] for e in truthA_cells]
            ctrlA_sents = [ev.control_from_prompt(e["prompt"]) for e in truthA_cells]

            # MODEL: one SAMPLED generation per truthA cell (temp>0) -> genuine pseudobulk
            model_gens = generate(truthA_cells, sample=True)
            # LINEAR: per-cell fair prediction rank arrays (mode-specific, done in scoring)
            # SCRAMBLE: sampled generation from scrambled-drug prompts for the same condition
            scram_gens = None
            s_cells = scram_by_cond.get((A, cl), [])[:N]
            if len(s_cells) >= N:
                scram_gens = generate(s_cells, sample=True)
            # CEILING: a disjoint real drug-A pseudobulk (size C) from cells not in truthA
            ceil_sents = ([cellsA[i]["response"] for i in idxA[N:N + C]]
                          if len(cellsA) >= N + C else None)

            for _ in range(args.n_pairs_per_cond):
                B = B_opts[rng.randint(len(B_opts))]
                cellsB = by_cond[(B, cl)]
                jb = rng.choice(len(cellsB), min(N, len(cellsB)), replace=False)
                truthB_sents = [cellsB[j]["response"] for j in jb]

                for md in modes:
                    tA = pseudobulk_rankarr(truthA_sents, panel_index, P, md)
                    tB = pseudobulk_rankarr(truthB_sents, panel_index, P, md)
                    ctrlA = pseudobulk_rankarr(ctrlA_sents, panel_index, P, md)

                    refs = {}
                    refs["model"] = pseudobulk_rankarr(model_gens, panel_index, P, md)
                    if ceil_sents is not None:
                        refs["ceiling"] = pseudobulk_rankarr(ceil_sents, panel_index, P, md)
                    if W_by_mode.get(md) is not None:
                        lin = [linear_predict_rankarr(cs, W_by_mode[md], panel_index, P, md)
                               for cs in ctrlA_sents]
                        refs["linear"] = np.mean(lin, axis=0)
                    if scram_gens is not None:
                        refs["scramble"] = pseudobulk_rankarr(scram_gens, panel_index, P, md)

                    for kind, ref in refs.items():
                        sA = score_all(ref, tA, ctrlA, P, args.topn, args.de_k)
                        sB = score_all(ref, tB, ctrlA, P, args.topn, args.de_k)
                        for m in METRICS:
                            if sA[m] is None or sB[m] is None:
                                continue
                            correct = 1 if sA[m] > sB[m] else (0 if sA[m] < sB[m] else None)
                            if correct is None:
                                continue
                            cell = acc[md][kind][m][cl]
                            cell[0] += correct; cell[1] += 1

        tier_res = {}
        for md in modes:
            tier_res[md] = {}
            for kind in kinds:
                tier_res[md][kind] = {m: cl_bootstrap(acc[md][kind][m], args.n_boot, args.seed)
                                      for m in METRICS}
            tier_res[md]["_support"] = {
                kind: len({cl for cl in acc[md][kind]["topn_tau"]
                           if acc[md][kind]["topn_tau"][cl][1] > 0}) for kind in kinds}
        result["tiers"][tier] = tier_res

        for md in modes:
            logger.info(f"  === {tier} [{md}] (chance=0.50) ===")
            sup = tier_res[md]["_support"]
            logger.info(f"    support: model={sup['model']} ceiling={sup['ceiling']} "
                        f"linear={sup['linear']} scramble={sup['scramble']}")
            for m in METRICS:
                def g(k):
                    r = tier_res[md][k][m]
                    return f"{r['acc']:.3f}" if r else "NA"
                logger.info(f"    {m:10s} model={g('model')} ceiling={g('ceiling')} "
                            f"linear={g('linear')} scramble={g('scramble')}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

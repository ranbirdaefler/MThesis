#!/usr/bin/env python
r"""
residual_eval.py — Arm 1b stage-1 evaluation: does the residual-trained model USE the drug?
============================================================================================
The residual model emits a signed DE signature ("<up> [DOWN] <down> [END_CELL]"), so the standard
nir_benchmark scoring path does not apply (its rank->expression decode is fitted for EXPRESSION ranks
and its truths are full pseudobulks). This scores in RESIDUAL space, which is the space the model was
trained in and — per target_divergence — the space where drug identity actually lives in the tokens.

METRIC (same logic as NIR, in residual space)
  Predicted and true residuals are both mapped to a SIGNED RANK vector (top-k up = positive scores by
  rank, top-k down = negative), so a generated sentence and a continuous truth are compared like for
  like. Similarity = cosine. NIR = fraction of OTHER drugs (same cell line) whose truth is LESS similar
  to the prediction than the drug's own truth. 0.5 = chance.

ARMS
  model    : generate from the real prompt.
  scramble : identical control cell, only the drug name+MoA swapped to another drug in the same cell
             line -> the decisive causal test. model >> scramble = genuine drug use.
  ceiling  : the drug's own half-B residual scoring against half-A truths (achievable bar).
  random   : a shuffled-gene signature (floor).
Clustered 95% CI over CELL LINES (never over drugs -- pseudoreplication).

LEAKAGE NOTE (read before quoting): the residual targets were built from this same cache, so sampled
conditions may have been TRAINED on. `model - scramble` is still a valid test of whether the model USES
the drug token (memorisation also requires reading the drug), but a high absolute model NIR may be
memorisation rather than generalisation. --held_out_only restricts scoring to conditions absent from
the training file, when such conditions exist.

USAGE
  python residual_eval.py --cache_dir /data/.../ot_cache \
      --model_path /data/.../checkpoints/pythia_sft_residual/final \
      --train_file /data/.../residual_targets/residual.jsonl \
      --n_conditions 200 --k_samples 4 --bf16 --out RESULTS/residual_eval.json
  python residual_eval.py --selftest
"""
import argparse, json, os, sys, re, ast, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
END, DOWN = "[END_CELL]", "[DOWN]"
CTRL_MARKER = "Control cell:"


# ----------------------------------------------------------------- signed-rank encoding
def signed_rank_from_sentence(sent, gene_index, P, k=100):
    """generated '<up> [DOWN] <down> [END_CELL]' -> signed rank-score vector."""
    v = np.zeros(P, dtype=np.float32)
    toks = [t for t in sent.replace(END, " ").split() if t]
    if DOWN in toks:
        d = toks.index(DOWN)
        up, dn = toks[:d], toks[d + 1:]
    else:
        up, dn = toks, []
    for blk, sgn in ((up, 1.0), (dn, -1.0)):
        r = 0
        for g in blk:
            gi = gene_index.get(g)
            if gi is None or v[gi] != 0:
                continue
            r += 1
            if r > k:
                break
            v[gi] = sgn / np.log2(r + 1.0)
    return v


def signed_rank_from_vector(res, P, k=100):
    """continuous residual -> the SAME signed rank encoding, so truths and generations are comparable."""
    v = np.zeros(P, dtype=np.float32)
    up = [j for j in np.argsort(-res)[:k].tolist() if res[j] > 0]
    dn = [j for j in np.argsort(res)[:k].tolist() if res[j] < 0]
    for blk, sgn in ((up, 1.0), (dn, -1.0)):
        for r, j in enumerate(blk, 1):
            v[j] = sgn / np.log2(r + 1.0)
    return v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def nir_from_sims(own, others):
    o = [x for x in others if x is not None]
    return float(np.mean([own > x for x in o])) if o else None


# ----------------------------------------------------------------- prompts
def scramble_prompt(prompt, orig_drug, new_drug, new_moa):
    i = prompt.find(CTRL_MARKER)
    if i == -1:
        return None
    pre, rest = prompt[:i], prompt[i:]
    if orig_drug not in pre:
        return None
    pre = pre.replace(orig_drug, new_drug, 1)
    if new_moa:
        pre = re.sub(r"(Mechanism:\s*)([^\n]*)", r"\1" + new_moa.replace("\\", "\\\\"), pre, count=1)
    return pre + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--model_path")
    ap.add_argument("--train_file", default=None, help="residual.jsonl (to flag/exclude trained conditions)")
    ap.add_argument("--held_out_only", action="store_true")
    ap.add_argument("--n_conditions", type=int, default=200)
    ap.add_argument("--k_samples", type=int, default=4)
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_drugs", type=int, default=4)
    ap.add_argument("--truth_repro_thr", type=float, default=0.2,
                    help="reliability bar for the TRUTH residuals we score against (scoring against "
                         "an irreproducible truth is meaningless)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/residual_eval.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args); return
    if not (args.cache_dir and args.model_path):
        ap.error("--cache_dir and --model_path required (unless --selftest)")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ot")
    import build_residual_targets as brt

    panel = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    gene_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    rng = np.random.RandomState(args.seed)

    # residuals + the half-split needed for the ceiling
    # Truths must be RELIABLE or the NIR is scored against noise -> same reproducibility bar as training.
    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control, repro_thr=args.truth_repro_thr,
        seed=args.seed)
    logger.info(f"conditions with residuals: {len(kept)}")
    cvcl, moa_of, conc_of = brt.load_meta_maps()

    trained = set()
    if args.train_file and os.path.exists(args.train_file):
        for line in open(args.train_file):
            m = json.loads(line).get("metadata", {})
            trained.add((m.get("drug"), m.get("cell_line_id"), m.get("plate"), m.get("dose_float")))
        logger.info(f"training file lists {len(trained)} conditions")

    by_cl = defaultdict(list)
    for k in kept:
        by_cl[k[1]].append(k)
    by_cl = {c: ks for c, ks in by_cl.items() if len(ks) >= args.min_drugs}
    truth = {k: signed_rank_from_vector(kept[k]["residual"], P, args.k_sig) for c in by_cl for k in by_cl[c]}

    # model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(dev).eval()
    ec = tok.encode(END, add_special_tokens=False)
    end_id = ec[0] if len(ec) == 1 else tok.convert_tokens_to_ids(END)
    eos = [end_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])

    def generate(prompts):
        prev = tok.padding_side; tok.padding_side = "left"; outs = []
        try:
            for i in range(0, len(prompts), args.gen_batch_size):
                enc = tok(prompts[i:i + args.gen_batch_size], return_tensors="pt", padding=True).to(dev)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tok.pad_token_id, eos_token_id=eos, do_sample=True,
                                       temperature=max(args.temperature, 1e-2), top_p=args.top_p)
                pl = enc["input_ids"].shape[1]
                for j in range(g.shape[0]):
                    ids = g[j][pl:].tolist()
                    if end_id in ids:
                        ids = ids[:ids.index(end_id)]
                    outs.append(tok.decode(ids, skip_special_tokens=False).strip())
        finally:
            tok.padding_side = prev
        return outs

    recs = []
    cond_list = [k for c in by_cl for k in by_cl[c]]
    if args.held_out_only and trained:
        cond_list = [k for k in cond_list if k not in trained]
        logger.info(f"held-out-only: {len(cond_list)} conditions never trained on")
    rng.shuffle(cond_list)
    cond_list = cond_list[:args.n_conditions]

    for n, key in enumerate(cond_list):
        d, c, p, ds = key
        others = [k for k in by_cl[c] if k != key]
        if len(others) < 3:
            continue
        rows = ctrl_rows.get(kept[key]["group"])
        if rows is None or len(rows) == 0:
            continue
        ctrl_vec = np.asarray(X[rows[rng.randint(len(rows))]].todense()).ravel()
        prompt = brt.format_prompt(cvcl.get(c, c), d, brt.parse_dose(conc_of.get(ds, "unknown")),
                                   moa_of.get(d, "unclear"), brt.expr_to_sentence(ctrl_vec, panel))
        # scramble: a different drug in the SAME cell line
        sk = others[rng.randint(len(others))]
        sp = scramble_prompt(prompt, d, sk[0], moa_of.get(sk[0], "unclear"))

        arms = {"model": generate([prompt] * args.k_samples)}
        if sp:
            arms["scramble"] = generate([sp] * args.k_samples)
        row = {"drug": d, "cell_line": c, "plate": p, "trained": key in trained, "swap_to": sk[0]}
        oth_truth = [truth[k] for k in others]
        for arm, gens in arms.items():
            v = np.mean(np.stack([signed_rank_from_sentence(g, gene_index, P, args.k_sig)
                                  for g in gens]), axis=0)
            row[arm] = nir_from_sims(cos(v, truth[key]), [cos(v, t) for t in oth_truth])
        # ceiling (real half-B) and random floor
        rB = signed_rank_from_vector(kept[key]["residual_B"], P, args.k_sig) \
            if "residual_B" in kept[key] else None
        if rB is not None:
            row["ceiling"] = nir_from_sims(cos(rB, truth[key]), [cos(rB, t) for t in oth_truth])
        rv = np.zeros(P, np.float32); idx = rng.choice(P, 2 * args.k_sig, replace=False)
        rv[idx[:args.k_sig]] = 1.0; rv[idx[args.k_sig:]] = -1.0
        row["random"] = nir_from_sims(cos(rv, truth[key]), [cos(rv, t) for t in oth_truth])
        recs.append(row)
        if n % 20 == 0:
            logger.info(f"  {n}/{len(cond_list)} conditions scored")

    if not recs:
        logger.error("no conditions scored"); return
    report(recs, args, rng)


def report(recs, args, rng):
    arms = ["model", "scramble", "ceiling", "random"]
    logger.info("=" * 96)
    logger.info(f"RESIDUAL-SPACE NIR  (chance 0.50)   n={len(recs)} conditions, "
                f"{len(set(r['cell_line'] for r in recs))} cell lines")
    means = {}
    for a in arms:
        v = [r[a] for r in recs if r.get(a) is not None]
        if v:
            means[a] = float(np.mean(v))
            logger.info(f"    {a:9s} {means[a]:.3f}   (n={len(v)})")
    # model - scramble with clustered bootstrap over CELL LINES
    pairs = [(r["cell_line"], r["model"] - r["scramble"]) for r in recs
             if r.get("model") is not None and r.get("scramble") is not None]
    if pairs:
        cls = np.array([c for c, _ in pairs]); dif = np.array([d for _, d in pairs])
        u = list(set(cls.tolist())); boot = []
        for _ in range(args.n_boot):
            take = rng.choice(len(u), len(u), replace=True)
            keep = np.concatenate([np.where(cls == u[t])[0] for t in take])
            boot.append(dif[keep].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        logger.info("-" * 96)
        logger.info(f"    >>> model - scramble = {dif.mean():+.4f}   clustered 95% CI [{lo:+.4f}, {hi:+.4f}]"
                    f"   ({len(u)} cell lines)")
        verdict = ("DRUG USE: swapping the drug degrades the prediction -> the model USES the drug"
                   if lo > 0 else
                   "NULL: swapping the drug changes nothing -> the model does NOT use the drug")
        logger.info(f"    >>> VERDICT: {verdict}")
    tr = [r for r in recs if r.get("trained")]
    ho = [r for r in recs if not r.get("trained")]
    if tr and ho:
        f = lambda s: np.mean([r["model"] for r in s if r.get("model") is not None])
        logger.info(f"    trained conditions n={len(tr)} model={f(tr):.3f} | "
                    f"held-out n={len(ho)} model={f(ho):.3f}  (gap => memorisation)")
    logger.info("=" * 96)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"config": vars(args), "means": means, "records": recs}, open(args.out, "w"),
              indent=2, default=float)
    logger.info(f"-> {args.out}")


def selftest(args):
    """A drug-AWARE synthetic generator must beat its scrambled twin; a drug-agnostic one must not."""
    P, D = 300, 8
    panel = [f"G{i}" for i in range(P)]
    gi = {g: i for i, g in enumerate(panel)}
    rng = np.random.RandomState(0)
    res = {d: rng.randn(P).astype(np.float32) for d in range(D)}
    truth = {d: signed_rank_from_vector(res[d], P, 40) for d in res}
    def sent(d):
        up = [panel[j] for j in np.argsort(-res[d])[:40]]
        dn = [panel[j] for j in np.argsort(res[d])[:40]]
        return " ".join(up) + f" {DOWN} " + " ".join(dn) + f" {END}"
    aware, scram, agn = [], [], []
    fixed = sent(0)
    for d in range(D):
        oth = [truth[o] for o in res if o != d]
        v = signed_rank_from_sentence(sent(d), gi, P, 40)
        aware.append(nir_from_sims(cos(v, truth[d]), [cos(v, t) for t in oth]))
        w = signed_rank_from_sentence(sent((d + 1) % D), gi, P, 40)
        scram.append(nir_from_sims(cos(w, truth[d]), [cos(w, t) for t in oth]))
        f = signed_rank_from_sentence(fixed, gi, P, 40)
        agn.append(nir_from_sims(cos(f, truth[d]), [cos(f, t) for t in oth]))
    a, s, g = np.mean(aware), np.mean(scram), np.mean(agn)
    logger.info(f"  drug-AWARE NIR={a:.3f} (expect ~1)  SCRAMBLED={s:.3f} (expect low)  "
                f"drug-AGNOSTIC={g:.3f} (expect ~chance)")
    ok = a > 0.95 and s < 0.5 and abs(g - 0.5) < 0.25
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

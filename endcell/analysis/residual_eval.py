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


def rank_value_profile(X, rows, P, kmax=400):
    """Empirical log1p(CP10K) value as a function of expression rank, from REAL cells. Lets us decode an
    ordinary cell sentence into the cache's units so a cell-sentence model can be scored in the SAME
    residual space as the residual model (the apples-to-apples control)."""
    import numpy as _np
    acc = _np.zeros(kmax, dtype=_np.float64); n = 0
    for i in rows:
        v = _np.asarray(X[i].todense()).ravel()
        s = _np.sort(v)[::-1][:kmax]
        acc[:len(s)] += s; n += 1
    return (acc / max(1, n)).astype(np.float32)


def sentence_to_expression(sent, gene_index, P, rank_prof):
    """ordinary '<genes ranked by expression> [END_CELL]' -> expression vector in cache units."""
    v = np.zeros(P, dtype=np.float32)
    r = 0
    for g in sent.replace(END, " ").replace(DOWN, " ").split():
        gi = gene_index.get(g)
        if gi is None or v[gi] != 0:
            continue
        if r >= len(rank_prof):
            break
        v[gi] = rank_prof[r]; r += 1
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
    ap.add_argument("--holdout", default=None,
                    help="holdout.json manifest from build_residual_targets -> report train / "
                         "unseen_combo (cross-context transfer) / unseen_drug separately")
    ap.add_argument("--held_out_only", action="store_true")
    ap.add_argument("--min_split_n", type=int, default=40,
                    help="minimum scorable conditions before a split gets a verdict (a clustered CI over "
                         "a handful of points can exclude zero by chance)")
    ap.add_argument("--min_split_cell_lines", type=int, default=10)
    ap.add_argument("--n_conditions", type=int, default=200)
    ap.add_argument("--k_samples", type=int, default=4)
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_drugs", type=int, default=4)
    ap.add_argument("--model_kind", choices=["residual", "cellsentence"], default="residual",
                    help="'residual' = model emits a signed [DOWN] signature (Arm 1b). 'cellsentence' = "
                         "an ORDINARY model (single-cell / consensus / OT): its generated treated cell is "
                         "decoded and converted to an IMPLIED residual (minus control, minus generic), so "
                         "old models are scored in the SAME space -- the apples-to-apples control.")
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

    split_of = {}
    if args.holdout and os.path.exists(args.holdout):
        hm = json.load(open(args.holdout))
        for kstr, v in hm["split"].items():
            split_of[tuple(kstr.split("|"))] = v
        from collections import Counter as _C
        logger.info(f"holdout manifest: {dict(_C(split_of.values()))}  "
                    f"({len(hm.get('holdout_drugs', []))} unseen drugs, "
                    f"{len(hm.get('holdout_combos', []))} unseen combos)")

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
    # CEILING must use DISJOINT halves: half-B scored against half-A truths. (Scoring half-B against the
    # FULL residual is self-overlapping -- that bug returned a meaningless ceiling of 1.000.)
    truth_A = {k: signed_rank_from_vector(kept[k]["residual_A"], P, args.k_sig)
               for c in by_cl for k in by_cl[c] if "residual_A" in kept[k]}

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

    # for the cell-sentence control we need: an expression rank->value profile, each condition's control
    # pseudobulk, and the per-cell-line generic shift (to convert a generated cell into a residual)
    rank_prof, ctrl_pb = None, {}
    if args.model_kind == "cellsentence":
        allrows = np.concatenate([r for r in ctrl_rows.values()])[:4000]
        rank_prof = rank_value_profile(X, allrows, P)
        for c in by_cl:
            for k in by_cl[c]:
                rows = ctrl_rows[kept[k]["group"]]
                ctrl_pb[k] = np.asarray(np.log1p(X[rows].todense()).mean(0)).ravel().astype(np.float32)
        logger.info("cell-sentence control mode: generations decoded -> implied residual "
                    "(minus control pseudobulk, minus per-cell-line generic shift)")

    # STRATIFIED SCRAMBLE PARTNERS. A random other drug may be a near-twin, in which case an unchanged
    # output is CORRECT rather than blind -- that biases model-scramble toward zero. We therefore swap to
    # three defined strata by residual cosine within the cell line:
    #   near     = most SIMILAR drug      (weakest test; a small gap here is expected and fine)
    #   orth     = most ORTHOGONAL (cos~0, unrelated program)
    #   opposite = most ANTI-correlated   (sharpest test: the drug with the opposite signature)
    # If the model truly reads the drug, the gap should GROW from near -> orth -> opposite.
    partners = {}
    for c, ks in by_cl.items():
        for a in ks:
            cs = [(cos(kept[a]["residual"], kept[b]["residual"]), b) for b in ks if b != a]
            if len(cs) < 3:
                continue
            partners[a] = {"near": max(cs)[1],
                           "orth": min(cs, key=lambda t: abs(t[0]))[1],
                           "opposite": min(cs)[1],
                           "cos_near": max(cs)[0],
                           "cos_orth": min(cs, key=lambda t: abs(t[0]))[0],
                           "cos_opposite": min(cs)[0]}
    logger.info(f"stratified scramble partners built for {len(partners)} conditions")

    # ---------------- BASELINES in residual space (model-vs-scramble alone says the model REACTS to the
    # drug token, not that it is any good). Each is a predictor of the drug-specific residual:
    #   drug_lookup : this drug's mean residual measured in OTHER cell lines -> the cross-context lookup
    #                 table. THE bar to beat: a lookup can memorise a drug, only a model can adapt it.
    #   moa_lookup  : mean residual of OTHER drugs sharing this drug's MoA in the same cell line.
    #                 Diagnostic for the MoA leak: our prompt contains 'Mechanism: {moa}', so if the model
    #                 merely reads the MoA it should not beat this.
    #   control_copy: predicting the control cell -> residual = -generic (constant) -> chance by
    #                 construction. Confirms the residual frame is leak-proof (control-copy scores 0.766
    #                 in full-profile space).
    #   generic     : predicting the average drug response -> residual = 0 -> chance by construction.
    by_drug_all = defaultdict(list)
    for k in kept:
        by_drug_all[k[0]].append(k)

    def bl_drug_lookup(key):
        d, c = key[0], key[1]
        oth = [kept[k2]["residual"] for k2 in by_drug_all[d] if k2[1] != c]
        return np.mean(np.stack(oth), 0) if oth else None

    def bl_moa_lookup(key):
        d, c = key[0], key[1]
        m = moa_of.get(d)
        if not m or m in ("unclear", "unknown", "nan", "None"):
            return None
        same = [kept[k2]["residual"] for k2 in by_cl.get(c, [])
                if k2[0] != d and moa_of.get(k2[0]) == m]
        return np.mean(np.stack(same), 0) if same else None

    gen_stats = {"n": 0, "has_down": 0, "up_len": [], "dn_len": [], "valid_frac": [], "dup_frac": []}

    def track(gens):
        for g in gens:
            toks = [t for t in g.replace(END, " ").split() if t]
            gen_stats["n"] += 1
            if DOWN in toks:
                gen_stats["has_down"] += 1
                d = toks.index(DOWN)
                gen_stats["up_len"].append(d); gen_stats["dn_len"].append(len(toks) - d - 1)
            genes = [t for t in toks if t != DOWN]
            if genes:
                gen_stats["valid_frac"].append(np.mean([t in gene_index for t in genes]))
                gen_stats["dup_frac"].append(1.0 - len(set(genes)) / len(genes))

    pred_by_cl = defaultdict(list)   # for the mode-collapse check
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
        # scramble arms across SIMILARITY STRATA (see partners{} above)
        if key not in partners:
            continue
        pinfo = partners[key]
        arms = {"model": generate([prompt] * args.k_samples)}
        for strat in ("near", "orth", "opposite"):
            bkey = pinfo[strat]
            sp = scramble_prompt(prompt, d, bkey[0], moa_of.get(bkey[0], "unclear"))
            if sp:
                arms[f"scramble_{strat}"] = generate([sp] * args.k_samples)
        for g in arms.values():
            track(g)
        row = {"drug": d, "cell_line": c, "plate": p, "trained": key in trained,
               "split": split_of.get(tuple(map(str, key)), "unknown"),
               "repro_cos": kept[key]["repro_cos"],
               "swap_near": pinfo["near"][0], "swap_orth": pinfo["orth"][0],
               "swap_opposite": pinfo["opposite"][0],
               "cos_near": pinfo["cos_near"], "cos_orth": pinfo["cos_orth"],
               "cos_opposite": pinfo["cos_opposite"]}
        oth_truth = [truth[k] for k in others]
        for arm, gens in arms.items():
            if args.model_kind == "residual":
                v = np.mean(np.stack([signed_rank_from_sentence(g, gene_index, P, args.k_sig)
                                      for g in gens]), axis=0)
            else:
                # ordinary model: decode the generated treated cell, subtract control + generic ->
                # implied residual, then the SAME signed-rank encoding as the residual model
                expr = np.mean(np.stack([sentence_to_expression(g, gene_index, P, rank_prof)
                                         for g in gens]), axis=0)
                v = signed_rank_from_vector(expr - ctrl_pb[key] - generic[c], P, args.k_sig)
            row[arm] = nir_from_sims(cos(v, truth[key]), [cos(v, t) for t in oth_truth])
            if arm == "model":
                row["cos_pred_truth"] = cos(v, truth[key])          # direct directional agreement
                row["cos_pred_others"] = float(np.mean([cos(v, t) for t in oth_truth]))
                pred_by_cl[c].append((key, v))                      # for the mode-collapse check
        # CEILING: real half-B replicate scored against DISJOINT half-A truths (no cell overlap)
        if "residual_B" in kept[key] and key in truth_A:
            rB = signed_rank_from_vector(kept[key]["residual_B"], P, args.k_sig)
            othA = [truth_A[k] for k in others if k in truth_A]
            if othA:
                row["ceiling"] = nir_from_sims(cos(rB, truth_A[key]), [cos(rB, t) for t in othA])
        rv = np.zeros(P, np.float32); idx = rng.choice(P, 2 * args.k_sig, replace=False)
        rv[idx[:args.k_sig]] = 1.0; rv[idx[args.k_sig:]] = -1.0
        row["random"] = nir_from_sims(cos(rv, truth[key]), [cos(rv, t) for t in oth_truth])
        # ---- BASELINES (scored identically: same truths, same NIR, same comparison set) ----
        for bname, bvec in (("drug_lookup", bl_drug_lookup(key)),
                            ("moa_lookup", bl_moa_lookup(key)),
                            ("control_copy", -generic[c]),
                            ("generic", np.zeros(P, np.float32))):
            if bvec is None:
                continue
            bs = signed_rank_from_vector(np.asarray(bvec, dtype=np.float32), P, args.k_sig)
            row[bname] = nir_from_sims(cos(bs, truth[key]), [cos(bs, t) for t in oth_truth])
        recs.append(row)
        if n % 20 == 0:
            logger.info(f"  {n}/{len(cond_list)} conditions scored")

    if not recs:
        logger.error("no conditions scored"); return
    report(recs, args, rng, gen_stats, pred_by_cl, kept)


def _clustered_ci(recs, key_a, key_b, rng, n_boot):
    pairs = [(r["cell_line"], r[key_a] - r[key_b]) for r in recs
             if r.get(key_a) is not None and r.get(key_b) is not None]
    if not pairs:
        return None
    cls = np.array([c for c, _ in pairs]); dif = np.array([d for _, d in pairs])
    u = list(set(cls.tolist())); boot = []
    for _ in range(n_boot):
        take = rng.choice(len(u), len(u), replace=True)
        boot.append(dif[np.concatenate([np.where(cls == u[t])[0] for t in take])].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(dif.mean()), float(lo), float(hi), len(pairs), len(u)


def report(recs, args, rng, gen_stats=None, pred_by_cl=None, kept=None):
    logger.info("=" * 100)
    logger.info(f"RESIDUAL-SPACE NIR  (chance 0.50)   n={len(recs)} conditions, "
                f"{len(set(r['cell_line'] for r in recs))} cell lines   model_kind={args.model_kind}")

    # --- (0) GENERATION VALIDITY: if the outputs are malformed, nothing below is meaningful ---
    if gen_stats and gen_stats["n"]:
        g = gen_stats
        logger.info(f"  [validity] {g['n']} generations | has [DOWN] {100*g['has_down']/g['n']:.0f}% | "
                    f"up-block {np.mean(g['up_len']) if g['up_len'] else 0:.0f} / "
                    f"down-block {np.mean(g['dn_len']) if g['dn_len'] else 0:.0f} genes | "
                    f"valid panel genes {100*np.mean(g['valid_frac']):.1f}% | "
                    f"duplicates {100*np.mean(g['dup_frac']):.1f}%")

    means = {}
    for a in ("ceiling", "model", "scramble_near", "scramble_orth", "scramble_opposite",
              "drug_lookup", "moa_lookup", "control_copy", "generic", "random"):
        v = [r[a] for r in recs if r.get(a) is not None]
        if v:
            means[a] = float(np.mean(v))
            tag = ""
            if a == "drug_lookup":
                tag = "   <- THE BAR: a lookup memorises a drug; only a model can adapt it"
            elif a == "moa_lookup":
                tag = "   <- MoA-leak diagnostic (our prompt contains 'Mechanism:')"
            elif a in ("control_copy", "generic"):
                tag = "   <- must be ~0.50 (drug-agnostic by construction)"
            logger.info(f"    {a:18s} {means[a]:.3f}   (n={len(v)}){tag}")

    # head-to-head against each baseline, clustered CI -- the "is the model any GOOD" question,
    # which is separate from "does the model react to the drug token" (model - scramble).
    logger.info("-" * 100)
    logger.info("  model - BASELINE (clustered CI over cell lines). Positive => the model beats a")
    logger.info("  predictor that needs no generation at all:")
    base_out = {}
    for b in ("drug_lookup", "moa_lookup", "control_copy", "generic"):
        r = _clustered_ci(recs, "model", b, rng, args.n_boot)
        if r:
            m, lo, hi, n, ncl = r
            base_out[b] = {"gap": m, "ci": [lo, hi], "n": n}
            logger.info(f"    model - {b:14s} {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  n={n}  "
                        f"{'model WINS' if lo > 0 else ('model LOSES' if hi < 0 else 'tie')}")
    means["vs_baselines"] = base_out

    # --- (1) STRATIFIED model - scramble: the gap should GROW near -> orth -> opposite ---
    logger.info("-" * 100)
    logger.info("  model - scramble BY SIMILARITY STRATUM (a random swap can land on a near-twin, where")
    logger.info("  an unchanged output is CORRECT; the opposite-signature swap is the sharpest test):")
    strat_out = {}
    for strat in ("near", "orth", "opposite"):
        r = _clustered_ci(recs, "model", f"scramble_{strat}", rng, args.n_boot)
        if r:
            m, lo, hi, n, ncl = r
            mc = np.mean([x[f"cos_{strat}"] for x in recs if x.get(f"cos_{strat}") is not None])
            strat_out[strat] = {"gap": m, "ci": [lo, hi], "n": n, "mean_cos": float(mc)}
            flag = "CI excludes 0" if lo > 0 else "CI spans 0"
            logger.info(f"    {strat:9s} (mean cos(A,B)={mc:+.2f})  gap = {m:+.4f}  "
                        f"CI [{lo:+.4f}, {hi:+.4f}]  {flag}")
    if len(strat_out) == 3:
        mono = strat_out["opposite"]["gap"] > strat_out["near"]["gap"]
        logger.info(f"    >>> gap grows with dissimilarity: {'YES' if mono else 'NO'} "
                    f"(near {strat_out['near']['gap']:+.4f} -> opposite {strat_out['opposite']['gap']:+.4f}). "
                    f"{'Consistent with genuine drug use.' if mono else 'A flat profile is a red flag.'}")
        best = strat_out["opposite"]
        logger.info(f"    >>> HEADLINE (opposite-signature swap): {best['gap']:+.4f} "
                    f"CI [{best['ci'][0]:+.4f}, {best['ci'][1]:+.4f}] -> "
                    f"{'DRUG USE' if best['ci'][0] > 0 else 'NULL'}")

    # --- (2) MODE COLLAPSE: are predictions for different drugs actually different? ---
    if pred_by_cl and kept:
        pc, tc = [], []
        for c, lst in pred_by_cl.items():
            if len(lst) < 3:
                continue
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    pc.append(cos(lst[i][1], lst[j][1]))
                    tc.append(cos(kept[lst[i][0]]["residual"], kept[lst[j][0]]["residual"]))
        if pc:
            logger.info("-" * 100)
            logger.info(f"  [diversity] mean pairwise cos between PREDICTIONS for different drugs = "
                        f"{np.mean(pc):+.3f}  vs between TRUTHS = {np.mean(tc):+.3f}")
            logger.info(f"              (predictions ~1.0 => mode collapse: the model emits one profile "
                        f"regardless of drug, which would make any gap trivially ~0)")

    # --- (3) INSTRUMENT VALIDITY: does the model do better where the truth is more reproducible? ---
    rc = [(r["repro_cos"], r["model"]) for r in recs
          if r.get("repro_cos") is not None and r.get("model") is not None]
    if len(rc) > 10:
        a = np.array([x for x, _ in rc]); b = np.array([y for _, y in rc])
        logger.info(f"  [validity] corr(condition reproducibility, model NIR) = {np.corrcoef(a, b)[0,1]:+.3f} "
                    f"(positive => the model does better where the drug effect is real)")
    cpt = [r["cos_pred_truth"] for r in recs if r.get("cos_pred_truth") is not None]
    cpo = [r["cos_pred_others"] for r in recs if r.get("cos_pred_others") is not None]
    if cpt:
        logger.info(f"  [direction] cos(prediction, OWN truth) = {np.mean(cpt):+.3f}  vs "
                    f"cos(prediction, OTHER drugs) = {np.mean(cpo):+.3f}")

    # --- (4) GENERALIZATION: the three-way split (the decisive test) ---
    splits = defaultdict(list)
    for r in recs:
        splits[r.get("split", "unknown")].append(r)
    if len(splits) > 1 or "unknown" not in splits:
        logger.info("-" * 100)
        logger.info("  GENERALIZATION — opposite-signature gap by split "
                    "(train = memorisation; unseen_combo = CROSS-CONTEXT TRANSFER, the real test;")
        logger.info("  unseen_drug = no drug information at all, expected ~0 given the SAR gate):")
        gen_out = {}
        # A clustered bootstrap over a handful of points produces a wide CI that can exclude zero by
        # chance. Require enough conditions AND cell lines before reporting a verdict at all -- a split
        # with n=6 is UNDERPOWERED, which is a different statement from "no effect".
        MIN_N, MIN_CL = args.min_split_n, args.min_split_cell_lines
        for name in ("train", "unseen_combo", "unseen_drug", "unknown"):
            sub = splits.get(name)
            if not sub:
                continue
            n_cl_sub = len({r["cell_line"] for r in sub})
            if len(sub) < MIN_N or n_cl_sub < MIN_CL:
                logger.info(f"    {name:14s} n={len(sub):4d} ({n_cl_sub:2d} cell lines)  "
                            f"UNDERPOWERED (need n>={MIN_N} and >={MIN_CL} cell lines) -- NO VERDICT")
                gen_out[name] = {"n": len(sub), "n_cell_lines": n_cl_sub, "underpowered": True}
                continue
            r = _clustered_ci(sub, "model", "scramble_opposite", rng, args.n_boot)
            mo = np.mean([x["model"] for x in sub if x.get("model") is not None])
            if r:
                m, lo, hi, n, ncl = r
                gen_out[name] = {"gap": m, "ci": [lo, hi], "n": n, "n_cell_lines": ncl, "model_nir": float(mo)}
                logger.info(f"    {name:14s} n={n:4d} ({ncl:2d} cell lines)  model NIR={mo:.3f}  "
                            f"gap = {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
                            f"{'USES DRUG' if lo > 0 else 'null'}")
        g = gen_out.get("unseen_combo")
        if g and g.get("underpowered"):
            logger.info(f"    >>> CROSS-CONTEXT TRANSFER: UNTESTED -- only {g['n']} scorable held-out "
                        f"combos ({g['n_cell_lines']} cell lines). This is NOT a negative result; the "
                        f"split is too small. Rebuild with more held-out (drug, cell_line) pairs.")
        elif g:
            logger.info(f"    >>> CROSS-CONTEXT TRANSFER: {'YES' if g['ci'][0] > 0 else 'NO'} "
                        f"({g['gap']:+.4f} CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], n={g['n']}). "
                        + ("The model applies a learned drug signature in a cell line it never saw it in."
                           if g['ci'][0] > 0 else
                           "Drug use does not transfer to new contexts -> memorisation only."))
        if all(k in gen_out and "gap" in gen_out[k] for k in ("train", "unseen_combo")):
            logger.info(f"    >>> memorisation premium: train {gen_out['train']['gap']:+.4f} vs "
                        f"unseen_combo {gen_out['unseen_combo']['gap']:+.4f}")
        means["generalization"] = gen_out
    else:
        tr = [r for r in recs if r.get("trained")]
        ho = [r for r in recs if not r.get("trained")]
        if tr and ho:
            f = lambda s: np.mean([r["model"] for r in s if r.get("model") is not None])
            logger.info(f"  [memorisation] trained n={len(tr)} model={f(tr):.3f} | "
                        f"held-out n={len(ho)} model={f(ho):.3f}")
    logger.info("=" * 100)
    means["strata"] = strat_out
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

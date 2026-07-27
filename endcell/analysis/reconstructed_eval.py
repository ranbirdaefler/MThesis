#!/usr/bin/env python
r"""
reconstructed_eval.py — score the residual model in the STANDARD full-profile space
====================================================================================
residual_eval.py found `model - scramble = +0.036` [+0.003, +0.072] in RESIDUAL space -- the first
non-null in this project. But every prior arm (single-cell +0.014, consensus -0.010, OT/T2 -0.001) was
measured in FULL-PROFILE expression space, so that comparison changes both the target AND the metric
and cannot support "residual training fixed drug-blindness" on its own.

This closes that gap: the residual model's generated signature is RECONSTRUCTED into a full expression
profile and scored the same way every earlier arm was.

    predicted_treated = control_pseudobulk + generic_shift(cell_line) + alpha * predicted_residual

  * control_pseudobulk : the condition's own plate-matched control (as in nir_benchmark)
  * generic_shift      : per-cell-line mean-over-drugs shift, from TRAIN data (reconstruction.npz)
  * predicted_residual : the model's signature, mapped back to a magnitude vector by the rank->value
                         profile of real residuals (so units are comparable), scaled by alpha.

FAIRNESS (state when quoting): the reconstruction hands the model the control and the generic program.
Those are IDENTICAL across the model and scramble arms and shared by all drugs in a cell line, so they
CANCEL in `model - scramble` and in NIR discrimination -- but they inflate absolute profile similarity,
so absolute numbers are NOT comparable across arms. Anchor to `model - scramble`.

Arms: model / scramble (same control, wrong drug) / control-copy (zero-drug-info leak baseline) /
generic (control + generic shift, drug-agnostic) / ceiling (real half-B pseudobulk). Clustered CI over
cell lines.

USAGE
  python reconstructed_eval.py --cache_dir ... --model_path ... --recon /data/.../residual_targets/reconstruction.npz \
      --n_conditions 200 --k_samples 4 --bf16 --out RESULTS/reconstructed_eval.json
  python reconstructed_eval.py --selftest
"""
import argparse, json, os, sys, re, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
END, DOWN = "[END_CELL]", "[DOWN]"
CTRL_MARKER = "Control cell:"


def sentence_to_residual_magnitude(sent, gene_index, P, rank_profile_up, rank_profile_dn):
    """Generated '<up> [DOWN] <down>' -> a MAGNITUDE residual vector. Position r in a block is assigned
    the empirical value that real residuals have at rank r (rank_profile_*), so the reconstruction is in
    the right units instead of arbitrary rank scores."""
    v = np.zeros(P, dtype=np.float32)
    toks = [t for t in sent.replace(END, " ").split() if t]
    if DOWN in toks:
        d = toks.index(DOWN); up, dn = toks[:d], toks[d + 1:]
    else:
        up, dn = toks, []
    for blk, prof, sgn in ((up, rank_profile_up, 1.0), (dn, rank_profile_dn, -1.0)):
        r = 0
        for g in blk:
            gi = gene_index.get(g)
            if gi is None or v[gi] != 0:
                continue
            if r >= len(prof):
                break
            v[gi] = sgn * abs(prof[r]); r += 1
    return v


def rank_value_profiles(residuals, k):
    """Empirical |residual| as a function of within-block rank, averaged over real conditions."""
    ups, dns = [], []
    for res in residuals:
        s = np.sort(res)[::-1]
        ups.append(s[:k])
        dns.append(np.abs(np.sort(res)[:k]))
    return np.mean(np.stack(ups), 0), np.mean(np.stack(dns), 0)


def nir_from_dists(own, others):
    o = [x for x in others if x is not None]
    return float(np.mean([own < x for x in o])) if o else None


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
    ap.add_argument("--recon", default=None, help="reconstruction.npz (generic shift per cell line)")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale on the predicted residual")
    ap.add_argument("--n_conditions", type=int, default=200)
    ap.add_argument("--k_samples", type=int, default=4)
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_drugs", type=int, default=4)
    ap.add_argument("--truth_repro_thr", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/reconstructed_eval.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not (args.cache_dir and args.model_path):
        ap.error("--cache_dir and --model_path required (unless --selftest)")

    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.join(os.path.dirname(here), "ot"), os.path.dirname(here)):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    import build_residual_targets as brt

    panel = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    gene_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    rng = np.random.RandomState(args.seed)

    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control, args.truth_repro_thr, args.seed)
    cvcl, moa_of, conc_of = brt.load_meta_maps()
    by_cl = defaultdict(list)
    for k in kept:
        by_cl[k[1]].append(k)
    by_cl = {c: ks for c, ks in by_cl.items() if len(ks) >= args.min_drugs}
    logger.info(f"{sum(len(v) for v in by_cl.values())} conditions across {len(by_cl)} cell lines")

    # rank->value profile so generated signatures reconstruct with realistic magnitudes
    up_prof, dn_prof = rank_value_profiles([kept[k]["residual"] for c in by_cl for k in by_cl[c]],
                                           args.k_sig)
    # TRUE full treated profile = control + shift ; and its half-B replicate for the ceiling
    truth_full, truth_B, ctrl_pb = {}, {}, {}
    for c in by_cl:
        for k in by_cl[c]:
            g = kept[k]["group"]
            rows = ctrl_rows[g]
            cp = np.asarray(np.log1p(X[rows].todense()).mean(0)).ravel().astype(np.float32)
            ctrl_pb[k] = cp
            truth_full[k] = cp + generic[c] + kept[k]["residual"]
            if "residual_B" in kept[k]:
                truth_B[k] = cp + generic[c] + kept[k]["residual_B"]

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

    cond = [k for c in by_cl for k in by_cl[c]]
    rng.shuffle(cond); cond = cond[:args.n_conditions]
    recs = []
    for n, key in enumerate(cond):
        d, c, p, ds = key
        others = [k for k in by_cl[c] if k != key]
        if len(others) < 3:
            continue
        rows = ctrl_rows[kept[key]["group"]]
        cell_vec = np.asarray(X[rows[rng.randint(len(rows))]].todense()).ravel()
        prompt = brt.format_prompt(cvcl.get(c, c), d, brt.parse_dose(conc_of.get(ds, "unknown")),
                                   moa_of.get(d, "unclear"), brt.expr_to_sentence(cell_vec, panel))
        sk = others[rng.randint(len(others))]
        sp = scramble_prompt(prompt, d, sk[0], moa_of.get(sk[0], "unclear"))

        base = ctrl_pb[key] + generic[c]                   # control + generic (drug-agnostic scaffold)
        oth = [truth_full[k] for k in others]
        row = {"drug": d, "cell_line": c, "plate": p, "swap_to": sk[0]}

        def score(vec):
            return nir_from_dists(float(np.linalg.norm(vec - truth_full[key])),
                                  [float(np.linalg.norm(vec - t)) for t in oth])
        for arm, pr in (("model", prompt), ("scramble", sp)):
            if pr is None:
                continue
            r = np.mean(np.stack([sentence_to_residual_magnitude(g, gene_index, P, up_prof, dn_prof)
                                  for g in generate([pr] * args.k_samples)]), axis=0)
            row[arm] = score(base + args.alpha * r)
        row["control_copy"] = score(ctrl_pb[key])          # zero drug info (the historic 0.766 leak)
        row["generic"] = score(base)                       # drug-agnostic scaffold alone
        if key in truth_B:
            row["ceiling"] = score(truth_B[key])
        recs.append(row)
        if n % 20 == 0:
            logger.info(f"  {n}/{len(cond)} conditions scored")

    if not recs:
        logger.error("no conditions scored"); return
    logger.info("=" * 96)
    logger.info(f"RECONSTRUCTED FULL-PROFILE NIR (chance 0.50)  n={len(recs)}, "
                f"{len(set(r['cell_line'] for r in recs))} cell lines")
    means = {}
    for a in ("model", "scramble", "control_copy", "generic", "ceiling"):
        v = [r[a] for r in recs if r.get(a) is not None]
        if v:
            means[a] = float(np.mean(v))
            logger.info(f"    {a:13s} {means[a]:.3f}  (n={len(v)})")
    pairs = [(r["cell_line"], r["model"] - r["scramble"]) for r in recs
             if r.get("model") is not None and r.get("scramble") is not None]
    if pairs:
        cls = np.array([c for c, _ in pairs]); dif = np.array([v for _, v in pairs])
        u = list(set(cls.tolist())); boot = []
        for _ in range(args.n_boot):
            take = rng.choice(len(u), len(u), replace=True)
            boot.append(dif[np.concatenate([np.where(cls == u[t])[0] for t in take])].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        logger.info("-" * 96)
        logger.info(f"    >>> model - scramble = {dif.mean():+.4f}  clustered 95% CI [{lo:+.4f}, {hi:+.4f}]"
                    f"  ({len(u)} cell lines)")
        logger.info(f"    >>> {'DRUG USE (CI excludes 0)' if lo > 0 else 'NULL (CI spans 0)'} "
                    f"-- directly comparable to prior arms: single-cell +0.014, consensus -0.010, OT -0.001")
        logger.info(f"    NOTE: absolute values are inflated by the reconstruction scaffold "
                    f"(control+generic handed to the model); anchor to the DIFFERENCE.")
    logger.info("=" * 96)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"config": vars(args), "means": means, "records": recs}, open(args.out, "w"),
              indent=2, default=float)
    logger.info(f"-> {args.out}")


def selftest():
    """A drug-aware signature must reconstruct closer to its own truth than a scrambled one."""
    P, K, D = 200, 20, 6
    panel = [f"G{i}" for i in range(P)]
    gi = {g: i for i, g in enumerate(panel)}
    rng = np.random.RandomState(0)
    res = {d: rng.randn(P).astype(np.float32) for d in range(D)}
    up_prof, dn_prof = rank_value_profiles(list(res.values()), K)
    base = np.abs(rng.rand(P)).astype(np.float32) * 3
    truth = {d: base + res[d] for d in res}
    def sent(d):
        up = [panel[j] for j in np.argsort(-res[d])[:K]]
        dn = [panel[j] for j in np.argsort(res[d])[:K]]
        return " ".join(up) + f" {DOWN} " + " ".join(dn) + f" {END}"
    aw, sc = [], []
    for d in range(D):
        oth = [truth[o] for o in res if o != d]
        for lab, s in (("aw", sent(d)), ("sc", sent((d + 1) % D))):
            v = base + sentence_to_residual_magnitude(s, gi, P, up_prof, dn_prof)
            n = nir_from_dists(float(np.linalg.norm(v - truth[d])),
                               [float(np.linalg.norm(v - t)) for t in oth])
            (aw if lab == "aw" else sc).append(n)
    a, s = np.mean(aw), np.mean(sc)
    logger.info(f"  reconstructed: drug-AWARE NIR={a:.3f} (expect high)  SCRAMBLED={s:.3f} (expect lower)")
    ok = a > 0.8 and a > s + 0.2
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

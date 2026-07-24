#!/usr/bin/env python
r"""
scramble_distance_sweep.py — is the scramble null real, or an artifact of swapping to SIMILAR drugs?
====================================================================================================
The scramble ablation swaps the drug token to a DIFFERENT drug and asks whether the output changes;
`model ~= scramble` is read as "the model ignores the drug". But if the swapped-in drug has a SIMILAR
true response, producing the same output is CORRECT, not blind — and Q11 showed MoA barely predicts
response (within-MoA dist ~= between-MoA), so the existing MoA-based scramble does NOT control for
response similarity. This script turns the single test into a CURVE: model - scramble as a function of
the RESPONSE DISTANCE between the real drug and the swapped-in drug.

  * If model - scramble stays ~0 across the whole distance range (incl. the FAR tail) -> drug-blindness
    is confirmed at the hardest setting (swapping to a genuinely different drug still changes nothing).
  * If model - scramble RISES with distance -> the model DOES discriminate dissimilar drugs and the
    flat aggregate was diluted by near-twin swaps (a real correction to the claim).

METHOD (within-plate, reusing the NIR machinery):
  per (cell_line, plate) group -> split each drug's cells into half-A (truth) / half-B (ceiling);
  truth_expr[d] = pb_expr(half-A). Drug-drug response distance D(A,B) = ||truth_expr_A - truth_expr_B||
  (the control cancels -> this is exactly the NIR 'other-drug' distance). For each sampled real drug A:
    - model:    generate from A's real prompts -> NIR vs truth (this is A's model score).
    - scramble: for B's spanning the distance quantiles from A, rebuild A's prompt with B's drug
                name + MoA, generate, and score NIR against A's OWN truth. Record (D(A,B), scramble_NIR).
  Aggregate model - scramble in distance BINS, clustered 95% CI (resample CELL LINES).

USAGE (GPU): drop-in on any endcell checkpoint (single-cell / consensus / OT).
  python scramble_distance_sweep.py --eval_dir DATA_endcell_big --model_path CKPT/final \
     --tier tier2_unseen_drugs --same_plate_only --n_drugs 40 --n_bins 5 --k_samples 8 \
     --bf16 --out RESULTS/scramble_sweep.json
"""
import argparse, json, os, sys, re, logging
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.join(os.path.dirname(os.path.dirname(_HERE)), "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import nir_benchmark as nb          # helpers: pb_expr, pb_rank, nir_from_dists, rank_corr, sentence_*
import evaluate_c2s_tahoe as ev     # control_from_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
CTRL_MARKER = "Control cell:"


def scramble_prompt(prompt, orig_drug, new_drug, new_moa):
    """A's prompt -> A's control cell, but B's drug name + MoA in the prefix (same edit as make_scramble)."""
    idx = prompt.find(CTRL_MARKER)
    if idx == -1:
        return None
    prefix, rest = prompt[:idx], prompt[idx:]
    if orig_drug not in prefix:
        return None
    prefix = prefix.replace(orig_drug, new_drug, 1)
    if new_moa is not None:
        prefix = re.sub(r"(Mechanism:\s*)([^\n]*)", r"\1" + new_moa.replace("\\", "\\\\"), prefix, count=1)
    return prefix + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--same_plate_only", action="store_true")
    ap.add_argument("--n_drugs", type=int, default=40, help="real drugs A sampled across groups")
    ap.add_argument("--n_bins", type=int, default=5, help="distance quantile bins to sweep")
    ap.add_argument("--min_cells", type=int, default=8)
    ap.add_argument("--min_drugs_per_cl", type=int, default=4)
    ap.add_argument("--k_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/scramble_sweep.json")
    args = ap.parse_args()

    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    lm = json.load(open(os.path.join(args.eval_dir, "linear_model.json")))
    rng = np.random.RandomState(args.seed)

    # drug -> MoA (for rebuilding the scrambled prompt)
    drug_moa = {}
    for line in open(os.path.join(args.eval_dir, f"eval_{args.tier}.jsonl")):
        m = json.loads(line).get("metadata", {})
        d = m.get("drug")
        if d and d not in drug_moa:
            drug_moa[d] = m.get("moa") or m.get("mechanism")

    # model + generate
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device).eval()
    ec = tok.encode(nb.SENTINEL, add_special_tokens=False)
    end_id = ec[0] if len(ec) == 1 else tok.convert_tokens_to_ids(nb.SENTINEL)
    eos = [end_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])

    def generate(prompts):
        prev = tok.padding_side; tok.padding_side = "left"
        outs = []
        try:
            for i in range(0, len(prompts), args.gen_batch_size):
                enc = tok(prompts[i:i + args.gen_batch_size], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                       do_sample=True, temperature=max(args.temperature, 1e-2), top_p=args.top_p)
                plen = enc["input_ids"].shape[1]
                for j in range(g.shape[0]):
                    ids = g[j][plen:].tolist()
                    if end_id in ids:
                        ids = ids[:ids.index(end_id)]
                    outs.append(tok.decode(ids, skip_special_tokens=True).strip() + " " + nb.SENTINEL)
        finally:
            tok.padding_side = prev
        return outs

    by_cl = nb.load_tier_by_drug(args.eval_dir, args.tier, ev, same_plate=args.same_plate_only)
    if not by_cl:
        logger.error("no eval data"); return

    # collect (distance, model_nir, scramble_nir, cell_line) records
    recs = []
    budget = args.n_drugs
    for gkey, dd in by_cl.items():
        drugs = [d for d, s in dd.items() if len(s["resp"]) >= args.min_cells]
        if len(drugs) < args.min_drugs_per_cl:
            continue
        cl = gkey[0]
        # half-split -> truth (half A); pairwise distances between drug truths
        truth = {}
        for d in drugs:
            resp = dd[d]["resp"]; idx = list(range(len(resp))); rng.shuffle(idx); h = len(idx) // 2
            truth[d] = nb.pb_expr([resp[i] for i in idx[:h]], panel_index, P, lm)
        drugs = [d for d in drugs if d in truth]
        if len(drugs) < args.min_drugs_per_cl:
            continue
        Dmat = {a: {b: float(np.linalg.norm(truth[a] - truth[b])) for b in drugs if b != a} for a in drugs}

        n_here = max(1, min(len(drugs), budget // max(1, (len(by_cl)))) or 1)
        pick_A = [drugs[i] for i in rng.choice(len(drugs), min(len(drugs), max(2, n_here)), replace=False)]
        for A in pick_A:
            others = [d for d in drugs if d != A]
            texpr = np.stack([truth[o] for o in others])
            # model score for A
            mgen = generate(dd[A]["prompts"][:args.k_samples])
            m_expr = nb.pb_expr(mgen, panel_index, P, lm)
            d_own = float(np.linalg.norm(m_expr - truth[A]))
            model_nir = nb.nir_from_dists(d_own, [float(np.linalg.norm(m_expr - t)) for t in texpr])
            # choose B's spanning distance quantiles
            Bs = sorted(others, key=lambda b: Dmat[A][b])
            qs = np.linspace(0, len(Bs) - 1, min(args.n_bins, len(Bs))).round().astype(int)
            for bi in np.unique(qs):
                B = Bs[bi]
                sp = scramble_prompt(dd[A]["prompts"][0], A, B, drug_moa.get(B))
                if sp is None:
                    continue
                sgen = generate([sp] * min(args.k_samples, 4))
                s_expr = nb.pb_expr(sgen, panel_index, P, lm)
                sd_own = float(np.linalg.norm(s_expr - truth[A]))
                scr_nir = nb.nir_from_dists(sd_own, [float(np.linalg.norm(s_expr - t)) for t in texpr])
                if model_nir is not None and scr_nir is not None:
                    recs.append({"cell_line": cl, "drug": A, "swap_to": B, "dist": Dmat[A][B],
                                 "model_nir": model_nir, "scramble_nir": scr_nir,
                                 "model_minus_scramble": model_nir - scr_nir})
            budget -= 1
        if budget <= 0:
            break

    if not recs:
        logger.error("no records collected"); return
    dists = np.array([r["dist"] for r in recs])
    mms = np.array([r["model_minus_scramble"] for r in recs])
    cls = np.array([r["cell_line"] for r in recs])

    # distance bins by quantile of the swap distance
    edges = np.quantile(dists, np.linspace(0, 1, args.n_bins + 1))
    edges[-1] += 1e-9
    logger.info("=" * 92)
    logger.info(f"SCRAMBLE DISTANCE SWEEP — {len(recs)} (drug,swap) pairs, {len(set(cls))} cell lines, tier {args.tier}")
    logger.info(f"  reading: model - scramble should RISE with swap-distance IF the model uses the drug;")
    logger.info(f"           flat ~0 across the FAR tail => drug-blindness confirmed at the hardest setting.")
    bins_out = []
    for k in range(args.n_bins):
        sel = (dists >= edges[k]) & (dists < edges[k + 1])
        if sel.sum() < 3:
            continue
        # clustered bootstrap over cell lines
        ucl = list(set(cls[sel]))
        boot = []
        for _ in range(args.n_boot):
            take = rng.choice(len(ucl), len(ucl), replace=True)
            keep = np.concatenate([np.where(sel & (cls == ucl[t]))[0] for t in take]) if ucl else np.array([], int)
            if len(keep):
                boot.append(mms[keep].mean())
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        row = {"bin": k, "dist_range": [float(edges[k]), float(edges[k + 1])], "n": int(sel.sum()),
               "mean_dist": float(dists[sel].mean()),
               "model_minus_scramble": float(mms[sel].mean()), "ci95": [float(lo), float(hi)],
               "model_nir": float(np.mean([r["model_nir"] for r, s in zip(recs, sel) if s])),
               "scramble_nir": float(np.mean([r["scramble_nir"] for r, s in zip(recs, sel) if s]))}
        bins_out.append(row)
        logger.info(f"  bin {k} dist[{edges[k]:.2f},{edges[k+1]:.2f}] n={row['n']:3d}  "
                    f"model-scramble = {row['model_minus_scramble']:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  "
                    f"(model {row['model_nir']:.3f} | scramble {row['scramble_nir']:.3f})")
    # trend: correlation of (distance, model-scramble)
    if len(recs) > 5:
        r = float(np.corrcoef(dists, mms)[0, 1])
        logger.info(f"  TREND: corr(swap-distance, model-scramble) = {r:+.3f}  "
                    f"(>0 and far-bin CI excluding 0 => model discriminates dissimilar drugs)")
    logger.info("=" * 92)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"config": vars(args), "bins": bins_out, "n_pairs": len(recs),
               "trend_corr": (r if len(recs) > 5 else None), "records": recs},
              open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

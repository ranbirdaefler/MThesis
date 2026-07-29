#!/usr/bin/env python
r"""
reward_calibration.py — Step 0 for the GRPO arm: is the reward worth optimizing? (no GPU, no training)
======================================================================================================
If the reward cannot distinguish a REAL drug-A cell from a REAL drug-B cell, GRPO would be optimizing
noise and no hyperparameter can rescue it. This probes the reward function *exactly as the trainer will
compute it* — including the lossy sentence→expression decode — and reports a set of pass/kill gates.

THE REWARD UNDER TEST (docs/proposals/grpo_training_plan.md §5)
    expr(g)     = decode(sentence)                       # empirical rank->value profile, as in training
    residual(g) = expr(g) - ctrl_pb - generic[cell_line] # drug-specific part
    R_disc      = sim(residual(g), residual_true) - max_j sim(residual(g), residual_other_j)
The subtraction cancels the generic program by construction; the max_j term is what makes it
discriminative rather than a similarity score.

GATES (each is a reason to stop before spending GPU-hours)
  G1 DISCRIMINATION   real own-drug cells must score above real other-drug cells, with a clear margin.
  G2 DECODE COST      the same test with true expression vs after the sentence round-trip. If the decode
                      destroys the margin, the reward is unusable no matter how good the policy is.
  G3 POOLING / K      margin vs number of pooled cells (1,2,4,8,16,40). A generation is ONE cell scored
                      against a 40-cell truth; this sets K (or the pooled-sample size m).
  G4 HACKABILITY      adversarial "generations" (control copy, generic response, orthogonal-to-everything,
                      one canonical maximally-contrastive vector reused for every drug, truncated stub,
                      duplicate spam) must all score BELOW real cells.
  G5 RANK-INVARIANCE  *the sharpest kill condition*. Rank real cells of drug A by reward against A's
                      truth, then re-rank the SAME cells against B's truth. Spearman ~ 1 means the reward
                      is sorting by generic cell quality, not by drug -> STOP. (Q10 showed every
                      rank/correlation metric on this data rewards the generic ordering.)
  G6 RELIABILITY      re-score identical cells against truths built from DISJOINT halves of the treated
                      population; sigma_signal / sigma_noise.
  G7 STRATA           the margin separately for reproducible vs unreproducible conditions — tells us which
                      conditions are trainable and feeds the curriculum.

USAGE
  python reward_calibration.py --selftest
  python reward_calibration.py --cache_dir /data/.../ot_cache --out RESULTS/reward_calibration.json
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
END, DOWN = "[END_CELL]", "[DOWN]"


# ----------------------------------------------------------------- similarity variants
def sim_cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def signed_rank(vec, k=100):
    """Signed rank encoding: top-k up positive, top-k down negative, weight 1/log2(rank+1).
    Cosine on THIS is the 'rank-weighted set overlap' variant."""
    v = np.zeros(len(vec), dtype=np.float32)
    up = [j for j in np.argsort(-vec)[:k].tolist() if vec[j] > 0]
    dn = [j for j in np.argsort(vec)[:k].tolist() if vec[j] < 0]
    for blk, s in ((up, 1.0), (dn, -1.0)):
        for r, j in enumerate(blk, 1):
            v[j] = s / np.log2(r + 1.0)
    return v


SIMS = {
    "cosine": lambda a, b, k=100: sim_cosine(a, b),
    "rank":   lambda a, b, k=100: sim_cosine(signed_rank(a, k), signed_rank(b, k)),
}


def r_disc(res_g, res_own, res_others, sim, k=100, neg_agg="mean", tau=0.1):
    """The contrastive reward. `neg_agg` chooses how the negatives are aggregated:

      max      : own - max_j sim_j          <- ORIGINAL, and BROKEN. The max over J noisy similarities
                 is biased upward (E[max] ~ sigma*sqrt(2 ln J)), which drags every REAL cell's reward
                 negative -- so a zero vector (sim=0 to everything) scores 0 and beats real cells.
                 Measured: generic_response and orthogonal_to_all both hacked this.
      mean     : own - mean_j sim_j         <- unbiased; a real cell has sim_own > 0 ~= mean_others,
                 so it scores positive while a zero vector still scores 0 and loses.
      opposite : own - sim to the single most ANTI-correlated drug (a fixed negative, no max bias)
      infonce  : log softmax(own / tau) over {own} u negatives -- bounded, and a vector equidistant
                 from everything gets the MINIMUM rather than 0.
    """
    own = sim(res_g, res_own, k)
    if not len(res_others):
        return own
    sims = [sim(res_g, o, k) for o in res_others]
    if neg_agg == "max":
        return own - max(sims)
    if neg_agg == "mean":
        return own - float(np.mean(sims))
    if neg_agg == "opposite":
        return own - sims[int(np.argmin([sim(res_own, o, k) for o in res_others]))]
    if neg_agg == "infonce":
        z = np.array([own] + sims, dtype=np.float64) / max(tau, 1e-6)
        z -= z.max()
        return float(z[0] - np.log(np.exp(z).sum()))
    raise ValueError(neg_agg)


# ----------------------------------------------------------------- the decode path (as in training)
def expr_to_sentence_idx(expr):
    """Real cell -> the gene ORDER a cell sentence would encode (expressed genes, desc expression)."""
    idx = np.where(expr > 0)[0]
    return idx[np.argsort(-expr[idx])] if len(idx) else idx


def rank_value_profile(X, rows, P, kmax=400):
    """Empirical log1p(CP10K) value as a function of expression rank, from real cells."""
    acc = np.zeros(kmax, dtype=np.float64); n = 0
    for i in rows:
        v = np.asarray(X[i].todense()).ravel()
        s = np.sort(v)[::-1][:kmax]
        acc[:len(s)] += s; n += 1
    return (acc / max(1, n)).astype(np.float32)


def decode_from_order(order, P, rank_prof):
    """Gene order -> expression vector using the empirical rank->value profile (the trainer's decode)."""
    v = np.zeros(P, dtype=np.float32)
    for r, j in enumerate(order[:len(rank_prof)]):
        v[j] = rank_prof[r]
    return v


def roundtrip(expr, P, rank_prof):
    """True expression -> sentence order -> decoded expression. Exercises the lossy step."""
    return decode_from_order(expr_to_sentence_idx(expr), P, rank_prof)


# ----------------------------------------------------------------- probes
def build_cell_residual(expr, ctrl_pb, generic):
    return expr - ctrl_pb - generic


def adversarial_probes(P, rank_prof, ctrl_pb, generic, truths_in_cl, ctrl_cell_expr, canonical, rng):
    """Cheap 'generations' a reward-hacking policy might emit. All must score BELOW real cells."""
    probes = {}
    # 1. copy the control verbatim (zero drug information; the historic 0.766 leak baseline)
    probes["control_copy"] = build_cell_residual(roundtrip(ctrl_cell_expr, P, rank_prof), ctrl_pb, generic)
    # 2. predict exactly the generic response -> residual is zero
    probes["generic_response"] = np.zeros(P, dtype=np.float32)
    # 3. orthogonal to EVERY profile in this cell line: drives max_other -> 0 with no drug knowledge
    v = rng.randn(P).astype(np.float32)
    for t in truths_in_cl:
        nt = np.linalg.norm(t)
        if nt > 1e-9:
            v -= (v @ t) / (nt ** 2) * t
    probes["orthogonal_to_all"] = v
    # 4. ONE canonical 'maximally contrastive' vector reused for every drug
    probes["canonical_contrastive"] = canonical
    # 5. truncated stub (only the top 20 genes survive)
    o = expr_to_sentence_idx(ctrl_cell_expr)
    probes["truncated_stub"] = build_cell_residual(decode_from_order(o[:20], P, rank_prof), ctrl_pb, generic)
    # 6. duplicate spam: 10 unique genes, so the decode fills only 10 slots
    probes["duplicate_spam"] = build_cell_residual(decode_from_order(o[:10], P, rank_prof), ctrl_pb, generic)
    return probes


def run(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ot"))
    import build_residual_targets as brt
    from scipy import sparse
    import pandas as pd

    panel = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    P = len(panel)
    rng = np.random.RandomState(args.seed)

    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control, args.repro_thr, args.seed)
    by_cl = defaultdict(list)
    for k in kept:
        by_cl[k[1]].append(k)
    by_cl = {c: ks for c, ks in by_cl.items() if len(ks) >= args.min_drugs}
    logger.info(f"{sum(len(v) for v in by_cl.values())} conditions across {len(by_cl)} cell lines")

    # treated-cell row indices per condition (our stand-in for 'generations': REAL cells)
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    plate = meta["plate"].astype(str).values
    drug = meta["drug"].astype(str).values
    dose = meta["dose"].astype(str).values
    rows_of = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            rows_of[(drug[i], cl[i], plate[i], dose[i])].append(i)

    allctrl = np.concatenate([r for r in ctrl_rows.values()])[:4000]
    rank_prof = rank_value_profile(X, allctrl, P)
    logger.info(f"rank->value profile from {len(allctrl)} real cells "
                f"(top rank {rank_prof[0]:.3f} -> rank 200 {rank_prof[min(199,len(rank_prof)-1)]:.3f})")

    ctrl_pb = {}
    for c in by_cl:
        for k in by_cl[c]:
            g = kept[k]["group"]
            ctrl_pb[k] = np.asarray(np.log1p(X[ctrl_rows[g]].todense()).mean(0)).ravel().astype(np.float32)

    # one canonical 'maximally contrastive' vector: mean over conditions of (own - mean of others)
    can = np.zeros(P, dtype=np.float64); n_can = 0
    for c in by_cl:
        for k in by_cl[c]:
            oth = [kept[o]["residual"] for o in by_cl[c] if o != k]
            if oth:
                can += kept[k]["residual"] - np.mean(np.stack(oth), 0); n_can += 1
    canonical = (can / max(1, n_can)).astype(np.float32)

    conds = [k for c in by_cl for k in by_cl[c]]
    rng.shuffle(conds)
    conds = [k for k in conds if len(rows_of.get(k, [])) >= args.min_treated][:args.n_conditions]
    logger.info(f"probing {len(conds)} conditions")

    out = {"sims": {}, "n_conditions": len(conds)}
    aggs = [a.strip() for a in args.neg_aggs.split(",") if a.strip()]
    for sim_name, sim in SIMS.items():
      for agg in aggs:
        NEG = agg
        logger.info("=" * 100)
        logger.info(f"VARIANT: similarity={sim_name}  negatives={agg}")
        own_true, own_rt, other_rt = [], [], []
        pool_curve = defaultdict(list)
        adv = defaultdict(list)
        rank_inv, reliab_sig, reliab_noise = [], [], []
        strata = defaultdict(list)

        for k in conds:
            c = k[1]
            others_k = [o for o in by_cl[c] if o != k]
            if len(others_k) < 3:
                continue
            res_own = kept[k]["residual"]
            res_oth = [kept[o]["residual"] for o in others_k]
            cpb, gen = ctrl_pb[k], generic[c]
            rows = np.array(rows_of[k]); rng.shuffle(rows)

            # ---- G1/G2: real own-drug cells, with TRUE expression vs after the SENTENCE ROUND-TRIP ----
            for i in rows[:args.n_cells]:
                e = np.asarray(np.log1p(X[i].todense())).ravel().astype(np.float32)
                own_true.append(r_disc(build_cell_residual(e, cpb, gen), res_own, res_oth, sim, args.k_sig, NEG))
                rt = build_cell_residual(roundtrip(e, P, rank_prof), cpb, gen)
                own_rt.append(r_disc(rt, res_own, res_oth, sim, args.k_sig, NEG))
            # real cells of ANOTHER drug, scored as if they were this drug (the discrimination contrast)
            ok = others_k[rng.randint(len(others_k))]
            orows = rows_of.get(ok, [])
            for i in list(orows)[:args.n_cells]:
                e = np.asarray(np.log1p(X[i].todense())).ravel().astype(np.float32)
                rt = build_cell_residual(roundtrip(e, P, rank_prof), cpb, gen)
                other_rt.append(r_disc(rt, res_own, res_oth, sim, args.k_sig, NEG))

            # ---- G3: pooling curve (how many cells to beat single-cell noise) ----
            for m in (1, 2, 4, 8, 16, 40):
                if len(rows) >= m:
                    e = np.asarray(np.log1p(X[rows[:m]].todense()).mean(0)).ravel().astype(np.float32)
                    rt = build_cell_residual(roundtrip(e, P, rank_prof), cpb, gen)
                    pool_curve[m].append(r_disc(rt, res_own, res_oth, sim, args.k_sig, NEG))

            # ---- G4: adversarial probes ----
            cell_e = np.asarray(np.log1p(X[ctrl_rows[kept[k]['group']][0]].todense())).ravel().astype(np.float32)
            for name, pv in adversarial_probes(P, rank_prof, cpb, gen, [res_own] + res_oth,
                                               cell_e, canonical, rng).items():
                adv[name].append(r_disc(pv, res_own, res_oth, sim, args.k_sig, NEG))

            # ---- G5: cross-drug RANK-INVARIANCE (the sharpest kill condition) ----
            take = rows[:args.n_rank_cells]
            if len(take) >= 6:
                cells = [build_cell_residual(roundtrip(
                    np.asarray(np.log1p(X[i].todense())).ravel().astype(np.float32), P, rank_prof), cpb, gen)
                    for i in take]
                rA = [r_disc(v, res_own, res_oth, sim, args.k_sig, NEG) for v in cells]
                res_B = kept[ok]["residual"]
                oth_B = [kept[o]["residual"] for o in by_cl[c] if o != ok]
                rB = [r_disc(v, res_B, oth_B, sim, args.k_sig, NEG) for v in cells]
                from scipy.stats import spearmanr
                rho = spearmanr(rA, rB).statistic
                if rho == rho:
                    rank_inv.append(rho)

            # ---- G6: reliability against truths from DISJOINT halves ----
            if "residual_A" in kept[k] and "residual_B" in kept[k]:
                e = np.asarray(np.log1p(X[rows[:8]].todense()).mean(0)).ravel().astype(np.float32)
                rt = build_cell_residual(roundtrip(e, P, rank_prof), cpb, gen)
                ra = r_disc(rt, kept[k]["residual_A"], res_oth, sim, args.k_sig, NEG)
                rb = r_disc(rt, kept[k]["residual_B"], res_oth, sim, args.k_sig, NEG)
                reliab_sig.append((ra + rb) / 2.0); reliab_noise.append(ra - rb)

            # ---- G7: strata by condition reproducibility ----
            strata["repro" if kept[k]["repro_cos"] > 0.35 else "weak"].append(
                float(np.mean(own_rt[-args.n_cells:])) if own_rt else np.nan)

        # -------- report --------
        m_own_t, m_own_r = float(np.mean(own_true)), float(np.mean(own_rt))
        m_oth = float(np.mean(other_rt))
        sd = float(np.std(own_rt))
        margin = m_own_r - m_oth
        d = margin / (sd + 1e-9)
        logger.info(f"  G1 DISCRIMINATION  own-drug cells {m_own_r:+.4f} vs other-drug cells {m_oth:+.4f}"
                    f"   margin {margin:+.4f}  (Cohen d {d:+.2f})   {'PASS' if margin > 0 else 'FAIL'}")
        logger.info(f"  G2 DECODE COST     own-drug reward: true expression {m_own_t:+.4f} -> after "
                    f"sentence round-trip {m_own_r:+.4f}  (delta {m_own_r-m_own_t:+.4f}; a large "
                    f"NEGATIVE delta means the decode is destroying signal)")
        pc = {m: float(np.mean(v)) for m, v in sorted(pool_curve.items())}
        logger.info(f"  G3 POOLING         reward vs #cells pooled: "
                    + "  ".join(f"m={m}:{v:+.3f}" for m, v in pc.items()))
        if 1 in pc and 40 in pc:
            frac = (pc[1] - m_oth) / (pc[40] - m_oth + 1e-9)
            logger.info(f"                     a SINGLE cell retains {100*frac:.0f}% of the 40-cell margin "
                        f"-> pooled-sample size m ~ {max(1, int(round(1/max(frac,1e-3)**2)))} for parity")
        logger.info("  G4 HACKABILITY     (all must be BELOW own-drug cells)")
        worst = None
        for name, v in adv.items():
            mv = float(np.mean(v))
            flag = "ok" if mv < m_own_r else "*** HACK ***"
            worst = mv if worst is None else max(worst, mv)
            logger.info(f"                       {name:24s} {mv:+.4f}   {flag}")
        ri = float(np.mean(rank_inv)) if rank_inv else float("nan")
        logger.info(f"  G5 RANK-INVARIANCE Spearman(rank by own truth, rank by OTHER drug's truth) = {ri:+.3f}"
                    f"   {'KILL: reward sorts by generic quality' if ri > 0.8 else 'ok (reward is drug-specific)'}")
        snr = (float(np.std(reliab_sig)) / (float(np.std(reliab_noise)) + 1e-9)) if reliab_sig else float("nan")
        logger.info(f"  G6 RELIABILITY     sigma_signal/sigma_noise across disjoint-half truths = {snr:.2f}")
        st = {k2: float(np.nanmean(v)) for k2, v in strata.items() if len(v)}
        logger.info(f"  G7 STRATA          {st}")

        out["sims"][f"{sim_name}|{agg}"] = {
            "own_true": m_own_t, "own_roundtrip": m_own_r, "other_drug": m_oth,
            "margin": margin, "cohen_d": d, "pooling": pc, "adversarial": {k2: float(np.mean(v)) for k2, v in adv.items()},
            "rank_invariance": ri, "reliability_snr": snr, "strata": st,
        }

    # -------- verdict --------
    logger.info("=" * 100)
    best = max(out["sims"], key=lambda s: out["sims"][s]["cohen_d"] if out["sims"][s]["margin"] > 0 else -9)
    b = out["sims"][best]
    ok_margin = b["margin"] > 0
    ok_hack = all(v < b["own_roundtrip"] for v in b["adversarial"].values())
    ok_rank = not (b["rank_invariance"] > 0.8)
    out["recommended_sim"] = best
    out["gates"] = {"discrimination": ok_margin, "no_hack": ok_hack, "rank_invariance": ok_rank}
    logger.info(f"RECOMMENDED SIMILARITY: {best}  (margin {b['margin']:+.4f}, d {b['cohen_d']:+.2f})")
    logger.info(f"GATES: discrimination={'PASS' if ok_margin else 'FAIL'}  "
                f"no-hack={'PASS' if ok_hack else 'FAIL'}  rank-invariance={'PASS' if ok_rank else 'FAIL'}")
    if ok_margin and ok_hack and ok_rank:
        logger.info(">>> PROCEED: the reward discriminates real drugs, resists the adversarial probes, and "
                    "does not merely sort by generic quality.")
    else:
        logger.info(">>> STOP: fix the reward before spending GPU-hours. A failing gate here means GRPO "
                    "would be optimizing noise or a hackable proxy.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")


def selftest():
    """Synthetic where drug identity IS in the residual: gates must PASS. Then a scrambled control where
    truths are shuffled: discrimination must COLLAPSE (proves the gates can actually fail)."""
    rng = np.random.RandomState(0)
    P, D, N = 300, 6, 30
    truths = {d: rng.randn(P).astype(np.float32) for d in range(D)}
    def cells(d, n):  # noisy realizations of the drug's residual
        return [(truths[d] + rng.randn(P) * 0.8).astype(np.float32) for _ in range(n)]
    for label, key in (("aligned", lambda d: d), ("scrambled", lambda d: (d + 1) % D)):
        own, oth = [], []
        for d in range(D):
            others = [truths[o] for o in truths if o != key(d)]
            for v in cells(d, N):
                own.append(r_disc(v, truths[key(d)], others, SIMS["cosine"]))
            for v in cells((d + 2) % D, N):
                oth.append(r_disc(v, truths[key(d)], others, SIMS["cosine"]))
        margin = np.mean(own) - np.mean(oth)
        logger.info(f"  {label:10s} own {np.mean(own):+.3f}  other {np.mean(oth):+.3f}  margin {margin:+.3f}")
        if label == "aligned" and margin <= 0.1:
            logger.error("  FAIL: gate cannot detect real drug signal"); sys.exit(1)
        if label == "scrambled" and margin > 0.1:
            logger.error("  FAIL: gate reports signal where the truths are shuffled"); sys.exit(1)
    logger.info("SELFTEST PASSED (detects real signal; collapses when truths are scrambled)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--n_conditions", type=int, default=150)
    ap.add_argument("--n_cells", type=int, default=8, help="real cells per condition used as 'generations'")
    ap.add_argument("--n_rank_cells", type=int, default=12, help="cells for the rank-invariance test")
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_drugs", type=int, default=4)
    ap.add_argument("--repro_thr", type=float, default=0.2)
    ap.add_argument("--neg_aggs", default="max,mean,opposite,infonce",
                    help="how to aggregate negatives; 'max' is the original and is biased")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/reward_calibration.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")
    run(args)


if __name__ == "__main__":
    main()

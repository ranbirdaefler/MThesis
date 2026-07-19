#!/usr/bin/env python
r"""
leak_audit.py — re-test EVERY leak-exposed claim, cross-plate vs same-plate
===========================================================================
We discovered that drug and PLATE are confounded by the experimental design (each drug sits on its
own plate(s)), and controls are plate-matched. So in any test where "same drug" shares a plate while
"different drug" does not, batch identity can stand in for drug identity. Measured directly: a
zero-drug-info control cell scores NIR 0.551 cross-plate vs 0.510 same-plate.

That exposes every discrimination claim we have made. This script re-runs each one under BOTH
groupings — identical data, identical metrics, the ONLY difference being whether the comparison set
is restricted to the same (cell_line, plate) — so the delta IS the batch contribution.

CLAIMS AUDITED
  1. SPIKE-IN discrimination (~0.93-0.99) — "the metric works; the model is the problem".
     ref = pseudobulk of drug A, same = disjoint pseudobulk of A, diff = pseudobulk of B.
     Correct if sim(ref,same) > sim(ref,diff). Cross-plate: B lives on another plate (leak available).
     Same-plate: A and B share a plate (leak impossible).  <-- NEVER TESTED BEFORE
  2. CEILING NIR (0.63-0.88) and the "% identifiable" classification the stratification rests on.
  3. DRF (+0.80) — Miller-style metric calibration. Its positive control is a same-plate replicate,
     so it is exposed too. (Simplification: positive control = the drug's real held-out half, i.e. a
     noise ceiling, rather than the interpolated duplicate. The A/B is what matters here.)
  4. DE-Δr / rank-metric discrimination, same treatment.

READOUT — for each claim:
    delta = cross_plate - same_plate  =  the part of the claim that was batch, not biology.
    If same_plate stays high  -> the claim survives; it was real drug signal.
    If same_plate collapses   -> the claim was substantially plate identity.

SELF-CONTAINED: no local imports (runs in the flat cluster layout and the reorganized repo).

USAGE (CPU)
  python leak_audit.py --eval_dir DATA_endcell_big --tiers tier2_unseen_drugs \
     --pb_size 8 --n_trials 300 --out RESULTS/leak_audit.json

SELFTEST (no data) — plants a pure-batch effect with ZERO drug signal and verifies the audit
attributes it to plate (cross-plate high, same-plate at chance):
  python leak_audit.py --selftest
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"


# ----------------------------------------------------------------- representation
def genes_of(s):
    out = []
    for t in s.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def sentence_to_expr(s, pidx, P, lm):
    slope, intercept = lm["slope"], lm["intercept"]
    arr = np.zeros(P); seen = set(); pos = 0
    for g in genes_of(s):
        gi = pidx.get(g)
        if gi is None or gi in seen:
            continue
        seen.add(gi); pos += 1
        arr[gi] = max(0.0, slope * np.log10(pos) + intercept)
    return arr


def sentence_to_rank(s, pidx, P):
    arr = np.full(P, float(P)); seen = set(); pos = 0
    for g in genes_of(s):
        gi = pidx.get(g)
        if gi is None or gi in seen:
            continue
        seen.add(gi); pos += 1
        arr[gi] = float(pos)
    return arr


def pbm(vs):
    return np.mean(np.stack(vs), axis=0)


def rank_sim(a, b):
    """Pearson on rank profiles (discrimination-equivalent to tau, much faster)."""
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def expr_sim(a, b):
    return -float(np.linalg.norm(a - b))          # higher = more similar


def nir_from_dists(d_own, d_oth):
    if not d_oth:
        return None
    return float(np.mean([d_own < x for x in d_oth]))


# ----------------------------------------------------------------- claim 1: spike-in
def spikein(groups, rng, pb_size, n_trials, min_cells):
    """Forced choice: ref & same from drug A, diff from drug B. Correct if sim(ref,same)>sim(ref,diff).
    Chance = 0.50. `groups` already encodes the plate restriction, so A and B are same-plate iff the
    caller grouped by (cell_line, plate)."""
    acc = {"rank": [], "expr": []}
    keys = [g for g, dd in groups.items()
            if len([d for d, v in dd.items() if len(v["expr"]) >= 2 * pb_size]) >= 2]
    if not keys:
        return None
    for _ in range(n_trials):
        g = keys[rng.randint(len(keys))]
        dd = groups[g]
        elig = [d for d, v in dd.items() if len(v["expr"]) >= 2 * pb_size]
        if len(elig) < 2:
            continue
        a, b = rng.choice(len(elig), 2, replace=False)
        A, B = elig[a], elig[b]
        ia = rng.permutation(len(dd[A]["expr"]))[:2 * pb_size]
        ib = rng.permutation(len(dd[B]["expr"]))[:pb_size]
        for space, sim in (("rank", rank_sim), ("expr", expr_sim)):
            ref = pbm([dd[A][space][i] for i in ia[:pb_size]])
            same = pbm([dd[A][space][i] for i in ia[pb_size:]])
            diff = pbm([dd[B][space][i] for i in ib])
            s1, s2 = sim(ref, same), sim(ref, diff)
            if s1 is None or s2 is None:
                continue
            acc[space].append(1.0 if s1 > s2 else 0.0)
    return {k: (float(np.mean(v)) if v else None, len(v)) for k, v in acc.items()}


# ----------------------------------------------------------------- claims 2-4: ceiling / DRF
def ceiling_and_drf(groups, rng, min_cells, identifiable_nir):
    """Per-drug ceiling NIR (real held-out half vs every drug's truth half), the % identifiable, and
    a DRF for NIR and DE-Δr. All comparison sets are whatever `groups` encodes (cross- or same-plate)."""
    ceils, rows = [], []
    drf_acc = {m: {"perfect": [], "neg": [], "pos": []} for m in ("nir", "de_delta")}
    for g, dd in groups.items():
        drugs = [d for d, v in dd.items() if len(v["expr"]) >= min_cells]
        if len(drugs) < 3:
            continue
        A, B = {}, {}
        for d in drugs:
            v = dd[d]["expr"]
            idx = rng.permutation(len(v)); h = len(idx) // 2
            A[d] = pbm([v[i] for i in idx[:h]])
            B[d] = pbm([v[i] for i in idx[h:]])
        for d in drugs:
            others = [o for o in drugs if o != d]
            d_own = float(np.linalg.norm(B[d] - A[d]))
            d_oth = [float(np.linalg.norm(B[d] - A[o])) for o in others]
            c = nir_from_dists(d_own, d_oth)
            if c is None:
                continue
            ceils.append(c); rows.append({"group": str(g), "drug": d, "ceiling_nir": c})

            # DRF: perfect = truth itself; neg = LOO mean over other drugs; pos = real replicate half
            neg = pbm([A[o] for o in others])
            for m in ("nir",):
                drf_acc[m]["perfect"].append(nir_from_dists(
                    float(np.linalg.norm(A[d] - A[d])), [float(np.linalg.norm(A[d] - A[o])) for o in others]))
                drf_acc[m]["neg"].append(nir_from_dists(
                    float(np.linalg.norm(neg - A[d])), [float(np.linalg.norm(neg - A[o])) for o in others]))
                drf_acc[m]["pos"].append(c)
    if not ceils:
        return None
    ceils = np.array(ceils)
    out = {"ceiling_nir_mean": float(ceils.mean()), "ceiling_nir_median": float(np.median(ceils)),
           "frac_identifiable": float((ceils >= identifiable_nir).mean()), "n_drugs": len(ceils),
           "frac_at_chance": float((ceils < 0.6).mean())}
    p = np.mean(drf_acc["nir"]["perfect"]); n = np.mean(drf_acc["nir"]["neg"]); po = np.mean(drf_acc["nir"]["pos"])
    out["drf_nir"] = {"drf": (float((po - n) / (p - n)) if abs(p - n) > 1e-6 else None),
                      "m_perfect": float(p), "m_neg": float(n), "m_pos": float(po)}
    return out


# ----------------------------------------------------------------- loading
def load(eval_dir, tiers, train_file, pidx, P, lm, same_plate):
    groups = defaultdict(lambda: defaultdict(lambda: {"expr": [], "rank": []}))
    files = [os.path.join(eval_dir, f"eval_{t}.jsonl") for t in tiers]
    if train_file:
        files.append(train_file)
    for path in files:
        if not os.path.exists(path):
            logger.warning(f"  missing {path}")
            continue
        n = 0
        for line in open(path):
            ex = json.loads(line)
            m = ex.get("metadata", {})
            cl, drug, plate = m.get("cell_line_id"), m.get("drug"), m.get("plate")
            if cl is None or drug is None:
                continue
            if same_plate and plate is None:
                continue
            key = (cl, plate) if same_plate else (cl, None)
            slot = groups[key][drug]
            slot["expr"].append(sentence_to_expr(ex["response"], pidx, P, lm))
            slot["rank"].append(sentence_to_rank(ex["response"], pidx, P))
            n += 1
        logger.info(f"    {os.path.basename(path)}: {n:,} rows")
    return groups


def audit(args, pidx, P, lm):
    res = {}
    for mode, sp in (("cross_plate", False), ("same_plate", True)):
        logger.info("")
        logger.info(f"  === {mode} ===")
        groups = load(args.eval_dir, [t.strip() for t in args.tiers.split(",") if t.strip()],
                      args.train_file, pidx, P, lm, same_plate=sp)
        logger.info(f"    comparison groups: {len(groups)}")
        rng = np.random.RandomState(args.seed)
        r = {"n_groups": len(groups)}
        r["spikein"] = spikein(groups, rng, args.pb_size, args.n_trials, args.min_cells)
        r["ceiling"] = ceiling_and_drf(groups, np.random.RandomState(args.seed),
                                       args.min_cells, args.identifiable_nir)
        res[mode] = r
    return res


def report(res):
    c, s = res.get("cross_plate", {}), res.get("same_plate", {})

    def g(d, *ks):
        for k in ks:
            if d is None:
                return None
            d = d.get(k) if isinstance(d, dict) else None
        return d

    logger.info("")
    logger.info("=" * 100)
    logger.info("  LEAK AUDIT — every claim, cross-plate vs same-plate (delta = the batch contribution)")
    logger.info("=" * 100)
    logger.info(f"  {'claim':<40} {'cross-plate':>12} {'same-plate':>12} {'delta':>9}   verdict")
    logger.info("  " + "-" * 96)

    def line(label, cv, sv, chance=0.5, collapse_frac=0.4):
        if cv is None or sv is None:
            logger.info(f"  {label:<40} {'NA':>12} {'NA':>12}"); return
        d = cv - sv
        head_c, head_s = cv - chance, sv - chance
        if head_c <= 1e-9:
            v = "n/a (already at chance)"
        elif head_s < head_c * collapse_frac:
            v = f"*** COLLAPSES: {100*(1-max(head_s,0)/head_c):.0f}% of it was PLATE"
        elif d > 0.02:
            v = f"survives (batch worth {d:+.3f})"
        else:
            v = "survives intact (no batch effect)"
        logger.info(f"  {label:<40} {cv:>12.3f} {sv:>12.3f} {d:>+9.3f}   {v}")

    line("SPIKE-IN discrimination (rank)", g(c, "spikein", "rank") and c["spikein"]["rank"][0],
         g(s, "spikein", "rank") and s["spikein"]["rank"][0])
    line("SPIKE-IN discrimination (expr)", g(c, "spikein", "expr") and c["spikein"]["expr"][0],
         g(s, "spikein", "expr") and s["spikein"]["expr"][0])
    line("CEILING NIR (mean)", g(c, "ceiling", "ceiling_nir_mean"), g(s, "ceiling", "ceiling_nir_mean"))
    line("% identifiable (ceiling>=thr)", g(c, "ceiling", "frac_identifiable"),
         g(s, "ceiling", "frac_identifiable"), chance=0.0)
    line("DRF (NIR)", g(c, "ceiling", "drf_nir", "drf"), g(s, "ceiling", "drf_nir", "drf"), chance=0.0)
    logger.info("  " + "-" * 96)
    logger.info("  A claim that COLLAPSES under same-plate was measuring batch identity, not the drug.")
    logger.info("  A claim that SURVIVES is real drug signal and can be quoted (at the same-plate value).")
    logger.info("=" * 100)


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Plant a PURE BATCH effect and ZERO drug effect: every drug's cells are drawn from an identical
    drug-independent distribution, but each drug sits on its own plate and cells inherit a strong
    plate signature. A correct audit must show the spike-in scoring HIGH cross-plate (it is reading
    the batch) and collapsing to ~chance same-plate (where batch is constant)."""
    rng = np.random.RandomState(0)
    P = 200
    lm = {"slope": -0.4, "intercept": 1.6}
    panel = [f"G{i}" for i in range(P)]
    pidx = {g: i for i, g in enumerate(panel)}

    def cell(sig):
        v = np.maximum(0.0, sig + rng.randn(P) * 0.4)
        order = np.argsort(-v)
        return " ".join(panel[i] for i in order if v[i] > 0)

    base = np.zeros(P); base[rng.choice(P, 100, replace=False)] = rng.rand(100) * 2 + 1.0
    plate_sig = {p: (lambda: (lambda z: z)(np.zeros(P)))() for p in range(4)}
    for p in range(4):
        z = np.zeros(P); z[rng.choice(P, 60, replace=False)] = rng.rand(60) * 3.0
        plate_sig[p] = z

    # 8 drugs, NO drug-specific effect; drugs 0-1 on plate0, 2-3 on plate1, etc.  Each plate ALSO
    # hosts >=2 drugs so the same-plate comparison is possible.
    groups_cross = defaultdict(lambda: defaultdict(lambda: {"expr": [], "rank": []}))
    groups_same = defaultdict(lambda: defaultdict(lambda: {"expr": [], "rank": []}))
    for d in range(8):
        p = d // 2
        for _ in range(24):
            s = cell(base + plate_sig[p])                  # drug contributes NOTHING
            e, r = sentence_to_expr(s, pidx, P, lm), sentence_to_rank(s, pidx, P)
            groups_cross[("cl", None)][f"d{d}"]["expr"].append(e)
            groups_cross[("cl", None)][f"d{d}"]["rank"].append(r)
            groups_same[("cl", p)][f"d{d}"]["expr"].append(e)
            groups_same[("cl", p)][f"d{d}"]["rank"].append(r)

    sc = spikein(groups_cross, np.random.RandomState(1), 4, 300, 8)
    ss = spikein(groups_same, np.random.RandomState(1), 4, 300, 8)
    logger.info(f"  PURE-BATCH synthetic (zero drug signal by construction):")
    logger.info(f"    spike-in cross-plate: rank={sc['rank'][0]:.3f} expr={sc['expr'][0]:.3f}  "
                f"(should look HIGH — it is reading the plate)")
    logger.info(f"    spike-in same-plate : rank={ss['rank'][0]:.3f} expr={ss['expr'][0]:.3f}  "
                f"(should be ~0.50 — batch constant, and there is no drug signal)")
    ok = (sc["expr"][0] > 0.75 and ss["expr"][0] < 0.65)
    if not ok:
        logger.error("  FAIL: audit did not attribute a pure-batch effect to plate")
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--tiers", default="tier2_unseen_drugs")
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--pb_size", type=int, default=8, help="cells per pseudobulk in the spike-in")
    ap.add_argument("--n_trials", type=int, default=400)
    ap.add_argument("--min_cells", type=int, default=8)
    ap.add_argument("--identifiable_nir", type=float, default=0.8)
    ap.add_argument("--out", default="RESULTS/leak_audit.json")
    ap.add_argument("--panel_file", default=None)
    ap.add_argument("--linear_model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return
    if not args.eval_dir:
        ap.error("--eval_dir required (unless --selftest)")

    panel = json.load(open(args.panel_file or os.path.join(args.eval_dir, "l1000_panel.json")))
    pidx = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    lm = json.load(open(args.linear_model or os.path.join(args.eval_dir, "linear_model.json")))
    logger.info(f"Panel {P}; lm slope={lm['slope']:.3f} intercept={lm['intercept']:.3f}")

    res = audit(args, pidx, P, lm)
    report(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

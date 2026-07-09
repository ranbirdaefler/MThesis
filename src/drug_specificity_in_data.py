#!/usr/bin/env python
"""
drug_specificity_in_data.py  (v2 — confound-controlled)
=======================================================
Model-free test of whether the DATA contains a drug-specific signal that our metrics can see.
Asks whether two REAL treated cells of the SAME drug agree more (in a metric) than two real
treated cells of DIFFERENT drugs, WITHIN the same cell line. Separates "the model can't capture
drug-specificity" from "there is no detectable drug-specific signal".

This version adds the following controls:
  * 3 metrics: DE-Δr, panel-τ (C2S-paper-style whole panel), topN-τ (expressed-only).
  * 2 aggregation levels: single-cell and pseudobulk (denoised; disjoint-half replicates).
  * DOSE control: a same-dose-matched comparison (same-drug-same-dose vs diff-drug-same-dose),
    so the gap isn't just "same drug shares dose".
  * BATCH control: a different-plate-only comparison (both cells from different plates), so the
    gap isn't inflated by shared-plate batch effects.
  * POSITIVE CONTROL (MOA): same-MOA vs different-MOA. If the test can detect the (coarser,
    expected-to-exist) MOA signal, a null on drug identity is interpretable as "small/absent"
    rather than "underpowered test".
  * EFFECT SIZE: Cohen's d (does the pair-distribution actually separate?) and gap-as-fraction-
    of-recoverable-signal (gap / (replicate_ceiling - diff_drug_floor)) so "0.02" is interpretable.
  * PSEUDOBULK SIZE SWEEP: gap vs #cells averaged, to see if/where recoverable drug signal plateaus.

All comparisons are within cell line; cell line is the unit of replication for the
cluster bootstrap + within-cell-line sign-flip permutation test.

USAGE
-----
  python drug_specificity_in_data.py --data_dir DATA \
     --sources train,eval_tier1_seen_conditions --out RESULTS/drug_specificity_v2.json \
     --de_k 50 --topn 100 --pseudobulk --pb_sizes 5,15,40 \
     --n_celllines 40 --drugs_per_cellline 30 --cells_per_drug 16 \
     --same_pairs_per_drug 20 --diff_pairs_per_cellline 400 \
     --n_boot 2000 --n_perm 2000 --seed 42
"""
import argparse, json, os, logging
from collections import defaultdict
import numpy as np
import evaluate_c2s_tahoe as ev

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["de_delta", "panel_tau", "topn_tau"]
MLABEL = {"de_delta": "DE-Δr", "panel_tau": "panel-τ", "topn_tau": "topN-τ"}


# --------------------------------------------------------------------------- loading
def load_cells(data_dir, sources):
    by_cl_drug = defaultdict(lambda: defaultdict(list))
    n = 0
    for src in sources:
        path = os.path.join(data_dir, src if src.endswith(".jsonl") else f"{src}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  source not found: {path}"); continue
        with open(path) as f:
            for line in f:
                ex = json.loads(line); m = ex.get("metadata", {})
                cl, drug = m.get("cell_line_id"), m.get("drug")
                if cl is None or drug is None: continue
                ctrl = ev.control_from_prompt(ex["prompt"])
                if not ctrl: continue
                by_cl_drug[cl][drug].append({
                    "resp": ex["response"], "ctrl": ctrl,
                    "moa": m.get("moa"), "dose": m.get("dose"),
                    "plate": m.get("plate") or m.get("sample"),
                })
                n += 1
    logger.info(f"  Loaded {n:,} treated cells across {len(by_cl_drug)} cell lines")
    return by_cl_drug


# --------------------------------------------------------------------------- metrics
def pair_metrics(recA, recB, panel, worst, de_k, topn):
    """A='truth', B='prediction', anchored on A's control. Returns the 3 metrics."""
    true_ranks = ev.cell_sentence_to_gene_ranks(recA["resp"])
    pred_ranks = ev.cell_sentence_to_gene_ranks(recB["resp"])
    control_ranks = ev.cell_sentence_to_gene_ranks(recA["ctrl"])
    de_genes = ev.select_top_de_genes(true_ranks, control_ranks, panel, de_k, worst)
    de_res = ev.delta_correlation(pred_ranks, true_ranks, control_ranks, de_genes, worst)
    de = de_res.get("delta_pearson") if isinstance(de_res, dict) else de_res
    ptau = ev.compute_rank_correlation(pred_ranks, true_ranks, gene_subset=panel)
    ptau = ptau.get("kendall_tau") if isinstance(ptau, dict) else None
    topn_genes = ev.select_top_expressed(true_ranks, panel, topn, worst)
    ttau = ev.compute_rank_correlation(pred_ranks, true_ranks, gene_subset=topn_genes)
    ttau = ttau.get("kendall_tau") if isinstance(ttau, dict) else None
    return {"de_delta": de, "panel_tau": ptau, "topn_tau": ttau}


def make_pseudobulk(recs, panel, worst):
    def consensus(key):
        acc = np.zeros(len(panel))
        for r in recs:
            rk = ev.cell_sentence_to_gene_ranks(r[key])
            acc += np.array([rk.get(g, worst) for g in panel], dtype=float)
        acc /= len(recs)
        order = np.argsort(acc, kind="stable")
        return " ".join(panel[i] for i in order)
    return {"resp": consensus("resp"), "ctrl": consensus("ctrl")}


# --------------------------------------------------------------------------- stats
def cluster_bootstrap_gap(per_cl_gaps, n_boot, seed):
    rng = np.random.RandomState(seed)
    vals = np.array([per_cl_gaps[c] for c in per_cl_gaps])
    boots = [np.nanmean(vals[rng.choice(len(vals), len(vals), replace=True)]) for _ in range(n_boot)]
    return dict(mean=float(np.nanmean(vals)), ci_low=float(np.nanpercentile(boots, 2.5)),
                ci_high=float(np.nanpercentile(boots, 97.5)), n_celllines=len(vals))


def perm_p(per_cl_gaps, obs, n_perm, seed):
    rng = np.random.RandomState(seed + 1)
    g = np.array(list(per_cl_gaps.values()))
    null = np.array([np.nanmean(g * rng.choice([-1, 1], size=len(g))) for _ in range(n_perm)])
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def cohens_d(same, diff):
    same, diff = np.array(same), np.array(diff)
    ns, nd = len(same), len(diff)
    if ns < 2 or nd < 2: return None
    sp = np.sqrt(((ns - 1) * same.var(ddof=1) + (nd - 1) * diff.var(ddof=1)) / (ns + nd - 2))
    return float((same.mean() - diff.mean()) / sp) if sp > 0 else None


def analyze_gap(per_cl, pooled_same, pooled_diff, n_boot, n_perm, seed, ceiling=None):
    if not per_cl: return None
    boot = cluster_bootstrap_gap(per_cl, n_boot, seed)
    p = perm_p(per_cl, boot["mean"], n_perm, seed)
    d = cohens_d(pooled_same, pooled_diff)
    out = {"same_mean": float(np.mean(pooled_same)), "diff_mean": float(np.mean(pooled_diff)),
           "same_n": len(pooled_same), "diff_n": len(pooled_diff),
           "gap": boot, "perm_p": p, "cohens_d": d, "n_celllines": len(per_cl)}
    # gap as fraction of recoverable signal = gap / (ceiling - diff_floor)
    if ceiling is not None:
        denom = ceiling - out["diff_mean"]
        out["frac_of_recoverable"] = float(out["gap"]["mean"] / denom) if denom > 1e-6 else None
    return out


# --------------------------------------------------------------------------- pairing utils
def usample(n, k, rng, groups=None, want_diff=False, want_same=False):
    """Yield up to k distinct unordered index pairs, optionally constrained by group equality."""
    seen = set(); tries = 0; cap = k * 25 + 50
    while len(seen) < k and tries < cap:
        i, j = rng.randint(n), rng.randint(n); tries += 1
        if i == j: continue
        key = (min(i, j), max(i, j))
        if key in seen: continue
        if groups is not None:
            if want_diff and groups[i] == groups[j]: continue
            if want_same and groups[i] != groups[j]: continue
        seen.add(key); yield i, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sources", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--de_k", type=int, default=50)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--pseudobulk", action="store_true")
    ap.add_argument("--pb_sizes", default="5,15,40", help="pseudobulk cell counts to sweep")
    ap.add_argument("--n_celllines", type=int, default=40)
    ap.add_argument("--drugs_per_cellline", type=int, default=30)
    ap.add_argument("--cells_per_drug", type=int, default=16)
    ap.add_argument("--same_pairs_per_drug", type=int, default=20)
    ap.add_argument("--diff_pairs_per_cellline", type=int, default=400)
    ap.add_argument("--min_drugs_per_cellline", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.abspath(args.out).startswith(os.path.abspath(args.data_dir) + os.sep):
        raise SystemExit("Refusing to write inside data_dir.")
    rng = np.random.RandomState(args.seed)
    panel = json.load(open(os.path.join(args.data_dir, "l1000_panel.json")))
    worst = len(panel) + 1
    pb_sizes = [int(x) for x in args.pb_sizes.split(",")]

    by_cl_drug = load_cells(args.data_dir, [s.strip() for s in args.sources.split(",")])
    cls = [cl for cl, dd in by_cl_drug.items()
           if sum(len(v) >= 2 for v in dd.values()) >= 2 and len(dd) >= args.min_drugs_per_cellline]
    rng.shuffle(cls); cls = cls[:args.n_celllines]
    logger.info(f"  Using {len(cls)} cell lines")

    # ---- accumulators: per analysis, per metric: {cellline: gap}, and pooled same/diff lists
    def newacc(): return {m: {"gap": {}, "same": [], "diff": []} for m in METRICS}
    A = {  # analysis name -> accumulator
        "single":            newacc(),   # same-drug vs diff-drug, single cell
        "single_samedose":   newacc(),   # + dose-matched
        "single_diffplate":  newacc(),   # + different-plate only
        "moa_poscontrol":    newacc(),   # same-MOA vs diff-MOA (positive control)
    }
    for s in pb_sizes:
        A[f"pb{s}"] = newacc()          # pseudobulk at size s
    # replicate ceiling: same (drug,dose,plate) cell pairs — the noise ceiling per metric
    CEIL = {m: {"gap": {}, "same": [], "diff": []} for m in METRICS}  # only 'same' used

    for cl in cls:
        dm = by_cl_drug[cl]
        drugs = list(dm.keys()); rng.shuffle(drugs); drugs = drugs[:args.drugs_per_cellline]
        pool = {}
        for d in drugs:
            recs = dm[d]
            if len(recs) > args.cells_per_drug:
                recs = [recs[i] for i in rng.choice(len(recs), args.cells_per_drug, replace=False)]
            pool[d] = recs
        flat = [(d, r) for d in drugs for r in pool[d]]
        if len(flat) < 2: continue
        gdrug = [d for d, _ in flat]
        gmoa = [r["moa"] for _, r in flat]

        def collect(acc_key, pairs, recs_list):
            for i, j in pairs:
                mm = pair_metrics(recs_list[i], recs_list[j], panel, worst, args.de_k, args.topn)
                for m in METRICS:
                    if mm[m] is not None and mm[m] == mm[m]:
                        A[acc_key][m].setdefault("_s_" + str(cl), []) if False else None
                        yield_target = A[acc_key][m]
                        yield_target.setdefault("cl_same", {})  # placeholder
            return

        # ---------- SINGLE-CELL same vs diff (+ pooled) ----------
        same_vals = {m: [] for m in METRICS}; diff_vals = {m: [] for m in METRICS}
        # same-drug pairs
        for d in drugs:
            recs = pool[d]
            if len(recs) < 2: continue
            k = min(args.same_pairs_per_drug, len(recs) * (len(recs) - 1) // 2)
            for i, j in usample(len(recs), k, rng):
                mm = pair_metrics(recs[i], recs[j], panel, worst, args.de_k, args.topn)
                for m in METRICS:
                    if mm[m] is not None and mm[m] == mm[m]: same_vals[m].append(mm[m])
        # diff-drug pairs
        kd = min(args.diff_pairs_per_cellline, len(flat) * (len(flat) - 1) // 2)
        for i, j in usample(len(flat), kd, rng, groups=gdrug, want_diff=True):
            mm = pair_metrics(flat[i][1], flat[j][1], panel, worst, args.de_k, args.topn)
            for m in METRICS:
                if mm[m] is not None and mm[m] == mm[m]: diff_vals[m].append(mm[m])
        for m in METRICS:
            if same_vals[m] and diff_vals[m]:
                A["single"][m]["gap"][cl] = np.mean(same_vals[m]) - np.mean(diff_vals[m])
                A["single"][m]["same"] += same_vals[m]; A["single"][m]["diff"] += diff_vals[m]

        # ---------- DOSE-MATCHED: same-drug-same-dose vs diff-drug-same-dose ----------
        sd_same = {m: [] for m in METRICS}; sd_diff = {m: [] for m in METRICS}
        # group cells by dose
        by_dose = defaultdict(list)
        for d, r in flat: by_dose[r["dose"]].append((d, r))
        for dose, items in by_dose.items():
            if len(items) < 2: continue
            gdr = [d for d, _ in items]
            # same-drug-same-dose
            for i, j in usample(len(items), args.same_pairs_per_drug, rng, groups=gdr, want_same=True):
                mm = pair_metrics(items[i][1], items[j][1], panel, worst, args.de_k, args.topn)
                for m in METRICS:
                    if mm[m] is not None and mm[m] == mm[m]: sd_same[m].append(mm[m])
            # diff-drug-same-dose
            for i, j in usample(len(items), args.same_pairs_per_drug * 3, rng, groups=gdr, want_diff=True):
                mm = pair_metrics(items[i][1], items[j][1], panel, worst, args.de_k, args.topn)
                for m in METRICS:
                    if mm[m] is not None and mm[m] == mm[m]: sd_diff[m].append(mm[m])
        for m in METRICS:
            if sd_same[m] and sd_diff[m]:
                A["single_samedose"][m]["gap"][cl] = np.mean(sd_same[m]) - np.mean(sd_diff[m])
                A["single_samedose"][m]["same"] += sd_same[m]; A["single_samedose"][m]["diff"] += sd_diff[m]

        # ---------- DIFFERENT-PLATE ONLY: same-drug vs diff-drug, both cells diff plate ----------
        dp_same = {m: [] for m in METRICS}; dp_diff = {m: [] for m in METRICS}
        for d in drugs:
            recs = pool[d]
            if len(recs) < 2: continue
            for i, j in usample(len(recs), args.same_pairs_per_drug, rng):
                if recs[i]["plate"] == recs[j]["plate"]: continue
                mm = pair_metrics(recs[i], recs[j], panel, worst, args.de_k, args.topn)
                for m in METRICS:
                    if mm[m] is not None and mm[m] == mm[m]: dp_same[m].append(mm[m])
        gpl = [r["plate"] for _, r in flat]
        for i, j in usample(len(flat), kd, rng, groups=gdrug, want_diff=True):
            if flat[i][1]["plate"] == flat[j][1]["plate"]: continue
            mm = pair_metrics(flat[i][1], flat[j][1], panel, worst, args.de_k, args.topn)
            for m in METRICS:
                if mm[m] is not None and mm[m] == mm[m]: dp_diff[m].append(mm[m])
        for m in METRICS:
            if dp_same[m] and dp_diff[m]:
                A["single_diffplate"][m]["gap"][cl] = np.mean(dp_same[m]) - np.mean(dp_diff[m])
                A["single_diffplate"][m]["same"] += dp_same[m]; A["single_diffplate"][m]["diff"] += dp_diff[m]

        # ---------- MOA POSITIVE CONTROL: same-MOA vs diff-MOA (different drugs) ----------
        moa_same = {m: [] for m in METRICS}; moa_diff = {m: [] for m in METRICS}
        # same-MOA but DIFFERENT drug (so it's not just same-drug leaking in)
        for i, j in usample(len(flat), kd, rng):
            if gmoa[i] is None or gmoa[j] is None: continue
            if gdrug[i] == gdrug[j]: continue
            same_moa = (gmoa[i] == gmoa[j])
            mm = pair_metrics(flat[i][1], flat[j][1], panel, worst, args.de_k, args.topn)
            for m in METRICS:
                if mm[m] is None or mm[m] != mm[m]: continue
                (moa_same if same_moa else moa_diff)[m].append(mm[m])
        for m in METRICS:
            if moa_same[m] and moa_diff[m]:
                A["moa_poscontrol"][m]["gap"][cl] = np.mean(moa_same[m]) - np.mean(moa_diff[m])
                A["moa_poscontrol"][m]["same"] += moa_same[m]; A["moa_poscontrol"][m]["diff"] += moa_diff[m]

        # ---------- REPLICATE CEILING: same (drug,dose,plate) pairs ----------
        by_cond = defaultdict(list)
        for d, r in flat: by_cond[(d, r["dose"], r["plate"])].append(r)
        for cond, recs in by_cond.items():
            if len(recs) < 2: continue
            for i, j in usample(len(recs), 10, rng):
                mm = pair_metrics(recs[i], recs[j], panel, worst, args.de_k, args.topn)
                for m in METRICS:
                    if mm[m] is not None and mm[m] == mm[m]: CEIL[m]["same"].append(mm[m])

        # ---------- PSEUDOBULK SIZE SWEEP ----------
        if args.pseudobulk:
            for s in pb_sizes:
                pb = {}
                for d in drugs:
                    recs = pool[d]
                    if len(recs) < 2 * s: continue
                    idx = list(range(len(recs))); rng.shuffle(idx)
                    pb[d] = [make_pseudobulk([recs[k] for k in idx[:s]], panel, worst),
                             make_pseudobulk([recs[k] for k in idx[s:2 * s]], panel, worst)]
                pbd = list(pb.keys())
                if len(pbd) < 2: continue
                ps = {m: [] for m in METRICS}; pdf = {m: [] for m in METRICS}
                for d in pbd:
                    mm = pair_metrics(pb[d][0], pb[d][1], panel, worst, args.de_k, args.topn)
                    for m in METRICS:
                        if mm[m] is not None and mm[m] == mm[m]: ps[m].append(mm[m])
                pflat = [(d, rep) for d in pbd for rep in pb[d]]
                pg = [d for d, _ in pflat]
                for i, j in usample(len(pflat), args.diff_pairs_per_cellline, rng, groups=pg, want_diff=True):
                    mm = pair_metrics(pflat[i][1], pflat[j][1], panel, worst, args.de_k, args.topn)
                    for m in METRICS:
                        if mm[m] is not None and mm[m] == mm[m]: pdf[m].append(mm[m])
                for m in METRICS:
                    if ps[m] and pdf[m]:
                        A[f"pb{s}"][m]["gap"][cl] = np.mean(ps[m]) - np.mean(pdf[m])
                        A[f"pb{s}"][m]["same"] += ps[m]; A[f"pb{s}"][m]["diff"] += pdf[m]

    # ---- ceiling per metric (mean of same-condition replicate agreement)
    ceiling = {m: (float(np.mean(CEIL[m]["same"])) if CEIL[m]["same"] else None) for m in METRICS}

    # ---- analyze every (analysis, metric)
    results = {}
    for aname, acc in A.items():
        results[aname] = {}
        for m in METRICS:
            results[aname][m] = analyze_gap(acc[m]["gap"], acc[m]["same"], acc[m]["diff"],
                                            args.n_boot, args.n_perm, args.seed, ceiling=ceiling[m])
    out = {"de_k": args.de_k, "topn": args.topn, "ceiling": ceiling,
           "pb_sizes": pb_sizes, "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    # ---- print
    def row(aname, m):
        r = results[aname][m]
        if not r: return f"  {aname:18s} {MLABEL[m]:8s}   (no data)"
        g = r["gap"]
        frac = r.get("frac_of_recoverable")
        fs = f"{frac*100:5.1f}%" if frac is not None else "  n/a"
        ds = f"{r['cohens_d']:+.2f}" if r["cohens_d"] is not None else " n/a"
        return (f"  {aname:18s} {MLABEL[m]:8s} same={r['same_mean']:.3f} diff={r['diff_mean']:.3f} "
                f"gap={g['mean']:+.3f}[{g['ci_low']:+.3f},{g['ci_high']:+.3f}] p={r['perm_p']:.3g} "
                f"d={ds} frac={fs}")

    order = ["single", "single_samedose", "single_diffplate", "moa_poscontrol"] + [f"pb{s}" for s in pb_sizes]
    logger.info("")
    logger.info("=" * 104)
    logger.info(f"  ceiling (same-cond replicate agreement): " +
                " ".join(f"{MLABEL[m]}={ceiling[m]:.3f}" if ceiling[m] else f"{MLABEL[m]}=NA" for m in METRICS))
    logger.info("  " + "-" * 100)
    for aname in order:
        for m in METRICS:
            logger.info(row(aname, m))
        logger.info("  " + "-" * 100)
    logger.info("  gap=same-diff; p=perm(cellline-clustered); d=Cohen's d; frac=gap/(ceiling-diff)")
    logger.info("  KEY: moa_poscontrol should show a CLEAR gap (test works). If drug gaps << MOA gap,")
    logger.info("       the drug-identity signal is genuinely small even where MOA is detectable.")
    logger.info("=" * 104)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

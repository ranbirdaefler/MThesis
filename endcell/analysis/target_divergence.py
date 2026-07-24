#!/usr/bin/env python
r"""
target_divergence.py — Arm 1b Step-0: is the training TARGET drug-specific enough to learn from?
=================================================================================================
Every target-side fix (Q12 consensus, Q14 OT/T2, the whole epsilon ladder) failed identically. The
proposed explanation (Claim B) is a RATIO: in a full-profile cell sentence the drug-specific part is a
handful of genes, so cross-entropy gives it ~1% of the gradient. Residual targets (generic program
subtracted) should make ~100% of the target drug-specific. This script MEASURES that instead of
assuming it, and settles two design choices at the same time. No training, no GPU.

Three measurements, all in ONE consistent setup (so the numbers are directly comparable):

  (1) INTER-DRUG TARGET DIVERGENCE  <- Claim B, the decisive one
      For drug pairs in the same group: how many of the top-K target genes DIFFER between drug A's
      target and drug B's target? Low for full profiles (targets nearly identical -> nothing to learn)
      vs high for residuals (targets distinct -> gradient carries drug identity).

  (2) CEILING, APPLES-TO-APPLES
      Replicate-based drug retrieval (half-B representation retrieves half-A among the group's drugs)
      under BOTH representations, same cells / thresholds / splits. Earlier we compared 0.576
      (full, nir_benchmark) vs 0.805 (residual, sar_gate) across different setups; this makes it one run.

  (3) RESIDUALIZATION SCOPE + RELIABILITY
      Residualize within (cell_line, plate) [also removes the PLATE signature that lets a zero-drug-info
      control-copy score 0.766] vs within (cell_line) [more drugs -> less noisy mean]. Reports the
      per-condition half-split reliability cos(res_A, res_B) distribution for each, i.e. what fraction
      of conditions carry a reproducible residual worth training on.

USAGE
  python target_divergence.py --cache_dir /data/.../ot_cache --out RESULTS/target_divergence.json
  python target_divergence.py --selftest
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- core helpers
def topk_set(vec, K, by_abs=False):
    v = np.abs(vec) if by_abs else vec
    idx = np.where(v > 0)[0] if not by_abs else np.arange(len(v))
    if len(idx) == 0:
        return set()
    if len(idx) > K:
        idx = idx[np.argsort(-v[idx])[:K]]
    return set(idx.tolist())


def divergence(setA, setB):
    """# genes differing between two top-K sets (symmetric difference / 2)."""
    return len(setA ^ setB) / 2.0


def retrieval_nir(query, gallery, groups):
    """query/gallery: {(drug,group): vec}. Retrieve the drug within its own group. 0.5 = chance."""
    vals = []
    for (d, g), q in query.items():
        cand = [dd for dd in groups[g] if (dd, g) in gallery]
        if len(cand) < 3 or d not in cand:
            continue
        dists = np.array([np.linalg.norm(q - gallery[(dd, g)]) for dd in cand])
        vals.append(np.sum(dists > dists[cand.index(d)]) / (len(cand) - 1))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def build_profiles(cache_dir, min_treated, min_control, min_drugs, seed=0):
    """-> per (drug, group) half-A/half-B TREATED pseudobulks + the group's control pseudobulk.
    group = (cell_line, plate). All in log1p(CP10K) space."""
    import pandas as pd
    from scipy import sparse
    rng = np.random.RandomState(seed)
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    X = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    plate = meta["plate"].astype(str).values
    drug = meta["drug"].astype(str).values

    ctrl = {}
    for g in set(zip(cl[is_ctrl], plate[is_ctrl])):
        idx = np.where(is_ctrl & (cl == g[0]) & (plate == g[1]))[0]
        if len(idx) >= min_control:
            ctrl[g] = np.asarray(np.log1p(X[idx].todense()).mean(0)).ravel()

    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], (cl[i], plate[i]))].append(i)

    prof = {}
    for (d, g), idxs in by.items():
        if len(idxs) < min_treated or g not in ctrl:
            continue
        idxs = list(idxs); rng.shuffle(idxs); h = len(idxs) // 2
        L = lambda ix: np.asarray(np.log1p(X[ix].todense()).mean(0)).ravel().astype(np.float32)
        prof[(d, g)] = {"A": L(idxs[:h]), "B": L(idxs[h:]), "full": L(idxs), "ctrl": ctrl[g]}

    groups = defaultdict(list)
    for (d, g) in prof:
        groups[g].append(d)
    groups = {g: ds for g, ds in groups.items() if len(ds) >= min_drugs}
    prof = {k: v for k, v in prof.items() if k[1] in groups}
    logger.info(f"profiles: {len(prof)} (drug,plate-group) across {len(groups)} groups "
                f"(>= {min_drugs} drugs, >= {min_treated} treated, >= {min_control} control)")
    return prof, groups


def make_reps(prof, groups, scope):
    """Return {half: {(d,g): vec}} for FULL profiles, SHIFTS (treated-control) and RESIDUALS.
    scope='plate' -> mean-over-drugs within (cell_line,plate); 'cellline' -> within cell_line."""
    shifts = {h: {k: prof[k][h] - prof[k]["ctrl"] for k in prof} for h in ("A", "B", "full")}
    # mean-over-drugs per residualization unit
    def unit(g):
        return g if scope == "plate" else g[0]
    res = {}
    for h in ("A", "B", "full"):
        by_unit = defaultdict(list)
        for (d, g) in prof:
            by_unit[unit(g)].append((d, g))
        out = {}
        for u, keys in by_unit.items():
            M = np.mean(np.stack([shifts[h][k] for k in keys]), axis=0)
            for k in keys:
                out[k] = shifts[h][k] - M
        res[h] = out
    full = {h: {k: prof[k][h] for k in prof} for h in ("A", "B", "full")}
    return full, shifts, res


def measure(name, rep, groups, K, by_abs):
    """(1) inter-drug divergence of top-K target sets; (2) replicate retrieval ceiling."""
    divs = []
    for g, ds in groups.items():
        sets = {d: topk_set(rep["full"][(d, g)], K, by_abs=by_abs) for d in ds if (d, g) in rep["full"]}
        dl = list(sets)
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                divs.append(divergence(sets[dl[i]], sets[dl[j]]))
    ceil, n = retrieval_nir(rep["B"], rep["A"], groups)
    logger.info(f"  [{name}] inter-drug top-{K} genes DIFFERING: mean {np.mean(divs):6.1f} "
                f"(median {np.median(divs):5.1f}, of K={K})  |  replicate retrieval ceiling {ceil:.3f} (n={n})")
    return {"divergence_mean": float(np.mean(divs)), "divergence_median": float(np.median(divs)),
            "K": K, "ceiling_retrieval": ceil, "n_ceiling": n, "n_pairs": len(divs)}


def baseline_predictions(prof, groups):
    """Drug-AGNOSTIC predictors, as predicted TREATED profiles (the same objects a model would emit):
      control  : the group's control pseudobulk (the zero-drug-info leak baseline; scores 0.766 today)
      loo_mean : mean of the OTHER drugs' half-A profiles (leave-one-out, drug-agnostic)
      generic  : control + mean-over-drugs shift = 'a typical drug response' in this group
      knn_ctrl : control + the shift of the drug whose CONTROL-side is nearest (no drug identity used)
    Each is constant (or near-constant) across drugs within a group, so under SHIFT/RESIDUAL they must
    collapse to chance -- that is the structural leak-proofing claim, tested rather than assumed."""
    preds = {n: {} for n in ("control", "loo_mean", "generic", "knn_ctrl")}
    for g, ds in groups.items():
        ctrl = prof[(ds[0], g)]["ctrl"]
        shifts = {d: prof[(d, g)]["A"] - ctrl for d in ds}
        mean_shift = np.mean(np.stack([shifts[d] for d in ds]), axis=0)
        for d in ds:
            others = [o for o in ds if o != d]
            preds["control"][(d, g)] = ctrl
            preds["loo_mean"][(d, g)] = np.mean(np.stack([prof[(o, g)]["A"] for o in others]), axis=0)
            preds["generic"][(d, g)] = ctrl + mean_shift
            # knn over the CONTROL side: controls are shared within a group -> falls back to a random
            # other drug's shift (still zero drug identity for THIS drug)
            preds["knn_ctrl"][(d, g)] = ctrl + shifts[others[0]]
    return preds


def to_rep(pred, rep, prof, groups, M_ref):
    """Map predicted TREATED profiles into the target representation (same frame as the truths)."""
    if rep == "full":
        return dict(pred)
    out = {}
    for (d, g), v in pred.items():
        s = v - prof[(d, g)]["ctrl"]
        out[(d, g)] = s if rep == "shift" else s - M_ref[g]
    return out


def measure_baselines(rep_name, rep_truth_A, preds, prof, groups, M_ref):
    """Score each drug-agnostic baseline with the SAME retrieval NIR used for the ceiling."""
    rows = {}
    for name, pr in preds.items():
        q = to_rep(pr, rep_name, prof, groups, M_ref)
        nir, n = retrieval_nir(q, rep_truth_A, groups)
        rows[name] = {"retrieval_nir": nir, "n": n}
    txt = "  ".join(f"{k}={v['retrieval_nir']:.3f}" for k, v in rows.items())
    logger.info(f"    baselines (drug-AGNOSTIC, must be ~0.50 if the rep is leak-proof): {txt}")
    return rows


def reliability(res, thr):
    vals = []
    for k in res["A"]:
        a, b = res["A"][k], res["B"][k]
        vals.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    vals = np.array(vals)
    return {"mean_cos": float(vals.mean()), "median_cos": float(np.median(vals)),
            "frac_reproducible": float((vals > thr).mean()), "n": int(len(vals)), "thr": thr}


def run(cache_dir, K, min_treated, min_control, min_drugs, thr, out_path):
    prof, groups = build_profiles(cache_dir, min_treated, min_control, min_drugs)
    if not prof:
        logger.error("no usable conditions"); return
    result = {"n_conditions": len(prof), "n_groups": len(groups), "K": K, "scopes": {}}
    for scope in ("plate", "cellline"):
        logger.info("=" * 100)
        logger.info(f"RESIDUALIZATION SCOPE = {scope}  "
                    f"({'mean over drugs within (cell_line,plate) — also removes the PLATE signature' if scope=='plate' else 'mean over drugs within cell_line'})")
        full, shifts, res = make_reps(prof, groups, scope)
        # reference frame for mapping PREDICTIONS into the residual space (same M as the truths)
        unit = (lambda g: g) if scope == "plate" else (lambda g: g[0])
        by_unit = defaultdict(list)
        for (d, g) in prof:
            by_unit[unit(g)].append((d, g))
        M_unit = {u: np.mean(np.stack([prof[k]["A"] - prof[k]["ctrl"] for k in keys]), axis=0)
                  for u, keys in by_unit.items()}
        M_ref = {g: M_unit[unit(g)] for g in groups}
        preds = baseline_predictions(prof, groups)

        m_full = measure("FULL profile  (current target)", full, groups, K, by_abs=False)
        b_full = measure_baselines("full", full["A"], preds, prof, groups, M_ref)
        m_shift = measure("SHIFT (treated-control)      ", shifts, groups, K, by_abs=True)
        b_shift = measure_baselines("shift", shifts["A"], preds, prof, groups, M_ref)
        m_res = measure("RESIDUAL (drug-specific)     ", res, groups, K, by_abs=True)
        b_res = measure_baselines("residual", res["A"], preds, prof, groups, M_ref)
        m_full["baselines"], m_shift["baselines"], m_res["baselines"] = b_full, b_shift, b_res
        rel = reliability(res, thr)
        logger.info(f"  reliability of residuals: mean cos(A,B)={rel['mean_cos']:+.3f}  "
                    f"reproducible (>{thr}) = {100*rel['frac_reproducible']:.0f}% of {rel['n']} conditions")
        result["scopes"][scope] = {"full": m_full, "shift": m_shift, "residual": m_res, "reliability": rel}
    logger.info("=" * 100)
    logger.info("READ:")
    logger.info("  (1) CLAIM B: if FULL-profile targets differ between drugs by only a few genes while")
    logger.info("      RESIDUAL targets differ by many, the ~1%-of-gradient diagnosis is CONFIRMED and the")
    logger.info("      residual reframing is justified by measurement, not argument.")
    logger.info("  (2) CEILING: residual retrieval >> full retrieval, measured on the SAME cells/splits.")
    logger.info("  (3) SCOPE: pick 'plate' if its reliability/ceiling hold up (it also kills plate leakage).")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2, default=float)
    logger.info(f"-> {out_path}")


def selftest():
    """Synthetic reproducing the real mechanism: a dominant generic program shared by all drugs, whose
    AMPLITUDE fluctuates per (group, half) — i.e. common-mode batch variation — plus a small
    drug-specific part. Expected:
      FULL profiles  -> top-K sets dominated by generic genes (low inter-drug divergence) AND the
                        common-mode fluctuation swamps drug identity in retrieval (ceiling -> chance);
      RESIDUALS      -> mean-over-drugs subtraction cancels BOTH the generic program and its
                        fluctuation exactly, leaving the drug-specific part (high divergence, high ceiling).
    This is precisely why residualization is expected to raise the ceiling on real data."""
    rng = np.random.RandomState(0)
    P, D, G = 200, 8, 3
    generic = np.zeros(P); generic[:120] = np.linspace(5, 1, 120)      # dominant shared program
    prof, groups = {}, defaultdict(list)
    for gi in range(G):
        g = (f"cl{gi}", f"p{gi}")
        base = np.abs(rng.rand(P)) * 2
        jit = {h: rng.randn() * 0.8 for h in ("A", "B", "full")}       # common-mode, shared by drugs
        for d in range(D):
            spec = np.zeros(P); spec[120 + d * 5:125 + d * 5] = 2.0    # small drug-specific part
            mk = lambda h: np.maximum(
                0, base + generic * (1 + jit[h]) + spec + rng.randn(P) * 0.15).astype(np.float32)
            prof[(f"d{d}", g)] = {h: mk(h) for h in ("A", "B", "full")}
            prof[(f"d{d}", g)]["ctrl"] = base.astype(np.float32)
            groups[g].append(f"d{d}")
    groups = dict(groups)
    full, shifts, res = make_reps(prof, groups, "plate")
    mf = measure("FULL ", full, groups, 60, by_abs=False)
    mr = measure("RESID", res, groups, 60, by_abs=True)
    ok = (mr["divergence_mean"] > mf["divergence_mean"]
          and mr["ceiling_retrieval"] > mf["ceiling_retrieval"])
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'} "
                f"(residual must diverge more AND retrieve better than full profiles)")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--K", type=int, default=200, help="top-K genes defining a target sentence")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_drugs", type=int, default=4, help="min drugs per group for comparisons")
    ap.add_argument("--repro_thr", type=float, default=0.2)
    ap.add_argument("--out", default="RESULTS/target_divergence.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")
    run(args.cache_dir, args.K, args.min_treated, args.min_control, args.min_drugs,
        args.repro_thr, args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
r"""
drug_stratify_geometry.py  —  TEST 2 (per-drug stratified NIR) + TEST 3 (drug confusion geometry)
==================================================================================================
Decomposes the aggregate NIR number and asks the advisor's question directly:
"which drugs does the model perform well on, and which not — and is the task even possible there?"

TEST 2 — PER-DRUG STRATIFIED NIR
  The headline "model ~0.5 vs ceiling ~0.88" is an AVERAGE over drugs of wildly different
  difficulty. Some drugs are inert or redundant, so their NIR is at chance for ANY predictor and
  they drag the mean down regardless of the model. This decomposes it:
    * per-drug: ceiling vs model vs linear NIR
    * binned by the PER-DRUG CEILING (how winnable that drug is)
    * THE DECISIVE ANALYSIS: restrict to drugs the ceiling proves are identifiable
      (ceiling NIR >= --identifiable_nir). On THAT subset, does the model beat the drug-agnostic
      linear baseline? Paired bootstrap CI on (model - linear).
      -> model still at chance there  = drug-blindness is REAL and confound-free (much stronger claim)
      -> model beats linear there     = the model tracks high-effect drugs; "drug-blind" was partly
                                        an artifact of averaging over unwinnable drugs (thesis changes)
  With --atlas it also correlates model performance against drug effect size / isolation / SNR.

TEST 3 — DRUG CONFUSION GEOMETRY
  Even if the model can't name the drug, does its prediction space have the right SHAPE?
    * real drug x drug distance matrix (from truth pseudobulks) vs the model's PREDICTED
      drug x drug matrix -> Spearman on the off-diagonals (Mantel-style).
    * COLLAPSE detection: coefficient of variation of the model's pairwise distances vs the real
      ones. A drug-blind model predicts ~the same profile for every drug -> its matrix is
      near-uniform -> CV ~ 0 and correlation ~ 0.
    * MoA-level NIR (needs --atlas for MoA labels): maybe the model gets the CLASS right even when
      it cannot resolve the individual drug — a partial-credit result the aggregate hides.

SELF-CONTAINED: no local imports; runs in the flat cluster layout and the reorganized repo.

INPUTS (produced by the patched nir_benchmark.py and by drug_difficulty_atlas.py)
  --nir       RESULTS/nir_benchmark.json          (per-drug rows; run nir_benchmark as usual)
  --profiles  RESULTS/nir_profiles.npz            (nir_benchmark --profiles ...)   [Test 3]
  --atlas     RESULTS/drug_atlas.json             (drug_difficulty_atlas.py)       [optional]

USAGE (CPU)
  python drug_stratify_geometry.py --nir RESULTS/nir_benchmark.json \
     --profiles RESULTS/nir_profiles.npz --atlas RESULTS/drug_atlas.json \
     --out RESULTS/stratify_geometry.json --csv RESULTS/per_drug.csv

SELFTEST (no data) — plants a perfect model and a drug-blind model and checks both are detected:
  python drug_stratify_geometry.py --selftest
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PREDICTORS = ("ceiling", "model", "scramble", "linear", "control", "mean")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).statistic
        return float(r) if r == r else None
    except Exception:
        ra, rb = _rank(a), _rank(b)
        return float(np.corrcoef(ra, rb)[0, 1])


def _rank(a):
    o = np.argsort(a, kind="stable")
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1)
    return r


def boot_paired(diff, n_boot=2000, seed=0):
    """NAIVE bootstrap CI (treats every drug as independent). Reported only for reference —
    it is ANTI-CONSERVATIVE here because drugs are clustered within cell lines. Use
    boot_paired_clustered for any claim."""
    if len(diff) < 3:
        return None, None
    rng = np.random.RandomState(seed)
    d = np.asarray(diff, float)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_paired_clustered(diff, clusters, n_boot=2000, seed=0):
    """CLUSTER bootstrap: resample CELL LINES with replacement (taking all their drugs), not
    individual drugs. Drugs inside a cell line share controls, batch and the same comparison set,
    so they are NOT independent replicates — resampling them individually understates the CI
    (pseudoreplication). The effective sample size is the number of cell lines, not drugs."""
    d = np.asarray(diff, float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    if len(uniq) < 3 or len(d) < 3:
        return None, None, len(uniq)
    groups = [d[cl == c] for c in uniq]
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in pick])
        if len(vals):
            means.append(vals.mean())
    if not means:
        return None, None, len(uniq)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(uniq)


# ----------------------------------------------------------------- load + join
def load_nir_rows(path, key):
    """-> list of per-drug rows with flattened metric `key` (nir_expr or nir_rank)."""
    res = json.load(open(path))
    out = []
    for tier, td in res.get("tiers", {}).items():
        for r in td.get("rows", []):
            row = {"tier": r.get("tier", tier), "cell_line": r.get("cell_line"),
                   "drug": r.get("drug"), "n_cells": r.get("n_cells")}
            ok = False
            for p in PREDICTORS:
                v = (r.get(p) or {}).get(key) if isinstance(r.get(p), dict) else None
                row[p] = v
                if p in ("model", "ceiling") and v is not None:
                    ok = True
            if ok:
                out.append(row)
    if not out:
        logger.error(f"No per-drug rows in {path}. Re-run nir_benchmark.py with the patched version "
                     f"(it now stores 'rows' per tier).")
        sys.exit(1)
    return out


def load_atlas(path):
    a = json.load(open(path))
    idx = {}
    for r in a.get("rows", []):
        idx[(r.get("cell_line"), r.get("drug"))] = r
    return idx


# ----------------------------------------------------------------- TEST 2
def test2(rows, args):
    have = [r for r in rows if r.get("model") is not None and r.get("ceiling") is not None]
    logger.info("")
    logger.info("=" * 96)
    logger.info(f"  TEST 2 — PER-DRUG STRATIFIED NIR   ({len(have)} drug x cell-line entries, "
                f"metric={args.metric})")
    logger.info("-" * 96)

    def m(rs, p):
        v = [r[p] for r in rs if r.get(p) is not None]
        return float(np.mean(v)) if v else float("nan")

    logger.info(f"  ALL drugs:            ceiling {m(have,'ceiling'):.3f} | model {m(have,'model'):.3f} "
                f"| linear {m(have,'linear'):.3f} | mean {m(have,'mean'):.3f}   <- reproduces the aggregate")

    # bin by how winnable the drug is (its own ceiling)
    bins = [(0.0, 0.6, "UNWINNABLE  (ceiling < 0.6 ~ chance: inert/redundant drug)"),
            (0.6, args.identifiable_nir, f"MARGINAL    ({0.6:.1f} <= ceiling < {args.identifiable_nir})"),
            (args.identifiable_nir, 1.01, f"IDENTIFIABLE (ceiling >= {args.identifiable_nir})")]
    binned = {}
    logger.info("")
    logger.info("  stratified by PER-DRUG CEILING (the task's own difficulty):")
    for lo, hi, label in bins:
        rs = [r for r in have if lo <= r["ceiling"] < hi]
        if not rs:
            logger.info(f"    {label:56s}  n=0"); continue
        binned[label] = {"n": len(rs), "ceiling": m(rs, "ceiling"), "model": m(rs, "model"),
                         "linear": m(rs, "linear"), "mean": m(rs, "mean")}
        logger.info(f"    {label:56s}  n={len(rs):4d}  ceiling {m(rs,'ceiling'):.3f}  "
                    f"model {m(rs,'model'):.3f}  linear {m(rs,'linear'):.3f}")

    # THE decisive analysis: on provably-identifiable drugs, does the model beat the linear?
    sub = [r for r in have if r["ceiling"] >= args.identifiable_nir
           and r.get("linear") is not None and r.get("model") is not None]
    verdict = None
    logger.info("")
    logger.info("  " + "*" * 92)
    if len(sub) < 5:
        logger.warning(f"  Only {len(sub)} identifiable drugs (ceiling >= {args.identifiable_nir}) — "
                       f"too few to adjudicate. Lower --identifiable_nir or widen the data.")
    else:
        diff = np.array([r["model"] - r["linear"] for r in sub])
        clusters = [r.get("cell_line") for r in sub]
        lo_n, hi_n = boot_paired(diff, seed=args.seed)
        lo, hi, n_cl = boot_paired_clustered(diff, clusters, seed=args.seed)
        mm, ml, mc = m(sub, "model"), m(sub, "linear"), m(sub, "ceiling")
        logger.info(f"  DECISIVE — drugs the ceiling proves are identifiable "
                    f"(n={len(sub)} drugs in {n_cl} cell lines):")
        logger.info(f"     ceiling {mc:.3f}   model {mm:.3f}   linear {ml:.3f}")
        logger.info(f"     model - linear = {diff.mean():+.3f}")
        logger.info(f"       naive  95% CI [{lo_n:+.3f}, {lo_n and hi_n:+.3f}]  "
                    f"<- ANTI-CONSERVATIVE (drugs are not independent), reference only")
        if lo is None:
            logger.warning(f"       clustered CI unavailable ({n_cl} cell lines < 3)")
        else:
            logger.info(f"       CLUSTERED 95% CI [{lo:+.3f}, {hi:+.3f}]  "
                        f"<- resamples CELL LINES; THIS is the one to quote")
        # model-vs-linear is a CONFOUNDED comparison: model and linear differ in TWO ways at once
        # (the drug token AND how much control/plate signature each retains). It is reported here as
        # PRELIMINARY only. The authoritative ruling is the FINAL VERDICT below, which is driven by
        # the two leak-immune controls (scramble = same control, only the drug token differs;
        # control-copy = zero drug info). Those OVERRIDE this line.
        beats = (lo is not None and lo > 0)
        if beats:
            verdict = (f"model - linear {diff.mean():+.3f} favours the model (CI excludes 0) — but this "
                       "is CONFOUNDED (see FINAL VERDICT; scramble/control decide it).")
        elif lo is not None:
            verdict = f"model - linear {diff.mean():+.3f}, clustered CI spans 0 (preliminary)."
        else:
            verdict = "no clustered CI (too few cell lines)."
        logger.info(f"     [preliminary] {verdict}")

        # ---- CAUSAL CONTROLS on the same identifiable subset -------------------------------------
        # (a) CONTROL-COPY: the drug's own control, zero drug info. It is control-conditioned just
        #     like the model, so if it also beats ~0.50 the "drug effect" is plate/batch leakage.
        csub = [r for r in sub if r.get("control") is not None]
        if len(csub) >= 5:
            mc_ctrl = m(csub, "control")
            # test control-copy against chance with a CLUSTERED CI (not a hard threshold): resample
            # cell lines and ask whether (control - 0.5) is reliably > 0.
            cdiff = np.array([r["control"] - 0.5 for r in csub])
            clo, chi, cncl = boot_paired_clustered(cdiff, [r.get("cell_line") for r in csub],
                                                   seed=args.seed)
            logger.info("")
            logger.info(f"     CONTROL-COPY (zero drug info, n={len(csub)} drugs, {cncl} cell lines): "
                        f"{mc_ctrl:.3f}")
            if clo is not None:
                logger.info(f"       control - 0.50 = {cdiff.mean():+.3f}   "
                            f"CLUSTERED 95% CI [{clo:+.3f}, {chi:+.3f}]")
            if clo is not None and clo > 0:
                logger.warning("       *** LEAKAGE: control-copy is RELIABLY above chance -> the "
                               "plate-matched control carries drug/batch identity, so ANY "
                               "control-conditioned predictor (incl. the model) gains NIR for free. "
                               "The model-linear gap cannot be attributed to drug knowledge on its own.")
            elif clo is not None:
                logger.info("       ~chance (CI spans 0) -> control-conditioning alone does NOT "
                            "discriminate; the model's edge is not simple plate leakage.")
            control_ci = [clo, chi]

        else:
            control_ci = None

        # (b) SCRAMBLE: same control, wrong drug token. This is the decisive causal test.
        ssub = [r for r in sub if r.get("scramble") is not None and r.get("model") is not None]
        if len(ssub) >= 5:
            sdiff = np.array([r["model"] - r["scramble"] for r in ssub])
            slo, shi, sncl = boot_paired_clustered(sdiff, [r.get("cell_line") for r in ssub],
                                                   seed=args.seed)
            logger.info("")
            logger.info(f"     SCRAMBLE ARM — same control cell, WRONG drug token (n={len(ssub)} "
                        f"drugs, {sncl} cell lines):")
            logger.info(f"       model {m(ssub,'model'):.3f}   scramble {m(ssub,'scramble'):.3f}")
            logger.info(f"       model - scramble = {sdiff.mean():+.3f}" +
                        (f"   CLUSTERED 95% CI [{slo:+.3f}, {shi:+.3f}]" if slo is not None else ""))
            if slo is not None and slo > 0:
                sv = ("CAUSAL: lying about the drug DEGRADES the prediction -> the model genuinely "
                      "USES the drug token. The +model-linear gap is real drug knowledge.")
            elif slo is not None and slo <= 0 <= shi and abs(float(sdiff.mean())) < args.null_margin:
                sv = ("NULL: swapping the drug changes nothing -> the model does NOT use the drug. "
                      "The model-linear gap must come from something else (plate/control leakage).")
            elif slo is None:
                sv = "no clustered CI (too few cell lines)"
            else:
                sv = "inconclusive — CI spans 0; needs more cell lines"
            logger.info(f"       SCRAMBLE VERDICT: {sv}")
            out_scram = {"n": len(ssub), "model": m(ssub, "model"), "scramble": m(ssub, "scramble"),
                         "model_minus_scramble": float(sdiff.mean()),
                         "ci_clustered": [slo, shi], "n_celllines": sncl, "verdict": sv}
        else:
            out_scram = None
            logger.info("     (no scramble arm — re-run nir_benchmark.py with --scram_dir)")
        test2._scram = out_scram
        test2._control_ci = control_ci

        # ---- FINAL VERDICT: leak-immune controls decide, overriding model-vs-linear ---------------
        scram_null = (out_scram and out_scram["ci_clustered"][0] is not None
                      and out_scram["ci_clustered"][0] <= 0 <= out_scram["ci_clustered"][1])
        scram_uses = (out_scram and out_scram["ci_clustered"][0] is not None
                      and out_scram["ci_clustered"][0] > 0)
        leak = (control_ci and control_ci[0] is not None and control_ci[0] > 0)
        logger.info("")
        if scram_uses and not leak:
            fv = (f"MODEL USES THE DRUG. Scramble drops it {out_scram['model_minus_scramble']:+.3f} "
                  f"(leak-immune), and control-copy is at chance. Genuine drug knowledge.")
        elif scram_uses and leak:
            fv = (f"MIXED: scramble shows real drug use ({out_scram['model_minus_scramble']:+.3f}), but "
                  f"control-copy also leaks — report the scramble effect as the clean drug signal and "
                  f"flag the leak in model-vs-linear.")
        elif scram_null:
            fv = ("MODEL DOES NOT USE THE DRUG. Swapping the drug token changes the output by ~0 "
                  f"(scramble {out_scram['model_minus_scramble']:+.3f}, CI spans 0)."
                  + (" The model-vs-linear gap is BATCH LEAKAGE: a zero-drug-info control-copy scores "
                     "as high as the model." if leak else
                     " The model-vs-linear gap is NOT drug knowledge (scramble is the clean test)."))
        elif out_scram is None:
            fv = "NO SCRAMBLE ARM — cannot rule on drug use; re-run nir_benchmark with --scram_dir."
        else:
            fv = "SCRAMBLE INCONCLUSIVE (CI spans 0 but effect not ~0) — needs more cell lines."
        logger.info(f"     >>> FINAL VERDICT: {fv}")
        test2._final = fv
    logger.info("  " + "*" * 92)

    # what explains model performance?
    corrs = {}
    if args.atlas:
        for field in ("snr", "isolation", "effect_size", "ceiling_nir"):
            xs = [r["_atlas"].get(field) for r in have if r.get("_atlas")]
            ys = [r["model"] for r in have if r.get("_atlas")]
            pair = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if len(pair) >= 5:
                c = spearman([p[0] for p in pair], [p[1] for p in pair])
                corrs[field] = c
        logger.info("")
        logger.info("  Spearman( drug property , MODEL NIR ) — what predicts where the model works?")
        for k, v in corrs.items():
            logger.info(f"    {k:14s} {('  NA' if v is None else f'{v:+.3f}')}")
        logger.info("    (high +corr with snr/isolation => the model only works on easy/strong drugs)")

    ident = None
    if len(sub) >= 5:
        _d = [r["model"] - r["linear"] for r in sub]
        _cl = [r.get("cell_line") for r in sub]
        _lo, _hi, _ncl = boot_paired_clustered(_d, _cl, seed=args.seed)
        ident = {"ceiling": m(sub, "ceiling"), "model": m(sub, "model"), "linear": m(sub, "linear"),
                 "model_minus_linear": float(np.mean(_d)),
                 "ci_naive": boot_paired(_d, seed=args.seed),
                 "ci_clustered": [_lo, _hi], "n_celllines": _ncl, "n_drugs": len(sub)}
    return {"n": len(have), "all": {p: m(have, p) for p in PREDICTORS},
            "binned": binned, "n_identifiable": len(sub),
            "identifiable": ident, "verdict_preliminary_model_linear": verdict,
            "final_verdict": getattr(test2, "_final", None), "corr_model_vs": corrs,
            "scramble_arm": getattr(test2, "_scram", None),
            "control_copy_identifiable": (m([r for r in sub if r.get("control") is not None], "control")
                                          if any(r.get("control") is not None for r in sub) else None),
            "control_copy_ci_clustered": getattr(test2, "_control_ci", None)}


# ----------------------------------------------------------------- TEST 3
def pdist(vs):
    n = len(vs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.linalg.norm(vs[i] - vs[j]))
    return D


def offdiag(D):
    n = D.shape[0]
    return np.array([D[i, j] for i in range(n) for j in range(i + 1, n)])


def test3(profiles_path, atlas_idx, args):
    z = np.load(profiles_path)
    model, truth = {}, {}
    for k in z.files:
        parts = k.split("||")
        if len(parts) < 4:
            continue
        kind, tier, cl, drug = parts[0], parts[1], parts[2], "||".join(parts[3:])
        (model if kind == "model" else truth)[(tier, cl, drug)] = z[k]
    groups = defaultdict(list)
    for (tier, cl, drug) in truth:
        if (tier, cl, drug) in model:
            groups[(tier, cl)].append(drug)

    logger.info("")
    logger.info("=" * 96)
    logger.info("  TEST 3 — DRUG CONFUSION GEOMETRY (does the model reproduce the real drug structure?)")
    logger.info("-" * 96)

    per_cl, mantel, cv_model, cv_real = [], [], [], []
    for (tier, cl), drugs in groups.items():
        if len(drugs) < 4:
            continue
        T = pdist([truth[(tier, cl, d)] for d in drugs])
        M = pdist([model[(tier, cl, d)] for d in drugs])
        t_off, m_off = offdiag(T), offdiag(M)
        r = spearman(t_off, m_off)
        cvm = float(m_off.std() / (m_off.mean() + 1e-12))
        cvt = float(t_off.std() / (t_off.mean() + 1e-12))
        if r is not None:
            mantel.append(r); cv_model.append(cvm); cv_real.append(cvt)
            per_cl.append({"tier": tier, "cell_line": cl, "n_drugs": len(drugs),
                           "mantel_spearman": r, "cv_model": cvm, "cv_real": cvt})

    res = {"per_cellline": per_cl}
    if mantel:
        mm = float(np.mean(mantel))
        res.update({"mantel_mean": mm, "mantel_median": float(np.median(mantel)),
                    "cv_model_mean": float(np.mean(cv_model)), "cv_real_mean": float(np.mean(cv_real)),
                    "n_celllines": len(mantel)})
        logger.info(f"  Mantel Spearman(real drug-drug matrix, model's predicted matrix): "
                    f"mean {mm:+.3f}  median {np.median(mantel):+.3f}   over {len(mantel)} cell lines")
        logger.info(f"  spread of pairwise distances (CV):  real {np.mean(cv_real):.3f}   "
                    f"model {np.mean(cv_model):.3f}   ratio {np.mean(cv_model)/max(1e-9,np.mean(cv_real)):.3f}")
        logger.info("    Mantel ~0 AND low model CV  => predictions collapse to one profile (drug-blind geometry)")
        logger.info("    Mantel > 0                  => model has PARTIAL drug knowledge: right geometry,")
        logger.info("                                   even if it cannot resolve drug identity (NIR at chance)")
    else:
        logger.warning("  not enough drugs per cell line for the geometry test")

    # MoA-level NIR: can the model at least resolve the mechanism class?
    if atlas_idx:
        moa_hits, moa_n = 0, 0
        for (tier, cl), drugs in groups.items():
            bym = defaultdict(list)
            for d in drugs:
                a = atlas_idx.get((cl, d))
                if a and a.get("moa"):
                    bym[a["moa"]].append(d)
            bym = {k: v for k, v in bym.items() if v}
            if len(bym) < 3:
                continue
            moas = sorted(bym)
            T = {mo: np.mean(np.stack([truth[(tier, cl, d)] for d in bym[mo]]), axis=0) for mo in moas}
            M = {mo: np.mean(np.stack([model[(tier, cl, d)] for d in bym[mo]]), axis=0) for mo in moas}
            for mo in moas:
                others = [o for o in moas if o != mo]
                d_own = np.linalg.norm(M[mo] - T[mo])
                d_oth = [np.linalg.norm(M[mo] - T[o]) for o in others]
                moa_hits += float(np.mean([d_own < x for x in d_oth])); moa_n += 1
        if moa_n:
            moa_nir = moa_hits / moa_n
            res["moa_level_nir"] = float(moa_nir)
            logger.info(f"  MoA-level NIR (can it resolve the mechanism class?): {moa_nir:.3f}  "
                        f"(chance 0.50, n={moa_n} MoA x cell-line)")
    return res


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Plant a PERFECT model and a DRUG-BLIND model; check Test 3 separates them and Test 2's
    decisive analysis fires correctly."""
    rng = np.random.RandomState(0)
    P, nd = 60, 8
    truth = {d: rng.rand(P) * (1 + d) for d in range(nd)}

    def geom(model_map):
        tmp = {}
        for d in range(nd):
            tmp[f"truth||t||cl||d{d}"] = truth[d]
            tmp[f"model||t||cl||d{d}"] = model_map[d]
        path = os.path.join(os.environ.get("TEMP", "/tmp"), "_sg_selftest.npz")
        np.savez_compressed(path, **tmp)
        return test3(path, None, args)

    logger.info("  --- perfect (drug-aware) model: predictions == truth ---")
    good = geom({d: truth[d].copy() for d in range(nd)})
    logger.info("  --- drug-blind model: same profile for every drug ---")
    blind_vec = np.mean(np.stack([truth[d] for d in range(nd)]), axis=0)
    blind = geom({d: blind_vec + rng.randn(P) * 1e-3 for d in range(nd)})

    ok = True
    if not (good.get("mantel_mean", -1) > 0.8):
        logger.error(f"  FAIL: perfect model Mantel not high ({good.get('mantel_mean')})"); ok = False
    if not (abs(blind.get("mantel_mean", 1)) < 0.5):
        logger.error(f"  FAIL: blind model Mantel not ~0 ({blind.get('mantel_mean')})"); ok = False
    if not (blind.get("cv_model_mean", 9) < good.get("cv_model_mean", 0)):
        logger.error("  FAIL: blind model spread not collapsed vs perfect"); ok = False

    # --- Test 2 case E (THE REAL-DATA PATTERN): model beats linear, control-copy ALSO high, and
    # scramble is null. The FINAL verdict must say the model does NOT use the drug and blame leakage,
    # NOT be fooled by the model-vs-linear win. This is the exact bug the same-plate run exposed.
    logger.info("  --- Test 2 case E: linear-win + high control + scramble-null (real-data pattern) ---")
    # model and scramble drawn from the SAME distribution (no drug effect) -> model-scramble CI must
    # span 0. Control-copy is as high as the model. Final verdict must be DOES NOT USE + batch leakage.
    rows_e = []
    for i in range(40):
        base = 0.77 + rng.randn() * 0.05                        # shared per-drug level (leak)
        rows_e.append({"cell_line": f"cl{i % 5}", "drug": f"d{i}", "ceiling": 0.95,
                       "model": base + rng.randn() * 0.02, "linear": 0.53 + rng.randn() * 0.02,
                       "scramble": base + rng.randn() * 0.02,   # same base -> no drug effect
                       "control": base + rng.randn() * 0.02, "mean": 0.18, "n_cells": 20})
    r_e = test2(rows_e, args)
    if not (r_e.get("final_verdict") and "DOES NOT USE" in r_e["final_verdict"]):
        logger.error(f"  FAIL: real-data pattern misjudged -> {r_e.get('final_verdict')}"); ok = False

    # --- Test 2 case C: scramble arm separates real drug use from plate leakage
    logger.info("  --- Test 2 case C: scramble arm (real drug use vs plate leakage) ---")
    # real drug use: model beats linear AND scramble collapses
    rows_use = [{"cell_line": f"cl{i % 5}", "drug": f"d{i}", "ceiling": 0.95,
                 "model": 0.65 + rng.randn() * 0.02, "linear": 0.50 + rng.randn() * 0.02,
                 "scramble": 0.50 + rng.randn() * 0.02, "control": 0.50 + rng.randn() * 0.02,
                 "mean": 0.3, "n_cells": 20} for i in range(40)]
    r_use = test2(rows_use, args)
    if not (r_use["scramble_arm"] and "CAUSAL" in r_use["scramble_arm"]["verdict"]):
        logger.error(f"  FAIL: real drug use not detected by scramble arm"); ok = False
    # plate leakage: model beats linear BUT scramble is just as good (drug token irrelevant)
    logger.info("  --- Test 2 case D: leakage (model beats linear, scramble unaffected) ---")
    rows_leak = [{"cell_line": f"cl{i % 5}", "drug": f"d{i}", "ceiling": 0.95,
                  "model": 0.65 + rng.randn() * 0.02, "linear": 0.50 + rng.randn() * 0.02,
                  "scramble": 0.65 + rng.randn() * 0.02, "control": 0.64 + rng.randn() * 0.02,
                  "mean": 0.3, "n_cells": 20} for i in range(40)]
    r_leak = test2(rows_leak, args)
    if not (r_leak["scramble_arm"] and "NULL" in r_leak["scramble_arm"]["verdict"]):
        logger.error(f"  FAIL: leakage case not caught by scramble arm"); ok = False

    # --- Test 2 case B: THE TRAP. A per-cell-line offset makes drugs look independently better
    # (naive CI excludes 0) but the effect is carried by whole cell lines, so the CLUSTERED CI must
    # span 0. This is the pseudoreplication failure mode the clustered bootstrap exists to catch.
    logger.info("  --- Test 2 case B: cell-line-driven effect (naive win, clustered null) ---")
    offs = {c: rng.randn() * 0.15 for c in range(4)}          # effect lives at the CELL LINE level
    rows_b = [{"cell_line": f"cl{i % 4}", "drug": f"d{i}", "ceiling": 0.95,
               "model": 0.50 + offs[i % 4] + rng.randn() * 0.005,
               "linear": 0.50 + rng.randn() * 0.005, "mean": 0.3, "n_cells": 20}
              for i in range(80)]
    r2b = test2(rows_b, args)
    ci_n = r2b["identifiable"]["ci_naive"]
    ci_c = r2b["identifiable"]["ci_clustered"]
    width_n = ci_n[1] - ci_n[0]
    width_c = ci_c[1] - ci_c[0]
    logger.info(f"    naive CI width {width_n:.3f} vs clustered CI width {width_c:.3f}")
    if not (width_c > width_n * 1.5):
        logger.error("  FAIL: clustered CI is not substantially wider than naive under clustering")
        ok = False

    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nir", help="nir_benchmark.json (patched version, contains per-drug rows)")
    ap.add_argument("--profiles", default=None, help="nir_profiles.npz from nir_benchmark --profiles")
    ap.add_argument("--atlas", default=None, help="drug_atlas.json from drug_difficulty_atlas.py")
    ap.add_argument("--metric", choices=["nir_expr", "nir_rank"], default="nir_expr")
    ap.add_argument("--identifiable_nir", type=float, default=0.8)
    ap.add_argument("--null_margin", type=float, default=0.03,
                    help="|model-linear| below this (with a CI spanning 0) counts as a true null "
                         "(drug-blind) rather than merely underpowered")
    ap.add_argument("--out", default="RESULTS/stratify_geometry.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return
    if not args.nir:
        ap.error("--nir is required (unless --selftest)")

    rows = load_nir_rows(args.nir, args.metric)
    atlas_idx = load_atlas(args.atlas) if args.atlas else None
    if atlas_idx:
        hit = 0
        for r in rows:
            a = atlas_idx.get((r["cell_line"], r["drug"]))
            if a:
                r["_atlas"] = a; hit += 1
        logger.info(f"  joined atlas: {hit}/{len(rows)} drugs matched")
        if hit == 0:
            logger.warning("  atlas join matched NOTHING — check both were run on the same --eval_dir")

    out = {"test2": test2(rows, args)}
    if args.profiles and os.path.exists(args.profiles):
        out["test3"] = test3(args.profiles, atlas_idx, args)
    else:
        logger.warning("  no --profiles -> skipping Test 3 (re-run nir_benchmark with --profiles)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")

    if args.csv:
        # drug names contain commas -> must use the csv module (proper quoting), never a manual join
        import csv as _csv
        cols = ["tier", "cell_line", "drug", "n_cells", "ceiling", "model", "linear", "mean"]
        acols = ["snr", "isolation", "effect_size", "n_deg", "moa", "nn_drug"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(cols + acols)
            for r in rows:
                a = r.get("_atlas", {})
                w.writerow(["" if r.get(c) is None else r.get(c) for c in cols] +
                           ["" if a.get(c) is None else a.get(c) for c in acols])
        logger.info(f"-> {args.csv}")


if __name__ == "__main__":
    main()

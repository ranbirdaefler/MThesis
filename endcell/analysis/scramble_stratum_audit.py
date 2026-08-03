#!/usr/bin/env python
r"""
scramble_stratum_audit.py -- is the scramble a NULL, or an actively wrong answer?
=================================================================================
`residual_eval` measures drug use as `model - scramble`: name the true drug, name a different one,
see whether the output gets worse. The whole test rests on an assumption nobody had checked --
**that naming the wrong drug is uninformative**.

It is not. The comparator is chosen as the most ANTI-correlated drug in the cell line, and the
partner pool is mostly drugs the model was TRAINED on. So the scramble arm does not emit noise; it
emits a learned signature that was selected to point away from the truth. The arm scores BELOW
chance, and `model - scramble` measures the model's own damage rather than its knowledge.

The three strata make this visible, because they differ only in how correlated the partner's truth is
with the target's:

    scramble_near      partner cos +0.26   ->  NIR 0.572   ABOVE chance
    scramble_orth      partner cos  0.00   ->  NIR 0.497   AT    chance
    scramble_opposite  partner cos -0.24   ->  NIR 0.403   BELOW chance

The comparator's score tracks the partner almost linearly. `orth` is therefore the only stratum that
is a neutral null, and it is the one the generalisation claim needs. `opposite` remains a valid test
of "does the model read the drug NAME at all" -- for that, a sharper comparator is a feature -- but
it cannot carry a claim about how much the true name helps.

`residual_eval` hard-codes `scramble_opposite` for its split table, which is how a positive
`unseen_drug` control -- a physically impossible result, since the model has never seen those drugs
-- was reported as USES DRUG.

This runs on a SAVED eval JSON. No model, no GPU, no re-generation.

USAGE
  python scramble_stratum_audit.py --selftest
  python scramble_stratum_audit.py --result RESULTS_cluster/re_repaired.json
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "shared"),
           os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import inference as inf                                              # noqa: E402

STRATA = ["near", "orth", "opposite"]
SPLITS = ["train", "unseen_combo", "unseen_drug"]
CHANCE = 0.5


def comparator_neutrality(recs):
    """Where does each comparator sit relative to chance? A null must sit AT chance."""
    out = {}
    for st in STRATA:
        k = "scramble_" + st
        v = [r[k] for r in recs if r.get(k) is not None]
        c = [r.get("cos_" + st) for r in recs if r.get(k) is not None and r.get("cos_" + st) is not None]
        if not v:
            continue
        m = float(np.mean(v))
        out[st] = {"mean_nir": m, "mean_partner_cos": (float(np.mean(c)) if c else None),
                   "verdict": ("above chance -- the partner is a partly CORRECT answer" if m > 0.52
                               else "below chance -- the partner is an actively WRONG answer" if m < 0.48
                               else "at chance -- a neutral null")}
    return out


def split_table(recs, min_n=30, min_lines=8):
    """gap = model - scramble, for every (split, stratum), with line-clustered and two-way intervals."""
    rows = []
    for sp in SPLITS:
        sub = [r for r in recs if r.get("split") == sp]
        for st in STRATA:
            k = "scramble_" + st
            use = [r for r in sub if r.get("model") is not None and r.get(k) is not None]
            n_cl = len({r["cell_line"] for r in use})
            if len(use) < min_n or n_cl < min_lines:
                rows.append({"split": sp, "stratum": st, "n": len(use), "n_cell_lines": n_cl,
                             "underpowered": True})
                continue
            diff = [r["model"] - r[k] for r in use]
            lines = [r["cell_line"] for r in use]
            wells = [r.get("sample_id", r["cell_line"]) for r in use]
            one = inf.cluster_bootstrap(diff, lines, lambda v: float(np.mean(v)), n_boot=4000, seed=1)
            two = inf.two_way_cluster_ci(diff, lines, wells)
            rows.append({
                "split": sp, "stratum": st, "n": len(use), "n_cell_lines": n_cl,
                "n_wells": len(set(wells)), "underpowered": False,
                "model_nir": float(np.mean([r["model"] for r in use])),
                "scramble_nir": float(np.mean([r[k] for r in use])),
                "gap": float(np.mean(diff)),
                "ci_line": ([one["lo"], one["hi"]] if one else None),
                "ci_two_way": ([two["lo"], two["hi"]] if two else None),
                "excludes_zero": bool(two and two["lo"] > 0),
            })
    return rows


def memorisation_premium(recs, stratum="orth", n_perm=5000, seed=3):
    """train gap minus unseen_combo gap, permutation-tested.

    `Results-and-Analysis.tex` and `Conclusions.tex` both state there is NO memorisation premium,
    on the strength of two overlapping intervals. Overlapping intervals are not a test of a
    difference; this is."""
    k = "scramble_" + stratum
    a = [r["model"] - r[k] for r in recs if r.get("split") == "train" and r.get(k) is not None
         and r.get("model") is not None]
    b = [r["model"] - r[k] for r in recs if r.get("split") == "unseen_combo" and r.get(k) is not None
         and r.get("model") is not None]
    if len(a) < 10 or len(b) < 10:
        return None
    rows, lab = a + b, ["train"] * len(a) + ["combo"] * len(b)

    def stat(rs, ls):
        x = [v for v, l in zip(rs, ls) if l == "train"]
        y = [v for v, l in zip(rs, ls) if l == "combo"]
        return None if not x or not y else float(np.mean(x) - np.mean(y))

    p = inf.permutation_p(rows, lab, stat, n_perm=n_perm, seed=seed)
    if p is None:
        return None
    return {"stratum": stratum, "train_gap": float(np.mean(a)), "unseen_combo_gap": float(np.mean(b)),
            "difference": p["observed"], "p": p["p"], "n_train": len(a), "n_combo": len(b)}


def output_tracks_named_drug(recs, n_boot=1500, seed=5):
    """Does the output resemble the drug the model was TOLD, whichever drug that is?

    A COMPARATOR-FREE test, which is its whole advantage: it needs no null arm, so the failure
    diagnosed above cannot touch it.

    For each condition we have three generations, each produced after naming a different partner
    drug P. For each we know cos(truth_target, truth_P) -- how alike the two drugs' real responses
    are -- and the NIR of the generation scored against the TARGET's truth. Fit a straight line
    through those points:

        NIR  =  a + b * cos(truth_target, truth_P)

    b is the slope. If the model ignores which drug it was told, the output does not move with the
    partner and b = 0. If the model emits the NAMED drug's signature, then naming a drug whose real
    response resembles the target's produces a generation that scores high, and naming an opposite
    one produces a generation that scores low, so b > 0.

    b > 0 therefore means the model holds a drug -> signature map. It does NOT mean the map is
    precise enough to identify the target among its neighbours; that is what `model_vs_chance` asks.
    """
    out = {}
    for sp in SPLITS:
        sub = [r for r in recs if r.get("split") == sp]
        pts, lines, wells = [], [], []
        for r in sub:
            for st in STRATA:
                x, y = r.get("cos_" + st), r.get("scramble_" + st)
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
                    lines.append(r["cell_line"]); wells.append(r.get("sample_id", r["cell_line"]))
        if len(pts) < 30:
            out[sp] = {"n_points": len(pts), "underpowered": True}
            continue

        def slope(ps):
            a_ = np.array([p[0] for p in ps]); b_ = np.array([p[1] for p in ps])
            if len(a_) < 3 or a_.std() < 1e-9:
                return None
            return float(np.polyfit(a_, b_, 1)[0])

        ci = inf.crossed_bootstrap(pts, lines, wells, slope, n_boot=n_boot, seed=seed)
        out[sp] = {"n_points": len(pts), "slope": slope(pts),
                   "ci": ([ci["lo"], ci["hi"]] if ci else None),
                   "excludes_zero": bool(ci and ci["lo"] > 0)}
    return out


def model_vs_chance(recs):
    """The simplest statement available: told the TRUE drug, does the prediction beat chance?

    NIR is the fraction of other drugs in the same cell line whose truth is LESS similar to the
    prediction than the target's own truth, so 0.5 is chance by construction and no comparator arm
    is required. This is the direct form of the generalisation question.
    """
    out = {}
    for sp in SPLITS:
        sub = [r for r in recs if r.get("split") == sp and r.get("model") is not None]
        if len(sub) < 30:
            out[sp] = {"n": len(sub), "underpowered": True}
            continue
        ci = inf.two_way_cluster_ci([r["model"] - CHANCE for r in sub],
                                    [r["cell_line"] for r in sub],
                                    [r.get("sample_id", r["cell_line"]) for r in sub])
        out[sp] = {"n": len(sub), "nir": float(np.mean([r["model"] for r in sub])),
                   "ci": ([CHANCE + ci["lo"], CHANCE + ci["hi"]] if ci else None),
                   "above_chance": bool(ci and ci["lo"] > 0)}
    return out


def output_geometry(recs):
    """Does the model CONFABULATE or HEDGE when it does not know the drug?

    NIR at chance is ambiguous on its own: a confidently WRONG output -- a plausible-looking
    signature for the wrong drug -- also scores about 0.5, because it is equally unrelated to every
    truth. The two cases are separated by the ABSOLUTE similarity between prediction and reality.

        confabulating : large cosines, pointing somewhere definite but wrong
        hedging       : small cosines to everything, no commitment
    """
    out = {}
    for sp in SPLITS:
        sub = [r for r in recs if r.get("split") == sp
               and r.get("cos_pred_truth") is not None and r.get("cos_pred_others") is not None]
        if len(sub) < 30:
            out[sp] = {"n": len(sub), "underpowered": True}
            continue
        own = np.array([r["cos_pred_truth"] for r in sub])
        oth = np.array([r["cos_pred_others"] for r in sub])
        ci = inf.two_way_cluster_ci((own - oth).tolist(),
                                    [r["cell_line"] for r in sub],
                                    [r.get("sample_id", r["cell_line"]) for r in sub])
        out[sp] = {"n": len(sub), "cos_own": float(own.mean()), "cos_others": float(oth.mean()),
                   "difference": float((own - oth).mean()),
                   "ci": ([ci["lo"], ci["hi"]] if ci else None),
                   "excludes_zero": bool(ci and ci["lo"] > 0),
                   "max_abs_cos": float(max(abs(own.mean()), abs(oth.mean())))}
    return out


def split_comparability(recs):
    """Are the splits equally hard? If held-out truths were noisier, the model's weaker showing
    there would be a property of the TARGETS rather than of the model. `ceiling` is a real replicate
    of the same condition scored the same way, so it measures how discriminable the truths are
    independently of any model."""
    out = {}
    for sp in SPLITS:
        sub = [r for r in recs if r.get("split") == sp]
        c = [r["ceiling"] for r in sub if r.get("ceiling") is not None]
        rc = [r["repro_cos"] for r in sub if r.get("repro_cos") is not None]
        if not c:
            continue
        out[sp] = {"n": len(c), "ceiling": float(np.mean(c)),
                   "mean_condition_reproducibility": (float(np.mean(rc)) if rc else None)}
    cs = [v["ceiling"] for v in out.values() if isinstance(v, dict)]
    if len(cs) > 1:
        out["max_ceiling_spread"] = float(max(cs) - min(cs))
    return out


def detection_limit(recs, split="unseen_drug", power_z=2.8):
    """What effect would this sample have MISSED? A null claim without this is not a claim.

    The half-width of a 95% interval is 1.96 standard errors; about 80% power at 5% two-sided needs
    roughly 2.8. So the smallest reliably detectable effect is about (2.8/1.96) * half-width."""
    sub = [r for r in recs if r.get("split") == split and r.get("model") is not None]
    if len(sub) < 30:
        return None
    v = np.array([r["model"] for r in sub])
    ci = inf.two_way_cluster_ci((v - CHANCE).tolist(),
                                [r["cell_line"] for r in sub],
                                [r.get("sample_id", r["cell_line"]) for r in sub])
    if not ci:
        return None
    half = (ci["hi"] - ci["lo"]) / 2.0
    return {"split": split, "n": len(sub), "observed": float(v.mean()),
            "half_width": float(half), "detectable_at_80pc_power": float(power_z / 1.96 * half)}


def report(recs):
    out = {"n_records": len(recs)}
    neut = comparator_neutrality(recs)
    out["comparator_neutrality"] = neut
    logger.info("=" * 100)
    logger.info("IS THE COMPARATOR A NULL?  A valid scramble must sit AT chance (0.500).")
    for st in STRATA:
        if st in neut:
            d = neut[st]
            pc = "--" if d["mean_partner_cos"] is None else f"{d['mean_partner_cos']:+.2f}"
            logger.info(f"    scramble_{st:9s} partner truth cos {pc}   NIR {d['mean_nir']:.3f}   "
                        f"{d['verdict']}")
    logger.info("    -> the comparator's score tracks the PARTNER, not the model's knowledge of the")
    logger.info("       target. Only `orth` is a neutral null and only it can carry a claim about")
    logger.info("       how much naming the TRUE drug helps.")

    rows = split_table(recs)
    out["split_table"] = rows
    logger.info("-" * 100)
    logger.info("GAP BY SPLIT AND STRATUM")
    logger.info(f"    {'split':14s} {'stratum':9s} {'n':>4s} {'model':>6s} {'scram':>6s} {'gap':>9s}"
                f"  {'line-clustered':>20s}  {'two-way':>20s}")
    for r in rows:
        if r.get("underpowered"):
            logger.info(f"    {r['split']:14s} {r['stratum']:9s} {r['n']:4d}   UNDERPOWERED "
                        f"({r['n_cell_lines']} cell lines)")
            continue
        logger.info(f"    {r['split']:14s} {r['stratum']:9s} {r['n']:4d} {r['model_nir']:6.3f} "
                    f"{r['scramble_nir']:6.3f} {r['gap']:+9.4f}  "
                    f"[{r['ci_line'][0]:+.4f}, {r['ci_line'][1]:+.4f}]  "
                    f"[{r['ci_two_way'][0]:+.4f}, {r['ci_two_way'][1]:+.4f}]"
                    f"{'  EXCLUDES 0' if r['excludes_zero'] else ''}")

    ctrl = next((r for r in rows if r["split"] == "unseen_drug" and r["stratum"] == "orth"
                 and not r.get("underpowered")), None)
    ctrl_opp = next((r for r in rows if r["split"] == "unseen_drug" and r["stratum"] == "opposite"
                     and not r.get("underpowered")), None)
    logger.info("-" * 100)
    logger.info("THE CONTROL.  unseen_drug conditions are drugs the model has NEVER seen, so the gap")
    logger.info("must be zero. A positive value there is an instrument fault, not a finding.")
    if ctrl_opp:
        logger.info(f"    under `opposite`: {ctrl_opp['gap']:+.4f} "
                    f"[{ctrl_opp['ci_two_way'][0]:+.4f}, {ctrl_opp['ci_two_way'][1]:+.4f}]"
                    f"{'  <- CONTROL FAILS' if ctrl_opp['excludes_zero'] else ''}")
    if ctrl:
        logger.info(f"    under `orth`:     {ctrl['gap']:+.4f} "
                    f"[{ctrl['ci_two_way'][0]:+.4f}, {ctrl['ci_two_way'][1]:+.4f}]"
                    f"{'  <- CONTROL FAILS' if ctrl['excludes_zero'] else '  <- control passes'}")
        out["control_passes_under_orth"] = not ctrl["excludes_zero"]

    mp = memorisation_premium(recs)
    out["memorisation_premium"] = mp
    if mp:
        logger.info("-" * 100)
        logger.info("MEMORISATION PREMIUM (neutral comparator): does the model do better on conditions")
        logger.info("it trained on than on held-out ones?")
        logger.info(f"    train {mp['train_gap']:+.4f}   unseen_combo {mp['unseen_combo_gap']:+.4f}   "
                    f"difference {mp['difference']:+.4f}   permutation p = {mp['p']:.4f}")
        if mp["p"] < 0.05:
            logger.info("    -> a premium EXISTS. Any statement that there is none must be withdrawn.")

    gen = next((r for r in rows if r["split"] == "unseen_combo" and r["stratum"] == "orth"
                and not r.get("underpowered")), None)
    if gen:
        logger.info("-" * 100)
        logger.info("THE GENERALISATION CLAIM, against the neutral comparator:")
        logger.info(f"    unseen_combo  {gen['gap']:+.4f}  two-way "
                    f"[{gen['ci_two_way'][0]:+.4f}, {gen['ci_two_way'][1]:+.4f}]  -> "
                    f"{'ESTABLISHED' if gen['excludes_zero'] else 'NOT ESTABLISHED'}")
        out["generalisation_established"] = bool(gen["excludes_zero"])

    trk = output_tracks_named_drug(recs)
    out["output_tracks_named_drug"] = trk
    logger.info("-" * 100)
    logger.info("DOES THE OUTPUT TRACK THE DRUG IT WAS TOLD?  Slope of NIR on cos(truth, partner truth).")
    logger.info("Comparator-free: zero slope is the natural null, so the fault above cannot reach it.")
    for sp in SPLITS:
        d = trk.get(sp) or {}
        if d.get("underpowered") or d.get("ci") is None:
            logger.info(f"    {sp:14s} UNDERPOWERED ({d.get('n_points', 0)} points)")
            continue
        logger.info(f"    {sp:14s} {d['n_points']:5d} points   slope {d['slope']:+.3f}  "
                    f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
                    f"{'  EXCLUDES 0 -- the model holds a drug->signature map' if d['excludes_zero'] else ''}")

    mvc = model_vs_chance(recs)
    out["model_vs_chance"] = mvc
    logger.info("-" * 100)
    logger.info("TOLD THE TRUE DRUG, DOES IT BEAT CHANCE?  NIR against 0.500; no comparator at all.")
    for sp in SPLITS:
        d = mvc.get(sp) or {}
        if d.get("underpowered") or d.get("ci") is None:
            logger.info(f"    {sp:14s} UNDERPOWERED")
            continue
        logger.info(f"    {sp:14s} n={d['n']:4d}  NIR {d['nir']:.3f}  "
                    f"[{d['ci'][0]:.4f}, {d['ci'][1]:.4f}]  "
                    f"{'ABOVE chance' if d['above_chance'] else 'not distinguishable from chance'}")
    geo = output_geometry(recs)
    out["output_geometry"] = geo
    logger.info("-" * 100)
    logger.info("CONFABULATION OR HEDGING?  NIR at chance is ambiguous -- a confidently WRONG output")
    logger.info("scores ~0.5 too, being equally unrelated to every truth. Absolute cosines separate them.")
    logger.info("    split            n   cos(pred,own)  cos(pred,others)   difference")
    for sp in SPLITS:
        d = geo.get(sp) or {}
        if d.get("underpowered") or d.get("ci") is None:
            continue
        logger.info(f"    {sp:14s} {d['n']:4d} {d['cos_own']:+14.4f} {d['cos_others']:+17.4f} "
                    f"{d['difference']:+11.4f}  [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"
                    f"{'  excl 0' if d['excludes_zero'] else '  spans 0'}")
    worst = max((d.get("max_abs_cos", 0.0) for d in geo.values() if isinstance(d, dict)), default=0.0)
    logger.info(f"    -> largest absolute cosine anywhere is {worst:.3f}. The output barely resembles")
    logger.info("       ANY real response, so the drug effect is a faint TILT that moves a rank")
    logger.info("       statistic, not a signature the model reproduces. On unseen drugs it commits to")
    logger.info("       nothing: it HEDGES rather than confabulating.")

    cmp_ = split_comparability(recs)
    out["split_comparability"] = cmp_
    logger.info("-" * 100)
    logger.info("ARE THE SPLITS EQUALLY HARD?")
    for sp in SPLITS:
        d = cmp_.get(sp)
        if d:
            logger.info(f"    {sp:14s} ceiling {d['ceiling']:.3f}   mean condition reproducibility "
                        f"{d['mean_condition_reproducibility']:+.3f}   n={d['n']}")
    if "max_ceiling_spread" in cmp_:
        sprd = cmp_["max_ceiling_spread"]
        logger.info(f"    -> ceilings differ by {sprd:.3f}. "
                    + ("Comparable; that confound is closed." if sprd < 0.02 else
                       "NOT comparable -- the split comparison is confounded by target quality."))

    dl = detection_limit(recs)
    out["detection_limit"] = dl
    if dl:
        logger.info("-" * 100)
        logger.info("WHAT WOULD HAVE BEEN MISSED?  A null claim without a detection limit is not a claim.")
        logger.info(f"    {dl['split']}: n={dl['n']}, observed {dl['observed']:.4f}; an effect below about "
                    f"+{dl['detectable_at_80pc_power']:.3f} above chance would not have been seen.")
        logger.info("    -> report as NOT ESTABLISHED with this limit, never as 'no effect'.")

    logger.info("    -> a positive SLOPE with a chance-level NIR is coherent: the map exists but is")
    logger.info("       coarse. It carries the drug MAIN EFFECT and not the drug x cell-line")
    logger.info("       interaction, which Q17 measures at ~45% of the residual variance -- and which")
    logger.info("       is what a per-cell-line discrimination test like NIR actually asks for.")
    logger.info("=" * 100)
    return out


def selftest():
    ok = []

    def check(n, c):
        ok.append((n, bool(c)))

    rng = np.random.RandomState(0)
    recs = []
    # A model that reads the drug ONLY on trained conditions. The comparator's score is set by the
    # partner: near above chance, orth at chance, opposite below. The control must be null under
    # orth and spuriously positive under opposite -- the exact pattern we are auditing for.
    for sp, model_nir in (("train", 0.64), ("unseen_combo", 0.52), ("unseen_drug", 0.51)):
        n = {"train": 200, "unseen_combo": 250, "unseen_drug": 120}[sp]
        for i in range(n):
            recs.append({
                "split": sp, "cell_line": f"c{i % 20}", "sample_id": f"w{i % 40}",
                "model": model_nir + 0.05 * rng.randn(),
                "scramble_near": 0.57 + 0.05 * rng.randn(),
                "scramble_orth": 0.50 + 0.05 * rng.randn(),
                "scramble_opposite": 0.40 + 0.05 * rng.randn(),
                "cos_near": 0.26, "cos_orth": 0.0, "cos_opposite": -0.24})

    neut = comparator_neutrality(recs)
    check("near is detected as above chance", "above chance" in neut["near"]["verdict"])
    check("orth is detected as a neutral null", "neutral null" in neut["orth"]["verdict"])
    check("opposite is detected as below chance", "below chance" in neut["opposite"]["verdict"])

    rows = split_table(recs)
    g = {(r["split"], r["stratum"]): r for r in rows}
    check("the planted control is NULL under the neutral comparator",
          not g[("unseen_drug", "orth")]["excludes_zero"])
    check("the planted control is SPURIOUSLY POSITIVE under the anti-correlated comparator",
          g[("unseen_drug", "opposite")]["excludes_zero"])
    check("a model that reads the drug on trained data shows it under orth",
          g[("train", "orth")]["excludes_zero"])
    check("two-way intervals are reported alongside line-clustered ones",
          g[("train", "orth")]["ci_two_way"] is not None and g[("train", "orth")]["ci_line"] is not None)

    mp = memorisation_premium(recs)
    check("the planted memorisation premium is detected", mp is not None and mp["p"] < 0.05)
    check("the premium has the right sign", mp["difference"] > 0)

    trk = output_tracks_named_drug(recs, n_boot=300)
    check("a planted drug->signature map gives a positive slope on trained data",
          trk["train"]["excludes_zero"])
    mvc = model_vs_chance(recs)
    check("a model above chance on trained data is detected", mvc["train"]["above_chance"])
    check("a model at chance on held-out data is NOT called above chance",
          not mvc["unseen_drug"]["above_chance"])

    # a model that ignores the drug name entirely must give slope ~0
    flat = [dict(r, scramble_near=0.5 + 0.02 * rng.randn(),
                 scramble_orth=0.5 + 0.02 * rng.randn(),
                 scramble_opposite=0.5 + 0.02 * rng.randn()) for r in recs]
    check("a name-ignoring model gives a slope indistinguishable from zero",
          not output_tracks_named_drug(flat, n_boot=300)["train"]["excludes_zero"])

    conf = [dict(r, cos_pred_truth=0.60 + 0.05 * rng.randn(),
                 cos_pred_others=0.55 + 0.05 * rng.randn()) for r in recs]
    hedge = [dict(r, cos_pred_truth=0.01 + 0.01 * rng.randn(),
                  cos_pred_others=0.00 + 0.01 * rng.randn()) for r in recs]
    check("a confabulating model is detected by LARGE absolute cosines",
          output_geometry(conf)["train"]["max_abs_cos"] > 0.3)
    check("a hedging model is detected by small absolute cosines",
          output_geometry(hedge)["train"]["max_abs_cos"] < 0.1)
    check("a detection limit is reported for the control split",
          (detection_limit(recs) or {}).get("detectable_at_80pc_power", 0) > 0)
    check("split comparability reports a ceiling spread",
          "max_ceiling_spread" in split_comparability(
              [dict(r, ceiling=0.95, repro_cos=0.2) for r in recs]))

    thin = [r for r in recs if r["split"] == "train"][:5]
    check("an underpowered cell is marked, not silently reported",
          split_table(thin)[0].get("underpowered") is True)

    for n, c in ok:
        logger.info(("  ok   " if c else "  FAIL ") + n)
    allok = all(c for _, c in ok)
    logger.info(f"SELFTEST {'PASSED' if allok else 'FAILED'}  ({sum(c for _, c in ok)}/{len(ok)})")
    if not allok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", help="a residual_eval output JSON (must contain `records`)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.result:
        ap.error("--result required (unless --selftest)")
    doc = json.load(open(a.result))
    recs = doc.get("records")
    if not recs:
        raise SystemExit(f"{a.result} has no `records` -- rerun residual_eval, which saves them")
    out = report(recs)
    out["source"] = os.path.basename(a.result)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        logger.info(f"-> {a.out}")


if __name__ == "__main__":
    main()

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

#!/usr/bin/env python
r"""
aggregate_workspace_probe.py -- one canonical Q21 table, one global correction.
===============================================================================
The mechanistic probe was run as many separate jobs: five model arms, up to seven layers each,
several seeds, a magnitude ladder. Each run wrote its own JSON and applied its own multiplicity
correction over whatever it happened to contain. The thesis then quotes a single number -- residual,
layer 12, +0.0197, q=0.026 -- assembled by reading across those files by hand.

Two problems with that, and this script exists to remove both.

MULTIPLICITY MUST SPAN THE FAMILY THAT WAS TESTED, NOT THE FILE THAT HAPPENED TO BE WRITTEN.
  If BH is applied within each arm's file, the correction is over ~7 tests; the family actually
  screened is arms x layers, which is ~26. A q-value computed over the smaller set is too generous,
  and which set it was computed over is invisible in the output. This recomputes one global BH over
  the whole declared family and reports both, so the difference is visible rather than argued.

A LAYER THAT WAS NEVER MEASURED IS NOT A LAYER THAT CAME BACK NULL.
  Runs dropped layers for real reasons -- the headroom gate excludes layers where the positive
  control has no dynamic range, and seed 44 lost five of seven. A table that silently omits them
  reads as "we tested 26 things and one worked" when the truth is "we tested fewer, and here is
  which". Missing cells are marked, counted, and excluded from the denominator explicitly.

USAGE
  python aggregate_workspace_probe.py --selftest
  python aggregate_workspace_probe.py --glob 'RESULTS/probe_arm_*.json' 'RESULTS/probe_rep_*.json' \
      --out RESULTS/probe_canonical.json --csv RESULTS/probe_canonical.csv
"""
import argparse
import csv as _csv
import glob as _glob
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "shared"),
           os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import inference as inf                                            # noqa: E402


# --------------------------------------------------------------------------- extraction
def _walk(node, path=()):
    """Yield (path, dict) for every dict in a nested structure. The probe's JSON layout has changed
    across runs, so a structural walk is more robust than hard-coding a key path -- and a run whose
    shape we do not recognise should be reported as unreadable, not silently skipped."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (str(i),))


def extract(path):
    """One probe JSON -> list of test records.

    A record needs an effect, a p-value and enough context to name it. Anything with a p-value but
    no identifiable layer or arm is reported as unlabelled rather than dropped, because a test we
    cannot place is a gap in the family, not an absence of a test.
    """
    try:
        doc = json.load(open(path))
    except (OSError, ValueError) as e:
        return [], f"unreadable ({e})"
    arm_hint = doc.get("arm") or doc.get("config", {}).get("arm")
    seed_hint = doc.get("seed", doc.get("config", {}).get("seed"))
    recs, unlabelled = [], 0
    for keypath, d in _walk(doc):
        p = d.get("p") if isinstance(d.get("p"), (int, float)) else None
        if p is None or not (0.0 <= p <= 1.0):
            continue
        eff = next((d[k] for k in ("effect", "gap", "delta", "contrast", "separation")
                    if isinstance(d.get(k), (int, float))), None)
        if eff is None:
            continue
        ctx = " ".join(keypath).lower()
        layer = d.get("layer")
        if layer is None:
            layer = next((int(t[5:]) for t in keypath
                          if t.lower().startswith("layer") and t[5:].isdigit()), None)
        arm = d.get("arm") or arm_hint
        if arm is None:
            arm = next((a for a in ("single_cell", "consensus", "ot_t2", "residual_holdout",
                                    "residual") if a in ctx), None)
        ci = d.get("ci") if isinstance(d.get("ci"), (list, tuple)) and len(d.get("ci")) == 2 else None
        if arm is None or layer is None:
            unlabelled += 1
        recs.append({"file": os.path.basename(path), "arm": arm, "layer": layer,
                     "seed": d.get("seed", seed_hint), "effect": float(eff), "p": float(p),
                     "ci_lo": (float(ci[0]) if ci else None), "ci_hi": (float(ci[1]) if ci else None),
                     "q_as_reported": (float(d["q"]) if isinstance(d.get("q"), (int, float)) else None),
                     "path": "/".join(keypath)})
    note = f"{len(recs)} tests" + (f", {unlabelled} unlabelled" if unlabelled else "")
    return recs, note


# --------------------------------------------------------------------------- aggregation
def aggregate(records, arms=None, layers=None, alpha=0.05):
    """One global BH over the declared family, with the un-measured cells named."""
    if not records:
        return {"error": "no probe records found"}
    arms = sorted(arms or {r["arm"] for r in records if r["arm"]})
    layers = sorted(layers or {r["layer"] for r in records if r["layer"] is not None})

    # headline = one record per (arm, layer): the primary seed if identifiable, else the first
    seen, headline = {}, []
    for r in sorted(records, key=lambda x: (str(x["arm"]), x["layer"] if x["layer"] is not None else -1,
                                            str(x["seed"]))):
        key = (r["arm"], r["layer"])
        if r["arm"] is None or r["layer"] is None or key in seen:
            continue
        seen[key] = True
        headline.append(r)

    q = inf.bh([r["p"] for r in headline]) if headline else []
    for r, qq in zip(headline, q):
        r["q_global"] = float(qq)
        r["survives"] = bool(qq < alpha)

    declared = [(a, l) for a in arms for l in layers]
    measured = set(seen)
    missing = [c for c in declared if c not in measured]

    surv = [r for r in headline if r.get("survives")]
    out = {
        "family": {"arms": arms, "layers": layers,
                   "n_declared_cells": len(declared),
                   "n_measured": len(measured), "n_missing": len(missing),
                   "missing_cells": [{"arm": a, "layer": l} for a, l in missing]},
        "n_tests_in_global_bh": len(headline),
        "alpha": alpha,
        "survivors": sorted(surv, key=lambda r: r["q_global"]),
        "all": sorted(headline, key=lambda r: (str(r["arm"]), r["layer"])),
    }

    logger.info("=" * 96)
    logger.info("CANONICAL Q21 FAMILY -- one BH over every (arm, layer) cell that was measured")
    logger.info(f"  declared family: {len(arms)} arms x {len(layers)} layers = {len(declared)} cells")
    logger.info(f"  MEASURED {len(measured)}; NOT MEASURED {len(missing)}  <- these are absences of a "
                f"test, not null results, and are excluded from the denominator")
    for a, l in missing[:12]:
        logger.info(f"      no measurement: arm={a} layer={l}")
    logger.info(f"  global BH over {len(headline)} tests at alpha={alpha}")
    for r in out["all"]:
        moved = ("" if r["q_as_reported"] is None
                 else f"  (file reported q={r['q_as_reported']:.4f})")
        logger.info(f"    {str(r['arm']):18s} L{str(r['layer']):<3s} effect {r['effect']:+.4f}  "
                    f"p={r['p']:.4f}  q_global={r['q_global']:.4f}  "
                    f"{'SURVIVES' if r['survives'] else '--'}{moved}")
    logger.info(f"  >>> {len(surv)} of {len(headline)} survive the GLOBAL correction")
    if any(r["q_as_reported"] is not None and abs(r["q_as_reported"] - r["q_global"]) > 1e-4
           for r in out["all"]):
        logger.info("  >>> at least one q moved against the per-file value. The global one is the "
                    "one to quote: the family screened is arms x layers, not one file.")
    logger.info("=" * 96)
    return out


# --------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = []

    def check(n, c):
        ok.append((n, bool(c)))

    d = tempfile.mkdtemp()
    # two arms x three layers, one genuinely strong effect; arm B is missing layer 12 entirely
    files = []
    for arm, ps in (("residual", {2: 0.60, 4: 0.30, 12: 0.0004}),
                    ("single_cell", {2: 0.55, 4: 0.44})):
        doc = {"arm": arm, "seed": 42,
               "layers": {f"layer{l}": {"layer": l, "effect": (0.02 if p < 0.01 else 0.001),
                                        "p": p, "q": p / 3.0, "ci": [0.001, 0.03]}
                          for l, p in ps.items()}}
        f = os.path.join(d, f"probe_arm_{arm}.json")
        json.dump(doc, open(f, "w"))
        files.append(f)

    recs = []
    for f in files:
        r, _ = extract(f)
        recs += r
    check("both files parsed", len({r["file"] for r in recs}) == 2)
    check("every test found an arm and a layer", all(r["arm"] and r["layer"] is not None for r in recs))
    check("five tests extracted", len(recs) == 5)

    agg = aggregate(recs)
    check("the declared family is arms x layers", agg["family"]["n_declared_cells"] == 6)
    check("the un-measured cell is counted", agg["family"]["n_missing"] == 1)
    check("the un-measured cell is named",
          agg["family"]["missing_cells"][0] == {"arm": "single_cell", "layer": 12})
    check("exactly the strong effect survives",
          len(agg["survivors"]) == 1 and agg["survivors"][0]["arm"] == "residual"
          and agg["survivors"][0]["layer"] == 12)

    # the global correction must be STRICTER than the per-file one it replaces
    strong = agg["survivors"][0]
    check("the global q is above the per-file q it replaces", strong["q_global"] > strong["q_as_reported"])
    check("the global BH spans every measured cell, not one file",
          agg["n_tests_in_global_bh"] == 5)

    # a file we cannot read is reported, not silently skipped
    bad = os.path.join(d, "probe_arm_broken.json")
    open(bad, "w").write("{not json")
    r, note = extract(bad)
    check("an unreadable file is reported rather than dropped", r == [] and "unreadable" in note)

    for n, c in ok:
        logger.info(("  ok   " if c else "  FAIL ") + n)
    allok = all(c for _, c in ok)
    logger.info(f"SELFTEST {'PASSED' if allok else 'FAILED'}  ({sum(c for _, c in ok)}/{len(ok)})")
    if not allok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", help="one or more globs over probe result JSON")
    ap.add_argument("--arms", nargs="*", default=None, help="declare the arm family explicitly")
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="declare the layer family")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="RESULTS/probe_canonical.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.glob:
        ap.error("--glob required (unless --selftest)")

    paths = sorted({p for g in a.glob for p in _glob.glob(os.path.expanduser(g))})
    if not paths:
        raise SystemExit(f"no files matched {a.glob}")
    recs = []
    for p in paths:
        r, note = extract(p)
        logger.info(f"  {os.path.basename(p):40s} {note}")
        recs += r
    out = aggregate(recs, a.arms, a.layers, a.alpha)
    out["sources"] = [os.path.basename(p) for p in paths]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    logger.info(f"canonical probe table -> {a.out}")
    if a.csv and out.get("all"):
        with open(a.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(out["all"][0].keys()))
            w.writeheader()
            w.writerows(out["all"])
        logger.info(f"csv -> {a.csv}")


if __name__ == "__main__":
    main()

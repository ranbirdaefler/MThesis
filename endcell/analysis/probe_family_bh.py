"""Pool the identity test across arms, correct once, and normalise the displacement ladder.

WHY THIS EXISTS. Two quantities the thesis quotes are computed nowhere in the pipeline:

  1. The multiplicity correction. Each probe_arm_*.json stores `q_perm_bh` computed WITHIN its own
     file, over that arm's layers only. The thesis makes a claim about a family of 26 tests across
     ALL arms -- "exactly one survives" -- and that needs BH applied once to the pooled p-values.
     The two are different: the consensus arm at layer 2 reads q = 0.048 within file and q = 0.156
     pooled, and quoting the within-file value would wrongly make it a survivor.

  2. The ladder in units the headline uses. `diff_perm` is in raw nats; the headline is normalised
     by `clean_ab_gap`. Only the top-level block stores the normalised form, and that block is the
     alpha = 0.5 rung, so the normalised value at any other alpha exists nowhere.

The second is what exposes the sensitivity worth knowing: the headline is quoted at HALF the natural
displacement, and applying the same correction to the alpha = 1 slice leaves nothing surviving.

Usage:  python endcell/analysis/probe_family_bh.py
"""
import glob
import io
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "RESULTS_cluster", "probe_family_bh.json")


def benjamini_hochberg(pvals):
    """Step-up BH. Returns q in the same order as the input."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    q = [None] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * m / (rank + 1))
        q[i] = prev
    return q


def collect(alpha=None):
    """One row per (arm, layer). alpha=None takes the stored top-level slice."""
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "RESULTS_cluster", "probe_arm_*.json"))):
        arm = os.path.basename(path)[len("probe_arm_"):-len(".json")]
        d = json.load(io.open(path, encoding="utf-8"))
        for layer, v in d["layers"].items():
            sw = v.get("swap") or {}
            if sw.get("p_perm") is None:
                continue
            gap = sw.get("clean_ab_gap")
            rec = {"arm": arm, "layer": int(layer), "clean_ab_gap": gap}
            if alpha is None:
                rec.update(alpha="stored (=0.5)", p=float(sw["p_perm"]),
                           diff=sw.get("diff_perm"), norm=sw.get("diff_perm_norm"),
                           ci_norm=sw.get("ci_perm_norm"), q_infile=sw.get("q_perm_bh"))
            else:
                lad = {r["alpha"]: r for r in sw.get("ladder", [])}
                if alpha not in lad or lad[alpha].get("p_perm") is None:
                    continue
                r = lad[alpha]
                ci = r.get("ci_perm") or [None, None]
                rec.update(alpha=alpha, p=float(r["p_perm"]), diff=r.get("diff_perm"),
                           norm=(r["diff_perm"] / gap) if gap else None,
                           ci_norm=([ci[0] / gap, ci[1] / gap] if gap and ci[0] is not None else None),
                           q_infile=None)
            rows.append(rec)
    if rows:
        for rec, q in zip(rows, benjamini_hochberg([r["p"] for r in rows])):
            rec["q_pooled"] = q
    return rows


def main():
    out = {"_why": "the pipeline stores BH within each arm's file; the thesis claims a family "
                   "across arms, and the ladder is stored in raw nats while the headline is "
                   "normalised by clean_ab_gap",
           "slices": {}}

    for alpha, key in ((None, "stored_alpha_0.5"), (1.0, "alpha_1.0"), (2.0, "alpha_2.0")):
        rows = collect(alpha)
        if not rows:
            continue
        surv = [r for r in rows if r["q_pooled"] < 0.05]
        out["slices"][key] = {"n_tests": len(rows), "survivors_q_lt_0.05": surv, "tests": rows}
        print("%-18s family=%d  survivors: %s" % (
            key, len(rows),
            ", ".join("%s@%d q=%.4f" % (r["arm"], r["layer"], r["q_pooled"]) for r in surv) or "NONE"))

    hl = next((r for r in out["slices"]["stored_alpha_0.5"]["tests"]
               if r["arm"] == "residual" and r["layer"] == 12), None)
    a1 = next((r for r in out["slices"].get("alpha_1.0", {}).get("tests", [])
               if r["arm"] == "residual" and r["layer"] == 12), None)
    out["headline_residual_layer12"] = {"stored_alpha_0.5": hl, "alpha_1.0": a1,
                                        "note": "the stored top-level block IS the alpha=0.5 rung"}
    if hl and a1:
        print("\nresidual@12   alpha 0.5: %+.4f %s p=%.4f q=%.4f"
              % (hl["norm"], [round(x, 4) for x in hl["ci_norm"]], hl["p"], hl["q_pooled"]))
        print("              alpha 1.0: %+.4f %s p=%.4f q=%.4f"
              % (a1["norm"], [round(x, 4) for x in a1["ci_norm"]], a1["p"], a1["q_pooled"]))

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()

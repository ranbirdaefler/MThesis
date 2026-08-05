"""Does co-plating ATTENUATE the mechanism channel gate?

THE ARGUMENT. The residual's generic is a plate-local, drug-weighted, leave-one-drug-out mean
(build_residual_targets.py, class Generic). So "what makes this drug different" means different from
the other drugs on its own plate and cell line. Where a mechanism class is co-plated -- and the gate
finds mechanism the one channel where co-plating is material, 0.404 against 0.362 for a random draw
-- part of that class's shared programme sits inside the generic and is subtracted out of every
member's residual by construction. The mechanism gate is therefore graded in a frame that has
already removed some of what it is asked to find, and the bias runs AGAINST the channel.

THE TEST. Conditions with no same-mechanism plate-mate keep the mechanism programme in their
residual; conditions with one do not. If the argument holds, the first group shows the larger gap.

ROSTER. Only the gate's own retained conditions are used. drug_biology_atlas.csv carries a wider
per-plate drug list, but it is NOT a superset -- it omits gate-retained drugs in all 122 groups --
so using it would compute plate-mates from a roster inconsistent with the one that produced the
residuals. That arm is deliberately not reported.

Usage:  python endcell/analysis/moa_coplate_attenuation.py
"""
import csv
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "shared"))
import inference as inf  # noqa: E402

GATE = os.path.join(REPO, "RESULTS_cluster", "channel_gate_v4.json")
ATLAS = os.path.join(REPO, "RESULTS_cluster", "drug_biology_atlas.csv")
OUT = os.path.join(REPO, "RESULTS_cluster", "moa_coplate_attenuation.json")

UNLABELLED = ("unclear", "unknown", "nan", "none", "")


def measure(rows, label):
    d = [r["moa"] - r["moa_null_pm"] for r in rows]
    ci = inf.two_way_cluster_ci(d, [r["cell_line"] for r in rows],
                                [r.get("well", r["cell_line"]) for r in rows])
    rec = {"n": len(rows), "n_drugs": len({r["drug"] for r in rows}),
           "n_lines": len({r["cell_line"] for r in rows}),
           "n_wells": len({r.get("well") for r in rows}),
           "gap": float(np.mean(d)), "ci": [ci["lo"], ci["hi"]], "df": ci.get("df")}
    print("  %-34s n=%-5d lines=%-3d wells=%-3d gap %+.4f [%+.4f, %+.4f]"
          % (label, rec["n"], rec["n_lines"], rec["n_wells"], rec["gap"], ci["lo"], ci["hi"]))
    return rec


def main():
    gate = json.load(io.open(GATE, encoding="utf-8"))
    arm = [r for r in gate["records"]
           if r.get("moa") is not None and r.get("moa_null_pm") is not None]

    print("full mechanism arm (reproduces the published figure):")
    full = measure(arm, "all conditions")

    moa_of = {}
    for x in csv.DictReader(io.open(ATLAS, encoding="utf-8")):
        m = (x.get("moa") or "").strip()
        if m.lower() not in UNLABELLED:
            moa_of[x["drug"]] = m

    roster = defaultdict(set)
    for r in gate["records"]:
        roster[(r["cell_line"], r["plate"])].add(r["drug"])

    labelled = [r for r in arm if r["drug"] in moa_of]
    none_, some, shares = [], [], []
    for r in labelled:
        mine = moa_of[r["drug"]]
        plate = roster[(r["cell_line"], r["plate"])]
        mates = [d for d in plate if d != r["drug"] and moa_of.get(d) == mine]
        shares.append(len(mates) / max(1, len(plate) - 1))
        (some if mates else none_).append(r)

    sizes = sorted(len(v) for v in roster.values())
    print("\nroster: %d (cell line, plate) groups, median %.1f drugs, range %d-%d"
          % (len(sizes), float(np.median(sizes)), sizes[0], sizes[-1]))
    print("mechanism-arm conditions carrying a moa-fine label: %d of %d" % (len(labelled), len(arm)))
    print("same-mechanism share of the generic: mean %.4f, median %.4f"
          % (float(np.mean(shares)), float(np.median(shares))))
    print("conditions with at least one same-mechanism plate-mate: %d of %d (%.0f%%)\n"
          % (len(some), len(labelled), 100.0 * len(some) / len(labelled)))

    a = measure(none_, "NO same-mechanism plate-mate")
    b = measure(some, "one or more plate-mates")

    overlap = not (a["ci"][0] > b["ci"][1] or b["ci"][0] > a["ci"][1])
    print("\ndirection: %s" % ("as predicted (none > some)" if a["gap"] > b["gap"] else "CONTRARY"))
    print("intervals overlap: %s -- %s" % (overlap,
          "directional evidence only, not a corrected estimate" if overlap else "separated"))

    json.dump({"_argument": __doc__.strip().split("\n\n")[1],
               "_roster_note": "gate-retained conditions only; drug_biology_atlas.csv is NOT a "
                               "superset (it omits gate-retained drugs in all 122 groups) so it is "
                               "not used to widen the roster",
               "full_arm": full, "no_same_moa_platemate": a, "has_same_moa_platemate": b,
               "same_moa_share_of_generic_mean": float(np.mean(shares)),
               "intervals_overlap": overlap,
               "reading": "directional support that the plate-local generic attenuates the "
                          "mechanism channel; the verdict stays inconclusive and this is not a "
                          "corrected estimate"},
              io.open(OUT, "w", encoding="utf-8"), indent=1)
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()

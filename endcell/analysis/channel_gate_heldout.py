"""Restrict the channel gate's TARGETS to held-out drugs.

WHY THIS EXISTS. The partner-pool repair restricted where the gate's PREDICTORS come from. It did
not change which conditions are SCORED: channel_gate.py iterates every retained condition, so the
gate's targets span all 4,131 of them. The question the section asks is about unseen drugs, and an
external review was right that the measured population is not that.

This joins the gate's per-record arms to the canonical evaluation's split labels and rescores each
channel on held-out target drugs only. It needs no cluster access.

The result is a limit on what the section can claim, not a correction to its numbers: the gate
measures CHANNEL AVAILABILITY over the atlas, and only a handful of its drugs are held out.

Usage:  python endcell/analysis/channel_gate_heldout.py
"""
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "shared"))
import inference as inf  # noqa: E402

GATE = os.path.join(REPO, "RESULTS_cluster", "channel_gate_v4.json")
EVAL = os.path.join(REPO, "RESULTS_cluster", "re_v3.json")
OUT = os.path.join(REPO, "RESULTS_cluster", "channel_gate_heldout_targets.json")

N_BOOT = 2000
SEED = 42


def main():
    gate = json.load(io.open(GATE, encoding="utf-8"))
    ev = json.load(io.open(EVAL, encoding="utf-8"))

    # A drug is held out only if it never appears in the training split.
    splits = {}
    for r in ev["records"]:
        splits.setdefault(r["drug"], set()).add(r.get("split"))
    heldout = {d for d, sp in splits.items() if sp and "train" not in sp}

    out = {"_why": "the gate's targets span every retained condition; this restricts them to drugs "
                   "held out of training, which is the population the unseen-drug question is about",
           "n_drugs_in_eval": len(splits), "n_drugs_heldout": len(heldout),
           "margin": gate["config"]["min_margin"], "channels": {}}

    print("evaluation carries %d drugs, %d of them held out of training\n"
          % (len(splits), len(heldout)))

    for ch in ("target", "moa", "chem"):
        arm, null = ch, ch + "_null_pm"
        use = [r for r in gate["records"]
               if r.get(arm) is not None and r.get(null) is not None]
        sub = [r for r in use if r["drug"] in heldout]
        row = {"all_targets": None, "heldout_targets": None}

        for label, rows in (("all_targets", use), ("heldout_targets", sub)):
            if len(rows) < 10:
                print("  %-7s %-18s n=%d -- too few to score" % (ch, label, len(rows)))
                continue
            d = [r[arm] - r[null] for r in rows]
            ci = inf.cluster_bootstrap(d, [r["cell_line"] for r in rows],
                                       lambda x: float(np.mean(x)), n_boot=N_BOOT, seed=SEED)
            row[label] = {"n": len(rows), "n_drugs": len({r["drug"] for r in rows}),
                          "gap": float(np.mean(d)), "ci": [ci["lo"], ci["hi"]],
                          "n_clusters": ci["n_clusters"],
                          "spans_zero": bool(ci["lo"] <= 0.0 <= ci["hi"])}
            print("  %-7s %-18s n=%-5d drugs=%-3d gap %+.4f [%+.4f, %+.4f]%s"
                  % (ch, label, len(rows), row[label]["n_drugs"], row[label]["gap"],
                     ci["lo"], ci["hi"], "  SPANS ZERO" if row[label]["spans_zero"] else ""))
        out["channels"][ch] = row

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()

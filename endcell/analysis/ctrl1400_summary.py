"""Are the two control arms flat at the MATCHED 1400-token budget?

The chapter's first non-null drug effect is the residual arm's gap against a neutral comparator.
Both controls were previously scored at 600 tokens against the residual arm's 1400, so the
three-column comparison was not budget-matched and the thesis had to say so. These are the
re-runs at 1400.

The comparison is model - scramble_orth (the neutral comparator), clustered two-way on cell line
and treatment well, exactly as the residual arm's gap is computed.
"""
import io
import json
import os
import sys

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")
sys.path.insert(0, "shared")
import numpy as np
import inference as inf

REF = json.load(io.open("RESULTS_cluster/re_v3.json", encoding="utf-8"))


def gap(recs, arm="scramble_orth"):
    u = [r for r in recs if r.get(arm) is not None and r.get("model") is not None]
    if len(u) < 10:
        return None
    d = [r["model"] - r[arm] for r in u]
    ci = inf.two_way_cluster_ci(d, [r["cell_line"] for r in u],
                                [r.get("sample_id", r["cell_line"]) for r in u])
    return {"n": len(u), "model": float(np.mean([r["model"] for r in u])),
            "comparator": float(np.mean([r[arm] for r in u])),
            "gap": float(np.mean(d)), "ci": [ci["lo"], ci["hi"]], "df": ci.get("df"),
            "spans_zero": bool(ci["lo"] <= 0.0 <= ci["hi"])}


print("model - scramble_orth, two-way clustered on cell line and well")
print("=" * 92)
rows = {}
for label, path, budget in (
        ("residual (the effect)", "RESULTS_cluster/re_v3.json", 1400),
        ("single-cell control", "RESULTS_cluster/re_singlecell_1400.json", 1400),
        ("optimal-transport control", "RESULTS_cluster/re_ot_1400.json", 1400),
        ("single-cell control", "RESULTS_cluster/re_singlecell_model.json", 600),
        ("optimal-transport control", "RESULTS_cluster/re_ot_model.json", 600)):
    try:
        d = json.load(io.open(path, encoding="utf-8"))
    except FileNotFoundError:
        print("  %-26s %4d  MISSING" % (label, budget))
        continue
    g = gap(d.get("records") or [])
    if g is None:
        print("  %-26s %4d  no scorable arm" % (label, budget))
        continue
    rows[(label, budget)] = g
    print("  %-26s %4d  n=%-4d model %.4f  comparator %.4f  gap %+.4f [%+.4f, %+.4f]  %s"
          % (label, budget, g["n"], g["model"], g["comparator"], g["gap"],
             g["ci"][0], g["ci"][1], "spans zero" if g["spans_zero"] else "EXCLUDES ZERO"))

print()
print("VERDICT")
print("-" * 92)
flat = []
for (label, budget), g in rows.items():
    if budget == 1400 and "control" in label:
        flat.append((label, g["spans_zero"], g))
if flat and all(s for _, s, _ in flat):
    print("  Both controls are FLAT at the matched budget: every interval spans zero.")
    print("  The three-column comparison IS budget-matched and the residual effect is not an")
    print("  artifact of generation length. The withdrawn sentence can be restored, with numbers.")
else:
    for label, spans, g in flat:
        if not spans:
            print("  %s does NOT span zero at 1400: %+.4f [%+.4f, %+.4f]"
                  % (label, g["gap"], g["ci"][0], g["ci"][1]))
    print("  The controls are not flat at the matched budget. The withdrawal stands and the")
    print("  chapter must say the control arms also move when given the same budget.")

out = {"_why": "re-run of both control arms at the residual arm's 1400-token budget, closing the "
               "token-budget confound the chapter previously had to leave open",
       "comparison": "model - scramble_orth, two-way clustered on cell line and well",
       "arms": {"%s@%d" % (k[0], k[1]): v for k, v in rows.items()}}
json.dump(out, io.open("RESULTS_cluster/ctrl1400_summary.json", "w", encoding="utf-8"), indent=1)
print("\n-> RESULTS_cluster/ctrl1400_summary.json")

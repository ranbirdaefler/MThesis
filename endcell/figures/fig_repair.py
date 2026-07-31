"""fig-repair: the gap grows with swap dissimilarity for the residual model, and stays flat for a
control scored through the identical pipeline.

The evidence here is the GRADIENT, not any single value. A metric artifact would not order itself
by how dissimilar the swapped-in drug's response is; a table makes the reader compute that ordering
for themselves.

MATCHED BUDGET IS THE PRIMARY CLAIM. The residual model's headline figures come from a run at 1400
generation tokens while both control arms ran at 600, and the token budget alone doubles the effect
(+0.0716 -> +0.1429 on the opposite stratum). Plotting the 1400 treatment against 600 controls would
overlay an arm at 2.3x its controls' budget and call the difference drug use. So the primary series
are all at 600, where the ordering survives, and the 1400 run is drawn as a separate, explicitly
labelled overlay.

The x-axis is the MEASURED mean swap-partner residual cosine (+0.406 / -0.002 / -0.329), not
categorical labels -- all four runs share byte-identical values, so the x-positions are exact.

Usage:  python endcell/figures/fig_repair.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

STRATA = ("near", "orth", "opposite")

RUNS = [
    ("re_residual_model.json",    "residual model",            style.OKABE["blue"],      "o", "-",  600),
    ("re_ot_model.json",          "optimal-transport control", style.OKABE["green"],     "D", "-",  600),
    ("re_singlecell_model.json",  "single-cell control",       style.OKABE["vermilion"], "s", "-",  600),
]
OVERLAY = ("re_residual_maxtok.json", "residual model, 1400 tokens", style.OKABE["blue"], "o", ":", 1400)


def series(fname):
    d = style.load(fname)
    s = d["means"]["strata"]
    x = [s[k]["mean_cos"] for k in STRATA]
    y = [s[k]["gap"] for k in STRATA]
    lo = [s[k]["ci"][0] for k in STRATA]
    hi = [s[k]["ci"][1] for k in STRATA]
    n = s[STRATA[0]]["n"]
    tok = d["config"]["max_new_tokens"]
    return x, y, lo, hi, n, tok


def main():
    style.apply()
    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH_IN, 3.15))
    drawn = {"strata": STRATA, "primary_budget_tokens": 600, "runs": {}}

    ax.axhline(0.0, color="black", lw=0.9, zorder=1)
    ax.annotate("no drug use", xy=(1.0, 0.0), xycoords=("axes fraction", "data"),
                xytext=(3, 3), textcoords="offset points", fontsize=8, ha="right")

    for fname, label, colour, marker, ls, want_tok in RUNS + [OVERLAY]:
        x, y, lo, hi, n, tok = series(fname)
        assert tok == want_tok, "%s ran at %s tokens, expected %s" % (fname, tok, want_tok)
        drawn["runs"][label] = {"file": fname, "max_new_tokens": tok, "n": n,
                                "mean_cos": x, "gap": y, "ci_lo": lo, "ci_hi": hi}
        for xi, l, h in zip(x, lo, hi):
            ax.plot([xi, xi], [l, h], color=colour, lw=1.0, alpha=0.75,
                    solid_capstyle="butt", zorder=2)
        ax.plot(x, y, marker=marker, ms=5, color=colour, ls=ls, label=label, zorder=3,
                mfc="none" if ls == ":" else colour)

    # stratum names sit at the FOOT of the axis: at the top they collide with the legend
    for xi, name in zip(series(RUNS[0][0])[0], STRATA):
        ax.annotate(name, xy=(xi, 0.035), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8.5, color="black")

    ax.set_xlabel("mean residual cosine between the real drug and the swapped-in drug"
                  "\n$\\longleftarrow$ more similar          more dissimilar $\\longrightarrow$")
    ax.set_ylabel("$\\Delta_{\\mathrm{scr}}$  =  model $-$ scramble")
    ax.invert_xaxis()
    ax.set_ylim(-0.09, 0.21)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.annotate("residual frame. Intervals are 95% bootstrap, clustered over cell lines; "
                "$n=250$ conditions per point.",
                xy=(0.5, -0.32), xycoords="axes fraction", ha="center", fontsize=7.5,
                color="black")

    style.save(fig, "fig-repair", drawn)
    for label, r in drawn["runs"].items():
        print("  %-28s tok=%4d  " % (label, r["max_new_tokens"])
              + "  ".join("%s %+.4f [%+.4f,%+.4f]" % (s, g, l, h)
                          for s, g, l, h in zip(STRATA, r["gap"], r["ci_lo"], r["ci_hi"])))


if __name__ == "__main__":
    main()

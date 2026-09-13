"""fig-repair (v2 rebuild): the model-minus-scramble gap against MEASURED swap dissimilarity.

The evidence here is the GRADIENT, not any single value. A metric artefact would not order itself
by how dissimilar the swapped-in drug's true response is; a table makes the reader compute that
ordering for themselves.

MATCHED BUDGET IS THE PRIMARY CLAIM. The residual model's headline figures come from a run at 1400
generation tokens while both control arms ran at 600, and the token budget alone roughly doubles
the effect (+0.0716 -> +0.1429 on the opposite stratum). Plotting the 1400 treatment against 600
controls would overlay an arm at 2.3x its controls' budget and call the difference drug use. So the
three primary series are all at 600, where the ordering survives, and the 1400 run is drawn as a
separate, explicitly labelled dotted overlay.

WHAT CHANGED FROM v1 (endcell/figures/fig_repair.py), and why:

  (1) The in-figure subtitle ("residual frame. Intervals are 95% bootstrap, clustered over cell
      lines; n = 250 conditions per point.") is GONE. The LaTeX caption below the figure carries
      the same sentence almost verbatim, so the page set one sentence twice, 15mm apart, in two
      different typefaces. The frame moved onto the y-axis label, where it belongs and where
      style_v2's convention already puts it; everything else is the caption's job.

  (2) The x-axis now runs LEFT TO RIGHT IN INCREASING DISSIMILARITY. v1 plotted the raw mean swap
      cosine and then called ax.invert_xaxis(), which produces an axis whose numbers decrease
      rightwards; v1 patched over that with a two-headed "<- more similar / more dissimilar ->"
      gloss under the axis label. Plotting 1 - cos instead makes the axis a genuine dissimilarity
      that increases rightwards, so the gloss is unnecessary and is deleted. 1 - cos is a
      monotone, information-preserving transform of the number v1 plotted; both are recorded in
      the sibling JSON. It also lands the orth stratum at 1.0016, i.e. visibly orthogonal.

  (3) v1 carried THREE horizontal reference systems for one variable: the numeric axis, a floating
      row of stratum names annotated at 3.5% of the axes height, and the arrow gloss. The stratum
      names are now the x tick labels, on the axis, at their exact measured positions, with the
      measured value on a second line. One variable, one axis.

  (4) Only three strata are measured. The connecting lines are kept because the gradient IS the
      claim and three disconnected triples do not read as a gradient, but the caption now states
      that they are guides between three measured strata and not a sampled continuum.

  (5) "no drug use" gets a short leader down from the zero line and sits clear of the
      optimal-transport control's green diamond. In v1 it was jammed against the top right of the
      zero line with 3pt of offset and touched the diamond's interval.

  (6) All four series share byte-identical x positions, so v1's four intervals were drawn on top of
      one another at each stratum -- at `near` they are a single vertical smear of four colours.
      The series are nudged sideways by +-0.008 / +-0.024 in x so the intervals separate. The nudge
      is recorded in the JSON and stated in the caption; the underlying positions are unchanged.

Usage:  python thesis_v2/figs/repair.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

STRATA = ("near", "orth", "opposite")

# (file, label, colour, marker, linestyle, expected token budget, x-nudge)
# Nudge order puts the two accent series (the same model at two budgets) adjacent.
RUNS = [
    ("re_residual_maxtok.json", "residual model, 1400 tokens",
     style.ACCENT, "o", ":", 1400, -0.024),
    ("re_residual_model.json", "residual model, 600 tokens",
     style.ACCENT, "o", "-", 600, -0.008),
    ("re_ot_model.json", "optimal-transport control",
     style.QUIETINK, "D", "-", 600, +0.008),
    ("re_singlecell_model.json", "single-cell control",
     style.INK, "s", "-", 600, +0.024),
]


def series(fname):
    """Pull the three measured strata out of one run. Nothing is typed in by hand."""
    d = style.load(fname)
    s = d["means"]["strata"]
    cos = [s[k]["mean_cos"] for k in STRATA]
    return {
        "file": fname,
        "max_new_tokens": d["config"]["max_new_tokens"],
        "n_conditions_config": d["config"]["n_conditions"],
        "mean_cos": cos,
        # CHANGE 2: the plotted x. A monotone transform of mean_cos, so the ordering and the
        # spacing of the three measured strata are preserved exactly.
        "dissimilarity_1_minus_cos": [1.0 - c for c in cos],
        "gap": [s[k]["gap"] for k in STRATA],
        "ci_lo": [s[k]["ci"][0] for k in STRATA],
        "ci_hi": [s[k]["ci"][1] for k in STRATA],
        "n": [s[k]["n"] for k in STRATA],
    }


def signed(v, _pos=None):
    if abs(v) < 1e-12:
        return "0"
    return ("%+.2f" % v).replace("-", "−")


def main():
    style.apply()
    print("style_v2 face: %s" % style.font_report())

    data = [series(f) for f, *_ in RUNS]
    for d, (fname, _, _, _, _, want_tok, _) in zip(data, RUNS):
        assert d["max_new_tokens"] == want_tok, \
            "%s ran at %s tokens, expected %s" % (fname, d["max_new_tokens"], want_tok)

    # The claim in the caption that the positions are exact rests on this being true.
    ref = data[0]["mean_cos"]
    for d in data[1:]:
        assert d["mean_cos"] == ref, "%s has different swap cosines: %r" % (d["file"], d["mean_cos"])
    xs = [1.0 - c for c in ref]

    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH_IN, 3.40), layout="constrained")

    # zero: the value the gap takes if the model is not using the drug in its prompt at all.
    ax.axhline(0.0, color=style.INK, lw=0.9, zorder=1)

    for d, (fname, label, colour, marker, ls, _, dx) in zip(data, RUNS):
        x = [xi + dx for xi in xs]
        open_marker = (ls == ":")
        for xi, lo, hi in zip(x, d["ci_lo"], d["ci_hi"]):
            style.whisker(ax, xi, None, lo, hi, orientation="v",
                          color=colour, lw=1.0, alpha=0.75, zorder=2)
        ax.plot(x, d["gap"], marker=marker, ms=4.5, mew=1.1, color=colour, ls=ls, lw=1.1,
                label=label, zorder=3, mfc="white" if open_marker else colour,
                dash_capstyle="butt")

    # CHANGE 3: the stratum names ARE the tick labels, at their exact measured positions, with the
    # measured value under them. v1 floated them inside the axes as a second categorical axis.
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.set_xticklabels(["%s\n%.2f" % (name, xi) for name, xi in zip(STRATA, xs)])
    ax.set_xlim(0.50, 1.53)
    ax.set_xlabel("swap dissimilarity, $1-\\cos$ (real vs. swapped-in drug)")

    ax.set_ylim(-0.072, 0.212)
    ax.yaxis.set_major_locator(FixedLocator([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20]))
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.set_ylabel("$\\Delta_{\\mathrm{scr}}$  =  model $-$ scramble\n(NIR, residual frame)")

    # CHANGE 5: leader down from the zero line, clear of the green diamond at (1.337, +0.026).
    ax.annotate("no drug use", xy=(1.396, 0.0), xytext=(1.378, -0.030),
                ha="left", va="center", fontsize=9, color=style.INK,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=style.QUIETINK,
                                shrinkA=3, shrinkB=0))

    ax.legend(loc="upper left", fontsize=9, borderaxespad=0.3, handlelength=2.4,
              labelspacing=0.35, borderpad=0.0)

    drawn = {
        "figure": "fig-repair",
        "quantity": "Delta_scr = NIR(model) - NIR(stratum-matched scramble), residual frame",
        "x_variable": "1 - mean_cos, where mean_cos is the measured mean residual cosine between "
                      "the real drug and the swapped-in drug within the cell line",
        "strata": list(STRATA),
        "x_positions_1_minus_cos": xs,
        "x_positions_mean_cos": ref,
        "x_positions_identical_across_runs": True,
        "x_nudge_per_series_for_legibility": {lbl: dx for _, lbl, _, _, _, _, dx in RUNS},
        "interval": "95% bootstrap, clustered over cell lines",
        "primary_budget_tokens": 600,
        "runs": {},
        "points": [],
    }
    for d, (_, label, _, _, _, _, dx) in zip(data, RUNS):
        drawn["runs"][label] = d
        for i, s in enumerate(STRATA):
            drawn["points"].append({
                "series": label, "stratum": s, "max_new_tokens": d["max_new_tokens"],
                "x_mean_cos": d["mean_cos"][i], "x_plotted_1_minus_cos": xs[i],
                "x_plotted_with_nudge": xs[i] + dx,
                "gap": d["gap"][i], "ci_lo": d["ci_lo"][i], "ci_hi": d["ci_hi"][i],
                "n": d["n"][i], "source": d["file"],
            })

    style.save(fig, "fig-repair", drawn)

    print("\n  %-28s tok   %s" % ("series", "   ".join("%-30s" % s for s in STRATA)))
    for label, r in drawn["runs"].items():
        print("  %-28s %4d  " % (label, r["max_new_tokens"])
              + "   ".join("%+.4f [%+.4f,%+.4f] n=%d" % (g, lo, hi, n)
                           for g, lo, hi, n in zip(r["gap"], r["ci_lo"], r["ci_hi"], r["n"])))
    print("\n  x (1-cos):  " + "  ".join("%s %.4f" % (s, x) for s, x in zip(STRATA, xs)))
    return fig


if __name__ == "__main__":
    main()

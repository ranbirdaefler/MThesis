"""fig-plateleak: the batch shortcut inflates the REFERENCE as well as the baseline.

WHAT THIS FIGURE IS FOR
===========================================================================================
Table \\ref{tab:plateleak} already prints these numbers. What a table cannot do is show that
the four arms move TOGETHER when the comparison set is forced within plate: a drug-agnostic
mean baseline collapses (0.399 -> 0.161) while the within-well split-half precision reference,
the yardstick every other expression-frame number is read against, ALSO falls (0.625 -> 0.575)
by about as much as control-copy does (0.551 -> 0.510). That is the whole argument for treating
plate leakage as a scope limit rather than an offset one could subtract from a baseline: the
shortcut is in the yardstick too, so there is no side of the comparison to apply a correction to.

A slope chart is the minimal drawing for that argument -- two positions, four connected pairs,
every value direct-labelled -- and it carries no mark that is not one of the eight numbers.

WHAT IS DELIBERATELY NOT DRAWN
  * The nir_rank frame. leak_*.json carries a *_nir_rank value for every arm; those are a
    DIFFERENT frame and putting them on this axis would invite the subtraction the thesis spends
    four pages forbidding. Not plotted, and not listed as plottable in the sibling JSON.
  * Any residual-frame number. Same reason, stated in the caption.
  * Intervals. NEITHER source file carries any interval, per-arm or otherwise: agg entries are
    [value, n] pairs only. This document's single CI convention is a 95% bootstrap clustered over
    cell lines, and none exists here. Bare markers; the caption says why.

THE TWO COLUMNS ARE NOT THE SAME CONDITIONS. Holding the plate constant costs every condition
with no same-plate comparison set, so the within-plate column is a SUBSET (n=998, 46 drugs,
41 cell lines) of the cross-plate column (n=1163, 50 drugs, 43 cell lines). The connecting
segments are therefore a comparison of two scorings of overlapping-but-unequal populations, not
a rescore of one set of conditions -- said in the caption, because a slope chart's line is the
one mark that could be misread as "the same thing moved".

Sources: RESULTS_cluster/leak_crossplate.json (same_plate_only=false) and
         RESULTS_cluster/leak_sameplate.json (same_plate_only=true),
         both at tiers.tier2_unseen_drugs.agg, *_nir_expr only.

Usage:  python thesis_v2/figs/plateleak.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# The values printed in Table \ref{tab:plateleak} (Sections/05-Limitations.tex, lines 148-151)
# and repeated in the prose at 05-Limitations.tex:129 and 04a-ruler.tex:461-462. The figure
# refuses to save if the JSONs do not round to them.
TABLE = {
    "mean":    (0.399, 0.161),
    "control": (0.551, 0.510),
    "ceiling": (0.625, 0.575),   # "precision reference (within-well split-half)" in the table
}
# linear is not in the table; it is read from the same agg block and is reported in the JSON.

# Draw order is back-to-front: the reference first, the mean baseline last (it is the mark the
# sentence is about). Labels are the document's names for these arms.
SERIES = [
    ("ceiling", "within-well split-half reference", style.OKABE["grey"], "_", (0, (4, 2.5))),
    ("control", "control-copy",                     style.OKABE["vermilion"], "s", "-"),
    ("linear",  "linear map",                       style.OKABE["green"], "D", "-"),
    ("mean",    "drug-agnostic mean baseline",      style.INK, ".", "-"),
]


def main():
    family = style.apply()
    print("font in force: %s" % style.font_report())

    cross = style.load("leak_crossplate.json")
    same = style.load("leak_sameplate.json")

    assert cross["config"]["same_plate_only"] is False
    assert same["config"]["same_plate_only"] is True
    assert cross["config"]["tier"] == same["config"]["tier"] == "tier2_unseen_drugs"

    tc = cross["tiers"]["tier2_unseen_drugs"]
    ts = same["tiers"]["tier2_unseen_drugs"]

    def agg(t, arm):
        v, n = t["agg"]["%s_nir_expr" % arm]
        return float(v), int(n)

    # population of each column, counted from the per-condition rows rather than trusted from a
    # summary field: leak_*.json's "n_drugs" is the ROW count (1163 / 998), not the drug count.
    def pop(t):
        r = t["rows"]
        return dict(n_conditions=len(r),
                    n_drugs=len(set(x["drug"] for x in r)),
                    n_cell_lines=len(set(x["cell_line"] for x in r)))

    pc, ps = pop(tc), pop(ts)
    assert pc == dict(n_conditions=1163, n_drugs=50, n_cell_lines=43), pc
    assert ps == dict(n_conditions=998, n_drugs=46, n_cell_lines=41), ps

    vals = {}
    for arm, _lab, _c, _m, _ls in SERIES:
        x0, n0 = agg(tc, arm)
        x1, n1 = agg(ts, arm)
        assert n0 == pc["n_conditions"] and n1 == ps["n_conditions"], arm
        vals[arm] = (x0, x1)

    # ------------------------------------------------------- guard: figure vs thesis text
    for arm, (a, b) in TABLE.items():
        got = vals[arm]
        assert round(got[0], 3) == a and round(got[1], 3) == b, (
            "%s reads %.4f -> %.4f, which does not round to the %.3f -> %.3f printed in "
            "tab:plateleak. STOP: do not pick one." % (arm, got[0], got[1], a, b))

    d_ref = vals["ceiling"][0] - vals["ceiling"][1]
    d_ctl = vals["control"][0] - vals["control"][1]
    d_mean = vals["mean"][0] - vals["mean"][1]
    d_lin = vals["linear"][0] - vals["linear"][1]

    drawn = {
        "figure": "fig-plateleak",
        "frame": "expression",
        "frames_not_drawn": ["nir_rank (a different frame; its values must not appear on this "
                             "axis)", "residual frame (not in these files at all)"],
        "tier": "tier2_unseen_drugs",
        "chance": style.CHANCE,
        "y_axis": "NIR (expression frame)",
        "x_positions": ["comparison set spans plates", "comparison set within plate"],
        "sources": {
            "cross_plate": "RESULTS_cluster/leak_crossplate.json :: "
                           "tiers.tier2_unseen_drugs.agg.<arm>_nir_expr[0]  "
                           "(config.same_plate_only = false)",
            "within_plate": "RESULTS_cluster/leak_sameplate.json :: "
                            "tiers.tier2_unseen_drugs.agg.<arm>_nir_expr[0]  "
                            "(config.same_plate_only = true)",
        },
        "population": {
            "cross_plate": pc,
            "within_plate": ps,
            "relation": "the within-plate column is a SUBSET of the cross-plate one, not the "
                        "same conditions rescored: holding the plate constant drops every "
                        "condition with no same-plate comparison set",
            "counted_from": "len(rows) and unique rows[].drug / rows[].cell_line; the file's "
                            "'n_drugs' field is the row count, not the drug count",
        },
        "series": {},
        "annotated_differences": {
            "within_well_split_half_reference": -d_ref,
            "control_copy": -d_ctl,
            "drug_agnostic_mean_baseline": -d_mean,
            "linear_map": -d_lin,
        },
        "intervals_drawn": False,
        "interval_reason": "neither leak_crossplate.json nor leak_sameplate.json carries any "
                           "interval: every agg entry is a [value, n] pair. The document's one "
                           "CI convention is a 95% bootstrap clustered over cell lines and no "
                           "such interval exists for these numbers, so the marks are bare.",
        "agrees_with": "Table tab:plateleak, Sections/05-Limitations.tex lines 148-151 and the "
                       "prose at 05-Limitations.tex:129, 04a-ruler.tex:461-462 "
                       "(mean 0.399/0.161, control-copy 0.551/0.510, reference 0.625/0.575)",
        "font": family,
    }
    for arm, lab, _c, _m, _ls in SERIES:
        drawn["series"][lab] = {
            "arm_key": arm,
            "cross_plate": vals[arm][0],
            "within_plate": vals[arm][1],
            "printed": [round(vals[arm][0], 3), round(vals[arm][1], 3)],
            "delta": vals[arm][1] - vals[arm][0],
        }

    # ------------------------------------------------------------------------------- canvas
    FW = style.TEXTWIDTH_IN          # 5.5906 in = 142 mm, placed at 1.0 and never scaled
    FH = 3.60
    fig = plt.figure(figsize=(FW, FH))

    # Explicit axes rectangle. The right-hand direct labels live OUTSIDE the axes (clip off), so
    # constrained layout would shrink the panel around them; the canvas has to stay exact.
    L, B, W, H = 0.62, 0.82, 2.55, FH - 0.82 - 0.30
    ax = fig.add_axes([L / FW, B / FH, W / FW, H / FH])

    X0, X1 = 0.0, 1.0
    XLIM = (-0.46, 1.46)
    YLIM = (0.13, 0.72)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    # the 0.50 rule, drawn on every NIR axis. Its own label would land in the right-hand label
    # column, between control-copy and the linear map; it goes under the rule instead.
    style.chance_line(ax, label=False)
    ax.annotate("chance", xy=(0.28, style.CHANCE - 0.006), ha="center", va="top",
                fontsize=9, color="black")

    # Label positions. control-copy (0.510) and the linear map (0.500) are 0.010 apart at the
    # right end, which is 0.045 in of paper -- a third of a 9pt line -- so those two labels are
    # nudged apart and joined to their marks by a leader. The linear map's LEFT value, 0.505,
    # would otherwise be struck through by the chance rule at 0.500, so it is nudged too. Every
    # other label sits at its own value.
    LEFT_Y = {"ceiling": vals["ceiling"][0],
              "control": vals["control"][0],
              "linear": 0.4670,
              "mean": vals["mean"][0]}
    RIGHT_Y = {"ceiling": vals["ceiling"][1],
               "control": 0.5225,
               "linear": 0.4720,
               "mean": vals["mean"][1]}

    for arm, lab, colour, marker, ls in SERIES:
        y0, y1 = vals[arm]
        ms = 7.0 if marker == "." else (7.5 if marker == "_" else 4.2)
        mew = 1.6 if marker == "_" else 1.0
        ax.plot([X0, X1], [y0, y1], ls=ls, lw=1.3, color=colour, zorder=3)
        ax.plot([X0, X1], [y0, y1], ls="none", marker=marker, ms=ms, mew=mew,
                color=colour, zorder=4, clip_on=False)

        # value at the left end
        ly0 = LEFT_Y[arm]
        if abs(ly0 - y0) > 1e-6:
            ax.plot([X0 - 0.05, X0 - 0.10], [y0, ly0], lw=0.5, color=colour,
                    clip_on=False, zorder=2)
        ax.annotate("%.3f" % y0, xy=(X0 - 0.13, ly0), ha="right", va="center",
                    fontsize=9, color=colour, annotation_clip=False)
        # value + name at the right end, in two aligned columns
        ly1 = RIGHT_Y[arm]
        if abs(ly1 - y1) > 1e-6:
            ax.plot([X1 + 0.05, X1 + 0.16], [y1, ly1], lw=0.5, color=colour,
                    clip_on=False, zorder=2)
        ax.annotate("%.3f   %s" % (y1, lab), xy=(X1 + 0.20, ly1), ha="left", va="center",
                    fontsize=9, color=colour, annotation_clip=False)

    # ---------------------------------------------------- the point the table cannot make
    ax.annotate("the reference falls too, $-%.3f$,\nabout as much as control-copy, $-%.3f$"
                % (d_ref, d_ctl),
                xy=(-0.42, 0.648), ha="left", va="bottom", fontsize=9, color=style.INK)
    xa = 0.36
    ya = vals["ceiling"][0] + xa * (vals["ceiling"][1] - vals["ceiling"][0])
    ax.annotate("", xy=(xa, ya + 0.006), xytext=(xa, 0.642),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=style.INK,
                                shrinkA=0, shrinkB=0))

    # -------------------------------------------------------------------------------- axes
    ax.set_xticks([X0, X1])
    ax.set_xticklabels(["comparison set\nspans plates", "comparison set\nwithin plate"])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticklabels(["0.20", "0.30", "0.40", "0.50", "0.60", "0.70"])
    # comparison_set=False: this axis carries BOTH comparison sets, and the x axis names them.
    ax.set_ylabel(style.nir_label("expression", comparison_set=False))
    ax.spines["bottom"].set_visible(False)

    style.save(fig, "fig-plateleak", drawn)

    print("  arm                              cross    within    delta")
    for arm, lab, _c, _m, _ls in SERIES:
        print("  %-30s  %.4f   %.4f   %+.4f" % (lab, vals[arm][0], vals[arm][1],
                                                vals[arm][1] - vals[arm][0]))
    print("  n: cross %s / within %s" % (pc, ps))


if __name__ == "__main__":
    main()

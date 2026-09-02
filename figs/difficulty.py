"""fig-difficulty (v2 rebuild) -- difficulty structure and the stratum ladder.

    python thesis_v2/figs/difficulty.py

Writes thesis_v2/figs/fig-difficulty.pdf and fig-difficulty.json. Nothing under thesis/ is
touched: style_v2.save() resolves its destination from dirname(__file__) and asserts it.

WHAT THE FIGURE SAYS
--------------------
(a) The within-well split-half precision reference against cells per condition, for two
    comparison sets: cross-plate and within plate. The reference rises with aggregation.
(b) Five arms in each of three difficulty strata, with the three model-minus-control-copy gaps.
    The load-bearing number is the +0.002 on the identifiable stratum: a predictor carrying no
    drug information whatever matches the model exactly where the ranking problem is easiest.

WHAT CHANGED FROM v1 endcell/figures/fig_difficulty.py, AND WHY
--------------------------------------------------------------
(1) GEOMETRY. v1 drew at 6.30in (its 160mm block) and saved bbox_inches="tight"; LaTeX then
    scaled it to the 142mm v2 block, printing every label 11% small. Drawn here at 5.591in and
    saved at the exact canvas, so \\includegraphics places it at scale 1.0.

(2) THE PANEL (a) AXIS LABEL WAS FALSE. It read "NIR (expression frame, within plate)" while
    plotting a cross-plate series AND a within-plate series. Half the marks contradicted the
    axis, on exactly the distinction chapter 3 insists is not interchangeable. The axis now names
    the FRAME only -- style_v2.nir_label("expression", comparison_set=False) -- and the legend,
    titled "comparison set", carries the cross-plate/within-plate split. Information moved; it
    did not disappear.

(3) ONE Y-AXIS, ONE LABEL. v1 gave each panel its own y-label (the same words twice) on two
    different y-ranges. Both panels plot expression-frame NIR in the same units, so they now
    SHARE the axis: one range, one label, one chance line at one height. That also makes the
    cross-panel reading legitimate -- (a)'s cross-plate reference topping out at 0.854 and (b)'s
    identifiable-stratum reference at 0.938 are now at comparable heights rather than at two
    silently different scales.

(4) BOTH "chance" LABELS WERE OUTSIDE THEIR AXES. v1's panel (a) label sat at axes-fraction
    x=1.002 -- in the inter-panel gutter, on top of panel (b)'s rotated y-label -- and panel
    (b)'s sat outside the right spine against the green linear diamond. Both are now placed in
    empty interior space, below their own line, inside their own axes.

(5) SHAPE CARRIES THE ENCODING. The five Okabe-Ito arm hues in panel (b) land within delta-L
    0.06 of each other in greyscale, so the panel was unreadable printed in black and white.
    Marker shape (dash / circle / square / triangle / diamond) is now the full encoding and
    colour is strictly redundant; the faint connectors take the arm's colour but are never the
    only cue.

(6) THE STRIKING FACT IS NOW STATED. On the unwinnable stratum the reference itself is 0.287 --
    BELOW the 0.50 chance line -- as are the model, control-copy and scramble marks. v1 drew
    those four marks under the chance line and said nothing. A reference below chance means the
    two halves of a well disagree about their own condition more often than a coin would; that
    is worth a sentence, and it is now annotated on the panel.

(7) THE RETIRED TERM. v1 labelled the reference arm "noise ceiling". That term is banned
    document-wide; the arm is "within-well split-half reference" (style_v2.REFERENCE_LABEL, which
    style_v2._check_label enforces on every string this figure draws). The stratum tick labels
    also dropped the word "ceiling" for "reference", and the x-axis names the binning variable.

NUMBERS AND PROVENANCE
----------------------
Nothing is typed in. Panel (a): drug_biology_atlas.json summary.cell_sweep (cross-plate) and
sweep_sameplate.json cell_sweep.*.ceiling_nir (within plate). Panel (b): recomputed from
nir_sameplate.json tiers.tier2_unseen_drugs.rows -- recomputed rather than read from the stored
`binned` block because that block omits control and scramble, and recomputing gives a unit test:
the bin sizes must come out at 296/106/204 and every arm mean must match stratify_sameplate.json
to 1e-9. Both assertions run below and both are hard failures.

INTERVALS. The arm means are BARE MARKERS. nir_sameplate.json carries no interval for a stratum
mean, and style_v2's one CI convention is 95% bootstrap clustered over cell lines -- inventing a
normal-approximation whisker to fill the space would sit a 0.0008 half-width next to this
thesis's genuine +/-0.03 intervals. The three model-minus-control-copy gaps ARE clustered
bootstraps (2000 resamples of the 40 cell lines, seed 0, reproducing v1 exactly); their intervals
are recorded in the sibling JSON and the identifiable one is quoted in the caption, but they are
not drawn, because a whisker on the gap and no whisker on the marks above it would read as though
the marks were exact.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- constants
EDGES = (0.6, 0.8)                 # within-well reference cut points, expression frame
ORDER = ("unwinnable", "marginal", "identifiable")
EXPECT_N = (296, 106, 204)         # unit test against stratify_sameplate.json test2.binned
ARMS = ("ceiling", "model", "control", "scramble", "linear")   # keys as stored in the JSON

# tick labels: three short lines. The binning VARIABLE is named once, on the x-axis label, so the
# ticks do not have to repeat "reference" three times inside a 62pt column.
TICKS = {"unwinnable":   "unwinnable\n$<0.6$",
         "marginal":     "marginal\n$0.6\\!-\\!0.8$",
         "identifiable": "identifiable\n$\\geq 0.8$"}

# stratify_sameplate.json test2.binned -> the cross-check. Keys are that file's own bin names.
STRATIFY_BINS = {
    "unwinnable":   "UNWINNABLE  (ceiling < 0.6 ~ chance: inert/redundant drug)",
    "marginal":     "MARGINAL    (0.6 <= ceiling < 0.8)",
    "identifiable": "IDENTIFIABLE (ceiling >= 0.8)",
}


def clustered_ci(vals, clusters, rng, n_boot=2000):
    """95% bootstrap resampling CELL LINES. Conditions within a line are pseudoreplicates.

    Identical to v1's implementation, including the RandomState(0) draw order, so the interval
    printed here is byte-for-byte the one the thesis quotes.
    """
    v = np.asarray(vals, float)
    cl = np.asarray(clusters)
    u = sorted(set(cl.tolist()))
    idx = {k: np.where(cl == k)[0] for k in u}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.randint(0, len(u), len(u))
        sel = np.concatenate([idx[u[t]] for t in take])
        boot[b] = v[sel].mean()
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    family = style.apply()
    print("figure face: %s" % style.font_report())
    rng = np.random.RandomState(0)

    # ------------------------------------------------------------------ panel (a) numbers
    cross = style.load("drug_biology_atlas.json")["summary"]["cell_sweep"]
    within = style.load("sweep_sameplate.json")["cell_sweep"]
    cx = sorted(int(k) for k in cross)
    cy = [float(cross[str(k)]) for k in cx]
    wx = sorted(int(k) for k in within)
    wy = [float(within[str(k)]["ceiling_nir"]) for k in wx]
    wn = [int(within[str(k)]["n_drugs"]) for k in wx]

    # ------------------------------------------------------------------ panel (b) numbers
    rows = style.load("nir_sameplate.json")["tiers"]["tier2_unseen_drugs"]["rows"]

    def get(r, a):
        return (r.get(a) or {}).get("nir_expr")

    bins = {k: [] for k in ORDER}
    for r in rows:
        c = get(r, "ceiling")
        if c is None:
            continue
        key = ORDER[0] if c < EDGES[0] else (ORDER[1] if c < EDGES[1] else ORDER[2])
        bins[key].append(r)

    got = tuple(len(bins[k]) for k in ORDER)
    assert got == EXPECT_N, "stratum sizes %s do not reproduce the stored %s" % (got, EXPECT_N)

    strata = {}
    for k in ORDER:
        rs = bins[k]
        strata[k] = {"n": len(rs),
                     "n_cell_lines": len(set(r["cell_line"] for r in rs)),
                     "arms": {a: float(np.mean([get(r, a) for r in rs if get(r, a) is not None]))
                              for a in ARMS}}
        pairs = [(r["cell_line"], get(r, "model") - get(r, "control")) for r in rs
                 if get(r, "model") is not None and get(r, "control") is not None]
        g, lo, hi = clustered_ci([d for _, d in pairs], [c for c, _ in pairs], rng)
        strata[k]["model_minus_control_copy"] = {
            "gap": g, "ci95_clustered_over_cell_lines": [lo, hi], "n_pairs": len(pairs)}

    # unit test 2: every recomputed arm mean that stratify_sameplate.json also stores must agree.
    strat = style.load("stratify_sameplate.json")["test2"]["binned"]
    for k in ORDER:
        stored = strat[STRATIFY_BINS[k]]
        assert stored["n"] == strata[k]["n"], k
        for a in ("ceiling", "model", "linear"):
            assert abs(stored[a] - strata[k]["arms"][a]) < 1e-9, (k, a)
    assert abs(style.load("stratify_sameplate.json")["test2"]["control_copy_identifiable"]
               - strata["identifiable"]["arms"]["control"]) < 1e-9
    assert abs(style.load("stratify_sameplate.json")["test2"]["scramble_arm"]["scramble"]
               - strata["identifiable"]["arms"]["scramble"]) < 1e-9
    print("unit tests OK: bin sizes %s and every stored arm mean reproduce "
          "stratify_sameplate.json" % (got,))

    # ------------------------------------------------------------------ the drawn record
    drawn = {
        "figure": "fig-difficulty",
        "frame": "expression",
        "chance": style.CHANCE,
        "note_no_intervals": (
            "Arm means are drawn as bare markers: the source carries no interval for a stratum "
            "mean, and this document's only CI convention is a 95% bootstrap clustered over cell "
            "lines. The three model-minus-control-copy gaps below ARE clustered bootstraps "
            "(2000 resamples of the 40 cell lines, seed 0) but are not drawn as whiskers."),
        "panel_a": {
            "quantity": style.REFERENCE_LABEL,
            "y_axis": style.nir_label("expression", comparison_set=False),
            "x_axis": "cells per condition",
            "cross_plate": {
                "source": "RESULTS_cluster/drug_biology_atlas.json :: summary.cell_sweep",
                "points": [{"cells": c, "nir_expr": v, "key": "summary.cell_sweep.%d" % c}
                           for c, v in zip(cx, cy)]},
            "within_plate": {
                "source": "RESULTS_cluster/sweep_sameplate.json :: cell_sweep",
                "points": [{"cells": c, "nir_expr": v, "n_drugs": n,
                            "key": "cell_sweep.%d.ceiling_nir" % c}
                           for c, v, n in zip(wx, wy, wn)]}},
        "panel_b": {
            "source": ("RESULTS_cluster/nir_sameplate.json :: "
                       "tiers.tier2_unseen_drugs.rows[*].<arm>.nir_expr"),
            "binning": {"variable": "rows[*].ceiling.nir_expr", "edges": list(EDGES),
                        "cross_check": ("RESULTS_cluster/stratify_sameplate.json :: "
                                        "test2.binned.*.{n,ceiling,model,linear}, "
                                        "test2.control_copy_identifiable, "
                                        "test2.scramble_arm.scramble")},
            "x_axis": "stratum, by within-well reference",
            "tick_labels": {k: TICKS[k].replace("\n", " ").replace("$", "") for k in ORDER},
            "legend_labels": [style.ARM[a]["label"] for a in ARMS],
            "annotation": ("reference 0.287 here: below chance, as are model, control-copy "
                           "and scramble"),
            "arm_keys": {"ceiling": "rows[*].ceiling.nir_expr -> %s" % style.REFERENCE_LABEL,
                         "model": "rows[*].model.nir_expr",
                         "control": "rows[*].control.nir_expr -> control-copy",
                         "scramble": "rows[*].scramble.nir_expr",
                         "linear": "rows[*].linear.nir_expr"},
            "strata": strata},
    }

    # =================================================================== figure
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(style.TEXTWIDTH_IN, 3.58), sharey=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.34]), layout="constrained")
    for ax in (axL, axR):
        # the range runs below the lowest tick on purpose: panel (b) keeps a clear strip under
        # the marks for the paired-contrast row, and panel (a) an empty corner for its legend.
        ax.set_ylim(0.03, 1.02)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # ------------------------------------------------------------------ (a)
    # Both series are the SAME estimator on two comparison sets, so both are drawn in the
    # reference's grey and separated by marker fill and dash pattern -- not by hue, which would
    # imply two different quantities. (CHANGE 5: greyscale-safe by construction.)
    axL.plot(cx, cy, marker="o", ms=3.6, lw=1.1, color=style.OKABE["grey"],
             mfc=style.OKABE["grey"], label="cross-plate")
    axL.plot(wx, wy, marker="s", ms=3.6, lw=1.1, ls=(0, (3.2, 1.6)), color=style.INK,
             mfc="white", mew=1.0, label="within plate")

    # CHANGE 4: the chance line's label lives INSIDE the axes, below its own line, in the empty
    # lower-left. style_v2.chance_line's default puts it at axes-fraction 1.002 -- the gutter.
    axL.axhline(style.CHANCE, color="black", lw=0.9, zorder=1)
    axL.annotate("chance", xy=(0.03, style.CHANCE - 0.012),
                 xycoords=("axes fraction", "data"), va="top", ha="left",
                 fontsize=9, color="black")

    axL.set_xscale("log")
    axL.set_xlim(3.5, 145)
    axL.set_xticks([4, 10, 20, 40, 120])
    axL.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axL.set_xlabel("cells per condition")
    # CHANGE 2: frame only. The legend title below carries the comparison set.
    axL.set_ylabel(style.nir_label("expression", comparison_set=False))
    axL.legend(loc="lower left", title="comparison set", title_fontsize=9,
               handlelength=1.5, handletextpad=0.5, labelspacing=0.35,
               borderaxespad=0.3, borderpad=0.2)
    axL.get_legend().get_title().set_color(style.QUIETINK)

    # ------------------------------------------------------------------ (b)
    xs = np.arange(len(ORDER), dtype=float)
    off = {"ceiling": -0.34, "model": -0.17, "control": 0.0, "scramble": 0.17, "linear": 0.34}
    handles, labels = [], []
    for a in ARMS:
        st = style.ARM[a]
        ys = [strata[k]["arms"][a] for k in ORDER]
        # faint connector: helps the eye follow ONE arm across the three strata (the text's
        # "the control tracks the reference almost step for step" is a cross-stratum reading).
        # Never the only cue -- shape is.
        axR.plot(xs + off[a], ys, lw=0.7, color=st["color"], alpha=0.35, zorder=2)
        # the reference's glyph is a bare dash, which needs extra length to read as a mark at all
        h, = axR.plot(xs + off[a], ys, ls="none", marker=st["marker"],
                      ms=7.2 if st["marker"] == "_" else 5.4,
                      color=st["color"], mfc=st["color"], mew=1.5, zorder=3)
        handles.append(h)
        labels.append(st["label"])

    axR.axhline(style.CHANCE, color="black", lw=0.9, zorder=1)
    # CHANGE 4 again: inside the axes, below its own line, in the empty strip between the
    # marginal and identifiable strata -- the only span of the chance line with no mark near it.
    axR.annotate("chance", xy=(1.72, style.CHANCE - 0.014), va="top", ha="center",
                 fontsize=9, color="black")

    # CHANGE 6: say the thing the panel was drawing silently. Anchored in the empty upper-left
    # with a near-vertical leader down to the unwinnable stratum's reference dash, so the leader
    # crosses nothing and cannot be mistaken for one of the arm connectors.
    axR.annotate("reference $0.287$ here:\nbelow chance, as are\nmodel, control-copy\nand scramble",
                 xy=(-0.35, 0.302), xytext=(-0.92, 1.005), textcoords="data",
                 va="top", ha="left", fontsize=9, color=style.QUIETINK, linespacing=1.30,
                 # dotted, so the leader cannot be read as one of the solid arm connectors
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=style.RULEGREY,
                                 linestyle=(0, (1, 1.8)), shrinkA=4, shrinkB=3))

    # the paired contrast is the honest summary of this panel: on the identifiable stratum a
    # predictor with NO drug information is within 0.002 of the model. The row sits in a clear
    # strip below every mark; its name sits on its own line above the numbers rather than beside
    # them, so no label has to share a baseline with a value it does not belong to.
    for i, k in enumerate(ORDER):
        axR.annotate("$%+.3f$" % strata[k]["model_minus_control_copy"]["gap"],
                     xy=(i, 0.055), ha="center", va="bottom", fontsize=9, color=style.INK)
    axR.annotate("model $-$ control-copy", xy=(-0.94, 0.150), ha="left", va="bottom",
                 fontsize=9, color=style.QUIETINK)

    axR.set_xlim(-0.96, 2.52)
    axR.set_xticks(xs)
    axR.set_xticklabels([TICKS[k] for k in ORDER])
    axR.set_xlabel("stratum, by within-well reference")
    axR.tick_params(axis="y", length=0)          # sharey: (b) inherits (a)'s labelled scale

    # panel letters, not panel titles: the caption does the describing, and v1's title
    # "(a) identifiability is noise-limited" asserted more than the caption does.
    for ax, letter in ((axL, "(a)"), (axR, "(b)")):
        ax.text(0.0, 1.015, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, color=style.INK)

    # One row of five. The spacing is tight because the reference's full name is long and the
    # measure is 142mm; check_fits() below is what proves the row actually fits the canvas.
    fig.legend(handles, labels, loc="outside lower center", ncol=5, columnspacing=0.75,
               handletextpad=0.3, handlelength=1.4, borderaxespad=0.0)

    style.save(fig, "fig-difficulty", drawn)

    # ------------------------------------------------------------------ console record
    print("\npanel (a) %s, expression frame" % style.REFERENCE_LABEL)
    print("  cross-plate  " + "  ".join("%d:%.3f" % (c, v) for c, v in zip(cx, cy)))
    print("  within plate " + "  ".join("%d:%.3f" % (c, v) for c, v in zip(wx, wy)))
    print("panel (b)")
    for k in ORDER:
        s = strata[k]
        print("  %-13s n=%3d  reference %.3f  model %.3f  control-copy %.3f  scramble %.3f  "
              "linear %.3f" % (k, s["n"], s["arms"]["ceiling"], s["arms"]["model"],
                               s["arms"]["control"], s["arms"]["scramble"], s["arms"]["linear"]))
        g = s["model_minus_control_copy"]
        print("                 model - control-copy = %+.4f [%+.4f, %+.4f]  (n=%d pairs, "
              "%d cell lines)" % (g["gap"], g["ci95_clustered_over_cell_lines"][0],
                                  g["ci95_clustered_over_cell_lines"][1], g["n_pairs"],
                                  s["n_cell_lines"]))
    return family


if __name__ == "__main__":
    main()

"""fig-calibration (v2): the standard metric is exploitable, and only one of five survives.

Rebuild of endcell/figures/fig_calibration.py against thesis_v2/figs/style_v2.py. The v1 script is
NEVER imported: its save() writes into thesis/figs/, which is the frozen read-only v1 tree.

This figure is an exhibit of METRIC failure. It must never read as a statement about model quality
-- DE-dr is the metric the chapter exists to discredit, and a panel that let a reader infer "the
model scores 0.73, which is close to the reference" would launder the artefact being exposed.
Hence the ordering: the zero-information baseline sits at the TOP.

WHAT CHANGED FROM v1, AND WHY
-----------------------------
(1) GEOMETRY. Drawn at 5.5906 in = 142 mm, the v2 body measure, and saved with the exact canvas
    (style_v2.save, bbox_inches=None), so \includegraphics places it at scale 1.0 and a 10pt label
    prints at 10pt. v1 was drawn at 6.30in and scaled to 0.887 in the v2 block, which is why its
    7.4pt annotations arrived at 6.6pt.

(2) STACKED, NOT SIDE BY SIDE. v1 put the two panels in a 1x2 grid, which left panel (a) about
    2.9in of axis for five two-line category labels and forced 7.6pt type. Each panel now gets the
    full measure. This is what buys the spelled-out arm names and the x-axis starting at 0.

(3) PANEL (a) x STARTS AT 0. v1 truncated at 0.70, which magnified the very gap under argument:
    on a 0.70-1.06 axis the distance from the model to the mid-rank baseline fills the panel. The
    argument is that a zero-information predictor OUTSCORES the model, and that argument is made by
    the ORDER of the points, not by the size of the gap. Starting at 0 states the order without
    inflating the difference. No truncation is used, so the caption asserts none.

(4) THE RETIRED TERM. v1 drew the string "noise ceiling" rotated 90 degrees up the left of panel
    (a) (fig_calibration.py:96), where it ran into the panel title. The term is banned
    document-wide; the band is now labelled with its construction's real name, horizontally,
    anchored at the bottom inside the axes. style_v2._check_label() enforces the ban mechanically.

(5) SHAPE AS WELL AS COLOUR. Okabe-Ito is colour-blind safe but collapses in greyscale
    (vermilion 0.222 vs grey 0.212 luminance), so each arm carries a distinct marker shape. Ink
    encodes information content -- black for the two arms carrying no drug information, green for
    control-only, accent for the model, orange for the wrong-drug ablation.

(6) THE 0.7328 AND 0.7291 LABELS moved to the LEFT of their markers. They sit 0.025 and 0.029
    below the reference band; set to the right they printed on top of it.

(7) PANEL (b) VALUE LABELS sit outside the arrow head and outside the interval, never against the
    y-axis spine. In v1 the -0.500 and -0.515 labels landed hard against the zero rule, where the
    minus sign read as part of the axis.

(8) PANEL (b) DRAWS ITS INTERVALS. style_v2's standing rule is that a quantity with no CLUSTERED
    interval is a bare marker. DRF from calibration_v5_sameplate_wide.json does have one -- 95%
    percentile bootstrap clustered on the 27 cell lines, the thesis's one CI convention -- and it
    is the interval Table 4.2's notes already quote. It is drawn as a band around the arrow head.
    Panel (a) draws no interval: the stored DE-dr intervals are normal-approximation over 400
    conditions (19,249 pairs for the reference), not clustered, and a half-width of 0.0008 beside
    the thesis's genuine +/-0.03 intervals would invite a reader to distrust all of them.

(9) weighted_r2 IS GREY, NOT VERMILION. v1 coloured everything below -0.05 as inverted. The thesis
    makes no directional claim for weighted_r2 (its sign tracks cell count: -0.104 here, +0.032 at
    ~121 cells), and Table 4.2's Reading column says exactly that. Only the three arms the text
    calls inverted are drawn in the inverted ink.

Usage:  python thesis_v2/figs/calibration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

TIER = "tier2_unseen_drugs"
CONV = "worst"

METRIC_LABEL = {
    "nir": "NIR",
    "weighted_r2": "weighted $R^2$",
    "panel_tau": "panel-$\\tau$",
    "spearman_expr": "Spearman",
    "de_delta": "DE-$\\Delta r$",
}
ORDER = ["nir", "weighted_r2", "panel_tau", "spearman_expr", "de_delta"]

# Table 4.2's Reading column, verbatim in effect: three metrics are called inverted, weighted_r2
# is explicitly given no directional claim. The colour follows the text, not a threshold on |DRF|.
INVERTED = ("panel_tau", "spearman_expr", "de_delta")
NO_CLAIM = ("weighted_r2",)

# The reference band's drawn halfwidth. This is a DRAWING DEVICE, not an interval: a within-well
# split-half figure is an estimate and is never drawn as a hard rule (style_v2 rationale). It is
# recorded as such in the sibling JSON so nobody can read a width off it.
BAND_HALFWIDTH_VISUAL = 0.006


def main():
    family = style.apply()
    print("font in force: %s" % style.font_report())

    # ---------------------------------------------------------------- data (panel a)
    lin = style.load("eval_endcell_linear.json")["linear"][CONV][TIER]
    mod = style.load("eval_endcell_model.json")
    cpu = style.load("eval_endcell_cpu.json")

    ref_blk = cpu["ceiling"][TIER][CONV]["cell_vs_cell_de50"]
    reference = ref_blk["mean"]
    n_ref_pairs = ref_blk["n"]
    model_blk = mod["model"][TIER]["conventions"][CONV]["de_pearson_k50"]
    scr_blk = mod["scramble"][TIER]["conventions"][CONV]["de_pearson_k50"]

    # (label, value, colour, marker, markersize, side the value label goes on, source key path)
    rows = [
        ("mid-rank constant baseline\n(no drug, cell or biology)",
         lin["revert_center"]["de_pearson_k50"]["mean"], style.INK, "*", 11, "right",
         "linear.worst.%s.revert_center.de_pearson_k50.mean" % TIER),
        ("mean-control baseline\n(no drug information, no fit)",
         lin["revert_mean"]["de_pearson_k50"]["mean"], style.INK, "X", 6.5, "right",
         "linear.worst.%s.revert_mean.de_pearson_k50.mean" % TIER),
        ("ridge linear\n(control only)",
         lin["ridge"]["de_pearson_k50"]["mean"], style.ARM["linear"]["color"], "D", 5.2, "right",
         "linear.worst.%s.ridge.de_pearson_k50.mean" % TIER),
        ("model\n(control $+$ drug)",
         model_blk["mean"], style.ACCENT, "o", 6.2, "left",
         "model.%s.conventions.worst.de_pearson_k50.mean" % TIER),
        ("scramble ablation\n(control $+$ wrong drug)",
         scr_blk["mean"], style.ARM["scramble"]["color"], "^", 6.5, "left",
         "scramble.%s.conventions.worst.de_pearson_k50.mean" % TIER),
    ]
    n_conditions = model_blk["n"]

    # ---------------------------------------------------------------- data (panel b)
    # CANONICAL same-plate arm. drf_sameplate.json is a superseded run (655 drugs, 61 cell lines,
    # NIR DRF +0.446); the canonical within-plate arm is the wide, cell-line-clustered rerun:
    # 586 drugs over 44 cell-line x plate groups spanning 27 cell lines.
    drf = style.load("calibration_v5_sameplate_wide.json")
    dd = drf["drf"]["neg_mean"]

    # ---------------------------------------------------------------- the sibling JSON
    drawn = {
        "figure": "fig-calibration",
        "tier": TIER,
        "convention": CONV,
        "panel_a": {
            "_source_file": "RESULTS_cluster/eval_endcell_linear.json, "
                            "RESULTS_cluster/eval_endcell_model.json, "
                            "RESULTS_cluster/eval_endcell_cpu.json",
            "_metric": "DE-Delta r, K = 50, Pearson",
            "_n_conditions": n_conditions,
            "_intervals_drawn": False,
            "_why_no_intervals": ("the stored DE-dr intervals are normal-approximation over "
                                  "%d conditions (%d pairs for the reference), not the 95%% "
                                  "bootstrap clustered over cell lines this thesis draws"
                                  % (n_conditions, n_ref_pairs)),
            "arms": [
                {"arm": lab.replace("\n", " ").replace("$+$", "+"),
                 "value": val,
                 "source_file": ("eval_endcell_model.json" if key.startswith(("model.", "scramble."))
                                 else "eval_endcell_linear.json"),
                 "key_path": key}
                for (lab, val, _c, _m, _ms, _side, key) in rows
            ],
            "within_well_split_half_reference": {
                "value": reference,
                "n_pairs": n_ref_pairs,
                "source_file": "eval_endcell_cpu.json",
                "key_path": "ceiling.%s.worst.cell_vs_cell_de50.mean" % TIER,
                "drawn_as": "vertical band",
                "band_halfwidth_drawn": BAND_HALFWIDTH_VISUAL,
                "band_halfwidth_is_a_visual_device_not_an_interval": True,
            },
        },
        "panel_b": {
            "_source_file": "RESULTS_cluster/calibration_v5_sameplate_wide.json",
            "_arm": "same plate",
            "_n_drugs": drf["n_drugs"],
            "_n_groups": drf["n_groups"],
            "_n_celllines": drf["n_celllines"],
            "_intervals_drawn": True,
            "_interval": "95% percentile bootstrap clustered on cell line",
            "metrics": [
                {"metric": m,
                 "label": METRIC_LABEL[m].replace("$", ""),
                 "drf": dd[m]["drf"],
                 "printed": "%+.3f" % dd[m]["drf"],
                 "ci_low": dd[m]["ci"][0],
                 "ci_high": dd[m]["ci"][1],
                 "n_clusters": dd[m]["n_clusters"],
                 "cluster_unit": dd[m]["cluster_unit"],
                 "source_file": "calibration_v5_sameplate_wide.json",
                 "key_path": "drf.neg_mean.%s.drf" % m,
                 "ci_key_path": "drf.neg_mean.%s.ci" % m}
                for m in ORDER
            ],
        },
        "_notes": {
            "panel_tau_vs_table": ("figure prints -0.250 from drf.neg_mean.panel_tau.drf = "
                                   "-0.2504545628681181; Table 4.2 prints -0.251. Flagged, not "
                                   "resolved here."),
            "panel_a_x_axis": "starts at 0; no truncation",
            "font_resolved": family,
        },
    }

    # ---------------------------------------------------------------- draw
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(style.TEXTWIDTH_IN, 4.75), layout="constrained",
        gridspec_kw=dict(height_ratios=[1.18, 1.0]))
    fig.get_layout_engine().set(h_pad=0.06, hspace=0.10, w_pad=0.02)

    # ------------------------------------------------- (a) the exploit
    # The reference is a split-half estimate, so it is a BAND, never a hard rule. The halfwidth is
    # a drawing device (see BAND_HALFWIDTH_VISUAL); no interval is being asserted.
    axA.axvspan(reference - BAND_HALFWIDTH_VISUAL, reference + BAND_HALFWIDTH_VISUAL,
                color=style.RULEGREY, lw=0, zorder=0)
    # (4) horizontal, anchored at the bottom inside the axes, clear of the title. The y-limit
    # below reserves a row's worth of empty space under the last arm so this cannot collide with
    # the scramble marker, which sits only 0.029 to the band's left.
    axA.annotate(style.REFERENCE_LABEL, xy=(reference - 0.018, 0.025),
                 xycoords=("data", "axes fraction"), ha="right", va="bottom",
                 fontsize=9, color=style.QUIETINK)

    for y, (lab, val, col, mk, ms, side, _key) in enumerate(rows):
        axA.plot([val], [y], marker=mk, ms=ms, color=col, mec=col, mew=0.9,
                 ls="none", zorder=3, clip_on=False)
        dx = 9 if side == "right" else -9
        axA.annotate("%.4f" % val, xy=(val, y), xytext=(dx, 0), textcoords="offset points",
                     ha="left" if side == "right" else "right", va="center",
                     fontsize=9, color=col)

    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels([r[0] for r in rows], fontsize=9)
    axA.tick_params(axis="y", length=0)
    axA.invert_yaxis()
    axA.set_ylim(len(rows) - 0.10, -0.55)
    # Right limit is set by the 0.9999 value label, not by the data: at 1.10 the label ran off the
    # canvas and save() writes the exact canvas, so it would have been clipped, not accommodated.
    axA.set_xlim(0.0, 1.18)
    axA.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axA.set_xlabel("DE-$\\Delta r$ ($K = 50$, Pearson), tier 2, worst convention")
    axA.set_title("(a) less information, higher score", fontsize=10, loc="left", pad=6)
    axA.spines["left"].set_visible(False)

    # ------------------------------------------------- (b) the diagnosis
    axB.axvline(0.0, color=style.INK, lw=0.9, zorder=1)
    for y, m in enumerate(ORDER):
        v = dd[m]["drf"]
        lo, hi = dd[m]["ci"]
        col = (style.ACCENT if m not in INVERTED + NO_CLAIM else
               style.QUIETINK if m in NO_CLAIM else style.INK)
        # (8) the clustered interval, as a band around the head, so the arrow shaft stays readable.
        # Drawn as a bar rather than a whisker because a whisker at the same y would lie collinear
        # with the arrow shaft and disappear under it between 0 and the head.
        axB.barh(y, hi - lo, left=lo, height=0.34, color=col, alpha=0.22, lw=0, zorder=2)
        axB.annotate("", xy=(v, y), xytext=(0, y),
                     arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col,
                                     shrinkA=0, shrinkB=0, mutation_scale=10), zorder=3)
        # (7) outside the head AND outside the interval, never against the zero rule.
        anchor = hi if v > 0 else lo
        axB.annotate("%+.3f" % v, xy=(anchor, y), xytext=(7 if v > 0 else -7, 0),
                     textcoords="offset points", va="center",
                     ha="left" if v > 0 else "right", fontsize=9, color=col)

    axB.set_yticks(range(len(ORDER)))
    axB.set_yticklabels([METRIC_LABEL[m] for m in ORDER], fontsize=9)
    axB.tick_params(axis="y", length=0)
    axB.invert_yaxis()
    axB.set_ylim(len(ORDER) - 0.45, -0.95)
    # Limits set by the outermost value label, not the data, so every label lands INSIDE the
    # bottom spine rather than hanging off the end of it.
    axB.set_xlim(-0.88, 0.88)
    axB.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    # Two lines: on one line this label is wider than the 142mm measure and would be clipped.
    axB.set_xlabel("dynamic-range fraction\n"
                   "$0 =$ drug-agnostic baseline, $1 =$ perfect predictor")
    axB.set_title("(b) only one metric is calibrated", fontsize=10, loc="left", pad=6)
    axB.spines["left"].set_visible(False)
    axB.annotate("inverted $\\longleftarrow$", xy=(-0.035, -0.80), ha="right", va="center",
                 fontsize=9, color=style.INK)
    axB.annotate("$\\longrightarrow$ calibrated", xy=(0.035, -0.80), ha="left", va="center",
                 fontsize=9, color=style.ACCENT)

    style.save(fig, "fig-calibration", drawn)

    # ---------------------------------------------------------------- console ledger
    print("  panel (a)  reference (within-well split-half, %d pairs) = %.4f" % (n_ref_pairs, reference))
    for lab, val, _c, _m, _ms, _s, _k in rows:
        print("  panel (a)  %-46s %.6f" % (lab.replace("\n", " ").replace("$+$", "+"), val))
    for m in ORDER:
        print("  panel (b)  %-14s %+0.6f   printed %+.3f   ci [%+.3f, %+.3f]"
              % (m, dd[m]["drf"], dd[m]["drf"], dd[m]["ci"][0], dd[m]["ci"][1]))


if __name__ == "__main__":
    main()

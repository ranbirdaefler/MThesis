"""fig:splits -- the whole-well holdout, on both clustering axes, with the null bounded.

The second pillar of the chapter's central negative. Panel (a) is a forest: where each split's
model NIR actually lands against chance and against its own within-well split-half reference, and
-- on a separate axis with its own zero -- the gap against the geometry-defined ``orth`` partner
drawn TWICE per row, once clustered on cell line and once clustered on drug. Drawing it twice is
the point: on ``unseen_combo`` the gap excludes zero on one clustering axis and not on the other,
and the verdict for that row depends on which axis is chosen. Panel (b) shows the per-condition
NIR distributions, so the pooled 0.537 reads as a mean over a nearly flat spread of conditions
rather than as a location where conditions sit.

EVERY NUMBER IS READ FROM RESULTS_cluster/. Nothing is typed in.
  * point estimates, intervals, df, references, n  <- CANONICAL_NUMBERS.json  splits{}, ceilings{}
  * the 80%-power detection limit                  <- CANONICAL_NUMBERS.json  detection_limit{}
  * per-condition NIR for panel (b)                <- re_v3.json  records[].model, by split
  * cross-check of the point estimates             <- re_v3.json  means.generalization{}

TWO DELIBERATE DEPARTURES FROM THE FIGURE BRIEF, both because style_v2 outranks it:
  1. The brief asks for a DOTTED chance rule. style_v2.chance_line() draws the 0.50 rule thin,
     solid and black, and its docstring says "Never dashed, never omitted" -- it is a document-wide
     convention, not a per-figure choice. Solid it is.
  2. re_v3.json's means.generalization[*].ci is a DIFFERENT interval from the one the thesis
     prints (train: [0.0471, 0.1324] there vs [0.0382, 0.1415] in tab:generalisation). The thesis
     table and CANONICAL_NUMBERS.json agree with each other; re_v3's own `ci` field is a third
     estimator and is NOT DRAWN. Only CANONICAL's gap_ci_line / gap_ci_drug reach the page.

THE DETECTION-LIMIT BAND SITS ON THE LEFT (ABSOLUTE NIR) AXIS, not on the gap axis. The stored
quantity is "detectable_at_80pc_power" against CHANCE at n = 199 (04c-reencoding.tex: "an
80%-power detection limit of 0.079 against chance"), so its frame is the absolute-NIR axis around
0.50, and putting it around zero on the gap axis would silently restate it as a limit on the
model-minus-comparator difference, which is not what was computed.
"""
import os
import sys

import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as S  # noqa: E402  (sets the Agg backend on import)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

SPLITS = ["train", "unseen_combo", "unseen_drug"]          # top row to bottom row
CHANCE = S.CHANCE


def mono_family():
    """The document's \\texttt face, by the name matplotlib actually knows it under.

    preamble.tex:50 loads \\usepackage[scaled=0.86]{inconsolata}, whose default variant is zi4.
    style_v2.MONO_STACK asks for the family "Inconsolata", but the OTFs in the TeX tree register
    themselves as "Inconsolatazi4" / "InconsolataN", so a bare request for "Inconsolata" silently
    lands on Consolas. The split names in this figure are \\texttt{} tokens in the running text
    (\\texttt{unseen\\_combo} etc.), so they are set in the same face the paragraph sets them in.
    """
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Inconsolatazi4", "InconsolataN", "Inconsolata", "Consolas", "DejaVu Sans Mono"):
        if cand in have:
            return cand
    return "monospace"


# --------------------------------------------------------------------------- data
def silverman_bw(v):
    """Silverman's rule on the ORIGINAL sample, so the reflection trick below does not silently
    change the bandwidth (the augmented sample is 3x as long and twice as wide)."""
    v = np.asarray(v, float)
    n = v.size
    q1, q3 = np.percentile(v, [25, 75])
    spread = min(v.std(ddof=1), (q3 - q1) / 1.349)
    return 0.9 * spread * n ** (-0.2)


def reflected_kde(v, grid, bw):
    """KDE on [0, 1] with reflection at both bounds. NIR is bounded; an unreflected kernel leaks
    mass past 0 and 1 and then under-states the density AT the bounds, which is where a
    near-uniform distribution has to be read."""
    v = np.asarray(v, float)
    aug = np.concatenate([v, -v, 2.0 - v])
    kde = gaussian_kde(aug, bw_method=bw / aug.std(ddof=1))
    return kde(grid) * 3.0


def collect():
    canon = S.load("CANONICAL_NUMBERS.json")
    rev3 = S.load("re_v3.json")

    per_condition = {s: [] for s in SPLITS}
    for rec in rev3["records"]:
        if rec.get("model") is None:
            continue
        if rec.get("split") in per_condition:
            per_condition[rec["split"]].append(float(rec["model"]))

    gen = rev3["means"]["generalization"]
    rows = {}
    for s in SPLITS:
        c = canon["splits"][s]
        v = np.asarray(per_condition[s], float)

        # cross-check CANONICAL against re_v3's own means, and against the records themselves.
        assert abs(c["model_nir"] - gen[s]["model_nir"]) < 1e-9, s
        assert abs(c["gap"] - gen[s]["gap"]) < 1e-9, s
        assert c["n"] == gen[s]["n"] == v.size, (s, c["n"], gen[s]["n"], v.size)
        assert c["n_lines"] == gen[s]["n_cell_lines"], s
        assert abs(v.mean() - c["model_nir"]) < 1e-9, (s, v.mean(), c["model_nir"])
        assert abs(c["ceiling"] - canon["ceilings"][s]) < 1e-12, s
        assert gen[s]["comparator"] == "scramble_orth", s

        rows[s] = dict(
            n=int(c["n"]), n_lines=int(c["n_lines"]), n_drugs=int(c["n_drugs"]),
            n_wells=int(c["n_wells"]),
            model_nir=float(c["model_nir"]), nir_ci=[float(x) for x in c["nir_ci"]],
            gap=float(c["gap"]),
            gap_ci_line=[float(x) for x in c["gap_ci_line"]],
            gap_ci_drug=[float(x) for x in c["gap_ci_drug"]],
            df_line=int(c["df_line"]), df_drug=int(c["df_drug"]),
            reference=float(c["ceiling"]),
            frac_at_or_below_chance=float((v <= CHANCE).mean()),
            median=float(np.median(v)), sd=float(v.std(ddof=1)),
            values=v,
        )

    dl = canon["detection_limit"]
    assert dl["split"] == "unseen_drug" and dl["n"] == rows["unseen_drug"]["n"]
    return rows, dl, canon


# --------------------------------------------------------------------------- draw
def main():
    family = S.apply()
    print("style_v2 face:", S.font_report())
    rows, dl, canon = collect()
    MONO = mono_family()
    print("mono face:", MONO)

    W = S.TEXTWIDTH_IN            # 5.5906 in = 142 mm. Placed at 1.0; never scaled.
    # 5.20, not the 5.05 this figure was first composed at. The annotations are set at the
    # module's 9pt and the axis labels at its 10pt, and at those sizes panel (a)'s two-line
    # x-label ended 1.8pt above panel (b)'s "chance" label. The canvas grows to hold the type;
    # the type never shrinks to fit the canvas (style_v2 CHANGE 1 -- and nothing here is scaled
    # in LaTeX, so a taller canvas costs height on the page and nothing else).
    H = 5.20
    fig = plt.figure(figsize=(W, H))

    # Explicit axes rectangles rather than constrained_layout: the row labels and the direct
    # labels live OUTSIDE the axes, and an automatic layout engine cannot see them.
    LM = 1.20 / W                 # left margin, holds the split names + n
    A_L_R = 2.95 / W              # panel (a) left sub-axis right edge
    DIV = 3.06 / W                # the hard divider between the two sub-axes
    A_R_L = 3.18 / W              # panel (a) right sub-axis left edge
    RM = 5.47 / W                 # right edge of everything

    # inches from the BOTTOM of the canvas. Panel (b) keeps its old coordinates, panel (a) rises
    # with the canvas, so the whole 0.15in goes into the gap between the two panels -- which is
    # the one place that was short of room -- and the top and bottom margins are unchanged.
    A_TOP_IN = 4.80
    a_top, a_h = A_TOP_IN / H, 1.80 / H
    b_top, b_h = 2.10 / H, 1.45 / H

    axL = fig.add_axes([LM, a_top - a_h, A_L_R - LM, a_h])
    axR = fig.add_axes([A_R_L, a_top - a_h, RM - A_R_L, a_h])
    axB = fig.add_axes([LM, b_top - b_h, RM - LM, b_h])

    Y = {"train": 3.0, "unseen_combo": 2.0, "unseen_drug": 1.0}
    OFF = 0.18                    # vertical offset of the two clustered intervals within a row

    # ---------------------------------------------------------------- (a) left: absolute NIR
    axL.set_xlim(0.38, 1.00)
    # the bottom limit reserves a strip under the unseen_drug row for the detection-limit label,
    # so that label clears the bottom edge of the band it names instead of notching into it.
    axL.set_ylim(0.02, 3.95)
    axL.set_yticks([])
    axL.spines["left"].set_visible(False)
    axL.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axL.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    # the 80%-power detection limit, unseen_drug row only, BEHIND everything. It is a bound on
    # what this arm could have seen, not an interval around what it did see.
    lim = float(dl["detectable_at_80pc_power"])
    yd = Y["unseen_drug"]
    axL.add_patch(Rectangle(
        (CHANCE - lim, yd - 0.38), 2 * lim, 0.76,
        facecolor=S.PANELBG, lw=0, zorder=0))

    S.chance_line(axL, orientation="v")

    # The label sits UNDER the band and centred on it, not beside it: the unseen_drug row's
    # within-well reference marker is at 0.836 and a right-hand label would run straight into it.
    # But the band is centred on chance, so a label centred under the band is also centred on the
    # chance rule, and the rule used to run straight through the words. Two changes fix that
    # without moving the label off the quantity it names or dropping the rule (style_v2: the 0.50
    # rule is never omitted): it is drawn AFTER chance_line(), and it carries an opaque white
    # bbox, so the rule is interrupted BY the label for the label's height and resumes below it.
    axL.annotate("80%-power detection limit", xy=(CHANCE, yd - 0.62),
                 va="center", ha="center", fontsize=9, color=S.QUIETINK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.4))

    for s in SPLITS:
        r, y = rows[s], Y[s]
        S.whisker(axL, None, y, r["nir_ci"][0], r["nir_ci"][1],
                  color=S.ACCENT, lw=1.3, zorder=3)
        axL.plot([r["model_nir"]], [y], marker="o", ms=4.2, color=S.ACCENT,
                 mec="white", mew=0.6, zorder=4, clip_on=False)
        # The within-well reference, in the ONE encoding style_v2 defines for it. This used to be
        # a hardcoded white-faced diamond, which made this figure the fifth different drawing of a
        # single quantity across the document -- and a diamond is the `linear` arm's glyph
        # elsewhere in the same palette. style_v2.ARM["reference"] is the series form of the
        # encoding (grey dash); reference_band() is the scalar form, and does not apply here
        # because each row carries its OWN reference value, not one level shared by the axis.
        # Drawn exactly as difficulty.py draws it: a bare dash needs extra length to read as a
        # mark at all. No interval is stored for it, so it stays a bare marker and the caption
        # says why.
        ref_st = S.ARM["reference"]
        axL.plot([r["reference"]], [y], ls="none", marker=ref_st["marker"],
                 ms=7.2 if ref_st["marker"] == "_" else 5.4,
                 color=ref_st["color"], mfc=ref_st["color"], mew=1.5,
                 zorder=4, clip_on=False)

    # direct label for the reference, with a short leader down to the train marker.
    # The leader used to stop at 3.22, a fifth of the row pitch clear of the dash it points at,
    # which read as a stray rule rather than a leader. It now runs to 3.06, just above the
    # marker's own half-height, so the eye carries the label onto the mark.
    ref_train = rows["train"]["reference"]
    axL.plot([ref_train, ref_train], [3.06, 3.42], color=S.QUIETINK, lw=0.6, zorder=2)
    axL.annotate(S.REFERENCE_LABEL.replace(" reference", "\nreference"),
                 xy=(0.998, 3.48), va="bottom", ha="right",
                 fontsize=9, color=S.QUIETINK, linespacing=1.25)

    # no fontsize= here or on the other two axis labels: axes.labelsize is style_v2's to set
    # (apply() puts it at 10), and this figure was overriding it down to 9.5.
    axL.set_xlabel("model NIR (residual frame,\ncell-line comparison set)",
                   linespacing=1.3, labelpad=3)

    # ---------------------------------------------------------------- (a) right: the gap, twice
    # the left limit is 0.015 wider than the leftmost drawn datum needs (-0.1049, the drug-
    # clustered interval on unseen_drug): the extra strip is what lets the two "cluster:" labels
    # below sit entirely on the negative side of the zero rule instead of running across it.
    axR.set_xlim(-0.150, 0.205)
    axR.set_ylim(0.02, 3.95)
    axR.set_yticks([])
    axR.spines["left"].set_visible(False)
    axR.set_xticks([-0.10, 0.0, 0.10])
    axR.set_xticklabels([u"−0.10", "0.00", "+0.10"])
    axR.set_xticks([-0.05, 0.05, 0.15], minor=True)

    axR.axvline(0.0, color=S.INK, lw=0.9, zorder=1)
    axR.annotate("zero", xy=(0.0, 1.004), xycoords=("data", "axes fraction"),
                 va="bottom", ha="center", fontsize=9, color=S.INK)

    for s in SPLITS:
        r, y = rows[s], Y[s]
        for ci, df, dy in ((r["gap_ci_line"], r["df_line"], +OFF),
                           (r["gap_ci_drug"], r["df_drug"], -OFF)):
            S.whisker(axR, None, y + dy, ci[0], ci[1], color=S.ACCENT, lw=1.3, zorder=3)
            axR.plot([r["gap"]], [y + dy], marker="o", ms=3.6, color=S.ACCENT,
                     mec="white", mew=0.5, zorder=4)
            axR.annotate("df %d" % df, xy=(ci[1] + 0.005, y + dy), va="center", ha="left",
                         fontsize=9, color=S.QUIETINK)

    # Direct labelling of the two clustering axes, once, on the top row. Right-aligned to a
    # common edge just LEFT of zero: anchored at +0.018 they ran leftwards and the zero rule --
    # the null this whole sub-axis is read against -- struck through both of them. The top row's
    # intervals are the two furthest from zero on the positive side, so the negative side of that
    # row is empty and the labels can terminate short of the rule rather than crossing it.
    for txt, dy in (("cluster: cell line", +OFF), ("cluster: drug", -OFF)):
        axR.annotate(txt, xy=(-0.008, Y["train"] + dy), va="center", ha="right",
                     fontsize=9, color=S.QUIETINK)

    axR.set_xlabel("gap vs. orth partner\n(model NIR − scrambled-partner NIR)",
                   linespacing=1.3, labelpad=3)

    # the hard divider: the two sub-axes carry different quantities and different nulls.
    fig.add_artist(Line2D(
        [DIV, DIV], [(a_top - a_h) - 0.055, a_top + 0.02],
        color=S.RULEGREY, lw=0.8, transform=fig.transFigure))

    # ---------------------------------------------------------------- row labels (both panels)
    tr = blended_transform_factory(axL.transAxes, axL.transData)
    for s in SPLITS:
        r, y = rows[s], Y[s]
        axL.text(-0.045, y + 0.10, s, transform=tr, ha="right", va="bottom",
                 fontsize=9, family=MONO, color=S.INK)
        axL.text(-0.045, y - 0.02, "n = %d, %d lines" % (r["n"], r["n_lines"]), transform=tr,
                 ha="right", va="top", fontsize=9, color=S.QUIETINK)

    # ---------------------------------------------------------------- (b) per-condition spread
    axB.set_xlim(0.0, 1.0)
    axB.set_ylim(-0.42, 2.90)
    axB.set_yticks([])
    axB.spines["left"].set_visible(False)
    axB.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axB.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    axB.set_xlabel("per-condition NIR (residual frame, cell-line comparison set)",
                   labelpad=3)

    grid = np.linspace(0.0, 1.0, 401)
    dens, bws = {}, {}
    for s in SPLITS:
        bws[s] = silverman_bw(rows[s]["values"])
        dens[s] = reflected_kde(rows[s]["values"], grid, bws[s])
    scale = 0.62 / max(d.max() for d in dens.values())

    BASE = {"train": 2.2, "unseen_combo": 1.1, "unseen_drug": 0.0}
    # The 0.50 rule is drawn on EVERY strip (style_v2's standing rule) but as a per-strip segment
    # rather than one full-height axvline: the summary lines sit just below each baseline, and a
    # rule running the whole height of the panel would cross the whitespace they need.
    axB.annotate("chance", xy=(CHANCE, 1.004), xycoords=("data", "axes fraction"),
                 va="bottom", ha="center", fontsize=9, color=S.INK)
    for s in SPLITS:
        r, b, d = rows[s], BASE[s], dens[s] * scale
        lo = grid <= CHANCE
        axB.fill_between(grid[lo], b, b + d[lo], color=S.RULEGREY, lw=0, zorder=2)
        axB.fill_between(grid[~lo], b, b + d[~lo], color=S.ACCENT, alpha=0.20, lw=0, zorder=2)
        axB.plot(grid, b + d, color=S.ACCENT, lw=1.1, zorder=3)
        axB.plot([0, 1], [b, b], color=S.RULEGREY, lw=0.7, zorder=1)
        axB.plot([CHANCE, CHANCE], [b, b + 0.68], color=S.INK, lw=0.9, zorder=5)
        m = r["model_nir"]
        axB.plot([m, m], [b, b + np.interp(m, grid, d)], color=S.ACCENT, lw=1.7, zorder=6)
        axB.annotate("mean %.3f  ·  %.1f%% of conditions at or below chance"
                     % (m, 100 * r["frac_at_or_below_chance"]),
                     xy=(0.995, b - 0.21), va="center", ha="right",
                     fontsize=9, color=S.QUIETINK)

    trb = blended_transform_factory(axB.transAxes, axB.transData)
    for s in SPLITS:
        axB.text(-0.045, BASE[s] + 0.06, s, transform=trb, ha="right", va="bottom",
                 fontsize=9, family=MONO, color=S.INK)

    # ---------------------------------------------------------------- panel letters
    fig.text(0.012, (A_TOP_IN + 0.16) / H, "(a)", fontsize=10, color=S.INK, va="bottom")
    fig.text(0.012, (2.10 - 0.05) / H, "(b)", fontsize=10, color=S.INK, va="bottom")

    # ---------------------------------------------------------------- ledger
    drawn = {
        "figure": "fig:splits",
        "frame": "residual",
        "comparison_set": "other drugs in the same cell line",
        "comparator_arm": "scramble_orth (geometry-defined orth partner)",
        "chance": CHANCE,
        "sources": {
            "point_estimates_intervals_df_references_n": "RESULTS_cluster/CANONICAL_NUMBERS.json"
                                                         " -> splits{}, ceilings{}",
            "detection_limit": "RESULTS_cluster/CANONICAL_NUMBERS.json -> detection_limit{}",
            "per_condition_nir_panel_b": "RESULTS_cluster/re_v3.json -> records[].model by split",
            "cross_check": "RESULTS_cluster/re_v3.json -> means.generalization{}",
            "canonical_provenance": canon["_source"],
        },
        "not_drawn": {
            "re_v3_means_generalization_ci": "re_v3.json stores a third interval per split "
                                             "(train [0.04712, 0.13243]) that disagrees with both "
                                             "CANONICAL_NUMBERS.json and tab:generalisation. Only "
                                             "gap_ci_line / gap_ci_drug are drawn.",
        },
        "encoding": {
            "model": "style_v2.ARM['model'] -- accent filled circle with a 95% clustered interval",
            "reference": "style_v2.ARM['reference'] -- grey dash, the module's SERIES form of the "
                         "within-well split-half reference. reference_band() is the scalar form "
                         "and does not apply here: each row carries its own reference value, not "
                         "one level shared by the axis. No interval is stored for it, so it is a "
                         "bare marker.",
            "annotation_type_size_pt": 9,
            "axis_label_type_size_pt": "style_v2 axes.labelsize (10)",
        },
        "panel_a": {},
        "panel_b": {
            "kernel": "gaussian, reflected at 0 and 1, Silverman bandwidth on the raw sample",
            "density_scale_factor_to_axis_units": float(scale),
            "splits": {},
        },
        "detection_limit_band": {
            "split": "unseen_drug",
            "n": int(dl["n"]),
            "quantity": "smallest departure from chance detectable at 80% power",
            "value": float(dl["detectable_at_80pc_power"]),
            "drawn_as": "grey band spanning 0.50 +/- 0.07898 on the absolute-NIR axis",
            "is_not": "a confidence interval",
            "nir_ci_half_width_same_arm": float(dl["half_width"]),
        },
        "verified_against_thesis_text": {
            "Sections/04c-reencoding.tex tab:generalisation": "all three rows of model NIR, both "
                "gap intervals, n and (lines; drugs; wells) agree to the printed 3-4 decimals",
            "Sections/04c-reencoding.tex sec:res-generalisation": "references 0.956 / 0.824 / "
                "0.836 and the 0.079 80%-power detection limit agree",
            "Sections/04d-baselines.tex, 05-Limitations.tex, 06-Conclusions.tex": "the same four "
                "intervals are quoted there and agree",
            "pooled": "0.5367 = the document's 0.537, and 0.5367 - 0.5 = the document's +0.0367",
            "disagreements_found": "none",
        },
        "caption_must_state": [
            "residual frame",
            "comparison set is other drugs in the same cell line",
            "n per split",
            "the reference is a within-well split-half precision figure, not a replicate",
            "both clustering axes are shown because the verdict depends on which is chosen",
            "the shaded band is the arm's 80%-power detection limit, not a confidence region",
        ],
    }
    for s in SPLITS:
        r = rows[s]
        drawn["panel_a"][s] = {k: r[k] for k in (
            "n", "n_lines", "n_drugs", "n_wells", "model_nir", "nir_ci", "gap",
            "gap_ci_line", "gap_ci_drug", "df_line", "df_drug", "reference")}
        drawn["panel_a"][s]["reference_note"] = ("raw within-well split-half precision reference "
                                                 "for this split; train is filtered at a "
                                                 "reliability line the held-out splits are not")
        drawn["panel_b"]["splits"][s] = {
            "n": r["n"],
            "mean": r["model_nir"],
            "median": r["median"],
            "sd": r["sd"],
            "frac_at_or_below_chance": r["frac_at_or_below_chance"],
            "pct_at_or_below_chance": round(100 * r["frac_at_or_below_chance"], 1),
            "bandwidth": float(bws[s]),
        }
    drawn["panel_b"]["pooled_mean_all_1394"] = float(
        np.concatenate([rows[s]["values"] for s in SPLITS]).mean())
    drawn["panel_b"]["pooled_n"] = int(sum(rows[s]["n"] for s in SPLITS))

    S.save(fig, "fig-splits", drawn)
    print("face in force:", family)
    for s in SPLITS:
        r = rows[s]
        print("%-13s NIR %.4f [%.4f, %.4f]  ref %.4f  gap %+.4f  line [%+.4f, %+.4f] df %d  "
              "drug [%+.4f, %+.4f] df %d  <=chance %.1f%%"
              % (s, r["model_nir"], r["nir_ci"][0], r["nir_ci"][1], r["reference"], r["gap"],
                 r["gap_ci_line"][0], r["gap_ci_line"][1], r["df_line"],
                 r["gap_ci_drug"][0], r["gap_ci_drug"][1], r["df_drug"],
                 100 * r["frac_at_or_below_chance"]))


if __name__ == "__main__":
    main()

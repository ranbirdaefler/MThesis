"""fig-mechanism (v2 rebuild): drug identity is encoded ever more explicitly with depth,
and the same slab carries almost none of the model's functional weight.

WHAT THIS FIGURE IS, AND WHAT WAS WRONG WITH THE v1 DRAWING
===========================================================================================
v1 (endcell/figures/fig_mechanism.py) put four series behind a four-row legend that sat on
top of the data, mixed two incommensurable quantities -- a probe accuracy and a subspace
fraction -- on one unlabelled 0-1 axis, drew layer as a categorical index so 9->12->16 was
compressed to the same step as 2->4, and drew panel (b) as BARS ON A LOG AXIS. That last one
is the serious error: on a log axis a bar's length is log(value) - log(axis floor), so the
"length" of the bar is set by wherever the axis happens to be cut, and the ratio of two bar
lengths is not the ratio of the two numbers. This rebuild:

  (a) splits the two quantities into two stacked sub-panels with separate, separately
      labelled scales, on a TRUE NUMERIC layer axis, with every series direct-labelled and
      no legend anywhere;
  (b) replaces the bars with paired dots joined by a stem (a dumbbell) on the same log axis.
      On a log axis a VERTICAL GAP is a RATIO, which is exactly the quantity the thesis
      quotes ("$3.6$--$18.6\\times$ the matched-dimension random null"), so each pair is
      annotated with its own ratio and the mark now means what the sentence means.

v1 ALSO DREW THE WRONG DATA IN PANEL (b). It plotted L[k]["swap"]["drug_b_kl"] against
L[k]["swap"]["random_inject_kl"] -- the drug-B INJECTION test -- while its own caption in
Sections/04b-representation.tex described "ablation of the purified drug subspace against a
matched-dimension random subspace", i.e. L[k]["kl"]["pure_drug"] against
L[k]["kl"]["random_matched"]. The caption, Table \\ref{tab:workspace} and the prose all
describe the ABLATION, and the ablation is what has the [mean, se, n=60] triples the brief
names. This script draws the ablation, and asserts its seven ratios reproduce the seven
ratios printed in Table \\ref{tab:workspace} before it will save.

NO INTERVALS ARE DRAWN. The kl entries are [mean, se, n=60] normal-approximation standard
errors over 60 prompts, not the thesis's one CI convention (95% bootstrap clustered over
cell lines). The thesis itself calls this column "a sign-and-magnitude reading and not an
estimate with an interval". Drawing a +/-0.003 whisker beside the thesis's genuine +/-0.03
intervals would invite a reader to distrust all of them. Bare markers; the caption says why.

Usage:  python thesis_v2/figs/mechanism.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

# The seven ratios printed in Table \ref{tab:workspace}, Sections/04b-representation.tex.
# The figure refuses to save if the JSON it reads does not reproduce them.
TABLE_RATIOS = {"2": 12.7, "4": 18.6, "6": 5.7, "8": 14.1, "9": 4.2, "12": 3.8, "16": 3.6}
# Decode columns as printed in the same table, raw and context-removed.
TABLE_DECODE_RAW = {"2": 0.408, "4": 0.585, "6": 0.692, "8": 0.779,
                    "9": 0.821, "12": 0.754, "16": 0.758}
TABLE_DECODE_CTX = {"2": 0.354, "4": 0.429, "6": 0.529, "8": 0.569,
                    "9": 0.602, "12": 0.873, "16": 0.883}


def mean(x):
    """kl entries are [mean, se, n] triples; shares and accuracies are bare floats."""
    return x[0] if isinstance(x, (list, tuple)) else x


def main():
    family = style.apply()
    print("font in force: %s" % style.font_report())

    P = style.load("workspace_probe.json")
    L = P["layers"]
    keys = sorted(L, key=int)
    x = np.array([int(k) for k in keys], dtype=float)   # TRUE numeric layer axis

    chance = P["chance"]
    n_classes = P["n_drug_classes"]
    n_kl = P["config"]["n_kl_prompts"]

    ctx = np.array([L[k]["drug_decode_context_removed"] for k in keys])
    raw = np.array([L[k]["drug_decode"] for k in keys])
    abl = np.array([L[k]["drug_decode_ablated"] for k in keys])
    gate = np.array([L[k]["frac_signal_removed"] for k in keys])
    vs_pure = np.array([L[k]["variance_share"]["drug_pure"] for k in keys])
    vs_cl = np.array([L[k]["variance_share"]["cell_line"] for k in keys])
    vs_rawdrug = np.array([L[k]["variance_share"]["drug"] for k in keys])

    kl_pure = np.array([mean(L[k]["kl"]["pure_drug"]) for k in keys])
    kl_rand = np.array([mean(L[k]["kl"]["random_matched"]) for k in keys])
    kl_cl = np.array([mean(L[k]["kl"]["cell_line"]) for k in keys])
    se_pure = np.array([L[k]["kl"]["pure_drug"][1] for k in keys])
    se_rand = np.array([L[k]["kl"]["random_matched"][1] for k in keys])
    n_pure = [L[k]["kl"]["pure_drug"][2] for k in keys]
    ratio = kl_pure / kl_rand

    # ---------------------------------------------------------------- guards on the numbers
    for i, k in enumerate(keys):
        assert round(ratio[i], 1) == TABLE_RATIOS[k], \
            "layer %s ratio %.3f disagrees with Table tab:workspace (%.1f)" % (k, ratio[i], TABLE_RATIOS[k])
        assert round(raw[i], 3) == TABLE_DECODE_RAW[k], "layer %s raw decode disagrees" % k
        assert round(ctx[i], 3) == TABLE_DECODE_CTX[k], "layer %s ctx decode disagrees" % k
        assert L[k]["kl"]["pure_drug"][2] == n_kl == 60, "n per KL entry is not 60 at layer %s" % k
    assert abs(chance - 1.0 / n_classes) < 1e-12 and n_classes == 12
    assert np.all(np.diff(ctx) > 0), "the thesis says context-removed decodability rises monotonically"
    assert gate.min() > 0.8, "the removal gate is quoted as passing at every layer"
    assert np.allclose(abl, abl[0]), "the thesis quotes a CONSTANT 0.121 ablated floor"
    # The panel (a) annotation prints "0.121" as a literal, and 04b-representation.tex says
    # "Accuracy falls from $0.41$--$0.82$ to a constant $0.121$ floor". The assert above only
    # pinned CONSTANCY, so the drawn label and the sentence could have drifted apart silently.
    assert round(float(abl[0]), 3) == 0.121, \
        "the ablated floor is %.4f but the figure label and 04b-representation.tex both say " \
        "0.121; do not change one without the other" % abl[0]

    drawn = {
        "source": "RESULTS_cluster/workspace_probe.json",
        "n_drug_classes": n_classes,
        "chance": chance,
        "n_per_kl_entry": n_pure,
        "intervals_drawn": False,
        "interval_reason": ("kl entries are [mean, se, n=60] normal-approximation SEs over "
                            "prompts, not the thesis's 95% bootstrap clustered over cell "
                            "lines; the text calls this a sign-and-magnitude reading and not "
                            "an estimate with an interval"),
        "layers": [int(k) for k in keys],
        "panel_a_top_probe_accuracy": {
            "drug_decode_context_removed": ctx.tolist(),
            "drug_decode_raw": raw.tolist(),
            "drug_decode_ablated": abl.tolist(),
            "chance": chance,
            "frac_signal_removed_gate": gate.tolist(),
        },
        "panel_a_bottom_descriptive_subspace_fraction": {
            "drug_purified": vs_pure.tolist(),
            "cell_line": vs_cl.tolist(),
            "drug_raw_not_drawn": vs_rawdrug.tolist(),
        },
        "panel_b_ablation_kl": {
            "pure_drug_mean": kl_pure.tolist(),
            "pure_drug_se_not_drawn": se_pure.tolist(),
            "random_matched_mean": kl_rand.tolist(),
            "random_matched_se_not_drawn": se_rand.tolist(),
            "ratio_pure_over_random": ratio.tolist(),
            "cell_line_positive_control_not_drawn": kl_cl.tolist(),
            "pure_over_cell_line": (kl_pure / kl_cl).tolist(),
        },
        "annotated": {
            "encoded_top": float(ctx[-1]),
            "encoded_bottom": float(vs_pure[-1]),
            "ratio_range": [float(ratio.min()), float(ratio.max())],
        },
        "font": family,
    }

    # ---------------------------------------------------------------- canvas
    # fullwidth, drawn at the final printed measure; never scaled in LaTeX.
    FW = style.FULLWIDTH_IN          # 6.8504 in = 174 mm
    FH = 3.95
    fig = plt.figure(figsize=(FW, FH))

    # Explicit axes rectangles. constrained/tight layout would resize the panels around the
    # out-of-axes direct labels, and the whole point of CHANGE 1 is that the canvas is exact.
    L_MARG, A_W, GUTTER = 0.52, 2.26, 0.68      # (a): ylabel+ticks, panel, out-of-axes labels
    B_MARG, B_W = 0.54, 2.68                    # (b): ylabel+ticks, panel
    BOT, TOP_PAD, VGAP = 0.42, 0.42, 0.24       # inches
    col_h = FH - BOT - TOP_PAD
    sub_h = (col_h - VGAP) / 2.0

    def rect(x0, y0, w, h):
        return [x0 / FW, y0 / FH, w / FW, h / FH]

    axA1 = fig.add_axes(rect(L_MARG, BOT + sub_h + VGAP, A_W, sub_h))   # accuracy
    axA2 = fig.add_axes(rect(L_MARG, BOT, A_W, sub_h))                  # subspace fraction
    bx0 = L_MARG + A_W + GUTTER + B_MARG
    axB = fig.add_axes(rect(bx0, BOT, B_W, col_h))
    assert bx0 + B_W < FW, "panel (b) runs off the 174 mm measure"

    XLIM = (0.6, 17.2)
    XT = [2, 4, 6, 8, 9, 12, 16]

    # ================================================================ (a) top: decodability
    axA1.axhline(chance, color=style.INK, lw=0.9, zorder=1)
    axA1.plot(XLIM, [abl[0], abl[0]], color=style.RULEGREY, lw=1.0, ls=(0, (4, 2.5)), zorder=1)
    axA1.plot(x, raw, ls=":", lw=1.2, marker="o", ms=3.6, mfc="white", mew=1.0,
              color=style.ACCENT, zorder=3)
    axA1.plot(x, ctx, ls="-", lw=1.4, marker="o", ms=4.0, color=style.ACCENT, zorder=4)

    # direct labels, no legend: the two curves cross once and are well separated elsewhere.
    axA1.annotate("raw probe", xy=(7.6, 0.905), ha="center", va="bottom",
                  fontsize=9, color=style.ACCENT)
    lbl_ctx = axA1.annotate("context removed", xy=(8.2, 0.44), ha="center", va="top",
                            fontsize=9, color=style.ACCENT)
    # The ablated-floor label rides just above its own dashed rule. It read "after ablating the
    # slab, ..." and that string sets 168pt at 9pt, inside a panel whose interior is 162.8pt: it
    # ran off the layer axis and the "encoded, not read" arrow at x=17.6 (x=202.5pt on the
    # canvas) struck through the word "layer". WRAPPING IT IS NOT THE FIX -- the rule sits at
    # y=0.121 and ctx reaches 0.354 by layer 2, so the free band is 24pt and a two-line block is
    # 19.5pt, which leaves ~3pt to the "context removed" label and to the ctx curve; that trades
    # one collision for two near-misses. "after" goes instead. The participle already carries the
    # tense, "the slab" is the load-bearing noun and is kept, and at 147pt the line clears the
    # arrow by 13pt with the whole 24pt band to itself. The guard below measures all of this.
    lbl_abl = axA1.annotate("ablating the slab, 0.121 at every layer",
                            xy=(1.1, abl[0] + 0.012), ha="left", va="bottom",
                            fontsize=9, color=style.QUIETINK)
    axA1.annotate("chance", xy=(17.95, chance), ha="left", va="center",
                  fontsize=9, color=style.INK, annotation_clip=False)

    axA1.set_ylim(0.0, 1.0)
    axA1.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axA1.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    axA1.set_ylabel("probe accuracy")
    axA1.set_xlim(*XLIM)
    axA1.set_xticks(XT)
    axA1.set_xticklabels([])
    axA1.set_title("(a) encoded more, used no more", loc="left", fontsize=10)

    # ================================================================ (a) bottom: causal share
    axA2.plot(x, vs_cl, ls="-", lw=1.3, marker="^", ms=4.2, color=style.QUIETINK, zorder=3)
    axA2.plot(x, vs_pure, ls="-", lw=1.4, marker="o", ms=4.0, color=style.ACCENT, zorder=4)
    axA2.annotate("cell line (positive control)", xy=(9.0, 0.70), ha="center", va="bottom",
                  fontsize=9, color=style.QUIETINK)
    axA2.annotate("purified drug slab", xy=(9.0, 0.115), ha="center", va="bottom",
                  fontsize=9, color=style.ACCENT)

    axA2.set_ylim(0.0, 0.85)
    axA2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axA2.set_yticklabels(["0", "0.20", "0.40", "0.60", "0.80"])
    axA2.set_ylabel("descriptive subspace fraction")
    axA2.set_xlim(*XLIM)
    axA2.set_xticks(XT)
    axA2.set_xticklabels([str(t) for t in XT])
    axA2.set_xlabel("layer")

    # ------------------------------------------------- the dissociation, drawn OUTSIDE the data
    # v1 put this arrow at layer 16 on top of the raw (dotted) series. It now lives in the
    # gutter to the right of both sub-panels, with two thin leaders showing what it spans.
    XB = 17.6
    axA1.plot([16, XB], [ctx[-1], ctx[-1]], lw=0.6, ls=(0, (1, 2)),
              color=style.QUIETINK, clip_on=False, zorder=2)
    axA2.plot([16, XB], [vs_pure[-1], vs_pure[-1]], lw=0.6, ls=(0, (1, 2)),
              color=style.QUIETINK, clip_on=False, zorder=2)
    arrow = fig.add_artist(ConnectionPatch(
        xyA=(XB, ctx[-1]), coordsA=axA1.transData,
        xyB=(XB, vs_pure[-1]), coordsB=axA2.transData,
        arrowstyle="<->", mutation_scale=8, lw=0.8, color=style.INK))
    # Text beside the arrow. NOT at the arrow's vertical middle: that is level with the chance
    # label, and the two would sit on each other in a 0.68in gutter. It goes high, beside the
    # arrow's upper end, which is the end the words "encoded" refers to.
    axA1.annotate("encoded,\nnot read", xy=(17.95, 0.50), ha="left", va="center",
                  fontsize=9, color=style.INK, annotation_clip=False)

    # ================================================================ (b) the ablation cost
    for xi, lo, hi, r in zip(x, kl_rand, kl_pure, ratio):
        axB.plot([xi, xi], [lo, hi], lw=0.8, color=style.RULEGREY, zorder=1,
                 solid_capstyle="butt")
        axB.annotate("%.1f$\\times$" % r, xy=(xi + 0.42, np.sqrt(lo * hi)), rotation=90,
                     ha="center", va="center", fontsize=9, color=style.QUIETINK)
    axB.plot(x, kl_rand, ls="none", marker="o", ms=4.0, mfc="white", mew=1.1,
             color=style.INK, zorder=3)
    axB.plot(x, kl_pure, ls="none", marker="o", ms=4.4, color=style.ACCENT, zorder=4)

    # direct labels with their own marker as the key glyph; both sit clear of every datum.
    axB.plot([1.15], [2.15], marker="o", ms=4.4, color=style.ACCENT, clip_on=False)
    axB.annotate("ablating the purified drug slab", xy=(1.55, 2.15), ha="left", va="center",
                 fontsize=9, color=style.ACCENT)
    axB.plot([1.15], [0.00285], marker="o", ms=4.0, mfc="white", mew=1.1, color=style.INK,
             clip_on=False)
    axB.annotate("matched-dimension random slab", xy=(1.55, 0.00285), ha="left", va="center",
                 fontsize=9, color=style.INK)

    axB.set_yscale("log")
    axB.set_ylim(0.0022, 3.4)
    axB.set_yticks([0.01, 0.1, 1.0])
    axB.set_yticklabels(["0.01", "0.10", "1.00"])
    axB.set_ylabel("KL from the clean forward pass")
    axB.set_xlim(0.6, 17.6)
    axB.set_xticks(XT)
    axB.set_xticklabels([str(t) for t in XT])
    axB.set_xlabel("layer")
    # "ratio = the vertical gap" is deliberately NOT written on the panel: it would have to sit
    # in the only free strip, which the lower direct label already occupies. The caption says it.
    axB.set_title("(b) ablation cost against a random null", loc="left", fontsize=10)

    # ---------------------------------------------------------------- guards on the layout
    # The numbers above are asserted against the table; the placement is asserted against the
    # canvas. The ablated-floor label previously overran panel (a) and the arrow at x=17.6 cut
    # its last word, which no check caught because nothing measured drawn text. This does.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    b_abl = lbl_abl.get_window_extent(renderer=rend)
    b_arrow = arrow.get_window_extent(renderer=rend)
    b_ctxlbl = lbl_ctx.get_window_extent(renderer=rend)
    pad = 4.0 * fig.dpi / 72.0                                  # 4pt of air, in device units
    assert b_abl.x1 < b_arrow.x0 - pad, \
        "the ablated-floor label reaches x=%.1fpt and the dissociation arrow starts at " \
        "%.1fpt: the arrow would strike through the text" \
        % (b_abl.x1 * 72.0 / fig.dpi, b_arrow.x0 * 72.0 / fig.dpi)
    assert b_abl.x1 <= axA1.get_window_extent(renderer=rend).x1, \
        "the ablated-floor label runs off the layer axis of panel (a)"
    assert b_abl.y1 < b_ctxlbl.y0 - pad or b_abl.x0 > b_ctxlbl.x1 or b_abl.x1 < b_ctxlbl.x0, \
        "the ablated-floor label collides with the 'context removed' direct label"

    style.save(fig, "fig-mechanism", drawn)

    print("  layer  raw    ctx    abl    gate   share_pure  share_cl  KL_pure  KL_rand  ratio")
    for i, k in enumerate(keys):
        print("  %5s  %.3f  %.3f  %.3f  %.3f   %.4f      %.4f    %.4f   %.4f   %.1fx"
              % (k, raw[i], ctx[i], abl[i], gate[i], vs_pure[i], vs_cl[i],
                 kl_pure[i], kl_rand[i], ratio[i]))


if __name__ == "__main__":
    main()

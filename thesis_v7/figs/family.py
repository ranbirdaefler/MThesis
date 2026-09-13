"""fig:family -- the 26-test family behind the identity result, at three rungs of the ladder.

WHY THIS FIGURE EXISTS
    sec:res-mechanism spends two pages discounting its own result, and the whole discount is an
    argument about a test family the reader has never been shown: "of twenty-six headline tests it
    is the only survivor of multiplicity correction", "applying the 26-test correction to the
    alpha = 1 slice leaves nothing surviving, and applying it to alpha = 2 promotes a different arm
    at a different depth". Those two sentences are the chapter's own reason for not leaning on this
    measurement, and until now the reader had to take them on trust. This draws the family.

WHAT IS DRAWN, AND WHY IT IS DRAWN THAT WAY
  * Three panels, one per stored rung (alpha = 0.5, 1, 2). Same grid, same colour scale, so the
    only thing that changes between panels is the rung -- which is exactly the claim.
  * 5 arms x 7 depths = 35 grid positions, of which only 26 were measured. The 9 that were never
    measured are drawn as EMPTY OUTLINES. Filling them with the colour of zero would assert 26
    nulls where there are 17 nulls and 9 non-measurements, and the surrounding prose is explicit
    that some depths are missing for reasons (the headroom gate, retained context) that are not
    "no effect".
  * The layer axis is ORDINAL. The pre-registered depths 2,4,6,8,9,12,16 are not evenly spaced;
    drawing them evenly and calling the axis "layer" would invite reading a depth trend off the
    spacing. The ticks carry the real layer numbers, the axis label says the spacing is ordinal,
    and the caption repeats it.
  * Colour is a symmetric diverging scale on `norm`, the effect as a fraction of that arm's own
    clean preference gap -- the only cross-arm-comparable form, because the five arms emit
    different target formats. Limits are symmetric about zero at +/- 0.07, which contains the
    largest value in all three slices (+0.0680) with no clipping. Most of the field is therefore
    near-white; that is the finding, not a defect of the scale.
  * SIGN IS ENCODED TWICE: in hue, and in a HATCH on every negative cell. A diverging scale is
    luminance-symmetric by construction -- it runs dark, through white, to dark -- so a greyscale
    print, a photocopy or a black-and-white e-reader collapses its two arms onto each other and a
    mid-grey cell becomes ambiguous between a strong negative and a strong positive. That is not
    hypothetical in this figure: the largest negative in the family (single cell, layer 16,
    alpha = 2, norm = -0.0190) and the headline positive (residual, layer 12, alpha = 0.5,
    norm = +0.0197) reduce to the SAME grey to three decimals. Sign and location of the surviving
    effect are the whole message here, so sign may not ride on hue alone.
    _worst_greyscale_collision() recomputes that pair on every build and save() writes it into
    family.json, so the claim above cannot rot silently if the numbers are ever refreshed.
    The hatch rule is UNCONDITIONAL -- every negative cell carries it, with no magnitude threshold
    -- so "unhatched" means "positive" everywhere on the grid and never "negative, but too small
    to have bothered marking". A threshold would have re-introduced exactly the ambiguity the
    hatch exists to remove, in the band just under it. The negative arm of the colourbar carries
    the same hatch, so the key is on the page and not only in the caption.
  * The other available fix -- swapping in a map whose two arms differ in luminance range as well
    as hue -- is REJECTED. It buys the sign by making a cell at -x print at a different darkness
    from a cell at +x, which corrupts the one thing the colour channel is here to carry (magnitude,
    comparably across sign) and leaves the colourbar's own two halves not comparable with each
    other. The hatch is a separate channel and leaves the magnitude mapping untouched.
    (v4: RdBu_r is replaced by a diverging map built from the document palette -- accent for
    positive, a luma-matched ink/quietink grey for negative -- so it stays luminance-symmetric.)
  * q_pooled is PRINTED, not encoded in opacity. Only three of the 78 cells have q < 0.10, so three
    numbers say more than a second visual channel would, and opacity on 78 cells would have fought
    the colour scale it sits on. The absence of any other printed number is itself the message.
  * The survivors (q_pooled < 0.05) are RINGED in black, outside the cell so the ring does not
    crowd the printed q. One ring at (residual, 12) in alpha = 0.5; none in alpha = 1; one at
    (residual-holdout, 4) in alpha = 2.
  * The strip beneath is the magnitude ladder at residual / layer 12 in RAW NATS, because the
    thesis quotes it in raw nats and because the 10% materiality margin is defined in those units.
    The margin is drawn as a vertical rule so the smallness of every rung against it is a distance
    on the page. Intervals are the cluster bootstrap over ordered (A,B) pairs, the study's only
    uncertainty model for this test; they are drawn rather than suppressed because whether a rung
    excludes zero is half of what the ladder is for.

NOT DRAWN, DELIBERATELY
  * The three seed replicates (+0.0197 / +0.0118 / +0.0085). They are a different question --
    does the anchor cell reproduce -- and the prose and tab:identity already carry them.
  * Any accuracy from probe_arm_*. Those are 24-class drug decodes at chance 0.0417 and must never
    share an axis with workspace_probe's 12-class numbers.

SOURCES
  RESULTS_cluster/probe_family_bh.json      -- slices.{stored_alpha_0.5, alpha_1.0, alpha_2.0}
  RESULTS_cluster/probe_arm_residual.json   -- layers.12.swap.ladder, .equiv_margin, cluster counts
"""
import json
import os
import sys

from matplotlib import colors
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- fixed design
# Row order is the spec's; residual on top because it is the arm the survivor first appears in.
ARMS = ["residual", "residual_holdout", "consensus", "ot_T2", "single_cell"]
# tab:trainconfig's own names for the five arms, shortened only where the gutter cannot hold them.
ARM_LABEL = {
    "residual":         "residual",
    "residual_holdout": "residual (holdout)",
    "consensus":        "consensus",
    "ot_T2":            "opt. transport (T2)",
    "single_cell":      "single cell",
}
# The pre-registered layer set, in order. Plotted at EQUAL spacing: the axis is ordinal.
LAYERS = [2, 4, 6, 8, 9, 12, 16]

SLICES = [("stored_alpha_0.5", r"$\alpha = 0.5$"),
          ("alpha_1.0",        r"$\alpha = 1$"),
          ("alpha_2.0",        r"$\alpha = 2$")]

VMAX = 0.07          # symmetric; max |norm| over all 78 cells is 0.0680, so nothing is clipped
Q_PRINT = 0.10       # print q_pooled below this
Q_SURVIVE = 0.05     # the pre-registered survivor criterion
# v4 palette: a diverging map built from the document's own inks, replacing RdBu_r. Positive arm
# ends in accent, centre is panelbg (as RdBu_r's centre was a pale grey, so a measured null never
# vanishes into the paper); negative arm ends in a warm grey on the ink -> quietink axis whose BT.601 luma is
# matched to accent's, so the two arms stay luminance-symmetric (the property the header relies on).
def _palette_diverging():
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    acc, ink, qi = to_rgb(S.ACCENT), to_rgb(S.INK), to_rgb(S.QUIETINK)
    t = (lum(acc) - lum(ink)) / (lum(qi) - lum(ink))
    neg = tuple(i + t * (q - i) for i, q in zip(ink, qi))
    return LinearSegmentedColormap.from_list("v4_diverging", [neg, to_rgb(S.PANELBG), acc], N=256)


CMAP = _palette_diverging()

# The redundant sign channel (see header). "//" lands two strokes across a cell this size: enough
# that a hatched cell is unmistakable in greyscale, light enough that it does not shout on the
# near-white cells, which are most of the grid and are the finding.
NEG_HATCH = "//"
HATCH_LW = 0.35      # points; matplotlib takes hatch weight from rcParams, not from the patch
GREY_TOL = 0.01      # luma gap under which two fills are "the same grey" on paper


def _grey(rgba):
    """BT.601 luma -- what a greyscale printer or a photocopier reduces this fill to."""
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _fill(norm):
    """The cell colour for an effect. One definition, shared by the grid and the greyscale check."""
    return CMAP((norm + VMAX) / (2 * VMAX))


def _txt_ink(rgba):
    """Black or white in-cell ink -- text or hatch -- whichever the cell's own fill can carry."""
    return "white" if _grey(rgba) < 0.55 else S.INK


def _worst_greyscale_collision(cells):
    """The opposite-signed pair of cells a greyscale printer renders most nearly identically.

    This is the measured form of the reason the hatch exists. Of all opposite-signed pairs whose
    fills fall within GREY_TOL luma of each other, it returns the pair with the largest magnitudes
    -- near-zero cells collide trivially and say nothing, the point is that LARGE effects of
    opposite sign collide too. Recomputed on every build and written to family.json, so if a data
    refresh ever separated the two arms in greyscale, the record would say so rather than the
    header quietly going stale.
    """
    items = [(k, a, l, t["norm"]) for (k, a, l), t in cells.items()]
    best = None
    for i, (ki, ai, li, ni) in enumerate(items):
        for kj, aj, lj, nj in items[i + 1:]:
            if (ni < 0) == (nj < 0):
                continue
            gap = abs(_grey(_fill(ni)) - _grey(_fill(nj)))
            if gap > GREY_TOL:
                continue
            score = min(abs(ni), abs(nj))
            if best is None or score > best["_score"]:
                best = {"_score": score, "luma_gap": gap,
                        "cells": [{"slice": k, "arm": a, "layer": l, "norm": n,
                                   "luma": _grey(_fill(n))}
                                  for k, a, l, n in ((ki, ai, li, ni), (kj, aj, lj, nj))]}
    if best is not None:
        best.pop("_score")
    return best


def _verify_against_thesis(cells, survivors, ladder, margin):
    """Every number this figure draws that the prose of sec:res-mechanism also quotes.

    Invariant 4 of the brief: a figure may not disagree with the text. The check is mechanical and
    it runs on every build, because the alternative is discovering the disagreement in a viva.
    Left-hand side: 04b-representation.tex. Right-hand side: what this script actually drew.
    """
    bad = []
    checked = [0]

    def eq(what, drawn, quoted, tol):
        checked[0] += 1
        if abs(drawn - quoted) > tol:
            bad.append("%s: figure draws %.6g, 04b-representation.tex says %.6g" %
                       (what, drawn, quoted))

    h = cells[("stored_alpha_0.5", "residual", 12)]                # the headline
    eq("headline effect (l.407 \\ci{+0.0197})", h["norm"], 0.0197, 5e-5)
    eq("headline CI lo (+0.0063)", h["ci_norm"][0], 0.0063, 5e-5)
    eq("headline CI hi (+0.0335)", h["ci_norm"][1], 0.0335, 5e-5)
    eq("headline p (l.500 p = 0.001)", h["p"], 0.001, 1e-9)
    eq("headline q (l.497 q = 0.026)", h["q_pooled"], 0.026, 5e-4)

    a1 = cells[("alpha_1.0", "residual", 12)]                      # l.508 the alpha=1 rung
    eq("alpha=1 effect (+0.0265)", a1["norm"], 0.0265, 5e-5)
    eq("alpha=1 CI lo (+0.0021)", a1["ci_norm"][0], 0.0021, 5e-5)
    eq("alpha=1 CI hi (+0.0524)", a1["ci_norm"][1], 0.0524, 5e-5)
    eq("alpha=1 p (l.509 p = 0.033)", a1["p"], 0.033, 1e-9)

    c2 = cells[("stored_alpha_0.5", "consensus", 2)]               # l.501 the runner-up
    eq("consensus layer 2 p (0.012)", c2["p"], 0.012, 1e-9)
    eq("consensus layer 2 q (0.156)", c2["q_pooled"], 0.156, 5e-4)

    rh = cells[("alpha_2.0", "residual_holdout", 4)]               # l.519 the alpha=2 promotion
    eq("alpha=2 residual-holdout layer 4 q (0.007)", round(rh["q_pooled"], 3), 0.007, 1e-9)

    rh12 = cells[("stored_alpha_0.5", "residual_holdout", 12)]     # l.591 the sibling arm
    eq("residual-holdout layer 12 at the headline rung (+0.0028)", rh12["norm"], 0.0028, 5e-5)

    # l.511: "In raw nats the rungs read +0.0013, +0.0017, +0.0044, +0.0061, +0.0046"
    for r, quoted in zip(ladder, [0.0013, 0.0017, 0.0044, 0.0061, 0.0046]):
        eq("ladder rung alpha=%g, raw nats" % r["alpha"], r["diff_perm"], quoted, 5e-5)
    # l.510: "Each of alpha = 0.5, 1 and 2 excludes zero; alpha = 5 and 10 span it"
    for r, should_exclude in zip(ladder, [True, True, True, False, False]):
        checked[0] += 1
        if (r["ci_perm"][0] > 0.0) is not should_exclude:
            bad.append("ladder alpha=%g: CI %r contradicts l.510" % (r["alpha"], r["ci_perm"]))
    # l.479, l.408: the margin is 10% of the arm's own clean preference gap
    eq("materiality margin", margin, 0.10 * cells[("stored_alpha_0.5", "residual", 12)]
       ["clean_ab_gap"], 1e-12)

    # l.495-l.502 and l.517-l.519: which slice yields which survivor
    expected = {"stored_alpha_0.5": [("residual", 12)],
                "alpha_1.0": [],
                "alpha_2.0": [("residual_holdout", 4)]}
    for k, v in expected.items():
        checked[0] += 1
        if sorted(survivors[k]) != sorted(v):
            bad.append("survivors of %s: figure has %r, text says %r" % (k, survivors[k], v))

    if bad:
        raise SystemExit("STOP -- figure and thesis text disagree:\n  " + "\n  ".join(bad))
    print("verified against 04b-representation.tex: %d quoted quantities agree" % checked[0])


def main():
    family = S.apply()
    # matplotlib takes hatch weight from the rcParams, never from the patch; set it once, after
    # apply(), so the negative-cell strokes are fine rather than the 1.0pt default.
    plt.rcParams["hatch.linewidth"] = HATCH_LW
    bh = S.load("probe_family_bh.json")
    resid = S.load("probe_arm_residual.json")
    swap12 = resid["layers"]["12"]["swap"]

    # ---------------------------------------------------------------- assemble the 78 cells
    cells = {}          # (slice_key, arm, layer) -> test dict
    survivors = {}
    for key, _ in SLICES:
        sl = bh["slices"][key]
        assert sl["n_tests"] == 26, "expected 26 tests in %s, found %d" % (key, sl["n_tests"])
        for t in sl["tests"]:
            cells[(key, t["arm"], t["layer"])] = t
        survivors[key] = [(s["arm"], s["layer"]) for s in sl["survivors_q_lt_0.05"]]
        # the survivor list and the q<0.05 cells must be the same set, or the file disagrees
        recomputed = sorted((t["arm"], t["layer"]) for t in sl["tests"]
                            if t["q_pooled"] < Q_SURVIVE)
        assert sorted(survivors[key]) == recomputed, \
            "%s: stored survivors %r != cells with q_pooled < %.2f %r" % (
                key, sorted(survivors[key]), Q_SURVIVE, recomputed)

    measured = sorted({(a, l) for (_, a, l) in cells})
    assert len(measured) == 26, "expected 26 measured (arm, layer) positions, got %d" % len(measured)
    allnorm = [t["norm"] for t in cells.values()]
    assert max(abs(min(allnorm)), abs(max(allnorm))) <= VMAX, \
        "colour scale clips: max |norm| = %.4f > VMAX = %.3f" % (
            max(abs(min(allnorm)), abs(max(allnorm))), VMAX)

    ladder = swap12["ladder"]
    margin = swap12["equiv_margin"]                 # 10% of the arm's clean preference gap

    _verify_against_thesis(cells, survivors, ladder, margin)

    # why the hatch exists, measured off this build's own numbers rather than asserted
    collision = _worst_greyscale_collision(cells)
    n_negative = sum(1 for t in cells.values() if t["norm"] < 0)
    n_hatched = [0]

    # ---------------------------------------------------------------- canvas
    # Full-width measure, drawn at final size. No LaTeX scaling, ever.
    W = S.FULLWIDTH_IN
    H = 4.28
    fig = plt.figure(figsize=(W, H))

    L = 1.18            # left gutter: holds "opt. transport (T2)" / "residual (holdout)" at 9pt
    R = 0.05
    GAP = 0.15
    HEADER = 0.52       # panel title (10pt) + two-line direct label (8.5pt) above each grid
    PANEL_W = (W - L - R - 2 * GAP) / 3.0
    CELL = PANEL_W / len(LAYERS)
    PANEL_H = CELL * len(ARMS)          # square cells
    PANEL_Y0 = (H - 0.06 - HEADER - PANEL_H) / H

    panel_axes = []
    for i, (key, title) in enumerate(SLICES):
        ax = fig.add_axes([(L + i * (PANEL_W + GAP)) / W, PANEL_Y0, PANEL_W / W, PANEL_H / H])
        panel_axes.append(ax)
        ax.set_xlim(-0.5, len(LAYERS) - 0.5)
        ax.set_ylim(len(ARMS) - 0.5, -0.5)          # row 0 at the top
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([str(l) for l in LAYERS], fontsize=8.5)
        ax.tick_params(axis="both", length=0, pad=2.5)
        if i == 0:
            ax.set_yticks(range(len(ARMS)))
            ax.set_yticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
        else:
            ax.set_yticks([])

        for r, arm in enumerate(ARMS):
            for c, lay in enumerate(LAYERS):
                t = cells.get((key, arm, lay))
                if t is None:
                    # never measured -- an outline, not a zero
                    ax.add_patch(Rectangle((c - 0.46, r - 0.46), 0.92, 0.92, facecolor="none",
                                           edgecolor=S.RULEGREY, lw=0.5, linestyle=(0, (1.2, 1.2)),
                                           zorder=2))
                    continue
                rgba = _fill(t["norm"])
                ax.add_patch(Rectangle((c - 0.46, r - 0.46), 0.92, 0.92, facecolor=rgba,
                                       edgecolor="white", lw=0.4, zorder=2))
                if t["norm"] < 0:
                    # the redundant sign channel. Unconditional: every negative cell, no threshold.
                    # lw=0 so this patch adds no border of its own -- matplotlib still draws the
                    # hatch, in `edgecolor`, at rcParams["hatch.linewidth"]. The ink is picked from
                    # the cell's own fill so the strokes survive however dark the fill gets.
                    n_hatched[0] += 1
                    ax.add_patch(Rectangle((c - 0.46, r - 0.46), 0.92, 0.92, facecolor="none",
                                           edgecolor=_txt_ink(rgba), lw=0.0, hatch=NEG_HATCH,
                                           zorder=3))
                if t["q_pooled"] < Q_PRINT:
                    ax.text(c, r, ("%.3f" % t["q_pooled"]).lstrip("0"), ha="center", va="center",
                            fontsize=7.5, color=_txt_ink(rgba), zorder=4)
                if (arm, lay) in survivors[key]:
                    # ring sits OUTSIDE the cell so it does not crowd the printed q
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor="none",
                                           edgecolor=S.INK, lw=1.6, zorder=5))

        # title clears the two-line direct label below it: 2 lines at 8.5pt / 1.25 = ~22pt
        ax.set_title(title, fontsize=10, pad=26, color=S.INK)
        n_s = len(survivors[key])
        head = "one survivor" if n_s == 1 else ("none survive" if n_s == 0 else "%d survivors" % n_s)
        where = ", ".join("%s, layer %d" % (ARM_LABEL[a], l) for a, l in survivors[key]) or "—"
        ax.annotate(head + "\n" + where, xy=(0.5, 1.008), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=8.5, color=S.QUIETINK, linespacing=1.25)

    # one ordinal-axis label for the whole row, centred under it
    fig.text((L + (3 * PANEL_W + 2 * GAP) / 2.0) / W, (H - 0.06 - HEADER - PANEL_H - 0.34) / H,
             "layer, in the pre-registered order — axis is ordinal, not proportional to depth",
             ha="center", va="baseline", fontsize=S.APPARATUS_PT, family="sans-serif",
             color=S.INK)   # v4: the row's shared axis label, so apparatus sans like the others

    # ---------------------------------------------------------------- shared colour scale
    CB_W, CB_H = 2.60, 0.085
    cax = fig.add_axes([(W - CB_W) / 2.0 / W, 1.80 / H, CB_W / W, CB_H / H])
    cb = ColorbarBase(cax, cmap=CMAP, norm=colors.Normalize(-VMAX, VMAX), orientation="horizontal")

    # The key for the sign channel: the negative ARM of the bar wears the same hatch the negative
    # CELLS do, so the rule is legible from the figure alone and does not depend on the caption.
    # The ink flips where the ramp goes too dark to carry a dark stroke -- the same per-fill choice
    # _txt_ink() makes in the cells -- so the hatch stays visible across the whole negative arm.
    _ramp = [-VMAX + i * (VMAX / 400.0) for i in range(401)]          # -VMAX .. 0
    _switch = next((x for x in _ramp if _grey(_fill(x)) >= 0.55), 0.0)
    for _x0, _x1 in ((-VMAX, _switch), (_switch, 0.0)):
        if _x1 <= _x0:
            continue
        _f0, _f1 = (_x0 + VMAX) / (2 * VMAX), (_x1 + VMAX) / (2 * VMAX)
        cax.add_patch(Rectangle((_f0, 0.0), _f1 - _f0, 1.0, transform=cax.transAxes,
                                facecolor="none", edgecolor=_txt_ink(_fill((_x0 + _x1) / 2.0)),
                                lw=0.0, hatch=NEG_HATCH, zorder=3, clip_on=True))

    cb.set_ticks([-0.06, -0.03, 0.0, 0.03, 0.06])
    cb.set_ticklabels(["−0.06", "−0.03", "0", "+0.03", "+0.06"])
    cax.tick_params(labelsize=8.5, length=2.5, width=0.7, pad=2, color=S.RULEGREY,
                    labelcolor=S.INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(S.RULEGREY)
    # the sign rule is stated in words as well as shown on the bar: hue alone cannot carry it
    cb.set_label("swap − permuted, as a fraction of that arm's own clean preference gap; "
                 "negative cells are hatched",
                 fontsize=9, color=S.INK, labelpad=4)

    # ---------------------------------------------------------------- the ladder strip
    LAD_H = 0.85
    lax = fig.add_axes([L / W, 0.50 / H, (3 * PANEL_W + 2 * GAP) / W, LAD_H / H])
    lax.spines["left"].set_visible(False)
    lax.spines["bottom"].set_color(S.RULEGREY)
    lax.tick_params(axis="y", length=0, pad=4)
    lax.tick_params(axis="x", labelsize=9)

    ys = list(range(len(ladder)))
    lax.axvline(0.0, color=S.RULEGREY, lw=0.7, zorder=1)
    lax.axvline(margin, color=S.INK, lw=0.9, zorder=1)   # the pre-registered materiality margin

    for y, rung in zip(ys, ladder):
        lo, hi = rung["ci_perm"]
        S.whisker(lax, None, y, lo, hi, orientation="h", color=S.ACCENT, lw=1.1, zorder=3)
        lax.plot([rung["diff_perm"]], [y], marker="o", ms=4.2, color=S.ACCENT,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=4)

    lax.set_ylim(len(ladder) - 0.5, -0.90)
    lax.set_yticks(ys)
    lax.set_yticklabels([r"$\alpha = %s$" % ("%g" % r["alpha"]) for r in ladder], fontsize=9)
    lax.annotate("10%% materiality margin (%.4f nats)" % margin,
                 xy=(margin, -0.85), xycoords=("data", "data"), xytext=(3, 0),
                 textcoords="offset points", ha="left", va="top", fontsize=9, color=S.INK)
    lax.set_xlabel("swap − permuted preference shift, raw nats — residual arm, layer 12",
                   fontsize=9, labelpad=4)

    # ---------------------------------------------------------------- sibling JSON
    drawn = {
        "figure": "fig:family",
        "what": ("the 26-test family of the identity probe at three rungs of the displacement "
                 "ladder, plus the alpha ladder at residual / layer 12 in raw nats"),
        "sources": {
            "grid": "RESULTS_cluster/probe_family_bh.json",
            "ladder": "RESULTS_cluster/probe_arm_residual.json :: layers.12.swap.ladder",
            "margin": "RESULTS_cluster/probe_arm_residual.json :: layers.12.swap.equiv_margin",
        },
        "grid": {
            "arms_top_to_bottom": ARMS,
            "layers_left_to_right": LAYERS,
            "axis_note": "layer axis is ordinal; the seven depths are not evenly spaced",
            "n_positions": len(ARMS) * len(LAYERS),
            "n_measured_positions": len(measured),
            "n_absent_positions": len(ARMS) * len(LAYERS) - len(measured),
            "absent_positions": [[a, l] for a in ARMS for l in LAYERS if (a, l) not in measured],
            "colour": {"field": "norm", "cmap": "v4_diverging (luma-matched grey / panelbg / accent)", "vmin": -VMAX, "vmax": VMAX,
                       "symmetric_about": 0.0,
                       "max_abs_norm_in_data": max(abs(v) for v in allnorm),
                       "clipped": False},
            "sign_encoding": {
                "channel": "hatch, redundant with hue",
                "rule": "hatched if and only if norm < 0 -- unconditional, no magnitude threshold",
                "pattern": NEG_HATCH,
                "hatch_linewidth_pt": HATCH_LW,
                "n_negative_cells": n_negative,
                "n_hatched_cells": n_hatched[0],
                "colourbar": "the negative arm of the shared bar carries the same hatch",
                "why": ("a diverging scale is luminance-symmetric, so in greyscale its two arms "
                        "collapse and the sign of a mid-grey cell is lost; the magnitude mapping "
                        "is left untouched rather than skewing the two arms' luminance ranges"),
                "greyscale_collision": collision,
                "greyscale_collision_test": (
                    "over all opposite-signed pairs of drawn cells, the pair with the largest "
                    "magnitudes whose BT.601 luma differs by <= %.3f; null if the two arms are "
                    "already separable in greyscale" % GREY_TOL),
            },
            "q_printed_below": Q_PRINT,
            "survivor_criterion": "q_pooled < %.2f (BH, pooled across arms within a slice)" % (
                Q_SURVIVE),
        },
        "cells": [
            {"slice": key, "alpha": (0.5 if key == "stored_alpha_0.5" else float(key.split("_")[1])),
             "arm": arm, "layer": lay,
             "norm": cells[(key, arm, lay)]["norm"],
             "ci_norm": cells[(key, arm, lay)]["ci_norm"],
             "p": cells[(key, arm, lay)]["p"],
             "q_pooled": cells[(key, arm, lay)]["q_pooled"],
             "clean_ab_gap": cells[(key, arm, lay)]["clean_ab_gap"],
             "printed_q": cells[(key, arm, lay)]["q_pooled"] < Q_PRINT,
             "hatched_negative": cells[(key, arm, lay)]["norm"] < 0,
             "ringed_survivor": (arm, lay) in survivors[key]}
            for key, _ in SLICES for arm in ARMS for lay in LAYERS if (key, arm, lay) in cells
        ],
        "survivors": {k: [{"arm": a, "layer": l} for a, l in survivors[k]] for k, _ in SLICES},
        "n_tests_per_slice": {k: bh["slices"][k]["n_tests"] for k, _ in SLICES},
        "ladder": {
            "arm": "residual", "layer": 12, "units": "raw nats (log-prob contrast, per-token mean)",
            "clean_ab_gap": swap12["clean_ab_gap"],
            "materiality_margin_nats": margin,
            "margin_definition": "10% of the arm's own clean preference gap (config.equiv_frac=0.1)",
            "rungs": [{"alpha": r["alpha"], "diff_perm_nats": r["diff_perm"],
                       "ci_perm_nats": r["ci_perm"], "p_perm": r["p_perm"],
                       "excludes_zero": r["ci_perm"][0] > 0.0} for r in ladder],
            "note": ("the alpha = 0.5 rung is the headline effect restated in raw nats, not an "
                     "independent point; its interval and p here are the ladder's own bootstrap "
                     "draw (p = %.3f), while the headline q = 0.026 rests on the file's other "
                     "stored draw (p = %.3f)" % (ladder[0]["p_perm"], swap12["p_perm"])),
        },
        "uncertainty": {
            "model": "cluster bootstrap over ordered (A,B) pairs, 4000 resamples, p floored at 1/4000",
            "rows_per_test": 60,
            "pair_clusters_range_over_the_26_tests": [55, 60],
            "pair_clusters_at_residual_layer12": swap12["n_pair_clusters"],
            "anchor_drug_clusters": swap12["n_drug_clusters"],
        },
        "not_drawn": ["the three seed replicates (+0.0197 / +0.0118 / +0.0085), which are in the "
                      "prose and tab:identity",
                      "any probe_arm_* accuracy (24 classes, chance 0.0417) -- must not share an "
                      "axis with workspace_probe's 12-class numbers"],
        "font_resolved": family,
        "figure_size_in": [W, H],
    }

    # the sign channel must be exhaustive, or "unhatched" would not mean "positive"
    assert n_hatched[0] == n_negative, \
        "sign channel is not exhaustive: %d cells have norm < 0 but %d were hatched" % (
            n_negative, n_hatched[0])

    S.save(fig, "family", drawn)
    print("font: %s" % S.font_report())
    print("cells drawn: %d (26 x 3 slices)" % len(drawn["cells"]))
    print("negative cells hatched: %d of %d" % (n_hatched[0], n_negative))
    if collision is None:
        print("greyscale: no opposite-signed pair within %.3f luma" % GREY_TOL)
    else:
        a, b = collision["cells"]
        print("greyscale: %+0.4f and %+0.4f both print at luma %.3f (gap %.4f) -- hence the hatch"
              % (a["norm"], b["norm"], a["luma"], collision["luma_gap"]))
    for k, _ in SLICES:
        print("  %-18s survivors: %s" % (k, survivors[k] or "none"))


if __name__ == "__main__":
    main()

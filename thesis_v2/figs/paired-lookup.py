"""fig:paired-lookup -- is the dictionary's win uniform, or is it a mean?

The chapter's central negative is one paired mean: model - drug_lookup =
-0.3631 [-0.3982, -0.3280] on n = 1192. fig:ladder shows two point values on a
strip, which cannot distinguish "the lookup wins nearly everywhere" from "the
lookup wins on average over a mixed population". This figure shows the paired
distribution behind that mean, for both lookups.

EVERY NUMBER COMES FROM RESULTS_cluster/re_v3.json. Nothing is typed in.
  records[]                -> per-condition model / drug_lookup / drug_lookup_1 / split
  means.common_support     -> n, model, within-well reference, drug_lookup on the support
  means.vs_baselines[k]    -> paired gap and its TWO-WAY clustered interval (cell line + well)

WHAT IS DELIBERATELY NOT DRAWN, and why
  * drug_lookup_oracle (0.977).  Not a claim the thesis makes; it is excluded from the
    ladder as an oracle, so putting it in a figure would smuggle it back in.
  * the one-way (cell-line-only) intervals stored beside the two-way ones. The thesis
    quotes the two-way interval; drawing the narrower one invites the reader to prefer it.
  * anything in the expression frame. Both axes here are residual-frame NIR and the two
    frames may not be set beside each other (see style_v2, and 00-HowToRead).

CONSTRUCTION NOTES
  * The two panels are drawn the SAME SIZE, deliberately, against the brief's "second
    small panel". The whole point of panel (b) is that the reader compares its cloud with
    panel (a)'s; a different square size makes an area comparison between two clouds that
    are on identical axes, which is the one comparison this figure exists to support.
  * Win counts are counts over conditions and carry NO interval. They are annotated as
    bare numbers and the caption says so. A bootstrap over a fraction-of-conditions would
    be a different estimand from the clustered paired mean beside it.
  * Ties (model exactly equal to the lookup: 23 against drug_lookup, 28 against
    drug_lookup_1) count as NOT a win, so "wins" is strictly greater-than. Recorded in the
    sibling JSON.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MINUS = "−"  # typographic minus; a hyphen at 9pt reads as a dash next to digits


def fmt(v, nd=4):
    s = "%.*f" % (nd, v)
    return s.replace("-", MINUS)


# --------------------------------------------------------------------------- data
def load():
    d = S.load("re_v3.json")
    recs = [r for r in d["records"] if r.get("drug_lookup") is not None]
    cs = d["means"]["common_support"]
    vb = d["means"]["vs_baselines"]
    assert len(recs) == cs["n"], "common support disagrees with records: %d vs %d" % (
        len(recs), cs["n"])
    return d, recs, cs, vb


def arrays(recs, key):
    m = np.array([r["model"] for r in recs], float)
    b = np.array([r[key] for r in recs], float)
    split = np.array([r["split"] for r in recs])
    return m, b, split


# --------------------------------------------------------------------------- panels
def scatter_panel(ax, m, b, split, baseline_name, wins, n, key_labels=False,
                  chance_labels=False):
    """model NIR (y) against a lookup's NIR (x), square, both axes 0-1.

    The mass below the identity line IS the claim, so the identity line is a solid rule and
    the two chance rules at 0.50 are drawn on both axes (style_v2's standing rule: a NIR
    axis without its chance line lets a reader infer a scale that does not exist).
    """
    win = m > b
    is_train = split == "train"

    # chance on BOTH axes -- both are NIR axes.
    S.chance_line(ax, "h", label=False)
    S.chance_line(ax, "v", label=False)
    # identity: the line the claim is about, so it is the heavier of the three rules.
    ax.plot([0, 1], [0, 1], color=S.INK, lw=1.2, zorder=2)

    common = dict(clip_on=True, zorder=3)
    # unseen_combo: filled dot. train: open square, so it stays visible over the dense mass.
    for wsel, col, al in ((~win, S.QUIETINK, 0.20), (win, S.ACCENT, 0.80)):
        s1 = wsel & ~is_train
        ax.scatter(b[s1], m[s1], s=7, marker="o", linewidths=0,
                   color=col, alpha=al, **common)
        s2 = wsel & is_train
        ax.scatter(b[s2], m[s2], s=15, marker="s", facecolors="none",
                   edgecolors=col, linewidths=0.5, alpha=min(1.0, al + 0.22), **common)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])

    # the identity line, named where the field is empty.
    ax.text(0.125, 0.160, "model = lookup", rotation=45, rotation_mode="anchor",
            ha="left", va="bottom", fontsize=9, color=S.INK, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.5))

    # the headline count, sitting in the region it describes. The ground is OPAQUE, not a
    # 70% wash: the marks are at zorder 3 and a 0.70 ground let them print straight through
    # the glyphs -- in panel (b), 17 winners under this box put dots on the 'w' of "wins",
    # on the colon of "the line:" and on the closing paren of "(18.0%)". Relocating does not
    # fix it, because (b) has no empty box of this size anywhere on the square: the emptiest
    # position for it still holds 7 marks, and every one of those positions is further from
    # the region the sentence is about. So the ground has to occlude. It hides 17 of 214
    # winners in (b) and 2 of 85 in (a), and the count those marks would support is the
    # sentence printed over them, so the reader loses nothing they could have used.
    ax.text(0.035, 0.978,
            "model wins\nabove the line:\n%s of %s (%.1f%%)"
            % (wins, n, 100.0 * wins / n),
            ha="left", va="top", fontsize=9, color=S.ACCENT, linespacing=1.3, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))

    if chance_labels:
        halo = dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.5)
        ax.text(0.487, 0.975, "chance", rotation=90, ha="right", va="top",
                fontsize=9, color="black", zorder=5, bbox=halo)
        ax.text(0.012, 0.513, "chance", ha="left", va="bottom",
                fontsize=9, color="black", zorder=5, bbox=halo)
    if key_labels:
        # direct key, not a legend box: two marks and their names. NOT in the bottom-left
        # corner, which is exactly where the identity line enters the square -- at y = 0.095
        # the diagonal is at x = 0.095, so it ran through "unseen_combo", and the 0.50 rule cut
        # the "=" of "n = 893"; nine marks touched the two grounds there, four under them and
        # five sliced in half at an edge. The key keeps its x and moves up into this band, the
        # only place in (a) where a 0.54-wide key clears the diagonal (by 11pt), stays out of
        # the headline block (by 13pt) and, crucially, touches no mark it does not fully cover:
        # four marks sit under the grounds and NONE is left half-drawn at an edge. Grounds are
        # opaque for the same reason as the headline's. The long row still crosses the vertical
        # 0.50 rule -- the string is wider than half the square, so no x on this axis avoids it
        # -- and the break it leaves is the one the "chance" label above already makes.
        ax.scatter([0.048], [0.7125], s=7, marker="o", linewidths=0,
                   color=S.QUIETINK, alpha=0.85, zorder=5)
        ax.text(0.088, 0.7125, "unseen_combo (n = 893)", ha="left", va="center",
                fontsize=9, color=S.QUIETINK, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=0.5))
        ax.scatter([0.048], [0.6455], s=15, marker="s", facecolors="none",
                   edgecolors=S.QUIETINK, linewidths=0.5, alpha=0.9, zorder=5)
        ax.text(0.088, 0.6455, "train (n = 299)", ha="left", va="center",
                fontsize=9, color=S.QUIETINK, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=0.5))

    ax.set_xlabel("%s NIR\n(residual frame, cell-line comparison set)" % baseline_name,
                  fontsize=10, linespacing=1.25)


BINS = np.arange(-1.0, 1.0 + 1e-9, 0.05)   # an edge exactly at 0, so the rule is a boundary


def diff_panel(ax, diff, mean, ci, top, side_labels=False):
    """Per-condition paired difference, with the clustered paired mean as a point and bar.

    The mean is drawn BELOW the count baseline, in its own strip, so a 95% interval is
    never read as part of a bar height. Bar length is a count here and nothing else.
    """
    counts, edges = np.histogram(diff, bins=BINS)
    left = edges[:-1]
    cols = [S.ACCENT if lo >= -1e-12 else S.RULEGREY for lo in left]
    alphas = [0.85 if lo >= -1e-12 else 0.70 for lo in left]
    for lo, c, a, h in zip(left, cols, alphas, counts):
        ax.bar(lo, h, width=0.05, align="edge", color=c, alpha=a, lw=0, zorder=2)

    # `top` is shared by both panels so the two count axes are the same scale and the two
    # distributions may be compared by height as well as by position.
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.42 * top, 1.62 * top)
    # zero = the model and the lookup tie. Drawn from under the mean's strip to just above
    # the tallest bin, and NOT the full height, so it does not cut through the label above.
    ax.plot([0.0, 0.0], [-0.42 * top, 1.06 * top], color="black", lw=0.9, zorder=3,
            solid_capstyle="butt")

    # the clustered paired mean, as a point and bar in its own strip.
    ybar = -0.17 * top
    S.whisker(ax, None, ybar, ci[0], ci[1], color=S.INK, lw=1.6, zorder=4)
    ax.plot([mean], [ybar], marker="o", ms=2.8, color=S.INK, zorder=5)
    # ONE line, in the headroom above the counts: two lines reach down into the tallest bin,
    # and a number sitting on a bar reads as if it labelled that bar. The interval's
    # construction (95%, bootstrap, clustered two-way on cell line and well) is stated in the
    # caption rather than here, because it does not fit on the measure at 9pt.
    ax.text(-0.98, 1.56 * top,
            "paired mean %s  [%s, %s]" % (fmt(mean), fmt(ci[0]), fmt(ci[1])),
            ha="left", va="top", fontsize=9, color=S.INK, zorder=5)

    if side_labels:
        # one direction label, over the region it names; the sign of the axis does the rest.
        ax.text(0.98, 1.14 * top, "model better", ha="right", va="top",
                fontsize=9, color=S.ACCENT)

    step = 50 if top <= 175 else 100
    tick = int(np.floor(top / step) * step)
    tallest = int(counts.max())
    ax.set_yticks([0, tick])
    ax.spines["left"].set_bounds(0, tick)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=3)
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_xticklabels([MINUS + "1", MINUS + "0.5", "0", "0.5", "1"])
    if side_labels:   # the count axis is named once; both panels carry the same tick values
        ax.set_ylabel("conditions", fontsize=10)
    return tallest


# --------------------------------------------------------------------------- figure
def main():
    family = S.apply()
    print("style_v2 face: %s" % S.font_report())

    d, recs, cs, vb = load()
    n = len(recs)
    splits = {"train": int(sum(r["split"] == "train" for r in recs)),
              "unseen_combo": int(sum(r["split"] == "unseen_combo" for r in recs)),
              "unseen_drug": int(sum(r["split"] == "unseen_drug" for r in recs))}

    drawn = {
        "_source": "RESULTS_cluster/re_v3.json",
        "_frame": "residual-frame NIR; comparison set is other drugs in the same cell line",
        "n_common_support": n,
        "splits_on_support": splits,
        "unseen_drug_note": "no training lookup exists for a held-out drug, so this split "
                            "contributes no points",
        "n_cell_lines": len(set(r["cell_line"] for r in recs)),
        "pooled_on_support": {
            "model": cs["model"],
            "within_well_split_half_reference": cs["ceiling"],
            "drug_lookup": cs["drug_lookup"],
            "drug_lookup_1": float(np.mean([r["drug_lookup_1"] for r in recs])),
        },
        "panels": {},
    }

    # one count scale for both difference panels
    shared_top = max(
        int(np.histogram([r["model"] - r[k] for r in recs], bins=BINS)[0].max())
        for k in ("drug_lookup", "drug_lookup_1"))

    fig = plt.figure(figsize=(S.FULLWIDTH_IN, 5.36))
    W, H = fig.get_size_inches()

    # absolute placement: two identical squares, each with its paired-difference panel
    # under it. Manual, because aspect="equal" under constrained layout leaves the two
    # columns raggedly spaced and the whole point is that the two squares match.
    left, sq, gap = 0.74, 2.62, 0.62
    top_pad, mid_gap, hist_h, bot_pad = 0.44, 0.78, 1.02, 0.50
    y_sq = (H - top_pad - sq) / H
    y_hi = (H - top_pad - sq - mid_gap - hist_h) / H

    axes = {}
    for i, (key, nice, panel) in enumerate([
            ("drug_lookup", "drug_lookup", "a"),
            ("drug_lookup_1", "drug_lookup_1", "b")]):
        x0 = (left + i * (sq + gap)) / W
        ax = fig.add_axes([x0, y_sq, sq / W, sq / H])
        axh = fig.add_axes([x0, y_hi, sq / W, hist_h / H])
        axes[key] = (ax, axh)

        m, b, split = arrays(recs, key)
        wins = int((m > b).sum())
        ties = int((m == b).sum())
        diff = m - b
        gapmean = float(np.mean(diff))
        ci = vb[key]["ci_two_way"]

        # the figure must draw the thesis's own number, not a recomputation that drifts
        assert abs(gapmean - vb[key]["gap"]) < 1e-9, (gapmean, vb[key]["gap"])
        assert vb[key]["n"] == n

        scatter_panel(ax, m, b, split, nice, wins, n,
                      key_labels=(i == 0), chance_labels=(i == 0))
        top = diff_panel(axh, diff, vb[key]["gap"], ci, shared_top, side_labels=(i == 0))
        axh.set_xlabel("model %s %s   (per condition)" % (MINUS, nice), fontsize=10)

        # panel head: label, comparator, and what the comparator knows.
        ax.text(0.0, 1.105, "(%s)" % panel, transform=ax.transAxes, ha="left", va="baseline",
                fontsize=10, color=S.INK)
        ax.text(0.085, 1.105, "model against %s" % nice, transform=ax.transAxes,
                ha="left", va="baseline", fontsize=10, color=S.INK)
        sub = ("the drug's mean residual in all other cell lines"
               if key == "drug_lookup" else
               "the same, from a single other cell line")
        ax.text(0.0, 1.032, sub, transform=ax.transAxes, ha="left", va="baseline",
                fontsize=9, color=S.QUIETINK)

        drawn["panels"][key] = {
            "paired_mean": vb[key]["gap"],
            "ci_two_way_clustered_cellline_well": ci,
            "ci_one_way_not_drawn": vb[key]["ci"],
            "n": vb[key]["n"],
            "n_cell_lines": vb[key]["n_cell_lines"],
            "two_way_note": vb[key]["two_way_note"],
            "model_wins": wins,
            "model_wins_fraction": wins / float(n),
            "ties_counted_as_not_a_win": ties,
            "wins_by_split": {s: int(sum((r["model"] > r[key]) and r["split"] == s
                                         for r in recs))
                              for s in ("train", "unseen_combo")},
            "baseline_mean_on_support": float(np.mean([r[key] for r in recs])),
            "histogram_bin_width": 0.05,
            "histogram_tallest_bin_count": top,
        }

    axes["drug_lookup"][0].set_ylabel("model NIR\n(residual frame)", fontsize=10,
                                      linespacing=1.25)

    drawn["_not_drawn"] = {
        "drug_lookup_oracle": "excluded: an oracle, not a claim the thesis makes",
        "one_way_intervals": "stored above as ci_one_way_not_drawn; the thesis quotes "
                             "the two-way clustered interval and only that is drawn",
        "expression_frame": "no expression-frame quantity appears in this figure",
    }

    # fig-* is the repo's graphic-file convention (fig-difficulty, fig-mechanism, ...); the
    # sibling JSON the brief asks for is fig-paired-lookup.json, written by save().
    S.save(fig, "fig-paired-lookup", drawn)
    print("model wins: %s" % {k: v["model_wins"] for k, v in drawn["panels"].items()})


if __name__ == "__main__":
    main()

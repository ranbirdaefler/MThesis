"""fig-repair (v3): two panels, one budget each, all three arms in both.

WHY TWO PANELS. v2 plotted three series at 600 tokens (the only budget shared by
all three checkpoints at the time) and overlaid the residual arm's own 1400-token
run as a dotted extra series on the SAME axis. That was a deliberate choice to
avoid comparing an arm at one budget against controls at another. Both controls
have since been re-run at 1400 tokens too, so a genuine same-budget comparison is
possible at 1400 as well as at 600, and drawing them side by side removes the
dotted-overlay compromise entirely: panel (a) is 1400 tokens for all three,
panel (b) is 600 tokens for all three, and nothing on either panel mixes budgets.

THE RESIDUAL SERIES IN BOTH PANELS IS THE SUPERSEDED, PRE-REPAIR CHECKPOINT. It
is the only residual checkpoint measured at 600 tokens at all, so it is the only
one that can appear in panel (b); it is kept in panel (a) too so the two panels
show the same three checkpoints at two budgets, not three checkpoints at one
budget and a different fourth thing at the other. It is not the checkpoint any
other number in \\S4.9's prose comes from.

THE X-AXIS IS CATEGORICAL, NOT A SHARED CONTINUOUS COSINE SCALE, AND THIS IS A
DELIBERATE CHANGE FROM v2. v2's exact-cosine positions relied on all four series
being computed from one identical 250-condition partner pool (verified
byte-identical mean_cos across all four v2 source files). That identity holds at
600 tokens: re_residual_model.json, re_ot_model.json and re_singlecell_model.json
share mean_cos = [0.4062, -0.0016, -0.3286] over the same 250 conditions, exactly
as v2 documented. It does NOT hold at 1400 tokens. re_residual_maxtok.json (the
superseded checkpoint) re-uses that same 250-condition pool, but re_ot_1400.json
and re_singlecell_1400.json score a DIFFERENT, smaller 248-condition pool (after
inventory filters) with substantially different partner geometry: mean_cos =
[0.4030, +0.0355, -0.0094]. The opposite stratum is the one that moves most --
-0.329 in the 600-token pool and in the residual arm's own 1400-token pool,
against only -0.009 in the controls' 1400-token pool, i.e. the "opposite" partner
available to the controls at 1400 tokens is barely anti-correlated at all. This
is a real difference in what was measured, not a plotting inconvenience, so it is
never plotted as though the three series in panel (a) share one continuous axis.
Exact mean cosines per series are written to the sidecar JSON and stated in the
caption instead. THIS IS A DELIBERATE HONESTY CHOICE: forcing all three onto one
shared cosine axis at 1400 would visually imply a shared difficulty ladder that
the data does not support.

Usage:  python repair_two_panel.py   (run from thesis_v4/figs/, writes fig-repair.pdf/.json here)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

STRATA = ("near", "orth", "opposite")
XPOS = [0, 1, 2]  # categorical: stratum RANK, not measured cosine (see module docstring)

# (file, label, colour, marker, token budget)
PANEL_1400 = [
    ("re_residual_maxtok.json", "residual model (superseded checkpoint)", style.ACCENT, "o"),
    ("re_ot_1400.json", "optimal-transport control", style.QUIETINK, "D"),
    ("re_singlecell_1400.json", "single-cell control", style.INK, "s"),
]
PANEL_600 = [
    ("re_residual_model.json", "residual model (superseded checkpoint)", style.ACCENT, "o"),
    ("re_ot_model.json", "optimal-transport control", style.QUIETINK, "D"),
    ("re_singlecell_model.json", "single-cell control", style.INK, "s"),
]
NUDGE = {"residual model (superseded checkpoint)": -0.06, "optimal-transport control": 0.0,
         "single-cell control": +0.06}


def series(fname):
    """Prefer the two-way (cell line + well) clustered interval where the source file has
    one -- currently only re_ot_1400.json and re_singlecell_1400.json -- so the plotted
    whisker matches the number quoted in prose. Falls back to the one-way (cell-line-only)
    interval for every other file, which has no two-way variant computed at all."""
    d = style.load(fname)
    s = d["means"]["strata"]
    cos = [s[k]["mean_cos"] for k in STRATA]
    has_2way = all("ci_two_way" in s[k] for k in STRATA)
    ci_key = "ci_two_way" if has_2way else "ci"
    return {
        "file": fname,
        "max_new_tokens": d["config"]["max_new_tokens"],
        "n_conditions_config": d["config"]["n_conditions"],
        "mean_cos": cos,
        "gap": [s[k]["gap"] for k in STRATA],
        "ci_lo": [s[k][ci_key][0] for k in STRATA],
        "ci_hi": [s[k][ci_key][1] for k in STRATA],
        "n": [s[k]["n"] for k in STRATA],
        "ci_clustering": "cell line and well (two-way)" if has_2way else "cell line only (one-way)",
    }


def signed(v, _pos=None):
    if abs(v) < 1e-12:
        return "0"
    return ("%+.2f" % v).replace("-", "−")


def draw_panel(ax, runs, budget, show_legend):
    ax.axhline(0.0, color=style.INK, lw=0.9, zorder=1)
    data = []
    for fname, label, colour, marker in runs:
        d = series(fname)
        assert d["max_new_tokens"] == budget, \
            "%s ran at %s tokens, expected %s" % (fname, d["max_new_tokens"], budget)
        data.append(d)
        dx = NUDGE[label]
        x = [xi + dx for xi in XPOS]
        for xi, lo, hi in zip(x, d["ci_lo"], d["ci_hi"]):
            style.whisker(ax, xi, None, lo, hi, orientation="v",
                          color=colour, lw=1.0, alpha=0.75, zorder=2)
        ax.plot(x, d["gap"], marker=marker, ms=4.5, mew=1.1, color=colour, ls="-", lw=1.1,
                label=label, zorder=3, mfc=colour)
    ax.xaxis.set_major_locator(FixedLocator(XPOS))
    ax.set_xticklabels(list(STRATA))
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.072, 0.212)
    ax.yaxis.set_major_locator(FixedLocator([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20]))
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.set_title("%d tokens" % budget, fontsize=10, color=style.INK, loc="left")
    if show_legend:
        ax.legend(loc="upper left", fontsize=8, borderaxespad=0.3, handlelength=2.2,
                  labelspacing=0.3, borderpad=0.0)
    return data


def main():
    style.apply()
    print("style_v2 face: %s" % style.font_report())

    fig, (ax1400, ax600) = plt.subplots(
        1, 2, figsize=(style.TEXTWIDTH_IN, 3.10), layout="constrained", sharey=True)

    data_1400 = draw_panel(ax1400, PANEL_1400, 1400, show_legend=True)
    data_600 = draw_panel(ax600, PANEL_600, 600, show_legend=False)

    fig.supxlabel("swap-partner stratum", fontsize=9, color=style.INK)
    ax1400.set_ylabel("$\\Delta_{\\mathrm{scr}}$  =  model $-$ scramble\n(NIR, residual frame)")

    drawn = {
        "figure": "fig-repair",
        "quantity": "Delta_scr = NIR(model) - NIR(stratum-matched scramble), residual frame",
        "x_variable": "categorical stratum rank (near=0, orth=1, opposite=2); NOT the measured "
                      "cosine, which differs across series at 1400 tokens -- see per-point "
                      "mean_cos below and the module docstring",
        "interval": "95% bootstrap, clustered over cell lines",
        "panels": {"1400_tokens": {}, "600_tokens": {}},
    }
    for panel_key, runs, data in (("1400_tokens", PANEL_1400, data_1400),
                                   ("600_tokens", PANEL_600, data_600)):
        for (fname, label, _, _), d in zip(runs, data):
            drawn["panels"][panel_key][label] = d

    style.save(fig, "fig-repair", drawn)

    for panel_key, runs, data in (("1400 tokens", PANEL_1400, data_1400),
                                   ("600 tokens", PANEL_600, data_600)):
        print("\n=== %s ===" % panel_key)
        print("  %-42s %s" % ("series", "  ".join("%-28s" % s for s in STRATA)))
        for (fname, label, _, _), d in zip(runs, data):
            print("  %-42s " % label
                  + "  ".join("%+.4f[%+.4f,%+.4f]" % (g, lo, hi)
                              for g, lo, hi in zip(d["gap"], d["ci_lo"], d["ci_hi"])))
            print("  %-42s mean_cos=%s n=%s" % ("", d["mean_cos"], d["n"]))
    return fig


if __name__ == "__main__":
    main()

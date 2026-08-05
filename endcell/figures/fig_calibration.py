"""fig-calibration: the standard metric is exploitable, and only one of five survives calibration.

This figure is an exhibit of METRIC failure. It must never read as a statement about model quality
-- DE-dr is the metric the thesis exists to discredit, and a panel that let a reader infer "the
model scores 0.73, which is close to the ceiling" would launder exactly the artifact being exposed.
Hence the ordering: the zero-information baseline sits at the TOP.

(a) The exploit. `revert_center` places every gene at the middle rank, carries no information about
    drug, cell or biology, and scores 0.9999 -- above the two-real-cell noise ceiling and above the
    model. The ordering is inverted: less information yields a higher score.

    No intervals are drawn here. The stored ones are normal-approximation over 400 examples
    (24,979 pairs for the ceiling), not clustered over cell lines like every other interval in the
    thesis, and a half-width of 0.0008 sitting beside the thesis's genuine +/-0.03 intervals would
    invite a reader to distrust all of them.

(b) The diagnosis. DRF is drawn on its own normalised scale, where 0 is the drug-agnostic baseline
    and 1 is a perfect predictor, so the plotted displacement IS the DRF. Raw m_neg -> m_pos gaps
    would NOT preserve the ordering -- on a raw axis panel_tau looks most inverted while DRF ranks
    de_delta worst -- so the figure would visually contradict its own number.

    weighted_r2 carries no directional tick: it reads -0.081 here and +0.032 at ~121 cells, so its
    sign tracks cell count and the thesis softens it to "approximately neutral".

Usage:  python endcell/figures/fig_calibration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

TIER = "tier2_unseen_drugs"
CONV = "worst"
METRIC_LABEL = {"nir": "NIR", "weighted_r2": "weighted $R^2$", "panel_tau": "panel-$\\tau$",
                "spearman_expr": "Spearman", "de_delta": "DE-$\\Delta r$"}


def main():
    style.apply()

    lin = style.load("eval_endcell_linear.json")["linear"][CONV][TIER]
    mod = style.load("eval_endcell_model.json")
    cpu = style.load("eval_endcell_cpu.json")

    de = lambda blk: blk["de_pearson_k50"]["mean"]
    ceiling = cpu["ceiling"][TIER][CONV]["cell_vs_cell_de50"]["mean"]
    model = mod["model"][TIER]["conventions"][CONV]["de_pearson_k50"]["mean"]
    try:
        scr = mod["scramble"][TIER]["conventions"][CONV]["de_pearson_k50"]["mean"]
    except (KeyError, TypeError):
        scr = None

    rows = [("mid-rank baseline\n(zero information)", de(lin["revert_center"]), style.OKABE["black"]),
            ("mean-control baseline\n(no fit)", de(lin["revert_mean"]), style.OKABE["grey"]),
            ("ridge linear\n(control only)", de(lin["ridge"]), style.OKABE["green"]),
            ("model\n(control + drug)", model, style.OKABE["blue"])]
    if scr is not None:
        rows.append(("scramble\n(wrong drug)", scr, style.OKABE["orange"]))

    # CANONICAL same-plate arm. drf_sameplate.json is a superseded run (655 drugs, 61 cell lines,
    # NIR DRF +0.446) and panel (b) was still drawing it while the caption quoted the current
    # figures -- a figure disagreeing with its own caption. The canonical within-plate arm is the
    # wide, cell-line-clustered rerun: 586 drugs over 44 cell-line x plate groups spanning 27 lines.
    drf = style.load("calibration_v5_sameplate_wide.json")
    ap = style.load("calibration_v4.json")          # all-plates arm, cited in the caption
    order = ["nir", "weighted_r2", "panel_tau", "spearman_expr", "de_delta"]
    dd = drf["drf"]["neg_mean"]

    drawn = {"tier": TIER, "convention": CONV,
             "panel_a": {"ceiling_cell_vs_cell": ceiling,
                         "values": {r[0].replace("\\", ""): r[1] for r in rows}},
             "panel_b": {m: {k: dd[m][k] for k in ("drf", "m_neg", "m_pos", "m_perfect")}
                         for m in order},
             "drf_n_drugs": drf["drf"]["neg_mean"]["nir"]["n"],
             "drf_n_celllines": drf.get("n_celllines"),
             "drf_n_groups": drf.get("n_groups"),
             "drf_source": "calibration_v5_sameplate_wide.json",
             # The caption contrasts panel (b) with the ALL-PLATES arm, which is not plotted.
             # Record it here so the figure's data file documents every number its caption asserts.
             "drf_all_plates_reference": {
                 "_source": "calibration_v4.json",
                 "n_drugs": ap.get("n_drugs"), "n_celllines": ap.get("n_celllines"),
                 **{m: {"drf": ap["drf"]["neg_mean"][m]["drf"],
                        "ci": ap["drf"]["neg_mean"][m].get("ci")}
                    for m in order}}}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(style.TEXTWIDTH_IN, 2.6),
                                   gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.95))

    # ---------------- (a) the exploit
    ys = range(len(rows))
    axL.axvspan(ceiling - 0.004, ceiling + 0.004, color=style.OKABE["grey"], alpha=0.30, lw=0)
    axL.annotate("noise ceiling", xy=(ceiling, -0.60), rotation=90,
                 ha="left", va="bottom", fontsize=7.4, color=style.OKABE["grey"])
    for y, (lab, val, col) in zip(ys, rows):
        axL.plot([val], [y], marker="o", ms=6, color=col, zorder=3)
        axL.annotate("%.4f" % val, xy=(val, y), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=7.5, color=col)
    axL.set_yticks(list(ys))
    axL.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    axL.invert_yaxis()
    axL.set_ylim(len(rows) - 0.4, -0.7)
    axL.set_xlim(0.70, 1.06)
    axL.set_xlabel("DE-$\\Delta r$ (K=50), tier 2")
    axL.set_title("(a) less information, higher score", fontsize=9.5, loc="left")

    # ---------------- (b) the diagnosis
    yb = range(len(order))
    axR.axvline(0.0, color="black", lw=0.9)
    for y, m in zip(yb, order):
        v = dd[m]["drf"]
        col = style.OKABE["blue"] if v > 0.05 else (
            style.OKABE["grey"] if abs(v) <= 0.05 else style.OKABE["vermilion"])
        axR.annotate("", xy=(v, y), xytext=(0, y),
                     arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col,
                                     shrinkA=0, shrinkB=0, mutation_scale=9))
        axR.annotate("%+.3f" % v, xy=(v, y), xytext=(9 if v > 0 else -9, 0),
                     textcoords="offset points", va="center",
                     ha="left" if v > 0 else "right", fontsize=7.5, color=col)
    axR.set_yticks(list(yb))
    axR.set_yticklabels([METRIC_LABEL[m] for m in order], fontsize=8.5)
    axR.invert_yaxis()
    axR.set_xlim(-1.02, 0.78)
    axR.set_ylim(len(order) - 0.4, -0.9)
    axR.set_xticks([-0.5, 0.0, 0.5])
    axR.set_xlabel("dynamic range fraction")
    axR.set_title("(b) only one metric is calibrated", fontsize=9.5, loc="left")
    axR.annotate("inverted $\\longleftarrow$", xy=(-0.10, -0.72), fontsize=7.4,
                 color=style.OKABE["vermilion"], ha="right", va="center")
    axR.annotate("$\\longrightarrow$ calibrated", xy=(0.10, -0.72), fontsize=7.4,
                 color=style.OKABE["blue"], ha="left", va="center")

    style.save(fig, "fig-calibration", drawn)
    print("  ceiling (cell vs cell, tier2) = %.4f" % ceiling)
    for lab, val, _ in rows:
        print("  %-34s %.5f" % (lab.replace("\\", ""), val))
    print("  DRF:", {m: round(dd[m]["drf"], 3) for m in order})


if __name__ == "__main__":
    main()

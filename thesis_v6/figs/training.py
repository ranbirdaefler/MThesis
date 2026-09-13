"""fig-training (v2 rebuild) -- undertraining is not the trivial explanation.

    python thesis_v2/figs/training.py

Writes thesis_v2/figs/fig-training.pdf and fig-training.json. Nothing under thesis/ is touched:
style_v2.save() resolves its destination from dirname(__file__) and asserts it.

WHAT THE FIGURE SAYS
--------------------
(a) Three one-epoch training arms, each in its own sub-panel with its own y-axis. Every one of
    them flattens well before its scheduled horizon ends, including inside the 50-70% band where
    the cosine schedule still has real step size left.
(b) A separate ten-epoch residual schedule. Held-out cross-entropy is BEST at epoch 1 (1.9199)
    and worsens monotonically thereafter to 2.2568, while training loss keeps falling to 1.4485.
    The canonical one-epoch run's validation loss on the same 3,600-example file (1.8986) is a
    level the ten-epoch run never reaches at any epoch.

    Together these exclude ordinary undertraining as a TRIVIAL explanation. They do not exclude
    it as a general one, and they say nothing about NIR -- see `interpretation_bounds` in the
    source artifact, reproduced verbatim into the sibling JSON and into the caption.

WHAT CHANGED FROM v1 endcell/figures/fig_training_curves.py, AND WHY
--------------------------------------------------------------------
(1) SCALE. This was the worst-scaled figure in the document. v1 drew at 10.2 x 4.0in and saved
    bbox_inches="tight"; the resulting PDF is 726.9pt wide and LaTeX included it at 404.03pt, a
    scale of 0.556. Its 7.5pt annotations therefore printed at 4.15pt and its 10pt panel titles
    at 5.54pt. Drawn here at FULLWIDTH_IN (174mm = 6.8504in) for \\begin{fullwidth} and saved at
    the exact canvas, so \\includegraphics places it at scale 1.0 and 9pt prints as 9pt.

(2) FACE. v1 set nothing, so this figure came out in DejaVu Sans while the other four v2 figures
    were in DejaVu Serif -- the five were not even consistent with each other. style_v2 puts all
    five in the document's own TeX Gyre Pagella.

(3) THE BROKEN ANNOTATION. v1's panel A carried the literal string "flat here at 22--52\\%\\nof
    peak learning rate". matplotlib is not TeX: the "\\%" printed as a backslash followed by a
    percent sign and the "--" printed as two hyphens, not an en dash. The annotation is gone
    entirely (see change 5); the band it described is stated in the caption instead.

(4) PANEL (a) NO LONGER INVITES THE COMPARISON ITS CAPTION FORBIDS. v1 drew all three arms on one
    shared y-axis, so "the residual curve sits above the other two" was the panel's loudest visual
    message -- while the caption said absolute losses are NOT comparable between arms because the
    three arms' target strings differ. Each arm now has its own sub-panel and its own y-axis.
    There is no shared scale left to over-read, and the within-arm shape, which IS the evidence,
    is what survives.

(5) NO TWINNED AXIS. v1 twinned a second y-axis onto panel A to carry the residual arm's learning
    rate as a fraction of peak. That curve made no claim of its own; it existed only to justify
    the shaded band. A decorative series does not earn an axis. The band stays -- it is the region
    the appendix calls load-bearing -- and the learning-rate fact travels in the caption, where it
    is a sentence rather than a second scale a reader has to hold.

(6) PANEL LETTERS. v1 used bold "A" and "B" in axes titles; every other multi-panel figure in this
    document uses "(a)" and "(b)". Matched, and demoted from title to letter: v1's titles
    ("Canonical short-run arms", "Longer residual schedule") described, which is the caption's job.

(7) ONE GRID TREATMENT. v1 gave panel A no grid and panel B a light grid, in the same figure. Both
    panels now carry the same faint horizontal rules at their own y-ticks, and nothing else.

(8) THE CROSSING IS MARKED AS AN INTERVAL, NOT A POINT. Training loss is above validation at
    epoch 1 and below it at epoch 2, so the crossing lies inside that epoch -- but WHERE inside it
    is not logged. Drawing a marker at the intersection of the two straight drawn segments would
    put an invented number (epoch 1.071, loss 1.930) on the page dressed as a measurement. The
    epoch is shaded instead, which states exactly what the two logged points support.

NUMBERS AND PROVENANCE
----------------------
Everything comes from RESULTS_cluster/training_adequacy_long.json. Nothing is typed in.
  panel (a): curve_series.<arm>.points[*].{step, cumulative_train_loss} and .nominal_total_steps
  panel (b): long_run.epochs[*].{epoch, train_loss, eval_loss} and canonical_one_epoch.eval_loss

DE-AVERAGING. The trainer logs a CUMULATIVE epoch mean, which flattens by arithmetic whatever the
model is doing. Every panel (a) series is de-averaged to its per-window value with the same
identity v1 used, (s_i*L_i - s_{i-1}*L_{i-1}) / (s_i - s_{i-1}), and the step sequence is asserted
strictly increasing before that division is trusted.

INTERVALS. None, and none are wanted. These are logged cross-entropies from single runs, not
estimates over a resampled population; a whisker here would have nothing to be an interval OF.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ARTIFACT = "training_adequacy_long.json"

# Canonical arm FIRST: it is the arm every behavioural result in this thesis is scored on, so it
# is the one a reader should meet first. The other two are context.
ORDER = ("retrain_614269", "ot_train_600856", "arm1a_consensus_592483")

# Short display names. The artifact's own `label` strings are kept in the sibling JSON verbatim;
# these are the two-line forms that fit a 3.1in sub-panel. "epsilon = 0.5" becomes real math.
NAMES = {
    "retrain_614269":         ("residual, the arm the results use", "%s scheduled steps"),
    "ot_train_600856":        ("optimal transport, $\\varepsilon = 0.5$", "%s scheduled steps"),
    "arm1a_consensus_592483": ("consensus", "%s scheduled steps, stopped at $61.5\\%%$"),
}

BAND = (0.50, 0.70)   # the fraction-of-horizon window the appendix calls load-bearing


def thousands(n):
    return format(int(n), ",")


def deaverage(row):
    """Cumulative epoch mean -> per-window loss. Fails closed on a non-increasing step sequence."""
    pts = row["points"]
    steps = [p["step"] for p in pts]
    cum = [p["cumulative_train_loss"] for p in pts]
    inst = [cum[0]]
    for i in range(1, len(steps)):
        d = steps[i] - steps[i - 1]
        if d <= 0:
            raise ValueError("curve steps must be strictly increasing (%s)" % row.get("label"))
        inst.append((steps[i] * cum[i] - steps[i - 1] * cum[i - 1]) / d)
    total = row["nominal_total_steps"]
    frac = [s / total for s in steps]
    return frac, inst, [p["learning_rate"] for p in pts], total


def window_mean(frac, vals, lo, hi):
    v = [x for f, x in zip(frac, vals) if lo < f <= hi]
    return (sum(v) / len(v)) if v else None


def main():
    style.apply()
    print("figure face: %s" % style.font_report())
    d = style.load(ARTIFACT)

    # ------------------------------------------------------------------ panel (a) numbers
    arms = {}
    for stem in ORDER:
        row = d["curve_series"][stem]
        frac, inst, lr, total = deaverage(row)
        peak = max(lr)
        if peak <= 0:
            raise ValueError("no positive learning rate logged for %s" % stem)
        inband = [(f, l / peak) for f, l in zip(frac, lr) if BAND[0] <= f <= BAND[1]]
        arms[stem] = {
            "label_in_artifact": row["label"],
            "canonical": row["canonical"],
            "completed": row["completed"],
            "nominal_total_steps": total,
            "n_logged_points": len(frac),
            "first_logged_step": row["points"][0]["step"],
            "last_logged_step": row["points"][-1]["step"],
            "fraction_of_horizon_reached": frac[-1],
            "deaveraged_first": inst[0],
            "deaveraged_last": inst[-1],
            "deaveraged_min": min(inst),
            "deaveraged_max": max(inst),
            "band_mean_45_55pct": window_mean(frac, inst, 0.45, 0.55),
            "band_mean_65_75pct": window_mean(frac, inst, 0.65, 0.75),
            "learning_rate_peak": peak,
            "learning_rate_last": lr[-1],
            "lr_fraction_of_peak_in_band": (
                [min(x[1] for x in inband), max(x[1] for x in inband)] if inband else None),
            "source": row["source"],
            "_frac": frac, "_inst": inst,      # popped before the JSON is written
        }

    # ------------------------------------------------------------------ panel (b) numbers
    lr_ = d["long_run"]
    rows = lr_["epochs"]
    epochs = [r["epoch"] for r in rows]
    train = [r["train_loss"] for r in rows]
    valid = [r["eval_loss"] for r in rows]
    canon = d["canonical_one_epoch"]["eval_loss"]
    best_ep, best_val = lr_["best_eval_epoch"], lr_["best_eval_loss"]

    # unit tests: the summary fields and the per-epoch table must agree with each other, and the
    # claim the figure is drawn to make must actually hold in the numbers.
    assert len(rows) == 10, "expected ten logged epochs, got %d" % len(rows)
    assert valid[best_ep - 1] == best_val == min(valid), "best_eval_loss is not the minimum"
    assert valid[-1] == lr_["final_eval_loss"] and train[-1] == lr_["final_train_loss"]
    assert min(valid) > canon, "the ten-epoch run reaches the canonical one-epoch validation loss"
    assert train[0] > valid[0] and train[1] < valid[1], "no crossing inside epoch 1->2"
    print("unit tests OK: epoch table reproduces long_run summary; min(valid)=%.4f > canonical "
          "%.4f; curves cross inside epoch 1->2" % (min(valid), canon))

    # ------------------------------------------------------------------ the drawn record
    drawn = {
        "figure": "fig-training",
        "source_file": "RESULTS_cluster/" + ARTIFACT,
        "source_logs": d["sources"],
        "no_intervals": (
            "None drawn and none available. Every value here is a logged cross-entropy from a "
            "single run, not an estimate over a resampled population, so there is nothing for an "
            "interval to be an interval of."),
        "panel_a": {
            "quantity": "training loss, de-averaged from the logged cumulative epoch mean",
            "deaveraging": "(s_i*L_i - s_{i-1}*L_{i-1}) / (s_i - s_{i-1})",
            "x_axis": "fraction of the arm's own nominal one-epoch horizon",
            "y_axes": "one per arm, independent -- absolute losses are NOT comparable between "
                      "arms because the three arms' target strings differ",
            "shaded_band": {"fraction_of_horizon": list(BAND),
                            "why": "the window the appendix calls load-bearing: the schedule "
                                   "still has real step size here and the loss has already flat"
                                   "tened"},
            "arms": {k: arms[k] for k in ORDER},
        },
        "panel_b": {
            "run_name": lr_["run_name"],
            "quantity": "epoch-mean cross-entropy",
            "full_epochs": lr_["full_epochs"],
            "nominal_scheduler_horizon": lr_["nominal_scheduler_horizon"],
            "actual_optimizer_updates": lr_["actual_optimizer_updates"],
            "actual_updates_per_epoch": lr_["actual_updates_per_epoch"],
            "validation_data": lr_["validation_data"],
            "epochs": [dict(epoch=r["epoch"], train_loss=r["train_loss"],
                            eval_loss=r["eval_loss"]) for r in rows],
            "best_eval_epoch": best_ep,
            "best_eval_loss": best_val,
            "final_eval_loss": lr_["final_eval_loss"],
            "final_train_loss": lr_["final_train_loss"],
            "canonical_one_epoch": {
                "run_name": d["canonical_one_epoch"]["run_name"],
                "eval_loss": canon,
                "train_loss": d["canonical_one_epoch"]["train_loss"],
                "validation_data": d["canonical_one_epoch"]["validation_data"],
            },
            "crossing": {
                "shaded_epoch_interval": [1, 2],
                "why_no_point_marker": (
                    "train 1.9456 > eval 1.9199 at epoch 1 and train 1.7176 < eval 2.0559 at "
                    "epoch 2, so the crossing lies inside epoch 1->2. Its position inside that "
                    "epoch is not logged; intersecting the two drawn straight segments would "
                    "produce epoch 1.071 / loss 1.930, which is an artefact of the drawing and "
                    "not a measurement. The interval is shaded instead."),
            },
        },
        "comparison_design": d["comparison_design"],
        "interpretation_bounds": d["interpretation_bounds"],
    }
    for k in ORDER:
        arms[k].pop("_frac")
        arms[k].pop("_inst")

    # =================================================================== figure
    fig = plt.figure(figsize=(style.FULLWIDTH_IN, 4.15), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.18], hspace=0.05, wspace=0.06)

    def grid(ax):
        """CHANGE 7: one grid treatment. Faint horizontal rules at the axis's own ticks, only."""
        ax.grid(axis="y", color=style.RULEGREY, lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    # ------------------------------------------------------------------ (a)
    axesA = []
    for i in range(len(ORDER)):
        axesA.append(fig.add_subplot(gs[i, 0], sharex=axesA[0] if axesA else None))

    # the raw series were popped out of `arms` for the JSON, so de-average again to draw
    for ax, stem in zip(axesA, ORDER):
        frac, inst, _, total = deaverage(d["curve_series"][stem])
        ax.axvspan(BAND[0], BAND[1], color=style.PANELBG, lw=0, zorder=0)
        # CHANGE 4: one arm, one axis. No hue distinguishes the arms because nothing needs to --
        # they are in different panels, and a colour here would imply a comparison the caption
        # explicitly forbids. Ink for all three.
        ax.plot(frac, inst, lw=1.1, color=style.INK, solid_joinstyle="round", zorder=3)
        grid(ax)
        ax.set_xlim(0, 1.0)
        # five ticks, not three: with three the residual panel's only ticks were 2.0/2.5/3.0 and
        # its plateau sat at 1.85, below the lowest one, with nothing to read it against. Five
        # also makes the three panels' tick VALUES visibly different from one another, which is
        # the point of giving each arm its own axis.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="y", pad=2)

        lo, hi = min(inst), max(inst)
        pad = 0.10 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 1.05 * pad)

        # direct labelling in the empty upper right; no legend, and the arm's own scheduled step
        # count sits with its name rather than in a key the reader has to walk back to.
        name, sub = NAMES[stem]
        ax.text(0.985, 0.97, name, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=style.ACCENT if stem == ORDER[0] else style.INK)
        ax.text(0.985, 0.71, sub % thousands(total), transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=style.QUIETINK)

        if not d["curve_series"][stem]["completed"]:
            # the line simply stops; say so at the stop rather than leaving a reader to wonder
            ax.plot([frac[-1]], [inst[-1]], marker="|", ms=6, mew=1.0, color=style.INK, zorder=4)

    for ax in axesA[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axesA[-1].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axesA[-1].set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    axesA[-1].set_xlabel("fraction of the arm's nominal one-epoch horizon")
    axesA[1].set_ylabel("training loss, de-averaged")

    # ------------------------------------------------------------------ (b)
    axB = fig.add_subplot(gs[:, 1])
    grid(axB)

    # CHANGE 8: the crossing is an INTERVAL. Shading epoch 1->2 says exactly what two logged
    # points support and no more.
    axB.axvspan(1, 2, color=style.PANELBG, lw=0, zorder=0)

    axB.plot(epochs, train, marker="o", ms=3.2, lw=1.3, color=style.INK, zorder=3)
    axB.plot(epochs, valid, marker="o", ms=3.2, lw=1.3, color=style.ACCENT, zorder=3)

    # the canonical one-epoch run's validation loss: a level, not a series. Grey and dashed, and
    # direct-labelled, so no hue is spent on a quantity whose whole message is its POSITION.
    axB.axhline(canon, color=style.RULEGREY, lw=1.0, ls=(0, (4.5, 2.2)), zorder=2)

    # the epoch-1 minimum, ringed
    axB.plot([best_ep], [best_val], marker="o", ms=7.6, mfc="none", mew=1.0,
             color=style.ACCENT, zorder=4)

    axB.set_xlim(0.45, 10.55)
    axB.set_ylim(1.38, 2.50)
    axB.set_xticks(epochs)
    axB.set_yticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    axB.set_xlabel("epoch of the ten-epoch schedule")
    axB.set_ylabel("epoch-mean cross-entropy")

    # --- annotations, all routed through empty space -------------------------------------
    # The epoch-1 label sits in the empty strip above the validation plateau, and its leader drops
    # STRAIGHT DOWN from the text box's bottom-left corner (relpos pins the departure point) into
    # the column left of the validation curve's rise, so it crosses neither series. v1 put this
    # text at (1.45, 1.76) -- underneath the training curve -- with a leader that crossed that
    # curve on its way out.
    axB.annotate("best validation of the ten-epoch run\n(epoch %d: %.4f)" % (best_ep, best_val),
                 xy=(best_ep, best_val), xytext=(1.02, 2.485), textcoords="data",
                 ha="left", va="top", fontsize=9, color=style.INK, linespacing=1.30,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=style.RULEGREY,
                                 linestyle=(0, (1, 1.8)), shrinkA=2, shrinkB=6,
                                 relpos=(0.008, 0.0)))

    # the crossing, named where it happens: in the wedge between the rising validation curve and
    # the reference level, which is the one clear span of that region
    axB.annotate("training loss falls below validation\nloss inside the shaded epoch",
                 xy=(2.30, 2.030), ha="left", va="top", fontsize=9,
                 color=style.QUIETINK, linespacing=1.30)

    # the reference level, labelled at the right end of its own line, in the empty band between
    # the training plateau and the line itself. Short enough that its left edge clears the
    # training curve's descent; the caption says that it is the same validation file.
    axB.annotate("canonical one-epoch validation: %.4f" % canon,
                 xy=(10.35, canon - 0.030), ha="right", va="top", fontsize=9,
                 color=style.QUIETINK)

    # the two series, direct-labelled at the end a reader's eye leaves them on
    axB.annotate("validation loss", xy=(10.35, 2.285), ha="right", va="bottom",
                 fontsize=9, color=style.ACCENT)
    axB.annotate("training loss", xy=(10.35, 1.478), ha="right", va="bottom",
                 fontsize=9, color=style.INK)

    # ------------------------------------------------------------------ panel letters
    axesA[0].text(0.0, 1.045, "(a)", transform=axesA[0].transAxes, ha="left", va="bottom",
                  fontsize=10, color=style.INK)
    axB.text(0.0, 1.015, "(b)", transform=axB.transAxes, ha="left", va="bottom",
             fontsize=10, color=style.INK)

    style.save(fig, "fig-training", drawn)

    # ------------------------------------------------------------------ console record
    print("\npanel (a)  de-averaged training loss, %s%%-%s%% band" %
          (int(BAND[0] * 100), int(BAND[1] * 100)))
    print("%-34s %9s %8s %8s %8s %14s" %
          ("arm", "sched", "L@45-55", "L@65-75", "L@end", "lr/peak in band"))
    for stem in ORDER:
        a = arms[stem]
        b = a["lr_fraction_of_peak_in_band"]
        print("%-34s %9s %8s %8s %8.4f %14s" % (
            a["label_in_artifact"][:34], thousands(a["nominal_total_steps"]),
            "%.4f" % a["band_mean_45_55pct"] if a["band_mean_45_55pct"] else "--",
            "%.4f" % a["band_mean_65_75pct"] if a["band_mean_65_75pct"] else "--",
            a["deaveraged_last"],
            "%.0f-%.0f%%" % (100 * b[0], 100 * b[1]) if b else "--"))
    print("\npanel (b)  ten-epoch residual schedule (%s updates, nominal horizon %s)" %
          (thousands(lr_["actual_optimizer_updates"]),
           thousands(lr_["nominal_scheduler_horizon"])))
    print("%-7s %10s %10s" % ("epoch", "train", "validation"))
    for r in rows:
        print("%-7d %10.4f %10.4f" % (r["epoch"], r["train_loss"], r["eval_loss"]))
    print("canonical one-epoch validation on the same file: %.4f" % canon)
    print("\ninterpretation_bounds.not_supported:")
    for s in d["interpretation_bounds"]["not_supported"]:
        print("  - %s" % s)


if __name__ == "__main__":
    main()

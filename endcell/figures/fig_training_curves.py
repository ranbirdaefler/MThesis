"""Render the training-adequacy figure from the committed structured artifact.

Raw cluster logs are needed only to refresh ``training_adequacy_long.json``.  The default figure
path is reproducible in a clean clone and fails closed on schema errors, truncated curves,
non-monotonic steps, or inconsistent production values.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from endcell.analysis.training_adequacy_long import (  # noqa: E402
    CURVE_SPECS,
    DEFAULT_OUT as ADEQUACY,
    load_artifact,
)

FIGS = os.path.join(ROOT, "thesis", "figs")
DEFAULT_PDF = os.path.join(FIGS, "fig-training.pdf")


def deaverage(row):
    """Return fraction, per-window loss and LR after validating one curve."""
    points = row["points"]
    total = row["nominal_total_steps"]
    steps = [point["step"] for point in points]
    cumulative = [point["cumulative_train_loss"] for point in points]
    learning_rate = [point["learning_rate"] for point in points]
    instantaneous = [cumulative[0]]
    for i in range(1, len(steps)):
        delta = steps[i] - steps[i - 1]
        if delta <= 0:
            raise ValueError("curve steps must be strictly increasing")
        instantaneous.append(
            (steps[i] * cumulative[i] - steps[i - 1] * cumulative[i - 1]) / delta
        )
    return [step / total for step in steps], instantaneous, learning_rate, total


def render(artifact_path=ADEQUACY, output_pdf=DEFAULT_PDF, output_png=None):
    adequacy = load_artifact(artifact_path)
    fig, (ax, axlong) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axlr = ax.twinx()

    for stem, spec in CURVE_SPECS.items():
        row = adequacy["curve_series"][stem]
        frac, inst, lr, total = deaverage(row)
        label = row["label"].replace("epsilon", "$\\varepsilon$")
        ax.plot(frac, inst, lw=2.0 if row["canonical"] else 1.3,
                color=row["colour"], alpha=1.0 if row["canonical"] else 0.8,
                zorder=3,
                label="%s  ($%s$ scheduled steps)" %
                      (label, format(total, ",").replace(",", "{,}")))
        if row["canonical"]:
            peak = max(lr)
            if peak <= 0:
                raise ValueError("canonical curve has no positive learning rate")
            axlr.plot(frac, [value / peak for value in lr], lw=1.0, ls="--",
                      color="#999999", zorder=1)

    ax.axvspan(0.50, 0.70, color="#f0f0f0", zorder=0)
    ax.annotate("flat here at 22--52\\%\nof peak learning rate",
                xy=(0.60, 2.00), xytext=(0.60, 2.32), ha="center",
                fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
    axlr.set_ylabel("learning rate (fraction of peak)", fontsize=8.5, color="#777777")
    axlr.tick_params(axis="y", labelsize=8, colors="#777777")
    axlr.set_ylim(0, 1.05)
    ax.set_xlabel("fraction of scheduled short run")
    ax.set_ylabel("training loss, de-averaged")
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("A  Canonical short-run arms", loc="left", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    axlr.spines["top"].set_visible(False)

    rows = adequacy["long_run"]["epochs"]
    epochs = [row["epoch"] for row in rows]
    train = [row["train_loss"] for row in rows]
    valid = [row["eval_loss"] for row in rows]
    canonical_valid = adequacy["canonical_one_epoch"]["eval_loss"]
    axlong.plot(epochs, train, marker="o", ms=4.0, lw=1.7, color="#1a1a1a",
                label="ten-epoch train")
    axlong.plot(epochs, valid, marker="o", ms=4.0, lw=1.7, color="#c44e52",
                label="ten-epoch validation")
    axlong.axhline(canonical_valid, color="#4c72b0", lw=1.2, ls="--",
                   label="canonical one-epoch validation")
    best = adequacy["long_run"]["best_eval_epoch"]
    best_loss = adequacy["long_run"]["best_eval_loss"]
    axlong.annotate("best long-run validation\n(epoch %d: %.4f)" % (best, best_loss),
                    xy=(best, best_loss), xytext=(1.45, 1.76), fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    axlong.set_xlabel("epoch in the ten-epoch schedule")
    axlong.set_ylabel("epoch-mean cross-entropy")
    axlong.set_xticks(epochs)
    axlong.grid(alpha=0.18, lw=0.6)
    axlong.legend(fontsize=7.5, frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.92, loc="center right")
    axlong.set_title("B  Longer residual schedule", loc="left", fontsize=10,
                     fontweight="bold")
    axlong.spines["top"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png:
        os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
        fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("->", output_pdf)

    print("\n%-38s %10s %8s %8s %8s" %
          ("arm", "scheduled", "L@50%", "L@70%", "L@end"))
    for stem in CURVE_SPECS:
        row = adequacy["curve_series"][stem]
        frac, inst, _, total = deaverage(row)

        def band(lo, hi):
            values = [value for f, value in zip(frac, inst) if lo < f <= hi]
            return sum(values) / len(values) if values else float("nan")

        print("%-38s %10s %8.4f %8.4f %8.4f" %
              (row["label"][:38], format(total, ","), band(0.45, 0.55),
               band(0.65, 0.75), inst[-1]))
    long = adequacy["long_run"]
    print("\nten-epoch actual optimizer updates: %s; nominal scheduler horizon: %s" %
          (format(long["actual_optimizer_updates"], ","),
           format(long["nominal_scheduler_horizon"], ",")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=ADEQUACY)
    ap.add_argument("--out", default=DEFAULT_PDF)
    ap.add_argument("--png", default=None, help="optional PNG preview path")
    args = ap.parse_args()
    render(args.artifact, args.out, args.png)


if __name__ == "__main__":
    main()

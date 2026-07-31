"""Shared plotting style for every thesis figure.

One module, applied by every script, so the figures read as one system. The conventions here are
not cosmetic -- several of them exist because the alternative would be misleading:

  * The chance line at 0.50 is drawn on EVERY NIR axis, always. Omitting it, or implying it through
    the axis limits, lets a reader infer a scale that does not exist.
  * Every NIR axis label carries its FRAME. Expression-frame NIR (ceiling 0.576) and residual-frame
    NIR (ceiling 0.968) otherwise appear in adjacent figures under one name, and a reader flipping
    between them reads 0.498 -> 0.704 as the model improving. They are not comparable.
  * A noise ceiling is a BAND, never a hard line -- it is a half-sample estimate.
  * One CI convention only: 95% bootstrap clustered over cell lines. Any interval that is not
    clustered (the normal-approximation intervals stored by evaluate_endcell) is NOT DRAWN. A
    0.0008 half-width sitting next to the thesis's genuine +/-0.03 intervals invites a reader to
    distrust all of them.
  * Quantities with no interval available (DRF, the mechanism swap) are bare markers, and the
    caption says why. Bar length must never be the message for an uncertain quantity.

Every figure script also writes a sibling .json of the exact numbers it drew, so a figure and the
table beside it can be diffed automatically. Three of this thesis's tables have already disagreed
with the JSONs they came from; that is not a risk worth carrying twice.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- geometry
# a4paper with \newgeometry{margin=2.5cm} -> 21.0 - 2*2.5 = 16.0 cm of text width.
TEXTWIDTH_IN = 16.0 / 2.54          # 6.30 in
HALFWIDTH_IN = TEXTWIDTH_IN / 2.0

# --------------------------------------------------------------------------- colour
# Okabe-Ito, colour-blind safe. The arm -> colour map is GLOBAL and must not vary between figures.
OKABE = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "grey":      "#7F7F7F",
}

ARM = {
    "ceiling":       dict(color=OKABE["grey"],      marker="_", label="noise ceiling"),
    "model":         dict(color=OKABE["blue"],      marker="o", label="model"),
    "control_copy":  dict(color=OKABE["vermilion"], marker="s", label="control-copy"),
    "control":       dict(color=OKABE["vermilion"], marker="s", label="control-copy"),
    "scramble":      dict(color=OKABE["orange"],    marker="^", label="scramble"),
    "linear":        dict(color=OKABE["green"],     marker="D", label="linear"),
    "drug_lookup":   dict(color=OKABE["purple"],    marker="P", label="drug lookup"),
    "drug_lookup_1": dict(color=OKABE["purple"],    marker="X", label="drug lookup (1 line)"),
    "moa_lookup":    dict(color=OKABE["skyblue"],   marker="v", label="MoA lookup"),
    "generic":       dict(color=OKABE["black"],     marker=".", label="generic"),
    "random":        dict(color=OKABE["black"],     marker=".", label="random"),
    "mean":          dict(color=OKABE["black"],     marker=".", label="mean"),
}

CHANCE = 0.50

# --------------------------------------------------------------------------- rc
def apply():
    """Serif, sized to match the 12pt body face. Call once at the top of every script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })


# --------------------------------------------------------------------------- primitives
def chance_line(ax, orientation="h", label=True):
    """The 0.50 rule. Thin, solid, black. Never dashed, never omitted."""
    fn = ax.axhline if orientation == "h" else ax.axvline
    fn(CHANCE, color="black", lw=0.9, zorder=1)
    if label:
        if orientation == "h":
            ax.annotate("chance", xy=(1.002, CHANCE), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=8, color="black")
        else:
            ax.annotate("chance", xy=(CHANCE, 1.004), xycoords=("data", "axes fraction"),
                        va="bottom", ha="center", fontsize=8, color="black")


def ceiling_band(ax, value, halfwidth=0.010, orientation="h", label="noise ceiling"):
    """A ceiling is a half-sample estimate, so it is a band and never a hard line."""
    fn = ax.axhspan if orientation == "h" else ax.axvspan
    fn(value - halfwidth, value + halfwidth, color=OKABE["grey"], alpha=0.22, lw=0, zorder=0)
    if orientation == "h":
        ax.annotate(label, xy=(0.995, value), xycoords=("axes fraction", "data"),
                    va="bottom", ha="right", fontsize=8, color=OKABE["grey"])
    else:
        ax.annotate(label, xy=(value, 0.99), xycoords=("data", "axes fraction"),
                    va="top", ha="center", fontsize=8, color=OKABE["grey"])


def nir_label(frame):
    """Every NIR axis names its frame. `frame` is 'expression' or 'residual'."""
    if frame == "expression":
        return "NIR (expression frame, within plate)"
    if frame == "residual":
        return "NIR (residual frame, cell-line comparison set)"
    raise ValueError("frame must be 'expression' or 'residual', got %r" % frame)


def whisker(ax, x, y, lo, hi, orientation="h", **kw):
    """Caps-off whisker. Only ever used for clustered bootstrap intervals."""
    if orientation == "h":
        ax.plot([lo, hi], [y, y], solid_capstyle="butt", **kw)
    else:
        ax.plot([x, x], [lo, hi], solid_capstyle="butt", **kw)


# --------------------------------------------------------------------------- output
def save(fig, name, drawn):
    """Write <name>.pdf into thesis/figs/ and <name>.json beside it.

    The sibling JSON holds the exact numbers the figure drew, so it can be diffed against the
    table that quotes them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.abspath(os.path.join(here, "..", "..", "thesis", "figs"))
    os.makedirs(figs, exist_ok=True)
    pdf = os.path.join(figs, name + ".pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    with open(os.path.join(figs, name + ".json"), "w") as fh:
        json.dump(drawn, fh, indent=2, default=float)
    print("wrote %s" % pdf)
    print("wrote %s" % os.path.join(figs, name + ".json"))
    return pdf


def results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "RESULTS_cluster"))


def load(name):
    with open(os.path.join(results_dir(), name), encoding="utf-8") as fh:
        return json.load(fh)

"""Shared plotting style for every thesis_v2 figure.

Derived from endcell/figures/style.py. One module, applied by every script, so the figures read
as one system. The conventions below are NOT cosmetic -- several of them exist because the
alternative would be misleading, and every one of them is carried forward from v1 unchanged:

  * The chance line at 0.50 is drawn on EVERY NIR axis, always. Omitting it, or implying it
    through the axis limits, lets a reader infer a scale that does not exist.
  * Every NIR axis label carries its FRAME. Expression-frame NIR (reference 0.576) and
    residual-frame NIR (reference 0.854) otherwise appear in adjacent figures under one name, and
    a reader flipping between them reads 0.498 -> 0.537 as the model improving. They are not
    comparable and they may not be subtracted.
  * A within-well reference is a BAND, never a hard line -- it is a split-half estimate.
  * One CI convention only: 95% bootstrap clustered over cell lines. Any interval that is not
    clustered (the normal-approximation intervals stored by evaluate_endcell) is NOT DRAWN. A
    0.0008 half-width sitting next to the thesis's genuine +/-0.03 intervals invites a reader to
    distrust all of them.
  * Quantities with no interval available (DRF, the mechanism swap) are bare markers, and the
    caption says why. Bar length must never be the message for an uncertain quantity.
  * Every save() writes a sibling .json of the exact numbers the figure drew, so a figure and the
    table beside it can be diffed automatically. Three of this thesis's tables have already
    disagreed with the JSONs they came from; that is not a risk worth carrying twice.

=============================================================================================
WHAT CHANGED FROM v1 style.py, AND WHY
=============================================================================================

(1) GEOMETRY. v1 set TEXTWIDTH_IN = 16.0/2.54 = 6.30in, which was v1's 160mm text block. The v2
    page has a 142mm block (5.591in) and a 174mm full-width measure. A 6.30in figure dropped into
    a 142mm block is scaled by LaTeX to 0.887, so every label in it prints 11% smaller than it
    was designed: a 9pt tick label lands at 8.0pt, an 8pt annotation at 7.1pt. That -- not the
    drawing -- is what read as "off". Figures are now drawn AT the final measure and are NEVER
    \resizebox'd or scaled in LaTeX. Include them with \includegraphics{} and no width= key.

(2) TYPE. v1 set font.serif to DejaVu Serif, so the figures were set in a face unrelated to the
    Palatino body and looked like they came from another document. v2 requests Palatino first.
    Text ink is set to the page's ink colours so the figure and the paragraph beside it are the
    same grey. NOTE the runtime check: resolve_serif() reports the face actually used, because
    "TeX Gyre Pagella" is frequently absent from a matplotlib font cache even where LaTeX has it.

(3) SIZES. Because nothing is scaled any more, a label at 10pt prints at 10pt. Base 10, axis
    labels 10, ticks 9, annotations 9, legend 9 -- which puts figure type at or just under the
    document's caption size instead of 1-2pt below it.

(4) OUTPUT PATH. v1's save() wrote into thesis/figs/. thesis/ is the frozen, read-only v1 tree
    and writing there is the single worst mistake available in this repo. This module lives IN
    the v2 output directory, so the destination is dirname(__file__) with no ".." traversal at
    all, and _figs_dir() additionally asserts the resolved path. There is no argument, no
    environment variable and no relative hop that can redirect it into thesis/.

(5) ACCENT. SUPERSEDED by (7). v2 kept Okabe-Ito for the arms and moved "model" to the old
    blue accent #0B5394.

(7) THE v4 MARGINALIA PALETTE (design/marginalia/01-tokens.md, 06-figures.md). Figures use the
    document palette and nothing else: ink, accent, quietink, rulegrey, panelbg. The model (the
    one series a figure is about) is the accent; references are quietink (marks) or rulegrey
    (rules, bands); every other arm is ink or quietink and keeps its own MARKER SHAPE, which is
    now the channel that tells arms apart. Okabe-Ito is retired -- its vermilion would read as the
    new red accent. Type: text in EB Garamond (the body face); axis labels and tick labels in the
    apparatus sans, Alegreya Sans, at 8pt (applied to every axes by save(), see
    apparatus_type()); maths in mathtext 'custom' on the serif.

(6) THE RETIRED TERM. v1's ARM entry for the reference was labelled "noise ceiling". That term
    is retired document-wide (see the glossary entry "noise ceiling (retired)" in
    00-HowToRead.tex): joining "ceiling" to a word for error invites reading a two-cell precision
    estimate as a bound on achievable performance. The label is now the construction's real name.
    _check_label() enforces this mechanically on every string this module draws.
"""
import glob
import json
import os
import subprocess
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

# --------------------------------------------------------------------------- geometry
# CHANGE 1. The v2 body text block is 142mm; \begin{fullwidth} gets 174mm. Draw at these widths
# and never scale in LaTeX -- scaling is what shrinks the type away from the sizes set below.
MM_PER_IN = 25.4
TEXTWIDTH_MM = 142.0
FULLWIDTH_MM = 174.0
TEXTWIDTH_IN = TEXTWIDTH_MM / MM_PER_IN      # 5.5906 in = 402.52 PostScript pt (72/in)
FULLWIDTH_IN = FULLWIDTH_MM / MM_PER_IN      # 6.8504 in = 493.23 PostScript pt (72/in)
HALFWIDTH_IN = TEXTWIDTH_IN / 2.0            # 2.7953 in, for a two-up pair inside the block

# --------------------------------------------------------------------------- ink
# CHANGE 7 (v4 palette). The preamble's own \definecolor values, so a figure's inks are the
# document's inks. Nothing else is defined: no hue outside these five may reach a figure.
ACCENT = "#A63D2F"    # \definecolor{accent}   -- the model / the one series the figure is about
INK = "#2B2A27"       # \definecolor{ink}      -- axes, primary marks, text (warm near-black)
QUIETINK = "#7A766D"  # \definecolor{quietink} -- secondary series, reference marks, annotations
RULEGREY = "#CFCBC0"  # \definecolor{rulegrey} -- grid, reference lines, bar tracks, hollow
                      #   points. Never text, never a series that has to be read.
PANELBG = "#F3F0E7"   # \definecolor{panelbg}  -- bands only (e.g. the 80%-power band); never a
                      #   plot background

# --------------------------------------------------------------------------- colour
# CHANGE 7. The Okabe-Ito map is retired (its vermilion collides with the red accent). The arm ->
# colour/marker map below is still GLOBAL and must not vary between figures; with only two inks
# for the non-model arms, the MARKER is what keeps arms apart, so no two arms share one.

# CHANGE 6. The document's own name for this quantity, from the glossary entry
# \glentry{within-well split-half precision reference}. Shortened only by dropping "precision",
# which no figure legend has room for and which no reader will misread.
REFERENCE_LABEL = "within-well split-half reference"

# Terms that must never appear in anything this module draws or writes. The check is mechanical
# because the retired term is short, natural to type, and reappeared three times during v1.
_BANNED = ("noise ceiling", "noise-ceiling", "noiseceiling")

ARM = {
    # CHANGE 6: label was "noise ceiling". CHANGE 7: a reference MARK is quietink.
    "reference":     dict(color=QUIETINK, marker="_", label=REFERENCE_LABEL),
    "ceiling":       dict(color=QUIETINK, marker="_", label=REFERENCE_LABEL),
    # CHANGE 7: the model is the accent (was the old blue accent #0B5394).
    "model":         dict(color=ACCENT,   marker="o", label="model"),
    # CHANGE 7: every other arm is ink or quietink (was Okabe vermilion / orange / green /
    # purple / sky blue); markers unchanged.
    "control_copy":  dict(color=INK,      marker="s", label="control-copy"),
    "control":       dict(color=INK,      marker="s", label="control-copy"),
    "scramble":      dict(color=QUIETINK, marker="^", label="scramble"),
    "linear":        dict(color=QUIETINK, marker="D", label="linear"),
    "drug_lookup":   dict(color=INK,      marker="P", label="drug lookup"),
    "drug_lookup_1": dict(color=QUIETINK, marker="X", label="drug lookup (1 line)"),
    "moa_lookup":    dict(color=QUIETINK, marker="v", label="MoA lookup"),
    "generic":       dict(color=INK,      marker=".", label="generic"),
    "random":        dict(color=INK,      marker=".", label="random"),
    "mean":          dict(color=INK,      marker=".", label="mean"),
}

CHANCE = 0.50

# --------------------------------------------------------------------------- font resolution
# CHANGE 2. Palatino first. "TeX Gyre Pagella" is the body face; matplotlib can only use it if
# the .otf is in a directory matplotlib scans, which on a MiKTeX/TeX Live box it usually is not.
# Palatino Linotype and URW Palladio L / P052 are the same design and metrics, so any of them
# keeps the figure in the body's face. DejaVu Serif is last and is a FAILURE state, not a choice.
# CHANGE 7: the v4 body face is EB Garamond (\usepackage[lining]{ebgaramond}); its OTFs are in the
# TeX tree and register_document_faces() hands them to matplotlib. The Palatino faces stay only as
# fallbacks behind it.
SERIF_STACK = [
    "EB Garamond",        # the document's actual body face (TeX tree OTFs)
    "EB Garamond 12",     # Georg Duffner's original release name
    "TeX Gyre Pagella",   # v2/v3 body face; fallback only
    "Palatino Linotype",
    "Book Antiqua",
    "DejaVu Serif",       # fallback only -- if this is what resolves, say so in the report
]

# The document's apparatus face (captions, table heads) and mono, for the rare figure that wants
# to match a caption or show a code token. Requested by name; absent is not fatal.
# CHANGE 7: axis labels and tick labels are apparatus, set in the document's sans.
SANS_STACK = ["Alegreya Sans", "TeX Gyre Heros", "Arial", "DejaVu Sans"]
APPARATUS_PT = 8          # axis labels and tick labels, sans (06-figures.md)
MONO_STACK = ["Inconsolata", "Consolas", "DejaVu Sans Mono"]

# Directories a TeX distribution keeps its OTFs in. TeX Gyre Pagella IS on a machine that builds
# this thesis -- it is the body face -- but a TeX tree is not on matplotlib's scan path, so
# matplotlib reports it missing and silently substitutes. That substitution is exactly how v1's
# figures ended up in DejaVu. Registering the TeX tree's OTFs is what lets the figures use the
# document's real face rather than a metric clone of it.
_TEX_FONT_GLOBS = [
    os.path.expanduser("~/AppData/Roaming/TinyTeX/texmf-dist/fonts/opentype/*/*/*.otf"),
    os.path.expanduser("~/AppData/Local/Programs/MiKTeX/fonts/opentype/*/*/*.otf"),
    "C:/Program Files/MiKTeX/fonts/opentype/*/*/*.otf",
    "C:/texlive/*/texmf-dist/fonts/opentype/*/*/*.otf",
    os.path.expanduser("~/texlive/*/texmf-dist/fonts/opentype/*/*/*.otf"),
    "/usr/share/texmf/fonts/opentype/*/*/*.otf",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/*/*/*.otf",
]

_registered = False


def register_document_faces(verbose=False):
    """Teach matplotlib about the TeX tree's OTFs so the body face is actually available.

    Idempotent; called once on import. If nothing is found the module falls back through
    SERIF_STACK as before, and font_report() will say which face won.
    """
    global _registered
    if _registered:
        return
    _registered = True
    files = []
    for pat in _TEX_FONT_GLOBS:
        files += glob.glob(pat)
    if not files:
        # Ask the TeX distribution itself where it put the faces, then take those directories.
        for probe in ("EBGaramond-Regular.otf", "AlegreyaSans-Regular.otf",
                      "texgyrepagella-regular.otf"):
            try:
                out = subprocess.run(["kpsewhich", probe],
                                     capture_output=True, text=True, timeout=10)
                hit = out.stdout.strip().splitlines()
                if hit:
                    files += glob.glob(os.path.join(os.path.dirname(hit[0]), "*.otf"))
            except Exception:
                pass
    known = {f.name for f in _fm.fontManager.ttflist}
    added = 0
    for f in files:
        base = os.path.basename(f).lower()
        # CHANGE 7: EB Garamond (body) and Alegreya Sans (apparatus) join the v2 faces. The
        # Initials face and the Alegreya Sans SC faces are separate families and are left out.
        if base.startswith("ebgaramond-initials"):
            continue
        if not (base.startswith("ebgaramond-") or base.startswith("alegreyasans-")
                or base.startswith("texgyrepagella") or base.startswith("texgyreheros")
                or base.startswith("inconsolata")):
            continue
        try:
            if _fm.get_font(f).family_name in known:
                continue
            _fm.fontManager.addfont(f)
            added += 1
        except Exception:
            continue
    if verbose:
        print("style_v2: registered %d TeX-tree font files" % added)


register_document_faces()


def resolve_serif():
    """Return (family_name, font_file) for the first face in SERIF_STACK that actually resolves.

    Never assume the requested face is the drawn face. matplotlib silently substitutes DejaVu
    Sans for a missing family, which is exactly how v1's figures ended up in a face nobody chose.
    """
    for name in SERIF_STACK:
        try:
            path = _fm.findfont(_fm.FontProperties(family=name), fallback_to_default=False)
        except Exception:
            continue
        if path and os.path.exists(path):
            return name, path
    return None, None


def font_report():
    """One-line human-readable statement of what the figures are actually set in."""
    name, path = resolve_serif()
    if name is None:
        return "NO SERIF RESOLVED -- matplotlib default in force"
    flag = "" if name == "EB Garamond" else "   <-- FALLBACK, not EB Garamond"
    return "%s  [%s]%s" % (name, path, flag)


# --------------------------------------------------------------------------- rc
def apply(strict_font=False):
    """EB Garamond text, Alegreya Sans apparatus, sized for a figure placed at 100%, never scaled.

    Call once at the top of every script. `strict_font=True` raises if EB Garamond does not
    resolve, for scripts that would rather fail than quietly print in the wrong face.
    """
    family, path = resolve_serif()
    if family != "EB Garamond":
        msg = ("style_v2: EB Garamond did not resolve (got %r). The figure will not match the "
               "body. Check the TeX tree's fonts/opentype/public/ebgaramond/." % family)
        if strict_font:
            raise RuntimeError(msg)
        warnings.warn(msg)

    serif = [family] if family else []
    serif += [s for s in SERIF_STACK if s != family]

    plt.rcParams.update({
        # CHANGE 2: family and face.
        "font.family": "serif",
        "font.serif": serif,
        # the page's apparatus face and mono, for the rare figure that quotes a caption or a token
        "font.sans-serif": SANS_STACK,
        "font.monospace": MONO_STACK,
        # matplotlib ships no Garamond mathtext set. 'custom' sets the math ROMAN/ITALIC/BOLD in
        # the resolved serif so digits and variables match the body, and falls back to STIX for
        # symbols the face has no glyph for. Maths inside a sans axis label stays in the serif;
        # 06-figures.md accepts that mismatch.
        "mathtext.fontset": "custom",
        "mathtext.rm": family or "DejaVu Serif",
        "mathtext.it": "%s:italic" % (family or "DejaVu Serif"),
        "mathtext.bf": "%s:bold" % (family or "DejaVu Serif"),
        "mathtext.fallback": "stix",

        # CHANGE 3: drawn at final size, so these ARE the printed point sizes. CHANGE 7: axis
        # and tick labels at APPARATUS_PT (8pt, sans -- the face is set by apparatus_type()).
        "font.size": 10,          # base
        "axes.labelsize": APPARATUS_PT,
        "axes.titlesize": 10,
        "xtick.labelsize": APPARATUS_PT,
        "ytick.labelsize": APPARATUS_PT,
        "legend.fontsize": 9,     # legend

        # CHANGE 7 (v4 palette): ink for axes and text, per 06-figures.md.
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.edgecolor": INK,

        # unchanged from v1
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,       # embed real glyphs, not Type 3 outlines
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return family


# --------------------------------------------------------------------------- size helpers
def figsize(width="text", ratio=0.618, height_in=None):
    """A figure at the final printed measure. width is 'text' (142mm), 'full' (174mm), 'half'."""
    w = {"text": TEXTWIDTH_IN, "full": FULLWIDTH_IN, "half": HALFWIDTH_IN}[width]
    return (w, height_in if height_in is not None else w * ratio)


# --------------------------------------------------------------------------- guards
def _check_label(*strings):
    """CHANGE 6 enforced mechanically: the retired term never reaches a rendered figure."""
    for s in strings:
        if s is None:
            continue
        low = str(s).lower()
        for bad in _BANNED:
            if bad in low:
                raise ValueError(
                    "style_v2: %r contains the retired term %r. It is banned document-wide "
                    "(00-HowToRead.tex glossary). Use %r, 'positive control', or name the "
                    "estimator." % (str(s), bad, REFERENCE_LABEL))


# --------------------------------------------------------------------------- primitives
def chance_line(ax, orientation="h", label=True):
    """The 0.50 rule. Thin, solid, ink. Never dashed, never omitted."""
    fn = ax.axhline if orientation == "h" else ax.axvline
    fn(CHANCE, color=INK, lw=0.9, zorder=1)
    if label:
        # annotations at 9pt (CHANGE 3); v1 used 8pt because v1 got shrunk to 0.887.
        if orientation == "h":
            ax.annotate("chance", xy=(1.002, CHANCE), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=9, color=INK)
        else:
            ax.annotate("chance", xy=(CHANCE, 1.004), xycoords=("data", "axes fraction"),
                        va="bottom", ha="center", fontsize=9, color=INK)


def reference_band(ax, value, halfwidth=0.010, orientation="h", label=REFERENCE_LABEL):
    """A within-well reference is a split-half estimate, so it is a BAND and never a hard line.

    (CHANGE 6: this was ceiling_band(..., label="noise ceiling").) `label` may be overridden to
    name the specific estimator, e.g. "split-half, expression frame, plate-mates".
    """
    _check_label(label)
    fn = ax.axhspan if orientation == "h" else ax.axvspan
    fn(value - halfwidth, value + halfwidth, color=RULEGREY, lw=0, zorder=0)   # CHANGE 7
    if orientation == "h":
        ax.annotate(label, xy=(0.995, value), xycoords=("axes fraction", "data"),
                    va="bottom", ha="right", fontsize=9, color=QUIETINK)
    else:
        ax.annotate(label, xy=(value, 0.99), xycoords=("data", "axes fraction"),
                    va="top", ha="center", fontsize=9, color=QUIETINK)


def ceiling_band(ax, value, **kw):
    """Deprecated v1 name. Kept so v1 call sites port without silently drawing nothing."""
    warnings.warn("ceiling_band() is the v1 name; use reference_band().", DeprecationWarning)
    return reference_band(ax, value, **kw)


def nir_label(frame, comparison_set=True):
    """Every NIR axis names its frame. `frame` is 'expression' or 'residual'.

    `comparison_set=False` drops the comparison-set clause and names the FRAME only. Use it -- and
    only it -- on an axis that carries more than one comparison set, e.g. fig-difficulty panel (a),
    which plots a cross-plate series and a within-plate series on one axis. Baking "within plate"
    into that axis label would have half the marks contradicting the axis, and the cross-plate /
    within-plate distinction is the one chapter 3 spends four pages saying is not interchangeable.
    When the label is shortened this way the series legend MUST name the comparison set instead;
    the information may move, it may not disappear.
    """
    if frame == "expression":
        return "NIR (expression frame)" if not comparison_set \
            else "NIR (expression frame, within plate)"
    if frame == "residual":
        return "NIR (residual frame)" if not comparison_set \
            else "NIR (residual frame, cell-line comparison set)"
    raise ValueError("frame must be 'expression' or 'residual', got %r" % frame)


def whisker(ax, x, y, lo, hi, orientation="h", **kw):
    """Caps-off whisker. Only ever used for 95% bootstrap intervals clustered over cell lines."""
    if orientation == "h":
        ax.plot([lo, hi], [y, y], solid_capstyle="butt", **kw)
    else:
        ax.plot([x, x], [lo, hi], solid_capstyle="butt", **kw)


# --------------------------------------------------------------------------- output
def _figs_dir():
    """CHANGE 4. The v2 figure directory, and nothing else, ever.

    This module lives IN the working thesis figs/ dir, so the destination is dirname(__file__): there is no
    ".." hop that could resolve into the frozen v1 tree at thesis/figs/. The asserts below make
    a wrong answer loud rather than destructive if the module is ever copied elsewhere.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parts = os.path.normpath(here).replace("\\", "/").split("/")
    if parts[-1] != "figs" or not parts[-2].startswith("thesis_v"):
        raise RuntimeError(
            "style_v2 must live in thesis_v<n>/figs/ -- resolved %r instead. Refusing to write; "
            "thesis/ is the read-only v1 tree." % here)
    if "thesis" in parts:
        raise RuntimeError("refusing to write inside the read-only v1 thesis/ tree: %r" % here)
    return here


def apparatus_type(fig):
    """CHANGE 7. Axis labels and tick labels are apparatus: Alegreya Sans at APPARATUS_PT.

    Applied to every axes of the figure by save(), so it overrides any per-script fontsize on an
    axis label or tick label. Everything else (annotations, legends, titles, panel letters) stays
    in the serif.
    """
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontfamily("sans-serif")
            axis.label.set_fontsize(APPARATUS_PT)
        ax.tick_params(axis="both", which="both", labelsize=APPARATUS_PT)
        # tick_params(labelfontfamily=) needs matplotlib >= 3.8 and the figures are built with
        # 3.7.2, so the face is set on the tick labels themselves; ticks created later inherit it
        # from the first tick (Axis._copy_tick_props).
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick.label1.set_fontfamily("sans-serif")
                tick.label2.set_fontfamily("sans-serif")
    for t in (getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel", None)):
        if t is not None:
            t.set_fontfamily("sans-serif")
            t.set_fontsize(APPARATUS_PT)


def check_fits(fig, tol_in=0.01):
    """Warn if any ink falls outside the canvas.

    New failure mode introduced by CHANGE 1: because save() writes the EXACT canvas rather than
    a tight crop, content that does not fit is silently CLIPPED instead of quietly enlarging the
    figure. A y-label longer than the figure is tall is the usual case. The fix is always to
    compose within the measure -- shorten the label, or make the figure taller -- never to crop
    and never to scale.
    """
    r = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else None
    try:
        bb = fig.get_tightbbox(r)
    except Exception:
        return True
    fw, fh = fig.get_size_inches()
    over = []
    if bb.x0 < -tol_in:
        over.append("left %.2fin" % -bb.x0)
    if bb.y0 < -tol_in:
        over.append("bottom %.2fin" % -bb.y0)
    if bb.x1 > fw + tol_in:
        over.append("right %.2fin" % (bb.x1 - fw))
    if bb.y1 > fh + tol_in:
        over.append("top %.2fin" % (bb.y1 - fh))
    if over:
        warnings.warn("style_v2: content overflows the %.3f x %.3f in canvas and WILL be clipped "
                      "(%s). Compose within the measure; do not crop and do not scale."
                      % (fw, fh, ", ".join(over)))
        return False
    return True


def save(fig, name, drawn, formats=("pdf",), tight=False):
    """Write <name>.pdf into thesis_v2/figs/ and <name>.json beside it.

    The sibling JSON holds the exact numbers the figure drew, so it can be diffed against the
    table that quotes them. It is not optional: three v1 tables disagreed with their own sources.

    CHANGE 1 (continued). v1 saved with bbox_inches="tight", which CROPS the canvas to the drawn
    content: a figure created at 5.591in came out of v1 at whatever width the ink happened to
    occupy. That is harmless only if LaTeX then places it at natural size -- and the habitual
    \\includegraphics[width=\\textwidth] promptly scales it back up, which is the 11% type shrink
    this module exists to remove. So the default here writes the EXACT canvas: a 5.591in figure
    is a 5.591in PDF, width=\\textwidth is a scale of 1.0, and the point sizes above are the
    printed point sizes either way. Use layout="constrained" on the figure to control margins;
    pass tight=True only for a figure whose final width genuinely does not matter.
    """
    _check_label(name, json.dumps(drawn, default=float))
    if drawn is None:
        raise ValueError("save() requires the dict of numbers drawn; the sibling JSON is the "
                         "only mechanical link between a figure and the table quoting it.")
    figs = _figs_dir()
    apparatus_type(fig)
    if not tight:
        fig.canvas.draw()
        check_fits(fig)
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    out = None
    for fmt in formats:
        out = os.path.join(figs, "%s.%s" % (name, fmt))
        fig.savefig(out, format=fmt, **kw)
        print("wrote %s" % out)
    jpath = os.path.join(figs, name + ".json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(drawn, fh, indent=2, default=float)
    print("wrote %s" % jpath)
    return out


def results_dir():
    """The only source of numbers. Nothing is ever typed in by hand."""
    return os.path.abspath(os.path.join(_figs_dir(), "..", "..", "RESULTS_cluster"))


def load(name):
    with open(os.path.join(results_dir(), name), encoding="utf-8") as fh:
        return json.load(fh)

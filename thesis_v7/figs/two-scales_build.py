#!/usr/bin/env python3
# ===========================================================================
# two-scales_build.py -- generator for thesis_v2/figs/two-scales.tex (+ .json)
#
# WHY A GENERATOR FOR A TIKZ FIGURE. The point of this figure is that the two
# scales are stretched by DIFFERENT factors, chosen so that chance lands at one
# shared x and each frame's own within-well split-half precision reference lands
# at another shared x. Every dot position is therefore an arithmetic consequence
# of the measured NIR values. Hand-placing them would invite exactly the kind of
# silent drift between a figure and its source JSON that this repo has already
# been bitten by three times. So: read the two source files, check every value
# against the number the thesis prints, compute the geometry, emit the .tex, and
# emit a sibling .json recording every number drawn and where it came from.
#
# SOURCES (nothing else is read; nothing is invented):
#   RESULTS_cluster/nir_sameplate.json
#       .tiers.tier2_unseen_drugs.agg.<arm>_nir_expr  -> expression frame
#       .tiers.tier2_unseen_drugs.n_drugs             -> n = 606
#   RESULTS_cluster/re_v3.json
#       .means.<arm>                                  -> residual frame
#       .means.vs_baselines.control_copy.n            -> n = 1394
#       .means.common_support.*                       -> the 13.9% annotation
#
# WRITES ONLY:  thesis_v2/figs/two-scales.tex, thesis_v2/figs/two-scales.json
# Never touches thesis/ .
# ===========================================================================
from __future__ import annotations
import json, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES  = os.path.join(REPO, "RESULTS_cluster")

# ---------------------------------------------------------------- read data
with open(os.path.join(RES, "nir_sameplate.json")) as fh:
    sp = json.load(fh)
with open(os.path.join(RES, "re_v3.json")) as fh:
    rv = json.load(fh)

T2  = sp["tiers"]["tier2_unseen_drugs"]
AGG = T2["agg"]
N_EXPR = T2["n_drugs"]                       # 606

# expression frame, calibrated metric, within plate, tier 2
E = {
    "reference":    AGG["ceiling_nir_expr"][0],
    "control_copy": AGG["control_nir_expr"][0],
    "linear":       AGG["linear_nir_expr"][0],
    "model":        AGG["model_nir_expr"][0],
    "global_mean":  AGG["mean_nir_expr"][0],
}
M = rv["means"]
N_RESID = M["vs_baselines"]["control_copy"]["n"]     # 1394
CS = M["common_support"]                             # n = 1192

# residual frame, cell-line sets
R = {
    "reference":     M["ceiling"],
    "drug_lookup":   M["drug_lookup"],
    "model":         M["model"],
    "control_copy":  M["control_copy"],
    "random":        M["random"],
    "scramble_orth": M["scramble_orth"],
    "generic":       M["generic"],
}

CHANCE = 0.50

# --------------------------------------------------- guard: text agreement
# Every value below is quoted, to three places, in Sections/04d-baselines.tex
# (tab:allarms and fig:ladder) and/or Sections/04a-ruler.tex. If rounding the
# measured number does not reproduce the printed number, STOP: do not draw.
TEXT = {   # (frame, arm): (value printed in the thesis, measured value here)
    ("expression", "reference"):     ("0.576", E["reference"]),
    ("expression", "control_copy"):  ("0.504", E["control_copy"]),
    ("expression", "linear"):        ("0.500", E["linear"]),
    ("expression", "model"):         ("0.498", E["model"]),
    ("expression", "global_mean"):   ("0.180", E["global_mean"]),
    ("residual",   "reference"):     ("0.854", R["reference"]),
    ("residual",   "drug_lookup"):   ("0.913", R["drug_lookup"]),
    ("residual",   "model"):         ("0.537", R["model"]),
    ("residual",   "control_copy"):  ("0.506", R["control_copy"]),
    ("residual",   "random"):        ("0.501", R["random"]),
    ("residual",   "scramble_orth"): ("0.501", R["scramble_orth"]),
    ("residual",   "generic"):       ("0.500", R["generic"]),
    ("common",     "model"):         ("0.549", CS["model"]),
    ("common",     "reference"):     ("0.857", CS["ceiling"]),
}
bad = [(k, p, v) for k, (p, v) in TEXT.items() if f"{v:.3f}" != p]
if bad:
    raise SystemExit("THESIS/DATA DISAGREEMENT, refusing to draw: %r" % (bad,))
assert N_EXPR == 606 and N_RESID == 1394 and CS["n"] == 1192

COVERAGE = CS["coverage_model"]              # 13.9%, printed in fig:ladder
assert f"{100*COVERAGE:.1f}" == "13.9", COVERAGE

# ------------------------------------------------------------- the geometry
# Body text block is 142 mm. Chance sits at X0 in BOTH scales and each frame's
# own reference sits at X1 in BOTH scales. That is the entire construction: the
# two frames are forced onto different rulers, and the ratio of those rulers is
# the argument.
X0, X1 = 3.60, 10.60          # cm
SPAN   = X1 - X0              # 7.00 cm

s_e = SPAN / (E["reference"] - CHANCE)      # cm per unit NIR, expression frame
s_r = SPAN / (R["reference"] - CHANCE)      # cm per unit NIR, residual frame
STRETCH = s_e / s_r                          # how much more the upper is stretched

def xe(v): return X0 + (v - CHANCE) * s_e
def xr(v): return X0 + (v - CHANCE) * s_r

# The global mean is far outside the drawable range of the expression ruler. It
# is drawn on a broken stub and the break is labelled with exactly how much it
# hides, in units of that frame's own chance-to-reference distance.
BREAK_MULT = (CHANCE - E["global_mean"]) / (E["reference"] - CHANCE)

CAL = 0.05                                   # the calibration bar, in NIR
cal_e, cal_r = CAL * s_e, CAL * s_r

# Vertical plan. Sized against a real build: \footnotesize sets on a 0.42 cm
# line here and \scriptsize on 0.36 cm. The first draft guessed 0.30 and
# collided in five places. Do not tighten without rebuilding
# figs/_harness_twoscales.tex and looking at the page.
YU, YL   = 8.85, 3.60                        # the two scale lines
YRULE    = 6.05                              # the hard rule between the frames
XL, XR   = 0.00, 14.20                       # exactly the 142 mm body block
# Where STROKES may end. pgf adds half a line width to the bounding box, so a
# 1.0 pt rule ending exactly on 14.20 cm pushed the figure 0.5 pt past the block.
XLD, XRD = 0.03, 14.17
YTOP     = 11.30                             # top of the two guide heads
GUIDETOP, GUIDEBOT = 10.50, 3.25             # the two shared vertical anchors
XGM      = 1.475                             # global mean, on the broken stub

MU, ML = xe(E["model"]), xr(R["model"])
def cut(y):                                  # x where the model connector is at y
    return MU + (YU - y) / (YU - YL) * (ML - MU)
FALSE_DELTA = R["model"] - E["model"]        # the subtraction the figure strikes out

f = lambda v: f"{v:.4f}"

# ---------------------------------------------------------------- emit .tex
L = []
A = L.append
HEADER = r"""% ===========================================================================
% two-scales.tex -- v2. fig:two-scales. GENERATED by two-scales_build.py from
% RESULTS_cluster/nir_sameplate.json and RESULTS_cluster/re_v3.json. Do not edit
% by hand: re-run the builder. Every number is listed in two-scales.json.
%
% BODY WIDTH, 142 mm exactly. Not \begin{fullwidth}. Contains a tikzpicture and
% nothing else, so a section \input's it inside its own figure float -- the
% pattern of figs/comparators.tex and figs/frames.tex, not of fig-frame.tex.
% The float wrapper and caption are at the foot of this file, commented out,
% ready to paste into Sections/04d-baselines.tex.
%
% THE CONSTRUCTION. Chance (0.50) is drawn at the same x in both scales and each
% frame's OWN within-well split-half precision reference is drawn at the same x
% as the other's. Nothing else is aligned. Because the two chance-to-reference
% distances are @@DE@@ and @@DR@@ of NIR, the two rulers come out @@ST@@ times
% apart, and that differential stretch is the whole argument. The calibration
% bars, flush to the right margin, make it measurable rather than merely felt.
%
% REGISTER, inherited from fig:ladder: mono for the identifiers that name a
% column in the results JSON (\texttt{generic}, \texttt{random},
% \texttt{control\_copy}, \texttt{orth}, \texttt{drug\_lookup}); roman for arms
% the thesis names in prose (model, linear map, control-copy, global mean,
% within-well reference). Accent is reserved for the model and for chance.
%
% WHAT IS DELIBERATELY ABSENT: every interval (no arm here is quoted with one at
% the pooled level, so all markers are bare); the identifiable-stratum
% expression rows 0.768 / 0.766, which are a different support; and the three
% supports of fig:ladder. The one derived quantity drawn, @@FD@@, appears ONLY
% as the thing being struck out.
% ==========================================================================="""
for _k, _v in (("@@DE@@", f"{E['reference']-CHANCE:.4f}"),
               ("@@DR@@", f"{R['reference']-CHANCE:.4f}"),
               ("@@ST@@", f"{STRETCH:.2f}"),
               ("@@FD@@", f"{FALSE_DELTA:+.3f}")):
    HEADER = HEADER.replace(_k, _v)
A(HEADER)

A(r"""\begin{tikzpicture}[
  x=1cm, y=1cm,
  scaleline/.style={ink, line width=0.7pt, line cap=round},
  guide/.style={rulegrey, densely dotted, line width=0.3pt},
  chanceguide/.style={accent!65, densely dotted, line width=0.45pt},
  leader/.style={rulegrey!85, line width=0.25pt},
  headrule/.style={rulegrey!60, line width=0.3pt},
  hardrule/.style={ink, line width=1.0pt},
  connector/.style={ink!60, densely dashed, line width=0.7pt},
  calib/.style={quietink, line width=0.9pt, line cap=butt},
  fh/.style={font=\sffamily\footnotesize\bfseries, text=accent, inner sep=0pt},
  fhs/.style={font=\footnotesize, text=quietink, inner sep=0pt},
  gh/.style={anchor=north, font=\sffamily\scriptsize, text=accent,
             align=center, inner sep=1pt},
  val/.style={font=\footnotesize, text=ink, align=center, inner sep=1.5pt},
  arm/.style={font=\scriptsize, text=quietink, inner sep=1.5pt},
  armM/.style={font=\footnotesize\bfseries, text=accent, inner sep=1.5pt},
  note/.style={font=\scriptsize, text=quietink, align=flush left, inner sep=0pt},
  noteR/.style={font=\scriptsize, text=quietink, align=flush right, inner sep=0pt},
]""")

A("\n% ---- the bounding box IS the body block; nothing may stick out of it")
A(r"\path (%s,1.60) rectangle (%s,%s);" % (f(XL), f(XR), f(YTOP)))

# ---------------- the two shared anchors, named once, at the top
A("\n% ================ the two alignment anchors")
A(r"\node[gh] at (%s,%s) {chance $=0.50$\\ the one value the two frames share};"
  % (f(X0), f(YTOP)))
A(r"\node[gh] at (%s,%s) {within-well split-half reference\\"
  r" each frame's own, drawn at the same place};" % (f(X1), f(YTOP)))
A(r"\draw[chanceguide] (%s,%s) -- (%s,%s);" % (f(X0), f(GUIDETOP), f(X0), f(GUIDEBOT)))
A(r"\draw[guide]       (%s,%s) -- (%s,%s);" % (f(X1), f(GUIDETOP), f(X1), f(GUIDEBOT)))

# ---------------- upper frame -------------------------------------------
A("\n% ================ UPPER SCALE -- expression frame")
A(r"\node[fh,  anchor=west] at (%s,10.30) {expression frame};" % f(XL))
A(r"\node[fhs, anchor=east] at (%s,10.30) {within plate, tier 2, $n=\num{%d}$};"
  % (f(XR), N_EXPR))
A(r"\draw[headrule] (%s,10.00) -- (%s,10.00);" % (f(XLD), f(XRD)))
A(r"\draw[scaleline] (2.35,%s) -- (12.30,%s);" % (f(YU), f(YU)))
A(r"\draw[scaleline] (1.15,%s) -- (1.80,%s);   %% the broken stub" % (f(YU), f(YU)))
A(r"\draw[ink, line width=0.5pt] (1.96,%s) -- (2.10,%s);" % (f(YU-0.13), f(YU+0.13)))
A(r"\draw[ink, line width=0.5pt] (2.08,%s) -- (2.22,%s);" % (f(YU-0.13), f(YU+0.13)))
A(r"\fill[quietink] (%s,%s) circle (1.4pt);   %% global mean, off scale" % (f(XGM), f(YU)))
A(r"\fill[quietink] (%s,%s) circle (1.4pt);   %% linear map" % (f(xe(E["linear"])), f(YU)))
A(r"\fill[quietink] (%s,%s) circle (1.4pt);   %% control-copy"
  % (f(xe(E["control_copy"])), f(YU)))
A(r"\fill[ink]      (%s,%s) circle (1.6pt);   %% within-well reference" % (f(X1), f(YU)))
A(r"\fill[accent]   (%s,%s) circle (1.9pt);   %% model" % (f(MU), f(YU)))
# chance drawn AFTER the dots: the linear map sits 0.4 mm off it and hid it
A(r"\draw[accent!85, line width=0.5pt] (%s,%s) -- (%s,%s);"
  % (f(X0), f(YU-0.11), f(X0), f(YU+0.11)))

A(r"\node[val, anchor=south] at (%s,%s) {global mean\\ $%.3f$};"
  % (f(XGM), f(YU + 0.15), E["global_mean"]))
A(r"\node[val, anchor=south] at (%s,%s) {$%.3f$};"
  % (f(X1), f(YU + 0.15), E["reference"]))

# fanned labels: model, linear map and control-copy sit inside 5.3 mm of scale
rows = [(8.53, "armM", r"model\;$%.3f$"        % E["model"],        MU),
        (8.18, "arm",  r"linear map\;$%.3f$"   % E["linear"],       xe(E["linear"])),
        (7.83, "arm",  r"control-copy\;$%.3f$" % E["control_copy"], xe(E["control_copy"]))]
for y, sty, txt, x in rows:
    A(r"\draw[leader] (%s,%s) -- (3.20,%s);" % (f(x), f(YU - 0.07), f(y)))
    A(r"\node[%s, anchor=east] at (3.16,%s) {%s};" % (sty, f(y), txt))

A(r"\node[note, anchor=north west, text width=3.30cm] at (0.00,7.58)"
  r" {Scale broken at the left: $%.3f$ lies $%.1f\times$ this frame's"
  r" chance-to-reference distance below chance.};" % (E["global_mean"], BREAK_MULT))

# Upper calibration bar, flush right. Its twin below shares this right edge, so
# the two rulers can be compared by eye without setting a number from one frame
# beside a number from the other.
A(r"\draw[calib] (%s,7.10) -- (%s,7.10);" % (f(XRD - cal_e), f(XRD)))
A(r"\draw[calib] (%s,7.02) -- (%s,7.18);" % (f(XRD - cal_e), f(XRD - cal_e)))
A(r"\draw[calib] (%s,7.02) -- (%s,7.18);" % (f(XRD), f(XRD)))
A(r"\node[noteR, anchor=north east] at (%s,6.96) {$0.05$ of $\NIR$ on this scale};"
  % f(XR))

# ---------------- the connector, and the callout that forbids it ----------
# HISTORY, so this is not undone by a later hand. The first version struck the
# equation with a 0.5 pt rule from (eq.west) to (eq.east) and stood the sentence
# loose on the page with a leader pointing back at the connector. It failed the
# only test that matters: the author, who knew what it meant, read it as a stray
# line. Two reasons. (a) A node's vertical centre in a digits-only math box sits
# almost exactly on the math axis, so the rule ran flush into the "-" and the
# "=" and the three glyphs fused into one long horizontal stroke. (b) Two
# dashed connectors and a leader stub crossed the same 2 cm of white space, so
# any additional short horizontal line read as more of the same debris.
#
# The replacement uses four independent signals, no one of which is load-bearing
# alone: the forbidden arithmetic is fenced in its own bordered panel, well
# clear of the connectors; the panel is titled, in words, with an imperative;
# the equation carries a filled no-entry badge; and the strike is heavy enough
# to read as an obliteration and OVERHANGS the equation at both ends, which is
# what stops a rule reading as an operator. The same badge is stamped where the
# connector meets the scope-limit rule, so the panel and the severed connector
# are visibly one statement.
#
# ON THE STRIKE WEIGHT AND POSITION, both of which were arrived at by printing
# six variants side by side (figs/_strike_probe.tex, since deleted). It stays
# CENTRED. Raising it by 1.1 pt to clear the "-" and the "=" opens a hairline of
# white under the rule and the pair then reads as an equals sign: the line
# printed "0.537 = 0.498 = +0.039", which is worse than the defect being fixed.
# Centred, the rule swallows both operators instead of pairing with them, and at
# 1.6 pt -- three times the old 0.5 pt, and heavier than every other stroke in
# the figure -- it can only be read as a deliberate score-through. Do not
# lighten it and do not move it off centre.
CX  = cut(YRULE)                      # where the connector meets the hard rule
BOXX, BOXW = 4.30, 4.70               # panel left edge; its TEXT width
BOXPAD = 0.17                         # panel inner sep, both axes

def badge(cx, cy, r, lw):
    """Filled no-entry disc with a knocked-out cross. Reads at 6 pt."""
    a = 0.50 * r
    A(r"\fill[ink] (%s,%s) circle (%.3f);" % (f(cx), f(cy), r))
    A(r"\draw[white, line width=%.1fpt, line cap=round] (%s,%s) -- (%s,%s);"
      % (lw, f(cx - a), f(cy - a), f(cx + a), f(cy + a)))
    A(r"\draw[white, line width=%.1fpt, line cap=round] (%s,%s) -- (%s,%s);"
      % (lw, f(cx - a), f(cy + a), f(cx + a), f(cy - a)))

A("\n% ================ the subtraction the figure exists to forbid")
# The upper connector crosses the linear-map and control-copy leaders about
# 6 mm below the scale line and tangled with them. A white casing over just that
# stretch puts the connector unambiguously in front; it stops above the chance
# guide, which the connector is content to cross as an equal.
A(r"\draw[white, line width=2.2pt] (%s,%s) -- (%s,%s);"
  % (f(MU), f(YU - 0.11), f(cut(7.75)), f(7.75)))
A(r"\draw[connector] (%s,%s) -- (%s,%s);"
  % (f(MU), f(YU - 0.11), f(cut(YRULE + 0.19)), f(YRULE + 0.19)))
A(r"\draw[connector] (%s,%s) -- (%s,%s);"
  % (f(cut(YRULE - 0.19)), f(YRULE - 0.19), f(ML), f(YL + 0.11)))

# The panel sits above the hard rule and to the right of the connector: its left
# edge clears the connector by 4 mm and its right edge clears the upper
# calibration bar by 2 mm. Nothing crosses it.
A(r"\node[draw=ink, line width=0.5pt, rounded corners=1.5pt, fill=panelbg,"
  r" inner sep=%.2fcm, text width=%.2fcm, anchor=south west, font=\scriptsize,"
  r" text=quietink, align=flush left] (nosub) at (%s,%s)"
  r" {\tikz[baseline=(eq.base)]{"
  r" \node[inner sep=0pt, font=\footnotesize, text=ink, anchor=west]"
  r" (eq) at (0.50,0) {$%.3f - %.3f = %+.3f$};"
  r" \draw[ink, line width=1.6pt]"
  r" ([xshift=-0.10cm]eq.west) -- ([xshift=0.10cm]eq.east);"
  r" \fill[ink] (0.15,0.075) circle (0.130);"
  r" \draw[white, line width=0.9pt, line cap=round] (0.085,0.010) -- (0.215,0.140);"
  r" \draw[white, line width=0.9pt, line cap=round] (0.085,0.140) -- (0.215,0.010);"
  r" }\\[3pt]"
  r" is the subtraction the shared chance value invites, and it is not a"
  r" quantity: two different frames, measured against two different"
  r" references.};"
  % (BOXPAD, BOXW, f(BOXX), f(YRULE + 0.17),
     R["model"], E["model"], FALSE_DELTA))
# The title rides ON the panel border, so it costs no height and cannot be read
# as a first line of the sentence. Words, not marks, carry the instruction.
A(r"\node[fill=panelbg, inner xsep=3pt, inner ysep=1pt, anchor=west,"
  r" font=\sffamily\scriptsize\bfseries, text=ink]"
  r" at ([xshift=0.30cm]nosub.north west) {Do not subtract};")

# ---------------- the hard rule -----------------------------------------
A("\n% ================ the scope limit, drawn as a rule and not as a caption")
A(r"\draw[hardrule] (%s,%s) -- (%s,%s);" % (f(XLD), f(YRULE), f(XRD), f(YRULE)))
A(r"\node[anchor=north west, font=\sffamily\scriptsize, text=ink, align=flush left,"
  r" inner sep=0pt, text width=9.90cm] at (4.30,%s) {\textbf{Scope limit.} No number"
  r" in one frame may be set beside a number in the other.};" % f(YRULE - 0.13))
# Stamped AFTER the rule so it sits on top of it: the connector is not merely
# interrupted at the scope limit, it is stopped there. Same mark as the panel.
A("% the connector is stopped at the rule, with the panel's own mark")
badge(CX, YRULE, 0.19, 1.0)

# ---------------- lower frame -------------------------------------------
A("\n% ================ LOWER SCALE -- residual frame")
A(r"\node[fh,  anchor=west] at (%s,5.05) {residual frame};" % f(XL))
A(r"\node[fhs, anchor=east] at (%s,5.05) {cell-line sets, $n=\num{%d}$};"
  % (f(XR), N_RESID))
A(r"\draw[headrule] (%s,4.75) -- (%s,4.75);" % (f(XLD), f(XRD)))
A(r"\draw[scaleline] (2.35,%s) -- (12.30,%s);" % (f(YL), f(YL)))

for k in ("generic", "scramble_orth", "random", "control_copy"):
    A(r"\fill[quietink] (%s,%s) circle (1.4pt);   %% %s" % (f(xr(R[k])), f(YL), k))
A(r"\fill[ink]    (%s,%s) circle (1.6pt);   %% within-well reference" % (f(X1), f(YL)))
A(r"\fill[ink]    (%s,%s) circle (1.6pt);   %% drug_lookup"
  % (f(xr(R["drug_lookup"])), f(YL)))
A(r"\fill[accent] (%s,%s) circle (1.9pt);   %% model" % (f(ML), f(YL)))
# chance drawn AFTER the dots: generic is exactly 0.500 and sat on top of it
A(r"\draw[accent!85, line width=0.5pt] (%s,%s) -- (%s,%s);"
  % (f(X0), f(YL-0.11), f(X0), f(YL+0.11)))

A(r"\draw[leader] (%s,%s) -- (4.46,%s);" % (f(ML), f(YL + 0.09), f(YL + 0.24)))
A(r"\node[val, anchor=south west, align=left, text=accent,"
  r" font=\footnotesize\bfseries] at (4.44,%s) {model\\ $%.3f$};"
  % (f(YL + 0.18), R["model"]))
A(r"\node[val, anchor=south] at (%s,%s) {$%.3f$};" % (f(X1), f(YL + 0.15), R["reference"]))
A(r"\draw[leader] (%s,%s) -- (11.98,%s);"
  % (f(xr(R["drug_lookup"])), f(YL + 0.09), f(YL + 0.24)))
A(r"\node[arm, anchor=south west, align=left, text=ink] at (11.96,%s)"
  r" {\texttt{drug\_lookup}\\ $%.3f$};" % (f(YL + 0.18), R["drug_lookup"]))

# the four-arm blur: one leader off the cluster, one spine, four labels
A(r"\draw[leader] (%s,%s) -- (3.44,3.26);"
  % (f(xr(R["control_copy"]) - 0.03), f(YL - 0.09)))
A(r"\draw[leader] (3.40,3.26) -- (3.40,2.29);")
srows = [(3.26, r"\texttt{generic}\;$%.3f$"        % R["generic"]),
         (2.94, r"scramble, \texttt{orth}\;$%.3f$" % R["scramble_orth"]),
         (2.62, r"\texttt{random}\;$%.3f$"         % R["random"]),
         (2.30, r"\texttt{control\_copy}\;$%.3f$"  % R["control_copy"])]
for y, txt in srows:
    A(r"\draw[leader] (3.40,%s) -- (3.34,%s);" % (f(y), f(y)))
    A(r"\node[arm, anchor=east] at (3.30,%s) {%s};" % (f(y), txt))

# the 13.9% annotation, with the support it is actually computed on
A(r"\draw[leader] (%s,%s) -- (4.66,3.30);" % (f(ML + 0.04), f(YL - 0.09)))
A(r"\node[note, anchor=north west, text width=5.50cm] at (4.70,3.36)"
  r" {The $%.1f\%%$ margin this chapter quotes for the model (\cref{fig:ladder})"
  r" is computed on the $n=\num{%d}$ common support, where the model is $%.3f$"
  r" against a reference of $%.3f$. It is not a measurement of the $n=\num{%d}$"
  r" gap drawn here.};"
  % (100 * COVERAGE, CS["n"], CS["model"], CS["ceiling"], N_RESID))

# lower calibration bar, flush right, sharing the upper bar's right edge
A(r"\draw[calib] (%s,3.00) -- (%s,3.00);" % (f(XRD - cal_r), f(XRD)))
A(r"\draw[calib] (%s,2.92) -- (%s,3.08);" % (f(XRD - cal_r), f(XRD - cal_r)))
A(r"\draw[calib] (%s,2.92) -- (%s,3.08);" % (f(XRD), f(XRD)))
A(r"\node[noteR, anchor=north east] at (%s,2.86) {$0.05$ of $\NIR$ on this scale};"
  % f(XR))
A(r"\node[noteR, anchor=north east, text width=3.85cm] at (%s,2.44)"
  r" {The two rulers differ by $%.1f\times$.};" % (f(XR), STRETCH))

A(r"\end{tikzpicture}")

# ---- the float wrapper and caption, ready to paste into the section --------
CAPTION = r"""%
% ---------------------------------------------------------------------------
% READY TO PASTE into Sections/04d-baselines.tex, at sec:res-summary, just after
% the paragraph "The same chance value, two different references".
%
% \begin{figure}[htbp]
% \centering
% \input{figs/two-scales}
% \caption[Two frames, one chance value]{\textbf{The two frames share a chance
% value and nothing else, and the shared value is exactly what makes the mistake
% easy.} Each scale is stretched so that chance falls at the same place in both
% and so that each frame's own within-well split-half precision reference falls
% at the same place as the other's. Nothing else is aligned, so the two frames
% carry different rulers --- the same $0.05$ of $\NIR$ is @@ST@@ times longer
% above than below --- and that differential stretch is the figure.
% \emph{Upper:} the expression frame, within plate, tier 2, $n = \num{606}$,
% where the model scores $0.498$ against a reference of $0.576$ and is matched by
% a drug-agnostic linear map ($0.500$) and a control-copy ($0.504$); the scale is
% broken at the left because the global mean, $0.180$, lies $4.2\times$ that
% frame's chance-to-reference distance below chance. \emph{Lower:} the residual
% frame, cell-line sets, $n = \num{1394}$, where the model scores $0.537$ against
% a reference of $0.854$, a training-only per-drug lookup reaches $0.913$ above
% that reference, and \texttt{generic}, \texttt{orth}, \texttt{random} and
% \texttt{control\_copy} all fall within $0.006$ of chance, one mark at this
% scale. Both references are within-well split-half precision figures --- two
% disjoint halves of one treated well --- and not biological replicates
% (\cref{sec:methods-data}). Chance is $0.50$ in both under the exchangeability
% condition audited in \cref{sec:res-metrics}, and because it is the one value
% the two frames share it is exactly what makes the struck-out subtraction in the
% middle of the figure easy to perform: $0.537$ and $0.498$ are not two attempts
% at one task, and the difference between them is neither an improvement nor a
% decline. Every value is a pooled mean drawn as a bare marker; no arm shown here
% is quoted with an interval at the pooled level, so none is drawn
% (\cref{tab:allarms}). The $13.9\%$ margin annotated below is computed on the
% $n = \num{1192}$ common support and is not a measurement of the
% $n = \num{1394}$ gap drawn here.}
% \label{fig:two-scales}
% \end{figure}
% ---------------------------------------------------------------------------"""
A(CAPTION.replace("@@ST@@", "%.1f" % STRETCH))

tex = "\n".join(L) + "\n"
with open(os.path.join(HERE, "two-scales.tex"), "w", encoding="utf-8") as fh:
    fh.write(tex)

# ---------------------------------------------------------------- emit .json
rec = {
  "figure": "fig:two-scales",
  "file": "thesis_v2/figs/two-scales.tex",
  "generated_by": "thesis_v2/figs/two-scales_build.py",
  "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
  "width_mm": 142.0,
  "note": ("Two horizontal scales, one per frame. Chance (0.50) is drawn at the "
           "same x in both and each frame's own within-well split-half precision "
           "reference is drawn at the same x as the other's, which forces the two "
           "scales onto different rulers. No interval is drawn: none of these arms "
           "is quoted with one at the pooled level."),
  "geometry": {"x_chance_cm": X0, "x_reference_cm": X1, "span_cm": SPAN,
               "cm_per_NIR_expression": s_e, "cm_per_NIR_residual": s_r,
               "stretch_ratio_expression_over_residual": STRETCH,
               "calibration_bar_NIR": CAL,
               "calibration_bar_cm_expression": cal_e,
               "calibration_bar_cm_residual": cal_r,
               "expression_break_multiple_of_chance_to_reference": BREAK_MULT},
  "expression_frame": {
     "source": "RESULTS_cluster/nir_sameplate.json"
               " .tiers.tier2_unseen_drugs.agg.<arm>_nir_expr",
     "scope": "within plate, tier 2, calibrated metric (NIR, expression)",
     "n": N_EXPR,
     "n_source": "nir_sameplate.json .tiers.tier2_unseen_drugs.n_drugs",
     "thesis_cross_check": "Sections/04d-baselines.tex tab:allarms lines 801-808;"
                           " Sections/04a-ruler.tex lines 441-446 and 688-690",
     "arms": [
       {"label": "within-well split-half reference", "key": "ceiling_nir_expr",
        "measured": E["reference"], "drawn_as": "0.576", "x_cm": X1},
       {"label": "control-copy", "key": "control_nir_expr",
        "measured": E["control_copy"], "drawn_as": "0.504",
        "x_cm": xe(E["control_copy"])},
       {"label": "linear map", "key": "linear_nir_expr",
        "measured": E["linear"], "drawn_as": "0.500", "x_cm": xe(E["linear"])},
       {"label": "model", "key": "model_nir_expr",
        "measured": E["model"], "drawn_as": "0.498", "x_cm": MU},
       {"label": "global mean", "key": "mean_nir_expr",
        "measured": E["global_mean"], "drawn_as": "0.180",
        "x_cm": XGM, "off_scale": True,
        "off_scale_note": ("drawn on a broken stub; %.4f x this frame's "
                           "chance-to-reference distance below chance"
                           % BREAK_MULT)}]},
  "residual_frame": {
     "source": "RESULTS_cluster/re_v3.json .means.<arm>",
     "scope": "cell-line sets, calibrated metric (NIR on the signed residual vector)",
     "n": N_RESID,
     "n_source": "re_v3.json .means.vs_baselines.control_copy.n",
     "thesis_cross_check": "Sections/04d-baselines.tex tab:allarms lines 810-825"
                           " and fig:ladder lines 122-131",
     "arms": [
       {"label": "drug_lookup", "measured": R["drug_lookup"], "drawn_as": "0.913",
        "x_cm": xr(R["drug_lookup"])},
       {"label": "within-well split-half reference", "key": "ceiling",
        "measured": R["reference"], "drawn_as": "0.854", "x_cm": X1},
       {"label": "model", "measured": R["model"], "drawn_as": "0.537", "x_cm": ML},
       {"label": "control_copy", "measured": R["control_copy"], "drawn_as": "0.506",
        "x_cm": xr(R["control_copy"])},
       {"label": "random", "measured": R["random"], "drawn_as": "0.501",
        "x_cm": xr(R["random"])},
       {"label": "scramble, orth", "key": "scramble_orth",
        "measured": R["scramble_orth"], "drawn_as": "0.501",
        "x_cm": xr(R["scramble_orth"])},
       {"label": "generic", "measured": R["generic"], "drawn_as": "0.500",
        "x_cm": xr(R["generic"])}]},
  "annotations": {
     "chance": {"value": CHANCE, "drawn_at_x_cm": X0,
                "basis": "exchangeability condition audited in sec:res-metrics"},
     "coverage_common_support": {
        "value_pct": 100 * COVERAGE, "drawn_as": "13.9%",
        "source": "re_v3.json .means.common_support.coverage_model",
        "support_n": CS["n"], "model": CS["model"], "reference": CS["ceiling"],
        "drawn_how": ("annotation text only; explicitly NOT a measured distance "
                      "on the n=1394 scale that is drawn"),
        "thesis_cross_check": "fig:ladder; Sections/04d-baselines.tex line 848"},
     "struck_subtraction": {
        "expression": "%.3f - %.3f = %+.3f" % (R["model"], E["model"], FALSE_DELTA),
        "value": FALSE_DELTA,
        "status": ("DERIVED AND STRUCK OUT. Drawn only as the misreading the "
                   "figure exists to block. It is not a result and appears "
                   "nowhere in the thesis as a quantity.")},
     "cluster_spread_residual": {
        "value": R["control_copy"] - CHANCE, "drawn_as": "0.006 (caption only)",
        "meaning": ("largest distance from chance among generic / scramble_orth "
                    "/ random / control_copy on the residual scale")}},
  "deliberately_not_drawn": [
     "any interval: no arm here is quoted with one at the pooled level",
     "the identifiable-stratum expression rows 0.768 / 0.766 (a different support)",
     "the n=1192 and n=218 supports of fig:ladder",
     "scramble near (0.550) and scramble opposite (0.423)"],
  "banned_term_check": "the string 'noise ceiling' does not appear in the figure"
}
with open(os.path.join(HERE, "two-scales.json"), "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=2)

assert "noise ceiling" not in tex.lower()
print("wrote two-scales.tex (%d blocks) and two-scales.json" % len(L))
print("  expression ruler %.3f cm/NIR   residual ruler %.3f cm/NIR   stretch %.3fx"
      % (s_e, s_r, STRETCH))
print("  break multiple %.4f   struck delta %+.4f   coverage %.4f"
      % (BREAK_MULT, FALSE_DELTA, COVERAGE))
print("  calibration bars: %.4f cm (expr) vs %.4f cm (resid)" % (cal_e, cal_r))

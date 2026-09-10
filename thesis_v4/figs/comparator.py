"""fig:comparator -- the swap comparator is calibrated, and most of the opposite-stratum gap is it.

Two things the prose of sec:res-generalisation states but cannot show.

(a) The comparator is CALIBRATED, not neutral. The scrambled model's NIR falls monotonically --
    0.5504, 0.5006, 0.4228 -- as the swapped-in partner's true residual moves from aligned
    (cos = +0.219) through orthogonal (+0.000) to anti-aligned (-0.202) with the target. So `near`
    is a partly correct answer and `opposite` is an actively wrong one. `orth` sits on chance, which
    makes it the least contaminated comparator available -- NOT an independently established null,
    because its stratum was defined using true residual geometry in the first place.

(b) The decomposition gap = (NIR_model - 0.5) + (0.5 - NIR_scramble). Pooled over all 1394 scored
    conditions the model's own advantage over chance is +0.0367 for every stratum, because the model
    arm is the same arm; the strata differ ONLY in the second term. On `opposite` that second term is
    +0.0772, i.e. 68% of the +0.1139 gap that arm reports. The thesis says "68%" in one clause
    (04c-reencoding.tex:744) and never shows it.

NUMBERS. Everything comes from RESULTS_cluster; nothing is typed in. Cross-checked against the
thesis text before drawing (see verify_against_thesis() below, which is asserted at run time):
    04c-reencoding.tex:730-734  "0.550, 0.501 and 0.423", cos "+0.22", "0.00", "-0.20"
    04c-reencoding.tex:744-745  "+0.0367", "+0.0772", "68%", "+0.1139"
    04c-reencoding.tex:658      orth gap \\ci{+0.0361}{+0.0159}{+0.0564}
All agree with re_v3.json to the printed precision.

DESIGN NOTES, each with its reason.

  * PANEL (a) CHANCE RULE IS SOLID, NOT DOTTED. The figure brief asked for a dotted rule at 0.50.
    style_v2.chance_line() draws it solid ("Thin, solid, black. Never dashed, never omitted") and
    that convention is document-wide: the 0.50 rule is the one line in this thesis that always
    looks the same. The cos = 0 rule IS dotted, so the two are still told apart -- by texture, and
    the dotted one is the one that is merely a coordinate.

  * THE GUIDE IN (a) IS THE CHORD JOINING THE TWO EXTREME MARKS, not a regression. A fitted line
    through three points would be an estimate the data cannot support and the caption would have to
    disown it. A chord is defined entirely by two plotted points, so it asserts nothing. `orth`
    lands 0.016 ABOVE the chord, which is visible and is honest: "almost linearly", not "linearly".

  * PANEL (b) SPLITS EACH STRATUM'S SPAN INTO TWO TIERS rather than one literally-stacked bar.
    A single span from 0.5367 down to the partner score, split at 0.50, is only a stack for
    `opposite`: on `near` the partner scores 0.5504, ABOVE both chance and the model, so its two
    terms run in OPPOSITE directions and a stacked bar would have to hide a sign to stay stacked.
    Two tiers per stratum, both starting at the chance rule, keep the sign visible: length is the
    size of a term, side of the rule is its sign, and gap = accent term + grey term with sign.
    On `opposite` the two tiers still abut at 0.50 and read as the intended single split span.

  * NO LEGEND. The two tiers are labelled once, on the top row, where the region left of the chance
    rule is empty on that row.

  * THE (b) HEADLINE IS RIGHT-ALIGNED TO THE CHANCE RULE AND CARRIES NO LEADER. Centred on the
    grey tier -- the obvious placement -- the sentence is wider than the tier it sits under (133pt
    of type against a 93pt tier) and the solid 0.50 rule ran straight through it, splitting
    "comparator" into "compa|rator". A white halo behind the text clears the collision only by
    punching a hole through the 0.50 rule, the bottom spine and two x-ticks, and style_v2's one
    inviolable convention is that the 0.50 rule is never dashed and never omitted. So the sentence
    is right-aligned and stops just short of the rule, ending exactly where the grey tier it
    describes ends: that alignment IS the referent. The old leader stub is gone with it. A leader's
    tail leaves the TEXT BOX CENTRE, so it was vertical only while the text was centred under the
    tier; right-aligned, the same stub runs diagonally and lands at 69% across the sentence, i.e.
    it points into the middle of a word. The ~6pt between tier and type is too little to route it
    around, and a stray hair under a bar reads as an errant mark, not as a leader.

  * THE INTERVALS ARE ci_two_way. re_v3.json also stores a one-way `ci`; the thesis quotes the
    two-way form everywhere (04c:658 matches ci_two_way exactly), so the one-way variants are not
    drawn. Mixing the two would put a narrow interval beside a wide one for the same quantity.

Run:  python thesis_v2/figs/comparator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

MINUS = "−"   # true minus, to match matplotlib's own tick labels
STRATA = ["near", "orth", "opposite"]


def signed(v, nd=4):
    """+0.0367 / -0.0137 with a real minus sign, never a hyphen."""
    return ("+" if v >= 0 else MINUS) + ("%.*f" % (nd, abs(v)))


# --------------------------------------------------------------------------- numbers
audit = S.load("scramble_stratum_audit_v4.json")
rev3 = S.load("re_v3.json")

n_records = audit["n_records"]                       # 1394
model_nir = rev3["means"]["model"]                   # 0.5367012554202398

# panel (a): the comparator's own score against the partner's true alignment
partner_cos = {k: audit["comparator_neutrality"][k]["mean_partner_cos"] for k in STRATA}
partner_nir = {k: audit["comparator_neutrality"][k]["mean_nir"] for k in STRATA}

# panel (b): the pooled gap per stratum and its two-way clustered interval
gap = {k: rev3["means"]["strata"][k]["gap"] for k in STRATA}
gap_ci = {k: rev3["means"]["strata"][k]["ci_two_way"] for k in STRATA}
gap_n = {k: rev3["means"]["strata"][k]["n"] for k in STRATA}
two_way_note = rev3["means"]["strata"]["opposite"]["two_way_note"]   # "42L/200W:two_way:t41"

model_over_chance = model_nir - S.CHANCE                          # +0.0367013
deficit = {k: S.CHANCE - partner_nir[k] for k in STRATA}          # the comparator term, signed
share_opposite = deficit["opposite"] / gap["opposite"]            # 0.6777 -> "68%"


# --------------------------------------------------------------------------- consistency
def verify_against_thesis():
    """Assert the drawn numbers against the values the thesis prints. STOP rather than draw.

    Every tuple is (what the figure will draw, what a thesis line prints, tolerance, where).
    A mismatch here means the figure and the text disagree, which is the one failure this
    project has already made three times.
    """
    checks = [
        (partner_nir["near"], 0.550, 5e-4, "04c-reencoding.tex:730 'is 0.550'"),
        (partner_nir["orth"], 0.501, 5e-4, "04c-reencoding.tex:730 '0.501'"),
        (partner_nir["opposite"], 0.423, 5e-4, "04c-reencoding.tex:730 '0.423'"),
        (partner_cos["near"], 0.22, 5e-3, "04c-reencoding.tex:731 'cos = +0.22'"),
        (partner_cos["orth"], 0.00, 5e-3, "04c-reencoding.tex:732 'orthogonal (0.00)'"),
        (partner_cos["opposite"], -0.20, 5e-3, "04c-reencoding.tex:732 'anti-aligned (-0.20)'"),
        (model_over_chance, 0.0367, 5e-5, "04c-reencoding.tex:744 '+0.0367'"),
        (deficit["opposite"], 0.0772, 5e-5, "04c-reencoding.tex:744 '+0.0772'"),
        (gap["opposite"], 0.1139, 5e-5, "04c-reencoding.tex:745 '+0.1139'"),
        (share_opposite, 0.68, 5e-3, "04c-reencoding.tex:744 '68%'"),
        (gap["orth"], 0.0361, 5e-5, "04c-reencoding.tex:658 '+0.0361'"),
        (gap_ci["orth"][0], 0.0159, 5e-5, "04c-reencoding.tex:658 lower '+0.0159'"),
        (gap_ci["orth"][1], 0.0564, 5e-5, "04c-reencoding.tex:658 upper '+0.0564'"),
        (n_records, 1394, 0, "04a-ruler.tex:503 'scores 1394 conditions'"),
    ]
    bad = [(a, b, w) for a, b, t, w in checks if abs(a - b) > t]
    if bad:
        raise SystemExit("STOP -- figure and thesis text disagree:\n" + "\n".join(
            "  drew %r, text says %r  (%s)" % (a, b, w) for a, b, w in bad))
    # internal identity: gap must be the sum of the two terms it is decomposed into
    for k in STRATA:
        assert abs(gap[k] - (model_over_chance + deficit[k])) < 1e-9, k
        assert gap_n[k] == n_records, k
    print("verify_against_thesis: %d checks pass" % len(checks))


verify_against_thesis()


# --------------------------------------------------------------------------- draw
family = S.apply()
print("face: %s" % S.font_report())
# the annotations use a real minus; a missing glyph would print as a box, silently
_name, _path = S.resolve_serif()
if _path and fm.get_font(_path).get_char_index(0x2212) == 0:
    raise SystemExit("STOP -- %s has no U+2212 MINUS glyph; the annotations would print boxes."
                     % _name)

FIG_W = S.TEXTWIDTH_IN            # 5.5906 in = 142 mm, drawn at final measure, never scaled
FIG_H = 4.72

fig = plt.figure(figsize=(FIG_W, FIG_H))
# Explicit axes rectangles rather than constrained layout: panel (b) needs a reserved right-hand
# column for the interval strings, and save() writes the EXACT canvas, so anything the layout
# engine does not know about would be clipped instead of accommodated.
AXL, AXW_A = 0.128, 0.787                     # panel (a): room at the right for the "chance" tag
AXW_B = 0.545                                 # panel (b): narrower, the rest is the CI column
ax_a = fig.add_axes([AXL, 0.6081, AXW_A, 0.3157])
ax_b = fig.add_axes([AXL, 0.1059, AXW_B, 0.2797])

# panel heads, as figure text so (b)'s can run wider than (b)'s axes
fig.text(AXL - 0.055, 0.9534, "(a) the comparator's score tracks the partner's true alignment",
         fontsize=10, color=S.INK, ha="left", va="baseline")
fig.text(AXL - 0.055, 0.4280, "(b) where each stratum's reported gap comes from",
         fontsize=10, color=S.INK, ha="left", va="baseline")

# ------------------------------------------------------------------ panel (a)
COMP = S.OKABE["orange"]                       # the scramble arm's global colour; all three strata
MARK = {"near": "^", "orth": "o", "opposite": "v"}   # shape encodes side of chance, not identity

# cos = 0 is a coordinate, so its rule is dotted; the 0.50 rule is solid, per style_v2
ax_a.axvline(0.0, color=S.RULEGREY, lw=0.7, ls=(0, (1, 3)), zorder=0)
S.chance_line(ax_a, "h")

# the guide: the chord between the two extreme marks. Not a fit -- it estimates nothing.
ax_a.plot([partner_cos["opposite"], partner_cos["near"]],
          [partner_nir["opposite"], partner_nir["near"]],
          color=S.RULEGREY, lw=0.7, zorder=1, solid_capstyle="butt")

LBL = {  # (text offset in points, ha) -- short leaders, no arrowheads
    "near":     ((-7, 9), "right"),
    "orth":     ((8, 9), "left"),
    "opposite": ((9, -2), "left"),
}
for k in STRATA:
    ax_a.plot([partner_cos[k]], [partner_nir[k]], marker=MARK[k], ms=6.0, ls="none",
              color=COMP, mec=COMP, zorder=3)
    off, ha = LBL[k]
    ax_a.annotate(k, xy=(partner_cos[k], partner_nir[k]), xytext=off,
                  textcoords="offset points", ha=ha, va="center", fontsize=9, color=S.INK,
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=S.RULEGREY,
                                  shrinkA=1.5, shrinkB=4.0))

ax_a.set_xlim(-0.25, 0.25)
ax_a.set_ylim(0.40, 0.60)
ax_a.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
ax_a.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])
ax_a.set_xlabel("mean partner residual cosine (partner's truth vs the target's)")
ax_a.set_ylabel("stratum mean NIR\n(residual frame)")

# ------------------------------------------------------------------ panel (b)
GREY_FILL = "#9A9A9A"        # the page's rule grey: quiet against the accent, solid so it reads
BH, OFF = 0.30, 0.175        # bar height and the two tiers' offsets from the row centre
XLO, XHI = 0.395, 0.578

S.chance_line(ax_b, "v")
for i, k in enumerate(STRATA):
    y_model, y_comp = i - OFF, i + OFF
    # tier 1, accent: the model's own advantage over chance. Identical on all three rows, because
    # it is the same model arm -- which is the point.
    ax_b.add_patch(Rectangle((S.CHANCE, y_model - BH / 2), model_nir - S.CHANCE, BH,
                             facecolor=S.ACCENT, edgecolor="none", zorder=2))
    # tier 2, quiet grey: the comparator's own displacement from chance, drawn on the side it
    # actually falls. Right of the rule it SUBTRACTS from the gap; left of it, it adds.
    ax_b.add_patch(Rectangle((min(S.CHANCE, partner_nir[k]), y_comp - BH / 2),
                             abs(partner_nir[k] - S.CHANCE), BH,
                             facecolor=GREY_FILL, edgecolor="none", zorder=2))
    # the partner's score, at the far end of its own bar
    if partner_nir[k] >= S.CHANCE:
        ax_b.annotate("%.4f" % partner_nir[k], xy=(partner_nir[k], y_comp), xytext=(4, 0),
                      textcoords="offset points", ha="left", va="center",
                      fontsize=9, color=S.QUIETINK)
    else:
        ax_b.annotate("%.4f" % partner_nir[k], xy=(partner_nir[k], y_comp), xytext=(-4, 0),
                      textcoords="offset points", ha="right", va="center",
                      fontsize=9, color=S.QUIETINK)
    # the gap and its two-way clustered interval, in the reserved right-hand column
    lo, hi = gap_ci[k]
    ax_b.annotate("%s  [%s, %s]" % (signed(gap[k]), signed(lo), signed(hi)),
                  xy=(1.035, i), xycoords=("axes fraction", "data"),
                  ha="left", va="center", fontsize=9, color=S.INK, annotation_clip=False)

# the two tiers named once, on the top row, in the region that row leaves empty
ax_b.annotate("model  %.4f" % model_nir, xy=(S.CHANCE, -OFF), xytext=(-5, 0),
              textcoords="offset points", ha="right", va="center", fontsize=9, color=S.ACCENT)
ax_b.annotate("comparator", xy=(S.CHANCE, OFF), xytext=(-5, 0),
              textcoords="offset points", ha="right", va="center", fontsize=9, color=S.QUIETINK)
ax_b.annotate("gap  [95% CI]", xy=(1.035, -0.72), xycoords=("axes fraction", "data"),
              ha="left", va="center", fontsize=9, color=S.QUIETINK, annotation_clip=False)

# the headline: on `opposite`, most of the reported gap is the grey tier. Right-aligned so it
# STOPS at the chance rule -- ending exactly where the grey tier it describes ends. Centring it
# on the tier is what put the solid 0.50 rule through the middle of the sentence; see the
# design note above.
ax_b.annotate("%d%% of this gap is the comparator" % round(share_opposite * 100),
              xy=(S.CHANCE - 0.004, 2.72),
              ha="right", va="center", fontsize=9, color=S.INK)

ax_b.set_xlim(XLO, XHI)
ax_b.set_ylim(2.92, -0.72)
ax_b.set_yticks(range(len(STRATA)))
ax_b.set_yticklabels(STRATA)
ax_b.set_xticks([0.42, 0.46, 0.50, 0.54])
ax_b.tick_params(axis="y", length=0)
ax_b.spines["left"].set_visible(False)
ax_b.set_xlabel(S.nir_label("residual"))

# --------------------------------------------------------------------------- write
drawn = {
    "_figure": "fig:comparator",
    "_sources": {
        "partner_cos, partner_nir, n_records":
            "RESULTS_cluster/scramble_stratum_audit_v4.json :: "
            "comparator_neutrality/<stratum>/{mean_partner_cos,mean_nir}, n_records",
        "model_nir":
            "RESULTS_cluster/re_v3.json :: means/model",
        "gap, ci":
            "RESULTS_cluster/re_v3.json :: means/strata/<stratum>/{gap,ci_two_way,n}",
    },
    "_interval_convention": "95%% two-way cluster-robust on cell line and well; re_v3 note %r "
                            "(42 cell lines, 200 treatment wells, Student t on 41 df). The "
                            "one-way `ci` field is NOT drawn." % two_way_note,
    "n_records": n_records,
    "chance": S.CHANCE,
    "panel_a": {
        "x_mean_partner_residual_cosine": {k: partner_cos[k] for k in STRATA},
        "y_stratum_mean_nir_residual_frame": {k: partner_nir[k] for k in STRATA},
        "guide_line": "chord joining the near and opposite marks; not a fit, estimates nothing",
        "orth_offset_above_chord": partner_nir["orth"] - (
            partner_nir["opposite"]
            + (partner_nir["near"] - partner_nir["opposite"])
            * (partner_cos["orth"] - partner_cos["opposite"])
            / (partner_cos["near"] - partner_cos["opposite"])),
    },
    "panel_b": {
        "model_nir_pooled": model_nir,
        "model_advantage_over_chance": model_over_chance,
        "comparator_displacement_from_chance_signed_as_deficit": {k: deficit[k] for k in STRATA},
        "gap": {k: gap[k] for k in STRATA},
        "gap_ci_two_way": {k: list(gap_ci[k]) for k in STRATA},
        "gap_n": {k: gap_n[k] for k in STRATA},
        "opposite_comparator_share_of_gap": share_opposite,
        "opposite_comparator_share_printed": "%d%%" % round(share_opposite * 100),
        "identity": "gap = (model_nir - 0.50) + (0.50 - partner_nir), exact to 1e-9 on all three",
    },
    "_thesis_crosscheck": {
        "04c-reencoding.tex:730": "0.550 / 0.501 / 0.423 -- matches",
        "04c-reencoding.tex:731-732": "+0.22 / 0.00 / -0.20 -- matches",
        "04c-reencoding.tex:744-745": "+0.0367, +0.0772, 68%, +0.1139 -- matches",
        "04c-reencoding.tex:658": "orth gap +0.0361 [+0.0159, +0.0564] -- matches ci_two_way",
    },
}

S.save(fig, "fig-comparator", drawn)
plt.close(fig)

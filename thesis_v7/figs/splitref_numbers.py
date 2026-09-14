"""fig-splitref: the six numbers the figure draws, and where each one comes from.

The figure itself is TikZ (figs/splitref.tex). This script exists so that the six
values in that drawing are not hand-typed constants with no provenance: it reads
RESULTS_cluster, recomputes both panels, checks them against what the thesis text
prints, and writes figs/fig-splitref.json next to the figure.

Run:  python thesis_v2/figs/splitref_numbers.py

WRITES ONLY inside thesis_v2/figs/. Nothing under thesis/ is read or written.

------------------------------------------------------------------------------
PANEL A -- the reference as each split is actually scored.
    Straight out of RESULTS_cluster/CANONICAL_NUMBERS.json -> ceilings{}, which
    is identical to re_v3.json aggregated per split and to
    scramble_stratum_audit_v4.json -> split_comparability{}. Spread 0.13178 is
    that file's own max_ceiling_spread. No reconstruction anywhere in this panel.

PANEL B -- the same reference with the 0.109 training line imposed on all three.
    THERE IS NO FILE IN RESULTS_cluster HOLDING THESE THREE NUMBERS. They appear
    only in the thesis prose (04c-reencoding.tex:843, 05-Limitations.tex:189).
    They are recomputed here from the per-condition records of re_v3.json by
    keeping every record whose repro_cos exceeds the 0.109 line and averaging
    that split's `ceiling`. That reproduces all three printed values exactly to
    three decimals and at no other threshold in a sweep of 0.10-0.20 do all
    three land on 0.955 / 0.961 / 0.960 at once, so the reconstruction is taken
    as the analysis the prose reports. It is still a reconstruction and the JSON
    says so.

    ONE DISAGREEMENT, NOT SILENTLY RESOLVED. The reconstructed spread is
    0.96056 - 0.95539 = 0.00517, which is 0.005 to three decimals. The thesis
    prints 0.006, which is max(rounded) - min(rounded) = 0.961 - 0.955. The
    left-hand 0.132 is NOT computed that way -- it is the raw 0.13178 rounded.
    So the two panels' spreads are quoted under two different conventions. The
    figure prints 0.006, i.e. what the text says, because a figure may not
    contradict its own chapter; the discrepancy is recorded in the JSON under
    `discrepancies` and reported upward for a decision.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "RESULTS_cluster")

SPLITS = ["train", "unseen_combo", "unseen_drug"]

# The reliability line. The thesis states it twice as the measured null's 95th
# percentile: "+0.109, resolving to +0.1086" (04c-reencoding.tex:211) and "the
# minimum is +0.109, and none falls at or below" (04c-reencoding.tex:219).
# Strict > 0.109 is the setting that reproduces all three printed values.
REPRO_LINE = 0.109

# What the thesis text prints, to three decimals. Panel A: 05-Limitations.tex
# lines 181-182 and 241-242. Panel B: 05-Limitations.tex line 189 and
# 04c-reencoding.tex line 843.
TEXT_A = {"train": 0.956, "unseen_combo": 0.824, "unseen_drug": 0.836}
TEXT_B = {"train": 0.955, "unseen_combo": 0.961, "unseen_drug": 0.960}
TEXT_SPREAD_A = 0.132
TEXT_SPREAD_B = 0.006


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    canon = load("CANONICAL_NUMBERS.json")
    audit = load("scramble_stratum_audit_v4.json")
    rev3 = load("re_v3.json")

    # ---- panel A: read, do not recompute -----------------------------------
    panel_a = {s: float(canon["ceilings"][s]) for s in SPLITS}
    spread_a = float(audit["split_comparability"]["max_ceiling_spread"])

    # cross-check panel A against a second file and against re_v3 aggregation
    for s in SPLITS:
        alt = float(audit["split_comparability"][s]["ceiling"])
        assert abs(alt - panel_a[s]) < 1e-12, (s, alt, panel_a[s])
        assert abs(round(panel_a[s], 3) - TEXT_A[s]) < 1e-9, (s, panel_a[s])
    assert abs(round(spread_a, 3) - TEXT_SPREAD_A) < 1e-9, spread_a

    # ---- panel B: reconstruct from the per-condition records ---------------
    by_split = defaultdict(list)
    for r in rev3["records"]:
        if r.get("split") in SPLITS:
            by_split[r["split"]].append(r)

    panel_b, kept, total = {}, {}, {}
    for s in SPLITS:
        rows = by_split[s]
        keep = [r for r in rows if r["repro_cos"] > REPRO_LINE]
        panel_b[s] = sum(r["ceiling"] for r in keep) / len(keep)
        kept[s], total[s] = len(keep), len(rows)
        assert abs(round(panel_b[s], 3) - TEXT_B[s]) < 1e-9, (s, panel_b[s])
    spread_b = max(panel_b.values()) - min(panel_b.values())

    # the one thing that does NOT reconcile
    discrepancies = []
    if abs(round(spread_b, 3) - TEXT_SPREAD_B) > 1e-9:
        discrepancies.append({
            "what": "right-panel spread",
            "thesis_text": TEXT_SPREAD_B,
            "reconstructed_raw": spread_b,
            "reconstructed_rounded": round(spread_b, 3),
            "explanation": (
                "0.006 is max(rounded) - min(rounded) = 0.961 - 0.955. The raw "
                "spread is %.5f, i.e. 0.005. The left panel's 0.132 is by "
                "contrast the raw 0.13178 rounded, so the two panels quote "
                "their spreads under different conventions."
            ) % spread_b,
            "drawn_in_figure": TEXT_SPREAD_B,
            "why_that_choice": (
                "The figure prints what Sections/05-Limitations.tex:189 and 243 "
                "and Sections/04c-reencoding.tex:844 already state. Changing it "
                "would put the figure at odds with three places in the prose. "
                "Reported upward rather than resolved here."
            ),
        })

    out = {
        "figure": "fig:splitref",
        "label": "fig:splitref",
        "source_tex": "thesis_v2/figs/splitref.tex",
        "estimator": ("within-well split-half precision reference; one half of a "
                      "condition's cells scored against the other half's truth, "
                      "residual frame, NIR scale"),
        "frame": "residual",
        "chance": 0.5,
        "y_axis_range_drawn": [0.82, 0.98],
        "y_axis_truncated": True,
        "shared_scale_across_panels": True,
        "drawing": {
            "kind": "tikz",
            "width_mm_measured": 132.90,
            "height_mm_measured": 77.74,
            "textwidth_mm": 142.00,
            "fullwidth_used": False,
            "measured_by": "thesis_v2/figs/_harness_splitref.tex",
        },
        "intervals_drawn": False,
        "intervals_note": ("No interval is carried for this reference in "
                           "CANONICAL_NUMBERS.json, scramble_stratum_audit_v4.json "
                           "or re_v3.json. Bare markers; none invented."),
        "sources": {
            "panel_a": ("RESULTS_cluster/CANONICAL_NUMBERS.json -> ceilings{}; "
                        "identical to scramble_stratum_audit_v4.json -> "
                        "split_comparability{}.<split>.ceiling"),
            "panel_a_spread": ("RESULTS_cluster/scramble_stratum_audit_v4.json -> "
                               "split_comparability.max_ceiling_spread"),
            "panel_b": ("RECONSTRUCTED. No file holds these. Recomputed from "
                        "RESULTS_cluster/re_v3.json -> records[], keeping "
                        "repro_cos > %.3f and averaging `ceiling` per split. "
                        "Reproduces the three values the thesis prints." % REPRO_LINE),
            "reliability_line": ("thesis_v2/Sections/04c-reencoding.tex:211,219 -- "
                                 "measured null 95th percentile +0.109"),
        },
        "panel_a_as_scored": {
            s: {
                "value_drawn": panel_a[s],
                "value_printed_in_figure": TEXT_A[s],
                "value_printed_in_thesis": TEXT_A[s],
                "n_conditions": int(canon["splits"][s]["n"]),
            } for s in SPLITS
        },
        "panel_a_spread": {
            "value_drawn": spread_a,
            "value_printed_in_figure": TEXT_SPREAD_A,
            "value_printed_in_thesis": TEXT_SPREAD_A,
        },
        "panel_b_training_line_imposed": {
            s: {
                "value_drawn": panel_b[s],
                "value_printed_in_figure": TEXT_B[s],
                "value_printed_in_thesis": TEXT_B[s],
                "n_conditions_kept": kept[s],
                "n_conditions_before": total[s],
                "reconstructed": True,
            } for s in SPLITS
        },
        "panel_b_spread": {
            "value_printed_in_figure": TEXT_SPREAD_B,
            "value_printed_in_thesis": TEXT_SPREAD_B,
            "value_reconstructed_raw": spread_b,
            "reconstructed": True,
        },
        "not_drawn": {
            "n_conditions_panel_b": ("The kept counts %s are a reconstruction and "
                                     "are not printed in the figure or caption. "
                                     "coda_errata/ERRATA.md:2368 quotes '499 of 895' "
                                     "discarded on unseen_combo, i.e. 396 kept, "
                                     "against %d here; the two differ by the two "
                                     "conditions sitting between 0.1086 and 0.109."
                                     % (kept, kept["unseen_combo"])),
            "chance_line": ("0.50 is off the truncated scale. It is marked at the "
                            "foot of a broken axis rather than drawn as a rule."),
        },
        "discrepancies": discrepancies,
    }

    dest = os.path.join(HERE, "fig-splitref.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    print("panel A (as scored)")
    for s in SPLITS:
        print("   %-13s %.7f  -> %.3f   n=%d" % (s, panel_a[s], TEXT_A[s], canon["splits"][s]["n"]))
    print("   spread        %.7f  -> %.3f" % (spread_a, TEXT_SPREAD_A))
    print("panel B (training line imposed, reconstructed at repro_cos > %.3f)" % REPRO_LINE)
    for s in SPLITS:
        print("   %-13s %.7f  -> %.3f   n=%d of %d" % (s, panel_b[s], TEXT_B[s], kept[s], total[s]))
    print("   spread        %.7f  -> text prints %.3f" % (spread_b, TEXT_SPREAD_B))
    for d in discrepancies:
        print("DISCREPANCY: %s  text=%s reconstructed=%.5f" %
              (d["what"], d["thesis_text"], d["reconstructed_raw"]))
    print("wrote", dest)

    # the exact y-coordinates the TikZ file must carry, so the two cannot drift
    print("\nTikZ dot values (paste into splitref.tex if the data ever changes):")
    for s in SPLITS:
        print("   A %-13s %.6f" % (s, panel_a[s]))
    for s in SPLITS:
        print("   B %-13s %.6f" % (s, panel_b[s]))


if __name__ == "__main__":
    main()

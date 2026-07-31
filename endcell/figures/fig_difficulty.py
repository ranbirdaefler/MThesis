"""fig-difficulty: the task is winnable where the ceiling is high -- and a control-copy matches
the model exactly there.

Two panels.

(a) Ceiling NIR against cells per condition, cross-plate and within-plate, with markers showing
    where the populations in panel (b) actually sit. The point is that identifiability is
    NOISE-limited, not absent: the signal is real and it emerges under aggregation.

(b) The stratum ladder. Five arms in EVERY stratum. Drawing the model alone across strata would
    make the visual claim "the model does better on winnable drugs", which the thesis explicitly
    denies -- a control-copy baseline carrying NO drug information sits within 0.002 of the model
    on the identifiable stratum. The paired model - control_copy contrast with a clustered
    interval is drawn beneath, because that is the honest summary of the panel.

The strata are recomputed from the per-drug rows rather than read from the stored `binned` block,
for two reasons: the stored block omits control_copy and scramble, and recomputing gives a unit
test -- the bin sizes must come out at 296 / 106 / 204.

Usage:  python endcell/figures/fig_difficulty.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

EDGES = (0.6, 0.8)                      # ceiling NIR cut points
EXPECT_N = (296, 106, 204)              # unit test: unwinnable / marginal / identifiable
ARMS = ("ceiling", "model", "control", "scramble", "linear")
NAMES = {"unwinnable": "unwinnable\n(ceiling < 0.6)",
         "marginal": "marginal\n(0.6–0.8)",
         "identifiable": "identifiable\n(ceiling ≥ 0.8)"}


def clustered_ci(vals, clusters, rng, n_boot=2000):
    """95% bootstrap resampling CELL LINES. Conditions within a line are pseudoreplicates."""
    v = np.asarray(vals, float)
    cl = np.asarray(clusters)
    u = sorted(set(cl.tolist()))
    idx = {k: np.where(cl == k)[0] for k in u}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.randint(0, len(u), len(u))
        sel = np.concatenate([idx[u[t]] for t in take])
        boot[b] = v[sel].mean()
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    style.apply()
    rng = np.random.RandomState(0)

    rows = style.load("nir_sameplate.json")["tiers"]["tier2_unseen_drugs"]["rows"]
    get = lambda r, a: (r.get(a) or {}).get("nir_expr")

    bins = {"unwinnable": [], "marginal": [], "identifiable": []}
    for r in rows:
        c = get(r, "ceiling")
        if c is None:
            continue
        key = "unwinnable" if c < EDGES[0] else ("marginal" if c < EDGES[1] else "identifiable")
        bins[key].append(r)

    got = tuple(len(bins[k]) for k in ("unwinnable", "marginal", "identifiable"))
    assert got == EXPECT_N, "stratum sizes %s do not reproduce the stored %s" % (got, EXPECT_N)
    print("unit test OK: bin sizes %s reproduce stratify_sameplate.json" % (got,))

    drawn = {"edges": EDGES, "n": dict(zip(bins, got)), "strata": {}, "sweep": {}}
    for k, rs in bins.items():
        drawn["strata"][k] = {a: float(np.mean([get(r, a) for r in rs
                                                if get(r, a) is not None])) for a in ARMS}
        pairs = [(r["cell_line"], get(r, "model") - get(r, "control")) for r in rs
                 if get(r, "model") is not None and get(r, "control") is not None]
        m, lo, hi = clustered_ci([d for _, d in pairs], [c for c, _ in pairs], rng)
        drawn["strata"][k]["model_minus_control"] = {"gap": m, "ci": [lo, hi], "n": len(pairs)}

    # ---------------------------------------------------------------- figure
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(style.TEXTWIDTH_IN, 2.75),
        gridspec_kw=dict(width_ratios=[1.0, 1.32], wspace=0.42))

    # ---- (a) ceiling vs cells
    cross = style.load("drug_biology_atlas.json")["summary"]["cell_sweep"]
    within = style.load("sweep_sameplate.json")["cell_sweep"]
    cx = sorted(int(k) for k in cross)
    cy = [cross[str(k)] for k in cx]
    wx = sorted(int(k) for k in within)
    wy = [within[str(k)]["ceiling_nir"] for k in wx]
    drawn["sweep"] = {"cross_plate": dict(zip(map(str, cx), cy)),
                      "within_plate": dict(zip(map(str, wx), wy))}

    axL.plot(cx, cy, marker="o", ms=3.5, color=style.OKABE["grey"], label="cross-plate")
    axL.plot(wx, wy, marker="s", ms=3.5, color=style.OKABE["blue"], label="within plate")
    style.chance_line(axL)
    axL.set_xscale("log")
    axL.set_xticks([4, 10, 20, 40, 120])
    axL.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axL.set_xlabel("cells per condition")
    axL.set_ylabel(style.nir_label("expression"), fontsize=8.5)
    axL.set_ylim(0.45, 0.92)
    axL.set_title("(a) identifiability is noise-limited", fontsize=9.5, loc="left")
    axL.legend(loc="lower right")

    # ---- (b) stratum ladder
    order = ["unwinnable", "marginal", "identifiable"]
    xs = np.arange(len(order))
    off = {"ceiling": -0.26, "model": -0.13, "control": 0.0, "scramble": 0.13, "linear": 0.26}
    for a in ARMS:
        st = style.ARM[a]
        axR.plot(xs + off[a], [drawn["strata"][k][a] for k in order],
                 ls="none", marker=st["marker"], ms=5.0, color=st["color"],
                 label=st["label"], mfc="none" if a == "ceiling" else st["color"], mew=1.2)
    handles = axR.get_legend_handles_labels()
    style.chance_line(axR)
    axR.set_xticks(xs)
    axR.set_xticklabels([NAMES[k] for k in order], fontsize=8)
    axR.set_ylabel(style.nir_label("expression"), fontsize=8.5)
    axR.set_ylim(0.15, 1.02)
    axR.set_title("(b) five arms in every stratum", fontsize=9.5, loc="left")
    fig.legend(*handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.20),
               columnspacing=1.4, handletextpad=0.35)

    # the paired contrast is the honest summary of this panel: on the identifiable stratum a
    # predictor with NO drug information is within 0.002 of the model.
    for i, k in enumerate(order):
        g = drawn["strata"][k]["model_minus_control"]
        axR.annotate("%+.3f" % g["gap"], xy=(i, 0.175), ha="center", fontsize=7.5,
                     color=style.OKABE["vermilion"])
    axR.annotate("model $-$ control-copy", xy=(0.05, 0.395), xycoords=("axes fraction", "data"),
                 ha="left", fontsize=7.5, color=style.OKABE["vermilion"])

    style.save(fig, "fig-difficulty", drawn)
    for k in order:
        s = drawn["strata"][k]
        print("  %-13s n=%3d  ceiling %.3f  model %.3f  control %.3f  scramble %.3f  linear %.3f"
              % (k, drawn["n"][k], s["ceiling"], s["model"], s["control"], s["scramble"], s["linear"]))
        print("                 model - control = %+.4f [%+.4f, %+.4f]"
              % (s["model_minus_control"]["gap"], *s["model_minus_control"]["ci"]))


if __name__ == "__main__":
    main()

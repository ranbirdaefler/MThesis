"""fig-sweep-arms: the swap-distance sweep of \\S4.2, extended to all three target arms.

WHY THIS EXISTS. \\S4.6 scored each arm against a SINGLE randomly drawn scramble
partner and reported one number per arm, so it could not show whether the gap grows
as the substituted drug becomes more dissimilar -- the shape \\S4.9's stratified
ladder rests on. The partner identity was recorded at build time
(make_scramble_endcell.py writes scrambled_from_drug / scrambled_to_drug into every
scrambled example) and nir_benchmark.py wrote per-drug truth pseudobulks to
*_profiles.npz, so the sweep is reconstructible POST HOC with no regeneration.

DISTANCE IS EUCLIDEAN ON DECODED PROFILES, matching \\S4.2's
scramble_distance_sweep.py exactly: D(A,B) = ||truth_expr_A - truth_expr_B||, both
half-A pseudobulks, within one (cell_line, plate) group so the control cancels.
Cosine was tried first and is the wrong metric here: on full expression profiles it
spans only +0.56..+0.89 and never reaches zero, because normalising away magnitude
discards where the drug signal lives. Euclidean recovers 2.85..6.21, matching the
2.75..6.25 of the \\S4.2 sweep.

THE PARTNER IS DRAWN PER CELL, NOT PER CONDITION (median 6 distinct partners per
condition), so a condition's scramble score already averages over a mixture of
distances. Each condition is therefore placed at its MEAN partner distance. This is
a property of the original \\S4.6 design, not a choice made here, and it compresses
the x-range relative to \\S4.9's fixed-extremum ladder.

SCORED ON ALL TIER-2 CONDITIONS, not the identifiable stratum tab:epsilon used.
The identifiable version was built first and dropped: restricting to ceiling
nir_expr >= 0.80 leaves 160 resolvable conditions, ~40 per bin, and the
cell-line-clustered intervals then span +-0.06 -- wide enough that no pattern
could be distinguished from none. The stratum restriction is retained in the
prose, which reports tab:epsilon's per-arm values; this figure trades that
population match for the power to see an ordering if one exists.

Intervals are cell-line-clustered bootstraps, 2000 resamples, matching \\S4.2.
Usage: python sweep_arms.py    (run from thesis_v4/figs/)
"""
import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_v2 as style
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ARMS = [("optimal transport", "nir_ot_T2", style.QUIETINK, "D"),
        ("consensus", "nir_consensus", style.INK, "^")]
NIRJ = {"nir_singlecell_t1t2": "nir_singlecell_t1t2.json",
        "nir_consensus": "nir_consensus.json", "nir_ot_T2": "nir_ot_T2.json"}
N_BOOT, SEED, IDENT = 2000, 42, 0.80


def records(stem, identifiable_only):
    pairs = json.load(open(os.path.join(style.results_dir(), "scramble_pairs_tier2.json")))
    by = defaultdict(list)
    for r in pairs:
        by[(r["cell_line_id"], r["plate"], r["scrambled_from_drug"])].append(r["scrambled_to_drug"])
    z = np.load(os.path.join(style.results_dir(), stem + "_profiles.npz"), allow_pickle=True)
    truth, grp = {}, defaultdict(dict)
    for k in z.keys():
        if not k.startswith("truth||"):
            continue
        _, _t, clp, d = k.split("||")
        cl, pl = clp.split("~")
        truth[(cl, pl, d)] = z[k]
        grp[(cl, pl)][d] = z[k]
    rows = style.load(NIRJ[stem])["tiers"]["tier2_unseen_drugs"]["rows"]
    if identifiable_only:
        rows = [r for r in rows if r["ceiling"]["nir_expr"] >= IDENT]
    out = []
    for r in rows:
        key = (r["cell_line"], r["plate"], r["drug"])
        if key not in truth or key not in by:
            continue
        cl, pl, a = key
        ta = truth[key]
        dd = [float(np.linalg.norm(ta - grp[(cl, pl)][b]))
              for b in by[key] if b in grp[(cl, pl)] and b != a]
        if not dd:
            continue
        out.append((float(np.mean(dd)),
                    r["model"]["nir_expr"] - r["scramble"]["nir_expr"], r["cell_line"]))
    return out


def binned(recs, n_bins, rng):
    d = np.array([x[0] for x in recs]); g = np.array([x[1] for x in recs])
    cls = np.array([x[2] for x in recs])
    q = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (d >= q[i]) & (d <= q[i + 1]) if i == n_bins - 1 else (d >= q[i]) & (d < q[i + 1])
        if m.sum() < 3:
            continue
        u = np.unique(cls[m])
        bs = [np.concatenate([g[m][cls[m] == c]
                              for c in rng.choice(u, len(u), replace=True)]).mean()
              for _ in range(N_BOOT)]
        out.append(dict(mean_dist=float(d[m].mean()), gap=float(g[m].mean()), n=int(m.sum()),
                        lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975))))
    return out, float(np.mean(g)), float(np.corrcoef(d, g)[0, 1]), len(recs)


def signed(v, _p=None):
    return "0" if abs(v) < 1e-12 else ("%+.2f" % v).replace("-", "−")


def main():
    style.apply()
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH_IN, 3.30), layout="constrained")
    ax.axhline(0.0, color=style.INK, lw=0.9, zorder=1)
    drawn = {"figure": "fig-sweep-arms",
             "quantity": "NIR(model) - NIR(scramble), expression frame, tier2_unseen_drugs",
             "x_variable": "mean Euclidean ||truth_A - truth_B|| over a condition's per-cell "
                           "scramble partners, within (cell_line, plate)",
             "population": "all tier-2 conditions with a resolvable partner (not the "
                           "identifiable stratum; see module docstring)",
             "interval": "95%% cell-line-clustered bootstrap, %d resamples" % N_BOOT,
             "arms": {}}
    for label, stem, colour, marker in ARMS:
        recs = records(stem, False)
        bins, overall, pear, n = binned(recs, 5, rng)
        drawn["arms"][label] = dict(n_conditions=n, overall_gap=overall, pearson=pear, bins=bins)
        for b in bins:
            style.whisker(ax, b["mean_dist"], None, b["lo"], b["hi"], orientation="v",
                          color=colour, lw=1.0, alpha=0.7, zorder=2)
        ax.plot([b["mean_dist"] for b in bins], [b["gap"] for b in bins],
                marker=marker, ms=4.5, mew=1.1, color=colour, ls="-", lw=1.1,
                label=label, zorder=3, mfc=colour)
    ax.set_ylim(-0.055, 0.105)
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.set_ylabel("$\\Delta_{\\mathrm{scr}}$  =  model $-$ scramble\n(NIR, expression frame)")
    ax.set_xlabel("swap distance $\\|$truth$_A$ $-$ truth$_B\\|$ "
                  "(mean over a condition's partners)")
    ax.legend(loc="upper left", fontsize=8.5, borderaxespad=0.3, handlelength=2.2,
              labelspacing=0.3, borderpad=0.0)
    style.save(fig, "fig-sweep-arms", drawn)
    for label, v in drawn["arms"].items():
        print("  %-19s n=%3d overall=%+.4f pearson=%+.4f" %
              (label, v["n_conditions"], v["overall_gap"], v["pearson"]))
        print("      " + "  ".join("d%.2f:%+.4f[%+.3f,%+.3f]" %
              (b["mean_dist"], b["gap"], b["lo"], b["hi"]) for b in v["bins"]))
    return fig


if __name__ == "__main__":
    main()

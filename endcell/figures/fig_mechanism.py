"""fig-mechanism: the drug is encoded ever more explicitly with depth, and never read for identity.

This is the thesis's most novel result and it is currently a seven-row table that no reader will
parse. It contains a clean dissociation that a table flattens:

(a) Two curves against a constant. Drug decodability with cell line, plate and dose projected out
    RISES monotonically with depth (0.354 -> 0.883), while the causal variance share of that same
    purified subspace stays FLAT near 0.03 against cell line's ~0.6. The late layers know the drug
    best and use it least. Note the RAW decode is also drawn and behaves differently -- it peaks at
    layer 9 and declines -- because the thesis's prose quotes the raw series and its table quotes
    the context-removed one, and conflating them is how the published table went wrong.

(b) The identity test, on a log axis because late-layer KLs are ~0.03 for everything and absolute
    magnitudes are not comparable across depth. Injecting drug B's code perturbs the output LESS
    than a random vector of identical norm, at all seven depths.

No error bars on the swap: workspace_probe.py reduces the per-prompt distributions to two means
before storing them. There is no dispersion behind the thesis's most load-bearing mechanistic
number, and inventing some would be the worst available failure. The evidence is seven independent
depths falling the same way, and the caption says so.

Usage:  python endcell/figures/fig_mechanism.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def first(x):
    """kl entries are stored as [mean, sem, n] triples; variance shares are bare floats."""
    return x[0] if isinstance(x, (list, tuple)) else x


def main():
    style.apply()
    L = style.load("workspace_probe.json")["layers"]
    layers = sorted(L, key=int)
    xi = np.arange(len(layers))

    raw = [L[k]["drug_decode"] for k in layers]
    ctx = [L[k]["drug_decode_context_removed"] for k in layers]
    chance = L[layers[0]]["chance"]
    vs_drug = [L[k]["variance_share"]["drug_pure"] for k in layers]
    vs_cl = [L[k]["variance_share"]["cell_line"] for k in layers]
    swap_b = [L[k]["swap"]["drug_b_kl"] for k in layers]
    swap_r = [L[k]["swap"]["random_inject_kl"] for k in layers]
    kl_pd = [first(L[k]["kl"]["pure_drug"]) for k in layers]
    kl_rm = [first(L[k]["kl"]["random_matched"]) for k in layers]
    gate = [L[k]["frac_signal_removed"] for k in layers]

    drawn = {"layers": layers, "drug_decode_raw": raw, "drug_decode_context_removed": ctx,
             "chance": chance, "var_share_pure_drug": vs_drug, "var_share_cell_line": vs_cl,
             "swap_drug_b": swap_b, "swap_random_matched_norm": swap_r,
             "kl_pure_drug": kl_pd, "kl_random_matched": kl_rm, "frac_signal_removed": gate,
             "swap_below_noise_at_every_layer": bool(all(b < r for b, r in zip(swap_b, swap_r)))}
    assert drawn["swap_below_noise_at_every_layer"], "the headline claim does not hold in the JSON"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(style.TEXTWIDTH_IN, 2.9),
                                   gridspec_kw=dict(wspace=0.46))

    # ---------------- (a) the dissociation
    axL.plot(xi, ctx, marker="o", ms=4, color=style.OKABE["blue"],
             label="drug decodable (context removed)")
    axL.plot(xi, raw, marker="o", ms=3, mfc="none", ls=":", color=style.OKABE["blue"],
             label="drug decodable (raw)")
    axL.axhline(chance, color="black", lw=0.9)
    axL.annotate("chance", xy=(len(layers) - 1, chance), xytext=(2, 3),
                 textcoords="offset points", fontsize=7.5, ha="right")
    axL.plot(xi, vs_drug, marker="s", ms=4, color=style.OKABE["vermilion"],
             label="causal share, drug subspace")
    axL.plot(xi, vs_cl, marker="^", ms=4, color=style.OKABE["grey"],
             label="causal share, cell line")
    axL.set_xticks(xi)
    axL.set_xticklabels(layers)
    axL.set_xlabel("layer")
    axL.set_ylabel("probe accuracy  /  variance share")
    axL.set_ylim(0, 1.0)
    axL.set_title("(a) encoded more, used no more", fontsize=9.5, loc="left")
    axL.legend(loc="upper left", fontsize=7.4, handlelength=1.6)

    axL.annotate("", xy=(6.0, 0.883), xytext=(6.0, vs_drug[-1]),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color="black"))
    axL.annotate("encoded, not read", xy=(5.85, 0.48), ha="right", fontsize=7.4, rotation=90,
                 va="center")

    # ---------------- (b) the identity test
    w = 0.34
    axR.bar(xi - w / 2, swap_b, width=w, color=style.OKABE["blue"], label="inject drug B")
    axR.bar(xi + w / 2, swap_r, width=w, color=style.OKABE["grey"],
            label="inject random, matched norm")
    axR.set_yscale("log")
    axR.set_xticks(xi)
    axR.set_xticklabels(layers)
    axR.set_xlabel("layer")
    axR.set_ylabel("KL from the clean forward pass")
    axR.set_title("(b) the identity test: swap $<$ noise, every depth", fontsize=9.5, loc="left")
    axR.legend(loc="lower left", fontsize=7.6)
    axR.set_ylim(0.02, 6.0)

    style.save(fig, "fig-mechanism", drawn)
    print("  layer  raw   ctx    var_pd  swapB   randinj  gate")
    for i, k in enumerate(layers):
        print("  %5s  %.3f %.3f  %.3f   %.3f   %.3f    %.0f%%"
              % (k, raw[i], ctx[i], vs_drug[i], swap_b[i], swap_r[i], 100 * gate[i]))


if __name__ == "__main__":
    main()

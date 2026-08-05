"""Does the panel index carry the gene identity it claims?

WHY THIS EXISTS. The construction was previously validated by a PCA cell-line probe. That probe
cannot do the job: permuting the gene columns is an orthogonal transform, so it preserves every
pairwise row distance and therefore preserves PCA and classification accuracy exactly. A panel with
every gene label scrambled scores identically. The probe shows the panel retains cell-line
discriminative structure; it says nothing about which column is which gene.

WHAT DOES WORK. Measured abundance is a property of the gene. Ribosomal proteins and canonical
housekeeping transcripts sit among the most abundant in essentially any mammalian cell, so their
position in an abundance ranking is not exchangeable. Under a permutation of the panel index their
ranks become uniform. That is a testable difference.

Runs off the stored `truth` profiles, so it needs no cluster access.

Usage:  python endcell/analysis/gene_identity_check.py
"""
import io
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_FILE = os.path.join(REPO, "shared", "l1000_panel.json")
PROFILES = os.path.join(REPO, "RESULTS_cluster", "nir_consensus_profiles.npz")
OUT = os.path.join(REPO, "RESULTS_cluster", "gene_identity_check.json")

# Abundance here is a property of the gene, not of this experiment or this panel.
CANONICAL_HIGH = ("RPS5", "RPL38", "RPS11", "RPS6", "RPL13A", "EEF1A1", "ACTB", "GAPDH",
                  "TPT1", "PABPC1", "HSP90AA1", "HSPA8", "NPM1", "EEF2")
N_DRAWS = 20000
SEED = 0


def main():
    panel = json.load(io.open(PANEL_FILE, encoding="utf-8"))
    g = len(panel)
    idx = {sym: i for i, sym in enumerate(panel)}

    z = np.load(PROFILES, allow_pickle=True)
    keys = [k for k in z.keys() if k.startswith("truth||")]
    if not keys:
        raise SystemExit("no `truth` arm in %s -- this check needs measured expression, "
                         "not model output" % os.path.basename(PROFILES))
    truth = np.stack([z[k] for k in keys])

    mean = truth.mean(axis=0)
    order = np.argsort(-mean)                 # rank 1 = most abundant
    rank = np.empty(g, dtype=int)
    rank[order] = np.arange(1, g + 1)

    present = [sym for sym in CANONICAL_HIGH if sym in idx]
    if len(present) < 3:
        raise SystemExit("only %d canonical high-expressors are on this panel; too few to test"
                         % len(present))
    obs = np.array([rank[idx[sym]] for sym in present], dtype=float)
    med = float(np.median(obs))

    # Under a permuted panel index these ranks are a uniform sample without replacement.
    rng = np.random.RandomState(SEED)
    draws = np.array([np.median(rng.choice(np.arange(1, g + 1), len(present), replace=False))
                      for _ in range(N_DRAWS)])
    p = float((np.sum(draws <= med) + 1) / (len(draws) + 1))

    out = {
        "n_truth_profiles": int(truth.shape[0]),
        "panel_size": g,
        "genes_tested": present,
        "ranks": {sym: int(rank[idx[sym]]) for sym in present},
        "median_rank_observed": med,
        "median_rank_expected_under_permutation": (g + 1) / 2.0,
        "p_one_sided": p,
        "n_draws": N_DRAWS,
        "seed": SEED,
        "verdict": "panel index matches gene identity" if p < 0.001 else "inconclusive",
        "_note": "A PCA cell-line probe cannot establish this: column permutation is orthogonal and "
                 "leaves PCA and classification invariant. Abundance rank is not permutation-"
                 "invariant, which is why this test can detect what that probe cannot.",
    }
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)

    print("truth profiles %s over a %d-gene panel" % (truth.shape, g))
    for sym in present:
        r = rank[idx[sym]]
        print("   %-10s rank %4d  (%.1f%%)" % (sym, r, 100.0 * r / g))
    print("median rank %.1f observed against %.1f expected under a permuted panel" % (med, (g + 1) / 2.0))
    print("one-sided p = %.5f over %d draws" % (p, N_DRAWS))
    print("VERDICT: %s" % out["verdict"])
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()

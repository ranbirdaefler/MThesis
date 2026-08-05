"""Calibrate the metric the residual chapter actually uses.

WHY THIS EXISTS. The DRF audit in sec:res-metrics graded `nir_expr`: Euclidean distance between
expression pseudobulks. The residual-frame headline numbers are computed with a THIRD variant --
signed rank-discounted residual vectors compared by COSINE (residual_eval.py:13, :78), over a
comparison set of other drugs in the same cell line. Grading one metric does not grade the other,
and an external review was right to say so.

DRF = (m_pos - m_neg) / (m_perfect - m_neg), following the expression-frame audit.

  positive control : `ceiling`, a split-half replicate of the same condition
  negative control : a predictor carrying no drug information
  perfect          : 1.0, the cosine of a vector with itself

THE NEGATIVE CONTROL MATTERS. `generic` scores exactly 0.5 with zero variance -- necessarily, since
the residual is defined by subtracting the generic, so its own residual is the zero vector. Using it
makes DRF a linear rescaling of the ceiling and tests nothing. The empirical baselines
(`control_copy`, `random`, `scramble_orth`) carry real variance and are the honest controls; all
three are reported so the figure cannot rest on the degenerate one.

Runs off the stored evaluation, so it needs no cluster access.

Usage:  python endcell/analysis/residual_frame_drf.py
"""
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "shared"))
import inference as inf  # noqa: E402

EVAL = os.path.join(REPO, "RESULTS_cluster", "re_v3.json")
OUT = os.path.join(REPO, "RESULTS_cluster", "residual_frame_drf.json")

NEGATIVES = (
    ("control_copy", "copies the untreated cell; zero drug information", True),
    ("random", "a random vector", True),
    ("scramble_orth", "the neutral scrambled partner", True),
    ("generic", "the zero vector BY CONSTRUCTION -- degenerate, reported for completeness", False),
)
N_BOOT = 2000
SEED = 42


def drf_of(rows):
    pos = np.array([r[0] for r in rows], dtype=float)
    neg = np.array([r[1] for r in rows], dtype=float)
    den = 1.0 - neg.mean()
    return None if abs(den) < 1e-12 else float((pos.mean() - neg.mean()) / den)


def main():
    rec = json.load(io.open(EVAL, encoding="utf-8"))["records"]
    out = {"_frame": "signed rank-discounted residual, cosine similarity (residual_eval.py)",
           "_perfect": 1.0, "arms": {}}

    print("%-15s %8s %7s  %-24s %s" % ("negative", "mean", "sd", "DRF [95% CI]", "empirical?"))
    for key, note, empirical in NEGATIVES:
        use = [r for r in rec if r.get("ceiling") is not None and r.get(key) is not None]
        if not use:
            continue
        pos = [float(r["ceiling"]) for r in use]
        neg = [float(r[key]) for r in use]
        lines = [r["cell_line"] for r in use]
        rows = list(zip(pos, neg))
        d = drf_of(rows)
        ci = inf.cluster_bootstrap(rows, lines, drf_of, n_boot=N_BOOT, seed=SEED)
        out["arms"][key] = {
            "n": len(use), "n_celllines": len(set(lines)),
            "m_pos_ceiling": float(np.mean(pos)),
            "m_neg": float(np.mean(neg)), "sd_neg": float(np.std(neg)),
            "drf": d, "ci": [ci["lo"], ci["hi"]], "n_clusters": ci["n_clusters"],
            "empirical": empirical, "note": note,
        }
        print("%-15s %8.4f %7.4f  %+0.4f [%+0.4f, %+0.4f]  %s"
              % (key, np.mean(neg), np.std(neg), d, ci["lo"], ci["hi"], "yes" if empirical else "NO"))

    out["headline"] = out["arms"].get("control_copy")
    out["_caveats"] = [
        "the positive control is a raw split-half replicate, not the denoised interpolated duplicate "
        "of the expression-frame protocol, so this figure is conservative by comparison",
        "the ceiling arm is scored against half-A truths while the negative arms use full-sample "
        "truths (residual_eval.py:631-635 vs :649)",
        "percentile cluster bootstrap over cell line; no small-sample correction",
    ]
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()

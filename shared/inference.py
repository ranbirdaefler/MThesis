#!/usr/bin/env python
r"""
inference.py -- the one place uncertainty is computed, so every interval names its unit.
========================================================================================
Three audits converged on the same complaint from different directions: intervals in this project
were computed ad hoc, each script rolling its own bootstrap, and none of them stated what it treated
as independent. This module exists so that a reviewer can read one file and know what every quoted
interval means.

FOUR THINGS IT FIXES, each of which was a real defect somewhere in the repository.

1. THE STATISTIC IS RECOMPUTED INSIDE EVERY RESAMPLE.
   `aggregate_drf` computes a RATIO -- (positive mean - negative mean) / denominator -- and the
   shipped code resampled the numerator while holding the denominator fixed. That understates the
   interval, because the denominator is estimated from the same data and moves with it. Every
   function here takes a `statistic` callable applied to the resampled rows, so a ratio, a
   disattenuated coefficient or a difference of differences is handled correctly by construction and
   there is no way to accidentally fix part of it.

2. THE DESIGN IS CROSSED, AND CROSSED IS NOT NESTED.
   The experimental-unit audit measured 48.6 cell lines per treated well across 289 wells. A well
   holds many cell lines AND a cell line appears in many wells, so neither grouping alone captures
   the dependence. Worse, there are more wells (289) than lines (~50), so "cluster on the well, it is
   the assignment unit" -- which is what we did first -- yields a NARROWER interval and is
   anti-conservative. `two_way_cluster_ci` implements Cameron-Gelbach-Miller:

       V = V_a + V_b - V_ab

   where the intersection of the two groupings is the observation itself, so V_ab is the ordinary
   independent-sampling variance. In finite samples V can go negative; we then fall back to the
   wider one-way variance and say which happened rather than silently returning a number.

3. ENERGY SHARES SUM TO ONE, INCLUDING THE CROSS TERM.
   `variance_decomposition` reported ||r||/||s|| = 0.786 and ||g||/||s|| = 0.620, squared them, and
   checked the sum came to about 1.002 as evidence of orthogonality. That is a diagnostic dressed as
   a decomposition. Since s = r + g exactly,

       ||s||^2 = ||r||^2 + ||g||^2 + 2<r,g>

   so the three shares sum to one BY CONSTRUCTION and the cross term is a number to report, not a
   residual to hope is small. `energy_shares` returns all three.

4. MULTIPLICITY IS APPLIED ACROSS THE FAMILY THAT WAS ACTUALLY TESTED.
   "Only NIR is calibrated" is a simultaneous statement about five metrics and needs Holm; the probe
   family is a screen across layers and arms and wants BH. Both are here, both return adjusted
   p-values rather than a bare reject/accept, and both refuse silently-wrong input.

USAGE
  python shared/inference.py --selftest
"""
import argparse
import logging
import math
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- multiplicity
def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved.

    Use when the claim is SIMULTANEOUS -- "of these five metrics, only NIR is calibrated" is a
    statement about all five at once, so controlling the family-wise error rate is the right
    guarantee and BH is too weak.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("holm expects a non-empty 1-D sequence of p-values")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("holm received a value outside [0, 1] or a NaN")
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])       # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def bh(pvalues):
    """Benjamini-Hochberg adjusted p-values (q-values), order preserved.

    Use for a SCREEN -- the probe tests many layers and arms expecting most to be null, and the
    claim is about which few survive, not about all of them jointly.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("bh expects a non-empty 1-D sequence of p-values")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("bh received a value outside [0, 1] or a NaN")
    n = len(p)
    order = np.argsort(p)[::-1]                            # largest first
    adj = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order):
        m = n - rank                                       # 1-based rank from the small end
        running = min(running, n * p[idx] / m)
        adj[idx] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- resampling
def _index_by(labels):
    g = defaultdict(list)
    for i, c in enumerate(labels):
        g[c].append(i)
    return {k: np.asarray(v) for k, v in g.items()}


def cluster_bootstrap(rows, clusters, statistic, n_boot=2000, seed=0, alpha=0.05):
    """Resample CLUSTERS with replacement; recompute `statistic` on each resample.

    `rows` is any sequence the statistic understands -- a list of dicts, an array of differences,
    whatever. `statistic(subset) -> float or None`. Returning None (a resample where the statistic is
    undefined, e.g. an empty stratum) drops that draw and is counted.

    The point of passing a callable rather than a vector of values is defect 1 in the header: a ratio
    must be recomputed, not assembled from separately-resampled parts.
    """
    rows = list(rows)
    if len(rows) != len(clusters):
        raise ValueError(f"rows ({len(rows)}) and clusters ({len(clusters)}) differ in length")
    idx = _index_by(clusters)
    keys = sorted(idx, key=str)
    if len(keys) < 2:
        return None
    point = statistic(rows)
    if point is None:
        return None
    rng = np.random.RandomState(seed)
    boot, n_undef = [], 0
    for _ in range(n_boot):
        take = rng.randint(0, len(keys), len(keys))
        sel = np.concatenate([idx[keys[t]] for t in take])
        v = statistic([rows[i] for i in sel])
        if v is None:
            n_undef += 1
        else:
            boot.append(v)
    if len(boot) < max(50, n_boot // 20):
        return None
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n": len(rows), "n_clusters": len(keys), "unit": "one-way cluster",
            "n_undefined_resamples": n_undef, "boot_sd": float(np.std(boot))}


def two_way_cluster_ci(values, cluster_a, cluster_b, alpha=0.05):
    """Cameron-Gelbach-Miller two-way cluster-robust interval for a MEAN.

    For the crossed design described in the header. Returns the branch taken in `unit`, because
    "the two-way estimator went negative and we used the wider one-way" is information a reader
    needs and a silent fallback would hide.

    Analytic rather than bootstrapped because a two-way cluster bootstrap has no single agreed form
    and this estimator is standard; `crossed_bootstrap` is available when the statistic is not a mean.
    """
    v = np.asarray(values, dtype=float)
    if len(v) != len(cluster_a) or len(v) != len(cluster_b):
        raise ValueError("values, cluster_a and cluster_b must be the same length")
    if len(v) < 3:
        return None
    n = len(v)
    m = float(v.mean())
    dev = v - m

    def var_of(labels):
        g = _index_by(labels)
        if len(g) < 2:
            return None
        s = sum(dev[i].sum() ** 2 for i in g.values())
        return (len(g) / (len(g) - 1)) * s / (n ** 2)

    va, vb = var_of(cluster_a), var_of(cluster_b)
    vab = float((dev ** 2).sum()) / (n ** 2)
    na, nb = len(set(map(str, cluster_a))), len(set(map(str, cluster_b)))

    if va is None and vb is None:
        return None
    if va is None or vb is None:
        var, unit = (vb if va is None else va), "one-way (only one grouping had >1 cluster)"
    else:
        var, unit = va + vb - vab, "two-way (Cameron-Gelbach-Miller)"
        if var <= 0:
            var = max(va, vb)
            unit = "two-way estimator non-positive; fell back to the wider one-way"
    z = Z95 if abs(alpha - 0.05) < 1e-12 else float(abs(_norm_ppf(alpha / 2)))
    se = math.sqrt(max(var, 0.0))
    return {"point": m, "lo": m - z * se, "hi": m + z * se, "se": se,
            "n": n, "n_clusters_a": na, "n_clusters_b": nb, "unit": unit,
            "var_a": va, "var_b": vb, "var_intersection": vab}


def crossed_bootstrap(rows, cluster_a, cluster_b, statistic, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap for a crossed design when the statistic is NOT a mean (so CGM does not apply).

    Resamples both groupings independently and keeps the observations whose BOTH labels were drawn.
    That is the intersection form: it is conservative relative to either one-way bootstrap, which is
    the direction we want given the design, and it degenerates gracefully to a one-way bootstrap when
    one grouping is a singleton.
    """
    rows = list(rows)
    ia, ib = _index_by(cluster_a), _index_by(cluster_b)
    ka, kb = sorted(ia, key=str), sorted(ib, key=str)
    if len(ka) < 2 or len(kb) < 2:
        return cluster_bootstrap(rows, cluster_a if len(ka) >= 2 else cluster_b,
                                 statistic, n_boot, seed, alpha)
    point = statistic(rows)
    if point is None:
        return None
    rng = np.random.RandomState(seed)
    boot, n_empty = [], 0
    for _ in range(n_boot):
        sa = {ka[t] for t in rng.randint(0, len(ka), len(ka))}
        sb = {kb[t] for t in rng.randint(0, len(kb), len(kb))}
        sel = [i for i in range(len(rows)) if cluster_a[i] in sa and cluster_b[i] in sb]
        if len(sel) < 3:
            n_empty += 1
            continue
        v = statistic([rows[i] for i in sel])
        if v is None:
            n_empty += 1
        else:
            boot.append(v)
    if len(boot) < max(50, n_boot // 20):
        return None
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n": len(rows),
            "n_clusters_a": len(ka), "n_clusters_b": len(kb),
            "unit": "crossed bootstrap (both groupings resampled)",
            "n_undefined_resamples": n_empty, "boot_sd": float(np.std(boot))}


def multiway_cluster_ci(values, node_sets, alpha=0.05):
    """Fafchamps-Gubert generalised: each observation carries a SET of nodes, and two observations
    are dependent if their sets intersect at all.

    The two-node version was not enough, and the failure was instructive. The transfer coefficient T
    is a SAME-DRUG statistic: every pair is (drug d in line c1, drug d in line c2). Using the two
    conditions as nodes captures pairs that share a condition -- but two pairs of the SAME drug in
    four different lines share no condition, while sharing the drug's main effect beta(d), which is
    the dominant dependence in the estimate. The estimator therefore returned an interval NARROWER
    than one-way clustering on the drug, and a dependence model that captures a superset of another's
    dependence cannot be narrower. That is the diagnostic.

    With node_sets = {drug, condition_1, condition_2} both channels are captured at once.

        V = (1/N^2) * sum over pairs of observations whose node sets intersect of e_i * e_j
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3 or len(node_sets) != n:
        return None
    m = float(v.mean())
    e = v - m

    holds = defaultdict(list)
    for i, ns in enumerate(node_sets):
        for nd in set(ns):
            holds[nd].append(i)

    # A node shared by EVERY observation makes the dependence graph one cluster, and a cluster-robust
    # variance with G = 1 is undefined: the single cluster's deviations sum to zero, so V collapses to
    # a floating-point residue and the interval comes out absurdly narrow rather than refusing. Catch
    # it on the structure, not on the arithmetic, because whether the residue lands at exactly zero
    # depends on accumulation order.
    if any(len(ix) >= n for ix in holds.values()):
        return {"point": m, "lo": float("nan"), "hi": float("nan"), "se": float("nan"),
                "n": n, "n_nodes": len(holds),
                "unit": "UNDEFINED: one node is shared by every observation, so there is a single "
                        "cluster and a cluster-robust variance does not exist"}

    total = 0.0
    for i, ns in enumerate(node_sets):
        nb = set()
        for nd in set(ns):
            nb.update(holds[nd])
        total += e[i] * float(e[list(nb)].sum())
    var = total / (n ** 2)

    n_nodes = len(holds)
    if var <= 0:
        var = float((e ** 2).sum()) / (n ** 2)
        unit = "multiway estimator non-positive; fell back to independent-sampling variance"
    else:
        unit = f"multiway (Fafchamps-Gubert) over {n_nodes} nodes"
    z = Z95 if abs(alpha - 0.05) < 1e-12 else float(abs(_norm_ppf(alpha / 2)))
    se = math.sqrt(var)
    return {"point": m, "lo": m - z * se, "hi": m + z * se, "se": se,
            "n": n, "n_nodes": n_nodes, "unit": unit}


def dyadic_cluster_ci(values, member_a, member_b, alpha=0.05):
    """Fafchamps-Gubert dyadic-robust interval, for statistics computed over PAIRS.

    The transfer coefficient is estimated from (condition, condition) pairs. Two such pairs are
    dependent if they share EITHER member -- pairs (A,B) and (A,C) both carry A's noise -- and no
    one-way clustering expresses that. Clustering on the drug, which is what the shipped code did,
    captures dependence through the drug and ignores dependence through the shared cell line.
    Cameron-Gelbach-Miller does not apply either: its two groupings must partition the observations,
    and here each observation belongs to two nodes at once.

        V = (1/N^2) * sum over pairs of dyads sharing at least one node of e_d * e_d'

    with e_d the deviation from the mean. Reduces to the ordinary variance when every dyad is
    node-disjoint, and grows as nodes are reused -- which is the behaviour the design requires.
    """
    if len(member_a) != len(values) or len(member_b) != len(values):
        return None
    out = multiway_cluster_ci(values, [{a, b} for a, b in zip(member_a, member_b)], alpha)
    if out:
        out["unit"] = out["unit"].replace("multiway", "dyadic")
    return out


# --------------------------------------------------------------------------- tests
def tost(ci, margin):
    """Two one-sided tests for equivalence, read off an interval.

    Equivalence requires the whole interval inside (-margin, +margin). Reported alongside whether the
    interval also contains zero, because those are different statements and conflating them is how a
    null result gets overclaimed: an interval can exclude zero and still be equivalent (a real but
    negligible effect), or contain zero and NOT be equivalent (simply underpowered).
    """
    if ci is None:
        return None
    lo, hi = ci["lo"], ci["hi"]
    return {"margin": float(margin),
            "equivalent": bool(lo > -margin and hi < margin),
            "contains_zero": bool(lo <= 0 <= hi),
            "verdict": ("equivalent to zero within the margin" if (lo > -margin and hi < margin)
                        else ("excludes zero" if not (lo <= 0 <= hi)
                              else "inconclusive: neither excludes zero nor fits the margin"))}


def permutation_p(rows, labels, statistic, n_perm=5000, seed=0, two_sided=True):
    """Permutation p-value: shuffle `labels` and recompute. Returns the (r+1)/(n+1) form, which
    cannot return exactly zero -- a p of 0 from a finite permutation set is a reporting error."""
    rows = list(rows)
    lab = list(labels)
    if len(rows) != len(lab):
        raise ValueError("rows and labels must be the same length")
    obs = statistic(rows, lab)
    if obs is None:
        return None
    rng = np.random.RandomState(seed)
    more = 0
    n_ok = 0
    for _ in range(n_perm):
        v = statistic(rows, list(rng.permutation(lab)))
        if v is None:
            continue
        n_ok += 1
        more += (abs(v) >= abs(obs)) if two_sided else (v >= obs)
    if n_ok < max(50, n_perm // 20):
        return None
    return {"observed": float(obs), "p": (more + 1) / (n_ok + 1), "n_perm": n_ok,
            "two_sided": bool(two_sided)}


# --------------------------------------------------------------------------- decompositions
def energy_shares(residual, generic, shift=None, atol=1e-6):
    """Exact variance decomposition of shift = residual + generic, cross term included.

    The three shares sum to 1 by construction, so this cannot be used to "check orthogonality" --
    which is what the shipped norm-ratio version was really doing. The cross term is the
    non-orthogonality, reported as a number.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    g = np.asarray(generic, dtype=np.float64).ravel()
    if r.shape != g.shape:
        raise ValueError(f"residual {r.shape} and generic {g.shape} must match")
    s = (r + g) if shift is None else np.asarray(shift, dtype=np.float64).ravel()
    if shift is not None and not np.allclose(s, r + g, atol=1e-4, rtol=1e-3):
        raise ValueError("shift is not residual + generic; the decomposition would be meaningless")
    denom = float(s @ s)
    if denom <= 0:
        return None
    out = {"residual_share": float(r @ r) / denom,
           "generic_share": float(g @ g) / denom,
           "cross_share": float(2.0 * (r @ g)) / denom,
           "cos_residual_generic": float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-12))}
    total = out["residual_share"] + out["generic_share"] + out["cross_share"]
    if abs(total - 1.0) > atol:
        raise AssertionError(f"energy shares sum to {total}, not 1 -- the identity is broken")
    out["sums_to"] = total
    return out


def spearman_brown(r_half):
    """Half-test reliability -> full-test reliability. Returns None outside (0, 1), where the
    correction is not defined and the shipped code silently produced values above 1."""
    if r_half is None or not np.isfinite(r_half) or r_half <= 0 or r_half >= 1:
        return None
    return float(2 * r_half / (1 + r_half))


def disattenuate(observed, reliability_a, reliability_b=None):
    """Correct a correlation for measurement error in one or both variables.

    Returns None rather than a number above 1 when the correction over-corrects: that happens when
    reliability is poorly estimated, and a "correlation" of 1.4 in a results table is worse than a
    gap. The caller should report the ceiling instead.
    """
    ra = reliability_a
    rb = reliability_a if reliability_b is None else reliability_b
    if observed is None or ra is None or rb is None or ra <= 0 or rb <= 0:
        return None
    d = observed / math.sqrt(ra * rb)
    return None if not np.isfinite(d) or abs(d) > 1.0 else float(d)


def _norm_ppf(p):
    """Inverse normal CDF (Acklam), so a non-95% alpha does not require scipy."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def fmt(ci, digits=4):
    """One consistent rendering, so intervals in logs and in the thesis cannot drift apart."""
    if ci is None:
        return "--"
    return (f"{ci['point']:+.{digits}f} [{ci['lo']:+.{digits}f}, {ci['hi']:+.{digits}f}]"
            f"  (n={ci.get('n', '?')}, {ci.get('unit', '?')})")


# --------------------------------------------------------------------------- selftest
def selftest():
    ok = []

    def check(name, cond):
        ok.append((name, bool(cond)))

    # ---- multiplicity, against hand-computable answers ------------------------------------
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    h = holm(p)
    check("holm scales the smallest p by n", abs(h[0] - 0.05) < 1e-12)
    check("holm is monotone in the original order", all(h[i] <= h[i + 1] + 1e-12 for i in range(4)))
    check("holm is never below the raw p", all(h[i] >= p[i] - 1e-12 for i in range(5)))
    q = bh(p)
    check("bh is uniformly no larger than holm", all(q[i] <= h[i] + 1e-12 for i in range(5)))
    check("bh at the largest p equals the raw p", abs(q[-1] - 0.05) < 1e-12)
    check("a single p-value is unchanged by either", holm([0.031])[0] == 0.031 and bh([0.031])[0] == 0.031)
    for bad in ([], [0.5, 1.5], [0.5, float("nan")], [-0.1]):
        try:
            holm(bad); check(f"holm rejects {bad}", False)
        except ValueError:
            check(f"holm rejects malformed input {bad}", True)

    # ---- the statistic really is recomputed inside the resample ---------------------------
    # A RATIO whose denominator is estimated. If the denominator were held fixed the interval would
    # be too narrow; this checks the callable is being applied to the resampled rows at all.
    rng = np.random.RandomState(0)
    rows = [{"num": float(x), "den": float(2 + 0.5 * rng.randn())} for x in rng.randn(400)]
    cl = [i % 20 for i in range(400)]
    seen = {"n": 0}

    def ratio(rs):
        seen["n"] += 1
        d = np.mean([r["den"] for r in rs])
        return float(np.mean([r["num"] for r in rs]) / d) if abs(d) > 1e-9 else None

    ci = cluster_bootstrap(rows, cl, ratio, n_boot=300, seed=1)
    check("cluster_bootstrap evaluates the statistic once per resample plus the point estimate",
          seen["n"] == 301)
    check("cluster_bootstrap returns a bracketing interval", ci["lo"] < ci["point"] < ci["hi"])
    check("cluster_bootstrap reports its unit", ci["n_clusters"] == 20)
    check("fewer than two clusters is refused", cluster_bootstrap(rows, [0] * 400, ratio, 50) is None)

    # ---- crossed design: the two-way interval must be wider than the narrower one-way ------
    # 40 wells x 10 lines. The effect is constant within a well, so wells carry all the dependence
    # and there are more wells than lines -- exactly the configuration that made "cluster on the
    # well" anti-conservative in the real data.
    rng = np.random.RandomState(3)
    well_eff = {w: rng.randn() for w in range(40)}
    line_eff = {l: 0.5 * rng.randn() for l in range(10)}
    vals, wells, lines = [], [], []
    for w in range(40):
        for l in range(10):
            vals.append(well_eff[w] + line_eff[l] + 0.2 * rng.randn())
            wells.append(f"w{w}"); lines.append(f"l{l}")
    tw = two_way_cluster_ci(vals, lines, wells)
    ow_line = cluster_bootstrap(vals, lines, lambda r: float(np.mean(r)), n_boot=400, seed=2)
    ow_well = cluster_bootstrap(vals, wells, lambda r: float(np.mean(r)), n_boot=400, seed=2)
    w_two = tw["hi"] - tw["lo"]
    w_line, w_well = ow_line["hi"] - ow_line["lo"], ow_well["hi"] - ow_well["lo"]
    check("two-way is at least as wide as the narrower one-way", w_two >= min(w_line, w_well) - 1e-9)
    check("the two-way branch is recorded", "two-way" in tw["unit"])
    check("two-way reports both cluster counts", tw["n_clusters_a"] == 10 and tw["n_clusters_b"] == 40)

    # independent data: two-way should NOT be dramatically wider than iid, i.e. it is not just
    # inflating everything
    rng = np.random.RandomState(4)
    iid = rng.randn(400).tolist()
    ti = two_way_cluster_ci(iid, list(range(400)), [i % 200 for i in range(400)])
    check("two-way does not inflate an independent sample", (ti["hi"] - ti["lo"]) < 0.5)

    # ---- dyadic clustering: pairs sharing a node must widen the interval -------------------
    # 12 nodes, every pair present. Node effects make dyads sharing a node correlated; an
    # independent-sampling interval would be far too narrow.
    rng = np.random.RandomState(7)
    node_eff = {i: rng.randn() for i in range(12)}
    dv, da, db = [], [], []
    for i in range(12):
        for j in range(i + 1, 12):
            dv.append(node_eff[i] + node_eff[j] + 0.1 * rng.randn())
            da.append(i); db.append(j)
    dy = dyadic_cluster_ci(dv, da, db)
    naive = float(np.std(dv, ddof=1) / math.sqrt(len(dv)))
    check("dyadic widens an interval when nodes are reused", dy["se"] > 1.5 * naive)
    check("dyadic counts its nodes", dy["n_nodes"] == 12)
    check("dyadic reports the estimator used", "Fafchamps" in dy["unit"])
    # node-disjoint dyads carry no shared noise, so it should agree with the naive interval
    rng = np.random.RandomState(8)
    k = 60
    iv = rng.randn(k).tolist()
    ia = [2 * i for i in range(k)]
    ib = [2 * i + 1 for i in range(k)]
    dj = dyadic_cluster_ci(iv, ia, ib)
    naive_dj = float(np.std(iv, ddof=0) / math.sqrt(k))
    # OMITTING A REAL DEPENDENCE CHANNEL gives a too-narrow interval. This is the production failure:
    # the transfer coefficient is a SAME-DRUG statistic, so two pairs of one drug share beta(drug)
    # while sharing no condition, and leaving the drug out of the node set made the "robust" interval
    # narrower than plain one-way clustering on the drug.
    #
    # Note the property being tested is NOT "more nodes always widen" -- that is false in general,
    # because the estimator sums signed cross terms. It is "including a channel that carries REAL
    # shared variance widens", which is the case that matters.
    rng = np.random.RandomState(11)
    n_drug, per = 8, 10
    drug_eff = {d: rng.randn() for d in range(n_drug)}
    mv, m_nodrug, m_drug = [], [], []
    uid = 0
    for d in range(n_drug):
        for _ in range(per):
            mv.append(drug_eff[d] + 0.25 * rng.randn())
            m_nodrug.append({f"c{uid}", f"c{uid + 1}"})            # every observation node-disjoint
            m_drug.append({f"c{uid}", f"c{uid + 1}", f"drug::{d}"})
            uid += 2
    without = multiway_cluster_ci(mv, m_nodrug)
    with_ = multiway_cluster_ci(mv, m_drug)
    check("omitting a real dependence channel understates the interval",
          with_["se"] > 1.5 * without["se"])
    check("the node count reflects the added channel", with_["n_nodes"] > without["n_nodes"])

    # degenerate case: if EVERY observation shares one node there is a single cluster, its deviations
    # sum to zero, the estimator is non-positive, and the fallback must be declared rather than silent
    allshare = multiway_cluster_ci(mv, [{"one"} for _ in mv])
    check("a single all-inclusive cluster is refused rather than given a number",
          allshare is not None and "UNDEFINED" in allshare["unit"]
          and not np.isfinite(allshare["se"]))

    check("dyadic reduces to the ordinary interval when no node is reused",
          abs(dj["se"] - naive_dj) < 0.02 * max(naive_dj, 1e-9) + 1e-9)

    # ---- TOST separates the three verdicts -------------------------------------------------
    check("a tight interval around zero is equivalent",
          tost({"lo": -0.01, "hi": 0.01}, 0.05)["equivalent"])
    check("an interval excluding zero but inside the margin is BOTH",
          tost({"lo": 0.01, "hi": 0.02}, 0.05)["equivalent"] and
          not tost({"lo": 0.01, "hi": 0.02}, 0.05)["contains_zero"])
    check("a wide interval containing zero is inconclusive, not equivalent",
          not tost({"lo": -0.4, "hi": 0.4}, 0.05)["equivalent"] and
          "inconclusive" in tost({"lo": -0.4, "hi": 0.4}, 0.05)["verdict"])

    # ---- permutation ------------------------------------------------------------------------
    rng = np.random.RandomState(5)
    x = rng.randn(200)
    lab_real = [(1 if v > 0 else 0) for v in x]         # perfectly informative labels

    def diff(rs, ls):
        a = [r for r, l in zip(rs, ls) if l == 1]
        b = [r for r, l in zip(rs, ls) if l == 0]
        return None if not a or not b else float(np.mean(a) - np.mean(b))

    pr = permutation_p(list(x), lab_real, diff, n_perm=400, seed=1)
    check("a real effect gets a small permutation p", pr["p"] < 0.01)
    pn = permutation_p(list(x), list(rng.permutation(lab_real)), diff, n_perm=400, seed=1)
    check("a scrambled label gets a large permutation p", pn["p"] > 0.05)
    check("permutation p can never be exactly zero", pr["p"] > 0)

    # ---- energy shares -----------------------------------------------------------------------
    rng = np.random.RandomState(6)
    r = rng.randn(500)
    g = rng.randn(500)
    es = energy_shares(r, g)
    check("energy shares sum to exactly one", abs(es["sums_to"] - 1.0) < 1e-9)
    check("a positive cross term is reported, not hidden",
          "cross_share" in es and abs(es["cross_share"]) > 0)
    # orthogonal by construction -> cross term vanishes and the shares are the squared norms
    a = np.zeros(4); a[0] = 3.0
    b = np.zeros(4); b[1] = 4.0
    eo = energy_shares(a, b)
    check("orthogonal parts give a zero cross term", abs(eo["cross_share"]) < 1e-12)
    check("orthogonal shares are the squared-norm fractions",
          abs(eo["residual_share"] - 9 / 25) < 1e-12 and abs(eo["generic_share"] - 16 / 25) < 1e-12)
    try:
        energy_shares(a, b, shift=np.ones(4)); check("a mismatched shift is refused", False)
    except ValueError:
        check("a shift that is not residual+generic is refused", True)

    # ---- reliability corrections ---------------------------------------------------------------
    check("spearman-brown raises a half-test reliability", abs(spearman_brown(0.5) - 2 / 3) < 1e-12)
    check("spearman-brown refuses values outside (0,1)",
          spearman_brown(0.0) is None and spearman_brown(1.0) is None and spearman_brown(-0.2) is None)
    check("disattenuation raises an observed correlation",
          disattenuate(0.5, 0.8) > 0.5)
    check("disattenuation returns None instead of a correlation above 1",
          disattenuate(0.95, 0.5) is None)

    for name, passed in ok:
        logger.info(("  ok   " if passed else "  FAIL ") + name)
    allok = all(p for _, p in ok)
    logger.info(f"SELFTEST {'PASSED' if allok else 'FAILED'}  ({sum(p for _, p in ok)}/{len(ok)})")
    if not allok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    ap.error("this module is a library; run --selftest to verify it")


if __name__ == "__main__":
    main()

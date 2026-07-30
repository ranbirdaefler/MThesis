#!/usr/bin/env python
"""
variance_decomposition.py — how much of the drug-specific residual is a per-drug CONSTANT
that transfers across cell lines, vs a drug x cell-line INTERACTION?

WHY THIS EXISTS
---------------
Q16 produced two numbers that point opposite ways and cannot both be read at face value:

    ceiling (within-condition replicate)          0.968
    drug_lookup   (~38 other conditions, avg)     0.963   <- ties the ceiling
    drug_lookup_1 (ONE other cell line)           0.832   <- 0.136 BELOW the ceiling

`drug_lookup_1` falling below the ceiling argues there IS cell-line-specific structure.
`drug_lookup` recovering to the ceiling argues there is NOT -- averaging over cell lines can
only remove NOISE, never the systematic miss of the target line's own interaction, so a large
interaction should have left it on a plateau well below the ceiling.

NIR cannot adjudicate this. At 0.96 the metric is in its compressive shoulder: a wide range of
cosines collapses into a few thousandths, and the ceiling is additionally scored against a
HALF-sample truth while every baseline is scored against the FULL-sample truth. So the question
has to be answered in cosine space, with the noise divided out explicitly.

THE MODEL
---------
    residual(d, c) = beta(d) + kappa(d, c) + eps

    beta(d)     per-drug main effect -- the same in every cell line
    kappa(d,c)  drug x cell-line interaction -- the ONLY thing a conditional model can win on
    eps         measurement noise

    T = var(beta) / (var(beta) + var(kappa))        the TRANSFER COEFFICIENT

T is the fraction of noise-free residual variance that is a per-drug constant.

    T -> 1   the drug's residual IS a per-drug constant. A lookup table is the correct model,
             there is nothing cell-line-specific to condition on, and the failure of every
             conditional arm in this project (Q12 consensus, Q14 OT, the whole eps-ladder) has a
             structural explanation rather than an engineering one.
    T -> 0   the response is mostly interaction. There is a real conditional target, the lookup's
             0.963 is an artifact of NIR compression, and one more modelling arm is justified.

ESTIMATOR (disattenuated cross-line correlation)
------------------------------------------------
    r_between(d; c1, c2) = cos(residual(d, c1), residual(d, c2))
    rel(d, c)            = reliability of the FULL-sample residual
                         = Spearman-Brown of cos(residual_A, residual_B)
    T_hat                = r_between / sqrt(rel(d,c1) * rel(d,c2))

Dividing by the geometric mean reliability removes measurement noise, which otherwise makes every
correlation look like interaction.

SIX CONTROLS, EACH OF WHICH CAN FLIP THE ANSWER
------------------------------------------------
(1) CONTROL HALF-SPLIT. `build_residual_targets.build_residuals` subtracts ONE `ctrl_pb` from both
    halves, so control-estimation error is COMMON to A and B. That inflates cos(rA, rB), inflates
    reliability, and therefore biases T DOWNWARD -- it would manufacture apparent interaction. We
    rebuild with DISJOINT control halves and report both, so the bias is measured, not assumed.
(2) SAME-PLATE DIFFERENT-DRUG negative control. If different drugs "transfer" too, what we are
    measuring is a shared artifact (incomplete generic removal, batch) and not drug biology.
(3) SIMULATED NULL. At kappa = 0 this estimator does NOT return 1.0 -- finite samples, the
    Spearman-Brown approximation, and non-normality all cost a little. We simulate at the MEASURED
    noise level and report T_null, so T is read against its own null and not against 1.0.
(4) DOSE STRATA. 62% of drugs are multi-dose (Q11). Same drug + same cell line + different dose is
    NOT cross-line transfer; pooling it would contaminate T. Reported separately -- and the
    same-line/different-dose value is itself the dose-interaction estimate.
(5) LEAVE-ONE-OUT GENERIC. generic(c) is a mean over drugs that INCLUDES drug d, leaving a -1/m
    self-term in its own residual. With m = 20-204 drugs per line this is a shrinkage, not a leak,
    but it is cheap to remove and it shifts small effects.
(6) ||residual|| / ||shift||. What FRACTION of the total perturbation response does this frame even
    cover? Never computed in this project. Every claim we make is scoped by it: closing the
    drug-specific component means little if that component is 5% of the response.

USAGE (CPU, ~1-2 h on the 620k-cell cache)
    python variance_decomposition.py --cache_dir /data/.../ot_cache --out RESULTS/variance_decomp.json

SELFTEST (no data, no network -- plants a known T and checks recovery)
    python variance_decomposition.py --selftest
"""
import argparse, json, logging, os, sys
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# dual path bootstrap: flat cluster layout (~/tahoe/*.py) and the split repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ot"))


# --------------------------------------------------------------------------- primitives
def _reject(x, g):
    """Component of x orthogonal to g. Removes the generic DIRECTION entirely, so no per-drug
    scaling of the generic program can survive into the residual."""
    gg = float(g @ g)
    return x if gg < 1e-12 else x - (float(x @ g) / gg) * g


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def spearman_brown(r_half):
    """Reliability of the FULL sample from the correlation between two HALVES.

    Doubling the sample size raises reliability as 2r/(1+r). Skipping this step would divide by
    the reliability of a half-sample, over-correcting T upward."""
    r = float(np.clip(r_half, -0.999, 0.999))
    return 2.0 * r / (1.0 + r)


def clustered_ci(values, clusters, rng, n_boot=2000):
    """Bootstrap resampling CLUSTERS, not observations. The unit here is the DRUG: a drug seen in
    12 cell lines contributes 66 pairs that are anything but independent, so an observation-level
    CI would be badly anti-conservative."""
    v = np.asarray(values, dtype=float)
    cl = np.asarray(clusters)
    if len(v) == 0:
        return None
    u = sorted(set(cl.tolist()))
    idx_of = {k: np.where(cl == k)[0] for k in u}
    boot = []
    for _ in range(n_boot):
        take = rng.randint(0, len(u), len(u))
        sel = np.concatenate([idx_of[u[t]] for t in take])
        boot.append(v[sel].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), len(v), len(u)


# --------------------------------------------------------------------------- residual construction
def build_conditions(cache_dir, min_treated=40, min_control=20, seed=0,
                     split_controls=True, loo_generic=False, generic_scope="cell_line",
                     project_generic=False):
    """Per-condition residuals with the halves built from DISJOINT cells AND (optionally) disjoint
    control cells.

    `split_controls=True` is the corrected build. The production builder
    (build_residual_targets.build_residuals) subtracts one shared `ctrl_pb` from both halves; the
    shared error then appears in both rA and rB, inflating cos(rA,rB). Since T divides by that
    reliability, the shared-control build biases T DOWN -- i.e. toward falsely concluding that
    interaction exists. Running both settings quantifies the bias instead of arguing about it.
    """
    import pandas as pd
    from scipy import sparse

    rng = np.random.RandomState(seed)
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    X = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    plate = meta["plate"].astype(str).values
    drug = meta["drug"].astype(str).values
    dose = meta["dose"].astype(str).values

    L = lambda ix: np.asarray(np.log1p(X[ix].todense()).mean(0)).ravel().astype(np.float32)

    # control pseudobulk per (cell_line, plate) -- full, and two disjoint halves
    ctrl = {}
    for g in set(zip(cl[is_ctrl], plate[is_ctrl])):
        idx = np.where(is_ctrl & (cl == g[0]) & (plate == g[1]))[0]
        if len(idx) < min_control:
            continue
        idx = list(idx)
        rng.shuffle(idx)
        h = len(idx) // 2
        ctrl[g] = {"full": L(idx), "A": L(idx[:h]), "B": L(idx[h:]), "n": len(idx)}

    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], cl[i], plate[i], dose[i])].append(i)

    cond = {}
    for key, idxs in by.items():
        g = (key[1], key[2])
        if len(idxs) < min_treated or g not in ctrl:
            continue
        idxs = list(idxs)
        rng.shuffle(idxs)
        h = len(idxs) // 2
        cA, cB = ("A", "B") if split_controls else ("full", "full")
        cond[key] = {
            "shift_full": L(idxs) - ctrl[g]["full"],
            "shift_A": L(idxs[:h]) - ctrl[g][cA],
            "shift_B": L(idxs[h:]) - ctrl[g][cB],
            "group": g, "n_cells": len(idxs), "n_ctrl": ctrl[g]["n"],
        }
    logger.info(f"conditions with enough cells: {len(cond)}  "
                f"(split_controls={split_controls}, loo_generic={loo_generic})")

    # GENERIC shift = mean over drugs. Scope is a DIAGNOSTIC, not a fixed choice:
    #   cell_line -- what the production pipeline uses (62% reproducible vs 19% at plate scope)
    #   plate     -- fewer drugs per group, so a noisier mean, BUT it removes plate/batch structure.
    # A cell-line-scoped generic leaves any plate-specific component in EVERY residual on that plate,
    # which makes unrelated same-plate conditions correlate. That is a candidate explanation for a
    # failing negative control, so both scopes must be runnable.
    by_cl = defaultdict(list)
    for k in cond:
        by_cl[k[1] if generic_scope == "cell_line" else (k[1], k[2])].append(k)

    out = {}
    for c, keys in by_cl.items():
        if len(keys) < 3:
            continue
        stack = {h: np.stack([cond[k][f"shift_{h}"] for k in keys]) for h in ("full", "A", "B")}
        tot = {h: stack[h].sum(0) for h in stack}
        m = len(keys)
        for j, k in enumerate(keys):
            if loo_generic and m > 1:
                # generic estimated WITHOUT this drug -> removes the -1/m self-term
                gen = {h: (tot[h] - stack[h][j]) / (m - 1) for h in stack}
            else:
                gen = {h: tot[h] / m for h in stack}
            if project_generic:
                # Remove the generic DIRECTION, not a fixed vector. Subtracting mean-over-drugs
                # removes the generic program's AVERAGE magnitude but not each drug's own scaling
                # of it: if shift(d,c) ~ a_d*g(c) + specific, then residual keeps
                # (a_d - mean(a))*g(c). Every potent drug then carries a positive multiple of g(c)
                # and every weak drug a negative one, so unrelated drugs correlate and the same
                # drug correlates with itself across lines for a reason that is not drug identity.
                # Projecting per condition removes that component exactly.
                rA = _reject(cond[k]["shift_A"], gen["A"])
                rB = _reject(cond[k]["shift_B"], gen["B"])
            else:
                rA = cond[k]["shift_A"] - gen["A"]
                rB = cond[k]["shift_B"] - gen["B"]
            r_full = (_reject(cond[k]["shift_full"], gen["full"]) if project_generic
                      else cond[k]["shift_full"] - gen["full"])
            out[k] = {
                "residual": r_full.astype(np.float32),
                "residual_A": rA.astype(np.float32), "residual_B": rB.astype(np.float32),
                "repro_cos": cos(rA, rB),
                "shift_norm": float(np.linalg.norm(cond[k]["shift_full"])),
                "generic_norm": float(np.linalg.norm(gen["full"])),
                "group": cond[k]["group"], "n_cells": cond[k]["n_cells"],
                "n_ctrl": cond[k]["n_ctrl"],
            }
    cs = np.array([v["repro_cos"] for v in out.values()])
    logger.info(f"residuals: {len(out)} conditions | mean cos(A,B) = {cs.mean():+.3f} "
                f"| frac > 0.2 = {100 * (cs > 0.2).mean():.0f}%")
    return out


# --------------------------------------------------------------------------- the estimator
def transfer_pairs(kept, same_drug=True, same_dose_only=False, cross_line=True, max_pairs_per_drug=60,
                   seed=0, min_rel=0.05):
    """Build (condition, condition) pairs and disattenuate.

    same_drug=True, cross_line=True   -> T, the quantity of interest
    same_drug=False                   -> the NEGATIVE CONTROL (different drugs, same plate)
    cross_line=False                  -> same cell line, different dose: the dose-interaction arm
    """
    rng = np.random.RandomState(seed)
    rows = []
    if same_drug:
        by_drug = defaultdict(list)
        for k in kept:
            by_drug[k[0]].append(k)
        groups = list(by_drug.items())
    else:
        # negative control: different drugs sharing a PLATE (batch held constant -- the hardest null)
        by_plate = defaultdict(list)
        for k in kept:
            by_plate[(k[1], k[2])].append(k)
        groups = [(f"plate::{g}", ks) for g, ks in by_plate.items()]

    for gname, keys in groups:
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if same_drug and k1[0] != k2[0]:
                    continue
                if not same_drug and k1[0] == k2[0]:
                    continue
                diff_line = k1[1] != k2[1]
                if cross_line and not diff_line:
                    continue
                if not cross_line and diff_line:
                    continue
                if same_dose_only and k1[3] != k2[3]:
                    continue
                pairs.append((k1, k2))
        if len(pairs) > max_pairs_per_drug:
            sel = rng.choice(len(pairs), max_pairs_per_drug, replace=False)
            pairs = [pairs[s] for s in sel]
        for k1, k2 in pairs:
            rel1 = spearman_brown(kept[k1]["repro_cos"])
            rel2 = spearman_brown(kept[k2]["repro_cos"])
            if rel1 < min_rel or rel2 < min_rel:
                continue
            r_between = cos(kept[k1]["residual"], kept[k2]["residual"])
            rows.append({
                "cluster": k1[0] if same_drug else gname,
                "drug": k1[0], "cl1": k1[1], "cl2": k2[1],
                "same_plate": k1[2] == k2[2], "same_dose": k1[3] == k2[3],
                "r_between": r_between,
                "rel_gm": float(np.sqrt(rel1 * rel2)),
                "T": float(r_between / np.sqrt(rel1 * rel2)),
            })
    return rows


def cross_line_diff_drug_pairs(kept, n_pairs=4000, seed=0, min_rel=0.05):
    """STRUCTURE-MATCHED negative control: DIFFERENT drug, DIFFERENT cell line.

    This differs from T in exactly ONE way -- the drug match is broken -- while preserving T's
    cross-cell-line (and hence cross-plate) structure. The same-plate control cannot do this job:
    it breaks the drug match AND holds the cell line fixed, so it differs from T in two ways at
    once and a difference between them is not attributable to drug identity.

    It needs its own sampler because transfer_pairs() groups different-drug pairs BY PLATE, which
    forces every such pair to share a cell line -- so asking it for cross_line=True silently
    returns nothing. (That is exactly what happened on the first plate-scope run: the verdict
    fell back to the same-plate control without saying so.)
    """
    rng = np.random.RandomState(seed + 31337)
    ks = list(kept.keys())
    if len(ks) < 4:
        return []
    rows = []
    for _ in range(n_pairs * 4):
        if len(rows) >= n_pairs:
            break
        k1 = ks[rng.randint(len(ks))]
        k2 = ks[rng.randint(len(ks))]
        if k1[0] == k2[0] or k1[1] == k2[1]:      # need different drug AND different cell line
            continue
        rel1 = spearman_brown(kept[k1]["repro_cos"])
        rel2 = spearman_brown(kept[k2]["repro_cos"])
        if rel1 < min_rel or rel2 < min_rel:
            continue
        r_between = cos(kept[k1]["residual"], kept[k2]["residual"])
        rows.append({"cluster": k1[0], "drug": k1[0], "cl1": k1[1], "cl2": k2[1],
                     "same_plate": False, "same_dose": k1[3] == k2[3],
                     "r_between": r_between, "rel_gm": float(np.sqrt(rel1 * rel2)),
                     "T": float(r_between / np.sqrt(rel1 * rel2))})
    return rows


def simulate_null(kept, n_drugs=60, n_lines=6, seed=0, kappa_frac=0.0):
    """What does the estimator return when the interaction is EXACTLY `kappa_frac`?

    Calibrated to the real data: gene dimension, per-condition cell counts and the empirical
    residual/noise scale are all taken from `kept`. Without this we would be reading T against
    1.0, which it never reaches even at kappa = 0.
    """
    rng = np.random.RandomState(seed)
    vals = list(kept.values())
    P = len(vals[0]["residual"])
    # empirical scales: signal from the full residual, noise from the A/B disagreement
    sig = float(np.mean([np.linalg.norm(v["residual"]) for v in vals]) / np.sqrt(P))
    noi = float(np.mean([np.linalg.norm(v["residual_A"] - v["residual_B"]) for v in vals])
                / (2 * np.sqrt(P)))
    ncell = int(np.median([v["n_cells"] for v in vals]))
    s_beta = sig * np.sqrt(max(0.0, 1.0 - kappa_frac))
    s_kap = sig * np.sqrt(kappa_frac)

    synth = {}
    for d in range(n_drugs):
        beta = rng.randn(P).astype(np.float32) * s_beta
        for c in range(n_lines):
            kap = rng.randn(P).astype(np.float32) * s_kap
            true = beta + kap
            # two disjoint halves, each with independent noise at the measured level
            eA = rng.randn(P).astype(np.float32) * noi * np.sqrt(2.0)
            eB = rng.randn(P).astype(np.float32) * noi * np.sqrt(2.0)
            rA, rB = true + eA, true + eB
            synth[(f"d{d}", f"c{c}", "p0", "1.0")] = {
                "residual": (0.5 * (rA + rB)).astype(np.float32),
                "residual_A": rA, "residual_B": rB, "repro_cos": cos(rA, rB),
                "shift_norm": 1.0, "generic_norm": 0.0,
                "group": (f"c{c}", "p0"), "n_cells": ncell, "n_ctrl": 100,
            }
    rows = transfer_pairs(synth, same_drug=True, cross_line=True, seed=seed)
    return float(np.mean([r["T"] for r in rows])) if rows else float("nan"), 1.0 - kappa_frac


# --------------------------------------------------------------------------- reporting
def report(kept, args, rng):
    out = {}
    logger.info("=" * 100)

    # ---- (6) SCOPE: what fraction of the response is the drug-specific residual? ----
    rn = np.array([np.linalg.norm(v["residual"]) for v in kept.values()])
    sn = np.array([v["shift_norm"] for v in kept.values()])
    gn = np.array([v["generic_norm"] for v in kept.values()])
    frac = float(np.mean(rn / np.maximum(sn, 1e-9)))
    out["scope"] = {"residual_over_shift": frac,
                    "generic_over_shift": float(np.mean(gn / np.maximum(sn, 1e-9))),
                    "mean_residual_norm": float(rn.mean()), "mean_shift_norm": float(sn.mean())}
    logger.info(f"  SCOPE  ||residual|| / ||shift|| = {frac:.3f}   "
                f"||generic|| / ||shift|| = {out['scope']['generic_over_shift']:.3f}")
    orth = frac ** 2 + out["scope"]["generic_over_shift"] ** 2
    logger.info(f"         => the drug-SPECIFIC component is {100*frac:.0f}% of the response this "
                f"project models. EVERY claim is scoped by this number.")
    logger.info(f"         orthogonality check: (r/s)^2 + (g/s)^2 = {orth:.3f}  "
                + ("(~1 => residual and generic are essentially ORTHOGONAL, so the decomposition is "
                   "clean and the two fractions are variance shares)" if abs(orth - 1) < 0.05 else
                   "(FAR from 1 => the components overlap; the fractions are NOT variance shares)"))
    out["scope"]["orthogonality"] = float(orth)

    # ---- the transfer coefficient, on the reliability-filtered set and unfiltered ----
    for tag, sub in (("repro-filtered (cos>0.2, the training set)",
                      {k: v for k, v in kept.items() if v["repro_cos"] > args.repro_thr}),
                     ("ALL conditions (no reliability filter)", kept)):
        rows = transfer_pairs(sub, same_drug=True, cross_line=True,
                              same_dose_only=args.same_dose_only, seed=args.seed)
        if not rows:
            continue
        ci = clustered_ci([r["T"] for r in rows], [r["cluster"] for r in rows], rng, args.n_boot)
        rb = float(np.mean([r["r_between"] for r in rows]))
        rl = float(np.mean([r["rel_gm"] for r in rows]))
        m, lo, hi, n, ncl = ci
        out[f"T::{tag}"] = {"T": m, "ci": [lo, hi], "n_pairs": n, "n_drugs": ncl,
                            "mean_r_between": rb, "mean_reliability": rl}
        logger.info(f"  T  [{tag}]")
        logger.info(f"       raw cross-line cos {rb:+.3f} / sqrt(rel {rl:.3f}) "
                    f"-> T = {m:.3f}  CI [{lo:.3f}, {hi:.3f}]  ({n} pairs, {ncl} drugs)")

    kept_f = {k: v for k, v in kept.items() if v["repro_cos"] > args.repro_thr}

    # ---- (2) NEGATIVE CONTROLS ----
    # TWO of them. The same-plate one holds batch constant but differs from T in TWO ways at once
    # (different drug AND same cell line), so it cannot be subtracted from T directly. The
    # STRUCTURE-MATCHED one differs from T in exactly one way -- the drug match is broken, the
    # cross-cell-line structure is preserved -- and it is the one the verdict is gated on.
    neg_sources = {
        "diff_drug_same_plate": lambda: transfer_pairs(kept_f, same_drug=False, cross_line=False,
                                                       seed=args.seed),
        "diff_drug_cross_line": lambda: cross_line_diff_drug_pairs(kept_f, seed=args.seed),
    }
    neg_T = {}
    for name, src in neg_sources.items():
        rows = src()
        if not rows:
            logger.warning(f"  negative control [{name}] produced NO PAIRS -- it cannot gate anything")
            continue
        ci = clustered_ci([r["T"] for r in rows], [r["cluster"] for r in rows], rng, args.n_boot)
        m, lo, hi, n, _ = ci
        raw = float(np.mean([r["r_between"] for r in rows]))
        neg_T[name] = {"T": m, "ci": [lo, hi], "n_pairs": n, "raw_cos": raw}
        tag = "  <- STRUCTURE-MATCHED: the verdict is gated on this" if "cross_line" in name else ""
        logger.info(f"  NEGATIVE CONTROL [{name}]  raw cos {raw:+.3f} -> T = {m:+.3f} "
                    f"CI [{lo:+.3f}, {hi:+.3f}]  ({n} pairs){tag}")
    out["negative_controls"] = neg_T
    logger.info("       Both must be ~0. If a negative control approaches T, then what T measures is "
                "NOT drug identity -- it is a component shared by unrelated conditions (incomplete "
                "generic removal, or plate/batch structure surviving a cell-line-scoped generic).")

    # ---- (4) DOSE: same drug, SAME line, different dose ----
    dose_rows = transfer_pairs(kept_f, same_drug=True, cross_line=False, seed=args.seed)
    dose_rows = [r for r in dose_rows if not r["same_dose"]]
    if dose_rows:
        ci = clustered_ci([r["T"] for r in dose_rows], [r["cluster"] for r in dose_rows], rng, args.n_boot)
        m, lo, hi, n, ncl = ci
        out["dose_transfer_same_line"] = {"T": m, "ci": [lo, hi], "n_pairs": n}
        logger.info(f"  DOSE (same drug, same cell line, DIFFERENT dose)  T = {m:.3f} "
                    f"CI [{lo:.3f}, {hi:.3f}]  ({n} pairs)")
        logger.info("       an upper reference for cross-line T: if cross-line T is close to this, "
                    "changing the cell line costs about as much as changing the dose.")

    # ---- (3) SIMULATED NULL ----
    sims = {}
    for kf in (0.0, 0.25, 0.5):
        t_hat, t_true = simulate_null(kept_f, seed=args.seed, kappa_frac=kf)
        sims[f"kappa_frac={kf}"] = {"T_hat": t_hat, "T_true": t_true}
        logger.info(f"  SIMULATED NULL  true T = {t_true:.2f}  ->  estimator returns {t_hat:.3f}")
    out["simulated_null"] = sims
    t_null = sims["kappa_frac=0.0"]["T_hat"]

    # ---- verdict -- GATED ON THE NEGATIVE CONTROL ----
    # An earlier version of this function compared T to the simulated null and printed a verdict
    # WITHOUT consulting the negative control. On the first real run that produced a confident
    # "a REAL interaction exists" while the structure-matched null sat within 0.03 of T. The gate
    # now runs first and can veto; a verdict is never printed on an uninterpretable T.
    key = [k for k in out if k.startswith("T::repro-filtered")]
    if key and np.isfinite(t_null):
        T = out[key[0]]["T"]
        lo, hi = out[key[0]]["ci"]
        gap = t_null - T
        nm = neg_T.get("diff_drug_cross_line") or neg_T.get("diff_drug_same_plate")
        logger.info("-" * 100)
        logger.info(f"  VERDICT  T = {T:.3f} [{lo:.3f}, {hi:.3f}]  vs  simulated kappa=0 null "
                    f"{t_null:.3f}   (shortfall {gap:+.3f})")
        verdict = None
        if nm is not None and (T - nm["T"]) < args.min_neg_margin:
            verdict = "VOID"
            logger.info(f"     >>> VERDICT VOID. The negative control is {nm['T']:+.3f}, only "
                        f"{T - nm['T']:+.3f} below T (need >= {args.min_neg_margin:+.3f}). Unrelated "
                        "conditions transfer about as well as the same drug does, so the shortfall "
                        "below the null is NOT evidence of a drug x cell-line interaction -- it is a "
                        "component shared by conditions that have nothing to do with each other.")
            logger.info("     >>> Do NOT report an interaction fraction from this run. Diagnose first: "
                        "(a) plate-scope generic (--generic_scope plate) -- a cell-line-scoped generic "
                        "leaves plate/batch structure in every residual; (b) control-split noise -- "
                        "Spearman-Brown corrects for halving the TREATED cells, not the CONTROL cells, "
                        "so rel is under-estimated and every T is inflated by the same factor, which is "
                        "exactly what makes all arms converge; (c) raise --min_control.")
        elif hi >= t_null - 0.03:
            verdict = "PER_DRUG_CONSTANT"
            logger.info("     => the residual is, to measurement precision, a PER-DRUG CONSTANT. There is "
                        "no cell-line-specific component for a conditional model to capture; a lookup "
                        "table is the correct model and the conditional arms failed for a structural "
                        "reason. The interaction budget is <= "
                        f"{100*max(0.0, gap):.0f}% of residual variance.")
        else:
            verdict = "REAL_INTERACTION"
            logger.info(f"     => a REAL interaction exists: roughly {100*gap:.0f}% of the residual "
                        "variance is drug x cell-line, ABOVE what unrelated conditions share "
                        f"(negative control {nm['T']:+.3f}). That is the conditional target, and it is "
                        "what drug_lookup structurally cannot reach.")
        out["verdict"] = {"T": T, "ci": [lo, hi], "T_null": t_null, "verdict": verdict,
                          "negative_control": (nm or {}).get("T"),
                          "interaction_frac": max(0.0, gap) if verdict == "REAL_INTERACTION" else None}
    logger.info("=" * 100)
    return out


# --------------------------------------------------------------------------- selftest
def selftest():
    """Plant a known T and check recovery. No data, no network."""
    rng = np.random.RandomState(0)
    P, ok = 946, True
    for true_kappa in (0.0, 0.3, 0.6):
        synth = {}
        sig, noi = 1.0, 0.6
        for d in range(50):
            beta = rng.randn(P) * sig * np.sqrt(1 - true_kappa)
            for c in range(5):
                kap = rng.randn(P) * sig * np.sqrt(true_kappa)
                true = beta + kap
                rA = true + rng.randn(P) * noi * np.sqrt(2)
                rB = true + rng.randn(P) * noi * np.sqrt(2)
                synth[(f"d{d}", f"c{c}", "p0", "1")] = {
                    "residual": 0.5 * (rA + rB), "residual_A": rA, "residual_B": rB,
                    "repro_cos": cos(rA, rB), "shift_norm": 1.0, "generic_norm": 0.0,
                    "group": (f"c{c}", "p0"), "n_cells": 50, "n_ctrl": 50}
        rows = transfer_pairs(synth, same_drug=True, cross_line=True, seed=1)
        T = float(np.mean([r["T"] for r in rows]))
        raw = float(np.mean([r["r_between"] for r in rows]))
        exp = 1 - true_kappa
        good = abs(T - exp) < 0.08
        ok &= good
        print(f"  kappa_frac={true_kappa:.1f}  expected T={exp:.2f}  raw cos={raw:.3f}  "
              f"disattenuated T={T:.3f}  {'OK' if good else 'FAIL'}")
        print(f"      (raw cos is depressed by noise to {raw:.3f}; without disattenuation we would "
              f"report {1-raw:.0%} interaction where the truth is {true_kappa:.0%})")

    # negative control must be ~0 on independent drugs
    synth = {}
    for d in range(40):
        for c in range(4):
            v = rng.randn(P)
            rA, rB = v + rng.randn(P) * 0.5, v + rng.randn(P) * 0.5
            synth[(f"d{d}", f"c{c}", "p0", "1")] = {
                "residual": 0.5 * (rA + rB), "residual_A": rA, "residual_B": rB,
                "repro_cos": cos(rA, rB), "shift_norm": 1.0, "generic_norm": 0.0,
                "group": (f"c{c}", "p0"), "n_cells": 50, "n_ctrl": 50}
    neg = transfer_pairs(synth, same_drug=False, cross_line=False, seed=1)
    nm = float(np.mean([r["T"] for r in neg])) if neg else float("nan")
    good = abs(nm) < 0.05
    ok &= good
    print(f"  negative control (independent drugs): T={nm:+.3f}  {'OK' if good else 'FAIL'}")

    # the structure-matched sampler must produce pairs AND read ~0 on independent drugs
    xl = cross_line_diff_drug_pairs(synth, n_pairs=800, seed=1)
    xm = float(np.mean([r["T"] for r in xl])) if xl else float("nan")
    good = len(xl) > 200 and abs(xm) < 0.05
    ok &= good
    print(f"  structure-matched control (diff drug, diff line): n={len(xl)} T={xm:+.3f}  "
          f"{'OK' if good else 'FAIL'}")

    # Spearman-Brown sanity
    good = abs(spearman_brown(0.5) - 2 / 3) < 1e-9
    ok &= good
    print(f"  spearman_brown(0.5) = {spearman_brown(0.5):.4f} (expect 0.6667)  {'OK' if good else 'FAIL'}")

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2)
    ap.add_argument("--same_dose_only", action="store_true",
                    help="restrict cross-line pairs to the SAME dose (cleaner T, fewer pairs)")
    ap.add_argument("--shared_controls", action="store_true",
                    help="reproduce the production build (one ctrl_pb for both halves) to MEASURE "
                         "the bias it induces; the default corrected build splits controls")
    ap.add_argument("--loo_generic", action="store_true")
    ap.add_argument("--generic_scope", choices=["cell_line", "plate"], default="cell_line",
                    help="scope of the mean-over-drugs subtraction. A cell-line-scoped generic "
                         "leaves PLATE structure in every residual on that plate, which makes "
                         "unrelated same-plate conditions correlate and can fail the negative "
                         "control. Run both.")
    ap.add_argument("--project_generic", action="store_true",
                    help="remove the generic DIRECTION per condition (orthogonal rejection) instead of subtracting the mean vector. Kills potency leakage: the mean subtraction leaves (a_d - mean(a))*g(c) in every residual, which makes unrelated drugs correlate.")
    ap.add_argument("--min_neg_margin", type=float, default=0.15,
                    help="T must exceed the structure-matched negative control by at least this "
                         "much, or the verdict is VOID. Guards the failure mode where every arm "
                         "converges because a shared component dominates.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/variance_decomp.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.cache_dir:
        ap.error("--cache_dir required (or use --selftest)")

    rng = np.random.RandomState(args.seed)
    results = {"args": vars(args)}

    # the corrected build (disjoint control halves) is the headline
    kept = build_conditions(args.cache_dir, args.min_treated, args.min_control, args.seed,
                            split_controls=not args.shared_controls, loo_generic=args.loo_generic,
                            generic_scope=args.generic_scope,
                            project_generic=args.project_generic)
    results["main"] = report(kept, args, rng)

    # (1) quantify the shared-control bias by rebuilding the production way
    if not args.shared_controls:
        logger.info("")
        logger.info("CONTROL-SPLIT BIAS CHECK — rebuilding the PRODUCTION way (shared ctrl_pb)")
        kept_shared = build_conditions(args.cache_dir, args.min_treated, args.min_control, args.seed,
                                       split_controls=False, loo_generic=args.loo_generic,
                                       generic_scope=args.generic_scope,
                                       project_generic=args.project_generic)
        f = {k: v for k, v in kept_shared.items() if v["repro_cos"] > args.repro_thr}
        rows = transfer_pairs(f, same_drug=True, cross_line=True,
                              same_dose_only=args.same_dose_only, seed=args.seed)
        if rows:
            ci = clustered_ci([r["T"] for r in rows], [r["cluster"] for r in rows], rng, args.n_boot)
            m, lo, hi, n, _ = ci
            main_key = [k for k in results["main"] if k.startswith("T::repro-filtered")]
            base = results["main"][main_key[0]]["T"] if main_key else float("nan")
            results["shared_control_build"] = {"T": m, "ci": [lo, hi], "n_pairs": n,
                                               "bias_vs_split": float(base - m)}
            logger.info(f"  shared-control T = {m:.3f} [{lo:.3f}, {hi:.3f}]  vs  split-control "
                        f"T = {base:.3f}   (shared build biases T by {base - m:+.3f})")
            logger.info("     A NEGATIVE bias here means the production reliability is inflated by the "
                        "shared control, which would have made the residual look more "
                        "cell-line-specific than it is.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()

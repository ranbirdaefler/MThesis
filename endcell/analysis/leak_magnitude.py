#!/usr/bin/env python
r"""
leak_magnitude.py -- how much does the transductive holdout actually change the targets?
=============================================================================================
An external audit found that `build_residual_targets.build_residuals` is transductive: it computes
the generic shift as a mean over ALL drugs in a cell line, and applies the reliability filter
cos(r_A, r_B) > thr to ALL conditions, and only THEN assigns the train/unseen_combo/unseen_drug
split. Held-out outcomes therefore enter both the centring of the training targets and the selection
of which conditions exist at all.

That is a real defect and the clean fix is to rebuild train-only and retrain. Before spending a day
of compute on it, this measures how large the contamination is, because two questions have different
answers:

  (1) HOW MUCH DO THE TARGETS MOVE? Each held-out drug contributes 1/m of the generic, and cell lines
      carry 20-204 drugs, so the held-out set supplies roughly 10-15% of it. Whether that materially
      changes a training target is an empirical question, not an arithmetic one.

  (2) DOES IT MOVE THE HEADLINE? The generalisation claim is a `gap` -- model minus scramble -- and
      both arms are scored against the SAME target. A shared additive perturbation of the target
      cancels to first order in a difference of that form. It does not cancel in the SELECTION,
      which decides which conditions exist to be scored at all.

So this reports two different things and they should not be conflated: how far the residuals move,
and how much the retained set changes. A small movement plus a stable retained set is a defensible
"the contamination is measurable and immaterial"; anything else means retrain.

  python leak_magnitude.py --cache_dir /data/.../ot_cache \
      --holdout /data/.../residual_targets_holdout2/holdout.json \
      --out RESULTS/leak_magnitude.json
"""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "src"), os.path.dirname(_HERE)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, json, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def build_shifts(cache_dir, min_treated, min_control, seed):
    """Per-condition treated-minus-control shifts and their two disjoint halves.

    Mirrors build_residual_targets.build_residuals up to the point where the generic is formed; the
    generic itself is computed twice by the caller, which is the whole point of this script.
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

    ctrl_pb = {}
    for g in set(zip(cl[is_ctrl], plate[is_ctrl])):
        idx = np.where(is_ctrl & (cl == g[0]) & (plate == g[1]))[0]
        if len(idx) >= min_control:
            ctrl_pb[g] = L(list(idx))

    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], cl[i], plate[i], dose[i])].append(i)

    cond = {}
    for key, idxs in by.items():
        g = (key[1], key[2])
        if len(idxs) < min_treated or g not in ctrl_pb:
            continue
        idxs = list(idxs); rng.shuffle(idxs); h = len(idxs) // 2
        cond[key] = {"full": L(idxs) - ctrl_pb[g], "A": L(idxs[:h]) - ctrl_pb[g],
                     "B": L(idxs[h:]) - ctrl_pb[g]}
    logger.info(f"conditions with enough cells: {len(cond)}")
    return cond


def generic_over(cond, keys_allowed):
    """Generic shift per cell line, formed over a specified subset of conditions."""
    by_cl = defaultdict(list)
    for k in cond:
        if k in keys_allowed:
            by_cl[k[1]].append(k)
    out = {}
    for c, keys in by_cl.items():
        out[c] = {h: np.mean(np.stack([cond[k][h] for k in keys]), axis=0)
                  for h in ("full", "A", "B")}
    return out, {c: len(v) for c, v in by_cl.items()}


def residuals(cond, generic, thr):
    """Residuals and the reliability decision, exactly as the production builder computes them."""
    res, keep = {}, set()
    for k, v in cond.items():
        c = k[1]
        if c not in generic:
            continue
        rA = v["A"] - generic[c]["A"]
        rB = v["B"] - generic[c]["B"]
        cs = cos(rA, rB)
        res[k] = {"full": v["full"] - generic[c]["full"], "cs": cs}
        if cs > thr:
            keep.add(k)
    return res, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--holdout", required=True, help="holdout.json written by build_residual_targets")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/leak_magnitude.json")
    args = ap.parse_args()

    cond = build_shifts(args.cache_dir, args.min_treated, args.min_control, args.seed)

    ho = json.load(open(args.holdout))
    split_raw = ho["split"] if "split" in ho else ho
    split = {}
    for k, v in split_raw.items():
        # the manifest key is "|".join((drug, cell_line, plate, dose)); split from the RIGHT so a
        # drug name containing the delimiter cannot silently drop the condition
        parts = k.rsplit("|", 3)
        if len(parts) == 4:
            split[tuple(parts)] = v
    logger.info(f"holdout manifest: {len(split)} conditions | "
                + ", ".join(f"{v}={sum(1 for x in split.values() if x == v)}"
                            for v in sorted(set(split.values()))))

    present = set(cond)
    train_keys = {k for k in present if split.get(k, "train") == "train"}
    held_keys = present - train_keys
    logger.info(f"of {len(present)} conditions in the cache: {len(train_keys)} train, "
                f"{len(held_keys)} held out")

    gen_all, n_all = generic_over(cond, present)
    gen_tr, n_tr = generic_over(cond, train_keys)

    res_all, keep_all = residuals(cond, gen_all, args.repro_thr)
    res_tr, keep_tr = residuals(cond, gen_tr, args.repro_thr)

    # ---- (1) how far do the targets move? -------------------------------------------------------
    rows = {"train": [], "held": []}
    for k in present:
        if k not in res_all or k not in res_tr:
            continue
        a, b = res_all[k]["full"], res_tr[k]["full"]
        d = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9)
        rows["train" if k in train_keys else "held"].append((cos(a, b), d))

    out = {"n_conditions": len(present), "n_train": len(train_keys), "n_held": len(held_keys),
           "generic_n_drugs_all": n_all, "generic_n_drugs_train": n_tr}
    logger.info("")
    logger.info("=" * 78)
    logger.info("(1) HOW FAR DO THE TARGETS MOVE when the generic is built train-only?")
    for grp in ("train", "held"):
        if not rows[grp]:
            continue
        cs = np.array([x[0] for x in rows[grp]])
        dl = np.array([x[1] for x in rows[grp]])
        out[f"cos_{grp}"] = {"mean": float(cs.mean()), "p05": float(np.percentile(cs, 5)),
                             "min": float(cs.min())}
        out[f"reldelta_{grp}"] = {"mean": float(dl.mean()), "p95": float(np.percentile(dl, 95)),
                                  "max": float(dl.max())}
        logger.info(f"  {grp:<6} n={len(cs):<5} cos(r_all, r_trainonly): mean {cs.mean():.4f} "
                    f"p05 {np.percentile(cs, 5):.4f} min {cs.min():.4f} | "
                    f"||delta||/||r||: mean {dl.mean():.4f} p95 {np.percentile(dl, 95):.4f}")
    logger.info("  A mean cosine at 0.999 means the centring leak is cosmetic; below ~0.99 it is not.")

    # ---- (2) does the RETAINED SET change? ------------------------------------------------------
    both = keep_all & keep_tr
    only_all = keep_all - keep_tr
    only_tr = keep_tr - keep_all
    jac = len(both) / max(1, len(keep_all | keep_tr))
    out["retained"] = {"all": len(keep_all), "train_only_generic": len(keep_tr),
                       "both": len(both), "only_under_all": len(only_all),
                       "only_under_trainonly": len(only_tr), "jaccard": jac}
    logger.info("")
    logger.info("(2) DOES THE RETAINED SET CHANGE? (this is the half a gap metric cannot cancel)")
    logger.info(f"  kept under the shipped build      : {len(keep_all)}")
    logger.info(f"  kept under a train-only generic   : {len(keep_tr)}")
    logger.info(f"  agreement (Jaccard)               : {jac:.4f}   "
                f"({len(only_all)} lost, {len(only_tr)} gained)")

    # the selection question that actually bears on the generalisation claim
    for grp, keys in (("train", train_keys), ("held", held_keys)):
        ka = len(keys & keep_all)
        kt = len(keys & keep_tr)
        out[f"retained_{grp}"] = {"all": ka, "train_only": kt, "n": len(keys)}
        logger.info(f"  {grp:<6}: retained {ka}/{len(keys)} shipped vs {kt}/{len(keys)} train-only")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info("")
    logger.info("READ: (1) bounds how much the TRAINING SIGNAL was contaminated. (2) bounds how much")
    logger.info("      the EVALUATION SET was. The generalisation claim is a model-minus-scramble")
    logger.info("      difference scored against a shared target, so (1) largely cancels in it and")
    logger.info("      (2) does not. If (1) is ~1.000 and (2) is ~1.000, the contamination is")
    logger.info("      measurable and immaterial and can be reported as such. Otherwise: rebuild")
    logger.info("      train-only and retrain, and do not quote the current generalisation number.")
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

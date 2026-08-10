#!/usr/bin/env python
"""Exact paired comparison of two frozen-manifest NIR artifacts.

K generations are collapsed within each biological manifest row before inference.
The primary dependence model is crossed drug + cell-line clustering.  Well-based
analyses are named sensitivities and explicitly detect nesting rather than pretending
that a nested pair of labels is a crossed design.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from freeze_nir_manifest import atomic_write_json, sha256_json


def _tier(document: Mapping[str, Any], tier: str) -> Mapping[str, Any]:
    if document.get("status") == "validity_failure":
        raise ValueError("input is a validity-failure artifact; NIR comparison is undefined")
    try:
        payload = document["tiers"][tier]
    except KeyError as exc:
        raise ValueError(f"artifact lacks tier {tier}") from exc
    if payload.get("status") == "validity_failure":
        raise ValueError(f"tier {tier} failed validity; NIR comparison is undefined")
    return payload


def _nir(row: Mapping[str, Any], arm: str) -> float:
    aliases = ("scramble", "wrong_condition") if arm == "wrong_condition" else (arm,)
    for key in aliases:
        payload = row.get(key)
        value = payload.get("nir_expr") if isinstance(payload, Mapping) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    raise ValueError(f"row {row.get('row_id')} lacks scalar {arm}.nir_expr")


def _row_map(document: Mapping[str, Any], tier: str, *, require_baselines: bool) -> Dict[str, Mapping[str, Any]]:
    rows = _tier(document, tier).get("rows", [])
    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = row.get("row_id")
        if not row_id:
            raise ValueError("all rows must carry frozen-manifest row_id")
        if row_id in out:
            raise ValueError(f"duplicate row_id {row_id}")
        _nir(row, "model")
        if require_baselines:
            for arm in ("control", "linear", "mean", "wrong_condition"):
                _nir(row, arm)
        if int(row.get("k_collapsed", document.get("config", {}).get("k_samples", 0))) < 1:
            raise ValueError(f"row {row_id} does not document collapsed K generations")
        out[str(row_id)] = row
    return out


def _cluster_meat(centered: np.ndarray, labels: Sequence[str]) -> Tuple[float, int]:
    sums: Dict[str, float] = defaultdict(float)
    for value, label in zip(centered, labels):
        sums[str(label)] += float(value)
    groups = len(sums)
    if groups < 2:
        raise ValueError("cluster-robust interval requires at least two clusters per axis")
    return (groups / (groups - 1)) * sum(value * value for value in sums.values()), groups


def _ci_from_variance(y: np.ndarray, variance: float, *, method: str,
                      axes: Sequence[str], details: Mapping[str, Any], alpha: float) -> Dict[str, Any]:
    if variance < 0 or not math.isfinite(variance):
        raise ValueError(f"invalid declared variance {variance}")
    se = math.sqrt(variance)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    mean = float(y.mean())
    return {"mean": mean, "se": se, "ci": [mean - z * se, mean + z * se],
            "n": len(y), "axes": list(axes), "method": method, **dict(details)}


def one_way_mean_ci(values: Sequence[float], labels: Sequence[str], *, axis: str,
                    alpha: float = 0.05) -> Dict[str, Any]:
    y = np.asarray(values, dtype=float)
    if len(y) != len(labels) or len(y) < 2:
        raise ValueError("values and cluster labels must have the same length >= 2")
    meat, groups = _cluster_meat(y - y.mean(), [str(x) for x in labels])
    return _ci_from_variance(y, meat / (len(y) ** 2),
                             method="one-way cluster-robust mean",
                             axes=[axis], details={"n_clusters": [groups]}, alpha=alpha)


def cgm_mean_ci(values: Sequence[float], axis_a: Sequence[str], axis_b: Sequence[str],
                *, axis_names: Sequence[str] = ("axis_a", "axis_b"),
                alpha: float = 0.05) -> Dict[str, Any]:
    """CGM interval; negative finite-sample variance uses a declared conservative fallback."""
    y = np.asarray(values, dtype=float)
    if len(y) != len(axis_a) or len(y) != len(axis_b) or len(y) < 2:
        raise ValueError("values and cluster axes must have the same length >= 2")
    centered = y - y.mean()
    meat_a, ga = _cluster_meat(centered, [str(x) for x in axis_a])
    meat_b, gb = _cluster_meat(centered, [str(x) for x in axis_b])
    intersections = [f"{a}\x1f{b}" for a, b in zip(axis_a, axis_b)]
    meat_ab, gab = _cluster_meat(centered, intersections)
    n2 = len(y) ** 2
    variance_cgm = (meat_a + meat_b - meat_ab) / n2
    fallback = variance_cgm < 0
    variance = max(meat_a / n2, meat_b / n2) if fallback else variance_cgm
    if fallback and variance <= 0:
        raise ValueError("negative CGM variance and both one-way fallbacks are zero; inference withheld")
    return _ci_from_variance(
        y, variance, method="Cameron-Gelbach-Miller two-way cluster-robust mean",
        axes=axis_names,
        details={"n_clusters": [ga, gb], "n_intersection_clusters": gab,
                 "raw_cgm_variance": float(variance_cgm),
                 "negative_variance_policy": (
                     "max of the two one-way cluster variances" if fallback else "not invoked"),
                 "conservative_fallback_used": bool(fallback)}, alpha=alpha)


def _nesting(axis_a: Sequence[str], axis_b: Sequence[str]) -> str:
    a_to_b: Dict[str, set] = defaultdict(set)
    b_to_a: Dict[str, set] = defaultdict(set)
    for a, b in zip(axis_a, axis_b):
        a_to_b[str(a)].add(str(b))
        b_to_a[str(b)].add(str(a))
    a_in_b = all(len(values) == 1 for values in a_to_b.values())
    b_in_a = all(len(values) == 1 for values in b_to_a.values())
    if a_in_b and b_in_a:
        return "one_to_one"
    if a_in_b:
        return "axis_a_nested_in_axis_b"
    if b_in_a:
        return "axis_b_nested_in_axis_a"
    return "crossed"


def dependence_ci(values: Sequence[float], axis_a: Sequence[str], axis_b: Sequence[str],
                  name_a: str, name_b: str) -> Dict[str, Any]:
    relationship = _nesting(axis_a, axis_b)
    if relationship == "crossed":
        result = cgm_mean_ci(values, axis_a, axis_b, axis_names=[name_a, name_b])
    elif relationship == "axis_a_nested_in_axis_b":
        result = one_way_mean_ci(values, axis_b, axis=name_b)
    elif relationship == "axis_b_nested_in_axis_a":
        result = one_way_mean_ci(values, axis_a, axis=name_a)
    else:
        # Identical partitions: one-way clustering is the exact non-duplicated estimand.
        labels, name = (axis_a, name_a) if len(set(axis_a)) <= len(set(axis_b)) else (axis_b, name_b)
        result = one_way_mean_ci(values, labels, axis=name)
    result["cluster_relationship"] = relationship
    result["requested_axes"] = [name_a, name_b]
    return result


def _all_intervals(values: Sequence[float], paired: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    drugs = [row["drug"] for row in paired]
    lines = [row["cell_line"] for row in paired]
    wells = [row["well"] for row in paired]
    return {
        "primary_drug_cell_line": dependence_ci(values, drugs, lines, "drug", "cell_line"),
        "sensitivity_drug_well": dependence_ci(values, drugs, wells, "drug", "treatment_well"),
        "sensitivity_cell_line_well": dependence_ci(
            values, lines, wells, "cell_line", "treatment_well"),
    }


def _bh(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [1.0] * m
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), 1):
        rank = m - rank_from_end + 1
        running = min(running, p_values[idx] * m / rank)
        q[idx] = min(1.0, running)
    return q


def compare(left: Mapping[str, Any], right: Mapping[str, Any], *, tier: str,
            left_name: str, right_name: str, per_drug_bh: bool = False) -> Dict[str, Any]:
    for key in ("manifest_sha256", "metric_config_hash"):
        if not left.get(key) or not right.get(key):
            raise ValueError(f"both artifacts must carry {key}")
        if left[key] != right[key]:
            raise ValueError(f"artifact {key} mismatch")
    lm = _row_map(left, tier, require_baselines=True)
    rm = _row_map(right, tier, require_baselines=False)
    left_only, right_only = sorted(set(lm) - set(rm)), sorted(set(rm) - set(lm))
    if left_only or right_only:
        raise ValueError(f"support mismatch: {len(left_only)} left-only, {len(right_only)} right-only")

    paired = []
    for row_id in sorted(lm):
        a, b = lm[row_id], rm[row_id]
        immutable = ("tier", "drug", "cell_line", "plate", "treatment_well_ids")
        mismatched = [key for key in immutable if a.get(key) != b.get(key)]
        if mismatched:
            raise ValueError(f"row {row_id} metadata mismatch: {mismatched}")
        wells = a.get("treatment_well_ids") or []
        if len(wells) != 1:
            raise ValueError(f"row {row_id} must identify exactly one treatment well, got {wells}")
        gemma = _nir(a, "model")
        pythia = _nir(b, "model")
        paired.append({
            "row_id": row_id, "drug": a["drug"], "cell_line": a["cell_line"],
            "plate": a.get("plate"), "well": str(wells[0]),
            left_name: gemma, right_name: pythia,
            "control": _nir(a, "control"), "linear": _nir(a, "linear"),
            "mean": _nir(a, "mean"), "wrong_condition": _nir(a, "wrong_condition"),
        })
    if not paired:
        raise ValueError("no exactly paired rows")

    estimand_values = {
        f"{left_name}_minus_chance": [row[left_name] - 0.5 for row in paired],
        f"{left_name}_minus_control": [row[left_name] - row["control"] for row in paired],
        f"{left_name}_minus_linear": [row[left_name] - row["linear"] for row in paired],
        f"{left_name}_minus_mean": [row[left_name] - row["mean"] for row in paired],
        f"{left_name}_minus_wrong_condition": [
            row[left_name] - row["wrong_condition"] for row in paired],
        f"{left_name}_minus_{right_name}": [row[left_name] - row[right_name] for row in paired],
    }
    estimands = {name: _all_intervals(values, paired)
                 for name, values in estimand_values.items()}

    by_drug: Dict[str, List[float]] = defaultdict(list)
    backbone_key = f"{left_name}_minus_{right_name}"
    for row, value in zip(paired, estimand_values[backbone_key]):
        by_drug[row["drug"]].append(value)
    exploratory, pvals = [], []
    for drug in sorted(by_drug):
        arr = np.asarray(by_drug[drug], float)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else None
        p = 2 * (1 - NormalDist().cdf(abs(mean / se))) if se and se > 0 else 1.0
        exploratory.append({"drug": drug, "mean_difference": mean, "n_rows": len(arr),
                            "naive_p": p, "label": "exploratory; not cluster-robust"})
        pvals.append(p)
    if per_drug_bh:
        for row, q in zip(exploratory, _bh(pvals)):
            row["q_bh"] = q

    backbone_primary = estimands[backbone_key]["primary_drug_cell_line"]
    return {
        "schema_version": 2, "status": "ok", "tier": tier,
        "left": left_name, "right": right_name,
        "manifest_sha256": left["manifest_sha256"],
        "metric_config_hash": left["metric_config_hash"],
        "row_weighting": "equal", "k_handling": "collapsed within row before inference",
        "support": {"common": len(paired), "left_only": 0, "right_only": 0},
        "estimands": estimands,
        "primary_cgm_drug_cell_line": backbone_primary,
        "per_drug": exploratory, "rows": paired,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--left_name", default="gemma")
    ap.add_argument("--right_name", default="pythia")
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--per_drug_bh", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.left, encoding="utf-8") as handle:
        left = json.load(handle)
    with open(args.right, encoding="utf-8") as handle:
        right = json.load(handle)
    result = compare(left, right, tier=args.tier, left_name=args.left_name,
                     right_name=args.right_name, per_drug_bh=args.per_drug_bh)
    result["comparison_config_hash"] = sha256_json({
        "tier": args.tier, "left": args.left_name, "right": args.right_name,
        "per_drug_bh": args.per_drug_bh,
    })
    atomic_write_json(args.out, result)
    primary = result["primary_cgm_drug_cell_line"]
    print(f"wrote {args.out}: n={result['support']['common']}, gap={primary['mean']:+.4f}")


if __name__ == "__main__":
    main()

"""The release gate for the generalisation chapter.

The defect this exists to catch: `build_residual_targets` used to fit the generic shift and the
reliability filter over EVERY condition and assign the train/holdout split afterwards. Each held-out
target was therefore defined partly by the other held-out conditions, and the reported cross-context
transfer was measured against targets that had already seen the holdout.

The test is a poison test, which is the only version of this check that cannot be satisfied by
writing the split earlier and leaving the arithmetic wrong. Corrupt every held-out expression vector
with large noise, rebuild, and require that

  * the training JSONL is byte-identical, and
  * the digest of every fitted quantity is unchanged.

If any held-out cell reaches the fit, one of the two moves. The negative control -- poisoning a
TRAINING condition instead, which MUST change both -- is what stops the test passing vacuously.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_split_before_fit.py -q
"""
import hashlib
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "endcell", "ot"))

import build_residual_targets as brt          # noqa: E402
import tahoe_design as td                     # noqa: E402

N_GENES = 12
N_CELLS = 6            # per condition and per control group
DRUGS = [f"Drug{i}" for i in range(8)]
LINES = [f"CVCL_{i:04d}" for i in range(4)]
PLATES = ["plate6", "plate14"]


# --------------------------------------------------------------------------- fixture cache
def _write_cache(cache_dir, meta_dir, seed=0):
    """A miniature Tahoe cache with the shipped defect reproduced exactly.

    One treated WELL carries every cell line, and the well identifier is stored in a column named
    `dose` -- which is what the real caches did and why nothing downstream ever held a concentration.
    """
    rng = np.random.RandomState(seed)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    rows, mat = [], []
    # controls: one DMSO group per (cell line, plate)
    for cl in LINES:
        for p in PLATES:
            for _ in range(N_CELLS):
                rows.append({"is_control": True, "cell_line_id": cl, "plate": p,
                             "drug": "DMSO_TF", "dose": "smp_ctrl"})
                mat.append(rng.poisson(4, N_GENES))
    # treated: sample = (drug, plate, dose index); every cell line sits in the same well
    samples = {}
    sid = 0
    for d in DRUGS:
        for p in PLATES:
            sid += 1
            samples[f"smp_{sid}"] = (d, 0.05 if p == "plate6" else 0.5)
            for cl in LINES:
                base = rng.poisson(4, N_GENES) + rng.randint(0, 3, N_GENES)
                for _ in range(N_CELLS):
                    rows.append({"is_control": False, "cell_line_id": cl, "plate": p,
                                 "drug": d, "dose": f"smp_{sid}"})
                    mat.append(base + rng.poisson(1, N_GENES))

    pd.DataFrame(rows).to_parquet(os.path.join(cache_dir, "meta.parquet"))
    sparse.save_npz(os.path.join(cache_dir, "panel_expr.npz"),
                    sparse.csr_matrix(np.vstack(mat).astype(np.float32)))
    json.dump([f"GENE{i}" for i in range(N_GENES)],
              open(os.path.join(cache_dir, "panel_genes.json"), "w"))

    pd.DataFrame([{"Cell_ID_Cellosaur": cl, "cell_name": f"LINE{i}"}
                  for i, cl in enumerate(LINES)]).to_parquet(
        os.path.join(meta_dir, "cell_line_metadata.parquet"))
    pd.DataFrame([{"drug": d, "moa-fine": "kinase inhibitor"} for d in DRUGS] +
                 [{"drug": "DMSO_TF", "moa-fine": "vehicle"}]).to_parquet(
        os.path.join(meta_dir, "drug_metadata.parquet"))
    sm = [{"sample": s, "drugname_drugconc": f"[('{d}', {v}, 'uM')]"} for s, (d, v) in samples.items()]
    sm.append({"sample": "smp_ctrl", "drugname_drugconc": "[('DMSO_TF', 0.0, 'uM')]"})
    pd.DataFrame(sm).to_parquet(os.path.join(meta_dir, "sample_metadata.parquet"))
    return samples


def _args(cache_dir, meta_dir, out_dir, **over):
    a = types.SimpleNamespace(
        cache_dir=cache_dir, meta_dir=meta_dir, out_dir=out_dir, repo=brt.TAHOE_REPO,
        min_treated=N_CELLS, min_control=N_CELLS, repro_thr=-1.0, k_up=5, k_down=5, max_ctrl=3,
        generic_scope="plate", shrink_k=0.0, scope_sensitivity=False,
        keep_combinations=False, no_require_sample_id=False, no_eval_repro_filter=False,
        tier2_file=None, tier3_file=None, holdout_combos=0.2, min_combo_conditions=1,
        holdout_drugs=0.15, split_unit="condition", prompt_order="drug_first",
        emit_fit_digest=os.path.join(out_dir, "fit.sha"), seed=42)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _build(tmp_path, tag, poison=None, **over):
    """Build once. `poison` is a callable(meta, split) -> row indices to corrupt."""
    cache = os.path.join(tmp_path, f"cache_{tag}")
    meta_dir = os.path.join(tmp_path, "meta")
    out = os.path.join(tmp_path, f"out_{tag}")
    os.makedirs(out, exist_ok=True)
    _write_cache(cache, meta_dir)
    if poison is not None:
        meta = pd.read_parquet(os.path.join(cache, "meta.parquet"))
        X = sparse.load_npz(os.path.join(cache, "panel_expr.npz")).tolil()
        bad = poison(meta)
        assert len(bad) > 0, "the poison selected no rows, so the test would prove nothing"
        rs = np.random.RandomState(7)
        for i in bad:
            X[i, :] = rs.uniform(500, 1000, N_GENES)
        sparse.save_npz(os.path.join(cache, "panel_expr.npz"), X.tocsr())
    brt.run(_args(cache, meta_dir, out, **over))
    jsonl = open(os.path.join(out, "residual.jsonl"), "rb").read()
    return {
        "dir": out,
        "jsonl_sha": hashlib.sha256(jsonl).hexdigest(),
        "jsonl": jsonl,
        "fit_digest": open(os.path.join(out, "fit.sha")).read().strip(),
        "report": json.load(open(os.path.join(out, "report.json"))),
        "holdout": json.load(open(os.path.join(out, "holdout.json"))),
    }


def _rows_of(meta, keys):
    """Row indices belonging to a set of (drug, cell_line, plate, sample) condition keys."""
    want = set(keys)
    return [i for i, r in meta.iterrows()
            if not r["is_control"] and (r["drug"], r["cell_line_id"], r["plate"], r["dose"]) in want]


def _split_keys(res, which):
    return [tuple(k.split("|")) for k, v in res["holdout"]["split"].items() if v == which]


# --------------------------------------------------------------------------- baseline
@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return _build(str(tmp_path_factory.mktemp("base")), "base")


def test_the_split_is_actually_three_way(baseline):
    """A poison test over an empty holdout proves nothing."""
    h = baseline["report"]["holdout"]
    assert h["n_train"] > 0
    assert h["n_unseen_combo"] > 0
    assert h["n_unseen_drug"] > 0


def test_holdout_conditions_are_never_written_to_the_training_file(baseline):
    held = {(k[0], k[1]) for k in _split_keys(baseline, "unseen_combo")}
    held_drugs = set(baseline["holdout"]["holdout_drugs"])
    for line in baseline["jsonl"].decode().splitlines():
        m = json.loads(line)["metadata"]
        assert m["drug"] not in held_drugs
        assert (m["drug"], m["cell_line_id"]) not in held


def test_rebuild_is_deterministic(tmp_path):
    a = _build(str(tmp_path), "a")
    b = _build(str(tmp_path), "b")
    assert a["jsonl_sha"] == b["jsonl_sha"]
    assert a["fit_digest"] == b["fit_digest"]


# --------------------------------------------------------------------------- the gate
def test_poisoning_held_out_cells_changes_nothing_that_was_fitted(tmp_path, baseline):
    """THE GATE. Corrupt every held-out expression vector; the training file and the fit must not move."""
    def poison(meta):
        return _rows_of(meta, _split_keys(baseline, "unseen_combo") + _split_keys(baseline, "unseen_drug"))

    poisoned = _build(str(tmp_path), "poisoned", poison=poison)
    assert poisoned["fit_digest"] == baseline["fit_digest"], (
        "held-out expression reached the fitted generic -- the split is still being assigned after "
        "the fit, or the generic is not restricted to train keys")
    assert poisoned["jsonl_sha"] == baseline["jsonl_sha"], (
        "held-out expression changed a training target")


def test_poisoning_a_training_condition_does_change_things(tmp_path, baseline):
    """Negative control. Without this the gate above passes for a builder that reads nothing at all."""
    def poison(meta):
        return _rows_of(meta, _split_keys(baseline, "train")[:3])

    poisoned = _build(str(tmp_path), "trainpoison", poison=poison)
    assert poisoned["fit_digest"] != baseline["fit_digest"]
    assert poisoned["jsonl_sha"] != baseline["jsonl_sha"]


def test_held_out_drugs_are_absent_from_the_fitted_generic(tmp_path):
    """Direct inspection of the fit, independent of the poison."""
    cache = os.path.join(str(tmp_path), "cache_g")
    meta_dir = os.path.join(str(tmp_path), "meta_g")
    _write_cache(cache, meta_dir)
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)
    conds, ctrl_rows, X, _, _ = brt.inventory(cache, N_CELLS, N_CELLS, conc)
    split, ho_drugs, _ = brt.make_holdout(conds, 0.2, 0.15, 42)
    shifts, _ = brt.compute_shifts(conds, ctrl_rows, X, 42)
    gen = brt.Generic(shifts, conds, [k for k in conds if split[k] == "train"])
    assert ho_drugs, "no drug was held out, so this asserts nothing"
    for store in (gen.fine, gen.coarse):
        for g, s in store.items():
            assert not (set(s["drug_mean"]) & set(ho_drugs)), f"held-out drug in the generic for {g}"


# --------------------------------------------------------------------------- the other two defects
def test_generic_is_plate_scoped_by_default_and_scope_is_recorded(baseline):
    assert baseline["report"]["scope"] == "plate"
    assert baseline["report"]["drug_weighted_generic"] is True
    assert baseline["report"]["leave_one_drug_out"] is True
    assert baseline["report"]["split_before_fit"] is True


def test_cell_line_scope_and_plate_scope_give_different_targets(tmp_path):
    a = _build(str(tmp_path), "sc_plate", generic_scope="plate")
    b = _build(str(tmp_path), "sc_line", generic_scope="cell_line")
    assert a["jsonl_sha"] != b["jsonl_sha"], "the scope flag is not reaching the target construction"


def test_dose_weighting_does_not_let_one_drug_count_twice():
    """Two doses of drug X and one of drug Y: the generic must be (X + Y)/2, not (X + X + Y)/3."""
    conds, shifts = {}, {}
    for i, (d, v) in enumerate([("X", 6.0), ("X", 6.0), ("Y", 0.0), ("Z", 0.0)]):
        k = (d, "c", "p", f"smp_{i}")
        conds[k] = {"cell_line": "c", "plate": "p", "drug": d, "group": ("c", "p"), "n_cells": 9}
        shifts[k] = {h: np.full(3, v, np.float32) for h in ("full", "A", "B")}
    g = brt.Generic(shifts, conds, list(conds))
    # leave Z out: mean over {X, Y} = 3.0.  Condition-weighted it would be (6+6+0)/3 = 4.0.
    assert g.value("c", "p", "Z", "full", "plate")[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------- the unit and the dose
def test_emitted_metadata_carries_the_sample_not_a_fake_dose(baseline):
    doses, samples = [], []
    for line in baseline["jsonl"].decode().splitlines():
        m = json.loads(line)["metadata"]
        assert "dose_float" not in m, "the mislabelled field is still being written"
        assert m["sample_id"].startswith("smp_")
        samples.append(m["sample_id"])
        doses.append(m["dose_molar"])
    assert not td.looks_like_sample_id(doses)
    assert set(doses) <= {5e-8, 5e-7}, f"doses did not resolve to molar: {sorted(set(doses))[:5]}"
    assert len(set(samples)) > 1


def test_the_sample_identifier_is_recovered_from_the_mislabelled_column(baseline):
    assert baseline["report"]["inventory"]["sample_id_source"].startswith("recovered")


def test_one_well_holds_many_cell_lines(baseline):
    assert baseline["report"]["inventory"]["cell_lines_per_sample_max"] == len(LINES)


def test_well_crossing_is_counted_rather_than_assumed(baseline):
    """A (drug, cell_line) holdout cannot avoid held-out conditions sharing a well with training
    ones. That is a limitation to report, not to hide, so the number must be in the manifest."""
    wc = baseline["holdout"]["well_crossing"]
    assert wc["n_samples_crossing_split"] > 0
    assert 0.0 < wc["frac_heldout_conditions_sharing_a_well_with_train"] <= 1.0


def test_sample_split_unit_removes_well_crossing(tmp_path):
    r = _build(str(tmp_path), "bysample", split_unit="sample")
    wc = r["holdout"]["well_crossing"]
    assert wc["n_samples_crossing_split"] == 0
    assert wc["n_heldout_conditions_sharing_a_well_with_train"] == 0


def test_combination_samples_are_dropped_not_truncated(tmp_path):
    cache = os.path.join(str(tmp_path), "cache_c")
    meta_dir = os.path.join(str(tmp_path), "meta_c")
    _write_cache(cache, meta_dir)
    sm = pd.read_parquet(os.path.join(meta_dir, "sample_metadata.parquet"))
    sm.loc[sm["sample"] == "smp_1", "drugname_drugconc"] = "[('Drug0', 0.05, 'uM'), ('Drug1', 1.0, 'uM')]"
    sm.to_parquet(os.path.join(meta_dir, "sample_metadata.parquet"))
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)

    dropped, _, _, _, notes = brt.inventory(cache, N_CELLS, N_CELLS, conc, drop_combinations=True)
    kept, _, _, _, notes2 = brt.inventory(cache, N_CELLS, N_CELLS, conc, drop_combinations=False)
    assert notes["n_dropped_combination"] == len(LINES)
    assert notes2["n_dropped_combination"] == 0
    assert not any(k[3] == "smp_1" for k in dropped)
    assert any(k[3] == "smp_1" for k in kept)
    # and when kept, it is flagged rather than silently reduced to its first component
    assert all(v["is_combination"] for k, v in kept.items() if k[3] == "smp_1")


def test_a_cache_with_no_recoverable_treatment_unit_is_refused(tmp_path):
    cache = os.path.join(str(tmp_path), "cache_n")
    meta_dir = os.path.join(str(tmp_path), "meta_n")
    _write_cache(cache, meta_dir)
    meta = pd.read_parquet(os.path.join(cache, "meta.parquet"))
    meta["dose"] = "0.05 uM"                      # a plausible-looking dose column, no sample anywhere
    meta.to_parquet(os.path.join(cache, "meta.parquet"))
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)
    with pytest.raises(SystemExit):
        brt.inventory(cache, N_CELLS, N_CELLS, conc)
    # explicit opt-out still works, and the report says the unit is unavailable
    _, _, _, _, notes = brt.inventory(cache, N_CELLS, N_CELLS, conc, require_sample_id=False)
    assert notes["sample_id_source"] == "unavailable"

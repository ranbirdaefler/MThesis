import inspect
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

from collections import defaultdict

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
        shared_control_reliability=False,
        keep_combinations=False, no_require_sample_id=False, eval_repro_filter=False,
        min_plate_drugs=0, val_frac=0.0,
        min_train_frac=0.05,
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
    shifts, _, _ = brt.compute_shifts(conds, ctrl_rows, X, 42)
    gen = brt.Generic(shifts, conds, [k for k in conds if split[k] == "train"])
    assert ho_drugs, "no drug was held out, so this asserts nothing"
    for store in (gen.fine, gen.coarse):
        for g, s in store.items():
            assert not (set(s["drug_mean"]) & set(ho_drugs)), f"held-out drug in the generic for {g}"


CONSUMERS = ["endcell/analysis/residual_eval.py", "endcell/analysis/channel_gate.py",
             "endcell/analysis/reconstructed_eval.py", "endcell/analysis/reward_calibration.py",
             "endcell/train/build_de_weights.py"]


def test_the_five_evaluation_consumers_still_have_their_api(tmp_path):
    """Restructuring run() into stages deleted `build_residuals`, and five scripts call it.

    Nothing caught that: the poison test exercises run(), and none of the five has a test. They are
    the entire evaluation stack, so the breakage would have surfaced on the cluster, after the queue
    wait, at the eval step of a job whose training had already finished.
    """
    missing = [f for f in CONSUMERS
               if "brt.build_residuals(" in open(os.path.join(ROOT, f), encoding="utf-8").read()
               and not hasattr(brt, "build_residuals")]
    assert not missing, f"these scripts call an API that no longer exists: {missing}"

    cache = os.path.join(str(tmp_path), "cache_shim")
    meta_dir = os.path.join(str(tmp_path), "meta_shim")
    _write_cache(cache, meta_dir)
    # the exact positional signature all five use
    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        cache, N_CELLS, N_CELLS, -1.0, seed=42, meta_dir=meta_dir)
    assert kept and len(next(iter(kept))) == 4, "condition keys must stay 4-tuples"
    for field in ("residual", "residual_A", "residual_B", "group", "n_cells", "repro_cos"):
        assert field in next(iter(kept.values())), f"consumers read kept[k]['{field}']"
    assert generic and all(isinstance(k, str) for k in generic), (
        "consumers index `generic` by cell line, so plate-scoped groups must be collapsed")
    assert len(ctrl_rows) > 0 and X.shape[0] == len(meta)


def test_the_shim_reproduces_the_published_semantics_by_default(tmp_path):
    """The published numbers came from the cell-line-scoped, non-LOO, shared-control build. The shim
    must default to exactly that, or previously quoted results become unreproducible."""
    cache = os.path.join(str(tmp_path), "cache_sem")
    meta_dir = os.path.join(str(tmp_path), "meta_sem")
    _write_cache(cache, meta_dir)
    old, _, _, _, _ = brt.build_residuals(cache, N_CELLS, N_CELLS, -1.0, seed=42, meta_dir=meta_dir)
    new, _, _, _, _ = brt.build_residuals(cache, N_CELLS, N_CELLS, -1.0, seed=42, meta_dir=meta_dir,
                                          generic_scope="plate", loo=True, split_controls=True)
    k = next(iter(old))
    assert not np.allclose(old[k]["residual"], new[k]["residual"]), (
        "the repaired frame must give different targets, else no flag is reaching the construction")
    assert old[k]["repro_cos"] != new[k]["repro_cos"]


def test_reliability_halves_do_not_share_a_control(tmp_path):
    """`repro_cos` is meant to be the correlation of two independent measurements of one condition.

    Subtracting the same control pseudobulk from both halves makes them share the whole control-noise
    term, which correlates them for a reason unrelated to the drug. FINDINGS.md:532 retracts a scope
    comparison made that way -- and the inflation is not neutral between scopes, so reproducing it
    here would bias the very sensitivity table that decides the rebuild.
    """
    cache = os.path.join(str(tmp_path), "cache_rel")
    meta_dir = os.path.join(str(tmp_path), "meta_rel")
    _write_cache(cache, meta_dir)
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)
    conds, ctrl_rows, X, _, _ = brt.inventory(cache, N_CELLS, N_CELLS, conc)

    split_shifts, _, prov = brt.compute_shifts(conds, ctrl_rows, X, 42)
    shared_shifts, _, prov_shared = brt.compute_shifts(conds, ctrl_rows, X, 42, shared_control=True)

    assert prov["shared_control_reliability"] is False
    assert prov_shared["shared_control_reliability"] is True
    assert prov["n_conditions_with_shared_control"] == 0, "every group here is large enough to halve"

    # the full shift is the training target and must be unaffected by how reliability is measured
    for k in conds:
        assert np.allclose(split_shifts[k]["full"], shared_shifts[k]["full"])

    # the halves must differ, i.e. the split control actually reached them
    assert any(not np.allclose(split_shifts[k]["A"], shared_shifts[k]["A"]) for k in conds)

    # and the shared-control version must report higher reliability on this noise-only fixture,
    # which is the inflation the retraction is about
    def mean_cos(sh):
        cs = []
        for k in conds:
            a, b = sh[k]["A"], sh[k]["B"]
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        return float(np.mean(cs))

    assert mean_cos(shared_shifts) > mean_cos(split_shifts), (
        "sharing a control across both halves should inflate repro_cos; if it does not, the fixture "
        "is not exercising the control-noise term and this test proves nothing")


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
    g = brt.Generic(shifts, conds, list(conds), min_plate_drugs=0)
    # leave Z out: mean over {X, Y} = 3.0.  Condition-weighted it would be (6+6+0)/3 = 4.0.
    assert g.value("c", "p", "Z", "full", "plate")[0] == pytest.approx(3.0)


def test_plate_scope_fails_closed_instead_of_falling_back_to_cell_line(tmp_path):
    """`--generic_scope plate --shrink_k 0` used to fall back to the cell-line generic whenever a
    plate group had no other training drug. That silently put an unreported subset of conditions in
    the contaminated frame the rebuild exists to leave."""
    conds, shifts = {}, {}
    for i, (d, p, v) in enumerate([("X", "p1", 6.0), ("Y", "p1", 0.0), ("Z", "p1", 3.0),
                                   ("W", "p2", 9.0)]):          # p2 holds one drug only
        k = (d, "c", p, f"smp_{i}")
        conds[k] = {"cell_line": "c", "plate": p, "drug": d, "group": ("c", p), "n_cells": 9}
        shifts[k] = {h: np.full(3, v, np.float32) for h in ("full", "A", "B")}

    strict = brt.Generic(shifts, conds, list(conds), shrink_k=0.0, min_plate_drugs=2)
    assert strict.value("c", "p2", "W", "full", "plate") is None, (
        "a plate with no other training drug must yield None, not the cell-line generic")
    assert strict.value("c", "p1", "X", "full", "plate") is not None, "p1 has two other drugs"
    assert strict.frame_name("plate") == "plate"

    # with shrinkage the blend toward cell-line scope is DECLARED, so a thin group is legitimate --
    # but the frame must then be named for what it is
    shrunk = brt.Generic(shifts, conds, list(conds), shrink_k=5.0, min_plate_drugs=2)
    assert shrunk.value("c", "p2", "W", "full", "plate") is not None
    assert "hierarchical" in shrunk.frame_name("plate")


def test_export_uses_the_same_frame_as_the_targets(tmp_path):
    """reconstruction.npz must invert the transform that produced the target. Export ignored
    shrinkage, so a shrunk build reconstructed with an unshrunk generic."""
    conds, shifts = {}, {}
    for i, (d, p, v) in enumerate([("X", "p1", 6.0), ("Y", "p1", 0.0), ("Z", "p2", 12.0)]):
        k = (d, "c", p, f"smp_{i}")
        conds[k] = {"cell_line": "c", "plate": p, "drug": d, "group": ("c", p), "n_cells": 9}
        shifts[k] = {h: np.full(3, v, np.float32) for h in ("full", "A", "B")}
    plain = brt.Generic(shifts, conds, list(conds), shrink_k=0.0, min_plate_drugs=0)
    shrunk = brt.Generic(shifts, conds, list(conds), shrink_k=5.0, min_plate_drugs=0)
    ep, es = plain.export(conds, "plate"), shrunk.export(conds, "plate")
    assert not np.allclose(ep[("c", "p2")], es[("c", "p2")]), (
        "shrinkage must reach the exported generic, or reconstruction uses a different definition "
        "of the quantity than the targets did")


def test_the_held_out_set_is_not_selected_on_its_own_outcome(tmp_path):
    """Dropping held-out conditions for failing their own reliability threshold selects the
    evaluation set on the outcome. Primary result keeps them; the filter is a labelled sensitivity."""
    held = lambda r: r["report"]["holdout"]["n_unseen_combo"] + r["report"]["holdout"]["n_unseen_drug"]

    # the guarantee: with the filter off, the held-out set is whatever the SPLIT assigned, and moving
    # the reliability threshold cannot change it
    loose = _build(str(tmp_path), "loose", repro_thr=-1.0)
    tight = _build(str(tmp_path), "tight", repro_thr=0.93)
    assert loose["report"]["eval_repro_filter"] is False, "unfiltered must be the default"
    assert held(loose) == held(tight) > 0, (
        "the held-out set moved when the reliability threshold moved, so it is still being selected "
        "on its own outcome")
    # and the training set DOES respond, which is what makes the check above meaningful
    assert tight["report"]["holdout"]["n_train"] < loose["report"]["holdout"]["n_train"]

    # opting in must actually filter, or the sensitivity arm is unavailable
    on = _build(str(tmp_path), "filtered", repro_thr=0.93, eval_repro_filter=True)
    assert held(on) < held(tight)


def test_sample_split_unit_is_honoured_even_when_tier_files_supply_the_split(tmp_path):
    """`--split_unit sample` was only honoured on the random-split path, so which estimand the
    thesis claims depended on which holdout source happened to fire."""
    cache = os.path.join(str(tmp_path), "cache_es")
    meta_dir = os.path.join(str(tmp_path), "meta_es")
    _write_cache(cache, meta_dir)
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)
    conds, _, _, _, _ = brt.inventory(cache, N_CELLS, N_CELLS, conc)

    # a split that straddles a well: one cell line held out, the rest training
    split = {k: "train" for k in conds}
    victim = next(k for k in conds)
    split[victim] = "unseen_combo"
    assert brt.sample_crossing_report(conds, split)["n_samples_crossing_split"] == 1

    promoted = brt.enforce_sample_split(conds, split)
    assert brt.sample_crossing_report(conds, promoted)["n_samples_crossing_split"] == 0
    # promotion never moves a condition INTO training
    assert all(not (split[k] != "train" and promoted[k] == "train") for k in split)


def test_promoting_a_tier_split_to_wells_would_collapse_training_and_is_refused(tmp_path):
    """A tier split holds out (drug, cell_line) PAIRS. A well spans many cell lines, so promoting
    that split to the well level holds out every well containing any held-out pair -- 1 - 0.85^N for
    N lines per well, i.e. essentially everything. The builder must draw at the well level instead,
    and must refuse outright rather than emit a near-empty training set."""
    cache = os.path.join(str(tmp_path), "cache_col")
    meta_dir = os.path.join(str(tmp_path), "meta_col")
    _write_cache(cache, meta_dir)
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)
    conds, _, _, _, _ = brt.inventory(cache, N_CELLS, N_CELLS, conc)

    # a condition-level holdout of a modest fraction of pairs...
    cond_split, _, _ = brt.make_holdout(conds, 0.2, 0.0, 42)
    frac_before = sum(1 for v in cond_split.values() if v == "train") / len(cond_split)
    # ...becomes a near-total holdout once promoted to the well
    promoted = brt.enforce_sample_split(conds, cond_split)
    frac_after = sum(1 for v in promoted.values() if v == "train") / len(promoted)
    assert frac_after < frac_before, "promotion must cost training data, or this proves nothing"
    assert frac_after < 0.5, (
        f"expected promotion to collapse the training set on a fixture where one well spans "
        f"{len(LINES)} cell lines; got {frac_after:.0%} still training")

    # drawing at the well level keeps a workable split
    by_well, _, _ = brt.make_holdout_by_sample(conds, 0.2, 0.0, 42)
    frac_well = sum(1 for v in by_well.values() if v == "train") / len(by_well)
    assert frac_well > 0.5, f"well-level draw should retain most conditions; got {frac_well:.0%}"

    # and the guard fires rather than shipping a collapsed build
    with pytest.raises(SystemExit, match="REFUSING TO BUILD"):
        _build(str(tmp_path), "collapse", split_unit="sample", min_train_frac=0.99)


def test_sample_holdout_never_takes_a_drugs_last_training_well(tmp_path):
    """A drug present in two wells could lose both, which silently converts it to an unseen DRUG
    while it sits in the unseen_combo arm -- the two arms then measure the same thing."""
    cache = os.path.join(str(tmp_path), "cache_last")
    meta_dir = os.path.join(str(tmp_path), "meta_last")
    _write_cache(cache, meta_dir)
    _, _, conc = brt.load_meta_maps(meta_dir=meta_dir)
    conds, _, _, _, _ = brt.inventory(cache, N_CELLS, N_CELLS, conc)

    # every drug here has exactly two wells, so an unguarded draw at a high fraction would strip some
    split, ho_drugs, _ = brt.make_holdout_by_sample(conds, 0.9, 0.0, 7)
    train_wells = defaultdict(set)
    for k, v in split.items():
        if v == "train":
            train_wells[k[0]].add(k[3])
    for d in {k[0] for k in conds} - set(ho_drugs):
        assert train_wells[d], f"drug {d} lost every training well and is now effectively unseen"


def test_auto_threshold_resolves_before_anything_reads_it(tmp_path):
    """`--repro_thr auto` is a string until it is resolved against the run's null. Anything that
    compares against it beforehand raises.

    That is not hypothetical: the first version resolved it AFTER `scope_sensitivity`, which does
    `cos > repro_thr`, so a cluster job with both flags died in stage 3. The local check missed it
    because the fixture defaults `scope_sensitivity=False` -- so this test sets BOTH.
    """
    r = _build(str(tmp_path), "autothr", repro_thr="auto", scope_sensitivity=True)
    rep = r["report"]
    assert isinstance(rep["repro_thr"], float), "the resolved threshold must be numeric in the report"
    cal = rep["reliability_calibration"]
    assert rep["repro_thr"] == pytest.approx(cal["null_p95"]), "auto must resolve to the null's 95th"
    assert cal["null_mean"] == pytest.approx(0.0, abs=0.15), (
        "the null pairs unrelated conditions and must sit near zero, or it is not a null")
    assert rep["scope_sensitivity"], "the sensitivity table must still be produced"
    for row in rep["scope_sensitivity"]:
        assert 0.0 <= row["retention"] <= 1.0

    # auto99 is stricter than auto95, and both are numeric
    r99 = _build(str(tmp_path), "autothr99", repro_thr="auto99", scope_sensitivity=True)
    assert r99["report"]["repro_thr"] > rep["repro_thr"]


def test_a_residual_format_validation_shard_is_written(tmp_path):
    """The retrain validated residual-format training against ordinary cell-sentence tier-1 data, so
    the validation loss measured a different output distribution than the one being trained."""
    r = _build(str(tmp_path), "valshard", val_frac=0.25)
    val = open(os.path.join(r["dir"], "residual_val.jsonl"), "rb").read().decode().splitlines()
    assert val, "no validation shard was written"
    train_samples = {json.loads(l)["metadata"]["sample_id"] for l in r["jsonl"].decode().splitlines()}
    val_samples = {json.loads(l)["metadata"]["sample_id"] for l in val}
    assert not (train_samples & val_samples), "a validation well also appears in training"
    for l in val:
        ex = json.loads(l)
        assert ex["metadata"]["target"] == "residual"
        assert brt.DOWN in ex["response"] or brt.END in ex["response"], (
            "the shard must be in the residual signed-DE format, not plain cell sentences")


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


# --- the by-split comparator must default to the NEUTRAL stratum ---------------------------------
# scramble_opposite is not a null: measured NIR 0.403 at partner cosine -0.24 against 0.497 at 0.00,
# so a gap against it mixes "the model used the drug" with "the comparator was pushed the other way".
# The by-split table hard-coded it, which is exactly where the transfer claim is read off.

def test_split_table_comparator_defaults_to_the_neutral_stratum():
    import argparse
    import endcell.analysis.residual_eval as re_
    src = inspect.getsource(re_)
    ap = [ln for ln in src.splitlines() if "--split_comparator" in ln]
    assert ap, "the by-split comparator is not configurable"

    parser = None
    for name in dir(re_):
        fn = getattr(re_, name)
        if callable(fn) and name in ("build_parser", "_parser", "make_parser"):
            parser = fn()
            break
    if parser is None:                      # parser is built inline in main(); read the default off source
        i = src.index("--split_comparator")
        window = src[i:i + 400]
        assert 'default="scramble_orth"' in window, "the default comparator is not the neutral stratum"
    else:
        assert parser.get_default("split_comparator") == "scramble_orth"
        assert isinstance(parser, argparse.ArgumentParser)

    # and the hard-coded call site is gone from the by-split block
    blk = src[src.index("GENERALIZATION"):]
    blk = blk[:blk.index("means[") if "means[" in blk else len(blk)]
    assert '_clustered_ci(sub, "model", "scramble_opposite"' not in blk, \
        "the by-split table still hard-codes scramble_opposite"

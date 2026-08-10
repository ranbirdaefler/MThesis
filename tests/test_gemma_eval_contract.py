"""Dependency-light contracts for the hardened Gemma/Pythia evaluation path."""
import copy
import inspect
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "endcell", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "endcell", "eval"))

import compare_backbones as cb  # noqa: E402
import evaluate_endcell as ee  # noqa: E402
import freeze_nir_manifest as fm  # noqa: E402
import nir_benchmark as nir  # noqa: E402
import residual_eval as residual  # noqa: E402


def test_expression_nir_is_tie_aware():
    assert nir.nir_from_sims(1.0, [0.0, 1.0, 2.0]) == pytest.approx(0.5)
    assert nir.nir_from_dists(1.0, [2.0, 1.0, 0.0]) == pytest.approx(0.5)
    assert nir.nir_from_sims(0.0, [0.0, 0.0]) == pytest.approx(0.5)
    assert nir.nir_from_dists(0.0, [0.0, 0.0]) == pytest.approx(0.5)


def test_legacy_generation_remains_default():
    assert ee.DEFAULT_GENERATION_CONTRACT == "legacy"
    assert residual.DEFAULT_GENERATION_CONTRACT == "legacy"


def test_inference_contract_restores_kv_cache_and_persists_attention_backend():
    for module in (nir, ee, residual):
        source = inspect.getsource(module)
        assert "attention_implementation" in source
        assert "model.config.use_cache = True" in source
        assert '"use_cache": True' in source


class FakeTokenizer:
    unk_token_id = 3
    unk_token = "<unk>"

    def __init__(self, ids):
        self.ids = ids

    def encode(self, token, add_special_tokens=False):
        return list(self.ids)

    def convert_ids_to_tokens(self, token_id):
        return "[END_CELL]" if token_id != 3 else "<unk>"


@pytest.mark.parametrize("ids", ([4, 5], [3]))
def test_atomic_end_cell_rejects_split_and_unknown(ids):
    with pytest.raises(ValueError):
        nir.atomic_sentinel_id(FakeTokenizer(ids))
    with pytest.raises(ValueError):
        ee.atomic_sentinel_id(FakeTokenizer(ids))
    assert nir.atomic_sentinel_id(FakeTokenizer([9])) == 9
    assert ee.atomic_sentinel_id(FakeTokenizer([9])) == 9


def test_seed_depends_only_on_frozen_identifiers():
    a = fm.stable_output_seed("model", "row", "prompt", 2, "v1")
    assert a == fm.stable_output_seed("model", "row", "prompt", 2, "v1")
    assert a != fm.stable_output_seed("model", "row", "prompt", 3, "v1")
    assert a != fm.stable_output_seed("other", "row", "prompt", 2, "v1")


def _rows(scramble=False, drugs=("DrugA", "DrugB", "DrugC"), with_well=True):
    rows = []
    for di, drug in enumerate(drugs):
        for cell in range(4):
            moa = f"M{di}"
            prompt_drug = drugs[(di + 1) % len(drugs)] if scramble else drug
            prompt_moa = f"M{(di + 1) % len(drugs)}" if scramble else moa
            prompt = (f"Predict the response of Line to {prompt_drug} at 1 uM. "
                      f"Mechanism: {prompt_moa}.\nControl cell: G0 G1 [END_CELL]\n\nResponse cell:")
            metadata = {"drug": drug, "cell_line_id": "CL1", "plate": "plate4",
                        "dose": "1 uM", "moa": moa}
            if with_well:
                metadata["sample_id"] = f"well-{di}"
            if scramble:
                metadata.update(scrambled_from_drug=drug, scrambled_to_drug=prompt_drug)
            rows.append({"prompt": prompt, "response": f"G{di} G9 [END_CELL]",
                         "metadata": metadata})
    return rows


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _manifest_fixture(tmp_path):
    evaluation = tmp_path / "eval.jsonl"
    scramble = tmp_path / "scramble.jsonl"
    train = tmp_path / "train.jsonl"
    _write_jsonl(evaluation, _rows())
    _write_jsonl(scramble, _rows(scramble=True))
    _write_jsonl(train, _rows(drugs=("TrainA", "TrainB", "TrainC")))
    return evaluation, scramble, train


def test_manifest_proves_disjointness_and_freezes_all_provenance(tmp_path):
    evaluation, scramble, train = _manifest_fixture(tmp_path)
    manifest = fm.build_manifest(
        str(evaluation), train_file=str(train), scramble_file=str(scramble), k_samples=3,
        min_cells=4, min_drugs_per_group=3, max_groups=1, same_plate_only=True,
        model_fingerprints=["pythia", "gemma"],
        model_declarations=[{"model_id": "parent", "fingerprint": "parent-fp",
                             "role": "validity_only_parent"}],
        expected_support={"rows": 3, "drugs": 3, "cell_lines": 1, "groups": 1})
    assert manifest["support"] == {"rows": 3, "drugs": 3, "cell_lines": 1, "groups": 1}
    assert manifest["tier2_disjointness"]["evaluation_drugs_in_training"] == []
    assert manifest["sources"]["training"]["sha256"] == fm.file_sha256(str(train))
    assert manifest["creator"]["sha256"]
    assert any(x["role"] == "validity_only_parent" for x in manifest["model_declarations"])
    for row in manifest["rows"]:
        assert len(row["truth_half_a"]) == 2 and len(row["truth_half_b"]) == 2
        assert len(row["treatment_well_ids"]) == 1
        assert len(row["prompt_specs"]) == 3
        for spec in row["prompt_specs"]:
            assert set(spec["seeds_by_model"]) == {"pythia", "gemma"}
            assert spec["wrong_condition"]["non_condition_bytes_verified"] is True
    fm.verify_manifest(manifest, str(evaluation), str(scramble), str(train))
    with open(evaluation, "a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        fm.verify_manifest(manifest, str(evaluation), str(scramble), str(train))


def test_manifest_rejects_tier2_leakage_missing_well_and_support_drift(tmp_path):
    evaluation, scramble, train = _manifest_fixture(tmp_path)
    _write_jsonl(train, _rows())
    with pytest.raises(ValueError, match="Tier-2 leakage"):
        fm.build_manifest(str(evaluation), train_file=str(train), scramble_file=str(scramble),
                          min_cells=4, min_drugs_per_group=3, max_groups=1)
    _write_jsonl(train, _rows(drugs=("TrainA", "TrainB", "TrainC")))
    _write_jsonl(evaluation, _rows(with_well=False))
    with pytest.raises(ValueError, match="genuine sample/well"):
        fm.build_manifest(str(evaluation), train_file=str(train), min_cells=4,
                          min_drugs_per_group=3, max_groups=1)
    _write_jsonl(evaluation, _rows())
    with pytest.raises(ValueError, match="preregistration"):
        fm.build_manifest(str(evaluation), train_file=str(train), min_cells=4,
                          min_drugs_per_group=3, max_groups=1,
                          expected_support={"rows": 606})


def test_manifest_rejects_training_row_without_drug_with_line_number(tmp_path):
    evaluation, scramble, train = _manifest_fixture(tmp_path)
    training_rows = _rows(drugs=("TrainA", "TrainB", "TrainC"))
    training_rows[1]["metadata"].pop("drug")
    _write_jsonl(train, training_rows)

    with pytest.raises(ValueError, match=r"lacks nonempty metadata\.drug.*train\.jsonl:2$"):
        fm.build_manifest(str(evaluation), train_file=str(train), scramble_file=str(scramble),
                          min_cells=4, min_drugs_per_group=3, max_groups=1)


def _cache_record(output_id="x", text="G1 [END_CELL]"):
    return {"type": "prediction", "output_id": output_id, "row_id": "row",
            "prompt_id": "prompt", "prompt_sha256": "abc", "draw_id": 0, "seed": 42,
            "raw_token_ids": [7, 8], "decoded_text": text,
            "termination_reason": "end_cell", "validity": {"complete": True}}


def test_prediction_cache_is_locked_schema_checked_and_fail_closed(tmp_path):
    path = tmp_path / "predictions.jsonl"
    cache = fm.PredictionCache(str(path), {"model": "m", "version": 1})
    row = _cache_record()
    cache.put(row)
    cache.put(row)
    with pytest.raises(ValueError, match="overwrite"):
        cache.put(_cache_record(text="G2 [END_CELL]"))
    with pytest.raises(RuntimeError, match="writer"):
        fm.PredictionCache(str(path), {"model": "m", "version": 1})
    cache.close()
    with pytest.raises(ValueError, match="contract mismatch"):
        fm.PredictionCache(str(path), {"model": "different", "version": 1})
    assert not os.path.exists(str(path) + ".lock")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"type":"prediction"')
    with pytest.raises(ValueError, match="corrupt prediction cache"):
        fm.PredictionCache(str(path), {"model": "m", "version": 1})


def test_atomic_json_refuses_immutable_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    fm.atomic_write_json(str(path), {"a": 1}, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fm.atomic_write_json(str(path), {"a": 2}, overwrite=False)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_validity_requires_genuine_end_cell_and_residual_down():
    panel = {"G1", "G2"}
    native_eos = ee.validity("G1 G2", "G1 G2 [END_CELL]", panel, min_genes=2,
                             generation={"end_cell_emitted": False, "native_eos_emitted": True,
                                         "cap_reached": False, "termination": "native_eos"})
    assert native_eos["complete"] is False
    capped = nir.generation_validity([7, 8], "G1 G2", end_id=9, eos_id=10,
                                     max_new_tokens=2, panel_index={"G1": 0, "G2": 1},
                                     model_kind="cellsentence", min_genes=2)
    assert capped["termination"] == "cap" and capped["complete"] is False
    incomplete_residual = residual.generation_validity(
        [7, 8], "G1 [DOWN] G2", end_id=9, eos_id=10, max_new_tokens=8,
        gene_index={"G1": 0, "G2": 1}, model_kind="residual", min_genes=2)
    assert incomplete_residual["has_down"] is True
    assert incomplete_residual["complete"] is False
    complete_residual = residual.generation_validity(
        [7, 9], "G1 [DOWN] G2 [END_CELL]", end_id=9, eos_id=10, max_new_tokens=8,
        gene_index={"G1": 0, "G2": 1}, model_kind="residual", min_genes=2)
    assert complete_residual["complete"] is True


def _artifact(name, offsets=None):
    offsets = offsets or {}
    rows = []
    drugs, lines = ("A", "B", "C"), ("L1", "L2", "L3")
    for di, drug in enumerate(drugs):
        for li, line in enumerate(lines):
            row_id = f"{drug}-{line}"
            model = 0.55 + 0.03 * di + 0.01 * li + offsets.get(row_id, 0.0)
            rows.append({
                "row_id": row_id, "tier": "tier2_unseen_drugs", "drug": drug,
                "cell_line": line, "plate": "p", "treatment_well_ids": [f"w-{drug}"],
                "k_collapsed": 8,
                "model": {"nir_expr": model},
                "control": {"nir_expr": 0.48 + 0.002 * li},
                "linear": {"nir_expr": 0.49 + 0.001 * di},
                "mean": {"nir_expr": 0.50},
                "scramble": {"nir_expr": 0.47 + 0.002 * di + 0.001 * li},
            })
    return {"manifest_sha256": "manifest", "metric_config_hash": "metric",
            "config": {"k_samples": 8},
            "tiers": {"tier2_unseen_drugs": {"rows": rows}}, "name": name}


def test_compare_backbones_reports_every_preregistered_clustered_contrast():
    gemma = _artifact("gemma", {"A-L1": .02, "B-L2": .03, "C-L3": .04})
    pythia = _artifact("pythia")
    result = cb.compare(gemma, pythia, tier="tier2_unseen_drugs",
                        left_name="gemma", right_name="pythia", per_drug_bh=True)
    assert result["support"]["common"] == 9
    expected = {"gemma_minus_chance", "gemma_minus_control", "gemma_minus_linear",
                "gemma_minus_mean", "gemma_minus_wrong_condition", "gemma_minus_pythia"}
    assert set(result["estimands"]) == expected
    for estimand in result["estimands"].values():
        assert set(estimand) == {"primary_drug_cell_line", "sensitivity_drug_well",
                                "sensitivity_cell_line_well"}
        assert estimand["primary_drug_cell_line"]["requested_axes"] == ["drug", "cell_line"]
        assert estimand["sensitivity_drug_well"]["cluster_relationship"] == "one_to_one"
    assert all("q_bh" in row for row in result["per_drug"])


def test_negative_cgm_variance_never_becomes_zero_width_interval():
    with pytest.raises(ValueError, match="inference withheld"):
        cb.cgm_mean_ci([0.0, 1.0, 1.0, 0.0], ["a", "a", "b", "b"],
                       ["x", "y", "x", "y"])


def test_compare_refuses_support_mismatch_duplicates_and_validity_failure():
    left, right = _artifact("left"), _artifact("right")
    right["tiers"]["tier2_unseen_drugs"]["rows"].pop()
    with pytest.raises(ValueError, match="support mismatch"):
        cb.compare(left, right, tier="tier2_unseen_drugs", left_name="l", right_name="r")
    dup = _artifact("dup")
    dup["tiers"]["tier2_unseen_drugs"]["rows"].append(
        copy.deepcopy(dup["tiers"]["tier2_unseen_drugs"]["rows"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        cb.compare(dup, _artifact("right"), tier="tier2_unseen_drugs",
                   left_name="l", right_name="r")
    failed = {"status": "validity_failure", "manifest_sha256": "manifest",
              "metric_config_hash": "metric", "tiers": {}}
    with pytest.raises(ValueError, match="validity-failure"):
        cb.compare(failed, _artifact("right"), tier="tier2_unseen_drugs",
                   left_name="l", right_name="r")

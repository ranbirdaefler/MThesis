"""Tests for the canonical Tahoe design layer (remediation plan, Workstream A4).

The fixtures the plan requires: same dose in different samples; different doses of one drug; the
same drug-dose on plates 6 and 14; mixed units converting to the same molar value; malformed or
absent units; and a sample wrongly carrying two drugs.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))

import tahoe_design as td


# --------------------------------------------------------------------------- dose parsing
@pytest.mark.parametrize("value,unit,expected", [
    (1, "M", 1.0),
    (1, "mM", 1e-3),
    (1, "uM", 1e-6),
    (1, "µM", 1e-6),        # micro sign
    (1, "μM", 1e-6),        # greek mu
    (1, "nM", 1e-9),
    (1, "pM", 1e-12),
    (0.05, "uM", 5e-8),
])
def test_molar_conversion(value, unit, expected):
    d = td.parse_dose(value, unit)
    assert d.ok
    assert d.molar == pytest.approx(expected, rel=1e-12)


def test_mixed_units_same_molar_group_together():
    """1000 nM and 1 uM are the same experiment and must not split into two dose groups."""
    a = td.parse_dose(1000, "nM").molar
    b = td.parse_dose(1, "uM").molar
    assert a != b, "if these ever become bit-identical this test is no longer testing anything"
    assert td.same_dose(a, b)
    assert td.molar_key(a) == td.molar_key(b)


def test_different_doses_of_one_drug_stay_different():
    lo = td.parse_dose(0.05, "uM").molar
    hi = td.parse_dose(5, "uM").molar
    assert not td.same_dose(lo, hi)
    assert td.molar_key(lo) != td.molar_key(hi)


@pytest.mark.parametrize("bad", [None, "unknown", "", "n/a", "0.05", "uM", "garbage"])
def test_unparseable_dose_is_never_zero(bad):
    """A missing dose and a zero dose are different experiments."""
    d = td.parse_dose(bad)
    assert d.molar is None
    assert d.reason
    assert d.molar != 0.0


def test_zero_dose_is_a_real_zero():
    d = td.parse_dose(0.0, "uM")
    assert d.ok and d.molar == 0.0


def test_mass_units_refuse_rather_than_guess():
    d = td.parse_dose(10, "mg/mL")
    assert d.molar is None
    assert "molecular weight" in d.reason


def test_raw_is_always_retained():
    assert td.parse_dose(0.05, "uM").raw == "0.05 uM"
    assert td.parse_dose("total garbage").raw == "total garbage"


def test_unknown_never_equals_anything():
    assert not td.same_dose(None, None)
    assert not td.same_dose(td.parse_dose(1, "uM").molar, None)


# --------------------------------------------------------------------------- treatments
def test_single_treatment():
    st = td.parse_treatment("[('Lapatinib', 0.05, 'uM')]", "smp_1")
    assert len(st.treatments) == 1
    assert st.treatments[0].drug == "Lapatinib"
    assert st.treatments[0].dose.molar == pytest.approx(5e-8)
    assert st.primary is not None
    assert not st.is_combination


def test_combination_keeps_every_component():
    """The shipped parser took parsed[0] and silently discarded the rest."""
    st = td.parse_treatment("[('A', 0.05, 'uM'), ('B', 1.0, 'uM')]", "smp_2")
    assert len(st.treatments) == 2
    assert st.is_combination
    assert {t.drug for t in st.treatments} == {"A", "B"}


def test_combination_has_no_primary():
    """Callers unable to handle combinations must be forced to notice them."""
    st = td.parse_treatment("[('A', 0.05, 'uM'), ('B', 1.0, 'uM')]", "smp_2")
    assert st.primary is None


def test_control_detection():
    assert td.parse_treatment("[('DMSO_TF', 0.0, 'uM')]", "s").is_control
    assert not td.parse_treatment("[('Lapatinib', 0.05, 'uM')]", "s").is_control


def test_malformed_treatment_yields_nothing():
    assert td.parse_treatment("not a list", "s").treatments == []
    assert td.parse_treatment("", "s").treatments == []


# --------------------------------------------------------------------------- keys and units
def test_condition_key_is_sample_and_line_not_the_old_tuple():
    row = {"sample_id": "smp_9", "cell_line_id": "CVCL_0031", "plate_id": "p6",
           "drug_id": "D", "dose_molar": 5e-8}
    assert td.treatment_key(row) == "smp_9"
    assert td.condition_key(row) == ("smp_9", "CVCL_0031")


def test_treatment_key_requires_a_sample():
    with pytest.raises(KeyError):
        td.treatment_key({"drug_id": "D", "cell_line_id": "c"})


def test_same_dose_different_samples_are_distinct_treatments():
    """Two wells at one concentration are two assignments, not one."""
    a = {"sample_id": "smp_1", "cell_line_id": "c1", "plate_id": "p6",
         "drug_id": "D", "dose_molar": 5e-8}
    b = {"sample_id": "smp_2", "cell_line_id": "c1", "plate_id": "p6",
         "drug_id": "D", "dose_molar": 5e-8}
    assert td.treatment_key(a) != td.treatment_key(b)
    assert td.validate_sample_mapping([a, b]).ok


def test_replicate_plates_six_and_fourteen():
    """The same drug-dose on two plates: legal, and two independent treatments."""
    rows = [{"sample_id": "smp_p6", "cell_line_id": "c1", "plate_id": "plate6",
             "drug_id": "D", "dose_molar": 5e-8},
            {"sample_id": "smp_p14", "cell_line_id": "c1", "plate_id": "plate14",
             "drug_id": "D", "dose_molar": 5e-8}]
    rep = td.validate_sample_mapping(rows)
    assert rep.ok
    assert rep.n_samples == 2


def test_many_cell_lines_in_one_sample_is_expected():
    rows = [{"sample_id": "s1", "cell_line_id": f"c{i}", "plate_id": "p1",
             "drug_id": "D", "dose_molar": 5e-8} for i in range(12)]
    rep = td.validate_sample_mapping(rows)
    assert rep.ok
    assert rep.detail["cell_lines_per_sample"]["max"] == 12


def test_sample_with_two_drugs_is_caught():
    rows = [{"sample_id": "s1", "cell_line_id": "c1", "plate_id": "p1",
             "drug_id": "D1", "dose_molar": 5e-8},
            {"sample_id": "s1", "cell_line_id": "c2", "plate_id": "p1",
             "drug_id": "D2", "dose_molar": 5e-8}]
    rep = td.validate_sample_mapping(rows)
    assert not rep.ok
    assert "sample_with_multiple_drugs" in rep.problems


def test_sample_on_two_plates_is_caught():
    rows = [{"sample_id": "s1", "cell_line_id": "c1", "plate_id": "p1",
             "drug_id": "D", "dose_molar": 5e-8},
            {"sample_id": "s1", "cell_line_id": "c2", "plate_id": "p2",
             "drug_id": "D", "dose_molar": 5e-8}]
    assert "sample_on_multiple_plates" in td.validate_sample_mapping(rows).problems


def test_mixed_units_do_not_trip_the_multiple_dose_check():
    """The validator groups on the canonical key, so a unit difference is not a design violation."""
    rows = [{"sample_id": "s1", "cell_line_id": "c1", "plate_id": "p1", "drug_id": "D",
             "dose_molar": td.parse_dose(1, "uM").molar},
            {"sample_id": "s1", "cell_line_id": "c2", "plate_id": "p1", "drug_id": "D",
             "dose_molar": td.parse_dose(1000, "nM").molar}]
    rep = td.validate_sample_mapping(rows)
    assert "sample_with_multiple_doses" not in rep.problems


# --------------------------------------------------------------------------- the shipped defect
def test_sample_ids_in_a_dose_column_are_detected():
    """The exact defect that reached production: dose_float holding 'smp_1841'."""
    assert td.looks_like_sample_id(["smp_1841", "smp_1882", "smp_1890"])
    assert td.looks_like_sample_id(["smp-1", "SMP_2", "smp_3"])


def test_a_real_dose_column_is_not_flagged():
    assert not td.looks_like_sample_id([5e-8, 1e-6, 5e-9])
    assert not td.looks_like_sample_id(["0.05 uM", "1 uM"])
    assert not td.looks_like_sample_id([])

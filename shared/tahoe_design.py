#!/usr/bin/env python
r"""
tahoe_design.py -- the one authoritative description of what Tahoe's experimental design contains.
=============================================================================================
Every analysis in this repository keys on `(drug, cell_line, plate, dose)` as though those four
attributes identify a physical experiment. They do not, and an audit found three consequences:

  1. `dose` in the OT and residual caches holds a SAMPLE IDENTIFIER (`smp_1841`), not a
     concentration. The raw column is `drugname_drugconc` in `sample_metadata.parquet`, the caches
     stored the sample key instead, and every "same dose" and "different dose" statistic computed
     downstream is therefore about sample identity.
  2. The existing `parse_dose` returns a DISPLAY STRING ("0.05 uM"). Nothing in the repository ever
     held a numeric concentration, so no analysis could compare doses on a common scale even in
     principle.
  3. `drugname_drugconc` is a LIST of (drug, value, unit) triples, and the existing parser takes
     `parsed[0]`. Combination treatments are silently reduced to their first component.

The assignment unit is also wrong. Tahoe treats a spheroid **sample/well** containing a mixture of
cell lines; the per-cell-line profiles are deconvolved observations nested inside one treatment.
Cells are not treatment replicates, and two cell lines from one treated well are not two independent
assignments of that drug.

This module is the single place those facts live. Downstream code should import it rather than
re-deriving any of it.

    python shared/tahoe_design.py --selftest
"""
import argparse
import ast
import logging
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- units
# Molar prefixes only. A concentration expressed by mass (mg/mL) or by fraction (%) cannot be
# converted without a molecular weight, and inventing one is worse than admitting the gap.
MOLAR_FACTORS = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6, "μm": 1e-6, "µm": 1e-6,     # U+03BC (greek mu) and U+00B5 (micro sign) both occur
    "nm": 1e-9,
    "pm": 1e-12,
    "fm": 1e-15,
}
NON_MOLAR_UNITS = {"mg/ml", "ug/ml", "μg/ml", "µg/ml", "ng/ml", "%", "v/v", "w/v", "x"}


@dataclass
class Dose:
    """A concentration that knows what it does and does not know.

    `molar` is None whenever the value could not be converted, and `reason` says why. It is never
    0.0 for an unparseable input: a missing dose and a zero dose are different experiments, and the
    difference matters for a control.
    """
    raw: str                       # the original cell, unmodified, always retained
    value: Optional[float] = None
    unit: Optional[str] = None
    molar: Optional[float] = None
    reason: Optional[str] = None   # populated iff molar is None

    @property
    def ok(self) -> bool:
        return self.molar is not None

    @property
    def log10_molar(self) -> Optional[float]:
        if self.molar is None or self.molar <= 0:
            return None
        return math.log10(self.molar)

    def display(self) -> str:
        if self.value is None or self.unit is None:
            return "unknown"
        v = f"{self.value:g}"
        return f"{v} {self.unit}"


@dataclass
class Treatment:
    """One drug applied at one concentration. A sample may carry several (a combination)."""
    drug: str
    dose: Dose


@dataclass
class SampleTreatment:
    """Everything a sample identifier resolves to."""
    sample_id: str
    treatments: List[Treatment] = field(default_factory=list)
    raw: str = ""

    @property
    def is_combination(self) -> bool:
        return len(self.treatments) > 1

    @property
    def is_control(self) -> bool:
        """DMSO/vehicle rows carry no active compound, however they are spelled."""
        if not self.treatments:
            return True
        return all(t.drug.strip().upper() in ("DMSO", "DMSO_TF", "VEHICLE", "UNTREATED", "")
                   for t in self.treatments)

    @property
    def primary(self) -> Optional[Treatment]:
        """The single treatment, for the common case. None for a combination -- callers that cannot
        handle combinations must skip them explicitly rather than silently take the first."""
        return self.treatments[0] if len(self.treatments) == 1 else None


# --------------------------------------------------------------------------- parsing
_NUM_UNIT = re.compile(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([A-Za-zμµ/%]+)\s*$")


def normalise_to_molar(value: Optional[float], unit: Optional[str]):
    """(value, unit) -> (molar or None, reason or None). Never returns 0.0 for an unknown input."""
    if value is None:
        return None, "no numeric value"
    if unit is None:
        return None, "no unit"
    u = str(unit).strip().lower().replace(" ", "")
    if u in NON_MOLAR_UNITS:
        return None, f"unit '{unit}' is not molar and needs a molecular weight to convert"
    f = MOLAR_FACTORS.get(u)
    if f is None:
        return None, f"unrecognised unit '{unit}'"
    try:
        return float(value) * f, None
    except (TypeError, ValueError):
        return None, f"value '{value}' is not numeric"


def molar_key(molar: Optional[float], sig: int = 6):
    """Canonical grouping key for a concentration.

    Unit conversion is not exact in floating point: 1000 nM evaluates to 1.0000000000000002e-06
    while 1 uM evaluates to 1e-06. Grouping on the raw float would split one concentration into two
    groups whenever the source rows happened to use different units, which is precisely the silent
    failure a "same dose" analysis must not have. Round to significant figures and group on that.
    """
    if molar is None:
        return None
    if molar == 0:
        return 0.0
    return float("%.*g" % (sig, molar))


def same_dose(a: Optional[float], b: Optional[float], rtol: float = 1e-6) -> bool:
    """Are two molar concentrations the same experiment? Unknown is never equal to anything."""
    if a is None or b is None:
        return False
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def parse_dose(raw_value, raw_unit=None) -> Dose:
    """Parse a concentration.

    Accepts a number plus a unit, or a single string carrying both ("0.05 uM"). Anything it cannot
    convert comes back with molar=None and a reason -- see the class docstring for why that is not
    zero.
    """
    raw_str = f"{raw_value} {raw_unit}" if raw_unit is not None else str(raw_value)
    if raw_value is None:
        return Dose(raw=raw_str, reason="value is None")
    if raw_unit is not None:
        try:
            v = float(raw_value)
        except (TypeError, ValueError):
            return Dose(raw=raw_str, reason=f"value '{raw_value}' is not numeric")
        molar, reason = normalise_to_molar(v, raw_unit)
        return Dose(raw=raw_str, value=v, unit=str(raw_unit), molar=molar, reason=reason)

    m = _NUM_UNIT.match(str(raw_value))
    if not m:
        return Dose(raw=raw_str, reason="could not split into a number and a unit")
    v = float(m.group(1))
    u = m.group(2)
    molar, reason = normalise_to_molar(v, u)
    return Dose(raw=raw_str, value=v, unit=u, molar=molar, reason=reason)


def parse_treatment(drugname_drugconc, sample_id: str = "") -> SampleTreatment:
    """Parse Tahoe's `drugname_drugconc`, e.g. `[('Lapatinib', 0.05, 'uM')]`.

    Returns EVERY component, not just the first. A two-drug sample is a combination experiment and
    an analysis that treats it as a single-drug condition is answering a different question; callers
    should branch on `is_combination` rather than receiving a silently truncated answer.
    """
    st = SampleTreatment(sample_id=str(sample_id), raw=str(drugname_drugconc))
    try:
        parsed = ast.literal_eval(str(drugname_drugconc))
    except (ValueError, SyntaxError):
        return st
    if isinstance(parsed, (list, tuple)):
        items = parsed if parsed and isinstance(parsed[0], (list, tuple)) else [parsed]
        for it in items:
            if not isinstance(it, (list, tuple)) or len(it) < 3:
                continue
            name, val, unit = it[0], it[1], it[2]
            st.treatments.append(Treatment(drug=str(name), dose=parse_dose(val, unit)))
    return st


# --------------------------------------------------------------------------- keys
def treatment_key(row: Dict[str, Any]) -> str:
    """The physical assignment unit: one drug-dose applied to one well/sample."""
    for k in ("sample_id", "sample"):
        if row.get(k) is not None:
            return str(row[k])
    raise KeyError("no sample_id/sample field -- the treatment unit cannot be identified")


def condition_key(row: Dict[str, Any]):
    """The observation unit: one cell line's deconvolved profile within one treated sample.

    Deliberately NOT (drug, cell_line, plate, dose). Two rows sharing that tuple can come from one
    physical well, and treating them as independent is the pseudoreplication the audit flagged.
    """
    cl = row.get("cell_line_id", row.get("cell_line"))
    if cl is None:
        raise KeyError("no cell_line_id field")
    return (treatment_key(row), str(cl))


# --------------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    n_rows: int = 0
    n_samples: int = 0
    ok: bool = True
    problems: Dict[str, int] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)

    def fail(self, name: str, n: int = 1, detail=None):
        self.problems[name] = self.problems.get(name, 0) + n
        if detail is not None:
            self.detail.setdefault(name, []).append(detail)
        self.ok = False

    def to_dict(self):
        return asdict(self)


def validate_sample_mapping(rows) -> ValidationReport:
    """Assert that a sample identifies exactly one treatment, on one plate.

    `rows` is any iterable of dict-like records carrying at least sample_id, and optionally
    plate_id, drug_id and dose_molar. Violations are counted rather than raised, so a caller can
    report the whole picture instead of dying on the first bad row.
    """
    rep = ValidationReport()
    by_sample = {}
    for r in rows:
        rep.n_rows += 1
        try:
            s = treatment_key(r)
        except KeyError:
            rep.fail("row_without_sample_id")
            continue
        rec = by_sample.setdefault(s, {"plates": set(), "drugs": set(), "doses": set(),
                                       "lines": set()})
        if r.get("plate_id") is not None:
            rec["plates"].add(str(r["plate_id"]))
        if r.get("drug_id") is not None:
            rec["drugs"].add(str(r["drug_id"]))
        if r.get("dose_molar") is not None:
            rec["doses"].add(molar_key(float(r["dose_molar"])))
        cl = r.get("cell_line_id", r.get("cell_line"))
        if cl is not None:
            rec["lines"].add(str(cl))

    rep.n_samples = len(by_sample)
    for s, rec in by_sample.items():
        if len(rec["plates"]) > 1:
            rep.fail("sample_on_multiple_plates", detail={"sample": s, "plates": sorted(rec["plates"])})
        if len(rec["drugs"]) > 1:
            rep.fail("sample_with_multiple_drugs", detail={"sample": s, "drugs": sorted(rec["drugs"])})
        if len(rec["doses"]) > 1:
            rep.fail("sample_with_multiple_doses", detail={"sample": s, "doses": sorted(rec["doses"])})
    rep.detail["cell_lines_per_sample"] = {
        "min": min((len(v["lines"]) for v in by_sample.values()), default=0),
        "max": max((len(v["lines"]) for v in by_sample.values()), default=0),
        "mean": (sum(len(v["lines"]) for v in by_sample.values()) / len(by_sample)) if by_sample else 0,
    }
    return rep


def sample_column(df):
    """Which column of a cache's `meta.parquet` actually carries the treatment/well identifier.

    Returns (column_name_or_None, how). `build_embeddings.py` wrote `row["sample"]` into a column it
    named `dose`, so caches built before that was fixed carry the identifier under the wrong name and
    carry no concentration at all. Recovering it is preferable to rebuilding, but only if the
    recovery is explicit -- hence the second return value, which callers should put in their report.
    """
    cols = list(getattr(df, "columns", df))
    for name in ("sample_id", "sample"):
        if name in cols:
            return name, "explicit column"
    if "dose" in cols:
        try:
            vals = df["dose"].astype(str).values
        except (TypeError, KeyError, AttributeError):
            vals = []
        if looks_like_sample_id(vals):
            return "dose", "recovered from the mislabelled `dose` column"
    return None, "unavailable"


def looks_like_sample_id(values) -> bool:
    """True if a column that is supposed to hold doses is actually holding sample identifiers.

    This is the specific defect that reached production: `dose_float` containing 'smp_1841'. Any
    builder writing a dose column should call this on its own output.
    """
    vals = [str(v) for v in values if v is not None][:2000]
    if not vals:
        return False
    hits = sum(1 for v in vals if re.match(r"^smp[_-]?\d+$", v.strip(), re.I))
    return hits > 0.5 * len(vals)


# --------------------------------------------------------------------------- selftest
def selftest():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if cond:
            logger.info(f"  ok   {msg}")
        else:
            logger.error(f"  FAIL {msg}")
            ok = False

    logger.info("units")
    check(parse_dose(0.05, "uM").molar == 5e-8, "0.05 uM -> 5e-8 M")
    check(parse_dose(0.05, "µM").molar == 5e-8, "micro sign U+00B5 parses")
    check(parse_dose(0.05, "μM").molar == 5e-8, "greek mu U+03BC parses")
    check(parse_dose(5, "nM").molar == 5e-9, "5 nM -> 5e-9 M")
    check(parse_dose("0.05 uM").molar == 5e-8, "combined string form parses")
    check(abs(parse_dose(1, "mM").molar - 1e-3) < 1e-18, "1 mM -> 1e-3 M")

    logger.info("unknowns are never zero")
    for bad, why in ((None, "None"), ("unknown", "unparseable string"), ("", "empty")):
        d = parse_dose(bad)
        check(d.molar is None and d.reason, f"{why} -> molar None with a reason")
    d = parse_dose(10, "mg/mL")
    check(d.molar is None and "molecular weight" in (d.reason or ""),
          "mass concentration refuses to convert and says why")
    check(parse_dose(0.0, "uM").molar == 0.0, "an actual zero dose IS zero, not missing")

    logger.info("raw is retained")
    check(parse_dose(0.05, "uM").raw == "0.05 uM", "raw string kept")
    check(parse_dose("garbage").raw == "garbage", "raw kept even when unparseable")

    logger.info("treatments, including combinations")
    st = parse_treatment("[('Lapatinib', 0.05, 'uM')]", "smp_1")
    check(len(st.treatments) == 1 and st.treatments[0].drug == "Lapatinib", "single treatment parses")
    check(st.primary is not None and not st.is_combination, "single treatment has a primary")
    st2 = parse_treatment("[('A', 0.05, 'uM'), ('B', 1.0, 'uM')]", "smp_2")
    check(len(st2.treatments) == 2 and st2.is_combination,
          "COMBINATION keeps both components (the old parser dropped the second)")
    check(st2.primary is None, "a combination has no primary, so callers must handle it explicitly")
    check(parse_treatment("[('DMSO_TF', 0.0, 'uM')]", "smp_3").is_control, "DMSO detected as control")
    check(parse_treatment("not a list", "smp_4").treatments == [], "garbage yields no treatments")

    logger.info("equal molarity from different units")
    a, b = parse_dose(1000, "nM").molar, parse_dose(1, "uM").molar
    check(a != b, "raw floats DIFFER after unit conversion (1000 nM vs 1 uM) -- this is why a "
                  "canonical comparison is required, not optional")
    check(same_dose(a, b), "same_dose() calls them equal")
    check(molar_key(a) == molar_key(b), "molar_key() groups them together")
    check(not same_dose(a, None), "unknown is never equal to a known dose")
    check(not same_dose(parse_dose(1, "uM").molar, parse_dose(2, "uM").molar),
          "genuinely different doses stay different")

    logger.info("keys")
    row = {"sample_id": "smp_9", "cell_line_id": "CVCL_0031", "plate_id": "p6"}
    check(treatment_key(row) == "smp_9", "treatment key is the sample")
    check(condition_key(row) == ("smp_9", "CVCL_0031"), "condition key is (sample, cell line)")

    logger.info("validation")
    good = [{"sample_id": "s1", "cell_line_id": "c1", "plate_id": "p1", "drug_id": "D", "dose_molar": 5e-8},
            {"sample_id": "s1", "cell_line_id": "c2", "plate_id": "p1", "drug_id": "D", "dose_molar": 5e-8}]
    check(validate_sample_mapping(good).ok, "two cell lines in one sample is legal (and expected)")
    bad = good + [{"sample_id": "s1", "cell_line_id": "c3", "plate_id": "p2",
                   "drug_id": "OTHER", "dose_molar": 1e-6}]
    rep = validate_sample_mapping(bad)
    check(not rep.ok and "sample_with_multiple_drugs" in rep.problems,
          "one sample with two drugs is caught")
    check("sample_on_multiple_plates" in rep.problems, "one sample on two plates is caught")

    logger.info("the defect that reached production")
    check(looks_like_sample_id(["smp_1841", "smp_1882", "smp_1890"]),
          "a dose column full of sample IDs is detected")
    check(not looks_like_sample_id([5e-8, 1e-6, 5e-9]), "a real dose column is not flagged")

    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        ap.print_help()

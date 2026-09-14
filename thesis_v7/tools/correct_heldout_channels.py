#!/usr/bin/env python3
"""Reproduce the v7 channel subgroups from the repository's saved scores.

Run from any directory: python thesis_v7/tools/correct_heldout_channels.py
Use --check to verify the committed v7 artifact and figure ledger without writing.
Requires numpy and scipy, as does shared/inference.py. No training or network
access is involved; RESULTS_cluster and the legacy analysis code remain intact.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path


THESIS = Path(__file__).resolve().parents[1]
REPO = THESIS.parent
OUTPUT = THESIS / "tools" / "data" / "heldout_channels_corrected.json"
SOURCE_COMMIT = "4696986b283dbbb74e4efaf126357981a3e9efb2"
SOURCES = {
    "RESULTS_cluster/re_v3.json": "29f701af70103796058ee8b2339ee802e96df48b",
    "RESULTS_cluster/channel_gate_v4.json": "cc1c4bb0c9244a45b871ddeb65c1147b60f710c4",
    "RESULTS_cluster/channel_gate_heldout_targets.json": "0835fe0538bdb896a019413aea52e824beb75f6f",
    "shared/inference.py": "538df1035ff21d5ae80c510f3f8128af7b2e1252",
}
EXPECTED_EXTRA = {
    "Artesunate", "Demeclocycline", "L-Thyroxine (sodium salt pentahydrate)",
    "Palmatine (chloride)",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_bytes(path):
    # Git may check out text as CRLF. Pin the LF-normalised Git text so the
    # provenance and output are identical on Windows and Unix checkouts.
    local = REPO / path
    raw = (local.read_bytes() if local.exists() else
           subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=REPO,
                          check=True, capture_output=True,
                          env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}).stdout)
    canonical = raw.replace(b"\r\n", b"\n")
    blob = hashlib.sha1(b"blob " + str(len(canonical)).encode() + b"\0" + canonical).hexdigest()
    require(blob == SOURCES[path], f"Source changed: {path} (Git blob {blob})")
    return canonical, {
        "path": path,
        "git_blob_sha1": blob,
        "sha256_lf": hashlib.sha256(canonical).hexdigest(),
    }


def close(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"{context}: {actual} != {expected}")


def compute():
    inputs, provenance = {}, []
    for path in SOURCES:
        payload, entry = source_bytes(path)
        provenance.append(entry)
        if path.endswith(".json"):
            inputs[path] = json.loads(payload)
    spec = importlib.util.spec_from_file_location("thesis_saved_inference", REPO / "shared/inference.py")
    inference = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(inference)

    ev = inputs["RESULTS_cluster/re_v3.json"]
    gate = inputs["RESULTS_cluster/channel_gate_v4.json"]
    published = inputs["RESULTS_cluster/channel_gate_heldout_targets.json"]
    records = ev["records"]
    require(ev["config"].get("train_file"), "Canonical evaluation must name its training file")
    require(sum(r["split"] == "train" for r in records) == 300,
            "Expected the canonical 300-condition sampled train tier")
    splits = {}
    for row in records:
        splits.setdefault(row["drug"], set()).add(row["split"])

    # A drug-level holdout is an explicit assignment, not absence from a sample.
    heldout = {d for d, labels in splits.items() if "unseen_drug" in labels}
    require(len(heldout) == 10, "Expected exactly ten assigned unseen_drug drugs")
    require(all(splits[d] == {"unseen_drug"} for d in heldout),
            "An unseen_drug drug has conflicting split labels")
    require(all(not r["trained"] and r.get("drug_lookup") is None
                for r in records if r["drug"] in heldout),
            "An assigned unseen_drug row is flagged trained or has a training lookup")

    previous = {d for d, labels in splits.items() if "train" not in labels}
    extra = previous - heldout
    require(extra == EXPECTED_EXTRA, "Unexpected drugs in the old sample-absence subgroup")
    extra_evidence = []
    for drug in sorted(extra):
        rows = [r for r in records if r["drug"] == drug]
        available = sum(r.get("drug_lookup") is not None for r in rows)
        require(available > 0, f"No training-only lookup evidence for {drug}")
        extra_evidence.append({"drug": drug, "n_scored": len(rows),
                               "n_training_lookup_available": available,
                               "split_labels": sorted(splits[drug])})

    def summarise(channel, drugs):
        null = channel + "_null_pm"
        rows = [r for r in gate["records"] if r["drug"] in drugs
                and r.get(channel) is not None and r.get(null) is not None]
        require(rows, f"No rows for {channel}")
        ci = inference.two_way_cluster_ci(
            [r[channel] - r[null] for r in rows],
            [r["cell_line"] for r in rows], [r["well"] for r in rows])
        require(ci is not None, f"Interval unavailable for {channel}")
        return {
            "n": len(rows), "n_drugs": len({r["drug"] for r in rows}),
            "drugs": sorted({r["drug"] for r in rows}),
            "n_cell_lines": ci["n_clusters_a"], "n_wells": ci["n_clusters_b"],
            "gap": ci["point"], "ci": [ci["lo"], ci["hi"]],
            "df": ci["df"], "unit": ci["unit"],
            "spans_zero": ci["lo"] <= 0 <= ci["hi"],
        }

    channels = {}
    for channel in ("target", "moa", "chem"):
        old = summarise(channel, previous)
        expected = published["channels"][channel]["heldout_targets"]
        for field in ("n", "n_drugs", "df"):
            require(old[field] == expected[field], f"Published {channel}.{field} did not reproduce")
        for field, actual, wanted in (("gap", old["gap"], expected["gap"]),
                                     ("lo", old["ci"][0], expected["ci"][0]),
                                     ("hi", old["ci"][1], expected["ci"][1])):
            close(actual, wanted, f"Published {channel}.{field}")
        channels[channel] = {"published_subgroup_reproduced": old,
                             "heldout_targets": summarise(channel, heldout)}
    require(channels["target"]["heldout_targets"] == channels["target"]["published_subgroup_reproduced"],
            "The two-drug target result should be unchanged")

    return {
        "description": "V7 held-out channel subgroups selected by explicit unseen_drug assignment",
        "source_repository": "https://github.com/ranbirdaefler/MThesis",
        "source_commit": SOURCE_COMMIT, "sources": provenance,
        "selection_rule": "drug has explicit unseen_drug label; reject any other split label for that drug",
        "training_file_recorded_in_evaluation": ev["config"]["train_file"],
        "n_scored_train_conditions": 300, "n_heldout_drugs": len(heldout),
        "heldout_drugs": sorted(heldout),
        "excluded_sample_absence_drugs": extra_evidence,
        "margin": gate["config"]["min_margin"],
        "published_subgroup_reproduced_before_correction": True,
        "channels": channels,
    }


def verify_figure(result):
    ledger = json.loads((THESIS / "figs/fig-gate.json").read_text())
    tex = (THESIS / "figs/gate.tex").read_text()
    require("\\def\\alo{-0.180}\\def\\ahi{0.265}" in tex,
            "Figure TeX axis differs from the corrected ledger range")
    rows = [r for r in ledger["rows"] if r["block"] == 3]
    require(len(rows) == 3, "Expected three held-out rows in the figure ledger")
    for row in rows:
        actual = result["channels"][row["channel"]]["heldout_targets"]
        for field in ("n", "n_drugs", "n_cell_lines", "n_wells", "df"):
            require(row[field] == actual[field], f"Figure {row['channel']}.{field} differs")
        for field, wanted in (("estimate", actual["gap"]), ("ci_lo", actual["ci"][0]), ("ci_hi", actual["ci"][1])):
            close(row[field], wanted, f"Figure {row['channel']}.{field}")
        drawn = [f"{actual['ci'][0]:+.4f}", f"{actual['gap']:+.4f}", f"{actual['ci'][1]:+.4f}"]
        require(row["drawn_in_tex"] == drawn, "Figure rounded values differ")
        literal = ("\\gaprow{" + str(row["row_y"]) + "}{" + "}{".join(drawn) + "}{"
                   + row["row_label"] + "}{\\num{" + str(row["n"]) + "}}")
        require(literal in tex, f"Missing corrected TeX row: {literal}")
        require(ledger["axis_range_drawn"]["alo"] <= actual["ci"][0]
                and ledger["axis_range_drawn"]["ahi"] >= actual["ci"][1],
                "Corrected interval extends outside the figure data range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify saved artifact and figure; write nothing")
    args = parser.parse_args()
    result = compute()
    if args.check:
        require(json.loads(OUTPUT.read_text()) == result, "Saved v7 artifact differs from recomputation")
        verify_figure(result)
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for channel, item in result["channels"].items():
        row = item["heldout_targets"]
        print(f"{channel}: n={row['n']}, drugs={row['n_drugs']}, "
              f"gap={row['gap']:+.4f} [{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}], df={row['df']}")
    print("Published subgroup reproduced; corrected artifact" + (" and figure verified." if args.check else " written."))


if __name__ == "__main__":
    main()

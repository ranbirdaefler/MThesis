#!/usr/bin/env python
r"""
artifact_manifest.py — can every number in the thesis be shown to an examiner?
=============================================================================
An audit found six result sets quoted in the thesis with no committed artifact: the channel gate,
the probe arms and their replication, the three kappa runs, the field decomposition, and the
variance decomposition carrying the matched-null / dose / shared-split arms. In a viva "show me this
number" is a fair request for any number in the document, and right now it cannot be answered for
roughly a sixth of the ledger.

The audit calls those results "not established". That is the right word for what it could verify,
but it is probably not what happened: the jobs ran on the cluster and wrote into `~/tahoe/RESULTS`,
and what is missing is the copy back, not the computation. So this script does not assume the
numbers are lost. It runs WHERE THE ARTIFACTS ARE, reports present / missing / superseded per claim,
and tars up everything present so it can be brought back and committed in one step.

A missing artifact after that is a real gap and gets one of two honest outcomes: re-run the job, or
withdraw the number.

USAGE
  python artifact_manifest.py --selftest
  python artifact_manifest.py --scan ~/tahoe/RESULTS --bundle /tmp/artifacts.tar.gz
  python artifact_manifest.py --scan RESULTS_cluster --strict      # exits 1 if anything is missing
"""
import argparse
import json
import logging
import os
import sys
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- the manifest
# claim id -> what it backs, which file, which script produced it, and whether the thesis quotes it.
# `quoted` is the field that matters: an artifact nobody cites is housekeeping, a quoted number with
# no artifact is a viva liability.
MANIFEST = [
    # ---- Q13 / Q21, the mechanistic probe --------------------------------------------------
    dict(id="probe_arms", quoted=True, files=["probe_arm_*.json"],
         script="endcell/analysis/workspace_probe.py",
         backs="Q21 headline: residual arm, layer 12, +0.0197 [+0.0063,+0.0335], q=0.026; "
               "single_cell null at all six layers"),
    dict(id="probe_replication", quoted=True, files=["probe_rep_*.json"],
         script="endcell/analysis/workspace_probe.py",
         backs="Q21 replication table, seeds 43/44/45"),
    # ---- Q18, the channel gate -------------------------------------------------------------
    dict(id="channel_gate", quoted=True, files=["channel_gate.json"],
         script="endcell/analysis/channel_gate.py",
         backs="Q18 'two live channels': MoA +0.078, target +0.084",
         superseded_by="channel_gate_platematched.json",
         note="the count-matched null is not plate-matched; see the plate-matched re-run"),
    dict(id="channel_gate_platematched", quoted=False, files=["channel_gate_platematched.json"],
         script="endcell/analysis/channel_gate.py",
         backs="the re-measurement that decides whether Q18's verdict survives"),
    # ---- Q17, transfer coefficient ---------------------------------------------------------
    dict(id="variance_decomposition", quoted=True,
         files=["vardecomp_*.json", "variance_decomp*.json"],
         script="endcell/analysis/variance_decomposition.py",
         backs="Q17 table: T=0.557 [0.513,0.601], same-plate null -0.018 vs +0.478, "
               "structure-matched null +0.000, shared-split bias, dose column",
         note="the committed vardecomp_plate.json carries generic_scope='plate' and T=0.549, "
              "not the quoted 0.557, and lacks the matched-null / dose / shared-split arms"),
    # ---- Q19, kappa ------------------------------------------------------------------------
    dict(id="kappa_plate", quoted=True, files=["kappa_plate.json"],
         script="endcell/analysis/kappa_channel.py", backs="Q19 kappa structure at plate scope"),
    dict(id="kappa_cellline", quoted=True, files=["kappa_cellline.json"],
         script="endcell/analysis/kappa_channel.py", backs="Q19 kappa structure at cell-line scope"),
    dict(id="kappa_channel", quoted=True, files=["kappa_channel*.json"],
         script="endcell/analysis/kappa_channel.py",
         backs="Q19 mechanism-matched vs mismatched cos(kappa), ceiling 0.195"),
    # ---- Q20, field decomposition ----------------------------------------------------------
    dict(id="field_decomp", quoted=True, files=["field_decomp.json"],
         script="endcell/analysis/residual_eval.py --field_decomp",
         backs="Q20: swap-drug-only, swap-MoA-only, swap-both; mechanism CI [-0.017,+0.038]"),
    # ---- Q15 / Q16, residual arm -----------------------------------------------------------
    dict(id="residual_eval_1400", quoted=True,
         files=["re_singlecell_model.json", "re_ot_model.json", "re_residual*.json"],
         script="endcell/analysis/residual_eval.py",
         backs="Q15/Q16 at max_new_tokens=1400",
         note="the local copies are the 600-token runs with the opposite-sign gap the thesis says "
              "was superseded; the 1400-token re-runs are the quoted ones"),
    dict(id="residual_holdout2", quoted=True, files=["re_holdout2.json"],
         script="endcell/analysis/residual_eval.py",
         backs="unseen_combo gap +0.1002 [+0.0661,+0.1368]; model NIR 0.657 / 0.650"),
    # ---- the repairs, not yet run ----------------------------------------------------------
    dict(id="experimental_unit_audit", quoted=False, files=["experimental_unit_audit.json"],
         script="endcell/analysis/experimental_unit_audit.py",
         backs="the clustering unit for every CI, and the dose_float collision check"),
    dict(id="residual_targets_repaired", quoted=False, files=["report.json", "fit.sha"],
         script="endcell/ot/build_residual_targets.py",
         backs="the split-before-fit rebuild and its scope-sensitivity gate"),
    # ---- already committed, listed so the manifest is the whole ledger ---------------------
    dict(id="calibration", quoted=True, files=["calibration.json", "calibration_hi.json"],
         script="endcell/analysis/nir_calibration.py",
         backs="the metric audit: NIR is the only calibrated metric, DRF within-plate +0.446"),
    dict(id="nir_benchmark", quoted=True, files=["nir_benchmark.json", "nir_benchmark_perdrug.json"],
         script="endcell/analysis/nir_benchmark.py", backs="NIR benchmark table"),
    dict(id="dose_coverage", quoted=True, files=["dose_coverage.json"],
         script="endcell/analysis/check_dose_coverage.py",
         backs="97.6% of (drug, cell line, plate) groups have exactly one dose value -- the evidence "
               "that the 'different dose' arm was not measuring dose"),
    dict(id="leak_audit", quoted=True, files=["leak_audit.json", "leak_sameplate.json",
                                              "leak_crossplate.json"],
         script="endcell/analysis/leak_audit.py", backs="the leakage audit"),
]


def _match(dirpath, pattern):
    import fnmatch
    try:
        names = os.listdir(dirpath)
    except OSError:
        return []
    return sorted(n for n in names if fnmatch.fnmatch(n, pattern))


def scan(dirpath):
    """Match every pattern independently and keep every hit.

    The first version tried to build `found` and `missing` in one pass with an index-splice, and the
    splice deleted an earlier hit whenever a later pattern in the same entry missed. So a multi-file
    entry that was partly present reported ZERO files and, worse, `bundle()` then omitted the files
    that did exist. That is exactly the failure mode this script is supposed to detect, produced by
    the script itself.
    """
    rows = []
    for m in MANIFEST:
        found, missing = [], []
        for pat in m["files"]:
            hits = _match(dirpath, pat)
            if hits:
                found.extend(hits)
            else:
                missing.append(pat)
        rows.append({**m, "found": sorted(set(found)), "missing_patterns": missing,
                     "ok": not missing, "partial": bool(found) and bool(missing)})
    return rows


def report(rows, dirpath):
    quoted_missing = [r for r in rows if r["quoted"] and not r["ok"]]
    logger.info("=" * 100)
    logger.info(f"ARTIFACT SCAN of {dirpath}")
    logger.info("=" * 100)
    for r in sorted(rows, key=lambda x: (x["ok"], not x["quoted"])):
        mark = ("ok  " if r["ok"] else
                ("PART" if r.get("partial") else ("MISS" if r["quoted"] else "----")))
        logger.info(f"  {mark}  {r['id']:28s} {'[quoted]' if r['quoted'] else '        '} "
                    f"{len(r['found'])} file(s)")
        if not r["ok"]:
            logger.info(f"        missing: {', '.join(r['missing_patterns'])}")
            logger.info(f"        produced by: {r['script']}")
            logger.info(f"        backs: {r['backs']}")
        if r.get("note"):
            logger.info(f"        NOTE: {r['note']}")
    logger.info("-" * 100)
    logger.info(f"  {sum(1 for r in rows if r['ok'])}/{len(rows)} entries complete; "
                f"{len(quoted_missing)} THESIS-QUOTED entries have no artifact here")
    if quoted_missing:
        logger.info("  Every one of these needs a re-run or a withdrawal. They are:")
        for r in quoted_missing:
            logger.info(f"    - {r['id']}: {r['backs']}")
    logger.info("=" * 100)
    return quoted_missing


def bundle(rows, dirpath, out):
    """Every file found, including from entries that are only partly satisfied -- a partial entry's
    files are exactly the ones most worth rescuing."""
    n = 0
    with tarfile.open(out, "w:gz") as tf:
        for r in rows:
            for f in r["found"]:
                p = os.path.join(dirpath, f)
                if os.path.isfile(p):
                    tf.add(p, arcname=f)
                    n += 1
    logger.info(f"bundled {n} artifacts -> {out}")
    return n


def selftest():
    import shutil
    import tempfile
    ok = True

    def check(c, m):
        nonlocal ok
        logger.info(("  ok   " if c else "  FAIL ") + m)
        ok = ok and c

    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "channel_gate.json"), "w").write("{}")
        open(os.path.join(d, "probe_arm_residual.json"), "w").write("{}")
        open(os.path.join(d, "probe_arm_single_cell.json"), "w").write("{}")
        rows = scan(d)
        by = {r["id"]: r for r in rows}
        check(by["channel_gate"]["ok"], "an exact filename is found")
        check(len(by["probe_arms"]["found"]) == 2, "a glob collects every matching artifact")
        check(not by["field_decomp"]["ok"], "an absent artifact is reported missing")
        check(by["field_decomp"]["quoted"] is True, "the missing one is flagged as thesis-quoted")

        missing = report(rows, d)
        check(all(r["quoted"] for r in missing), "only quoted entries drive the exit status")
        check("channel_gate" not in {r["id"] for r in missing}, "a present entry is not reported")

        tgz = os.path.join(d, "b.tar.gz")
        n = bundle(rows, d, tgz)
        check(n == 3, "the bundle contains exactly the artifacts that were found")
        with tarfile.open(tgz) as tf:
            names = set(tf.getnames())
        check("channel_gate.json" in names and "probe_arm_residual.json" in names,
              "bundled members are stored under their bare filenames")

        empty = scan(os.path.join(d, "does_not_exist"))
        check(all(not r["ok"] for r in empty), "a nonexistent directory reports everything missing")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="directory holding result JSON (e.g. ~/tahoe/RESULTS)")
    ap.add_argument("--bundle", help="write a .tar.gz of everything found, for copying back")
    ap.add_argument("--out", default=None, help="write the scan as JSON here")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any thesis-quoted artifact is missing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return
    if not a.scan:
        ap.error("--scan required (unless --selftest)")

    rows = scan(os.path.expanduser(a.scan))
    missing = report(rows, a.scan)
    if a.bundle:
        bundle(rows, os.path.expanduser(a.scan), os.path.expanduser(a.bundle))
    if a.out:
        json.dump({"scanned": a.scan,
                   "entries": [{k: v for k, v in r.items() if k != "files"} for r in rows]},
                  open(a.out, "w"), indent=2)
        logger.info(f"scan -> {a.out}")
    if a.strict and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

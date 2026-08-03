#!/usr/bin/env python
r"""
build_thesis_assets.py -- every quoted number comes from an artifact, or it does not get quoted.
================================================================================================
Two audits asked for the same thing in different words: a reader should be able to take any figure
in the thesis and trace it to the file that produced it. Right now the numbers are typed into LaTeX
by hand, which is how `+0.078` survived a change of control construction and how a retracted 62%/19%
comparison stayed in a document arguing against itself.

This emits two things from the result JSONs:

  1. `thesis/generated/numbers.tex` -- LaTeX macros. `\NIRResidualUnseenCombo` expands to the value
     in the artifact. A number that changes upstream changes in the document at the next compile,
     and a number with no artifact cannot be written at all, because the macro will not exist and
     LaTeX will fail loudly rather than silently keeping a stale figure.

  2. `thesis/generated/claim_artifact_table.tex` -- the claim -> artifact -> script -> commit table
     for the appendix. This is the cheap version of the artifact-lock system the remediation scope
     cut; it captures the part a viva actually tests ("show me this number") without the weeks of
     infrastructure.

DELIBERATELY STRICT. A claim whose artifact is missing is written into the table as MISSING and, with
`--strict`, fails the build. A silent gap in a provenance document is worse than no provenance
document, because it looks like coverage.

USAGE
  python build_thesis_assets.py --selftest
  python build_thesis_assets.py --results RESULTS_cluster --out_dir thesis/generated
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "shared"),
           os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- the registry
# macro name -> (artifact file, dotted path inside it, format, what it backs)
# Dotted paths may index lists numerically. A path that does not resolve is a MISSING claim.
CLAIMS = [
    # ---- Q17, the transfer coefficient -------------------------------------------------
    dict(macro="TransferCoefficient", file="vardecomp_matched.json",
         path="main.T::repro-filtered (cos>0.2, the training set).T", fmt="{:.3f}",
         backs="T, the drug x cell-line interaction share",
         script="endcell/analysis/variance_decomposition.py"),
    dict(macro="TransferCoefficientCI", file="vardecomp_matched.json",
         path="main.T::repro-filtered (cos>0.2, the training set).ci_dyadic", fmt="ci3",
         backs="T's dyadic-robust interval",
         script="endcell/analysis/variance_decomposition.py"),
    dict(macro="ResidualEnergyShare", file="vardecomp_matched.json",
         path="main.scope.residual_energy_share", fmt="{:.1%}",
         backs="fraction of response ENERGY that is drug-specific",
         script="endcell/analysis/variance_decomposition.py"),
    dict(macro="CrossEnergyShare", file="vardecomp_matched.json",
         path="main.scope.cross_energy_share", fmt="{:+.3f}",
         backs="the residual/generic cross term, previously assumed away",
         script="endcell/analysis/variance_decomposition.py"),
    # ---- Q18, the channel gate ----------------------------------------------------------
    dict(macro="ChannelTargetGap", file="channel_gate_platematched.json",
         path="channels.target.plate_matched.gap", fmt="{:+.4f}",
         backs="protein-target channel over a PLATE-MATCHED null",
         script="endcell/analysis/channel_gate.py"),
    dict(macro="ChannelMoaGap", file="channel_gate_platematched.json",
         path="channels.moa.plate_matched.gap", fmt="{:+.4f}",
         backs="mechanism channel over a PLATE-MATCHED null",
         script="endcell/analysis/channel_gate.py"),
    dict(macro="ChannelMoaCoplateExcess", file="channel_gate_platematched.json",
         path="coplate.moa.channel", fmt="{:.3f}",
         backs="how co-plated mechanism partners are -- the size of the confound",
         script="endcell/analysis/channel_gate.py"),
    # ---- Q21, the probe -------------------------------------------------------------------
    dict(macro="ProbeSurvivors", file="probe_canonical.json",
         path="survivors", fmt="len",
         backs="how many (arm, layer) cells survive ONE global BH",
         script="endcell/analysis/aggregate_workspace_probe.py"),
    dict(macro="ProbeFamilySize", file="probe_canonical.json",
         path="n_tests_in_global_bh", fmt="{:d}",
         backs="the size of the family the correction spans",
         script="endcell/analysis/aggregate_workspace_probe.py"),
    dict(macro="ProbeUnmeasuredCells", file="probe_canonical.json",
         path="family.n_missing", fmt="{:d}",
         backs="cells never measured -- absences of a test, not null results",
         script="endcell/analysis/aggregate_workspace_probe.py"),
    # ---- the rebuild ------------------------------------------------------------------------
    dict(macro="ReproThreshold", file="report.json", path="repro_thr", fmt="{:.4f}",
         backs="reliability threshold, resolved from the run's own null",
         script="endcell/ot/build_residual_targets.py"),
    dict(macro="FracReproducible", file="report.json",
         path="reliability_calibration.frac_above_null_p95", fmt="{:.0%}",
         backs="fraction of conditions whose residual reproduces at all",
         script="endcell/ot/build_residual_targets.py"),
    dict(macro="TrainConditions", file="report.json", path="holdout.n_train", fmt="{:d}",
         backs="training conditions after the repaired build",
         script="endcell/ot/build_residual_targets.py"),
    dict(macro="WellCrossing", file="report.json",
         path="well_crossing.frac_heldout_conditions_sharing_a_well_with_train", fmt="{:.1%}",
         backs="held-out conditions sharing a treated well with training data",
         script="endcell/ot/build_residual_targets.py"),
    # ---- the experimental unit ----------------------------------------------------------------
    dict(macro="CellLinesPerWell", file="experimental_unit_audit.json",
         path="cell_lines_per_sample.mean", fmt="{:.1f}",
         backs="the pseudoreplication factor",
         script="endcell/analysis/experimental_unit_audit.py"),
    dict(macro="FracTreatmentsReplicated", file="experimental_unit_audit.json",
         path="frac_treatments_replicated", fmt="{:.1%}",
         backs="independent-well replication -- why repro_cos is not biological reproducibility",
         script="endcell/analysis/experimental_unit_audit.py"),
    # ---- the metric audit ------------------------------------------------------------------
    dict(macro="NIRDRF", file="calibration.json", path="drf.neg_mean.nir.drf", fmt="{:+.3f}",
         backs="NIR's discrimination-recovery fraction",
         script="endcell/analysis/calibration_eval.py"),
    dict(macro="NIRDRFHolm", file="calibration.json", path="drf.neg_mean.nir.p_holm", fmt="{:.4f}",
         backs="NIR's Holm-adjusted p across the five-metric family",
         script="endcell/analysis/calibration_eval.py"),
]


# --------------------------------------------------------------------------- resolution
def dig(doc, dotted):
    """Resolve a dotted path, allowing literal dots inside a key (the T:: keys contain them)."""
    if dotted in doc:
        return doc[dotted]
    cur, parts = doc, dotted.split(".")
    i = 0
    while i < len(parts) and cur is not None:
        if isinstance(cur, list):
            if not parts[i].lstrip("-").isdigit():
                return None
            idx = int(parts[i])
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
            i += 1
            continue
        if not isinstance(cur, dict):
            return None
        for j in range(len(parts), i, -1):        # greedy: longest key first
            cand = ".".join(parts[i:j])
            if cand in cur:
                cur = cur[cand]
                i = j
                break
        else:
            return None
    return cur


def render(value, fmt):
    if value is None:
        return None
    if fmt == "len":
        return str(len(value)) if hasattr(value, "__len__") else None
    if fmt == "ci3":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            return None
        return f"[{value[0]:+.3f}, {value[1]:+.3f}]"
    try:
        return fmt.format(value)
    except (ValueError, TypeError, KeyError):
        return None


def _texify(s):
    """LaTeX-safe. % in particular is a comment character and would eat the rest of the line."""
    return (str(s).replace("\\", r"\textbackslash{}").replace("%", r"\%")
            .replace("_", r"\_").replace("&", r"\&").replace("#", r"\#"))


def _git_commit(root):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def resolve(results_dir, claims=None):
    claims = claims or CLAIMS
    cache, rows = {}, []
    for c in claims:
        p = os.path.join(results_dir, c["file"])
        if p not in cache:
            try:
                cache[p] = json.load(open(p))
            except (OSError, ValueError):
                cache[p] = None
        doc = cache[p]
        raw = dig(doc, c["path"]) if doc is not None else None
        val = render(raw, c["fmt"])
        rows.append({**c, "value": val, "raw": raw,
                     "status": ("ok" if val is not None
                                else ("no artifact" if doc is None else "path not found"))})
    return rows


def emit(rows, out_dir, commit):
    os.makedirs(out_dir, exist_ok=True)
    npath = os.path.join(out_dir, "numbers.tex")
    with open(npath, "w", encoding="utf-8") as fh:
        fh.write("% GENERATED by endcell/analysis/build_thesis_assets.py -- do not edit.\n")
        fh.write(f"% repository commit {commit}\n")
        fh.write("% A macro absent from this file is a number with no artifact. LaTeX will fail on\n")
        fh.write("% it, which is the intended behaviour: better a build error than a stale figure.\n")
        for r in rows:
            if r["value"] is not None:
                fh.write(f"\\newcommand{{\\{r['macro']}}}{{{_texify(r['value'])}}}\n")
    tpath = os.path.join(out_dir, "claim_artifact_table.tex")
    with open(tpath, "w", encoding="utf-8") as fh:
        fh.write("% GENERATED -- claim -> artifact -> script -> commit\n")
        fh.write("\\begin{tabular}{p{4.6cm}p{2.9cm}p{4.4cm}p{1.9cm}}\n\\hline\n")
        fh.write("Claim & Value & Artifact / script & Status \\\\\n\\hline\n")
        for r in sorted(rows, key=lambda x: (x["status"] != "ok", x["macro"])):
            fh.write(f"{_texify(r['backs'])} & {_texify(r['value'] or '--')} & "
                     f"\\texttt{{{_texify(r['file'])}}} \\newline "
                     f"\\texttt{{{_texify(os.path.basename(r['script']))}}} & "
                     f"{_texify(r['status'])} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
        fh.write(f"\n% repository commit {commit}\n")
    return npath, tpath


def report(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    logger.info("=" * 96)
    for r in rows:
        mark = "ok  " if r["status"] == "ok" else "MISS"
        logger.info(f"  {mark}  \\{r['macro']:28s} {str(r['value'] or '--'):>22s}   "
                    f"{r['file']}  ({r['status']})")
    logger.info(f"  {len(ok)}/{len(rows)} claims resolved to an artifact")
    missing = [r for r in rows if r["status"] != "ok"]
    if missing:
        logger.info("  UNRESOLVED -- each of these is a number that must not be quoted until its "
                    "artifact exists:")
        for r in missing:
            logger.info(f"      {r['macro']}: {r['backs']}  (wanted {r['file']}:{r['path']})")
    logger.info("=" * 96)
    return missing


# --------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = []

    def check(n, c):
        ok.append((n, bool(c)))

    d = tempfile.mkdtemp()
    res, out = os.path.join(d, "res"), os.path.join(d, "gen")
    os.makedirs(res)
    json.dump({"main": {"T::repro-filtered (cos>0.2, the training set)":
                        {"T": 0.5571, "ci_dyadic": [0.4903, 0.6239]},
                        "scope": {"residual_energy_share": 0.617, "cross_energy_share": -0.021}}},
              open(os.path.join(res, "vardecomp_matched.json"), "w"))
    json.dump({"repro_thr": 0.10857, "holdout": {"n_train": 2523},
               "reliability_calibration": {"frac_above_null_p95": 0.505},
               "well_crossing": {"frac_heldout_conditions_sharing_a_well_with_train": 0.0}},
              open(os.path.join(res, "report.json"), "w"))

    rows = resolve(res)
    by = {r["macro"]: r for r in rows}
    check("a dotted path resolves", by["ResidualEnergyShare"]["value"] == "61.7%")
    check("a key containing dots resolves", by["TransferCoefficient"]["value"] == "0.557")
    check("an interval renders as a pair", by["TransferCoefficientCI"]["value"] == "[+0.490, +0.624]")
    check("a signed format keeps its sign", by["CrossEnergyShare"]["value"] == "-0.021")
    check("an integer renders as one", by["TrainConditions"]["value"] == "2523")
    check("zero percent is not confused with missing",
          by["WellCrossing"]["value"] == "0.0%" and by["WellCrossing"]["status"] == "ok")
    check("a missing artifact is reported, not skipped",
          by["ChannelMoaGap"]["status"] == "no artifact" and by["ChannelMoaGap"]["value"] is None)
    check("a present artifact with a wrong path says so",
          resolve(res, [dict(macro="X", file="report.json", path="nope.nope", fmt="{:.2f}",
                             backs="b", script="s")])[0]["status"] == "path not found")

    npath, tpath = emit(rows, out, "deadbee")
    tex = open(npath, encoding="utf-8").read()
    check("resolved macros are emitted", "\\newcommand{\\TransferCoefficient}{0.557}" in tex)
    check("unresolved macros are NOT emitted, so LaTeX fails loudly",
          "\\ChannelMoaGap}" not in tex)
    check("percent is escaped, or it would comment out the line",
          r"{61.7\%}" in tex)
    check("the commit is recorded in the generated file", "deadbee" in tex)
    table = open(tpath, encoding="utf-8").read()
    check("the table lists unresolved claims too", "no artifact" in table)
    check("the table escapes underscores in filenames", r"vardecomp\_matched.json" in table)

    missing = report(rows)
    check("report returns exactly the unresolved claims",
          len(missing) == len([r for r in rows if r["status"] != "ok"]) and len(missing) > 0)

    for n, c in ok:
        logger.info(("  ok   " if c else "  FAIL ") + n)
    allok = all(c for _, c in ok)
    logger.info(f"SELFTEST {'PASSED' if allok else 'FAILED'}  ({sum(c for _, c in ok)}/{len(ok)})")
    if not allok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS_cluster")
    ap.add_argument("--out_dir", default="thesis/generated")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any claim has no artifact")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = resolve(a.results)
    missing = report(rows)
    npath, tpath = emit(rows, a.out_dir, _git_commit(root))
    logger.info(f"macros -> {npath}")
    logger.info(f"claim table -> {tpath}")
    logger.info(r"add \input{generated/numbers.tex} to the preamble")
    if a.strict and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

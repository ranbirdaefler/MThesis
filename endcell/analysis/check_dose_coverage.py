#!/usr/bin/env python
r"""
check_dose_coverage.py — did the dose-blind sampling cap (A-05) actually reduce dose diversity?
===============================================================================================
READ-ONLY diagnostic. The v2 preprocessor caps cells per (drug, cell_line, plate) — dose is NOT in
the key — and rows are streamed in storage order, so an early dose can in principle consume a
group's cap before later doses are seen. This script measures whether that happened in the ALREADY
GENERATED JSONLs. It does not re-preprocess, modify, or delete anything: it only streams the
existing files and counts.

The A-05 failure signature, made concrete:
  a (drug, cell_line, plate) group that (i) HIT the cap, (ii) contains only ONE dose, while
  (iii) that drug is multi-dose elsewhere in the data. Such a group *might* have lost later doses
  to the cap. We report how many groups match, and — the real downstream harm — whether the
  Tier-4 dose-interpolation split still has ≥2 train doses for each of its drugs.

USAGE
-----
  python check_dose_coverage.py --eval_dir DATA_endcell_big --cap 30 --out RESULTS/dose_coverage.json
  python check_dose_coverage.py --selftest         # synthetic, no data

If --cap is omitted it is inferred as the maximum observed group size (and flagged as inferred).
"""
import argparse, json, os, glob, sys, tempfile
from collections import defaultdict, Counter


def _meta(row):
    m = row.get("metadata", row)
    return m.get("drug"), m.get("cell_line_id"), m.get("plate"), m.get("dose_float", m.get("dose"))


def scan_file(path, per_group, drug_cl_doses, drug_doses, tier_doses, tier_name):
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            drug, cl, plate, dose = _meta(row)
            if drug is None:
                continue
            n += 1
            g = (drug, cl, plate)
            per_group[g]["n"] += 1
            per_group[g]["doses"][dose] += 1
            drug_cl_doses[(drug, cl)].add(dose)
            drug_doses[drug].add(dose)
            if tier_name:
                tier_doses[tier_name][drug].add(dose)
    return n


def analyze(per_group, drug_cl_doses, drug_doses, tier_doses, cap, cap_inferred):
    groups = list(per_group.values())
    n_groups = len(groups)
    cap_hit = [g for g in groups if g["n"] >= cap]
    multi_dose_drugs = {d for d, s in drug_doses.items() if len(s) > 1}

    # dose-count distribution per group (all groups, and cap-hit groups)
    dist_all = Counter(len(g["doses"]) for g in groups)
    dist_caphit = Counter(len(g["doses"]) for g in cap_hit)

    # A-05 AT-RISK groups: hit the cap, hold a single dose, but the drug is multi-dose elsewhere.
    # NOTE this is a design-confounded UPPER BOUND — a drug being multi-dose on OTHER plates is the
    # normal Tahoe design (one dose per plate/well), not proof this group lost a dose.
    at_risk = 0
    for (drug, cl, plate), g in per_group.items():
        if g["n"] >= cap and len(g["doses"]) == 1 and drug in multi_dose_drugs:
            at_risk += 1
    # The REAL ceiling on possible loss: how often a single (drug,cl,plate) even holds >1 dose. If
    # within-plate dose is near-degenerate, a dose-blind cap can harm at most that slice.
    n_multidose_groups = sum(1 for g in groups if len(g["doses"]) > 1)
    n_caphit_multidose = sum(1 for g in cap_hit if len(g["doses"]) > 1)

    # Tier-4 interpolation integrity: every tier4 drug should still have >=2 doses in TRAIN
    t4 = tier_doses.get("tier4_dose_interpolation", {})
    train = tier_doses.get("train", {})
    t4_drugs = list(t4.keys())
    t4_train_ok = sum(1 for d in t4_drugs if len(train.get(d, set())) >= 2)
    t4_train_single = [d for d in t4_drugs if len(train.get(d, set())) < 2]

    return {
        "cap": cap, "cap_inferred": cap_inferred,
        "n_groups": n_groups,
        "n_cap_hit": len(cap_hit),
        "frac_cap_hit": round(len(cap_hit) / max(1, n_groups), 4),
        "n_multi_dose_drugs": len(multi_dose_drugs),
        "n_total_drugs": len(drug_doses),
        "dose_count_per_group_all": dict(sorted(dist_all.items())),
        "dose_count_per_group_caphit": dict(sorted(dist_caphit.items())),
        "at_risk_groups": at_risk,
        "frac_caphit_at_risk": round(at_risk / max(1, len(cap_hit)), 4),
        "n_multidose_groups": n_multidose_groups,
        "frac_groups_multidose": round(n_multidose_groups / max(1, n_groups), 4),
        "n_caphit_multidose": n_caphit_multidose,
        "tier4_n_drugs": len(t4_drugs),
        "tier4_drugs_with_ge2_train_doses": t4_train_ok,
        "tier4_drugs_missing_train_doses": t4_train_single[:50],
    }


def _verdict(r):
    lines = []
    hit = r["frac_cap_hit"]
    md = r["frac_groups_multidose"]
    t4_ok = (r["tier4_n_drugs"] == 0) or (not r["tier4_drugs_missing_train_doses"])
    lines.append(f"cap={r['cap']}{' (INFERRED as max group size)' if r['cap_inferred'] else ''}  "
                 f"groups={r['n_groups']:,}  cap-hit={r['n_cap_hit']:,} ({hit:.1%})  "
                 f"multi-dose drugs={r['n_multi_dose_drugs']}/{r['n_total_drugs']}")
    lines.append(f"dose-count/group (all):     {r['dose_count_per_group_all']}")
    lines.append(f"dose-count/group (cap-hit): {r['dose_count_per_group_caphit']}")
    lines.append(f"within-plate multi-dose groups: {r['n_multidose_groups']:,} ({md:.1%} of all) "
                 f"<- CEILING on possible dose loss; a (drug,cl,plate) with one dose cannot lose one. "
                 f"{r['n_caphit_multidose']:,} multi-dose groups survived the cap.")
    lines.append(f"at-risk UPPER BOUND (design-confounded: single-dose cap-hit groups of drugs that are "
                 f"multi-dose ON OTHER plates = normal design, not loss): {r['at_risk_groups']:,}")
    if r["tier4_n_drugs"]:
        lines.append(f"Tier-4 interpolation: {r['tier4_drugs_with_ge2_train_doses']}/{r['tier4_n_drugs']} "
                     f"drugs still have >=2 train doses"
                     + ("  (all OK)" if t4_ok else f"  MISSING: {r['tier4_drugs_missing_train_doses']}"))
    if md < 0.05 and t4_ok:
        lines.append(f"VERDICT: A-05 does NOT materially bite. Within-plate dose is near-degenerate "
                     f"({md:.1%} multi-dose), so a dose-blind cap can touch at most that slice, and the only "
                     f"dose-dependent split (Tier-4) is fully intact. Record as a design caveat; the "
                     f"preprocessor now keys the cap on dose for FUTURE builds. No regeneration needed.")
    elif md < 0.20 and t4_ok:
        lines.append("VERDICT: minor - some within-plate multi-dose groups exist but Tier-4 is intact. "
                     "Footnote it; regenerate only for a dose-resolved analysis.")
    else:
        lines.append("VERDICT: material - within-plate multi-dose is common and/or Tier-4 lost doses. "
                     "Regenerate with the dose-aware cap key for any dose-dependent claim.")
    return "\n".join(lines)


def selftest():
    """Synthetic JSONLs where drug D1 is multi-dose but one (cl,plate) group is capped on a single
    dose -> the at-risk detector must flag exactly that group."""
    d = tempfile.mkdtemp()
    cap = 5
    # train: D1 on (clA, p1) all dose 1.0 and capped (5 rows); D1 on (clA, p2) doses 1.0 & 2.0;
    #        D1 also appears at dose 2.0 on p2 -> D1 is multi-dose. D2 single-dose everywhere.
    rows = []
    for _ in range(5):
        rows.append({"metadata": {"drug": "D1", "cell_line_id": "clA", "plate": "p1", "dose_float": 1.0}})
    for dose in (1.0, 2.0):
        for _ in range(2):
            rows.append({"metadata": {"drug": "D1", "cell_line_id": "clA", "plate": "p2", "dose_float": dose}})
    for _ in range(3):
        rows.append({"metadata": {"drug": "D2", "cell_line_id": "clA", "plate": "p1", "dose_float": 5.0}})
    with open(os.path.join(d, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # tier4: D1 needs interpolation; train has D1 at doses {1.0, 2.0} -> OK
    with open(os.path.join(d, "eval_tier4_dose_interpolation.jsonl"), "w") as f:
        f.write(json.dumps({"metadata": {"drug": "D1", "cell_line_id": "clB", "plate": "p9", "dose_float": 1.5}}) + "\n")

    per_group = defaultdict(lambda: {"n": 0, "doses": Counter()})
    drug_cl_doses, drug_doses = defaultdict(set), defaultdict(set)
    tier_doses = defaultdict(lambda: defaultdict(set))
    for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        base = os.path.basename(path)
        tname = "train" if base == "train.jsonl" else base.replace("eval_", "").replace(".jsonl", "")
        scan_file(path, per_group, drug_cl_doses, drug_doses, tier_doses, tname)
    r = analyze(per_group, drug_cl_doses, drug_doses, tier_doses, cap, False)
    print(_verdict(r))
    ok = (r["at_risk_groups"] == 1 and r["n_multi_dose_drugs"] == 1
          and r["tier4_drugs_with_ge2_train_doses"] == 1 and r["tier4_n_drugs"] == 1)
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--cap", type=int, default=None, help="cells_per_condition used at preprocess time; "
                                                          "inferred as max group size if omitted")
    ap.add_argument("--out", default="RESULTS/dose_coverage.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.eval_dir:
        ap.error("--eval_dir required (unless --selftest)")

    per_group = defaultdict(lambda: {"n": 0, "doses": Counter()})
    drug_cl_doses, drug_doses = defaultdict(set), defaultdict(set)
    tier_doses = defaultdict(lambda: defaultdict(set))
    files = sorted(glob.glob(os.path.join(args.eval_dir, "*.jsonl")))
    if not files:
        ap.error(f"no .jsonl files in {args.eval_dir}")
    total = 0
    for path in files:
        base = os.path.basename(path)
        tname = "train" if base == "train.jsonl" else base.replace("eval_", "").replace(".jsonl", "")
        n = scan_file(path, per_group, drug_cl_doses, drug_doses, tier_doses, tname)
        total += n
        print(f"  scanned {base}: {n:,} rows")
    cap = args.cap
    cap_inferred = cap is None
    if cap_inferred:
        cap = max((g["n"] for g in per_group.values()), default=1)
    r = analyze(per_group, drug_cl_doses, drug_doses, tier_doses, cap, cap_inferred)
    r["total_rows"] = total
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(r, open(args.out, "w"), indent=2, default=str)
    print("\n" + _verdict(r))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

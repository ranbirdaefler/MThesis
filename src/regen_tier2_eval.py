#!/usr/bin/env python
"""
regen_tier2_eval.py — regenerate the Tier-2 (unseen-drug) eval file with PER-DRUG
stratification, cheaply, without a full preprocessing rerun.

Why this exists
---------------
The original Tier-2 eval file was filled under a flat total cap, so it ended up
spanning only ~10 of the ~50 held-out drugs. This script rebuilds Tier-2 so every
held-out drug is represented (cap N cells per drug), which makes the effective
sample size of the unseen-drug clustered bootstrap = number of held-out drugs.

How it stays cheap and consistent
---------------------------------
- Controls are HARVESTED from the existing train.jsonl prompts (real plate-matched
  DMSO control sentences), so no DMSO rescan is needed. The held-out drugs share
  (cell_line, plate) contexts with non-held-out drugs that ARE in train, so a
  plate-matched control already exists for almost all of them.
- Response sentences are built with the SAME `build_panel_sentence` used in training
  (imported from tahoe_c2s_preprocess), so eval and train construction cannot diverge.
- The held-out drug set is reconstructed deterministically (same shards + seed 42),
  and then FILTERED against the drugs actually present in train.jsonl as a hard
  leak guard: no drug that appears in training can enter the unseen-drug eval.

Run on a CPU node (defq) via sbatch — it streams the treated shards twice
(observe + build), ~30-40 min, no GPU needed.
"""
import argparse
import json
import os
import logging
from collections import defaultdict

import numpy as np
from datasets import load_dataset

import tahoe_c2s_preprocess as prep  # reuse the EXACT builders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAHOE_REPO = "tahoebio/Tahoe-100M"
MAX_CTRL_PER_KEY = 8
MAX_CTRL_PER_CL = 16


def harvest_controls(train_path):
    """From train.jsonl: (cell_line_id, plate) -> [control_sentence], cell_line ->
    [control_sentence] (fallback), and the set of drugs present in train (leak guard)."""
    by_key = defaultdict(list)
    by_cl = defaultdict(list)
    train_drugs = set()
    n = 0
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            p = ex["prompt"]
            m = ex.get("metadata", {})
            cl, plate = m.get("cell_line_id"), m.get("plate")
            if m.get("drug"):
                train_drugs.add(m["drug"])
            if cl is None or plate is None or "\nControl cell: " not in p:
                continue
            ctrl = p.split("\nControl cell: ", 1)[1].split("\n\nResponse cell:", 1)[0]
            key = (cl, plate)
            if len(by_key[key]) < MAX_CTRL_PER_KEY:
                by_key[key].append(ctrl)
            if len(by_cl[cl]) < MAX_CTRL_PER_CL:
                by_cl[cl].append(ctrl)
            n += 1
    logger.info(f"  Harvested controls from {n:,} train rows: {len(by_key):,} "
                f"(cell_line,plate) keys, {len(by_cl):,} cell lines, "
                f"{len(train_drugs):,} train drugs")
    return by_key, by_cl, train_drugs


def iterate_shards(shard_list, per_shard_cap=None):
    for shard in shard_list:
        url = f"hf://datasets/{TAHOE_REPO}/{shard}"
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        sc = 0
        for row in ds:
            yield row
            sc += 1
            if per_shard_cap and sc >= per_shard_cap:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Dataset dir (must contain train.jsonl + l1000_panel.json).")
    ap.add_argument("--num_shards", type=int, default=32)
    ap.add_argument("--rows_per_shard", type=int, default=400000)
    ap.add_argument("--shard_seed", type=int, default=7)
    ap.add_argument("--held_out_drugs", type=int, default=50)
    ap.add_argument("--max_eval_per_tier", type=int, default=5000)
    ap.add_argument("--cells_per_condition", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="Output jsonl (default: overwrite eval_tier2_unseen_drugs.jsonl "
                         "in data_dir, backing up the old one to .bak).")
    args = ap.parse_args()

    panel = json.load(open(os.path.join(args.data_dir, "l1000_panel.json")))
    pidx = {g: i for i, g in enumerate(panel)}
    gene_id_to_symbol, sample_to_conc, drug_info, cvcl_to_name = prep.load_metadata()

    by_key, by_cl, train_drugs = harvest_controls(os.path.join(args.data_dir, "train.jsonl"))

    all_shards = prep.discover_expression_shards()
    treated = prep.select_shards(all_shards, args.num_shards, args.shard_seed)
    logger.info(f"Treated shards: {len(treated)} (seed {args.shard_seed})")

    # --- OBSERVE pass: same shards -> same observed drugs -> same held-out set ---
    logger.info("Observe pass (reconstruct held-out drug set)...")
    observed = set()
    for row in iterate_shards(treated, per_shard_cap=args.rows_per_shard):
        d = row["drug"]
        if d != "DMSO_TF" and d != "DMSO":
            observed.add(d)
    observed_list = sorted(observed)
    np.random.seed(42)
    n_hold = min(args.held_out_drugs, max(1, len(observed_list) // 5))
    held = set(np.random.choice(observed_list, size=n_hold, replace=False)) \
        if observed_list and n_hold > 0 else set()

    # HARD LEAK GUARD: a held-out (unseen) drug must not appear in train.
    leaked = held & train_drugs
    if leaked:
        logger.warning(f"  {len(leaked)} reconstructed held-out drugs are present in train "
                       f"(determinism drift) — dropping them from the eval set: "
                       f"{sorted(leaked)[:5]}...")
        held = held - train_drugs
    logger.info(f"Reconstructed {len(held)} held-out drugs of {len(observed_list)} observed "
                f"(all disjoint from the {len(train_drugs)} train drugs)")

    cap = max(1, args.max_eval_per_tier // max(1, len(held)))
    logger.info(f"Per-drug cap: {cap} (target up to {cap * len(held):,} across {len(held)} drugs)")

    # --- BUILD pass: only held-out-drug cells, per-drug + per-condition capped ---
    logger.info("Build pass (Tier-2 examples)...")
    cnt = defaultdict(int)
    cond = defaultdict(int)
    tier2 = []
    total = qc = noctrl = 0
    np.random.seed(0)  # reproducible control draw
    for row in iterate_shards(treated, per_shard_cap=args.rows_per_shard):
        drug = row["drug"]
        if drug not in held or cnt[drug] >= cap:
            continue
        cl, plate, sample = row["cell_line_id"], row["plate"], row["sample"]
        ckey = (drug, cl, plate)
        if cond[ckey] >= args.cells_per_condition:
            continue
        pool = by_key.get((cl, plate))
        plate_matched = True
        if not pool:
            pool = by_cl.get(cl)
            plate_matched = False
        if not pool:
            noctrl += 1
            continue
        ctrl_sent = pool[np.random.randint(len(pool))]
        resp = prep.build_panel_sentence(
            row["genes"], row["expressions"], gene_id_to_symbol, panel, pidx,
            min_expressed=200, min_panel_expressed=50,
        )
        if resp is None:
            qc += 1
            continue
        cell_line_name = cvcl_to_name.get(cl, cl)
        dose_str = prep.parse_dose(sample_to_conc.get(sample, "unknown"))
        moa = row.get("moa-fine", drug_info.get(drug, {}).get("moa", "unknown"))
        try:
            dose_float = float(dose_str.split()[0])
        except Exception:
            dose_float = None
        prompt = prep.format_prompt(cell_line_name, drug, dose_str, moa, ctrl_sent)
        tier2.append({
            "prompt": prompt,
            "response": resp,
            "metadata": {
                "drug": drug, "cell_line_id": cl, "cell_line_name": cell_line_name,
                "plate": plate, "sample": sample, "dose": dose_str,
                "dose_float": dose_float, "moa": moa,
                "control_plate_matched": plate_matched,
            },
        })
        cnt[drug] += 1
        cond[ckey] += 1
        total += 1
        if total % 1000 == 0:
            logger.info(f"  built {total:,} tier-2 across {len(cnt)} drugs")

    out = args.out or os.path.join(args.data_dir, "eval_tier2_unseen_drugs.jsonl")
    if args.out is None and os.path.exists(out):
        os.rename(out, out + ".bak")
        logger.info(f"  backed up old tier-2 -> {out}.bak")
    with open(out, "w") as f:
        for ex in tier2:
            f.write(json.dumps(ex) + "\n")
    with open(os.path.join(args.data_dir, "held_out_drugs.json"), "w") as f:
        json.dump(sorted(held), f, indent=2)

    logger.info(f"\nWrote {len(tier2):,} Tier-2 examples across {len(cnt)}/{len(held)} "
                f"held-out drugs (qc_fail={qc}, no_control={noctrl})")
    logger.info(f"  -> {out}")
    fb = sum(1 for e in tier2 if not e['metadata']['control_plate_matched'])
    logger.info(f"  control_plate_matched: {len(tier2)-fb}/{len(tier2)} "
                f"({100.0*(len(tier2)-fb)/max(1,len(tier2)):.1f}%)")


if __name__ == "__main__":
    main()

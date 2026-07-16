#!/usr/bin/env python
r"""
make_scramble_endcell.py
========================
Build a scrambled-drug version of the [END_CELL] eval tiers for the scramble ablation.

For each eval example, replace the DRUG and MECHANISM in the prompt with those of a DIFFERENT,
different-mechanism drug, while keeping the control cell and the ground-truth response UNCHANGED.
The model is thus asked to predict the (real) drug-A response but told it is a different drug. If
the model uses the drug, its prediction should move toward the wrong drug and score worse against
truth_A; if it ignores the drug, the scrambled prediction ~ the real prediction.

We swap to a drug whose MOA differs from the original (a "different-mechanism" scramble), matching
the earlier scramble ablation. MOA is read from metadata if present; otherwise we swap to any
different drug (still a valid negative, just not MOA-controlled).

Determinism: seeded, so the scramble is reproducible.

USAGE
-----
  python make_scramble_endcell.py \
     --in_dir  /data/.../data_diverse2_endcell_big \
     --out_dir /data/.../data_diverse2_endcell_big_scram \
     --tiers tier1_seen_conditions,tier2_unseen_drugs --seed 42
"""
import argparse, json, os, re, logging
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# The prompt template (from preprocessing):
#   "Predict the response of {cell_line} to {drug} at {dose}. Mechanism: {moa}\nControl cell: {ctrl}\n\nResponse cell:"
# We replace the "{drug}" and "Mechanism: {moa}" spans using the metadata, which is the robust way
# (string surgery on the prompt is brittle). We rebuild the prompt prefix from metadata + keep the
# control-cell text verbatim.

CTRL_MARKER = "Control cell:"


def split_prompt(prompt):
    """Split into (prefix_before_control, control_and_after). Keep everything from 'Control cell:'
    onward verbatim (that's the control sentence + the trailing 'Response cell:')."""
    idx = prompt.find(CTRL_MARKER)
    if idx == -1:
        return None, None
    return prompt[:idx], prompt[idx:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tiers", default="tier1_seen_conditions,tier2_unseen_drugs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.abspath(args.out_dir) == os.path.abspath(args.in_dir):
        raise SystemExit("out_dir must differ from in_dir")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    for tier in [t.strip() for t in args.tiers.split(",")]:
        path = os.path.join(args.in_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"missing {path}"); continue
        examples = [json.loads(l) for l in open(path)]

        # catalog drugs -> their MOA, and MOA -> drugs, from metadata
        drug_moa = {}
        moa_drugs = defaultdict(set)
        for e in examples:
            m = e.get("metadata", {})
            d, moa = m.get("drug"), m.get("moa") or m.get("mechanism") or None
            if d is not None:
                drug_moa[d] = moa
                moa_drugs[moa].add(d)
        all_drugs = sorted(drug_moa.keys())
        logger.info(f"{tier}: {len(examples)} examples, {len(all_drugs)} drugs, "
                    f"{len(moa_drugs)} distinct MOAs")

        def pick_scramble_drug(orig_drug):
            orig_moa = drug_moa.get(orig_drug)
            # prefer a drug with a DIFFERENT moa
            diff_moa_drugs = [d for d in all_drugs
                              if d != orig_drug and drug_moa.get(d) != orig_moa]
            pool = diff_moa_drugs if diff_moa_drugs else [d for d in all_drugs if d != orig_drug]
            return rng.choice(pool) if pool else orig_drug

        out_path = os.path.join(args.out_dir, f"eval_{tier}.jsonl")
        n_ok, n_skip = 0, 0
        with open(out_path, "w") as out:
            for e in examples:
                m = e.get("metadata", {})
                orig_drug = m.get("drug")
                prefix, ctrl_and_after = split_prompt(e["prompt"])
                if prefix is None or orig_drug is None:
                    n_skip += 1; continue
                new_drug = pick_scramble_drug(orig_drug)
                new_moa = drug_moa.get(new_drug)
                # rebuild the prefix: replace the drug name and the mechanism text.
                # We do targeted replacement on the prefix using the known metadata strings.
                new_prefix = prefix
                # replace drug (first occurrence of the exact original drug string)
                if orig_drug in new_prefix:
                    new_prefix = new_prefix.replace(orig_drug, new_drug, 1)
                else:
                    n_skip += 1; continue
                # replace mechanism after "Mechanism:" up to end-of-line, if present
                if new_moa is not None:
                    new_prefix = re.sub(r"(Mechanism:\s*)([^\n]*)", r"\1" + new_moa.replace("\\", "\\\\"),
                                        new_prefix, count=1)
                new_ex = dict(e)
                new_ex["prompt"] = new_prefix + ctrl_and_after
                # keep response (truth) and metadata UNCHANGED, but record the scramble
                new_ex["metadata"] = dict(m)
                new_ex["metadata"]["scrambled_from_drug"] = orig_drug
                new_ex["metadata"]["scrambled_to_drug"] = new_drug
                # IMPORTANT: metadata['drug'] stays the ORIGINAL so grouping/truth still keys on drug A
                out.write(json.dumps(new_ex) + "\n")
                n_ok += 1
        logger.info(f"  wrote {out_path}: {n_ok} scrambled, {n_skip} skipped")

    logger.info("done.")


if __name__ == "__main__":
    main()

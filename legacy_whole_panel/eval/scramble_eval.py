#!/usr/bin/env python
"""
scramble_eval.py — build a drug-scrambled copy of the eval tiers for the
drug-specificity ablation.

For each eval example, the drug name + MOA in the PROMPT are replaced with a
different drug/MOA sampled from the training pool; the control cell, cell-line,
dose, the RESPONSE (truth), and all metadata stay unchanged. If the model is
drug-specific, feeding the wrong drug should degrade DE-Δr against the unchanged
truth; if the model ignores the drug token, DE-Δr is unchanged.

Two modes (run both as separate dirs):
  --mode diff_moa   replacement drug has a DIFFERENT MOA (hardest, cleanest test)
  --mode rand_drug  replacement is any different training drug (may share MOA)

The scrambled directory is self-contained: scrambled eval_tier*.jsonl files are
written, and every other file in the source eval_dir (panel, linear_model, gene
maps, train.jsonl, ...) is symlinked, so the eval harness can point --eval_dir
straight at it.

Usage:
  python scramble_eval.py --eval_dir DATA --out_dir DATA_scram_diffmoa \
      --train_file DATA/train.jsonl --mode diff_moa --seed 42
"""
import argparse, json, os, re, random, glob

TIERS = ["tier1_seen_conditions", "tier2_unseen_drugs",
         "tier3_unseen_combos", "tier4_dose_interpolation"]


def norm_moa(moa):
    moa = moa or "unclear"
    if moa in ("unknown", "nan", "None", "", None):
        moa = "unclear"
    return moa


def build_drug_pool(train_file, limit=200000):
    """Unique (drug, normalized_moa) pairs seen in training, with the set of MOAs."""
    pool = {}
    n = 0
    with open(train_file) as f:
        for line in f:
            if n >= limit:
                break
            m = json.loads(line).get("metadata", {})
            drug = m.get("drug")
            if drug:
                pool[drug] = norm_moa(m.get("moa"))
            n += 1
    pairs = [(d, mo) for d, mo in pool.items()]
    return pairs


def scramble_prompt(prompt, orig_drug, orig_dose, orig_moa, new_drug, new_moa):
    """Replace 'to {drug} at {dose}. Mechanism: {moa}.' with the new drug/moa.
    Robust: try exact-segment replace first, then a regex fallback. Returns
    (new_prompt, ok)."""
    orig_seg = f"to {orig_drug} at {orig_dose}. Mechanism: {orig_moa}."
    new_seg = f"to {new_drug} at {orig_dose}. Mechanism: {new_moa}."
    if orig_seg in prompt:
        return prompt.replace(orig_seg, new_seg, 1), True
    # fallback: regex on the fixed template skeleton
    pat = re.compile(r"to (.+?) at (.+?)\. Mechanism: (.+?)\.")
    def _sub(mobj):
        return f"to {new_drug} at {mobj.group(2)}. Mechanism: {new_moa}."
    new_prompt, n = pat.subn(_sub, prompt, count=1)
    return new_prompt, (n == 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--mode", choices=["diff_moa", "rand_drug"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = build_drug_pool(args.train_file)
    moas = sorted({mo for _, mo in pool})
    print(f"Training pool: {len(pool)} unique drugs, {len(moas)} MOA classes")

    os.makedirs(args.out_dir, exist_ok=True)

    # symlink every non-tier file so the harness finds panel/linear_model/etc.
    for src in glob.glob(os.path.join(args.eval_dir, "*")):
        base = os.path.basename(src)
        if base.startswith("eval_tier"):
            continue
        dst = os.path.join(args.out_dir, base)
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                pass

    summary = {}
    for tier in TIERS:
        src = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(src):
            continue
        out = os.path.join(args.out_dir, f"eval_{tier}.jsonl")
        n_ok = n_fail = 0
        same_moa_count = 0
        with open(src) as fin, open(out, "w") as fout:
            for line in fin:
                ex = json.loads(line)
                m = ex.get("metadata", {})
                od, omoa = m.get("drug"), norm_moa(m.get("moa"))
                odose = m.get("dose", "unknown")

                # choose replacement
                if args.mode == "diff_moa":
                    cands = [(d, mo) for d, mo in pool if mo != omoa and d != od]
                else:  # rand_drug
                    cands = [(d, mo) for d, mo in pool if d != od]
                if not cands:
                    fout.write(line); n_fail += 1; continue
                nd, nmoa = rng.choice(cands)
                if nmoa == omoa:
                    same_moa_count += 1

                new_prompt, ok = scramble_prompt(ex["prompt"], od, odose, omoa, nd, nmoa)
                if not ok:
                    fout.write(line); n_fail += 1; continue
                ex["prompt"] = new_prompt
                # record what we injected; KEEP original drug/moa in metadata so
                # truth + DE-gene selection are unaffected.
                ex.setdefault("scramble", {})
                ex["scramble"] = {"orig_drug": od, "orig_moa": omoa,
                                  "scram_drug": nd, "scram_moa": nmoa, "mode": args.mode}
                fout.write(json.dumps(ex) + "\n")
                n_ok += 1
        summary[tier] = {"ok": n_ok, "fail": n_fail, "same_moa_injected": same_moa_count}
        print(f"  {tier}: scrambled {n_ok}, fallback-failed {n_fail}, "
              f"accidental same-MOA {same_moa_count}")

    with open(os.path.join(args.out_dir, "scramble_summary.json"), "w") as f:
        json.dump({"mode": args.mode, "seed": args.seed, "tiers": summary}, f, indent=2)
    print(f"Wrote scrambled eval to {args.out_dir}")
    if any(v["fail"] for v in summary.values()):
        print("WARNING: some prompts could not be scrambled (left unchanged) — check template match.")


if __name__ == "__main__":
    main()

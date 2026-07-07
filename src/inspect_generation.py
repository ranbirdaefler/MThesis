#!/usr/bin/env python
"""
inspect_generation.py — print the ACTUAL generated text from one or two models on a
handful of eval examples, side by side, so you can see WHAT a model emits (the eval
harness only stores metrics, not text). Useful for showing that the base model fails
the prompt format while the fine-tuned model emits a proper panel sentence.

Run as a short GPU job (loads the model(s) and generates a few examples).
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path, bf16=True, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if bf16 else torch.float32
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt)
    except TypeError:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dt)
    return tok, mdl.to(device).eval()


def gen(tok, mdl, prompt, max_new_tokens, device="cuda"):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        out = mdl.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def fmt_stats(text, panel_set):
    toks = text.split()
    if not toks:
        return "empty"
    emp = [t for t in toks if t in panel_set]
    uniq = set(emp)
    return (f"len={len(toks)} coverage={len(uniq)/len(panel_set):.2f} "
            f"halluc={(len(toks)-len(emp))/len(toks):.2f} valid_genes={len(uniq)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--models", nargs="+", required=True,
                    help="One or more model paths/labels, e.g. FT=/path/ckpt BASE=repo_id "
                         "(use LABEL=PATH or just PATH).")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=3800)
    ap.add_argument("--preview_genes", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_set = set(panel)

    examples = []
    with open(os.path.join(args.eval_dir, f"eval_{args.tier}.jsonl")) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= args.n:
                break

    parsed = []
    for spec in args.models:
        label, path = (spec.split("=", 1) if "=" in spec else (spec, spec))
        parsed.append((label, path))

    for label, path in parsed:
        print(f"\n{'#'*70}\n# MODEL: {label}  ({path})\n{'#'*70}")
        tok, mdl = load(path, device=device)
        for i, ex in enumerate(examples):
            meta = ex.get("metadata", {})
            print(f"\n--- example {i}: {meta.get('drug')} / {meta.get('cell_line_name')} "
                  f"@ {meta.get('dose')} ---")
            true_preview = " ".join(ex["response"].split()[:args.preview_genes])
            print(f"  TRUE (first {args.preview_genes}): {true_preview} ...")
            g = gen(tok, mdl, ex["prompt"], args.max_new_tokens, device)
            gen_preview = " ".join(g.split()[:args.preview_genes])
            print(f"  GEN  (first {args.preview_genes}): {gen_preview} ...")
            print(f"  GEN stats: {fmt_stats(g, panel_set)}")
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

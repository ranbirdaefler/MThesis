#!/usr/bin/env python
r"""
causal_drug_probe.py
====================
Turns "the drug is read into the representation but not used" from correlation into CAUSATION.

The mechanistic probe shows drug identity is linearly decodable from the residual stream (~76-82%).
The output-invariance result shows swapping the drug in the prompt does not change the output. Those
together IMPLY "present but not used" — but a reviewer can object that linear decodability is not
functional use. This script tests it causally by STEERING: we take the drug-A−drug-B direction in
activation space and ADD it into a drug-B forward pass at layer L. If the model uses the drug, the
generated cell sentence should move toward drug-A's native output; if the drug is causally inert for
generation, the output stays at drug-B (up to sampling noise) no matter how hard we push.

Why steering (not positional patching): drug names tokenize to different lengths, so patching a
token span misaligns everything after it. A steering vector added at the last prompt position (and
each generated position) sidesteps alignment entirely and asks the cleaner question — is generation
sensitive to movement along the drug axis?

CONTROLS (so a null is interpretable):
  * RANDOM-direction steer at matched norm  -> should perturb the output (proves the hook works and
    the layer is upstream of generation). If even this does nothing, the hook/layer is wrong.
  * SAMPLING-noise floor (two temperature samples of the same drug) -> the yardstick: a real causal
    effect must exceed this.

READOUT per (layer, scale), averaged over drug pairs/contexts:
  * effect_toward_A = topn_tau(steered, A_native) − topn_tau(steered, B_native)   (>0 => drug used)
  * output_change_drug = 1 − topn_tau(steered_drug, B_native)   (how far drug-steer moved the output)
  * output_change_rand = 1 − topn_tau(steered_rand, B_native)   (hook/positive control)
  * noise_floor        = 1 − topn_tau(B_sample1, B_sample2)      (sampling yardstick)
Interpretation: output_change_rand >> noise_floor (hook works) AND effect_toward_A ≈ 0 AND
output_change_drug ≈ noise_floor  =>  the drug direction is causally inert for generation.

USAGE (GPU)
  python causal_drug_probe.py --eval_dir DATA_endcell_big --model_path CKPT_endcell/final \
     --tier tier2_unseen_drugs --n_contexts 4 --n_pairs 8 --layers 4,8,12 --scales 1,4,8 \
     --temperature 0.8 --top_p 0.9 --topn 100 --max_new_tokens 1200 --bf16 \
     --out RESULTS/causal_drug_probe.json --seed 42

SELFTEST (no model/data) — validates the readout logic on synthetic generations
  python causal_drug_probe.py --selftest --out /tmp/causal_selftest.json
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

# --- repo path bootstrap (reorg): make shared/ + sibling pipeline dirs importable ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
for _p in [os.path.join(_ROOT, "shared"), *sorted(glob.glob(os.path.join(_PIPE, "*")))]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"

# module-level steering state, toggled around model.generate; the per-layer hooks read it
_STEER = {"layer_idx": None, "vec": None}


def _genes(s):
    out = []
    for t in s.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def topn_tau(a, b, topn):
    import evaluate_c2s_tahoe as ev
    ra = ev.cell_sentence_to_gene_ranks(" ".join(_genes(a)))
    rb = ev.cell_sentence_to_gene_ranks(" ".join(_genes(b)))
    if len(ra) < 3 or len(rb) < 3:
        return None
    top = sorted(ra, key=lambda g: ra[g])[:topn]
    res = ev.compute_rank_correlation(rb, ra, gene_subset=top)
    return res.get("kendall_tau") if isinstance(res, dict) else None


# ----------------------------------------------------------------- readout aggregation (selftestable)
def readout(gens, contexts, topn):
    """gens keyed by tags; see build in run(). Returns per-(layer,scale) aggregate readout."""
    agg = defaultdict(lambda: defaultdict(list))
    noise = []
    for ci, ctx in enumerate(contexts):
        for (A, B) in ctx["pairs"]:
            a_nat = gens.get((ci, "native", A, 0))
            b_nat = gens.get((ci, "native", B, 0))
            b_nat2 = gens.get((ci, "native", B, 1))
            if a_nat is None or b_nat is None:
                continue
            if b_nat2 is not None:
                nf = topn_tau(b_nat, b_nat2, topn)
                if nf is not None:
                    noise.append(1 - nf)
            for (kind, L, s) in ctx["configs"]:
                g = gens.get((ci, kind, (A, B), (L, s)))
                if g is None:
                    continue
                t_a = topn_tau(g, a_nat, topn)
                t_b = topn_tau(g, b_nat, topn)
                if t_a is None or t_b is None:
                    continue
                key = (L, s)
                if kind == "drug":
                    agg[key]["effect_toward_A"].append(t_a - t_b)
                    agg[key]["output_change_drug"].append(1 - t_b)
                elif kind == "rand":
                    agg[key]["output_change_rand"].append(1 - t_b)
    out = {"noise_floor": float(np.mean(noise)) if noise else None, "by_layer_scale": {}}
    for (L, s), d in agg.items():
        out["by_layer_scale"][f"L{L}_s{s}"] = {
            k: (float(np.mean(v)) if v else None) for k, v in d.items()}
    return out


# ----------------------------------------------------------------- context construction
def build_contexts(examples, n_contexts, n_pairs, layers, scales, rng):
    import evaluate_c2s_tahoe as ev
    by_cl = defaultdict(list)
    for e in examples:
        cl = e.get("metadata", {}).get("cell_line_id")
        if cl is not None:
            by_cl[cl].append(e)
    cls = sorted(by_cl, key=lambda cl: -len({e["metadata"]["drug"] for e in by_cl[cl]}))
    contexts = []
    for cl in cls:
        if len(contexts) >= n_contexts:
            break
        exs = by_cl[cl]
        drug_header = {}
        for e in exs:
            d = e["metadata"]["drug"]
            drug_header.setdefault(d, e["prompt"].split("\nControl cell:", 1)[0])
        if len(drug_header) < 4:
            continue
        ctrl = ""
        for e in exs:
            ctrl = ev.control_from_prompt(e["prompt"])
            if ctrl:
                break
        if not ctrl:
            continue
        drugs = list(drug_header)
        rng.shuffle(drugs)
        pairs = [(drugs[2 * i], drugs[2 * i + 1]) for i in range(min(n_pairs, len(drugs) // 2))]
        prompts = {d: f"{drug_header[d]}\nControl cell: {ctrl}\n\nResponse cell:" for d in drug_header}
        configs = [("drug", L, s) for L in layers for s in scales] + \
                  [("rand", L, s) for L in layers for s in scales]
        contexts.append({"cell_line": str(cl), "pairs": pairs, "prompts": prompts, "configs": configs})
    logger.info(f"  built {len(contexts)} contexts")
    return contexts


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic readout check. Two worlds:
      * drug-USED:   steering toward A makes the output look like A (effect_toward_A > 0)
      * drug-INERT:  steering does nothing beyond the random/noise controls (effect ~ 0)
    Confirms the aggregation distinguishes them."""
    rng = np.random.RandomState(0)
    P = 300
    panel = [f"G{i}" for i in range(P)]

    def sent(pool_seed, draw_seed):
        pool = np.random.RandomState(pool_seed).choice(P, 120, replace=False)
        base = np.random.RandomState(draw_seed).choice(pool, 90, replace=False)
        base = sorted(base)  # consistent gene order so topn_tau (a rank metric) is sensitive,
        #                      as real model outputs are (expression order); synthetic sets alone aren't
        return " ".join(panel[i] for i in base) + " " + SENTINEL

    def world(mode):
        contexts = []
        gens = {}
        call = 1000
        for ci in range(3):
            drugs = [f"d{k}" for k in range(8)]
            pairs = [(drugs[2 * i], drugs[2 * i + 1]) for i in range(4)]
            configs = [("drug", 8, 4), ("rand", 8, 4)]
            contexts.append({"cell_line": f"cl{ci}", "pairs": pairs, "configs": configs})
            for d in drugs:
                gens[(ci, "native", d, 0)] = sent(hash((ci, d)) % 99991, call); call += 1
                gens[(ci, "native", d, 1)] = sent(hash((ci, d)) % 99991, call); call += 1  # resample
            for (A, B) in pairs:
                # random steer: pushes output to a garbled pool (changes output, not toward A)
                gens[(ci, "rand", (A, B), (8, 4))] = sent(hash((ci, "rand", A, B)) % 99991, call); call += 1
                if mode == "used":
                    # drug steer moves output to A's pool
                    gens[(ci, "drug", (A, B), (8, 4))] = sent(hash((ci, A)) % 99991, call); call += 1
                else:  # inert: drug steer stays at B's pool
                    gens[(ci, "drug", (A, B), (8, 4))] = sent(hash((ci, B)) % 99991, call); call += 1
        return readout(gens, contexts, args.topn)

    used = world("used")
    inert = world("inert")
    eu = used["by_layer_scale"]["L8_s4"]["effect_toward_A"]
    ei = inert["by_layer_scale"]["L8_s4"]["effect_toward_A"]
    logger.info(f"  drug-USED  effect_toward_A = {eu:+.3f} (steer -> A: expect strongly > 0)")
    logger.info(f"  drug-INERT effect_toward_A = {ei:+.3f} (steer -> stays B: expect < USED)")
    logger.info(f"  INERT output_change_rand={inert['by_layer_scale']['L8_s4']['output_change_rand']:.3f} "
                f"vs noise_floor={inert['noise_floor']:.3f} (rand should exceed noise)")
    # the readout must SEPARATE the two worlds; the exact null value depends on the noise regime
    ok = (eu > 0.1) and (eu > ei + 0.2) and \
         (inert["by_layer_scale"]["L8_s4"]["output_change_rand"] > inert["noise_floor"])
    out = {"selftest": True, "passed": bool(ok), "used": used, "inert": inert}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def _find_layers(model):
    """Return the ModuleList of transformer blocks for GPTNeoX/Pythia (or a best-effort fallback)."""
    for attr in ("gpt_neox", "transformer", "model"):
        base = getattr(model, attr, None)
        if base is not None:
            for lname in ("layers", "h", "blocks"):
                layers = getattr(base, lname, None)
                if layers is not None:
                    return layers
    raise RuntimeError("Could not locate transformer layers for steering hooks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--n_contexts", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--scales", default="1,2,4",
                    help="steering magnitude as a FRACTION of the residual-stream norm at the layer "
                         "(s=1 adds a vector as large as the residual; strong enough to move the output)")
    ap.add_argument("--do_sample", action="store_true",
                    help="sample instead of greedy; leave OFF — greedy gives a deterministic baseline "
                         "so steering effects aren't drowned by sampling noise")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    import torch
    import evaluate_c2s_tahoe as ev
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.RandomState(args.seed)
    layers = [int(x) for x in args.layers.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    examples = [json.loads(l) for l in open(os.path.join(args.eval_dir, f"eval_{args.tier}.jsonl"))]
    contexts = build_contexts(examples, args.n_contexts, args.n_pairs, layers, scales, rng)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()
    blocks = _find_layers(model)
    logger.info(f"  {len(blocks)} transformer blocks; steering layers {layers}")

    # register a steering hook on every block; each adds _STEER['vec'] at the last position
    def make_hook(idx):
        def hook(module, inp, out):
            if _STEER["layer_idx"] != idx or _STEER["vec"] is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            hs[:, -1, :] = hs[:, -1, :] + _STEER["vec"].to(hs.dtype)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook
    for i, blk in enumerate(blocks):
        blk.register_forward_hook(make_hook(i))

    def last_pos_acts(prompt):
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        # hidden_states[i+1] == output of block i
        return {i: o.hidden_states[i + 1][0, -1, :].float().cpu().numpy() for i in range(len(blocks))}

    ec = tok.encode(SENTINEL, add_special_tokens=False)
    end_cell_id = ec[0] if len(ec) == 1 else tok.convert_tokens_to_ids(SENTINEL)
    eos = [end_cell_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    logger.info(f"  [END_CELL] id {end_cell_id}; generation stops at it")

    def _run_gen(prompt):
        """Generation, stopping at [END_CELL], truncated at the token level. Respects the
        globally-registered steering hooks via _STEER. GREEDY by default (--do_sample off): a
        deterministic baseline means the noise floor is ~0 and any steering-induced change is cleanly
        causal — temperature sampling swamps the effect (resamples of one drug agree at only ~0.2)."""
        prev = tok.padding_side
        tok.padding_side = "left"
        kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        if args.do_sample:
            kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            kw.update(do_sample=False)
        try:
            enc = tok([prompt], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**enc, **kw)
            ids = out[0][enc["input_ids"].shape[1]:].tolist()
            if end_cell_id in ids:
                ids = ids[:ids.index(end_cell_id)]
            return tok.decode(ids, skip_special_tokens=True).strip()
        finally:
            tok.padding_side = prev

    def gen(prompt):
        _STEER["layer_idx"] = None
        _STEER["vec"] = None
        return _run_gen(prompt)

    def gen_steer(prompt, idx, vec):
        _STEER["layer_idx"] = idx
        _STEER["vec"] = torch.tensor(vec, device=device)
        try:
            return _run_gen(prompt)
        finally:
            _STEER["layer_idx"] = None
            _STEER["vec"] = None

    gens = {}
    for ci, ctx in enumerate(contexts):
        logger.info(f"  context {ci} ({ctx['cell_line']}): {len(ctx['pairs'])} pairs")
        acts = {d: last_pos_acts(ctx["prompts"][d]) for d in ctx["prompts"]}
        # native generations (2 samples each drug for the noise floor)
        drugs_used = {d for pair in ctx["pairs"] for d in pair}
        for d in drugs_used:
            gens[(ci, "native", d, 0)] = gen(ctx["prompts"][d])
            gens[(ci, "native", d, 1)] = gen(ctx["prompts"][d])
        for (A, B) in ctx["pairs"]:
            for L in layers:
                diff = acts[A][L] - acts[B][L]
                unit_drug = diff / (np.linalg.norm(diff) + 1e-8)
                rnd = rng.randn(*diff.shape).astype(np.float32)
                unit_rand = rnd / (np.linalg.norm(rnd) + 1e-8)
                # Scale RELATIVE TO THE RESIDUAL NORM at the steer point (not the tiny drug-diff
                # norm — that was too weak in run 583489, so even random steering didn't move the
                # output past sampling noise). s is now a fraction of ‖residual‖: s>=1 is a large,
                # output-changing intervention, which lets the rand control clear the noise floor.
                resid_norm = float(np.linalg.norm(acts[B][L]))
                for s in scales:
                    mag = s * resid_norm
                    gens[(ci, "drug", (A, B), (L, s))] = gen_steer(ctx["prompts"][B], L, mag * unit_drug)
                    gens[(ci, "rand", (A, B), (L, s))] = gen_steer(ctx["prompts"][B], L, mag * unit_rand)

    result = readout(gens, contexts, args.topn)
    result["config"] = {k: v for k, v in vars(args).items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)

    logger.info("")
    logger.info("=" * 100)
    logger.info(f"  CAUSAL DRUG STEERING — noise_floor (sampling) = {result['noise_floor']}")
    logger.info("  effect_toward_A>0 => drug used | output_change_rand>>noise => hook works")
    for k, d in sorted(result["by_layer_scale"].items()):
        logger.info(f"  {k}: effect_toward_A={d.get('effect_toward_A')}  "
                    f"out_change_drug={d.get('output_change_drug')}  "
                    f"out_change_rand={d.get('output_change_rand')}")
    logger.info("=" * 100)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

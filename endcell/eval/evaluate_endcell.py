#!/usr/bin/env python
r"""
evaluate_endcell.py
===================
[END_CELL]-aware standard evaluation for the retrained C2S-Scale model. Replaces the full-panel
`evaluate_c2s_tahoe.py` scoring for the [END_CELL] representation, where a *correct* sentence
contains only the expressed genes (mean ~123 of 946) + the [END_CELL] sentinel — so the old
full-panel `coverage = emitted/946` guard would flag every correct prediction as degenerate and
null its DE-Δr. Here validity is redefined for variable-length outputs, and every score is reported
under BOTH advisor conventions for absent genes.

TWO ABSENT-GENE CONVENTIONS (the only axis that varies; topN-τ/panel-τ collapse to one panel-τ):
  * worst      : absent panel genes tied at the worst rank (P)         [Federico: absent -> bottom]
  * francesca  : absent panel genes tied at a fixed mid-rank (P//2)     [Francesca: fixed mid-rank]
Present genes always keep their emitted ranks 1..k. The convention is just the fill value handed to
the rank vector — one knob, so a bucket-position sweep is free if the interpretation is disputed.

METRICS reported per (tier, convention):
  * DE-Δr  — Pearson AND Spearman of the rank-shift (vs control) over top-K DE genes, K-sweep 20/50/100/200
  * panel-τ — Kendall τ over all 946 genes (the absent-sensitive τ)

MODES (select with --mode; comma-separated to run several):
  * model     : generate on the real eval tiers, score both conventions, validity, per-drug/MOA/dose breakdown   [GPU]
  * scramble  : generate on the scrambled-drug eval set, score vs the SAME truth -> DE-Δr(real) vs DE-Δr(scramble)  [GPU]
  * baselines : mean-shift ladder (control / global / per-cellline / per-MOA / per-MOA×cellline) from train        [CPU]
  * ceiling   : replicate noise ceiling (cell-vs-cell and cell-vs-consensus) from the eval files                    [CPU]

USAGE
  # GPU (model + scramble):
  python evaluate_endcell.py --mode model,scramble \
     --eval_dir DATA_endcell_big --scram_dir DATA_endcell_big_scram \
     --model_path CKPT_endcell/final --tiers tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos \
     --scram_tiers tier1_seen_conditions,tier2_unseen_drugs \
     --max_eval 400 --bf16 --out RESULTS/eval_endcell_model.json
  # CPU (baselines + ceiling), no GPU:
  python evaluate_endcell.py --mode baselines,ceiling \
     --eval_dir DATA_endcell_big --train_file DATA_endcell_big/train.jsonl \
     --tiers tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos \
     --out RESULTS/eval_endcell_cpu.json

SELFTEST (no model/data/network)
  python evaluate_endcell.py --selftest --out /tmp/eval_endcell_selftest.json
"""
import argparse, hashlib, json, os, sys, logging
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
DE_K_LIST = (20, 50, 100, 200)
DEFAULT_GENERATION_CONTRACT = "legacy"

try:
    from freeze_nir_manifest import (PredictionCache, atomic_write_json, control_from_prompt,
                                     file_sha256, sha256_json, sha256_text, stable_output_seed)
except ImportError:  # pragma: no cover
    from endcell.analysis.freeze_nir_manifest import (PredictionCache, atomic_write_json,
                                                      control_from_prompt, file_sha256, sha256_json,
                                                      sha256_text, stable_output_seed)


# ----------------------------------------------------------------- rank representation
def genes_of(sentence):
    """Gene tokens up to the FIRST [END_CELL] — the model's answer ends there; anything the model
    keeps emitting afterward is not part of the predicted cell and must not pollute scoring."""
    out = []
    for t in sentence.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def atomic_sentinel_id(tokenizer, token=SENTINEL):
    """Reject split and unknown sentinel encodings; never manufacture a stop token."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{token} must tokenize atomically, got {ids}")
    token_id = int(ids[0])
    if tokenizer.unk_token_id is not None and token_id == int(tokenizer.unk_token_id):
        raise ValueError(f"{token} maps to unk_token_id={tokenizer.unk_token_id}")
    return token_id


def sentence_to_rankarr(sentence, panel_index, P, fill):
    """Length-P rank array. Present panel genes keep emitted rank 1..k (first occurrence);
    absent panel genes tied at `fill` (P for 'worst', P//2 for 'francesca'). [END_CELL] stripped."""
    arr = np.full(P, float(fill), dtype=np.float64)
    seen = set()
    pos = 0
    for g in genes_of(sentence):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        pos += 1
        seen.add(gi)
        arr[gi] = float(pos)
    return arr


def expressed_panel_set(sentence, panel_set):
    return {g for g in genes_of(sentence) if g in panel_set}


# ----------------------------------------------------------------- metrics
def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    return r


def _partial_corr(a, b, c):
    """corr(a, b | c) — the DE-Δr with the control regressed OUT of both the predicted and true shift.
    Removes the shared −control (regression-to-mean) component that makes plain DE-Δr exploitable:
    a predictor that is pure control-reversion (e.g. every gene at rank P/2) has a ≈ −c, so r_ac ≈ ±1
    and the partial is undefined/0 — it earns no residual 'skill'. What survives is only the agreement
    between the prediction's structure BEYOND control-reversion and the truth's."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    rab, rac, rbc = _corr(a, b), _corr(a, c), _corr(b, c)
    if rab is None or rac is None or rbc is None:
        return None
    denom = np.sqrt(max(0.0, (1 - rac ** 2) * (1 - rbc ** 2)))
    if denom < 1e-6:                       # a or b is (anti)collinear with control -> pure reversion
        return None
    return float((rab - rac * rbc) / denom)


def score_pair(pred_arr, true_arr, ctrl_arr, de_k_list):
    """DE-Δr (Pearson + Spearman) over top-K DE genes, K-sweep, plus panel-τ (Kendall).
    DE genes = top-K by |true_rank - control_rank|. true-shift constant -> None (degenerate)."""
    out = {}
    dshift_true_full = np.abs(true_arr - ctrl_arr)
    order = np.argsort(-dshift_true_full)
    for k in de_k_list:
        de = order[:k]
        pred_shift = pred_arr[de] - ctrl_arr[de]
        true_shift = true_arr[de] - ctrl_arr[de]
        if np.std(true_shift) < 1e-12:
            out[f"de_pearson_k{k}"] = None
            out[f"de_spearman_k{k}"] = None
            out[f"de_partial_k{k}"] = None
            continue
        out[f"de_pearson_k{k}"] = _corr(pred_shift, true_shift)
        out[f"de_spearman_k{k}"] = _corr(_rankdata(pred_shift), _rankdata(true_shift))
        # control-partialled DE-Δr: skill BEYOND control-reversion (kills the revert_center exploit)
        out[f"de_partial_k{k}"] = _partial_corr(pred_shift, true_shift, ctrl_arr[de])
    # panel-τ (Kendall) over all P — the absent-sensitive ordering metric
    try:
        from scipy.stats import kendalltau
        tau = kendalltau(true_arr, pred_arr).statistic
        out["panel_tau"] = float(tau) if tau == tau else None
    except Exception:
        out["panel_tau"] = _corr(_rankdata(true_arr), _rankdata(pred_arr))  # fallback
    return out


def validity(pred_sentence, true_sentence, panel_set, min_genes=20, generation=None):
    """[END_CELL]-appropriate well-formedness. coverage is recall against the TRUTH expressed set
    (not fraction of 946); precision = fraction of emitted genes that are truly expressed."""
    toks = pred_sentence.strip().split()
    gene_toks = genes_of(pred_sentence)  # truncated at first [END_CELL]
    pred_set = {t for t in gene_toks if t in panel_set}
    true_set = expressed_panel_set(true_sentence, panel_set)
    n_pred, n_true = len(pred_set), len(true_set)
    inter = len(pred_set & true_set)
    recall = inter / n_true if n_true else None
    precision = inter / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (recall and precision and (precision + recall) > 0) else 0.0)
    halluc = (len(gene_toks) - len([t for t in gene_toks if t in panel_set])) / max(1, len(gene_toks))
    generation = generation or {}
    end_emitted = bool(generation.get("end_cell_emitted", SENTINEL in toks))
    native_eos = bool(generation.get("native_eos_emitted", False))
    cap = bool(generation.get("cap_reached", False))
    malformed = bool(generation.get("malformed", halluc > 0))
    duplicates = len([t for t in gene_toks if t in panel_set]) - n_pred
    complete = bool(end_emitted and n_pred >= min_genes and not cap)
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "hallucination_rate": halluc,
        "n_pred_genes": n_pred, "n_true_genes": n_true,
        "len_ratio": (n_pred / n_true) if n_true else None,
        "emits_end_cell": end_emitted, "native_eos": native_eos,
        "cap_reached": cap, "malformed": malformed,
        "duplicate_gene_count": generation.get("duplicate_gene_count", duplicates),
        "recognized_gene_count": generation.get("recognized_gene_count", n_pred),
        "recognized_gene_rate": generation.get("recognized_gene_rate",
                                                  n_pred / max(1, len(gene_toks))),
        "termination_reason": generation.get("termination", "unknown"),
        "complete": complete,
        "degenerate": n_pred < min_genes,
    }


CONV = None  # set in main: {"worst": P, "francesca": P//2}


def score_example(pred_sentence, ex, panel_index, panel_set, P, de_k_list, generation=None):
    """Score one prediction under BOTH conventions + validity. Returns a per-example record."""
    true_s = ex["response"]
    ctrl_s = control_from_prompt(ex["prompt"])
    rec = {"metadata": ex.get("metadata", {}),
           "validity": validity(pred_sentence, true_s, panel_set, generation=generation)}
    if not ctrl_s:
        rec["scores"] = None
        return rec
    scores = {}
    for cname, fill in CONV.items():
        pred_arr = sentence_to_rankarr(pred_sentence, panel_index, P, fill)
        true_arr = sentence_to_rankarr(true_s, panel_index, P, fill)
        ctrl_arr = sentence_to_rankarr(ctrl_s, panel_index, P, fill)
        scores[cname] = score_pair(pred_arr, true_arr, ctrl_arr, de_k_list)
    rec["scores"] = scores
    return rec


# ----------------------------------------------------------------- aggregation
def aggregate(records, de_k_list, invalid_policy="score_as_emitted"):
    """Aggregate under an explicit invalid-output policy; never silently repair outputs."""
    if invalid_policy == "drop_degenerate_legacy":
        good = [r for r in records if r.get("scores") and not r["validity"]["degenerate"]]
    else:
        good = [r for r in records if r.get("scores")]
    out = {"n_total": len(records), "n_scored": len(good), "conventions": {}}
    for cname in CONV:
        mdict = {}
        for k in de_k_list:
            for base in (f"de_pearson_k{k}", f"de_spearman_k{k}", f"de_partial_k{k}"):
                vals = [r["scores"][cname][base] for r in good
                        if r["scores"][cname].get(base) is not None]
                mdict[base] = _meanci(vals)
        vals = [r["scores"][cname]["panel_tau"] for r in good
                if r["scores"][cname].get("panel_tau") is not None]
        mdict["panel_tau"] = _meanci(vals)
        out["conventions"][cname] = mdict
    # validity summary (convention-independent)
    def vmean(key):
        vs = [r["validity"][key] for r in records if r["validity"].get(key) is not None]
        return _meanci(vs)
    out["validity"] = {k: vmean(k) for k in ("recall", "precision", "f1", "hallucination_rate",
                                             "len_ratio", "n_pred_genes", "n_true_genes")}
    out["validity"]["emits_end_cell_rate"] = float(np.mean([r["validity"]["emits_end_cell"] for r in records]))
    out["validity"]["native_eos_rate"] = float(np.mean([r["validity"]["native_eos"] for r in records]))
    out["validity"]["cap_rate"] = float(np.mean([r["validity"]["cap_reached"] for r in records]))
    out["validity"]["malformed_rate"] = float(np.mean([r["validity"]["malformed"] for r in records]))
    out["validity"]["complete_rate"] = float(np.mean([r["validity"]["complete"] for r in records]))
    out["validity"]["duplicate_gene_count"] = vmean("duplicate_gene_count")
    out["validity"]["recognized_gene_count"] = vmean("recognized_gene_count")
    out["validity"]["recognized_gene_rate"] = vmean("recognized_gene_rate")
    out["validity"]["degenerate_rate"] = float(np.mean([r["validity"]["degenerate"] for r in records]))
    out["invalid_policy"] = invalid_policy
    return out


def _meanci(vals):
    if not vals:
        return None
    a = np.asarray(vals, float)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "ci_low": float(a.mean() - 1.96 * se),
            "ci_high": float(a.mean() + 1.96 * se), "n": int(len(a))}


def breakdown(records, key, de_k_list, headline_k=50, conv="worst", min_n=8,
              invalid_policy="score_as_emitted"):
    """Per-group (drug/moa/dose) headline DE-Δr (Pearson) — turns the aggregate into a finding."""
    groups = defaultdict(list)
    for r in records:
        if not r.get("scores") or (invalid_policy == "drop_degenerate_legacy" and
                                   r["validity"]["degenerate"]):
            continue
        g = r["metadata"].get(key)
        v = r["scores"][conv].get(f"de_pearson_k{headline_k}")
        if g is not None and v is not None:
            groups[g].append(v)
    out = {str(g): {"mean_de": float(np.mean(v)), "n": len(v)}
           for g, v in groups.items() if len(v) >= min_n}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["mean_de"]))


# ----------------------------------------------------------------- generation modes
def load_tier(data_dir, tier):
    path = os.path.join(data_dir, f"eval_{tier}.jsonl")
    if not os.path.exists(path):
        logger.warning(f"  missing {path}")
        return None
    examples = []
    with open(path, "rb") as handle:
        for line_index, raw in enumerate(handle):
            if not raw.strip():
                continue
            ex = json.loads(raw.decode("utf-8"))
            ex["_source_line_index"] = line_index
            ex["_source_line_sha256"] = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
            examples.append(ex)
    return examples


def _generation_meta(raw_ids, decoded, end_cell_id, eos_id, max_new_tokens, panel_set):
    ids = list(map(int, raw_ids))
    end_pos = ids.index(end_cell_id) if end_cell_id in ids else None
    eos_pos = ids.index(eos_id) if eos_id is not None and eos_id in ids else None
    stops = [(p, "end_cell") for p in [end_pos] if p is not None]
    stops += [(p, "native_eos") for p in [eos_pos] if p is not None]
    termination = min(stops)[1] if stops else (
        "cap" if len(ids) >= max_new_tokens else "malformed_stop")
    toks = decoded.strip().split()
    if SENTINEL in toks:
        toks = toks[:toks.index(SENTINEL)]
    recognized = [t for t in toks if t in panel_set]
    malformed = [t for t in toks if t not in panel_set]
    return {
        "termination": termination, "end_cell_emitted": end_pos is not None,
        "native_eos_emitted": eos_pos is not None, "cap_reached": termination == "cap",
        "malformed": bool(malformed), "malformed_token_count": len(malformed),
        "duplicate_gene_count": len(recognized) - len(set(recognized)),
        "recognized_gene_count": len(set(recognized)),
        "recognized_gene_rate": len(recognized) / max(1, len(toks)),
    }


def generate_endcell_batch(model, tok, prompts, args, device, end_cell_id, panel_set,
                           specs=None, cache=None):
    """Return raw generated IDs, decoded text, and the observed native termination.

    Nothing is truncated or repaired before auditing. In particular `[END_CELL]` is never
    appended in post-processing: native EOS, a token cap, and malformed stopping stay distinct.
    """
    import torch
    prev = tok.padding_side
    tok.padding_side = "left"
    eos = [end_cell_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    try:
        if args.generation_contract == "deterministic":
            records = []
            for prompt, spec in zip(prompts, specs or []):
                row_id, prompt_id, draw_id = spec["row_id"], spec["prompt_id"], spec["draw_id"]
                seed = stable_output_seed(args.model_fingerprint, row_id, prompt_id, draw_id,
                                          args.generation_contract_version)
                output_id = sha256_json({"model": args.model_fingerprint, "row_id": row_id,
                                         "prompt_id": prompt_id, "draw_id": draw_id,
                                         "contract": args.generation_contract_version})
                old = cache.get(output_id) if cache else None
                if old is not None:
                    records.append(old)
                    continue
                enc = tok(prompt, return_tensors="pt").to(device)
                kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                          eos_token_id=eos)
                if args.do_sample:
                    kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
                else:
                    kw.update(do_sample=False)
                devices = ([torch.cuda.current_device()] if str(device).startswith("cuda") else [])
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    if devices:
                        torch.cuda.manual_seed_all(seed)
                    with torch.no_grad():
                        out = model.generate(**enc, **kw)
                raw_ids = out[0, enc["input_ids"].shape[1]:].tolist()
                decoded = tok.decode(raw_ids, skip_special_tokens=False).strip()
                meta = _generation_meta(raw_ids, decoded, end_cell_id, tok.eos_token_id,
                                        args.max_new_tokens, panel_set)
                record = {"type": "prediction", "output_id": output_id, "row_id": row_id,
                          "prompt_id": prompt_id, "prompt_sha256": sha256_text(prompt),
                          "draw_id": draw_id, "seed": seed,
                          "raw_token_ids": list(map(int, raw_ids)), "decoded_text": decoded,
                          "termination_reason": meta["termination"], "validity": meta}
                if cache:
                    cache.put(record)
                records.append(record)
            return records
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        if args.do_sample:
            kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        plen = enc["input_ids"].shape[1]
        records = []
        for i in range(len(prompts)):
            ids = out[i][plen:].tolist()
            txt = tok.decode(ids, skip_special_tokens=False).strip()
            meta = _generation_meta(ids, txt, end_cell_id, tok.eos_token_id,
                                    args.max_new_tokens, panel_set)
            records.append({"decoded_text": txt, "raw_token_ids": list(map(int, ids)),
                            "termination_reason": meta["termination"], "validity": meta})
        return records
    finally:
        tok.padding_side = prev


def run_generate(model, tok, examples, args, device, panel_set, tier, cache=None):
    gens = []
    for i in range(0, len(examples), args.gen_batch_size):
        chunk = examples[i:i + args.gen_batch_size]
        batch = [e["prompt"] for e in chunk]
        specs = [{"row_id": sha256_json({"tier": tier, "line": e["_source_line_index"],
                                           "line_sha256": e["_source_line_sha256"]}),
                  "prompt_id": sha256_text(e["prompt"]), "draw_id": 0} for e in chunk]
        gens.extend(generate_endcell_batch(model, tok, batch, args, device, args.end_cell_id,
                                           panel_set, specs=specs, cache=cache))
        if (i // args.gen_batch_size) % 10 == 0:
            logger.info(f"    generated {min(i + args.gen_batch_size, len(examples))}/{len(examples)}")
    return gens


def subsample(examples, max_eval, seed):
    if max_eval and len(examples) > max_eval:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(examples), max_eval, replace=False)
        return [examples[i] for i in sorted(idx)]
    return examples


def mode_model(args, panel_index, panel_set, P, model, tok, device, scram=False):
    data_dir = args.scram_dir if scram else args.eval_dir
    tiers = (args.scram_tiers if scram else args.tiers).split(",")
    out = {}
    for tier in [t.strip() for t in tiers if t.strip()]:
        examples = load_tier(data_dir, tier)
        if not examples:
            continue
        examples = subsample(examples, args.max_eval, args.seed)
        logger.info(f"  [{'scramble' if scram else 'model'}] tier {tier}: {len(examples)} cells")
        predictions = run_generate(model, tok, examples, args, device, panel_set, tier,
                                   cache=getattr(args, "prediction_cache_obj", None))
        records = [score_example(pred["decoded_text"], ex, panel_index, panel_set, P, DE_K_LIST,
                                 generation=pred.get("validity"))
                   for pred, ex in zip(predictions, examples)]
        agg = aggregate(records, DE_K_LIST, args.invalid_policy)
        agg["prediction_records"] = predictions
        gate_pass = (agg["validity"]["complete_rate"] >= args.min_complete_rate and
                     agg["validity"]["recognized_gene_rate"]["mean"] >=
                     args.min_recognized_gene_rate)
        agg["validity_gate"] = {"passed": bool(gate_pass),
                                "min_complete_rate": args.min_complete_rate,
                                "min_recognized_gene_rate": args.min_recognized_gene_rate}
        if args.invalid_policy == "validity_failure" and not gate_pass:
            agg = {"status": "validity_failure", "n_total": len(records),
                   "failure_reason": "checkpoint failed cell-sentence validity gate",
                   "validity": agg["validity"], "validity_gate": agg["validity_gate"],
                   "prediction_records": predictions}
            out[tier] = agg
            continue
        agg["breakdown_by_drug"] = breakdown(records, "drug", DE_K_LIST,
                                               invalid_policy=args.invalid_policy)
        agg["breakdown_by_moa"] = breakdown(records, "moa", DE_K_LIST,
                                              invalid_policy=args.invalid_policy)
        agg["breakdown_by_dose"] = breakdown(records, "dose", DE_K_LIST,
                                               invalid_policy=args.invalid_policy)
        out[tier] = agg
    return out


# ----------------------------------------------------------------- baselines (CPU)
def mode_baselines(args, panel_index, panel_set, P):
    """Mean-shift ladder from train, scored under both conventions. Predicts control + group-mean
    rank-shift; beating per-MOA×cellline reflects CONTROL-CONDITIONING (per-cell tailoring), not
    drug knowledge — the corrected results.md §3 framing."""
    import evaluate_c2s_tahoe as ev
    logger.info("  [baselines] loading train for mean-shift estimation ...")
    train = [json.loads(l) for l in open(args.train_file)]
    if args.max_train and len(train) > args.max_train:
        rng = np.random.RandomState(args.seed)
        train = [train[i] for i in sorted(rng.choice(len(train), args.max_train, replace=False))]

    def key_of(meta, mode):
        moa = meta.get("moa") or "unclear"
        cl = meta.get("cell_line_id")
        if mode == "global":
            return "_G"
        if mode == "cellline":
            return cl
        if mode == "moa":
            return moa
        if mode == "moa_cellline":
            return (moa, cl)
        return None

    modes = ["global", "cellline", "moa", "moa_cellline"]
    # accumulate mean shift (in the 'worst' rank space; convention only changes scoring fill)
    out = {}
    for cname, fill in CONV.items():
        shift_sum = {m: defaultdict(lambda: np.zeros(P)) for m in modes}
        shift_cnt = {m: defaultdict(int) for m in modes}
        for ex in train:
            meta = ex.get("metadata", {})
            ctrl = ev.control_from_prompt(ex["prompt"])
            if not ctrl:
                continue
            t_arr = sentence_to_rankarr(ex["response"], panel_index, P, fill)
            c_arr = sentence_to_rankarr(ctrl, panel_index, P, fill)
            s = t_arr - c_arr
            for m in modes:
                k = key_of(meta, m)
                shift_sum[m][k] += s
                shift_cnt[m][k] += 1
        shift_mean = {m: {k: shift_sum[m][k] / shift_cnt[m][k] for k in shift_sum[m]} for m in modes}

        conv_out = {}
        for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
            examples = subsample(load_tier(args.eval_dir, tier) or [], args.max_eval, args.seed)
            if not examples:
                continue
            per = {("control"): [], **{m: [] for m in modes}}
            for ex in examples:
                meta = ex.get("metadata", {})
                ctrl = ev.control_from_prompt(ex["prompt"])
                if not ctrl:
                    continue
                t_arr = sentence_to_rankarr(ex["response"], panel_index, P, fill)
                c_arr = sentence_to_rankarr(ctrl, panel_index, P, fill)
                # control-as-prediction (DE-Δr ≡ 0 by construction)
                per["control"].append(score_pair(c_arr, t_arr, c_arr, DE_K_LIST))
                for m in modes:
                    k = key_of(meta, m)
                    sm = shift_mean[m].get(k)
                    pred_arr = c_arr + sm if sm is not None else c_arr
                    per[m].append(score_pair(pred_arr, t_arr, c_arr, DE_K_LIST))
            conv_out[tier] = {b: _meanci([s[f"de_pearson_k50"] for s in per[b]
                                          if s.get("de_pearson_k50") is not None])
                              for b in per}
        out[cname] = conv_out
    return out


# ----------------------------------------------------------------- fair linear baseline (CPU)
def mode_linear(args, panel_index, panel_set, P):
    """Fair linear control->shift baseline on the HEADLINE metric — the 'does the LLM beat a linear
    predictor?' comparator. Ridge-regress the rank-shift vector on the control rank vector:
        shift ≈ W·(control − μ_c) + μ_s,   prediction = control + shift_pred.
    MEAN-CENTERING is essential on [END_CELL]: ~820/946 genes sit at the same constant fill (absent),
    so those columns are near-constant and a naive fit is degenerate; centering + ridge handles it
    (a column that is always the fill value contributes nothing, as it should). Scored with the same
    DE-Δr (K-sweep) as the model, under both conventions. Unlike the mean-shift ladder (group
    averages), this uses each cell's OWN control vector, so it captures the same control-conditioning
    the model exploits — hence it is the fair 'is the model just a linear map of the control?' test."""
    import evaluate_c2s_tahoe as ev
    logger.info("  [linear] loading train for ridge control->shift fit ...")
    train = [json.loads(l) for l in open(args.train_file)]
    rng = np.random.RandomState(args.seed)
    if args.max_linear_fit and len(train) > args.max_linear_fit:
        train = [train[i] for i in sorted(rng.choice(len(train), args.max_linear_fit, replace=False))]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    out = {}
    for cname, fill in CONV.items():
        # build control (X) and shift (Y = treated - control) matrices over train
        Xs, Ys = [], []
        for ex in train:
            ctrl = ev.control_from_prompt(ex["prompt"])
            if not ctrl:
                continue
            c = sentence_to_rankarr(ctrl, panel_index, P, fill)
            t = sentence_to_rankarr(ex["response"], panel_index, P, fill)
            Xs.append(c)
            Ys.append(t - c)
        X = np.asarray(Xs, dtype=np.float64)
        Y = np.asarray(Ys, dtype=np.float64)
        mu_c, mu_s = X.mean(0), Y.mean(0)
        Xc, Yc = X - mu_c, Y - mu_s
        XtX = Xc.T @ Xc
        lam = args.ridge_lambda * (np.trace(XtX) / P + 1e-9)      # scale-relative ridge
        W = np.linalg.solve(XtX + lam * np.eye(P), Xc.T @ Yc)     # (P, P)
        logger.info(f"  [linear/{cname}] fit on {X.shape[0]:,} pairs, ridge λ≈{lam:.3g}")
        # Baselines to compare against the model — ALL drug-AGNOSTIC (they never see the drug):
        #   ridge         : the fitted linear control->shift map
        #   revert_mean   : predict treated = mean control profile (no real fit; shift = μ_c − control)
        #   revert_center : predict every gene at the middle rank P/2 (pure regression-to-the-mean)
        # If the trivial reverts already score high on DE-Δr, that PROVES DE-Δr is dominated by the
        # on/off regression-to-mean control structure, not by any learned perturbation/drug signal.
        conv_out = {}
        mid = np.full(P, P / 2.0)
        MK = ("de_pearson_k50", "de_partial_k50", "panel_tau")
        for tier in tiers:
            examples = subsample(load_tier(args.eval_dir, tier) or [], args.max_eval, args.seed)
            acc = {b: {k: [] for k in MK} for b in ("ridge", "revert_mean", "revert_center")}
            for ex in examples:
                ctrl = ev.control_from_prompt(ex["prompt"])
                if not ctrl:
                    continue
                c = sentence_to_rankarr(ctrl, panel_index, P, fill)
                t = sentence_to_rankarr(ex["response"], panel_index, P, fill)
                preds = {"ridge": c + mu_s + (c - mu_c) @ W,
                         "revert_mean": mu_c,
                         "revert_center": mid}
                for b, pred in preds.items():
                    s = score_pair(pred, t, c, DE_K_LIST)
                    for k in MK:
                        if s.get(k) is not None:
                            acc[b][k].append(s[k])
            conv_out[tier] = {b: {k: _meanci(v) for k, v in acc[b].items()} for b in acc}
        out[cname] = conv_out
    return out


# ----------------------------------------------------------------- noise ceiling (CPU)
def mode_ceiling(args, panel_index, panel_set, P):
    """Replicate ceiling: agreement between two REAL treated cells of the same (drug, cell line).
    cell-vs-cell (single-cell truth) and cell-vs-consensus (denoised). Both conventions."""
    import evaluate_c2s_tahoe as ev
    out = {}
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        examples = load_tier(args.eval_dir, tier)
        if not examples:
            continue
        by_cond = defaultdict(list)
        for ex in examples:
            m = ex.get("metadata", {})
            ctrl = ev.control_from_prompt(ex["prompt"])
            if not ctrl:
                continue
            by_cond[(m.get("drug"), m.get("cell_line_id"))].append((ex["response"], ctrl))
        rng = np.random.RandomState(args.seed)
        tier_out = {}
        for cname, fill in CONV.items():
            cvc, cvcons = [], []
            for cond, cells in by_cond.items():
                if len(cells) < 2:
                    continue
                arrs = [sentence_to_rankarr(r, panel_index, P, fill) for r, _ in cells]
                carrs = [sentence_to_rankarr(c, panel_index, P, fill) for _, c in cells]
                n = len(arrs)
                # cell-vs-cell: a few random disjoint pairs
                for _ in range(min(10, n * (n - 1) // 2)):
                    i, j = rng.choice(n, 2, replace=False)
                    cvc.append(score_pair(arrs[i], arrs[j], carrs[j], DE_K_LIST)["de_pearson_k50"])
                # cell-vs-consensus: each cell vs mean of the others
                for i in range(min(n, 12)):
                    others = [arrs[k] for k in range(n) if k != i]
                    cons = np.mean(np.stack(others), axis=0)
                    cvcons.append(score_pair(cons, arrs[i], carrs[i], DE_K_LIST)["de_pearson_k50"])
            tier_out[cname] = {
                "cell_vs_cell_de50": _meanci([v for v in cvc if v is not None]),
                "cell_vs_consensus_de50": _meanci([v for v in cvcons if v is not None]),
                "n_conditions": len([c for c in by_cond.values() if len(c) >= 2]),
            }
        out[tier] = tier_out
    return out


# ----------------------------------------------------------------- reporting
def print_report(result):
    logger.info("")
    logger.info("=" * 100)
    for mode, data in result.items():
        if mode in ("config", "selftest", "passed", "conventions"):
            continue
        logger.info(f"  ===== MODE: {mode} =====")
        if mode in ("model", "scramble"):
            for tier, agg in data.items():
                if agg.get("status") == "validity_failure":
                    logger.info(f"  [{tier}] VALIDITY FAILURE; NIR/metric scoring withheld "
                                f"({agg['failure_reason']})")
                    continue
                logger.info(f"  [{tier}] scored {agg['n_scored']}/{agg['n_total']}  "
                            f"recall={_g(agg['validity']['recall'])} "
                            f"prec={_g(agg['validity']['precision'])} "
                            f"halluc={_g(agg['validity']['hallucination_rate'])} "
                            f"endcell={agg['validity']['emits_end_cell_rate']:.2f} "
                            f"len_ratio={_g(agg['validity']['len_ratio'])}")
                for cname in agg["conventions"]:
                    c = agg["conventions"][cname]
                    logger.info(f"      [{cname:9s}] DE-Δr(P) K20/50/100/200="
                                f"{_g(c['de_pearson_k20'])}/{_g(c['de_pearson_k50'])}/"
                                f"{_g(c['de_pearson_k100'])}/{_g(c['de_pearson_k200'])}  "
                                f"partial-DE(K50)={_g(c['de_partial_k50'])}  panel-τ={_g(c['panel_tau'])}")
        elif mode == "baselines":
            for cname, tiers in data.items():
                logger.info(f"  [{cname}]")
                for tier, lad in tiers.items():
                    logger.info(f"    {tier}: " + "  ".join(
                        f"{b}={_g(v)}" for b, v in lad.items()))
        elif mode == "linear":
            logger.info("  drug-AGNOSTIC baselines — DE-Δr(K50) | partial-DE(K50, control removed) | panel-τ.")
            logger.info("  raw DE-Δr high + partial-DE ~0 => the metric is a control regression-to-mean artifact.")
            for cname, tiers in data.items():
                for tier, bl in tiers.items():
                    for b, ks in bl.items():
                        logger.info(f"  [{tier}/{cname}/{b:13s}] DE-Δr={_g(ks['de_pearson_k50'])}  "
                                    f"partial-DE={_g(ks['de_partial_k50'])}  panel-τ={_g(ks['panel_tau'])}")
        elif mode == "ceiling":
            for tier, tc in data.items():
                for cname, c in tc.items():
                    logger.info(f"  [{tier}/{cname}] cell-vs-cell={_g(c['cell_vs_cell_de50'])} "
                                f"cell-vs-consensus={_g(c['cell_vs_consensus_de50'])} "
                                f"(n_cond={c['n_conditions']})")
    logger.info("=" * 100)


def _g(x):
    if x is None:
        return "NA"
    if isinstance(x, dict):
        return f"{x['mean']:.3f}" if x.get("mean") is not None else "NA"
    return f"{x:.3f}"


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic check of the scoring core (no model/data). Confirms:
      * a perfect prediction scores DE-Δr ~ 1 under both conventions
      * a shuffled prediction scores lower
      * a garbage (few-gene) output is flagged degenerate (would have been nulled)
      * scramble wiring: scoring a wrong-drug prediction vs the true response runs and drops."""
    global CONV
    P = 200
    CONV = {"worst": P, "francesca": P // 2}
    panel = [f"G{i}" for i in range(P)]
    panel_index = {g: i for i, g in enumerate(panel)}
    panel_set = set(panel)
    rng = np.random.RandomState(0)

    def make_sentence(expr_idx):
        order = sorted(expr_idx, key=lambda i: -rng.rand())  # arbitrary but stable-ish order
        return " ".join(panel[i] for i in order) + " " + SENTINEL

    # a "true" cell: 120 expressed genes; control: overlapping but shifted
    true_idx = rng.choice(P, 120, replace=False)
    ctrl_idx = rng.choice(P, 110, replace=False)
    true_s = make_sentence(true_idx)
    ctrl_s = make_sentence(ctrl_idx)
    ex = {"response": true_s, "prompt": f"x\nControl cell: {ctrl_s}\n\nResponse cell:",
          "metadata": {"drug": "D", "moa": "M", "cell_line_id": "CL"}}

    perfect = score_example(true_s, ex, panel_index, panel_set, P, DE_K_LIST)
    shuffled_idx = rng.choice(P, 120, replace=False)
    shuffled = score_example(make_sentence(shuffled_idx), ex, panel_index, panel_set, P, DE_K_LIST)
    garbage = score_example(" ".join(panel[i] for i in true_idx[:5]) + " " + SENTINEL,
                            ex, panel_index, panel_set, P, DE_K_LIST)

    pe = {c: perfect["scores"][c]["de_pearson_k50"] for c in CONV}
    sh = {c: shuffled["scores"][c]["de_pearson_k50"] for c in CONV}
    logger.info(f"  perfect  DE-Δr(K50): {pe}   (expect ~1 both)")
    logger.info(f"  shuffled DE-Δr(K50): {sh}   (expect << perfect)")
    logger.info(f"  perfect recall={perfect['validity']['recall']:.2f} (expect 1.0); "
                f"garbage degenerate={garbage['validity']['degenerate']} (expect True)")
    ok = (all(pe[c] is not None and pe[c] > 0.9 for c in CONV)
          and all(sh[c] < pe[c] - 0.2 for c in CONV)
          and abs(perfect["validity"]["recall"] - 1.0) < 1e-6
          and garbage["validity"]["degenerate"] is True
          and perfect["validity"]["degenerate"] is False)
    out = {"selftest": True, "passed": bool(ok),
           "perfect_de50": pe, "shuffled_de50": sh}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    atomic_write_json(args.out, out)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", default="model", help="comma-sep: model,scramble,baselines,ceiling")
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--scram_dir", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--tiers", default="tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos")
    ap.add_argument("--scram_tiers", default="tier1_seen_conditions,tier2_unseen_drugs")
    ap.add_argument("--max_eval", type=int, default=400, help="cells per tier (subsample)")
    ap.add_argument("--max_train", type=int, default=200000, help="train rows for baseline means")
    ap.add_argument("--max_linear_fit", type=int, default=40000, help="train pairs for the ridge linear fit")
    ap.add_argument("--ridge_lambda", type=float, default=0.1, help="ridge penalty (scaled by mean diag of X'X)")
    ap.add_argument("--francesca_rank", type=int, default=None, help="override the mid-rank (default P//2)")
    ap.add_argument("--panel_file", default=None)
    # generation
    ap.add_argument("--do_sample", action="store_true", help="sample (default greedy for eval)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--gen_batch_size", type=int, default=48)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generation_contract", choices=["deterministic", "legacy"],
                    default=DEFAULT_GENERATION_CONTRACT,
                    help="legacy preserves existing Pythia commands; deterministic opts into the "
                         "hardened paired-evaluation contract")
    ap.add_argument("--generation_contract_version", default="endcell-validity-v1")
    ap.add_argument("--model_fingerprint", default=None)
    ap.add_argument("--manifest_sha256", default=None,
                    help="frozen evaluation-manifest hash; required for deterministic caching")
    ap.add_argument("--parent_validity_only", action="store_true")
    ap.add_argument("--attention_implementation", choices=["eager", "sdpa", "flash_attention_2"],
                    default=None)
    ap.add_argument("--prediction_cache", default=None)
    ap.add_argument("--invalid_policy",
                    choices=["score_as_emitted", "validity_failure", "drop_degenerate_legacy"],
                    default="score_as_emitted")
    ap.add_argument("--min_complete_rate", type=float, default=0.95)
    ap.add_argument("--min_recognized_gene_rate", type=float, default=0.95)
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    global CONV
    panel_file = args.panel_file
    if panel_file is None:
        for cand in (os.path.join(args.eval_dir or ".", "l1000_panel.json"),
                     "l1000_panel.json", os.path.join("src", "l1000_panel.json")):
            if cand and os.path.exists(cand):
                panel_file = cand
                break
    panel = json.load(open(panel_file))
    panel_index = {g: i for i, g in enumerate(panel)}
    panel_set = set(panel)
    P = len(panel)
    CONV = {"worst": P, "francesca": args.francesca_rank or (P // 2)}
    logger.info(f"Panel {P} genes | conventions: {CONV}")

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    result = {"config": {k: v for k, v in vars(args).items()}, "conventions": CONV}

    need_gpu = any(m in ("model", "scramble") for m in modes)
    model = tok = device = None
    if need_gpu:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        load_kwargs = {"torch_dtype": torch.bfloat16 if args.bf16 else torch.float32}
        if args.attention_implementation:
            load_kwargs["attn_implementation"] = args.attention_implementation
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs).to(device)
        model.config.use_cache = True
        model.eval()
        try:
            args.end_cell_id = atomic_sentinel_id(tok)
        except ValueError as exc:
            if args.generation_contract == "deterministic" or args.parent_validity_only:
                failure = {
                    "status": "validity_failure", "nir_eligible": False,
                    "failure_stage": "tokenizer_contract", "failure_reason": str(exc),
                    "parent_validity_only": args.parent_validity_only,
                    "config": {k: v for k, v in vars(args).items()},
                }
                atomic_write_json(args.out, failure, default=float)
                logger.warning("atomic END_CELL contract failed; wrote validity-failure artifact")
                return
            raise
        if args.generation_contract == "deterministic":
            if not args.model_fingerprint:
                ap.error("deterministic generation requires immutable --model_fingerprint")
            if not args.manifest_sha256:
                ap.error("deterministic generation requires --manifest_sha256")
            cache_path = args.prediction_cache or (args.out + ".predictions.jsonl")
            args.prediction_cache_obj = PredictionCache(cache_path, {
                "model_fingerprint": args.model_fingerprint,
                "manifest_sha256": args.manifest_sha256,
                "panel_sha256": file_sha256(panel_file),
                "generation_contract_version": args.generation_contract_version,
                "do_sample": args.do_sample, "temperature": args.temperature,
                "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                "end_cell_id": args.end_cell_id, "native_eos_id": tok.eos_token_id,
                "min_complete_rate": args.min_complete_rate,
                "min_recognized_gene_rate": args.min_recognized_gene_rate,
                "min_recognized_genes": 20,
                "invalid_policy": args.invalid_policy,
                "attention_implementation": args.attention_implementation,
                "use_cache": True,
            })
        ec = [args.end_cell_id]
        logger.info(f"  model on {device}; [END_CELL] -> {ec} (id {args.end_cell_id}, "
                    f"{'atomic' if len(ec) == 1 else 'SPLIT — check tokenizer'}); generation stops at it")

    if "model" in modes:
        result["model"] = mode_model(args, panel_index, panel_set, P, model, tok, device, scram=False)
    if "scramble" in modes:
        result["scramble"] = mode_model(args, panel_index, panel_set, P, model, tok, device, scram=True)
    if "baselines" in modes:
        result["baselines"] = mode_baselines(args, panel_index, panel_set, P)
    if "linear" in modes:
        result["linear"] = mode_linear(args, panel_index, panel_set, P)
    if "ceiling" in modes:
        result["ceiling"] = mode_ceiling(args, panel_index, panel_set, P)

    failures = []
    for mode in ("model", "scramble"):
        for tier, payload in (result.get(mode) or {}).items():
            if payload.get("status") == "validity_failure":
                failures.append({"mode": mode, "tier": tier,
                                 "reason": payload.get("failure_reason")})
    result["status"] = ("validity_failure" if failures else
                        ("validity_only" if args.parent_validity_only else "ok"))
    result["nir_eligible"] = not failures and not args.parent_validity_only
    result["validity_failures"] = failures
    atomic_write_json(args.out, result, default=float)
    if getattr(args, "prediction_cache_obj", None) is not None:
        args.prediction_cache_obj.close()
    print_report(result)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

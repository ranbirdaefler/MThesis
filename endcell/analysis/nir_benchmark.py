#!/usr/bin/env python
r"""
nir_benchmark.py
================
Half 2 of the calibration story: benchmark the model on the ONE calibrated metric (NIR), against
fair baselines, at pseudobulk, on the held-out eval tiers.

NIR (Normalized Inverse Rank; ties score 0.5) = discrimination: for a predictor's pseudobulk profile, rank its
similarity to its OWN drug's truth against its similarity to ALL OTHER drugs' truths. 1.0 = own is
the single closest (perfect identifiability); ~0.5 = chance. Reported under TWO distances:
  * rank-NIR : rank correlation over ALL genes expressed in the own-drug truth (no top-N cap; the
               ~few-hundred expressed genes of [END_CELL]). Clean — the model's native output, no decode.
  * expr-NIR : Euclidean distance after decoding ranks -> expression via linear_model.json
               (matches the distance used to establish NIR as calibrated; inherits the lossy decode).

PREDICTORS scored per drug x cell line:
  * model   : K temperature-sampled predictions from the drug's held-out prompts, pseudobulk-averaged.
  * linear  : ridge control->shift fit on train, applied to the drug's control pseudobulk (drug-AGNOSTIC
              -> same output for every drug in a cell line -> chance by construction).
  * mean    : leave-one-out drug-agnostic mean profile (chance by construction).
  * ceiling : a real disjoint half of the drug's cells (the achievable discrimination bar).

USAGE (GPU)
  python nir_benchmark.py --eval_dir DATA_endcell_big --model_path CKPT/final \
     --tiers tier2_unseen_drugs,tier3_unseen_combos,tier4_dose_interpolation \
     --train_file DATA_endcell_big/train.jsonl --k_samples 8 --temperature 0.8 \
     --min_cells 20 --min_drugs_per_cl 6 --bf16 --out RESULTS/nir_benchmark.json

TEMP SWEEP diagnostic (how output diversity changes with temperature)
  python nir_benchmark.py --temp_sweep --eval_dir ... --model_path ... --tier tier2_unseen_drugs \
     --temps 0,0.5,0.8,1.0 --k_samples 8 --bf16 --out RESULTS/temp_sweep.json

SELFTEST (no model/data) — validates the NIR + similarity machinery
  python nir_benchmark.py --selftest --out /tmp/nir_selftest.json
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

# --- repo path bootstrap: works in BOTH the reorganized repo AND the flat cluster layout ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
_cands = [_HERE, os.path.join(_HERE, "src")]                    # flat layout: ~/tahoe and ~/tahoe/src
if os.path.isdir(os.path.join(_ROOT, "shared")):                # reorganized layout
    _cands += [os.path.join(_ROOT, "shared")] + sorted(glob.glob(os.path.join(_PIPE, "*")))
for _p in _cands:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"
GENERATION_CONTRACT_VERSION = "nir-generation-v1"

try:
    from freeze_nir_manifest import (PredictionCache, atomic_write_json, file_sha256,
                                     sha256_json, sha256_text, stable_output_seed,
                                     verify_manifest)
except ImportError:  # pragma: no cover - flat cluster layout
    from endcell.analysis.freeze_nir_manifest import (PredictionCache, atomic_write_json,
                                                      file_sha256, sha256_json, sha256_text,
                                                      stable_output_seed, verify_manifest)


# ----------------------------------------------------------------- representations
def genes_of(sentence):
    out = []
    for t in sentence.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def sentence_to_rankarr(sentence, panel_index, P, fill=None):
    fill = P if fill is None else fill
    arr = np.full(P, float(fill))
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


def sentence_to_expr(sentence, panel_index, P, lm):
    """Decode a cell sentence to an expression vector via the C2S linear model
    (expr = slope*log10(rank) + intercept, clamped >=0; absent genes -> 0)."""
    slope, intercept = lm["slope"], lm["intercept"]
    arr = np.zeros(P)
    seen = set()
    pos = 0
    for g in genes_of(sentence):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        pos += 1
        seen.add(gi)
        arr[gi] = max(0.0, slope * np.log10(pos) + intercept)
    return arr


def pb_rank(sentences, panel_index, P):
    return np.mean(np.stack([sentence_to_rankarr(s, panel_index, P) for s in sentences]), axis=0)


def pb_expr(sentences, panel_index, P, lm):
    return np.mean(np.stack([sentence_to_expr(s, panel_index, P, lm) for s in sentences]), axis=0)


# ----------------------------------------------------------------- similarities + NIR
def rank_corr(pred_rank, true_rank, expressed_idx):
    """Correlation of two rank profiles over the reference's expressed genes (all of them)."""
    if len(expressed_idx) < 3:
        return None
    a, b = pred_rank[expressed_idx], true_rank[expressed_idx]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def nir_from_sims(sim_own, sims_other):
    """Tie-aware NIR under a similarity (win=1, loss=0, tie=0.5)."""
    s = [x for x in sims_other if x is not None]
    if sim_own is None or not s:
        return None
    return float(np.mean([1.0 if sim_own > x else (0.5 if sim_own == x else 0.0) for x in s]))


def nir_from_dists(dist_own, dists_other):
    """Tie-aware NIR under a distance (win=1, loss=0, tie=0.5)."""
    d = [x for x in dists_other if x is not None]
    if dist_own is None or not d:
        return None
    return float(np.mean([1.0 if dist_own < x else (0.5 if dist_own == x else 0.0) for x in d]))


def generation_validity(raw_ids, decoded_text, *, end_id, eos_id, max_new_tokens,
                        panel_index, model_kind="cellsentence", min_genes=20):
    """Classify exactly what the model emitted; never infer or append a sentinel."""
    ids = list(map(int, raw_ids))
    end_pos = ids.index(end_id) if end_id is not None and end_id in ids else None
    eos_pos = ids.index(eos_id) if eos_id is not None and eos_id in ids else None
    stops = [(p, "end_cell") for p in [end_pos] if p is not None]
    stops += [(p, "native_eos") for p in [eos_pos] if p is not None]
    if stops:
        stop_pos, termination = min(stops)
        content_ids = ids[:stop_pos]
    else:
        stop_pos = None
        content_ids = ids
        termination = "cap" if len(ids) >= max_new_tokens else "malformed_stop"
    toks = decoded_text.strip().split()
    if SENTINEL in toks:
        toks = toks[:toks.index(SENTINEL)]
    has_down = "[DOWN]" in toks
    gene_toks = [t for t in toks if t != "[DOWN]"]
    recognized = [t for t in gene_toks if t in panel_index]
    unique = list(dict.fromkeys(recognized))
    malformed = [t for t in gene_toks if t not in panel_index]
    complete = ((termination == "end_cell") if model_kind == "cellsentence" else has_down)
    complete = bool(complete and len(unique) >= min_genes and termination != "cap")
    return {
        "termination": termination, "end_cell_emitted": end_pos is not None,
        "native_eos_emitted": eos_pos is not None, "cap_reached": termination == "cap",
        "malformed": bool(malformed), "malformed_token_count": len(malformed),
        "duplicate_gene_count": len(recognized) - len(unique),
        "recognized_gene_count": len(unique), "recognized_gene_rate": (
            len(recognized) / max(1, len(gene_toks))), "has_down": has_down,
        "complete": complete, "content_token_count": len(content_ids),
    }


def atomic_sentinel_id(tokenizer, token=SENTINEL):
    """Return a strict sentinel ID; split or UNK encodings are never accepted."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{token} must tokenize atomically, got {ids}")
    token_id = int(ids[0])
    if tokenizer.unk_token_id is not None and token_id == int(tokenizer.unk_token_id):
        raise ValueError(f"{token} maps to unk_token_id={tokenizer.unk_token_id}")
    if tokenizer.convert_ids_to_tokens(token_id) == tokenizer.unk_token:
        raise ValueError(f"{token} resolves to the unknown token")
    return token_id


def summarize_generation(records):
    if not records:
        return {"n": 0, "complete_rate": 0.0}
    validity = [r["validity"] for r in records]
    mean = lambda key: float(np.mean([float(v.get(key, 0)) for v in validity]))
    return {
        "n": len(records), "end_cell_rate": mean("end_cell_emitted"),
        "native_eos_rate": mean("native_eos_emitted"), "cap_rate": mean("cap_reached"),
        "malformed_rate": mean("malformed"),
        "duplicate_gene_rate": float(np.mean([
            v["duplicate_gene_count"] / max(1, v["recognized_gene_count"] + v["duplicate_gene_count"])
            for v in validity])),
        "mean_recognized_genes": mean("recognized_gene_count"),
        "mean_recognized_gene_rate": mean("recognized_gene_rate"),
        "complete_rate": mean("complete"),
    }


def _manifest_groups(manifest, eval_file, ev):
    examples = [json.loads(line) for line in open(eval_file, encoding="utf-8")]
    groups = defaultdict(dict)
    for row in manifest["rows"]:
        specs = [{**spec, "row_id": row["row_id"]} for spec in row["prompt_specs"]]
        slot = {
            "resp": [examples[i]["response"] for i in row["source_line_indices"]],
            "ctrl": [ev.control_from_prompt(examples[i]["prompt"])
                     for i in row["source_line_indices"]],
            "prompts": [spec["prompt"] for spec in specs],
            "prompt_specs": specs, "manifest_row": row,
            "truth_A": [examples[x["index"]]["response"] for x in row["truth_half_a"]],
            "truth_B": [examples[x["index"]]["response"] for x in row["truth_half_b"]],
        }
        groups[(row["cell_line"], row.get("plate"))][row["drug"]] = slot
    return groups


# ----------------------------------------------------------------- ridge linear (control->shift)
def fit_ridge(train, panel_index, P, ev, max_fit, ridge_lambda, rng):
    Xs, Ys = [], []
    idx = rng.permutation(len(train))[:max_fit]
    for i in idx:
        ex = train[i]
        ctrl = ev.control_from_prompt(ex["prompt"])
        if not ctrl:
            continue
        c = sentence_to_rankarr(ctrl, panel_index, P)
        t = sentence_to_rankarr(ex["response"], panel_index, P)
        Xs.append(c); Ys.append(t - c)
    X, Y = np.asarray(Xs), np.asarray(Ys)
    mu_c, mu_s = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_c, Y - mu_s
    XtX = Xc.T @ Xc
    lam = ridge_lambda * (np.trace(XtX) / P + 1e-9)
    W = np.linalg.solve(XtX + lam * np.eye(P), Xc.T @ Yc)
    return mu_c, mu_s, W


# ------------------------------ train-only per-drug displacement lookup (auditor A-03) -------------
def fit_drug_displacement(train, panel_index, P, ev, min_cells=5):
    """The simplest predictor that actually USES drug identity: the drug's mean rank displacement
    (response - control), measured on TRAIN CELLS ONLY. Returns {(drug, cell_line): delta} plus the
    cell-line-POOLED {(drug, None): delta} used for cross-context transfer when the (drug,cell_line)
    pair is unseen. Eval tiers live in separate files, so no evaluation response can contribute to any
    fitted delta (the auditor's explicit leak requirement)."""
    from collections import defaultdict
    acc = defaultdict(lambda: [np.zeros(P), 0])
    for ex in train:
        m = ex.get("metadata", {}) or {}
        d, cl = m.get("drug"), m.get("cell_line_id")
        if not d:
            continue
        ctrl = ev.control_from_prompt(ex["prompt"])
        if not ctrl:
            continue
        c = sentence_to_rankarr(ctrl, panel_index, P)
        t = sentence_to_rankarr(ex["response"], panel_index, P)
        s = t - c
        acc[(d, cl)][0] += s; acc[(d, cl)][1] += 1
        acc[(d, None)][0] += s; acc[(d, None)][1] += 1
    delta = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= min_cells}
    n_dc = sum(1 for k in delta if k[1] is not None)
    logger.info(f"drug-displacement lookup fitted on TRAIN ONLY: {n_dc} (drug,cell_line) entries + "
                f"{len(delta) - n_dc} pooled-drug entries (>= {min_cells} cells each)")
    return delta


# ----------------------------------------------------------------- benchmark one cell line
def score_cellline(by_drug, panel_index, P, lm, model_pb_fn, lin_fn, rng, scram_pb_fn=None,
                   lookup_fn=None):
    """by_drug: {drug: {"resp":[sentences], "ctrl":[sentences]}}. Returns per-drug NIR per predictor.

    CONSISTENT DENOISING is essential: split each drug into two disjoint halves; use half A as EVERY
    drug's held-out truth, and half B as the ceiling's real-replicate predictor. All predictors
    (ceiling / model / linear / mean) are then scored against the same half-A truths at the same noise
    level. (The earlier version compared the ceiling's own noisy half to other drugs' clean FULL
    profiles, which inverted it.)"""
    drugs0 = list(by_drug.keys())
    A, B = {}, {}
    for d in drugs0:
        if by_drug[d].get("truth_A") is not None and by_drug[d].get("truth_B") is not None:
            A[d] = list(by_drug[d]["truth_A"])
            B[d] = list(by_drug[d]["truth_B"])
            continue
        resp = by_drug[d]["resp"]
        if len(resp) < 4:
            continue
        idx = list(range(len(resp))); rng.shuffle(idx); h = len(idx) // 2
        A[d] = [resp[i] for i in idx[:h]]
        B[d] = [resp[i] for i in idx[h:]]
    drugs = list(A.keys())
    if len(drugs) < 3:
        return [], {"model": {}, "truth": {}}
    truth_rank = {d: pb_rank(A[d], panel_index, P) for d in drugs}          # held-out truth = half A
    truth_expr = {d: pb_expr(A[d], panel_index, P, lm) for d in drugs}
    expressed = {d: np.where(truth_rank[d] < P)[0] for d in drugs}          # all expressed genes of the truth
    ceil_rank = {d: pb_rank(B[d], panel_index, P) for d in drugs}           # disjoint real replicate = half B
    ceil_expr = {d: pb_expr(B[d], panel_index, P, lm) for d in drugs}

    rows, profiles = [], {"model": {}, "truth": {}}
    for d in drugs:
        others = [dd for dd in drugs if dd != d]
        if len(others) < 2:
            continue
        preds = {"ceiling": (ceil_rank[d], ceil_expr[d]),
                 "linear": lin_fn(by_drug[d]["ctrl"]),
                 "mean": (np.mean(np.stack([truth_rank[o] for o in others]), axis=0),
                          np.mean(np.stack([truth_expr[o] for o in others]), axis=0))}
        # CONTROL-COPY leakage baseline: the drug's own plate-matched control pseudobulk, unmodified.
        # It contains ZERO drug information, so it must score ~0.50. If it scores above chance, the
        # control itself carries drug/plate identity, and then ANY control-conditioned predictor
        # (including the model) gets NIR for free — i.e. an apparent "drug effect" that is batch leakage.
        ctrl_sents = by_drug[d].get("ctrl") or []
        if ctrl_sents:
            preds["control"] = (pb_rank(ctrl_sents, panel_index, P),
                                pb_expr(ctrl_sents, panel_index, P, lm))
        # DRUG-LOOKUP baseline (A-03): control pseudobulk + the drug's TRAIN-only mean displacement.
        # The simplest predictor that uses drug identity -> the yardstick the model must beat. Returns
        # None when the drug is unseen in train (e.g. tier2), where no lookup is legitimately possible.
        if lookup_fn is not None and ctrl_sents:
            _lk = lookup_fn(d, ctrl_sents)
            if _lk is not None:
                preds["drug_lookup"] = _lk
        mr, me = model_pb_fn(d)
        if mr is not None:
            preds["model"] = (mr, me)
        # SCRAMBLE arm: same control cell, same truth, only the drug token in the prompt is swapped
        # to a different-mechanism drug. Drug knowledge => scramble scores LOWER than model.
        # Plate leakage => scramble ~= model (the control never moved).
        if scram_pb_fn is not None:
            sr, se = scram_pb_fn(d)
            if sr is not None:
                preds["scramble"] = (sr, se)

        # per-drug identity is kept on the row so the aggregate can be decomposed later
        # (drug_stratify_geometry.py): which drugs the model wins/loses on, vs their difficulty.
        row = {"drug": d, "n_cells": len(by_drug[d]["resp"])}
        frozen = by_drug[d].get("manifest_row")
        if frozen:
            row.update({
                "row_id": frozen["row_id"], "group_id": frozen["group_id"],
                "dose_values": frozen.get("dose_values", []),
                "treatment_well_ids": frozen.get("treatment_well_ids", []),
                "k_collapsed": len(frozen.get("prompt_specs", [])),
            })
        exp_idx = expressed[d]
        for name, (pr, pe) in preds.items():
            s_own = rank_corr(pr, truth_rank[d], exp_idx)
            s_oth = [rank_corr(pr, truth_rank[o], exp_idx) for o in others]
            d_own = float(np.linalg.norm(pe - truth_expr[d]))
            d_oth = [float(np.linalg.norm(pe - truth_expr[o])) for o in others]
            row[name] = {"nir_rank": nir_from_sims(s_own, s_oth),
                         "nir_expr": nir_from_dists(d_own, d_oth)}
        rows.append(row)
        if "model" in preds:
            profiles["model"][d] = preds["model"][1]      # predicted pseudobulk (expression)
        profiles["truth"][d] = truth_expr[d]              # real held-out pseudobulk (half A)
    return rows, profiles


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic: drug-aware predictor should get NIR ~1 (identifies its drug); drug-agnostic
    predictor (same output for all) ~0.5 (chance), under BOTH distances."""
    rng = np.random.RandomState(0)
    P = 300
    panel = [f"G{i}" for i in range(P)]
    lm = {"slope": -0.4, "intercept": 1.6}
    pidx = {g: i for i, g in enumerate(panel)}

    # each drug: a fixed pool + a fixed per-gene expression level, so cells of the same drug emit genes
    # in a CONSISTENT rank order (real model outputs are ordered by expression; random order can't test
    # a rank metric). Different drugs -> different genes/order -> low cross-drug similarity.
    pools, levels = {}, {}
    for d in range(15):
        pool = rng.choice(P, 130, replace=False)
        pools[f"d{d}"] = pool
        levels[f"d{d}"] = {int(g): rng.rand() for g in pool}

    def make_cell(dname):
        pool, lev = pools[dname], levels[dname]
        drawn = sorted(rng.choice(pool, 90, replace=False), key=lambda g: -lev[int(g)])
        return " ".join(panel[g] for g in drawn) + " " + SENTINEL

    def make_random():
        genes = sorted(rng.choice(P, 90, replace=False))
        return " ".join(panel[g] for g in genes) + " " + SENTINEL

    by_drug = {f"d{d}": {"resp": [make_cell(f"d{d}") for _ in range(20)],
                         "ctrl": [make_random() for _ in range(20)]} for d in range(15)}

    def model_aware(d):
        s = [make_cell(d) for _ in range(8)]
        return pb_rank(s, pidx, P), pb_expr(s, pidx, P, lm)
    fixed = [make_random() for _ in range(8)]           # drug-AGNOSTIC: same output for every drug
    def lin_agnostic(ctrl_sents):
        return pb_rank(fixed, pidx, P), pb_expr(fixed, pidx, P, lm)

    # scramble arm: a drug-AWARE model fed the WRONG drug -> generates the wrong drug's cell
    def scram_aware(d):
        wrong = f"d{(int(d[1:]) + 1) % 15}"
        s = [make_cell(wrong) for _ in range(8)]
        return pb_rank(s, pidx, P), pb_expr(s, pidx, P, lm)

    rows, _ = score_cellline(by_drug, pidx, P, lm, model_aware, lin_agnostic, rng,
                             scram_pb_fn=scram_aware)
    def agg(name, key):
        v = [r[name][key] for r in rows if name in r and r[name][key] is not None]
        return float(np.mean(v)) if v else None
    m_rank, m_expr = agg("model", "nir_rank"), agg("model", "nir_expr")
    l_rank, l_expr = agg("linear", "nir_rank"), agg("linear", "nir_expr")
    s_rank, s_expr = agg("scramble", "nir_rank"), agg("scramble", "nir_expr")
    c_expr = agg("control", "nir_expr")
    logger.info(f"  drug-AWARE model  NIR: rank={m_rank:.3f} expr={m_expr:.3f} (expect ~1, identifies drug)")
    logger.info(f"  drug-AGNOSTIC lin NIR: rank={l_rank:.3f} expr={l_expr:.3f} (expect ~chance, cannot discriminate)")
    logger.info(f"  SCRAMBLE (wrong drug) NIR: rank={s_rank:.3f} expr={s_expr:.3f} "
                f"(expect << model: lying about the drug must destroy discrimination)")
    logger.info(f"  CONTROL-copy NIR: expr={c_expr if c_expr is None else f'{c_expr:.3f}'} "
                f"(expect ~chance: the control carries no drug identity in this synthetic data)")
    # machinery is right if: the drug-aware predictor identifies its drug, the agnostic one cannot,
    # and the scramble arm collapses (proving the arm can detect genuine drug USE, not just fit)
    ok = (m_rank > 0.85 and m_expr > 0.85 and l_rank < m_rank - 0.25 and l_expr < m_expr - 0.25
          and l_rank < 0.7 and l_expr < 0.7
          and s_expr is not None and s_expr < m_expr - 0.3)
    out = {"selftest": True, "passed": bool(ok),
           "model": {"nir_rank": m_rank, "nir_expr": m_expr},
           "linear": {"nir_rank": l_rank, "nir_expr": l_expr},
           "scramble": {"nir_rank": s_rank, "nir_expr": s_expr},
           "control": {"nir_expr": c_expr}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    atomic_write_json(args.out, out)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def load_tier_by_drug(eval_dir, tier, ev, same_plate=False, exclude_lines=None):
    """-> {group_key: {drug: {...}}}, where group_key = (cell_line, plate) if same_plate else
    (cell_line, None).

    WHY same_plate MATTERS (plate/batch leakage): controls are plate-matched and, in this design,
    each drug sits on its own plate(s) — so WITHIN a cell line, 'which plate' is close to a proxy for
    'which drug'. A predictor conditioned on the plate-matched control therefore inherits the plate
    signature and scores above chance on NIR with ZERO drug knowledge (measured: control-copy 0.659).
    Restricting every comparison set to drugs on the SAME plate holds the plate signature constant
    across all candidates, so batch identity carries no information and the leak is structurally
    impossible."""
    path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
    if not os.path.exists(path):
        logger.warning(f"  missing {path}")
        return None
    by_cl = defaultdict(lambda: defaultdict(lambda: {"resp": [], "ctrl": [], "prompts": []}))
    n_no_plate = n_excluded = 0
    for _li, line in enumerate(open(path)):
        # SPLIT-SAMPLE: skip the line indices reserved for SELECTION (perturbation_strength.py), so
        # the cells used to DEFINE strength/distinctiveness are disjoint from the cells the model is
        # SCORED against. Without this, subsetting on a selection statistic is circular (double dipping).
        if exclude_lines is not None and _li in exclude_lines:
            n_excluded += 1
            continue
        ex = json.loads(line)
        m = ex.get("metadata", {})
        cl, drug, plate = m.get("cell_line_id"), m.get("drug"), m.get("plate")
        ctrl = ev.control_from_prompt(ex["prompt"])
        if cl is None or drug is None or not ctrl:
            continue
        if same_plate and plate is None:
            n_no_plate += 1
            continue
        key = (cl, plate) if same_plate else (cl, None)
        slot = by_cl[key][drug]
        slot["resp"].append(ex["response"]); slot["ctrl"].append(ctrl); slot["prompts"].append(ex["prompt"])
    if n_no_plate:
        logger.warning(f"  {n_no_plate} rows dropped (no plate in metadata)")
    if n_excluded:
        logger.info(f"  [{tier}] excluded {n_excluded} SELECTION cells (split-sample; scoring only "
                    f"on disjoint evaluation cells)")
    return by_cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--temp_sweep", action="store_true")
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--no_model", action="store_true",
                    help="skip generation entirely; score only ceiling/linear/mean/control (all from "
                         "real cells). Runs on CPU in minutes — use it with --same_plate_only to test "
                         "whether the control-copy leak disappears and whether the ceiling survives.")
    ap.add_argument("--scram_dir", default=None,
                    help="scrambled-drug eval dir (make_scramble_endcell.py output). Adds the "
                         "'scramble' arm: same control + same truth, only the drug token swapped. "
                         "model >> scramble => real drug use; model ~= scramble => plate leakage.")
    ap.add_argument("--manifest", default=None,
                    help="frozen manifest from freeze_nir_manifest.py; enables deterministic, paired, "
                         "cache-backed evaluation and refuses source/config drift")
    ap.add_argument("--prediction_cache", default=None,
                    help="append-only generation JSONL; required with --manifest when a model is run")
    ap.add_argument("--model_fingerprint", default=None,
                    help="immutable checkpoint fingerprint used in per-output seed derivation")
    ap.add_argument("--parent_validity_only", action="store_true",
                    help="declare this frozen parent validity-only; an absent atomic END_CELL writes "
                         "validity_failure and NIR is never computed")
    ap.add_argument("--attention_implementation", choices=["eager", "sdpa", "flash_attention_2"],
                    default=None, help="explicit inference backend; hardened Gemma jobs use eager")
    ap.add_argument("--generation_contract_version", default=GENERATION_CONTRACT_VERSION)
    ap.add_argument("--invalid_policy", choices=["score_as_emitted", "validity_failure"],
                    default="score_as_emitted",
                    help="retain every output; validity_failure writes an audit artifact and skips NIR "
                         "when the predeclared gate fails")
    ap.add_argument("--min_complete_rate", type=float, default=0.95)
    ap.add_argument("--min_recognized_gene_rate", type=float, default=0.95)
    ap.add_argument("--min_recognized_genes", type=int, default=20)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--drug_lookup", action="store_true",
                    help="add the A-03 DRUG-SPECIFIC baseline: control pseudobulk + the drug's "
                         "TRAIN-only mean displacement (per drug x cell_line, pooled-drug fallback). "
                         "The simplest predictor that uses drug identity -> the yardstick the model "
                         "must beat. NA where the drug is unseen in train (e.g. tier2).")
    ap.add_argument("--lookup_min_cells", type=int, default=5,
                    help="min train cells per (drug,cell_line) to fit its displacement")
    ap.add_argument("--tiers", default="tier2_unseen_drugs,tier3_unseen_combos")
    ap.add_argument("--tier", default="tier2_unseen_drugs", help="single tier for --temp_sweep")
    ap.add_argument("--temps", default="0,0.5,0.8,1.0")
    ap.add_argument("--k_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--gen_batch_size", type=int, default=48)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--min_drugs_per_cl", type=int, default=6,
                    help="min drugs per comparison group. With --same_plate_only, groups are "
                         "(cell_line, plate) and hold ~4 drugs in tier2, so use 3.")
    ap.add_argument("--n_celllines", type=int, default=20,
                    help="deprecated alias for --max_groups")
    ap.add_argument("--max_groups", type=int, default=None,
                    help="max comparison groups to score. With --same_plate_only there are many more "
                         "groups (one per cell_line x plate), so raise this (e.g. 200).")
    ap.add_argument("--exclude_manifest", default=None,
                    help="split_manifest.json from perturbation_strength.py: line indices reserved "
                         "for SELECTION. Excluding them makes selection (strength/distinctiveness) "
                         "and evaluation (model scoring) disjoint — required to avoid circular "
                         "analysis when stratifying on a selection statistic.")
    ap.add_argument("--same_plate_only", action="store_true",
                    help="LEAKAGE FIX: restrict every NIR comparison set to drugs on the SAME "
                         "(cell_line, plate). Drug and plate are confounded by the experimental "
                         "design (each drug sits on its own plate), so cross-plate comparisons let a "
                         "control-conditioned predictor identify the drug from batch alone "
                         "(control-copy scores 0.659 vs 0.50 chance). Same-plate comparisons hold the "
                         "plate signature constant, making the leak structurally impossible.")
    ap.add_argument("--max_fit", type=int, default=40000)
    ap.add_argument("--ridge_lambda", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--profiles", default=None,
                    help="optional .npz of per-drug model + truth pseudobulk profiles, consumed by "
                         "drug_stratify_geometry.py (Test 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_groups is None:                       # back-compat with --n_celllines
        args.max_groups = args.n_celllines

    if args.selftest:
        selftest(args)
        return

    if args.manifest and not args.no_model:
        if not args.prediction_cache or not args.model_fingerprint:
            ap.error("--manifest model evaluation requires --prediction_cache and --model_fingerprint")

    import evaluate_c2s_tahoe as ev

    panel_path = os.path.join(args.eval_dir, "l1000_panel.json")
    panel = json.load(open(panel_path))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    lm_path = os.path.join(args.eval_dir, "linear_model.json")
    lm = json.load(open(lm_path))
    logger.info(f"Panel {P}; linear_model slope={lm['slope']:.3f} intercept={lm['intercept']:.3f}")

    # --no_model: skip all generation. ceiling / linear / mean / control are computed from REAL cells
    # only, so the plate-leakage diagnostic (does control-copy fall to ~0.50 under --same_plate_only?)
    # and the clean ceiling need no GPU at all — minutes on CPU instead of hours.
    generate = None
    generate_specs = None
    generation_records = []
    cache = None
    if args.no_model:
        logger.info("--no_model: skipping generation; scoring ceiling/linear/mean/control only")
        if args.scram_dir:
            logger.warning("  --scram_dir ignored under --no_model (the scramble arm needs generation)")
            args.scram_dir = None
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not args.model_path:
            raise SystemExit("--model_path is required unless --no_model")
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
            end_id = atomic_sentinel_id(tok)
        except ValueError as exc:
            if args.manifest or args.parent_validity_only:
                manifest_hash = None
                if args.manifest:
                    with open(args.manifest, encoding="utf-8") as handle:
                        parent_manifest = json.load(handle)
                    frozen_tier = parent_manifest["config"]["tier"]
                    eval_file = os.path.join(args.eval_dir, f"eval_{frozen_tier}.jsonl")
                    scram_file = (os.path.join(args.scram_dir, f"eval_{frozen_tier}.jsonl")
                                  if args.scram_dir else None)
                    verify_manifest(parent_manifest, eval_file, scram_file, args.train_file)
                    declaration = next((x for x in parent_manifest.get("model_declarations", [])
                                        if x.get("fingerprint") == args.model_fingerprint), None)
                    expected_role = "validity_only_parent" if args.parent_validity_only else "scored"
                    if declaration is None or declaration.get("role") != expected_role:
                        raise ValueError("failed-tokenizer model is not frozen under the requested role")
                    manifest_hash = parent_manifest["manifest_sha256"]
                failure = {
                    "status": "validity_failure", "tiers": {},
                    "failure_reason": str(exc), "failure_stage": "tokenizer_contract",
                    "nir_computed": False, "parent_validity_only": args.parent_validity_only,
                    "model_fingerprint": args.model_fingerprint,
                    "manifest_sha256": manifest_hash,
                    "config": {k: v for k, v in vars(args).items()},
                }
                atomic_write_json(args.out, failure, default=float)
                logger.warning("atomic END_CELL contract failed; wrote validity-failure artifact")
                return
            raise
        eos = [end_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])

        if args.manifest:
            cache_contract = {
                "model_fingerprint": args.model_fingerprint,
                "generation_contract_version": args.generation_contract_version,
                "temperature": args.temperature, "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0,
                "end_cell_id": end_id, "native_eos_id": tok.eos_token_id,
                "manifest_sha256": None,  # filled after manifest verification below
                "panel_sha256": file_sha256(panel_path),
                "min_recognized_genes": args.min_recognized_genes,
                "min_complete_rate": args.min_complete_rate,
                "min_recognized_gene_rate": args.min_recognized_gene_rate,
                "invalid_policy": args.invalid_policy,
                "attention_implementation": args.attention_implementation,
                "use_cache": True,
            }
            # The manifest is loaded below; defer cache opening so its hash is part of the contract.
            pending_cache_contract = cache_contract

            def generate_specs(specs, arm="model"):
                """Generate each frozen output in its own seed stream; batch/order/resume invariant."""
                prev = tok.padding_side
                tok.padding_side = "left"
                outputs = []
                try:
                    for spec in specs:
                        prompt_spec = spec.get("wrong_condition") if arm == "wrong_condition" else spec
                        if not prompt_spec:
                            continue
                        prompt_id = prompt_spec["prompt_id"]
                        row_id = spec["row_id"]
                        draw_id = int(spec["draw_id"])
                        frozen_seed = prompt_spec.get("seeds_by_model", {}).get(args.model_fingerprint)
                        seed = stable_output_seed(args.model_fingerprint, row_id, prompt_id, draw_id,
                                                  args.generation_contract_version)
                        if frozen_seed is not None and int(frozen_seed) != seed:
                            raise ValueError(f"frozen seed mismatch for {row_id}/{prompt_id}")
                        output_id = sha256_json({
                            "model": args.model_fingerprint, "row_id": row_id,
                            "prompt_id": prompt_id, "draw_id": draw_id, "arm": arm,
                            "contract": args.generation_contract_version,
                        })
                        old = cache.get(output_id)
                        if old is not None:
                            if old.get("prompt_sha256") != sha256_text(prompt_spec["prompt"]):
                                raise ValueError(f"cached prompt mismatch for {output_id}")
                            record = old
                        else:
                            enc = tok(prompt_spec["prompt"], return_tensors="pt").to(device)
                            devices = ([torch.cuda.current_device()] if str(device).startswith("cuda")
                                       else [])
                            with torch.random.fork_rng(devices=devices):
                                torch.manual_seed(seed)
                                if devices:
                                    torch.cuda.manual_seed_all(seed)
                                with torch.no_grad():
                                    generated = model.generate(
                                        **enc, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                        do_sample=(args.temperature > 0),
                                        temperature=max(args.temperature, 1e-2), top_p=args.top_p)
                            raw_ids = generated[0, enc["input_ids"].shape[1]:].tolist()
                            decoded = tok.decode(raw_ids, skip_special_tokens=False).strip()
                            val = generation_validity(
                                raw_ids, decoded, end_id=end_id, eos_id=tok.eos_token_id,
                                max_new_tokens=args.max_new_tokens, panel_index=panel_index,
                                model_kind="cellsentence", min_genes=args.min_recognized_genes)
                            record = {
                                "type": "prediction", "output_id": output_id, "row_id": row_id,
                                "prompt_id": prompt_id, "prompt_sha256": sha256_text(prompt_spec["prompt"]),
                                "draw_id": draw_id, "arm": arm, "seed": seed,
                                "raw_token_ids": list(map(int, raw_ids)), "decoded_text": decoded,
                                "termination_reason": val["termination"], "validity": val,
                            }
                            cache.put(record)
                        generation_records.append(record)
                        outputs.append(record["decoded_text"])
                finally:
                    tok.padding_side = prev
                return outputs

        def generate(prompts, temperature):
            prev = tok.padding_side; tok.padding_side = "left"
            outs = []
            try:
                for i in range(0, len(prompts), args.gen_batch_size):
                    batch = prompts[i:i + args.gen_batch_size]
                    enc = tok(batch, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                           pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                           do_sample=(temperature > 0), temperature=max(temperature, 1e-2),
                                           top_p=args.top_p)
                    plen = enc["input_ids"].shape[1]
                    for j in range(len(batch)):
                        ids = g[j][plen:].tolist()
                        # Decode exactly what was emitted.  In particular, never manufacture
                        # [END_CELL] for a native-EOS or max-token termination.
                        outs.append(tok.decode(ids, skip_special_tokens=False).strip())
            finally:
                tok.padding_side = prev
            return outs

    if args.temp_sweep:
        run_temp_sweep(args, ev, panel_index, P, generate)
        return

    # fit the drug-agnostic ridge linear once
    logger.info("Fitting ridge control->shift on train ...")
    train = [json.loads(l) for l in open(args.train_file)]
    mu_c, mu_s, W = fit_ridge(train, panel_index, P, ev, args.max_fit, args.ridge_lambda,
                              np.random.RandomState(args.seed))

    def lin_fn(ctrl_sents):
        c = pb_rank(ctrl_sents, panel_index, P)
        pred = c + mu_s + (c - mu_c) @ W
        # decode the predicted rank profile to expression (approx: order genes by predicted rank)
        order = np.argsort(pred)
        expr = np.zeros(P)
        for r, gi in enumerate(order, 1):
            if pred[gi] < P:  # treat fill-level as absent
                expr[gi] = max(0.0, lm["slope"] * np.log10(r) + lm["intercept"])
        return pred, expr

    # drug-specific lookup baseline (A-03), fitted on TRAIN only
    DELTA = fit_drug_displacement(train, panel_index, P, ev, args.lookup_min_cells) \
        if args.drug_lookup else None

    def lookup_fn_base(drug, cell_line, ctrl_sents):
        if not DELTA:
            return None
        dl = DELTA.get((drug, cell_line))
        if dl is None:
            dl = DELTA.get((drug, None))     # fall back to cell-line-pooled = cross-context transfer
        if dl is None:
            return None                      # drug unseen in train -> no legitimate lookup (tier2)
        c = pb_rank(ctrl_sents, panel_index, P)
        pred = c + dl
        order = np.argsort(pred)
        expr = np.zeros(P)
        for r, gi in enumerate(order, 1):
            if pred[gi] < P:
                expr[gi] = max(0.0, lm["slope"] * np.log10(r) + lm["intercept"])
        return pred, expr

    # split-sample manifest: line indices reserved for SELECTION, to be excluded from scoring
    _exclude_manifest = None
    if args.exclude_manifest:
        _exclude_manifest = json.load(open(args.exclude_manifest))
        logger.info(f"Split-sample: excluding selection cells listed in {args.exclude_manifest} "
                    f"({ {k: len(v) for k, v in _exclude_manifest.items()} })")

    frozen_manifest = None
    if args.manifest:
        if args.exclude_manifest:
            raise ValueError("--manifest already freezes support; --exclude_manifest cannot be combined")
        with open(args.manifest, encoding="utf-8") as handle:
            frozen_manifest = json.load(handle)
        if frozen_manifest.get("generation_contract_version") != args.generation_contract_version:
            raise ValueError("generation-contract version differs from frozen manifest")
        if len({r.get("tier") for r in frozen_manifest.get("rows", [])}) != 1:
            raise ValueError("one frozen manifest must contain exactly one tier")
        frozen_tier = frozen_manifest["config"]["tier"]
        requested = [x.strip() for x in args.tiers.split(",") if x.strip()]
        if requested != [frozen_tier]:
            raise ValueError(f"--tiers {requested} differs from frozen tier {frozen_tier}")
        eval_file = os.path.join(args.eval_dir, f"eval_{frozen_tier}.jsonl")
        scram_file = (os.path.join(args.scram_dir, f"eval_{frozen_tier}.jsonl")
                      if args.scram_dir else None)
        verify_manifest(frozen_manifest, eval_file, scram_file, args.train_file)
        expected_cfg = {
            "k_samples": args.k_samples, "min_cells": args.min_cells,
            "min_drugs_per_group": args.min_drugs_per_cl,
            "max_groups": args.max_groups, "same_plate_only": args.same_plate_only,
            "seed": args.seed,
        }
        for key, value in expected_cfg.items():
            if frozen_manifest["config"].get(key) != value:
                raise ValueError(f"runtime {key}={value!r} differs from manifest "
                                 f"{frozen_manifest['config'].get(key)!r}")
        if not args.no_model:
            declarations = frozen_manifest.get("model_declarations", [])
            declaration = next((x for x in declarations
                                if x.get("fingerprint") == args.model_fingerprint), None)
            if declaration is None:
                raise ValueError("model fingerprint was not frozen in the manifest")
            expected_role = "validity_only_parent" if args.parent_validity_only else "scored"
            if declaration.get("role") != expected_role:
                raise ValueError(f"model declaration role {declaration.get('role')!r} != {expected_role!r}")
            if args.parent_validity_only:
                failure = {
                    "status": "validity_failure", "tiers": {}, "nir_computed": False,
                    "failure_stage": "parent_gate",
                    "failure_reason": "frozen parent is declared validity-only; NIR intentionally skipped",
                    "manifest_sha256": frozen_manifest["manifest_sha256"],
                    "model_fingerprint": args.model_fingerprint,
                    "config": {k: v for k, v in vars(args).items()},
                }
                atomic_write_json(args.out, failure, default=float)
                logger.warning("validity-only parent declaration: NIR was intentionally skipped")
                return
            pending_cache_contract["manifest_sha256"] = frozen_manifest["manifest_sha256"]
            cache = PredictionCache(args.prediction_cache, pending_cache_contract)
        logger.info(f"frozen manifest verified: {frozen_manifest['manifest_sha256']} "
                    f"({len(frozen_manifest['rows'])} rows)")

    rng = np.random.RandomState(args.seed)
    metric_config = {
        "metric": "tie-aware NIR", "rank_similarity": "Pearson over own-truth expressed genes",
        "expression_distance": "Euclidean", "row_weighting": "equal",
        "same_plate_only": args.same_plate_only, "min_cells": args.min_cells,
        "min_drugs_per_group": args.min_drugs_per_cl,
        "k_samples": args.k_samples, "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "generation_contract_version": args.generation_contract_version,
        "invalid_policy": args.invalid_policy,
        "panel_sha256": file_sha256(panel_path),
        "min_complete_rate": args.min_complete_rate,
        "min_recognized_gene_rate": args.min_recognized_gene_rate,
        "min_recognized_genes": args.min_recognized_genes,
        "attention_implementation": args.attention_implementation,
        "use_cache": True if not args.no_model else None,
    }
    result = {"tiers": {}, "config": {k: v for k, v in vars(args).items()},
              "manifest_sha256": (frozen_manifest or {}).get("manifest_sha256"),
              "metric_config_hash": sha256_json(metric_config), "metric_config": metric_config}
    model_profiles, truth_profiles = {}, {}     # for the drug-geometry test (Test 3)
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        _excl = set(_exclude_manifest.get(tier, [])) if _exclude_manifest else None
        if frozen_manifest:
            eval_file = os.path.join(args.eval_dir, f"eval_{tier}.jsonl")
            by_cl = _manifest_groups(frozen_manifest, eval_file, ev)
        else:
            by_cl = load_tier_by_drug(args.eval_dir, tier, ev, same_plate=args.same_plate_only,
                                      exclude_lines=_excl)
        if not by_cl:
            continue
        scram_by_cl = (None if frozen_manifest else
                       (load_tier_by_drug(args.scram_dir, tier, ev, same_plate=args.same_plate_only,
                                          exclude_lines=_excl) if args.scram_dir else None))
        if args.scram_dir and not scram_by_cl:
            logger.warning(f"  [{tier}] no scramble data in {args.scram_dir} — skipping scramble arm")
        all_rows = []
        used = 0
        for gkey, dd in by_cl.items():
            cl, plate = gkey
            drugs = {d: s for d, s in dd.items() if len(s["resp"]) >= args.min_cells}
            if len(drugs) < args.min_drugs_per_cl:
                continue

            def model_pb_fn(d, _drugs=drugs):
                if generate is None:                      # --no_model
                    return None, None
                if frozen_manifest:
                    specs = _drugs[d]["prompt_specs"]
                    gens = generate_specs(specs, "model")
                else:
                    prompts = _drugs[d]["prompts"][:args.k_samples]
                    if not prompts:
                        return None, None
                    gens = generate(prompts, args.temperature)
                if not gens:
                    return None, None
                return pb_rank(gens, panel_index, P), pb_expr(gens, panel_index, P, lm)

            scram_fn = None
            if frozen_manifest and args.scram_dir:
                def scram_fn(d, _drugs=drugs):
                    specs = _drugs[d]["prompt_specs"]
                    gens = generate_specs(specs, "wrong_condition")
                    if not gens:
                        return None, None
                    return pb_rank(gens, panel_index, P), pb_expr(gens, panel_index, P, lm)
            elif scram_by_cl and gkey in scram_by_cl:
                _sdd = scram_by_cl[gkey]

                def scram_fn(d, _s=_sdd):
                    slot = _s.get(d)
                    if not slot or not slot["prompts"]:
                        return None, None
                    gens = generate(slot["prompts"][:args.k_samples], args.temperature)
                    return pb_rank(gens, panel_index, P), pb_expr(gens, panel_index, P, lm)

            lk_fn = (lambda d, ctrl, _cl=cl: lookup_fn_base(d, _cl, ctrl)) if DELTA else None
            rows, profs = score_cellline(drugs, panel_index, P, lm, model_pb_fn, lin_fn, rng,
                                         scram_pb_fn=scram_fn, lookup_fn=lk_fn)
            # cell_line and plate are kept SEPARATE: the clustered bootstrap must resample CELL LINES
            # (plates within a cell line are still correlated), not (cell_line, plate) groups.
            for r in rows:
                r["cell_line"] = cl
                r["plate"] = plate
                r["tier"] = tier
            all_rows.extend(rows)
            gtag = f"{cl}~{plate}" if plate is not None else str(cl)
            for d, v in profs["model"].items():
                model_profiles[f"{tier}||{gtag}||{d}"] = v
            for d, v in profs["truth"].items():
                truth_profiles[f"{tier}||{gtag}||{d}"] = v
            used += 1
            logger.info(f"  [{tier}] {gtag[:30]:30s} {len(drugs)} drugs -> {len(rows)} scored")
            if used >= args.max_groups:
                break

        agg = {}
        for name in ("model", "scramble", "linear", "drug_lookup", "control", "mean", "ceiling"):
            for key in ("nir_rank", "nir_expr"):
                vals = [r[name][key] for r in all_rows if name in r and r[name][key] is not None]
                agg[f"{name}_{key}"] = (float(np.mean(vals)), len(vals)) if vals else (None, 0)
        # per-drug rows are kept alongside the aggregate so the headline number can be decomposed
        # (aggregates are computed exactly as before — unchanged).
        result["tiers"][tier] = {"n_drugs": len(all_rows), "agg": agg, "rows": all_rows}

    if generation_records:
        by_arm = {arm: summarize_generation([r for r in generation_records if r.get("arm") == arm])
                  for arm in sorted({r.get("arm") for r in generation_records})}
        result["generation_validity"] = by_arm
        model_validity = by_arm.get("model", {"complete_rate": 0.0,
                                               "mean_recognized_gene_rate": 0.0})
        gate_pass = (model_validity["complete_rate"] >= args.min_complete_rate and
                     model_validity["mean_recognized_gene_rate"] >= args.min_recognized_gene_rate)
        result["validity_gate"] = {
            "passed": bool(gate_pass), "policy": args.invalid_policy,
            "min_complete_rate": args.min_complete_rate,
            "min_recognized_gene_rate": args.min_recognized_gene_rate,
        }
        if args.invalid_policy == "validity_failure" and not gate_pass:
            result["status"] = "validity_failure"
            result["failure_reason"] = "checkpoint failed predeclared cell-sentence validity gate"
            result["tiers"] = {}
            logger.warning("validity gate failed; writing validity-failure artifact and skipping NIR")
        else:
            result["status"] = "ok"
    else:
        result["status"] = "no_model" if args.no_model else "legacy_no_generation_audit"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    atomic_write_json(args.out, result, default=float)
    if cache is not None:
        cache.close()
    if args.profiles:
        np.savez_compressed(args.profiles,
                            **{f"model||{k}": v for k, v in model_profiles.items()},
                            **{f"truth||{k}": v for k, v in truth_profiles.items()})
        logger.info(f"-> {args.profiles}  ({len(model_profiles)} model + {len(truth_profiles)} truth "
                    f"profiles for the geometry test)")

    logger.info("")
    logger.info("=" * 100)
    logger.info("  NIR BENCHMARK (chance ~0.50; ceiling = achievable) — rank-NIR | expr-NIR")
    for tier, td in result["tiers"].items():
        logger.info(f"  [{tier}] (n={td['n_drugs']})")
        for name in ("model", "scramble", "linear", "drug_lookup", "control", "mean", "ceiling"):
            r = td["agg"][f"{name}_nir_rank"][0]
            e = td["agg"][f"{name}_nir_expr"][0]
            g = lambda x: f"{x:.3f}" if x is not None else "NA"
            logger.info(f"      {name:8s}  rank-NIR={g(r)}   expr-NIR={g(e)}")
    logger.info("=" * 100)
    logger.info("  Read: ceiling >> chance (drug signal exists); model ~ linear ~ mean ~ chance => drug-blind.")
    logger.info("  CONTROL arm: the drug's own control, zero drug info -> MUST be ~0.50. Above chance")
    logger.info("               => the control leaks drug/plate identity and inflates every")
    logger.info("               control-conditioned predictor (including the model).")
    logger.info("  SCRAMBLE arm: same control, wrong drug token. model >> scramble => genuine drug use;")
    logger.info("               model ~= scramble => the apparent drug effect is NOT from the drug.")
    logger.info(f"-> {args.out}")


def run_temp_sweep(args, ev, panel_index, P, generate):
    """Diagnostic: for a few (drug,control) contexts, K samples at each temperature; report the
    'core' genes (present in ALL K samples), 'variable' genes (in exactly one), and mean pairwise
    Jaccard. Shows whether higher temperature adds signal or just noise."""
    by_cl = load_tier_by_drug(args.eval_dir, args.tier, ev)
    contexts = []
    for cl, dd in by_cl.items():
        for d, s in dd.items():
            if s["prompts"]:
                contexts.append(s["prompts"][0])
        if len(contexts) >= 12:
            break
    contexts = contexts[:12]
    temps = [float(x) for x in args.temps.split(",")]
    out = {"temps": temps, "per_temp": {}}
    for T in temps:
        cores, variables, jaccs = [], [], []
        for prompt in contexts:
            gens = generate([prompt] * args.k_samples, T)
            sets = [set(genes_of(g)) for g in gens]
            allg = set().union(*sets) if sets else set()
            core = set.intersection(*sets) if sets else set()
            counts = {g: sum(g in s for s in sets) for g in allg}
            variable = [g for g, c in counts.items() if c == 1]
            cores.append(len(core)); variables.append(len(variable))
            js = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
                  for i in range(len(sets)) for j in range(i + 1, len(sets))]
            jaccs.append(float(np.mean(js)) if js else None)
        out["per_temp"][str(T)] = {
            "mean_core_genes": float(np.mean(cores)), "mean_variable_genes": float(np.mean(variables)),
            "mean_pairwise_jaccard": float(np.mean([j for j in jaccs if j is not None]))}
        logger.info(f"  T={T}: core(all K)={np.mean(cores):.0f}  variable(1 of K)={np.mean(variables):.0f}  "
                    f"pairwise-Jaccard={out['per_temp'][str(T)]['mean_pairwise_jaccard']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    atomic_write_json(args.out, out)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()

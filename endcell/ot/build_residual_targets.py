#!/usr/bin/env python
r"""
build_residual_targets.py — Arm 1b stage 1: drug-specific RESIDUAL training targets
====================================================================================
Measured rationale (target_divergence.py, 6,602 conditions / 125 groups): the current cell-sentence
target encodes almost no drug identity — only **34.6 of 200 tokens differ** between two drugs in the
same context, because ranking genes by expression puts the same housekeeping genes on top for every
drug. Under a drug-specific RESIDUAL the same data yields **120.3 of 200 tokens differing**. The
information was never missing (replicate retrieval ceiling ~0.742 under every representation) — the
TOKENIZATION discarded it. This builder changes what the tokens encode.

  NOTE ON THE TOKEN COUNTS ABOVE. `target_divergence.py` compares gene SETS, not orders. A rank
  sentence can share every gene with another and still differ in every position, so the direction of
  that error is not established. The rebuild does not depend on the number; it is retained here for
  provenance only.

TARGET
  residual = (treated_pseudobulk - control_pseudobulk) - generic(scope, excluding this drug)
    * the control subtraction removes the cell's own state;
    * the generic subtraction removes the drug-AGNOSTIC program (Q8: ~0.26 of "skill" matched by
      trivial baselines). What remains is what makes THIS drug different.

ORDER OF OPERATIONS — this is the part that was wrong and is the reason for the rebuild
  inventory  -> split -> fit -> transform
  Every step reads only what the step before it is allowed to expose:
    inventory   metadata and cell COUNTS only. No expression is aggregated.
    split       assigned from metadata only, before a single pseudobulk is computed.
    fit         the generic and its drug inventory come from TRAIN conditions only.
    transform   held-out conditions are projected through the fitted generic; they never enter it.
  The previous build fitted the generic over every condition and assigned the split afterwards, so
  each held-out target was defined partly by the other held-out conditions. `--emit_fit_digest`
  writes a hash of everything fitted, which `tests/test_split_before_fit.py` uses as its gate.

THREE DEFECTS FIXED IN THE SAME PASS
  1. LEAKAGE      generic + reliability filter now fitted on train only (above).
  2. PLATE SCOPE  the generic defaults to (cell_line, plate) scope. Cell-line scope leaves same-plate
                  structure at +0.478 against -0.018 at plate scope, and the shipped targets were
                  built the contaminated way; FINDINGS.md:532 concludes plate scope is the better
                  build. The old counter-argument -- a plate holds too few drugs, so plate scope
                  loses most of the data -- rested on a SHARED-control reliability measurement and is
                  retracted with it: under split controls retention is comparable, 20% at plate scope
                  against 16% at cell-line scope. Both are low in absolute terms, so `--shrink_k`
                  still blends the plate generic toward the cell-line one with weight n/(n+k), where
                  n is the number of OTHER training drugs in the plate group (leave-one-drug-out
                  applies here too) — the standard hierarchical compromise. The report prints the
                  reliability retention at every setting so the choice is made on numbers.
  3. DRUG WEIGHT  the generic is a mean over DRUGS: each drug's doses and plates are averaged first.
                  Previously it was a mean over conditions, so a five-dose drug carried five times
                  the weight of a single-dose drug.
  Plus LEAVE-ONE-DRUG-OUT: a condition's own drug is excluded from its own generic. Without it a
  train condition is centred on a mean containing itself while a held-out condition is not, and the
  two splits would be scored against differently-defined targets.

EXPERIMENTAL UNIT (see shared/tahoe_design.py)
  Tahoe assigns a treatment to a SAMPLE/well holding a mixture of cell lines. The caches stored the
  sample identifier in a column named `dose`, so nothing downstream ever held a concentration. This
  builder resolves the sample identifier explicitly, recovers a real molar dose from
  `drugname_drugconc`, refuses to emit a dose field that is actually a sample identifier, and drops
  combination samples instead of silently analysing them as their first component.

SENTENCE ENCODING (sign is biology, so we keep it)
  "<up genes, most up-regulated first> [DOWN] <down genes, most down-regulated first> [END_CELL]"
  i.e. a signed DE signature. NOTE: register [DOWN] as a special token in the trainer alongside
  [END_CELL] (both are plain strings that cannot collide with a gene symbol).

RELIABILITY FILTER (decided: filter, not weight)
  Keep a condition only if its residual reproduces across a half-split: cos(res_A, res_B) > --repro_thr.
  The halves are measured against SPLIT controls, so res_A and res_B share nothing. Sharing one
  control pseudobulk across both halves correlates them for a reason unrelated to the drug, and
  FINDINGS.md:532 retracts an earlier scope comparison for exactly that (62%/19% shared vs
  20%/16% split). See `compute_shifts`.
  This is a MEASUREMENT-QUALITY criterion, not a performance one, and it is applied to held-out
  conditions too — but through the train-fitted generic, and the per-split retention rates are
  reported so the selection is visible.

Also writes `reconstruction.npz` (generic shift per scope group + panel order) so evaluation can
rebuild full profiles:  predicted_treated = control_cell + generic_shift(group) + predicted_residual.

USAGE
  python build_residual_targets.py --selftest
  python build_residual_targets.py --cache_dir /data/.../ot_cache --out_dir /data/.../residual_targets
"""
import argparse, hashlib, json, os, sys, zlib, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# shared/ locally; flat alongside the script on the cluster
for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "shared"),
           os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tahoe_design as td

TAHOE_REPO = "tahoebio/Tahoe-100M"
END = "[END_CELL]"
DOWN = "[DOWN]"


# ----------------------------------------------------------------- sentence encoding
def expr_to_sentence(vec, panel_genes):
    """control cell -> ordinary [END_CELL] sentence (expressed genes, highest expression first)."""
    idx = np.where(vec > 0)[0]
    if len(idx) == 0:
        return END
    order = sorted(idx.tolist(), key=lambda j: (-vec[j], j))
    return " ".join(panel_genes[j] for j in order) + " " + END


def residual_to_sentence(res, panel_genes, k_up, k_down):
    """signed DE signature: up-regulated block, [DOWN], down-regulated block, [END_CELL]."""
    up = [j for j in np.argsort(-res)[:k_up].tolist() if res[j] > 0]
    dn = [j for j in np.argsort(res)[:k_down].tolist() if res[j] < 0]
    parts = [" ".join(panel_genes[j] for j in up), DOWN, " ".join(panel_genes[j] for j in dn), END]
    return " ".join(p for p in parts if p)


def parse_dose(conc_str):
    """Display string for the prompt. Retained under its original name because residual_eval imports
    it; the numeric concentration now comes from `tahoe_design.parse_treatment` instead."""
    st = td.parse_treatment(conc_str)
    if st.primary is None:
        return "unknown"
    return st.primary.dose.display()


def format_prompt(cell_line_name, drug, dose_str, moa, control_sentence, order="drug_first"):
    """Build the prompt. `order` decides where the drug sits relative to generation.

    drug_first (the original): the instruction leads, then the control cell sentence, then the
      generation point. The control sentence is ~123 gene symbols, i.e. several hundred BPE tokens,
      so the drug name ends up buried that far upstream of the first generated token.

    drug_last: the control cell leads and the instruction is moved to sit immediately before
      generation. Nothing about the CONTENT changes -- identical fields, identical values, identical
      target -- only the distance between the drug token and the token that has to condition on it.

    This is a genuinely separate hypothesis from the two already tested. Q13 says the readout is
    direction-blind; Q15 says the target's tokens are diluted. Neither addresses the possibility that
    the drug is simply too far away for attention to carry it to the generation point, and that
    possibility is cheap to falsify.
    """
    if not moa or moa in ("unknown", "nan", "None"):
        moa = "unclear"
    instr = f"Predict the response of {cell_line_name} to {drug} at {dose_str}. Mechanism: {moa}."
    ctrl = f"Control cell: {control_sentence}"
    if order == "drug_last":
        return f"{ctrl}\n{instr}\n\nResponse cell:"
    return f"{instr}\n{ctrl}\n\nResponse cell:"


def load_meta_maps(repo=TAHOE_REPO, meta_dir=None):
    """Exact columns as the preprocessor, so prompts match the model's training/eval prompts.

    `meta_dir` reads the three parquets from disk instead of the Hub, which is what makes this
    builder runnable offline and therefore testable."""
    import pandas as pd
    if meta_dir:
        L = lambda n: pd.read_parquet(os.path.join(meta_dir, n))
    else:
        from huggingface_hub import hf_hub_download
        L = lambda n: pd.read_parquet(hf_hub_download(repo, f"metadata/{n}", repo_type="dataset"))
    cl, dr, sm = L("cell_line_metadata.parquet"), L("drug_metadata.parquet"), L("sample_metadata.parquet")
    cvcl = {}
    for _, r in cl.iterrows():
        cid, nm = r.get("Cell_ID_Cellosaur"), r.get("cell_name")
        if nm is None or (isinstance(nm, float) and pd.isna(nm)):
            nm = str(cid)
        if cid is not None:
            cvcl[str(cid)] = str(nm)
    moa = {str(r.get("drug")): str(r.get("moa-fine", r.get("moa_fine", "unknown")))
           for _, r in dr.iterrows() if r.get("drug")}
    conc = {str(r.get("sample")): str(r.get("drugname_drugconc", "unknown"))
            for _, r in sm.iterrows() if r.get("sample") is not None}
    logger.info(f"meta maps: {len(cvcl)} cell lines, {len(moa)} drugs, {len(conc)} samples")
    return cvcl, moa, conc


# ----------------------------------------------------------------- stage 1: INVENTORY
def inventory(cache_dir, min_treated, min_control, conc_of, drop_combinations=True,
              require_sample_id=True):
    """Metadata and cell COUNTS only -- no expression is aggregated here.

    Returns (conds, ctrl_rows, X, meta, notes). `conds[key]` describes a condition well enough to
    split on it, and nothing in it depends on an outcome."""
    import pandas as pd
    from scipy import sparse
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    X = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    plate = meta["plate"].astype(str).values
    drug = meta["drug"].astype(str).values

    scol, how = td.sample_column(meta)
    if scol is None:
        msg = ("no treatment/well identifier in the cache -- conditions cannot be keyed to a physical "
               "experiment. Rebuild the cache with a `sample_id` column, or pass "
               "--no_require_sample_id to key on the opaque `dose` column and accept that neither "
               "the dose nor the assignment unit is recoverable.")
        if require_sample_id:
            raise SystemExit(msg)
        logger.warning(msg)
        scol = "dose"
    logger.info(f"treatment identifier: column '{scol}' ({how})")
    sample = meta[scol].astype(str).values

    # resolve every sample once: real molar dose, display string, combination flag
    treat = {}
    for sid in set(sample.tolist()):
        treat[sid] = td.parse_treatment(conc_of.get(sid, "unknown"), sid)

    # control pseudobulk groups + the control cell row indices (prompts come from these)
    ctrl_rows = {}
    for g in set(zip(cl[is_ctrl], plate[is_ctrl])):
        idx = np.where(is_ctrl & (cl == g[0]) & (plate == g[1]))[0]
        if len(idx) >= min_control:
            ctrl_rows[g] = idx

    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], cl[i], plate[i], sample[i])].append(i)

    conds, n_combo, n_nodose, n_small, n_noctrl = {}, 0, 0, 0, 0
    for (d, c, p, sid), idxs in by.items():
        st = treat.get(sid)
        if drop_combinations and st is not None and st.is_combination:
            n_combo += 1
            continue
        if len(idxs) < min_treated:
            n_small += 1
            continue
        if (c, p) not in ctrl_rows:
            n_noctrl += 1
            continue
        dose = st.primary.dose if (st is not None and st.primary is not None) else None
        if dose is None or dose.molar is None:
            n_nodose += 1
        conds[(d, c, p, sid)] = {
            "rows": idxs, "group": (c, p), "drug": d, "cell_line": c, "plate": p,
            "sample_id": sid, "n_cells": len(idxs),
            "dose_molar": (dose.molar if dose is not None else None),
            "dose_raw": (dose.raw if dose is not None else "unknown"),
            "dose_display": (dose.display() if dose is not None else "unknown"),
            "is_combination": bool(st.is_combination) if st is not None else False,
        }
    notes = {"sample_id_source": how, "sample_column": scol,
             "n_conditions": len(conds), "n_dropped_combination": n_combo,
             "n_dropped_too_few_cells": n_small, "n_dropped_no_control_group": n_noctrl,
             "n_without_molar_dose": n_nodose,
             "n_samples": len({k[3] for k in conds}),
             "cell_lines_per_sample_max": max([len({k[1] for k in conds if k[3] == s})
                                               for s in {k[3] for k in conds}] or [0])}
    logger.info(f"inventory: {len(conds)} conditions over {notes['n_samples']} treatment samples "
                f"(max {notes['cell_lines_per_sample_max']} cell lines nested in one sample); "
                f"dropped {n_combo} combination, {n_small} too-few-cells, {n_noctrl} no-control-group; "
                f"{n_nodose} conditions have no recoverable molar dose")
    return conds, ctrl_rows, X, meta, notes


# ----------------------------------------------------------------- stage 2: SPLIT
def holdout_from_tiers(conds, tier2_file, tier3_file):
    """Use the ORIGINAL preprocessing's held-out sets so our numbers are directly comparable to every
    prior tier-2 / tier-3 result in the thesis, instead of a private random split.
      tier2_unseen_drugs.jsonl  -> that drug set becomes `unseen_drug`
      tier3_unseen_combos.jsonl -> those (drug, cell_line) pairs become `unseen_combo`
    Returns (split, ho_drugs, ho_combos) or None if the overlap with our cache is too small to test."""
    def read(path, combo=False):
        out = set()
        if not (path and os.path.exists(path)):
            return out
        for line in open(path):
            m = json.loads(line).get("metadata", {})
            d, c = m.get("drug"), m.get("cell_line_id")
            if d:
                out.add((str(d), str(c)) if combo else str(d))
        return out
    t2_drugs, t3_combos = read(tier2_file), read(tier3_file, combo=True)
    cache_drugs = {k[0] for k in conds}
    cache_combos = {(k[0], k[1]) for k in conds}
    ho_drugs = t2_drugs & cache_drugs
    ho_combos = (t3_combos & cache_combos) - {(d, c) for (d, c) in cache_combos if d in ho_drugs}
    logger.info(f"tier-aligned holdout: tier2 file lists {len(t2_drugs)} drugs -> {len(ho_drugs)} present "
                f"in our cache ({len(cache_drugs)} drugs); tier3 lists {len(t3_combos)} combos -> "
                f"{len(ho_combos)} present")
    if len(ho_drugs) < 3 and len(ho_combos) < 10:
        logger.warning("tier overlap with the cache is too small to test -> falling back to a random split")
        return None
    split = {}
    for k in conds:
        d, c = k[0], k[1]
        split[k] = "unseen_drug" if d in ho_drugs else ("unseen_combo" if (d, c) in ho_combos else "train")
    n = defaultdict(int)
    for v in split.values():
        n[v] += 1
    logger.info(f"holdout (tier-aligned): {n['train']} train | {n['unseen_combo']} unseen_combo (tier3) | "
                f"{n['unseen_drug']} unseen_drug (tier2)")
    return split, sorted(ho_drugs), sorted(ho_combos)


def make_holdout(conds, frac_combos, frac_drugs, seed):
    """Three-way split for the generalization test.
      * unseen_drug  : every condition of a held-out DRUG -> the model never sees the token. Given the
                       SAR gate (structure does not predict response) this SHOULD fail; it is the
                       control proving that any transfer in `unseen_combo` comes from having seen the drug.
      * unseen_combo : a held-out (drug, cell_line) pair whose drug IS seen in OTHER cell lines ->
                       CROSS-CONTEXT TRANSFER, the scientifically meaningful test (Q11 found
                       identifiability swings by 0.83 across cell lines, so this is genuinely open).
      * train        : everything else.
    Held out at the (drug, cell_line) level, so no plate or sample of a held-out combo leaks into
    training as the same (drug, cell_line) pair. It does NOT prevent a held-out condition sharing a
    treated well with a training condition of a different cell line -- see `sample_crossing_report`,
    which counts exactly that and puts the number in the manifest rather than leaving it implicit."""
    rng = np.random.RandomState(seed)
    drugs = sorted({k[0] for k in conds})
    combos = sorted({(k[0], k[1]) for k in conds})
    n_d = int(round(frac_drugs * len(drugs)))
    ho_drugs = set(rng.choice(drugs, n_d, replace=False).tolist()) if n_d else set()
    # candidate combos: drug not already fully held out, and the drug must survive in >=2 other lines
    per_drug = defaultdict(list)
    for (d, c) in combos:
        per_drug[d].append(c)
    cand = [(d, c) for (d, c) in combos if d not in ho_drugs and len(per_drug[d]) >= 3]
    n_c = int(round(frac_combos * len(combos)))
    idx = rng.choice(len(cand), min(n_c, len(cand)), replace=False) if cand else []
    ho_combos = {cand[i] for i in np.atleast_1d(idx).tolist()} if len(cand) else set()
    split = {}
    for k in conds:
        d, c = k[0], k[1]
        split[k] = "unseen_drug" if d in ho_drugs else ("unseen_combo" if (d, c) in ho_combos else "train")
    n = defaultdict(int)
    for v in split.values():
        n[v] += 1
    logger.info(f"holdout: {n['train']} train | {n['unseen_combo']} unseen_combo "
                f"({len(ho_combos)} drug x cell-line pairs) | {n['unseen_drug']} unseen_drug "
                f"({len(ho_drugs)} drugs)")
    return split, sorted(ho_drugs), sorted(ho_combos)


def make_holdout_by_sample(conds, frac_combos, frac_drugs, seed):
    """Hold out whole treated WELLS. Cleaner -- no held-out condition shares a physical assignment
    with a training one -- but it answers a different question: an unseen well of a seen drug is a
    dose/replicate generalisation test, not cross-context transfer. Offered so the estimand is a
    choice rather than an accident."""
    rng = np.random.RandomState(seed)
    drugs = sorted({k[0] for k in conds})
    n_d = int(round(frac_drugs * len(drugs)))
    ho_drugs = set(rng.choice(drugs, n_d, replace=False).tolist()) if n_d else set()
    samples = sorted({k[3] for k in conds if k[0] not in ho_drugs})
    n_s = int(round(frac_combos * len(samples)))
    ho_samples = set(rng.choice(samples, min(n_s, len(samples)), replace=False).tolist()) if n_s else set()
    split = {}
    for k in conds:
        split[k] = ("unseen_drug" if k[0] in ho_drugs else
                    ("unseen_combo" if k[3] in ho_samples else "train"))
    n = defaultdict(int)
    for v in split.values():
        n[v] += 1
    logger.info(f"holdout (by sample): {n['train']} train | {n['unseen_combo']} unseen_combo "
                f"({len(ho_samples)} wells) | {n['unseen_drug']} unseen_drug ({len(ho_drugs)} drugs)")
    return split, sorted(ho_drugs), sorted({(k[0], k[1]) for k, v in split.items() if v == "unseen_combo"})


def enforce_sample_split(conds, split):
    """Promote a condition-level split to the treated WELL, so no well straddles the boundary.

    `--split_unit sample` was only honoured on the random-split path: whenever tier files produced a
    split it was silently ignored, and the estimand quietly reverted to condition-level leave-pairs-out
    with wells shared across the boundary. The flag now applies to every split, because which of the
    two estimands the thesis claims is a scientific decision and must not depend on which holdout
    source happened to fire.

    A well is held out if ANY of its conditions is, and it inherits the strictest label present
    (unseen_drug > unseen_combo > train), so promotion never moves a condition INTO training.
    """
    rank = {"train": 0, "unseen_combo": 1, "unseen_drug": 2}
    inv = {v: k for k, v in rank.items()}
    worst = defaultdict(int)
    for k, v in split.items():
        worst[k[3]] = max(worst[k[3]], rank.get(v, 0))
    out = {k: inv[worst[k[3]]] for k in split}
    moved = sum(1 for k in split if out[k] != split[k])
    n = defaultdict(int)
    for v in out.values():
        n[v] += 1
    logger.info(f"split promoted to the treated well: {moved} conditions relabelled -> "
                f"{n['train']} train | {n['unseen_combo']} unseen_combo | {n['unseen_drug']} unseen_drug")
    return out


def sample_crossing_report(conds, split):
    """How many held-out conditions sit in a well that also contributes training conditions.

    This is unavoidable under a (drug, cell_line) holdout, because one Tahoe well carries many cell
    lines. It is not a bug, but it does mean a held-out condition shares its well's technical
    context with training data, and that belongs in the thesis as a number rather than a hedge."""
    by_sample = defaultdict(set)
    for k, v in split.items():
        by_sample[k[3]].add(v)
    crossing = {s for s, vs in by_sample.items() if len(vs) > 1}
    n_cond = sum(1 for k, v in split.items() if v != "train" and k[3] in crossing)
    rep = {"n_samples": len(by_sample), "n_samples_crossing_split": len(crossing),
           "n_heldout_conditions_sharing_a_well_with_train": n_cond,
           "frac_heldout_conditions_sharing_a_well_with_train":
               round(n_cond / max(1, sum(1 for v in split.values() if v != "train")), 4)}
    logger.info(f"well crossing: {len(crossing)}/{len(by_sample)} wells contribute to both sides; "
                f"{rep['frac_heldout_conditions_sharing_a_well_with_train']:.1%} of held-out "
                f"conditions share a well with training data")
    return rep


# ----------------------------------------------------------------- stage 3: FIT
def _stable_rng(seed, key):
    return np.random.RandomState((seed * 1000003 + zlib.crc32("|".join(map(str, key)).encode())) % (2 ** 32))


def compute_shifts(conds, ctrl_rows, X, seed, shared_control=False):
    """Per-condition treated pseudobulk minus its group control, full and on two disjoint halves.

    Uses each condition's OWN cells only, so running it over held-out conditions leaks nothing --
    what must not happen is a held-out condition entering the generic, and that is enforced in
    `Generic.__init__`, which is handed the train keys explicitly.

    SPLIT CONTROLS, and why this is not a detail. `repro_cos` is cos(res_A, res_B), and the two
    halves are supposed to be independent measurements of the same condition. Subtracting the SAME
    control pseudobulk from both makes them share the whole control-noise term, which correlates
    them for a reason that has nothing to do with the drug. FINDINGS.md:532 retracts an earlier
    scope comparison for exactly this: measured with shared controls, reliability read 62%
    (cell line) against 19% (plate); measured with SPLIT controls the two are comparable, 20%
    against 16%. The inflation is not neutral between the scopes -- it favours cell-line scope,
    which is the choice this rebuild exists to abandon. Reproducing it here would have biased
    `--scope_sensitivity`, the gate deciding the rebuild, against the rebuild.

    So half A is measured against control half A and half B against control half B, and the two
    residuals then share nothing. The FULL shift still uses every control cell, because that is the
    training target rather than a reliability estimate and there is no reason to halve its precision.
    `--shared_control_reliability` restores the old behaviour for a side-by-side.
    """
    L = lambda ix: np.asarray(np.log1p(X[ix].todense()).mean(0)).ravel().astype(np.float32)
    ctrl_pb, ctrl_half = {}, {}
    for g, idx in ctrl_rows.items():
        ctrl_pb[g] = L(idx)
        ci = list(idx)
        _stable_rng(seed + 2, g).shuffle(ci)
        m = len(ci) // 2
        # a group too small to halve falls back to the shared control, and says so in the report
        ctrl_half[g] = (L(ci[:m]), L(ci[m:])) if m >= 1 and len(ci) >= 2 else (ctrl_pb[g], ctrl_pb[g])

    shifts, n_shared = {}, 0
    for k, info in conds.items():
        idxs = list(info["rows"])
        _stable_rng(seed, k).shuffle(idxs)
        h = len(idxs) // 2
        g = info["group"]
        cpb = ctrl_pb[g]
        if shared_control:
            cA = cB = cpb
        else:
            cA, cB = ctrl_half[g]
            if cA is cpb:
                n_shared += 1
        shifts[k] = {"full": L(idxs) - cpb, "A": L(idxs[:h]) - cA, "B": L(idxs[h:]) - cB}
    if shared_control:
        logger.warning("reliability measured with SHARED controls: res_A and res_B share the whole "
                       "control-noise term, which inflates repro_cos and does so unevenly across "
                       "scopes. Use for comparison only.")
    elif n_shared:
        logger.warning(f"{n_shared} conditions sit in a control group too small to halve; their "
                       f"reliability is shared-control and therefore optimistic")
    return shifts, ctrl_pb, {"n_conditions_with_shared_control": n_shared,
                             "shared_control_reliability": bool(shared_control)}


class Generic:
    """The drug-agnostic program, fitted on TRAIN conditions only.

    Drug-weighted: each drug's conditions inside a scope group are averaged before drugs are averaged
    together, so a drug measured at five doses does not count five times.

    Leave-one-drug-out: `value(group, drug, half)` excludes `drug` from its own generic. Without that,
    a train condition would be centred on a mean containing itself while a held-out condition would
    not, and the two splits would be scored against differently defined targets.
    """

    def __init__(self, shifts, conds, train_keys, shrink_k=0.0, loo=True, min_plate_drugs=3):
        self.shrink_k = float(shrink_k)
        self.loo = bool(loo)
        self.min_plate_drugs = int(min_plate_drugs)
        self.fine, self.coarse = {}, {}      # (cell_line, plate) and cell_line
        for level, keyfn in (("fine", lambda k: (conds[k]["cell_line"], conds[k]["plate"])),
                             ("coarse", lambda k: conds[k]["cell_line"])):
            acc = defaultdict(lambda: defaultdict(lambda: {h: None for h in "fAB"}))
            cnt = defaultdict(lambda: defaultdict(int))
            for k in train_keys:
                g, d = keyfn(k), conds[k]["drug"]
                cnt[g][d] += 1
                for h, name in (("f", "full"), ("A", "A"), ("B", "B")):
                    v = shifts[k][name]
                    acc[g][d][h] = v.copy() if acc[g][d][h] is None else acc[g][d][h] + v
            store = {}
            for g, drugs in acc.items():
                dm = {d: {h: drugs[d][h] / cnt[g][d] for h in "fAB"} for d in drugs}
                tot = {h: np.sum(np.stack([dm[d][h] for d in dm]), axis=0) for h in "fAB"}
                store[g] = {"drug_mean": dm, "total": tot, "n_drugs": len(dm)}
            setattr(self, level, store)

    def _loo(self, store, g, drug, h):
        s = store.get(g)
        if s is None:
            return None
        n, tot = s["n_drugs"], s["total"][h]
        if self.loo and drug in s["drug_mean"]:
            tot = tot - s["drug_mean"][drug][h]
            n -= 1
        return None if n <= 0 else tot / n

    def _n_other(self, g, drug):
        """Training drugs available to a plate group's generic, after leave-one-drug-out."""
        s = self.fine.get(g)
        if s is None:
            return 0
        return s["n_drugs"] - (1 if (self.loo and drug in s["drug_mean"]) else 0)

    def value(self, cell_line, plate, drug, half, scope="plate"):
        """The generic for one condition, or None when the requested scope cannot supply one.

        FAIL CLOSED AT PLATE SCOPE. An earlier version fell back to the cell-line generic whenever a
        plate group had no other training drug. That made `--generic_scope plate --shrink_k 0` a lie:
        an unreported subset of conditions silently received cell-line-scoped targets, which is the
        contaminated frame this build exists to leave, and nothing in the output said which ones. A
        plate group with fewer than `min_plate_drugs` other training drugs now yields None, the
        condition is dropped, and the count is reported.

        With `--shrink_k > 0` the blend toward the cell-line generic is the DECLARED estimator rather
        than a hidden fallback, so a thin group is legitimate -- but then the frame is hierarchical,
        not plate-scoped, and `report.json` names it that way.
        """
        h = {"full": "f", "A": "A", "B": "B"}[half]
        coarse = self._loo(self.coarse, cell_line, drug, h)
        if scope == "cell_line":
            return coarse
        g = (cell_line, plate)
        n = self._n_other(g, drug)
        fine = self._loo(self.fine, g, drug, h)
        if self.shrink_k <= 0:
            return fine if n >= self.min_plate_drugs else None
        if coarse is None:
            return fine if n >= self.min_plate_drugs else None
        if fine is None or n <= 0:
            return coarse
        w = n / (n + self.shrink_k)
        return w * fine + (1.0 - w) * coarse

    def frame_name(self, scope):
        if scope == "cell_line":
            return "cell_line"
        return "plate" if self.shrink_k <= 0 else f"hierarchical(shrink_k={self.shrink_k:g})"

    def digest(self):
        """Hash of everything that was fitted. The poison test asserts this is unchanged when
        held-out expression is corrupted."""
        hsh = hashlib.sha256()
        for level in ("coarse", "fine"):
            for g in sorted(getattr(self, level), key=lambda x: str(x)):
                s = getattr(self, level)[g]
                hsh.update(str(g).encode()); hsh.update(str(s["n_drugs"]).encode())
                for d in sorted(s["drug_mean"]):
                    hsh.update(d.encode())
                    for h in "fAB":
                        hsh.update(np.ascontiguousarray(s["drug_mean"][d][h]).tobytes())
        return hsh.hexdigest()

    def export(self, conds, scope):
        """Group -> generic shift, for reconstruction. Averaged over drugs WITHOUT leave-one-out,
        because evaluation reconstructs a profile for a condition whose drug it does not know.

        Shrinkage is applied here too. It was not, which meant reconstruction used a pure plate mean
        while the targets it was reconstructing had been built from a blended one -- two different
        definitions of the same quantity, so `predicted_treated` would not have been the inverse of
        the transform that produced the target.
        """
        if scope == "cell_line":
            return {g: s["total"]["f"] / s["n_drugs"] for g, s in self.coarse.items()}
        out = {}
        for g, s in self.fine.items():
            fine = s["total"]["f"] / s["n_drugs"]
            n = s["n_drugs"]
            if self.shrink_k <= 0:
                if n >= self.min_plate_drugs:
                    out[g] = fine
                continue
            c = self.coarse.get(g[0])
            if c is None:
                out[g] = fine
            else:
                w = n / (n + self.shrink_k)
                out[g] = w * fine + (1.0 - w) * (c["total"]["f"] / c["n_drugs"])
        return out


# ----------------------------------------------------------------- stage 4: TRANSFORM
def transform(conds, shifts, gen, split, scope, repro_thr, eval_filter=True):
    """Project every condition through the fitted generic and apply the reliability filter."""
    kept, stats = {}, defaultdict(lambda: {"n": 0, "n_kept": 0, "cos": []})
    for k, info in conds.items():
        c, p, d = info["cell_line"], info["plate"], info["drug"]
        gf = gen.value(c, p, d, "full", scope)
        gA = gen.value(c, p, d, "A", scope)
        gB = gen.value(c, p, d, "B", scope)
        s = split.get(k, "train")
        st = stats[s]
        st["n"] += 1
        if gf is None or gA is None or gB is None:
            continue                     # no other training drug in scope: the residual is undefined
        rA, rB = shifts[k]["A"] - gA, shifts[k]["B"] - gB
        cs = float(rA @ rB / (np.linalg.norm(rA) * np.linalg.norm(rB) + 1e-9))
        st["cos"].append(cs)
        if cs <= repro_thr and (s == "train" or eval_filter):
            continue
        st["n_kept"] += 1
        kept[k] = {"residual": (shifts[k]["full"] - gf).astype(np.float32),
                   # half-split residuals kept so evaluation can compute the achievable CEILING
                   # (a real replicate scoring against the other half's truths)
                   "residual_A": rA.astype(np.float32), "residual_B": rB.astype(np.float32),
                   "group": info["group"], "n_cells": info["n_cells"], "repro_cos": cs}
    out = {}
    for s, v in stats.items():
        cos = np.array(v["cos"]) if v["cos"] else np.array([np.nan])
        out[s] = {"n": v["n"], "n_kept": v["n_kept"],
                  "retention": round(v["n_kept"] / max(1, v["n"]), 4),
                  "mean_repro_cos": round(float(np.nanmean(cos)), 4)}
        logger.info(f"reliability [{s}]: mean cos(A,B)={out[s]['mean_repro_cos']:+.3f}; "
                    f"KEPT {v['n_kept']}/{v['n']} ({out[s]['retention']:.0%}) at cos > {repro_thr}")
    return kept, out


def build_residuals(cache_dir, min_treated, min_control, repro_thr, seed=0,
                    generic_scope="cell_line", shrink_k=0.0, loo=False, split_controls=False,
                    train_keys=None, holdout=None, meta_dir=None, repo=TAHOE_REPO,
                    drop_combinations=False):
    """COMPATIBILITY SHIM for the five analysis scripts that consume residual truth.

    `residual_eval.py`, `channel_gate.py`, `reconstructed_eval.py`, `reward_calibration.py` and
    `build_de_weights.py` all call this with the same positional signature and unpack
    `(kept, generic, ctrl_rows, X, meta)`. Restructuring `run()` into inventory/split/fit/transform
    removed the function they were calling; this restores it on top of the new stages.

    THE DEFAULTS REPRODUCE THE OLD, DEFECTIVE SEMANTICS ON PURPOSE. An audit asked for every number
    quoted in the thesis to be regenerable from the repository, and those numbers were produced by
    the cell-line-scoped, non-leave-one-out, shared-control build. Silently "fixing" them here would
    make the published figures unreproducible and hide the size of each repair. So:

        generic_scope="cell_line"   as published; pass "plate" for the repaired frame
        loo=False                   as published; the generic contains the condition's own drug
        split_controls=False        as published; res_A and res_B share one control pseudobulk,
                                    which inflates repro_cos (variance_decomposition.py:55 already
                                    documented this, and FINDINGS.md:532 retracts a scope comparison
                                    made this way)

    Pass `train_keys` (or `holdout`, a path to a holdout.json) to fit the generic on training
    conditions only. WITHOUT ONE OF THOSE THE GENERIC IS FITTED OVER EVERY CONDITION, so a caller
    scoring a held-out split against this truth is scoring against targets partly defined by the
    holdout. That is the transductive defect, on the evaluation side rather than the training side,
    and closing it in `residual_eval` is Step 6.
    """
    _, _, conc_of = load_meta_maps(repo, meta_dir)
    conds, ctrl_rows, X, meta, notes = inventory(
        cache_dir, min_treated, min_control, conc_of,
        drop_combinations=drop_combinations, require_sample_id=False)
    shifts, _, _ = compute_shifts(conds, ctrl_rows, X, seed, shared_control=not split_controls)

    if train_keys is None and holdout and os.path.exists(holdout):
        hm = json.load(open(holdout))
        want = {k for k, v in hm.get("split", {}).items() if v == "train"}
        train_keys = [k for k in conds if "|".join(map(str, k)) in want]
        logger.info(f"generic fitted on {len(train_keys)} training conditions from {holdout}")
    if train_keys is None:
        train_keys = list(conds)
        logger.warning("build_residuals: the generic is being fitted over EVERY condition. Any "
                       "held-out split scored against this truth is transductive. Pass holdout= or "
                       "train_keys= to close that.")

    gen = Generic(shifts, conds, train_keys, shrink_k=shrink_k, loo=loo)
    split = {k: "train" for k in conds}
    kept, _ = transform(conds, shifts, gen, split, generic_scope, repro_thr, eval_filter=True)
    gexp = gen.export(conds, generic_scope)
    if generic_scope == "cell_line":
        generic = gexp
    else:                       # collapse (cell_line, plate) -> cell_line so old callers still index
        acc = defaultdict(list)
        for g, v in gexp.items():
            acc[g[0]].append(v)
        generic = {c: np.mean(np.stack(v), axis=0) for c, v in acc.items()}
    return kept, generic, ctrl_rows, X, meta


def scope_sensitivity(conds, shifts, train_keys, repro_thr):
    """What each scope choice costs, measured on TRAIN conditions only.

    Cell-line scope was chosen originally because it retains far more conditions; plate scope is
    required because cell-line scope leaves same-plate structure in the target. This prints the
    trade-off so the configuration is decided on numbers rather than on either argument alone."""
    rows = []
    for label, scope, k_shrink in (("cell_line", "cell_line", 0.0), ("plate", "plate", 0.0),
                                   ("plate+shrink5", "plate", 5.0), ("plate+shrink20", "plate", 20.0)):
        g = Generic(shifts, conds, train_keys, shrink_k=k_shrink)
        cos = []
        for k in train_keys:
            info = conds[k]
            gA = g.value(info["cell_line"], info["plate"], info["drug"], "A", scope)
            gB = g.value(info["cell_line"], info["plate"], info["drug"], "B", scope)
            if gA is None or gB is None:
                continue
            rA, rB = shifts[k]["A"] - gA, shifts[k]["B"] - gB
            cos.append(float(rA @ rB / (np.linalg.norm(rA) * np.linalg.norm(rB) + 1e-9)))
        cos = np.array(cos) if cos else np.array([np.nan])
        rows.append({"scope": label, "n": len(cos), "mean_repro_cos": round(float(np.nanmean(cos)), 4),
                     "retention": round(float(np.nanmean(cos > repro_thr)), 4)})
        logger.info(f"  scope {label:15s} mean cos={rows[-1]['mean_repro_cos']:+.3f}  "
                    f"retained {rows[-1]['retention']:.0%} of {len(cos)} train conditions")
    return rows


# ----------------------------------------------------------------- driver
def run(args):
    panel_genes = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    cvcl, moa_of, conc_of = load_meta_maps(args.repo, args.meta_dir)

    logger.info("=== [1/4] inventory (metadata and cell counts only) ===")
    conds, ctrl_rows, X, meta, notes = inventory(
        args.cache_dir, args.min_treated, args.min_control, conc_of,
        drop_combinations=not args.keep_combinations, require_sample_id=not args.no_require_sample_id)
    if not conds:
        logger.error("no conditions survived the inventory filters"); return

    logger.info("=== [2/4] split (assigned from metadata, before anything is fitted) ===")
    split, ho_drugs, ho_combos = None, [], []
    if args.tier2_file or args.tier3_file:      # preferred: reuse the ORIGINAL held-out tiers
        res = holdout_from_tiers(conds, args.tier2_file, args.tier3_file)
        if res:
            split, ho_drugs, ho_combos = res
    if split is None and (args.holdout_combos > 0 or args.holdout_drugs > 0):
        mk = make_holdout_by_sample if args.split_unit == "sample" else make_holdout
        split, ho_drugs, ho_combos = mk(conds, args.holdout_combos, args.holdout_drugs, args.seed)
    # TOP-UP: the original tier-3 set overlaps our cache on only ~17 combos, far too few to test
    # cross-context transfer (a clustered CI over a handful of points is meaningless). Add RANDOM
    # (drug, cell_line) combos on top of the tier-aligned ones until the split is powered. The
    # tier-aligned subset stays identifiable in the manifest for comparability.
    if split is not None and args.holdout_combos > 0 and args.split_unit == "condition":
        n_combo = sum(1 for v in split.values() if v == "unseen_combo")
        cur_combos = {(k[0], k[1]) for k, v in split.items() if v == "unseen_combo"}
        if n_combo < args.min_combo_conditions:
            rng2 = np.random.RandomState(args.seed + 7)
            per_drug = defaultdict(set)
            for k in conds:
                per_drug[k[0]].add(k[1])
            cand = sorted({(k[0], k[1]) for k, v in split.items()
                           if v == "train" and len(per_drug[k[0]]) >= 2} - cur_combos)
            need = args.min_combo_conditions - n_combo
            add, i = set(), 0
            order = rng2.permutation(len(cand))
            while i < len(order) and sum(1 for k in conds if (k[0], k[1]) in add) < need:
                add.add(cand[order[i]]); i += 1
            for k in conds:
                if split[k] == "train" and (k[0], k[1]) in add:
                    split[k] = "unseen_combo"
            ho_combos = sorted(cur_combos | add)
            n2 = sum(1 for v in split.values() if v == "unseen_combo")
            logger.info(f"combo top-up: {n_combo} tier-aligned -> {n2} conditions "
                        f"({len(cur_combos)} tier-aligned + {len(add)} random pairs) so the "
                        f"cross-context transfer test is powered")
    if split is None:
        logger.warning("NO HOLDOUT REQUESTED: the generic will be fitted on every condition, which "
                       "is only valid when nothing downstream is scored as held out")
        split = {k: "train" for k in conds}
    if args.split_unit == 'sample':
        split = enforce_sample_split(conds, split)
    crossing = sample_crossing_report(conds, split)
    train_keys = [k for k in conds if split[k] == "train"]
    logger.info(f"fitting on {len(train_keys)} train conditions "
                f"({len({conds[k]['drug'] for k in train_keys})} drugs)")

    logger.info("=== [3/4] fit (train conditions only) ===")
    shifts, _, relprov = compute_shifts(conds, ctrl_rows, X, args.seed,
                                        shared_control=args.shared_control_reliability)
    sens = scope_sensitivity(conds, shifts, train_keys, args.repro_thr) if args.scope_sensitivity else []
    gen = Generic(shifts, conds, train_keys, shrink_k=args.shrink_k,
                  min_plate_drugs=args.min_plate_drugs)
    fit_digest = gen.digest()
    logger.info(f"generic scope={args.generic_scope} shrink_k={args.shrink_k} "
                f"leave-one-drug-out=on  fit_digest={fit_digest[:16]}")

    logger.info("=== [4/4] transform ===")
    kept, relstats = transform(conds, shifts, gen, split, args.generic_scope, args.repro_thr,
                               eval_filter=args.eval_repro_filter)
    if not kept:
        logger.error("no conditions survived the reliability filter"); return

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "residual.jsonl")
    # written to a temp path and renamed only once the dose check passes, so "REFUSING TO WRITE"
    # is literally true and a failed run cannot leave a usable-looking training file behind
    tmp_path = out_path + ".partial"
    # VALIDATION SHARD, in the SAME format as training. The retrain was validating residual-format
    # training against ordinary cell-sentence tier-1 data, so the validation loss was measuring a
    # different output distribution than the one being learned -- it could not detect overfitting on
    # the actual objective and was not comparable across arms. Carved from TRAINING conditions and
    # held out by WELL, so a validation condition never shares a treated well with a training one.
    val_path = os.path.join(args.out_dir, "residual_val.jsonl")
    tmp_val = val_path + ".partial"
    train_samples = sorted({k[3] for k in kept if split[k] == "train"})
    n_val_s = int(round(args.val_frac * len(train_samples)))
    val_samples = set()
    if n_val_s > 0:
        vr = np.random.RandomState(args.seed + 11)
        val_samples = {train_samples[i] for i in vr.choice(len(train_samples), n_val_s, replace=False)}
    logger.info(f"validation shard: {len(val_samples)}/{len(train_samples)} training wells "
                f"({args.val_frac:.1%}) -> {val_path}")
    n_val = 0
    n_ex, n_cond, n_skipped_holdout, emitted_doses = 0, 0, 0, []
    with open(tmp_path, "w") as out, open(tmp_val, "w") as out_val:
        for k, v in kept.items():
            d, c, p, sid = k
            info = conds[k]
            # held-out conditions are NEVER written to the training file (the manifest records them
            # so evaluation can score each split separately)
            if split[k] != "train":
                n_skipped_holdout += 1
                continue
            rows = ctrl_rows.get(v["group"])
            if rows is None or len(rows) == 0:
                continue
            r = _stable_rng(args.seed + 1, k)
            take = rows if len(rows) <= args.max_ctrl else rows[r.choice(len(rows), args.max_ctrl,
                                                                         replace=False)]
            resp = residual_to_sentence(v["residual"], panel_genes, args.k_up, args.k_down)
            cname = cvcl.get(c, c)
            moa = moa_of.get(d, "unclear")
            emitted_doses.append(info["dose_molar"] if info["dose_molar"] is not None else info["dose_raw"])
            for ri in take:
                ctrl_vec = np.asarray(X[ri].todense()).ravel()
                prompt = format_prompt(cname, d, info["dose_display"], moa,
                                       expr_to_sentence(ctrl_vec, panel_genes), order=args.prompt_order)
                rec = json.dumps({
                    "prompt": prompt, "response": resp,
                    "metadata": {"drug": d, "cell_line_id": c, "plate": p,
                                 "sample_id": sid, "dose_molar": info["dose_molar"],
                                 "dose_raw": info["dose_raw"],
                                 "target": "residual", "repro_cos": v["repro_cos"]}}) + "\n"
                if sid in val_samples:
                    out_val.write(rec); n_val += 1
                else:
                    out.write(rec); n_ex += 1
            n_cond += 1
            if n_cond % 200 == 0:
                logger.info(f"  {n_cond} conditions, {n_ex} examples")

    # the defect this builder exists to prevent: never ship a dose field holding sample identifiers
    if td.looks_like_sample_id(emitted_doses):
        for t in (tmp_path, tmp_val):
            if os.path.exists(t):
                os.remove(t)
        raise SystemExit("REFUSING TO WRITE: the emitted dose field holds sample identifiers. "
                         "This is the exact defect shared/tahoe_design.py was written to stop.")
    os.replace(tmp_path, out_path)
    os.replace(tmp_val, val_path)

    # reconstruction assets: predicted_treated = control + generic_shift(group) + predicted_residual
    gexp = gen.export(conds, args.generic_scope)
    gkeys = sorted(gexp, key=lambda x: str(x))
    np.savez_compressed(
        os.path.join(args.out_dir, "reconstruction.npz"),
        scope=np.array(args.generic_scope, dtype=object),
        group_keys=np.array(["|".join(g) if isinstance(g, tuple) else str(g) for g in gkeys], dtype=object),
        generic_shift=np.stack([gexp[g] for g in gkeys]),
        # kept under its old name so an unpatched residual_eval still finds a cell-line axis
        cell_lines=np.array([g[0] if isinstance(g, tuple) else g for g in gkeys], dtype=object),
        panel_genes=np.array(panel_genes, dtype=object))

    report = {"n_conditions_inventoried": len(conds), "n_conditions_kept": len(kept),
              "n_examples": n_ex, "k_up": args.k_up, "k_down": args.k_down,
              "repro_thr": args.repro_thr, "scope": args.generic_scope, "shrink_k": args.shrink_k,
              "frame": gen.frame_name(args.generic_scope),
              "min_plate_drugs": args.min_plate_drugs,
              "n_validation_examples": n_val, "n_validation_wells": len(val_samples),
              "leave_one_drug_out": True, "drug_weighted_generic": True,
              "split_before_fit": True, "split_unit": args.split_unit,
              "reliability_controls": relprov,
              "eval_repro_filter": bool(args.eval_repro_filter),
              "fit_digest": fit_digest, "reliability_by_split": relstats,
              "scope_sensitivity": sens, "well_crossing": crossing,
              "inventory": notes, "down_token": DOWN, "end_token": END}
    json.dump({"split": {"|".join(map(str, k)): v for k, v in split.items() if k in kept},
               "holdout_drugs": ho_drugs,
               "holdout_combos": ["|".join(map(str, kk)) for kk in ho_combos],
               "key_fields": ["drug", "cell_line_id", "plate", "sample_id"],
               "well_crossing": crossing},
              open(os.path.join(args.out_dir, "holdout.json"), "w"), indent=2)
    report["holdout"] = {"n_train": sum(1 for k, v in split.items() if v == "train" and k in kept),
                         "n_unseen_combo": sum(1 for k, v in split.items()
                                               if v == "unseen_combo" and k in kept),
                         "n_unseen_drug": sum(1 for k, v in split.items()
                                              if v == "unseen_drug" and k in kept),
                         "n_conditions_excluded_from_training": n_skipped_holdout}
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w"), indent=2)
    if args.emit_fit_digest:
        open(args.emit_fit_digest, "w").write(fit_digest + "\n")
    logger.info(f"holdout manifest -> {args.out_dir}/holdout.json "
                f"({n_skipped_holdout} conditions withheld from training)")
    logger.info(f"wrote {n_ex} examples from {n_cond} conditions -> {out_path}")
    logger.info(f"wrote {n_val} validation examples ({len(val_samples)} wells) -> {val_path}")
    logger.info(f"reconstruction assets -> {args.out_dir}/reconstruction.npz")
    logger.info(f"NOTE: register '{DOWN}' as a special token in the trainer alongside '{END}'.")


def selftest():
    """Synthetic checks on the pieces that carry the rebuild: the sentence encoding, the reliability
    contrast, drug-weighting, leave-one-drug-out, and that a held-out condition cannot move the fit."""
    ok = []
    panel = [f"G{i}" for i in range(20)]
    res = np.zeros(20, dtype=np.float32)
    res[3], res[7] = 5.0, 3.0          # up
    res[11], res[15] = -4.0, -2.0      # down
    s = residual_to_sentence(res, panel, 5, 5)
    toks = s.split()
    up_block = toks[:toks.index(DOWN)]
    dn_block = toks[toks.index(DOWN) + 1:toks.index(END)]
    ok.append(("sentence lists up genes, [DOWN], then down genes",
               up_block == ["G3", "G7"] and dn_block == ["G11", "G15"] and s.endswith(END)))

    rng = np.random.RandomState(0)
    sig = rng.randn(50)
    good = float((sig + rng.randn(50) * 0.3) @ (sig + rng.randn(50) * 0.3) /
                 (np.linalg.norm(sig + rng.randn(50) * 0.3) ** 2 + 1e-9))
    a, b = rng.randn(50), rng.randn(50)
    bad = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok.append(("reproducible halves are kept and noise-only halves are dropped", good > 0.2 > bad))

    # --- generic: drug weighting, leave-one-out, and train-only fitting -----------------
    # drug X appears at three doses with shift 3, drug Y once with shift 0. A condition-weighted
    # mean gives 2.25; a drug-weighted mean gives 1.5, and only the second is the intended estimand.
    conds, shifts = {}, {}
    for i, (d, val) in enumerate([("X", 3.0), ("X", 3.0), ("X", 3.0), ("Y", 0.0)]):
        k = (d, "c1", "p1", f"smp_{i}")
        conds[k] = {"cell_line": "c1", "plate": "p1", "drug": d, "group": ("c1", "p1"),
                    "n_cells": 50, "sample_id": f"smp_{i}"}
        shifts[k] = {h: np.full(4, val, dtype=np.float32) for h in ("full", "A", "B")}
    g = Generic(shifts, conds, list(conds), shrink_k=0.0)
    # leave-one-out for drug Y: the generic it sees is drug X's mean alone = 3.0
    v_y = g.value("c1", "p1", "Y", "full", "plate")[0]
    # leave-one-out for drug X: only drug Y remains = 0.0
    v_x = g.value("c1", "p1", "X", "full", "plate")[0]
    ok.append(("generic is a mean over DRUGS, not conditions (X's 3 doses count once)",
               abs(v_y - 3.0) < 1e-6))
    ok.append(("leave-one-drug-out removes the condition's own drug", abs(v_x - 0.0) < 1e-6))

    # a held-out condition must not move the fit, however extreme it is
    k_ho = ("Z", "c1", "p1", "smp_99")
    conds[k_ho] = {"cell_line": "c1", "plate": "p1", "drug": "Z", "group": ("c1", "p1"),
                   "n_cells": 50, "sample_id": "smp_99"}
    shifts[k_ho] = {h: np.full(4, 1e6, dtype=np.float32) for h in ("full", "A", "B")}
    train_only = [k for k in conds if k != k_ho]
    ok.append(("a poisoned held-out condition leaves the fit digest unchanged",
               Generic(shifts, conds, train_only, shrink_k=0.0).digest() == g.digest()))

    # shrinkage: a plate with one other drug is pulled most of the way to the cell-line generic
    gs = Generic(shifts, conds, train_only, shrink_k=5.0)
    blended = gs.value("c1", "p1", "X", "full", "plate")[0]
    ok.append(("shrink_k pulls a thin plate generic toward the cell-line generic",
               abs(blended - 0.0) < 1e-6))     # only one scope level here, so fine == coarse

    # the shipped defect: a dose column full of sample identifiers must be refused
    ok.append(("a dose field holding sample identifiers is detected",
               td.looks_like_sample_id(["smp_1841", "smp_1882"]) and
               not td.looks_like_sample_id([5e-8, 1e-6])))

    for name, passed in ok:
        logger.info(f"  {'ok  ' if passed else 'FAIL'} {name}")
    allok = all(p for _, p in ok)
    logger.info(f"SELFTEST {'PASSED' if allok else 'FAILED'}")
    if not allok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--meta_dir", default=None,
                    help="read cell_line/drug/sample metadata parquets from this directory instead "
                         "of the Hub (offline runs and tests)")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2, help="cos(res_A,res_B) filter")
    ap.add_argument("--k_up", type=int, default=100)
    ap.add_argument("--k_down", type=int, default=100)
    ap.add_argument("--max_ctrl", type=int, default=60, help="control cells (=examples) per condition")
    ap.add_argument("--generic_scope", choices=["plate", "cell_line"], default="plate",
                    help="plate = (cell_line, plate), which removes the same-plate structure that "
                         "cell-line scope leaves in the target (+0.478 vs -0.018)")
    ap.add_argument("--shrink_k", type=float, default=0.0,
                    help="blend the plate generic toward the cell-line generic with weight "
                         "n/(n+k), n = OTHER training drugs in the plate group. 0 = pure plate "
                         "scope. Use --scope_sensitivity first.")
    ap.add_argument("--shared_control_reliability", action="store_true",
                    help="measure repro_cos with ONE control pseudobulk shared by both halves. "
                         "Inflates reliability, and unevenly across scopes -- FINDINGS.md:532 "
                         "retracts a scope comparison made this way. For side-by-side only.")
    ap.add_argument("--scope_sensitivity", action="store_true",
                    help="report reliability retention under every scope setting (train conditions "
                         "only) before choosing one")
    ap.add_argument("--keep_combinations", action="store_true",
                    help="keep multi-drug samples, which the prompt cannot express. Off by default: "
                         "the old parser took the first component and called it a single-drug condition")
    ap.add_argument("--no_require_sample_id", action="store_true",
                    help="proceed even if no treatment identifier can be recovered (the dose and the "
                         "assignment unit are then both unavailable, and the report says so)")
    ap.add_argument("--eval_repro_filter", action="store_true",
                    help="ALSO drop held-out conditions whose own residual fails the reliability "
                         "threshold. OFF by default: that is selection on the outcome, and it makes "
                         "the evaluation set unrepresentative of the conditions the split defined. "
                         "The primary result uses the complete metadata-eligible holdout; "
                         "repro_cos is written per example so a filtered SENSITIVITY can be computed "
                         "afterwards and labelled as one.")
    ap.add_argument("--val_frac", type=float, default=0.02,
                    help="fraction of TRAINING wells carved into residual_val.jsonl, in the same "
                         "format as training. Validating residual targets against ordinary "
                         "cell-sentence data measures a different output distribution and cannot "
                         "detect overfitting on the objective actually being trained.")
    ap.add_argument("--min_plate_drugs", type=int, default=3,
                    help="a plate group with fewer OTHER training drugs than this cannot supply a "
                         "plate-scoped generic. At --shrink_k 0 the condition is dropped rather than "
                         "silently falling back to the cell-line generic, which would put part of the "
                         "build back in the contaminated frame without saying so.")
    ap.add_argument("--tier2_file", default=None,
                    help="eval_tier2_unseen_drugs.jsonl -> hold out THE SAME drugs the original "
                         "preprocessing held out, so tier-2 numbers are comparable to prior results")
    ap.add_argument("--tier3_file", default=None,
                    help="eval_tier3_unseen_combos.jsonl -> hold out the same (drug, cell_line) combos")
    ap.add_argument("--holdout_combos", type=float, default=0.0,
                    help="fraction of (drug, cell_line) pairs withheld from training -> CROSS-CONTEXT "
                         "transfer test (drug seen in other cell lines)")
    ap.add_argument("--min_combo_conditions", type=int, default=200,
                    help="top up unseen_combo with random (drug,cell_line) pairs until this many "
                         "conditions are held out (tier-3 overlap alone gives ~27, far too few)")
    ap.add_argument("--holdout_drugs", type=float, default=0.0,
                    help="fraction of DRUGS withheld entirely -> unseen-drug control (expected to fail "
                         "given the SAR gate; proves transfer requires having seen the drug)")
    ap.add_argument("--split_unit", choices=["condition", "sample"], default="condition",
                    help="condition = hold out (drug, cell_line) pairs, the cross-context transfer "
                         "estimand. sample = hold out whole treated wells: no shared-well contact, "
                         "but it becomes a dose/replicate generalisation test instead.")
    ap.add_argument("--prompt_order", choices=["drug_first", "drug_last"], default="drug_first",
                    help="drug_last moves the instruction to sit immediately before generation, "
                         "instead of leaving it several hundred tokens upstream behind the control "
                         "cell sentence. Content is identical; only the distance changes. The eval "
                         "MUST be run with the same value.")
    ap.add_argument("--emit_fit_digest", default=None,
                    help="write the hash of every fitted quantity here (used by the poison test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="RESULTS/residual_targets")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")
    run(args)


if __name__ == "__main__":
    main()

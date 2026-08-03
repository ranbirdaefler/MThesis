#!/usr/bin/env python
r"""
residual_eval.py — Arm 1b stage-1 evaluation: does the residual-trained model USE the drug?
============================================================================================
The residual model emits a signed DE signature ("<up> [DOWN] <down> [END_CELL]"), so the standard
nir_benchmark scoring path does not apply (its rank->expression decode is fitted for EXPRESSION ranks
and its truths are full pseudobulks). This scores in RESIDUAL space, which is the space the model was
trained in and — per target_divergence — the space where drug identity actually lives in the tokens.

METRIC (same logic as NIR, in residual space)
  Predicted and true residuals are both mapped to a SIGNED RANK vector (top-k up = positive scores by
  rank, top-k down = negative), so a generated sentence and a continuous truth are compared like for
  like. Similarity = cosine. NIR = fraction of OTHER drugs (same cell line) whose truth is LESS similar
  to the prediction than the drug's own truth. 0.5 = chance.

ARMS
  model    : generate from the real prompt.
  scramble : identical control cell, only the drug name+MoA swapped to another drug in the same cell
             line -> the decisive causal test. model >> scramble = genuine drug use.
  ceiling  : the drug's own half-B residual scoring against half-A truths (achievable bar).
  random   : a shuffled-gene signature (floor).
Clustered 95% CI over CELL LINES (never over drugs -- pseudoreplication).

LEAKAGE NOTE (read before quoting): the residual targets were built from this same cache, so sampled
conditions may have been TRAINED on. `model - scramble` is still a valid test of whether the model USES
the drug token (memorisation also requires reading the drug), but a high absolute model NIR may be
memorisation rather than generalisation. --held_out_only restricts scoring to conditions absent from
the training file, when such conditions exist.

USAGE
  python residual_eval.py --cache_dir /data/.../ot_cache \
      --model_path /data/.../checkpoints/pythia_sft_residual/final \
      --train_file /data/.../residual_targets/residual.jsonl \
      --n_conditions 200 --k_samples 4 --bf16 --out RESULTS/residual_eval.json
  python residual_eval.py --selftest
"""
import argparse, json, os, sys, re, ast, zlib, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "shared"),
           os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import inference as inf                                              # noqa: E402

END, DOWN = "[END_CELL]", "[DOWN]"
CTRL_MARKER = "Control cell:"


# ----------------------------------------------------------------- signed-rank encoding
def signed_rank_from_sentence(sent, gene_index, P, k=100):
    """generated '<up> [DOWN] <down> [END_CELL]' -> signed rank-score vector."""
    v = np.zeros(P, dtype=np.float32)
    toks = [t for t in sent.replace(END, " ").split() if t]
    if DOWN in toks:
        d = toks.index(DOWN)
        up, dn = toks[:d], toks[d + 1:]
    else:
        up, dn = toks, []
    for blk, sgn in ((up, 1.0), (dn, -1.0)):
        r = 0
        for g in blk:
            gi = gene_index.get(g)
            if gi is None or v[gi] != 0:
                continue
            r += 1
            if r > k:
                break
            v[gi] = sgn / np.log2(r + 1.0)
    return v


def signed_rank_from_vector(res, P, k=100):
    """continuous residual -> the SAME signed rank encoding, so truths and generations are comparable."""
    v = np.zeros(P, dtype=np.float32)
    up = [j for j in np.argsort(-res)[:k].tolist() if res[j] > 0]
    dn = [j for j in np.argsort(res)[:k].tolist() if res[j] < 0]
    for blk, sgn in ((up, 1.0), (dn, -1.0)):
        for r, j in enumerate(blk, 1):
            v[j] = sgn / np.log2(r + 1.0)
    return v


def rank_value_profile(X, rows, P, kmax=400):
    """Empirical log1p(CP10K) value as a function of expression rank, from REAL cells. Lets us decode an
    ordinary cell sentence into the cache's units so a cell-sentence model can be scored in the SAME
    residual space as the residual model (the apples-to-apples control)."""
    import numpy as _np
    acc = _np.zeros(kmax, dtype=_np.float64); n = 0
    for i in rows:
        v = _np.asarray(X[i].todense()).ravel()
        s = _np.sort(v)[::-1][:kmax]
        acc[:len(s)] += s; n += 1
    return (acc / max(1, n)).astype(np.float32)


def sentence_to_expression(sent, gene_index, P, rank_prof):
    """ordinary '<genes ranked by expression> [END_CELL]' -> expression vector in cache units."""
    v = np.zeros(P, dtype=np.float32)
    r = 0
    for g in sent.replace(END, " ").replace(DOWN, " ").split():
        gi = gene_index.get(g)
        if gi is None or v[gi] != 0:
            continue
        if r >= len(rank_prof):
            break
        v[gi] = rank_prof[r]; r += 1
    return v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def nir_from_sims(own, others):
    """Tie-aware (Mann-Whitney convention): ties count half. Without this a DEGENERATE predictor -- one
    whose similarity to every truth is identical -- scores 0.000 instead of the 0.500 it deserves. That is
    not hypothetical: 'predict the average drug response' IS the all-zero vector in residual space (the
    generic program is subtracted out by construction), every cosine against it is 0.0, and the strict
    `>` scored it 0.000 while the header claimed it must be ~0.50."""
    o = [x for x in others if x is not None]
    if not o:
        return None
    return float(np.mean([1.0 if own > x else (0.5 if own == x else 0.0) for x in o]))


# ----------------------------------------------------------------- prompts
INSTR_MARKER = "Predict the response of"


def scramble_prompt(prompt, orig_drug, new_drug, new_moa, fields="both"):
    """Replace the drug name and/or the mechanism, leaving every other byte identical.

    Operates on the INSTRUCTION LINE wherever it sits, rather than on everything preceding the
    control marker. Under `--prompt_order drug_last` the instruction comes AFTER the control
    sentence, and the old pre/rest split on "Control cell:" would have found no drug name to
    replace and silently returned None -- dropping the arm without saying so.

    fields:
      "drug" -- swap the identity token only, keep the mechanism. Isolates whether the model reads
                the drug NAME.
      "moa"  -- swap the mechanism only, keep the name. Isolates whether the model reads the
                KNOWLEDGE CHANNEL it is already being given. This is the question the channel gate
                raises (mechanism carries real signal) and that the unseen-drug arm, being
                underpowered, cannot answer.
      "both" -- the original manipulation, retained for continuity.
    """
    lines = prompt.split("\n")
    idx = next((i for i, l in enumerate(lines) if l.startswith(INSTR_MARKER)), None)
    if idx is None:
        return None
    line = lines[idx]
    if fields in ("drug", "both"):
        if orig_drug not in line:
            return None
        line = line.replace(orig_drug, new_drug, 1)
    if fields in ("moa", "both") and new_moa:
        # lambda replacement: a plain string would re-interpret backslashes in the MoA text
        line = re.sub(r"(Mechanism:\s*).*$", lambda m: m.group(1) + new_moa + ".", line)
    lines[idx] = line
    out = "\n".join(lines)
    return out if out != prompt else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--model_path")
    ap.add_argument("--train_file", default=None, help="residual.jsonl (to flag/exclude trained conditions)")
    ap.add_argument("--holdout", default=None,
                    help="holdout.json manifest from build_residual_targets -> report train / "
                         "unseen_combo (cross-context transfer) / unseen_drug separately")
    ap.add_argument("--held_out_only", action="store_true")
    ap.add_argument("--min_split_n", type=int, default=40,
                    help="minimum scorable conditions before a split gets a verdict (a clustered CI over "
                         "a handful of points can exclude zero by chance)")
    ap.add_argument("--min_split_cell_lines", type=int, default=10)
    ap.add_argument("--partner_policy", choices=["within_plate", "any_plate"],
                    default="within_plate",
                    help="scramble partners are always a DIFFERENT drug. within_plate also requires "
                         "the same plate, so the swap differs from the target in the drug alone; "
                         "any_plate reproduces the published cross-plate pool for comparison.")
    ap.add_argument("--n_conditions", type=int, default=200)
    ap.add_argument("--prompt_order", choices=["drug_first", "drug_last"], default="drug_first",
                    help="MUST match the order the checkpoint was trained on. drug_last places the "
                         "instruction immediately before generation instead of several hundred "
                         "tokens upstream behind the control cell sentence.")
    ap.add_argument("--field_decomp", action="store_true",
                    help="add drug-only and mechanism-only scramble arms on the opposite stratum, "
                         "to separate 'reads the identity token' from 'reads the knowledge channel "
                         "it is already given'. Costs ~1.5x generation.")
    ap.add_argument("--split_quota", default=None,
                    help="stratified sampling by holdout split, e.g. "
                         "'train=200,unseen_combo=250,unseen_drug=150'. Overrides --n_conditions. A "
                         "UNIFORM sample reproduces the pool's proportions, so the held-out splits the "
                         "run exists to test arrive starved (250 built unseen_combo conditions yielded "
                         "only 33 scored in a 500-of-4091 draw). Spend the budget where n is short.")
    ap.add_argument("--k_samples", type=int, default=4)
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_drugs", type=int, default=4)
    ap.add_argument("--model_kind", choices=["residual", "cellsentence"], default="residual",
                    help="'residual' = model emits a signed [DOWN] signature (Arm 1b). 'cellsentence' = "
                         "an ORDINARY model (single-cell / consensus / OT): its generated treated cell is "
                         "decoded and converted to an IMPLIED residual (minus control, minus generic), so "
                         "old models are scored in the SAME space -- the apples-to-apples control.")
    ap.add_argument("--truth_from", default=None,
                    help="path to the target build's report.json. Reproduces that build's frame "
                         "exactly -- scope, leave-one-drug-out, split controls, reliability threshold "
                         "-- so the truth the model is scored against is the one it was trained on. "
                         "Omit ONLY when reproducing a published number in the published frame.")
    ap.add_argument("--truth_repro_thr", type=float, default=0.2,
                    help="reliability bar for the TRUTH residuals we score against (scoring against "
                         "an irreproducible truth is meaningless)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1400,
                    help="NEVER lower this. A signature is 100 up + [DOWN] + 100 down ~= 201 gene "
                         "symbols at 2-4 BPE tokens each. At 600 the old default, 26%% of generations "
                         "never reached [DOWN] and were scored with the whole down-block missing, which "
                         "HALVED a measured effect (Q15). One forgotten flag was worth +0.0716 vs +0.1429.")
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--split_comparator", default="scramble_orth",
                    choices=["scramble_orth", "scramble_opposite", "scramble_near"],
                    help="Comparator for the by-split table. Defaults to the NEUTRAL stratum. "
                         "scramble_opposite is NOT a null: it tracks the partner drug (measured NIR "
                         "0.403 at partner cosine -0.24 against 0.497 at 0.00), so a gap against it "
                         "mixes 'the model used the drug' with 'the comparator was pushed the other "
                         "way'. The old behaviour is reproduced with --split_comparator "
                         "scramble_opposite, and both are reported either way.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/residual_eval.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args); return
    if not (args.cache_dir and args.model_path):
        ap.error("--cache_dir and --model_path required (unless --selftest)")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ot")
    import build_residual_targets as brt

    panel = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    gene_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    rng = np.random.RandomState(args.seed)

    # THE TRUTH MUST BE BUILT IN THE FRAME THE MODEL WAS TRAINED IN.
    #
    # `build_residuals` is a compatibility shim whose defaults reproduce the PUBLISHED build -- cell-line
    # scope, no leave-one-drug-out, shared controls, a generic fitted over every condition. Those
    # defaults exist so old numbers stay regenerable, and they are exactly wrong for scoring a NEW
    # checkpoint: a model trained to emit plate-scoped, leave-one-drug-out residuals would be scored
    # against cell-line-scoped, non-LOO ones, and the resulting NIR would measure the mismatch between
    # two definitions rather than the model's skill.
    #
    # `--truth_from <report.json>` reads the build's own report and reproduces its frame exactly, so the
    # two can never drift. Without it the published frame is used and the run says so loudly, because
    # silently scoring against the wrong target is the failure this whole remediation is about.
    truth_cfg = dict(generic_scope="cell_line", loo=False, split_controls=False,
                     repro_thr=args.truth_repro_thr, holdout=None, eval_filter=True)
    if args.truth_from:
        rep_doc = json.load(open(args.truth_from))
        truth_cfg.update(
            generic_scope=rep_doc.get("scope", "plate"),
            loo=bool(rep_doc.get("leave_one_drug_out", True)),
            split_controls=not bool(rep_doc.get("reliability_controls", {})
                                    .get("shared_control_reliability", False)),
            repro_thr=float(rep_doc.get("repro_thr", args.truth_repro_thr)),
            holdout=(args.holdout if args.holdout and os.path.exists(args.holdout) else None),
            # THE FLAG THE BUILD RECORDED AND NOBODY READ. `report.json` has carried
            # `eval_repro_filter` since the split-before-fit rebuild, but this branch propagated
            # scope / loo / split_controls / repro_thr and silently dropped it -- so held-out truth
            # was reliability-filtered even when the build had deliberately not filtered it. That is
            # selection on the outcome, on the evaluation set.
            eval_filter=bool(rep_doc.get("eval_repro_filter", False)),
        )
        logger.info(f"truth frame taken from {args.truth_from}: scope={truth_cfg['generic_scope']} "
                    f"loo={truth_cfg['loo']} split_controls={truth_cfg['split_controls']} "
                    f"repro_thr={truth_cfg['repro_thr']:.4f} "
                    f"eval_repro_filter={truth_cfg['eval_filter']} "
                    f"generic fitted on {'TRAIN ONLY' if truth_cfg['holdout'] else 'all conditions'}")
        if truth_cfg["eval_filter"]:
            logger.warning("the build filtered HELD-OUT conditions on their own split-half "
                           "reproducibility, so this evaluation runs on the reliability-surviving "
                           "subset rather than the full holdout. That is outcome selection on the "
                           "evaluation set: it retains the cleanest conditions, so it flatters "
                           "absolute scores and is CONSERVATIVE for a null result. Rebuild without "
                           "--eval_repro_filter for the primary number.")
        if truth_cfg["holdout"] is None:
            logger.warning("no --holdout given, so the truth's generic is fitted over every condition "
                           "and any held-out split scored against it is transductive")
    else:
        logger.warning("=" * 96)
        logger.warning("NO --truth_from: scoring against the PUBLISHED truth frame (cell-line scope, "
                       "no leave-one-drug-out, shared controls, generic over all conditions).")
        logger.warning("That is correct ONLY when reproducing a published number. For a checkpoint "
                       "trained on repaired targets it scores the model against a DIFFERENT quantity "
                       "than it learned. Pass --truth_from <build>/report.json.")
        logger.warning("=" * 96)

    # The manifest's COMPLETE pre-filter split, so `transform` sees the real labels. Without this
    # the shim forces every label to "train", which makes its reliability filter fire on the
    # `s == "train"` disjunct for every condition and renders eval_filter inert.
    truth_split = None
    if truth_cfg["holdout"]:
        _hm = json.load(open(truth_cfg["holdout"]))
        _sa = _hm.get("split_all")
        if _sa:
            # passed with the manifest's own string keys; build_residuals.normalise_split maps them
            # onto condition keys in string space, because tuple(k.split("|")) is type-lossy
            truth_split = dict(_sa)
            logger.info(f"split labels for the truth build: {len(truth_split)} conditions from "
                        f"split_all (the complete pre-filter assignment)")
        else:
            logger.warning("this holdout.json predates `split_all`; the truth build cannot "
                           "distinguish train from held-out conditions and will filter both. "
                           "Rebuild the targets before quoting held-out numbers.")

    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control,
        repro_thr=truth_cfg["repro_thr"], seed=args.seed,
        generic_scope=truth_cfg["generic_scope"], loo=truth_cfg["loo"],
        split_controls=truth_cfg["split_controls"], holdout=truth_cfg["holdout"],
        eval_filter=truth_cfg["eval_filter"], split=truth_split)
    logger.info(f"conditions with residuals: {len(kept)}")
    cvcl, moa_of, conc_of = brt.load_meta_maps()

    split_of = {}
    if args.holdout and os.path.exists(args.holdout):
        hm = json.load(open(args.holdout))
        for kstr, v in hm["split"].items():
            split_of[tuple(kstr.split("|"))] = v
        from collections import Counter as _C
        logger.info(f"holdout manifest: {dict(_C(split_of.values()))}  "
                    f"({len(hm.get('holdout_drugs', []))} unseen drugs, "
                    f"{len(hm.get('holdout_combos', []))} unseen combos)")

    trained = set()
    if args.train_file and os.path.exists(args.train_file):
        for line in open(args.train_file):
            m = json.loads(line).get("metadata", {})
            # the fourth element is the treated WELL. Builds before the sample/dose fix wrote it
            # under the name `dose_float`, so accept either -- but never mix them into one key.
            trained.add((m.get("drug"), m.get("cell_line_id"), m.get("plate"),
                         m.get("sample_id", m.get("dose_float"))))
        logger.info(f"training file lists {len(trained)} conditions")

    by_cl = defaultdict(list)
    for k in kept:
        by_cl[k[1]].append(k)
    by_cl = {c: ks for c, ks in by_cl.items() if len(ks) >= args.min_drugs}
    truth = {k: signed_rank_from_vector(kept[k]["residual"], P, args.k_sig) for c in by_cl for k in by_cl[c]}
    # CEILING must use DISJOINT halves: half-B scored against half-A truths. (Scoring half-B against the
    # FULL residual is self-overlapping -- that bug returned a meaningless ceiling of 1.000.)
    truth_A = {k: signed_rank_from_vector(kept[k]["residual_A"], P, args.k_sig)
               for c in by_cl for k in by_cl[c] if "residual_A" in kept[k]}

    # model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(dev).eval()
    ec = tok.encode(END, add_special_tokens=False)
    end_id = ec[0] if len(ec) == 1 else tok.convert_tokens_to_ids(END)
    eos = [end_id] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])

    def generate(prompts, tag=""):
        """Sampled generation, seeded from `tag` so a rerun reproduces byte-for-byte.

        Without this the sampler advanced from global torch state, so two runs of the same command
        gave different generations and no reported number could be reproduced exactly -- including
        the model-minus-scramble contrasts, whose two arms were drawn from different points of one
        stream. `tag` is (condition, arm, replicate), so every arm of every condition gets its own
        deterministic stream and the arms stay independent of each other's batch position.
        """
        prev = tok.padding_side; tok.padding_side = "left"; outs = []
        try:
            for i in range(0, len(prompts), args.gen_batch_size):
                enc = tok(prompts[i:i + args.gen_batch_size], return_tensors="pt", padding=True).to(dev)
                gen = torch.Generator(device=dev)
                gen.manual_seed((zlib.crc32(f"{args.seed}|{tag}|{i}".encode()) % (2 ** 31)) or 1)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                       pad_token_id=tok.pad_token_id, eos_token_id=eos, do_sample=True,
                                       temperature=max(args.temperature, 1e-2), top_p=args.top_p)
                pl = enc["input_ids"].shape[1]
                for j in range(g.shape[0]):
                    ids = g[j][pl:].tolist()
                    if end_id in ids:
                        ids = ids[:ids.index(end_id)]
                    outs.append(tok.decode(ids, skip_special_tokens=False).strip())
        finally:
            tok.padding_side = prev
        return outs

    # for the cell-sentence control we need: an expression rank->value profile, each condition's control
    # pseudobulk, and the per-cell-line generic shift (to convert a generated cell into a residual)
    rank_prof, ctrl_pb = None, {}
    if args.model_kind == "cellsentence":
        allrows = np.concatenate([r for r in ctrl_rows.values()])[:4000]
        rank_prof = rank_value_profile(X, allrows, P)
        for c in by_cl:
            for k in by_cl[c]:
                rows = ctrl_rows[kept[k]["group"]]
                ctrl_pb[k] = np.asarray(np.log1p(X[rows].todense()).mean(0)).ravel().astype(np.float32)
        logger.info("cell-sentence control mode: generations decoded -> implied residual "
                    "(minus control pseudobulk, minus per-cell-line generic shift)")

    # STRATIFIED SCRAMBLE PARTNERS. A random other drug may be a near-twin, in which case an unchanged
    # output is CORRECT rather than blind -- that biases model-scramble toward zero. We therefore swap to
    # three defined strata by residual cosine within the cell line:
    #   near     = most SIMILAR drug      (weakest test; a small gap here is expected and fine)
    #   orth     = most ORTHOGONAL (cos~0, unrelated program)
    #   opposite = most ANTI-correlated   (sharpest test: the drug with the opposite signature)
    # If the model truly reads the drug, the gap should GROW from near -> orth -> opposite.
    #
    # TWO CONSTRAINTS ON THE PARTNER POOL, both required for the stratum labels to mean what they say.
    #   DIFFERENT DRUG. `b != a` allowed the SAME drug at another dose or plate to be the partner. A
    #     "scramble" that substitutes a drug for itself is not a scramble, an unchanged output is
    #     correct rather than blind, and the `near` stratum -- the one the monotone-gradient argument
    #     leans on -- was the stratum most likely to be contaminated that way, since a drug's own
    #     other doses are usually its nearest neighbours.
    #   SAME PLATE. Residuals retain plate structure, so a cross-plate partner differs from the target
    #     in batch as well as in drug, and the gap would absorb both.
    partners, n_no_pool = {}, 0
    for c, ks in by_cl.items():
        for a in ks:
            pool = [b for b in ks if b[0] != a[0]
                    and (args.partner_policy == "any_plate" or b[2] == a[2])]
            cs = [(cos(kept[a]["residual"], kept[b]["residual"]), b) for b in pool]
            if len(cs) < 3:
                n_no_pool += 1
                continue
            partners[a] = {"near": max(cs)[1],
                           "orth": min(cs, key=lambda t: abs(t[0]))[1],
                           "opposite": min(cs)[1],
                           "cos_near": max(cs)[0],
                           "cos_orth": min(cs, key=lambda t: abs(t[0]))[0],
                           "cos_opposite": min(cs)[0]}
    logger.info(f"stratified scramble partners built for {len(partners)} conditions "
                f"(policy={args.partner_policy}, different drug enforced); {n_no_pool} conditions "
                f"have too few eligible partners and are skipped")

    # ---------------- BASELINES in residual space (model-vs-scramble alone says the model REACTS to the
    # drug token, not that it is any good). Each is a predictor of the drug-specific residual:
    #   drug_lookup : this drug's mean residual measured in OTHER cell lines -> the cross-context lookup
    #                 table. THE bar to beat: a lookup can memorise a drug, only a model can adapt it.
    #   moa_lookup  : mean residual of OTHER drugs sharing this drug's MoA in the same cell line.
    #                 Diagnostic for the MoA leak: our prompt contains 'Mechanism: {moa}', so if the model
    #                 merely reads the MoA it should not beat this.
    #   control_copy: predicting the control cell -> residual = -generic (constant) -> chance by
    #                 construction. Confirms the residual frame is leak-proof (control-copy scores 0.766
    #                 in full-profile space).
    #   generic     : predicting the average drug response -> residual = 0 -> chance by construction.
    # TRAINING-ONLY BY DEFAULT. These lookups were fitted from the full cache, which means that on a
    # held-out condition they could read residuals the model was never shown -- an ORACLE, not a
    # baseline, and the bar the thesis says the model fails to clear. `by_drug_all` is retained so the
    # oracle can still be reported, explicitly labelled, alongside the fair version.
    by_drug_all, by_drug_train = defaultdict(list), defaultdict(list)
    for k in kept:
        by_drug_all[k[0]].append(k)
        if not trained or k in trained:
            by_drug_train[k[0]].append(k)
    if trained:
        logger.info(f"lookup baselines fitted from {sum(len(v) for v in by_drug_train.values())} "
                    f"TRAINING conditions (oracle variants over all "
                    f"{sum(len(v) for v in by_drug_all.values())} reported separately)")
    else:
        logger.warning("no --train_file given: the lookup baselines are ORACLES over the full cache "
                       "and are labelled as such")

    def bl_drug_lookup(key, pool=None):
        d, c = key[0], key[1]
        oth = [kept[k2]["residual"] for k2 in (pool or by_drug_train)[d] if k2[1] != c]
        return np.mean(np.stack(oth), 0) if oth else None

    rng_bl = np.random.RandomState(args.seed + 977)   # private: must not shift the generation stream

    def bl_drug_lookup_1(key):
        """Same drug, ONE randomly chosen other CELL LINE -- no cross-line averaging. drug_lookup averages
        over many cell lines and so is DENOISED, while the ceiling is one noisy half against another; that
        alone could explain a lookup sitting at the ceiling. This arm removes the averaging advantage.

        Pick a LINE and average that line's conditions, do NOT pick one condition: 62% of drugs are
        multi-dose (Q11), so a single condition confounds 'different cell line' with 'different dose' and
        a low score would be unreadable.

        READ IT AGAINST THE CEILING, NOT AGAINST drug_lookup. Under exact context-independence this is a
        full-sample estimate of what the ceiling estimates from a half, so it should MEET or EXCEED the
        ceiling; drug_lookup_1 < ceiling is conservative evidence that the residual IS cell-line-specific."""
        d, c = key[0], key[1]
        lines = sorted({k2[1] for k2 in by_drug_train[d] if k2[1] != c})
        if not lines:
            return None
        pick = lines[rng_bl.randint(len(lines))]
        v = [kept[k2]["residual"] for k2 in by_drug_train[d] if k2[1] == pick]
        return np.mean(np.stack(v), 0) if v else None

    def bl_moa_lookup(key, oracle=False):
        d, c = key[0], key[1]
        m = moa_of.get(d)
        if not m or m in ("unclear", "unknown", "nan", "None"):
            return None
        same = [kept[k2]["residual"] for k2 in by_cl.get(c, [])
                if k2[0] != d and moa_of.get(k2[0]) == m
                and (oracle or not trained or k2 in trained)]
        return np.mean(np.stack(same), 0) if same else None

    gen_stats = {"n": 0, "has_down": 0, "up_len": [], "dn_len": [], "valid_frac": [], "dup_frac": []}

    def track(gens):
        for g in gens:
            toks = [t for t in g.replace(END, " ").split() if t]
            gen_stats["n"] += 1
            if DOWN in toks:
                gen_stats["has_down"] += 1
                d = toks.index(DOWN)
                gen_stats["up_len"].append(d); gen_stats["dn_len"].append(len(toks) - d - 1)
            genes = [t for t in toks if t != DOWN]
            if genes:
                gen_stats["valid_frac"].append(np.mean([t in gene_index for t in genes]))
                gen_stats["dup_frac"].append(1.0 - len(set(genes)) / len(genes))

    pred_by_cl = defaultdict(list)   # for the mode-collapse check
    recs = []
    cond_list = [k for c in by_cl for k in by_cl[c]]
    if args.held_out_only and trained:
        cond_list = [k for k in cond_list if k not in trained]
        logger.info(f"held-out-only: {len(cond_list)} conditions never trained on")
    rng.shuffle(cond_list)
    if split_of and args.split_quota:
        quota = {}
        for part in args.split_quota.split(","):
            k, _, v = part.partition("=")
            quota[k.strip()] = int(v)
        by_split_pool = defaultdict(list)
        for k in cond_list:
            by_split_pool[split_of.get(tuple(map(str, k)), "unknown")].append(k)
        picked, short = [], []
        for name, want in quota.items():
            avail = by_split_pool.get(name, [])
            picked.extend(avail[:want])
            if len(avail) < want:
                short.append(f"{name} {len(avail)}/{want}")
        rng.shuffle(picked)
        cond_list = picked
        logger.info("stratified sample: " + ", ".join(
            f"{n}={min(len(by_split_pool.get(n, [])), w)}" for n, w in quota.items())
            + (f"   [SHORT: {'; '.join(short)}]" if short else ""))
    else:
        cond_list = cond_list[:args.n_conditions]

    for n, key in enumerate(cond_list):
        d, c, p, ds = key
        # NEGATIVES EXCLUDE SAME-DRUG SIBLINGS. The comparison set asks "is this prediction closer
        # to its own truth than to OTHER DRUGS' truths". Leaving the drug's own other doses in the
        # gallery asks a different and much harder question -- can you tell this drug's 0.5 uM from
        # its own 5 uM -- and every arm was being scored against that instead.
        others = [k for k in by_cl[c] if k[0] != key[0]]
        if len(others) < 3:
            continue
        rows = ctrl_rows.get(kept[key]["group"])
        if rows is None or len(rows) == 0:
            continue
        ctrl_vec = np.asarray(X[rows[rng.randint(len(rows))]].todense()).ravel()
        prompt = brt.format_prompt(cvcl.get(c, c), d, brt.parse_dose(conc_of.get(ds, "unknown")),
                                   moa_of.get(d, "unclear"), brt.expr_to_sentence(ctrl_vec, panel),
                                   order=args.prompt_order)
        # scramble arms across SIMILARITY STRATA (see partners{} above)
        if key not in partners:
            continue
        pinfo = partners[key]
        arms = {"model": generate([prompt] * args.k_samples, tag=f"{key}|model")}
        for strat in ("near", "orth", "opposite"):
            bkey = pinfo[strat]
            sp = scramble_prompt(prompt, d, bkey[0], moa_of.get(bkey[0], "unclear"))
            if sp:
                arms[f"scramble_{strat}"] = generate([sp] * args.k_samples,
                                                     tag=f"{key}|scramble_{strat}")
        # FIELD DECOMPOSITION: which part of the prompt is actually read? Run on the OPPOSITE
        # partner only -- the sharpest stratum -- to keep the generation cost to 1.5x.
        if args.field_decomp:
            bkey = pinfo["opposite"]
            for fld in ("drug", "moa"):
                sp = scramble_prompt(prompt, d, bkey[0], moa_of.get(bkey[0], "unclear"), fields=fld)
                if sp:                       # None when the swap would not change the prompt,
                    arms[f"scramble_{fld}only"] = generate([sp] * args.k_samples,
                                                          tag=f"{key}|scramble_{fld}only")
        for g in arms.values():
            track(g)
        row = {"drug": d, "cell_line": c, "plate": p, "sample_id": ds,
               "trained": key in trained,
               "split": split_of.get(tuple(map(str, key)), "unknown"),
               "repro_cos": kept[key]["repro_cos"],
               "swap_near": pinfo["near"][0], "swap_orth": pinfo["orth"][0],
               "swap_opposite": pinfo["opposite"][0],
               "cos_near": pinfo["cos_near"], "cos_orth": pinfo["cos_orth"],
               "cos_opposite": pinfo["cos_opposite"]}
        oth_truth = [truth[k] for k in others]
        for arm, gens in arms.items():
            if args.model_kind == "residual":
                v = np.mean(np.stack([signed_rank_from_sentence(g, gene_index, P, args.k_sig)
                                      for g in gens]), axis=0)
            else:
                # ordinary model: decode the generated treated cell, subtract control + generic ->
                # implied residual, then the SAME signed-rank encoding as the residual model
                expr = np.mean(np.stack([sentence_to_expression(g, gene_index, P, rank_prof)
                                         for g in gens]), axis=0)
                v = signed_rank_from_vector(expr - ctrl_pb[key] - generic[c], P, args.k_sig)
            row[arm] = nir_from_sims(cos(v, truth[key]), [cos(v, t) for t in oth_truth])
            if arm == "model":
                row["cos_pred_truth"] = cos(v, truth[key])          # direct directional agreement
                row["cos_pred_others"] = float(np.mean([cos(v, t) for t in oth_truth]))
                pred_by_cl[c].append((key, v))                      # for the mode-collapse check
        # CEILING: real half-B replicate scored against DISJOINT half-A truths (no cell overlap)
        if "residual_B" in kept[key] and key in truth_A:
            rB = signed_rank_from_vector(kept[key]["residual_B"], P, args.k_sig)
            othA = [truth_A[k] for k in others if k in truth_A]
            if othA:
                row["ceiling"] = nir_from_sims(cos(rB, truth_A[key]), [cos(rB, t) for t in othA])
        rv = np.zeros(P, np.float32); idx = rng.choice(P, 2 * args.k_sig, replace=False)
        rv[idx[:args.k_sig]] = 1.0; rv[idx[args.k_sig:]] = -1.0
        row["random"] = nir_from_sims(cos(rv, truth[key]), [cos(rv, t) for t in oth_truth])
        # ---- BASELINES (scored identically: same truths, same NIR, same comparison set) ----
        for bname, bvec in (("drug_lookup", bl_drug_lookup(key)),
                            ("drug_lookup_1", bl_drug_lookup_1(key)),
                            ("moa_lookup", bl_moa_lookup(key)),
                            # ORACLES: fitted over the full cache, so on a held-out condition they
                            # read residuals the model was never shown. Reported so the gap between
                            # a fair lookup and an oracle one is visible instead of being the number.
                            ("drug_lookup_oracle", bl_drug_lookup(key, by_drug_all)),
                            ("moa_lookup_oracle", bl_moa_lookup(key, oracle=True)),
                            ("control_copy", -generic[c]),
                            ("generic", np.zeros(P, np.float32))):
            if bvec is None:
                continue
            bs = signed_rank_from_vector(np.asarray(bvec, dtype=np.float32), P, args.k_sig)
            row[bname] = nir_from_sims(cos(bs, truth[key]), [cos(bs, t) for t in oth_truth])
        recs.append(row)
        if n % 20 == 0:
            logger.info(f"  {n}/{len(cond_list)} conditions scored")

    if not recs:
        logger.error("no conditions scored"); return
    report(recs, args, rng, gen_stats, pred_by_cl, kept)


def _two_way_ci(recs, key_a, key_b):
    """Cameron-Gelbach-Miller two-way cluster-robust interval for the mean paired difference.

    THE DESIGN IS CROSSED, NOT NESTED, and that is why neither single clustering is correct. The
    experimental-unit audit measured 48.6 cell lines per treated well across 289 wells: a well
    contains many cell lines AND a cell line appears in many wells. Two conditions sharing a well
    share the drug assignment and the batch; two conditions sharing a cell line share the biology.
    Clustering on cell lines alone (about 50 clusters) ignores the first; clustering on wells alone
    (289 clusters) ignores the second and, because there are more wells than lines, actually gives a
    NARROWER interval -- so "cluster on the well instead" would have been the anti-conservative move.

        V_2way = V_line + V_well - V_intersection

    The intersection of the two groupings is the condition itself, so V_intersection is the ordinary
    independent-sampling variance. The estimator can go negative in finite samples; when it does we
    fall back to the wider of the two one-way variances and say so.
    """
    rows = [r for r in recs if r.get(key_a) is not None and r.get(key_b) is not None]
    if len(rows) < 3:
        return None
    d = np.array([r[key_a] - r[key_b] for r in rows], float)
    n = len(d)
    m = float(d.mean())
    dev = d - m

    def v_of(keyname):
        g = defaultdict(list)
        for i, r in enumerate(rows):
            g[r.get(keyname, r["cell_line"])].append(i)
        if len(g) < 2:
            return None
        s = sum(dev[idx].sum() ** 2 for idx in g.values())
        return (len(g) / max(1, len(g) - 1)) * s / (n ** 2)

    v_line, v_well = v_of("cell_line"), v_of("sample_id")
    v_int = float((dev ** 2).sum()) / (n ** 2)
    if v_line is None or v_well is None:
        v = v_line if v_well is None else v_well
        note = "one_way_only"
    else:
        v = v_line + v_well - v_int
        note = "two_way"
        if v <= 0:
            v, note = max(v_line, v_well), "two_way_negative_fell_back_to_wider_one_way"
    se = float(np.sqrt(max(v, 0.0)))
    n_line = len({r["cell_line"] for r in rows})
    n_well = len({r.get("sample_id") for r in rows})
    # t, not 1.96: a cluster-robust variance is asymptotic in the CLUSTER COUNT, and the binding
    # count here is the wells -- as few as 25 on unseen_drug, where t(24)=2.064 is 5% wider.
    df = max(1, min(n_line, n_well) - 1)
    z = inf.crit(0.05, df)
    return m, m - z * se, m + z * se, n, f"{n_line}L/{n_well}W:{note}:t{df}"


def _clustered_ci(recs, key_a, key_b, rng, n_boot, cluster="cell_line"):
    """Clustered bootstrap. `cluster` selects the independence unit.

    Every interval in the thesis resampled CELL LINES. But Tahoe assigns a treatment to a WELL that
    carries many cell lines, so two conditions from one well are not two independent assignments of
    that drug -- they are one assignment observed twice. Resampling cell lines treats them as
    independent and produces intervals that are too narrow by whatever the well-level ICC is.
    `cluster="sample_id"` resamples wells instead, which is the unit the design supports.
    """
    if cluster == "two_way":
        return _two_way_ci(recs, key_a, key_b)
    pairs = [(r.get(cluster, r["cell_line"]), r[key_a] - r[key_b]) for r in recs
             if r.get(key_a) is not None and r.get(key_b) is not None]
    if not pairs:
        return None
    cls = np.array([c for c, _ in pairs]); dif = np.array([d for _, d in pairs])
    u = list(set(cls.tolist())); boot = []
    for _ in range(n_boot):
        take = rng.choice(len(u), len(u), replace=True)
        boot.append(dif[np.concatenate([np.where(cls == u[t])[0] for t in take])].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(dif.mean()), float(lo), float(hi), len(pairs), len(u)


def report(recs, args, rng, gen_stats=None, pred_by_cl=None, kept=None):
    logger.info("=" * 100)
    logger.info(f"RESIDUAL-SPACE NIR  (chance 0.50)   n={len(recs)} conditions, "
                f"{len(set(r['cell_line'] for r in recs))} cell lines   model_kind={args.model_kind}")

    # --- (0) GENERATION VALIDITY: if the outputs are malformed, nothing below is meaningful ---
    if gen_stats and gen_stats["n"]:
        g = gen_stats
        logger.info(f"  [validity] {g['n']} generations | has [DOWN] {100*g['has_down']/g['n']:.0f}% | "
                    f"up-block {np.mean(g['up_len']) if g['up_len'] else 0:.0f} / "
                    f"down-block {np.mean(g['dn_len']) if g['dn_len'] else 0:.0f} genes | "
                    f"valid panel genes {100*np.mean(g['valid_frac']):.1f}% | "
                    f"duplicates {100*np.mean(g['dup_frac']):.1f}%")

    means = {}
    for a in ("ceiling", "model", "scramble_near", "scramble_orth", "scramble_opposite",
              "scramble_drugonly", "scramble_moaonly",
              "drug_lookup", "drug_lookup_1", "moa_lookup", "control_copy", "generic", "random"):
        v = [r[a] for r in recs if r.get(a) is not None]
        if v:
            means[a] = float(np.mean(v))
            tag = ""
            if a == "drug_lookup":
                tag = "   <- THE BAR: a lookup memorises a drug; only a model can adapt it"
            elif a == "drug_lookup_1":
                tag = "   <- same, ONE other cell line (no averaging) -> isolates transfer from denoising"
            elif a == "moa_lookup":
                tag = "   <- MoA-leak diagnostic (our prompt contains 'Mechanism:')"
            elif a in ("control_copy", "generic"):
                tag = "   <- must be ~0.50 (drug-agnostic by construction)"
            logger.info(f"    {a:18s} {means[a]:.3f}   (n={len(v)}){tag}")

    # COMMON SUPPORT. Each arm above is averaged over its OWN non-null records, so `model` (every
    # condition) and `drug_lookup` (only conditions with a lookup) are means over DIFFERENT SETS.
    # Comparing their chance-to-ceiling coverage across unequal supports is not a comparison; on the
    # 449 conditions where both exist the model reads 18.6% rather than 15.2%. The paired contrast
    # further down needs none of this and is the number to quote.
    common = [r for r in recs if r.get("drug_lookup") is not None and r.get("model") is not None
              and r.get("ceiling") is not None]
    if common:
        cm = float(np.mean([r["model"] for r in common]))
        cc = float(np.mean([r["ceiling"] for r in common]))
        cl = float(np.mean([r["drug_lookup"] for r in common]))
        span = cc - 0.5
        means["common_support"] = {
            "n": len(common), "model": cm, "ceiling": cc, "drug_lookup": cl,
            "coverage_model": ((cm - 0.5) / span if span > 0 else None),
            "coverage_lookup": ((cl - 0.5) / span if span > 0 else None)}
        if span > 0:
            logger.info(f"    -- on the {len(common)} conditions where BOTH model and lookup exist: "
                        f"model {cm:.3f}, lookup {cl:.3f}, ceiling {cc:.3f}")
            logger.info(f"       chance-to-ceiling coverage  model {(cm - 0.5) / span:.1%}   "
                        f"lookup {(cl - 0.5) / span:.1%}   (quote the PAIRED gap, not this ratio)")

    # HEADROOM: what is left for ANY conditional model above a cross-cell-line lookup table. If this is
    # ~0 the task has collapsed to retrieval -- the drug residual carries no cell-line-specific component
    # resolvable above replicate noise, and no amount of modelling can recover one that is not there.
    if "ceiling" in means and "drug_lookup" in means:
        hr = means["ceiling"] - means["drug_lookup"]
        logger.info(f"    >>> ceiling - drug_lookup = {hr:+.3f}  (NOT a headroom estimate -- see below)")
        logger.info("        This difference CANNOT be read as 'how much a model could add'. Three "
                    "asymmetries all favour the lookup: (i) the ceiling is scored against a HALF-sample "
                    "truth while every baseline is scored against the FULL-sample truth; (ii) drug_lookup "
                    "averages many cell lines while the ceiling is one noisy half vs another; (iii) NIR "
                    "near 0.96 is compressive, so a wide range of cosines collapses into a few thousandths. "
                    "A lookup that merely TIES a split-half ceiling in cosine is still compatible with much "
                    "of the residual being cell-line-specific. Use drug_lookup_1 vs ceiling, and a "
                    "noise-corrected transfer coefficient, to answer that question.")

    # head-to-head against each baseline, clustered CI -- the "is the model any GOOD" question,
    # which is separate from "does the model react to the drug token" (model - scramble).
    logger.info("-" * 100)
    logger.info("  model - BASELINE. Two intervals: clustered over CELL LINES (as previously")
    logger.info("  reported), and TWO-WAY cluster-robust over lines AND wells. The design is crossed --")
    logger.info("  48.6 cell lines per well, each line in many wells -- so neither grouping alone")
    logger.info("  captures the dependence, and there are MORE wells than lines, which means")
    logger.info("  clustering on wells alone would have been the anti-conservative choice.")
    logger.info("  `_oracle` rows are fitted over the FULL cache and are not baselines the model could")
    logger.info("  fairly be asked to beat; they bound how much the fair lookups were inflated.")
    base_out = {}
    for b in ("drug_lookup", "drug_lookup_1", "moa_lookup",
              "drug_lookup_oracle", "moa_lookup_oracle", "control_copy", "generic"):
        r = _clustered_ci(recs, "model", b, rng, args.n_boot)
        if not r:
            continue
        m, lo, hi, n, ncl = r
        base_out[b] = {"gap": m, "ci": [lo, hi], "n": n, "n_cell_lines": ncl}
        w = _clustered_ci(recs, "model", b, rng, args.n_boot, cluster="two_way")
        if w:
            _, wlo, whi, _, note = w
            base_out[b]["ci_two_way"] = [wlo, whi]
            base_out[b]["two_way_note"] = note
            widen = (whi - wlo) / max(1e-9, hi - lo)
            base_out[b]["two_way_widening"] = round(widen, 3)
            logger.info(f"    model - {b:20s} {m:+.4f}  line [{lo:+.4f}, {hi:+.4f}]  "
                        f"2way [{wlo:+.4f}, {whi:+.4f}] (x{widen:.2f}, {note})  n={n}  "
                        f"{'model WINS' if wlo > 0 else ('model LOSES' if whi < 0 else 'tie')}")
        else:
            logger.info(f"    model - {b:20s} {m:+.4f}  line [{lo:+.4f}, {hi:+.4f}]  n={n}")
    means["vs_baselines"] = base_out

    # --- (0b) FIELD DECOMPOSITION: which part of the prompt does the model actually read? ---
    if any(r.get("scramble_drugonly") is not None for r in recs):
        logger.info("-" * 100)
        logger.info("  WHICH FIELD IS READ? Each arm swaps ONE part of the instruction to the "
                    "opposite-signature drug and leaves the rest byte-identical.")
        fd = {}
        for a, what in (("scramble_drugonly", "drug NAME only (mechanism kept)"),
                        ("scramble_moaonly", "MECHANISM only (name kept)"),
                        ("scramble_opposite", "both (the standard arm)")):
            r = _clustered_ci(recs, "model", a, rng, args.n_boot)
            if r:
                m, lo, hi, n, _ = r
                fd[a] = {"gap": m, "ci": [lo, hi], "n": n}
                logger.info(f"    swap {what:34s} gap = {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  n={n}  "
                            f"{'READ' if lo > 0 else 'not detected'}")
        means["field_decomposition"] = fd
        dm = fd.get("scramble_moaonly")
        if dm is not None:
            if dm["ci"][0] > 0:
                logger.info("    >>> the model DOES read the mechanism string it is given. A "
                            "channel-conditioned prompt (adding protein targets) is therefore "
                            "well motivated.")
            else:
                logger.info("    >>> the model does NOT measurably read the mechanism string it is "
                            "already given -- and the channel gate shows that field carries real "
                            "signal. The bottleneck for unseen drugs is the READOUT, not the "
                            "information, so adding a target field to the prompt would likely "
                            "change nothing.")

    # --- (1) STRATIFIED model - scramble: the gap should GROW near -> orth -> opposite ---
    logger.info("-" * 100)
    logger.info("  model - scramble BY SIMILARITY STRATUM (a random swap can land on a near-twin, where")
    logger.info("  an unchanged output is CORRECT; the opposite-signature swap is the sharpest test):")
    strat_out = {}
    for strat in ("near", "orth", "opposite"):
        r = _clustered_ci(recs, "model", f"scramble_{strat}", rng, args.n_boot)
        rw = _clustered_ci(recs, "model", f"scramble_{strat}", rng, args.n_boot,
                           cluster="two_way")
        if r:
            m, lo, hi, n, ncl = r
            mc = np.mean([x[f"cos_{strat}"] for x in recs if x.get(f"cos_{strat}") is not None])
            strat_out[strat] = {"gap": m, "ci": [lo, hi], "n": n, "mean_cos": float(mc)}
            if rw:
                _, wlo, whi, _, note = rw
                strat_out[strat]["ci_two_way"] = [wlo, whi]
                strat_out[strat]["two_way_note"] = note
            # the TWO-WAY interval decides: the design is crossed, so neither cell lines nor wells
            # alone capture the dependence (see _two_way_ci)
            dlo, dhi = (rw[1], rw[2]) if rw else (lo, hi)
            flag = "CI excludes 0" if dlo > 0 else "CI spans 0"
            logger.info(f"    {strat:9s} (mean cos(A,B)={mc:+.2f})  gap = {m:+.4f}  "
                        f"line [{lo:+.4f}, {hi:+.4f}]  2way [{dlo:+.4f}, {dhi:+.4f}]  {flag}")
    if len(strat_out) == 3:
        mono = strat_out["opposite"]["gap"] > strat_out["near"]["gap"]
        logger.info(f"    >>> gap grows with dissimilarity: {'YES' if mono else 'NO'} "
                    f"(near {strat_out['near']['gap']:+.4f} -> opposite {strat_out['opposite']['gap']:+.4f}). "
                    f"{'Consistent with genuine drug use.' if mono else 'A flat profile is a red flag.'}")
        best = strat_out["opposite"]
        blo, bhi = best.get("ci_two_way", best["ci"])
        logger.info(f"    >>> HEADLINE (opposite-signature swap, different drug, same plate): "
                    f"{best['gap']:+.4f} two-way clustered CI [{blo:+.4f}, {bhi:+.4f}] -> "
                    f"{'DRUG USE' if blo > 0 else 'NULL'}")

    # --- (2) MODE COLLAPSE: are predictions for different drugs actually different? ---
    if pred_by_cl and kept:
        pc, tc = [], []
        for c, lst in pred_by_cl.items():
            if len(lst) < 3:
                continue
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    pc.append(cos(lst[i][1], lst[j][1]))
                    tc.append(cos(kept[lst[i][0]]["residual"], kept[lst[j][0]]["residual"]))
        if pc:
            logger.info("-" * 100)
            logger.info(f"  [diversity] mean pairwise cos between PREDICTIONS for different drugs = "
                        f"{np.mean(pc):+.3f}  vs between TRUTHS = {np.mean(tc):+.3f}")
            logger.info(f"              (predictions ~1.0 => mode collapse: the model emits one profile "
                        f"regardless of drug, which would make any gap trivially ~0)")

    # --- (3) INSTRUMENT VALIDITY: does the model do better where the truth is more reproducible? ---
    rc = [(r["repro_cos"], r["model"]) for r in recs
          if r.get("repro_cos") is not None and r.get("model") is not None]
    if len(rc) > 10:
        a = np.array([x for x, _ in rc]); b = np.array([y for _, y in rc])
        logger.info(f"  [validity] corr(condition reproducibility, model NIR) = {np.corrcoef(a, b)[0,1]:+.3f} "
                    f"(positive => the model does better where the drug effect is real)")
    cpt = [r["cos_pred_truth"] for r in recs if r.get("cos_pred_truth") is not None]
    cpo = [r["cos_pred_others"] for r in recs if r.get("cos_pred_others") is not None]
    if cpt:
        logger.info(f"  [direction] cos(prediction, OWN truth) = {np.mean(cpt):+.3f}  vs "
                    f"cos(prediction, OTHER drugs) = {np.mean(cpo):+.3f}")

    # --- (4) GENERALIZATION: the three-way split (the decisive test) ---
    splits = defaultdict(list)
    for r in recs:
        splits[r.get("split", "unknown")].append(r)
    if len(splits) > 1 or "unknown" not in splits:
        logger.info("-" * 100)
        # The comparator is NOT fixed at scramble_opposite any more. That stratum tracks the
        # partner drug rather than sitting at chance, so a gap measured against it is inflated by
        # however far the partner was pushed. The neutral stratum is the honest denominator.
        cmp_arm = args.split_comparator
        if not any(r.get(cmp_arm) is not None for r in recs):
            fb = "scramble_opposite"
            logger.warning(f"  {cmp_arm} is not populated in these records; falling back to {fb}. "
                           f"That arm is NOT neutral -- read the gap as an upper bound.")
            cmp_arm = fb
        alt_arm = "scramble_opposite" if cmp_arm != "scramble_opposite" else "scramble_orth"
        logger.info(f"  GENERALIZATION — gap vs {cmp_arm} by split "
                    "(train = memorisation; unseen_combo = CROSS-CONTEXT TRANSFER, the real test;")
        logger.info("  unseen_drug = no drug information at all, expected ~0 given the SAR gate):")
        gen_out = {}
        # A clustered bootstrap over a handful of points produces a wide CI that can exclude zero by
        # chance. Require enough conditions AND cell lines before reporting a verdict at all -- a split
        # with n=6 is UNDERPOWERED, which is a different statement from "no effect".
        MIN_N, MIN_CL = args.min_split_n, args.min_split_cell_lines
        for name in ("train", "unseen_combo", "unseen_drug", "unknown"):
            sub = splits.get(name)
            if not sub:
                continue
            n_cl_sub = len({r["cell_line"] for r in sub})
            if len(sub) < MIN_N or n_cl_sub < MIN_CL:
                logger.info(f"    {name:14s} n={len(sub):4d} ({n_cl_sub:2d} cell lines)  "
                            f"UNDERPOWERED (need n>={MIN_N} and >={MIN_CL} cell lines) -- NO VERDICT")
                gen_out[name] = {"n": len(sub), "n_cell_lines": n_cl_sub, "underpowered": True}
                continue
            r = _clustered_ci(sub, "model", cmp_arm, rng, args.n_boot)
            r_alt = _clustered_ci(sub, "model", alt_arm, rng, args.n_boot)
            mo = np.mean([x["model"] for x in sub if x.get("model") is not None])
            if r:
                m, lo, hi, n, ncl = r
                gen_out[name] = {"gap": m, "ci": [lo, hi], "n": n, "n_cell_lines": ncl,
                                 "model_nir": float(mo), "comparator": cmp_arm}
                if r_alt:
                    gen_out[name]["gap_vs_" + alt_arm] = {"gap": r_alt[0], "ci": [r_alt[1], r_alt[2]]}
                logger.info(f"    {name:14s} n={n:4d} ({ncl:2d} cell lines)  model NIR={mo:.3f}  "
                            f"gap = {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
                            f"{'USES DRUG' if lo > 0 else 'null'}")
                if r_alt:
                    logger.info(f"    {'':14s} {'':4s}  {'':16s} for reference, vs {alt_arm}: "
                                f"{r_alt[0]:+.4f}  CI [{r_alt[1]:+.4f}, {r_alt[2]:+.4f}]")
                # Baselines PER SPLIT. Pooling them mixes regimes: on unseen_drug the lookup is an
                # ORACLE -- it reads the held-out drug's true residuals, information the model is
                # definitionally denied -- so only the train / unseen_combo rows are fair contests.
                for b in ("drug_lookup", "drug_lookup_1", "moa_lookup"):
                    rb = _clustered_ci(sub, "model", b, rng, args.n_boot)
                    if rb:
                        mb, lob, hib, nb, _ = rb
                        gen_out[name][f"vs_{b}"] = {"gap": mb, "ci": [lob, hib], "n": nb}
                        # Only the same-DRUG lookups are oracles on unseen_drug: they read the held-out
                        # drug's own residuals, which the model is definitionally denied. moa_lookup uses
                        # OTHER drugs that WERE in training, so it is a fair contest in every split.
                        note = ("  (lookup is an ORACLE here -- reads the held-out drug's own residuals)"
                                if name == "unseen_drug" and b.startswith("drug_lookup") else "")
                        logger.info(f"        vs {b:14s} {mb:+.4f}  CI [{lob:+.4f}, {hib:+.4f}]  "
                                    f"{'WINS' if lob > 0 else ('LOSES' if hib < 0 else 'tie')}{note}")
        g = gen_out.get("unseen_combo")
        if g and g.get("underpowered"):
            logger.info(f"    >>> CROSS-CONTEXT TRANSFER: UNTESTED -- only {g['n']} scorable held-out "
                        f"combos ({g['n_cell_lines']} cell lines). This is NOT a negative result; the "
                        f"split is too small. Rebuild with more held-out (drug, cell_line) pairs.")
        elif g:
            logger.info(f"    >>> CROSS-CONTEXT TRANSFER: {'YES' if g['ci'][0] > 0 else 'NO'} "
                        f"({g['gap']:+.4f} CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], n={g['n']}). "
                        + ("The model applies a learned drug signature in a cell line it never saw it in."
                           if g['ci'][0] > 0 else
                           "Drug use does not transfer to new contexts -> memorisation only."))
        if all(k in gen_out and "gap" in gen_out[k] for k in ("train", "unseen_combo")):
            logger.info(f"    >>> memorisation premium: train {gen_out['train']['gap']:+.4f} vs "
                        f"unseen_combo {gen_out['unseen_combo']['gap']:+.4f}")
        means["generalization"] = gen_out
    else:
        tr = [r for r in recs if r.get("trained")]
        ho = [r for r in recs if not r.get("trained")]
        if tr and ho:
            f = lambda s: np.mean([r["model"] for r in s if r.get("model") is not None])
            logger.info(f"  [memorisation] trained n={len(tr)} model={f(tr):.3f} | "
                        f"held-out n={len(ho)} model={f(ho):.3f}")
    logger.info("=" * 100)
    means["strata"] = strat_out
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"config": vars(args), "means": means, "records": recs}, open(args.out, "w"),
              indent=2, default=float)
    logger.info(f"-> {args.out}")


def selftest(args):
    """A drug-AWARE synthetic generator must beat its scrambled twin; a drug-agnostic one must not."""
    P, D = 300, 8
    panel = [f"G{i}" for i in range(P)]
    gi = {g: i for i, g in enumerate(panel)}
    rng = np.random.RandomState(0)
    res = {d: rng.randn(P).astype(np.float32) for d in range(D)}
    truth = {d: signed_rank_from_vector(res[d], P, 40) for d in res}
    def sent(d):
        up = [panel[j] for j in np.argsort(-res[d])[:40]]
        dn = [panel[j] for j in np.argsort(res[d])[:40]]
        return " ".join(up) + f" {DOWN} " + " ".join(dn) + f" {END}"
    aware, scram, agn = [], [], []
    fixed = sent(0)
    for d in range(D):
        oth = [truth[o] for o in res if o != d]
        v = signed_rank_from_sentence(sent(d), gi, P, 40)
        aware.append(nir_from_sims(cos(v, truth[d]), [cos(v, t) for t in oth]))
        w = signed_rank_from_sentence(sent((d + 1) % D), gi, P, 40)
        scram.append(nir_from_sims(cos(w, truth[d]), [cos(w, t) for t in oth]))
        f = signed_rank_from_sentence(fixed, gi, P, 40)
        agn.append(nir_from_sims(cos(f, truth[d]), [cos(f, t) for t in oth]))
    a, s, g = np.mean(aware), np.mean(scram), np.mean(agn)
    logger.info(f"  drug-AWARE NIR={a:.3f} (expect ~1)  SCRAMBLED={s:.3f} (expect low)  "
                f"drug-AGNOSTIC={g:.3f} (expect ~chance)")
    ok = a > 0.95 and s < 0.5 and abs(g - 0.5) < 0.25
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

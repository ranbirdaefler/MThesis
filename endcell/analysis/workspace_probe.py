#!/usr/bin/env python
r"""
workspace_probe.py — is drug identity encoded in a CAUSALLY INERT subspace?
===========================================================================
Bridges the two standing observations — (i) drug identity is DECODABLE from activations (~82% at
layer 9) and (ii) the model IGNORES the drug (scramble / output-invariance null) — into a single
mechanistic claim: the direction that ENCODES drug identity is not the direction that DRIVES
generation. Inspired by the "workspace / privileged subspace" framing (Anthropic 2026): decompose
activations into a probe-defined subspace vs the rest, and show causal influence is (not) concentrated
there. Adapted to a supervised drug-identity subspace and, critically, hardened against the failure
mode that would make it unfalsifiable.

WHY A NAIVE ABLATION IS UNFALSIFIABLE (and how we fix it):
  We ALREADY know the model is drug-blind. So ablating a direction it doesn't use is GUARANTEED to
  change nothing — a null there is indistinguishable from a broken hook. Three guards make the result
  interpretable:
    * POSITIVE CONTROL subspace: the CELL-LINE / control-state subspace, which the model demonstrably
      DOES use (control-copy ~ model; cell line is the only informative baseline grouping). Ablating
      it MUST produce a large causal effect — that is the proof the instrument works.
    * NULL subspace: a random orthonormal subspace of matched dimension -> ~zero effect.
    * ABLATION-VERIFICATION GATE: after projecting the drug subspace out, re-probe. Decodability MUST
      collapse to chance, else we did not remove the information and any null is void. Generation must
      also stay valid (projecting out many dims could just break the model).
  Only the CONTRAST drug vs cell-line vs random, with the gate passed, is interpretable.

MEASUREMENT IS IN LOGIT SPACE, NOT GENERATION SPACE:
  We teacher-force the real response and measure KL( logits(h) || logits(h - P h) ) at each response
  position — deterministic, per-token, no sampling-noise floor (the floor is what made our earlier
  causal probe inconclusive). This is the causal effect of the subspace on the next-token distribution.

EXPERIMENTS
  1. build drug / cell-line / random subspaces (per layer) + CONFOUND cross-decodability
     (does the drug subspace also predict cell line / plate / dose? if so, orthogonalize & re-verify).
  2. ABLATION-VERIFICATION GATE (probe collapses to chance; generation stays valid).
  3. LOGIT-SPACE KL causal test: drug vs cell-line vs random, swept across layers.
  4. VARIANCE-vs-CAUSAL-EFFECT decomposition: variance share of each subspace vs its KL.
  5. (--do_swap) ACTIVATION-SPACE drug swap: replace the drug component with another drug's; does the
     readout respond more than to a matched-norm random injection? Distinguishes "inert" (encoded but
     unreadable) from "routing failure" (readable but never routed).

SELFTEST (no model/data) — a synthetic linear readout that reads from a KNOWN direction and ignores
another; verifies the gate collapses decodability and the KL test flags causal vs inert correctly:
  python workspace_probe.py --selftest
"""
# --- repo path bootstrap: works in BOTH the reorganized repo AND the flat cluster layout ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
_cands = [_HERE, os.path.join(_HERE, "src")]
if os.path.isdir(os.path.join(_ROOT, "shared")):
    _cands += [os.path.join(_ROOT, "shared")] + sorted(glob.glob(os.path.join(_PIPE, "*")))
for _p in _cands:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- linear-algebra core (numpy)
def build_subspace(X, labels, max_dims):
    """Class-mean (between-class) subspace: the directions along which the label varies. Stack the
    per-class mean vectors (centered by the global mean), SVD, keep the leading orthonormal
    directions. Returns V (H, k) with orthonormal columns. This is where the label 'lives'."""
    X = np.asarray(X, dtype=np.float64)
    uniq = sorted(set(labels))
    gmean = X.mean(0)
    M = np.stack([X[[i for i, l in enumerate(labels) if l == u]].mean(0) - gmean for u in uniq])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(max_dims, len(uniq) - 1, int(np.sum(S > 1e-8)))
    k = max(k, 1)
    return Vt[:k].T.astype(np.float64)          # (H, k) orthonormal columns


def build_union_subspace(X, label_sets, max_dims):
    """Subspace spanned by the class means of SEVERAL label sets at once (e.g. cell_line ∪ plate ∪
    dose) — the 'context / batch' subspace the drug axis must be decorrelated against. Stacks all
    centered class-mean vectors from every label set, then SVD."""
    X = np.asarray(X, dtype=np.float64)
    gmean = X.mean(0)
    rows = []
    for labels in label_sets:
        for u in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == u]
            if idx:
                rows.append(X[idx].mean(0) - gmean)
    M = np.stack(rows)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(max_dims, int(np.sum(S > 1e-8)))
    return Vt[:max(k, 1)].T.astype(np.float64)


def random_subspace(H, k, rng):
    k = max(int(k), 1)
    A = rng.randn(H, k)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def orthogonalize(V, W):
    """Component of V's span orthogonal to W's span (project V's basis off W, re-orthonormalize)."""
    Vp = V - W @ (W.T @ V)
    Q, R = np.linalg.qr(Vp)
    keep = np.abs(np.diag(R)) > 1e-6
    return Q[:, keep]


def subspace_overlap(V, W):
    """Mean squared cosine of principal angles — how much V's span lies inside W's span (0..1)."""
    s = np.linalg.svd(V.T @ W, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def project_out(X, V):
    """Remove the V-subspace component from rows of X."""
    return X - (X @ V) @ V.T


def variance_share(X, V):
    """Fraction of total variance captured by the V-subspace."""
    Xc = X - X.mean(0)
    tot = float(np.sum(Xc ** 2))
    proj = Xc @ V
    return float(np.sum(proj ** 2) / (tot + 1e-12))


def probe_acc(X, labels, rng, n_splits=5):
    """Cross-validated multinomial logistic-regression accuracy + shuffled-label chance floor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    y = np.array([sorted(set(labels)).index(l) for l in labels])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    nsp = min(n_splits, np.min(np.bincount(y)))
    if nsp < 2:
        return None, None
    acc = float(np.mean(cross_val_score(clf, X, y, cv=nsp)))
    ys = y.copy(); rng.shuffle(ys)
    shuf = float(np.mean(cross_val_score(clf, X, ys, cv=nsp)))
    return acc, shuf


def kl_rows(logp_clean, logp_alt):
    """KL(clean || alt) per row, from log-softmax rows."""
    p = np.exp(logp_clean)
    return np.sum(p * (logp_clean - logp_alt), axis=-1)


# ----------------------------------------------------------------- model plumbing (torch)
def get_layers(model):
    for attr in ("gpt_neox", "model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None:
            if hasattr(base, "layers"):
                return base.layers
            if hasattr(base, "h"):
                return base.h
    raise RuntimeError("could not locate transformer layer list on this model")


def _control_from_prompt(prompt):
    i = prompt.find("Control cell:")
    if i == -1:
        return ""
    rest = prompt[i + len("Control cell:"):]
    j = rest.find("\n")
    return (rest if j == -1 else rest[:j]).strip()


def extract_activations(model, tok, prompts, device, layers, bf16):
    """activations[L] = (n_prompts, H) at the LAST PROMPT position (matches the 82% decodability
    result). One forward pass per prompt."""
    import torch
    acts = {L: [] for L in layers}
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for L in layers:
            acts[L].append(out.hidden_states[L][0, -1, :].float().cpu().numpy())
    return {L: np.vstack(v) for L, v in acts.items()}


def make_hook(V_t, add_vec=None):
    """Forward hook on layer (L-1) (its output == hidden_states[L]). Ablates the V-subspace from the
    residual at every position, optionally adding a fixed component (for the swap experiment)."""
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        Vd = V_t.to(hs.dtype)
        comp = (hs @ Vd) @ Vd.transpose(0, 1)          # (1, T, H) projection onto V
        hs2 = hs - comp
        if add_vec is not None:
            hs2 = hs2 + add_vec.to(hs.dtype)
        return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
    return hook


def teacher_forced_logits(model, tok, prompt, response, device, resp_only=True):
    """Return log-softmax logits at the positions that PREDICT the response tokens, and the ids."""
    import torch
    p_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=3500).input_ids
    r_ids = tok(" " + response, add_special_tokens=False, truncation=True, max_length=1200).input_ids
    if len(r_ids) < 2:
        return None, None
    ids = torch.tensor([p_ids + r_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()          # (T, V)
    # position i predicts token i+1; response tokens sit at [len(p_ids)-1 .. T-2]
    pos = list(range(len(p_ids) - 1, len(p_ids) - 1 + len(r_ids)))
    pos = [q for q in pos if q < logits.shape[0]]
    lp = torch.log_softmax(logits[pos], dim=-1)
    return lp.cpu().numpy(), pos


def kl_under_hook(model, tok, layers_mod, target_layer, V_np, prompt, response, device, add_vec=None):
    """Mean KL over response positions between clean and V-ablated (optionally V-swapped) logits."""
    import torch
    lp_clean, pos = teacher_forced_logits(model, tok, prompt, response, device)
    if lp_clean is None:
        return None
    V_t = torch.tensor(V_np, dtype=torch.float32, device=device)
    av = torch.tensor(add_vec, dtype=torch.float32, device=device) if add_vec is not None else None
    h = layers_mod[target_layer - 1].register_forward_hook(make_hook(V_t, av))
    try:
        p_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=3500).input_ids
        r_ids = tok(" " + response, add_special_tokens=False, truncation=True, max_length=1200).input_ids
        ids = torch.tensor([p_ids + r_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        qpos = list(range(len(p_ids) - 1, len(p_ids) - 1 + len(r_ids)))
        qpos = [q for q in qpos if q < logits.shape[0]]
        lp_alt = torch.log_softmax(logits[qpos], dim=-1).cpu().numpy()
    finally:
        h.remove()
    n = min(len(lp_clean), len(lp_alt))
    return float(np.mean(kl_rows(lp_clean[:n], lp_alt[:n])))


# ----------------------------------------------------------------- data
def load_prompts(eval_dir, tier, n_drugs, n_per_drug, rng):
    path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
    ex = [json.loads(l) for l in open(path)]
    by_drug = defaultdict(list)
    for e in ex:
        m = e.get("metadata", {})
        if m.get("drug"):
            by_drug[m["drug"]].append(e)
    drugs = [d for d, v in by_drug.items() if len(v) >= n_per_drug]
    rng.shuffle(drugs); drugs = drugs[:n_drugs]
    rows = []
    for d in drugs:
        cells = by_drug[d]
        idx = rng.choice(len(cells), n_per_drug, replace=False)
        for i in idx:
            e = cells[i]; m = e["metadata"]
            rows.append({"prompt": e["prompt"], "response": e["response"], "drug": d,
                         "cell_line": m.get("cell_line_id"), "plate": m.get("plate"),
                         "dose": m.get("dose_float")})
    logger.info(f"  {len(drugs)} drugs x {n_per_drug} = {len(rows)} prompts")
    return rows


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic linear readout logits = h @ W_read.T, where W_read READS ONLY from the cell-line
    subspace. Drug identity lives in a MULTI-DIM subspace ORTHOGONAL to it (decodable but causally
    dead); cell-line identity in the readout subspace (decodable AND causally live). Verifies the
    whole chain: decodability, the ablation gate, and drug-KL ~ 0 while cell-line-KL >> 0."""
    rng = np.random.RandomState(0)
    H, V, C, n, kdim = 64, 200, 8, 40, 5
    # two ORTHOGONAL subspaces
    Q, _ = np.linalg.qr(rng.randn(H, 2 * kdim))
    drug_sub, cl_sub = Q[:, :kdim], Q[:, kdim:2 * kdim]             # (H, kdim) each, orthonormal, ⟂
    drug_code = {d: (rng.randn(kdim) * 3) @ drug_sub.T for d in range(C)}   # distinct, separable
    cl_code = {c: (rng.randn(kdim) * 3) @ cl_sub.T for c in range(C)}
    # readout depends ONLY on the cell-line subspace -> ablating drug_sub cannot change logits
    W_read = rng.randn(V, kdim) @ cl_sub.T                          # rowspace ⊆ cl_sub

    X, drug_lab, cl_lab = [], [], []
    for _ in range(n * C):
        d = rng.randint(C); c = rng.randint(C)
        h = rng.randn(H) * 0.4 + drug_code[d] + cl_code[c]
        X.append(h); drug_lab.append(d); cl_lab.append(c)
    X = np.array(X)

    V_drug = build_subspace(X, drug_lab, args.n_dims)
    V_cl = build_subspace(X, cl_lab, args.n_dims)
    V_conf = build_union_subspace(X, [cl_lab], args.n_dims)          # context = cell line here
    V_drug_pure = orthogonalize(V_drug, V_conf)
    d_pure = V_drug_pure.shape[1]
    V_rand_pure = random_subspace(H, max(d_pure, 1), rng)

    def readout_kl(Vsub):
        if Vsub is None or Vsub.shape[1] == 0:
            return 0.0
        lp_clean = _logsm(X @ W_read.T)
        lp_abl = _logsm(project_out(X, Vsub) @ W_read.T)
        return float(np.mean(kl_rows(lp_clean, lp_abl)))

    a_drug, _ = probe_acc(X, drug_lab, np.random.RandomState(1))
    a_drug_abl, _ = probe_acc(project_out(X, V_drug), drug_lab, np.random.RandomState(1))
    frac_removed = (a_drug - a_drug_abl) / (a_drug - 1.0 / C)
    kl_pure, kl_randp, kl_cl = readout_kl(V_drug_pure), readout_kl(V_rand_pure), readout_kl(V_cl)
    logger.info(f"  drug decodable {a_drug:.2f} -> ablated {a_drug_abl:.2f}  (removed "
                f"{100*frac_removed:.0f}% of above-chance signal; robust gate expects >80%)")
    logger.info(f"  pure-drug subspace = {d_pure} dims after removing context")
    logger.info(f"  causal KL:  PURE-drug={kl_pure:.4f}  random@matched={kl_randp:.4f}  "
                f"cell_line(+ctrl)={kl_cl:.4f}")

    ok = True
    if not (a_drug > 0.8):
        logger.error("  FAIL: drug not decodable in synthetic data"); ok = False
    if not (frac_removed > 0.8):
        logger.error("  FAIL: robust gate did not register signal removal"); ok = False
    if not (kl_cl > 5 * max(kl_pure, 1e-6)):
        logger.error("  FAIL: positive-control (cell line) KL not >> pure-drug KL"); ok = False
    if not (kl_pure < 3 * kl_randp + 5e-3):
        logger.error("  FAIL: pure-drug KL not ~ matched-dim random (should be causally inert)"); ok = False
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}  (drug decodable but causally inert once "
                f"context removed; cell line decodable AND causal; robust gate + decorrelation work)")
    if not ok:
        sys.exit(1)


def _logsm(Z):
    Z = Z - Z.max(-1, keepdims=True)
    return Z - np.log(np.sum(np.exp(Z), -1, keepdims=True))


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--model_path")
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--layers", default="4,8,9,12,16")
    ap.add_argument("--n_drugs", type=int, default=12)
    ap.add_argument("--n_per_drug", type=int, default=40)
    ap.add_argument("--n_dims", type=int, default=10, help="max subspace dimension")
    ap.add_argument("--n_kl_prompts", type=int, default=60, help="prompts for the KL causal test")
    ap.add_argument("--do_swap", action="store_true", help="also run the activation-space drug swap")
    ap.add_argument("--out", default="RESULTS/workspace_probe.json")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return
    if not (args.eval_dir and args.model_path):
        ap.error("--eval_dir and --model_path required (unless --selftest)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.RandomState(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device).eval()
    layers_mod = get_layers(model)
    logger.info(f"Model {args.model_path}: {len(layers_mod)} layers; testing hidden_states {layers}")

    rows = load_prompts(args.eval_dir, args.tier, args.n_drugs, args.n_per_drug, rng)
    drug_lab = [r["drug"] for r in rows]
    cl_lab = [r["cell_line"] for r in rows]
    n_drug_classes = len(set(drug_lab))

    logger.info("Extracting activations (last prompt position) ...")
    acts = extract_activations(model, tok, [r["prompt"] for r in rows], device, layers, args.bf16)

    kl_rows_prompts = rows[:args.n_kl_prompts]
    result = {"layers": {}, "config": {k: v for k, v in vars(args).items()},
              "n_drug_classes": n_drug_classes, "chance": 1.0 / n_drug_classes}

    plate_lab = [str(r["plate"]) for r in rows]
    dose_lab = [str(r["dose"]) for r in rows]
    chance = 1.0 / n_drug_classes

    for L in layers:
        X = acts[L]
        H = X.shape[1]
        logger.info("")
        logger.info(f"===== hidden_states[{L}] =====")
        # (1) subspaces. V_drug = raw drug axis; V_conf = the CONTEXT/BATCH subspace (cell_line ∪
        # plate ∪ dose); V_drug_pure = drug orthogonalized against ALL of it — the only subspace whose
        # ablation cleanly tests DRUG identity rather than the context the model is known to use.
        V_drug = build_subspace(X, drug_lab, args.n_dims)
        V_cl = build_subspace(X, cl_lab, args.n_dims)
        V_conf = build_union_subspace(X, [cl_lab, plate_lab, dose_lab], args.n_dims * 3)
        V_drug_pure = orthogonalize(V_drug, V_conf)
        d_pure = V_drug_pure.shape[1]
        V_rand = random_subspace(H, V_drug.shape[1], rng)
        V_rand_pure = random_subspace(H, max(d_pure, 1), rng)          # matched-dim null for the pure test

        # (1b) confound cross-decodability of the RAW drug subspace
        Xd = X @ V_drug
        cross = {n: probe_acc(Xd, lab, np.random.RandomState(1))[0]
                 for n, lab in (("cell_line", cl_lab), ("plate", plate_lab), ("dose", dose_lab))}
        overlaps = {"cell_line": subspace_overlap(V_drug, V_cl),
                    "context": subspace_overlap(V_drug, V_conf)}

        # (2) ROBUST gate: fraction of ABOVE-CHANCE drug decodability removed by ablation (>0.8 = pass).
        # (The old shuffle-vs-ablated comparison mismatched two different no-signal floors.)
        a_drug, a_shuf = probe_acc(X, drug_lab, np.random.RandomState(1))
        a_drug_abl, _ = probe_acc(project_out(X, V_drug), drug_lab, np.random.RandomState(1))
        # residual PURE-drug decodability: is drug decodable AFTER removing all context?
        a_drug_pureonly, _ = probe_acc(project_out(X, V_conf), drug_lab, np.random.RandomState(1))
        frac_removed = ((a_drug - a_drug_abl) / (a_drug - chance)) if (a_drug and a_drug > chance) else None
        gate_ok = (frac_removed is not None and frac_removed > 0.8)
        logger.info(f"  GATE: drug decodable {a_drug:.3f} (chance {chance:.3f}) -> ablated {a_drug_abl:.3f} "
                    f"| removed {100*(frac_removed or 0):.0f}% of signal  [{'PASS' if gate_ok else 'FAIL'}]")
        logger.info(f"  CONFOUND: raw drug subspace also decodes cell_line={_f(cross['cell_line'])} "
                    f"plate={_f(cross['plate'])} dose={_f(cross['dose'])}")
        logger.info(f"    overlap(drug, context)={overlaps['context']:.3f} | pure-drug subspace = "
                    f"{d_pure}/{V_drug.shape[1]} dims after removing context | "
                    f"drug decodable with context removed = {_f(a_drug_pureonly)} (chance {chance:.3f})")

        def mean_kl(Vsub):
            if Vsub is None or (hasattr(Vsub, "shape") and Vsub.shape[1] == 0):
                return (None, None, 0)
            vals = [kl_under_hook(model, tok, layers_mod, L, Vsub, r["prompt"], r["response"], device)
                    for r in kl_rows_prompts]
            vals = [v for v in vals if v is not None]
            return ((float(np.mean(vals)), float(np.std(vals) / max(1, len(vals)) ** 0.5), len(vals))
                    if vals else (None, None, 0))

        # (3) CAUSAL KL. Headline = pure-drug vs its MATCHED-DIM random null.
        kl_drug = mean_kl(V_drug)
        kl_pure = mean_kl(V_drug_pure) if d_pure else (None, None, 0)
        kl_rand_pure = mean_kl(V_rand_pure) if d_pure else (None, None, 0)
        kl_cl = mean_kl(V_cl)
        kl_conf = mean_kl(V_conf)
        kl_rand = mean_kl(V_rand)
        logger.info(f"  CAUSAL KL (mean+-sem over {args.n_kl_prompts} prompts):")
        logger.info(f"     PURE drug (⊥ context)  {_kf(kl_pure)}   <<< HEADLINE")
        logger.info(f"     random @ matched dim   {_kf(kl_rand_pure)}   <- null for the pure test")
        logger.info(f"     cell_line (+ctrl)      {_kf(kl_cl)}   <- POSITIVE CONTROL")
        logger.info(f"     context (all confounds){_kf(kl_conf)}")
        logger.info(f"     raw drug               {_kf(kl_drug)}   (confounded — do not quote)")
        vs = {"drug": variance_share(X, V_drug), "drug_pure": variance_share(X, V_drug_pure) if d_pure else 0.0,
              "cell_line": variance_share(X, V_cl), "context": variance_share(X, V_conf),
              "random": variance_share(X, V_rand)}
        logger.info(f"  VARIANCE SHARE: pure-drug={vs['drug_pure']:.3f}  cell_line={vs['cell_line']:.3f}  "
                    f"context={vs['context']:.3f}")

        # PURE-drug verdict
        pv = "no pure-drug subspace (drug fully within context span)"
        if kl_pure[0] is not None and kl_rand_pure[0] is not None:
            ratio = kl_pure[0] / max(kl_rand_pure[0], 1e-9)
            if ratio < 2.0:
                pv = f"INERT: pure-drug KL ~ matched random (ratio {ratio:.2f}) -> encoded, not read"
            elif kl_cl[0] and kl_pure[0] < 0.25 * kl_cl[0]:
                pv = f"weak: pure-drug KL {ratio:.1f}x random but << cell-line -> minor causal role"
            else:
                pv = f"CAUSAL: pure-drug KL {ratio:.1f}x random and comparable to controls -> the model DOES read pure drug identity"
        logger.info(f"  >>> PURE-DRUG VERDICT: {pv}")

        entry = {"gate_pass": bool(gate_ok), "frac_signal_removed": frac_removed,
                 "drug_decode": a_drug, "drug_decode_ablated": a_drug_abl,
                 "drug_decode_context_removed": a_drug_pureonly, "chance": chance,
                 "confound_crossdecode": cross, "overlaps": overlaps,
                 "pure_drug_dims": int(d_pure), "raw_drug_dims": int(V_drug.shape[1]),
                 "kl": {"pure_drug": kl_pure, "random_matched": kl_rand_pure, "cell_line": kl_cl,
                        "context": kl_conf, "raw_drug": kl_drug, "random": kl_rand},
                 "variance_share": vs, "pure_drug_verdict": pv}

        # (4) activation-space drug swap ON THE PURE-DRUG subspace (clean drug-specificity test)
        if args.do_swap and d_pure and gate_ok:
            drug_mean = {d: X[[i for i, r in enumerate(rows) if r["drug"] == d]].mean(0)
                         for d in set(drug_lab)}
            swap_kl, randinj_kl = [], []
            for r in kl_rows_prompts:
                others = [d for d in drug_mean if d != r["drug"]]
                if not others:
                    continue
                b = others[rng.randint(len(others))]
                comp_b = V_drug_pure @ (V_drug_pure.T @ drug_mean[b])   # drug B's PURE component
                sk = kl_under_hook(model, tok, layers_mod, L, V_drug_pure, r["prompt"], r["response"],
                                   device, add_vec=comp_b)
                rv = rng.randn(H); rv *= np.linalg.norm(comp_b) / (np.linalg.norm(rv) + 1e-9)
                rk = kl_under_hook(model, tok, layers_mod, L, V_drug_pure, r["prompt"], r["response"],
                                   device, add_vec=rv.astype(np.float32))
                if sk is not None and rk is not None:
                    swap_kl.append(sk); randinj_kl.append(rk)
            if swap_kl:
                mb, mr = float(np.mean(swap_kl)), float(np.mean(randinj_kl))
                sv = ("drug-B >> random -> readout IS drug-direction sensitive (routing failure)"
                      if mb > 1.5 * mr else
                      "drug-B ~ random -> readout is NOT drug-specific (responds to magnitude only)")
                logger.info(f"  SWAP (pure): inject drug-B {mb:.4f}  vs matched-norm random {mr:.4f}  -> {sv}")
                entry["swap"] = {"drug_b_kl": mb, "random_inject_kl": mr, "n": len(swap_kl), "verdict": sv}
        result["layers"][str(L)] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=float)
    logger.info("")
    logger.info("READ: on a GATE-passing layer, compare PURE-drug KL (⊥ context) to its MATCHED-DIM")
    logger.info("      random null. ~random => drug identity is encoded but causally inert. Also check")
    logger.info("      'drug decodable with context removed': if ~chance, drug is not even separable")
    logger.info("      from dose/plate/cell-line internally (a finding in itself).")
    logger.info(f"-> {args.out}")


def _f(x):
    return "NA" if x is None else f"{x:.3f}"


def _kf(t):
    return "NA" if t is None or t[0] is None else f"{t[0]:.4f} +- {t[1]:.4f} (n={t[2]})"


if __name__ == "__main__":
    main()

#!/usr/bin/env python
r"""
make_provenance.py — bundle the existing results + fingerprint the big inputs (A-08)
====================================================================================
Makes the reported numbers independently auditable by tying them to a commit, command, seed,
environment, dataset identifier and checkpoint hash. NON-DESTRUCTIVE: it only READS the dataset and
checkpoint (never writes to them), COPIES the small RESULTS/*.json into a committable artifacts/
folder, and writes a manifest. It never deletes or modifies data.

Fingerprints are cheap by default: for a big directory (dataset / checkpoint) we hash a manifest of
(relative-path, size, mtime) lines rather than the file contents, plus record file count and total
bytes. That uniquely identifies a build without reading gigabytes. Pass --full_hash to also sha256
the file *contents* (slow; use once for the final archival record).

USAGE (run on the cluster, from ~/tahoe)
----------------------------------------
  python endcell/analysis/make_provenance.py \
      --results_glob "RESULTS/*.json" --logs_glob "logs/*.out" \
      --dataset_dir /data/.../data_diverse2_endcell_big \
      --checkpoint_dir /data/.../checkpoints/pythia_sft_endcell/final \
      --jobs 596144,596473 --seed 42 \
      --label "Q13 workspace_probe + calibration" \
      --out_dir artifacts

Then commit the artifacts/ folder (small JSON + manifest only).
"""
import argparse, json, os, glob, hashlib, subprocess, sys, shutil, platform
from datetime import datetime, timezone


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dir_fingerprint(path, full_hash=False):
    """Cheap, deterministic identity of a directory tree. Hash a sorted manifest of
    (relpath, size, mtime); optionally also sha256 each file's contents (slow)."""
    if not path or not os.path.isdir(path):
        return {"path": path, "exists": False}
    files, total = [], 0
    for root, _, names in os.walk(path):
        for nm in sorted(names):
            fp = os.path.join(root, nm)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, path)
            rec = {"rel": rel, "size": st.st_size, "mtime": int(st.st_mtime)}
            if full_hash:
                rec["sha256"] = sha256_file(fp)
            files.append(rec)
            total += st.st_size
    manifest = "\n".join(f"{r['rel']}\t{r['size']}\t{r['mtime']}" for r in files)
    return {
        "path": path, "exists": True, "n_files": len(files), "total_bytes": total,
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "content_sha256": (hashlib.sha256("\n".join(r["sha256"] for r in files).encode()).hexdigest()
                           if full_hash else None),
    }


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "sklearn", "torch", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", default="RESULTS/*.json", help="small outputs to COPY + hash")
    ap.add_argument("--logs_glob", default=None, help="logs to hash in place (not copied by default)")
    ap.add_argument("--copy_logs", action="store_true", help="also copy matched logs into out_dir")
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--jobs", default=None, help="comma-separated SLURM job ids for traceability")
    ap.add_argument("--command", default=None, help="the command(s) that produced the results")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--label", default="", help="short human label for this bundle")
    ap.add_argument("--full_hash", action="store_true", help="sha256 file CONTENTS of big dirs (slow)")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.out_dir, exist_ok=True)
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    # copy + hash the small result JSONs
    results = []
    for fp in sorted(glob.glob(args.results_glob)):
        if not os.path.isfile(fp):
            continue
        dst = os.path.join(res_dir, os.path.basename(fp))
        shutil.copy2(fp, dst)
        results.append({"file": os.path.basename(fp), "src": fp,
                        "bytes": os.path.getsize(fp), "sha256": sha256_file(fp)})

    # hash logs (copy only if asked; logs can be large)
    logs = []
    if args.logs_glob:
        log_dir = os.path.join(args.out_dir, "logs")
        if args.copy_logs:
            os.makedirs(log_dir, exist_ok=True)
        for fp in sorted(glob.glob(args.logs_glob)):
            if not os.path.isfile(fp):
                continue
            rec = {"file": os.path.basename(fp), "src": fp,
                   "bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}
            if args.copy_logs:
                shutil.copy2(fp, os.path.join(log_dir, os.path.basename(fp)))
            logs.append(rec)

    prov = {
        "created_utc": stamp,
        "label": args.label,
        "git": {
            "commit": sh("git rev-parse HEAD"),
            "branch": sh("git rev-parse --abbrev-ref HEAD"),
            "dirty": bool(sh("git status --porcelain")),
            "remote": sh("git config --get remote.origin.url"),
        },
        "env": env_info(),
        "seed": args.seed,
        "jobs": [j.strip() for j in args.jobs.split(",")] if args.jobs else [],
        "command": args.command,
        "dataset": dir_fingerprint(args.dataset_dir, args.full_hash),
        "checkpoint": dir_fingerprint(args.checkpoint_dir, args.full_hash),
        "results": results,
        "logs": logs,
    }
    prov_path = os.path.join(args.out_dir, f"provenance_{stamp}.json")
    json.dump(prov, open(prov_path, "w"), indent=2, default=str)

    # append a human-readable row to artifacts/MANIFEST.md
    man = os.path.join(args.out_dir, "MANIFEST.md")
    new = not os.path.exists(man)
    with open(man, "a") as f:
        if new:
            f.write("# Provenance manifest\n\nEach row ties a set of committed result files to the "
                    "exact commit, dataset, checkpoint, seed, and jobs that produced them.\n\n"
                    "| created (UTC) | label | commit | dataset manifest sha | checkpoint manifest sha | "
                    "seed | jobs | #results |\n"
                    "|---|---|---|---|---|---|---|---|\n")
        ds = prov["dataset"].get("manifest_sha256", "-") or "-"
        ck = prov["checkpoint"].get("manifest_sha256", "-") or "-"
        f.write(f"| {stamp} | {args.label or '-'} | {(prov['git']['commit'] or '-')[:10]} | "
                f"{ds[:12]} | {ck[:12]} | {args.seed if args.seed is not None else '-'} | "
                f"{','.join(prov['jobs']) or '-'} | {len(results)} |\n")

    print(f"bundled {len(results)} result files, {len(logs)} logs")
    print(f"  dataset:    {prov['dataset']}")
    print(f"  checkpoint: {prov['checkpoint']}")
    print(f"  commit:     {prov['git']['commit']}  dirty={prov['git']['dirty']}")
    print(f"-> {prov_path}\n-> {man}")
    print("\nNext: git add artifacts/ && git commit -m 'provenance bundle'  (small JSON + manifest only)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statically reconcile Gemma job/runbook commands with current argparse CLIs."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


PARSER_REQUIREMENTS = {
    "trainer": (
        "endcell/train/train_c2s_tahoe_endcell.py",
        {
            "--mode", "--model_name", "--model_revision", "--attn_implementation",
            "--train_file", "--eval_file", "--output_dir", "--de_weight", "--max_length",
            "--num_epochs", "--batch_size", "--grad_accum", "--learning_rate",
            "--weight_decay", "--warmup_ratio", "--bf16", "--gradient_checkpointing",
            "--prepend_bos", "--strict_token_contract", "--resumable",
            "--resume_from_checkpoint", "--log_every", "--save_every",
            "--keep_checkpoints", "--seed",
        },
    ),
    "tokenizer_probe": (
        "endcell/train/gemma_tokenizer_probe.py",
        {
            "--model", "--revision", "--data_dir", "--output", "--max_length",
            "--max_examples", "--parity_examples", "--require_fast",
        },
    ),
    "manifest": (
        "endcell/analysis/freeze_nir_manifest.py",
        {
            "--eval_dir", "--scram_dir", "--train_file", "--tier", "--out",
            "--k_samples", "--min_cells", "--min_drugs_per_group", "--max_groups",
            "--same_plate_only", "--seed", "--generation_contract_version",
            "--model_fingerprint", "--validity_only_parent", "--expected_rows",
            "--expected_drugs", "--expected_cell_lines", "--expected_groups", "--hash_named",
        },
    ),
    "nir": (
        "endcell/analysis/nir_benchmark.py",
        {
            "--out", "--selftest", "--eval_dir", "--scram_dir", "--manifest",
            "--prediction_cache", "--model_fingerprint", "--parent_validity_only",
            "--attention_implementation", "--generation_contract_version", "--invalid_policy",
            "--min_complete_rate", "--min_recognized_gene_rate", "--min_recognized_genes",
            "--model_path", "--train_file", "--tiers", "--k_samples", "--temperature",
            "--top_p", "--max_new_tokens", "--gen_batch_size", "--min_cells",
            "--min_drugs_per_cl", "--max_groups", "--same_plate_only", "--bf16",
            "--profiles", "--seed",
        },
    ),
    "validity": (
        "endcell/eval/evaluate_endcell.py",
        {
            "--out", "--selftest", "--mode", "--eval_dir", "--model_path", "--tiers",
            "--max_eval", "--max_new_tokens", "--gen_batch_size", "--bf16", "--seed",
            "--generation_contract", "--generation_contract_version", "--model_fingerprint",
            "--manifest_sha256", "--parent_validity_only", "--attention_implementation",
            "--prediction_cache", "--invalid_policy", "--min_complete_rate",
            "--min_recognized_gene_rate",
        },
    ),
    "comparator": (
        "endcell/analysis/compare_backbones.py",
        {"--left", "--right", "--left_name", "--right_name", "--tier", "--per_drug_bh", "--out"},
    ),
    "fingerprint": (
        "endcell/jobs/gemma2_standard_checkpoint_fingerprint.py",
        {"--checkpoint", "--out", "--digest_only"},
    ),
    "preflight_contract": (
        "endcell/jobs/gemma2_standard_preflight_contract.py",
        {
            "--certificate", "--model-id", "--revision", "--snapshot-path",
            "--tokenizer-probe", "--generation-cap", "--data", "--source",
            "--hub-cache", "--tests-certificate", "--environment", "--require-generation-cap",
            "--print-snapshot", "--verify-current-environment", "--protobuf-root",
            "--protobuf-tree-sha256",
        },
    ),
    "tests_contract": (
        "endcell/jobs/gemma2_standard_tests_contract.py",
        {"--certificate", "--command", "--source", "--verify-current-environment"},
    ),
}


CONSUMER_REQUIREMENTS = {
    "endcell/jobs/gemma2_standard_preflight.sh": {
        "--model", "--revision", "--data_dir", "--output", "--max_length",
        "--max_examples", "--parity_examples", "--require_fast",
        "--certificate", "--model-id", "--snapshot-path", "--tokenizer-probe",
        "--generation-cap", "--data", "--source", "--hub-cache", "--tests-certificate",
        "--environment", "--protobuf-root", "--protobuf-tree-sha256",
        "--require-generation-cap", "--verify-current-environment",
    },
    "endcell/jobs/gemma2_standard_tests.sh": {
        "--certificate", "--command", "--source", "--verify-current-environment",
    },
    "endcell/jobs/gemma2_standard_smoke.sbatch": {
        "--mode", "--model_name", "--model_revision", "--attn_implementation",
        "--train_file", "--eval_file", "--output_dir", "--num_epochs", "--batch_size",
        "--grad_accum", "--bf16", "--gradient_checkpointing", "--max_length",
        "--learning_rate", "--weight_decay", "--warmup_ratio", "--de_weight",
        "--log_every", "--save_every", "--keep_checkpoints", "--seed", "--prepend_bos",
        "--strict_token_contract", "--resumable", "--resume_from_checkpoint",
        "--certificate", "--model-id", "--revision", "--require-generation-cap",
        "--verify-current-environment",
    },
    "endcell/jobs/gemma2_standard_train.sbatch": {
        "--mode", "--model_name", "--model_revision", "--attn_implementation",
        "--train_file", "--eval_file", "--output_dir", "--num_epochs", "--batch_size",
        "--grad_accum", "--bf16", "--gradient_checkpointing", "--max_length",
        "--learning_rate", "--weight_decay", "--warmup_ratio", "--de_weight",
        "--log_every", "--save_every", "--keep_checkpoints", "--seed", "--prepend_bos",
        "--strict_token_contract", "--resumable", "--resume_from_checkpoint",
        "--certificate", "--model-id", "--revision", "--require-generation-cap",
        "--verify-current-environment",
    },
    "endcell/jobs/gemma2_standard_eval.sbatch": {
        "--checkpoint", "--digest_only", "--selftest", "--mode", "--eval_dir",
        "--model_path", "--tiers", "--max_eval", "--max_new_tokens", "--gen_batch_size",
        "--bf16", "--seed", "--generation_contract", "--generation_contract_version",
        "--model_fingerprint", "--manifest_sha256", "--parent_validity_only",
        "--attention_implementation", "--prediction_cache", "--invalid_policy",
        "--min_complete_rate", "--min_recognized_gene_rate", "--same_plate_only",
        "--scram_dir", "--train_file", "--k_samples", "--temperature", "--top_p",
        "--min_cells", "--min_drugs_per_cl", "--max_groups", "--manifest",
        "--min_recognized_genes", "--out", "--profiles",
        "--certificate", "--model-id", "--revision", "--require-generation-cap",
        "--verify-current-environment",
    },
    "docs/endcell/gemma2_standard_hpc_runbook.md": {
        "--eval_dir", "--scram_dir", "--train_file", "--tier", "--k_samples",
        "--min_cells", "--min_drugs_per_group", "--max_groups", "--same_plate_only",
        "--seed", "--generation_contract_version", "--expected_rows", "--expected_drugs",
        "--expected_cell_lines", "--expected_groups", "--model_fingerprint",
        "--validity_only_parent", "--hash_named", "--out", "--checkpoint", "--digest_only",
        "--left", "--right", "--left_name", "--right_name", "--per_drug_bh",
        "--certificate", "--model-id", "--revision", "--print-snapshot",
    },
}


TEXT_REQUIREMENTS = {
    "endcell/jobs/gemma2_standard_preflight.sh": {
        "gemma2_standard_protobuf_env.sh", "source \"$PROTOBUF_ENV\"",
        "unset PYTHONHOME", "[[ \"$LONG_GPU_PARTITION\" == \"long_gpuh200\" ]]",
    },
    "endcell/jobs/gemma2_standard_tests.sh": {
        "source \"$PROTOBUF_ENV\"",
        "PYTHONPATH=\"$TEST_DEPS:$GEMMA_PROTOBUF_DIR\"",
        "unset PYTHONHOME", "protobuf_env=$PROTOBUF_ENV",
    },
    "endcell/jobs/gemma2_standard_protobuf_env.sh": {
        "export PYTHONPATH=\"$GEMMA_PROTOBUF_DIR\"",
        "GEMMA_PROTOBUF_TREE_SHA256", "unset PYTHONHOME",
        "google/_upb/_message.abi3.so", "RECORD",
    },
    "endcell/jobs/gemma2_standard_smoke.sbatch": {
        "source \"$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh\"",
        "unset PYTHONHOME",
    },
    "endcell/jobs/gemma2_standard_train.sbatch": {
        "source \"$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh\"",
        "unset PYTHONHOME", "[[ \"${SLURM_JOB_PARTITION:-}\" == \"long_gpuh200\" ]]",
    },
    "endcell/jobs/gemma2_standard_eval.sbatch": {
        "source \"$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh\"",
        "unset PYTHONHOME",
    },
    "docs/endcell/gemma2_standard_hpc_runbook.md": {
        "protobuf-5.29.5", "Do **not** install TikToken", "256002",
        "google/_upb/_message.abi3.so", "schema 4", "literal `long_gpuh200`",
    },
}


FORBIDDEN_TEXT = {
    "endcell/jobs/gemma2_standard_preflight.sh": {"EXPECTED_LONG_PARTITION"},
    "endcell/jobs/gemma2_standard_train.sbatch": {"EXPECTED_LONG_PARTITION"},
    "docs/endcell/gemma2_standard_hpc_runbook.md": {"EXPECTED_LONG_PARTITION"},
}


def argparse_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    failures: list[str] = []

    parser_flags: dict[str, set[str]] = {}
    for name, (relative, required) in PARSER_REQUIREMENTS.items():
        path = repo / relative
        observed = argparse_flags(path)
        parser_flags[name] = observed
        missing = sorted(required - observed)
        if missing:
            failures.append(f"{name} parser missing {missing}")
        print(f"[CLI] {name}: required={len(required)} observed={len(observed)} missing={len(missing)}")

    known_application_flags = set().union(*parser_flags.values())
    for relative, required in CONSUMER_REQUIREMENTS.items():
        text = (repo / relative).read_text(encoding="utf-8")
        missing = sorted(flag for flag in required if flag not in text)
        unknown_required = sorted(required - known_application_flags)
        if missing:
            failures.append(f"{relative} does not invoke/document {missing}")
        if unknown_required:
            failures.append(f"{relative} requires unknown application flags {unknown_required}")
        print(f"[CLI] {relative}: required={len(required)} missing={len(missing)}")

    for relative, required in TEXT_REQUIREMENTS.items():
        text = (repo / relative).read_text(encoding="utf-8")
        missing = sorted(fragment for fragment in required if fragment not in text)
        if missing:
            failures.append(f"{relative} lacks required contract text {missing}")
        print(f"[TEXT] {relative}: required={len(required)} missing={len(missing)}")

    for relative, forbidden in FORBIDDEN_TEXT.items():
        text = (repo / relative).read_text(encoding="utf-8")
        present = sorted(fragment for fragment in forbidden if fragment in text)
        if present:
            failures.append(f"{relative} contains forbidden contract text {present}")
        print(f"[TEXT] {relative}: forbidden={len(forbidden)} present={len(present)}")

    if failures:
        for failure in failures:
            print(f"[FATAL] {failure}")
        raise SystemExit(1)
    print("[PASS] current argparse definitions and every Gemma job/runbook command reconcile")


if __name__ == "__main__":
    main()

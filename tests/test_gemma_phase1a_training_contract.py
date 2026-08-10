"""Offline contract tests for the opt-in Gemma trainer path.

These tests use a tiny fake tokenizer and tiny Torch modules; they never access Hugging Face or
Tahoe data.  Run on the cluster environment with:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
        tests/test_gemma_phase1a_training_contract.py -q
"""

import copy
import json
import os
import random
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "endcell", "train"))

import train_c2s_tahoe_endcell as trainer  # noqa: E402
import gemma_tokenizer_probe as probe  # noqa: E402


class FakeTokenizer:
    def __init__(self, pad_id=0, eos_id=1, bos_id=2, is_fast=False):
        self.vocab = {"<pad>": pad_id, "<eos>": eos_id, "<bos>": bos_id,
                      "<unk>": 3, "PROMPT": 10, "GENE1": 11, "GENE2": 12}
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self.bos_token_id = bos_id
        self.unk_token_id = 3
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.additional_special_tokens = []
        self.padding_side = "left"
        self.truncation_side = "right"
        self.is_fast = bool(is_fast)

    @property
    def vocab_size(self):
        return 13

    @property
    def all_special_ids(self):
        return ([self.pad_token_id, self.eos_token_id, self.bos_token_id, self.unk_token_id] +
                [self.vocab[x] for x in self.additional_special_tokens])

    @property
    def special_tokens_map(self):
        return {"pad_token": self.pad_token, "eos_token": self.eos_token,
                "bos_token": self.bos_token, "unk_token": self.unk_token,
                "additional_special_tokens": list(self.additional_special_tokens)}

    def __len__(self):
        return max(self.vocab.values()) + 1

    def get_added_vocab(self):
        return {x: self.vocab[x] for x in self.additional_special_tokens}

    def add_special_tokens(self, value):
        added = 0
        for token in value["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.convert_tokens_to_ids(x) for x in text.strip().split()]

    def __call__(self, texts, add_special_tokens=False, **_kwargs):
        return {"input_ids": [self.encode(x, add_special_tokens=add_special_tokens)
                              for x in texts]}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab,
                       "additional_special_tokens": self.additional_special_tokens,
                       "pad": self.pad_token_id, "eos": self.eos_token_id,
                       "bos": self.bos_token_id, "is_fast": self.is_fast}, f)

    @classmethod
    def from_pretrained(cls, path):
        value = json.load(open(os.path.join(path, "fake_tokenizer.json"), encoding="utf-8"))
        out = cls(value["pad"], value["eos"], value["bos"], value.get("is_fast", False))
        out.vocab = value["vocab"]
        out.additional_special_tokens = value["additional_special_tokens"]
        return out


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path, **_kwargs):
        return FakeTokenizer.from_pretrained(path)


class TinySaveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"architectures": [self.__class__.__name__]}, f)
        torch.save(self.state_dict(), os.path.join(path, "model.safetensors"))


def _checkpoint_state(step, optimizer=None, scheduler=None, rng=None, completed=False,
                      microbatch_position=None):
    return {
        "schema_version": 1,
        "optimizer": {} if optimizer is None else optimizer.state_dict(),
        "scheduler": {} if scheduler is None else scheduler.state_dict(),
        "epoch": 0,
        "microbatch_position": (step if microbatch_position is None else
                                microbatch_position),
        "accumulation_position": 0,
        "global_step": step,
        "best_eval_loss": 1.0,
        "rng": trainer._capture_rng_state() if rng is None else rng,
        "contract": {"fixture": "phase1a"},
        "contract_fingerprint": "fixture-contract",
        "epoch_loss_sum": 0.0,
        "epoch_tokens": 0,
        "window_loss_sum": 0.0,
        "window_tokens": 0,
        "accounting": {},
        "completed": completed,
    }


def _checkpoint_provenance():
    return {"contract_fingerprint": "fixture-contract", "fixture": True}


def _write_jsonl(tmp_path, n=1):
    path = tmp_path / "tiny.jsonl"
    rows = [{"prompt": "PROMPT", "response": "GENE1 GENE2 [END_CELL]"} for _ in range(n)]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return str(path)


def test_pythia_legacy_tokens_and_labels_are_unchanged(tmp_path):
    tok = FakeTokenizer()
    trainer.register_sentinels(tok)
    ds = trainer.C2SDataset(_write_jsonl(tmp_path), tok, max_length=32)
    item = ds[0]
    end = tok.convert_tokens_to_ids("[END_CELL]")
    assert item["input_ids"].tolist() == [10, 11, 12, end, 1]
    assert item["labels"].tolist() == [-100, 11, 12, end, 1]
    assert 2 not in item["input_ids"].tolist()


def test_gemma_path_has_exactly_one_masked_bos(tmp_path):
    tok = FakeTokenizer()
    trainer.register_sentinels(tok)
    ds = trainer.C2SDataset(_write_jsonl(tmp_path), tok, max_length=32, prepend_bos=True)
    item = ds[0]
    assert item["input_ids"].tolist().count(tok.bos_token_id) == 1
    assert item["input_ids"][0].item() == tok.bos_token_id
    assert item["labels"][0].item() == -100


def test_gemma_contract_rejects_pad_equal_to_eos():
    tok = FakeTokenizer(pad_id=1, eos_id=1)
    with pytest.raises(ValueError, match="PAD distinct from EOS"):
        trainer.validate_training_token_contract(tok, prepend_bos=True,
                                                 require_distinct_pad_eos=True)


def test_strict_sentinels_are_atomic_distinct_non_unk_and_survive_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "AutoTokenizer", FakeAutoTokenizer)
    tok = FakeTokenizer()
    result = trainer.register_sentinels(tok, strict=True)
    assert set(result["ids"]) == set(trainer.SENTINELS)
    assert len(set(result["ids"].values())) == 2
    assert tok.unk_token_id not in result["ids"].values()


def test_strict_reload_accepts_semantic_equivalence_across_tokenizer_class(monkeypatch):
    class ReloadedFastTokenizer(FakeTokenizer):
        pass

    seen = {}

    class ClassChangingAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.update(kwargs)
            base = FakeTokenizer.from_pretrained(path)
            out = ReloadedFastTokenizer(
                base.pad_token_id, base.eos_token_id, base.bos_token_id, is_fast=True)
            out.vocab = base.vocab
            out.additional_special_tokens = base.additional_special_tokens
            return out

    monkeypatch.setattr(trainer, "AutoTokenizer", ClassChangingAutoTokenizer)
    tok = FakeTokenizer(is_fast=True)
    result = trainer.register_sentinels(tok, strict=True, reload_use_fast=True)
    assert seen == {"use_fast": True}
    assert result["ids"] == {x: tok.convert_tokens_to_ids(x) for x in trainer.SENTINELS}
    reloaded = ReloadedFastTokenizer(is_fast=True)
    reloaded.vocab = dict(tok.vocab)
    reloaded.additional_special_tokens = list(tok.additional_special_tokens)
    assert trainer.tokenizer_fingerprint(tok) != trainer.tokenizer_fingerprint(reloaded)
    assert (trainer.semantic_tokenizer_fingerprint(tok) ==
            trainer.semantic_tokenizer_fingerprint(reloaded))


def test_probe_preserves_explicit_slow_tokenizer_on_serialization(monkeypatch):
    calls = []

    class RecordingAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(kwargs.get("use_fast"))
            if os.path.isdir(path):
                return FakeTokenizer.from_pretrained(path)
            return FakeTokenizer(is_fast=bool(kwargs.get("use_fast")))

    monkeypatch.setattr(probe, "AutoTokenizer", RecordingAutoTokenizer)
    monkeypatch.setattr(trainer, "AutoTokenizer", RecordingAutoTokenizer)
    result = probe.check_slow_fast_parity(
        "fixture-model", "fixture-revision", FakeTokenizer(is_fast=True), [])
    assert result["available"] is True
    assert result["mismatch_count"] == 0
    assert calls == [False, False]


def test_strict_sentinel_rejects_non_atomic_token(monkeypatch):
    tok = FakeTokenizer()
    original = tok.encode

    def split_end(text, add_special_tokens=False):
        if text == "[END_CELL]":
            return [20, 21]
        return original(text, add_special_tokens=add_special_tokens)

    tok.encode = split_end
    with pytest.raises(ValueError, match="not atomic"):
        trainer.register_sentinels(tok, strict=True)


def test_resume_fingerprint_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="REFUSING TO RESUME"):
        trainer._validate_resume_fingerprints(
            {"train_sha256": "a", "seed": 42},
            {"train_sha256": "b", "seed": 42})


def test_auto_resume_without_published_checkpoint_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="no valid published"):
        trainer.resolve_resume_checkpoint("auto", str(tmp_path))


def test_checkpoint_publication_is_immutable_verified_and_timed(tmp_path):
    torch.manual_seed(4)
    model = TinySaveModel()
    tok = FakeTokenizer()
    target = tmp_path / "checkpoint-5"
    state = _checkpoint_state(5)
    provenance = _checkpoint_provenance()

    first = trainer._save_checkpoint_atomic(
        str(target), model, tok, state, provenance)
    assert first["reused"] is False
    assert first["duration_seconds"] >= 0
    assert first["hash_duration_seconds"] >= 0
    manifest = json.loads((target / trainer.CHECKPOINT_MANIFEST).read_text(
        encoding="utf-8"))
    assert manifest["schema_version"] == trainer.CHECKPOINT_MANIFEST_SCHEMA
    for name in (trainer.TRAINING_STATE, trainer.PROVENANCE_FILE, "config.json",
                 "fake_tokenizer.json", "model.safetensors"):
        assert manifest["files"][name]["size"] == (target / name).stat().st_size
        assert manifest["files"][name]["sha256"] == trainer.sha256_file(target / name)
    before = {name: trainer.sha256_file(target / name)
              for name in (trainer.TRAINING_STATE, "model.safetensors",
                           trainer.CHECKPOINT_MANIFEST)}

    second = trainer._save_checkpoint_atomic(
        str(target), model, tok, state, provenance)
    assert second["reused"] is True
    assert second["identity"] == first["identity"]
    assert before == {name: trainer.sha256_file(target / name) for name in before}

    conflicting = dict(state, global_step=6, microbatch_position=6)
    with pytest.raises(RuntimeError, match="REFUSING TO OVERWRITE"):
        trainer._save_checkpoint_atomic(
            str(target), model, tok, conflicting, provenance)
    assert before == {name: trainer.sha256_file(target / name) for name in before}
    assert not list(tmp_path.glob(".checkpoint-5.tmp-*"))
    assert not (tmp_path / "checkpoint-5.publish.lock").exists()


def test_checkpoint_content_hash_rejects_same_size_corruption(tmp_path):
    target = tmp_path / "checkpoint-5-mb80"
    trainer._save_checkpoint_atomic(
        str(target), TinySaveModel(), FakeTokenizer(),
        _checkpoint_state(5, microbatch_position=80), _checkpoint_provenance())
    config = target / "config.json"
    original = config.read_bytes()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    assert len(replacement) == len(original)
    config.write_bytes(replacement)
    with pytest.raises(RuntimeError, match="inventory/content mismatch"):
        trainer._validate_published_checkpoint(str(target))


def test_checkpoint_name_and_auto_resume_distinguish_same_step_microbatches(tmp_path):
    assert trainer._checkpoint_dir_name(42_198, 675_168) == \
        "checkpoint-42198-mb675168"
    assert trainer._parse_checkpoint_dir_name("checkpoint-42198") == (42_198, None)
    assert trainer._parse_checkpoint_dir_name("checkpoint-42198-mb675183") == \
        (42_198, 675_183)

    provenance = _checkpoint_provenance()
    earlier = tmp_path / trainer._checkpoint_dir_name(42_198, 675_168)
    later = tmp_path / trainer._checkpoint_dir_name(42_198, 675_183)
    trainer._save_checkpoint_atomic(
        str(earlier), TinySaveModel(), FakeTokenizer(),
        _checkpoint_state(42_198, microbatch_position=675_168), provenance)
    trainer._save_checkpoint_atomic(
        str(later), TinySaveModel(), FakeTokenizer(),
        _checkpoint_state(42_198, microbatch_position=675_183), provenance)

    assert earlier.is_dir() and later.is_dir()
    assert trainer.resolve_resume_checkpoint("auto", str(tmp_path)) == str(later)


def test_checkpoint_failure_cleans_only_its_own_temp(tmp_path):
    class BrokenModel(TinySaveModel):
        def save_pretrained(self, path):
            super().save_pretrained(path)
            raise RuntimeError("planted save failure")

    unrelated = tmp_path / ".checkpoint-8.tmp-unrelated"
    unrelated.mkdir()
    with pytest.raises(RuntimeError, match="planted save failure"):
        trainer._save_checkpoint_atomic(
            str(tmp_path / "checkpoint-8"), BrokenModel(), FakeTokenizer(),
            _checkpoint_state(8), _checkpoint_provenance())
    assert unrelated.is_dir()
    assert not (tmp_path / "checkpoint-8").exists()
    assert not (tmp_path / "checkpoint-8.publish.lock").exists()


def test_immediate_signal_reuses_just_written_boundary():
    saved = {"global_step": 100, "epoch": 0, "microbatch_position": 1600,
             "result": {"duration_seconds": 12.5}}
    assert trainer._saved_checkpoint_matches(
        saved, global_step=100, epoch=0, microbatch_position=1600)
    assert not trainer._saved_checkpoint_matches(
        saved, global_step=100, epoch=0, microbatch_position=1601)


def test_deterministic_sampler_reconstructs_epoch_and_offset():
    data = list(range(31))
    full = trainer.DeterministicEpochSampler(data, seed=42, epoch=3)
    resumed = trainer.DeterministicEpochSampler(data, seed=42, epoch=3, start_index=17)
    assert list(resumed) == list(full)[17:]
    assert list(trainer.DeterministicEpochSampler(data, 42, 3)) == list(full)
    assert list(trainer.DeterministicEpochSampler(data, 42, 4)) != list(full)


def _toy_run(interrupt_at=None, resume=None):
    random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.2),
                                torch.nn.Linear(4, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 - step / 12.0)
    start = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        start = resume["next_step"]
        trainer._restore_rng_state(resume["rng"])
    x = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 10
    y = torch.arange(12, dtype=torch.float32).reshape(12, 1) / 10
    for step in range(start, 12):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x[step:step + 1]), y[step:step + 1])
        loss.backward()
        opt.step()
        sched.step()
        if interrupt_at is not None and step + 1 == interrupt_at:
            return {"model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(opt.state_dict()),
                    "scheduler": copy.deepcopy(sched.state_dict()),
                    "next_step": step + 1, "rng": trainer._capture_rng_state()}
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), sched.state_dict()


def test_interrupted_resume_matches_uninterrupted_exactly():
    uninterrupted = _toy_run()
    partial = _toy_run(interrupt_at=5)
    resumed = _toy_run(resume=partial)
    for name in uninterrupted[0]:
        assert torch.equal(uninterrupted[0][name], resumed[0][name]), name
    assert uninterrupted[1]["param_groups"] == resumed[1]["param_groups"]
    assert uninterrupted[2] == resumed[2]


def _writer_backed_toy_run(tmp_path, interrupt_at=None, resume_dir=None):
    random.seed(7)
    torch.manual_seed(7)
    model = TinySaveModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 - step / 12.0)
    start = 0
    if resume_dir is not None:
        published = trainer._validate_published_checkpoint(str(resume_dir))
        model.load_state_dict(trainer._torch_load(resume_dir / "model.safetensors"))
        state = published["state"]
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start = state["microbatch_position"]
        trainer._restore_rng_state(state["rng"])
    x = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 10
    y = torch.arange(12, dtype=torch.float32).reshape(12, 1) / 10
    for step in range(start, 12):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x[step:step + 1]), y[step:step + 1])
        loss.backward()
        opt.step()
        sched.step()
        if interrupt_at is not None and step + 1 == interrupt_at:
            target = tmp_path / f"checkpoint-{step + 1}"
            state = _checkpoint_state(
                step + 1, optimizer=opt, scheduler=sched,
                rng=trainer._capture_rng_state())
            trainer._save_checkpoint_atomic(
                str(target), model, FakeTokenizer(), state, _checkpoint_provenance())
            return target
    return (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()),
            copy.deepcopy(sched.state_dict()))


def test_actual_checkpoint_writer_resume_matches_uninterrupted(tmp_path):
    uninterrupted = _writer_backed_toy_run(tmp_path / "full")
    checkpoint = _writer_backed_toy_run(tmp_path / "split", interrupt_at=5)
    assert trainer.resolve_resume_checkpoint("auto", str(tmp_path / "split")) == str(checkpoint)
    resumed = _writer_backed_toy_run(tmp_path / "split", resume_dir=checkpoint)
    for name in uninterrupted[0]:
        assert torch.equal(uninterrupted[0][name], resumed[0][name]), name
    assert uninterrupted[1]["param_groups"] == resumed[1]["param_groups"]
    assert uninterrupted[2] == resumed[2]


def test_canonical_remainder_is_fifteen_and_not_flushed():
    n, grad_accum = 675_183, 16
    assert n // grad_accum == 42_198
    assert n % grad_accum == 15


def test_probe_detects_semantic_truncation_and_sentinel_loss():
    tok = FakeTokenizer()
    trainer.register_sentinels(tok)
    stats = probe._semantic_lengths(
        tok, "PROMPT", "GENE1 GENE2 [END_CELL]", prepend_bos=True, max_length=4)
    assert stats["truncated_response_tokens"] > 0
    assert stats["sentinel_loss"] is True

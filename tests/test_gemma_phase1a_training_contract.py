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
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "endcell", "train"))
sys.path.insert(0, os.path.join(ROOT, "endcell", "jobs"))

import train_c2s_tahoe_endcell as trainer  # noqa: E402
import gemma_tokenizer_probe as probe  # noqa: E402
import gemma2_standard_preflight_contract as preflight_contract  # noqa: E402
import gemma2_standard_tests_contract as tests_contract  # noqa: E402
import gemma2_standard_checkpoint_fingerprint as checkpoint_fingerprint  # noqa: E402
import gemma2_standard_provenance as shared_provenance  # noqa: E402


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
    contract = {"fixture": "phase1a"}
    return {
        "schema_version": trainer.TRAINING_STATE_SCHEMA,
        "optimizer": {} if optimizer is None else optimizer.state_dict(),
        "scheduler": {} if scheduler is None else scheduler.state_dict(),
        "epoch": 0,
        "microbatch_position": (step if microbatch_position is None else
                                microbatch_position),
        "accumulation_position": 0,
        "global_step": step,
        "best_eval_loss": 1.0,
        "rng": trainer._capture_rng_state() if rng is None else rng,
        "contract": contract,
        "contract_fingerprint": trainer._stable_json_hash(contract),
        "epoch_loss_sum": 0.0,
        "epoch_tokens": 0,
        "window_loss_sum": 0.0,
        "window_tokens": 0,
        "accounting": {},
        "completed": completed,
    }


def _checkpoint_provenance():
    contract = {"fixture": "phase1a"}
    return {
        "schema_version": trainer.PROVENANCE_SCHEMA,
        "contract": contract,
        "contract_fingerprint": trainer._stable_json_hash(contract),
        "fixture": True,
    }


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


def test_strict_sentinels_are_idempotent_when_resume_reload_omits_additional_list(monkeypatch):
    class ResumeTokenizer(FakeTokenizer):
        forced_special_ids = []

        @property
        def all_special_ids(self):
            return super().all_special_ids + list(self.forced_special_ids)

    monkeypatch.setattr(trainer, "AutoTokenizer", FakeAutoTokenizer)
    tok = ResumeTokenizer()
    trainer.register_sentinels(tok)
    sentinel_ids = {tok.convert_tokens_to_ids(token) for token in trainer.SENTINELS}
    tok.additional_special_tokens = []
    tok.forced_special_ids = sorted(sentinel_ids)
    tok.convert_ids_to_tokens = lambda token_id: next(
        (token for token, value in tok.vocab.items() if value == token_id), "<unk>")

    result = trainer.register_sentinels(tok, strict=True)

    assert result["added"] == 0
    assert set(result["ids"].values()) == sentinel_ids


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
        "fixture-model", {"revision": "fixture-revision"},
        FakeTokenizer(is_fast=True), [])
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


class GemmaContractTokenizer:
    def __init__(self, vocab_size=256_000, length=256_000, unknown=False):
        self.vocab_size = vocab_size
        self._length = length
        self.unk_token_id = 3
        self.unknown = unknown

    def __len__(self):
        return self._length

    def encode(self, _text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.unk_token_id] if self.unknown else [10, 11, 12, 13]


class GemmaEmbeddingModel:
    def __init__(self, rows, config_vocab):
        self.embedding = SimpleNamespace(weight=torch.empty(rows, 1))
        self.config = SimpleNamespace(vocab_size=config_vocab)

    def get_input_embeddings(self):
        return self.embedding


def test_gemma_base_contract_rejects_planted_five_token_collapse():
    collapsed = GemmaContractTokenizer(vocab_size=5, length=5, unknown=True)
    with pytest.raises(ValueError, match="collapsed or mismatched"):
        trainer.validate_gemma_base_tokenizer_contract(collapsed, 256_000)


def test_gemma_base_contract_rejects_model_vocabulary_mismatch():
    tokenizer = GemmaContractTokenizer()
    with pytest.raises(ValueError, match="config vocab_size"):
        trainer.validate_gemma_base_tokenizer_contract(tokenizer, 255_999)


def test_gemma_registered_and_embedding_contracts_split_cold_start_from_resume():
    tokenizer = GemmaContractTokenizer(length=256_002)
    sentinel_result = {
        "added": 2,
        "ids": {"[END_CELL]": 256_000, "[DOWN]": 256_001},
    }
    registered = trainer.validate_gemma_registered_tokenizer_contract(
        tokenizer, sentinel_result)
    assert registered["length"] == 256_002

    cold = GemmaEmbeddingModel(256_000, 256_000)
    result = trainer.validate_model_embedding_rows_before_resize(
        cold, tokenizer, strict_gemma=True, resume=False,
        base_tokenizer_length=256_000)
    assert result["embedding_rows"] == 256_000

    resumed = GemmaEmbeddingModel(256_002, 256_002)
    result = trainer.validate_model_embedding_rows_before_resize(
        resumed, tokenizer, strict_gemma=True, resume=True,
        base_tokenizer_length=256_002)
    assert result["embedding_rows"] == 256_002


@pytest.mark.parametrize(
    "tokenizer,sentinel_result,match",
    [
        (GemmaContractTokenizer(length=256_001),
         {"added": 2, "ids": {"[END_CELL]": 256_000, "[DOWN]": 256_001}},
         "tokenizer length"),
        (GemmaContractTokenizer(length=256_002),
         {"added": 2, "ids": {"[END_CELL]": 256_001, "[DOWN]": 256_000}},
         "sentinel ids"),
    ],
)
def test_gemma_registered_contract_rejects_wrong_length_or_ids(
        tokenizer, sentinel_result, match):
    with pytest.raises(ValueError, match=match):
        trainer.validate_gemma_registered_tokenizer_contract(tokenizer, sentinel_result)


def test_gemma_embedding_contract_rejects_collapse_before_resize():
    tokenizer = GemmaContractTokenizer(length=7)
    parent = GemmaEmbeddingModel(256_000, 256_000)
    with pytest.raises(ValueError, match="refusing Gemma resize"):
        trainer.validate_model_embedding_rows_before_resize(
            parent, tokenizer, strict_gemma=True, resume=False,
            base_tokenizer_length=5)


@pytest.mark.parametrize(
    "model,resume,match",
    [
        (GemmaEmbeddingModel(255_999, 256_000), False, "parent embedding/config mismatch"),
        (GemmaEmbeddingModel(256_001, 256_002), True, "resume checkpoint is not already resized"),
        (GemmaEmbeddingModel(256_002, 256_001), True, "resume checkpoint is not already resized"),
    ],
)
def test_gemma_embedding_contract_rejects_cold_or_resume_row_mismatch(model, resume, match):
    tokenizer = GemmaContractTokenizer(length=256_002)
    with pytest.raises(ValueError, match=match):
        trainer.validate_model_embedding_rows_before_resize(
            model, tokenizer, strict_gemma=True, resume=resume,
            base_tokenizer_length=256_002 if resume else 256_000)


def test_gemma_identity_enforces_flags_and_never_misclassifies_pythia():
    gemma = SimpleNamespace(model_type="gemma2")
    pythia = SimpleNamespace(model_type="gpt_neox")
    with pytest.raises(ValueError, match="requires both"):
        trainer.gemma_mode_from_config(
            gemma, prepend_bos=False, strict_token_contract=True)
    assert trainer.gemma_mode_from_config(
        gemma, prepend_bos=True, strict_token_contract=True) is True
    assert trainer.gemma_mode_from_config(
        pythia, prepend_bos=True, strict_token_contract=True) is False


def _canonical_gemma_snapshot(tmp_path):
    hub = tmp_path / "hub"
    snapshot = (hub / "models--vandijklab--C2S-Scale-Gemma-2-2B" / "snapshots" /
                trainer.GEMMA_MODEL_REVISION)
    snapshot.mkdir(parents=True)
    return hub, snapshot


def _planted_official_snapshot(monkeypatch, tmp_path):
    """Build the exact official filename tree with tiny independently pinned bytes."""
    hub, snapshot = _canonical_gemma_snapshot(tmp_path)
    expected = {}
    for index, name in enumerate(sorted(shared_provenance.OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES)):
        payload = f"planted-official-{index}-{name}".encode("utf-8")
        path = snapshot / name
        path.write_bytes(payload)
        if name in shared_provenance.AUTHORITATIVE_REVISION_FILES:
            expected[name] = {
                "size": len(payload),
                "sha256": shared_provenance.sha256_file(path),
                "upstream_hash_kind": "planted-authoritative-sha256",
            }
    monkeypatch.setattr(shared_provenance, "AUTHORITATIVE_REVISION_FILES", expected)
    monkeypatch.setattr(preflight_contract, "AUTHORITATIVE_REVISION_FILES", expected)
    return hub, snapshot, expected


def test_gemma_cold_start_retains_logical_identity_but_loads_exact_snapshot(tmp_path):
    hub, snapshot = _canonical_gemma_snapshot(tmp_path)
    contract = trainer.resolve_model_load_contract(
        trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, str(snapshot),
        hub_cache=str(hub))
    assert contract["logical_model_name"] == trainer.GEMMA_MODEL_ID
    assert contract["logical_model_revision"] == trainer.GEMMA_MODEL_REVISION
    assert contract["parent_snapshot_path"] == str(snapshot.resolve())
    assert contract["active_source"] == str(snapshot.resolve())
    assert contract["active_role"] == "exact_parent_snapshot"
    assert contract["loader_kwargs"] == {"local_files_only": True}


def test_gemma_model_id_route_and_arbitrary_local_path_are_rejected(tmp_path):
    hub, snapshot = _canonical_gemma_snapshot(tmp_path)
    with pytest.raises(ValueError, match="model-ID tokenizer route is forbidden"):
        trainer.resolve_model_load_contract(
            trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, None,
            hub_cache=str(hub))
    arbitrary = tmp_path / "arbitrary-model"
    arbitrary.mkdir()
    with pytest.raises(ValueError, match="not the pinned snapshot"):
        trainer.resolve_model_load_contract(
            trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, str(arbitrary),
            hub_cache=str(hub))
    assert snapshot.is_dir()


def test_authoritative_revision_mismatch_cannot_be_blessed_by_new_certificate(
        monkeypatch, tmp_path):
    hub, snapshot, expected = _planted_official_snapshot(monkeypatch, tmp_path)
    payload = snapshot / "config.json"
    certified = preflight_contract.certified_snapshot(
        trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, snapshot, hub)
    assert certified["authoritative_revision_files"] == expected

    payload.write_bytes(b"tampered-bytes!")
    with pytest.raises(SystemExit, match="authoritative Gemma revision mismatch"):
        preflight_contract.certified_snapshot(
            trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, snapshot, hub)


@pytest.mark.parametrize("override_name", ["model.safetensors", "tokenizer.json"])
def test_certificate_creation_rejects_transformers_override_file(
        monkeypatch, tmp_path, override_name):
    hub, snapshot, _ = _planted_official_snapshot(monkeypatch, tmp_path)
    (snapshot / override_name).write_bytes(b"planted-override")
    args = SimpleNamespace(
        model_id=trainer.GEMMA_MODEL_ID,
        revision=trainer.GEMMA_MODEL_REVISION,
        snapshot_path=str(snapshot),
        hub_cache=str(hub),
    )
    with pytest.raises(SystemExit, match="unexpected=.*" + override_name.replace(".", r"\.")):
        preflight_contract.create(args)


@pytest.mark.parametrize("override_name", ["model.safetensors", "tokenizer.json"])
def test_certificate_verification_rejects_transformers_override_file(
        monkeypatch, tmp_path, override_name):
    hub, snapshot, _ = _planted_official_snapshot(monkeypatch, tmp_path)
    (snapshot / override_name).write_bytes(b"planted-override")
    certificate = tmp_path / "PREFLIGHT_PASSED.json"
    certificate.write_text(json.dumps({
        "schema_version": preflight_contract.SCHEMA_VERSION,
        "preflight_passed": True,
        "model": {
            "repo_id": trainer.GEMMA_MODEL_ID,
            "revision": trainer.GEMMA_MODEL_REVISION,
            "snapshot_path": str(snapshot),
            "hub_cache": str(hub),
        },
    }), encoding="utf-8")
    args = SimpleNamespace(
        certificate=str(certificate),
        model_id=trainer.GEMMA_MODEL_ID,
        revision=trainer.GEMMA_MODEL_REVISION,
    )
    with pytest.raises(SystemExit, match="unexpected=.*" + override_name.replace(".", r"\.")):
        preflight_contract.verify(args)


def test_snapshot_guard_rechecks_after_load_and_rejects_persistent_mutation(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload = snapshot / "tokenizer.model"
    payload.write_bytes(b"stable")
    digest = shared_provenance.snapshot_inventory_sha256(snapshot)

    def mutating_load():
        payload.write_bytes(b"changed")
        return "loaded"

    with pytest.raises(shared_provenance.ProvenanceError, match="after planted load"):
        shared_provenance.guarded_snapshot_load(
            mutating_load, snapshot, digest, stage="planted load")


def test_frozen_parent_accepts_beegfs_alias_but_retains_certificate_path(
        monkeypatch, tmp_path):
    _, snapshot, expected = _ancestry_fixture(tmp_path)
    alias = tmp_path / "mnt-beegfsnew-alias"
    try:
        alias.symlink_to(snapshot, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"platform cannot create directory symlink: {exc}")
    monkeypatch.setattr(
        checkpoint_fingerprint, "verify_authoritative_snapshot", lambda _path: {})
    assert shared_provenance.paths_refer_to_same_location(alias, snapshot)
    result = checkpoint_fingerprint.validate_gemma_parent_snapshot(alias, **expected)
    assert result["parent_snapshot"] == str(snapshot.resolve())
    different = tmp_path / "genuinely-different-parent"
    different.mkdir()
    assert not shared_provenance.paths_refer_to_same_location(different, snapshot)
    with pytest.raises(
            checkpoint_fingerprint.CheckpointValidationError,
            match="not the certificate snapshot"):
        checkpoint_fingerprint.validate_gemma_parent_snapshot(different, **expected)


def test_gemma_resume_loads_checkpoint_not_parent_and_retains_parent_provenance(tmp_path):
    hub, snapshot = _canonical_gemma_snapshot(tmp_path)
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    contract = trainer.resolve_model_load_contract(
        trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, str(snapshot),
        resume_dir=str(checkpoint), hub_cache=str(hub))
    assert contract["active_source"] == str(checkpoint.resolve())
    assert contract["active_role"] == "resume_checkpoint"
    assert contract["parent_snapshot_path"] == str(snapshot.resolve())
    assert contract["loader_kwargs"] == {"local_files_only": True}


def test_pythia_loader_contract_is_unchanged_by_gemma_snapshot_option():
    contract = trainer.resolve_model_load_contract(
        "vandijklab/C2S-Scale-Pythia-1b-pt", probe.PYTHIA_REVISION, None)
    assert contract["active_source"] == "vandijklab/C2S-Scale-Pythia-1b-pt"
    assert contract["parent_snapshot_path"] is None
    assert contract["loader_kwargs"] == {"revision": probe.PYTHIA_REVISION}


def test_any_gemma2_config_rejects_arbitrary_model_name_before_tokenizer(tmp_path):
    arbitrary = tmp_path / "arbitrary-gemma2"
    arbitrary.mkdir()
    with pytest.raises(ValueError, match="canonical logical model and revision"):
        trainer.validate_gemma_runtime_provenance(
            SimpleNamespace(model_type="gemma2"),
            model_name=str(arbitrary), model_revision="different", model_load_path=None,
            preflight_certificate_sha256="a" * 64,
            parent_snapshot_inventory_sha256="b" * 64,
            parent_authoritative_files_sha256="c" * 64,
            prepend_bos=True, strict_token_contract=True,
            hub_cache=str(tmp_path / "hub"),
        )
    assert trainer.validate_gemma_runtime_provenance(
        SimpleNamespace(model_type="gpt_neox"),
        model_name="vandijklab/C2S-Scale-Pythia-1b-pt",
        model_revision=probe.PYTHIA_REVISION, model_load_path=None,
        preflight_certificate_sha256=None,
        parent_snapshot_inventory_sha256=None,
        parent_authoritative_files_sha256=None,
        prepend_bos=False, strict_token_contract=False,
    ) is None


def _ancestry_fixture(tmp_path):
    _, snapshot = _canonical_gemma_snapshot(tmp_path)
    (snapshot / "config.json").write_text('{"model_type":"gemma2"}\n', encoding="utf-8")
    (snapshot / "tokenizer.model").write_bytes(b"sentencepiece")
    (snapshot / "model.safetensors").write_bytes(b"parent-weights")
    snapshot_digest = shared_provenance.snapshot_inventory_sha256(snapshot)
    authoritative_digest = shared_provenance.authoritative_revision_files_sha256()
    certificate_digest = "a" * 64
    contract = {
        "model_name": trainer.GEMMA_MODEL_ID,
        "model_revision": trainer.GEMMA_MODEL_REVISION,
        "parent_model_load_path": str(snapshot.resolve()),
        "preflight_certificate_sha256": certificate_digest,
        "parent_snapshot_inventory_sha256": snapshot_digest,
        "parent_authoritative_files_sha256": authoritative_digest,
    }
    contract_fingerprint = trainer._stable_json_hash(contract)
    provenance = {
        "schema_version": trainer.PROVENANCE_SCHEMA,
        "model": {
            "requested_source": trainer.GEMMA_MODEL_ID,
            "requested_revision": trainer.GEMMA_MODEL_REVISION,
            "certified_parent_snapshot_path": str(snapshot.resolve()),
            "preflight_certificate_sha256": certificate_digest,
            "parent_snapshot_inventory_sha256": snapshot_digest,
            "parent_authoritative_files_sha256": authoritative_digest,
        },
        "contract": contract,
        "contract_fingerprint": contract_fingerprint,
    }
    state = _checkpoint_state(5, completed=False)
    state["contract"] = contract
    state["contract_fingerprint"] = contract_fingerprint
    checkpoint = tmp_path / "checkpoint-5-mb5"
    trainer._save_checkpoint_atomic(
        str(checkpoint), TinySaveModel(), FakeTokenizer(), state, provenance)
    expected = {
        "model_id": trainer.GEMMA_MODEL_ID,
        "revision": trainer.GEMMA_MODEL_REVISION,
        "parent_snapshot": str(snapshot.resolve()),
        "preflight_certificate_sha256": certificate_digest,
        "snapshot_inventory_sha256": snapshot_digest,
        "authoritative_files_sha256": authoritative_digest,
    }
    return checkpoint, snapshot, expected


def test_shared_checkpoint_ancestry_accepts_exact_parent_and_rejects_cert_substitution(tmp_path):
    checkpoint, _, expected = _ancestry_fixture(tmp_path)
    checkpoint_fingerprint.validate_checkpoint_manifest(
        checkpoint, required=True, verify_state_identity=True)
    checkpoint_fingerprint.validate_gemma_checkpoint_ancestry(
        checkpoint, **expected, verify_parent_snapshot=False)
    substituted = dict(expected, preflight_certificate_sha256="b" * 64)
    with pytest.raises(
            checkpoint_fingerprint.CheckpointValidationError, match="ancestry differs"):
        checkpoint_fingerprint.validate_gemma_checkpoint_ancestry(
            checkpoint, **substituted, verify_parent_snapshot=False)


def test_final_checkpoint_ancestry_uses_the_same_shared_validator(tmp_path):
    checkpoint, _, expected = _ancestry_fixture(tmp_path)
    final = tmp_path / "final"
    checkpoint.rename(final)
    checkpoint_fingerprint.validate_checkpoint_manifest(
        final, required=True, verify_state_identity=True)
    checkpoint_fingerprint.validate_gemma_checkpoint_ancestry(
        final, **expected, verify_parent_snapshot=False)


def test_resume_substitution_and_snapshot_mutation_fail_closed(monkeypatch, tmp_path):
    checkpoint, snapshot, expected = _ancestry_fixture(tmp_path)
    wrong_parent = tmp_path / "other-parent"
    wrong_parent.mkdir()
    with pytest.raises(
            checkpoint_fingerprint.CheckpointValidationError, match="model ancestry differs"):
        checkpoint_fingerprint.validate_gemma_checkpoint_ancestry(
            checkpoint, **dict(expected, parent_snapshot=str(wrong_parent)),
            verify_parent_snapshot=False)
    with open(snapshot / "tokenizer.model", "ab") as handle:
        handle.write(b"mutation")
    monkeypatch.setattr(
        checkpoint_fingerprint, "verify_authoritative_snapshot", lambda _path: {})
    with pytest.raises(
            checkpoint_fingerprint.CheckpointValidationError, match="snapshot changed"):
        checkpoint_fingerprint.validate_gemma_checkpoint_ancestry(checkpoint, **expected)


def test_gemma_parent_role_rejects_substituted_path(monkeypatch, tmp_path):
    _, snapshot, expected = _ancestry_fixture(tmp_path)
    monkeypatch.setattr(
        checkpoint_fingerprint, "verify_authoritative_snapshot", lambda _path: {})
    checkpoint_fingerprint.validate_gemma_parent_snapshot(snapshot, **expected)
    other = tmp_path / "other-parent"
    other.mkdir()
    with pytest.raises(
            checkpoint_fingerprint.CheckpointValidationError,
            match="not the certificate snapshot"):
        checkpoint_fingerprint.validate_gemma_parent_snapshot(other, **expected)


def test_inference_fingerprint_excludes_mutable_training_outputs(tmp_path):
    for name in ("training_state.pt", "run_provenance.json", "metrics.json", "notes.txt"):
        assert checkpoint_fingerprint.included(tmp_path / name) is False
    for name in ("config.json", "tokenizer.model", "model.safetensors"):
        assert checkpoint_fingerprint.included(tmp_path / name) is True


def test_probe_uses_exact_snapshot_for_config_fast_and_slow_loaders(monkeypatch, tmp_path):
    _, snapshot = _canonical_gemma_snapshot(tmp_path)
    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt":"p","response":"r"}\n', encoding="utf-8")
    calls = []

    class ProbeTokenizer:
        is_fast = True
        vocab_size = 256_000
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2
        unk_token_id = 3

        def __init__(self):
            self.length = 256_000
            self.vocab_file = str(snapshot / "tokenizer.model")

        def __len__(self):
            return self.length

    tokenizer = ProbeTokenizer()

    class ConfigLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("config", source, kwargs))
            return SimpleNamespace(model_type="gemma2", vocab_size=256_000)

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("fast", source, kwargs))
            return tokenizer

    monkeypatch.setattr(probe, "AutoConfig", ConfigLoader)
    monkeypatch.setattr(probe, "AutoTokenizer", TokenizerLoader)
    monkeypatch.setattr(probe, "require_gemma_protobuf_runtime", lambda: {})
    monkeypatch.setattr(probe, "verify_authoritative_snapshot", lambda _path: {})
    monkeypatch.setattr(probe, "snapshot_inventory_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(
        probe, "authoritative_revision_files_sha256", lambda: "e" * 64)
    monkeypatch.setattr(
        probe, "guarded_snapshot_load", lambda load, *_args, **_kwargs: load())
    monkeypatch.setattr(
        probe, "validate_gemma_model_load_path",
        lambda model, revision, source: str(snapshot.resolve()))
    monkeypatch.setattr(
        probe, "validate_gemma_base_tokenizer_contract",
        lambda tok, vocab: {"vocab_size": vocab, "length": len(tok)})

    def register(tok, **_kwargs):
        tok.length = 256_002
        return {"added": 2, "ids": {"[END_CELL]": 256_000, "[DOWN]": 256_001}}

    monkeypatch.setattr(probe, "register_sentinels", register)
    monkeypatch.setattr(probe, "validate_gemma_registered_tokenizer_contract", lambda *_: {})
    monkeypatch.setattr(probe, "tokenizer_fingerprint", lambda _tok: "tokenizer-fingerprint")
    monkeypatch.setattr(
        probe, "audit_file",
        lambda path, *_args, **_kwargs: {
            "path": str(data.resolve()), "sha256": "data-sha",
            "counts": {"prompt_unk_tokens": 0, "response_unk_tokens": 0},
            "distributions": {
                "prompt_tokens": {"min": 395}, "response_tokens": {"min": 481}},
        })

    def parity(source, kwargs, _tokenizer, _rows):
        calls.append(("slow", source, kwargs))
        return {"available": True, "mismatch_count": 0, "rows_checked": 1,
                "slow_vocab_file": str(snapshot / "tokenizer.model")}

    monkeypatch.setattr(probe, "check_slow_fast_parity", parity)
    args = SimpleNamespace(
        parity_examples=1, max_length=8192, max_examples=0, batch_size=1,
        require_fast=True, load_model=False, attn_implementation="eager")
    result = probe.audit_model(
        trainer.GEMMA_MODEL_ID, trainer.GEMMA_MODEL_REVISION, str(snapshot),
        [str(data)], {str(data): "data-sha"}, args)

    assert [entry[0] for entry in calls] == ["config", "fast", "slow"]
    assert all(entry[1] == str(snapshot.resolve()) for entry in calls)
    assert calls[0][2] == {"local_files_only": True}
    assert calls[1][2] == {"local_files_only": True, "use_fast": True}
    assert calls[2][2] == {"local_files_only": True}
    assert result["model_name"] == trainer.GEMMA_MODEL_ID
    assert result["revision"] == trainer.GEMMA_MODEL_REVISION
    assert result["load_source"]["path"] == str(snapshot.resolve())
    assert result["snapshot_load_guard"]["before_and_after_each_load"] is True


def test_missing_protobuf_fails_before_gemma_tokenizer_load(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("planted missing protobuf")

    monkeypatch.setattr(trainer.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="refusing any tiktoken fallback"):
        trainer.require_gemma_protobuf_runtime()


def test_wrong_protobuf_version_fails_before_gemma_tokenizer_load(monkeypatch):
    monkeypatch.setattr(
        trainer.importlib, "import_module", lambda _name: SimpleNamespace(__version__="6.0.0"))
    with pytest.raises(RuntimeError, match="requires Python protobuf 5.29.5"):
        trainer.require_gemma_protobuf_runtime()


def _protobuf_distribution_fixture(root):
    package = root / "google" / "protobuf"
    upb = root / "google" / "_upb"
    metadata = root / "protobuf-5.29.5.dist-info"
    package.mkdir(parents=True)
    upb.mkdir(parents=True)
    metadata.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "5.29.5"\n', encoding="utf-8")
    (upb / "_message.abi3.so").write_bytes(b"planted compiled payload\n")
    (metadata / "METADATA").write_text("Version: 5.29.5\n", encoding="utf-8")
    record = metadata / "RECORD"
    record.write_text(
        "google/protobuf/__init__.py,,\n"
        "google/_upb/_message.abi3.so,,\n"
        "protobuf-5.29.5.dist-info/METADATA,,\n"
        "protobuf-5.29.5.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    return package, upb, metadata, record


def test_protobuf_record_digest_binds_compiled_upb_payload(tmp_path):
    root = tmp_path / "protobuf"
    _, upb, _, _ = _protobuf_distribution_fixture(root)
    before, count = preflight_contract.protobuf_tree_digest(root)
    assert count == 4
    assert trainer.protobuf_tree_digest(root) == before
    (upb / "_message.abi3.so").write_bytes(b"mutated compiled payload\n")
    after, _ = preflight_contract.protobuf_tree_digest(root)
    assert before != after
    assert trainer.protobuf_tree_digest(root) == after


def test_protobuf_record_digest_rejects_missing_record(tmp_path):
    root = tmp_path / "protobuf"
    _, _, _, record = _protobuf_distribution_fixture(root)
    record.unlink()
    with pytest.raises(SystemExit, match="RECORD is missing"):
        preflight_contract.protobuf_tree_digest(root)


def test_protobuf_record_digest_rejects_path_traversal(tmp_path):
    root = tmp_path / "protobuf"
    _, _, _, record = _protobuf_distribution_fixture(root)
    with record.open("a", encoding="utf-8") as handle:
        handle.write("../outside-payload,,\n")
    with pytest.raises(SystemExit, match="unsafe protobuf RECORD path"):
        preflight_contract.protobuf_tree_digest(root)


def test_helper_bound_protobuf_rejects_wrong_payload(monkeypatch, tmp_path):
    root = tmp_path / "protobuf"
    package, _, _, _ = _protobuf_distribution_fixture(root)
    module_file = package / "__init__.py"
    module = SimpleNamespace(__version__="5.29.5", __file__=str(module_file))
    monkeypatch.setattr(trainer.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(trainer, "GEMMA_PROTOBUF_ROOT", root)
    monkeypatch.setenv("PYTHONPATH", str(root.resolve()))
    monkeypatch.setenv("GEMMA_PROTOBUF_TREE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="content digest changed"):
        trainer.require_gemma_protobuf_runtime()


def test_gemma_runtime_accepts_literal_alias_pythonpath(monkeypatch, tmp_path):
    root = tmp_path / "protobuf"
    package, _, _, _ = _protobuf_distribution_fixture(root)
    module = SimpleNamespace(
        __version__="5.29.5", __file__=str(package / "__init__.py"))
    lexical_alias = root / "nonexistent-alias-component" / ".."
    assert str(lexical_alias) != str(lexical_alias.resolve())
    assert lexical_alias.resolve() == root.resolve()
    monkeypatch.setattr(trainer.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(trainer, "GEMMA_PROTOBUF_ROOT", lexical_alias)
    monkeypatch.setenv("PYTHONPATH", str(lexical_alias))
    monkeypatch.setenv("GEMMA_PROTOBUF_TREE_SHA256", trainer.protobuf_tree_digest(root))
    monkeypatch.delenv("PYTHONHOME", raising=False)

    runtime = trainer.require_gemma_protobuf_runtime()

    assert runtime["tree_sha256"] == trainer.protobuf_tree_digest(root)


def test_gemma_runtime_rejects_different_pythonpath(monkeypatch, tmp_path):
    root = tmp_path / "protobuf"
    package, _, _, _ = _protobuf_distribution_fixture(root)
    module = SimpleNamespace(
        __version__="5.29.5", __file__=str(package / "__init__.py"))
    monkeypatch.setattr(trainer.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(trainer, "GEMMA_PROTOBUF_ROOT", root)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "different-protobuf"))
    monkeypatch.setenv("GEMMA_PROTOBUF_TREE_SHA256", trainer.protobuf_tree_digest(root))
    monkeypatch.delenv("PYTHONHOME", raising=False)

    with pytest.raises(RuntimeError, match=r"observed=.*expected="):
        trainer.require_gemma_protobuf_runtime()


def test_gemma_runtime_rejects_inherited_pythonhome(monkeypatch, tmp_path):
    root = tmp_path / "protobuf"
    package, _, _, _ = _protobuf_distribution_fixture(root)
    module = SimpleNamespace(
        __version__="5.29.5", __file__=str(package / "__init__.py"))
    monkeypatch.setattr(trainer.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(trainer, "GEMMA_PROTOBUF_ROOT", root)
    monkeypatch.setenv("PYTHONPATH", str(root.resolve()))
    monkeypatch.setenv("GEMMA_PROTOBUF_TREE_SHA256", trainer.protobuf_tree_digest(root))
    monkeypatch.setenv("PYTHONHOME", "/planted/shadow/runtime")
    with pytest.raises(RuntimeError, match="PYTHONHOME to be unset"):
        trainer.require_gemma_protobuf_runtime()


def _valid_probe_document(tmp_path):
    snapshot = (tmp_path / "hub" /
                "models--vandijklab--C2S-Scale-Gemma-2-2B" / "snapshots" /
                probe.GEMMA_REVISION).resolve()
    snapshot.mkdir(parents=True)
    reports = []
    expected_data = {}
    input_hashes = {}
    for index, name in enumerate(("train", "tier1", "tier2", "tier3", "tier4")):
        path = (tmp_path / f"{name}.jsonl").resolve()
        path.write_text(f'{{"fixture": {index}}}\n', encoding="utf-8")
        digest = preflight_contract.sha256_file(path)
        expected_data[name] = {"path": str(path), "sha256": digest}
        input_hashes[str(path)] = digest
        reports.append({
            "path": str(path),
            "sha256": digest,
            "counts": {
                "rows": 1,
                "semantic_truncations": 0,
                "prompt_truncations": 0,
                "sentinel_losses": 0,
                "response_missing_end_cell": 0,
                "response_contains_down": 0,
                "prompt_unk_tokens": 0,
                "response_unk_tokens": 0,
                "prompt_token_count": 395,
                "response_token_count": 481,
            },
            "distributions": {
                "prompt_tokens": {"min": 395, "max": 395},
                "response_tokens": {"min": 481, "max": 481},
            },
        })
    document = {
        "schema_version": 3,
        "max_examples_per_file": 0,
        "input_hashes": input_hashes,
        "models": [{
            "model_name": "vandijklab/C2S-Scale-Gemma-2-2B",
            "revision": probe.GEMMA_REVISION,
            "load_source": {
                "kind": "exact_local_snapshot",
                "path": str(snapshot),
                "local_files_only": True,
                "revision_argument": None,
            },
            "parent_snapshot_inventory_sha256": "b" * 64,
            "authoritative_revision_files_sha256": (
                shared_provenance.authoritative_revision_files_sha256()),
            "snapshot_load_guard": {
                "before_and_after_each_load": True,
                "operations": ["config", "fast_tokenizer", "slow_tokenizer"],
            },
            "model_config_vocab_size": 256_000,
            "base_tokenizer_contract": {"vocab_size": 256_000, "length": 256_000},
            "tokenizer": {
                "is_fast": True,
                "vocab_size": 256_000,
                "length_after_sentinels": 256_002,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "bos_token_id": 2,
                "vocab_file": str(snapshot / "tokenizer.model"),
                "sentinels": {"[END_CELL]": 256_000, "[DOWN]": 256_001},
            },
            "slow_fast_parity": {
                "available": True, "mismatch_count": 0, "rows_checked": 512,
                "slow_vocab_file": str(snapshot / "tokenizer.model")},
            "files": reports,
        }],
    }
    return document, expected_data


def _write_probe(tmp_path, document):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _probe_snapshot(document):
    return document["models"][0]["load_source"]["path"]


def _validate_probe(tmp_path, document, expected_data):
    model = document["models"][0]
    return preflight_contract.validate_probe(
        _write_probe(tmp_path, document), 1600,
        "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION,
        _probe_snapshot(document), model["parent_snapshot_inventory_sha256"],
        model["authoritative_revision_files_sha256"], expected_data)


def test_preflight_contract_accepts_only_complete_gemma_vocabulary(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    result = _validate_probe(tmp_path, document, expected_data)
    assert result["generation_cap_authorized"] is True
    assert result["maximum_truth_response_tokens"] == 481


@pytest.mark.parametrize("mutation,match", [
    ("load_path", "exact certified snapshot"),
    ("fast_vocab_file", "fast Gemma tokenizer did not open"),
    ("slow_vocab_file", "parity was not proved"),
    ("logical_model", "different model or revision"),
])
def test_preflight_contract_binds_logical_identity_to_exact_snapshot(
        tmp_path, mutation, match):
    document, expected_data = _valid_probe_document(tmp_path)
    expected_snapshot = _probe_snapshot(document)
    if mutation == "load_path":
        document["models"][0]["load_source"]["path"] = str(
            (tmp_path / "arbitrary-snapshot").resolve())
    elif mutation == "fast_vocab_file":
        document["models"][0]["tokenizer"]["vocab_file"] = None
    elif mutation == "slow_vocab_file":
        document["models"][0]["slow_fast_parity"]["slow_vocab_file"] = None
    else:
        document["models"][0]["model_name"] = "attacker/substituted-model"
    with pytest.raises(SystemExit, match=match):
        model = document["models"][0]
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION,
            expected_snapshot, model["parent_snapshot_inventory_sha256"],
            model["authoritative_revision_files_sha256"], expected_data)


def test_preflight_contract_rejects_five_token_probe(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    model = document["models"][0]
    model["base_tokenizer_contract"] = {"vocab_size": 5, "length": 5}
    model["tokenizer"]["vocab_size"] = 5
    model["tokenizer"]["length_after_sentinels"] = 7
    with pytest.raises(SystemExit, match="base tokenizer is collapsed"):
        _validate_probe(tmp_path, document, expected_data)


def test_preflight_contract_rejects_model_config_vocabulary_mismatch(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["model_config_vocab_size"] = 255_999
    with pytest.raises(SystemExit, match="model config vocab_size"):
        _validate_probe(tmp_path, document, expected_data)


def test_preflight_contract_rejects_unknown_token_collapse(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["files"][2]["counts"]["prompt_unk_tokens"] = 1
    with pytest.raises(SystemExit, match="contains unknown tokens"):
        _validate_probe(tmp_path, document, expected_data)


def test_preflight_contract_rejects_negative_unknown_counter(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["files"][0]["counts"]["prompt_unk_tokens"] = -1
    with pytest.raises(SystemExit, match="invalid prompt_unk_tokens"):
        _validate_probe(tmp_path, document, expected_data)


def test_preflight_contract_rejects_opposing_unknown_counters(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    counts = document["models"][0]["files"][0]["counts"]
    counts["prompt_unk_tokens"] = 1
    counts["response_unk_tokens"] = -1
    with pytest.raises(SystemExit, match="prompt_unk_tokens=1"):
        _validate_probe(tmp_path, document, expected_data)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicated or incomplete"),
    ("missing_counter", "omitted counters"),
    ("missing_input_hash", "input_hashes do not match"),
    ("zero_parity", "parity was not proved"),
    ("slow_tokenizer", "required fast tokenizer"),
])
def test_preflight_contract_rejects_incomplete_probe_binding(tmp_path, mutation, match):
    document, expected_data = _valid_probe_document(tmp_path)
    model = document["models"][0]
    if mutation == "duplicate":
        model["files"][-1] = copy.deepcopy(model["files"][0])
    elif mutation == "missing_counter":
        del model["files"][0]["counts"]["prompt_unk_tokens"]
    elif mutation == "missing_input_hash":
        document["input_hashes"].pop(next(iter(document["input_hashes"])))
    elif mutation == "zero_parity":
        model["slow_fast_parity"]["rows_checked"] = 0
    elif mutation == "slow_tokenizer":
        model["tokenizer"]["is_fast"] = False
    with pytest.raises(SystemExit, match=match):
        _validate_probe(tmp_path, document, expected_data)


@pytest.mark.parametrize("field,value,match", [
    ("length_after_sentinels", 256_001, "tokenizer contract"),
    ("sentinels", {"[END_CELL]": 256_001, "[DOWN]": 256_000}, "sentinel ids"),
])
def test_preflight_contract_rejects_wrong_sentinel_contract(tmp_path, field, value, match):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["tokenizer"][field] = value
    with pytest.raises(SystemExit, match=match):
        _validate_probe(tmp_path, document, expected_data)


@pytest.mark.parametrize("schema", [1, 2, 3, 4, 5, 6, 7])
def test_preflight_verify_rejects_older_schema_before_trusting_contents(tmp_path, schema):
    certificate = tmp_path / "certificate.json"
    certificate.write_text(
        json.dumps({"schema_version": schema, "preflight_passed": True}), encoding="utf-8")
    args = SimpleNamespace(
        certificate=str(certificate),
        model_id="vandijklab/C2S-Scale-Gemma-2-2B",
        revision=probe.GEMMA_REVISION,
        require_generation_cap=None,
        print_snapshot=False,
        verify_current_environment=False,
    )
    with pytest.raises(SystemExit, match="invalid or unsuccessful"):
        preflight_contract.verify(args)


def test_canonical_tahoe_hashes_are_exactly_pinned():
    assert shared_provenance.CANONICAL_TAHOE_DATA_SHA256 == {
        "train": "4bed186da4c5348dbb899182f2ab0f635c0f73ce7994be8738a23f8e6c61d000",
        "tier1": "bb2d8f45c32b7f64f0e874c3e29d3e4e086194c3cc18d97fea5016c494f8bb21",
        "tier2": "054dc5370103fb9da381046b13cc7235b556bb39d7ffced4976fd45768727c39",
        "tier3": "29c8cb31e82ce39d988983457ebea34b78a15369b5dbd69cb2db76185ca1d2d8",
        "tier4": "0015f09ce47bc45a574ef7d9dbd54e8342929d56fc965b50869a61fb75722d75",
    }


def test_fresh_certificate_cannot_bless_mutated_tahoe_data(monkeypatch, tmp_path):
    entries = {
        name: {"path": str(tmp_path / f"{name}.jsonl"), "sha256": digest}
        for name, digest in shared_provenance.CANONICAL_TAHOE_DATA_SHA256.items()
    }
    mutated = tmp_path / "tier2.jsonl"
    mutated.write_text('{"mutated": true}\n', encoding="utf-8")
    entries["tier2"]["sha256"] = preflight_contract.sha256_file(mutated)
    monkeypatch.setattr(preflight_contract, "certified_snapshot", lambda *_args: {})
    monkeypatch.setattr(preflight_contract, "parse_named_paths", lambda _items: entries)
    args = SimpleNamespace(
        model_id=trainer.GEMMA_MODEL_ID,
        revision=trainer.GEMMA_MODEL_REVISION,
        snapshot_path=str(tmp_path / "snapshot"),
        hub_cache=str(tmp_path / "hub"),
        data=["not-used"],
    )
    with pytest.raises(SystemExit, match="new certificate cannot bless different data"):
        preflight_contract.create(args)


def test_runtime_contract_binds_exact_certificate_bytes_and_snapshot_digest(tmp_path):
    certificate = tmp_path / "PREFLIGHT_PASSED.json"
    certificate.write_text('{"schema_version":8}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = preflight_contract.runtime_contract_document(
        certificate, model_id=trainer.GEMMA_MODEL_ID,
        revision=trainer.GEMMA_MODEL_REVISION, snapshot_path=snapshot,
        snapshot_inventory_sha256="b" * 64,
        authoritative_revision_files_sha256="c" * 64)
    assert first["schema_version"] == 2
    assert first["preflight_certificate_sha256"] == preflight_contract.sha256_file(certificate)
    assert first["snapshot_inventory_sha256"] == "b" * 64
    certificate.write_text('{"schema_version":8,"substituted":true}\n', encoding="utf-8")
    second = preflight_contract.runtime_contract_document(
        certificate, model_id=trainer.GEMMA_MODEL_ID,
        revision=trainer.GEMMA_MODEL_REVISION, snapshot_path=snapshot,
        snapshot_inventory_sha256="b" * 64,
        authoritative_revision_files_sha256="c" * 64)
    assert second["preflight_certificate_sha256"] != first["preflight_certificate_sha256"]


@pytest.mark.parametrize("mutation", ["omit", "extra"])
def test_tests_certificate_rejects_nonexact_source_set(mutation):
    sources = {name: {} for name in tests_contract.EXPECTED_SOURCE_KEYS}
    if mutation == "omit":
        sources.pop("protobuf_env")
    else:
        sources["unexpected"] = {}
    with pytest.raises(SystemExit, match="source keys differ from contract"):
        tests_contract.require_exact_source_keys(sources)


def test_jobs_make_model_id_tokenizer_route_impossible():
    preflight = open(
        os.path.join(ROOT, "endcell", "jobs", "gemma2_standard_preflight.sh"),
        encoding="utf-8").read()
    smoke = open(
        os.path.join(ROOT, "endcell", "jobs", "gemma2_standard_smoke.sbatch"),
        encoding="utf-8").read()
    train_job = open(
        os.path.join(ROOT, "endcell", "jobs", "gemma2_standard_train.sbatch"),
        encoding="utf-8").read()
    assert '--load-source "$MODEL_ID=$SNAPSHOT_PATH"' in preflight
    assert "AutoConfig.from_pretrained(snapshot_path, local_files_only=True)" in preflight
    for job in (smoke, train_job):
        assert "--runtime-contract-out" in job
        assert '--model_load_path "$SNAPSHOT_PATH"' in job
        assert '--model_name "$MODEL"' in job
    assert "verify_parent_load_provenance" not in train_job
    assert "--require_gemma_ancestry" in train_job
    trainer_source = open(
        os.path.join(ROOT, "endcell", "train", "train_c2s_tahoe_endcell.py"),
        encoding="utf-8").read()
    for stage in ("trainer AutoConfig load", "trainer AutoTokenizer load",
                  "trainer AutoModelForCausalLM load"):
        assert stage in trainer_source


def test_eval_role_fixes_label_and_model_family_without_losing_pythia_comparator():
    source = open(
        os.path.join(ROOT, "endcell", "jobs", "gemma2_standard_eval.sbatch"),
        encoding="utf-8").read()
    expected = {
        "gemma_sft": ("gemma", "gemma2"),
        "gemma_parent": ("gemma_parent", "gemma2"),
        "pythia_sft_legacy": ("pythia", "gpt_neox"),
        "pythia_parent": ("pythia_parent", "gpt_neox"),
    }
    for role, (label, model_type) in expected.items():
        assert (f"{role}) EXPECTED_MODEL_LABEL={label}; "
                f"EXPECTED_MODEL_TYPE={model_type}") in source
    assert 'MODEL_LABEL="$EXPECTED_MODEL_LABEL"' in source
    assert "requires MODEL_LABEL=$EXPECTED_MODEL_LABEL" in source
    assert 'document.get("model_type")' in source
    assert 'VALIDITY="$REPO/endcell/eval/evaluate_endcell.py"' in source
    assert 'NIR="$REPO/endcell/analysis/nir_benchmark.py"' in source
    assert "VALIDITY_ENTRYPOINT" not in source
    assert "NIR_ENTRYPOINT" not in source
    assert "fixed production evaluator entrypoints match preflight path and SHA-256" in source

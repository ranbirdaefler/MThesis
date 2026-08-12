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
        "schema_version": 1,
        "max_examples_per_file": 0,
        "input_hashes": input_hashes,
        "models": [{
            "model_name": "vandijklab/C2S-Scale-Gemma-2-2B",
            "revision": probe.GEMMA_REVISION,
            "model_config_vocab_size": 256_000,
            "base_tokenizer_contract": {"vocab_size": 256_000, "length": 256_000},
            "tokenizer": {
                "is_fast": True,
                "vocab_size": 256_000,
                "length_after_sentinels": 256_002,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "bos_token_id": 2,
                "sentinels": {"[END_CELL]": 256_000, "[DOWN]": 256_001},
            },
            "slow_fast_parity": {
                "available": True, "mismatch_count": 0, "rows_checked": 512},
            "files": reports,
        }],
    }
    return document, expected_data


def _write_probe(tmp_path, document):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_preflight_contract_accepts_only_complete_gemma_vocabulary(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    path = _write_probe(tmp_path, document)
    result = preflight_contract.validate_probe(
        path, 1600, "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION,
        expected_data)
    assert result["generation_cap_authorized"] is True
    assert result["maximum_truth_response_tokens"] == 481


def test_preflight_contract_rejects_five_token_probe(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    model = document["models"][0]
    model["base_tokenizer_contract"] = {"vocab_size": 5, "length": 5}
    model["tokenizer"]["vocab_size"] = 5
    model["tokenizer"]["length_after_sentinels"] = 7
    with pytest.raises(SystemExit, match="base tokenizer is collapsed"):
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION, expected_data)


def test_preflight_contract_rejects_model_config_vocabulary_mismatch(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["model_config_vocab_size"] = 255_999
    with pytest.raises(SystemExit, match="model config vocab_size"):
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION, expected_data)


def test_preflight_contract_rejects_unknown_token_collapse(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["files"][2]["counts"]["prompt_unk_tokens"] = 1
    with pytest.raises(SystemExit, match="contains unknown tokens"):
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION, expected_data)


def test_preflight_contract_rejects_negative_unknown_counter(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["files"][0]["counts"]["prompt_unk_tokens"] = -1
    with pytest.raises(SystemExit, match="invalid prompt_unk_tokens"):
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION, expected_data)


def test_preflight_contract_rejects_opposing_unknown_counters(tmp_path):
    document, expected_data = _valid_probe_document(tmp_path)
    counts = document["models"][0]["files"][0]["counts"]
    counts["prompt_unk_tokens"] = 1
    counts["response_unk_tokens"] = -1
    with pytest.raises(SystemExit, match="prompt_unk_tokens=1"):
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION, expected_data)


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
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION,
            expected_data)


@pytest.mark.parametrize("field,value,match", [
    ("length_after_sentinels", 256_001, "tokenizer contract"),
    ("sentinels", {"[END_CELL]": 256_001, "[DOWN]": 256_000}, "sentinel ids"),
])
def test_preflight_contract_rejects_wrong_sentinel_contract(tmp_path, field, value, match):
    document, expected_data = _valid_probe_document(tmp_path)
    document["models"][0]["tokenizer"][field] = value
    with pytest.raises(SystemExit, match=match):
        preflight_contract.validate_probe(
            _write_probe(tmp_path, document), 1600,
            "vandijklab/C2S-Scale-Gemma-2-2B", probe.GEMMA_REVISION,
            expected_data)


@pytest.mark.parametrize("schema", [2, 3])
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


@pytest.mark.parametrize("mutation", ["omit", "extra"])
def test_tests_certificate_rejects_nonexact_source_set(mutation):
    sources = {name: {} for name in tests_contract.EXPECTED_SOURCE_KEYS}
    if mutation == "omit":
        sources.pop("protobuf_env")
    else:
        sources["unexpected"] = {}
    with pytest.raises(SystemExit, match="source keys differ from contract"):
        tests_contract.require_exact_source_keys(sources)

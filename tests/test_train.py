from __future__ import annotations

import os
from pathlib import Path

import torch
import transformers
from transformers.trainer_pt_utils import LengthGroupedSampler

from genterp.runtime import TorchRuntime
from genterp.train import (
    EVAL_STEPS,
    GenterpForCausalLM,
    GenterpHFConfig,
    GenterpTrainer,
    LOGGING_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    _limit_eval_worker,
    build_training_args,
    checkpoint_is_complete,
    checkpoint_matches_runtime,
    checkpoint_n_atoms,
    checkpoint_runtime_state,
    final_model_path,
    gradient_checkpointing_enabled,
    latest_checkpoint,
    model_dir_is_complete,
    save_final_model,
    write_runtime_state,
)


class _LengthDataset:
    lengths = [8, 2, 4]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict:
        return {"event_atoms": [1] * self.lengths[idx]}


class _SaveModelTrainer:
    state = type("State", (), {"global_step": 7})()

    def save_model(self, path: str) -> None:
        _write_model_dir(Path(path))


def _runtime(*, bf16: bool = True, fp16: bool = False, optim: str = "adamw_torch_fused") -> TorchRuntime:
    return TorchRuntime(
        device=torch.device("cuda", 0),
        cuda_device_count=1,
        cuda_name="accelerator",
        cuda_capability=(8, 0),
        per_device_train_batch_size=4,
        bf16=bf16,
        fp16=fp16,
        tf32=True,
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="reduce-overhead",
        optim=optim,
        use_data_parallel=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        auto_find_batch_size=True,
    )


def _write_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


def _write_complete_checkpoint(path: Path, runtime: TorchRuntime | None = None) -> None:
    _write_model_dir(path)
    (path / "trainer_state.json").write_text("{}")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    write_runtime_state(path, runtime or _runtime())


def test_latest_checkpoint_empty_dir(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_selects_highest_complete_step(tmp_path: Path):
    _write_complete_checkpoint(tmp_path / "checkpoint-2")
    _write_complete_checkpoint(tmp_path / "checkpoint-10")

    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-10")


def test_latest_checkpoint_ignores_partial_higher_step(tmp_path: Path):
    _write_complete_checkpoint(tmp_path / "checkpoint-2")
    partial = tmp_path / "checkpoint-10"
    _write_model_dir(partial)
    (partial / "trainer_state.json").write_text("{}")

    assert not checkpoint_is_complete(partial)
    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-2")


def test_final_model_path_requires_saved_config(tmp_path: Path):
    assert final_model_path(tmp_path) is None

    final = tmp_path / "final"
    final.mkdir()
    assert final_model_path(tmp_path) is None

    (final / "config.json").write_text("{}")
    assert final_model_path(tmp_path) is None

    (final / "model.safetensors").write_bytes(b"weights")
    assert model_dir_is_complete(final)
    assert final_model_path(tmp_path) == str(final)


def test_final_model_path_uses_atomic_pointer(tmp_path: Path):
    old_final = tmp_path / "final-1"
    new_final = tmp_path / "final-2"
    _write_model_dir(old_final)
    _write_model_dir(new_final)
    (tmp_path / "final_checkpoint.json").write_text('{"path": "final-2"}')

    assert final_model_path(tmp_path) == str(new_final)


def test_save_final_model_writes_versioned_final_and_pointer(tmp_path: Path):
    save_final_model(_SaveModelTrainer(), tmp_path, _runtime())

    final_path = final_model_path(tmp_path)

    assert final_path is not None
    assert Path(final_path).name.startswith("final-7-")
    assert checkpoint_runtime_state(final_path) is not None


def test_runtime_state_round_trips(tmp_path: Path):
    runtime = _runtime()
    write_runtime_state(tmp_path, runtime)

    assert checkpoint_runtime_state(tmp_path) is not None
    assert checkpoint_matches_runtime(tmp_path, runtime)


def test_runtime_state_detects_hardware_profile_change(tmp_path: Path):
    write_runtime_state(tmp_path, _runtime(bf16=True, fp16=False))

    assert not checkpoint_matches_runtime(tmp_path, _runtime(bf16=False, fp16=True))
    assert not checkpoint_matches_runtime(tmp_path, _runtime(bf16=False, fp16=False, optim="adamw_torch"))


def test_training_args_use_wsd_and_cuda_gradient_checkpointing(tmp_path: Path):
    runtime = _runtime()
    args = build_training_args(tmp_path, runtime)

    assert gradient_checkpointing_enabled(runtime)
    from genterp.train import MAX_STEPS, WSD_DECAY_STEPS
    assert args["lr_scheduler_type"] == "warmup_stable_decay"
    assert args["warmup_steps"] == 500
    assert args["max_steps"] == MAX_STEPS
    assert args["lr_scheduler_kwargs"] == {
        "num_decay_steps": WSD_DECAY_STEPS,
        "decay_type": "linear",
        "min_lr_ratio": 0.0,
    }
    assert args["logging_steps"] == LOGGING_STEPS
    assert args["eval_steps"] == EVAL_STEPS
    assert args["save_steps"] == SAVE_STEPS
    assert args["save_total_limit"] == SAVE_TOTAL_LIMIT
    assert args["gradient_checkpointing"] is True
    assert args["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_trainer_uses_dataset_lengths_for_grouped_sampling(tmp_path: Path):
    dataset = _LengthDataset()
    trainer = GenterpTrainer(
        model=torch.nn.Linear(1, 1),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            train_sampling_strategy="group_by_length",
        ),
        train_dataset=dataset,
    )

    sampler = trainer._get_train_sampler()

    assert isinstance(sampler, LengthGroupedSampler)
    assert sampler.lengths == dataset.lengths


def test_trainer_skips_incompatible_optimizer_and_scaler_state(tmp_path: Path, monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError("incompatible training state should not load")

    monkeypatch.setattr(transformers.Trainer, "_load_optimizer_and_scheduler", fail_load)
    monkeypatch.setattr(transformers.Trainer, "_load_scaler", fail_load)

    trainer = GenterpTrainer(
        model=torch.nn.Linear(1, 1),
        args=transformers.TrainingArguments(output_dir=str(tmp_path)),
        reset_training_state_on_resume=True,
    )

    trainer._load_optimizer_and_scheduler(str(tmp_path))
    trainer._load_scaler(str(tmp_path))


def _tiny_genterp_model(n_atoms: int) -> GenterpForCausalLM:
    cfg_dict = {
        "n_atoms": n_atoms,
        "dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "n_static_blocks": 1,
        "k_static_summary": 2,
        "n_time_mix": 2,
        "time_phi_dim": 4,
    }
    return GenterpForCausalLM(GenterpHFConfig(genterp_cfg=cfg_dict))


def test_load_state_dict_partial_prefix_copy_on_vocab_growth(capsys):
    """Vocab-grown checkpoint loads into a larger model without raising; the
    vocab-shaped params get a partial-prefix copy (existing atoms keep their
    trained vectors, new atoms keep their fresh init) and the rest of the
    weights copy in cleanly.

    Critical for the death + demographics ETL change: those add new atoms to
    the vocab, but the existing 27k atoms' trained embeddings must survive
    the resume. The prior behavior dropped ALL embedding rows on any shape
    mismatch — destroying every step of training. The partial-prefix path
    preserves the prefix and only the new atoms reset.
    """
    n_old, n_new = 4, 10
    old = _tiny_genterp_model(n_atoms=n_old)
    new = _tiny_genterp_model(n_atoms=n_new)

    norm_key = "model.norm.weight"
    emb_key = "model.embed.embedding.weight"
    assert norm_key in new.state_dict() and emb_key in new.state_dict()

    # Set known values so we can verify per-row preservation.
    with torch.no_grad():
        old.state_dict()[norm_key].fill_(0.42)
        # Sentinel: row 0..n_old-1 set to row-index values so we can verify
        # they survive the partial copy.
        for i in range(n_old):
            old.state_dict()[emb_key][i].fill_(float(i) + 1.0)

    snapshot_new_tail = new.state_dict()[emb_key][n_old:].clone()
    result = new.load_state_dict(old.state_dict(), strict=True)

    # Vocab prefix: copied from checkpoint.
    for i in range(n_old):
        assert torch.allclose(
            new.state_dict()[emb_key][i],
            torch.full_like(new.state_dict()[emb_key][i], float(i) + 1.0),
        ), f"row {i} should have been copied from checkpoint"
    # Vocab suffix (new atoms): kept their fresh init.
    assert torch.equal(new.state_dict()[emb_key][n_old:], snapshot_new_tail)
    # Shape-compatible param did load normally.
    assert torch.allclose(
        new.state_dict()[norm_key], torch.full_like(new.state_dict()[norm_key], 0.42)
    )
    # The partial-prefix-copied key is in the filtered set passed to load,
    # so it's NOT in missing_keys. (Only dimensionally-incompatible keys
    # would be reported as missing.)
    assert emb_key not in set(result.missing_keys)
    captured = capsys.readouterr().out
    assert "[warm-start]" in captured
    assert "prefix-copied" in captured


def test_load_state_dict_partial_prefix_copy_on_vocab_shrink(capsys):
    """Shrinking vocab (rare but possible if a CDR refresh removes concepts)
    also takes the prefix-copy path: copy the first min(old, new) rows."""
    n_old, n_new = 10, 4
    old = _tiny_genterp_model(n_atoms=n_old)
    new = _tiny_genterp_model(n_atoms=n_new)

    emb_key = "model.embed.embedding.weight"
    with torch.no_grad():
        for i in range(n_old):
            old.state_dict()[emb_key][i].fill_(float(i) + 1.0)

    new.load_state_dict(old.state_dict(), strict=True)

    # Only the first 4 rows are kept; row 4..9 are gone.
    for i in range(n_new):
        assert torch.allclose(
            new.state_dict()[emb_key][i],
            torch.full_like(new.state_dict()[emb_key][i], float(i) + 1.0),
        )
    captured = capsys.readouterr().out
    assert "prefix-copied" in captured


def test_load_state_dict_preserves_strict_when_shapes_match(tmp_path: Path):
    """Normal resume (same vocab) is a no-op for the filter — strict semantics
    pass through unchanged, so missing/unexpected keys still raise as before."""
    a = _tiny_genterp_model(n_atoms=4)
    b = _tiny_genterp_model(n_atoms=4)

    # Same shapes everywhere — load_state_dict with strict=True must succeed.
    result = b.load_state_dict(a.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_checkpoint_n_atoms_reads_config(tmp_path: Path):
    """checkpoint_n_atoms parses the saved HF config.json correctly so the
    resume path can detect vocab growth without loading the full state dict."""
    import json
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"genterp_cfg": {"n_atoms": 12345}}))
    assert checkpoint_n_atoms(ckpt) == 12345

    bad = tmp_path / "checkpoint-2"
    bad.mkdir()
    (bad / "config.json").write_text("not-json")
    assert checkpoint_n_atoms(bad) is None

    missing = tmp_path / "checkpoint-3"
    missing.mkdir()
    assert checkpoint_n_atoms(missing) is None


def test_checkpoint_ancestor_rows_distinguishes_absent_from_unreadable(tmp_path: Path):
    """checkpoint_ancestor_rows must distinguish three cases:
      - parameter present  → returns row count (>=1)
      - safetensors readable but parameter ABSENT → returns 0 sentinel
      - safetensors not readable at all → returns -1 sentinel

    The ABSENT vs UNREADABLE distinction is what fixes the pre-hierarchical
    checkpoint resume crash: ABSENT means the live model has an
    ancestor_embedding parameter the checkpoint's optimizer state doesn't,
    so optimizer state must be rebuilt; UNREADABLE means we can't tell and
    should be conservative.
    """
    from safetensors.torch import save_file
    from genterp.train import (
        _ANCESTOR_ROWS_ABSENT,
        _ANCESTOR_ROWS_UNREADABLE,
        checkpoint_ancestor_rows,
    )

    # Case 1: parameter present with 5 rows.
    present = tmp_path / "ckpt-present"
    present.mkdir()
    save_file(
        {"model.embed.ancestor_embedding.weight": torch.zeros(5, 8)},
        str(present / "model.safetensors"),
    )
    assert checkpoint_ancestor_rows(present) == 5

    # Case 2: safetensors readable, parameter absent — the
    # pre-hierarchical checkpoint case that triggered the crash.
    absent = tmp_path / "ckpt-absent"
    absent.mkdir()
    save_file(
        {"model.embed.embedding.weight": torch.zeros(10, 8)},  # vocab only
        str(absent / "model.safetensors"),
    )
    assert checkpoint_ancestor_rows(absent) == _ANCESTOR_ROWS_ABSENT

    # Case 3: no safetensors file at all → unreadable.
    unreadable = tmp_path / "ckpt-unreadable"
    unreadable.mkdir()
    assert checkpoint_ancestor_rows(unreadable) == _ANCESTOR_ROWS_UNREADABLE


def test_trainer_keeps_lengths_on_host_for_attention_control_flow(tmp_path: Path):
    runtime = TorchRuntime(
        device=torch.device("meta"),
        cuda_device_count=0,
        cuda_name=None,
        cuda_capability=None,
        per_device_train_batch_size=1,
        bf16=False,
        fp16=False,
        tf32=False,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        optim="adamw_torch",
        use_data_parallel=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=None,
        auto_find_batch_size=False,
    )
    trainer = GenterpTrainer(
        model=torch.nn.Linear(1, 1),
        args=transformers.TrainingArguments(output_dir=str(tmp_path)),
        runtime=runtime,
    )
    batch = {"event_atoms": torch.ones(2, 4), "length": torch.tensor([4, 2])}

    prepared = trainer._prepare_input(batch)

    assert prepared["event_atoms"].device == runtime.device
    assert prepared["length"].device.type == "cpu"


def test_dataloader_worker_thread_limits(monkeypatch):
    set_threads = []
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(torch, "set_num_threads", set_threads.append)

    _limit_eval_worker(0)

    assert set_threads == [1]
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"

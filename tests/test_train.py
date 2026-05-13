from __future__ import annotations

from pathlib import Path

import torch
import transformers
from transformers.trainer_pt_utils import LengthGroupedSampler

from genterp.runtime import TorchRuntime
from genterp.train import (
    GenterpTrainer,
    checkpoint_is_complete,
    checkpoint_matches_runtime,
    checkpoint_runtime_state,
    final_model_path,
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

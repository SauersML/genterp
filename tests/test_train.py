from __future__ import annotations

from pathlib import Path

import torch
import transformers
from transformers.trainer_pt_utils import LengthGroupedSampler

from genterp.train import GenterpTrainer, final_model_path, latest_checkpoint


class _LengthDataset:
    lengths = [8, 2, 4]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict:
        return {"event_atoms": [1] * self.lengths[idx]}


def test_latest_checkpoint_empty_dir(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_selects_highest_step(tmp_path: Path):
    for step in (2, 10):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")

    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-10")


def test_final_model_path_requires_saved_config(tmp_path: Path):
    assert final_model_path(tmp_path) is None

    final = tmp_path / "final"
    final.mkdir()
    assert final_model_path(tmp_path) is None

    (final / "config.json").write_text("{}")
    assert final_model_path(tmp_path) == str(final)


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

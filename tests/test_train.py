from __future__ import annotations

from pathlib import Path

from genterp.train import latest_checkpoint


def test_latest_checkpoint_empty_dir(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_selects_highest_step(tmp_path: Path):
    for step in (2, 10):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")

    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-10")

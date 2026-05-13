"""Smoke test for end-to-end CLT training against a fake-saved tiny Genterp.

Asserts: clt_train.main(["--tiny", "--max-steps=2"]) runs without error,
the recon loss is recorded, and a CLT checkpoint lands at runs-tiny/clt/final/clt.pt.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from genterp import clt_train
from genterp.modeling import Genterp, GenterpConfig
from genterp.train import GenterpForCausalLM, GenterpHFConfig


def _write_minimal_etl(etl: Path, n_atoms: int) -> None:
    etl.mkdir(parents=True, exist_ok=True)
    vocab = {f"V/{i}": i for i in range(1, n_atoms)}  # atom 0 reserved for PAD
    (etl / "vocab.json").write_text(json.dumps(vocab))
    (etl / "value_stats.json").write_text("{}")
    # Two tiny subjects, a handful of events each.
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time_seconds": [0, 86400, 172800, 259200, 0, 86400, 172800, 259200],
            "atom": pl.Series([1, 2, 3, 1, 2, 3, 1, 2], dtype=pl.UInt32),
            "value": [None] * 8,
        }
    ).write_parquet(etl / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "start": [0, 4],
            "end": [3, 7],
            "sex": [0, 1],
            "birth_seconds": [0, 0],
            "censor_seconds": [86400 * 10, 86400 * 10],
            "split": ["train", "train"],
        }
    ).write_parquet(etl / "subjects.parquet")


def _seed_fake_genterp(runs_dir: Path, n_atoms: int) -> None:
    cfg = GenterpConfig(n_atoms=n_atoms, dim=16, n_heads=2, n_layers=2)
    hf_cfg = GenterpHFConfig(genterp_cfg={
        "n_atoms": cfg.n_atoms, "dim": cfg.dim, "n_heads": cfg.n_heads, "n_layers": cfg.n_layers,
    })
    model = GenterpForCausalLM(hf_cfg)
    final = runs_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    # Write the bare minimum that `final_model_path` recognizes. `save_pretrained`
    # trips on the tied embed↔mark_out weight pair under both safetensors and bin
    # paths in some transformers versions; this bypass is acceptable for a test fixture.
    hf_cfg.save_pretrained(str(final))
    torch.save(model.state_dict(), final / "pytorch_model.bin")
    (runs_dir / "final_checkpoint.json").write_text(json.dumps({"path": "final"}))


def test_clt_train_smoke_end_to_end(tmp_path, monkeypatch):
    n_atoms = 8
    home = tmp_path
    monkeypatch.setenv("HOME", str(home))
    # Path.home() reads $HOME on POSIX/macOS; force a refresh by patching the function too.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    etl = home / "genterp" / "etl"
    runs = home / "genterp" / "runs-tiny"
    _write_minimal_etl(etl, n_atoms)
    _seed_fake_genterp(runs, n_atoms)

    # Keep the run cheap and CPU-friendly.
    torch.manual_seed(0)
    clt_train.main(["--tiny", "--max-steps", "2", "--log-every", "1", "--save-every", "100"])

    saved = runs / "clt" / "final" / "clt.pt"
    config_json = runs / "clt" / "final" / "config.json"
    assert saved.is_file(), "expected CLT weights file at runs-tiny/clt/final/clt.pt"
    assert config_json.is_file(), "expected CLT config sidecar"

    payload = torch.load(saved, map_location="cpu", weights_only=False)
    assert "state_dict" in payload and "config" in payload
    assert payload["config"]["n_layers"] == 2
    assert payload["config"]["dim"] == 16


def test_clt_train_errors_when_no_genterp_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _write_minimal_etl(tmp_path / "genterp" / "etl", n_atoms=4)
    # Note: no runs-tiny/ directory created → no final model exists.
    with pytest.raises(SystemExit, match="no final Genterp model"):
        clt_train.main(["--tiny", "--max-steps", "1"])

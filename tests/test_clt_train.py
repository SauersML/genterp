from __future__ import annotations

from pathlib import Path

import pytest
import torch

from genterp.clt_train import CLTTrainingConfig, iter_activation_chunks, load_clt_artifact, train_clt
from genterp.modeling import Genterp
from genterp.runtime import TorchRuntime
from genterp.transcoder import CLTConfig, CrossLayerTranscoder
from tests._factories import make_batch, tiny_config


def _cpu_runtime() -> TorchRuntime:
    return TorchRuntime(
        device=torch.device("cpu"),
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


def test_iter_activation_chunks_bounds_tokens_and_preserves_pairs():
    pre_mlp = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3)
    mlp_out = pre_mlp + 1000

    chunks = list(iter_activation_chunks(pre_mlp, mlp_out, chunk_tokens=4, shuffle=False))

    assert [chunk[0].shape[0] for chunk in chunks] == [4, 4, 2]
    assert torch.equal(chunks[0][0], pre_mlp[:4])
    assert torch.equal(chunks[2][1], mlp_out[8:])


def test_train_clt_end_to_end_saves_loadable_artifact(tmp_path: Path):
    torch.manual_seed(0)
    cfg = tiny_config(dim=16, n_heads=4, n_layers=2)
    base = Genterp(cfg).eval()
    clt = CrossLayerTranscoder(CLTConfig(n_layers=cfg.n_layers, dim=cfg.dim, n_features=16, off_diagonal_rank=2))
    batch = make_batch(B=2, T=10, n_atoms=cfg.n_atoms)
    training_cfg = CLTTrainingConfig(
        steps=4,
        learning_rate=1e-2,
        activation_batch_tokens=8,
        subject_batch_size=2,
        eval_every=2,
        save_every=3,
        eval_batches=1,
        max_events=10,
    )

    metrics = train_clt(
        base,
        clt,
        [batch],
        [batch],
        runtime=_cpu_runtime(),
        training_cfg=training_cfg,
        output_dir=tmp_path,
    )

    artifact_dir = Path(str(metrics["artifact_dir"]))
    loaded = load_clt_artifact(artifact_dir)
    assert artifact_dir.is_dir()
    assert (tmp_path / "final_clt.json").is_file()
    assert loaded.cfg == clt.cfg
    assert metrics["loss"] > 0
    assert "eval_loss" in metrics


def test_clt_train_cli_rejects_unknown_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The CLI surface is intentionally minimal: only --tiny is accepted.

    Every other knob (steps, lr, features, ranks, batch sizing) lives in
    CLTTrainingConfig defaults so library callers stay flexible while operators
    have one switch. This test pins that contract.
    """
    from genterp import clt_train

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for flag in ("--steps", "--model-dir", "--n-features", "--off-diagonal-rank"):
        with pytest.raises(SystemExit):
            clt_train.main([flag, "1"])


def test_clt_train_errors_when_no_genterp_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from genterp import clt_train

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(FileNotFoundError, match="no Genterp final checkpoint"):
        clt_train.main([])

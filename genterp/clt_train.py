"""Train a Cross-Layer Transcoder on a frozen Genterp checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler

from genterp.data import CohortDataset, collate
from genterp.progress import ProgressLogger, count_parameters
from genterp.runtime import TorchRuntime, accelerator_label, configure_torch_runtime
from genterp.train import GenterpForCausalLM, atomic_write_json, final_model_path
from genterp.transcoder import CLTConfig, CrossLayerTranscoder, harvest_transcoder_acts, unwrap_genterp_model

CLT_FINAL_POINTER_FILE = "final_clt.json"
CLT_CONFIG_FILE = "clt_config.json"
CLT_STATE_FILE = "clt_state.pt"
CLT_TRAINING_STATE_FILE = "clt_training_state.json"
DEFAULT_STEPS = 10_000
DEFAULT_EVAL_EVERY = 250
DEFAULT_SAVE_EVERY = 1_000


@dataclass(frozen=True)
class CLTTrainingConfig:
    steps: int = DEFAULT_STEPS
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    activation_batch_tokens: int = 8192
    subject_batch_size: int = 1
    eval_every: int = DEFAULT_EVAL_EVERY
    save_every: int = DEFAULT_SAVE_EVERY
    eval_batches: int = 8
    max_events: int = 4096
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.activation_batch_tokens <= 0:
            raise ValueError("activation_batch_tokens must be > 0")
        if self.subject_batch_size <= 0:
            raise ValueError("subject_batch_size must be > 0")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be > 0")
        if self.save_every <= 0:
            raise ValueError("save_every must be > 0")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be > 0")
        if self.max_events <= 0:
            raise ValueError("max_events must be > 0")


def _autocast_dtype(runtime: TorchRuntime) -> torch.dtype | None:
    if runtime.bf16:
        return torch.bfloat16
    if runtime.fp16:
        return torch.float16
    return None


def _autocast_context(runtime: TorchRuntime) -> torch.autocast:
    dtype = _autocast_dtype(runtime)
    return torch.autocast(device_type=runtime.device.type, dtype=dtype, enabled=dtype is not None)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def iter_activation_chunks(
    pre_mlp: torch.Tensor,
    mlp_out: torch.Tensor,
    *,
    chunk_tokens: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield bounded token chunks from harvested CLT activations."""
    if pre_mlp.shape != mlp_out.shape:
        raise ValueError(f"pre_mlp and mlp_out shapes differ: {tuple(pre_mlp.shape)} vs {tuple(mlp_out.shape)}")
    if pre_mlp.ndim != 3:
        raise ValueError("pre_mlp and mlp_out must have shape (tokens, layers, dim)")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    n_tokens = pre_mlp.shape[0]
    if n_tokens == 0:
        return
    if shuffle:
        order = torch.randperm(n_tokens, generator=generator).to(pre_mlp.device)
        pre_mlp = pre_mlp.index_select(0, order)
        mlp_out = mlp_out.index_select(0, order)
    for start in range(0, n_tokens, chunk_tokens):
        stop = min(start + chunk_tokens, n_tokens)
        yield pre_mlp[start:stop], mlp_out[start:stop]


def build_clt_dataloader(
    dataset: CohortDataset,
    *,
    batch_size: int,
    runtime: TorchRuntime,
    training: bool,
) -> DataLoader:
    sampler = LengthGroupedSampler(batch_size, lengths=dataset.lengths) if training else None
    workers = runtime.dataloader_num_workers
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": workers,
        "pin_memory": runtime.dataloader_pin_memory,
        "persistent_workers": workers > 0,
        "sampler": sampler,
        "drop_last": training,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = runtime.dataloader_prefetch_factor
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate_clt(
    base_model: torch.nn.Module,
    clt: CrossLayerTranscoder,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    runtime: TorchRuntime,
    activation_batch_tokens: int,
    max_batches: int,
) -> dict[str, float]:
    clt_was_training = clt.training
    clt.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_sparsity = 0.0
    total_active = 0.0
    total_chunks = 0
    try:
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            batch = _move_batch_to_device(batch, runtime.device)
            with _autocast_context(runtime):
                pre_mlp, mlp_out = harvest_transcoder_acts(base_model, batch)
                for pre_chunk, target_chunk in iter_activation_chunks(
                    pre_mlp,
                    mlp_out,
                    chunk_tokens=activation_batch_tokens,
                    shuffle=False,
                ):
                    metrics = clt.loss(pre_chunk, target_chunk)
                    total_loss += float(metrics["loss"].detach().float().item())
                    total_recon += float(metrics["recon"].detach().float().item())
                    total_sparsity += float(metrics["sparsity"].detach().float().item())
                    total_active += float(metrics["n_active"].detach().float().item())
                    total_chunks += 1
    finally:
        clt.train(clt_was_training)
    if total_chunks == 0:
        raise ValueError("evaluation produced no activation chunks")
    return {
        "loss": total_loss / total_chunks,
        "recon": total_recon / total_chunks,
        "sparsity": total_sparsity / total_chunks,
        "n_active": total_active / total_chunks,
        "chunks": float(total_chunks),
    }


def save_clt_artifact(
    clt: CrossLayerTranscoder,
    output_dir: str | Path,
    *,
    step: int,
    training_cfg: CLTTrainingConfig,
    runtime: TorchRuntime,
    metrics: dict[str, float] | None,
    final: bool,
) -> Path:
    logger = ProgressLogger("clt_save", total_units=5)
    output_dir = Path(output_dir)
    name = f"{'final' if final else 'checkpoint'}-{step}-{uuid.uuid4().hex[:12]}"
    tmp_dir = output_dir / f".{name}.tmp"
    artifact_dir = output_dir / name

    logger.start_unit("prepare CLT artifact directory", f"tmp_dir={tmp_dir} artifact_dir={artifact_dir}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    logger.finish_unit("prepare CLT artifact directory", f"artifact_dir={artifact_dir}")

    logger.start_unit("write CLT config", f"file={tmp_dir / CLT_CONFIG_FILE}")
    atomic_write_json(tmp_dir / CLT_CONFIG_FILE, asdict(clt.cfg))
    logger.finish_unit("write CLT config", f"features={clt.cfg.n_features:,} layers={clt.cfg.n_layers}")

    logger.start_unit("write CLT weights", f"file={tmp_dir / CLT_STATE_FILE}")
    torch.save(clt.state_dict(), tmp_dir / CLT_STATE_FILE)
    logger.finish_unit("write CLT weights", f"params={count_parameters(clt):,}")

    logger.start_unit("write CLT training state", f"file={tmp_dir / CLT_TRAINING_STATE_FILE}")
    state = {
        "step": step,
        "training_config": asdict(training_cfg),
        "runtime": {
            "device_type": runtime.device.type,
            "device_index": runtime.device.index,
            "precision": "bf16" if runtime.bf16 else "fp16" if runtime.fp16 else "fp32",
        },
        "metrics": metrics or {},
    }
    atomic_write_json(tmp_dir / CLT_TRAINING_STATE_FILE, state)
    logger.finish_unit("write CLT training state", f"step={step:,}")

    logger.start_unit("publish CLT artifact", f"from={tmp_dir} to={artifact_dir}")
    tmp_dir.replace(artifact_dir)
    if final:
        atomic_write_json(output_dir / CLT_FINAL_POINTER_FILE, {"path": artifact_dir.name})
    logger.finish_unit("publish CLT artifact", f"path={artifact_dir}")
    return artifact_dir


def load_clt_artifact(path: str | Path, *, map_location: torch.device | str = "cpu") -> CrossLayerTranscoder:
    path = Path(path)
    cfg = CLTConfig(**json.loads((path / CLT_CONFIG_FILE).read_text()))
    clt = CrossLayerTranscoder(cfg)
    state = torch.load(path / CLT_STATE_FILE, map_location=map_location, weights_only=True)
    clt.load_state_dict(state)
    return clt


def train_clt(
    base_model: torch.nn.Module,
    clt: CrossLayerTranscoder,
    train_dataloader: Iterable[dict[str, torch.Tensor]],
    eval_dataloader: Iterable[dict[str, torch.Tensor]] | None,
    *,
    runtime: TorchRuntime,
    training_cfg: CLTTrainingConfig,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    logger = ProgressLogger("clt_train", total_units=training_cfg.steps)
    torch.manual_seed(training_cfg.seed)

    base = unwrap_genterp_model(base_model)
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)

    clt.train()
    opt = torch.optim.AdamW(
        clt.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(training_cfg.seed)

    last_metrics: dict[str, float] = {}
    step = 0
    while step < training_cfg.steps:
        had_batch = False
        for batch in train_dataloader:
            had_batch = True
            batch = _move_batch_to_device(batch, runtime.device)
            with _autocast_context(runtime):
                pre_mlp, mlp_out = harvest_transcoder_acts(base, batch)

            for pre_chunk, target_chunk in iter_activation_chunks(
                pre_mlp,
                mlp_out,
                chunk_tokens=training_cfg.activation_batch_tokens,
                shuffle=True,
                generator=generator,
            ):
                step += 1
                opt.zero_grad(set_to_none=True)
                with _autocast_context(runtime):
                    metrics = clt.loss(pre_chunk, target_chunk)
                metrics["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(clt.parameters(), training_cfg.grad_clip_norm)
                opt.step()

                last_metrics = {
                    "loss": float(metrics["loss"].detach().float().item()),
                    "recon": float(metrics["recon"].detach().float().item()),
                    "sparsity": float(metrics["sparsity"].detach().float().item()),
                    "n_active": float(metrics["n_active"].detach().float().item()),
                    "grad_norm": float(grad_norm.detach().float().item()),
                }
                logger.set_progress(step, training_cfg.steps)
                logger.log(
                    "CLT optimizer step",
                    " ".join(
                        [
                            f"loss={last_metrics['loss']:.5g}",
                            f"recon={last_metrics['recon']:.5g}",
                            f"sparsity={last_metrics['sparsity']:.5g}",
                            f"n_active={last_metrics['n_active']:.2f}",
                            f"grad_norm={last_metrics['grad_norm']:.3g}",
                        ]
                    ),
                )

                if eval_dataloader is not None and step % training_cfg.eval_every == 0:
                    eval_metrics = evaluate_clt(
                        base,
                        clt,
                        eval_dataloader,
                        runtime=runtime,
                        activation_batch_tokens=training_cfg.activation_batch_tokens,
                        max_batches=training_cfg.eval_batches,
                    )
                    last_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
                    logger.log(
                        "CLT evaluation",
                        f"eval_loss={eval_metrics['loss']:.5g} eval_recon={eval_metrics['recon']:.5g} "
                        f"eval_n_active={eval_metrics['n_active']:.2f} chunks={int(eval_metrics['chunks'])}",
                    )

                if output_dir is not None and step % training_cfg.save_every == 0:
                    save_clt_artifact(
                        clt,
                        output_dir,
                        step=step,
                        training_cfg=training_cfg,
                        runtime=runtime,
                        metrics=last_metrics,
                        final=False,
                    )

                if step >= training_cfg.steps:
                    break
            if step >= training_cfg.steps:
                break
        if not had_batch:
            raise ValueError("training dataloader produced no batches")
    if output_dir is not None:
        artifact = save_clt_artifact(
            clt,
            output_dir,
            step=step,
            training_cfg=training_cfg,
            runtime=runtime,
            metrics=last_metrics,
            final=True,
        )
        last_metrics["artifact_dir"] = str(artifact)
    return last_metrics


def _default_activation_batch_tokens(runtime: TorchRuntime) -> int:
    if runtime.device.type == "cuda":
        return 8192
    if runtime.device.type == "mps":
        return 2048
    return 1024


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLT on a frozen Genterp checkpoint.")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Load from ~/genterp/runs-tiny/ and use the tiny CLT preset. Pair with "
        "`scripts.aou_etl --tiny` + `genterp.train --tiny`.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    setup = ProgressLogger("clt_setup", total_units=9)

    setup.start_unit("configure torch runtime", "selecting accelerator and precision")
    runtime = configure_torch_runtime()
    setup.finish_unit("configure torch runtime", f"accelerator={accelerator_label(runtime)}")

    data_dir = Path.home() / "genterp" / "etl"
    runs_dir = Path.home() / "genterp" / ("runs-tiny" if args.tiny else "runs")
    output_dir = runs_dir / "clt"

    setup.start_unit("resolve Genterp checkpoint", f"runs_dir={runs_dir}")
    resolved = final_model_path(runs_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"no Genterp final checkpoint under {runs_dir}; run `python -m genterp.train"
            f"{' --tiny' if args.tiny else ''}` first"
        )
    model_dir = Path(resolved)
    setup.finish_unit("resolve Genterp checkpoint", f"model_dir={model_dir}")

    setup.start_unit("load frozen Genterp model", f"path={model_dir}")
    model = GenterpForCausalLM.from_pretrained(model_dir).to(runtime.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    base = unwrap_genterp_model(model)
    setup.finish_unit(
        "load frozen Genterp model",
        f"layers={base.cfg.n_layers} dim={base.cfg.dim} params={count_parameters(model):,}",
    )

    setup.start_unit("build CLT datasets", f"data_dir={data_dir}")
    train_dataset = CohortDataset(data_dir, split="train")
    eval_dataset = CohortDataset(data_dir, split="test")
    setup.finish_unit("build CLT datasets", f"train={len(train_dataset):,} eval={len(eval_dataset):,}")

    subject_batch_size = runtime.per_device_train_batch_size
    activation_batch_tokens = _default_activation_batch_tokens(runtime)
    training_cfg = CLTTrainingConfig(
        steps=500 if args.tiny else DEFAULT_STEPS,
        activation_batch_tokens=activation_batch_tokens,
        subject_batch_size=subject_batch_size,
    )

    setup.start_unit("build CLT dataloaders", f"subject_batch_size={subject_batch_size}")
    train_loader = build_clt_dataloader(train_dataset, batch_size=subject_batch_size, runtime=runtime, training=True)
    eval_loader = build_clt_dataloader(eval_dataset, batch_size=subject_batch_size, runtime=runtime, training=False)
    setup.finish_unit("build CLT dataloaders", f"workers={runtime.dataloader_num_workers}")

    n_features = 256 if args.tiny else 8192
    off_diagonal_rank = 4 if args.tiny else 32
    clt_cfg = CLTConfig(
        n_layers=base.cfg.n_layers,
        dim=base.cfg.dim,
        n_features=n_features,
        off_diagonal_rank=off_diagonal_rank,
    )
    setup.start_unit("initialize CLT", f"features={n_features:,} off_diagonal_rank={off_diagonal_rank}")
    clt = CrossLayerTranscoder(clt_cfg).to(runtime.device)
    setup.finish_unit("initialize CLT", f"params={count_parameters(clt):,}")

    setup.start_unit("prepare CLT output directory", f"path={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup.finish_unit("prepare CLT output directory", f"path={output_dir}")

    setup.start_unit(
        "train CLT",
        f"steps={training_cfg.steps:,} activation_batch_tokens={training_cfg.activation_batch_tokens:,}",
    )
    metrics = train_clt(
        model,
        clt,
        train_loader,
        eval_loader,
        runtime=runtime,
        training_cfg=training_cfg,
        output_dir=output_dir,
    )
    setup.finish_unit(
        "train CLT",
        f"loss={metrics.get('loss', float('nan')):.5g} recon={metrics.get('recon', float('nan')):.5g} "
        f"artifact={metrics.get('artifact_dir')}",
    )


if __name__ == "__main__":
    main()

"""Train a Cross-Layer Transcoder on the activations of a frozen, trained Genterp.

Pipeline:
  1. Find the most recent final Genterp checkpoint in ``~/genterp/runs[-tiny]/``.
  2. Freeze it; eval mode; move to runtime device.
  3. Stream real batches from ``CohortDataset(split="train")``.
  4. For each batch, run a no-grad forward with ``return_transcoder_acts=True``
     and harvest ``(pre_mlp, mlp_out)`` per real token per layer.
  5. Step AdamW on a ``CrossLayerTranscoder`` against ``clt.loss(...)``
     (per-layer-std-normalized recon + JumpReLU tanh sparsity).
  6. Save the CLT to ``<runs>/clt/final/clt.pt`` so the interpretability
     helpers (``top_activating_examples``, ``feature_to_output_attribution``,
     ``feature_to_feature_attribution_graph``) have something to consume.

Only the CLT receives gradients. Genterp is frozen, so this is interpretability
post-training, not a co-trained autoencoder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genterp.data import CohortDataset, collate
from genterp.progress import ProgressLogger, count_parameters
from genterp.runtime import TorchRuntime, accelerator_label, configure_torch_runtime, move_batch_to_device
from genterp.train import GenterpForCausalLM, final_model_path
from genterp.transcoder import CLTConfig, CrossLayerTranscoder, harvest_transcoder_acts


@dataclass
class CLTTrainingConfig:
    """All knobs for a CLT training run. Build with ``CLTTrainingConfig.from_args(argv)``."""

    runs_dir: Path
    etl_dir: Path
    tiny: bool = False
    max_steps: int = 20_000
    lr: float = 1e-3
    log_every: int = 50
    save_every: int = 2_000
    feature_mult: int = 16
    sparsity_coef: float = 1e-3
    off_diagonal_rank: int | None = None
    # Optional Genterp-config override; normally inferred from the loaded checkpoint.
    genterp_cfg_override: dict = field(default_factory=dict)

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> CLTTrainingConfig:
        parser = argparse.ArgumentParser(description="Train a Cross-Layer Transcoder on a frozen Genterp.")
        parser.add_argument("--tiny", action="store_true")
        parser.add_argument("--max-steps", type=int, default=None)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--log-every", type=int, default=50)
        parser.add_argument("--save-every", type=int, default=2_000)
        args = parser.parse_args(argv)
        tiny = args.tiny
        max_steps = args.max_steps if args.max_steps is not None else (500 if tiny else 20_000)
        return cls(
            runs_dir=_resolve_runs_dir(tiny),
            etl_dir=Path.home() / "genterp" / "etl",
            tiny=tiny,
            max_steps=max_steps,
            lr=args.lr,
            log_every=args.log_every,
            save_every=args.save_every,
            feature_mult=4 if tiny else 16,
        )


def _resolve_runs_dir(tiny: bool) -> Path:
    return Path.home() / "genterp" / ("runs-tiny" if tiny else "runs")


def _load_frozen_genterp(runs_dir: Path, runtime: TorchRuntime, logger: ProgressLogger) -> tuple[GenterpForCausalLM, dict]:
    logger.start_unit("locate final Genterp checkpoint", f"runs_dir={runs_dir}")
    path = final_model_path(runs_dir)
    if path is None:
        raise SystemExit(
            f"no final Genterp model found under {runs_dir}; run `python -m genterp.train`"
            f"{' --tiny' if 'tiny' in runs_dir.name else ''} first"
        )
    logger.finish_unit("locate final Genterp checkpoint", f"path={path}")

    logger.start_unit("load Genterp weights + config", f"from_pretrained({path})")
    model = GenterpForCausalLM.from_pretrained(path)
    genterp_cfg = dict(getattr(model.config, "genterp_cfg", {}))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(runtime.device)
    logger.finish_unit(
        "load Genterp weights + config",
        f"params={count_parameters(model):,} dim={genterp_cfg.get('dim')} layers={genterp_cfg.get('n_layers')} (frozen)",
    )
    return model, genterp_cfg


def _build_clt_config(genterp_cfg: dict, cfg: CLTTrainingConfig) -> CLTConfig:
    dim = int(genterp_cfg["dim"])
    n_layers = int(genterp_cfg["n_layers"])
    return CLTConfig(
        n_layers=n_layers,
        dim=dim,
        n_features=cfg.feature_mult * dim,
        sparsity_coef=cfg.sparsity_coef,
        off_diagonal_rank=cfg.off_diagonal_rank,
    )


def _save_clt(clt: CrossLayerTranscoder, runs_dir: Path, logger: ProgressLogger) -> Path:
    out_dir = runs_dir / "clt" / "final"
    tmp_dir = runs_dir / "clt" / ".final.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.start_unit("save CLT state_dict", f"tmp_dir={tmp_dir}")
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in clt.state_dict().items()}, "config": asdict(clt.cfg)},
        tmp_dir / "clt.pt",
    )
    (tmp_dir / "config.json").write_text(json.dumps(asdict(clt.cfg), indent=2, sort_keys=True))
    logger.finish_unit("save CLT state_dict", "clt.pt + config.json written")

    logger.start_unit("atomic replace final CLT dir", f"out_dir={out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.replace(out_dir)
    logger.finish_unit("atomic replace final CLT dir", f"out_dir={out_dir}")
    return out_dir


def load_clt_artifact(clt_dir: str | Path) -> tuple[CrossLayerTranscoder, CLTConfig]:
    """Load a CLT saved by :func:`train_clt` from ``runs_dir/clt/final/``."""
    clt_dir = Path(clt_dir)
    payload = torch.load(clt_dir / "clt.pt", map_location="cpu", weights_only=False)
    clt_cfg = CLTConfig(**payload["config"])
    clt = CrossLayerTranscoder(clt_cfg)
    clt.load_state_dict(payload["state_dict"])
    return clt, clt_cfg


def _harvest_and_step(
    genterp: GenterpForCausalLM,
    clt: CrossLayerTranscoder,
    optim: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    runtime: TorchRuntime,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, float]:
    with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype, enabled=use_amp):
        pre_mlp, mlp_out = harvest_transcoder_acts(genterp, batch)
    # CLT in fp32 for stable JumpReLU thresholds + sparsity arithmetic; activation tensors are small.
    pre_mlp = pre_mlp.detach().float()
    mlp_out = mlp_out.detach().float()
    if pre_mlp.shape[0] == 0:
        return {"skipped_empty_batch": 1.0}
    out = clt.loss(pre_mlp, mlp_out)
    optim.zero_grad(set_to_none=True)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(clt.parameters(), 1.0)
    optim.step()
    return {
        "loss": float(out["loss"].item()),
        "recon": float(out["recon"].item()),
        "sparsity": float(out["sparsity"].item()),
        "n_active": float(out["n_active"].item()),
        "tokens": float(pre_mlp.shape[0]),
    }


def train_clt(cfg: CLTTrainingConfig) -> Path:
    """Run the CLT training loop end-to-end. Returns the path of the saved final CLT."""
    setup = ProgressLogger("clt_setup", total_units=9)

    setup.start_unit("configure torch runtime", "selecting accelerator, precision, optimizer")
    runtime = configure_torch_runtime()
    setup.finish_unit(
        "configure torch runtime",
        f"accelerator={accelerator_label(runtime)} batch_per_device={runtime.per_device_train_batch_size}",
    )

    setup.start_unit("resolve directories", "runs-tiny when --tiny, else runs")
    setup.finish_unit("resolve directories", f"etl={cfg.etl_dir} runs_dir={cfg.runs_dir}")

    genterp, genterp_cfg = _load_frozen_genterp(cfg.runs_dir, runtime, setup)

    setup.start_unit("build train dataset", "split=train")
    train_dataset = CohortDataset(cfg.etl_dir, split="train")
    setup.finish_unit("build train dataset", f"subjects={len(train_dataset):,}")

    setup.start_unit("build CLT config + module", f"--tiny={cfg.tiny} sizing off loaded Genterp")
    clt_cfg = _build_clt_config(genterp_cfg, cfg)
    clt = CrossLayerTranscoder(clt_cfg).to(runtime.device)
    setup.finish_unit(
        "build CLT config + module",
        f"n_layers={clt_cfg.n_layers} dim={clt_cfg.dim} n_features={clt_cfg.n_features:,} "
        f"params={count_parameters(clt):,} off_diagonal_rank={clt_cfg.off_diagonal_rank}",
    )

    setup.start_unit("initialize AdamW on CLT params", f"lr={cfg.lr}")
    optim = torch.optim.AdamW(
        clt.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=runtime.optim == "adamw_torch_fused",
    )
    setup.finish_unit("initialize AdamW on CLT params", f"fused={runtime.optim == 'adamw_torch_fused'}")

    setup.start_unit(
        "build DataLoader",
        f"batch_per_device={runtime.per_device_train_batch_size} num_workers={runtime.dataloader_num_workers}",
    )
    loader = DataLoader(
        train_dataset,
        batch_size=runtime.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=runtime.dataloader_num_workers,
        pin_memory=runtime.dataloader_pin_memory,
        persistent_workers=runtime.dataloader_num_workers > 0,
        drop_last=True,
    )
    setup.finish_unit("build DataLoader", f"len={len(loader)} batches/epoch")

    setup.start_unit("configure AMP", f"bf16={runtime.bf16} fp16={runtime.fp16}")
    amp_dtype = torch.bfloat16 if runtime.bf16 else torch.float16
    use_amp = runtime.device.type == "cuda" and (runtime.bf16 or runtime.fp16)
    setup.finish_unit("configure AMP", f"use_amp={use_amp} amp_dtype={amp_dtype}")

    setup.start_unit(
        "run CLT training loop",
        f"max_steps={cfg.max_steps:,} log_every={cfg.log_every} save_every={cfg.save_every}",
    )
    train_log = ProgressLogger("clt_train", total_units=cfg.max_steps)
    step = 0
    init_recon: float | None = None
    last_recon: float | None = None
    t0 = time.monotonic()
    done = False
    while not done:
        for batch in loader:
            batch = move_batch_to_device(batch, runtime.device)
            metrics = _harvest_and_step(genterp, clt, optim, batch, runtime, amp_dtype, use_amp)
            step += 1
            if "loss" in metrics:
                last_recon = metrics["recon"]
                if init_recon is None:
                    init_recon = last_recon
                if step % cfg.log_every == 0 or step == 1:
                    elapsed = time.monotonic() - t0
                    rate = step / max(elapsed, 1e-6)
                    train_log.set_progress(step, cfg.max_steps)
                    train_log.log(
                        "clt step",
                        f"step={step:,}/{cfg.max_steps:,} loss={metrics['loss']:.4f} recon={metrics['recon']:.4f} "
                        f"sparsity={metrics['sparsity']:.4f} n_active={metrics['n_active']:.1f} "
                        f"tokens={int(metrics['tokens'])} rate={rate:.2f} steps/s",
                    )
            if step > 0 and step % cfg.save_every == 0:
                _save_clt(clt, cfg.runs_dir, train_log)
            if step >= cfg.max_steps:
                done = True
                break
    setup.finish_unit(
        "run CLT training loop",
        f"steps={step:,} init_recon={init_recon} final_recon={last_recon} elapsed={time.monotonic()-t0:.1f}s",
    )

    return _save_clt(clt, cfg.runs_dir, setup)


def main(argv: list[str] | None = None) -> None:
    cfg = CLTTrainingConfig.from_args(argv)
    train_clt(cfg)


if __name__ == "__main__":
    main()

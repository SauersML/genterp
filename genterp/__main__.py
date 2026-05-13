"""End-to-end demo: build a small Genterp, train 100 steps on synthetic batches with joint marked-TPP loss."""

from __future__ import annotations

import time

import torch

from genterp import Genterp, GenterpConfig
from genterp._synthetic import make_batch
from genterp.progress import ProgressLogger, count_parameters
from genterp.runtime import accelerator_label, configure_torch_runtime, move_batch_to_device


def main() -> None:
    setup = ProgressLogger("demo_setup", total_units=10)
    setup.start_unit("configure runtime", "selecting accelerator and precision settings")
    runtime = configure_torch_runtime()
    setup.finish_unit("configure runtime", f"accelerator={accelerator_label(runtime)}")

    setup.start_unit("seed torch RNG", "seed=0")
    torch.manual_seed(0)
    setup.finish_unit("seed torch RNG", "seed=0")

    setup.start_unit("build model config", "n_atoms=512 dim=128 heads=4 layers=4")
    n_atoms = 512
    cfg = GenterpConfig(n_atoms=n_atoms, dim=128, n_heads=4, n_layers=4, n_static_blocks=2, k_static_summary=8)
    setup.finish_unit("build model config", f"n_atoms={cfg.n_atoms:,} dim={cfg.dim} layers={cfg.n_layers}")

    setup.start_unit("initialize model", f"device={runtime.device}")
    model = Genterp(cfg).to(runtime.device)
    setup.finish_unit("initialize model", f"params={count_parameters(model):,}")

    setup.start_unit("build synthetic value-stat flags", "marking half of atoms as magnitude-bearing")
    has_mag = torch.zeros(n_atoms, dtype=torch.bool)
    has_mag[torch.randperm(n_atoms)[: n_atoms // 2]] = True
    has_mag[0] = False
    setup.finish_unit("build synthetic value-stat flags", f"magnitude_atoms={int(has_mag.sum().item()):,}")

    setup.start_unit("apply value modulation stats", "mu=0 sigma=1 for synthetic demo")
    model.value_mod.set_stats(torch.zeros(n_atoms), torch.ones(n_atoms), has_mag)
    setup.finish_unit("apply value modulation stats", f"magnitude_atoms={int(has_mag.sum().item()):,}")

    setup.start_unit("initialize AdamW optimizer", f"optim={runtime.optim} lr=3e-4")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        fused=runtime.optim == "adamw_torch_fused",
    )
    setup.finish_unit("initialize AdamW optimizer", f"fused={runtime.optim == 'adamw_torch_fused'}")

    setup.start_unit("configure AMP scaler", f"bf16={runtime.bf16} fp16={runtime.fp16}")
    amp_dtype = torch.bfloat16 if runtime.bf16 else torch.float16
    use_amp = runtime.device.type == "cuda" and (runtime.bf16 or runtime.fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.fp16)
    setup.finish_unit("configure AMP scaler", f"use_amp={use_amp} amp_dtype={amp_dtype} scaler_enabled={runtime.fp16}")

    params = count_parameters(model)
    print(f"genterp demo  device={accelerator_label(runtime)}  params={params:,}")

    setup.start_unit("build synthetic batch", "B=max(4, runtime batch) M=4 T=24")
    batch_size = max(4, runtime.per_device_train_batch_size)
    batch = make_batch(B=batch_size, M=4, T=24, n_atoms=n_atoms, seed=0)
    setup.finish_unit("build synthetic batch", f"batch_size={batch_size} fields={len(batch):,}")

    setup.start_unit("move synthetic batch to runtime device", f"device={runtime.device}")
    batch = move_batch_to_device(batch, runtime.device)
    setup.finish_unit("move synthetic batch to runtime device", f"fields={len(batch):,} device={runtime.device}")

    t0 = time.time()
    train_log = ProgressLogger("demo_train", total_units=100)
    for step in range(100):
        train_log.start_unit("optimization step", f"step={step + 1}/100 forward loss, backward, clip, optimizer step")
        with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype, enabled=use_amp):
            ld = model.loss(**batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(ld["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        train_log.finish_unit(
            "optimization step",
            f"step={step + 1}/100 loss={ld['loss'].item():.4f} time={ld['time_nll'].item():.4f} "
            f"mark={ld['mark_nll'].item():.4f} value={ld['value_nll'].item():.4f}",
        )
    train_log.log("demo training complete", f"elapsed={time.time() - t0:.1f}s steps=100")


if __name__ == "__main__":
    main()

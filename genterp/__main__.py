"""End-to-end demo: build a small Genterp, train 100 steps on synthetic batches with joint marked-TPP loss."""

from __future__ import annotations

import time

import torch

from genterp import Genterp, GenterpConfig
from genterp._synthetic import make_batch
from genterp.runtime import configure_torch_runtime, move_batch_to_device


def main() -> None:
    runtime = configure_torch_runtime()
    torch.manual_seed(0)
    n_atoms = 512
    cfg = GenterpConfig(n_atoms=n_atoms, dim=128, n_heads=4, n_layers=4, n_static_blocks=2, k_static_summary=8)
    model = Genterp(cfg).to(runtime.device)
    has_mag = torch.zeros(n_atoms, dtype=torch.bool)
    has_mag[torch.randperm(n_atoms)[: n_atoms // 2]] = True
    has_mag[0] = False
    model.value_mod.set_stats(torch.zeros(n_atoms), torch.ones(n_atoms), has_mag)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        fused=runtime.optim == "adamw_torch_fused",
    )
    amp_dtype = torch.bfloat16 if runtime.bf16 else torch.float16
    use_amp = runtime.device.type == "cuda" and (runtime.bf16 or runtime.fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.fp16)
    params = sum(p.numel() for p in model.parameters())
    print(f"genterp demo  device={runtime.device.type}  params={params:,}")

    batch_size = max(4, runtime.per_device_train_batch_size)
    batch = move_batch_to_device(make_batch(B=batch_size, M=4, T=24, n_atoms=n_atoms, seed=0), runtime.device)

    t0 = time.time()
    for step in range(100):
        with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype, enabled=use_amp):
            ld = model.loss(**batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(ld["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 10 == 0 or step == 99:
            print(
                f"step {step:3d}  loss {ld['loss'].item():.4f}  time {ld['time_nll'].item():.4f}  "
                f"mark {ld['mark_nll'].item():.4f}  value {ld['value_nll'].item():.4f}"
            )
    print(f"ok  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

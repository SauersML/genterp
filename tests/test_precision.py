"""Forward is dtype/device-clean: CPU fp32, CPU bf16 autocast, MPS, CUDA."""

from __future__ import annotations

import torch

from genterp import CrosscoderConfig, Genterp, MultiLayerCrosscoder, harvest_activations
from tests._factories import make_batch, tiny_config


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _forward_finite(model: Genterp, batch: dict, autocast_dtype: torch.dtype | None, device_type: str) -> torch.Tensor:
    ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_dtype else torch.autocast(device_type=device_type, enabled=False)
    with torch.no_grad(), ctx:
        out = model(**batch)
    assert torch.isfinite(out).all()
    return out


def test_cpu_fp32_bf16_consistency():
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    batch = make_batch(n_atoms=cfg.n_atoms)
    out_fp32 = _forward_finite(model, batch, None, "cpu")
    out_bf16 = _forward_finite(model, batch, torch.bfloat16, "cpu")
    drift = (out_fp32 - out_bf16.float()).abs().max().item()
    rel = drift / max(out_fp32.abs().max().item(), 1e-8)
    assert drift < 2.0 and rel < 0.5, f"unexpected fp32↔bf16 drift {drift} (rel {rel})"


def test_crosscoder_under_bf16():
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    batch = make_batch(n_atoms=cfg.n_atoms)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        acts = harvest_activations(model, batch)
    assert torch.isfinite(acts).all()

    cc = MultiLayerCrosscoder(CrosscoderConfig(n_layers=acts.shape[1], dim=acts.shape[2], n_features=64, l1_coef=1e-3))
    opt = torch.optim.Adam(cc.parameters(), lr=5e-3)
    init = cc.loss(acts)["recon"].item()
    for _ in range(30):
        out = cc.loss(acts)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert cc.loss(acts)["recon"].item() < init


def test_cuda_bf16():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    cfg = tiny_config()
    model = Genterp(cfg).to(device).eval()
    batch = _to_device(make_batch(n_atoms=cfg.n_atoms), device)
    out = _forward_finite(model, batch, torch.bfloat16, "cuda")
    assert out.is_cuda


def test_mps():
    if not torch.backends.mps.is_available():
        return
    device = torch.device("mps")
    cfg = tiny_config()
    model = Genterp(cfg).to(device).eval()
    batch = _to_device(make_batch(n_atoms=cfg.n_atoms), device)
    with torch.no_grad():
        out = model(**batch)
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()

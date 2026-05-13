from __future__ import annotations

from types import SimpleNamespace

import torch

from genterp.runtime import GIB, batch_size_for_cuda_memory, get_torch_runtime


def test_cuda_batch_size_scales_with_visible_memory():
    assert batch_size_for_cuda_memory(12 * GIB) == 1
    assert batch_size_for_cuda_memory(20 * GIB) == 2
    assert batch_size_for_cuda_memory(40 * GIB) == 4
    assert batch_size_for_cuda_memory(64 * GIB) == 8
    assert batch_size_for_cuda_memory(80 * GIB) == 16
    assert batch_size_for_cuda_memory(180 * GIB) == 32


def test_cuda_runtime_uses_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=180 * GIB))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.per_device_train_batch_size == 32
    assert runtime.bf16
    assert not runtime.fp16
    assert runtime.tf32
    assert runtime.torch_compile
    assert runtime.optim == "adamw_torch_fused"


def test_cuda_runtime_uses_fp16_when_bf16_is_not_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=24 * GIB))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.per_device_train_batch_size == 4
    assert not runtime.bf16
    assert runtime.fp16


def test_mps_runtime_avoids_cuda_only_options(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device.type == "mps"
    assert not runtime.bf16
    assert not runtime.fp16
    assert not runtime.tf32
    assert not runtime.torch_compile
    assert runtime.optim == "adamw_torch"
    assert not runtime.dataloader_pin_memory

from __future__ import annotations

from types import SimpleNamespace

import torch

from genterp.runtime import GIB, accelerator_label, batch_size_for_cuda_memory, get_torch_runtime


def test_cuda_batch_size_scales_with_visible_memory():
    assert batch_size_for_cuda_memory(12 * GIB) == 1
    assert batch_size_for_cuda_memory(20 * GIB) == 2
    assert batch_size_for_cuda_memory(40 * GIB) == 4
    assert batch_size_for_cuda_memory(64 * GIB) == 8
    assert batch_size_for_cuda_memory(80 * GIB) == 16
    assert batch_size_for_cuda_memory(180 * GIB) == 32


def test_cuda_runtime_uses_bf16_when_supported(monkeypatch):
    set_devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=180 * GIB, name=f"NVIDIA H200 {index}", major=9, minor=0),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.device.index == 0
    assert runtime.cuda_device_count == 8
    assert runtime.cuda_name == "NVIDIA H200 0"
    assert runtime.cuda_capability == (9, 0)
    assert runtime.per_device_train_batch_size == 32
    assert runtime.bf16
    assert not runtime.fp16
    assert runtime.tf32
    assert runtime.torch_compile
    assert runtime.torch_compile_mode == "max-autotune"
    assert runtime.optim == "adamw_torch_fused"
    assert runtime.use_data_parallel
    assert set_devices == [0]
    assert "visible_gpus=8" in accelerator_label(runtime)
    assert "strategy=data_parallel" in accelerator_label(runtime)


def test_cuda_runtime_uses_fp16_when_bf16_is_not_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=16 * GIB, name="NVIDIA T4", major=7, minor=5),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.cuda_name == "NVIDIA T4"
    assert runtime.cuda_capability == (7, 5)
    assert runtime.per_device_train_batch_size == 2
    assert not runtime.bf16
    assert runtime.fp16
    assert not runtime.tf32
    assert not runtime.torch_compile
    assert runtime.torch_compile_mode is None
    assert not runtime.use_data_parallel


def test_cuda_runtime_picks_best_visible_mixed_gpu(monkeypatch):
    set_devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    props = [
        SimpleNamespace(total_memory=16 * GIB, name="NVIDIA T4", major=7, minor=5),
        SimpleNamespace(total_memory=180 * GIB, name="NVIDIA H200", major=9, minor=0),
        SimpleNamespace(total_memory=80 * GIB, name="NVIDIA A100", major=8, minor=0),
        SimpleNamespace(total_memory=24 * GIB, name="NVIDIA L4", major=8, minor=9),
    ]
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: props[index],
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device == torch.device("cuda", 1)
    assert runtime.cuda_name == "NVIDIA H200"
    assert not runtime.use_data_parallel
    assert set_devices == [1]
    assert "strategy=single_gpu" in accelerator_label(runtime)


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

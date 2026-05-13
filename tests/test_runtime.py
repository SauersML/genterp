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


def test_cuda_runtime_uses_all_homogeneous_bf16_devices(monkeypatch):
    set_devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            total_memory=180 * GIB,
            name=f"accelerator-{index}",
            major=9,
            minor=0,
            multi_processor_count=120,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.device.index == 0
    assert runtime.cuda_device_count == 8
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


def test_cuda_runtime_uses_fp16_without_unsupported_newer_cuda_features(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            total_memory=16 * GIB,
            name="accelerator",
            major=7,
            minor=5,
            multi_processor_count=40,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.cuda_capability == (7, 5)
    assert runtime.per_device_train_batch_size == 2
    assert not runtime.bf16
    assert runtime.fp16
    assert not runtime.tf32
    assert not runtime.torch_compile
    assert runtime.torch_compile_mode is None
    assert not runtime.use_data_parallel


def test_cuda_runtime_keeps_pre_tensor_core_devices_in_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            total_memory=8 * GIB,
            name="accelerator",
            major=6,
            minor=1,
            multi_processor_count=20,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    runtime = get_torch_runtime()

    assert runtime.cuda_capability == (6, 1)
    assert runtime.per_device_train_batch_size == 1
    assert not runtime.bf16
    assert not runtime.fp16
    assert not runtime.tf32
    assert not runtime.torch_compile
    assert runtime.optim == "adamw_torch"


def test_cuda_runtime_picks_best_visible_mixed_gpu(monkeypatch):
    set_devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    props = [
        SimpleNamespace(total_memory=16 * GIB, name="small", major=7, minor=5, multi_processor_count=40),
        SimpleNamespace(total_memory=180 * GIB, name="largest-newest", major=9, minor=0, multi_processor_count=120),
        SimpleNamespace(total_memory=80 * GIB, name="large", major=8, minor=0, multi_processor_count=108),
        SimpleNamespace(total_memory=24 * GIB, name="newer-small", major=8, minor=9, multi_processor_count=60),
    ]
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: props[index],
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device == torch.device("cuda", 1)
    assert not runtime.use_data_parallel
    assert set_devices == [1]
    assert "strategy=single_gpu" in accelerator_label(runtime)


def test_cuda_runtime_prefers_larger_fp16_device_over_smaller_fp16_device(monkeypatch):
    set_devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    props = [
        SimpleNamespace(total_memory=8 * GIB, name="legacy", major=6, minor=1, multi_processor_count=20),
        SimpleNamespace(total_memory=16 * GIB, name="compact-fp16", major=7, minor=5, multi_processor_count=40),
        SimpleNamespace(total_memory=16 * GIB, name="wide-fp16", major=7, minor=0, multi_processor_count=80),
    ]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props[index])
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    runtime = get_torch_runtime()

    assert runtime.device == torch.device("cuda", 2)
    assert runtime.cuda_capability == (7, 0)
    assert runtime.fp16
    assert not runtime.use_data_parallel
    assert set_devices == [2]


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

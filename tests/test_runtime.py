from __future__ import annotations

from types import SimpleNamespace

import torch

from genterp.runtime import GIB, accelerator_label, batch_size_for_cuda_memory, get_torch_runtime, should_launch_distributed


def test_cuda_batch_size_scales_with_visible_memory():
    assert batch_size_for_cuda_memory(12 * GIB) == 1
    assert batch_size_for_cuda_memory(20 * GIB) == 2
    assert batch_size_for_cuda_memory(40 * GIB) == 4
    assert batch_size_for_cuda_memory(64 * GIB) == 8
    assert batch_size_for_cuda_memory(80 * GIB) == 16
    assert batch_size_for_cuda_memory(180 * GIB) == 32


def test_cuda_runtime_uses_bf16_when_supported(monkeypatch):
    set_devices = []
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=180 * GIB, name="NVIDIA H200", major=9, minor=0),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device.type == "cuda"
    assert runtime.device.index == 0
    assert runtime.cuda_device_count == 8
    assert runtime.cuda_name == "NVIDIA H200"
    assert runtime.cuda_capability == (9, 0)
    assert runtime.per_device_train_batch_size == 32
    assert runtime.bf16
    assert not runtime.fp16
    assert runtime.tf32
    assert runtime.torch_compile
    assert runtime.torch_compile_mode == "max-autotune"
    assert runtime.optim == "adamw_torch_fused"
    assert runtime.ddp_find_unused_parameters is False
    assert runtime.ddp_bucket_cap_mb == 256
    assert set_devices == [0]
    assert "visible_gpus=8" in accelerator_label(runtime)


def test_cuda_runtime_uses_fp16_when_bf16_is_not_supported(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
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


def test_cuda_runtime_binds_local_rank(monkeypatch):
    set_devices = []
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=80 * GIB, name=f"GPU {index}", major=8, minor=0),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = get_torch_runtime()

    assert runtime.device == torch.device("cuda", 3)
    assert runtime.cuda_name == "GPU 3"
    assert set_devices == [3]


def test_should_launch_distributed_for_parent_with_multiple_cuda_gpus(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert should_launch_distributed()


def test_should_not_launch_distributed_inside_worker(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert not should_launch_distributed()


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

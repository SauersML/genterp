"""Torch accelerator/runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

GIB = 1024**3


@dataclass(frozen=True)
class TorchRuntime:
    device: torch.device
    per_device_train_batch_size: int
    bf16: bool
    fp16: bool
    tf32: bool
    torch_compile: bool
    optim: str
    dataloader_pin_memory: bool
    auto_find_batch_size: bool


def batch_size_for_cuda_memory(total_memory: int) -> int:
    if total_memory < 16 * GIB:
        return 1
    if total_memory < 24 * GIB:
        return 2
    if total_memory < 48 * GIB:
        return 4
    if total_memory < 80 * GIB:
        return 8
    if total_memory < 120 * GIB:
        return 16
    return 32


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def _cuda_bf16_supported() -> bool:
    return bool(torch.cuda.is_bf16_supported())


def get_torch_runtime() -> TorchRuntime:
    if torch.cuda.is_available():
        props: Any = torch.cuda.get_device_properties(torch.cuda.current_device())
        bf16 = _cuda_bf16_supported()
        return TorchRuntime(
            device=torch.device("cuda"),
            per_device_train_batch_size=batch_size_for_cuda_memory(props.total_memory),
            bf16=bf16,
            fp16=not bf16,
            tf32=True,
            torch_compile=True,
            optim="adamw_torch_fused",
            dataloader_pin_memory=True,
            auto_find_batch_size=True,
        )
    if _mps_available():
        return TorchRuntime(
            device=torch.device("mps"),
            per_device_train_batch_size=2,
            bf16=False,
            fp16=False,
            tf32=False,
            torch_compile=False,
            optim="adamw_torch",
            dataloader_pin_memory=False,
            auto_find_batch_size=False,
        )
    return TorchRuntime(
        device=torch.device("cpu"),
        per_device_train_batch_size=1,
        bf16=False,
        fp16=False,
        tf32=False,
        torch_compile=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        auto_find_batch_size=False,
    )


def configure_torch_runtime(runtime: TorchRuntime | None = None) -> TorchRuntime:
    runtime = runtime or get_torch_runtime()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = runtime.tf32
    torch.backends.cudnn.allow_tf32 = runtime.tf32
    torch.backends.cudnn.benchmark = runtime.device.type == "cuda"
    return runtime


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    non_blocking = device.type == "cuda"
    return {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

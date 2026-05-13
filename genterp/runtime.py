"""Torch accelerator/runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

GIB = 1024**3


@dataclass(frozen=True)
class TorchRuntime:
    device: torch.device
    cuda_device_count: int
    cuda_name: str | None
    cuda_capability: tuple[int, int] | None
    per_device_train_batch_size: int
    bf16: bool
    fp16: bool
    tf32: bool
    torch_compile: bool
    torch_compile_backend: str | None
    torch_compile_mode: str | None
    optim: str
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    dataloader_prefetch_factor: int | None
    ddp_find_unused_parameters: bool | None
    ddp_bucket_cap_mb: int | None
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


def is_distributed_worker() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def should_launch_distributed() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 1 and not is_distributed_worker()


def _cuda_device_index(device_count: int) -> int:
    return torch.cuda.current_device()


def _cuda_capability(index: int, props: Any) -> tuple[int, int]:
    major = getattr(props, "major", None)
    minor = getattr(props, "minor", None)
    if major is not None and minor is not None:
        return int(major), int(minor)
    capability = torch.cuda.get_device_capability(index)
    return int(capability[0]), int(capability[1])


def _cuda_dataloader_workers(device_count: int) -> int:
    cpu_count = os.cpu_count() or 8
    workers_per_rank = max(2, cpu_count // max(device_count, 1))
    return min(8, workers_per_rank)


def get_torch_runtime() -> TorchRuntime:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = _cuda_device_index(device_count)
        torch.cuda.set_device(device_index)
        props: Any = torch.cuda.get_device_properties(device_index)
        capability = _cuda_capability(device_index, props)
        name = str(getattr(props, "name", "CUDA"))
        bf16 = _cuda_bf16_supported()
        tf32 = capability[0] >= 8
        compile_model = capability[0] >= 8
        large_hopper = capability[0] >= 9
        return TorchRuntime(
            device=torch.device("cuda", device_index),
            cuda_device_count=device_count,
            cuda_name=name,
            cuda_capability=capability,
            per_device_train_batch_size=batch_size_for_cuda_memory(props.total_memory),
            bf16=bf16,
            fp16=not bf16,
            tf32=tf32,
            torch_compile=compile_model,
            torch_compile_backend="inductor" if compile_model else None,
            torch_compile_mode="max-autotune" if large_hopper else "reduce-overhead" if compile_model else None,
            optim="adamw_torch_fused",
            dataloader_num_workers=_cuda_dataloader_workers(device_count),
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4,
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=256 if large_hopper else 100,
            auto_find_batch_size=True,
        )
    if _mps_available():
        return TorchRuntime(
            device=torch.device("mps"),
            cuda_device_count=0,
            cuda_name=None,
            cuda_capability=None,
            per_device_train_batch_size=2,
            bf16=False,
            fp16=False,
            tf32=False,
            torch_compile=False,
            torch_compile_backend=None,
            torch_compile_mode=None,
            optim="adamw_torch",
            dataloader_num_workers=4,
            dataloader_pin_memory=False,
            dataloader_prefetch_factor=2,
            ddp_find_unused_parameters=None,
            ddp_bucket_cap_mb=None,
            auto_find_batch_size=False,
        )
    return TorchRuntime(
        device=torch.device("cpu"),
        cuda_device_count=0,
        cuda_name=None,
        cuda_capability=None,
        per_device_train_batch_size=1,
        bf16=False,
        fp16=False,
        tf32=False,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        optim="adamw_torch",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=None,
        ddp_bucket_cap_mb=None,
        auto_find_batch_size=False,
    )


def configure_torch_runtime(runtime: TorchRuntime | None = None) -> TorchRuntime:
    runtime = runtime or get_torch_runtime()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = runtime.tf32
    torch.backends.cudnn.allow_tf32 = runtime.tf32
    torch.backends.cudnn.benchmark = runtime.device.type == "cuda"
    if runtime.device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    return runtime


def accelerator_label(runtime: TorchRuntime) -> str:
    if runtime.device.type != "cuda":
        return runtime.device.type
    capability = runtime.cuda_capability or (0, 0)
    return (
        f"{runtime.device.type}:{runtime.device.index} {runtime.cuda_name} "
        f"cc={capability[0]}.{capability[1]} visible_gpus={runtime.cuda_device_count}"
    )


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    non_blocking = device.type == "cuda"
    return {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

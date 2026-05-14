"""Torch accelerator/runtime selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from genterp.progress import ProgressLogger

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
    use_data_parallel: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    dataloader_prefetch_factor: int | None
    auto_find_batch_size: bool


def batch_size_for_cuda_memory(total_memory: int) -> int:
    if total_memory < 14 * GIB:
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
    try:
        return bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        return False


def _cuda_fp16_supported(capability: tuple[int, int]) -> bool:
    return capability[0] >= 7


def _cuda_backend_is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _cuda_bf16_runtime_supported(capability: tuple[int, int]) -> bool:
    return capability[0] >= 8 and _cuda_bf16_supported()


def _cuda_capability(index: int, props: Any) -> tuple[int, int]:
    major = getattr(props, "major", None)
    minor = getattr(props, "minor", None)
    if major is not None and minor is not None:
        return int(major), int(minor)
    try:
        capability = torch.cuda.get_device_capability(index)
    except (AttributeError, RuntimeError):
        return 0, 0
    return int(capability[0]), int(capability[1])


def _cuda_sm_count(props: Any) -> int:
    return int(getattr(props, "multi_processor_count", 0))


def _cuda_precision_tier(capability: tuple[int, int]) -> int:
    if capability[0] >= 8:
        return 2
    if _cuda_fp16_supported(capability):
        return 1
    return 0


def _cuda_score(index: int, props: Any) -> tuple[int, int, int, int]:
    capability = _cuda_capability(index, props)
    return _cuda_precision_tier(capability), int(getattr(props, "total_memory", 0)), _cuda_sm_count(props), -index


def _cuda_devices_are_homogeneous(props: list[Any]) -> bool:
    if len(props) < 2:
        return False
    first_capability = _cuda_capability(0, props[0])
    first_memory = int(getattr(props[0], "total_memory", 0))
    first_sms = _cuda_sm_count(props[0])
    for index, item in enumerate(props[1:], start=1):
        if _cuda_capability(index, item) != first_capability:
            return False
        memory = int(getattr(item, "total_memory", 0))
        if abs(memory - first_memory) / max(first_memory, 1) > 0.05:
            return False
        sms = _cuda_sm_count(item)
        if sms and first_sms and abs(sms - first_sms) / max(first_sms, 1) > 0.05:
            return False
    return True


def _best_cuda_device(props: list[Any]) -> int:
    return max(range(len(props)), key=lambda index: _cuda_score(index, props[index]))


def _cuda_dataloader_workers(active_device_count: int) -> int:
    cpu_count = os.cpu_count() or 8
    if active_device_count > 1:
        return min(16, max(4, cpu_count // 2))
    return min(8, max(2, cpu_count // 4))


def _cuda_compile_supported(capability: tuple[int, int]) -> bool:
    return capability[0] >= 8 and hasattr(torch, "compile") and not _cuda_backend_is_rocm()


def _cuda_fused_optimizer_supported(capability: tuple[int, int]) -> bool:
    return _cuda_fp16_supported(capability) and not _cuda_backend_is_rocm()


def _set_attr_if_present(obj: object, attr: str, value: bool) -> bool:
    if not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
    except RuntimeError:
        return False
    return True


def _call_if_present(obj: object, name: str, *args: object) -> bool:
    fn = getattr(obj, name, None)
    if fn is None:
        return False
    try:
        fn(*args)
    except RuntimeError:
        return False
    return True


def get_torch_runtime() -> TorchRuntime:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return _non_cuda_runtime()
        all_props = [torch.cuda.get_device_properties(index) for index in range(device_count)]
        use_data_parallel = _cuda_devices_are_homogeneous(all_props)
        device_index = 0 if use_data_parallel else _best_cuda_device(all_props)
        torch.cuda.set_device(device_index)
        props: Any = all_props[device_index]
        capability = _cuda_capability(device_index, props)
        name = str(getattr(props, "name", "CUDA"))
        bf16 = _cuda_bf16_runtime_supported(capability)
        fp16 = not bf16 and _cuda_fp16_supported(capability)
        tf32 = capability[0] >= 8 and not _cuda_backend_is_rocm()
        compile_model = _cuda_compile_supported(capability)
        large_hopper = capability[0] >= 9
        fused_optimizer = _cuda_fused_optimizer_supported(capability)
        active_device_count = device_count if use_data_parallel else 1
        return TorchRuntime(
            device=torch.device("cuda", device_index),
            cuda_device_count=device_count,
            cuda_name=name,
            cuda_capability=capability,
            per_device_train_batch_size=batch_size_for_cuda_memory(int(getattr(props, "total_memory", 0))),
            bf16=bf16,
            fp16=fp16,
            tf32=tf32,
            torch_compile=compile_model,
            torch_compile_backend="inductor" if compile_model else None,
            torch_compile_mode="max-autotune" if large_hopper else "reduce-overhead" if compile_model else None,
            optim="adamw_torch_fused" if fused_optimizer else "adamw_torch",
            use_data_parallel=use_data_parallel,
            dataloader_num_workers=_cuda_dataloader_workers(active_device_count),
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4,
            auto_find_batch_size=True,
        )

    return _non_cuda_runtime()


def _non_cuda_runtime() -> TorchRuntime:
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
            use_data_parallel=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=False,
            dataloader_prefetch_factor=2,
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
        use_data_parallel=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
        auto_find_batch_size=False,
    )


def configure_torch_runtime(runtime: TorchRuntime | None = None) -> TorchRuntime:
    logger = ProgressLogger("torch_runtime", total_units=5)
    logger.start_unit("select torch runtime", "using provided runtime or probing local accelerator")
    runtime = runtime or get_torch_runtime()
    logger.finish_unit("select torch runtime", f"device={runtime.device} cuda_gpus={runtime.cuda_device_count}")

    logger.start_unit("set float32 matmul precision", "precision=high")
    torch.set_float32_matmul_precision("high")
    logger.finish_unit("set float32 matmul precision", "precision=high")

    logger.start_unit("configure TF32 flags", f"enabled={runtime.tf32}")
    cuda_backend = getattr(torch.backends, "cuda", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    matmul_set = _set_attr_if_present(matmul_backend, "allow_tf32", runtime.tf32) if matmul_backend is not None else False
    cudnn_set = _set_attr_if_present(cudnn_backend, "allow_tf32", runtime.tf32) if cudnn_backend is not None else False
    logger.finish_unit("configure TF32 flags", f"enabled={runtime.tf32} matmul={matmul_set} cudnn={cudnn_set}")

    logger.start_unit("configure cuDNN benchmark", f"enabled={runtime.device.type == 'cuda'}")
    benchmark_set = _set_attr_if_present(cudnn_backend, "benchmark", runtime.device.type == "cuda") if cudnn_backend is not None else False
    logger.finish_unit("configure cuDNN benchmark", f"enabled={runtime.device.type == 'cuda'} set={benchmark_set}")

    logger.start_unit("configure CUDA scaled-dot-product attention kernels", f"device_type={runtime.device.type}")
    if runtime.device.type == "cuda" and cuda_backend is not None:
        flash = _call_if_present(cuda_backend, "enable_flash_sdp", True)
        mem_efficient = _call_if_present(cuda_backend, "enable_mem_efficient_sdp", True)
        math = _call_if_present(cuda_backend, "enable_math_sdp", True)
        logger.finish_unit(
            "configure CUDA scaled-dot-product attention kernels",
            f"flash={flash} mem_efficient={mem_efficient} math={math}",
        )
    else:
        logger.finish_unit("configure CUDA scaled-dot-product attention kernels", "skipped for non-CUDA runtime")
    return runtime


def accelerator_label(runtime: TorchRuntime) -> str:
    if runtime.device.type != "cuda":
        return runtime.device.type
    capability = runtime.cuda_capability or (0, 0)
    return (
        f"{runtime.device.type}:{runtime.device.index} {runtime.cuda_name} "
        f"cc={capability[0]}.{capability[1]} visible_gpus={runtime.cuda_device_count} "
        f"strategy={'data_parallel' if runtime.use_data_parallel else 'single_gpu'}"
    )


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    logger = ProgressLogger("device_transfer", total_units=len(batch))
    non_blocking = device.type == "cuda"
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.start_unit("move tensor to device", f"key={key} shape={tuple(value.shape)} device={device}")
            moved[key] = value.to(device, non_blocking=non_blocking)
            logger.finish_unit("move tensor to device", f"key={key} device={device}")
        else:
            logger.start_unit("leave non-tensor batch field on host", f"key={key} type={type(value).__name__}")
            moved[key] = value
            logger.finish_unit("leave non-tensor batch field on host", f"key={key}")
    return moved

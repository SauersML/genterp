"""Train Genterp on AoU OMOP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint

from genterp.data import AtomVocab, CodeAtomMap, CohortDataset, collate
from genterp.modeling import Genterp, GenterpConfig
from genterp.runtime import TorchRuntime, accelerator_label, configure_torch_runtime

RUNTIME_STATE_FILE = "genterp_runtime.json"


class GenterpHFConfig(transformers.PretrainedConfig):
    model_type = "genterp"

    def __init__(self, **kwargs):
        self.genterp_cfg: dict = kwargs.pop("genterp_cfg", {})
        super().__init__(**kwargs)


class GenterpForCausalLM(transformers.PreTrainedModel):
    config_class = GenterpHFConfig
    main_input_name = "event_atoms"

    def __init__(self, config: GenterpHFConfig):
        super().__init__(config)
        self.model = Genterp(GenterpConfig(**config.genterp_cfg))

    def forward(self, **batch: torch.Tensor) -> transformers.modeling_outputs.CausalLMOutput:
        ld = self.model.loss(**batch)
        return transformers.modeling_outputs.CausalLMOutput(loss=ld["loss"], logits=ld["loss"].detach().reshape(1))


class RuntimeStateCallback(transformers.TrainerCallback):
    def __init__(self, runtime: TorchRuntime):
        self.runtime = runtime

    def on_save(self, args, state, control, **kwargs):
        write_runtime_state(Path(args.output_dir) / f"checkpoint-{state.global_step}", self.runtime)
        return control


class GenterpTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        runtime: TorchRuntime | None = None,
        reset_training_state_on_resume: bool = False,
        **kwargs,
    ):
        self.runtime = runtime
        self.reset_training_state_on_resume = reset_training_state_on_resume
        super().__init__(*args, **kwargs)
        if runtime is not None and runtime.device.type == "cuda" and not runtime.use_data_parallel:
            self.args._n_gpu = 1

    def _get_train_sampler(self, train_dataset=None) -> torch.utils.data.Sampler | None:
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        if (
            train_dataset is not None
            and self.args.train_sampling_strategy == "group_by_length"
            and hasattr(train_dataset, "lengths")
        ):
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
            return LengthGroupedSampler(
                batch_size,
                lengths=train_dataset.lengths,
            )
        return super()._get_train_sampler(train_dataset)

    def _prepare_input(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            device = self.runtime.device if self.runtime is not None else self.args.device
            return data.to(device, non_blocking=device.type == "cuda")
        if isinstance(data, Mapping):
            return type(data)((key, self._prepare_input(value)) for key, value in data.items())
        if isinstance(data, tuple):
            return tuple(self._prepare_input(value) for value in data)
        if isinstance(data, list):
            return [self._prepare_input(value) for value in data]
        return data

    def _move_model_to_device(self, model: torch.nn.Module, device: torch.device) -> None:
        if self.runtime is not None:
            device = self.runtime.device
        super()._move_model_to_device(model, device)

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        if checkpoint is not None and self.reset_training_state_on_resume:
            return
        super()._load_optimizer_and_scheduler(checkpoint)

    def _load_scaler(self, checkpoint: str | None) -> None:
        if checkpoint is not None and self.reset_training_state_on_resume:
            return
        super()._load_scaler(checkpoint)


def latest_checkpoint(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


def final_model_path(output_dir: str | Path) -> str | None:
    final_dir = Path(output_dir) / "final"
    if (final_dir / "config.json").is_file():
        return str(final_dir)
    return None


def runtime_state(runtime: TorchRuntime) -> dict[str, object]:
    return {
        "device_type": runtime.device.type,
        "cuda_capability": list(runtime.cuda_capability) if runtime.cuda_capability is not None else None,
        "precision": "bf16" if runtime.bf16 else "fp16" if runtime.fp16 else "fp32",
        "tf32": runtime.tf32,
        "torch_compile": runtime.torch_compile,
        "torch_compile_backend": runtime.torch_compile_backend,
        "torch_compile_mode": runtime.torch_compile_mode,
        "optim": runtime.optim,
        "use_data_parallel": runtime.use_data_parallel,
    }


def write_runtime_state(path: str | Path, runtime: TorchRuntime) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / RUNTIME_STATE_FILE).write_text(json.dumps(runtime_state(runtime), indent=2, sort_keys=True))


def checkpoint_runtime_state(path: str | Path) -> dict[str, object] | None:
    state_path = Path(path) / RUNTIME_STATE_FILE
    if not state_path.is_file():
        return None
    return dict(json.loads(state_path.read_text()))


def checkpoint_matches_runtime(path: str | Path, runtime: TorchRuntime) -> bool:
    return checkpoint_runtime_state(path) == runtime_state(runtime)


def _load_value_stats(path: Path, vocab: AtomVocab) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(vocab)
    mu = torch.zeros(n)
    sigma = torch.ones(n)
    has_mag = torch.zeros(n, dtype=torch.bool)
    for code, s in json.loads(path.read_text()).items():
        a = vocab.encode(code)
        if a == 0:
            continue
        mu[a] = float(s["mu"])
        sigma[a] = max(float(s["sigma"]), 1e-6)
        has_mag[a] = True
    return mu, sigma, has_mag


def main() -> None:
    runtime = configure_torch_runtime()
    if runtime.device.type != "cuda" or runtime.device.index in (None, 0):
        print(f"genterp train accelerator={accelerator_label(runtime)} batch_per_device={runtime.per_device_train_batch_size}")
    etl = Path.home() / "genterp" / "etl"
    output_dir = Path.home() / "genterp" / "runs"
    vocab = AtomVocab(dict(json.loads((etl / "vocab.json").read_text())))
    dataset = CohortDataset(etl, CodeAtomMap.from_vocab(vocab))

    cfg = GenterpConfig(n_atoms=len(vocab), dim=512, n_heads=8, n_layers=8)
    resume_checkpoint = latest_checkpoint(output_dir)
    reset_training_state = resume_checkpoint is not None and not checkpoint_matches_runtime(resume_checkpoint, runtime)
    warm_start_path = final_model_path(output_dir) if resume_checkpoint is None else None
    if warm_start_path is not None:
        model = GenterpForCausalLM.from_pretrained(warm_start_path)
    else:
        model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))
    if reset_training_state:
        print("genterp train hardware profile changed; resuming model weights and rebuilding optimizer state")
    mu, sigma, has_mag = _load_value_stats(etl / "value_stats.json", vocab)
    model.model.value_mod.set_stats(mu, sigma, has_mag)

    training_args = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=runtime.per_device_train_batch_size,
        learning_rate=3e-4,
        warmup_steps=500,
        max_steps=50_000,
        lr_scheduler_type="cosine",
        bf16=runtime.bf16,
        fp16=runtime.fp16,
        tf32=runtime.tf32,
        torch_compile=runtime.torch_compile,
        save_steps=2_000,
        logging_steps=50,
        optim=runtime.optim,
        dataloader_num_workers=runtime.dataloader_num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=runtime.dataloader_pin_memory,
        dataloader_prefetch_factor=runtime.dataloader_prefetch_factor,
        dataloader_drop_last=True,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        remove_unused_columns=False,
        report_to="none",
        skip_memory_metrics=True,
        restore_callback_states_from_checkpoint=True,
        save_safetensors=True,
        auto_find_batch_size=runtime.auto_find_batch_size,
    )
    if runtime.torch_compile:
        training_args["torch_compile_backend"] = runtime.torch_compile_backend
        training_args["torch_compile_mode"] = runtime.torch_compile_mode

    trainer = GenterpTrainer(
        model=model,
        args=transformers.TrainingArguments(**training_args),
        train_dataset=dataset,
        data_collator=collate,
        runtime=runtime,
        reset_training_state_on_resume=reset_training_state,
        callbacks=[RuntimeStateCallback(runtime)],
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    write_runtime_state(output_dir / "final", runtime)


if __name__ == "__main__":
    main()

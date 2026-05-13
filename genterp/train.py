"""Train Genterp on AoU OMOP."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint

from genterp.data import AtomVocab, CodeAtomMap, CohortDataset, collate
from genterp.modeling import Genterp, GenterpConfig
from genterp.runtime import accelerator_label, configure_torch_runtime, should_launch_distributed


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


class GenterpTrainer(transformers.Trainer):
    def _get_train_sampler(self, train_dataset=None) -> torch.utils.data.Sampler | None:
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        if (
            train_dataset is not None
            and self.args.train_sampling_strategy == "group_by_length"
            and hasattr(train_dataset, "lengths")
        ):
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
            if self.args.world_size > 1:
                return DistributedLengthGroupedSampler(
                    batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                    drop_last=self.args.dataloader_drop_last,
                    lengths=train_dataset.lengths,
                )
            return LengthGroupedSampler(
                batch_size,
                lengths=train_dataset.lengths,
            )
        return super()._get_train_sampler(train_dataset)

    def _prepare_input(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.args.device, non_blocking=self.args.device.type == "cuda")
        if isinstance(data, Mapping):
            return type(data)((key, self._prepare_input(value)) for key, value in data.items())
        if isinstance(data, tuple):
            return tuple(self._prepare_input(value) for value in data)
        if isinstance(data, list):
            return [self._prepare_input(value) for value in data]
        return data


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


def distributed_launch_command(module: str = "genterp.train") -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={torch.cuda.device_count()}",
        "-m",
        module,
    ]


def launch_distributed_if_needed() -> bool:
    if not should_launch_distributed():
        return False
    command = distributed_launch_command()
    print(f"genterp train launching {torch.cuda.device_count()} GPU workers automatically")
    raise SystemExit(subprocess.run(command).returncode)


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
    warm_start_path = resume_checkpoint or final_model_path(output_dir)
    if warm_start_path is not None and resume_checkpoint is None:
        model = GenterpForCausalLM.from_pretrained(warm_start_path)
    else:
        model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))
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
        ddp_find_unused_parameters=runtime.ddp_find_unused_parameters,
        ddp_bucket_cap_mb=runtime.ddp_bucket_cap_mb,
    )
    if runtime.torch_compile:
        training_args["torch_compile_backend"] = runtime.torch_compile_backend
        training_args["torch_compile_mode"] = runtime.torch_compile_mode

    trainer = GenterpTrainer(
        model=model,
        args=transformers.TrainingArguments(**training_args),
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    launch_distributed_if_needed()
    main()

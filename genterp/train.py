"""Train Genterp on AoU OMOP."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import trainer as hf_trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

from genterp.data import AtomVocab, CohortDataset, collate
from genterp.modeling import Genterp, GenterpConfig
from genterp.progress import ProgressLogger, count_parameters
from genterp.runtime import TorchRuntime, accelerator_label, configure_torch_runtime

RUNTIME_STATE_FILE = "genterp_runtime.json"
FINAL_POINTER_FILE = "final_checkpoint.json"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
MAX_STEPS = 50_000
WARMUP_STEPS = 500
WSD_DECAY_STEPS = MAX_STEPS // 10


class GenterpHFConfig(transformers.PretrainedConfig):
    model_type = "genterp"

    def __init__(self, **kwargs):
        self.genterp_cfg: dict = kwargs.pop("genterp_cfg", {})
        super().__init__(**kwargs)


class GenterpForCausalLM(transformers.PreTrainedModel):
    config_class = GenterpHFConfig
    main_input_name = "event_atoms"
    supports_gradient_checkpointing = True
    # tpp.mark_out shares storage with the atom embedding (input/output weight tie).
    # transformers 5.x expects a dict mapping tied → source so save_pretrained can
    # dedup on disk and re-tie on load.
    _tied_weights_keys = {"model.tpp.mark_out.weight": "model.embed.embedding.weight"}
    all_tied_weights_keys = _tied_weights_keys

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
        logger = ProgressLogger("trainer_save", total_units=1)
        logger.start_unit("write checkpoint runtime profile", f"checkpoint=checkpoint-{state.global_step}")
        write_runtime_state(Path(args.output_dir) / f"checkpoint-{state.global_step}", self.runtime)
        logger.finish_unit("write checkpoint runtime profile", f"global_step={state.global_step:,}")
        return control


class VerboseTrainerProgressCallback(transformers.TrainerCallback):
    def __init__(self) -> None:
        self.logger = ProgressLogger("trainer", total_units=None)

    def on_train_begin(self, args, state, control, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log(
            "training loop begins",
            f"max_steps={total:,} batch_per_device={args.per_device_train_batch_size} "
            f"grad_accum={args.gradient_accumulation_steps} logging_steps={args.logging_steps} "
            f"save_steps={args.save_steps} eval_steps={args.eval_steps}",
        )
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log(
            "training step starts",
            f"next_step={state.global_step + 1:,}; Trainer will load a batch, run forward loss, "
            "backpropagate, clip/update gradients, and advance scheduler",
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log("training step complete", f"global_step={state.global_step:,}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        metrics = ", ".join(f"{key}={value}" for key, value in sorted((logs or {}).items()))
        self.logger.log("training metrics emitted", metrics or "no metrics payload")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        payload = ", ".join(f"{key}={value}" for key, value in sorted((metrics or {}).items()))
        self.logger.log("evaluation complete", payload or "no eval metrics payload")
        return control

    def on_save(self, args, state, control, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log("checkpoint save complete", f"checkpoint=checkpoint-{state.global_step}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log("training loop ends", f"global_step={state.global_step:,}")
        return control


def _limit_dataloader_worker_threads() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)


def _seed_and_limit_train_worker(worker_id: int, *, num_workers: int, rank: int) -> None:
    _limit_dataloader_worker_threads()
    hf_trainer.seed_worker(worker_id, num_workers=num_workers, rank=rank)


def _limit_eval_worker(worker_id: int) -> None:
    del worker_id
    _limit_dataloader_worker_threads()


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

    def _get_eval_sampler(self, eval_dataset) -> torch.utils.data.Sampler | None:
        """Mirror the train sampler: feed our pre-computed lengths so LengthGroupedSampler
        doesn't try to auto-infer them by probing ``dataset[0]['input_ids']`` (we don't have
        that key — our items are plain dicts of tensors keyed by event/static fields)."""
        if (
            eval_dataset is not None
            and self.args.train_sampling_strategy == "group_by_length"
            and hasattr(eval_dataset, "lengths")
        ):
            return LengthGroupedSampler(self.args.eval_batch_size, lengths=eval_dataset.lengths)
        return super()._get_eval_sampler(eval_dataset)

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Callable[[Dataset], torch.utils.data.Sampler] | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        data_collator = self.data_collator
        if hf_trainer.is_datasets_available() and isinstance(dataset, hf_trainer.datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        worker_init_fn = None
        if self.args.dataloader_num_workers > 0:
            worker_init_fn = (
                partial(
                    _seed_and_limit_train_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
                if is_training
                else _limit_eval_worker
            )

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
            "worker_init_fn": worker_init_fn,
        }

        if not isinstance(dataset, IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

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
    checkpoints = []
    for path in output_dir.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if path.is_dir() and match and checkpoint_is_complete(path):
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    return str(max(checkpoints)[1])


def final_model_path(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    pointer = output_dir / FINAL_POINTER_FILE
    if pointer.is_file():
        try:
            final_dir = output_dir / str(json.loads(pointer.read_text())["path"])
        except (KeyError, TypeError, json.JSONDecodeError):
            final_dir = output_dir / "final"
        if model_dir_is_complete(final_dir):
            return str(final_dir)
    final_dir = output_dir / "final"
    if model_dir_is_complete(final_dir):
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
    logger = ProgressLogger("runtime_state", total_units=2)
    logger.start_unit("prepare runtime state directory", f"path={path}")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.finish_unit("prepare runtime state directory", f"path={path}")
    logger.start_unit("write runtime state json", f"file={path / RUNTIME_STATE_FILE}")
    atomic_write_json(path / RUNTIME_STATE_FILE, runtime_state(runtime))
    logger.finish_unit("write runtime state json", f"file={path / RUNTIME_STATE_FILE}")


def checkpoint_runtime_state(path: str | Path) -> dict[str, object] | None:
    state_path = Path(path) / RUNTIME_STATE_FILE
    if not state_path.is_file():
        return None
    try:
        return dict(json.loads(state_path.read_text()))
    except json.JSONDecodeError:
        return None


def checkpoint_matches_runtime(path: str | Path, runtime: TorchRuntime) -> bool:
    return checkpoint_runtime_state(path) == runtime_state(runtime)


def atomic_write_json(path: str | Path, data: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def model_dir_is_complete(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    )


def checkpoint_is_complete(path: str | Path) -> bool:
    path = Path(path)
    return (
        model_dir_is_complete(path)
        and (path / "trainer_state.json").is_file()
        and ((path / "optimizer.pt").is_file() or (path / "optimizer.bin").is_file())
        and (path / "scheduler.pt").is_file()
        and checkpoint_runtime_state(path) is not None
    )


def save_final_model(trainer: transformers.Trainer, output_dir: str | Path, runtime: TorchRuntime) -> None:
    logger = ProgressLogger("final_save", total_units=5)
    output_dir = Path(output_dir)
    final_name = f"final-{trainer.state.global_step}-{uuid.uuid4().hex[:12]}"
    tmp_dir = output_dir / f".{final_name}.tmp"
    final_dir = output_dir / final_name
    logger.start_unit("prepare final model directories", f"tmp_dir={tmp_dir} final_dir={final_dir}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    logger.finish_unit("prepare final model directories", f"tmp_exists_after_cleanup={tmp_dir.exists()}")

    logger.start_unit("save model to temporary final directory", f"global_step={trainer.state.global_step:,}")
    trainer.save_model(str(tmp_dir))
    logger.finish_unit("save model to temporary final directory", f"tmp_dir={tmp_dir}")

    logger.start_unit("write runtime profile beside final model", f"tmp_dir={tmp_dir}")
    write_runtime_state(tmp_dir, runtime)
    logger.finish_unit("write runtime profile beside final model", f"tmp_dir={tmp_dir}")

    logger.start_unit("publish final model directory atomically", f"from={tmp_dir} to={final_dir}")
    tmp_dir.replace(final_dir)
    logger.finish_unit("publish final model directory atomically", f"final_dir={final_dir}")

    logger.start_unit("write final checkpoint pointer", f"file={output_dir / FINAL_POINTER_FILE}")
    atomic_write_json(output_dir / FINAL_POINTER_FILE, {"path": final_dir.name})
    logger.finish_unit("write final checkpoint pointer", f"path={final_dir.name}")


def _load_value_stats(path: Path, vocab: AtomVocab) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger = ProgressLogger("value_stats", total_units=3)
    logger.start_unit("initialize default value statistics", f"n_atoms={len(vocab):,}")
    n = len(vocab)
    mu = torch.zeros(n)
    sigma = torch.ones(n)
    has_mag = torch.zeros(n, dtype=torch.bool)
    logger.finish_unit("initialize default value statistics", f"n_atoms={n:,}")

    logger.start_unit("read value stats json", f"path={path}")
    payload = json.loads(path.read_text())
    logger.finish_unit("read value stats json", f"codes_with_stats={len(payload):,}")

    logger.start_unit("map value stats to atom tensors", "unknown or PAD codes are skipped")
    skipped = 0
    for code, s in payload.items():
        a = vocab.encode(code)
        if a == 0:
            skipped += 1
            continue
        mu[a] = float(s["mu"])
        sigma[a] = max(float(s["sigma"]), 1e-6)
        has_mag[a] = True
    logger.finish_unit("map value stats to atom tensors", f"magnitude_atoms={int(has_mag.sum().item()):,} skipped={skipped:,}")
    return mu, sigma, has_mag


def gradient_checkpointing_enabled(runtime: TorchRuntime) -> bool:
    return runtime.device.type == "cuda" and runtime.auto_find_batch_size


def build_training_args(output_dir: str | Path, runtime: TorchRuntime) -> dict[str, object]:
    use_gradient_checkpointing = gradient_checkpointing_enabled(runtime)
    training_args: dict[str, object] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": runtime.per_device_train_batch_size,
        "per_device_eval_batch_size": runtime.per_device_train_batch_size,
        "learning_rate": 3e-4,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "lr_scheduler_type": "warmup_stable_decay",
        "lr_scheduler_kwargs": {
            "num_decay_steps": WSD_DECAY_STEPS,
            "decay_type": "linear",
            "min_lr_ratio": 0.0,
        },
        "bf16": runtime.bf16,
        "fp16": runtime.fp16,
        "tf32": runtime.tf32,
        "torch_compile": runtime.torch_compile,
        "save_strategy": "steps",
        "save_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "prediction_loss_only": True,
        "logging_steps": 1,
        "logging_first_step": True,
        "optim": runtime.optim,
        "dataloader_num_workers": runtime.dataloader_num_workers,
        "dataloader_persistent_workers": True,
        "dataloader_pin_memory": runtime.dataloader_pin_memory,
        "dataloader_prefetch_factor": runtime.dataloader_prefetch_factor,
        "dataloader_drop_last": True,
        "train_sampling_strategy": "group_by_length",
        "length_column_name": "length",
        "remove_unused_columns": False,
        "report_to": "none",
        "skip_memory_metrics": True,
        "restore_callback_states_from_checkpoint": True,
        # save_safetensors removed in transformers 5.x — safetensors is the only format now.
        "auto_find_batch_size": runtime.auto_find_batch_size,
        "gradient_checkpointing": use_gradient_checkpointing,
    }
    if use_gradient_checkpointing:
        training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if runtime.torch_compile:
        training_args["torch_compile_backend"] = runtime.torch_compile_backend
        training_args["torch_compile_mode"] = runtime.torch_compile_mode
    return training_args


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Genterp on AoU OMOP.")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Build a ~1000× smaller model (dim=32, heads=2, layers=2) for quick iteration. "
        "Pair with `scripts.aou_etl --tiny` to also use the 10× downsampled cohort.",
    )
    args = parser.parse_args(argv)
    tiny = args.tiny

    setup = ProgressLogger("train_setup", total_units=15)
    setup.start_unit("configure torch runtime", "selecting accelerator, precision, optimizer, and dataloader settings")
    runtime = configure_torch_runtime()
    setup.finish_unit(
        "configure torch runtime",
        f"accelerator={accelerator_label(runtime)} batch_per_device={runtime.per_device_train_batch_size}",
    )
    if runtime.device.type != "cuda" or runtime.device.index in (None, 0):
        print(f"genterp train accelerator={accelerator_label(runtime)} batch_per_device={runtime.per_device_train_batch_size}")

    setup.start_unit("resolve ETL and output directories", "tiny runs land under runs-tiny so they don't warm-start the full pipeline")
    etl = Path.home() / "genterp" / "etl"
    output_dir = Path.home() / "genterp" / ("runs-tiny" if tiny else "runs")
    setup.finish_unit("resolve ETL and output directories", f"etl={etl} output_dir={output_dir}")

    setup.start_unit("load vocabulary", f"path={etl / 'vocab.json'}")
    vocab = AtomVocab(dict(json.loads((etl / "vocab.json").read_text())))
    setup.finish_unit("load vocabulary", f"atoms={len(vocab):,} mapped_codes={len(vocab.code_to_atom):,}")

    setup.start_unit("build training dataset", "split=train")
    train_dataset = CohortDataset(etl, split="train")
    setup.finish_unit("build training dataset", f"subjects={len(train_dataset):,}")

    setup.start_unit("build evaluation dataset", "split=test")
    eval_dataset = CohortDataset(etl, split="test")
    setup.finish_unit("build evaluation dataset", f"subjects={len(eval_dataset):,}")

    if tiny:
        # ~1000× fewer transformer params: dim 32 vs 512 (16×), 2 vs 8 layers (4×).
        # Per-layer dense weights scale as L·dim² → 4·256 = 1024×.
        setup.start_unit("construct model config", "--tiny → dim=32 heads=2 layers=2")
        cfg = GenterpConfig(n_atoms=len(vocab), dim=32, n_heads=2, n_layers=2)
    else:
        setup.start_unit("construct model config", "dim=512 heads=8 layers=8")
        cfg = GenterpConfig(n_atoms=len(vocab), dim=512, n_heads=8, n_layers=8)
    setup.finish_unit("construct model config", f"n_atoms={cfg.n_atoms:,} dim={cfg.dim} layers={cfg.n_layers}")

    setup.start_unit("inspect checkpoints", f"output_dir={output_dir}")
    resume_checkpoint = latest_checkpoint(output_dir)
    reset_training_state = resume_checkpoint is not None and not checkpoint_matches_runtime(resume_checkpoint, runtime)
    warm_start_path = final_model_path(output_dir) if resume_checkpoint is None else None
    setup.finish_unit(
        "inspect checkpoints",
        f"resume_checkpoint={resume_checkpoint} warm_start_path={warm_start_path} "
        f"reset_training_state={reset_training_state}",
    )

    setup.start_unit("load or initialize model", "preferring resume checkpoint, then previous final model, then fresh init")
    if warm_start_path is not None:
        model = GenterpForCausalLM.from_pretrained(warm_start_path)
        model_source = f"warm_start={warm_start_path}"
    else:
        model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))
        model_source = "fresh_init" if resume_checkpoint is None else f"resume_weights_from={resume_checkpoint}"
    if reset_training_state:
        print("genterp train hardware profile changed; resuming model weights and rebuilding optimizer state")
    setup.finish_unit("load or initialize model", f"{model_source} params={count_parameters(model):,}")

    setup.start_unit("configure mark negative sampler", "frequency-weighted atom negatives from training events")
    model.model.tpp.set_mark_noise_distribution(torch.from_numpy(train_dataset.atom_counts(model.model.cfg.n_atoms)))
    setup.finish_unit("configure mark negative sampler", f"negatives={model.model.cfg.sampled_mark_negatives:,}")

    setup.start_unit("load value modulation stats", f"path={etl / 'value_stats.json'}")
    mu, sigma, has_mag = _load_value_stats(etl / "value_stats.json", vocab)
    setup.finish_unit("load value modulation stats", f"magnitude_atoms={int(has_mag.sum().item()):,}")

    setup.start_unit("apply value modulation stats", "copying mu/sigma/has_magnitude tensors into model buffers")
    model.model.value_mod.set_stats(mu, sigma, has_mag)
    setup.finish_unit("apply value modulation stats", f"magnitude_atoms={int(has_mag.sum().item()):,}")

    setup.start_unit("build training arguments", "max_steps=50000 with per-step logging and checkpoint saves")
    training_args = build_training_args(output_dir, runtime)
    setup.finish_unit(
        "build training arguments",
        f"max_steps={training_args['max_steps']:,} logging_steps={training_args['logging_steps']} "
        f"save_steps={training_args['save_steps']} eval_steps={training_args['eval_steps']} "
        f"lr_scheduler={training_args['lr_scheduler_type']} gradient_checkpointing={training_args['gradient_checkpointing']}",
    )

    setup.start_unit("instantiate Trainer", "attaching runtime-state and verbose progress callbacks")
    trainer = GenterpTrainer(
        model=model,
        args=transformers.TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        runtime=runtime,
        reset_training_state_on_resume=reset_training_state,
        callbacks=[RuntimeStateCallback(runtime), VerboseTrainerProgressCallback()],
    )
    setup.finish_unit("instantiate Trainer", "trainer ready")

    setup.start_unit("run training loop", "Trainer owns batch loading, forward/backward, optimizer, eval, and checkpoint steps")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    setup.finish_unit("run training loop", f"global_step={trainer.state.global_step:,}")

    setup.start_unit("save final model", f"output_dir={output_dir}")
    save_final_model(trainer, output_dir, runtime)
    setup.finish_unit("save final model", f"global_step={trainer.state.global_step:,}")


if __name__ == "__main__":
    main()

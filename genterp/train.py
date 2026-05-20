"""Train Genterp on AoU OMOP."""

from __future__ import annotations

import atexit
import faulthandler
import json
import os
import re
import shutil
import signal
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genterp.eval_cindex import CindexCohort

import numpy as np
import psutil
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from transformers import trainer as hf_trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

from genterp.data import AtomVocab, CohortDataset, EventStore, collate
from genterp.modeling import Genterp, GenterpConfig
from genterp.progress import ProgressLogger, count_parameters
from genterp.runtime import TorchRuntime, accelerator_label, configure_torch_runtime


_PROC = psutil.Process()


def _rss_str() -> str:
    return f"RSS={_PROC.memory_info().rss/1e9:.2f}GB"


def _install_crash_diagnostics() -> None:
    """Same recipe as scripts/aou_etl.py: make non-OOM crashes loud.

    The eval CohortDataset loading the events.parquet for the second time blew
    the box's RAM ceiling and got SIGKILL'd silently. SIGKILL is uncatchable,
    so the *mitigation* is to share the events store (see EventStore) — but
    any *other* crash (segfault, unhandled exception, SIGTERM) should at least
    print a Python traceback before exit. faulthandler covers segfaults/abort
    signals; the signal handlers cover clean termination; sys.excepthook
    catches anything that escapes the main module.
    """
    faulthandler.enable()
    for name in ("SIGTERM", "SIGHUP", "SIGUSR1"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, all_threads=True, chain=True)
        except (ValueError, OSError):
            pass

    def _on_signal(signum: int, frame) -> None:
        try:
            label = signal.Signals(signum).name
        except ValueError:
            label = str(signum)
        print(f"[train] FATAL: received {label} ({signum}); {_rss_str()}", file=sys.stderr, flush=True)
        traceback.print_stack(frame)
        sys.stderr.flush()
        sys.exit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    def _on_unhandled(exc_type, exc, tb) -> None:
        print(f"[train] FATAL: unhandled {exc_type.__name__}: {exc}; {_rss_str()}", file=sys.stderr, flush=True)
        traceback.print_exception(exc_type, exc, tb)
        sys.stderr.flush()

    sys.excepthook = _on_unhandled
    atexit.register(lambda: print(f"[train] process exiting; final {_rss_str()}", file=sys.stderr, flush=True))

RUNTIME_STATE_FILE = "genterp_runtime.json"
FINAL_POINTER_FILE = "final_checkpoint.json"
ANCESTORS_FILE = "ancestors.npz"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# Model is undertrained at 50K (loss still dropping fast late in training);
# Chinchilla-style scaling on this token budget argues for substantially
# more steps. The WSD scheduler's decay tail is set to 10% of total so the
# stable phase stretches with longer runs and the decay phase still has
# enough room to polish.
MAX_STEPS = 200_000
WARMUP_STEPS = 500
WSD_DECAY_STEPS = MAX_STEPS // 10
LOGGING_STEPS = 50
EVAL_STEPS = 2_000
SAVE_STEPS = 2_000
SAVE_TOTAL_LIMIT = 3
# In-loop eval subsample size. Full ~12K test cohort is too slow to score
# every EVAL_STEPS; the random subsample stays large enough that per-disease
# C-index has stable sampling.
EVAL_SUBSAMPLE_SIZE = 1024
EVAL_SUBSAMPLE_SEED = 0


class GenterpHFConfig(transformers.PretrainedConfig):
    model_type = "genterp"

    def __init__(self, **kwargs):
        self.genterp_cfg: dict = kwargs.pop("genterp_cfg", {})
        super().__init__(**kwargs)


class _LengthAwareSubset(Subset):
    """Subset that exposes ``.lengths`` so ``LengthGroupedSampler`` keeps working."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        super().__init__(dataset, indices)
        parent_lengths = getattr(dataset, "lengths", None)
        if parent_lengths is not None:
            self.lengths = np.asarray(parent_lengths, dtype=np.int64)[indices]


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

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Warm-load with partial-prefix copy along vocab axis.

        When the ETL vocab grows (e.g., death and demographics events added
        → n_atoms jumps from 27k to 27.005k), the vocab-shaped params don't
        match in shape. Naively dropping them destroys ALL trained
        embeddings, not just the new atoms. Instead we detect axis-0-only
        mismatches (vocab axis) and copy the overlapping prefix: existing
        atoms keep their trained vectors, new atoms keep their fresh init.

        Affected parameters:
          - embed.embedding.weight       : (n_atoms, dim)
          - embed.ancestor_embedding.weight : (n_ancestor_rows+1, dim)
          - value_mod.value_mu / value_sigma / atom_has_mag : (n_atoms,)
          - tpp.mark_out.weight (tied to embed.embedding.weight) — inherits
            the partial-prefix copy via the tying.

        Genuinely incompatible mismatches (different dim, different number
        of attention heads, etc.) still drop and the new tensor keeps its
        init — same as the prior behavior. HF Trainer's optimizer-state
        load is gated separately by `reset_training_state_on_resume`, which
        we set to True whenever any of these axis-0 shapes changed so the
        Adam moments for the resized parameters are rebuilt cleanly.
        """
        own = super().state_dict()
        actions: list[tuple[str, tuple[int, ...], tuple[int, ...], str]] = []
        filtered: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k not in own:
                filtered[k] = v
                continue
            own_v = own[k]
            if tuple(own_v.shape) == tuple(v.shape):
                filtered[k] = v
                continue
            # Vocab-axis-only mismatch: same ndim, same shape[1:], differ only on axis 0.
            if (
                v.ndim == own_v.ndim
                and tuple(v.shape[1:]) == tuple(own_v.shape[1:])
            ):
                n = min(int(v.shape[0]), int(own_v.shape[0]))
                merged = own_v.detach().clone()
                merged[:n] = v[:n].to(dtype=own_v.dtype, device=own_v.device)
                filtered[k] = merged
                actions.append((k, tuple(v.shape), tuple(own_v.shape), f"prefix-copied[:n={n}]"))
            else:
                actions.append((k, tuple(v.shape), tuple(own_v.shape), "dropped (incompatible)"))
        if actions:
            preview = ", ".join(f"{k} ckpt{c}→cur{o} [{a}]" for k, c, o, a in actions[:5])
            if len(actions) > 5:
                preview += f", +{len(actions) - 5} more"
            print(f"[warm-start] handled {len(actions)} shape-mismatched param(s): {preview}")
        return super().load_state_dict(
            filtered,
            strict=False if actions else strict,
            assign=assign,
        )


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

    @staticmethod
    def _format_metrics(logs: dict[str, object] | None) -> str:
        if not logs:
            return "no metrics payload"
        ordered = [
            "loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "time_nll",
            "mark_nll",
            "value_nll",
            "value_nll_max",
            "value_z_abs_max",
            "value_z_clip_pct",
            "censor_nll",
            "n_real",
            "n_censor",
            "n_mag",
            "real_per_subject",
            "censor_per_subject",
            "mag_per_real",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
            "eval_time_nll",
            "eval_mark_nll",
            "eval_value_nll",
            "eval_value_nll_max",
            "eval_value_z_abs_max",
            "eval_value_z_clip_pct",
            "eval_censor_nll",
            "eval_n_real",
            "eval_n_censor",
            "eval_n_mag",
            "eval_real_per_subject",
            "eval_censor_per_subject",
            "eval_mag_per_real",
            "epoch",
        ]
        pieces = [f"{key}={logs[key]}" for key in ordered if key in logs]
        pieces.extend(f"{key}={value}" for key, value in sorted(logs.items()) if key not in ordered)
        alerts = []
        n_censor = logs.get("n_censor")
        if isinstance(n_censor, (int, float)) and n_censor == 0 and "loss" in logs:
            alerts.append("NO_CENSOR_TOKENS")
        eval_n_censor = logs.get("eval_n_censor")
        if isinstance(eval_n_censor, (int, float)) and eval_n_censor == 0:
            alerts.append("EVAL_NO_CENSOR_TOKENS")
        value_nll_max = logs.get("value_nll_max")
        if isinstance(value_nll_max, (int, float)) and value_nll_max > 8:
            alerts.append("VALUE_OUTLIER")
        eval_value_nll_max = logs.get("eval_value_nll_max")
        if isinstance(eval_value_nll_max, (int, float)) and eval_value_nll_max > 8:
            alerts.append("EVAL_VALUE_OUTLIER")
        if alerts:
            pieces.append(f"alerts={'+'.join(alerts)}")
        return ", ".join(pieces)

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

    def on_step_end(self, args, state, control, **kwargs):
        """Lightweight per-step pulse: only update the progress counter.

        We deliberately do NOT log on every step — at 8 it/s × 50K steps that's
        50K lines of noise. The on_log callback fires at logging_steps cadence
        and emits the per-step metrics; that's the user-visible signal.
        """
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log("training metrics emitted", self._format_metrics(logs))
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        total = int(state.max_steps or args.max_steps or 0)
        self.logger.set_progress(int(state.global_step), total)
        self.logger.log("evaluation complete", self._format_metrics(metrics))
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
        cindex_cohort: "CindexCohort | None" = None,
        cindex_every_n_evals: int = 1,
        **kwargs,
    ):
        self.runtime = runtime
        self.reset_training_state_on_resume = reset_training_state_on_resume
        # Per-component loss accumulator drained by `log()`. Keys are the metric
        # names emitted by `marked_tpp_value_loss` (time_nll, mark_nll, value_nll,
        # censor_nll, n_real, n_censor, n_mag). One list of per-step floats per key
        # for {train, eval}; we average on emit so each logging_steps interval (or
        # full eval pass) reports the mean component NLL over its tokens.
        self._loss_accum_train: dict[str, list[torch.Tensor]] = {}
        self._loss_accum_eval: dict[str, list[torch.Tensor]] = {}
        # Lazy-imported cohort artifact for periodic C-index scoring; mixed into
        # eval metrics every `cindex_every_n_evals` calls to evaluation_loop so
        # the C-index curves end up in trainer_state.json alongside eval_loss.
        self._cindex_cohort = cindex_cohort
        self._cindex_every_n_evals = max(1, int(cindex_every_n_evals))
        self._cindex_eval_count = 0
        super().__init__(*args, **kwargs)
        if runtime is not None and runtime.device.type == "cuda" and not runtime.use_data_parallel:
            self.args._n_gpu = 1

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        """Compute the rich Genterp loss dict; stash component NLLs for `log()` to surface.

        HF Trainer normally only sees `loss`. Here we also capture time/mark/value/censor
        NLLs and token counts every forward — train side accumulates into
        `_loss_accum_train`, eval side into `_loss_accum_eval`, and `log()` averages
        whichever bucket matches the call.
        """
        inner = model.model if hasattr(model, "model") else model
        ld = inner.loss(**inputs)
        bucket = self._loss_accum_train if model.training else self._loss_accum_eval
        for key in (
            "time_nll",
            "mark_nll",
            "value_nll",
            "value_nll_max",
            "value_z_abs_max",
            "value_z_clipped",
            "censor_nll",
            "n_real",
            "n_censor",
            "n_mag",
            "n_subject",
        ):
            tensor = ld.get(key)
            if tensor is None:
                continue
            bucket.setdefault(key, []).append(tensor.detach().float())
        loss = ld["loss"]
        if return_outputs:
            outputs = transformers.modeling_outputs.CausalLMOutput(loss=loss, logits=loss.detach().reshape(1))
            return loss, outputs
        return loss

    def _drain_loss_bucket(self, bucket: dict[str, list[torch.Tensor]], prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        for key, values in bucket.items():
            if not values:
                continue
            stacked = torch.stack(values)
            reducer = torch.max if key.endswith("_max") else torch.mean
            out[f"{prefix}{key}"] = float(reducer(stacked).cpu().item())
        self._add_derived_metrics(out, prefix)
        bucket.clear()
        return out

    @staticmethod
    def _add_derived_metrics(logs: dict[str, float], prefix: str = "") -> None:
        n_subject = logs.get(f"{prefix}n_subject", 0.0)
        n_real = logs.get(f"{prefix}n_real", 0.0)
        n_censor = logs.get(f"{prefix}n_censor", 0.0)
        n_mag = logs.get(f"{prefix}n_mag", 0.0)
        n_clipped = logs.get(f"{prefix}value_z_clipped", 0.0)
        if n_subject > 0:
            logs[f"{prefix}real_per_subject"] = n_real / n_subject
            logs[f"{prefix}censor_per_subject"] = n_censor / n_subject
        if n_real > 0:
            logs[f"{prefix}mag_per_real"] = n_mag / n_real
        if n_mag > 0:
            logs[f"{prefix}value_z_clip_pct"] = 100.0 * n_clipped / n_mag

    def log(self, logs, *args, **kwargs):
        """Merge accumulated component metrics into the next emitted log payload.

        The presence of an "eval_loss" key marks an eval log emission; otherwise
        it's a train-side metrics flush. Component aggregates come along for the
        ride so on-disk train history and trainer_pt's TensorBoard both see them.
        """
        is_eval = any(k.startswith("eval_") for k in logs)
        if is_eval:
            logs.update(self._drain_loss_bucket(self._loss_accum_eval, prefix="eval_"))
        else:
            logs.update(self._drain_loss_bucket(self._loss_accum_train))
        return super().log(logs, *args, **kwargs)

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

        # Eval gets fewer workers and never keeps them resident: each worker copies
        # the events parquet into its own numpy arrays (~1.7GB for the tiny CDR; way
        # more for full). Keeping 4 of those alive between every 500-step eval cycle
        # is what was OOMing the box.
        num_workers = self.args.dataloader_num_workers if is_training else min(2, self.args.dataloader_num_workers)
        persistent_workers = self.args.dataloader_persistent_workers if is_training else False

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
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
            return type(data)(
                (key, value if key == "length" and isinstance(value, torch.Tensor) else self._prepare_input(value))
                for key, value in data.items()
            )
        if isinstance(data, tuple):
            return tuple(self._prepare_input(value) for value in data)
        if isinstance(data, list):
            return [self._prepare_input(value) for value in data]
        return data

    def _move_model_to_device(self, model: torch.nn.Module, device: torch.device) -> None:
        if self.runtime is not None:
            device = self.runtime.device
        super()._move_model_to_device(model, device)

    def evaluation_loop(self, *args, **kwargs):
        """Augment HF's eval metrics with C-index per disease so the survival
        head's external utility shows up alongside loss in trainer_state.json.

        Runs once per N evaluation_loop calls (configurable via
        `cindex_every_n_evals`). On the tiny cohort the extra cost is ~6s; on
        full it's ~40s. Per-disease C-index and incidence rate are logged with
        the `eval_` prefix so HF's log() routes them to log_history alongside
        eval_loss and friends.
        """
        output = super().evaluation_loop(*args, **kwargs)
        if self._cindex_cohort is None:
            return output
        self._cindex_eval_count += 1
        if self._cindex_eval_count % self._cindex_every_n_evals != 0:
            return output
        try:
            # Local import to avoid the import cycle: eval_cindex imports from train.
            from genterp.eval_cindex import run_cindex
            autocast_dtype = (
                torch.bfloat16 if (self.runtime and self.runtime.bf16)
                else torch.float16 if (self.runtime and self.runtime.fp16)
                else None
            )
            device = self.runtime.device if self.runtime is not None else next(self.model.parameters()).device
            cindex_results = run_cindex(
                self.model, self._cindex_cohort,
                device=device, autocast_dtype=autocast_dtype,
            )
            c_values: list[float] = []
            for name, m in cindex_results.items():
                # run_cindex tucks a "__summary__" entry into the results dict
                # with a different schema (no "c_index" key) — skip it.
                if name.startswith("_"):
                    continue
                key_safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
                c = m.get("c_index")
                if c is None:
                    continue
                c_f = float(c)  # type: ignore[arg-type]
                output.metrics[f"eval_cindex_{key_safe}"] = c_f
                c_values.append(c_f)
            if c_values:
                # Weighted by event count would be cleaner but float-mean is fine
                # as a single-number summary for early stopping / dashboards.
                output.metrics["eval_cindex_mean"] = float(np.mean(c_values))
                output.metrics["eval_cindex_n_diseases"] = len(c_values)
        except Exception as exc:  # noqa: BLE001 — never break training on eval failure
            print(f"[cindex] in-loop eval skipped: {type(exc).__name__}: {exc}")
        return output

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        if checkpoint is not None and self.reset_training_state_on_resume:
            return
        super()._load_optimizer_and_scheduler(checkpoint)

    def _load_scaler(self, checkpoint: str | None) -> None:
        if checkpoint is not None and self.reset_training_state_on_resume:
            return
        super()._load_scaler(checkpoint)

    def _load_rng_state(self, checkpoint: str | None) -> None:
        # When optimizer/scheduler are being rebuilt anyway (vocab grew or
        # hardware changed), RNG continuity is already broken — skip the load
        # entirely. Otherwise let the parent class handle it.
        if checkpoint is None or self.reset_training_state_on_resume:
            return
        super()._load_rng_state(checkpoint)


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


def checkpoint_ancestor_rows(path: str | Path) -> int | None:
    """How many rows did the checkpoint's ``embed.ancestor_embedding`` have?

    1 means flat mode (placeholder); >1 means hierarchical was active. Returned
    by sniffing the safetensors header (no full weight load). Used to decide
    whether activating ancestors in the current run is a config change that
    requires rebuilding optimizer state.
    """
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    weights = Path(path) / "model.safetensors"
    if not weights.is_file():
        return None
    try:
        with safe_open(str(weights), framework="pt") as f:  # type: ignore[no-untyped-call]
            for key in ("model.embed.ancestor_embedding.weight",):
                if key in f.keys():
                    return int(f.get_slice(key).get_shape()[0])
    except (OSError, ValueError):
        return None
    return None


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


def checkpoint_n_atoms(path: str | Path) -> int | None:
    """Read n_atoms from a saved checkpoint's config.json. None if unreadable.

    Used to detect when the ETL vocab has grown/shrunk between runs; a mismatch
    forces optimizer-state reset (the optimizer's per-parameter moments are
    sized to the old vocab and would be garbage for the new one) even though
    the partial state-dict load itself succeeds via the warm-start filter in
    GenterpForCausalLM.load_state_dict.
    """
    cfg_path = Path(path) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        n = cfg.get("genterp_cfg", {}).get("n_atoms")
        return int(n) if n is not None else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


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


def _load_ancestors(path: Path) -> tuple[torch.Tensor, int] | None:
    """Read the optional hierarchical-embedding ancestor table from ETL.

    Layout (produced by ``scripts/build_ancestors.py``):
      - ``ancestor_ids`` : (n_atoms, max_anc) int64; row a is the ancestor-node
        ids for atom a, right-padded with 0. Node id 0 means "no ancestor here".
      - ``n_ancestor_rows`` : scalar int — distinct non-pad ancestor nodes.

    Missing file is fine: the model stays in flat-embedding mode and the
    existing checkpoint warm-starts unchanged. Present file activates the
    hierarchical path with zero-init ancestor vectors, so the very first
    forward after attaching ancestors reproduces the flat-embedding output
    bit-for-bit; ancestor gradients are then learned during continued training.
    """
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    if "ancestor_ids" not in data.files or "n_ancestor_rows" not in data.files:
        raise ValueError(f"ancestors file {path} is missing required keys")
    ancestor_ids = torch.from_numpy(np.asarray(data["ancestor_ids"], dtype=np.int64))
    n_ancestor_rows = int(np.asarray(data["n_ancestor_rows"]).item())
    return ancestor_ids, n_ancestor_rows


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
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "eval_strategy": "steps",
        "eval_steps": EVAL_STEPS,
        "prediction_loss_only": True,
        "logging_steps": LOGGING_STEPS,
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


def tensor_core_padding_multiple(runtime: TorchRuntime) -> int | None:
    if runtime.device.type == "cuda" and (runtime.fp16 or runtime.bf16):
        return 8
    return None


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

    _install_crash_diagnostics()
    setup = ProgressLogger("train_setup", total_units=16)
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

    setup.start_unit(
        "load events parquet (shared)",
        "single decompressed copy reused by train + eval — prevents the ~20GB×2 OOM",
    )
    event_store = EventStore.from_parquet(etl / "events.parquet")
    setup.finish_unit(
        "load events parquet (shared)",
        f"rows={event_store.num_rows:,} chunks={event_store.num_chunks} {_rss_str()}",
    )

    setup.start_unit("build training dataset", "split=train; events shared from event_store")
    train_dataset = CohortDataset(etl, split="train", events=event_store)
    setup.finish_unit("build training dataset", f"subjects={len(train_dataset):,} {_rss_str()}")

    setup.start_unit("build evaluation dataset", "split=test; events shared from event_store")
    eval_full = CohortDataset(etl, split="test", events=event_store)
    # Each in-loop eval pass scores a deterministic random subsample of the
    # held-out test cohort. Random (not longest-first) so the slice is
    # representative — longest-first biased the eval toward heavy utilizers
    # whose compressed risk profile suppressed measurable C-index.
    if len(eval_full) > EVAL_SUBSAMPLE_SIZE:
        order = np.random.default_rng(EVAL_SUBSAMPLE_SEED).permutation(len(eval_full))[:EVAL_SUBSAMPLE_SIZE]
        eval_dataset: Dataset = _LengthAwareSubset(eval_full, order.tolist())
        setup.finish_unit(
            "build evaluation dataset",
            f"subjects={len(eval_dataset):,} (random subsample from {len(eval_full):,}; "
            f"seed={EVAL_SUBSAMPLE_SEED})",
        )
    else:
        eval_dataset = eval_full
        setup.finish_unit("build evaluation dataset", f"subjects={len(eval_dataset):,}")

    setup.start_unit(
        "pre-scan ancestor table (optional)",
        f"path={etl / ANCESTORS_FILE} — determines hierarchical embedding shape at construction",
    )
    pending_ancestors = _load_ancestors(etl / ANCESTORS_FILE)
    target_n_ancestor_rows = pending_ancestors[1] if pending_ancestors is not None else 0
    setup.finish_unit(
        "pre-scan ancestor table (optional)",
        f"n_ancestor_rows={target_n_ancestor_rows} ({'hierarchical' if target_n_ancestor_rows > 0 else 'flat'} embedding)",
    )

    if tiny:
        # ~1000× fewer transformer params: dim 32 vs 512 (16×), 2 vs 8 layers (4×).
        # Per-layer dense weights scale as L·dim² → 4·256 = 1024×.
        setup.start_unit("construct model config", "--tiny → dim=32 heads=2 layers=2")
        cfg = GenterpConfig(n_atoms=len(vocab), dim=32, n_heads=2, n_layers=2, n_ancestor_rows=target_n_ancestor_rows)
    else:
        setup.start_unit("construct model config", "dim=512 heads=8 layers=8")
        cfg = GenterpConfig(n_atoms=len(vocab), dim=512, n_heads=8, n_layers=8, n_ancestor_rows=target_n_ancestor_rows)
    setup.finish_unit(
        "construct model config",
        f"n_atoms={cfg.n_atoms:,} dim={cfg.dim} layers={cfg.n_layers} n_ancestor_rows={cfg.n_ancestor_rows}",
    )

    setup.start_unit("inspect checkpoints", f"output_dir={output_dir}")
    resume_checkpoint = latest_checkpoint(output_dir)
    hardware_changed = resume_checkpoint is not None and not checkpoint_matches_runtime(resume_checkpoint, runtime)
    ckpt_n_atoms = checkpoint_n_atoms(resume_checkpoint) if resume_checkpoint else None
    vocab_changed = ckpt_n_atoms is not None and ckpt_n_atoms != cfg.n_atoms
    # Detect ancestor-table activation: if the ETL now has an ancestors file
    # but the checkpoint was saved with the flat placeholder (or vice versa),
    # the embed.ancestor_embedding parameter has a new shape on this run and
    # the saved Adam moments for it are unusable. Treat it like a vocab change
    # — load whatever weights still fit, rebuild optimizer state.
    ckpt_ancestor_rows = checkpoint_ancestor_rows(resume_checkpoint) if resume_checkpoint else None
    target_ancestor_rows = target_n_ancestor_rows + 1
    ancestors_changed = (
        ckpt_ancestor_rows is not None and ckpt_ancestor_rows != target_ancestor_rows
    )
    # Any shape-axis change in the model invalidates the optimizer state (Adam
    # moments are per-parameter and per-shape). Force a state reset so we
    # rebuild fresh moments around the partially-loaded weights.
    reset_training_state = bool(
        resume_checkpoint and (hardware_changed or vocab_changed or ancestors_changed)
    )
    warm_start_path = final_model_path(output_dir) if resume_checkpoint is None else None
    setup.finish_unit(
        "inspect checkpoints",
        f"resume_checkpoint={resume_checkpoint} warm_start_path={warm_start_path} "
        f"reset_training_state={reset_training_state} hardware_changed={hardware_changed} "
        f"vocab_changed={vocab_changed} (ckpt_n_atoms={ckpt_n_atoms} cur_n_atoms={cfg.n_atoms}) "
        f"ancestors_changed={ancestors_changed} (ckpt_anc_rows={ckpt_ancestor_rows} cur_anc_rows={target_ancestor_rows})",
    )

    setup.start_unit("load or initialize model", "preferring resume checkpoint, then previous final model, then fresh init")
    if warm_start_path is not None:
        # ignore_mismatched_sizes lets transformers fall through vocab-shaped
        # tensors (embedding / mark_out / value_mod buffers) that don't match;
        # they keep their fresh init while everything else loads.
        model = GenterpForCausalLM.from_pretrained(warm_start_path, ignore_mismatched_sizes=True)
        model_source = f"warm_start={warm_start_path}"
    else:
        model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))
        model_source = "fresh_init" if resume_checkpoint is None else f"resume_weights_from={resume_checkpoint}"
    if reset_training_state:
        reason = "hardware profile changed" if hardware_changed else f"vocab changed ({ckpt_n_atoms} → {cfg.n_atoms} atoms)"
        print(f"genterp train {reason}; resuming model weights (partial) and rebuilding optimizer state")
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

    setup.start_unit(
        "attach hierarchical ancestor lookup (optional)",
        "pre-scanned table is now wired into the constructed embedding",
    )
    if pending_ancestors is not None:
        ancestor_ids, n_ancestor_rows = pending_ancestors
        if ancestor_ids.shape[0] != cfg.n_atoms:
            print(
                f"[ancestors] shape mismatch: file has {ancestor_ids.shape[0]} atoms, "
                f"model has {cfg.n_atoms}; skipping. Rebuild {ANCESTORS_FILE} from current vocab."
            )
            setup.finish_unit("attach hierarchical ancestor lookup (optional)", "skipped (atom-count mismatch)")
        else:
            model.model.embed.set_ancestor_ids(ancestor_ids)
            setup.finish_unit(
                "attach hierarchical ancestor lookup (optional)",
                f"n_atoms={ancestor_ids.shape[0]:,} max_anc_per_atom={int(ancestor_ids.shape[1])} "
                f"n_ancestor_rows={n_ancestor_rows:,} (zero-init at first activation keeps warm-start exact)",
            )
    else:
        setup.finish_unit("attach hierarchical ancestor lookup (optional)", "no ancestors file; flat embedding mode")

    setup.start_unit("build training arguments", "max_steps=50000 with lower-overhead logging, eval, and checkpoint cadence")
    training_args = build_training_args(output_dir, runtime)
    setup.finish_unit(
        "build training arguments",
        f"max_steps={training_args['max_steps']:,} logging_steps={training_args['logging_steps']} "
        f"save_steps={training_args['save_steps']} save_total_limit={training_args['save_total_limit']} "
        f"eval_steps={training_args['eval_steps']} "
        f"lr_scheduler={training_args['lr_scheduler_type']} gradient_checkpointing={training_args['gradient_checkpointing']}",
    )

    setup.start_unit(
        "prepare C-index cohort",
        "SNOMED-descendant phenotypes + outcome table for periodic in-loop scoring",
    )
    cindex_cohort = None
    try:
        from genterp.eval_cindex import prepare_cindex_cohort
        cindex_cohort = prepare_cindex_cohort(
            etl, vocab,
            events=event_store,
            pin_memory=runtime.dataloader_pin_memory,
        )
        setup.finish_unit(
            "prepare C-index cohort",
            f"subjects={len(cindex_cohort.subjects):,}  diseases={len(cindex_cohort.disease_names)}",
        )
    except Exception as exc:  # noqa: BLE001 — survival eval is optional; never block training
        print(f"[cindex] cohort prep skipped: {type(exc).__name__}: {exc}")
        setup.finish_unit("prepare C-index cohort", f"skipped ({type(exc).__name__})")

    setup.start_unit("instantiate Trainer", "attaching runtime-state and verbose progress callbacks")
    data_collator = partial(collate, pad_to_multiple_of=tensor_core_padding_multiple(runtime))
    trainer = GenterpTrainer(
        model=model,
        args=transformers.TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        runtime=runtime,
        reset_training_state_on_resume=reset_training_state,
        cindex_cohort=cindex_cohort,
        cindex_every_n_evals=1,
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

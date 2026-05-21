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
import time
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
from genterp.vocab import validate_atom_registry


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
ATOM_REGISTRY_FILE = "atom_registry.json"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# Model is undertrained at 50K (loss still dropping fast late in training);
# Chinchilla-style scaling on this token budget argues for substantially
# more steps. The WSD scheduler's decay tail is set to 10% of total so the
# stable phase stretches with longer runs and the decay phase still has
# enough room to polish.
MAX_STEPS = 200_000
# 1% warmup is the conservative default for pretraining-scale runs under
# mixed precision: too short a warmup has been a recurring source of early-
# step loss spikes (large residual branches × under-warmed Adam moments ×
# narrow fp16 range). bf16 is more forgiving but the upside of extra warmup
# is a tiny LR underrun at the start vs. the much larger downside if a
# single spike wipes the run.
WARMUP_STEPS = max(MAX_STEPS // 100, 500)
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
# Cap the C-index cohort. The eligible test pool is ~30k+ subjects; scoring
# every one of them every EVAL_STEPS dominates training wall-clock. But
# Harrell's C power scales with *event count* per disease, not subject
# count: a 1%-incidence cancer in 2k subjects only fires ~20 events, which
# gives noisy per-disease estimates. 8192 hits ~80 events for a 1%-incidence
# disease and ~800 for a 10%-incidence one — well-powered across the
# leaderboard. Eval wall-clock at batch_size=16 ≈ 4-5 min per cycle, ~5%
# of training time at the 2000-step cadence. Standalone eval_cindex /
# eval_rollout CLIs leave the cap off (None) and score the full cohort.
CINDEX_MAX_SUBJECTS_IN_LOOP = 8192


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
    _vocab_axis_keys = {
        "model.embed.embedding.weight",
        "model.tpp.mark_out.weight",
        "model.value_mod.value_mu",
        "model.value_mod.value_sigma",
        "model.value_mod.atom_has_mag",
    }

    def __init__(self, config: GenterpHFConfig):
        super().__init__(config)
        self.model = Genterp(GenterpConfig(**config.genterp_cfg))
        self._atom_id_remap: list[tuple[int, int]] | None = None
        self._atom_id_remap_summary = ""

    def configure_atom_registry_remap(
        self,
        checkpoint_registry: dict[str, int] | None,
        current_registry: dict[str, int] | None,
    ) -> None:
        """Tell ``load_state_dict`` how to map vocab rows by canonical atom code."""
        self._atom_id_remap = None
        self._atom_id_remap_summary = ""
        if checkpoint_registry is None or current_registry is None:
            return
        validate_atom_registry(checkpoint_registry)
        validate_atom_registry(current_registry)
        if checkpoint_registry == current_registry:
            return
        shared_codes = sorted(set(checkpoint_registry) & set(current_registry))
        remap = [(0, 0)]
        remap.extend((checkpoint_registry[code], current_registry[code]) for code in shared_codes)
        self._atom_id_remap = remap
        self._atom_id_remap_summary = (
            f"shared_codes={len(shared_codes):,} "
            f"ckpt_atoms={len(checkpoint_registry):,} current_atoms={len(current_registry):,}"
        )

    def forward(self, **batch: torch.Tensor) -> transformers.modeling_outputs.CausalLMOutput:
        ld = self.model.loss(**batch)
        return transformers.modeling_outputs.CausalLMOutput(loss=ld["loss"], logits=ld["loss"].detach().reshape(1))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Warm-load vocab-shaped tensors by canonical atom code, never prefix order."""
        ancestor_key = "model.embed.ancestor_ids"
        if ancestor_key in state_dict:
            ancestor_ids = state_dict[ancestor_key]
            if (
                torch.is_tensor(ancestor_ids)
                and ancestor_ids.ndim == 2
                and int(ancestor_ids.shape[0]) == self.model.embed.embedding.num_embeddings
                and (
                    ancestor_ids.numel() == 0
                    or int(ancestor_ids.max().item()) < self.model.embed.ancestor_embedding.num_embeddings
                )
            ):
                self.model.embed.set_ancestor_ids(ancestor_ids)

        own = super().state_dict()
        actions: list[tuple[str, tuple[int, ...], tuple[int, ...], str]] = []
        filtered: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k not in own:
                filtered[k] = v
                continue
            own_v = own[k]
            if (
                self._atom_id_remap is not None
                and k in self._vocab_axis_keys
                and v.ndim == own_v.ndim
                and tuple(v.shape[1:]) == tuple(own_v.shape[1:])
            ):
                merged = own_v.detach().clone()
                copied = 0
                for old_id, new_id in self._atom_id_remap:
                    if old_id < int(v.shape[0]) and new_id < int(own_v.shape[0]):
                        merged[new_id] = v[old_id].to(dtype=own_v.dtype, device=own_v.device)
                        copied += 1
                filtered[k] = merged
                actions.append((k, tuple(v.shape), tuple(own_v.shape), f"registry-remapped[rows={copied}]"))
                continue
            if tuple(own_v.shape) == tuple(v.shape):
                filtered[k] = v
                continue
            if (
                k in self._vocab_axis_keys
                and v.ndim == own_v.ndim
                and tuple(v.shape[1:]) == tuple(own_v.shape[1:])
            ):
                n = min(int(v.shape[0]), int(own_v.shape[0]))
                merged = own_v.detach().clone()
                merged[:n] = v[:n].to(dtype=own_v.dtype, device=own_v.device)
                filtered[k] = merged
                actions.append((k, tuple(v.shape), tuple(own_v.shape), f"prefix-copied[:n={n}]"))
            elif k in self._vocab_axis_keys:
                actions.append((k, tuple(v.shape), tuple(own_v.shape), "dropped (no atom registry remap)"))
            else:
                actions.append((k, tuple(v.shape), tuple(own_v.shape), "dropped (incompatible)"))
        if actions:
            preview = ", ".join(f"{k} ckpt{c}→cur{o} [{a}]" for k, c, o, a in actions[:5])
            if len(actions) > 5:
                preview += f", +{len(actions) - 5} more"
            print(f"[warm-start] handled {len(actions)} shape-mismatched param(s): {preview}")
            if self._atom_id_remap_summary:
                print(f"[warm-start] atom registry remap: {self._atom_id_remap_summary}")
        return super().load_state_dict(
            filtered,
            strict=False if actions else strict,
            assign=assign,
        )


class RuntimeStateCallback(transformers.TrainerCallback):
    def __init__(self, runtime: TorchRuntime, atom_registry: dict[str, int] | None = None):
        self.runtime = runtime
        self.atom_registry = atom_registry

    def on_save(self, args, state, control, **kwargs):
        logger = ProgressLogger("trainer_save", total_units=1)
        logger.start_unit("write checkpoint runtime profile", f"checkpoint=checkpoint-{state.global_step}")
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        write_runtime_state(checkpoint_dir, self.runtime)
        write_atom_registry_snapshot(checkpoint_dir, self.atom_registry)
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
            "time_modeled_nll",
            "mark_nll",
            "sampled_mark_nll",
            "exact_mark_nll",
            "mark_z_loss",
            "weighted_mark_z_loss",
            "value_nll",
            "value_loss_weight",
            "value_nll_max",
            "value_z_abs_max",
            "value_z_clip_pct",
            "censor_nll",
            "censor_loss_weight",
            "n_real",
            "n_time",
            "n_censor",
            "n_mag",
            "cum_n_real",
            "cum_n_time",
            "cum_n_censor",
            "cum_n_mag",
            "cum_n_subject",
            "cum_transitions",
            "effective_event_epochs",
            "real_tokens_per_hour",
            "transitions_per_hour",
            "real_per_subject",
            "time_per_real",
            "censor_per_subject",
            "mag_per_real",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
            "eval_time_nll",
            "eval_time_modeled_nll",
            "eval_mark_nll",
            "eval_sampled_mark_nll",
            "eval_exact_mark_nll",
            "eval_mark_z_loss",
            "eval_weighted_mark_z_loss",
            "eval_value_nll",
            "eval_value_loss_weight",
            "eval_value_nll_max",
            "eval_value_z_abs_max",
            "eval_value_z_clip_pct",
            "eval_censor_nll",
            "eval_censor_loss_weight",
            "eval_n_real",
            "eval_n_time",
            "eval_n_censor",
            "eval_n_mag",
            "eval_real_per_subject",
            "eval_time_per_real",
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
        self._train_count_totals = {
            "n_real": 0.0,
            "n_time": 0.0,
            "n_censor": 0.0,
            "n_mag": 0.0,
            "n_subject": 0.0,
        }
        self._token_clock_start = time.monotonic()
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
            "time_modeled_nll",
            "mark_nll",
            "sampled_mark_nll",
            "exact_mark_nll",
            "mark_z_loss",
            "weighted_mark_z_loss",
            "value_nll",
            "value_loss_weight",
            "value_nll_max",
            "value_z_abs_max",
            "value_z_clipped",
            "censor_nll",
            "censor_loss_weight",
            "n_real",
            "n_time",
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

    def _drain_loss_bucket(
        self,
        bucket: dict[str, list[torch.Tensor]],
        prefix: str = "",
        *,
        update_train_totals: bool = False,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        count_sums: dict[str, float] = {}
        for key, values in bucket.items():
            if not values:
                continue
            stacked = torch.stack(values)
            reducer = torch.max if key.endswith("_max") else torch.mean
            out[f"{prefix}{key}"] = float(reducer(stacked).cpu().item())
            if key in self._train_count_totals:
                count_sums[key] = float(stacked.sum().cpu().item())
        if update_train_totals:
            for key, value in count_sums.items():
                self._train_count_totals[key] += value
            self._add_cumulative_token_metrics(out)
        self._add_derived_metrics(out, prefix)
        bucket.clear()
        return out

    def _add_cumulative_token_metrics(self, logs: dict[str, float]) -> None:
        elapsed_hours = max((time.monotonic() - self._token_clock_start) / 3600.0, 1e-9)
        n_real = self._train_count_totals["n_real"]
        n_time = self._train_count_totals["n_time"]
        n_censor = self._train_count_totals["n_censor"]
        n_mag = self._train_count_totals["n_mag"]
        n_subject = self._train_count_totals["n_subject"]
        transitions = n_real + n_censor
        logs["cum_n_real"] = n_real
        logs["cum_n_time"] = n_time
        logs["cum_n_censor"] = n_censor
        logs["cum_n_mag"] = n_mag
        logs["cum_n_subject"] = n_subject
        logs["cum_transitions"] = transitions
        logs["real_tokens_per_hour"] = n_real / elapsed_hours
        logs["transitions_per_hour"] = transitions / elapsed_hours
        token_budget = getattr(self.train_dataset, "event_token_budget", None)
        if token_budget:
            logs["effective_event_epochs"] = n_real / float(token_budget)

    @staticmethod
    def _add_derived_metrics(logs: dict[str, float], prefix: str = "") -> None:
        n_subject = logs.get(f"{prefix}n_subject", 0.0)
        n_real = logs.get(f"{prefix}n_real", 0.0)
        n_time = logs.get(f"{prefix}n_time", 0.0)
        n_censor = logs.get(f"{prefix}n_censor", 0.0)
        n_mag = logs.get(f"{prefix}n_mag", 0.0)
        n_clipped = logs.get(f"{prefix}value_z_clipped", 0.0)
        if n_subject > 0:
            logs[f"{prefix}real_per_subject"] = n_real / n_subject
            logs[f"{prefix}censor_per_subject"] = n_censor / n_subject
        if n_real > 0:
            logs[f"{prefix}mag_per_real"] = n_mag / n_real
            logs[f"{prefix}time_per_real"] = n_time / n_real
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
            logs.update(self._drain_loss_bucket(self._loss_accum_train, update_train_totals=True))
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
            # Use the actively-trained model from accelerator-wrapped reference,
            # not self.model which can be a stale snapshot in some HF Trainer +
            # accelerator configurations. The C-index eval was returning
            # bit-identical results across thousands of training steps because
            # self.model was a frozen reference disconnected from the param
            # tensors actually being updated by optimizer.step.
            wrapped = getattr(self, "model_wrapped", None)
            if wrapped is not None and hasattr(self, "accelerator"):
                try:
                    live_model = self.accelerator.unwrap_model(wrapped)
                except Exception:
                    live_model = self.model
            else:
                live_model = self.model
            cindex_results = run_cindex(
                live_model, self._cindex_cohort,
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


_ANCESTOR_ROWS_UNREADABLE = -1   # couldn't sniff the checkpoint at all
_ANCESTOR_ROWS_ABSENT = 0        # checkpoint sniffed fine; the parameter just isn't there


def checkpoint_ancestor_rows(path: str | Path) -> int:
    """How many rows did the checkpoint's ``embed.ancestor_embedding`` have?

    ``>=1`` means the parameter is present with that many rows (1 = placeholder
    aka flat mode, >1 = hierarchical was active). ``_ANCESTOR_ROWS_ABSENT``
    (0) means the safetensors file was readable but the parameter is missing
    entirely — a pre-hierarchical checkpoint, saved before the
    ancestor_embedding module existed. ``_ANCESTOR_ROWS_UNREADABLE`` (-1)
    means we couldn't sniff at all (no safetensors lib, no model file, etc.).

    The distinction matters for the optimizer-state reset decision: a
    pre-hierarchical checkpoint has FEWER parameters than the current model
    (no ancestor_embedding.weight) and so the saved optimizer parameter
    groups don't match the live ones — the load must reset. An unreadable
    sniff can't tell us either way, so we conservatively assume the live
    state and don't reset on its account.
    """
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _ANCESTOR_ROWS_UNREADABLE
    weights = Path(path) / "model.safetensors"
    if not weights.is_file():
        return _ANCESTOR_ROWS_UNREADABLE
    try:
        with safe_open(str(weights), framework="pt") as f:  # type: ignore[no-untyped-call]
            key = "model.embed.ancestor_embedding.weight"
            keys = set(f.keys())
            if key in keys:
                return int(f.get_slice(key).get_shape()[0])
            return _ANCESTOR_ROWS_ABSENT
    except (OSError, ValueError):
        return _ANCESTOR_ROWS_UNREADABLE


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


def load_atom_registry(path: str | Path) -> dict[str, int] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        registry = {str(code): int(atom) for code, atom in json.loads(path.read_text()).items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    validate_atom_registry(registry)
    return registry


def write_atom_registry_snapshot(path: str | Path, registry: dict[str, int] | None) -> None:
    if registry is None:
        return
    validate_atom_registry(registry)
    atomic_write_json(Path(path) / ATOM_REGISTRY_FILE, registry)


def load_model_state_from_dir(model: GenterpForCausalLM, path: str | Path) -> None:
    path = Path(path)
    safetensors_path = path / "model.safetensors"
    pytorch_path = path / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path), device="cpu")
    elif pytorch_path.is_file():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"no model weights found under {path}")
    model.load_state_dict(state_dict, strict=False)


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


def save_final_model(
    trainer: transformers.Trainer,
    output_dir: str | Path,
    runtime: TorchRuntime,
    atom_registry: dict[str, int] | None = None,
) -> None:
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
    write_atom_registry_snapshot(tmp_dir, atom_registry)
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


def ensure_loaded_ancestors(model: GenterpForCausalLM, etl_dir: Path) -> str:
    """Ensure a hierarchical model has its per-atom ancestor lookup attached."""
    embed = model.model.embed
    if embed.has_ancestors() or model.model.cfg.n_ancestor_rows <= 0:
        return "checkpoint"
    path = etl_dir / ANCESTORS_FILE
    loaded = _load_ancestors(path)
    if loaded is None:
        raise FileNotFoundError(
            f"model config expects n_ancestor_rows={model.model.cfg.n_ancestor_rows}, "
            f"but checkpoint has no ancestor_ids and {path} is missing"
        )
    ancestor_ids, n_ancestor_rows = loaded
    if n_ancestor_rows != model.model.cfg.n_ancestor_rows:
        raise ValueError(
            f"{path} has n_ancestor_rows={n_ancestor_rows}, "
            f"but model config expects {model.model.cfg.n_ancestor_rows}"
        )
    embed.set_ancestor_ids(ancestor_ids)
    return str(path)


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
        # Pretraining-style optimizer hardening:
        #   • adam_beta2 = 0.95 (vs HF default 0.999) — Adam's second-moment
        #     EMA half-life at β₂ = 0.999 is ~700 steps, so noisy gradients
        #     from sampled-softmax mark loss or rare-atom value tokens leak
        #     into the variance estimate for hundreds of steps. β₂ = 0.95
        #     halves in ~14 steps, recovering faster from outlier batches.
        #   • weight_decay = 0.05 — AdamW-style decoupled decay at a
        #     pretraining-typical magnitude; ties the unused mark-out bias-
        #     free linear back to zero and keeps the residual stream's
        #     RMSNorm scales from drifting unboundedly.
        #   • max_grad_norm = 1.0 — explicit, since HF's default is 1.0 but
        #     we want to make the contract visible.
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.05,
        "max_grad_norm": 1.0,
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
    setup = ProgressLogger("train_setup", total_units=17)
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

    setup.start_unit("load atom registry", f"path={etl / ATOM_REGISTRY_FILE}")
    current_atom_registry = load_atom_registry(etl / ATOM_REGISTRY_FILE)
    if current_atom_registry is None:
        raise FileNotFoundError(
            f"{etl / ATOM_REGISTRY_FILE} is required; rerun scripts/aou_etl.py to build stable atom ids"
        )
    setup.finish_unit(
        "load atom registry",
        f"registry_atoms={len(current_atom_registry):,} max_id={max(current_atom_registry.values(), default=0):,}",
    )

    setup.start_unit(
        "load events parquet (shared)",
        "single decompressed copy reused by train + eval — prevents the ~20GB×2 OOM",
    )
    event_store = EventStore.from_parquet(etl / "events.parquet")
    setup.finish_unit(
        "load events parquet (shared)",
        f"rows={event_store.num_rows:,} chunks={event_store.num_chunks} {_rss_str()}",
    )

    setup.start_unit("build training dataset", "split=train; random windows over long histories")
    train_dataset = CohortDataset(
        etl,
        split="train",
        events=event_store,
        window_policy="random",
        last_window_fraction=0.1,
    )
    setup.finish_unit("build training dataset", f"subjects={len(train_dataset):,} {_rss_str()}")

    setup.start_unit("build evaluation dataset", "split=validation; deterministic last windows")
    eval_full = CohortDataset(etl, split="validation", events=event_store, window_policy="last")
    # Each in-loop eval pass scores a deterministic random subsample of the
    # held-out validation cohort. Random (not longest-first) so the slice is
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
    warm_start_path = final_model_path(output_dir) if resume_checkpoint is None else None
    source_checkpoint_for_registry = resume_checkpoint or warm_start_path
    checkpoint_atom_registry = (
        load_atom_registry(Path(source_checkpoint_for_registry) / ATOM_REGISTRY_FILE)
        if source_checkpoint_for_registry is not None else None
    )
    hardware_changed = resume_checkpoint is not None and not checkpoint_matches_runtime(resume_checkpoint, runtime)
    ckpt_n_atoms = checkpoint_n_atoms(resume_checkpoint) if resume_checkpoint else None
    vocab_changed = ckpt_n_atoms is not None and ckpt_n_atoms != cfg.n_atoms
    registry_changed = bool(
        source_checkpoint_for_registry is not None
        and checkpoint_atom_registry is not None
        and checkpoint_atom_registry != current_atom_registry
    )
    # Detect model-graph or shape change for ancestor_embedding. Three cases:
    #   ABSENT       : pre-hierarchical checkpoint. The current model has the
    #                  ancestor_embedding parameter; the saved optimizer state
    #                  doesn't. Loading optimizer state would raise
    #                  "parameter group that doesn't match the size of
    #                  optimizer's group". Must reset.
    #   row count mismatch : checkpoint had ancestor_embedding but with a
    #                        different number of rows (vocab grew via
    #                        scripts.build_ancestors). Param shape changed,
    #                        Adam moments unusable. Must reset.
    #   row count match    : everything aligns. Resume cleanly.
    #   UNREADABLE   : conservative no-op; assume safe.
    ckpt_ancestor_rows = (
        checkpoint_ancestor_rows(resume_checkpoint)
        if resume_checkpoint else _ANCESTOR_ROWS_UNREADABLE
    )
    target_ancestor_rows = target_n_ancestor_rows + 1
    ancestors_changed = bool(
        resume_checkpoint
        and ckpt_ancestor_rows != _ANCESTOR_ROWS_UNREADABLE
        and ckpt_ancestor_rows != target_ancestor_rows
    )
    # Any shape-axis change in the model invalidates the optimizer state (Adam
    # moments are per-parameter and per-shape). Force a state reset so we
    # rebuild fresh moments around the partially-loaded weights.
    reset_training_state = bool(
        resume_checkpoint and (hardware_changed or vocab_changed or registry_changed or ancestors_changed)
    )
    setup.finish_unit(
        "inspect checkpoints",
        f"resume_checkpoint={resume_checkpoint} warm_start_path={warm_start_path} "
        f"reset_training_state={reset_training_state} hardware_changed={hardware_changed} "
        f"vocab_changed={vocab_changed} (ckpt_n_atoms={ckpt_n_atoms} cur_n_atoms={cfg.n_atoms}) "
        f"registry_changed={registry_changed} "
        f"ancestors_changed={ancestors_changed} (ckpt_anc_rows={ckpt_ancestor_rows} cur_anc_rows={target_ancestor_rows})",
    )

    setup.start_unit("load or initialize model", "preferring resume checkpoint, then previous final model, then fresh init")
    model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))
    model.configure_atom_registry_remap(checkpoint_atom_registry, current_atom_registry)
    if warm_start_path is not None:
        load_model_state_from_dir(model, warm_start_path)
        model_source = f"warm_start={warm_start_path}"
    else:
        model_source = "fresh_init" if resume_checkpoint is None else f"resume_weights_from={resume_checkpoint}"
    if reset_training_state:
        if hardware_changed:
            reason = "hardware profile changed"
        elif registry_changed:
            reason = "atom registry changed"
        elif ancestors_changed:
            reason = "ancestor table shape changed"
        else:
            reason = f"vocab changed ({ckpt_n_atoms} -> {cfg.n_atoms} atoms)"
        print(f"genterp train {reason}; resuming model weights (partial) and rebuilding optimizer state")
    setup.finish_unit("load or initialize model", f"{model_source} params={count_parameters(model):,}")

    setup.start_unit("configure mark negative sampler", "frequency-weighted atom negatives from training events")
    train_atom_counts = train_dataset.atom_counts(model.model.cfg.n_atoms)
    model.model.tpp.set_mark_noise_distribution(torch.from_numpy(train_atom_counts))
    setup.finish_unit(
        "configure mark negative sampler",
        f"negatives={model.model.cfg.sampled_mark_negatives:,} "
        f"train_event_token_budget={int(train_atom_counts.sum()):,}",
    )

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
        "OHDSI PhenotypeLibrary canonical disease sweep (cached by aou_etl.py)",
    )
    cindex_cohort = None
    try:
        from genterp.eval_cindex import (
            DEFAULT_SWEEP_TOP_N,
            build_cohort_condition_phenotypes,
            prepare_cindex_cohort,
        )
        sweep_phenotypes = build_cohort_condition_phenotypes(etl, top_n=DEFAULT_SWEEP_TOP_N)
        if not sweep_phenotypes:
            raise SystemExit(
                "OHDSI sweep returned no phenotypes — ohdsi_disease_phenotypes.json "
                "missing from ETL cache. Re-run scripts/aou_etl.py to build the "
                "OHDSI PhenotypeLibrary canonical disease list."
            )
        cindex_cohort = prepare_cindex_cohort(
            etl, vocab,
            events=event_store,
            pin_memory=runtime.dataloader_pin_memory,
            phenotypes=sweep_phenotypes,
            max_subjects=CINDEX_MAX_SUBJECTS_IN_LOOP,
            split="validation",
        )
        setup.finish_unit(
            "prepare C-index cohort",
            f"mode=sweep (top-{DEFAULT_SWEEP_TOP_N} OHDSI Conditions)  "
            f"subjects={len(cindex_cohort.subjects):,}  "
            f"diseases={len(cindex_cohort.disease_names)}",
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
        callbacks=[RuntimeStateCallback(runtime, current_atom_registry), VerboseTrainerProgressCallback()],
    )
    setup.finish_unit("instantiate Trainer", "trainer ready")

    setup.start_unit("run training loop", "Trainer owns batch loading, forward/backward, optimizer, eval, and checkpoint steps")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    setup.finish_unit("run training loop", f"global_step={trainer.state.global_step:,}")

    setup.start_unit("save final model", f"output_dir={output_dir}")
    save_final_model(trainer, output_dir, runtime, current_atom_registry)
    setup.finish_unit("save final model", f"global_step={trainer.state.global_step:,}")


if __name__ == "__main__":
    main()

"""Train Genterp on AoU OMOP."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint

from genterp.data import AncestorMap, AtomVocab, CohortDataset, collate
from genterp.modeling import Genterp, GenterpConfig


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


def latest_checkpoint(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


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


def configure_torch_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    configure_torch_runtime()
    etl = Path.home() / "genterp" / "etl"
    output_dir = Path.home() / "genterp" / "runs"
    vocab = AtomVocab(dict(json.loads((etl / "vocab.json").read_text())))
    ancestors = AncestorMap.from_omop_concept_ancestor(vocab, json.loads((etl / "ancestors.json").read_text()))
    dataset = CohortDataset(etl, ancestors)

    cfg = GenterpConfig(n_atoms=len(vocab), dim=512, n_heads=8, n_layers=8)
    model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))
    mu, sigma, has_mag = _load_value_stats(etl / "value_stats.json", vocab)
    model.model.value_mod.set_stats(mu, sigma, has_mag)

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=4,
            learning_rate=3e-4,
            warmup_steps=500,
            max_steps=50_000,
            lr_scheduler_type="cosine",
            bf16=True,
            tf32=True,
            torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_mode="max-autotune",
            save_steps=2_000,
            logging_steps=50,
            optim="adamw_torch_fused",
            dataloader_num_workers=4,
            dataloader_persistent_workers=True,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="none",
            skip_memory_metrics=True,
            restore_callback_states_from_checkpoint=True,
            save_safetensors=True,
        ),
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir))
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    main()

"""Train Genterp on AoU MEDS."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

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

    def forward(self, target_atoms: torch.Tensor | None = None, **batch: Any):
        logits = self.model(**batch)
        loss = _next_atom_loss(logits, target_atoms, batch["event_pad"]) if target_atoms is not None else None
        return transformers.modeling_outputs.CausalLMOutput(loss=loss, logits=logits)


def _next_atom_loss(logits: torch.Tensor, targets: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    pred = logits[:, :-1]
    tgt = targets[:, 1:]
    tgt_pad = pad[:, 1:]
    ce = F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
    return ce.masked_fill(tgt_pad, 0.0).sum() / (~tgt_pad).sum().clamp(min=1)


def main() -> None:
    etl = Path.home() / "genterp" / "etl"
    vocab = AtomVocab(dict(json.loads((etl / "vocab.json").read_text())))
    ancestors = AncestorMap.from_omop_concept_ancestor(vocab, json.loads((etl / "ancestors.json").read_text()))
    dataset = CohortDataset(etl, ancestors)

    cfg = GenterpConfig(n_atoms=len(vocab), dim=512, n_heads=8, n_layers=8)
    model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(Path.home() / "genterp" / "runs"),
            per_device_train_batch_size=4,
            learning_rate=3e-4,
            warmup_steps=500,
            max_steps=50_000,
            lr_scheduler_type="cosine",
            bf16=True,
            torch_compile=True,
            save_steps=2_000,
            logging_steps=50,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(Path.home() / "genterp" / "runs" / "final"))


if __name__ == "__main__":
    main()

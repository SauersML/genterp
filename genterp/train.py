"""Train Genterp on AoU OMOP."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers

from genterp.data import AncestorMap, AtomVocab, CohortDataset, collate
from genterp.modeling import Genterp, GenterpConfig, marked_tpp_loss


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

    def forward(self, target_atoms: torch.Tensor | None = None, censor_age: torch.Tensor | None = None, **batch: Any):
        out = self.model(**batch)
        loss = None
        if target_atoms is not None and censor_age is not None:
            ld = marked_tpp_loss(
                self.model.tpp,
                out["hidden"],
                batch["event_ages"],
                target_atoms,
                batch["event_pad"],
                censor_age,
            )
            loss = ld["loss"]
        return transformers.modeling_outputs.CausalLMOutput(loss=loss, logits=out["hidden"])


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

"""Train Genterp on AoU MEDS using HuggingFace Trainer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

from genterp.data import AncestorMap, AtomVocab, MEDSDataset, collate
from genterp.modeling import Genterp, GenterpConfig


HOME = Path.home()
MEDS_DIR = HOME / "meds"
ETL_DIR = HOME / "genterp" / "etl"
VOCAB_PATH = ETL_DIR / "vocab.json"
ANCESTOR_PATH = ETL_DIR / "ancestors.json"
OUTPUT_DIR = HOME / "genterp" / "runs"

DIM = 512
N_HEADS = 8
N_LAYERS = 8
BATCH_SIZE = 4
GRAD_ACCUM = 1
LR = 3e-4
WARMUP_STEPS = 500
MAX_STEPS = 50_000
BF16 = True
COMPILE = False


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


def _load_vocab() -> AtomVocab:
    return AtomVocab(dict(json.loads(VOCAB_PATH.read_text())))


def _load_ancestors(vocab: AtomVocab) -> AncestorMap:
    raw: dict[str, list[str]] = json.loads(ANCESTOR_PATH.read_text())
    return AncestorMap.from_omop_concept_ancestor(vocab, raw)


def main() -> None:
    vocab = _load_vocab()
    ancestors = _load_ancestors(vocab)
    dataset = MEDSDataset(MEDS_DIR, vocab, ancestors)
    cfg = GenterpConfig(n_atoms=len(vocab), dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS)
    model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))

    if COMPILE:
        model.model = torch.compile(model.model)

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            lr_scheduler_type="cosine",
            bf16=BF16,
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
    trainer.save_model(str(OUTPUT_DIR / "final"))


if __name__ == "__main__":
    main()

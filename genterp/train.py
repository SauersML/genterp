"""HF Trainer driver for Genterp."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

from genterp.data import AncestorMap, AtomVocab, MEDSDataset, collate
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


@dataclass
class TrainArgs:
    meds_dir: str
    vocab_path: str
    ancestor_path: str
    output_dir: str
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    batch_size: int = 4
    lr: float = 3e-4
    warmup_steps: int = 500
    max_steps: int = 50_000
    bf16: bool = True
    compile: bool = False
    grad_accum: int = 1


def _load_vocab(path: str) -> AtomVocab:
    with open(path) as f:
        return AtomVocab(dict(json.load(f)))


def _load_ancestors(path: str, vocab: AtomVocab) -> AncestorMap:
    with open(path) as f:
        raw: dict[str, list[str]] = json.load(f)
    return AncestorMap.from_omop_concept_ancestor(vocab, raw)


def build_dataset(args: TrainArgs) -> tuple[MEDSDataset, AtomVocab]:
    vocab = _load_vocab(args.vocab_path)
    ancestors = _load_ancestors(args.ancestor_path, vocab)
    return MEDSDataset(args.meds_dir, vocab, ancestors), vocab


def run(args: TrainArgs) -> None:
    dataset, vocab = build_dataset(args)
    cfg = GenterpConfig(n_atoms=len(vocab), dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers)
    model = GenterpForCausalLM(GenterpHFConfig(genterp_cfg=asdict(cfg)))

    if args.compile:
        model.model = torch.compile(model.model)

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            lr_scheduler_type="cosine",
            bf16=args.bf16,
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
    trainer.save_model(str(Path(args.output_dir) / "final"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meds-dir", required=True)
    p.add_argument("--vocab", dest="vocab_path", required=True)
    p.add_argument("--ancestors", dest="ancestor_path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", dest="n_heads", type=int, default=8)
    p.add_argument("--layers", dest="n_layers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", dest="warmup_steps", type=int, default=500)
    p.add_argument("--steps", dest="max_steps", type=int, default=50_000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--compile", action="store_true")
    run(TrainArgs(**vars(p.parse_args())))


if __name__ == "__main__":
    main()

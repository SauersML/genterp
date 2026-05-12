"""Cohort timelines from OMOP-derived Parquet shards (no MEDS layer)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


PAD_ATOM = 0


@dataclass
class AtomVocab:
    """MEDS-style 'VOCAB/CODE' string -> atom index. Index 0 reserved for PAD."""

    code_to_atom: dict[str, int]

    def __len__(self) -> int:
        return len(self.code_to_atom) + 1

    def encode(self, code: str) -> int:
        return self.code_to_atom.get(code, PAD_ATOM)


@dataclass
class AncestorMap:
    """Code -> [leaf_atom, *strict_ancestor_atoms]. bag[0] is the next-event target."""

    closure: dict[str, list[int]]

    def bag(self, code: str) -> list[int]:
        return self.closure.get(code, [])

    @classmethod
    def from_omop_concept_ancestor(
        cls, vocab: AtomVocab, code_to_ancestor_codes: dict[str, Iterable[str]]
    ) -> "AncestorMap":
        out: dict[str, list[int]] = {}
        for code, ancestors in code_to_ancestor_codes.items():
            leaf = vocab.encode(code)
            if leaf == PAD_ATOM:
                continue
            anc = [vocab.encode(a) for a in ancestors if a in vocab.code_to_atom and a != code]
            out[code] = [leaf, *anc]
        return cls(out)


class CohortDataset(Dataset):
    """events.parquet sorted by (subject_id, time_seconds); subjects.parquet holds per-subject row offsets, sex, birth."""

    def __init__(self, data_dir: str | Path, ancestors: AncestorMap, max_events: int = 4096):
        data_dir = Path(data_dir)
        self.events = pq.read_table(data_dir / "events.parquet", memory_map=True)
        subjects = pl.read_parquet(data_dir / "subjects.parquet").sort("subject_id")
        self.start = subjects["start"].to_numpy()
        self.end = subjects["end"].to_numpy()
        self.sex = subjects["sex"].to_numpy()
        self.birth_seconds = subjects["birth_seconds"].to_numpy()
        self.censor_seconds = subjects["censor_seconds"].to_numpy()
        self.ancestors = ancestors
        self.max_events = max_events

    def __len__(self) -> int:
        return len(self.start)

    def __getitem__(self, idx: int) -> dict:
        s, e = int(self.start[idx]), int(self.end[idx])
        slice_ = self.events.slice(s, e - s + 1)
        times = slice_.column("time_seconds").to_numpy().tolist()
        codes = slice_.column("code").to_pylist()
        birth = float(self.birth_seconds[idx])

        static_bags: list[list[int]] = []
        event_bags: list[list[int]] = []
        event_ages: list[float] = []
        for t, code in zip(times, codes):
            bag = self.ancestors.bag(code)
            if not bag:
                continue
            delta_days = (t - birth) / 86400.0
            if delta_days <= 0.5:
                static_bags.append(bag)
            else:
                event_bags.append(bag)
                event_ages.append(delta_days)

        if len(event_bags) > self.max_events:
            event_bags = event_bags[: self.max_events]
            event_ages = event_ages[: self.max_events]

        censor_age_days = (float(self.censor_seconds[idx]) - birth) / 86400.0
        return {
            "sex": int(self.sex[idx]),
            "static_bags": static_bags,
            "event_bags": event_bags,
            "event_ages": np.asarray(event_ages, dtype=np.float32),
            "censor_age_days": float(censor_age_days),
        }


def _pack(bags_per_seq: list[list[list[int]]]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Pack [[bag, ...], ...] into (atoms, offsets, (B, S)) for nn.EmbeddingBag. Always emits S >= 1."""
    B = len(bags_per_seq)
    S = max((len(seq) for seq in bags_per_seq), default=0)
    S = max(S, 1)
    flat_offsets: list[int] = []
    flat_atoms: list[int] = []
    running = 0
    for seq in bags_per_seq:
        for s in range(S):
            flat_offsets.append(running)
            if s < len(seq):
                flat_atoms.extend(seq[s])
                running += len(seq[s])
            else:
                flat_atoms.append(PAD_ATOM)
                running += 1
    return (
        torch.tensor(flat_atoms, dtype=torch.long),
        torch.tensor(flat_offsets, dtype=torch.long),
        (B, S),
    )


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    static_atoms, static_offsets, static_shape = _pack([b["static_bags"] for b in batch])
    event_atoms, event_offsets, event_shape = _pack([b["event_bags"] for b in batch])
    M, T = static_shape[1], event_shape[1]

    static_pad = torch.ones(B, M, dtype=torch.bool)
    event_pad = torch.ones(B, T, dtype=torch.bool)
    event_ages = torch.zeros(B, T, dtype=torch.float32)
    target_atoms = torch.zeros(B, T, dtype=torch.long)
    for i, b in enumerate(batch):
        static_pad[i, : len(b["static_bags"])] = False
        n_ev = len(b["event_bags"])
        event_pad[i, :n_ev] = False
        if n_ev:
            event_ages[i, :n_ev] = torch.from_numpy(b["event_ages"])
            target_atoms[i, :n_ev] = torch.tensor([bag[0] for bag in b["event_bags"]], dtype=torch.long)
    static_pad[:, 0] = False

    return {
        "static_atoms": static_atoms,
        "static_offsets": static_offsets,
        "static_pad": static_pad,
        "static_shape": static_shape,
        "event_atoms": event_atoms,
        "event_offsets": event_offsets,
        "event_pad": event_pad,
        "event_ages": event_ages,
        "target_atoms": target_atoms,
        "censor_age": torch.tensor([b["censor_age_days"] for b in batch], dtype=torch.float32),
        "sex": torch.tensor([b["sex"] for b in batch], dtype=torch.long),
    }

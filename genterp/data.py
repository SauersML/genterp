"""MEDS data loaders and collator. Subjects come from meds_reader; codes are MEDS strings; ancestors are OMOP IS-A closures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_ATOM = 0


@dataclass
class AtomVocab:
    """MEDS code (str) -> atom index. Index 0 reserved for PAD."""

    code_to_atom: dict[str, int]

    @classmethod
    def from_codes(cls, codes: Iterable[str]) -> "AtomVocab":
        m: dict[str, int] = {}
        for c in codes:
            if c not in m:
                m[c] = len(m) + 1
        return cls(m)

    def __len__(self) -> int:
        return len(self.code_to_atom) + 1

    def encode(self, code: str) -> int:
        return self.code_to_atom.get(code, PAD_ATOM)


@dataclass
class AncestorMap:
    """MEDS code -> [leaf_atom, *strict_ancestor_atoms]. bag[0] is the next-event target."""

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


_OMOP_MALE = frozenset({"Gender/MALE", "Gender/M", "Gender/8507"})
_OMOP_FEMALE = frozenset({"Gender/FEMALE", "Gender/F", "Gender/8532"})


def _subject_sex(subject, male_codes: frozenset[str], female_codes: frozenset[str]) -> int:
    for ev in subject.events:
        if ev.code in male_codes:
            return 1
        if ev.code in female_codes:
            return 0
    return 0


def _split_static_event(subject, ancestors: AncestorMap):
    birth_time = next((ev.time for ev in subject.events if ev.code == "MEDS_BIRTH"), None)
    if birth_time is None:
        return [], [], np.zeros(0, dtype=np.float32)

    static_bags: list[list[int]] = []
    event_bags: list[list[int]] = []
    event_ages: list[float] = []
    for ev in subject.events:
        bag = ancestors.bag(ev.code)
        if not bag:
            continue
        delta_days = (ev.time - birth_time).total_seconds() / 86400.0
        if delta_days <= 0.5:
            static_bags.append(bag)
        else:
            event_bags.append(bag)
            event_ages.append(delta_days)
    return static_bags, event_bags, np.asarray(event_ages, dtype=np.float32)


class MEDSDataset(Dataset):
    def __init__(
        self,
        db_path: str | Path,
        vocab: AtomVocab,
        ancestors: AncestorMap,
        subject_ids: list[int] | None = None,
        max_events: int = 4096,
        male_codes: Iterable[str] = _OMOP_MALE,
        female_codes: Iterable[str] = _OMOP_FEMALE,
    ):
        import meds_reader  # noqa: PLC0415 — optional heavy dep, only needed when MEDSDataset is constructed

        self.db = meds_reader.SubjectDatabase(str(db_path))
        self.vocab = vocab
        self.ancestors = ancestors
        self.max_events = max_events
        self.male_codes = frozenset(male_codes)
        self.female_codes = frozenset(female_codes)
        all_ids = set(self.db.subject_ids())
        if subject_ids is None:
            self.subject_ids = sorted(all_ids)
        else:
            missing = [s for s in subject_ids if s not in all_ids]
            if missing:
                raise KeyError(missing[:5])
            self.subject_ids = list(subject_ids)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        subject = self.db[self.subject_ids[idx]]
        sex = _subject_sex(subject, self.male_codes, self.female_codes)
        static, events, ages = _split_static_event(subject, self.ancestors)
        if len(events) > self.max_events:
            events = events[: self.max_events]
            ages = ages[: self.max_events]
        return {"sex": sex, "static_bags": static, "event_bags": events, "event_ages": ages}


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
    static_pad[:, 0] = False  # guarantee at least one attendable static key per subject

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
        "sex": torch.tensor([b["sex"] for b in batch], dtype=torch.long),
    }

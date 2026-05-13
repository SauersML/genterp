"""Cohort timelines from OMOP-derived Parquet shards (no MEDS layer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from genterp.progress import ProgressLogger

PAD_ATOM = 0


@dataclass
class AtomVocab:
    """MEDS-style 'VOCAB/CODE' string -> atom index. Index 0 reserved for PAD."""

    code_to_atom: dict[str, int]

    def __len__(self) -> int:
        return len(self.code_to_atom) + 1

    def encode(self, code: str) -> int:
        return self.code_to_atom.get(code, PAD_ATOM)


class CohortDataset(Dataset):
    """events.parquet sorted by (subject_id, time_seconds); subjects.parquet holds per-subject row offsets, sex, birth.

    ``split`` filters subjects.parquet by the ETL-assigned split column (e.g. "train", "test").
    The shared events.parquet is unchanged — splits just expose different subject row-ranges.
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_events: int = 4096,
        split: str | None = None,
    ):
        data_dir = Path(data_dir)
        logger = ProgressLogger(f"cohort_dataset:{split or 'all'}", total_units=6)
        events_path = data_dir / "events.parquet"
        logger.start_unit("read events parquet", f"path={events_path} columns=time_seconds,atom,value")
        events = pq.read_table(events_path, columns=["time_seconds", "atom", "value"], memory_map=True)
        logger.finish_unit("read events parquet", f"rows={events.num_rows:,}")

        logger.start_unit("materialize event columns", "combining Arrow chunks into numpy columns")
        self.event_times = events.column("time_seconds").combine_chunks().to_numpy(zero_copy_only=False)
        self.event_atoms = events.column("atom").combine_chunks().to_numpy(zero_copy_only=True)
        self.event_values = events.column("value").combine_chunks().to_numpy(zero_copy_only=False)
        logger.finish_unit("materialize event columns", f"atoms={len(self.event_atoms):,} values={len(self.event_values):,}")

        logger.start_unit("read subjects parquet", f"path={data_dir / 'subjects.parquet'}")
        subjects = pl.read_parquet(data_dir / "subjects.parquet").sort("subject_id")
        logger.finish_unit("read subjects parquet", f"subjects={subjects.height:,}")

        logger.start_unit("apply subject split filter", f"requested split={split!r}")
        if split is not None:
            if "split" not in subjects.columns:
                raise ValueError(
                    f"subjects.parquet has no 'split' column; rerun aou_etl to materialize it (requested split={split!r})"
                )
            subjects = subjects.filter(pl.col("split") == split)
            if subjects.height == 0:
                raise ValueError(f"no subjects in split={split!r}")
        logger.finish_unit("apply subject split filter", f"remaining_subjects={subjects.height:,}")

        logger.start_unit("materialize subject arrays", "start/end offsets, sex, birth time, censor time")
        self.split = split
        self.start = subjects["start"].to_numpy()
        self.end = subjects["end"].to_numpy()
        self.sex = subjects["sex"].to_numpy()
        self.birth_seconds = subjects["birth_seconds"].to_numpy()
        self.censor_seconds = subjects["censor_seconds"].to_numpy()
        logger.finish_unit("materialize subject arrays", f"subjects={len(self.start):,}")

        logger.start_unit("compute per-subject sequence lengths", f"max_events={max_events:,}")
        self.max_events = max_events
        self.lengths = np.minimum(np.maximum(self.end - self.start + 1, 1), max_events).astype(np.int64).tolist()
        mean_length = float(np.mean(self.lengths)) if self.lengths else 0.0
        logger.finish_unit("compute per-subject sequence lengths", f"subjects={len(self.lengths):,} mean_length={mean_length:.1f}")

    def __len__(self) -> int:
        return len(self.start)

    def atom_counts(self, n_atoms: int) -> np.ndarray:
        counts = np.bincount(np.asarray(self.event_atoms), minlength=n_atoms)[:n_atoms].astype(np.float32, copy=False)
        counts[PAD_ATOM] = 0.0
        if counts.sum() <= 0:
            raise ValueError("event atom cache has no non-PAD atoms")
        return counts

    def __getitem__(self, idx: int) -> dict:
        s, e = int(self.start[idx]), int(self.end[idx])
        stop = e + 1
        birth = float(self.birth_seconds[idx])
        max_events = self.max_events

        atoms = np.asarray(self.event_atoms[s:stop])
        times = np.asarray(self.event_times[s:stop])
        delta_days = (times - birth) / 86400.0
        real_atom = atoms != PAD_ATOM

        static_idx = np.where((delta_days <= 0.5) & real_atom)[0]
        event_idx = np.where((delta_days > 0.5) & real_atom)[0][-max_events:]

        static_atoms_arr = atoms[static_idx]
        event_atoms_arr = atoms[event_idx]
        event_ages_arr = delta_days[event_idx]
        event_values_arr = np.asarray(self.event_values[s:stop])[event_idx]

        censor_age_days = (float(self.censor_seconds[idx]) - birth) / 86400.0
        return {
            "sex": int(self.sex[idx]),
            "static_atoms": [int(a) for a in static_atoms_arr.tolist()],
            "event_atoms": [int(a) for a in event_atoms_arr.tolist()],
            "event_ages": event_ages_arr.astype(np.float32, copy=False),
            "event_values": event_values_arr.astype(np.float32, copy=False),
            "censor_age_days": float(censor_age_days),
            "length": int(event_atoms_arr.shape[0]),
        }


def _pad_atoms(seqs: list[list[int]]) -> torch.Tensor:
    """Pad per-subject atom sequences. Always emits S >= 1."""
    B = len(seqs)
    S = max((len(seq) for seq in seqs), default=0)
    S = max(S, 1)
    out = torch.full((B, S), PAD_ATOM, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    static_atoms = _pad_atoms([b["static_atoms"] for b in batch])
    event_atoms = _pad_atoms([b["event_atoms"] for b in batch])
    M, T = static_atoms.shape[1], event_atoms.shape[1]

    static_pad = torch.ones(B, M, dtype=torch.bool)
    event_pad = torch.ones(B, T, dtype=torch.bool)
    event_ages = torch.zeros(B, T, dtype=torch.float32)
    event_values = torch.full((B, T), float("nan"), dtype=torch.float32)
    target_atoms = torch.zeros(B, T, dtype=torch.long)
    for i, b in enumerate(batch):
        static_pad[i, : len(b["static_atoms"])] = False
        n_ev = len(b["event_atoms"])
        event_pad[i, :n_ev] = False
        if n_ev:
            event_ages[i, :n_ev] = torch.from_numpy(b["event_ages"])
            event_values[i, :n_ev] = torch.from_numpy(b["event_values"])
            target_atoms[i, :n_ev] = event_atoms[i, :n_ev]
    static_pad[:, 0] = False

    return {
        "static_atoms": static_atoms,
        "static_pad": static_pad,
        "event_atoms": event_atoms,
        "event_pad": event_pad,
        "event_ages": event_ages,
        "event_values": event_values,
        "target_atoms": target_atoms,
        "censor_age": torch.tensor([b["censor_age_days"] for b in batch], dtype=torch.float32),
        "sex": torch.tensor([b["sex"] for b in batch], dtype=torch.long),
        "length": torch.tensor([b.get("length", len(b["event_atoms"])) for b in batch], dtype=torch.long),
    }

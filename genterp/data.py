"""Cohort timelines from OMOP-derived Parquet shards (no MEDS layer)."""

from __future__ import annotations

import hashlib
import json
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


@dataclass
class CodeAtomMap:
    """Code -> collapsed atom index."""

    code_to_atom: dict[str, int]

    def atom(self, code: str) -> int:
        return self.code_to_atom.get(code, PAD_ATOM)

    @classmethod
    def from_vocab(cls, vocab: AtomVocab) -> CodeAtomMap:
        return cls({code: atom for code, atom in vocab.code_to_atom.items() if atom != PAD_ATOM})


class CohortDataset(Dataset):
    """events.parquet sorted by (subject_id, time_seconds); subjects.parquet holds per-subject row offsets, sex, birth.

    ``split`` filters subjects.parquet by the ETL-assigned split column (e.g. "train", "test").
    The shared events.parquet is unchanged — splits just expose different subject row-ranges.
    """

    def __init__(
        self,
        data_dir: str | Path,
        code_atoms: CodeAtomMap,
        max_events: int = 4096,
        split: str | None = None,
    ):
        data_dir = Path(data_dir)
        logger = ProgressLogger(f"cohort_dataset:{split or 'all'}", total_units=7)
        events_path = data_dir / "events.parquet"
        logger.start_unit("read events parquet", f"path={events_path} columns=time_seconds,code,value")
        events = pq.read_table(events_path, columns=["time_seconds", "code", "value"], memory_map=True)
        logger.finish_unit("read events parquet", f"rows={events.num_rows:,}")

        logger.start_unit("materialize event columns", "combining Arrow chunks into numpy/list columns")
        self.event_times = events.column("time_seconds").combine_chunks().to_numpy(zero_copy_only=False)
        event_codes = events.column("code").to_pylist()
        self.event_values = events.column("value").combine_chunks().to_numpy(zero_copy_only=False)
        logger.finish_unit("materialize event columns", f"codes={len(event_codes):,} values={len(self.event_values):,}")

        logger.start_unit("encode event atom cache", "mapping OMOP code strings to collapsed atom ids")
        self.event_atoms = _cached_event_atoms(data_dir, events_path, event_codes, code_atoms, logger)
        logger.finish_unit("encode event atom cache", f"atoms={len(self.event_atoms):,}")

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

    def __getitem__(self, idx: int) -> dict:
        s, e = int(self.start[idx]), int(self.end[idx])
        stop = e + 1
        times = self.event_times[s:stop]
        birth = float(self.birth_seconds[idx])
        atoms = self.event_atoms
        max_events = self.max_events

        static_atoms: list[int] = []
        event_atoms: list[int] = []
        event_ages: list[float] = []
        event_values: list[float] = []
        values = self.event_values
        for offset, t in enumerate(times, start=s):
            atom = int(atoms[offset])
            if atom == PAD_ATOM:
                continue
            delta_days = (t - birth) / 86400.0
            if delta_days <= 0.5:
                static_atoms.append(atom)
            elif len(event_atoms) < max_events:
                event_atoms.append(atom)
                event_ages.append(delta_days)
                event_values.append(float(values[offset]))
                if len(event_atoms) == max_events:
                    break

        censor_age_days = (float(self.censor_seconds[idx]) - birth) / 86400.0
        return {
            "sex": int(self.sex[idx]),
            "static_atoms": static_atoms,
            "event_atoms": event_atoms,
            "event_ages": np.asarray(event_ages, dtype=np.float32),
            "event_values": np.asarray(event_values, dtype=np.float32),
            "censor_age_days": float(censor_age_days),
            "length": len(event_atoms),
        }


def _code_atoms_fingerprint(code_atoms: CodeAtomMap) -> str:
    payload = json.dumps(sorted(code_atoms.code_to_atom.items()), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _events_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _cached_event_atoms(
    data_dir: Path,
    events_path: Path,
    codes: list[str],
    code_atoms: CodeAtomMap,
    logger: ProgressLogger,
) -> np.ndarray:
    cache_dir = data_dir / ".genterp_cache"
    cache_name = f"event_atoms-{_events_fingerprint(events_path)}-{_code_atoms_fingerprint(code_atoms)}.npy"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        logger.log("event atom cache hit", f"path={cache_path}")
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == (len(codes),):
            logger.log("event atom cache validated", f"shape={cached.shape} expected_codes={len(codes):,}")
            return cached
        logger.log("event atom cache stale", f"shape={cached.shape} expected_codes={len(codes):,}; rebuilding")
        cache_path.unlink()

    logger.log("event atom cache miss", f"encoding codes={len(codes):,} cache_path={cache_path}")
    encoded = np.fromiter((code_atoms.atom(code) for code in codes), dtype=np.uint32, count=len(codes))
    non_pad = int(np.count_nonzero(encoded))
    logger.log("event atom encoding complete", f"encoded={len(encoded):,} non_pad={non_pad:,} pad={len(encoded) - non_pad:,}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, encoded)
    tmp.replace(cache_path)
    logger.log("event atom cache written", f"path={cache_path}")
    return np.load(cache_path, mmap_mode="r")


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

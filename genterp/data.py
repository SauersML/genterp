"""Cohort timelines from OMOP-derived Parquet shards (no MEDS layer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from genterp.progress import ProgressLogger

PAD_ATOM = 0
WindowPolicy = Literal["last", "random", "mixed"]
# 'last'  : tail window (most-recent max_events). Stable, biases toward
#           end-of-record / late-disease / high-utilization trajectories.
# 'random': random-anchor window. Exposes earlier slices but still capped at
#           max_events, so long records are still seen at full length.
# 'mixed' : per-sample choice between {tail, random_anchor} drawn from
#           ``mixed_window_weights``. Long windows are preserved — each draw
#           still emits exactly max_events events when the trajectory has at
#           least that many. Switching from 'last' to 'mixed' improves
#           coverage of early-trajectory dynamics without shortening windows.

# Role enum mirrors ``scripts/aou_etl.py``. Kept in sync by hand because
# importing the ETL script at training time would drag BigQuery deps in.
ROLE_UNKNOWN = 255  # sentinel for events.parquet predating the role column
ROLE_DEMOGRAPHIC_RACE = 10
ROLE_DEMOGRAPHIC_ETHNICITY = 11
STATIC_ROLES = frozenset({ROLE_DEMOGRAPHIC_RACE, ROLE_DEMOGRAPHIC_ETHNICITY})


def _derive_event_groups(times: np.ndarray) -> np.ndarray:
    """Dense per-subject group index: contiguous identical timestamps share an id.

    Events are stored sorted by (subject_id, time_seconds, atom), so any run
    of equal ``time_seconds`` within a subject's slice is a single clinical
    timestamp / visit-like atom set. The model-side bag loss uses this to
    treat same-time atoms as a set instead of an arbitrary serialization.
    """
    if times.size == 0:
        return np.zeros(0, dtype=np.int32)
    diff = np.concatenate(([0], (np.diff(times) != 0).astype(np.int32)))
    return np.cumsum(diff, dtype=np.int32)


def _assert_arrow_integer_range(array: pa.ChunkedArray, *, name: str, min_value: int, max_value: int) -> None:
    bounds = pc.min_max(array).as_py()
    observed_min = bounds["min"]
    observed_max = bounds["max"]
    if observed_min is None or observed_max is None:
        return
    if observed_min < min_value or observed_max > max_value:
        raise OverflowError(
            f"{name} cannot be compacted safely: observed range "
            f"[{observed_min}, {observed_max}] exceeds [{min_value}, {max_value}]"
        )


def _select_window(
    event_idx: np.ndarray,
    *,
    max_events: int,
    policy: WindowPolicy,
    last_window_fraction: float,
    mixed_tail_weight: float = 0.5,
) -> np.ndarray:
    if event_idx.size <= max_events:
        return event_idx
    if policy == "mixed":
        # Bernoulli draw between tail and random-anchor. Long windows are
        # preserved either way — each branch emits exactly max_events events.
        tail = np.random.random() < float(mixed_tail_weight)
    else:
        tail = policy == "last" or np.random.random() < last_window_fraction
    if tail:
        return event_idx[-max_events:]
    max_start = event_idx.size - max_events
    start = int(np.random.randint(0, max_start + 1))
    return event_idx[start : start + max_events]


def _shuffle_same_time(event_idx: np.ndarray, times: np.ndarray) -> np.ndarray:
    if event_idx.size <= 1:
        return event_idx
    out = event_idx.copy()
    window_times = times[out]
    boundaries = np.flatnonzero(window_times[1:] != window_times[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [out.size]))
    for start, stop in zip(starts, stops, strict=True):
        if stop - start > 1:
            out[start:stop] = np.random.permutation(out[start:stop])
    return out


def _drop_events(event_idx: np.ndarray, *, drop_prob: float) -> np.ndarray:
    if drop_prob <= 0.0 or event_idx.size <= 1:
        return event_idx
    keep = np.random.random(event_idx.size) >= drop_prob
    if keep.any():
        return event_idx[keep]
    return event_idx[-1:]


@dataclass
class AtomVocab:
    """MEDS-style 'VOCAB/CODE' string -> atom index. Index 0 reserved for PAD.

    After vocabulary collapse, many distinct codes map to the same atom (an
    ancestor concept covering its descendants). ``len(self)`` must return the
    number of distinct *atoms* (i.e. ``max_atom_id + 1``), not the number of
    codes — that's what the model's embedding/output tables are sized against.
    Pre-fix the model was being allocated at ``len(code_to_atom) + 1`` slots,
    ~4× the actual vocab and a giant chunk of wasted parameters/optimizer state.
    """

    code_to_atom: dict[str, int]

    def __len__(self) -> int:
        if not self.code_to_atom:
            return 1  # PAD only
        return max(self.code_to_atom.values()) + 1

    def encode(self, code: str) -> int:
        return self.code_to_atom.get(code, PAD_ATOM)


@dataclass
class EventStore:
    """Decompressed events.parquet columns, intended to be loaded *once* and shared.

    Why this exists: ``pq.read_table(..., memory_map=True)`` does NOT avoid
    decompression — it mmaps the on-disk *compressed* parquet bytes, then
    decompresses every value into fresh Arrow heap buffers. For our 3.74GB
    zstd events.parquet that's ~20GB of resident Arrow ChunkedArrays per call.

    Train+eval datasets used to call ``pq.read_table`` independently, so the
    second one tried to allocate another ~20GB on top of the first. On the
    V100 box (~60GB RAM) the eval load got SIGKILLed by the kernel mid-read,
    silently — no Python frame to dump a traceback.

    Load once via ``EventStore.from_parquet``, hand the same instance to
    every ``CohortDataset``, and the second dataset is effectively free.
    """

    time_seconds: pa.ChunkedArray
    atom: pa.ChunkedArray
    value: pa.ChunkedArray
    role: pa.ChunkedArray

    @property
    def num_rows(self) -> int:
        return int(self.atom.length())

    @property
    def num_chunks(self) -> int:
        return int(self.atom.num_chunks)

    @classmethod
    def from_parquet(cls, path: str | Path) -> EventStore:
        """Load the shared event store, compacting dtypes column-by-column so
        peak RAM stays bounded on the full-cohort (1.1B-row) pull.

        The on-disk schema is int64 time / uint32 atom / float64 value (= 20
        B/row). At 1.1B rows that's ~22 GB decompressed when read in one go,
        and ``pq.read_table`` with all three columns at once OOM-killed the
        process before training could start. We instead:

          1. Read each column independently via ParquetFile.read().
          2. Immediately cast to a compact dtype (int32 / uint16 / float32 =
             10 B/row total, ~11 GB resident).
          3. Drop the original wide-typed table reference so peak transient
             is bounded by the single biggest column (~9 GB) rather than the
             sum of all three (~22 GB).

        Dtype safety:
          * int32 seconds spans ±68 y around 1970 — fine for AoU dates.
          * uint16 atom holds 65k ids; current vocab is ≤42k.
          * float32 value preserves ~7 sig figs; clinical labs only need ~3.
        """
        path = Path(path)
        logger = ProgressLogger("event_store", total_units=2)
        logger.start_unit("read events parquet (shared)", f"path={path}")
        pf = pq.ParquetFile(str(path), memory_map=True)
        time_table = pf.read(columns=["time_seconds"])
        time_wide = time_table.column("time_seconds")
        _assert_arrow_integer_range(
            time_wide,
            name="time_seconds",
            min_value=np.iinfo(np.int32).min,
            max_value=np.iinfo(np.int32).max,
        )
        time_seconds = time_wide.cast(pa.int32())
        del time_wide
        del time_table
        atom_table = pf.read(columns=["atom"])
        atom_wide = atom_table.column("atom")
        _assert_arrow_integer_range(
            atom_wide,
            name="atom",
            min_value=0,
            max_value=np.iinfo(np.uint16).max,
        )
        atom = atom_wide.cast(pa.uint16())
        del atom_wide
        del atom_table
        value_table = pf.read(columns=["value"])
        value = value_table.column("value").cast(pa.float32())
        del value_table
        # role column is optional. Older events.parquet files (built before
        # the role enum landed in scripts/aou_etl.py) don't have it. When
        # absent we synthesize a sentinel ROLE_UNKNOWN array and the
        # downstream static/event split falls back to the delta_days
        # heuristic. This preserves warm-start: no ETL re-pull needed when
        # only the role column was added.
        schema_field_names = set(pf.schema_arrow.names)
        if "role" in schema_field_names:
            role_table = pf.read(columns=["role"])
            role_wide = role_table.column("role")
            _assert_arrow_integer_range(
                role_wide,
                name="role",
                min_value=0,
                max_value=np.iinfo(np.uint8).max,
            )
            role = role_wide.cast(pa.uint8())
            del role_wide
            del role_table
        else:
            n_rows_pre = int(atom.length())
            role_np = np.full(n_rows_pre, ROLE_UNKNOWN, dtype=np.uint8)
            role = pa.chunked_array([pa.array(role_np, type=pa.uint8())])
            del role_np
        n_rows = int(atom.length())
        logger.finish_unit("read events parquet (shared)", f"rows={n_rows:,}")
        logger.start_unit(
            "register event columns (shared)",
            "compact dtypes (i32/u16/f32) — per-subject slices land in numpy on demand",
        )
        store = cls(time_seconds=time_seconds, atom=atom, value=value, role=role)
        logger.finish_unit("register event columns (shared)", f"chunks={store.num_chunks}")
        return store


class CohortDataset(Dataset):
    """events.parquet sorted by (subject_id, time_seconds, atom); subjects.parquet holds per-subject row offsets, sex, birth.

    ``split`` filters subjects.parquet by the ETL-assigned split column (e.g. "train", "test").
    The shared events.parquet is unchanged — splits just expose different subject row-ranges.

    Pass ``events`` (an :class:`EventStore` already loaded by the caller) to skip
    re-reading the parquet — essential when constructing multiple datasets in
    the same process (train + eval).
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_events: int = 4096,
        split: str | None = None,
        events: EventStore | None = None,
        window_policy: WindowPolicy = "last",
        last_window_fraction: float = 0.0,
        mixed_tail_weight: float = 0.5,
        max_windows_per_subject: int = 1,
        same_time_shuffle: bool = False,
        event_drop_prob: float = 0.0,
        temporal_ood: bool | None = None,
    ):
        if window_policy not in ("last", "random", "mixed"):
            raise ValueError("window_policy must be 'last', 'random', or 'mixed'")
        if not 0.0 <= last_window_fraction <= 1.0:
            raise ValueError("last_window_fraction must be between 0 and 1")
        if not 0.0 <= mixed_tail_weight <= 1.0:
            raise ValueError("mixed_tail_weight must be between 0 and 1")
        if max_windows_per_subject < 1:
            raise ValueError("max_windows_per_subject must be >= 1")
        if not 0.0 <= event_drop_prob < 1.0:
            raise ValueError("event_drop_prob must be in [0, 1)")
        data_dir = Path(data_dir)
        logger = ProgressLogger(f"cohort_dataset:{split or 'all'}", total_units=4)
        if events is None:
            events = EventStore.from_parquet(data_dir / "events.parquet")
        self.event_times = events.time_seconds
        self.event_atoms = events.atom
        self.event_values = events.value
        self.event_roles = events.role

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
                # Legacy subjects.parquet from the pre-validation-split era only
                # has 'train'/'test' labels. Resuming a previous run against an
                # older ETL output would crash here when asked for split=
                # 'validation'. Fall back to re-deriving splits from subject_id
                # using the current split_for_subject() rule. The deterministic
                # hash means subjects keep their old train/test buckets — the
                # validation bucket is carved out of the old train pool only.
                from scripts.aou_etl import split_for_subject as _split_for_subject  # noqa: PLC0415
                full_subjects = pl.read_parquet(data_dir / "subjects.parquet").sort("subject_id")
                derived = (
                    full_subjects
                    .with_columns(
                        pl.col("subject_id").map_elements(_split_for_subject, return_dtype=pl.Utf8).alias("_derived_split")
                    )
                    .filter(pl.col("_derived_split") == split)
                    .drop("_derived_split")
                )
                if derived.height == 0:
                    raise ValueError(f"no subjects in split={split!r}")
                subjects = derived
                logger.log(
                    "fallback: derived split from subject_id (legacy subjects.parquet)",
                    f"derived_subjects={subjects.height:,}",
                )
        if temporal_ood is not None:
            if "temporal_ood" not in subjects.columns:
                # Legacy subjects.parquet without the OOD flag: cannot honor
                # a temporal_ood filter. Surface the gap explicitly rather
                # than silently returning the wrong cohort.
                raise ValueError(
                    "subjects.parquet has no 'temporal_ood' column; rerun aou_etl to materialize it"
                )
            subjects = subjects.filter(pl.col("temporal_ood") == bool(temporal_ood))
            if subjects.height == 0:
                raise ValueError(f"no subjects with temporal_ood={temporal_ood} in split={split!r}")
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
        self.window_policy = window_policy
        self.last_window_fraction = float(last_window_fraction)
        self.mixed_tail_weight = float(mixed_tail_weight)
        self.same_time_shuffle = bool(same_time_shuffle)
        self.event_drop_prob = float(event_drop_prob)
        self.event_token_budget: float | None = None
        physical_rows = np.maximum(self.end - self.start + 1, 1)
        physical_lengths = np.minimum(physical_rows, max_events).astype(np.int64)
        if window_policy in ("random", "mixed") and max_windows_per_subject > 1:
            repeats = np.ceil(physical_rows / max_events).astype(np.int64)
            repeats = np.clip(repeats, 1, max_windows_per_subject)
        else:
            repeats = np.ones(len(self.start), dtype=np.int64)
        self.subject_indices = np.repeat(np.arange(len(self.start), dtype=np.int64), repeats)
        self.lengths = physical_lengths[self.subject_indices].astype(np.int64).tolist()
        mean_length = float(np.mean(self.lengths)) if self.lengths else 0.0
        logger.finish_unit(
            "compute per-subject sequence lengths",
            f"subjects={len(self.start):,} samples={len(self.lengths):,} mean_length={mean_length:.1f} "
            f"window_policy={window_policy} max_windows_per_subject={max_windows_per_subject}",
        )

    def __len__(self) -> int:
        return int(self.subject_indices.shape[0])

    def atom_counts(self, n_atoms: int) -> np.ndarray:
        """Atom counts over the materialized subject split.

        The events store is shared by train and eval datasets, so counting raw
        Arrow chunks would silently include held-out subjects. Count this split's
        clinical event tokens after the same static/event split used by
        ``__getitem__``. Random-window training can eventually expose any token
        in this pool, so this intentionally does not last-window truncate.
        """
        counts = np.zeros(n_atoms, dtype=np.float64)
        static_role_list = list(STATIC_ROLES)
        for start, end, birth in zip(self.start, self.end, self.birth_seconds, strict=True):
            length = int(end) + 1 - int(start)
            if length <= 0:
                continue
            atoms = self.event_atoms.slice(int(start), length).to_numpy()
            times = self.event_times.slice(int(start), length).to_numpy()
            roles = self.event_roles.slice(int(start), length).to_numpy()
            delta_days = (times - float(birth)) / 86400.0
            # Same role-vs-delta_days fallback as __getitem__: pre-role
            # events.parquet files use the legacy heuristic.
            if roles.size and bool((roles == ROLE_UNKNOWN).all()):
                non_static_mask = delta_days > 0.5
            else:
                non_static_mask = ~np.isin(roles, static_role_list)
            event_atoms = atoms[non_static_mask & (delta_days >= 0.0) & (atoms != PAD_ATOM)]
            if event_atoms.size:
                counts += np.bincount(event_atoms, minlength=n_atoms)[:n_atoms]
        counts[PAD_ATOM] = 0.0
        if counts.sum() <= 0:
            raise ValueError("events.parquet has no non-PAD atoms")
        self.event_token_budget = float(counts.sum())
        return counts.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict:
        idx = int(self.subject_indices[idx])
        s, e = int(self.start[idx]), int(self.end[idx])
        length = e + 1 - s
        birth = float(self.birth_seconds[idx])
        max_events = self.max_events

        # ChunkedArray.slice is O(1); .to_numpy() copies only the per-subject
        # window (≈1.7K rows), well below any memory concern.
        atoms = self.event_atoms.slice(s, length).to_numpy()
        times = self.event_times.slice(s, length).to_numpy()
        values = self.event_values.slice(s, length).to_numpy()
        roles = self.event_roles.slice(s, length).to_numpy()
        delta_days = (times - birth) / 86400.0
        real_atom = atoms != PAD_ATOM

        # When events.parquet predates the role column, roles are all
        # ROLE_UNKNOWN — fall back to the legacy delta_days <= 0.5 heuristic
        # so static (demographic) tokens still get routed to the static
        # prefix instead of the event stream.
        if roles.size and bool((roles == ROLE_UNKNOWN).all()):
            is_static = delta_days <= 0.5
        else:
            is_static = np.isin(roles, list(STATIC_ROLES))
        static_idx = np.where(is_static & real_atom)[0]
        event_idx_all = np.where((~is_static) & (delta_days >= 0.0) & real_atom)[0]
        event_idx = _select_window(
            event_idx_all,
            max_events=max_events,
            policy=self.window_policy,
            last_window_fraction=self.last_window_fraction,
            mixed_tail_weight=self.mixed_tail_weight,
        )
        if self.same_time_shuffle:
            event_idx = _shuffle_same_time(event_idx, times)
        event_idx = _drop_events(event_idx, drop_prob=self.event_drop_prob)

        static_atoms_arr = atoms[static_idx]
        event_atoms_arr = atoms[event_idx]
        event_ages_arr = delta_days[event_idx]
        event_values_arr = values[event_idx]
        event_times_arr = times[event_idx]
        # Dense per-subject group ids over the *windowed* event slice. Run
        # the derivation on the windowed times so groups are dense in
        # [0, n_groups) — the model-side bag loss assumes that.
        event_groups_arr = _derive_event_groups(event_times_arr)

        censor_age_days = (float(self.censor_seconds[idx]) - birth) / 86400.0
        return {
            "sex": int(self.sex[idx]),
            "static_atoms": [int(a) for a in static_atoms_arr.tolist()],
            "event_atoms": [int(a) for a in event_atoms_arr.tolist()],
            "event_ages": event_ages_arr.astype(np.float32, copy=False),
            "event_values": event_values_arr.astype(np.float32, copy=False),
            "event_groups": event_groups_arr.astype(np.int32, copy=False),
            "censor_age_days": float(censor_age_days),
            "length": int(event_atoms_arr.shape[0]),
        }


def _pad_size(size: int, pad_to_multiple_of: int | None) -> int:
    if pad_to_multiple_of is None:
        return size
    if pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of must be > 0")
    return ((size + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of


def _pad_atoms(
    seqs: list[list[int]],
    *,
    pad_to_multiple_of: int | None = None,
    min_size: int = 1,
) -> torch.Tensor:
    """Pad per-subject atom sequences. Always emits S >= min_size."""
    B = len(seqs)
    S = max((len(seq) for seq in seqs), default=0)
    S = _pad_size(max(S, min_size), pad_to_multiple_of)
    out = torch.full((B, S), PAD_ATOM, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def collate(batch: list[dict], *, pad_to_multiple_of: int | None = None) -> dict:
    B = len(batch)
    static_atoms = _pad_atoms([b["static_atoms"] for b in batch], pad_to_multiple_of=pad_to_multiple_of)
    # Always reserve at least one trailing pad slot on the event axis so the
    # censor transition (last real event -> first pad slot) is representable.
    # Without min_size=max(n_ev)+1, batch_size=1 packs the row exactly to n_ev
    # and the censor head sees zero tokens (n_censor == 0 every step).
    event_seqs = [b["event_atoms"] for b in batch]
    n_ev_max = max((len(seq) for seq in event_seqs), default=0)
    event_atoms = _pad_atoms(
        event_seqs,
        pad_to_multiple_of=pad_to_multiple_of,
        min_size=n_ev_max + 1,
    )
    M, T = static_atoms.shape[1], event_atoms.shape[1]

    static_pad = torch.ones(B, M, dtype=torch.bool)
    event_pad = torch.ones(B, T, dtype=torch.bool)
    event_ages = torch.zeros(B, T, dtype=torch.float32)
    event_values = torch.full((B, T), float("nan"), dtype=torch.float32)
    target_atoms = torch.zeros(B, T, dtype=torch.long)
    # ``-1`` marks padded / no-group positions; bag-loss code masks those
    # out. We deliberately don't reuse 0 because 0 is a valid group id.
    event_groups = torch.full((B, T), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        static_pad[i, : len(b["static_atoms"])] = False
        n_ev = len(b["event_atoms"])
        event_pad[i, :n_ev] = False
        if n_ev:
            groups = b.get("event_groups")
            if groups is not None and len(groups):
                event_groups[i, :n_ev] = torch.from_numpy(np.asarray(groups, dtype=np.int64))
            event_ages[i, :n_ev] = torch.from_numpy(b["event_ages"])
            event_values[i, :n_ev] = torch.from_numpy(b["event_values"])
            target_atoms[i, :n_ev] = event_atoms[i, :n_ev]
    static_pad[:, 0] = False

    out = {
        "static_atoms": static_atoms,
        "static_pad": static_pad,
        "event_atoms": event_atoms,
        "event_pad": event_pad,
        "event_ages": event_ages,
        "event_values": event_values,
        "event_groups": event_groups,
        "target_atoms": target_atoms,
        "censor_age": torch.tensor([b["censor_age_days"] for b in batch], dtype=torch.float32),
        "sex": torch.tensor([b["sex"] for b in batch], dtype=torch.long),
        "length": torch.tensor([b.get("length", len(b["event_atoms"])) for b in batch], dtype=torch.long),
    }
    if all("landmark_age_days" in b for b in batch):
        out["landmark_age"] = torch.tensor([b["landmark_age_days"] for b in batch], dtype=torch.float32)
    return out

"""Collator contract: padded atom tensors, target_atoms, NaN-safe static pad, S>=1 invariant."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from genterp.data import AtomVocab, CohortDataset, _pad_atoms, collate


def test_atom_vocab_encodes_known_codes_and_pad_for_missing():
    vocab = AtomVocab({"A": 5, "B": 5, "P": 9})

    assert vocab.encode("A") == 5
    assert vocab.encode("B") == 5
    assert vocab.encode("P") == 9
    assert vocab.encode("missing") == 0


def test_pad_atoms_always_emits_at_least_one_slot():
    atoms = _pad_atoms([[]])
    assert atoms.shape == (1, 1)
    assert atoms[0, 0].item() == 0  # PAD


def test_pad_atoms_can_align_to_multiple():
    atoms = _pad_atoms([[1, 2, 3]], pad_to_multiple_of=8)

    assert atoms.shape == (1, 8)
    assert atoms[0, :3].tolist() == [1, 2, 3]
    assert atoms[0, 3:].tolist() == [0, 0, 0, 0, 0]


def test_collate_shapes_and_targets():
    batch = [
        {
            "sex": 1,
            "static_atoms": [7, 13],
            "event_atoms": [21, 23, 24],
            "event_ages": np.array([100.0, 200.0, 365.25 * 20], dtype=np.float32),
            "event_values": np.array([1.5, np.nan, -2.0], dtype=np.float32),
            "censor_age_days": 30000.0,
        },
        {
            "sex": 0,
            "static_atoms": [31],
            "event_atoms": [41],
            "event_ages": np.array([50.0], dtype=np.float32),
            "event_values": np.array([0.25], dtype=np.float32),
            "censor_age_days": 25000.0,
        },
    ]
    out = collate(batch)

    # Event axis pads to max(n_ev)+1 so the censor transition (last real -> first pad)
    # is always representable. Longest subject has 3 events -> T=4.
    assert out["static_atoms"].shape == (2, 2)
    assert out["event_ages"].shape == (2, 4)
    assert out["static_atoms"].tolist() == [[7, 13], [31, 0]]
    assert out["event_atoms"].tolist() == [[21, 23, 24, 0], [41, 0, 0, 0]]
    assert out["target_atoms"].tolist() == [[21, 23, 24, 0], [41, 0, 0, 0]]
    assert torch.allclose(out["event_values"][0, :3], torch.tensor([1.5, torch.nan, -2.0]), equal_nan=True)
    assert torch.isnan(out["event_values"][0, 3]).item()
    assert out["event_values"][1, 0].item() == 0.25
    assert torch.isnan(out["event_values"][1, 1:]).all()
    assert out["event_pad"].tolist() == [[False, False, False, True], [False, True, True, True]]
    assert out["static_pad"][:, 0].tolist() == [False, False]
    assert out["sex"].tolist() == [1, 0]
    assert out["censor_age"].tolist() == [30000.0, 25000.0]


def test_collate_empty_static_is_nan_safe():
    batch = [
        {
            "sex": 0,
            "static_atoms": [],
            "event_atoms": [1],
            "event_ages": np.array([10.0], dtype=np.float32),
            "event_values": np.array([np.nan], dtype=np.float32),
            "censor_age_days": 100.0,
        }
    ]
    out = collate(batch)
    assert not out["static_pad"][0, 0].item(), "first static slot must be attendable to avoid all-masked softmax NaN"
    assert torch.isfinite(out["event_ages"]).all()


def test_cohort_dataset_keeps_most_recent_events_when_history_exceeds_max(tmp_path):
    """Older subjects (10k events, max_events=4096) must train on the most RECENT events,
    not the first 4096 (which would be childhood records for a 70-year-old)."""
    n_static = 3
    n_events = 100
    times = list(range(n_static)) + [86400 * (n_static + 1 + i) for i in range(n_events)]
    pl.DataFrame(
        {
            "subject_id": [1] * len(times),
            "time_seconds": times,
            "atom": [1] * n_static + [i + 2 for i in range(n_events)],
            "value": [None] * len(times),
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1],
            "start": [0],
            "end": [len(times) - 1],
            "sex": [0],
            "birth_seconds": [0],
            "censor_seconds": [86400 * 10_000],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    max_events = 10
    item = CohortDataset(tmp_path, max_events=max_events)[0]

    # Should be the *last* 10 events (E90..E99 → atoms 92..101), not the first 10.
    assert item["event_atoms"] == [i + 2 for i in range(n_events - max_events, n_events)]
    assert item["length"] == max_events


def test_cohort_dataset_reads_materialized_atoms(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time_seconds": [0, 86400, 172800],
            "atom": [5, 6, 0],
            "value": [None, 1.5, None],
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1],
            "start": [0],
            "end": [2],
            "sex": [1],
            "birth_seconds": [0],
            "censor_seconds": [86400 * 10],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    item = CohortDataset(tmp_path)[0]

    assert item["static_atoms"] == [5]
    assert item["event_atoms"] == [6]

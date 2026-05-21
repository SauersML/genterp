"""Collator contract: padded atom tensors, target_atoms, NaN-safe static pad, S>=1 invariant."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
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
            "landmark_age_days": 20000.0,
        },
        {
            "sex": 0,
            "static_atoms": [31],
            "event_atoms": [41],
            "event_ages": np.array([50.0], dtype=np.float32),
            "event_values": np.array([0.25], dtype=np.float32),
            "censor_age_days": 25000.0,
            "landmark_age_days": 15000.0,
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
    assert out["landmark_age"].tolist() == [20000.0, 15000.0]


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
    """Deterministic last-window mode keeps the most recent events."""
    n_static = 3
    n_events = 100
    times = list(range(n_static)) + [86400 * (n_static + 1 + i) for i in range(n_events)]
    pl.DataFrame(
        {
            "subject_id": [1] * len(times),
            "time_seconds": times,
            "atom": [1] * n_static + [i + 2 for i in range(n_events)],
            "value": [None] * len(times),
            "role": [10] * n_static + [0] * n_events,
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


def test_cohort_dataset_random_window_samples_long_history(tmp_path, monkeypatch):
    n_events = 20
    times = [0] + [86400 * (i + 1) for i in range(n_events)]
    pl.DataFrame(
        {
            "subject_id": [1] * len(times),
            "time_seconds": times,
            "atom": [1] + [i + 2 for i in range(n_events)],
            "value": [None] * len(times),
            "role": [10] + [0] * n_events,
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
    monkeypatch.setattr(np.random, "random", lambda: 1.0)
    monkeypatch.setattr(np.random, "randint", lambda low, high: 4)

    item = CohortDataset(tmp_path, max_events=5, window_policy="random")[0]

    assert item["event_atoms"] == [6, 7, 8, 9, 10]
    assert item["length"] == 5


def test_cohort_dataset_reads_materialized_atoms(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time_seconds": [0, 86400, 172800],
            "atom": [5, 6, 0],
            "value": [None, 1.5, None],
            "role": [10, 0, 0],
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


def test_cohort_dataset_uses_role_not_age_for_static_split(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time_seconds": [0, 86400, 86400 * 30],
            "atom": [5, 6, 7],
            "value": [None, None, None],
            "role": [0, 10, 0],
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1],
            "start": [0],
            "end": [2],
            "sex": [1],
            "birth_seconds": [0],
            "censor_seconds": [86400 * 100],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    item = CohortDataset(tmp_path)[0]

    assert item["static_atoms"] == [6]
    assert item["event_atoms"] == [5, 7]


def test_cohort_dataset_repeats_long_subjects_for_random_windows(tmp_path):
    n_events = 11
    pl.DataFrame(
        {
            "subject_id": [1] * n_events + [2, 2],
            "time_seconds": [86400 * i for i in range(n_events)] + [0, 86400],
            "atom": list(range(1, n_events + 1)) + [30, 31],
            "value": [None] * (n_events + 2),
            "role": [0] * (n_events + 2),
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "start": [0, n_events],
            "end": [n_events - 1, n_events + 1],
            "sex": [0, 1],
            "birth_seconds": [0, 0],
            "censor_seconds": [86400 * 100, 86400 * 100],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    ds = CohortDataset(tmp_path, max_events=5, window_policy="random", max_windows_per_subject=4)

    assert len(ds) == 4
    assert ds.subject_indices.tolist() == [0, 0, 0, 1]


def test_cohort_dataset_repeats_long_subjects_for_mixed_windows(tmp_path):
    n_events = 12
    pl.DataFrame(
        {
            "subject_id": [1] * n_events + [2, 2],
            "time_seconds": [86400 * i for i in range(n_events)] + [0, 86400],
            "atom": list(range(1, n_events + 1)) + [99, 100],
            "value": [None] * (n_events + 2),
            "role": [0] * (n_events + 2),
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "start": [0, n_events],
            "end": [n_events - 1, n_events + 1],
            "sex": [0, 1],
            "birth_seconds": [0, 0],
            "censor_seconds": [86400 * 100, 86400 * 100],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    ds = CohortDataset(tmp_path, max_events=5, window_policy="mixed", max_windows_per_subject=4)

    assert len(ds) == 4
    assert ds.subject_indices.tolist() == [0, 0, 0, 1]


def test_atom_counts_only_use_materialized_split(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "time_seconds": [0, 86400, 0, 86400],
            "atom": [5, 6, 7, 8],
            "value": [None, None, None, None],
            "role": [10, 0, 10, 0],
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "start": [0, 2],
            "end": [1, 3],
            "sex": [0, 1],
            "birth_seconds": [0, 0],
            "censor_seconds": [86400 * 10, 86400 * 10],
            "split": ["train", "test"],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    train = CohortDataset(tmp_path, split="train")
    counts = train.atom_counts(16)

    assert counts[5] == 0
    assert counts[6] == 1
    assert counts[7] == 0
    assert counts[8] == 0


def test_mixed_window_policy_alternates_between_tail_and_random_anchor(tmp_path, monkeypatch):
    """Mixed policy: tail vs random-anchor decided per draw; long windows preserved."""
    n_events = 20
    times = [0] + [86400 * (i + 1) for i in range(n_events)]
    pl.DataFrame(
        {
            "subject_id": [1] * len(times),
            "time_seconds": times,
            "atom": [1] + [i + 2 for i in range(n_events)],
            "value": [None] * len(times),
            "role": [10] + [0] * n_events,
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

    ds = CohortDataset(tmp_path, max_events=5, window_policy="mixed", mixed_tail_weight=0.5)

    # tail branch: random()<0.5
    monkeypatch.setattr(np.random, "random", lambda: 0.1)
    item_tail = ds[0]
    assert item_tail["event_atoms"][-1] == n_events + 1  # ends on the last event

    # random-anchor branch: random()>=0.5, randint picks anchor=4
    monkeypatch.setattr(np.random, "random", lambda: 0.9)
    monkeypatch.setattr(np.random, "randint", lambda low, high: 4)
    item_random = ds[0]
    assert item_random["event_atoms"] == [6, 7, 8, 9, 10]
    # Both windows are full length — long windows are preserved either way.
    assert item_tail["length"] == 5 and item_random["length"] == 5


def test_mixed_window_policy_rejects_out_of_range_weight(tmp_path):
    pl.DataFrame({"subject_id": [1], "time_seconds": [0], "atom": [1], "value": [None], "role": [10]}).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame({"subject_id": [1], "start": [0], "end": [0], "sex": [0], "birth_seconds": [0], "censor_seconds": [86400]}).write_parquet(tmp_path / "subjects.parquet")
    with pytest.raises(ValueError):
        CohortDataset(tmp_path, max_events=5, window_policy="mixed", mixed_tail_weight=1.5)

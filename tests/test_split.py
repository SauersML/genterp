from __future__ import annotations

import polars as pl
import pytest

from genterp.data import CohortDataset
from scripts.aou_etl import TEST_SPLIT_PERCENT, VALIDATION_SPLIT_PERCENT, split_for_subject


def test_split_assignment_is_deterministic():
    for sid in [0, 1, 7, 42, 10_000_000, 2**31 - 1]:
        assert split_for_subject(sid) == split_for_subject(sid)


def test_split_only_emits_train_validation_or_test():
    for sid in range(200):
        assert split_for_subject(sid) in {"train", "validation", "test"}


def test_split_ratio_is_roughly_70_10_20_on_large_sample():
    n = 50_000
    test_count = sum(split_for_subject(sid) == "test" for sid in range(n))
    validation_count = sum(split_for_subject(sid) == "validation" for sid in range(n))
    test_frac = test_count / n
    validation_frac = validation_count / n
    target = TEST_SPLIT_PERCENT / 100
    validation_target = VALIDATION_SPLIT_PERCENT / 100
    assert abs(test_frac - target) < 0.01, f"observed test fraction {test_frac:.4f} drifted from {target}"
    assert abs(validation_frac - validation_target) < 0.01, (
        f"observed validation fraction {validation_frac:.4f} drifted from {validation_target}"
    )


def _write_split_fixture(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time_seconds": [0, 86400, 0, 86400, 0, 86400, 0, 86400],
            "atom": [5, 6] * 4,
            "value": [None, 1.0, None, 2.0, None, 3.0, None, 4.0],
            "role": [10, 0] * 4,
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "start": [0, 2, 4, 6],
            "end": [1, 3, 5, 7],
            "sex": [0, 1, 0, 1],
            "birth_seconds": [0, 0, 0, 0],
            "censor_seconds": [86400 * 10] * 4,
            "split": ["train", "validation", "test", "train"],
        }
    ).write_parquet(tmp_path / "subjects.parquet")


def test_cohort_dataset_filters_to_requested_split(tmp_path):
    _write_split_fixture(tmp_path)

    train = CohortDataset(tmp_path, split="train")
    validation = CohortDataset(tmp_path, split="validation")
    test = CohortDataset(tmp_path, split="test")

    assert len(train) == 2
    assert len(validation) == 1
    assert len(test) == 1
    assert train.split == "train"
    assert validation.split == "validation"
    assert test.split == "test"


def test_cohort_dataset_split_partition_is_disjoint_and_exhaustive(tmp_path):
    _write_split_fixture(tmp_path)

    train = CohortDataset(tmp_path, split="train")
    validation = CohortDataset(tmp_path, split="validation")
    test = CohortDataset(tmp_path, split="test")
    full = CohortDataset(tmp_path)

    train_starts = set(train.start.tolist())
    validation_starts = set(validation.start.tolist())
    test_starts = set(test.start.tolist())
    full_starts = set(full.start.tolist())

    assert train_starts.isdisjoint(test_starts)
    assert train_starts.isdisjoint(validation_starts)
    assert validation_starts.isdisjoint(test_starts)
    assert train_starts | validation_starts | test_starts == full_starts


def test_cohort_dataset_raises_when_subjects_lack_split_column(tmp_path):
    pl.DataFrame(
        {"subject_id": [1], "time_seconds": [0], "atom": [5], "value": [None], "role": [10]}
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1],
            "start": [0],
            "end": [0],
            "sex": [0],
            "birth_seconds": [0],
            "censor_seconds": [86400],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    with pytest.raises(ValueError, match="no 'split' column"):
        CohortDataset(tmp_path, split="train")


def test_cohort_dataset_validation_falls_back_to_derived_split_for_legacy_subjects(tmp_path):
    """Pre-validation-era subjects.parquet (only train/test) still resumes for split='validation'."""
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "time_seconds": [0, 0, 0, 0, 0],
            "atom": [1, 1, 1, 1, 1],
            "value": [None, None, None, None, None],
            "role": [10, 10, 10, 10, 10],
        }
    ).write_parquet(tmp_path / "events.parquet")
    # Find a subject_id that the new split rule maps to 'validation'.
    from scripts.aou_etl import split_for_subject
    val_ids = [sid for sid in range(1, 1000) if split_for_subject(sid) == "validation"][:3]
    train_ids = [sid for sid in range(1, 1000) if split_for_subject(sid) == "train"][:3]
    legacy_ids = val_ids + train_ids
    pl.DataFrame(
        {
            "subject_id": legacy_ids,
            "start": list(range(len(legacy_ids))),
            "end": list(range(len(legacy_ids))),
            "sex": [0] * len(legacy_ids),
            "birth_seconds": [0] * len(legacy_ids),
            "censor_seconds": [86400] * len(legacy_ids),
            # Legacy: only 'train' / 'test' labels exist.
            "split": ["train"] * len(legacy_ids),
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    # Fresh subjects.parquet must have validation events.parquet → the fallback
    # carves a validation bucket out of subject_ids deterministically.
    ds = CohortDataset(tmp_path, split="validation")
    assert len(ds) == len(val_ids)


def test_cohort_dataset_filters_temporal_ood(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time_seconds": [0, 0, 0],
            "atom": [1, 1, 1],
            "value": [None, None, None],
            "role": [10, 10, 10],
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "start": [0, 1, 2],
            "end": [0, 1, 2],
            "sex": [0, 1, 0],
            "birth_seconds": [0, 0, 0],
            "censor_seconds": [86400, 86400, 86400],
            "split": ["train", "train", "train"],
            "temporal_ood": [True, False, True],
        }
    ).write_parquet(tmp_path / "subjects.parquet")

    ood = CohortDataset(tmp_path, split="train", temporal_ood=True)
    in_dist = CohortDataset(tmp_path, split="train", temporal_ood=False)
    assert len(ood) == 2
    assert len(in_dist) == 1


def test_cohort_dataset_raises_when_temporal_ood_requested_but_missing(tmp_path):
    pl.DataFrame(
        {
            "subject_id": [1],
            "time_seconds": [0],
            "atom": [1],
            "value": [None],
            "role": [10],
        }
    ).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {
            "subject_id": [1],
            "start": [0],
            "end": [0],
            "sex": [0],
            "birth_seconds": [0],
            "censor_seconds": [86400],
            "split": ["train"],
        }
    ).write_parquet(tmp_path / "subjects.parquet")
    with pytest.raises(ValueError, match="temporal_ood"):
        CohortDataset(tmp_path, split="train", temporal_ood=True)

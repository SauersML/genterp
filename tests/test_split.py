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

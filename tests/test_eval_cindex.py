"""Tests for the eval_cindex pieces that don't need an AoU cohort.

Covers:
- build_cohort_condition_phenotypes filter and top-N selection
- legacy concept_codes.json shape detected and produces empty sweep
- _build_outcome_table vectorized invariants on a synthetic cohort
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from genterp.eval_cindex import (
    DEFAULT_SWEEP_TOP_N,
    DiseasePhenotype,
    SubjectIndex,
    _build_outcome_table,
    build_cohort_condition_phenotypes,
)


def _write_etl_cache(
    etl_dir: Path,
    concept_codes: list[list[object]],
    coverage: list[list[int]],
    ancestors: list[list[object]],
) -> None:
    """Build the minimum ETL cache layout that eval_cindex looks for."""
    cache_dir = etl_dir / "cache" / "fake-cdr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "concept_codes.json").write_text(json.dumps(concept_codes))
    payload = {"coverage": coverage, "ancestors": ancestors}
    (cache_dir / "coverage_and_ancestors-deadbeef.json").write_text(json.dumps(payload))


def test_build_cohort_condition_phenotypes_filters_and_ranks(tmp_path: Path):
    """domain==Condition filter keeps diseases, drops drugs/labs; top-N keeps
    the highest-coverage entries; concept_name lands in DiseasePhenotype.name.
    """
    concept_codes: list[list[object]] = [
        # cid, code, domain, class, name
        [1001, "SNOMED/A", "Condition", "Clinical Finding", "Disease A"],
        [1002, "SNOMED/B", "Condition", "Clinical Finding", "Disease B"],
        [1003, "SNOMED/C", "Condition", "Clinical Finding", "Disease C"],
        [2001, "RxNorm/X", "Drug", "Ingredient", "Drug X"],
        [3001, "LOINC/L", "Measurement", "Lab", "Lab L"],
    ]
    coverage = [
        [1001, 5_000],   # most common condition
        [1002, 1_500],
        [1003, 200],
        [2001, 20_000],  # most common atom overall but not a condition
        [3001, 10_000],
    ]
    # Empty ancestor closure — descendants_of will just have self-references
    # which is fine for this test (the function only needs concept_meta +
    # coverage to pick candidates).
    _write_etl_cache(tmp_path, concept_codes, coverage, ancestors=[])

    phenotypes = build_cohort_condition_phenotypes(tmp_path, top_n=10)

    names = [p.name for p in phenotypes]
    codes = [p.root_code for p in phenotypes]
    assert names == ["Disease A", "Disease B", "Disease C"], names
    assert codes == ["SNOMED/A", "SNOMED/B", "SNOMED/C"], codes
    assert all(isinstance(p, DiseasePhenotype) for p in phenotypes)


def test_build_cohort_condition_phenotypes_applies_top_n_cap(tmp_path: Path):
    concept_codes: list[list[object]] = [
        [cid, f"SNOMED/{cid}", "Condition", "Clinical Finding", f"Disease {cid}"]
        for cid in range(1, 101)
    ]
    coverage = [[cid, 100 - cid] for cid in range(1, 101)]
    _write_etl_cache(tmp_path, concept_codes, coverage, ancestors=[])

    phenotypes = build_cohort_condition_phenotypes(tmp_path, top_n=5)
    assert len(phenotypes) == 5
    # Ordered by coverage descending — cid=1 has the highest coverage (99).
    assert [p.root_code for p in phenotypes] == [f"SNOMED/{i}" for i in range(1, 6)]


def test_build_cohort_condition_phenotypes_handles_legacy_cache(tmp_path: Path, capsys):
    """Old 2-tuple [[cid, code], ...] cache produces no sweep candidates and
    surfaces a clear re-run-ETL warning instead of crashing or silently
    returning hierarchy-inferred candidates.
    """
    concept_codes_legacy: list[list[object]] = [
        [1001, "SNOMED/A"],
        [1002, "SNOMED/B"],
    ]
    coverage = [[1001, 5_000], [1002, 1_500]]
    _write_etl_cache(tmp_path, concept_codes_legacy, coverage, ancestors=[])

    phenotypes = build_cohort_condition_phenotypes(tmp_path, top_n=10)
    assert phenotypes == []
    captured = capsys.readouterr().out
    assert "re-run aou_etl" in captured.lower()


def test_default_sweep_top_n_is_reasonable():
    """Guard against accidental knob drift — the user explicitly asked for 50."""
    assert DEFAULT_SWEEP_TOP_N == 50


class _SyntheticEventStore:
    """Minimal EventStore-like object that supports .atom.slice(s, n).to_numpy()
    and .time_seconds.slice(s, n).to_numpy() — the only methods _build_outcome_table
    calls. Pure-Python so the test doesn't depend on pyarrow.ChunkedArray's
    construction quirks."""

    class _Col:
        def __init__(self, data: list[int]):
            self._data = np.asarray(data, dtype=np.int64)

        def slice(self, offset: int, length: int) -> "_SyntheticEventStore._Sub":
            return _SyntheticEventStore._Sub(self._data[offset : offset + length])

    class _Sub:
        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def to_numpy(self) -> np.ndarray:
            return self._arr

    def __init__(self, atoms: list[int], time_seconds: list[int]):
        self.atom = self._Col(atoms)
        self.time_seconds = self._Col(time_seconds)


def test_build_outcome_table_basic_phenotype_resolution():
    """Vectorized outcome table — same logic as before, just sanity-checked
    on a tiny hand-built cohort so we can be sure no off-by-one snuck in.

    Setup: 2 subjects, 2 diseases.
      - Subject 0: age 30y has a disease-A atom (post-landmark @ landmark=40y? no, pre).
        Then disease-A atom at age 50y and 50.2y — both post-landmark.
        Prior_case[0, A]=False, observed[0, A]=True (two hits ≥ 30 days apart).
      - Subject 1: age 35y has disease-B atom (PRE landmark).
        Then disease-A atom at age 45y — single hit, doesn't qualify min_occurrences=2.
        Prior_case[1, B]=True, observed[1, A]=False (only 1 post-landmark hit).
    """
    SECONDS_PER_DAY = 86400
    YEAR_SECONDS = int(365.25 * SECONDS_PER_DAY)

    # birth_seconds=0 → ages_days = time_seconds / 86400
    subjects = [
        SubjectIndex(
            subject_id=0, start=0, end=2,
            birth_seconds=0.0,
            censor_seconds=float(60 * YEAR_SECONDS),
            sex=0,
            last_event_idx_local=2,
            last_event_age_days=30 * 365.25,
            first_event_age_days=30 * 365.25,
            landmark_age_days=40 * 365.25,
            gap_to_landmark_days=10 * 365.25,
            censor_age_days=60 * 365.25,
        ),
        SubjectIndex(
            subject_id=1, start=3, end=4,
            birth_seconds=0.0,
            censor_seconds=float(60 * YEAR_SECONDS),
            sex=1,
            last_event_idx_local=1,
            last_event_age_days=35 * 365.25,
            first_event_age_days=35 * 365.25,
            landmark_age_days=40 * 365.25,
            gap_to_landmark_days=5 * 365.25,
            censor_age_days=60 * 365.25,
        ),
    ]
    # Atom layout (subject 0 rows 0–2, subject 1 rows 3–4):
    #  row 0: atom 10 (disease-A) at age 30y  — pre-landmark
    #  row 1: atom 10 (disease-A) at age 50.0y — post, hit #1
    #  row 2: atom 10 (disease-A) at age 50.2y — post, hit #2 (≥30d apart)
    #  row 3: atom 20 (disease-B) at age 35y   — pre-landmark
    #  row 4: atom 10 (disease-A) at age 45y   — single post hit
    atoms = [10, 10, 10, 20, 10]
    times_sec = [
        int(30 * YEAR_SECONDS),
        int(50.0 * YEAR_SECONDS),
        int(50.2 * YEAR_SECONDS),
        int(35 * YEAR_SECONDS),
        int(45 * YEAR_SECONDS),
    ]
    events = _SyntheticEventStore(atoms, times_sec)

    phenotypes = [
        DiseasePhenotype(name="A", root_code="SNOMED/A", min_occurrences=2, min_gap_days=30.0),
        DiseasePhenotype(name="B", root_code="SNOMED/B", min_occurrences=2, min_gap_days=30.0),
    ]
    atom_sets = [{10}, {20}]

    t2e, observed, prior_case, sex_eligible = _build_outcome_table(
        events,  # type: ignore[arg-type]
        subjects, phenotypes, atom_sets, n_atoms_total=64,
    )

    # Subject 0: disease A — pre-event existed (age 30y < landmark 40y) → prior_case
    assert prior_case[0, 0] is np.True_ or bool(prior_case[0, 0]) is True
    assert not bool(observed[0, 0])

    # Subject 0: disease B — never seen
    assert not bool(prior_case[0, 1])
    assert not bool(observed[0, 1])

    # Subject 1: disease A — only one post-landmark hit, qualifies fail
    assert not bool(prior_case[1, 0])
    assert not bool(observed[1, 0])

    # Subject 1: disease B — prior at 35y
    assert bool(prior_case[1, 1])
    assert not bool(observed[1, 1])

    # No sex-restricted phenotypes here → eligible everywhere
    assert sex_eligible.all()
    # time_to_event for non-cases defaults to (horizon_age - landmark) where
    # horizon_age = min(censor, landmark + HORIZON_DAYS). With HORIZON_DAYS = 10y
    # and landmark=40y, censor=60y, horizon_age = 50y → 10y window.
    expected_window_days = 10 * 365.25
    np.testing.assert_allclose(t2e[0, 1], expected_window_days)
    np.testing.assert_allclose(t2e[1, 0], expected_window_days)

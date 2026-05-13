"""Hierarchical collapse: leaf survives, leaf collapses, DAG tie-breaking, drop on orphan."""

from __future__ import annotations

import pytest

from genterp.vocab import collapse_vocabulary


def test_high_count_leaf_keeps_specificity():
    own = {"A": 100}
    coverage = {"A": 100, "B": 100, "C": 100}
    ancestors = {"A": {"B": 1, "C": 2}}
    atom_idx = collapse_vocabulary(own, coverage, ancestors, threshold=50)
    assert set(atom_idx) == {"A"}


def test_rare_leaf_collapses_to_first_eligible_ancestor():
    own = {"A": 10}
    coverage = {"A": 10, "B": 10, "C": 100}
    ancestors = {"A": {"B": 1, "C": 2}}
    atom_idx = collapse_vocabulary(own, coverage, ancestors, threshold=50)
    assert atom_idx == {"A": 1}


def test_dag_picks_most_specific_parent_by_coverage():
    # A has two parents P1 (coverage 100) and P2 (coverage 1000). Pick P1 — more specific.
    own = {"A": 10, "P1": 100}  # include P1 so we can verify atom mapping
    coverage = {"A": 10, "P1": 100, "P2": 1000, "R": 1500}
    ancestors = {"A": {"P1": 1, "P2": 1, "R": 2}, "P1": {"R": 1}}
    atom_idx = collapse_vocabulary(own, coverage, ancestors, threshold=50)
    # A's atom == P1's atom (A collapsed to P1)
    assert atom_idx["A"] == atom_idx["P1"]
    # P2 didn't survive (nothing maps to it)
    assert "P2" not in atom_idx


def test_dag_ties_break_by_hops_then_code():
    # Both parents coverage 100. P1 is 1 hop, P2 is 2 hops. Pick P1.
    own = {"A": 10}
    coverage = {"A": 10, "P1": 100, "P2": 100}
    ancestors = {"A": {"P1": 1, "P2": 2}}
    atom_idx = collapse_vocabulary(own, coverage, ancestors, threshold=50)
    assert atom_idx == {"A": 1}
    # Both at same hops, same coverage → lexicographic
    ancestors2 = {"A": {"P1": 1, "P2": 1}}
    atom_idx2 = collapse_vocabulary(own, coverage, ancestors2, threshold=50)
    assert atom_idx2 == {"A": 1}


def test_orphan_dropped():
    own = {"A": 10}
    coverage = {"A": 10, "B": 20}
    ancestors = {"A": {"B": 1}}
    atom_idx = collapse_vocabulary(own, coverage, ancestors, threshold=50)
    assert "A" not in atom_idx


def test_common_leaf_and_rare_sibling_map_to_their_own_targets_only():
    # A survives. B is rare and collapses to P. Event embedding stays at the mapped atom.
    own = {"A": 100, "B": 10}
    coverage = {"A": 100, "B": 10, "P": 110, "R": 200}
    ancestors = {"A": {"P": 1, "R": 2}, "B": {"P": 1, "R": 2}}
    atom_idx = collapse_vocabulary(own, coverage, ancestors, threshold=50)
    # A targets A, B targets P
    assert atom_idx["A"] != atom_idx["B"]


def test_threshold_validation():
    with pytest.raises(ValueError):
        collapse_vocabulary({}, {}, {}, threshold=0)


def test_dense_atoms():
    own = {"A": 100, "B": 100, "C": 100}
    coverage = {"A": 100, "B": 100, "C": 100}
    atom_idx = collapse_vocabulary(own, coverage, {}, threshold=50)
    assert sorted(atom_idx.values()) == [1, 2, 3]

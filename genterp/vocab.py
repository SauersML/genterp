"""Hierarchical SNOMED collapse: rare cohort concepts roll up to their most-specific high-coverage ancestor."""

from __future__ import annotations


def collapse_vocabulary(
    own_counts: dict[str, int],
    coverage: dict[str, int],
    ancestors: dict[str, dict[str, int]],
    threshold: int = 50,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Roll rare cohort concepts up the IS-A DAG.

    coverage[c] is the number of distinct patients with at least one event whose
    concept is c or any descendant of c — a fixed quantity, computed once from
    OMOP `concept_ancestor`, independent of which descendants end up in vocab.
    Using coverage (not bottom-up rolling sums) avoids the multi-parent
    double-counting that breaks greedy aggregation in a DAG.

    For each cohort concept c:
      - if own_counts[c] >= threshold, c is its own target
      - else, among c's ancestors a with coverage[a] >= threshold, pick the one
        minimizing (coverage[a], hops(c→a), a). Most-specific surviving
        ancestor, deterministic tie-break.

    Returns (atom_idx, ancestor_codes):
      atom_idx[code]       = dense atom in [1..V] of code's target
      ancestor_codes[code] = strict vocab ancestors of code's target
    Codes with no eligible ancestor are absent (dropped).
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")

    targets: dict[str, str] = {}
    for c, own in own_counts.items():
        if own >= threshold:
            targets[c] = c
            continue
        eligible = [
            (a, h) for a, h in ancestors.get(c, {}).items()
            if coverage.get(a, 0) >= threshold
        ]
        if not eligible:
            continue
        best, _ = min(eligible, key=lambda ah: (coverage[ah[0]], ah[1], ah[0]))
        targets[c] = best

    vocab = sorted(set(targets.values()))
    atom_of = {v: i + 1 for i, v in enumerate(vocab)}
    vocab_set = set(vocab)

    atom_idx = {c: atom_of[t] for c, t in targets.items()}
    ancestor_codes = {
        c: sorted(a for a in ancestors.get(t, {}) if a in vocab_set)
        for c, t in targets.items()
    }
    return atom_idx, ancestor_codes

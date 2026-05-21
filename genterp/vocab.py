"""Hierarchical SNOMED collapse with stable append-only atom identity."""

from __future__ import annotations

from collections.abc import Iterable

PAD_ATOM_ID = 0
FIRST_ATOM_ID = 1


def collapse_targets(
    own_counts: dict[str, int],
    coverage: dict[str, int],
    ancestors: dict[str, dict[str, int]],
    threshold: int = 500,
) -> dict[str, str]:
    """Return ``source_code -> canonical_atom_code`` after hierarchical collapse."""
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
    return targets


def validate_atom_registry(registry: dict[str, int]) -> None:
    """Reject registries that cannot safely define atom-table row identity."""
    seen: set[int] = set()
    for code, atom_id in registry.items():
        if not isinstance(code, str) or not code:
            raise ValueError("atom registry codes must be non-empty strings")
        if not isinstance(atom_id, int) or atom_id < FIRST_ATOM_ID:
            raise ValueError(f"atom registry id for {code!r} must be >= {FIRST_ATOM_ID}")
        if atom_id in seen:
            raise ValueError(f"duplicate atom registry id: {atom_id}")
        seen.add(atom_id)


def build_atom_registry(
    canonical_atom_codes: Iterable[str],
    existing_registry: dict[str, int] | None = None,
) -> dict[str, int]:
    """Append new canonical atom codes without moving existing ids.

    ``PAD_ATOM_ID`` is reserved for padding and is never represented in the
    registry. Historical atoms can remain in the registry even if a later CDR
    refresh no longer emits them; keeping their rows is what makes checkpoint
    identity stable.
    """
    registry = dict(existing_registry or {})
    validate_atom_registry(registry)
    next_id = max(registry.values(), default=PAD_ATOM_ID) + 1
    for code in sorted(set(canonical_atom_codes)):
        if code in registry:
            continue
        registry[code] = next_id
        next_id += 1
    return registry


def materialize_atom_indices(targets: dict[str, str], registry: dict[str, int]) -> dict[str, int]:
    """Convert collapsed targets into ``source_code -> stable_atom_id``."""
    validate_atom_registry(registry)
    missing = sorted(set(targets.values()) - set(registry))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f", +{len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(f"atom registry is missing canonical codes: {preview}{suffix}")
    return {code: registry[target] for code, target in targets.items()}


def collapse_vocabulary(
    own_counts: dict[str, int],
    coverage: dict[str, int],
    ancestors: dict[str, dict[str, int]],
    threshold: int = 500,
    atom_registry: dict[str, int] | None = None,
) -> dict[str, int]:
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

    Returns:
      atom_idx[code] = dense atom in [1..V] of code's target
    Codes with no eligible ancestor are absent (dropped).
    """
    targets = collapse_targets(own_counts, coverage, ancestors, threshold=threshold)
    registry = build_atom_registry(targets.values(), atom_registry)
    return materialize_atom_indices(targets, registry)

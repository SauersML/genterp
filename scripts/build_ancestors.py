"""Materialize ``etl/ancestors.npz`` for hierarchical atom embeddings.

The model's :class:`genterp.modeling.AtomEmbedding` can sum ancestor-node
vectors into each atom's input embedding. To do that it needs a per-atom
list of ancestor-node ids. This script produces that table from the
existing ETL cache (no BigQuery hit required) so the training run can
pick it up via the optional ``etl/ancestors.npz`` path.

What it does
------------
1. Read ``etl/vocab.json``  (code → atom_id).
2. Read the latest ``etl/cache/<key>/concept_codes.json``  (cid → "VOCAB/CODE").
3. Read the latest ``etl/cache/<key>/coverage_and_ancestors-*.json``
   (descendant_cid → [(ancestor_cid, hops), …]).
4. For each atom_id ``a``, collect every source code that collapses to ``a``,
   resolve them to cids, union their proper ancestors (hops > 0), and write
   the resulting ancestor-cid set as a row in ``ancestor_ids``. Cids that
   appear as ancestors get fresh contiguous node ids starting at 1 (id 0 is
   reserved as the pad row of ``ancestor_embedding``).

The union-over-source-codes rule is deliberate: the hierarchical-embedding
inductive bias only cares that an atom share parameters with semantically-
related nodes; using every source code's ancestor closure gives the model
more parameter sharing without losing information.

Output schema (numpy ``.npz``):
  ancestor_ids    : (n_atoms, max_anc) int64  — right-padded with 0
  n_ancestor_rows : ()            int64       — distinct non-pad nodes
  node_to_cid     : (n_ancestor_rows,) int64  — mapping for debuggability

Idempotent: if the output exists and was produced from the current vocab.json
+ cache snapshot, exits without rewriting.

CLI:
  python -m scripts.build_ancestors            # reads ~/genterp/etl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _find_etl_cache(etl_dir: Path) -> Path:
    cache_root = etl_dir / "cache"
    candidates = [d for d in cache_root.iterdir() if d.is_dir() and (d / "concept_codes.json").is_file()]
    if not candidates:
        raise SystemExit(f"no ETL cache under {cache_root} — run aou_etl.py first")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _latest_coverage_and_ancestors(cache_dir: Path) -> Path:
    files = sorted(
        cache_dir.glob("coverage_and_ancestors-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise SystemExit(f"no coverage_and_ancestors-*.json under {cache_dir}")
    return files[0]


def _fingerprint(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(str(int(p.stat().st_mtime_ns)).encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def build_ancestors(etl_dir: Path, output: Path) -> dict[str, object]:
    """Return summary stats; writes ``output`` atomically."""
    vocab_path = etl_dir / "vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"missing {vocab_path}")
    cache_dir = _find_etl_cache(etl_dir)
    ca_path = _latest_coverage_and_ancestors(cache_dir)
    cc_path = cache_dir / "concept_codes.json"
    print(f"  vocab        : {vocab_path}")
    print(f"  cache_dir    : {cache_dir}")
    print(f"  ancestors    : {ca_path}")
    print(f"  concept_codes: {cc_path}")

    fp = _fingerprint(vocab_path, ca_path, cc_path)
    if output.is_file():
        try:
            existing = np.load(output, allow_pickle=False)
            if "source_fingerprint" in existing.files:
                stored = bytes(existing["source_fingerprint"]).decode().strip("\x00")
                if stored == fp:
                    print(f"  output up to date (fingerprint={fp}); skipping rebuild")
                    return {
                        "n_atoms": int(existing["ancestor_ids"].shape[0]),
                        "max_anc": int(existing["ancestor_ids"].shape[1]),
                        "n_ancestor_rows": int(existing["n_ancestor_rows"].item()),
                        "skipped": True,
                    }
        except (OSError, ValueError, KeyError):
            pass

    code_to_atom: dict[str, int] = dict(json.loads(vocab_path.read_text()))
    n_atoms = max(code_to_atom.values()) + 1
    print(f"  loaded vocab : codes={len(code_to_atom):,} atoms={n_atoms:,}")

    cc_pairs = json.loads(cc_path.read_text())
    # Backward-compatible: support both old [[cid, code]] and new
    # [[cid, code, domain, class, name]] schemas. Only the (cid, code) pair
    # is needed here.
    cid_to_code: dict[int, str] = {int(entry[0]): str(entry[1]) for entry in cc_pairs}
    code_to_cid: dict[str, int] = {code: cid for cid, code in cid_to_code.items()}
    print(f"  loaded codes : {len(cid_to_code):,} concept ids")

    cov_anc = json.loads(ca_path.read_text())
    proper_ancestors_of: dict[int, set[int]] = defaultdict(set)
    for entry in cov_anc.get("ancestors", []):
        desc_cid = int(entry[0])
        for anc_pair in entry[1]:
            anc_cid, hops = int(anc_pair[0]), int(anc_pair[1])
            if hops <= 0:
                # Skip self-ancestor (hops=0). Self is the leaf and is already
                # represented by embed.embedding; ancestor_embedding only
                # carries the *additive* hierarchy beyond the leaf.
                continue
            proper_ancestors_of[desc_cid].add(anc_cid)
    print(f"  loaded ancestors: {len(proper_ancestors_of):,} concepts with ≥1 proper ancestor")

    # For each atom, gather source codes that map to it; union their ancestors.
    codes_per_atom: dict[int, list[str]] = defaultdict(list)
    for code, atom in code_to_atom.items():
        codes_per_atom[int(atom)].append(code)

    ancestor_cids_per_atom: list[set[int]] = [set() for _ in range(n_atoms)]
    for atom_id, codes in codes_per_atom.items():
        ancs: set[int] = set()
        for code in codes:
            cid = code_to_cid.get(code)
            if cid is None:
                # Synthetic survey-answer codes ("VOCAB/CODE=str:...") have no
                # cid because they're not in OMOP concept; they have no
                # ancestors. Fine — they fall through with empty ancs.
                continue
            ancs |= proper_ancestors_of.get(cid, set())
        ancestor_cids_per_atom[atom_id] = ancs

    # Assign node ids: id 0 = pad, then deterministic sorted-cid order so
    # repeated builds produce the same mapping (warm-start friendly).
    all_anc_cids = sorted({c for s in ancestor_cids_per_atom for c in s})
    cid_to_node = {c: i + 1 for i, c in enumerate(all_anc_cids)}
    n_ancestor_rows = len(all_anc_cids)
    print(f"  distinct ancestor nodes: {n_ancestor_rows:,}")

    max_anc = max((len(s) for s in ancestor_cids_per_atom), default=0)
    print(f"  max ancestors per atom : {max_anc}")
    if max_anc == 0:
        # Degenerate but valid: emit an empty hierarchy and let the model run
        # in flat mode.
        ancestor_ids = np.zeros((n_atoms, 0), dtype=np.int64)
    else:
        ancestor_ids = np.zeros((n_atoms, max_anc), dtype=np.int64)
        for atom_id, ancs in enumerate(ancestor_cids_per_atom):
            for i, cid in enumerate(sorted(ancs)):
                ancestor_ids[atom_id, i] = cid_to_node[cid]

    node_to_cid = np.asarray(all_anc_cids, dtype=np.int64)
    tmp = output.with_suffix(output.suffix + ".tmp")
    np.savez_compressed(
        tmp,
        ancestor_ids=ancestor_ids,
        n_ancestor_rows=np.int64(n_ancestor_rows),
        node_to_cid=node_to_cid,
        source_fingerprint=np.frombuffer(fp.encode().ljust(16, b"\x00"), dtype=np.uint8),
    )
    tmp.replace(output)
    print(f"  wrote {output}")

    nonzero_atoms = int(sum(1 for s in ancestor_cids_per_atom if s))
    return {
        "n_atoms": n_atoms,
        "atoms_with_ancestors": nonzero_atoms,
        "max_anc": max_anc,
        "n_ancestor_rows": n_ancestor_rows,
        "skipped": False,
    }


def main(argv: list[str] | None = None) -> None:
    # parse_known_args (not parse_args) so run.sh can forward "$@" uniformly
    # to all entrypoints — flags meant for aou_etl / train (e.g. --tiny) just
    # pass through without crashing here.
    parser = argparse.ArgumentParser(description="Build ETL ancestors.npz for hierarchical embeddings.")
    parser.parse_known_args(argv)
    etl_dir = Path.home() / "genterp" / "etl"
    output = etl_dir / "ancestors.npz"
    summary = build_ancestors(etl_dir, output)
    print(
        "  summary: "
        + " ".join(f"{k}={v}" for k, v in summary.items())
    )


if __name__ == "__main__":
    main()

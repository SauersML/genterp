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
    # Track min-hops per (descendant, ancestor) so we can cap per-atom at the
    # CLOSEST ancestors (most specific = most informative). The raw transitive
    # closure has up to 4676 ancestors for the deepest atom, and AtomEmbedding.
    # effective_weight materializes a (n_atoms, max_anc, dim) tensor every
    # forward — at max_anc=4676 that's 400 GB. Cap aggressively.
    proper_ancestors_of: dict[int, dict[int, int]] = defaultdict(dict)
    for entry in cov_anc.get("ancestors", []):
        desc_cid = int(entry[0])
        for anc_pair in entry[1]:
            anc_cid, hops = int(anc_pair[0]), int(anc_pair[1])
            if hops <= 0:
                # Skip self-ancestor (hops=0). Self is the leaf and is already
                # represented by embed.embedding; ancestor_embedding only
                # carries the *additive* hierarchy beyond the leaf.
                continue
            existing = proper_ancestors_of[desc_cid].get(anc_cid)
            if existing is None or hops < existing:
                proper_ancestors_of[desc_cid][anc_cid] = hops
    print(f"  loaded ancestors: {len(proper_ancestors_of):,} concepts with ≥1 proper ancestor")

    # For each atom, union ancestors from all source codes that collapse to it,
    # keeping the min hops across source codes (most specific seen anywhere).
    codes_per_atom: dict[int, list[str]] = defaultdict(list)
    for code, atom in code_to_atom.items():
        codes_per_atom[int(atom)].append(code)

    MAX_ANC_PER_ATOM = 16  # SNOMED IS-A chains are typically 5–12 deep; 16 is comfortable
    capped_anc_per_atom: list[list[int]] = [[] for _ in range(n_atoms)]
    raw_max_anc = 0
    for atom_id, codes in codes_per_atom.items():
        ancs: dict[int, int] = {}
        for code in codes:
            cid = code_to_cid.get(code)
            if cid is None:
                # Synthetic survey-answer codes ("VOCAB/CODE=str:...") have no
                # cid because they're not in OMOP concept; they have no
                # ancestors. Fine — they fall through with empty ancs.
                continue
            for anc_cid, hops in proper_ancestors_of.get(cid, {}).items():
                existing = ancs.get(anc_cid)
                if existing is None or hops < existing:
                    ancs[anc_cid] = hops
        raw_max_anc = max(raw_max_anc, len(ancs))
        # Sort by hops ASC (closest first), then by cid for deterministic ordering
        # across rebuilds. Keep top-K.
        sorted_ancs = sorted(ancs.items(), key=lambda kv: (kv[1], kv[0]))[:MAX_ANC_PER_ATOM]
        capped_anc_per_atom[atom_id] = [cid for cid, _ in sorted_ancs]
    print(f"  raw max ancestors per atom: {raw_max_anc:,} (uncapped)")
    print(f"  capped to closest {MAX_ANC_PER_ATOM} per atom")

    # Only keep node ids for ancestors that actually appear after capping.
    # This dramatically reduces ancestor_embedding parameter count vs the
    # raw transitive closure (most deep ancestors fall out of the top-16).
    used_anc_cids = sorted({cid for ancs in capped_anc_per_atom for cid in ancs})
    cid_to_node = {c: i + 1 for i, c in enumerate(used_anc_cids)}
    n_ancestor_rows = len(used_anc_cids)
    print(f"  distinct ancestor nodes (used after cap): {n_ancestor_rows:,}")

    max_anc = max((len(s) for s in capped_anc_per_atom), default=0)
    print(f"  max ancestors per atom (after cap) : {max_anc}")
    if max_anc == 0:
        # Degenerate but valid: emit an empty hierarchy and let the model run
        # in flat mode.
        ancestor_ids = np.zeros((n_atoms, 0), dtype=np.int64)
    else:
        ancestor_ids = np.zeros((n_atoms, max_anc), dtype=np.int64)
        for atom_id, ancs in enumerate(capped_anc_per_atom):
            for i, cid in enumerate(ancs):
                ancestor_ids[atom_id, i] = cid_to_node[cid]
    all_anc_cids = used_anc_cids

    node_to_cid = np.asarray(all_anc_cids, dtype=np.int64)
    tmp = output.with_suffix(output.suffix + ".tmp")
    # np.savez_compressed auto-appends ".npz" to STRING/Path filenames,
    # so passing "foo.npz.tmp" silently writes "foo.npz.tmp.npz" and the
    # atomic-rename target doesn't exist. Pass a file object to suppress
    # that behavior — numpy writes to exactly the path we opened.
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            ancestor_ids=ancestor_ids,
            n_ancestor_rows=np.int64(n_ancestor_rows),
            node_to_cid=node_to_cid,
            source_fingerprint=np.frombuffer(fp.encode().ljust(16, b"\x00"), dtype=np.uint8),
        )
    tmp.replace(output)
    print(f"  wrote {output}")

    nonzero_atoms = int(sum(1 for ancs in capped_anc_per_atom if ancs))
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

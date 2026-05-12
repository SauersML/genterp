"""Build genterp's atom vocab + ancestor map from the AoU OMOP CDR in BigQuery.

Follows the gnomon biobank pattern: SNOMED concepts are pulled from `concept`,
IS-A closure from `concept_ancestor`. Runs inside the AoU Researcher Workbench
where `WORKSPACE_CDR` points at the CDR dataset.

Outputs two JSON files consumed by `genterp.train`:
  - vocab.json:     {"<vocabulary>/<code>": atom_idx, ...}
  - ancestors.json: {"<vocabulary>/<code>": ["<ancestor-code>", ...], ...}

The vocabulary captures every standard SNOMED concept actually used in the
cohort (so the model never spends embedding capacity on dead codes); ancestors
include the full transitive IS-A closure, used input-side by AncestorEmbedding.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from google.cloud import bigquery


def _client() -> bigquery.Client:
    return bigquery.Client()


def _cohort_concept_ids(client: bigquery.Client, cdr: str) -> set[int]:
    sql = f"""
    SELECT DISTINCT condition_concept_id AS cid FROM `{cdr}.condition_occurrence` WHERE condition_concept_id > 0
    UNION DISTINCT SELECT DISTINCT drug_concept_id      FROM `{cdr}.drug_exposure`         WHERE drug_concept_id      > 0
    UNION DISTINCT SELECT DISTINCT procedure_concept_id FROM `{cdr}.procedure_occurrence`  WHERE procedure_concept_id > 0
    UNION DISTINCT SELECT DISTINCT measurement_concept_id FROM `{cdr}.measurement`         WHERE measurement_concept_id > 0
    UNION DISTINCT SELECT DISTINCT observation_concept_id FROM `{cdr}.observation`         WHERE observation_concept_id > 0
    """
    return {int(r["cid"]) for r in client.query(sql).result()}


def _concept_codes(client: bigquery.Client, cdr: str, concept_ids: set[int]) -> dict[int, str]:
    """concept_id -> 'VOCAB/CODE' for the MEDS-style string code."""
    if not concept_ids:
        return {}
    sql = f"""
    SELECT concept_id, vocabulary_id, concept_code
    FROM `{cdr}.concept`
    WHERE concept_id IN UNNEST(@cids)
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("cids", "INT64", list(concept_ids)),
        ]),
    )
    return {int(r["concept_id"]): f"{r['vocabulary_id']}/{r['concept_code']}" for r in job.result()}


def _ancestor_pairs(client: bigquery.Client, cdr: str, descendant_ids: set[int]) -> dict[int, list[int]]:
    """descendant_concept_id -> [ancestor_concept_id, ...] over `concept_ancestor`."""
    sql = f"""
    SELECT descendant_concept_id AS d, ancestor_concept_id AS a
    FROM `{cdr}.concept_ancestor`
    WHERE descendant_concept_id IN UNNEST(@d)
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("d", "INT64", list(descendant_ids)),
        ]),
    )
    out: dict[int, list[int]] = defaultdict(list)
    for r in job.result():
        out[int(r["d"])].append(int(r["a"]))
    return dict(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cdr", default=os.environ.get("WORKSPACE_CDR"))
    args = p.parse_args()
    if not args.cdr:
        raise SystemExit("WORKSPACE_CDR not set and --cdr not provided")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = _client()

    cohort_ids = _cohort_concept_ids(client, args.cdr)
    code_of = _concept_codes(client, args.cdr, cohort_ids)
    descendant_ancestors = _ancestor_pairs(client, args.cdr, cohort_ids)
    extra_ancestor_ids = {a for v in descendant_ancestors.values() for a in v} - cohort_ids
    code_of.update(_concept_codes(client, args.cdr, extra_ancestor_ids))

    vocab: dict[str, int] = {}
    for cid in sorted(code_of):
        code = code_of[cid]
        if code not in vocab:
            vocab[code] = len(vocab) + 1

    ancestors: dict[str, list[str]] = {}
    for d_cid, a_cids in descendant_ancestors.items():
        d_code = code_of.get(d_cid)
        if d_code is None:
            continue
        ancestors[d_code] = sorted({code_of[a] for a in a_cids if a in code_of and a != d_cid})

    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out / "ancestors.json").write_text(json.dumps(ancestors, indent=2))
    print(f"wrote {len(vocab):,} atoms, {len(ancestors):,} ancestor entries to {out}")


if __name__ == "__main__":
    main()

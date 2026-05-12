"""Build genterp's atom vocab + ancestor map from the AoU OMOP CDR."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genterp.vocab import collapse_vocabulary  # noqa: E402

from google.cloud import bigquery  # noqa: E402


OUT_DIR = Path.home() / "genterp" / "etl"
THRESHOLD = 500

EVENT_TABLES = [
    ("condition_occurrence", "condition_concept_id"),
    ("drug_exposure", "drug_concept_id"),
    ("procedure_occurrence", "procedure_concept_id"),
    ("measurement", "measurement_concept_id"),
    ("observation", "observation_concept_id"),
]


def _events_cte(cdr: str) -> str:
    parts = [
        f"SELECT person_id, {col} AS cid FROM `{cdr}.{tbl}` WHERE {col} > 0"
        for tbl, col in EVENT_TABLES
    ]
    return "events AS (\n  " + "\n  UNION ALL ".join(parts) + "\n)"


def own_counts(client: bigquery.Client, cdr: str) -> dict[int, int]:
    sql = f"""
    WITH {_events_cte(cdr)}
    SELECT cid, COUNT(DISTINCT person_id) AS n FROM events GROUP BY cid
    """
    return {int(r["cid"]): int(r["n"]) for r in client.query(sql).result()}


def coverage(client: bigquery.Client, cdr: str) -> dict[int, int]:
    sql = f"""
    WITH {_events_cte(cdr)}
    SELECT ca.ancestor_concept_id AS aid, COUNT(DISTINCT events.person_id) AS n
    FROM events
    JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = events.cid
    GROUP BY ca.ancestor_concept_id
    """
    return {int(r["aid"]): int(r["n"]) for r in client.query(sql).result()}


def ancestor_closure(client: bigquery.Client, cdr: str, cohort: list[int]) -> dict[int, dict[int, int]]:
    sql = f"""
    WITH relevant AS (
      SELECT DISTINCT cid FROM UNNEST(@cohort) AS cid
      UNION DISTINCT
      SELECT DISTINCT ancestor_concept_id FROM `{cdr}.concept_ancestor`
      WHERE descendant_concept_id IN UNNEST(@cohort)
    )
    SELECT descendant_concept_id AS d, ancestor_concept_id AS a, min_levels_of_separation AS hops
    FROM `{cdr}.concept_ancestor`
    WHERE descendant_concept_id IN (SELECT cid FROM relevant)
      AND min_levels_of_separation > 0
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("cohort", "INT64", cohort),
        ]),
    )
    out: dict[int, dict[int, int]] = {}
    for r in job.result():
        out.setdefault(int(r["d"]), {})[int(r["a"])] = int(r["hops"])
    return out


def concept_codes(client: bigquery.Client, cdr: str, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    sql = f"SELECT concept_id, vocabulary_id, concept_code FROM `{cdr}.concept` WHERE concept_id IN UNNEST(@ids)"
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "INT64", list(ids)),
        ]),
    )
    return {int(r["concept_id"]): f"{r['vocabulary_id']}/{r['concept_code']}" for r in job.result()}


def main() -> None:
    cdr = os.environ.get("WORKSPACE_CDR")
    if not cdr:
        raise SystemExit("WORKSPACE_CDR not set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client()

    own = own_counts(client, cdr)
    cov = coverage(client, cdr)
    anc = ancestor_closure(client, cdr, list(own))

    all_ids = set(own) | set(cov) | {a for d in anc.values() for a in d}
    code_of = concept_codes(client, cdr, all_ids)

    own_by_code = {code_of[c]: n for c, n in own.items() if c in code_of}
    cov_by_code = {code_of[c]: n for c, n in cov.items() if c in code_of}
    anc_by_code = {
        code_of[d]: {code_of[a]: h for a, h in ancs.items() if a in code_of}
        for d, ancs in anc.items() if d in code_of
    }

    atom_idx, ancestor_codes = collapse_vocabulary(
        own_by_code, cov_by_code, anc_by_code, threshold=THRESHOLD,
    )

    (OUT_DIR / "vocab.json").write_text(json.dumps(atom_idx, indent=2, sort_keys=True))
    (OUT_DIR / "ancestors.json").write_text(json.dumps(ancestor_codes, indent=2, sort_keys=True))

    n_vocab = len(set(atom_idx.values()))
    n_codes = len(atom_idx)
    print(f"threshold={THRESHOLD}  vocab={n_vocab:,}  routed_codes={n_codes:,}  dropped={len(own_by_code) - n_codes:,}  -> {OUT_DIR}")


if __name__ == "__main__":
    main()

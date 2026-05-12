"""Build genterp's atom vocab + ancestor map from the AoU OMOP CDR."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genterp.vocab import collapse_vocabulary  # noqa: E402

from google.cloud import bigquery  # noqa: E402


EVENT_TABLES = [
    ("condition_occurrence", "condition_concept_id"),
    ("drug_exposure", "drug_concept_id"),
    ("procedure_occurrence", "procedure_concept_id"),
    ("measurement", "measurement_concept_id"),
    ("observation", "observation_concept_id"),
]


def _events_cte(cdr: str) -> str:
    parts = [f"SELECT person_id, {col} AS cid FROM `{cdr}.{tbl}` WHERE {col} > 0" for tbl, col in EVENT_TABLES]
    return "events AS (\n  " + "\n  UNION ALL ".join(parts) + "\n)"


def main() -> None:
    cdr = os.environ["WORKSPACE_CDR"]
    out_dir = Path.home() / "genterp" / "etl"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client()

    own = {
        int(r["cid"]): int(r["n"])
        for r in client.query(f"WITH {_events_cte(cdr)} SELECT cid, COUNT(DISTINCT person_id) AS n FROM events GROUP BY cid").result()
    }
    cov = {
        int(r["aid"]): int(r["n"])
        for r in client.query(f"""
        WITH {_events_cte(cdr)}
        SELECT ca.ancestor_concept_id AS aid, COUNT(DISTINCT events.person_id) AS n
        FROM events JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = events.cid
        GROUP BY ca.ancestor_concept_id
        """).result()
    }
    anc: dict[int, dict[int, int]] = {}
    job = client.query(
        f"""
        WITH relevant AS (
          SELECT DISTINCT cid FROM UNNEST(@cohort) AS cid
          UNION DISTINCT
          SELECT DISTINCT ancestor_concept_id FROM `{cdr}.concept_ancestor` WHERE descendant_concept_id IN UNNEST(@cohort)
        )
        SELECT descendant_concept_id AS d, ancestor_concept_id AS a, min_levels_of_separation AS hops
        FROM `{cdr}.concept_ancestor`
        WHERE descendant_concept_id IN (SELECT cid FROM relevant) AND min_levels_of_separation > 0
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("cohort", "INT64", list(own)),
        ]),
    )
    for r in job.result():
        anc.setdefault(int(r["d"]), {})[int(r["a"])] = int(r["hops"])

    all_ids = set(own) | set(cov) | {a for d in anc.values() for a in d}
    code_of = {
        int(r["concept_id"]): f"{r['vocabulary_id']}/{r['concept_code']}"
        for r in client.query(
            f"SELECT concept_id, vocabulary_id, concept_code FROM `{cdr}.concept` WHERE concept_id IN UNNEST(@ids)",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ArrayQueryParameter("ids", "INT64", list(all_ids)),
            ]),
        ).result()
    }

    own_by_code = {code_of[c]: n for c, n in own.items() if c in code_of}
    cov_by_code = {code_of[c]: n for c, n in cov.items() if c in code_of}
    anc_by_code = {
        code_of[d]: {code_of[a]: h for a, h in ancs.items() if a in code_of}
        for d, ancs in anc.items() if d in code_of
    }

    atom_idx, ancestor_codes = collapse_vocabulary(own_by_code, cov_by_code, anc_by_code, threshold=500)

    (out_dir / "vocab.json").write_text(json.dumps(atom_idx, indent=2, sort_keys=True))
    (out_dir / "ancestors.json").write_text(json.dumps(ancestor_codes, indent=2, sort_keys=True))

    n_vocab = len(set(atom_idx.values()))
    print(f"vocab={n_vocab:,}  routed={len(atom_idx):,}  dropped={len(own_by_code) - len(atom_idx):,}  -> {out_dir}")


if __name__ == "__main__":
    main()

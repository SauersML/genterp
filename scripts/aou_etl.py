"""Build genterp's vocab, ancestors, event timelines, and subject metadata from AoU OMOP."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genterp.vocab import collapse_vocabulary  # noqa: E402

import polars as pl  # noqa: E402
from google.cloud import bigquery  # noqa: E402


EVENT_TABLES = [
    ("condition_occurrence", "condition_concept_id", "condition_start_datetime"),
    ("drug_exposure", "drug_concept_id", "drug_exposure_start_datetime"),
    ("procedure_occurrence", "procedure_concept_id", "procedure_datetime"),
    ("measurement", "measurement_concept_id", "measurement_datetime"),
    ("observation", "observation_concept_id", "observation_datetime"),
]


def _events_cte(cdr: str, with_time: bool) -> str:
    if with_time:
        parts = [
            f"SELECT person_id, {col} AS cid, {tcol} AS t FROM `{cdr}.{tbl}` WHERE {col} > 0 AND {tcol} IS NOT NULL"
            for tbl, col, tcol in EVENT_TABLES
        ]
    else:
        parts = [
            f"SELECT person_id, {col} AS cid FROM `{cdr}.{tbl}` WHERE {col} > 0"
            for tbl, col, _ in EVENT_TABLES
        ]
    return "events AS (\n  " + "\n  UNION ALL ".join(parts) + "\n)"


def main() -> None:
    cdr = os.environ["WORKSPACE_CDR"]
    out_dir = Path.home() / "genterp" / "etl"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client()

    own = {
        int(r["cid"]): int(r["n"])
        for r in client.query(
            f"WITH {_events_cte(cdr, False)} SELECT cid, COUNT(DISTINCT person_id) AS n FROM events GROUP BY cid"
        ).result()
    }
    cov = {
        int(r["aid"]): int(r["n"])
        for r in client.query(
            f"""WITH {_events_cte(cdr, False)}
            SELECT ca.ancestor_concept_id AS aid, COUNT(DISTINCT events.person_id) AS n
            FROM events JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = events.cid
            GROUP BY ca.ancestor_concept_id"""
        ).result()
    }
    anc: dict[int, dict[int, int]] = {}
    for r in client.query(
        f"""WITH relevant AS (
          SELECT DISTINCT cid FROM UNNEST(@cohort) AS cid
          UNION DISTINCT SELECT DISTINCT ancestor_concept_id FROM `{cdr}.concept_ancestor` WHERE descendant_concept_id IN UNNEST(@cohort)
        )
        SELECT descendant_concept_id AS d, ancestor_concept_id AS a, min_levels_of_separation AS hops
        FROM `{cdr}.concept_ancestor`
        WHERE descendant_concept_id IN (SELECT cid FROM relevant) AND min_levels_of_separation > 0""",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("cohort", "INT64", list(own)),
        ]),
    ).result():
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

    events = pl.from_arrow(
        client.query(
            f"""WITH {_events_cte(cdr, True)}
            SELECT
              CAST(events.person_id AS INT64) AS subject_id,
              UNIX_SECONDS(events.t) AS time_seconds,
              CONCAT(c.vocabulary_id, '/', c.concept_code) AS code
            FROM events JOIN `{cdr}.concept` c ON c.concept_id = events.cid"""
        ).to_arrow()
    )
    events = events.filter(pl.col("code").is_in(set(atom_idx.keys()))).sort(["subject_id", "time_seconds"])
    events.write_parquet(out_dir / "events.parquet")

    persons = pl.from_arrow(
        client.query(
            f"""SELECT
              CAST(person_id AS INT64) AS subject_id,
              IF(gender_concept_id = 8507, 1, 0) AS sex,
              UNIX_SECONDS(COALESCE(
                birth_datetime,
                TIMESTAMP(DATE(year_of_birth, COALESCE(month_of_birth, 1), COALESCE(day_of_birth, 1)))
              )) AS birth_seconds
            FROM `{cdr}.person`"""
        ).to_arrow()
    )

    offsets = events.with_row_index("row").group_by("subject_id", maintain_order=True).agg(
        pl.col("row").min().alias("start"),
        pl.col("row").max().alias("end"),
    )
    subjects = offsets.join(persons, on="subject_id", how="inner").sort("subject_id")
    subjects.write_parquet(out_dir / "subjects.parquet")

    print(
        f"vocab={len(set(atom_idx.values())):,}  "
        f"events={events.height:,}  "
        f"subjects={subjects.height:,}  "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()

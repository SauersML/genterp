"""Build genterp's vocab, ancestors, event timelines, and subject metadata from AoU OMOP.

Stages:
  1. Standard event coverage via concept_ancestor (Conditions/Procedures/Observations).
  2. Drug events expanded to RxNorm ingredient atoms via drug_strength
     (skipping noisy intermediate RxNorm levels — clinicians think at ingredient level).
  3. Measurement values quantile-binned per concept (Q=10) into VOCAB/CODE@Q{n} tokens,
     ancestors pointing back to the bare concept.
  4. Hierarchical collapse at threshold=500 patients.
  5. observation_period_end_date drives per-subject right-censoring.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genterp.vocab import collapse_vocabulary  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from google.cloud import bigquery  # noqa: E402


THRESHOLD = 500
N_QUANTILES = 10

NON_DRUG_TABLES = [
    ("condition_occurrence", "condition_concept_id", "condition_start_datetime"),
    ("procedure_occurrence", "procedure_concept_id", "procedure_datetime"),
    ("observation", "observation_concept_id", "observation_datetime"),
]


def _non_drug_events_cte(cdr: str, with_time: bool) -> str:
    sel = "person_id, {c} AS cid, {t} AS t" if with_time else "person_id, {c} AS cid"
    where = "{c} > 0 AND {t} IS NOT NULL" if with_time else "{c} > 0"
    parts = [
        f"SELECT {sel.format(c=col, t=tcol)} FROM `{cdr}.{tbl}` WHERE {where.format(c=col, t=tcol)}"
        for tbl, col, tcol in NON_DRUG_TABLES
    ]
    return "\n  UNION ALL ".join(parts)


def _drug_events_arrow(client: bigquery.Client, cdr: str):
    """Drug exposures, exploded to ingredient codes via drug_strength."""
    sql = f"""
    SELECT
      CAST(de.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(de.drug_exposure_start_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      ds.ingredient_concept_id AS cid
    FROM `{cdr}.drug_exposure` de
    JOIN `{cdr}.drug_strength` ds ON ds.drug_concept_id = de.drug_concept_id
    JOIN `{cdr}.concept` c ON c.concept_id = ds.ingredient_concept_id
    WHERE de.drug_concept_id > 0 AND de.drug_exposure_start_datetime IS NOT NULL
    """
    return client.query(sql).to_arrow()


def _non_drug_events_arrow(client: bigquery.Client, cdr: str):
    sql = f"""
    WITH events AS (
      {_non_drug_events_cte(cdr, with_time=True)}
    )
    SELECT
      CAST(events.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(events.t) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      events.cid AS cid
    FROM events JOIN `{cdr}.concept` c ON c.concept_id = events.cid
    """
    return client.query(sql).to_arrow()


def _measurement_events_arrow(client: bigquery.Client, cdr: str):
    sql = f"""
    SELECT
      CAST(m.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(m.measurement_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      m.measurement_concept_id AS cid,
      m.value_as_number AS value
    FROM `{cdr}.measurement` m JOIN `{cdr}.concept` c ON c.concept_id = m.measurement_concept_id
    WHERE m.measurement_concept_id > 0 AND m.measurement_datetime IS NOT NULL
    """
    return client.query(sql).to_arrow()


def _coverage_and_ancestors(client: bigquery.Client, cdr: str, cohort_ids: list[int]):
    cov = {
        int(r["aid"]): int(r["n"])
        for r in client.query(
            f"""
            WITH events AS (
              {_non_drug_events_cte(cdr, with_time=False)}
              UNION ALL SELECT person_id, drug_concept_id AS cid FROM `{cdr}.drug_exposure` WHERE drug_concept_id > 0
              UNION ALL SELECT person_id, measurement_concept_id AS cid FROM `{cdr}.measurement` WHERE measurement_concept_id > 0
            )
            SELECT ca.ancestor_concept_id AS aid, COUNT(DISTINCT events.person_id) AS n
            FROM events JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = events.cid
            GROUP BY ca.ancestor_concept_id
            """
        ).result()
    }
    anc: dict[int, dict[int, int]] = {}
    for r in client.query(
        f"""
        WITH relevant AS (
          SELECT DISTINCT cid FROM UNNEST(@cohort) AS cid
          UNION DISTINCT SELECT DISTINCT ancestor_concept_id FROM `{cdr}.concept_ancestor` WHERE descendant_concept_id IN UNNEST(@cohort)
        )
        SELECT descendant_concept_id AS d, ancestor_concept_id AS a, min_levels_of_separation AS hops
        FROM `{cdr}.concept_ancestor`
        WHERE descendant_concept_id IN (SELECT cid FROM relevant) AND min_levels_of_separation > 0
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("cohort", "INT64", cohort_ids),
        ]),
    ).result():
        anc.setdefault(int(r["d"]), {})[int(r["a"])] = int(r["hops"])
    return cov, anc


def _concept_codes(client: bigquery.Client, cdr: str, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    rows = client.query(
        f"SELECT concept_id, vocabulary_id, concept_code FROM `{cdr}.concept` WHERE concept_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "INT64", list(ids)),
        ]),
    ).result()
    return {int(r["concept_id"]): f"{r['vocabulary_id']}/{r['concept_code']}" for r in rows}


def _bucket_measurements(meas: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[float]]]:
    """Replace measurement codes with VOCAB/CODE@Q{n}. Returns (transformed events, per-code quantile edges)."""
    with_value = meas.filter(pl.col("value").is_not_null())
    edges_by_code: dict[str, list[float]] = {}
    parts = []
    for code, group in with_value.group_by("code"):
        code_str = code[0] if isinstance(code, tuple) else code
        vals = group["value"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) < THRESHOLD:
            continue
        edges = np.quantile(vals, np.linspace(0, 1, N_QUANTILES + 1)).tolist()
        edges_by_code[code_str] = edges
        inner = np.asarray(edges[1:-1])
        bins = np.searchsorted(inner, group["value"].to_numpy(), side="right")
        parts.append(group.with_columns(
            (pl.lit(code_str) + "@Q" + pl.Series(bins, dtype=pl.Int32).cast(pl.Utf8)).alias("code")
        ))
    bucketed = pl.concat(parts) if parts else with_value.head(0)
    no_value = meas.filter(pl.col("value").is_null())
    out = pl.concat([
        bucketed.select(["subject_id", "time_seconds", "code"]),
        no_value.select(["subject_id", "time_seconds", "code"]),
    ])
    return out, edges_by_code


def main() -> None:
    cdr = os.environ["WORKSPACE_CDR"]
    out_dir = Path.home() / "genterp" / "etl"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client()

    non_drug = pl.from_arrow(_non_drug_events_arrow(client, cdr))
    drug = pl.from_arrow(_drug_events_arrow(client, cdr))
    meas_raw = pl.from_arrow(_measurement_events_arrow(client, cdr))
    meas, q_edges = _bucket_measurements(meas_raw)

    events_all = pl.concat([
        non_drug.select(["subject_id", "time_seconds", "code", "cid"]),
        drug.select(["subject_id", "time_seconds", "code", "cid"]),
        meas_raw.select(["subject_id", "time_seconds", "code", "cid"]),
    ])
    own_by_cid = {
        int(r["cid"]): int(r["n"])
        for r in events_all.group_by("cid").agg(pl.col("subject_id").n_unique().alias("n")).iter_rows(named=True)
    }
    cov, anc = _coverage_and_ancestors(client, cdr, list(own_by_cid))

    all_ids = set(own_by_cid) | set(cov) | {a for d in anc.values() for a in d}
    code_of = _concept_codes(client, cdr, all_ids)
    own_by_code = {code_of[c]: n for c, n in own_by_cid.items() if c in code_of}
    cov_by_code = {code_of[c]: n for c, n in cov.items() if c in code_of}
    anc_by_code = {
        code_of[d]: {code_of[a]: h for a, h in ancs.items() if a in code_of}
        for d, ancs in anc.items() if d in code_of
    }

    quantile_own = (
        meas.filter(pl.col("code").str.contains("@Q"))
        .group_by("code").agg(pl.col("subject_id").n_unique().alias("n"))
    )
    for r in quantile_own.iter_rows(named=True):
        qtoken = r["code"]
        bare = qtoken.split("@Q")[0]
        own_by_code[qtoken] = int(r["n"])
        cov_by_code[qtoken] = int(r["n"])
        if bare in anc_by_code:
            anc_by_code[qtoken] = {bare: 1, **{a: h + 1 for a, h in anc_by_code[bare].items()}}

    atom_idx, ancestor_codes = collapse_vocabulary(own_by_code, cov_by_code, anc_by_code, threshold=THRESHOLD)
    (out_dir / "vocab.json").write_text(json.dumps(atom_idx, indent=2, sort_keys=True))
    (out_dir / "ancestors.json").write_text(json.dumps(ancestor_codes, indent=2, sort_keys=True))
    (out_dir / "measurement_quantiles.json").write_text(json.dumps(q_edges, indent=2, sort_keys=True))

    final_events = pl.concat([
        non_drug.select(["subject_id", "time_seconds", "code"]),
        drug.select(["subject_id", "time_seconds", "code"]),
        meas,
    ])
    final_events = final_events.filter(pl.col("code").is_in(set(atom_idx.keys()))).sort(["subject_id", "time_seconds"])
    final_events.write_parquet(out_dir / "events.parquet")

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
    censor = pl.from_arrow(
        client.query(
            f"""SELECT CAST(person_id AS INT64) AS subject_id,
                       UNIX_SECONDS(TIMESTAMP(MAX(observation_period_end_date))) AS censor_seconds
                FROM `{cdr}.observation_period` GROUP BY person_id"""
        ).to_arrow()
    )

    offsets = final_events.with_row_index("row").group_by("subject_id", maintain_order=True).agg(
        pl.col("row").min().alias("start"),
        pl.col("row").max().alias("end"),
    )
    subjects = (
        offsets.join(persons, on="subject_id", how="inner")
        .join(censor, on="subject_id", how="inner")
        .sort("subject_id")
    )
    subjects.write_parquet(out_dir / "subjects.parquet")

    print(
        f"vocab={len(set(atom_idx.values())):,}  "
        f"events={final_events.height:,}  "
        f"subjects={subjects.height:,}  "
        f"measurement_concepts_with_quantiles={len(q_edges):,}  "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()

"""Build genterp's vocab, ancestors, value stats, event timelines, and subject metadata from AoU OMOP.

  - Drug events expanded to RxNorm ingredient atoms via drug_strength.
  - Measurement raw values flow through to events.parquet; per-atom (μ, σ) for
    magnitude-bearing codes are written to value_stats.json for the model's
    ValueModulator at training start.
  - Hierarchical collapse at threshold=500 patients across all domains.
  - observation_period_end_date drives per-subject right-censoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl
from google.cloud import bigquery


_T0 = time.monotonic()


def _log(msg: str) -> None:
    elapsed = time.monotonic() - _T0
    print(f"[aou_etl t+{elapsed:6.1f}s] {msg}", flush=True)


def _query_to_arrow(client: bigquery.Client, sql: str, label: str):
    _log(f"  bq query: {label}")
    t0 = time.monotonic()
    job = client.query(sql)
    table = job.to_arrow(progress_bar_type="tqdm")
    _log(f"  bq done:  {label} rows={table.num_rows:,} in {time.monotonic() - t0:.1f}s")
    return table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genterp.vocab import collapse_vocabulary

THRESHOLD = 500
TEST_SPLIT_PERCENT = 20


def split_for_subject(subject_id: int) -> str:
    """Deterministic per-person 80/20 split, stable across CDR refreshes."""
    digest = hashlib.sha256(str(int(subject_id)).encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "test" if bucket < TEST_SPLIT_PERCENT else "train"

NON_DRUG_TABLES = [
    ("condition_occurrence", "condition_concept_id", "condition_start_datetime"),
    ("procedure_occurrence", "procedure_concept_id", "procedure_datetime"),
    ("observation", "observation_concept_id", "observation_datetime"),
    ("visit_occurrence", "visit_concept_id", "visit_start_datetime"),
    ("device_exposure", "device_concept_id", "device_exposure_start_datetime"),
]


def _cache_key(cdr: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", cdr).strip("_")
    return f"{key}_threshold-{THRESHOLD}_values-v2"


def _write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _stable_json_fingerprint(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _path_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_parquet(path: Path, build: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    if path.exists():
        _log(f"cache hit:  {path.name}")
        return pl.read_parquet(path)
    _log(f"cache miss: {path.name} — building")
    t0 = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    tmp = path.with_suffix(path.suffix + ".tmp")
    data.write_parquet(tmp)
    tmp.replace(path)
    _log(f"cached:     {path.name} rows={data.height:,} in {time.monotonic() - t0:.1f}s")
    return data


def _cache_json(path: Path, build: Callable[[], object]) -> object:
    if path.exists():
        _log(f"cache hit:  {path.name}")
        return json.loads(path.read_text())
    _log(f"cache miss: {path.name} — building")
    t0 = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    _write_json(path, data)
    _log(f"cached:     {path.name} in {time.monotonic() - t0:.1f}s")
    return data


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
      ds.ingredient_concept_id AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM `{cdr}.drug_exposure` de
    JOIN `{cdr}.drug_strength` ds ON ds.drug_concept_id = de.drug_concept_id
    JOIN `{cdr}.concept` c ON c.concept_id = ds.ingredient_concept_id
    WHERE de.drug_concept_id > 0 AND de.drug_exposure_start_datetime IS NOT NULL
    """
    return _query_to_arrow(client, sql, "drug_exposure → ingredient")


def _non_drug_events_arrow(client: bigquery.Client, cdr: str):
    sql = f"""
    WITH events AS (
      {_non_drug_events_cte(cdr, with_time=True)}
    )
    SELECT
      CAST(events.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(events.t) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      events.cid AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM events JOIN `{cdr}.concept` c ON c.concept_id = events.cid
    """
    return _query_to_arrow(client, sql, "condition+procedure+observation+visit+device")


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
    return _query_to_arrow(client, sql, "measurement (with value_as_number)")


def _coverage_and_ancestors(client: bigquery.Client, cdr: str, cohort_ids: list[int]):
    _log(f"  bq query: coverage (concept_ancestor JOIN cohort events, {len(cohort_ids):,} cohort ids)")
    t0 = time.monotonic()
    cov_rows = list(client.query(
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
    ).result())
    cov = {int(r["aid"]): int(r["n"]) for r in cov_rows}
    _log(f"  bq done:  coverage rows={len(cov):,} in {time.monotonic() - t0:.1f}s")

    _log("  bq query: ancestor closure (concept_ancestor for cohort+ancestors)")
    t0 = time.monotonic()
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
    _log(f"  bq done:  ancestors descendants={len(anc):,} in {time.monotonic() - t0:.1f}s")
    return cov, anc


def _concept_codes(client: bigquery.Client, cdr: str, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    _log(f"  bq query: concept codes for {len(ids):,} ids")
    t0 = time.monotonic()
    rows = list(client.query(
        f"SELECT concept_id, vocabulary_id, concept_code FROM `{cdr}.concept` WHERE concept_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "INT64", list(ids)),
        ]),
    ).result())
    out = {int(r["concept_id"]): f"{r['vocabulary_id']}/{r['concept_code']}" for r in rows}
    _log(f"  bq done:  concept codes resolved={len(out):,} in {time.monotonic() - t0:.1f}s")
    return out


def _cached_coverage_and_ancestors(client: Callable[[], bigquery.Client], cdr: str, cohort_ids: list[int], cache_dir: Path):
    cohort_key = _stable_json_fingerprint(sorted(cohort_ids))

    def build() -> dict[str, list]:
        coverage, ancestors = _coverage_and_ancestors(client(), cdr, cohort_ids)
        return {
            "coverage": [[cid, count] for cid, count in sorted(coverage.items())],
            "ancestors": [
                [desc, [[anc, hops] for anc, hops in sorted(desc_ancestors.items())]]
                for desc, desc_ancestors in sorted(ancestors.items())
            ],
        }

    payload = _cache_json(
        cache_dir / f"coverage_and_ancestors-{cohort_key}.json",
        build,
    )
    return (
        {int(cid): int(count) for cid, count in payload["coverage"]},
        {
            int(desc): {int(anc): int(hops) for anc, hops in ancestors}
            for desc, ancestors in payload["ancestors"]
        },
    )


def _cached_concept_codes(client: Callable[[], bigquery.Client], cdr: str, ids: set[int], cache_dir: Path) -> dict[int, str]:
    path = cache_dir / "concept_codes.json"
    cached = {int(cid): str(code) for cid, code in json.loads(path.read_text())} if path.exists() else {}
    missing = ids - set(cached)
    if not missing:
        return cached

    cached.update(_concept_codes(client(), cdr, missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, [[cid, code] for cid, code in sorted(cached.items())])
    return cached


def _cached_own_counts(events_all: pl.DataFrame, cache_dir: Path, source_key: str) -> dict[int, int]:
    payload = _cache_json(
        cache_dir / f"own_counts-{source_key}.json",
        lambda: [
            [int(r["cid"]), int(r["n"])]
            for r in events_all.group_by("cid").agg(pl.col("subject_id").n_unique().alias("n")).iter_rows(named=True)
        ],
    )
    return {int(cid): int(n) for cid, n in payload}


def _cached_value_stats(
    events_all: pl.DataFrame,
    atom_idx: dict[str, int],
    cache_dir: Path,
    source_key: str,
    vocab_key: str,
) -> dict[str, dict[str, float]]:
    def build() -> dict[str, dict[str, float]]:
        stats_df = (
            events_all.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("code")
            .agg(
                pl.col("value").mean().alias("mu"),
                pl.col("value").std().alias("sigma"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= THRESHOLD)
        )
        return {
            r["code"]: {"mu": float(r["mu"]), "sigma": float(r["sigma"] or 1.0)}
            for r in stats_df.iter_rows(named=True)
            if r["code"] in atom_idx
        }

    payload = _cache_json(cache_dir / f"value_stats-{source_key}-{vocab_key}.json", build)
    return {str(code): {"mu": float(stats["mu"]), "sigma": float(stats["sigma"])} for code, stats in payload.items()}


def main() -> None:
    cdr = os.environ.get("WORKSPACE_CDR")
    if not cdr:
        raise SystemExit(
            "WORKSPACE_CDR is not set. On the AoU Researcher Workbench it's set automatically; "
            "outside AoU, export WORKSPACE_CDR=<project>.<dataset> before running."
        )
    out_dir = Path.home() / "genterp" / "etl"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache" / _cache_key(cdr)
    _log(f"CDR={cdr}")
    _log(f"cache_dir={cache_dir}")
    client_instance: bigquery.Client | None = None

    def client() -> bigquery.Client:
        nonlocal client_instance
        if client_instance is None:
            _log("creating BigQuery client")
            client_instance = bigquery.Client()
        return client_instance

    _log("step 1/8: pull non-drug events (condition, procedure, observation, visit, device)")
    non_drug = _cache_parquet(cache_dir / "non_drug_events.parquet", lambda: pl.from_arrow(_non_drug_events_arrow(client(), cdr)))
    _log("step 2/8: pull drug events (drug_exposure JOIN drug_strength → ingredient)")
    drug = _cache_parquet(cache_dir / "drug_events.parquet", lambda: pl.from_arrow(_drug_events_arrow(client(), cdr)))
    _log("step 3/8: pull measurement events (with value_as_number)")
    meas = _cache_parquet(cache_dir / "measurement_events.parquet", lambda: pl.from_arrow(_measurement_events_arrow(client(), cdr)))

    _log("step 4/8: concatenate event streams")
    source_events_path = cache_dir / "all_source_events.parquet"
    events_all = _cache_parquet(
        source_events_path,
        lambda: pl.concat([
            non_drug.select(["subject_id", "time_seconds", "code", "cid", "value"]),
            drug.select(["subject_id", "time_seconds", "code", "cid", "value"]),
            meas.select(["subject_id", "time_seconds", "code", "cid", "value"]),
        ]),
    )
    source_key = _path_fingerprint(source_events_path)
    _log(f"  total events: {events_all.height:,}")
    own_by_cid = _cached_own_counts(events_all, cache_dir, source_key)
    _log(f"  unique concept_ids: {len(own_by_cid):,}")

    _log("step 5/8: vocab collapse")
    vocab_cache = cache_dir / f"collapsed_vocab-{source_key}.json"
    if vocab_cache.exists():
        _log(f"  cache hit: {vocab_cache.name}")
        atom_idx = {str(code): int(atom) for code, atom in json.loads(vocab_cache.read_text()).items()}
    else:
        _log("  cache miss — running coverage+ancestors queries and threshold-500 collapse")
        cov, anc = _cached_coverage_and_ancestors(client, cdr, list(own_by_cid), cache_dir)

        all_ids = set(own_by_cid) | set(cov) | {a for d in anc.values() for a in d}
        code_of = _cached_concept_codes(client, cdr, all_ids, cache_dir)
        own_by_code = {code_of[c]: n for c, n in own_by_cid.items() if c in code_of}
        cov_by_code = {code_of[c]: n for c, n in cov.items() if c in code_of}
        anc_by_code = {
            code_of[d]: {code_of[a]: h for a, h in ancs.items() if a in code_of}
            for d, ancs in anc.items() if d in code_of
        }

        atom_idx = collapse_vocabulary(own_by_code, cov_by_code, anc_by_code, threshold=THRESHOLD)
        _log(f"  collapsed vocab: {len(set(atom_idx.values())):,} atoms covering {len(atom_idx):,} codes")
        _write_json(vocab_cache, atom_idx)
    _write_json(out_dir / "vocab.json", atom_idx)

    _log("step 6/8: per-atom value stats (μ, σ over numeric measurements)")
    vocab_key = _stable_json_fingerprint(atom_idx)
    stats = _cached_value_stats(events_all, atom_idx, cache_dir, source_key, vocab_key)
    _write_json(out_dir / "value_stats.json", stats)
    _log(f"  magnitude-bearing atoms: {len(stats):,}")

    _log("step 7/8: filter+sort final events.parquet")
    final_events = _cache_parquet(
        cache_dir / f"events-{source_key}-{vocab_key}.parquet",
        lambda: events_all.select(["subject_id", "time_seconds", "code", "value"])
        .filter(pl.col("code").is_in(set(atom_idx.keys())))
        .sort(["subject_id", "time_seconds"]),
    )
    final_events.write_parquet(out_dir / "events.parquet")

    _log("step 8/8: pull person + observation_period (sex, birth, censoring)")
    persons = _cache_parquet(
        cache_dir / "persons.parquet",
        lambda: pl.from_arrow(
            _query_to_arrow(client(), f"""SELECT
                  CAST(person_id AS INT64) AS subject_id,
                  IF(gender_concept_id = 8507, 1, 0) AS sex,
                  UNIX_SECONDS(COALESCE(
                    birth_datetime,
                    TIMESTAMP(DATE(year_of_birth, COALESCE(month_of_birth, 1), COALESCE(day_of_birth, 1)))
                  )) AS birth_seconds
                FROM `{cdr}.person`""", "person")
        ),
    )
    censor = _cache_parquet(
        cache_dir / "censor.parquet",
        lambda: pl.from_arrow(
            _query_to_arrow(client(), f"""SELECT CAST(person_id AS INT64) AS subject_id,
                           UNIX_SECONDS(TIMESTAMP(MAX(observation_period_end_date))) AS censor_seconds
                    FROM `{cdr}.observation_period` GROUP BY person_id""", "observation_period")
        ),
    )

    offsets = final_events.with_row_index("row").group_by("subject_id", maintain_order=True).agg(
        pl.col("row").min().alias("start"),
        pl.col("row").max().alias("end"),
    )
    subjects = (
        offsets.join(persons, on="subject_id", how="inner")
        .join(censor, on="subject_id", how="inner")
        .with_columns(
            pl.col("subject_id")
            .map_elements(split_for_subject, return_dtype=pl.Utf8)
            .alias("split")
        )
        .sort("subject_id")
    )
    subjects = _cache_parquet(cache_dir / f"subjects-{source_key}-{vocab_key}-split{TEST_SPLIT_PERCENT}.parquet", lambda: subjects)
    subjects.write_parquet(out_dir / "subjects.parquet")

    split_counts = subjects.group_by("split").agg(pl.len().alias("n")).sort("split")
    split_summary = "  ".join(f"{row['split']}={row['n']:,}" for row in split_counts.iter_rows(named=True))
    print(
        f"vocab={len(set(atom_idx.values())):,}  "
        f"events={final_events.height:,}  "
        f"subjects={subjects.height:,}  "
        f"magnitude_atoms={len(stats):,}  "
        f"split[{split_summary}]  "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()

"""Build genterp's vocab, ancestors, value stats, event timelines, and subject metadata from AoU OMOP.

  - Drug events expanded to RxNorm ingredient atoms via drug_strength.
  - Measurement raw values flow through to events.parquet; collapsed atom ids
    are materialized as uint32 so training never string-encodes events.
  - Per-atom (μ, σ) for magnitude-bearing codes are written to
    value_stats.json for the model's ValueModulator at training start.
  - Hierarchical collapse at threshold=500 patients across all domains.
  - observation_period_end_date drives per-subject right-censoring.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery, bigquery_storage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genterp.progress import ProgressLogger
from genterp.vocab import collapse_vocabulary


_PROC = psutil.Process()


def _mem_str() -> str:
    return f"RSS={_PROC.memory_info().rss / 1e9:.2f}GB"

_WORK = ProgressLogger("aou_etl", total_units=11)


def _log(msg: str) -> None:
    _WORK.log(f"{msg} [{_mem_str()}]")


_BQ_JOB_CACHE_DIR: Path | None = None


def _set_bq_job_cache_dir(p: Path) -> None:
    global _BQ_JOB_CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    _BQ_JOB_CACHE_DIR = p


def _bq_job_id_file(sql: str) -> Path | None:
    if _BQ_JOB_CACHE_DIR is None:
        return None
    key = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
    return _BQ_JOB_CACHE_DIR / f"{key}.txt"


def _submit_or_reuse_job(client: bigquery.Client, sql: str, label: str) -> bigquery.QueryJob:
    """Resume a prior server-side BQ job for this exact SQL if still available, else submit a fresh one."""
    job_id_file = _bq_job_id_file(sql)
    if job_id_file is not None and job_id_file.exists():
        prev_id = job_id_file.read_text().strip()
        try:
            job = client.get_job(prev_id)
            if job.state == "DONE" and job.error_result is None:
                _log(f"  bq reuse:  {label} job_id={prev_id} (server-cached result)")
                return job
            _log(f"  bq prior job state={job.state}; resubmitting")
        except Exception as exc:
            _log(f"  bq prior job lookup failed ({exc.__class__.__name__}); resubmitting")
    _log(f"  bq submit: {label}")
    job = client.query(sql)
    if job_id_file is not None:
        job_id_file.write_text(job.job_id or "")
        _log(f"  bq job_id={job.job_id} (recorded for resume)")
    return job


def _query_to_arrow(client: bigquery.Client, sql: str, label: str):
    """One-shot Arrow Table fetch — only safe for small results (concept lookup, censor, person)."""
    t0 = time.monotonic()
    job = _submit_or_reuse_job(client, sql, label)
    table = job.to_arrow(progress_bar_type="tqdm", create_bqstorage_client=False)
    _log(f"  bq done:   {label} rows={table.num_rows:,} in {time.monotonic() - t0:.1f}s")
    return table


def _wait_with_progress(job: bigquery.QueryJob, label: str, poll_s: float = 5.0) -> None:
    """Block until ``job`` is DONE, logging state and bytes-processed every poll_s seconds.

    Without this, a long aggregation query (coverage over 1B events, etc.) goes silent
    for minutes between submit and result — indistinguishable from a hang.
    """
    t0 = time.monotonic()
    last_log = t0
    while not job.done():
        time.sleep(min(poll_s, 1.0))
        if time.monotonic() - last_log < poll_s:
            continue
        try:
            job.reload()
        except Exception as exc:
            _log(f"  bq poll:    {label} reload failed ({exc.__class__.__name__}); retrying")
            last_log = time.monotonic()
            continue
        bytes_proc = (job.total_bytes_processed or 0) / 1e9
        slot_ms = getattr(job, "slot_millis", None) or 0
        _log(
            f"  bq running: {label} state={job.state} bytes_scanned={bytes_proc:.2f}GB "
            f"slot_time={slot_ms/1000:.1f}s elapsed={time.monotonic()-t0:.1f}s"
        )
        last_log = time.monotonic()
    if job.error_result:
        raise RuntimeError(f"BQ {label} failed: {job.error_result}")


def _run_aggregation(
    client: bigquery.Client,
    sql: str,
    label: str,
    parameters: Sequence[bigquery.ArrayQueryParameter] | None = None,
) -> pa.Table:
    """Submit a BQ aggregation query, log live progress, download via Storage API.

    For analytic queries (coverage, ancestors, concept codes) where we need to *see*
    the query making progress. Job-id caching is skipped because parameters change the
    result, and these queries are anyway one-shot per CDR (the outer JSON cache handles
    persistence across runs).
    """
    t0 = time.monotonic()
    job_config = bigquery.QueryJobConfig(query_parameters=list(parameters or []))
    _log(f"  bq submit:  {label}")
    job = client.query(sql, job_config=job_config)
    _log(f"  bq job_id:  {label} {job.job_id}")

    _wait_with_progress(job, label)
    t_done = time.monotonic()
    _log(f"  bq query:   {label} completed in {t_done-t0:.1f}s; downloading via Storage API")

    bqs_client = bigquery_storage.BigQueryReadClient()
    try:
        batches = list(job.result().to_arrow_iterable(bqstorage_client=bqs_client))
        table = pa.Table.from_batches(batches) if batches else pa.table({})
    finally:
        try:
            bqs_client.transport.close()
        except Exception:
            pass
    _log(
        f"  bq done:    {label} rows={table.num_rows:,} download={time.monotonic()-t_done:.1f}s "
        f"total={time.monotonic()-t0:.1f}s"
    )
    return table


def _stream_query_to_parquet(client: bigquery.Client, sql: str, label: str, out_path: Path) -> None:
    """Submit/reuse a BQ job and stream its Arrow batches directly to a Parquet file.

    Uses the BigQuery Storage API (gRPC + parallel streams) — typically 20–50×
    faster than REST pagination for large results. Never materializes the full
    Arrow table in process memory.
    """
    if out_path.exists():
        _log(f"cache hit:  {out_path.name}")
        return
    _log(f"streaming:  {out_path.name} (BQ Storage API, parallel streams)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    t0 = time.monotonic()
    job = _submit_or_reuse_job(client, sql, label)

    bqs_client = bigquery_storage.BigQueryReadClient()
    try:
        iterable = job.result().to_arrow_iterable(bqstorage_client=bqs_client)
        writer: pq.ParquetWriter | None = None
        rows = 0
        batches = 0
        t_first: float | None = None
        t_last_log = time.monotonic()
        try:
            for batch in iterable:
                if writer is None:
                    writer = pq.ParquetWriter(tmp, batch.schema, compression="zstd")
                    t_first = time.monotonic()
                    _log(f"  stream:    {label} first batch in {t_first - t0:.1f}s rows={batch.num_rows:,}")
                assert writer is not None
                writer.write_batch(batch)
                rows += batch.num_rows
                batches += 1
                # Log every 2 seconds — fast enough to feel live, sparse enough for big runs.
                now = time.monotonic()
                if now - t_last_log >= 2.0:
                    elapsed = now - (t_first or t0)
                    rate = rows / max(elapsed, 1e-6)
                    _log(f"  stream:    {label} rows={rows:,} batches={batches} rate={rate/1e6:.2f}M/s")
                    t_last_log = now
        finally:
            if writer is not None:
                writer.close()
    finally:
        try:
            bqs_client.transport.close()
        except Exception:
            pass
    tmp.replace(out_path)
    elapsed = time.monotonic() - t0
    rate = rows / max(elapsed, 1e-6)
    _log(f"streamed:   {out_path.name} rows={rows:,} batches={batches} in {elapsed:.1f}s ({rate/1e6:.2f}M rows/s)")


THRESHOLD = 500
TEST_SPLIT_PERCENT = 20
TINY_PERSON_MOD = 10  # --tiny samples 1 in N person_ids end-to-end
TINY = False  # set by main() from --tiny; module-level so SQL builders read it


def _tiny_predicate(person_col: str) -> str:
    """SQL fragment restricting to 1/TINY_PERSON_MOD of person_ids when --tiny is set.

    Pushed inside each per-domain WHERE so BigQuery prunes on the source tables and
    scans ~10% of rows — keeps the same code path as full runs but ~10× cheaper /
    faster end-to-end.
    """
    return f" AND MOD({person_col}, {TINY_PERSON_MOD}) = 0" if TINY else ""


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
    suffix = f"_tiny{TINY_PERSON_MOD}x" if TINY else ""
    return f"{key}_threshold-{THRESHOLD}_values-v2{suffix}"


def _write_json(path: Path, data: object) -> None:
    _log(f"writing json atomically: {path} ({_payload_units(data)})")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)
    _log(f"json write complete: {path} ({tmp.name} replaced target)")


def _payload_units(data: object) -> str:
    if isinstance(data, dict):
        return f"items={len(data):,}"
    if isinstance(data, (list, tuple, set)):
        return f"items={len(data):,}"
    return f"type={type(data).__name__}"


def _stable_json_fingerprint(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _path_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_parquet(path: Path, build: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    if path.exists():
        _log(f"cache hit:  {path.name}; reading parquet from disk")
        t0 = time.monotonic()
        data = pl.read_parquet(path)
        _log(f"cache read complete: {path.name} rows={data.height:,} columns={len(data.columns):,} in {time.monotonic() - t0:.1f}s")
        return data
    _log(f"cache miss: {path.name}; building dataframe")
    t_build = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    _log(f"cache build complete: {path.name} rows={data.height:,} columns={len(data.columns):,} in {time.monotonic() - t_build:.1f}s")
    tmp = path.with_suffix(path.suffix + ".tmp")
    _log(f"parquet write starting: {tmp} rows={data.height:,}")
    t_write = time.monotonic()
    data.write_parquet(tmp)
    bytes_written = tmp.stat().st_size
    _log(f"parquet write complete: {tmp.name} bytes={bytes_written:,} in {time.monotonic() - t_write:.1f}s; renaming to final path")
    tmp.replace(path)
    _log(f"cached:     {path.name} rows={data.height:,} total={time.monotonic() - t_build:.1f}s")
    return data


def _arrow_to_polars(arrow_table, label: str) -> pl.DataFrame:
    """Used only for small results (person, observation_period)."""
    _log(f"  arrow→polars: {label} (rows={arrow_table.num_rows:,})")
    t0 = time.monotonic()
    df = pl.from_arrow(arrow_table)
    _log(f"  arrow→polars done: {label} rows={df.height:,} columns={len(df.columns):,} in {time.monotonic() - t0:.1f}s")
    return df


def _sink_parquet(lf: pl.LazyFrame, path: Path, label: str) -> None:
    """Streaming sink: never materialize the full frame in memory."""
    if path.exists():
        _log(f"cache hit:  {path.name}")
        return
    _log(f"sink:       {path.name} (streaming)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    t0 = time.monotonic()
    lf.sink_parquet(tmp, compression="zstd")
    tmp.replace(path)
    _log(f"sunk:       {path.name} bytes={path.stat().st_size:,} in {time.monotonic() - t0:.1f}s")


def _cache_json(path: Path, build: Callable[[], object]) -> object:
    if path.exists():
        _log(f"cache hit:  {path.name}; reading json from disk")
        payload = json.loads(path.read_text())
        _log(f"cache read complete: {path.name} ({_payload_units(payload)})")
        return payload
    _log(f"cache miss: {path.name}; building json payload")
    t0 = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    _write_json(path, data)
    _log(f"cached:     {path.name} in {time.monotonic() - t0:.1f}s")
    return data


def _non_drug_events_cte(cdr: str, with_time: bool) -> str:
    sel = "person_id, {c} AS cid, {t} AS t" if with_time else "person_id, {c} AS cid"
    where = "{c} > 0 AND {t} IS NOT NULL" if with_time else "{c} > 0"
    tiny = _tiny_predicate("person_id")
    parts = [
        f"SELECT {sel.format(c=col, t=tcol)} FROM `{cdr}.{tbl}` WHERE {where.format(c=col, t=tcol)}{tiny}"
        for tbl, col, tcol in NON_DRUG_TABLES
    ]
    return "\n  UNION ALL ".join(parts)


def _drug_events_sql(cdr: str) -> str:
    return f"""
    SELECT
      CAST(de.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(de.drug_exposure_start_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      ds.ingredient_concept_id AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM `{cdr}.drug_exposure` de
    JOIN `{cdr}.drug_strength` ds ON ds.drug_concept_id = de.drug_concept_id
    JOIN `{cdr}.concept` c ON c.concept_id = ds.ingredient_concept_id
    WHERE de.drug_concept_id > 0 AND de.drug_exposure_start_datetime IS NOT NULL{_tiny_predicate("de.person_id")}
    """


def _non_drug_events_sql(cdr: str) -> str:
    return f"""
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


def _measurement_events_sql(cdr: str) -> str:
    return f"""
    SELECT
      CAST(m.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(m.measurement_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      m.measurement_concept_id AS cid,
      m.value_as_number AS value
    FROM `{cdr}.measurement` m JOIN `{cdr}.concept` c ON c.concept_id = m.measurement_concept_id
    WHERE m.measurement_concept_id > 0 AND m.measurement_datetime IS NOT NULL{_tiny_predicate("m.person_id")}
    """


def _all_events_sql(cdr: str) -> str:
    """One combined event stream, unsorted.

    A global ``ORDER BY`` at BigQuery looks attractive but defeats the BQ Storage API: a
    sorted result requires a single final-stage worker, so ``to_arrow_iterable`` falls
    back to a single read stream and throughput collapses to REST-API levels
    (~25K rows/sec for 434M rows ≈ hours). With no ORDER BY, BQ writes results in
    parallel shards and the Storage API streams them concurrently — typically 1–5M
    rows/sec on AoU workspaces. We sort downstream with polars streaming sort,
    which spills to disk and never materializes the full table in RAM.
    """
    return f"""
    WITH events AS (
      ({_non_drug_events_sql(cdr)})
      UNION ALL ({_drug_events_sql(cdr)})
      UNION ALL ({_measurement_events_sql(cdr)})
    )
    SELECT subject_id, time_seconds, code, cid, value
    FROM events
    """


def _coverage_sql(cdr: str) -> str:
    """Coverage = approximate number of distinct subjects with any descendant of each concept.

    Uses ``APPROX_COUNT_DISTINCT`` (HyperLogLog++) instead of exact ``COUNT(DISTINCT)``.
    On 1B+ events the exact path takes minutes of single-shuffle BQ time; the HLL++ path
    runs in parallel and finishes in seconds with ≤2% relative error — well below the
    granularity that matters at threshold=500.
    """
    return f"""
    WITH events AS (
      {_non_drug_events_cte(cdr, with_time=False)}
      UNION ALL SELECT person_id, drug_concept_id AS cid FROM `{cdr}.drug_exposure` WHERE drug_concept_id > 0
      UNION ALL SELECT person_id, measurement_concept_id AS cid FROM `{cdr}.measurement` WHERE measurement_concept_id > 0
    )
    SELECT ca.ancestor_concept_id AS aid, APPROX_COUNT_DISTINCT(events.person_id) AS n
    FROM events JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = events.cid
    GROUP BY ca.ancestor_concept_id
    """


def _ancestor_closure_sql(cdr: str) -> str:
    return f"""
    WITH relevant AS (
      SELECT DISTINCT cid FROM UNNEST(@cohort) AS cid
      UNION DISTINCT SELECT DISTINCT ancestor_concept_id FROM `{cdr}.concept_ancestor` WHERE descendant_concept_id IN UNNEST(@cohort)
    )
    SELECT descendant_concept_id AS d, ancestor_concept_id AS a, min_levels_of_separation AS hops
    FROM `{cdr}.concept_ancestor`
    WHERE descendant_concept_id IN (SELECT cid FROM relevant) AND min_levels_of_separation > 0
    """


def _coverage_and_ancestors(client: bigquery.Client, cdr: str, cohort_ids: list[int]):
    """Run coverage + ancestor closure in PARALLEL; both stream Arrow via Storage API.

    Independent queries — no reason to wait on coverage before submitting ancestor
    closure. ThreadPoolExecutor is fine here: BigQuery client objects are thread-safe
    and the work is I/O-bound (GIL is released during gRPC waits).
    """
    cohort_param = bigquery.ArrayQueryParameter("cohort", "INT64", cohort_ids)

    def run_coverage() -> pa.Table:
        return _run_aggregation(client, _coverage_sql(cdr), "coverage")

    def run_ancestors() -> pa.Table:
        return _run_aggregation(
            client, _ancestor_closure_sql(cdr), "ancestor_closure", parameters=[cohort_param]
        )

    _log(f"submitting coverage + ancestor closure in parallel (cohort={len(cohort_ids):,})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="bq") as pool:
        cov_future = pool.submit(run_coverage)
        anc_future = pool.submit(run_ancestors)
        cov_table = cov_future.result()
        anc_table = anc_future.result()

    _log(f"converting coverage Arrow → dict (rows={cov_table.num_rows:,})")
    cov_aids = cov_table.column("aid").to_pylist()
    cov_ns = cov_table.column("n").to_pylist()
    cov = {int(a): int(n) for a, n in zip(cov_aids, cov_ns, strict=True)}

    _log(f"converting ancestor closure Arrow → nested dict (rows={anc_table.num_rows:,})")
    anc: dict[int, dict[int, int]] = {}
    anc_d = anc_table.column("d").to_pylist()
    anc_a = anc_table.column("a").to_pylist()
    anc_h = anc_table.column("hops").to_pylist()
    for d, a, h in zip(anc_d, anc_a, anc_h, strict=True):
        anc.setdefault(int(d), {})[int(a)] = int(h)
    _log(f"  derived: coverage_ancestors={len(cov):,} descendants_with_ancestors={len(anc):,}")
    return cov, anc


def _concept_codes(client: bigquery.Client, cdr: str, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    sql = f"SELECT concept_id, vocabulary_id, concept_code FROM `{cdr}.concept` WHERE concept_id IN UNNEST(@ids)"
    table = _run_aggregation(
        client,
        sql,
        f"concept_codes (n={len(ids):,})",
        parameters=[bigquery.ArrayQueryParameter("ids", "INT64", list(ids))],
    )
    cids = table.column("concept_id").to_pylist()
    vocabs = table.column("vocabulary_id").to_pylist()
    codes = table.column("concept_code").to_pylist()
    return {int(c): f"{v}/{k}" for c, v, k in zip(cids, vocabs, codes, strict=True)}


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
        _log(f"concept code cache satisfied: cached={len(cached):,} requested={len(ids):,} missing=0")
        return cached

    _log(f"concept code cache incomplete: cached={len(cached):,} requested={len(ids):,} missing={len(missing):,}")
    cached.update(_concept_codes(client(), cdr, missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, [[cid, code] for cid, code in sorted(cached.items())])
    return cached


def _cached_own_counts(events_lf: pl.LazyFrame, cache_dir: Path, source_key: str) -> dict[int, int]:
    def build() -> list[list[int]]:
        df = (
            events_lf.group_by("cid")
            .agg(pl.col("subject_id").n_unique().alias("n"))
            .collect(engine="streaming")
        )
        return [[int(r["cid"]), int(r["n"])] for r in df.iter_rows(named=True)]

    payload = _cache_json(cache_dir / f"own_counts-{source_key}.json", build)
    return {int(cid): int(n) for cid, n in payload}


def _cached_value_stats(
    events_lf: pl.LazyFrame,
    atom_idx: dict[str, int],
    cache_dir: Path,
    source_key: str,
    vocab_key: str,
) -> dict[str, dict[str, float]]:
    def build() -> dict[str, dict[str, float]]:
        stats_df = (
            events_lf.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .group_by("code")
            .agg(
                pl.col("value").mean().alias("mu"),
                pl.col("value").std().alias("sigma"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= THRESHOLD)
            .collect(engine="streaming")
        )
        return {
            r["code"]: {"mu": float(r["mu"]), "sigma": float(r["sigma"] or 1.0)}
            for r in stats_df.iter_rows(named=True)
            if r["code"] in atom_idx
        }

    payload = _cache_json(cache_dir / f"value_stats-{source_key}-{vocab_key}.json", build)
    return {str(code): {"mu": float(stats["mu"]), "sigma": float(stats["sigma"])} for code, stats in payload.items()}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build genterp ETL artifacts from AoU OMOP.")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help=f"Sample 1/{TINY_PERSON_MOD} of person_ids end-to-end for quick iteration. "
        f"Cache lands under a separate _tiny{TINY_PERSON_MOD}x namespace so it doesn't "
        "collide with full-CDR artifacts.",
    )
    args = parser.parse_args(argv)
    global TINY
    TINY = args.tiny

    _WORK.start_unit("validate AoU CDR configuration", "reading WORKSPACE_CDR and preparing output paths")
    cdr = os.environ.get("WORKSPACE_CDR")
    if not cdr:
        raise SystemExit(
            "WORKSPACE_CDR is not set. On the AoU Researcher Workbench it's set automatically; "
            "outside AoU, export WORKSPACE_CDR=<project>.<dataset> before running."
        )
    out_dir = Path.home() / "genterp" / "etl"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache" / _cache_key(cdr)
    _set_bq_job_cache_dir(cache_dir / "bq_jobs")
    _log(f"CDR={cdr}")
    _log(f"cache_dir={cache_dir}")
    if TINY:
        _log(f"--tiny active: sampling 1/{TINY_PERSON_MOD} of person_ids end-to-end")
    _WORK.finish_unit("validate AoU CDR configuration", f"out_dir={out_dir} cache_dir={cache_dir}")
    client_instance: bigquery.Client | None = None

    def client() -> bigquery.Client:
        nonlocal client_instance
        if client_instance is None:
            _log("creating BigQuery client")
            client_instance = bigquery.Client()
        return client_instance

    all_events_path = cache_dir / "all_events.parquet"

    _WORK.start_unit(
        "pull all events (combined, unsorted)",
        "single query: UNION ALL non-drug/drug/measurement; no ORDER BY so BQ Storage API parallel streams kick in",
    )
    _stream_query_to_parquet(client(), _all_events_sql(cdr), "all_events", all_events_path)
    _WORK.finish_unit("pull all events (combined, unsorted)", f"path={all_events_path.name}")

    _WORK.start_unit("build lazy event scan", "scanning all-events parquet as a lazy frame")
    source_key = _path_fingerprint(all_events_path)
    events_lf = pl.scan_parquet(str(all_events_path)).select(
        ["subject_id", "time_seconds", "code", "cid", "value"]
    )
    _log(f"source event fingerprint={source_key}")
    _WORK.finish_unit("build lazy event scan", f"source_key={source_key}")

    _WORK.start_unit("count distinct subjects per source concept", "lazy group_by over scanned event parquets")
    own_by_cid = _cached_own_counts(events_lf, cache_dir, source_key)
    _WORK.finish_unit("count distinct subjects per source concept", f"unique concept_ids={len(own_by_cid):,}")

    _WORK.start_unit("collapse vocabulary", f"threshold={THRESHOLD:,}; resolving ancestors and concept codes if cache is cold")
    vocab_cache = cache_dir / f"collapsed_vocab-{source_key}.json"
    if vocab_cache.exists():
        _log(f"vocab cache hit: {vocab_cache.name}; loading collapsed code-to-atom map")
        atom_idx = {str(code): int(atom) for code, atom in json.loads(vocab_cache.read_text()).items()}
    else:
        _log("vocab cache miss; running coverage+ancestors queries and threshold collapse")
        cov, anc = _cached_coverage_and_ancestors(client, cdr, list(own_by_cid), cache_dir)
        _log(f"coverage+ancestor payload ready: coverage_concepts={len(cov):,} descendant_maps={len(anc):,}")

        all_ids = set(own_by_cid) | set(cov) | {a for d in anc.values() for a in d}
        _log(f"resolving OMOP concept codes: own={len(own_by_cid):,} coverage={len(cov):,} ancestor_ids={len(all_ids):,}")
        code_of = _cached_concept_codes(client, cdr, all_ids, cache_dir)
        own_by_code = {code_of[c]: n for c, n in own_by_cid.items() if c in code_of}
        cov_by_code = {code_of[c]: n for c, n in cov.items() if c in code_of}
        anc_by_code = {
            code_of[d]: {code_of[a]: h for a, h in ancs.items() if a in code_of}
            for d, ancs in anc.items() if d in code_of
        }

        atom_idx = collapse_vocabulary(own_by_code, cov_by_code, anc_by_code, threshold=THRESHOLD)
        _log(f"collapsed vocab: atoms={len(set(atom_idx.values())):,} covered_codes={len(atom_idx):,}")
        _write_json(vocab_cache, atom_idx)
    _write_json(out_dir / "vocab.json", atom_idx)
    _WORK.finish_unit("collapse vocabulary", f"atoms={len(set(atom_idx.values())):,} covered_codes={len(atom_idx):,}")

    _WORK.start_unit("compute per-atom value stats", "lazy mean/stddev over finite numeric measurements")
    vocab_key = _stable_json_fingerprint(atom_idx)
    stats = _cached_value_stats(events_lf, atom_idx, cache_dir, source_key, vocab_key)
    _write_json(out_dir / "value_stats.json", stats)
    _WORK.finish_unit("compute per-atom value stats", f"magnitude-bearing atoms={len(stats):,} vocab_key={vocab_key}")

    _WORK.start_unit(
        "filter and sort final events",
        "streaming filter + atom-encode + (subject_id, time_seconds) sort to final parquet (polars spills to disk)",
    )
    final_events_path = cache_dir / f"events-{source_key}-{vocab_key}-atom-v1.parquet"
    keep_codes = pl.Series("code", list(atom_idx.keys()))
    atom_expr = pl.col("code").replace_strict(atom_idx, return_dtype=pl.UInt32).alias("atom")
    final_lf = (
        events_lf.select(["subject_id", "time_seconds", "code", "value"])
        .filter(pl.col("code").is_in(keep_codes.implode()))
        .select(["subject_id", "time_seconds", atom_expr, "value"])
        .sort(["subject_id", "time_seconds"])
    )
    _sink_parquet(final_lf, final_events_path, "final_events")
    _log(f"copying final events to: {out_dir / 'events.parquet'}")
    pl.scan_parquet(str(final_events_path)).sink_parquet(out_dir / "events.parquet", compression="zstd")
    final_events_lf = pl.scan_parquet(str(final_events_path))
    final_rows = final_events_lf.select(pl.len()).collect(engine="streaming").item()
    _WORK.finish_unit("filter and sort final events", f"rows={final_rows:,}")

    _WORK.start_unit("pull person demographics", "sex and birth timestamp from OMOP person")
    persons = _cache_parquet(
        cache_dir / "persons.parquet",
        lambda: _arrow_to_polars(
            _query_to_arrow(client(), f"""SELECT
                  CAST(person_id AS INT64) AS subject_id,
                  IF(gender_concept_id = 8507, 1, 0) AS sex,
                  UNIX_SECONDS(COALESCE(
                    birth_datetime,
                    TIMESTAMP(DATE(year_of_birth, COALESCE(month_of_birth, 1), COALESCE(day_of_birth, 1)))
                  )) AS birth_seconds
                FROM `{cdr}.person`
                WHERE TRUE{_tiny_predicate("person_id")}""", "person"),
            "person",
        ),
    )
    _WORK.finish_unit("pull person demographics", f"rows={persons.height:,}")

    _WORK.start_unit("pull observation-period censoring", "latest observation_period_end_date per subject")
    censor = _cache_parquet(
        cache_dir / "censor.parquet",
        lambda: _arrow_to_polars(
            _query_to_arrow(client(), f"""SELECT CAST(person_id AS INT64) AS subject_id,
                           UNIX_SECONDS(TIMESTAMP(MAX(observation_period_end_date))) AS censor_seconds
                    FROM `{cdr}.observation_period`
                    WHERE TRUE{_tiny_predicate("person_id")}
                    GROUP BY person_id""", "observation_period"),
            "observation_period",
        ),
    )
    _WORK.finish_unit("pull observation-period censoring", f"rows={censor.height:,}")

    _WORK.start_unit("build subject metadata", "row offsets, demographics, censoring, deterministic split labels")
    offsets = (
        final_events_lf.with_row_index("row")
        .group_by("subject_id", maintain_order=True)
        .agg(pl.col("row").min().alias("start"), pl.col("row").max().alias("end"))
        .collect(engine="streaming")
    )
    _log(f"subject offsets computed: subjects_with_events={offsets.height:,}")
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
    subjects = _cache_parquet(
        cache_dir / f"subjects-{source_key}-{vocab_key}-split{TEST_SPLIT_PERCENT}.parquet",
        lambda: subjects,
    )
    _log(f"writing final subjects parquet: {out_dir / 'subjects.parquet'} rows={subjects.height:,}")
    subjects.write_parquet(out_dir / "subjects.parquet")
    _WORK.finish_unit("build subject metadata", f"subjects={subjects.height:,}")

    _WORK.start_unit("summarize ETL artifacts", "counting split rows and final artifact dimensions")
    split_counts = subjects.group_by("split").agg(pl.len().alias("n")).sort("split")
    split_summary = "  ".join(f"{row['split']}={row['n']:,}" for row in split_counts.iter_rows(named=True))
    _WORK.finish_unit(
        "summarize ETL artifacts",
        f"vocab={len(set(atom_idx.values())):,}  "
        f"events={final_rows:,}  "
        f"subjects={subjects.height:,}  "
        f"magnitude_atoms={len(stats):,}  "
        f"split[{split_summary}]  "
        f"out_dir={out_dir}",
    )


if __name__ == "__main__":
    main()

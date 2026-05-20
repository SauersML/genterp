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

import atexit
import concurrent.futures
import csv
import faulthandler
import hashlib
import io
import json
import os
import re
import signal
import sys
import time
import traceback
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import duckdb
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery, bigquery_storage, storage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genterp.progress import ProgressLogger
from genterp.vocab import collapse_vocabulary


_PROC = psutil.Process()


def _mem_str() -> str:
    return f"RSS={_PROC.memory_info().rss / 1e9:.2f}GB"

_WORK = ProgressLogger("aou_etl", total_units=11)


def _log(msg: str) -> None:
    _WORK.log(f"{msg} [{_mem_str()}]")


def _log_version_banner() -> None:
    """Loud, unambiguous statement of *which on-disk source* this process is running.

    The "OLD code keeps showing up after I pushed" mystery happens when run.sh's
    git reset succeeds but, for whatever reason (network FS lag, an open file
    handle from a parent process, manual GENTERP_REEXEC=1 invocation that
    skipped self-update), the .py we actually execute is from a different
    commit. This banner makes that impossible to miss: every run logs the
    source path, its mtime, byte size, and SHA-256 of the actual file contents.
    Grep the log for VERSION_TAG to confirm what's running.
    """
    src_path = Path(__file__).resolve()
    try:
        data = src_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()[:16]
        size = len(data)
        mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(src_path.stat().st_mtime))
    except OSError as exc:
        _log(f"VERSION_TAG: source-readable-failed path={src_path} exc={exc.__class__.__name__}")
        return
    _log(f"VERSION_TAG: path={src_path} size={size} mtime={mtime} sha256_16={sha}")
    # If the chunked-sort marker is missing, the source on disk is stale.
    if b"FINAL_EVENTS_SHARDS" not in data:
        _log("VERSION_TAG: WARNING: FINAL_EVENTS_SHARDS marker absent — running pre-chunked-sort code")


def _install_crash_diagnostics() -> None:
    """Make non-OOM crashes loud and OOM crashes diagnosable after the fact.

    What this covers:
      * faulthandler.enable: prints a C-level traceback to stderr on SIGSEGV,
        SIGFPE, SIGABRT, SIGBUS, SIGILL — the polars/pyarrow Rust panic surface.
      * faulthandler.register on SIGTERM/SIGHUP: dumps a Python traceback when
        we get terminated cleanly (e.g. parent shell hangup, job manager kill).
      * sys.excepthook: catches the last unhandled Python exception so it lands
        in the same _log channel as every other line. Otherwise Trainer-style
        async output ordering can hide the traceback above the prompt return.
      * atexit: logs the final RSS so a graceful exit leaves a footprint.

    What this does NOT cover: SIGKILL from the kernel OOM-killer. Nothing
    in-process can catch SIGKILL — the only mitigation is to not run
    memory-hungry steps locally. That's why ``own_counts`` was moved off
    polars and onto BigQuery in the same change as installing this.
    """
    faulthandler.enable()
    for sig_name in ("SIGTERM", "SIGHUP", "SIGUSR1"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, all_threads=True, chain=True)
        except (ValueError, OSError):
            pass

    def _on_signal(signum: int, frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        _log(f"FATAL: received {name} ({signum}); flushing logs and exiting")
        traceback.print_stack(frame)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(128 + signum)

    for sig_name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    def _on_unhandled(exc_type, exc, tb) -> None:
        _log(f"FATAL: unhandled {exc_type.__name__}: {exc}")
        traceback.print_exception(exc_type, exc, tb)
        sys.stdout.flush()
        sys.stderr.flush()

    sys.excepthook = _on_unhandled

    def _on_exit() -> None:
        try:
            _log(f"process exiting; final {_mem_str()}")
        except Exception:
            pass

    atexit.register(_on_exit)


_BQ_JOB_CACHE_DIR: Path | None = None


def _set_bq_job_cache_dir(p: Path) -> None:
    global _BQ_JOB_CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    _BQ_JOB_CACHE_DIR = p


def _bq_param_fingerprint(parameters: Sequence[bigquery.ArrayQueryParameter] | None) -> str:
    if not parameters:
        return ""
    parts: list[str] = []
    for p in parameters:
        name = getattr(p, "name", "") or ""
        ptype = getattr(p, "array_type", None) or getattr(p, "type_", None) or ""
        values = getattr(p, "values", None)
        if values is None:
            values = getattr(p, "value", None)
        try:
            values_repr = json.dumps(values, default=str, sort_keys=True)
        except (TypeError, ValueError):
            values_repr = repr(values)
        parts.append(f"{name}:{ptype}:{values_repr}")
    return "\n".join(parts)


def _bq_job_id_file(
    sql: str,
    parameters: Sequence[bigquery.ArrayQueryParameter] | None = None,
) -> Path | None:
    if _BQ_JOB_CACHE_DIR is None:
        return None
    blob = sql + "\n--params--\n" + _bq_param_fingerprint(parameters)
    key = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return _BQ_JOB_CACHE_DIR / f"{key}.txt"


def _query_job_result_table_ref(job: bigquery.QueryJob) -> bigquery.TableReference | None:
    """Return the anonymous result table for a completed query job, or None
    if BQ didn't materialize one (small/cached/stateless queries sometimes
    skip the temp table and only expose results inline).

    The official ``job.destination`` property is checked first; if missing,
    we refresh the job (destination metadata sometimes lags the initial
    response) and fall back to scraping the raw _properties dict.
    """
    if job.destination is not None:
        return job.destination
    try:
        job.reload()
    except Exception:
        pass
    if job.destination is not None:
        return job.destination
    resource = (job._properties or {}).get("statistics", {}).get("query", {}).get("destinationTable")
    if resource:
        return bigquery.TableReference.from_api_repr(resource)
    return None


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"expected bucket in gs:// URI, got {uri!r}")
    return bucket, prefix.rstrip("/")


def _gcs_export_target(sql: str, label: str) -> tuple[str, str, str]:
    workspace_bucket = os.environ.get("WORKSPACE_BUCKET")
    if not workspace_bucket:
        raise RuntimeError("WORKSPACE_BUCKET is not set; cannot export oversized BigQuery result to GCS")
    bucket, base_prefix = _parse_gs_uri(workspace_bucket)
    safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_") or "query"
    key = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
    run_id = f"{time.time_ns():x}"
    prefix_parts = [p for p in (base_prefix, "genterp", "etl", "bq_exports", f"{safe_label}-{key}-{run_id}") if p]
    prefix = "/".join(prefix_parts)
    return bucket, prefix, f"gs://{bucket}/{prefix}/part-*.parquet"


def _bq_string_literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _submit_or_reuse_job(
    client: bigquery.Client,
    sql: str,
    label: str,
    parameters: Sequence[bigquery.ArrayQueryParameter] | None = None,
) -> bigquery.QueryJob:
    """Resume a prior server-side BQ job for this exact SQL+params if still available."""
    job_id_file = _bq_job_id_file(sql, parameters)
    if job_id_file is not None and job_id_file.exists():
        prev_id = job_id_file.read_text().strip()
        try:
            job = client.get_job(prev_id)
            if job.state == "DONE" and job.error_result is None:
                _log(f"  bq reuse:   {label} job_id={prev_id} (server-cached result)")
                return job
            _log(f"  bq prior:   {label} state={job.state}; resubmitting")
        except Exception as exc:
            _log(f"  bq prior:   {label} lookup failed ({exc.__class__.__name__}); resubmitting")
    _log(f"  bq submit:  {label}")
    job_config = bigquery.QueryJobConfig(query_parameters=list(parameters or [])) if parameters else None
    job = client.query(sql, job_config=job_config) if job_config is not None else client.query(sql)
    if job_id_file is not None:
        job_id_file.write_text(job.job_id or "")
        _log(f"  bq job_id:  {label} {job.job_id} (recorded for resume)")
    return job


def _query_to_arrow(client: bigquery.Client, sql: str, label: str):
    """One-shot Arrow Table fetch — only safe for small results (concept lookup, censor, person)."""
    t0 = time.monotonic()
    job = _submit_or_reuse_job(client, sql, label)
    table = job.to_arrow(progress_bar_type="tqdm", create_bqstorage_client=False)
    _log(f"  bq done:   {label} rows={table.num_rows:,} in {time.monotonic() - t0:.1f}s")
    return table


def _job_progress_text(job: bigquery.QueryJob) -> str:
    """Best-available progress description from a running BQ query job.

    BigQuery's ``total_bytes_processed`` is only populated at completion, so polling
    that field looks like the query is doing nothing. The real-time signals are:

      * ``job.timeline`` — periodic snapshots of (pending, active, completed) work units.
        Computes a real progress percentage if at least one snapshot has landed.
      * ``job.slot_millis`` / ``job.query_plan`` — cumulative parallel-slot consumption
        and per-stage completion. Useful when ``timeline`` hasn't populated yet.
    """
    parts: list[str] = []
    timeline = list(getattr(job, "timeline", None) or ())
    if timeline:
        latest = timeline[-1]
        done = int(getattr(latest, "completed_units", 0) or 0)
        active = int(getattr(latest, "active_units", 0) or 0)
        pending = int(getattr(latest, "pending_units", 0) or 0)
        total = done + active + pending
        if total > 0:
            pct = 100.0 * done / total
            parts.append(f"progress={pct:.1f}% (done={done:,} active={active:,} pending={pending:,})")
    plan = list(getattr(job, "query_plan", None) or ())
    if plan:
        finished = sum(1 for s in plan if getattr(s, "end_time", None) is not None)
        parts.append(f"stages={finished}/{len(plan)}")
    slot_ms = getattr(job, "slot_millis", None) or 0
    if slot_ms:
        parts.append(f"slot_time={slot_ms/1000:.1f}s")
    return "  ".join(parts) if parts else "(no progress snapshot yet)"


def _wait_with_progress(job: bigquery.QueryJob, label: str, poll_s: float = 5.0) -> None:
    """Block until ``job`` is DONE, logging timeline/stage progress every poll_s seconds.

    Without this, a long aggregation query (coverage over 1B events, etc.) goes silent
    for minutes between submit and result — indistinguishable from a hang.
    """
    t0 = time.monotonic()
    last_log = 0.0
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
        _log(
            f"  bq running: {label} state={job.state} "
            f"{_job_progress_text(job)} elapsed={time.monotonic()-t0:.1f}s"
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
    """Submit (or reuse) a BQ aggregation, log live progress, download via Storage API.

    Hits the SQL+params job-id cache, so a Python crash *after* the BQ job completed
    but *before* the result was consumed costs nothing on retry — BQ keeps job
    results server-side for 24h. The outer JSON cache (``_cache_json``) handles
    persistence across longer windows.
    """
    t0 = time.monotonic()
    job = _submit_or_reuse_job(client, sql, label, parameters)
    _wait_with_progress(job, label)
    t_done = time.monotonic()
    _log(f"  bq query:   {label} completed in {t_done-t0:.1f}s; downloading via Storage API")

    destination = _query_job_result_table_ref(job)
    if destination is not None:
        # Large result with a materialized anonymous temp table — read via
        # Storage API to bypass the REST "Response too large" cap.
        bqs_client = bigquery_storage.BigQueryReadClient()
        try:
            row_iter = client.list_rows(destination, selected_fields=job.schema or [])
            batches = list(row_iter.to_arrow_iterable(bqstorage_client=bqs_client))
            table = pa.Table.from_batches(batches) if batches else pa.table({})
        finally:
            try:
                bqs_client.transport.close()
            except Exception:
                pass
    else:
        # Small/cached result — BQ didn't bother with a destination. Fetch
        # inline; this is safe because no-destination implies small enough
        # to fit in the REST response.
        _log(f"  bq inline:  {label} has no destination table; fetching results inline")
        table = job.to_arrow(create_bqstorage_client=False)
    _log(
        f"  bq done:    {label} rows={table.num_rows:,} download={time.monotonic()-t_done:.1f}s "
        f"total={time.monotonic()-t0:.1f}s"
    )
    return table


def _export_query_to_gcs_parquet(client: bigquery.Client, sql: str, label: str) -> tuple[str, str]:
    """Run a large query through BigQuery EXPORT DATA into the workspace bucket."""
    bucket, prefix, export_uri = _gcs_export_target(sql, label)
    export_sql = f"""
EXPORT DATA OPTIONS (
  uri = {_bq_string_literal(export_uri)},
  format = 'PARQUET',
  overwrite = true
) AS
{sql.rstrip().rstrip(";")}
"""
    _log(f"  bq export:  {label} uri={export_uri}")
    job = client.query(export_sql)
    _log(f"  bq job_id:  {label}_export {job.job_id}")
    _wait_with_progress(job, f"{label}_export")
    return bucket, prefix


def _compose_gcs_parquet_export(bucket_name: str, prefix: str, label: str, out_path: Path, tmp: Path) -> None:
    """Download exported GCS parquet shards and rewrite them as one local parquet."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = sorted(
        (blob for blob in storage_client.list_blobs(bucket, prefix=f"{prefix}/") if blob.name.endswith(".parquet")),
        key=lambda blob: blob.name,
    )
    if not blobs:
        raise RuntimeError(f"BigQuery export for {label} produced no parquet shards under gs://{bucket_name}/{prefix}/")

    writer: pq.ParquetWriter | None = None
    rows = 0
    batches = 0
    t0 = time.monotonic()
    t_last_log = t0
    shard_path = tmp.with_suffix(tmp.suffix + ".shard")
    try:
        for i, blob in enumerate(blobs, start=1):
            size = (blob.size or 0) / 1e9
            _log(f"  gcs shard: {label} {i}/{len(blobs)} {blob.name} ({size:.2f}GB)")
            blob.download_to_filename(shard_path)
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(batch_size=262_144):
                if writer is None:
                    writer = pq.ParquetWriter(tmp, batch.schema, compression="zstd")
                    _log(f"  compose:   {label} first batch rows={batch.num_rows:,}")
                writer.write_batch(batch)
                rows += batch.num_rows
                batches += 1
                now = time.monotonic()
                if now - t_last_log >= 2.0:
                    elapsed = now - t0
                    rate = rows / max(elapsed, 1e-6)
                    _log(f"  compose:   {label} rows={rows:,} batches={batches} rate={rate/1e6:.2f}M/s")
                    t_last_log = now
            try:
                shard_path.unlink()
            except FileNotFoundError:
                pass
    finally:
        if writer is not None:
            writer.close()
        try:
            shard_path.unlink()
        except FileNotFoundError:
            pass

    if writer is None:
        raise RuntimeError(f"BigQuery export for {label} contained no rows")
    tmp.replace(out_path)
    elapsed = time.monotonic() - t0
    rate = rows / max(elapsed, 1e-6)
    _log(f"streamed:   {out_path.name} rows={rows:,} batches={batches} in {elapsed:.1f}s ({rate/1e6:.2f}M rows/s)")

    deleted = 0
    for blob in blobs:
        try:
            blob.delete()
            deleted += 1
        except Exception as exc:
            _log(f"  gcs cleanup: {label} {blob.name} delete failed ({exc.__class__.__name__})")
    _log(f"  gcs cleanup: {label} deleted {deleted}/{len(blobs)} export shards")


def _stream_query_to_parquet(client: bigquery.Client, sql: str, label: str, out_path: Path) -> None:
    """Run a large BQ query and materialize it as one local Parquet file.

    BigQuery rejects this result as an anonymous query result with
    ``responseTooLarge``. AoU service accounts also cannot create scratch BQ
    datasets, so the large pull uses ``EXPORT DATA`` to write Parquet shards to
    ``WORKSPACE_BUCKET`` and then composes those shards locally.
    """
    if out_path.exists():
        _log(f"cache hit:  {out_path.name} ({out_path.stat().st_size/1e9:.2f}GB)")
        return
    _log(f"streaming:  {out_path.name} (BQ EXPORT DATA to GCS parquet)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    bucket, prefix = _export_query_to_gcs_parquet(client, sql, label)
    _compose_gcs_parquet_export(bucket, prefix, label, out_path, tmp)


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

# Tables whose only cid-bearing column is the *_concept_id and whose events
# carry no associated value. Observation is handled separately so we can emit
# both the question concept (observation_concept_id) AND the answer concept
# (value_as_concept_id) — see _observation_events_sql below.
NON_DRUG_TABLES = [
    ("condition_occurrence", "condition_concept_id", "condition_start_datetime"),
    ("procedure_occurrence", "procedure_concept_id", "procedure_datetime"),
    ("visit_occurrence", "visit_concept_id", "visit_start_datetime"),
    ("device_exposure", "device_concept_id", "device_exposure_start_datetime"),
]


def _cache_key(cdr: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", cdr).strip("_")
    suffix = f"_tiny{TINY_PERSON_MOD}x" if TINY else ""
    return f"{key}_threshold-{THRESHOLD}_values-v4{suffix}"


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
        size_gb = path.stat().st_size / 1e9
        _log(f"cache hit:  {path.name} ({size_gb:.2f}GB); reading parquet from disk")
        t0 = time.monotonic()
        data = pl.read_parquet(path)
        _log(f"cache read: {path.name} rows={data.height:,} columns={len(data.columns):,} in {time.monotonic() - t0:.1f}s")
        return data
    _log(f"cache miss: {path.name}; building dataframe")
    t_build = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    tmp = path.with_suffix(path.suffix + ".tmp")
    t_write = time.monotonic()
    data.write_parquet(tmp)
    tmp.replace(path)
    _log(
        f"cached:     {path.name} ({path.stat().st_size/1e9:.2f}GB) "
        f"rows={data.height:,} build={t_write - t_build:.1f}s write={time.monotonic() - t_write:.1f}s"
    )
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
        _log(f"cache hit:  {path.name} ({path.stat().st_size/1e9:.2f}GB)")
        return
    _log(f"sink:       {path.name} (streaming)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    t0 = time.monotonic()
    lf.sink_parquet(tmp, compression="zstd")
    tmp.replace(path)
    _log(f"sunk:       {path.name} ({path.stat().st_size/1e9:.2f}GB) in {time.monotonic() - t0:.1f}s")


def _sweep_stale_tmp(cache_dir: Path) -> None:
    """Delete ``*.tmp`` artifacts from previously crashed runs.

    The atomic-write pattern is ``write_path.with_suffix(.tmp)`` then ``rename``.
    If a process dies between the two, the .tmp survives — wasting disk and
    confusing the cache banner ("HIT events-…parquet.tmp 1.30GB"). At startup,
    no .tmp file is ever load-bearing, so they're always safe to remove.
    """
    if not cache_dir.exists():
        return
    swept = 0
    bytes_freed = 0
    for tmp in cache_dir.rglob("*.tmp"):
        try:
            bytes_freed += tmp.stat().st_size
            tmp.unlink()
            swept += 1
        except OSError:
            continue
    if swept:
        _log(f"swept {swept} stale .tmp file(s) from {cache_dir.name}, freed {bytes_freed/1e9:.2f}GB")


def _summarize_cache_state(out_dir: Path, cache_dir: Path) -> None:
    """Print a single block of HIT/MISS status for every artifact this run will produce.

    Anything tagged HIT below will short-circuit its build step at the cache layer.
    Anything MISS will be computed fresh; if a build step crashes, its BQ job-id
    (under ``cache_dir/bq_jobs/``) is still recorded so the next run reuses BQ's
    server-cached result rather than re-billing the same query.
    """
    candidates: list[tuple[str, Path]] = [
        ("publish", out_dir / "events.parquet"),
        ("publish", out_dir / "subjects.parquet"),
        ("publish", out_dir / "vocab.json"),
        ("publish", out_dir / "value_stats.json"),
        ("cache", cache_dir / "all_events.parquet"),
        ("cache", cache_dir / "persons.parquet"),
        ("cache", cache_dir / "censor.parquet"),
        ("cache", cache_dir / "concept_codes.json"),
    ]
    # Match keyed paths by stable prefix — keys aren't known until later steps run.
    keyed_prefixes = [
        "own_counts-",
        "string_value_counts-",
        "coverage_and_ancestors-",
        "collapsed_vocab-",
        "value_stats-",
        "events-",
        "subjects-",
    ]
    matched_by_prefix: dict[str, list[Path]] = {p: [] for p in keyed_prefixes}
    if cache_dir.exists():
        for entry in cache_dir.iterdir():
            for prefix in keyed_prefixes:
                if entry.name.startswith(prefix):
                    matched_by_prefix[prefix].append(entry)
                    break

    _log("cache state at run start:")
    n_hit = n_miss = 0
    for kind, path in candidates:
        if path.exists():
            n_hit += 1
            unit = "GB" if path.stat().st_size >= 1e9 else "MB"
            size = path.stat().st_size / (1e9 if unit == "GB" else 1e6)
            _log(f"  [{kind:7s}] HIT  {path.name} ({size:.2f}{unit})")
        else:
            n_miss += 1
            _log(f"  [{kind:7s}] MISS {path.name}")
    for prefix, entries in matched_by_prefix.items():
        if entries:
            for entry in sorted(entries):
                unit = "GB" if entry.stat().st_size >= 1e9 else "MB"
                size = entry.stat().st_size / (1e9 if unit == "GB" else 1e6)
                _log(f"  [keyed  ] HIT  {entry.name} ({size:.2f}{unit})")
                n_hit += 1
        else:
            _log(f"  [keyed  ] MISS {prefix}*")
            n_miss += 1

    job_cache = cache_dir / "bq_jobs"
    n_jobs = sum(1 for _ in job_cache.iterdir()) if job_cache.exists() else 0
    _log(f"cache state summary: {n_hit} hits, {n_miss} misses, {n_jobs} BQ job-ids on file for resume")


def _publish(source: Path, dest: Path) -> None:
    """Make ``dest`` an instant alias for ``source`` (hardlink if same filesystem, else copy).

    The final ETL outputs (events.parquet, subjects.parquet) live in ``out_dir`` but
    their authoritative copies are under ``cache_dir``. Hardlinking is O(1) and
    zero-disk; only falls back to a streaming copy if the two live on different
    filesystems (cross-device EXDEV).
    """
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
        _log(f"publish:    {dest.name} ← {source.name} (hardlink, 0B)")
    except OSError as exc:
        if getattr(exc, "errno", None) != 18:  # EXDEV
            raise
        _log(f"publish:    {dest.name} ← {source.name} (cross-device, streaming copy)")
        pl.scan_parquet(str(source)).sink_parquet(dest, compression="zstd")


def _cache_json(path: Path, build: Callable[[], object]) -> object:
    if path.exists():
        _log(f"cache hit:  {path.name} ({path.stat().st_size/1e6:.2f}MB); reading json from disk")
        payload = json.loads(path.read_text())
        _log(f"cache read: {path.name} ({_payload_units(payload)})")
        return payload
    _log(f"cache miss: {path.name}; building json payload")
    t0 = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    _write_json(path, data)
    _log(f"cached:     {path.name} ({path.stat().st_size/1e6:.2f}MB) in {time.monotonic() - t0:.1f}s")
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


def _observation_string_value_events_sql(cdr: str) -> str:
    """Emit synthetic events for observations whose answer is a value_as_string.

    AoU stores Zip3 / postal-region answers, free-text/other-specify responses,
    and similar string-valued observations in `value_as_string` — they have no
    value_as_concept_id, no value_as_number, and the question's own concept_id
    captures only "this field was filled in", not what was filled in.

    Each such observation contributes a synthesized code of the form

        VOCAB/CODE=str:<value_as_string>

    where VOCAB/CODE is the question concept. The cid column is NULL because
    there is no OMOP concept for the (question, answer) pair. Downstream vocab
    collapse cannot find ancestors for these synthetic codes (they're not in
    `concept_ancestor`), so they only survive collapse if their own subject
    count clears the threshold — at AoU scale, Zip3 buckets and frequently-
    chosen "other" responses do. Rare strings get dropped.
    """
    return f"""
    SELECT
      CAST(o.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(o.observation_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code, '=str:', o.value_as_string) AS code,
      CAST(NULL AS INT64) AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM `{cdr}.observation` o
    JOIN `{cdr}.concept` c ON c.concept_id = o.observation_concept_id
    WHERE o.observation_concept_id > 0
      AND o.observation_datetime IS NOT NULL
      AND o.value_as_string IS NOT NULL
      AND o.value_as_string != ''{_tiny_predicate("o.person_id")}
    """


def _observation_string_value_counts_sql(cdr: str) -> str:
    """Distinct-subject counts for each synthetic (question, value_as_string) code.

    Mirrors `_own_counts_sql` for cid-based concepts but operates on the
    synthesized code strings. Used to feed `collapse_vocabulary` so the
    threshold cut is applied consistently with the rest of the vocab.
    """
    return f"""
    SELECT
      CONCAT(c.vocabulary_id, '/', c.concept_code, '=str:', o.value_as_string) AS code,
      COUNT(DISTINCT o.person_id) AS n
    FROM `{cdr}.observation` o
    JOIN `{cdr}.concept` c ON c.concept_id = o.observation_concept_id
    WHERE o.observation_concept_id > 0
      AND o.value_as_string IS NOT NULL
      AND o.value_as_string != ''{_tiny_predicate("o.person_id")}
    GROUP BY code
    """


def _observation_events_sql(cdr: str) -> str:
    """Emit one row per observation token actually present in OMOP:

      1. The question concept (observation_concept_id) at the encounter time.
         For numeric surveys we carry value_as_number on this row so the value
         head sees it like any measurement.
      2. The categorical answer concept (value_as_concept_id) at the same time,
         if present. Without this branch the model can see "patient was asked
         about X" but never "they answered Y" — half the survey signal is lost.

    Both rows go through the same vocab collapse and atom encoding downstream;
    a row with the same (subject, time, atom) twice — e.g. an observation that
    coincidentally has answer == question — is deduped by the final-events SQL.
    """
    return f"""
    SELECT
      CAST(o.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(o.observation_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      o.observation_concept_id AS cid,
      CAST(o.value_as_number AS FLOAT64) AS value
    FROM `{cdr}.observation` o
    JOIN `{cdr}.concept` c ON c.concept_id = o.observation_concept_id
    WHERE o.observation_concept_id > 0 AND o.observation_datetime IS NOT NULL{_tiny_predicate("o.person_id")}
    UNION ALL
    SELECT
      CAST(o.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(o.observation_datetime) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      o.value_as_concept_id AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM `{cdr}.observation` o
    JOIN `{cdr}.concept` c ON c.concept_id = o.value_as_concept_id
    WHERE o.value_as_concept_id IS NOT NULL AND o.value_as_concept_id > 0
      AND o.observation_datetime IS NOT NULL{_tiny_predicate("o.person_id")}
    """


DEATH_CONCEPT_ID = 4306655  # OMOP standard concept for Death (SNOMED 419620001).


def _death_events_sql(cdr: str) -> str:
    """One synthetic Death event per person with a death record.

    Sources from ``aou_death``, picking the row with the highest
    ``primary_death_record`` flag and (as a tiebreaker) the earliest date —
    same precedence pgsEngine uses for death-date resolution. We emit the
    event at the resolved death date using the OMOP standard concept id for
    Death (4306655), so the concept passes through normal vocab collapse and
    inherits SNOMED ancestor closure rather than being a synthetic-code
    outlier. The downstream censor query (below) makes death_date a hard
    censor whenever it predates the observation-period end, so the model is
    not asked to predict any events past death.
    """
    death_concept = DEATH_CONCEPT_ID
    return f"""
    WITH ranked AS (
      SELECT
        person_id,
        COALESCE(death_date, DATE(death_datetime)) AS death_date,
        ROW_NUMBER() OVER (
          PARTITION BY person_id
          ORDER BY
            IFNULL(primary_death_record, FALSE) DESC,
            COALESCE(death_date, DATE(death_datetime)) ASC
        ) AS row_num
      FROM `{cdr}.aou_death`
      WHERE COALESCE(death_date, DATE(death_datetime)) IS NOT NULL{_tiny_predicate("person_id")}
    )
    SELECT
      CAST(r.person_id AS INT64) AS subject_id,
      UNIX_SECONDS(TIMESTAMP(r.death_date)) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      {death_concept} AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM ranked r
    JOIN `{cdr}.concept` c ON c.concept_id = {death_concept}
    WHERE r.row_num = 1
    """


def _demographics_events_sql(cdr: str) -> str:
    """Race and ethnicity as static-prefix events at birth-time.

    Currently AoU's race_concept_id and ethnicity_concept_id sit in the
    person table but never reach the model — sex is the only signal
    propagated to the static prefix. Emitting these at ``time = birth``
    (delta_days ≈ 0) lands them in ``static_atoms`` where the
    SetTransformer pools them alongside sex. Real OMOP concept ids → real
    ancestor closures via concept_ancestor → real vocab atoms.
    """
    birth_ts = (
        "TIMESTAMP(DATE(p.year_of_birth, "
        "COALESCE(p.month_of_birth, 1), COALESCE(p.day_of_birth, 1)))"
    )
    return f"""
    SELECT
      CAST(p.person_id AS INT64) AS subject_id,
      UNIX_SECONDS({birth_ts}) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      p.race_concept_id AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM `{cdr}.person` p
    JOIN `{cdr}.concept` c ON c.concept_id = p.race_concept_id
    WHERE p.race_concept_id > 0{_tiny_predicate("p.person_id")}
    UNION ALL
    SELECT
      CAST(p.person_id AS INT64) AS subject_id,
      UNIX_SECONDS({birth_ts}) AS time_seconds,
      CONCAT(c.vocabulary_id, '/', c.concept_code) AS code,
      p.ethnicity_concept_id AS cid,
      CAST(NULL AS FLOAT64) AS value
    FROM `{cdr}.person` p
    JOIN `{cdr}.concept` c ON c.concept_id = p.ethnicity_concept_id
    WHERE p.ethnicity_concept_id > 0{_tiny_predicate("p.person_id")}
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
      UNION ALL ({_observation_events_sql(cdr)})
      UNION ALL ({_observation_string_value_events_sql(cdr)})
      UNION ALL ({_death_events_sql(cdr)})
      UNION ALL ({_demographics_events_sql(cdr)})
    )
    SELECT subject_id, time_seconds, code, cid, value
    FROM events
    """


def _coverage_sql(cdr: str) -> str:
    """Coverage = approximate distinct subjects with any descendant of each ancestor.

    Computed with the BigQuery **HLL pre-aggregation pattern** instead of a naive
    ``APPROX_COUNT_DISTINCT`` over the JOIN. Why this matters:

      Naive path (``APPROX_COUNT_DISTINCT(person_id)`` over events ⨯ concept_ancestor):
        scan 1B+ events → JOIN ~10 ancestors per event → 10B intermediate rows →
        hash-distinct → group. The JOIN explosion dominates: minutes of slot time.

      Pre-aggregated path:
        1. ``HLL_COUNT.INIT(person_id, 12)`` per cid produces ONE sketch per concept
           (~60K rows after the events scan; each sketch ~1KB at precision 12, ~2% err).
        2. JOIN the 60K cid_sketches with concept_ancestor (~10 ancestors per cid →
           ~600K (sketch, ancestor) rows — three orders of magnitude smaller than
           materialising the per-event JOIN).
        3. ``HLL_COUNT.MERGE`` to combine sketches per ancestor → 100K results.

    Same accuracy (≤2% relative error, well below the granularity that matters at
    threshold=500). Typically 10–50× faster than ``APPROX_COUNT_DISTINCT``.
    """
    return f"""
    WITH events AS (
      {_non_drug_events_cte(cdr, with_time=False)}
      UNION ALL SELECT person_id, drug_concept_id AS cid FROM `{cdr}.drug_exposure` WHERE drug_concept_id > 0
      UNION ALL SELECT person_id, measurement_concept_id AS cid FROM `{cdr}.measurement` WHERE measurement_concept_id > 0
      UNION ALL SELECT person_id, observation_concept_id AS cid FROM `{cdr}.observation` WHERE observation_concept_id > 0
      UNION ALL SELECT person_id, value_as_concept_id AS cid FROM `{cdr}.observation` WHERE value_as_concept_id IS NOT NULL AND value_as_concept_id > 0
      UNION ALL SELECT person_id, {DEATH_CONCEPT_ID} AS cid FROM `{cdr}.aou_death` WHERE COALESCE(death_date, DATE(death_datetime)) IS NOT NULL
      UNION ALL SELECT person_id, race_concept_id AS cid FROM `{cdr}.person` WHERE race_concept_id > 0
      UNION ALL SELECT person_id, ethnicity_concept_id AS cid FROM `{cdr}.person` WHERE ethnicity_concept_id > 0
    ),
    cid_sketches AS (
      SELECT cid, HLL_COUNT.INIT(person_id, 12) AS sketch
      FROM events
      GROUP BY cid
    )
    SELECT ca.ancestor_concept_id AS aid,
           HLL_COUNT.MERGE(cid_sketches.sketch) AS n
    FROM cid_sketches
    JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = cid_sketches.cid
    GROUP BY ca.ancestor_concept_id
    """


def _ancestor_closure_sql(cdr: str) -> str:
    """Transitive ancestor closure for cohort descendants.

    OMOP's ``concept_ancestor`` table is already the full transitive closure: for any
    descendant D, the rows where ``descendant_concept_id=D`` enumerate *every*
    ancestor of D at any depth, with ``min_levels_of_separation`` recording the
    shortest path length. So a single filter on ``descendant_concept_id IN @cohort``
    yields every (descendant, ancestor, hops) edge ``collapse_vocabulary`` needs.

    The previous version UNIONed in "ancestors of cohort" as additional descendants
    and re-walked from there — redundant for transitive-closure tables, and the
    correlated ``IN (SELECT … FROM relevant)`` form was rejected by the BQ planner
    ("Correlated subqueries that reference other tables are not supported").
    """
    return f"""
    SELECT descendant_concept_id AS d,
           ancestor_concept_id   AS a,
           min_levels_of_separation AS hops
    FROM `{cdr}.concept_ancestor`
    WHERE descendant_concept_id IN UNNEST(@cohort)
      AND min_levels_of_separation > 0
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


_CONCEPT_CODES_CHUNK = 100_000  # BQ jobs.insert POST body cap is ~10 MB; ~100k INT64 ids fits safely.


def _concept_codes(client: bigquery.Client, cdr: str, ids: set[int]) -> dict[int, dict[str, str]]:
    """Resolve concept_id → metadata dict via BQ.

    Returns ``{cid: {"code": "VOCAB/CODE", "domain": ..., "class": ..., "name": ...}}``.
    The eval-side disease sweep filters candidates by ``domain == "Condition"``
    so we score against OMOP-normalized disease concepts rather than guessing
    via SNOMED hierarchy descent. ``concept_name`` makes the leaderboard
    human-readable.

    Chunks large id sets because BigQuery's jobs.insert endpoint rejects POST
    bodies > ~10 MB (HTTP 413). At AoU full-cohort scale we routinely see
    ≥460k cids in one batch; pre-chunking turns that into a few quick 100k-id
    queries.
    """
    if not ids:
        return {}
    sql = (
        f"SELECT concept_id, vocabulary_id, concept_code, domain_id, "
        f"concept_class_id, standard_concept, concept_name FROM `{cdr}.concept` "
        f"WHERE concept_id IN UNNEST(@ids)"
    )
    id_list = list(ids)
    result: dict[int, dict[str, str]] = {}
    n_chunks = (len(id_list) + _CONCEPT_CODES_CHUNK - 1) // _CONCEPT_CODES_CHUNK
    for chunk_idx in range(n_chunks):
        chunk = id_list[chunk_idx * _CONCEPT_CODES_CHUNK : (chunk_idx + 1) * _CONCEPT_CODES_CHUNK]
        label = f"concept_codes (chunk {chunk_idx + 1}/{n_chunks}, n={len(chunk):,})"
        table = _run_aggregation(
            client, sql, label,
            parameters=[bigquery.ArrayQueryParameter("ids", "INT64", chunk)],
        )
        cids = table.column("concept_id").to_pylist()
        vocabs = table.column("vocabulary_id").to_pylist()
        codes = table.column("concept_code").to_pylist()
        domains = table.column("domain_id").to_pylist()
        classes = table.column("concept_class_id").to_pylist()
        standards = table.column("standard_concept").to_pylist()
        names = table.column("concept_name").to_pylist()
        for c, v, k, dom, cls, std, name in zip(cids, vocabs, codes, domains, classes, standards, names, strict=True):
            result[int(c)] = {
                "code": f"{v}/{k}",
                "domain": str(dom) if dom is not None else "",
                "class": str(cls) if cls is not None else "",
                "standard_concept": str(std) if std is not None else "",
                "name": str(name) if name is not None else "",
            }
    return result


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


def _cached_concept_codes(
    client: Callable[[], bigquery.Client], cdr: str, ids: set[int], cache_dir: Path,
) -> dict[int, dict[str, str]]:
    """Cached concept metadata lookup keyed by concept_id.

    On-disk format is a JSON list of 6-tuples:
    ``[[cid, code, domain, class, standard_concept, name], ...]``.
    Older 5-tuple (no standard_concept) and 2-tuple (cid+code only) shapes are
    accepted on read; entries missing standard_concept get re-fetched from BQ
    so the OHDSI sweep filter (domain==Condition AND standard_concept=='S')
    always sees populated values.
    """
    path = cache_dir / "concept_codes.json"
    cached: dict[int, dict[str, str]] = {}
    if path.exists():
        raw: list[list[object]] = json.loads(path.read_text())
        for entry in raw:
            row = list(entry)
            cid = int(row[0])  # type: ignore[arg-type]
            if len(row) >= 6:
                cached[cid] = {
                    "code": str(row[1]),
                    "domain": str(row[2]) if row[2] is not None else "",
                    "class": str(row[3]) if row[3] is not None else "",
                    "standard_concept": str(row[4]) if row[4] is not None else "",
                    "name": str(row[5]) if row[5] is not None else "",
                }
            elif len(row) >= 5:
                # Legacy 5-tuple: no standard_concept column → trigger re-fetch.
                cached[cid] = {
                    "code": str(row[1]),
                    "domain": str(row[2]) if row[2] is not None else "",
                    "class": str(row[3]) if row[3] is not None else "",
                    "standard_concept": "",
                    "name": str(row[4]) if row[4] is not None else "",
                }
            else:
                # Legacy [cid, code] shape — leave metadata empty; will be
                # backfilled by the next BQ fetch if this cid is in `ids`.
                cached[cid] = {
                    "code": str(row[1]), "domain": "", "class": "",
                    "standard_concept": "", "name": "",
                }

    # A cid counts as "cached" only if we have all the metadata. Legacy entries
    # are missing domain or standard_concept → re-fetch.
    fully_cached = {
        cid for cid, meta in cached.items()
        if meta["domain"] and meta.get("standard_concept", "")
    }
    missing = ids - fully_cached
    if not missing:
        _log(f"concept code cache satisfied: cached={len(cached):,} requested={len(ids):,} missing=0")
        return cached

    _log(f"concept code cache incomplete: cached={len(cached):,} requested={len(ids):,} missing={len(missing):,}")
    cached.update(_concept_codes(client(), cdr, missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, [
        [cid, meta["code"], meta["domain"], meta["class"],
         meta.get("standard_concept", ""), meta["name"]]
        for cid, meta in sorted(cached.items())
    ])
    return cached


# ---------------------------------------------------------------------------
# OHDSI PhenotypeLibrary canonical disease list
# ---------------------------------------------------------------------------
# OMOP's ``concept_class_id`` collapses SNOMED's "(disorder)" and "(finding)"
# semantic tags into the single value "Clinical Finding", so it can't be used
# to separate diseases from symptoms. The functionally-equivalent OHDSI signal
# is the SNOMED hierarchy: every "(disorder)" concept descends from SNOMED
# 'Disease (disorder)' (concept_code 64572001). We combine that ancestor
# filter with the peer-curated OHDSI Phenotype Library's Reference cohorts
# to land at ~235 canonical disease phenotypes, each rooted at a single
# standard SNOMED concept_id.

_OHDSI_PL_URL = "https://github.com/OHDSI/PhenotypeLibrary/archive/refs/heads/main.zip"
_OHDSI_PL_DIRNAME = "PhenotypeLibrary-main"
_SNOMED_DISEASE_ROOT_CODE = "64572001"  # SNOMED 'Disease (disorder)'


def _ensure_ohdsi_phenotype_library(cache_dir: Path) -> Path:
    pl_dir = cache_dir / _OHDSI_PL_DIRNAME
    if (pl_dir / "inst" / "cohorts").exists():
        return pl_dir
    _log(f"downloading OHDSI PhenotypeLibrary → {pl_dir}")
    with urllib.request.urlopen(_OHDSI_PL_URL) as resp:
        data = resp.read()
    _log(f"  PL zip downloaded ({len(data) / 1e6:.1f} MB); extracting")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache_dir)
    if not (pl_dir / "inst" / "cohorts").exists():
        raise RuntimeError(f"OHDSI PL extract failed: {pl_dir}/inst/cohorts missing")
    return pl_dir


def _parse_ohdsi_truthy_flag(raw: object) -> int:
    """Accept any plausible truthy serialization in OHDSI's CSV columns.

    OHDSI's R-exported metadata has used 0/1, TRUE/FALSE (R style), Yes/No,
    and quoted variants across releases. Anything we don't recognize as
    truthy → 0.
    """
    s = str(raw if raw is not None else "").strip().strip('"').lower()
    if not s or s in {"na", "nan", "null", "none"}:
        return 0
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    try:
        return 1 if int(float(s)) == 1 else 0
    except ValueError:
        return 0


def _parse_ohdsi_reference_condition_cohorts(
    pl_dir: Path,
) -> tuple[dict[int, dict[str, object]], dict[str, int]]:
    """Walk PL → {include_root_cid: cohort meta}, plus a skip-reason counter.

    Filters applied here:
      (a) ``isReferenceCohort == 1``        — canonical OHDSI variants only.
      (b) ConditionOccurrence primary       — diagnosis-driven phenotypes.
      (c) Single include, zero excludes     — 1:1 cohort↔SNOMED root mapping.

    Filter (d) — root must descend from SNOMED 'Disease (disorder)' — runs
    later because it needs a BigQuery concept_ancestor lookup.
    """
    cohorts_csv = pl_dir / "inst" / "Cohorts.csv"
    json_dir = pl_dir / "inst" / "cohorts"

    meta_by_id: dict[int, dict[str, object]] = {}
    csv_columns: list[str] = []
    first_row_sample: dict[str, str] | None = None
    n_csv_rows = 0
    # encoding='utf-8-sig' strips the UTF-8 BOM the OHDSI PL ships at byte 0;
    # without it csv.DictReader treats the BOM as part of the first column
    # name ('﻿"cohortId"') and every row.get("cohortId") returns ''.
    with cohorts_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        csv_columns = list(reader.fieldnames or [])
        for row in reader:
            n_csv_rows += 1
            if first_row_sample is None:
                first_row_sample = {k: str(v)[:40] for k, v in row.items() if v}
            try:
                cohort_id = int(str(row.get("cohortId", "")).strip())
            except (TypeError, ValueError):
                continue
            is_ref = _parse_ohdsi_truthy_flag(row.get("isReferenceCohort"))
            meta_by_id[cohort_id] = {
                "cohort_name": row.get("cohortName", ""),
                "status": row.get("status", ""),
                "is_reference": is_ref,
            }

    ref_ids = sorted(cid for cid, m in meta_by_id.items() if m["is_reference"] == 1)
    if not ref_ids:
        _log(
            f"OHDSI PL Cohorts.csv parsed {n_csv_rows} rows but found 0 "
            f"isReferenceCohort==1 entries. columns={csv_columns} "
            f"sample_row={first_row_sample}"
        )

    root_to_cohort: dict[int, dict[str, object]] = {}
    skipped = {"missing_json": 0, "parse": 0, "non_condition_primary": 0,
               "multi_include_or_exclude": 0}

    for cohort_id in ref_ids:
        path = json_dir / f"{cohort_id}.json"
        if not path.exists():
            skipped["missing_json"] += 1
            continue
        try:
            raw = path.read_bytes()
            try:
                j = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                j = json.loads(raw.decode("latin-1"))
        except Exception:
            skipped["parse"] += 1
            continue

        codeset_id = None
        for crit in j.get("PrimaryCriteria", {}).get("CriteriaList", []):
            if "ConditionOccurrence" in crit:
                codeset_id = crit["ConditionOccurrence"].get("CodesetId")
                break
        if codeset_id is None:
            skipped["non_condition_primary"] += 1
            continue

        cs = next((c for c in j.get("ConceptSets", []) if c.get("id") == codeset_id), None)
        if cs is None:
            skipped["parse"] += 1
            continue
        items = cs.get("expression", {}).get("items", [])
        incs = [it for it in items if not it.get("isExcluded")]
        excs = [it for it in items if it.get("isExcluded")]
        if len(incs) != 1 or len(excs) != 0:
            skipped["multi_include_or_exclude"] += 1
            continue

        root_cid = int(incs[0]["concept"]["CONCEPT_ID"])
        meta = meta_by_id[cohort_id]
        if root_cid not in root_to_cohort or cohort_id < int(root_to_cohort[root_cid]["cohort_id"]):
            root_to_cohort[root_cid] = {
                "cohort_id": cohort_id,
                "cohort_name": meta["cohort_name"],
                "status": meta["status"],
            }

    return root_to_cohort, skipped


def _resolve_snomed_disease_root_cid(client: bigquery.Client, cdr: str) -> int:
    sql = f"""
        SELECT concept_id FROM `{cdr}.concept`
        WHERE vocabulary_id = 'SNOMED'
          AND concept_code = '{_SNOMED_DISEASE_ROOT_CODE}'
          AND standard_concept = 'S'
    """
    table = _run_aggregation(client, sql, "ohdsi: resolve SNOMED 'Disease (disorder)' root")
    cids = table.column("concept_id").to_pylist()
    if not cids:
        raise RuntimeError(
            f"SNOMED {_SNOMED_DISEASE_ROOT_CODE} not found in {cdr}.concept"
        )
    return int(cids[0])


def _filter_disease_descendants(
    client: bigquery.Client, cdr: str, roots: list[int], disease_root_cid: int,
) -> set[int]:
    if not roots:
        return set()
    sql = f"""
        WITH roots AS (SELECT concept_id FROM UNNEST(@roots) AS concept_id),
             dis   AS (SELECT descendant_concept_id AS concept_id
                       FROM `{cdr}.concept_ancestor`
                       WHERE ancestor_concept_id = {disease_root_cid})
        SELECT r.concept_id
        FROM roots r INNER JOIN dis USING (concept_id)
    """
    table = _run_aggregation(
        client, sql, "ohdsi: filter roots to SNOMED Disease descendants",
        parameters=[bigquery.ArrayQueryParameter("roots", "INT64", roots)],
    )
    if "concept_id" not in table.schema.names:
        return set()
    return {int(c) for c in table.column("concept_id").to_pylist()}


def _lookup_disease_concept_codes(
    client: bigquery.Client, cdr: str, cids: list[int],
) -> dict[int, tuple[str, str]]:
    if not cids:
        return {}
    sql = f"""
        SELECT concept_id, vocabulary_id, concept_code, concept_name
        FROM `{cdr}.concept`
        WHERE concept_id IN UNNEST(@cids)
    """
    table = _run_aggregation(
        client, sql, "ohdsi: lookup disease concept codes + names",
        parameters=[bigquery.ArrayQueryParameter("cids", "INT64", cids)],
    )
    if "concept_id" not in table.schema.names:
        return {}
    out: dict[int, tuple[str, str]] = {}
    for cid, vocab, code, name in zip(
        table.column("concept_id").to_pylist(),
        table.column("vocabulary_id").to_pylist(),
        table.column("concept_code").to_pylist(),
        table.column("concept_name").to_pylist(),
    ):
        out[int(cid)] = (f"{vocab}/{code}", str(name) if name else "")
    return out


def _cached_ohdsi_disease_phenotypes(
    client_factory: Callable[[], bigquery.Client], cdr: str, cache_dir: Path,
) -> list[dict[str, object]]:
    """Build (or load cached) OHDSI canonical disease phenotype list.

    Output JSON shape: list of
        {"concept_id", "concept_code", "concept_name",
         "ohdsi_cohort_id", "ohdsi_cohort_name", "status"}
    """
    path = cache_dir / "ohdsi_disease_phenotypes.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if isinstance(cached, list) and cached:
            _log(f"OHDSI disease phenotype cache hit: {path.name} ({len(cached):,} phenotypes)")
            return cached

    _log("OHDSI disease phenotype cache cold — building from PhenotypeLibrary + BigQuery")
    pl_dir = _ensure_ohdsi_phenotype_library(cache_dir)
    root_to_cohort, skipped = _parse_ohdsi_reference_condition_cohorts(pl_dir)
    _log(
        f"OHDSI PL Reference cohorts → single-include condition cohorts: "
        f"kept={len(root_to_cohort):,}  "
        f"skipped: missing_json={skipped['missing_json']:,} "
        f"parse_err={skipped['parse']:,} "
        f"non_condition_primary={skipped['non_condition_primary']:,} "
        f"multi_include_or_exclude={skipped['multi_include_or_exclude']:,}"
    )
    if not root_to_cohort:
        raise RuntimeError(
            "OHDSI PL parse yielded 0 Reference condition cohorts — see preceding "
            f"diagnostic. Check Cohorts.csv schema at {pl_dir}/inst/Cohorts.csv."
        )

    client = client_factory()
    disease_root_cid = _resolve_snomed_disease_root_cid(client, cdr)
    roots_all = sorted(root_to_cohort.keys())
    disease_roots = _filter_disease_descendants(client, cdr, roots_all, disease_root_cid)
    dropped = len(roots_all) - len(disease_roots)
    _log(
        f"SNOMED 'Disease (disorder)' ancestor filter: kept {len(disease_roots):,}, "
        f"dropped {dropped:,} symptom/finding-rooted Reference cohorts"
    )
    final_cids = [r for r in roots_all if r in disease_roots]
    codes_names = _lookup_disease_concept_codes(client, cdr, final_cids)

    out: list[dict[str, object]] = []
    for cid in final_cids:
        meta = root_to_cohort[cid]
        code, name = codes_names.get(cid, ("", ""))
        if not code:
            continue
        out.append({
            "concept_id": cid,
            "concept_code": code,
            "concept_name": name or str(meta["cohort_name"]),
            "ohdsi_cohort_id": meta["cohort_id"],
            "ohdsi_cohort_name": meta["cohort_name"],
            "status": meta["status"],
        })
    out.sort(key=lambda r: str(r["concept_name"]).lower())
    _write_json(path, out)
    _log(f"OHDSI disease phenotype list written: {len(out):,} phenotypes → {path.name}")
    return out


def _own_counts_sql(cdr: str) -> str:
    """Distinct subjects per source concept_id, computed at BigQuery.

    Computing this locally on 1B rows tried to keep a hash set of subject_ids per
    cid (n_unique aggregation, 104K groups) and got SIGKILL'd by the OOM killer
    even with the polars streaming engine. BQ has unlimited memory for this kind
    of distinct count — and the result is tiny (one row per cid, ~100K rows).

    The UNION must mirror every source in ``_all_events_sql`` exactly, so the
    threshold filter in ``collapse_vocabulary`` sees the same per-concept
    patient counts that the events stream will materialize. The death and
    demographics sources are included here so their concept ids survive the
    threshold cut and end up as vocab atoms.
    """
    death_concept = DEATH_CONCEPT_ID
    return f"""
    WITH events AS (
      {_non_drug_events_cte(cdr, with_time=False)}
      UNION ALL SELECT person_id, drug_concept_id AS cid FROM `{cdr}.drug_exposure` WHERE drug_concept_id > 0
      UNION ALL SELECT person_id, measurement_concept_id AS cid FROM `{cdr}.measurement` WHERE measurement_concept_id > 0
      UNION ALL SELECT person_id, observation_concept_id AS cid FROM `{cdr}.observation` WHERE observation_concept_id > 0
      UNION ALL SELECT person_id, value_as_concept_id AS cid FROM `{cdr}.observation` WHERE value_as_concept_id IS NOT NULL AND value_as_concept_id > 0
      UNION ALL SELECT person_id, {death_concept} AS cid FROM `{cdr}.aou_death` WHERE COALESCE(death_date, DATE(death_datetime)) IS NOT NULL
      UNION ALL SELECT person_id, race_concept_id AS cid FROM `{cdr}.person` WHERE race_concept_id > 0
      UNION ALL SELECT person_id, ethnicity_concept_id AS cid FROM `{cdr}.person` WHERE ethnicity_concept_id > 0
    )
    SELECT cid, COUNT(DISTINCT person_id) AS n
    FROM events
    GROUP BY cid
    """


def _cached_own_counts(
    client: bigquery.Client,
    cdr: str,
    cache_dir: Path,
    source_key: str,
) -> dict[int, int]:
    def build() -> list[list[int]]:
        table = _run_aggregation(client, _own_counts_sql(cdr), "own_counts")
        cids = table.column("cid").to_pylist()
        ns = table.column("n").to_pylist()
        return [[int(c), int(n)] for c, n in zip(cids, ns, strict=True)]

    payload = _cache_json(cache_dir / f"own_counts-{source_key}.json", build)
    return {int(cid): int(n) for cid, n in payload}


def _cached_string_value_counts(
    client: bigquery.Client,
    cdr: str,
    cache_dir: Path,
    source_key: str,
) -> dict[str, int]:
    """Distinct-subject counts for synthetic `VOCAB/CODE=str:VALUE` codes.

    Cached alongside own_counts (keyed by source_key so it busts whenever the
    upstream BQ pull changes). Returns a {code_string: n_subjects} map merged
    into own_by_code before vocab collapse.
    """
    def build() -> list[list[object]]:
        table = _run_aggregation(
            client, _observation_string_value_counts_sql(cdr), "observation_string_value_counts"
        )
        codes = table.column("code").to_pylist()
        ns = table.column("n").to_pylist()
        return [[str(code), int(n)] for code, n in zip(codes, ns, strict=True) if code]

    payload = _cache_json(cache_dir / f"string_value_counts-{source_key}.json", build)
    return {str(code): int(n) for code, n in payload}


def _duckdb_connection(cache_dir: Path, label: str) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with bounded RAM and a real disk spill directory.

    Polars' streaming engine silently failed to spill on the 1B-row sort and the
    100K-group mean/std aggregation, sending RSS to 10GB+ before kernel OOM-killed
    the process. DuckDB respects ``PRAGMA memory_limit`` honestly: when the working
    set exceeds the cap it writes hash-table partitions / sort runs to ``temp_directory``
    on disk. The same applies to the GROUP BY in value_stats.

    ``preserve_insertion_order=false`` is essential here: with it on, DuckDB keeps
    extra row-order bookkeeping that roughly doubles sort working memory. We do
    *not* care about insertion order — every output of these passes is either
    ungrouped (value_stats) or has its own ORDER BY that defines the output order.
    """
    spill_dir = cache_dir / "duckdb_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    available = psutil.virtual_memory().available
    # Reserve headroom so other processes (BQ client, parquet writers) don't get squeezed.
    cap_gb = max(4, int(available * 0.5 / 1e9))
    con.execute(f"PRAGMA memory_limit='{cap_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{spill_dir.as_posix()}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA threads={os.cpu_count() or 4}")
    _log(
        f"duckdb[{label}] opened memory_limit={cap_gb}GB threads={os.cpu_count() or 4} "
        f"spill_dir={spill_dir.name}"
    )
    return con


def _cached_value_stats(
    all_events_path: Path,
    atom_idx: dict[str, int],
    cache_dir: Path,
    source_key: str,
    vocab_key: str,
) -> dict[str, dict[str, float]]:
    def build() -> dict[str, dict[str, float]]:
        con = _duckdb_connection(cache_dir, "value_stats")
        try:
            t0 = time.monotonic()
            rows = con.execute(
                f"""
                SELECT code,
                       AVG(value)         AS mu,
                       STDDEV_SAMP(value) AS sigma,
                       COUNT(*)           AS n
                FROM read_parquet('{all_events_path.as_posix()}')
                WHERE value IS NOT NULL
                  AND value = value                 -- NaN check (NaN != NaN)
                  AND ABS(value) < 1e308            -- ±inf check
                GROUP BY code
                HAVING COUNT(*) >= {THRESHOLD}
                """
            ).fetchall()
            _log(f"duckdb[value_stats] aggregated rows={len(rows):,} in {time.monotonic()-t0:.1f}s")
        finally:
            con.close()
        return {
            row[0]: {"mu": float(row[1]), "sigma": float(row[2] or 1.0)}
            for row in rows
            if row[0] in atom_idx
        }

    payload = _cache_json(cache_dir / f"value_stats-{source_key}-{vocab_key}.json", build)
    return {str(code): {"mu": float(stats["mu"]), "sigma": float(stats["sigma"])} for code, stats in payload.items()}


FINAL_EVENTS_SHARDS = 128


def _build_final_events_with_duckdb(
    all_events_path: Path,
    atom_idx: dict[str, int],
    final_path: Path,
    cache_dir: Path,
) -> None:
    """Filter + atom-encode + sort to final parquet via DuckDB *chunked* external sort.

    A single global ORDER BY on the full 1B-row input asked DuckDB to spill ~30GB
    of sort runs to a 24GB temp dir → OutOfMemoryException at "failed to offload
    data block (24.1GiB/24.1GiB used)". The workspace disk is the limit, not RAM.

    Chunked plan, three passes, each bounded by ``1/FINAL_EVENTS_SHARDS`` of the
    working set:

      1. ONE scan over ``all_events.parquet`` joining ``atom_map`` and writing
         a hive-partitioned dataset to disk, partitioned by
         ``CAST(subject_id % K AS INTEGER)``. No ORDER BY — single linear pass,
         minimal spill. Output is K parquet directories.

      2. For each shard k = 0..K-1, run an ORDER BY (subject_id, time_seconds)
         over that shard alone. Each shard holds ~1/K of the data, so sort
         working set is ~30GB / K — fits in memory at K=128.

      3. Append each sorted shard to the final parquet via a pyarrow
         ``ParquetWriter``. No sort, no JOIN — streaming row-batch copy.

    Subjects with ``id % K == k`` are *atomically* assigned to shard k, so
    every subject's events live in exactly one shard. The concatenation order
    is shard-by-shard (subjects bucketed by ``id % K``, then by id within
    bucket). Downstream offsets in ``subjects.parquet`` only care that each
    subject's events are *contiguous* in the file, which this guarantees.

    Cleanup is per-shard: each raw shard is deleted as soon as its sorted
    counterpart lands; each sorted shard is deleted as soon as it's been
    appended to the final file. Peak disk is bounded by the raw partition
    output (Pass 1) plus one shard's worth of sort spill.
    """
    if final_path.exists():
        _log(f"cache hit:  {final_path.name} ({final_path.stat().st_size/1e9:.2f}GB)")
        return

    import shutil

    K = FINAL_EVENTS_SHARDS
    work_dir = cache_dir / "duckdb_final_work"
    raw_dir = work_dir / "raw_shards"
    sorted_dir = work_dir / "sorted_shards"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    sorted_dir.mkdir(parents=True, exist_ok=True)

    con = _duckdb_connection(cache_dir, "final_events")
    try:
        codes = pa.array(list(atom_idx.keys()), type=pa.string())
        atoms = pa.array(list(atom_idx.values()), type=pa.uint32())
        atom_map = pa.table({"code": codes, "atom": atoms})
        con.register("atom_map", atom_map)
        _log(f"duckdb[final_events] atom_map registered: {len(atom_idx):,} codes; chunks={K}")

        # Pass 1: single scan + JOIN + partition_by (no sort).
        _log(f"duckdb[final_events] Pass 1/3: scan + atom-encode + partition into {K} shards")
        t0 = time.monotonic()
        con.execute(
            f"""
            COPY (
                SELECT e.subject_id,
                       e.time_seconds,
                       am.atom AS atom,
                       e.value,
                       CAST(e.subject_id % {K} AS INTEGER) AS shard
                FROM read_parquet('{all_events_path.as_posix()}') e
                JOIN atom_map am USING (code)
            ) TO '{raw_dir.as_posix()}'
            (FORMAT PARQUET, PARTITION_BY (shard), COMPRESSION ZSTD)
            """
        )
        _log(f"duckdb[final_events] Pass 1 done in {time.monotonic()-t0:.1f}s")

        # Pass 2 + Pass 3 fused: sort each shard, append to final, drop intermediates.
        tmp = final_path.with_suffix(final_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        writer: pq.ParquetWriter | None = None
        total_rows = 0
        t_phase = time.monotonic()
        for k in range(K):
            shard_subdir = raw_dir / f"shard={k}"
            if not shard_subdir.exists():
                continue
            shard_files = sorted(shard_subdir.glob("*.parquet"))
            if not shard_files:
                continue
            sorted_path = sorted_dir / f"sorted_{k:04d}.parquet"
            shard_list_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in shard_files) + "]"
            t_sort = time.monotonic()
            # Collapse exact duplicates: OMOP routinely re-asserts the same
            # condition/observation across visits, and the drug_exposure ->
            # ingredient fan-out can hit the same ingredient twice for a single
            # combo drug row. After atom encoding those all share
            # (subject_id, time_seconds, atom). MAX(value) keeps the numeric
            # value if any row carried one (NULLs are ignored).
            con.execute(
                f"""
                COPY (
                    SELECT subject_id, time_seconds, atom, MAX(value) AS value
                    FROM read_parquet({shard_list_sql})
                    GROUP BY subject_id, time_seconds, atom
                    ORDER BY subject_id, time_seconds
                ) TO '{sorted_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            # Append sorted shard rows to the final file.
            pf = pq.ParquetFile(sorted_path)
            rows_this_shard = 0
            for batch in pf.iter_batches(batch_size=1_000_000):
                if writer is None:
                    writer = pq.ParquetWriter(tmp, batch.schema, compression="zstd")
                writer.write_batch(batch)
                rows_this_shard += batch.num_rows
            total_rows += rows_this_shard
            shutil.rmtree(shard_subdir, ignore_errors=True)
            sorted_path.unlink()
            if (k + 1) % max(1, K // 16) == 0 or k == K - 1:
                _log(
                    f"duckdb[final_events] Pass 2-3: shard {k+1}/{K} sorted+appended "
                    f"(+{rows_this_shard:,} rows, total={total_rows:,}) in {time.monotonic()-t_sort:.1f}s"
                )
        if writer is not None:
            writer.close()
        tmp.replace(final_path)
        shutil.rmtree(work_dir, ignore_errors=True)
        _log(
            f"duckdb[final_events] done: {final_path.name} "
            f"({final_path.stat().st_size/1e9:.2f}GB) rows={total_rows:,} "
            f"in {time.monotonic()-t_phase:.1f}s (Pass 2-3)"
        )
    finally:
        con.close()


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

    _install_crash_diagnostics()
    _log_version_banner()
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
    _sweep_stale_tmp(cache_dir)
    _summarize_cache_state(out_dir, cache_dir)
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

    _WORK.start_unit(
        "count distinct subjects per source concept",
        "COUNT(DISTINCT person_id) GROUP BY cid at BigQuery — local polars n_unique on 1B rows OOM'd",
    )
    own_by_cid = _cached_own_counts(client(), cdr, cache_dir, source_key)
    # Synthetic codes built from (question_concept, value_as_string) — no cid,
    # so they need their own subject count and live alongside own_by_cid in the
    # code-string keyed view used by collapse_vocabulary below.
    string_value_counts = _cached_string_value_counts(client(), cdr, cache_dir, source_key)
    _WORK.finish_unit(
        "count distinct subjects per source concept",
        f"unique concept_ids={len(own_by_cid):,} string_value_codes={len(string_value_counts):,}",
    )

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
        meta_of = _cached_concept_codes(client, cdr, all_ids, cache_dir)
        code_of = {cid: meta["code"] for cid, meta in meta_of.items()}
        own_by_code = {code_of[c]: n for c, n in own_by_cid.items() if c in code_of}
        # Synthetic string-value codes get added with no coverage/ancestors —
        # collapse_vocabulary treats them as standalone (ancestors.get(c, {})
        # returns empty, so they survive iff own_count >= threshold).
        for code, n in string_value_counts.items():
            own_by_code[code] = own_by_code.get(code, 0) + int(n)
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

    # Refresh concept_codes metadata if it's still in the legacy 2-tuple
    # format. The eval-side OHDSI Condition sweep needs ``domain_id`` per
    # concept; the older cache shape only has ``[cid, code]``. When the
    # collapsed_vocab cache hits (above) the concept-codes fetcher never
    # runs, so on long-running workspaces the metadata can lag the code
    # schema. Force a check here that only hits BQ for cids missing
    # metadata (no-op if everything is already 5-tuple).
    _WORK.start_unit(
        "refresh concept_codes metadata (legacy → 5-tuple)",
        "ensures eval-side OHDSI sweep can filter by concept.domain_id",
    )
    concept_codes_path = cache_dir / "concept_codes.json"
    if concept_codes_path.exists():
        existing = json.loads(concept_codes_path.read_text())
        existing_cids = {int(row[0]) for row in existing}
        # Triggers a BQ refetch for any cid whose cached entry lacks domain.
        meta_after = _cached_concept_codes(client, cdr, existing_cids, cache_dir)
        with_domain = sum(1 for m in meta_after.values() if m.get("domain"))
        _WORK.finish_unit(
            "refresh concept_codes metadata (legacy → 5-tuple)",
            f"cids_total={len(meta_after):,} with_domain={with_domain:,}",
        )
    else:
        _WORK.finish_unit(
            "refresh concept_codes metadata (legacy → 5-tuple)",
            "skipped — no concept_codes.json on disk yet",
        )

    _WORK.start_unit(
        "build OHDSI canonical disease phenotype list",
        "OHDSI PhenotypeLibrary Reference cohorts → ConditionOccurrence primary "
        "→ single-include/zero-exclude → descend from SNOMED 'Disease (disorder)'",
    )
    ohdsi_diseases = _cached_ohdsi_disease_phenotypes(client, cdr, cache_dir)
    _WORK.finish_unit(
        "build OHDSI canonical disease phenotype list",
        f"phenotypes={len(ohdsi_diseases):,}",
    )

    _WORK.start_unit(
        "compute per-atom value stats",
        "DuckDB mean/stddev/count GROUP BY code with HAVING n>=THRESHOLD; spills to disk under PRAGMA memory_limit",
    )
    vocab_key = _stable_json_fingerprint(atom_idx)
    stats = _cached_value_stats(all_events_path, atom_idx, cache_dir, source_key, vocab_key)
    _write_json(out_dir / "value_stats.json", stats)
    _WORK.finish_unit("compute per-atom value stats", f"magnitude-bearing atoms={len(stats):,} vocab_key={vocab_key}")

    # Free polars LazyFrame state before the big DuckDB op so we don't carry
    # polars working memory through the sort. The downstream offsets step opens
    # a fresh scan on the final parquet anyway.
    del events_lf
    import gc
    gc.collect()

    _WORK.start_unit(
        "filter and sort final events",
        "DuckDB: JOIN atom_map for code→atom encode + ORDER BY (subject_id, time_seconds) with on-disk external sort",
    )
    final_events_path = cache_dir / f"events-{source_key}-{vocab_key}-atom-v1.parquet"
    _build_final_events_with_duckdb(all_events_path, atom_idx, final_events_path, cache_dir)
    _publish(final_events_path, out_dir / "events.parquet")
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

    _WORK.start_unit(
        "pull censor (observation-period end + aou_death)",
        "censor_seconds = LEAST(observation_period_end, death_date) — death is a hard censor",
    )
    censor = _cache_parquet(
        cache_dir / "censor.parquet",
        lambda: _arrow_to_polars(
            _query_to_arrow(client(), f"""
                WITH obs_end AS (
                  SELECT person_id,
                         MAX(observation_period_end_date) AS observation_end
                  FROM `{cdr}.observation_period`
                  WHERE TRUE{_tiny_predicate("person_id")}
                  GROUP BY person_id
                ),
                death AS (
                  SELECT person_id,
                         MIN(COALESCE(death_date, DATE(death_datetime))) AS death_date
                  FROM `{cdr}.aou_death`
                  WHERE COALESCE(death_date, DATE(death_datetime)) IS NOT NULL
                    {_tiny_predicate("person_id")}
                  GROUP BY person_id
                )
                SELECT
                  CAST(o.person_id AS INT64) AS subject_id,
                  UNIX_SECONDS(TIMESTAMP(LEAST(
                    o.observation_end,
                    COALESCE(d.death_date, o.observation_end)
                  ))) AS censor_seconds
                FROM obs_end o
                LEFT JOIN death d ON d.person_id = o.person_id
            """, "censor"),
            "censor",
        ),
    )
    _WORK.finish_unit("pull censor (observation-period end + aou_death)", f"rows={censor.height:,}")

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
    subjects_path = cache_dir / f"subjects-{source_key}-{vocab_key}-split{TEST_SPLIT_PERCENT}.parquet"
    subjects = _cache_parquet(subjects_path, lambda: subjects)
    _publish(subjects_path, out_dir / "subjects.parquet")
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

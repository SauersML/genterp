#!/usr/bin/env bash
set -euo pipefail

RUN_TOTAL_UNITS=10
RUN_COMPLETED_UNITS=0
RUN_STARTED_AT="$(date +%s)"

log_run() {
  local message="$1"
  local now
  now="$(date +%s)"
  local elapsed=$((now - RUN_STARTED_AT))
  printf '[run.sh t+%6ss units=%s/%s] %s\n' "$elapsed" "$RUN_COMPLETED_UNITS" "$RUN_TOTAL_UNITS" "$message" >&2
}

finish_run_unit() {
  RUN_COMPLETED_UNITS=$((RUN_COMPLETED_UNITS + 1))
  log_run "DONE  $1"
}

# Force unbuffered stdout/stderr so logs stream in real time.
log_run "START enable unbuffered Python output"
export PYTHONUNBUFFERED=1
# Don't write .pyc files: a stale __pycache__ entry can shadow an updated .py
# (e.g. after git pull) if the disk mtime hasn't propagated. Belt-and-suspenders
# against "I pushed but the box is running old code" mysteries.
export PYTHONDONTWRITEBYTECODE=1
finish_run_unit "enable unbuffered Python output"

log_run "START resolve repository directory"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
finish_run_unit "resolve repository directory: $SCRIPT_DIR"

# Reap any lingering Python processes from a prior aborted run so we don't end
# up with two ETL processes writing the same parquet, two trainers fighting
# over the same GPU, or a zombie holding the events file mmap open. Match by
# module path so we don't accidentally pkill unrelated python processes.
log_run "START reap leftover genterp processes from prior runs"
for pattern in "python.*scripts\.aou_etl" "python.*genterp\.train" "python.*genterp\.clt_train"; do
  if pkill -f "$pattern" 2>/dev/null; then
    echo "[run.sh] killed leftover process matching: $pattern" >&2
    sleep 1
    pkill -9 -f "$pattern" 2>/dev/null || true
  fi
done
finish_run_unit "reap leftover genterp processes from prior runs"

# Wipe any __pycache__ from a previous code version. Python invalidates .pyc by
# .py mtime, but on shared/network filesystems mtime can lag — wiping the cache
# is cheap and guarantees the next interpreter compiles from the .py we just
# pulled. Combined with PYTHONDONTWRITEBYTECODE above, no new caches are created.
log_run "START wipe stale bytecode caches under $SCRIPT_DIR"
find "$SCRIPT_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name '*.pyc' -delete 2>/dev/null || true
finish_run_unit "wipe stale bytecode caches"

# Self-update: pull latest main, then re-exec the refreshed script so changes
# to run.sh / pyproject.toml / source files are picked up on every invocation.
# The GENTERP_REEXEC guard prevents an infinite re-exec loop after one update.
log_run "START self-update from git if this is the first run.sh invocation"
if [ -z "${GENTERP_REEXEC:-}" ] && git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[run.sh] git pull --ff-only in $SCRIPT_DIR" >&2
  if git -C "$SCRIPT_DIR" pull --ff-only; then
    finish_run_unit "self-update succeeded; re-executing refreshed run.sh"
    export GENTERP_REEXEC=1
    exec bash "$0" "$@"
  else
    echo "[run.sh] git pull failed; continuing with on-disk code" >&2
  fi
fi
finish_run_unit "self-update check complete"

# Bootstrap uv if missing.
log_run "START ensure uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
finish_run_unit "ensure uv is installed"

# Sync deps (after pull, so pyproject.toml changes flow through).
log_run "START sync Python dependencies with uv"
uv sync
finish_run_unit "sync Python dependencies with uv"

# Forward run.sh args (e.g. --tiny) to both ETL and training entrypoints so
# `~/genterp/run.sh --tiny` exercises the downsampled cohort end-to-end.
# ETL is internally cached per-CDR; rerun is cheap if outputs are warm.
log_run "START run AoU ETL workflow"
uv run python -m scripts.aou_etl "$@"
finish_run_unit "run AoU ETL workflow"

log_run "START run Genterp training workflow"
uv run python -m genterp.train "$@"
finish_run_unit "run Genterp training workflow"

log_run "START run CLT training workflow"
uv run python -m genterp.clt_train "$@"
finish_run_unit "run CLT training workflow"

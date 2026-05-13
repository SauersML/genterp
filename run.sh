#!/usr/bin/env bash
set -euo pipefail

RUN_TOTAL_UNITS=7
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
finish_run_unit "enable unbuffered Python output"

log_run "START resolve repository directory"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
finish_run_unit "resolve repository directory: $SCRIPT_DIR"

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

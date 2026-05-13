#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Self-update: pull latest main, then re-exec the refreshed script so changes
# to run.sh / pyproject.toml / source files are picked up on every invocation.
# The GENTERP_REEXEC guard prevents an infinite re-exec loop after one update.
if [ -z "${GENTERP_REEXEC:-}" ] && git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[run.sh] git pull --ff-only in $SCRIPT_DIR" >&2
  if git -C "$SCRIPT_DIR" pull --ff-only; then
    export GENTERP_REEXEC=1
    exec bash "$0" "$@"
  else
    echo "[run.sh] git pull failed; continuing with on-disk code" >&2
  fi
fi

# Bootstrap uv if missing.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Sync deps (after pull, so pyproject.toml changes flow through).
uv sync

# ETL is internally cached per-CDR; rerun is cheap if outputs are warm.
uv run python -m scripts.aou_etl
uv run python -m genterp.train

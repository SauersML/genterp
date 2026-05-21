#!/usr/bin/env bash
set -euo pipefail

RUN_TOTAL_UNITS=11
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

# Self-update: ff-merge from origin/main, then re-exec the refreshed script
# so every invocation runs the latest committed code. Workflow:
#   1. If there are uncommitted local edits or untracked files we care about,
#      skip the update entirely (loud log) so working state is never lost.
#      Set GENTERP_RUN_FORCE_UPDATE=1 to override (use git reset --hard).
#   2. Otherwise git fetch + ff-only merge. ff-only refuses to rewrite history
#      so a diverged local branch errors out instead of silently losing work.
#   3. On success, re-exec the refreshed run.sh so the rest of the run uses
#      the new code. GENTERP_REEXEC guard prevents infinite re-exec loops.
log_run "START self-update from git origin/main"
if [ -z "${GENTERP_REEXEC:-}" ] && git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  CURRENT_HEAD="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "[run.sh] current HEAD before update: $CURRENT_HEAD" >&2
  WT_DIRTY=""
  if ! git -C "$SCRIPT_DIR" diff --quiet 2>/dev/null; then
    WT_DIRTY="unstaged"
  fi
  if ! git -C "$SCRIPT_DIR" diff --cached --quiet 2>/dev/null; then
    WT_DIRTY="${WT_DIRTY:+$WT_DIRTY,}staged"
  fi
  if [ -n "$WT_DIRTY" ] && [ -z "${GENTERP_RUN_FORCE_UPDATE:-}" ]; then
    echo "[run.sh] working tree has $WT_DIRTY changes; skipping self-update to preserve them" >&2
    echo "[run.sh] commit/stash the changes (or set GENTERP_RUN_FORCE_UPDATE=1) to enable self-update" >&2
  elif ! git -C "$SCRIPT_DIR" fetch --quiet origin '+refs/heads/main:refs/remotes/origin/main'; then
    echo "[run.sh] WARNING: git fetch origin main failed; continuing with current checkout" >&2
  else
    TARGET="$(git -C "$SCRIPT_DIR" rev-parse --short origin/main 2>/dev/null || echo unknown)"
    if [ "$CURRENT_HEAD" = "$TARGET" ]; then
      echo "[run.sh] already at origin/main ($TARGET); no re-exec needed" >&2
    elif [ -n "${GENTERP_RUN_FORCE_UPDATE:-}" ]; then
      echo "[run.sh] resetting $CURRENT_HEAD -> $TARGET (origin/main) [force]" >&2
      if ! git -C "$SCRIPT_DIR" reset --hard origin/main; then
        echo "[run.sh] FATAL: git reset --hard origin/main failed" >&2
        exit 1
      fi
      finish_run_unit "self-update succeeded (force); re-executing refreshed run.sh at $TARGET"
      export GENTERP_REEXEC=1
      exec bash "$0" "$@"
    elif git -C "$SCRIPT_DIR" merge --ff-only --quiet origin/main; then
      echo "[run.sh] fast-forwarded $CURRENT_HEAD -> $TARGET (origin/main)" >&2
      finish_run_unit "self-update succeeded; re-executing refreshed run.sh at $TARGET"
      export GENTERP_REEXEC=1
      exec bash "$0" "$@"
    else
      echo "[run.sh] local branch diverged from origin/main ($CURRENT_HEAD vs $TARGET); fast-forward refused" >&2
      echo "[run.sh] rebase or reset locally, then re-run (or set GENTERP_RUN_FORCE_UPDATE=1 to hard-reset)" >&2
    fi
  fi
else
  echo "[run.sh] $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) (re-exec or non-git)" >&2
fi
finish_run_unit "self-update check complete"

# Loud confirmation of what the running source actually is. If the
# "VERSION_TAG" line from aou_etl.main() ever shows a different commit/sha
# than what's printed here, the workspace is running stale code.
HEAD_NOW="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
ETL_COMMIT="$(git -C "$SCRIPT_DIR" log -1 --format=%h --abbrev=10 -- scripts/aou_etl.py 2>/dev/null || echo unknown)"
TRAIN_COMMIT="$(git -C "$SCRIPT_DIR" log -1 --format=%h --abbrev=10 -- genterp/train.py 2>/dev/null || echo unknown)"
RUN_COMMIT="$(git -C "$SCRIPT_DIR" log -1 --format=%h --abbrev=10 -- run.sh 2>/dev/null || echo unknown)"
echo "[run.sh] commit summary: HEAD=$HEAD_NOW aou_etl.py=$ETL_COMMIT train.py=$TRAIN_COMMIT run.sh=$RUN_COMMIT" >&2

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

# Build the per-atom ancestor table from the ETL cache (idempotent — exits
# fast when the fingerprint matches the existing ancestors.npz). Activates
# hierarchical embeddings in the model: rare leaf atoms inherit signal from
# their SNOMED IS-A ancestors via additive ancestor-sum embeddings. Warm-
# start safe: ancestor_embedding initializes to zero, so the first forward
# after activation is bit-identical to the flat-embedding checkpoint and
# gradient pressure then learns the hierarchy from there.
log_run "START build hierarchical ancestor table"
uv run python -m scripts.build_ancestors "$@"
finish_run_unit "build hierarchical ancestor table"

log_run "START run Genterp training workflow"
GPU_COUNT="$(uv run python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [ "${GPU_COUNT:-0}" -gt 1 ]; then
  echo "[run.sh] launching training with Accelerate DDP across ${GPU_COUNT} GPUs" >&2
  uv run accelerate launch --num_processes "$GPU_COUNT" -m genterp.train "$@"
else
  uv run python -m genterp.train "$@"
fi
finish_run_unit "run Genterp training workflow"

log_run "START run CLT training workflow"
uv run python -m genterp.clt_train "$@"
finish_run_unit "run CLT training workflow"

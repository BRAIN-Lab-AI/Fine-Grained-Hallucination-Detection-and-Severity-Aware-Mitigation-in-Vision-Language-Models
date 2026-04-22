#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

INPUT="${INPUT:-output/fghd/stage2/rewrites.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/stage3}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/vote_records.jsonl}"
PREFERENCES_OUT="${PREFERENCES_OUT:-${OUTPUT_DIR}/preference_pairs.jsonl}"
STATS_OUT="${STATS_OUT:-${OUTPUT_DIR}/stats.json}"
BACKEND="${BACKEND:-heuristic}"

if [ ! -f "${INPUT}" ]; then
  echo "Stage 3 input not found: ${INPUT}" >&2
  echo "Run Stage 2 first:  bash scripts/run_stage2_rewrites.sh" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m fg_pipeline.stage3.run_stage3
  --input "${INPUT}"
  --output "${OUTPUT}"
  --preferences-out "${PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}" "$@"

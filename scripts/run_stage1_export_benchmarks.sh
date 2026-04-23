#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

BENCHMARK="${BENCHMARK:-mhalubench}"
STAGE1_INPUT="${STAGE1_INPUT:-output/fghd/stage1/detection_critiques.jsonl}"
ANNOTATION_INPUT="${ANNOTATION_INPUT:-playground/data/eval/${BENCHMARK}/annotations.jsonl}"
OUTPUT="${OUTPUT:-playground/data/eval/${BENCHMARK}/predictions.jsonl}"

python -m fg_pipeline.stage1.run_stage1_export_benchmarks \
  --benchmark "${BENCHMARK}" \
  --stage1-input "${STAGE1_INPUT}" \
  --annotation-input "${ANNOTATION_INPUT}" \
  --output "${OUTPUT}" \
  "$@"

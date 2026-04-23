#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

INPUT="${INPUT:-fg_pipeline/data/hsa_dpo_detection.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/stage1/detector_train.json}"

CMD=(
  python -m fg_pipeline.stage1.run_stage1_detector_dataset
  --input "${INPUT}"
  --output "${OUTPUT}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}" "$@"

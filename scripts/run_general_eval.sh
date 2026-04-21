#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-output/eval}"
RUN_NAME="${RUN_NAME:-general_eval}"
MODEL_MANIFEST="${MODEL_MANIFEST:-models.eval.json}"
BENCHMARKS="${BENCHMARKS:-pope_adv,llava_bench_wild}"
OPENAI_JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-${OPENAI_MODEL:-}}"

if [ ! -f "${MODEL_MANIFEST}" ]; then
  echo "Missing MODEL_MANIFEST: ${MODEL_MANIFEST}" >&2
  exit 1
fi

CMD=(
  python -m fg_pipeline.eval.run_eval
  --run-name "${RUN_NAME}"
  --models-json "${MODEL_MANIFEST}"
  --benchmarks "${BENCHMARKS}"
  --general
  --skip-missing-datasets
  --output-root "${OUTPUT_ROOT}"
)

if [ -n "${OPENAI_JUDGE_MODEL}" ]; then
  CMD+=(--openai-judge-model "${OPENAI_JUDGE_MODEL}")
fi

"${CMD[@]}" "$@"

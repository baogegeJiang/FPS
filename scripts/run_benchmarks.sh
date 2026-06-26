#!/usr/bin/env bash
set -euo pipefail

# Run FPS-UDA benchmark training tasks and write summary.json/summary.csv.
#
# Examples:
#   bash scripts/run_benchmarks.sh
#   DEVICE=cuda:0 RESUME=1 bash scripts/run_benchmarks.sh
#   DATASETS=office31 BACKBONES=vit LIMIT=2 DRY_RUN=1 bash scripts/run_benchmarks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/training}"
OUT_ROOT="${OUT_ROOT:-runs/benchmarks}"
DATASETS="${DATASETS:-}"
BACKBONES="${BACKBONES:-}"
TASKS="${TASKS:-}"
DEVICE="${DEVICE:-cuda:0}"
ITER_NUM="${ITER_NUM:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
KEEP_GOING="${KEEP_GOING:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

cmd=(
  "${PYTHON_BIN}" scripts/run_benchmarks.py
  --config-root "${CONFIG_ROOT}"
  --out-root "${OUT_ROOT}"
)

if [[ -n "${DATASETS}" ]]; then
  cmd+=(--datasets "${DATASETS}")
fi
if [[ -n "${BACKBONES}" ]]; then
  cmd+=(--backbones "${BACKBONES}")
fi
if [[ -n "${TASKS}" ]]; then
  cmd+=(--tasks "${TASKS}")
fi
if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi
if [[ -n "${ITER_NUM}" ]]; then
  cmd+=(--iter-num "${ITER_NUM}")
fi
if [[ -n "${EVAL_INTERVAL}" ]]; then
  cmd+=(--eval-interval "${EVAL_INTERVAL}")
fi
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "${KEEP_GOING}" == "1" ]]; then
  cmd+=(--keep-going)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${NO_PROGRESS}" == "1" ]]; then
  cmd+=(--no-progress)
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"

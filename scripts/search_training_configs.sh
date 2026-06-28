#!/usr/bin/env bash
set -euo pipefail

# Batch-search FPS-UDA training configs and copy each best.yaml into
# configs/training_autosearch/{dataset}/{task}/{suffix}.yaml.
#
# Examples:
#   bash scripts/search_training_configs.sh
#   BACKBONES="resnet vit siglip2" DEVICE=cuda:0 bash scripts/search_training_configs.sh
#   SUFFIXES=vit DATASETS=office31 TASKS=amazon_to_webcam DRY_RUN=1 bash scripts/search_training_configs.sh
#   SEARCH_SPACE=configs/search/default_search_space.yaml bash scripts/search_training_configs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BANK_ROOT="${BANK_ROOT:-fps_h5cache/banks}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/training_autosearch}"
if [[ -n "${SUFFIXES:-}" ]]; then
  BACKBONES="${SUFFIXES}"
else
BACKBONES="${BACKBONES:-siglip2}"
fi
SEARCH_ROOT="${SEARCH_ROOT:-runs/search_configs}"
SEARCH_SPACE="${SEARCH_SPACE:-configs/search/default_search_space.yaml}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
METRIC="${METRIC:-acc}"
ROUNDS="${ROUNDS:-3}"
GRID_MIN="${GRID_MIN:-0.05}"
GRID_MAX="${GRID_MAX:-1.0}"
GRID_STEP="${GRID_STEP:-0.05}"
PATIENCE="${PATIENCE:-10}"
ITER_NUM="${ITER_NUM:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-}"
CROSS_NORM_SCALE="${CROSS_NORM_SCALE:-2.5}"
LR_CANDIDATES="${LR_CANDIDATES:-}"
LAMBDA_LCR_GRID="${LAMBDA_LCR_GRID:-}"
DATASETS="${DATASETS:-office31 office_home visda17}"
TASKS="${TASKS:-}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-1}"
KEEP_GOING="${KEEP_GOING:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

RUN_COUNT=0
SKIP_COUNT=0
FAILURES=()

contains_item() {
  local list="${1//,/ }"
  local needle="$2"
  local item
  for item in ${list}; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

dataset_enabled() {
  contains_item "${DATASETS}" "$1"
}

task_enabled() {
  if [[ -z "${TASKS}" ]]; then
    return 0
  fi
  contains_item "${TASKS}" "$1"
}

backbone_enabled() {
  contains_item "${BACKBONES}" "$1"
}

print_cmd() {
  printf '%q ' "$@"
  echo
}

atomic_copy() {
  local source_path="$1"
  local target_path="$2"
  local target_dir
  local target_name
  local tmp_path
  target_dir="$(dirname "${target_path}")"
  target_name="$(basename "${target_path}")"
  mkdir -p "${target_dir}"
  tmp_path="$(mktemp "${target_dir}/.${target_name}.tmp.XXXXXX")"
  cp "${source_path}" "${tmp_path}"
  mv "${tmp_path}" "${target_path}"
}

record_failure() {
  local message="$1"
  FAILURES+=("${message}")
  echo "ERROR: ${message}" >&2
  if [[ "${KEEP_GOING}" != "1" ]]; then
    exit 1
  fi
}

bank_file_for() {
  local dataset="$1"
  local suffix="$2"
  case "${dataset}:${suffix}" in
    office31:resnet) echo "office31_resnet50.h5" ;;
    office31:vit) echo "office31_vit.h5" ;;
    office31:siglip2) echo "office31_siglip2.h5" ;;
    office_home:resnet) echo "office_home_resnet50.h5" ;;
    office_home:vit) echo "office_home_vit.h5" ;;
    office_home:siglip2) echo "office_home_siglip2.h5" ;;
    visda17:resnet) echo "visda17_resnet101.h5" ;;
    visda17:vit) echo "visda17_vit.h5" ;;
    visda17:siglip2) echo "visda17_siglip2.h5" ;;
    *)
      return 1
      ;;
  esac
}

run_task() {
  local dataset="$1"
  local suffix="$2"
  local task="$3"
  local source_domain="$4"
  local target_domain="$5"
  local bank_file
  if ! bank_file="$(bank_file_for "${dataset}" "${suffix}")"; then
    record_failure "${dataset}/${task}/${suffix}: unknown suffix. Known suffixes: resnet vit siglip2"
    return 0
  fi
  local bank="${BANK_ROOT}/${bank_file}"
  local search_dir="${SEARCH_ROOT}/${suffix}/${dataset}/${task}"
  local config_path="${CONFIG_ROOT}/${dataset}/${task}/${suffix}.yaml"
  local best_yaml="${search_dir}/best.yaml"

  if ! backbone_enabled "${suffix}"; then
    return 0
  fi
  if ! dataset_enabled "${dataset}"; then
    return 0
  fi
  if ! task_enabled "${task}"; then
    return 0
  fi
  if [[ -n "${LIMIT}" && "${RUN_COUNT}" -ge "${LIMIT}" ]]; then
    return 0
  fi
  if [[ "${OVERWRITE}" != "1" && -f "${config_path}" ]]; then
    echo "==> Skipping ${dataset}/${task}/${suffix}: ${config_path} exists and OVERWRITE=0"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return 0
  fi
  if [[ ! -f "${bank}" ]]; then
    record_failure "${dataset}/${task}/${suffix}: missing feature bank ${bank}"
    return 0
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  echo
  echo "==> Searching ${dataset}/${task}/${suffix}: '${source_domain}' -> '${target_domain}'"
  echo "    bank:   ${bank}"
  echo "    search: ${search_dir}"
  echo "    config: ${config_path}"

  if [[ "${RESUME}" == "1" && -f "${best_yaml}" ]]; then
    echo "    Reusing existing ${best_yaml} because RESUME=1."
  else
    local cmd=(
      "${PYTHON_BIN}" scripts/search_fps_hyperparams.py
      --feature-bank "${bank}"
      --source-domain "${source_domain}"
      --target-domain "${target_domain}"
      --out "${search_dir}"
      --device "${DEVICE}"
      --seed "${SEED}"
      --metric "${METRIC}"
      --rounds "${ROUNDS}"
      --grid-min "${GRID_MIN}"
      --grid-max "${GRID_MAX}"
      --grid-step "${GRID_STEP}"
      --patience "${PATIENCE}"
      --cross-norm-scale "${CROSS_NORM_SCALE}"
      --search-space "${SEARCH_SPACE}"
    )
    if [[ -n "${ITER_NUM}" ]]; then
      cmd+=(--iter-num "${ITER_NUM}")
    fi
    if [[ -n "${EVAL_INTERVAL}" ]]; then
      cmd+=(--eval-interval "${EVAL_INTERVAL}")
    fi
    if [[ -n "${LR_CANDIDATES}" ]]; then
      cmd+=(--lr-candidates "${LR_CANDIDATES}")
    fi
    if [[ -n "${LAMBDA_LCR_GRID}" ]]; then
      cmd+=(--lambda-lcr-grid "${LAMBDA_LCR_GRID}")
    fi
    if [[ "${NO_PROGRESS}" == "1" ]]; then
      cmd+=(--no-progress)
    fi

    echo "    command:"
    echo -n "      "
    print_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${search_dir}"
      if ! "${cmd[@]}"; then
        record_failure "${dataset}/${task}/${suffix}: search command failed"
        return 0
      fi
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    DRY_RUN=1: would copy ${best_yaml} -> ${config_path}"
    return 0
  fi
  if [[ ! -f "${best_yaml}" ]]; then
    record_failure "${dataset}/${task}/${suffix}: expected output ${best_yaml} was not created"
    return 0
  fi
  atomic_copy "${best_yaml}" "${config_path}"
  echo "    Wrote ${config_path}"
}

run_office31() {
  local suffix="$1"
  run_task office31 "${suffix}" amazon_to_dslr amazon dslr
  run_task office31 "${suffix}" amazon_to_webcam amazon webcam
  run_task office31 "${suffix}" dslr_to_amazon dslr amazon
  run_task office31 "${suffix}" webcam_to_amazon webcam amazon
}

run_office_home() {
  local suffix="$1"
  run_task office_home "${suffix}" art_to_clipart Art Clipart
  run_task office_home "${suffix}" art_to_product Art Product
  run_task office_home "${suffix}" art_to_real_world Art "Real World"
  run_task office_home "${suffix}" clipart_to_art Clipart Art
  run_task office_home "${suffix}" clipart_to_product Clipart Product
  run_task office_home "${suffix}" clipart_to_real_world Clipart "Real World"
  run_task office_home "${suffix}" product_to_art Product Art
  run_task office_home "${suffix}" product_to_clipart Product Clipart
  run_task office_home "${suffix}" product_to_real_world Product "Real World"
  run_task office_home "${suffix}" real_world_to_art "Real World" Art
  run_task office_home "${suffix}" real_world_to_clipart "Real World" Clipart
  run_task office_home "${suffix}" real_world_to_product "Real World" Product
}

run_visda17() {
  local suffix="$1"
  run_task visda17 "${suffix}" syn_to_real train validation
}

for suffix in ${BACKBONES//,/ }; do
  case "${suffix}" in
    resnet | vit | siglip2)
      run_office31 "${suffix}"
      run_office_home "${suffix}"
      run_visda17 "${suffix}"
      ;;
    *)
      record_failure "${suffix}: unknown suffix. Known suffixes: resnet vit siglip2"
      ;;
  esac
done

echo
echo "Search tasks considered: ${RUN_COUNT}; skipped: ${SKIP_COUNT}."
if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo "Failures:" >&2
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
echo "Done. Search artifacts are under ${SEARCH_ROOT}; configs are under ${CONFIG_ROOT}."

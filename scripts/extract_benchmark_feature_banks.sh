#!/usr/bin/env bash
set -euo pipefail

# Extract the benchmark dataset-level feature banks with the current YAML
# settings. Each dataset has a ResNet and a ViT bank.
#
# Examples:
#   bash scripts/extract_benchmark_feature_banks.sh
#   DEVICE=cuda:0 NUM_WORKERS=16 bash scripts/extract_benchmark_feature_banks.sh
#   BANKS="office31_resnet office_home_vit" bash scripts/extract_benchmark_feature_banks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-fps_h5cache/banks}"
NUM_WORKERS="${NUM_WORKERS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NO_PROGRESS="${NO_PROGRESS:-0}"
BANKS="${BANKS:-office31_resnet office31_vit office_home_resnet office_home_vit visda17_resnet visda17_vit}"

mkdir -p "${OUT_DIR}"

run_bank() {
  local name="$1"
  local config="$2"
  local output="$3"
  shift 3

  local cmd=(
    "${PYTHON_BIN}" -m fps_uda
    extract-feature-bank
    --dataset-config "${config}"
    --out "${OUT_DIR}/${output}"
    --device "${DEVICE}"
  )
  if [[ -n "${NUM_WORKERS}" ]]; then
    cmd+=(--num-workers "${NUM_WORKERS}")
  fi
  if [[ -n "${BATCH_SIZE}" ]]; then
    cmd+=(--batch-size "${BATCH_SIZE}")
  fi
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    cmd+=(--no-progress)
  fi

  echo
  echo "==> Extracting ${name}"
  printf '    %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

for bank in ${BANKS}; do
  case "${bank}" in
    office31_resnet)
      run_bank "${bank}" "configs/datasets/office31_resnet.yaml" "office31_resnet50.h5"
      ;;
    office31_vit)
      run_bank "${bank}" "configs/datasets/office31_vit.yaml" "office31_vit.h5"
      ;;
    office_home_resnet)
      run_bank "${bank}" "configs/datasets/office_home_resnet.yaml" "office_home_resnet50.h5"
      ;;
    office_home_vit)
      run_bank "${bank}" "configs/datasets/office_home_vit.yaml" "office_home_vit.h5"
      ;;
    visda17_resnet)
      run_bank "${bank}" "configs/datasets/visda17_resnet.yaml" "visda17_resnet101.h5"
      ;;
    visda17_vit)
      run_bank "${bank}" "configs/datasets/visda17_vit.yaml" "visda17_vit.h5"
      ;;
    *)
      echo "Unknown bank '${bank}'." >&2
      echo "Known banks: office31_resnet office31_vit office_home_resnet office_home_vit visda17_resnet visda17_vit" >&2
      exit 2
      ;;
  esac
done

echo
echo "All requested feature banks were extracted into ${OUT_DIR}."

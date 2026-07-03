#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/script"
PROGRAM_NAME="$(basename "$0")"

INPUT_PATH=""
CONFIG_PATH="${ROOT_DIR}/config/registration.conf"
GRAYSCALE_MODE=""
ROTATION="0"

usage() {
    cat <<EOF
Usage:
  $PROGRAM_NAME rgb|nissl --input-path /path/brain.tif [--config /path/registration.conf] [--rotation 0|90|180|270]

Required:
  rgb|nissl                         Grayscale conversion mode
  --input-path PATH
  --config PATH                     Optional; default: config/registration.conf
  --rotation 0|90|180|270           Counterclockwise degrees; default: 0

Set ROI_TXT_PATH in the configuration file to run Step5. Leave it empty to skip Step5.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
case "$1" in
    -h|--help) usage; exit 0 ;;
esac
GRAYSCALE_MODE="$1"
shift
if [[ "$GRAYSCALE_MODE" != "rgb" && "$GRAYSCALE_MODE" != "nissl" ]]; then
    echo "The first argument must be rgb or nissl." >&2
    usage >&2
    exit 2
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-path) INPUT_PATH="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --rotation) ROTATION="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$INPUT_PATH" ]]; then
    echo "--input-path is required." >&2
    usage >&2
    exit 2
fi
case "$ROTATION" in
    0|90|180|270) ;;
    *) echo "--rotation must be 0, 90, 180, or 270." >&2; exit 2 ;;
esac
for required_path in "$INPUT_PATH" "$CONFIG_PATH"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Input file does not exist: $required_path" >&2
        exit 2
    fi
done
CLI_INPUT_PATH="$INPUT_PATH"
CLI_GRAYSCALE_MODE="$GRAYSCALE_MODE"
CLI_ROTATION="$ROTATION"
# shellcheck source=/dev/null
source "$CONFIG_PATH"
INPUT_PATH="$CLI_INPUT_PATH"
GRAYSCALE_MODE="$CLI_GRAYSCALE_MODE"
ROTATION="$CLI_ROTATION"

: "${ATLAS_NISSL:?ATLAS_NISSL is required in the configuration file}"
: "${ATLAS_ANNOTATION:?ATLAS_ANNOTATION is required in the configuration file}"
: "${OUTPUT_PATH:=}"
: "${INPUT_RES:=0.294}"
: "${TARGET_RES:=10.0}"
: "${BRAIN_LAYOUT:=auto}"
: "${REPLACE_BACKGROUND_VALUES:=1}"
: "${ATLAS_SLICE:=}"
: "${SLICE_SEARCH_RADIUS:=200}"
: "${SLICE_SEARCH_STEP:=20}"
: "${SEARCH_RESIZE_MAX:=0.6}"
: "${RANDOM_SEED:=42}"
: "${WORKERS:=8}"
: "${NEIGHBOR_SMOOTH_SIGMA:=3.0}"
: "${METRIC_MODE:=weighted}"
: "${WEIGHT_FLOOR:=0.25}"
: "${MESH_SIZE:=8}"
: "${SAMPLING_PERCENTAGE:=0.25}"
: "${ROI_TXT_PATH:=}"
: "${STRUCTURE_TREE_CSV:=}"

for atlas_path in "$ATLAS_NISSL" "$ATLAS_ANNOTATION"; do
    if [[ ! -f "$atlas_path" ]]; then
        echo "Configured atlas file does not exist: $atlas_path" >&2
        exit 2
    fi
done

if [[ -n "$ROI_TXT_PATH" && -z "$STRUCTURE_TREE_CSV" ]]; then
    echo "STRUCTURE_TREE_CSV is required in the configuration when ROI is provided." >&2
    exit 2
fi
if [[ -n "$ROI_TXT_PATH" && ! -f "$ROI_TXT_PATH" ]]; then
    echo "ROI file does not exist: $ROI_TXT_PATH" >&2
    exit 2
fi
if [[ -n "$ROI_TXT_PATH" && ! -f "$STRUCTURE_TREE_CSV" ]]; then
    echo "Configured structure tree does not exist: $STRUCTURE_TREE_CSV" >&2
    exit 2
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    OUTPUT_PATH="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
fi

INPUT_FILE="$(basename "$INPUT_PATH")"
SAMPLE_NAME="${INPUT_FILE%.*}"
STEP1_DIR="${OUTPUT_PATH}/01.preprocess"
STEP2_DIR="${OUTPUT_PATH}/02.select_slice"
STEP3_DIR="${OUTPUT_PATH}/03.registration"
STEP4_DIR="${OUTPUT_PATH}/04.apply_label"
STEP5_DIR="${OUTPUT_PATH}/05.roi_mask"
mkdir -p "$STEP1_DIR" "$STEP2_DIR" "$STEP3_DIR" "$STEP4_DIR"

PYTHON_CMD=(pixi run --manifest-path "${ROOT_DIR}/pixi.toml" python)

STEP1_ARGS=(
    --input-path "$INPUT_PATH"
    --output-path "$STEP1_DIR"
    --input-res "$INPUT_RES"
    --target-res "$TARGET_RES"
    --grayscale-mode "$GRAYSCALE_MODE"
    --rotation "$ROTATION"
    --brain-layout "$BRAIN_LAYOUT"
)
if [[ "$REPLACE_BACKGROUND_VALUES" == "1" ]]; then
    STEP1_ARGS+=(--replace-background-values)
else
    STEP1_ARGS+=(--no-replace-background-values)
fi
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/step1_preprocess.py" "${STEP1_ARGS[@]}"

STEP1_JSON="${STEP1_DIR}/${SAMPLE_NAME}_step1_record.json"
STEP2_ARGS=(
    --step1-record-json "$STEP1_JSON"
    --moving-path "$ATLAS_NISSL"
    --output-path "$STEP2_DIR"
    --slice-search-radius "$SLICE_SEARCH_RADIUS"
    --slice-search-step "$SLICE_SEARCH_STEP"
    --search-resize-max "$SEARCH_RESIZE_MAX"
    --random-seed "$RANDOM_SEED"
    --workers "$WORKERS"
    --neighbor-smooth-sigma "$NEIGHBOR_SMOOTH_SIGMA"
)
if [[ -n "$ATLAS_SLICE" ]]; then
    STEP2_ARGS+=(--atlas-slice "$ATLAS_SLICE")
fi
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/step2_select_slice.py" "${STEP2_ARGS[@]}"

STEP2_JSON="${STEP2_DIR}/step2_record.json"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/step3_registration.py" \
    --step2-record-json "$STEP2_JSON" \
    --output-path "$STEP3_DIR" \
    --metric-mode "$METRIC_MODE" \
    --weight-floor "$WEIGHT_FLOOR" \
    --mesh-size "$MESH_SIZE" \
    --sampling-percentage "$SAMPLING_PERCENTAGE" \
    --random-seed "$RANDOM_SEED"

STEP3_JSON="${STEP3_DIR}/${SAMPLE_NAME}_step3_record.json"
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/step4_apply_label.py" \
    --label-path "$ATLAS_ANNOTATION" \
    --step3-record-json "$STEP3_JSON" \
    --output-path "$STEP4_DIR"

STEP4_JSON="${STEP4_DIR}/${SAMPLE_NAME}_step4_record.json"
if [[ -n "$ROI_TXT_PATH" ]]; then
    mkdir -p "$STEP5_DIR"
    "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/step5_roi_mask.py" \
        --step4-record-json "$STEP4_JSON" \
        --roi-txt-path "$ROI_TXT_PATH" \
        --structure-tree-csv "$STRUCTURE_TREE_CSV" \
        --output-path "$STEP5_DIR"
else
    echo "Step5 skipped: --roi-txt-path was not provided"
fi

echo "Registration pipeline finished: ${OUTPUT_PATH}"

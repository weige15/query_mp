#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SHORT="${MODEL_SHORT:-${MODEL_ID##*/}}"
MATGPTQ_DIR="${MATGPTQ_DIR:-$HOME/MatGPTQ}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/runs}"
GPU_DEVICE="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-5}}"

BITS_LIST="${BITS_LIST:-2 3 4 6 8}"
BITS_WEIGHTS="${BITS_WEIGHTS:-1 1 1 1 1}"
SLICE_BITS="${SLICE_BITS:-2 3 4 6 8}"
LOW_BITS="${LOW_BITS:-2 3 4 6}"
HIGH_BIT="${HIGH_BIT:-8}"
EVAL_DATASETS="${EVAL_DATASETS:-c4}"
COLLECT_DATASET="${COLLECT_DATASET:-${EVAL_DATASETS%% *}}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
EVAL_TOKENS="${EVAL_TOKENS:-8192}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
MAX_MODULES="${MAX_MODULES:-0}"
MODULE_REGEX="${MODULE_REGEX:-.*layers.*((q|k|v|o|gate|up|down)_proj)$}"

export MATGPTQ_DIR MODEL_ID MODEL_SHORT RUN_DIR GPU_DEVICE
export BITS_LIST BITS_WEIGHTS SLICE_BITS EVAL_DATASETS
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

cd "$REPO_DIR"

if [[ "${SKIP_QUANT:-0}" != "1" ]]; then
  bash "$SCRIPT_DIR/run_matgptq_quant.sh"
fi

if [[ "${SKIP_PPL:-0}" != "1" ]]; then
  bash "$SCRIPT_DIR/run_matgptq_ppl_slices.sh"
fi

QUANT_WEIGHTS="${QUANT_WEIGHTS:-$RUN_DIR/weights/$MODEL_SHORT}"
SENS_DIR="$RUN_DIR/h1_sensitivity"
SENS_CSV="$SENS_DIR/sensitivity.csv"

python "$SCRIPT_DIR/scripts/collect_h1_sensitivity.py" \
  --model_name_or_path "$MODEL_ID" \
  --matgptq_dir "$MATGPTQ_DIR" \
  --quant_weights_path "$QUANT_WEIGHTS" \
  --output_csv "$SENS_CSV" \
  --dataset "$COLLECT_DATASET" \
  --sequence_length "$SEQUENCE_LENGTH" \
  --eval_tokens "$EVAL_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --max_modules "$MAX_MODULES" \
  --module_regex "$MODULE_REGEX" \
  --high_bit "$HIGH_BIT" \
  --low_bits $LOW_BITS

python "$SCRIPT_DIR/scripts/analyze_h1_sensitivity.py" \
  --input_csv "$SENS_CSV" \
  --out_dir "$SENS_DIR/analysis" \
  --high_bit "$HIGH_BIT"

echo "H1 outputs:"
echo "$SENS_CSV"
echo "$SENS_DIR/analysis/summary.md"

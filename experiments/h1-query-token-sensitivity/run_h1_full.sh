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
PROMPTS_JSONL="${PROMPTS_JSONL:-}"
ANALYSIS_STEPS="${ANALYSIS_STEPS:-500}"
ANALYSIS_SAMPLE_FRACTION="${ANALYSIS_SAMPLE_FRACTION:-1}"
ANALYSIS_NO_SCORED_CSV="${ANALYSIS_NO_SCORED_CSV:-0}"

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

collect_args=(
  --model_name_or_path "$MODEL_ID" \
  --matgptq_dir "$MATGPTQ_DIR" \
  --quant_weights_path "$QUANT_WEIGHTS" \
  --dataset "$COLLECT_DATASET" \
  --sequence_length "$SEQUENCE_LENGTH" \
  --eval_tokens "$EVAL_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --max_modules "$MAX_MODULES" \
  --module_regex "$MODULE_REGEX" \
  --high_bit "$HIGH_BIT" \
  --low_bits $LOW_BITS
)

if [[ -n "$PROMPTS_JSONL" ]]; then
  collect_args+=(--prompts_jsonl "$PROMPTS_JSONL")
fi

if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  IFS=',' read -r -a SENS_GPUS <<< "$GPU_DEVICE"
  if (( ${#SENS_GPUS[@]} > 1 )); then
    mkdir -p "$SENS_DIR"
    rm -f "$SENS_DIR"/sensitivity_shard_*.csv "$SENS_DIR"/sensitivity_shard_*.log "$SENS_CSV"
    pids=()
    cleanup_shards() {
      if (( ${#pids[@]} > 0 )); then
        kill "${pids[@]}" 2>/dev/null || true
      fi
    }
    trap cleanup_shards INT TERM EXIT
    for shard_index in "${!SENS_GPUS[@]}"; do
      shard_csv="$SENS_DIR/sensitivity_shard_${shard_index}.csv"
      shard_log="$SENS_DIR/sensitivity_shard_${shard_index}.log"
      CUDA_VISIBLE_DEVICES="${SENS_GPUS[$shard_index]}" python "$SCRIPT_DIR/scripts/collect_h1_sensitivity.py" \
        "${collect_args[@]}" \
        --output_csv "$shard_csv" \
        --sample_shard_count "${#SENS_GPUS[@]}" \
        --sample_shard_index "$shard_index" \
        > "$shard_log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    trap - INT TERM EXIT
    first_shard="$SENS_DIR/sensitivity_shard_0.csv"
    head -n 1 "$first_shard" > "$SENS_CSV"
    for shard_csv in "$SENS_DIR"/sensitivity_shard_*.csv; do
      tail -n +2 "$shard_csv" >> "$SENS_CSV"
    done
  else
    python "$SCRIPT_DIR/scripts/collect_h1_sensitivity.py" \
      "${collect_args[@]}" \
      --output_csv "$SENS_CSV"
  fi
elif [[ ! -f "$SENS_CSV" ]]; then
  echo "SKIP_COLLECT=1 but missing $SENS_CSV" >&2
  exit 1
fi

analysis_args=(
  --input_csv "$SENS_CSV" \
  --out_dir "$SENS_DIR/analysis" \
  --high_bit "$HIGH_BIT" \
  --steps "$ANALYSIS_STEPS" \
  --sample_fraction "$ANALYSIS_SAMPLE_FRACTION"
)

if [[ "$ANALYSIS_NO_SCORED_CSV" == "1" ]]; then
  analysis_args+=(--no_scored_csv)
fi

python3 "$SCRIPT_DIR/scripts/analyze_h1_sensitivity.py" "${analysis_args[@]}"

echo "H1 outputs:"
echo "$SENS_CSV"
echo "$SENS_DIR/analysis/summary.md"

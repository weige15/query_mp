#!/usr/bin/env bash
set -euo pipefail

MATGPTQ_DIR="${MATGPTQ_DIR:-/home/kuotzuwei15/MatGPTQ}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
OUT_DIR="${OUT_DIR:-$PWD/experiments/h1-query-token-sensitivity/runs}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
CALIB_DATA="${CALIB_DATA:-fineweb_edu}"
CALIB_TOKENS="${CALIB_TOKENS:-2097152}"
BITS_LIST="${BITS_LIST:-3 4 8}"
BITS_WEIGHTS="${BITS_WEIGHTS:-1 1 1}"
MASTER_BITWIDTH="${MASTER_BITWIDTH:-8}"
GROUP_SIZE="${GROUP_SIZE:-128}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

export PYTHONPATH="$MATGPTQ_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR/weights"

torchrun --nnodes=1 --nproc-per-node="$NPROC" --master_port "$MASTER_PORT" "$MATGPTQ_DIR/quant.py" \
  --model_name_or_path "$MODEL_ID" \
  --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
  --pre_block_modules model.embed_tokens \
  --block_modules model.layers \
  --post_block_modules model.norm lm_head \
  --calibration_data "$CALIB_DATA" \
  --calibration_tokens "$CALIB_TOKENS" \
  --calibration_sequence_length "$SEQUENCE_LENGTH" \
  --method matgptq \
  --master_bitwidth "$MASTER_BITWIDTH" \
  --bitwidth_options $BITS_LIST \
  --bitwidth_weights $BITS_WEIGHTS \
  --group_size "$GROUP_SIZE" \
  --perchannel \
  --sym \
  --verbose \
  --dtype float16 \
  --attn_implementation eager \
  --save_dir "$OUT_DIR/weights"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

MATGPTQ_DIR="${MATGPTQ_DIR:-$HOME/MatGPTQ}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/runs}"
MODEL_SHORT="${MODEL_SHORT:-${MODEL_ID##*/}}"
QUANT_WEIGHTS="${QUANT_WEIGHTS:-$RUN_DIR/weights/$MODEL_SHORT}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MASTER_BITWIDTH="${MASTER_BITWIDTH:-8}"
SLICE_BITS="${SLICE_BITS:-3 4 8}"

export PYTHONPATH="$MATGPTQ_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_DIR"

mkdir -p "$RUN_DIR/ppl"

for bit in $SLICE_BITS; do
  log="$RUN_DIR/ppl/ppl_${MODEL_SHORT}_${bit}bit.log"
  python "$MATGPTQ_DIR/eval_ppl.py" \
    --model_name_or_path "$MODEL_ID" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --eval_datasets $EVAL_DATASETS \
    --eval_tokens "$EVAL_TOKENS" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --method matgptq \
    --quant_weights_path "$QUANT_WEIGHTS" \
    --quant_master_bitwidth "$MASTER_BITWIDTH" \
    --quant_uniform_bitwidth "$bit" \
    --dtype float16 \
    --attn_implementation eager \
    2>&1 | tee "$log"
done

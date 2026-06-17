# MatGPTQ Experiment

This repo can run MatGPTQ without editing `~/MatGPTQ`. The scripts are path-safe for the lab server path:

```text
/nfs/home/s314511048/query_mp
```

## Quantize

```bash
cd /nfs/home/s314511048/query_mp
GPU_DEVICE=5 \
bash experiments/h1-query-token-sensitivity/run_matgptq_quant.sh
```

Outputs go to:

```text
experiments/h1-query-token-sensitivity/runs/weights/
```

The quantization wrapper defaults to MatGPTQ CPU offload so Llama-3.1-8B can run on a 24 GB GPU. For a quicker smoke run:

```bash
GPU_DEVICE=5 CALIB_TOKENS=131072 \
bash experiments/h1-query-token-sensitivity/run_matgptq_quant.sh
```

## Evaluate Uniform Slices

```bash
cd /nfs/home/s314511048/query_mp
GPU_DEVICE=5 \
bash experiments/h1-query-token-sensitivity/run_matgptq_ppl_slices.sh
```

Logs go to:

```text
experiments/h1-query-token-sensitivity/runs/ppl/
```

## Run H1 End-to-End

This runs every H1 step that produces evidence for the query-token-layer hypothesis:

1. Quantize one MatGPTQ parent over the requested bit set.
2. Evaluate uniform PPL slices.
3. Collect per-token, per-module sensitivity rows by swapping one module to a lower bit.
4. Train layer-only and dynamic-feature predictors.
5. Produce sensitivity tables, calibration data, and a routing proxy curve.

```bash
cd /nfs/home/s314511048/query_mp
GPU_DEVICE=5 \
BITS_LIST="2 3 4 6 8" \
BITS_WEIGHTS="1 1 1 1 1" \
SLICE_BITS="2 3 4 6 8" \
LOW_BITS="2 3 4 6" \
EVAL_DATASETS=c4 \
COLLECT_DATASET=c4 \
SEQUENCE_LENGTH=512 \
EVAL_TOKENS=8192 \
MAX_SAMPLES=8 \
bash experiments/h1-query-token-sensitivity/run_h1_full.sh
```

For a smoke run that only checks the pipeline shape:

```bash
SKIP_QUANT=1 SKIP_PPL=1 MAX_SAMPLES=1 MAX_MODULES=2 LOW_BITS="3 4" \
bash experiments/h1-query-token-sensitivity/run_h1_full.sh
```

Main outputs:

```text
experiments/h1-query-token-sensitivity/runs/h1_sensitivity/sensitivity.csv
experiments/h1-query-token-sensitivity/runs/h1_sensitivity/analysis/summary.md
experiments/h1-query-token-sensitivity/runs/h1_sensitivity/analysis/predictor_metrics.json
experiments/h1-query-token-sensitivity/runs/h1_sensitivity/analysis/routing_proxy.csv
```

What each output answers:

- `sensitive_modules.csv` / `sensitive_layers.csv`: which layer/module is sensitive.
- `sensitive_token_positions.csv`: which token positions are sensitive.
- `sensitive_categories.csv`: which query categories are sensitive, if `--prompts_jsonl` is used.
- `predictor_metrics.json`: whether runtime features beat a layer-only baseline.
- `routing_proxy.csv`: whether dynamic routing improves the quality/bit proxy frontier.

## Useful Overrides

```bash
cd /nfs/home/s314511048/query_mp
MODEL_ID=/path/or/hf-id \
MATGPTQ_DIR=$HOME/MatGPTQ \
GPU_DEVICE=5 \
CALIB_TOKENS=131072 \
EVAL_TOKENS=65536 \
SLICE_BITS="3 4 8" \
bash experiments/h1-query-token-sensitivity/run_matgptq_quant.sh
```

`~/MatGPTQ` is read as code only. Generated weights, logs, and configs stay here.

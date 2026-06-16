# Local MatGPTQ Experiment

This repo can run MatGPTQ without editing `~/MatGPTQ`.

## Quantize

```bash
bash experiments/h1-query-token-sensitivity/run_matgptq_quant.sh
```

Outputs go to:

```text
experiments/h1-query-token-sensitivity/runs/weights/
```

## Evaluate Uniform Slices

```bash
bash experiments/h1-query-token-sensitivity/run_matgptq_ppl_slices.sh
```

Logs go to:

```text
experiments/h1-query-token-sensitivity/runs/ppl/
```

## Useful Overrides

```bash
MODEL_ID=/path/or/hf-id \
CALIB_TOKENS=131072 \
EVAL_TOKENS=65536 \
SLICE_BITS="3 4 8" \
bash experiments/h1-query-token-sensitivity/run_matgptq_quant.sh
```

`~/MatGPTQ` is read as code only. Generated weights, logs, and configs stay here.

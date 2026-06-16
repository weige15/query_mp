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

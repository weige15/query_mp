# Public Codebases To Try

Date: 2026-06-15

## Recommendation

The best starting point depends on whether the immediate goal is scientific signal or faithful any-precision execution.

For this project, I would start with:

1. **MatGPTQ** if you have a recent CUDA environment and want a modern sliceable/mixed-precision substrate.
2. **Any-Precision LLM** if you specifically want to build on the ICML 2024 any-precision representation.
3. **GPTQModel** if you want the fastest path to practical sensitivity tracing and static mixed-precision baselines.

Then use **lm-evaluation-harness** for downstream task evaluation.

## Best Fits

### 1. MatGPTQ

Repository: https://github.com/IST-DASLab/MatGPTQ

Why it is useful:

- Implements post-training Matryoshka/GPTQ quantization.
- Produces a single parent model optimized for multiple target precisions.
- Includes `eval_ppl.py`, `lmeval.py`, `quant.py`, and `evo_quant_search.py`.
- Includes custom kernels and a vLLM plugin path.
- Good substrate for testing dynamic bit-slice or layer-wise routing.

Best use in this project:

- Build static mixed-precision and sliceable-precision baselines.
- Add logging for per-layer/per-token sensitivity.
- Add an experimental router that changes `inference_bitwidth` or per-layer configuration by token/query.

Main caveat:

- This is newer code and expects a modern CUDA/PyTorch stack. The README says it was tested with CUDA 12.4 and PyTorch 2.7.1.

### 2. Any-Precision LLM

Repository: https://github.com/SNU-ARC/any-precision-llm

Why it is useful:

- Official implementation for Any-Precision LLM.
- Supports bitplane-based any-precision weight representation.
- Includes `quantize.py`, `run_eval.py`, `evaluate.sh`, `demo.py`, and custom CUDA kernels.
- Directly matches the storage substrate assumed by DP-LLM-style runtime adaptation.

Best use in this project:

- Reproduce uniform any-precision baselines.
- Add a layer/token router on top of the any-precision execution path.
- Compare uniform precision, static layer precision, and dynamic precision at matched average bits.

Main caveats:

- Requires CUDA Toolkit 12 or higher and gcc 9 or higher.
- Quantization needs roughly 2x FP16 model size in RAM, VRAM, and disk for larger models.

### 3. GPTQModel

Repository: https://github.com/ModelCloud/GPTQModel

Why it is useful:

- Practical quantization toolkit with GPTQ, AWQ, GGUF, FP8, EXL3, and other formats.
- Supports HF, vLLM, SGLang, CUDA, ROCm, XPU, CPU, and Apple Silicon paths.
- Exposes dynamic per-module mixed quantization control.
- Has simple APIs for quantization and inference.

Best use in this project:

- Generate 3-bit, 4-bit, 5-bit, FP8, or mixed baselines quickly.
- Instrument layer outputs and logits to collect sensitivity traces.
- Build a first H1 experiment without needing true any-precision kernels.

Main caveat:

- It is not a DP-LLM/MoBiQuant implementation. It is a strong infrastructure base for baselines and instrumentation.

### 4. llm-awq

Repository: https://github.com/mit-han-lab/llm-awq

Why it is useful:

- Official AWQ codebase.
- Stable baseline for activation-aware W4A16 quantization.
- Includes model zoo/search results and real CUDA kernels.
- Good for static quantization baselines.

Best use in this project:

- Compare adaptive routing against a strong static 4-bit baseline.
- Use activation-aware saliency as one router feature.

Main caveat:

- AWQ intentionally avoids hardware-inefficient mixed precision, so it is more a baseline than a dynamic routing substrate.

### 5. lm-evaluation-harness

Repository: https://github.com/EleutherAI/lm-evaluation-harness

Why it is useful:

- Standard downstream evaluation framework.
- Supports Hugging Face, vLLM, GPTQModel, AutoGPTQ, and many tasks.
- Good for PIQA, HellaSwag, ARC, WinoGrande, GSM8K, LongBench/RULER extras, and custom tasks.

Best use in this project:

- Keep evaluation comparable and reproducible.
- Run the same benchmark set across FP16, uniform quantization, static mixed precision, and adaptive routing.

## Seed-Paper Code Availability

### FlexQuant

Repository: https://github.com/ZongwuWang/FlexQuant

Status:

- Public, but at inspection time it contained only a license and no runnable implementation.

Use:

- Cite/read as a method reference, but do not depend on the repo for experiments yet.

### DP-LLM

Paper: https://arxiv.org/abs/2508.06041

Status:

- I did not find an official public GitHub implementation.

Use:

- Implement its core idea on top of Any-Precision LLM or MatGPTQ if needed: collect low/high error estimates, learn thresholds, and route low/high precision per layer.

### MoBiQuant

Paper: https://arxiv.org/abs/2602.20191

Status:

- I did not find an official public GitHub implementation.

Use:

- Use it as conceptual guidance for token-aware residual bit-slice routing.

### MixKVQ

Paper: https://arxiv.org/abs/2512.19206

Status:

- I did not find an official public GitHub implementation.

Use:

- Treat as a second-stage KV-cache direction after weight precision routing works.

### Progressive Mixed-Precision Decoding

Paper: https://arxiv.org/abs/2410.13461

Status:

- I did not find a public implementation.

Use:

- Useful baseline idea: progressive precision lowering during decoding.

## Suggested First Runnable Experiment

Start with **GPTQModel** or **MatGPTQ**, not DP-LLM/MoBiQuant directly.

Minimum experiment:

1. Pick a small model: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-1.5B`, or a Pythia model.
2. Produce or load 4-bit and higher-bit variants.
3. Run the same prompts through low/high precision.
4. Log per-token entropy, top-token margin, selected layer output norms, and logit KL.
5. Train a simple classifier for "high precision needed".
6. Compare:
   - uniform low precision,
   - uniform high precision,
   - static layer mixed precision,
   - query/token/layer router.

The first useful artifact is not a full router. It is a sensitivity trace dataset that proves whether routing is predictable.

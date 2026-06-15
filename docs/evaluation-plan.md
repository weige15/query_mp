# Evaluation Plan

Date: 2026-06-15

## Goal

Evaluate whether query-token adaptive precision routing improves the quality-efficiency tradeoff of LLM inference compared with uniform and static mixed-precision baselines.

## Primary Metric

The primary metric is Pareto improvement:

> For a fixed quality degradation, does the adaptive router use lower average effective bits, lower memory, or lower latency than baselines?

Report all three axes:

- Quality: perplexity or downstream accuracy.
- Compression: average effective bits and peak memory.
- Runtime: prefill latency, decode ms/token, throughput, and router overhead.

## Pilot Benchmarks

Start with cheap, repeatable benchmarks:

- WikiText-2 perplexity.
- C4 perplexity subset.
- PIQA, HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande via lm-evaluation-harness.

Then add harder stress tests:

- GSM8K or MATH subset for reasoning.
- HumanEval or MBPP subset for code if model supports code.
- LongBench or synthetic retrieval for long-context KV-cache routing.
- Prompt categories with rare tokens, high entropy, or long contexts.

## Baselines

### Reference

- FP16 or BF16 model.

### Uniform Quantization

- Uniform 3-bit, 4-bit, and 5-bit if supported.
- Uniform any-precision slices if using Any-Precision or Matryoshka-style quantization.

### Static Mixed Precision

- Static layer allocation from calibration sensitivity.
- Static layer allocation from average per-layer output error.
- Static layer allocation under the same average effective bit budget as the adaptive router.

### Runtime Adaptive Baselines

If feasible:

- Entropy/KL threshold routing.
- DP-LLM-style low/high selector.
- MoBiQuant-style token-aware bit-slice selection.

If not feasible, include simplified approximations and clearly label them.

## Ablations

Router feature ablations:

- Layer-only.
- Token + layer.
- Query + layer.
- Query + token + layer.
- Add runtime budget signal.
- Add activation outlier features.
- Add logit entropy / margin features.

Granularity ablations:

- Request-level precision.
- Token-level uniform precision.
- Layer-level static precision.
- Layer x token dynamic precision.
- KV-cache-only routing.
- Joint weight + KV routing.

Overhead ablations:

- Simulated routing with oracle schedule.
- Replay schedule without online router.
- Online router with measured feature extraction.
- Online router with full precision switching.

## Sensitivity Labels

Possible labels for training or evaluating the router:

- Layer output error: norm difference between low-bit and high-bit layer outputs.
- Logit KL: KL divergence between low-bit and high-bit logits after a token.
- Rank flip: whether the top token or top-k set changes.
- Loss delta: next-token negative log likelihood difference.
- Task flip: whether final answer correctness changes.

Use cheap labels first. Task-flip labels are more expensive and noisier but more semantically meaningful.

## Reporting Standard

Every result should report:

- Model checkpoint and quantization method.
- Calibration dataset and size.
- Evaluation dataset and size.
- Average effective bitwidth.
- Peak memory.
- Latency and hardware.
- Router overhead as absolute time and percent of decode time.
- Random seed.
- Whether the run is confirmatory or exploratory.

## Success Criteria

A strong pilot result:

- Adaptive routing matches static mixed precision quality with at least 5-10 percent lower average effective bits, or
- Adaptive routing improves perplexity/accuracy at matched average effective bits, and
- Router overhead is small enough that end-to-end latency is not worse.

A negative but useful result:

- Query/token features do not predict sensitivity better than static layer identity.
- Router overhead dominates.
- Gains appear only in simulation and disappear with real kernels.

Any of these outcomes should be documented because they decide whether to deepen or pivot.

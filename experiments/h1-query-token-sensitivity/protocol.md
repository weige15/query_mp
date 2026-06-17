# H1 Protocol: Query-Token-Layer Sensitivity Is Predictable

Status: planned
Type: confirmatory once run
Created: 2026-06-15

## Hypothesis

Per-layer quantization sensitivity varies systematically with query type and decoding position, and can be predicted better than a static layer average using cheap activation/logit features.

## Motivation

If sensitivity is not predictable from runtime features, adaptive precision routing is unlikely to be useful. This experiment tests the foundation of the research direction before building a full router.

## Setup

Model:

- Primary: Llama-3.1-8B-Instruct with the existing MatGPTQ 3/4/8-bit slices.
- Replication: at least one second open model family or size when compute allows.
- Reference: FP16/BF16 if feasible, otherwise 8-bit is only a high-precision proxy and must be labeled that way.

Precision variants:

- FP16/BF16 reference.
- Uniform 8-bit, 4-bit, and 3-bit slices.
- Static layer-wise mixed precision under matched average-bit budgets.
- Dynamic schedules for request-level, token-level, layer-level, and layer x token precision.

Datasets:

- WikiText-2 and C4 for language modeling.
- PIQA, HellaSwag, ARC-Easy, ARC-Challenge, and WinoGrande for broad downstream behavior.
- GSM8K or MATH subset for reasoning sensitivity.
- HumanEval or MBPP subset for code sensitivity if the model supports code generation.
- LongBench or synthetic retrieval for long-context and KV-cache follow-up.
- Categorized prompt set with commonsense, math, code, long-context retrieval, rare-token, and simple factual prompts.

## Measurements

For each query, generated token, and selected layer/module:

- Activation norm.
- Activation outlier score.
- Logit entropy or margin.
- Low/high layer output difference.
- Low/high logit KL divergence.
- Next-token loss delta.
- Rank flip and top-k set change.
- Task-level answer flip where an automatic checker exists.
- Query category and decoding position.

Collect enough examples per category to support confidence intervals and held-out evaluation. Do not treat aggregate perplexity alone as evidence for H1.

## Prediction

Dynamic features will predict high-sensitivity events better than a static per-layer average. Expected ranking:

1. Query + token + layer features.
2. Token + layer features.
3. Query + layer features.
4. Layer-only static average.

## Success Criteria

- Dynamic features improve sensitivity prediction over layer-only static baselines on held-out prompts, held-out documents, and at least one task family beyond language modeling.
- The improvement remains under matched average-bit budgets and includes confidence intervals or bootstrap uncertainty.
- Sensitivity plots show nontrivial variation across token position, layer, prompt category, and task family.
- A routing policy built from those features improves the quality/bit Pareto frontier versus uniform and static mixed precision, not just versus 3-bit.
- Router overhead is measured end to end, including feature extraction and precision-switching cost.

## Failure Criteria

- Static layer identity explains nearly all useful sensitivity variation.
- Dynamic features are too noisy or too expensive to collect.
- Gains appear only on WikiText2/C4 aggregate perplexity and disappear on downstream tasks or held-out prompt categories.
- Simulated routing gains disappear once real runtime overhead is included.

## Analysis

Produce:

- Heatmap: layer x token sensitivity.
- Box plots by prompt category.
- Feature importance table.
- ROC/PR curve for high-sensitivity event prediction.
- Calibration curves for predicted sensitivity.
- Held-out generalization table across datasets, task families, and model/checkpoint variants.
- Pareto curves for uniform precision, static mixed precision, oracle schedules, replayed schedules, and online routing.
- Tail-risk analysis: worst decile prompts, answer flips, and categories where low precision fails.

## Notes

The existing MatGPTQ uniform-slice run is baseline evidence only: WikiText2 PPL is 7.09 at 8-bit, 7.30 at 4-bit, and 9.51 at 3-bit. It shows 4-bit is strong and 3-bit is risky, but it does not test query-token-layer predictability.

This protocol should be locked before the next confirmatory execution if a functional git repository becomes available.

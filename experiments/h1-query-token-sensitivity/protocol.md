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

- Prefer an open model in the 1B-8B range depending on hardware.
- Start smaller if only CPU is available.

Precision variants:

- High precision: FP16/BF16 or higher-bit quantized path.
- Low precision: 3-bit or 4-bit weight-only path.

Datasets:

- WikiText-2 or C4 subset for language modeling.
- Small categorized prompt set: commonsense, math, code, long-context retrieval, simple factual prompts.

## Measurements

For each query, generated token, and selected layer/module:

- Activation norm.
- Activation outlier score.
- Logit entropy or margin.
- Low/high layer output difference.
- Low/high logit KL divergence.
- Next-token loss delta.
- Query category and decoding position.

## Prediction

Dynamic features will predict high-sensitivity events better than a static per-layer average. Expected ranking:

1. Query + token + layer features.
2. Token + layer features.
3. Query + layer features.
4. Layer-only static average.

## Success Criteria

- A simple classifier/regressor using dynamic features improves sensitivity prediction over static layer baselines.
- Sensitivity plots show nontrivial variation across token position and query category.

## Failure Criteria

- Static layer identity explains nearly all useful sensitivity variation.
- Dynamic features are too noisy or too expensive to collect.

## Analysis

Produce:

- Heatmap: layer x token sensitivity.
- Box plots by prompt category.
- Feature importance table.
- ROC/PR curve for high-sensitivity event prediction.

## Notes

No experiment has been run yet. This protocol should be locked before execution if a functional git repository becomes available.

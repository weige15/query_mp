# H3 Protocol: Joint Weight and KV-Cache Precision Routing

Status: planned
Type: exploratory until H1/H2 are validated
Created: 2026-06-15

## Hypothesis

Joint routing over weight precision and KV-cache precision improves long-context reasoning efficiency more than optimizing either component independently.

## Motivation

For short-context inference, weight memory and compute often dominate. For long-context inference, KV cache becomes a major memory and bandwidth cost. Query-aware KV-cache methods suggest relevance matters; weight routers suggest token/layer sensitivity matters. A shared budget allocator may outperform separate policies.

## Setup

Datasets:

- LongBench subset.
- Synthetic needle-in-a-haystack retrieval.
- Long-context QA with distractors.

Baselines:

- Weight-only adaptive routing.
- KV-only adaptive routing.
- Independent weight and KV routing.
- Joint routing under a shared memory budget.

## Prediction

Joint routing helps most when:

- Context length is long.
- Only a small part of the context is query-relevant.
- The answer requires reasoning over retrieved evidence.

## Success Criteria

- Joint routing improves answer accuracy or perplexity under the same memory budget.
- Query relevance features explain which KV channels/tokens receive higher precision.

## Failure Criteria

- KV routing dominates and weight routing adds little.
- Weight routing dominates and KV routing adds little.
- Joint policy overhead is too high for long-context decoding.

## Notes

This is a second-stage direction. Do not start here until H1/H2 establish that dynamic routing is worthwhile.

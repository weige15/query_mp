# H2 Protocol: Lightweight Precision Router Beats Static Mixed Precision

Status: planned
Type: confirmatory once run
Created: 2026-06-15

## Hypothesis

A lightweight precision router conditioned on query-level difficulty, token-level uncertainty, layer identity, and activation statistics can match static mixed precision quality at lower average effective bits.

## Motivation

This is the core algorithmic claim. It tests whether predictability from H1 can become an actual inference-efficiency improvement.

## Router Candidates

Start simple:

- Threshold on activation outlier score.
- Threshold on entropy or logit margin.
- Logistic regression over query-token-layer features.
- Tiny MLP only if linear/rule-based routers are insufficient.

Avoid reinforcement learning in the first pilot unless simpler routers fail.

## Baselines

- FP16/BF16 reference.
- Uniform low precision.
- Uniform high precision.
- Static layer-wise mixed precision at the same average effective bitwidth.
- Oracle dynamic routing using sensitivity labels, as an upper bound.

## Budget Matching

All comparisons must be matched by average effective bitwidth or latency target. Do not compare a dynamic method that simply spends more precision.

## Prediction

The router will improve perplexity or downstream accuracy at the same average effective bitwidth, especially for heterogeneous prompt sets containing both easy and hard queries.

## Success Criteria

- Adaptive router improves the Pareto frontier over static mixed precision.
- Router overhead is measured and small enough that end-to-end latency is not worse.
- Ablations show which features matter.

## Failure Criteria

- Gains vanish after measuring routing and switching overhead.
- Router only helps on calibration prompts and fails out of distribution.
- Static mixed precision is effectively as good as dynamic routing.

## Analysis

Produce:

- Pareto plot: quality delta versus effective bits.
- Latency plot: ms/token including router overhead.
- Feature ablation table.
- Failure-case table for prompts where adaptive routing under-allocates precision.

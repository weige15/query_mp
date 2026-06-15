# Research Direction Brief

Date: 2026-06-15

## What You Are Researching

You are researching query-token adaptive precision routing for LLM inference.

The concrete question:

> Can an LLM dynamically decide which layers, tokens, or cache components need higher numerical precision for the current query, and use lower precision everywhere else without losing output quality?

## Why This Is A Good Narrowing

The seed papers are not just about mixed precision. They are about adaptation:

- DP-LLM: layer precision changes during decoding.
- MoBiQuant: token sensitivity controls how many bit slices are used.
- QAQ: the title suggests query-level adaptation, but the exact public paper could not be verified.

The shared direction is runtime precision choice.

## Main Claim To Test

> Precision demand in LLM inference is sparse, structured, and predictable from cheap query-token-layer features.

If true, a router can spend high precision only on sensitive events.

## First Experiment

Before building a full method, measure sensitivity:

1. Run high-precision and low-precision paths.
2. Log layer output error, logit KL, entropy, activation outliers, and query category.
3. Test whether dynamic features predict high-sensitivity events better than static layer averages.

## Best First Paper Shape

1. Show dynamic sensitivity exists.
2. Build a lightweight router.
3. Beat static mixed precision at matched average bitwidth.
4. Measure router overhead and failures.

## Main Risk

Adaptive routing may look good in bitwidth metrics but fail in actual latency if routing and precision switching are too expensive. The evaluation must include real overhead.

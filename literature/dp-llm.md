# DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment

URL: https://arxiv.org/abs/2508.06041

## Citation Metadata

- Authors: Sangwoo Kwon, Seong Hoon Seo, Jae W. Lee, Yeonhong Park.
- Year: 2025.
- Status: arXiv preprint.

## Core Idea

DP-LLM assigns precision dynamically at runtime for each layer during decoding. It uses the observation that layer sensitivity to quantization changes across decoding iterations. Each linear layer gets a lightweight precision selector that estimates whether low precision will introduce too much relative error for the current input.

## Relevance

This is the strongest seed for the current research direction. It shifts mixed precision from an offline allocation problem to an online decision problem.

## What It Suggests For This Project

- Use layer x token as the first practical routing granularity.
- Treat relative error or output-difference proxies as candidate sensitivity labels.
- Evaluate whether a simpler or more query-aware router can improve over layer-local selection.

## Open Questions For Follow-Up

- Does query type explain when DP-LLM's layer choices change?
- Can query-level features predict thresholds before generation starts?
- How much of the gain comes from dynamic routing versus improved calibration?
- What is the measured router overhead on real hardware?

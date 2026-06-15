# RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference

URL: https://arxiv.org/abs/2603.17891

## Citation Metadata

- Authors: Arpit Singh Gautam, Saurabh Jha.
- Year: 2026.
- Status: arXiv preprint.

## Core Idea

RAMP frames per-layer bitwidth allocation as a sequential decision problem and uses Soft Actor-Critic to learn a mixed-precision allocation under a global bit budget. It emphasizes transfer through normalized layer embeddings.

## Relevance

RAMP is most useful as a contrast case. It is adaptive during search/calibration, but the resulting allocation is still primarily static during inference. Its feature design is relevant for transfer across models.

## What It Suggests For This Project

- Include a static learned-policy baseline.
- Use normalized structural layer features rather than raw layer ids.
- Treat transferability as a secondary research question.

## Open Questions For Follow-Up

- Can RAMP-style structural features be combined with query-token features?
- Does dynamic routing outperform a strong learned static policy?
- Is reinforcement learning necessary, or do simpler routers perform similarly?

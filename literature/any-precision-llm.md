# Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs

URL: https://arxiv.org/abs/2402.10517

## Citation Metadata

- Authors: Yeonhong Park, Jake Hyun, SangLyul Cho, Bonggeun Sim, Jae W. Lee.
- Year: 2024.
- Status: ICML 2024 / arXiv.

## Core Idea

Any-Precision LLM stores multiple quantized bitwidth variants in a memory-efficient overlaid representation. This lets a single deployed LLM support multiple precision levels without storing multiple full checkpoints.

## Relevance

Any-precision storage makes runtime precision adaptation practical. Without a multi-precision representation, a router may require too much memory.

## What It Suggests For This Project

- Separate the storage problem from the routing problem.
- Use any-precision or nested-bit quantization as the substrate if available.
- The research contribution can be the policy for selecting precision rather than the quantizer itself.

## Open Questions For Follow-Up

- How should precision be selected under a latency target rather than just a memory target?
- Can any-precision models support non-uniform layer-wise or token-wise effective bitwidths efficiently?

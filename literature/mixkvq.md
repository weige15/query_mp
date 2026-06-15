# MixKVQ: Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning

URL: https://arxiv.org/abs/2512.19206

## Citation Metadata

- Authors: Tao Zhang, Ziqian Zeng, Hao Peng, Huiping Zhuang, Cen Chen.
- Year: 2025.
- Status: arXiv preprint.

## Core Idea

MixKVQ argues that effective low-bit KV-cache quantization should consider both intrinsic quantization difficulty and relevance to the current query. It preserves critical key channels in higher precision while quantizing values per token.

## Relevance

This paper brings query-awareness into mixed precision, but for KV cache rather than weights. It is highly relevant if the project expands from weight precision routing to long-context memory routing.

## What It Suggests For This Project

- Query relevance can be a real precision signal.
- Long-context reasoning is a good stress setting because not all cached information is equally relevant.
- A future method could allocate a shared precision budget across weights and KV cache.

## Open Questions For Follow-Up

- Does query relevance also predict weight precision demand?
- How should the system trade off preserving KV cache precision versus increasing weight precision?
- Can query-aware KV routing and token-aware weight routing share features?

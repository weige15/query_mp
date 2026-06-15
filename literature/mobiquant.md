# MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM

URL: https://arxiv.org/abs/2602.20191

## Citation Metadata

- Authors: Dongwei Wang, Jinhee Kim, Seokho Han, Denis Gudovskiy, Yohei Nakata, Tomoyuki Okuno, KhayTze Peong, Kang Eun Jeon, Jong Hwan Ko, Yiran Chen, Huanrui Yang.
- Year: 2026.
- Status: arXiv preprint.

## Core Idea

MoBiQuant uses a mixture-of-bits framework for any-precision LLM inference. It introduces recursive residual quantization so higher precision can be reconstructed from additional bit slices, and a token-aware router selects how many bit slices to use based on token sensitivity.

## Relevance

MoBiQuant directly supports the idea that precision demand is token-dependent. It also gives a concrete implementation pattern: a router over residual bit slices rather than a fully separate model for each precision.

## What It Suggests For This Project

- Token sensitivity is a key signal, not just layer sensitivity.
- Routing over bit slices may be more efficient than switching independent quantized weights.
- Outlier migration across precisions is a likely mechanism to study.

## Open Questions For Follow-Up

- Does the token-aware router benefit from query-level context?
- Is the router interpretable enough to predict failure cases?
- Can similar routing be extended to KV cache or activations?
- Does token-adaptive bit selection remain efficient with real kernels?

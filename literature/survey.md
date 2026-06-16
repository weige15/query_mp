# Literature Survey: Adaptive Precision for LLM Inference

Date: 2026-06-15

## Scope

This survey focuses on literature relevant to query-token adaptive mixed-precision inference for LLMs. It intentionally does not cover every quantization paper. The goal is to position a narrow research direction around runtime precision assignment.

## Search Notes

Seed papers from the user:

- DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment.
- MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM.
- QAQ: Query-adaptive Mixed-precision Quantization for Large Language Models.

Verification status:

- DP-LLM: verified on arXiv, arXiv:2508.06041.
- MoBiQuant: verified on arXiv, arXiv:2602.20191.
- QAQ exact title: available locally as `QAQ.pdf`; public indexing was not found in web search on 2026-06-15/2026-06-16.

## Key Papers

| Paper | Year | Relevance |
|---|---:|---|
| Any-Precision LLM | 2024 | Multi-bitwidth deployment in one overlaid model representation. |
| Matryoshka Quantization | 2025 | Nested integer bit structure for serving multiple precisions from one model. |
| FlexQuant | 2025 | Dynamic precision switching during token generation using uncertainty/divergence signals. |
| DP-LLM | 2025 | Runtime layer-wise precision assignment based on estimated relative error. |
| MixKVQ | 2025 | Query-aware mixed precision for KV cache in long-context reasoning. |
| QAQ | 2025 | Query-conditioned block/layer bit-width routing with bit-plane weights and CPU/GPU on-demand loading. |
| MoBiQuant | 2026 | Token-aware mixture-of-bits routing for any-precision LLMs. |
| RAMP | 2026 | Learned static per-layer mixed precision with transfer claims. |
| TileFuse | 2026 | Hardware/kernel reality for mixed-precision quantized inference. |

## Synthesis

The literature suggests a field transition:

1. Static low-bit PTQ established that 3-4 bit LLM inference can work for many models.
2. Any-precision and Matryoshka-style methods make multiple precisions available from one representation.
3. Static mixed precision uses layer sensitivity to allocate bitwidths, but still fixes those choices.
4. Runtime-adaptive methods argue that precision demand changes by token, layer, query, and context.

The research gap is not simply "better quantization". It is:

> Better online policies for deciding when and where to spend precision.

## Gaps

### Gap 1: Beyond Query-Level Adaptation

QAQ already targets query-level adaptation for block/layer precision. DP-LLM and MoBiQuant are strongly token/layer oriented, while MixKVQ introduces query relevance for KV cache. The remaining question is not whether query adaptation is possible, but whether hierarchical query-token-layer routing improves over QAQ-style query-only routing.

### Gap 2: Unified Weight and KV Precision

Most methods target weights or KV cache separately. Long-context workloads may require a budget split between weights and KV cache based on query/context relevance.

### Gap 3: Router Overhead and Kernel Costs

Adaptive precision can be attractive in average-bit metrics but poor in end-to-end latency if switching and routing are costly. Hardware-aware evaluation is mandatory.

### Gap 4: Failure Risk

Average perplexity can hide the prompts where under-allocation causes severe answer errors. Query-wise failure analysis should be part of the contribution.

## Working Research Direction

Query-token adaptive precision routing:

- Build sensitivity traces across query, token, layer, and possibly KV-cache channel.
- Train or design cheap routers.
- Compare against static mixed precision under matched budgets.
- Measure overhead and tail failures.

## Source URLs

- QAQ: local file `QAQ.pdf`
- Any-Precision LLM: https://arxiv.org/abs/2402.10517
- Matryoshka Quantization: https://arxiv.org/abs/2502.06786
- FlexQuant: https://arxiv.org/abs/2506.12024
- DP-LLM: https://arxiv.org/abs/2508.06041
- MixKVQ: https://arxiv.org/abs/2512.19206
- MoBiQuant: https://arxiv.org/abs/2602.20191
- RAMP: https://arxiv.org/abs/2603.17891
- TileFuse: https://arxiv.org/abs/2606.11357

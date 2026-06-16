# QAQ Positioning

Date: 2026-06-16

## What QAQ Actually Covers

The local paper `QAQ.pdf` proposes Query-Adaptive Quantization for LLM inference. Its main components are:

- Bit-plane decomposition of model weights.
- A trainable query-conditioned router.
- Block/layer-wise bit-width selection.
- CPU-to-GPU on-demand loading of selected bit-planes.

In its current reported results, QAQ matches static 8-bit accuracy on Qwen3 and LLaMA-3.1 style evaluations, lowers GPU memory when on-demand loading is enabled, and pays a latency cost from synchronous transfers.

## Alignment With This Project

QAQ aligns strongly with the project objective, but only with one axis:

> Query-level adaptive precision.

The broader project objective remains:

> Query-token adaptive precision routing under explicit runtime budgets.

That broader objective includes query-level routing, token-level routing, layer/block routing, runtime budget constraints, failure-risk control, and possibly KV-cache precision.

## What Not To Claim As Novel

Do not claim these as the sole novelty:

- Query-conditioned mixed-precision quantization.
- Block/layer-wise bit-width routing from query features.
- Bit-plane based flexible precision.
- CPU/GPU on-demand loading of selected bit-planes.

QAQ already covers those at least at a workshop-paper level.

## Strong Novelty Directions

### 1. Hierarchical Query-Token-Layer Routing

QAQ selects precision from query-conditioned features. Extend this into a two-level policy:

1. A query router predicts the budget and risk class.
2. A token/layer router spends the budget dynamically during decoding.

Novel claim:

> Query-level adaptation sets the budget, but token-level runtime sensitivity decides where the budget is spent.

### 2. Risk-Aware Precision Escalation

QAQ optimizes average accuracy/memory/latency. Add failure-risk control:

- Detect high-risk prompts.
- Escalate precision when uncertainty, OOD score, or low/high-bit disagreement crosses a threshold.
- Report query-wise catastrophic failures, not only average benchmarks.

Novel claim:

> Adaptive precision should optimize tail correctness, not just average effective bit-width.

### 3. Predictive Prefetching For QAQ-Style Loading

QAQ's on-demand loading saves memory but increases latency due to synchronous transfers. A strong systems contribution is:

- Predict future bit-plane needs.
- Prefetch asynchronously.
- Cache high-value bit-planes across requests.
- Constrain router actions to schedules that can be prefetched.

Novel claim:

> Dynamic precision routing needs a predictive memory scheduler; otherwise memory savings become latency regressions.

### 4. Joint Weight And KV-Cache Precision Budgeting

QAQ routes weight precision. MixKVQ routes KV-cache precision. Combine them:

- Query router decides whether the budget should go to weights, KV cache, or both.
- Long-context tasks become the primary evaluation.

Novel claim:

> In long-context inference, the optimal precision target is query-dependent: sometimes weights matter, sometimes cached context matters.

### 5. Sensitivity Dataset And Mechanistic Analysis

Before building a complex router, collect traces:

- Query category.
- Token position.
- Layer/block.
- Activation outliers.
- Entropy and logit margin.
- Low/high-bit disagreement.
- Task correctness flips.

Novel claim:

> Precision demand is sparse, structured, and predictable, but the useful predictors differ by task and decoding phase.

## Recommended Positioning

The safest next project title is:

> Budget-Aware Hierarchical Precision Routing for LLM Inference

Two-sentence pitch:

> QAQ shows that query-conditioned precision selection is feasible, but it treats precision mostly as a per-query block/layer decision and pays latency overhead under on-demand loading. We extend this into a budget-aware hierarchical router that assigns a query-level precision budget, spends it token-by-token and layer-by-layer, and uses risk-aware escalation or predictive prefetching to preserve tail accuracy and real latency.

# Research Direction: Query-Token Adaptive Precision Routing for Elastic LLM Inference

Date: 2026-06-15

## Short Answer

You are researching adaptive precision routing for LLM inference.

More exactly:

> You are studying how to choose the numerical precision of LLM computation at runtime, conditioned on the current query, generated token, model layer, and resource budget, so the model uses high precision only when it matters for output quality.

The central scientific object is not the quantized model itself. It is the policy that decides precision.

## Two-Sentence Pitch

LLM quantization usually compresses every request with a fixed bitwidth or a static layer-wise allocation, even though the precision a model needs can change by query, token, layer, and decoding phase. We study lightweight runtime routers that predict this changing sensitivity and allocate high precision only to components that are likely to affect the output, improving quality under memory and latency constraints.

## What This Is

This project sits at the intersection of:

- LLM post-training quantization.
- Mixed-precision inference.
- Runtime model adaptation.
- Elastic serving under latency, memory, or energy budgets.
- Query-aware and token-aware sensitivity prediction.

The narrowed direction is:

> Query-token adaptive precision routing for elastic LLM inference.

Alternative names:

- Dynamic precision routing for LLMs.
- Query-adaptive mixed-precision inference.
- Token-aware any-precision LLM serving.
- Sensitivity-guided precision allocation.

## What This Is Not

This project is not primarily:

- Inventing a new quantization datatype.
- Training LLMs in mixed precision.
- A generic survey of mixed precision.
- Static layer-wise bit allocation only.
- Hardware kernel development only.
- KV-cache quantization only.

Those topics are adjacent. The core question is: when several precisions are available, how should the system decide which one to use right now?

## Why This Direction Is Coherent

Your seed papers all point toward a single transition in the field.

Early LLM quantization asks:

> How do we quantize a model to one low-bit representation without too much quality loss?

Any-precision and mixed-precision work asks:

> How do we support several bitwidths or heterogeneous bitwidths efficiently?

Your direction asks:

> Given multiple possible precisions, can the model choose precision adaptively from the input and current decoding state?

That is the next decision layer above quantization.

## Literature Positioning

| Thread | Representative papers | What they give us | What remains open |
|---|---|---|---|
| Strong static PTQ | GPTQ, AWQ, OmniQuant, SqueezeLLM | Low-bit LLMs with acceptable degradation | Mostly fixed precision once calibrated |
| Multi-scale / any-precision quantization | Any-Precision LLM, Matryoshka Quantization, MatGPTQ | One model can expose multiple bitwidths | Selection policy can be coarse or offline |
| Static mixed precision | LLM-MQ-style and RAMP-style methods | Layer-wise bit allocation under a budget | Assumes the allocation is fixed across queries/tokens |
| Runtime layer/token routing | FlexQuant, DP-LLM, MoBiQuant | Evidence that runtime precision choice can help | Need better routing signals, ablations, overhead accounting, and generalization |
| Query-aware KV-cache precision | MixKVQ | Query relevance matters for KV-cache precision | Weight and KV precision are usually studied separately |
| Hardware execution | TileFuse and mixed-precision kernel work | Mixed precision must be executable efficiently | Algorithms often under-measure switching and routing overhead |

## Seed Paper Interpretation

### DP-LLM

DP-LLM is the closest conceptual anchor. It argues that layer-wise sensitivity changes across decoding iterations and uses lightweight relative-error estimation to choose between low and high precision for each layer at runtime.

Your takeaway:

> Static layer sensitivity is insufficient. Runtime features can guide precision decisions.

### MoBiQuant

MoBiQuant focuses on token-adaptive any-precision. It frames sensitivity around token-level outliers and routes over residual bit slices to reconstruct higher precision when needed.

Your takeaway:

> Token sensitivity can be a practical routing signal when the quantized representation supports elastic precision.

### QAQ

The local file `QAQ.pdf` is available in the workspace. It introduces Query-Adaptive Quantization: bit-plane weight decomposition, a trainable query-conditioned router for block/layer bit-width selection, and CPU/GPU on-demand bit-plane loading.

Your takeaway:

> Basic query-level adaptation is no longer the novelty target. The project should move beyond QAQ toward hierarchical query-token-layer routing, risk-aware fallback, KV-cache/weight budget allocation, or hardware-aware prefetching.

### MixKVQ

MixKVQ studies query-aware mixed precision for KV cache in long-context reasoning. Its key relevance is that query relevance and intrinsic quantization difficulty are both needed to decide which key channels deserve higher precision.

Your takeaway:

> Query-aware routing may matter most when the model must decide which stored context is relevant.

## Core Research Question

Main RQ:

> Can query-token adaptive precision routing improve LLM inference efficiency at the same quality level compared with uniform precision and static mixed precision?

Subquestions:

1. Predictability: Can we predict quantization sensitivity from cheap online signals?
2. Granularity: Which routing granularity gives the best quality-overhead tradeoff?
3. Generalization: Does a router trained or calibrated on one distribution transfer to new prompts, tasks, context lengths, or models?
4. Systems reality: Do measured latency and memory improve after router overhead and kernel costs are included?
5. Risk: Which queries fail when precision is under-allocated?

## Proposed Thesis

The working thesis is:

> LLM precision demand is sparse and structured. Most tokens and layers can tolerate low precision, but a small set of query-token-layer events require higher precision. A lightweight router can identify many of those events early enough to improve the quality-efficiency Pareto frontier.

This thesis can be refuted. If sensitivity is too noisy, if router overhead is too high, or if query-level signals add little beyond static layer sensitivity, the result may be negative. That would still be useful if measured rigorously.

## Concrete Research Object

Define a precision routing policy:

```
router(query_features, token_features, layer_features, runtime_budget) -> precision_action
```

Where:

- `query_features`: prompt length, task type proxy, retrieval/context structure, initial uncertainty, calibration-domain similarity.
- `token_features`: decoding step, entropy, logit margin, activation norm, outlier score, low/high-bit disagreement proxy.
- `layer_features`: layer index, module type, calibration sensitivity, historical error distribution.
- `runtime_budget`: target average bitwidth, latency SLA, memory budget, battery/thermal mode.
- `precision_action`: choose bitwidth, choose residual bit slices, choose high/low precision path, or preserve selected KV channels.

## Best Initial Scope

Start narrow:

- Model: one open LLM in the 1B-8B range, depending on available hardware.
- Quantization: weight-only, any-precision or paired low/high precision if implementation is easier.
- Decision granularity: layer x decoding token.
- Router action: choose low precision or high precision for each eligible linear layer.
- Baselines: uniform precision and static layer-wise mixed precision.
- Primary metric: quality degradation at matched average effective bits.

Do not start with joint weight + activation + KV + hardware kernels. That is too broad for a first publishable result.

## Minimum Publishable Unit

A credible first paper could be:

> We show that quantization sensitivity is predictable from query-token-layer features, then use this predictability to build a lightweight precision router that improves quality at matched average bitwidth over static mixed precision.

Required evidence:

1. Sensitivity analysis showing dynamic variation by query, token, and layer.
2. Router prediction quality compared with static sensitivity baselines.
3. End-to-end quality and efficiency comparison under matched bit budgets.
4. Router overhead and latency accounting.
5. Error analysis of under-routed queries.

## Candidate Contributions

### Contribution A: Sensitivity Dataset

Build a dataset of per-query, per-token, per-layer sensitivity labels by comparing low-bit and high-bit execution.

Possible labels:

- Difference in layer output norm.
- KL divergence between low-bit and high-bit logits.
- Change in next-token rank or probability.
- Downstream answer correctness flip.
- PPL delta for a token span.

This dataset itself may reveal publishable structure.

### Contribution B: Lightweight Precision Router

Train or calibrate a router using cheap features.

Possible routers:

- Threshold rules over activation norm and entropy.
- Logistic regression or gradient-boosted trees.
- Tiny MLP per module type.
- Contextual bandit for budgeted precision allocation.
- Reinforcement learning only after simpler baselines fail.

Start simple. A rule-based router that beats static mixed precision is stronger than a complex router with unclear mechanism.

### Contribution C: Query-Risk Analysis

Show which prompts require precision:

- Math/reasoning.
- Code.
- Rare-token generation.
- Long-context retrieval.
- Ambiguous queries with high uncertainty.
- Safety/refusal boundary prompts.

This makes the paper more than an engineering benchmark.

### Contribution D: Joint Weight/KV Budgeting

After weight-only routing works, extend to long context:

> Given a fixed memory budget, should extra precision be spent on weights, KV cache, or both for this query?

This is a natural follow-up and may become the main contribution if long-context results are strong.

## Initial Hypotheses

H1: Per-layer quantization sensitivity varies systematically with query type and decoding position, and can be predicted better than a static layer average using cheap activation/logit features.

H2: A lightweight precision router conditioned on query-level difficulty, token-level uncertainty, layer identity, and activation statistics can match static mixed precision quality at lower average effective bits.

H3: Joint routing over weight precision and KV-cache precision improves long-context reasoning efficiency more than optimizing either component independently.

H4: A router trained on one model family can transfer partially to another when features encode structural role and normalized sensitivity rather than raw layer indices.

## Evaluation Plan Summary

Primary comparisons:

- FP16/BF16 reference.
- Uniform 4-bit quantization.
- Uniform any-precision model selected per request.
- Static layer-wise mixed precision.
- Query-token adaptive routing.

Metrics:

- Perplexity delta on WikiText-2 and C4.
- Downstream accuracy on reasoning and commonsense tasks.
- Average effective bitwidth.
- Peak memory.
- Tokens/sec or ms/token.
- Router overhead.
- Tail latency.
- Query failure rate.

A result is meaningful only if it improves the Pareto frontier after overhead is included.

## Strongest Objections

### Objection 1: The router overhead may erase the gains.

Response: Measure overhead explicitly. Start with cheap features that are already computed during inference. Include ablations where the router is disabled but precision schedule is replayed.

### Objection 2: Query-level signals may add little beyond token/layer signals.

Response: Treat this as an empirical ablation. Compare layer-only, token-layer, query-layer, and query-token-layer routers.

### Objection 3: Dynamic bit switching may be hard to implement efficiently.

Response: Start with simulation for scientific signal, then validate on an executable subset. If hardware execution is limiting, report both oracle/simulated and real-runtime results.

### Objection 4: Gains may not transfer across models.

Response: Make transfer a secondary hypothesis. The first paper can still contribute if the router is per-model but cheap and interpretable.

## Two-Week Pilot

Week 1:

1. Choose a model and quantization stack.
2. Implement or reuse low/high precision variants.
3. Collect per-token/layer sensitivity traces on a small calibration set.
4. Plot sensitivity by layer, token position, prompt category, and entropy.

Week 2:

1. Build static baseline allocation.
2. Build a rule-based router.
3. Compare uniform, static, and dynamic routing at matched average bitwidth.
4. Analyze failure cases.

Go/no-go signal:

- Continue if dynamic features predict sensitivity and dynamic routing beats static at matched bits.
- Pivot if sensitivity is unpredictable or overhead dominates.

## Final Narrowed Statement

The research direction is:

> Query-token adaptive precision routing for LLM inference: learning or designing lightweight policies that allocate quantization precision dynamically across layers, tokens, and possibly KV-cache components under explicit runtime budgets.

The simplest first claim to test is:

> Runtime precision demand is predictable enough that a cheap router can outperform static mixed precision at the same average bitwidth.

# Research Findings

## Research Question

Can an LLM choose precision dynamically from query, token, layer, and runtime signals so that it spends high precision only where it is needed, improving the quality-latency-memory Pareto frontier over static mixed precision?

## Current Understanding

The research direction is not "mixed precision" in the broad sense. It is specifically about runtime precision routing for LLM inference: deciding which parts of the model should run at higher or lower precision for a particular request and decoding step.

The seed papers point to the same emerging thesis from different angles:

- Any-precision and Matryoshka-style quantization make multiple bitwidths available within one model representation.
- Static mixed precision assigns bitwidths based on offline layer sensitivity, but assumes sensitivity is stable.
- DP-LLM, FlexQuant, and MoBiQuant challenge that assumption by showing that useful precision decisions can vary at runtime.
- MixKVQ suggests query relevance matters for KV-cache compression, especially in long-context reasoning.

The narrowed research object is therefore the precision assignment policy:

> Given a model that can execute at several bitwidths, what cheap online signal tells us where precision matters right now?

This is a stronger and more publishable framing than "make quantization better", because it isolates a concrete unsolved decision problem.

## Key Results

The first MatGPTQ uniform-slice baseline on Llama-3.1-8B-Instruct produced WikiText2 perplexity of 7.09 at 8-bit, 7.30 at 4-bit, and 9.51 at 3-bit.

This is preliminary baseline evidence, not a test of H1. It shows that uniform 4-bit is already close to 8-bit on aggregate language-modeling PPL, while uniform 3-bit is substantially worse. That pattern makes the research bar higher: a router must beat strong 4-bit/static baselines or selectively recover 3-bit failures under matched average-bit budgets.

The initial literature-backed direction is:

> Query-token adaptive precision routing for elastic LLM inference.

The minimum viable research result would be evidence that a lightweight router can improve the quality-efficiency Pareto frontier over static layer-wise mixed precision at matched average effective bitwidth.

## Patterns and Insights

Three patterns emerged from the literature scan.

1. Precision sensitivity is dynamic.

Layer sensitivity is not only a static property of the layer. It can depend on decoding iteration, activation values, token outliers, and query relevance.

2. Multi-precision storage is becoming available.

Any-precision and Matryoshka-style methods reduce the storage cost of supporting multiple bitwidths. This shifts the bottleneck from "can we store many precisions?" to "can we choose the right precision cheaply?"

3. Practical routing must be hardware-aware.

Fine-grained routing can look good in quality/bit metrics while losing in wall-clock latency if it creates router overhead, kernel fragmentation, or excessive bit-switching cost. The research must measure actual latency and not only perplexity.

## Lessons and Constraints

- Do not frame the work as generic quantization. The field already has many strong PTQ methods such as GPTQ, AWQ, SmoothQuant, and OmniQuant.
- Do not frame the work as static mixed precision only. The seed papers are about runtime adaptation.
- Do not optimize only average perplexity. The relevant deployment question includes tail risk: which queries fail when the router under-allocates precision?
- Do not treat cheap WikiText2 aggregate PPL as sufficient evidence. H1 needs per-token/per-layer sensitivity labels, downstream task flips, held-out prompt categories, uncertainty estimates, and runtime overhead.
- Start with layer-by-token routing before channel- or element-level routing. It is more implementable and closer to existing mixed-precision execution.
- Treat KV cache as an extension, not the first dependency. Weight-only routing gives a cleaner first pilot; KV-cache routing becomes important for long-context reasoning.
- QAQ is now available locally as `QAQ.pdf`. It validates that query-adaptive block/layer-wise precision routing is directly aligned with this project, but also means the novelty should move beyond basic query-conditioned bit-width selection.

## Open Questions

- Can query difficulty be predicted early enough to influence precision before the model has already done most of the expensive work?
- Which online features are cheap and reliable: input norm, activation outlier score, entropy, KL divergence between quantized logits, calibration sensitivity, or routing history?
- Does query-aware routing provide gains beyond token-aware routing?
- How much overhead can a router spend before it destroys latency gains?
- Can a learned router transfer across models, tasks, or context lengths?
- Are failures concentrated in reasoning/math/code tasks, long-context retrieval, rare-token generation, or safety-sensitive refusals?
- How much does QAQ's query-level router add beyond static 8-bit and 4-bit baselines, given that its reported QAQ accuracy is identical to static 8-bit in the current table?
- Can QAQ's synchronous CPU-to-GPU loading overhead be reduced with predictive prefetching, cache-aware routing, or hardware-constrained schedules?

## Optimization Trajectory

The first recorded baseline trajectory is:

1. 8-bit uniform MatGPTQ on WikiText2: PPL 7.09.
2. 4-bit uniform MatGPTQ on WikiText2: PPL 7.30.
3. 3-bit uniform MatGPTQ on WikiText2: PPL 9.51.

The robust trajectory should next track quality degradation versus average effective bits and latency for:

1. Uniform precision.
2. Static layer-wise mixed precision.
3. Query-token adaptive routing.
4. Router ablations with selected feature groups removed.

# QAQ: Query-adaptive Mixed-precision Quantization for Large Language Models

## Verification Status

The user provided the local file `QAQ.pdf` on 2026-06-16. I still could not verify a public indexed arXiv/web record for the exact title, so cite the local PDF or obtain a stable URL before manuscript submission.

PDF metadata:

- Pages: 5.
- PDF creation date: 2025-11-13.
- Venue line: ML For Systems workshop at NeurIPS 2025.

## Core Idea

QAQ introduces Query-Adaptive Quantization, a dynamic-precision inference framework that:

- Decomposes weights into bit-planes.
- Uses a trainable query-conditioned router to select block/layer bit-widths.
- Supports CPU-to-GPU on-demand loading of selected bit-planes.
- Evaluates on Qwen3 and LLaMA-3.1 models.

## Reported Result Pattern

The paper reports accuracy close to or matching static 8-bit baselines, lower GPU memory when on-demand loading is enabled, and latency overhead from synchronous CPU-to-GPU transfers.

## Relevance To This Project

QAQ is highly aligned with the query-adaptive part of the project. It means a future contribution should avoid claiming basic query-conditioned bit-width routing as the sole novelty.

## Open Novelty Space

- Hierarchical query-token-layer routing, not query-only routing.
- Risk-aware routing and hard-query fallback.
- Joint weight and KV-cache precision budgeting.
- Predictive asynchronous prefetching to reduce QAQ's on-demand loading overhead.
- Sensitivity trace datasets and analysis explaining when query adaptation matters.

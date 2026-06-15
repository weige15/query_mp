# TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs

URL: https://arxiv.org/abs/2606.11357

## Citation Metadata

- Authors: Wesley Pang, Gregory Hyegang Jun, Feiyang Liu, Deming Chen.
- Year: 2026.
- Status: arXiv preprint.

## Core Idea

TileFuse develops a fused mixed-precision kernel library for quantized LLM inference on AMD NPUs, co-designing weight layout, metadata placement, microkernels, and dataflow.

## Relevance

This is not the core algorithmic direction, but it is a systems constraint. Dynamic precision policies must eventually run efficiently on real hardware. Otherwise average-bit savings may not translate to latency or energy gains.

## What It Suggests For This Project

- Always report measured latency, not just average effective bits.
- Avoid overly fine-grained decisions until there is a feasible execution path.
- Consider hardware-friendly routing granularity as part of the method.

## Open Questions For Follow-Up

- Which precision switching granularity is realistic on the target hardware?
- Can routing decisions be batched or fused to avoid kernel fragmentation?
- Should the router optimize for hardware-supported precision patterns rather than arbitrary bitwidths?

# Kernel And Memory-Layout Direction

Date: 2026-06-16

## Hypothesis

Fine-grained adaptive precision can lose its theoretical efficiency gains because heterogeneous bit-width assignments disrupt hardware-friendly memory layout and kernel execution.

This is especially relevant when precision is assigned at token, channel, head, or small block granularity. Layer-level precision is easier to handle because each layer can use a specialized static kernel. Token/layer/channel-level precision creates more layout and scheduling pressure.

## Verified Assessment

The hypothesis is broadly correct, but the precise issue is larger than cache-line alignment.

The main systems issues are:

- Packed low-bit formats require predictable grouping and alignment.
- GPU loads are fastest when accesses are coalesced and vectorized.
- Tensor-core kernels expect fixed tile shapes and supported operand types.
- Fine-grained precision changes can create branch divergence or require multiple kernel launches.
- Mixed bit-width tensors need metadata, offsets, scales, and zero points that must also be laid out efficiently.
- Dynamic CPU/GPU loading, as in QAQ, can save memory but add latency if transfers are synchronous.
- KV-cache precision variation can create physical fragmentation unless it is aligned with paging/block allocation.

## Why This Is A Research Direction

QAQ shows query-adaptive routing but reports latency overhead when on-demand loading is enabled. That leaves a clear gap:

> Adaptive precision needs a hardware-constrained routing and layout policy, not only an accuracy-driven router.

This direction can be framed as systems co-design:

- The router should choose precision only from schedules that map well to kernels.
- The memory layout should make common mixed-precision patterns contiguous and coalesced.
- The runtime should prefetch or cache bit-planes before they are needed.

## Research Question

Can a hardware-aware adaptive precision router preserve the quality gains of query/token/layer routing while enforcing memory-layout and kernel constraints that keep latency competitive?

## Candidate Method

Use constrained routing:

1. Define hardware-friendly precision units: layer, tile, block, KV page, or group-of-tokens.
2. Prepack weights into aligned bit-plane groups.
3. Restrict routing decisions to a small set of kernel-supported precision patterns.
4. Add an alignment/fragmentation penalty to the router objective.
5. Prefetch predicted high-precision planes asynchronously.
6. Measure quality, memory, latency, cache misses, and effective bandwidth.

## Baselines

- QAQ-style unconstrained query router.
- Static 4-bit and 8-bit.
- Static mixed precision.
- Hardware-constrained adaptive routing.
- Oracle adaptive routing without layout constraints.

## Success Criterion

The method is worth pursuing if hardware-constrained routing keeps most of the quality/bit-width benefit of adaptive precision while substantially reducing latency overhead versus unconstrained or synchronous on-demand routing.

## Positioning

Possible title:

> Layout-Aware Adaptive Precision Routing for LLM Inference

Two-sentence pitch:

> Query-adaptive quantization can reduce memory, but fine-grained heterogeneous precision creates memory-layout, cache, and kernel-execution overheads that erase theoretical gains. We study layout-aware precision routing that constrains bit-width decisions to hardware-friendly blocks and uses aligned packing/prefetching to preserve both accuracy and real latency.

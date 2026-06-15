# FlexQuant: A Flexible and Efficient Dynamic Precision Switching Framework for LLM Quantization

URL: https://arxiv.org/abs/2506.12024

## Citation Metadata

- Authors: Fangxin Liu, Zongwu Wang, JinHong Xia, Junping Zhao, Shouren Zhao, Jinjin Li, Jian Liu, Li Jiang, Haibing Guan.
- Year: 2025.
- Status: arXiv preprint.

## Core Idea

FlexQuant dynamically switches precision during token generation. It uses signals such as model perplexity entropy and KL divergence to guide fine-grained layer-wise mixed-precision decisions.

## Relevance

FlexQuant is useful as a feature-design reference. It suggests that uncertainty and divergence signals can indicate when higher precision is needed.

## What It Suggests For This Project

- Include entropy, logit margin, and KL-style disagreement as token-level router features.
- Compare query-token routing against entropy-only routing.
- Measure whether dynamic precision helps across tasks, not only on language modeling loss.

## Open Questions For Follow-Up

- Are entropy/KL features cheap enough in the target runtime?
- Do they predict sensitivity before low-bit errors have already propagated?
- Are gains robust on reasoning and long-context tasks?

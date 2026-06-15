# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-06-15 | bootstrap | Started autoresearch workspace for mixed precision LLM inference. Since no cron or /loop tool is exposed in this session, recorded continuity status in research-state.yaml. |
| 2 | 2026-06-15 | bootstrap | Searched current literature around DP-LLM, MoBiQuant, QAQ, dynamic precision switching, any-precision LLMs, Matryoshka quantization, query-aware KV-cache quantization, and hardware support. |
| 3 | 2026-06-15 | outer-loop | Narrowed topic from generic mixed precision to query-token adaptive precision routing for elastic LLM inference. The central object is the runtime precision assignment policy, not the quantizer alone. |
| 4 | 2026-06-15 | bootstrap | Created initial hypotheses H1-H4 and experiment protocols. QAQ exact title could not be verified in public indexed sources as of 2026-06-15, so it is treated as an unverified seed. |

# Published V3 LoCoMo Evidence (Architecture Carried into V3.8.0)

This page is the canonical public reference for SuperLocalMemory benchmark
figures. It reports the published V3 architecture experiments that remain the
foundation of V3.7. These are not a newly rerun V3.7 package benchmark and are
not a cross-vendor leaderboard.

Source: [SuperLocalMemory V3 preprint (arXiv:2603.14588)](https://arxiv.org/abs/2603.14588).

## Published LoCoMo results

| Published configuration | LoCoMo aggregate | Original protocol scope | What the result establishes |
|---|---:|---|---|
| **Mode A Raw** | **60.4%** | 10 conversations; 1,276 scored questions; local embeddings, local retrieval, and zero-LLM answer construction | End-to-end local answer construction under the published V3 protocol. |
| **Mode A Retrieval** | **74.8%** | 10 conversations; 1,276 scored questions; local retrieval, then GPT-4.1-mini answer synthesis | Retrieval evidence from the local system, with the disclosed external model constructing the final answer. |
| **Mode C** | **87.7%** | Conv-30 only; 81 scored questions; text-embedding-3-large plus GPT-4.1-mini answer generation and judge | Cloud-assisted configuration on one disclosed conversation; not a full-dataset result. |

The Mode A Retrieval category results were 72.0% single-hop, 70.3% multi-hop,
80.0% temporal, and 85.0% open-domain. Mode C does not report a temporal
category in this one-conversation scope.

## Mathematical ablation evidence

Across six LoCoMo conversations, the paper reports **71.7%** with the
information-geometric layers and **58.9%** without them: **+12.7pp**. This is
an ablation result with its own protocol; it must not be substituted for the
10-conversation aggregate rows above.

## How to compare responsibly

A LoCoMo percentage is comparable only when the conversation subset, question
count, retrieval stack, answer-construction model, judge, and release artifact
are declared and compatible. Do not use these figures to claim that V3.7 has
been freshly rerun, that every mode has the same score, or that SLM outranks a
competitor tested under another protocol.

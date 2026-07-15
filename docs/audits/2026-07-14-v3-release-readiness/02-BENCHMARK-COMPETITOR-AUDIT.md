# Benchmark and competitor audit

## Decision

There is no defensible global “LoCoMo leaderboard.” Current public numbers mix retrieval recall with end-to-end QA, omit different categories, use different answerers and judges, expose radically different context budgets, and sometimes contain artifacts that disagree with the headline.

SLM should not chase the largest percentage. It should own the **accuracy–context–latency–privacy–auditability Pareto frontier** under a frozen protocol.

## Official LoCoMo lock

- Repository: [snap-research/locomo](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)
- Dataset: `data/locomo10.json`
- SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- Total QA pairs: 1,986
- Categories: 282 multi-hop, 321 temporal, 96 open-domain, 841 single-hop, and 446 adversarial
- Original benchmark: [project page](https://snap-research.github.io/locomo/) and [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)

Any run over 1,540 questions excludes the entire adversarial category. Any retrieval-recall number is not QA accuracy. Any result without the dataset hash, question count, answerer, judge, prompts, context cap, and raw outputs is a vendor claim, not a release gate.

## SLM’s current benchmark truth

The current source does not contain a release-grade, post-change LoCoMo harness or raw full-dataset artifact. Historical backup runs cover different commits and mostly two conversations. The best recorded historical two-conversation result was 70.4%; other runs ranged around 66.8–68.4, and the handoff itself notes judge variance and the unfinished ten-conversation run.

The paper associated with the current headline distinguishes configurations that the website merges:

| Historical result | Evidence-safe interpretation |
|---|---|
| 60.4% | Mode A raw result described as zero-LLM. |
| 74.8% | Local retrieval followed by GPT-4.1-mini answer construction; not an end-to-end zero-LLM result. |
| 87.7% | Mode C cloud-assisted result on 81 questions from one conversation; not a full LoCoMo score. |

Because ingestion, retrieval, and scoring changed after those experiments, the current `3.6.23` score is **UNKNOWN**. Historical results may appear in research history with their original protocol; they must not be presented as current shipped-product measurements.

## Protocol-normalized evidence matrix

`ARTIFACT-RECOMPUTED` means the audit counted a checked-in or public per-query artifact. It does not mean proprietary APIs were rerun.

| System and source lock | Task actually measured | Dataset and protocol | Answerer / judge | Best defensible result | Verdict |
|---|---|---|---|---:|---|
| [Hindsight `b6c7b2a`](https://github.com/vectorize-io/hindsight/tree/b6c7b2a2e94d5e6ea6aad4c3de8913d7059f0c58) | End-to-end single-query RAG QA | LoCoMo categories 1–4; 1,540; average 36,235 context tokens | Gemini 3.1 Pro Preview / Gemini 2.5 Flash Lite | **92.013%, 1,417/1,540** | `ARTIFACT-RECOMPUTED`. Strongest transparent current artifact found, but no adversarial questions and a large context budget. The benchmark operator is the vendor. |
| [Mem0 benchmark `4b61c5d`](https://github.com/mem0ai/memory-benchmarks/tree/4b61c5d31b9c668a12b4f5e78064248a02c82d2b) | Managed V3 end-to-end QA, top 200 | Categories 1–4; 1,540 | GPT-5 / GPT-5 through Azure | **91.558%, 1,410/1,540** | `ARTIFACT-RECOMPUTED`. README says 92.5% and 1,425; the committed artifact contains 1,410 passes. |
| Mem0 LongMemEval, same lock | Managed V3 end-to-end QA | LongMemEval-S; 500 | GPT-5 / GPT-5 | **93.4%, 467/500 top 200; 90.4%, 452/500 top 50** | `ARTIFACT-RECOMPUTED`. README instead says 94.4% and 94.8%, so headline-to-artifact traceability is broken. |
| [Letta active `4056e94`](https://github.com/letta-ai/letta-code/tree/4056e947bc751aa901f796f2b4cafb3777329c02) | Agentic file-search QA with repeated search | Categories 1–4; 1,540 | GPT-4o-mini; archived grader | **ABSTAIN; vendor reports 74.0%** | Public harness, no matching per-query artifact found. This is agentic file search, not a standard memory API pipeline. |
| [Synthius-Mem](https://arxiv.org/abs/2604.11563) | End-to-end QA | Claims full LoCoMo but reports 1,813; 173 missing unexplained | GPT-4.1-mini for Synthius; Gemini 3 Flash for baselines | **ABSTAIN; paper reports 94.37%** | No code, raw predictions, dataset hash, or judge artifact; model-confounded comparison. |
| [Context-mem `2a55af0`](https://github.com/JubaKitiashvili/context-mem/tree/2a55af0a4bf3467df89f1315a74bb2e15ad903f7) | Evidence-session retrieval recall, not QA | 1,977; top 10; any evidence session counts | None in headline metric | **98.0–98.1% R@10 as documented** | Ingests provided/generated summaries and uses benchmark-tuned synonyms. Not comparable to raw-dialogue QA. |
| [MemPalace `5b1c32e`](https://github.com/MemPalace/mempalace/tree/5b1c32eee133b5c480ce3bdd8927ce265a1bb3e6) | Evidence/session retrieval recall | Full 1,986; top 10 | None in honest run | **88.9% fractional evidence recall; raw 60.3%** | Its older 100% run used top 50 when conversations had only 19–32 sessions, effectively returning the corpus; the repository now acknowledges the caveat. |
| [Cognee `252f2c3`](https://github.com/topoteretes/cognee/tree/252f2c3efb184533a0955e31e83a28ea7db9813d) | Single-query RAG QA | 152 questions, apparently one conversation | Benchmark-standard models for the run | **80.26%, 122/152** | Public aggregate, not full LoCoMo. |
| [EverOS `a1e21ca`](https://github.com/EverMind-AI/EverOS/tree/a1e21ca676519b11653aaf3293c77a53254a7018) | Agentic top-10 QA harness | Categories 1–4; 1,540 | GPT-4.1-mini / GPT-4o-mini ×3 | **ABSTAIN** | No committed raw full-run artifact found; the 93.3 sample report is illustrative. |
| [Supermemory `2cebe81`](https://github.com/supermemoryai/supermemory/tree/2cebe81512a56272a453afe08d8c00b6888d2b7b) | README claims across three benchmarks | Current matching LoCoMo protocol not located | Unknown | **ABSTAIN** | “#1” is not tied to a current raw artifact and conflicts with the current Hindsight evidence. |
| [Graphiti `526dcad`](https://github.com/getzep/graphiti/tree/526dcad7a300f3c5c506ff96a68bcdc7ca9f97ed) / Zep | No normalized current artifact established | — | — | **ABSTAIN** | Graphiti OSS engine quality and Zep cloud product claims must not be conflated. |

Mem0’s judge is also permissive: one correct list item can count, dates receive broad tolerance, durations receive proportional tolerance, and extra detail is not penalized. This makes a “same dataset” comparison insufficient without a common judge and human audit.

## Architecture cohort

GitHub popularity was captured on 2026-07-14. Stars and forks are developer-attention signals, not users, installations, revenue, production deployments, or retention. Auditable active-user numbers were not found, so user counts are `ABSTAIN`.

| Product | Visible attention | Mechanics that matter | SLM decision |
|---|---:|---|---|
| [Mem0](https://github.com/mem0ai/mem0/tree/42cf18c4e6adb448e981aa1c7b55c1602b0cb670) | 60,770 stars; 7,077 forks | LLM fact extraction, semantic lookup of existing memory, deduplication/history, entity linking, broad providers and stores. OSS hybrid retrieval begins from semantic candidates and adds lexical/entity signals. | Match adapter simplicity. Beat its vector-defined candidate universe with truly independent lexical, graph, and temporal generators. |
| [Graphiti](https://github.com/getzep/graphiti/tree/526dcad7a300f3c5c506ff96a68bcdc7ca9f97ed) | 28,692; 2,894 | Episodic source nodes, deduplicated entities/edges, contradiction invalidation, valid time vs ingestion time, parallel node/edge/episode/community search, RRF and multiple rerankers. | Adopt bi-temporal validity, explicit supersession, provenance edges, and independent search pools. |
| [Supermemory](https://github.com/supermemoryai/supermemory/tree/2cebe81512a56272a453afe08d8c00b6888d2b7b) | 28,370; 2,470 | Consumer app, hosted API, connectors, multimodal processing, profiles, local binary, SDKs and MCP. The audited repository does not expose enough of the hosted/local engine to validate its internals. | Compete on transparent engine code, package provenance, local trust, and cross-IDE continuity. Copy its onboarding breadth, not its remote-script installer. |
| [Cognee](https://github.com/topoteretes/cognee/tree/252f2c3efb184533a0955e31e83a28ea7db9813d) | 27,799; 2,751 | Chunk/entity/relation/summary graph construction and a large registry of retrieval modes across many databases. | Preserve extensibility but expose fewer composable retrieval primitives with routing evidence and bounded operational complexity. |
| [Letta Code](https://github.com/letta-ai/letta-code/tree/4056e947bc751aa901f796f2b4cafb3777329c02) | 2,839 active; 23,780 legacy | Git-backed Markdown MemFS, full system-memory files, progressive disclosure for external files, per-edit commits, dream/reflection, branches/worktrees and rollback. Local search is currently substring/FTS-lite until vector indexing exists. | Add portable, versioned, reviewable memory views and deterministic rebuild. The market is moving beyond opaque vector facts. |
| [Hindsight](https://github.com/vectorize-io/hindsight/tree/b6c7b2a2e94d5e6ea6aad4c3de8913d7059f0c58) | 18,320; 1,120 | Defense/redaction, delta dedupe, world/experience facts, causal/temporal links, async observations, independent semantic/BM25/graph/temporal arms, RRF, cross-encoder, proof-count and budget trimming. | Its four-arm traceable retrieval is the strongest immediate reference. Beat it on bounded context, p95, local operation, and auditability. |
| [EverOS](https://github.com/EverMind-AI/EverOS/tree/a1e21ca676519b11653aaf3293c77a53254a7018) | 10,950; 854 | Markdown/YAML source of truth; SQLite queues/audit; Lance derived index; typed episodes/facts/foresight/profiles/cases/skills; offline reflection; dense+sparse RRF and agentic retrieval. | Adopt source/derived separation, typed memory, durable queues, index-freshness reporting, and read-your-write guarantees. |

## Mechanics SLM must adopt or beat

1. **Independent candidate generators.** Dense, lexical, graph, temporal, and associative search must contribute candidates before fusion.
2. **Bi-temporal memory.** Store `occurred_at`, `mentioned_at`, `valid_from`, `valid_to`, `ingested_at`, source revision, and supersession.
3. **Immutable evidence and rebuildable derivations.** Every extracted object points to exact source spans and can be regenerated deterministically.
4. **Typed memory.** Separate episodes, facts, preferences/profiles, procedures/skills, project state, and agent lessons without benchmark-specific ontology.
5. **Contradiction and negative evidence.** Preserve competing claims and their time/source; an empty search is not proof that a premise is false.
6. **Read-your-write plus asynchronous enrichment.** A durable fast representation is queryable immediately while idempotent enrichment completes.
7. **Scoring traces.** Return per-channel ranks, fusion contribution, reranker score, policy changes, source, token count, latency, and final calibration.
8. **Portable versioned memory.** Provide Markdown/JSON export, git-friendly diffs, rollback, migration, and deterministic index rebuild.
9. **Mesh contract tests.** Prove shared identity, project isolation, concurrent updates, offline recovery, conflict resolution, delete propagation, and source-attributed handoff.
10. **Package trust.** Signed artifacts, checksums, SBOM, provenance, and package-manager installation without remote scripts or system-Python override.

## Required benchmark protocol for V3.7

### Track 1 — retrieval

Run all 1,986 questions. Report evidence Recall-any@K, Recall-all@K, MRR, nDCG, precision, and retrieved tokens for K=5/10/20. No answerer is involved.

### Track 2 — bounded QA

Use the same frozen answerer and judge for every provider. Run all 1,986 with one retrieval call and hard 4K and 8K context caps. Publish adversarial/abstention separately and in the overall score.

### Track 3 — agentic QA

Allow at most three retrieval calls under a fixed total context, wall-time, and tool-call budget. Never place this score in the bounded-QA table.

### Manifest and raw artifacts

Every run must publish dataset URL/commit/hash, included categories, ingestion unit, extractor, embedder, top-K, token budget, prompts, models, temperatures, retries, hardware, source commit, container digest, per-query retrieved IDs and scores, final prompt, answer, judge output, latency, tokens, cost, and failures. Ban gold evidence, dataset summaries, event summaries, category labels, hard-coded question IDs, and failure-derived synonyms from ingestion.

At least 10% of the result set must receive blinded human adjudication, including all adversarial, contradiction, abstention, and temporal errors. Report bootstrap confidence intervals and paired differences.

### Additional gates

LoCoMo alone is insufficient. Add [LongMemEval](https://github.com/xiaowu0162/LongMemEval), [BEAM](https://github.com/mohammadtavakoli78/BEAM), [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench), [PersonaMem-v2](https://github.com/bowen-upenn/PersonaMem-v2), [EvoMemBench](https://github.com/DSAIL-Memory/EvoMemBench), and action-oriented memory evaluation such as [MemoryArena](https://arxiv.org/abs/2602.16313). Add SLM-owned cache/compression fidelity and cross-IDE mesh suites; these test the actual product differentiation competitors can avoid on conversational QA.

## Benchmark conclusion

The correct market claim is not “we scored 100%.” The correct claim, after execution, is: **the full dataset, a bounded context budget, raw outputs, reproducible packages, and memory that survives across real tools.** That is harder to game and more valuable to buyers.

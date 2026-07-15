# Website and public-claims audit

## Verdict

The live site and README are a **P0 credibility blocker**. This is not a request for softer copy. Several prominent statements are factually inconsistent with the papers, code, defaults, registry artifacts, or current benchmark evidence. A product positioned under AI Reliability Engineering cannot ask buyers to trust claims that its own repository disproves.

Live page audited: [superlocalmemory.com](https://www.superlocalmemory.com/) on 2026-07-14.

## Claim ledger

| Current claim | Decision | Evidence | Required replacement |
|---|---|---|---|
| “74.8% Mode A · zero cloud · TF-IDF · pure local retrieval” and README “zero-LLM” | **Remove from headline** | Paper configuration uses GPT-4.1-mini to construct answers for the 74.8 result. The paper’s raw zero-LLM result is 60.4. Current Mode A uses a local Nomic sentence transformer and ONNX reranker, not TF-IDF-only. | “Historical research result: 74.8% with local retrieval and GPT-4.1-mini answer construction. Current-release benchmark pending.” |
| “87.7% Mode C” without scope | **Qualify** | The paper result covers 81 questions from one conversation and uses cloud-assisted components. | “Historical Mode C result: 87.7% on 81 questions from one LoCoMo conversation; not a full-dataset score.” |
| “3 peer-reviewed papers” / “peer-reviewed proofs” | **Remove** | The linked works are arXiv preprints. No peer-reviewed venue acceptance was found. | “3 public arXiv preprints with code-linked experiments.” |
| `2603.02240` as “Bounded Persistent Memory,” lifecycle proof, or scale evidence | **Correct mapping** | [arXiv:2603.02240](https://arxiv.org/abs/2603.02240) is *Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning*. | Use the exact title and associate it only with the evaluated trust, privacy, search, graph, and multi-agent claims. |
| `2604.06392` as SLM compression theory | **Remove mapping** | [arXiv:2604.06392](https://arxiv.org/abs/2604.06392) is *Qualixar OS: A Universal Operating System for AI Agent Orchestration*. It is not the third SLM paper or a compression benchmark. | Link the actual third SLM paper, [arXiv:2604.04514](https://arxiv.org/abs/2604.04514), and do not call it a compression proof. |
| “Every recall uses Fisher-Rao … not cosine heuristics” | **Replace** | Semantic candidate generation uses cosine similarity. Fisher-derived terms modify scoring, and fast/full-scan paths apply them differently. | “Hybrid retrieval uses dense cosine candidates plus lexical, temporal, associative, graph-derived, and Fisher-informed scoring.” |
| “Not a heuristic in sight” / “mathematical guarantees” | **Remove** | Ranking includes RRF, regular expressions, thresholds, recency, floors, reranker availability, and other heuristics. Mathematical properties of a component do not guarantee end-to-end recall correctness. | “Research-informed ranking with inspectable heuristic and mathematical components.” |
| “1M+ memories, zero slowdown” / “5 years daily use” | **Remove** | No release-linked raw 1M-memory benchmark, five-year trace, hardware manifest, latency distribution, or retrieval-quality result supports it. A cited trust paper reports scaling only to thousands and does not prove constant performance. | Publish a scale envelope only after a reproducible corpus, ingest rate, p50/p95/p99, quality-at-scale, disk/RAM, and compaction artifact exists. |
| “32× cold compression” | **Remove or narrowly qualify** | The value refers to embedding precision/representation, not full memory, prompt, or database compression. Current lifecycle/backend wiring is not proven end to end. | “Up to 32× reduction for selected quantized embedding representations under the paper’s experiment.” |
| “Exact-match cache on by default” | **Correct** | Optimize master, proxy, cache, semantic cache, and compression default off in shipped configuration. | “Exact cache is available when Optimize/proxy caching is enabled.” |
| “Zero cloud, zero proxy” for cache | **Rewrite by surface** | MCP KV tools can cache explicitly routed results without a proxy; transparent full-turn LLM caching requires the proxy. | “Use explicit MCP result caching without a proxy, or enable the proxy for transparent provider-response caching.” |
| “60–95% prompt compression” and “structured payloads” | **Remove headline** | Shipped synthetic corpus measured JSON 0%, code 0%, safe prose 0.5%, aggressive lossy prose 40.68%. Proxy disables the lossy second layer. | “Safe mode preserves JSON and code; measured reduction varies by content. Publish corpus-specific results.” |
| “No cloud. No telemetry. Yours.” | **Qualify** | Local Mode A can avoid provider calls, but Mode C, cloud backup, connectors/downloads, and proxy providers use networks. Packages also acquire models/dependencies. | “Local Mode A can run without sending memory content to a cloud provider; optional modes and integrations make network calls.” |
| “One SQLite file” or equivalent portability language | **Qualify** | Memory, behavioral state, cache, optional backends, models, config, logs, and derived files occupy multiple locations. | “Core memory is SQLite-backed; additional databases, indexes, models, configuration, and logs are documented in the storage map.” |
| “17+ tools/IDEs auto-configured” | **Replace** | Portable matrix has 12 entries and 11 config writers; legacy CLI lists seven. A name/template is not a verified integration. | “Verified integrations: publish only clients that pass the install/session/remember/recall/update/forget matrix for this release.” |
| “Six-channel” / “seven-channel” retrieval | **Replace** | Current recall has five independent candidate producers; entity graph is a post-fusion enhancer. Other surfaces say four. | “Five candidate channels plus graph-based score enhancement,” until the architecture itself changes. |
| MCP profiles `core14`, `mesh8`, `full38`, `power50`, `whole81` | **Correct** | Source accepts `core`, `code`, `full`, `power`, `mesh`, and `whole`; stale documented names silently fall back. | Document source names, actual tool counts, and fail on unknown profiles rather than silently defaulting. |
| “Everything included” via npm | **Qualify and harden** | Runtime requires Node 18+/npm 9+ and Python 3.11–3.14. Post-install can invoke pip, hooks, setup, and model acquisition, with an unsafe system-Python fallback. | State exact prerequisites and side effects. Remove `--break-system-packages` and require explicit setup consent. |
| “MIT,” “Elastic,” and “AGPL” across surfaces | **Normalize** | Root license is AGPL, while source headers, npm scripts, paper-era copy, and `llms.txt` contain stale MIT/Elastic language. | One canonical license matrix: repository, packages, optional commercial license, third-party components, and historical paper snapshot. |
| LoCoMo link `2402.09714` | **Correct** | Official LoCoMo paper is [arXiv:2402.17753](https://arxiv.org/abs/2402.17753). | Use the official project and paper URLs everywhere. |

## Replacement home-page position

### Headline

> Private, persistent memory across your coding agents.

### Subhead

> Run one local memory service for Codex, Claude Code, Cursor, and MCP clients. Keep source-attributed project facts on your machine, inspect what was remembered, and carry context across sessions.

This copy should ship only after the cross-client contract passes for every named client. Until then, name only the clients that pass.

### Proof bar

Use proof users can reproduce:

- `pip` and npm package version plus signed checksums.
- “Full source suite: 6,025 passed” with the exclusions stated, followed by separate slow/integration results.
- A live two-client handoff showing write, recall, update, provenance, and forget.
- Benchmark cards that show task, subset, model, context cap, commit, and raw artifact—not a naked percentage.
- “Local Mode A available” rather than “no cloud” for the entire product.

## Research page structure

List the three SLM preprints accurately:

1. [Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning](https://arxiv.org/abs/2603.02240).
2. [Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory](https://arxiv.org/abs/2603.14588).
3. [The Living Brain: Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval](https://arxiv.org/abs/2604.04514).

Each card must state the code/tag evaluated, dataset/subset, model dependencies, license at the evaluated snapshot, current implementation drift, and raw-artifact availability. Qualixar OS should appear separately under related research.

## CTA and commercial gaps

The site has no strong commercial path: no paid design-partner offer, commercial-license CTA, support scope, buyer qualification, case study, or newsletter capture. The fix is not to add generic “Contact sales.” Add one precise offer:

> Deploy private cross-agent memory in one engineering team. We audit your agent workflows, integrate SLM with two tools, run the memory-reliability suite, and deliver a measured handoff report. Fixed scope. Apply for a founding design-partner slot.

Route the form to the verified founder email already used by the project; do not invent a `qualixar.com` mailbox.

## Publication gate

Before any site or README change is published, run a final claim scanner over the rendered text. Every number, superlative, product count, benchmark name, paper, model, and license must link to a live entailing artifact for the released commit. If the artifact does not exist, remove the claim or mark it explicitly as historical/research-only.

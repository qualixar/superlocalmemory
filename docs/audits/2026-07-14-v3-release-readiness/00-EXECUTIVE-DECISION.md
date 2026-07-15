# Executive decision

## Verdict

SuperLocalMemory is a serious and unusually broad local-memory codebase, but the current product narrative is ahead of the shipped default behavior. The next public push must be an **evidence-correction and integrity release**, not a “best memory in the world” launch.

The current release is a no-go for three reasons:

1. The dominant MCP/HTTP/CLI `remember` path writes a reduced record and its materializer normally marks that record complete without running the full ingestion pipeline. Graph, canonical entities, provenance, temporal enrichment, trust hooks, consolidation, and optional backend synchronization are therefore absent from the default path.
2. Global/shared facts can be stored correctly and still disappear during semantic, BM25, and temporal candidate generation. This breaks the universal/cross-profile contract at recall time.
3. Public claims contain material inaccuracies: 74.8% is not a zero-LLM end-to-end result; the website calls arXiv preprints peer reviewed; paper links are assigned to the wrong product claims; cache defaults, compression ratios, Fisher-Rao usage, scale, and integration counts are overstated.

Additional release blockers include cross-home daemon attachment, unsafe context injection, confidence-score collapse, stale MCP profile names, an unsafe npm post-install fallback, missing slow-test isolation, and registry/GitHub release divergence.

## What is genuinely strong

The audit does not support dismissing SLM. It supports narrowing its promise to what can become defensible:

- Local-first, inspectable memory with a large operating surface and substantial automated coverage.
- One memory service accessible through MCP, CLI, hooks, HTTP, dashboard, LangChain, and LlamaIndex adapters.
- Exact cache replay with tenant-separated keys, byte-identical responses, invalidation, and stampede controls.
- Safe compression paths that preserve JSON and code and fail open.
- Research-backed experiments in information geometry, lifecycle behavior, associative retrieval, and quantization.
- A credible product wedge that most benchmark leaders do not own: private memory carried across coding agents and IDEs with provenance.

Those strengths become marketable only after the default path and the public claims agree.

## Alternatives

### Alternative A — publish another benchmark-led patch

Cost and time are lowest, but this is the wrong choice. It leaves the product contract broken, compounds credibility risk, and invites competitors to reproduce the mismatches publicly. It is easy to reverse technically and hard to reverse reputationally.

### Alternative B — rebuild as a new V4 before shipping again

This offers a clean architecture but has the highest cost, longest time to revenue, and greatest delivery risk. It discards a working base with thousands of tests and delays market learning. It is the least reversible option.

### Alternative C — corrective patch followed by a V3.7 integrity relaunch

First ship a narrowly scoped corrective patch for the P0 data-path, isolation, and claim defects. Then ship V3.7 only after normalized benchmarks, cross-IDE contract tests, security gates, and release provenance pass. This is moderate in cost, staged, reversible, and turns the audit itself into proof.

## L99 recommendation

Choose **Alternative C**.

The immediate release should be `3.6.24`, limited to release-blocking correctness and truth fixes. The public relaunch should be `3.7.0`, named around **Memory Integrity** rather than a new score. Do not start V4. Do not add more retrieval channels before fixing the five that currently produce candidates. Do not market “mesh” as distributed memory until facts, updates, deletes, conflicts, and identity are proven across machines.

## Hard decisions

1. **Freeze 74.8%, 82.3%, and 87.7% in product headlines.** Retain historical paper results only inside a protocol-labeled research table. The current post-change score is unknown.
2. **Remove “peer-reviewed,” “1M memories, zero slowdown,” “five years daily use,” “every recall uses Fisher-Rao,” “exact cache on by default,” “60–95% compression,” and “17+ auto-configured IDEs” until each has a release-linked evidence artifact.**
3. **Define one canonical ingestion contract and one canonical retrieval contract.** Fast paths may defer enrichment, but they may not mark an incomplete memory complete.
4. **Make raw evidence immutable and derived state rebuildable.** Facts, summaries, profiles, graph edges, and compressed forms must preserve source spans, versions, and supersession.
5. **Treat daemon identity as a security boundary.** A CLI home must never attach to a daemon merely because an expected port returns HTTP 200.
6. **Sell reliability, privacy, and portability—not the largest percentage.** “Private, persistent memory shared across coding agents and IDEs” is the correct wedge after the cross-client contract passes.
7. **Pursue revenue through a fixed-scope integration and reliability audit before building a cloud platform.** This creates paid learning without committing to multi-tenant operations prematurely.

## Exit decision

The repository may produce `3.6.24` when every `P0` row in the gap register has an executable regression test, package artifacts are reproducible, and public claims are corrected. The `3.7.0` relaunch may proceed only after the benchmark protocol, cross-IDE suite, security suite, upgrade/rollback suite, and evidence publication gates in the release plan are green.

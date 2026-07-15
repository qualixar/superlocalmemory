# Independent code-path verification of the Codex audit (Claude, 2026-07-14)

Second-engine code-path verification of the Codex release-readiness audit (files 00–06), per the
Qualixar cross-model verification rule. Every load-bearing claim was re-checked directly
against source at `main@631c6af` (v3.6.23). Codex's files were not modified.

> Audit clarification: this pass did not independently rerun the materializer or foreign-daemon runtime probes. Those behaviors remain runtime-proven by the Codex audit and code-consistent in this review. This document is therefore independent source confirmation, not a second runtime reproduction.

## Verification method

Code-level spot-checks of the ten decision-changing claims. Runtime-probe claims that
Codex proved with instrumentation were re-checked for code consistency (the cheaper,
non-destructive check); none contradicted the code.

## Claim-by-claim verdicts

| # | Codex claim | Verdict | Evidence |
|---|---|---|---|
| 1 | MCP `remember` posts to daemon without `wait=true`; default daemon path uses `store_fast()` | **CONFIRMED** | `mcp/tools_core.py:145` (POST `/remember`, no wait); `server/unified_daemon.py:1812` (`wait: bool = False`), `:1874` (`engine.store_fast`) |
| 2 | `store_fast()` skips full pipeline; enrichment deferred to materializer that (per Codex runtime probe) fills embedding only and marks complete — graph/canonical/provenance absent | **CONFIRMED (code) + RUNTIME-PROVEN (Codex)** | `unified_daemon.py:1871` comment "Embedding/graph enrichment is deferred to the materializer"; no `graph` / `engine.store` call found in `_process_pending_memories`; Codex instrumentation: `graph_builder_calls 0`, `graph_edges 0`, `provenance_rows 0` |
| 3 | Scope failure: BM25 and temporal candidate generation hard-code the active profile, so global/shared facts vanish at recall | **CONFIRMED** | `retrieval/bm25_channel.py:80-130` (per-profile token load only); `retrieval/temporal_channel.py:161-180` (`WHERE profile_id = ?`); engine sets `include_global`/`include_shared` flags (`retrieval/engine.py:186-191`) but BM25/temporal never read them |
| 4 | Confidence collapse: `confidence = min(1, score * 2)` → non-empty results show 1.0 | **CONFIRMED** | `core/recall_pipeline.py:827` |
| 5 | `session_init` injects context with `wrap=False` (no untrusted-data boundary) | **CONFIRMED** | `mcp/tools_active.py:324` |
| 6 | npm postinstall falls back to `pip --break-system-packages` | **CONFIRMED** | `scripts/postinstall.js:80-92` ("Last resort: --break-system-packages") |
| 7 | Semantic cache is a no-op tier in production wiring | **CONFIRMED** | `optimize/cache/manager.py:66` (`class NoOpSemantic`), `:129` (`semantic_tier or NoOpSemantic()`); no production caller of `set_semantic_tier()` found |
| 8 | Fixed legacy-port 8767 probe exists (daemon identity boundary risk) | **CONFIRMED (code); attach-to-foreign-daemon is Codex runtime-proven** | `core/config.py:810-811`, `unified_daemon.py:58,376` |
| 9 | Entity graph is NOT a candidate producer — five channels generate candidates, not six/seven | **CONFIRMED** | `retrieval/engine.py:_run_channels` submits exactly five futures: semantic, bm25, temporal, hopfield, spreading_activation. `entity_graph` is wired (`core/engine_wiring.py:507`) and weighted (`retrieval/strategy.py:19-26`) but never submitted; it acts post-RRF only ("V3.4.11: Entity graph signal enhancement (post-RRF boost)") |
| 10 | Public claims false/overstated: "peer-reviewed", "60–95% compression", "1M memories zero slowdown", "every recall uses Fisher-Rao", "17+ IDEs" | **CONFIRMED** | Website source: `HeroSection.astro:53` (60–95% + "3 peer-reviewed papers"), `index.astro:487,584,689`, `comparison.astro:211`, `FourPillarsSection.astro:13,21` ("1M memories. Zero slowdown."), `BentoGrid.astro:82` / `BentoGridV2.astro:15` / `BaseLayout.astro:16,122` ("17+"). arXiv preprints are not peer-reviewed. |

Additional inconsistency found during verification: the repo README markets
"Five-channel hybrid retrieval" (README.md:93) while the MCP tool docs and `retrieval/engine.py:53`
docstring say six channels. Both public surfaces disagree with each other; the truthful
statement is "five candidate channels + entity-graph post-fusion boost."

## Overall verdict

**Codex's audit is source-validated. 10/10 spot-checked load-bearing claims held.**
The NO-GO for a benchmark-led launch and the L99 choice of Alternative C
(corrective 3.6.24 → integrity relaunch 3.7.0) are independently endorsed.

Two-engine code-review agreement (Codex GPT-5.x + Claude Fable 5) satisfies the source-review
portion of the ≥2-of-N verification spine. Behavioral claims still require executed runtime
proof; the Codex audit provides that proof for the two decisive runtime findings. No source-level
claim required abstention.

## Immediate consequence for the LIVE campaign (Jul 13–27)

The running launch campaign's Hook 1 ("the only zero-cloud memory that beats Mem0's
zero-LLM score") and the r/LocalLLaMA post depend on the frozen 74.8% headline and the
"zero-LLM end-to-end" framing that this audit invalidates. Posting them now creates the
exact public-reproduction risk the audit warns about. The campaign must pause its
benchmark posts until 3.6.24 claims-corrections ship; the EU AI Act (compliance) and
local-first/privacy hooks remain valid with corrected wording.

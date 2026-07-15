# Launch, positioning, and revenue plan

## Premise challenge

“Make it viral everywhere” is not a strategy. Virality is an outcome of a compact story, a visible proof, and a distribution loop. Trying to sell memory, cache, compression, mathematics, mesh, 17+ integrations, enterprise compliance, and benchmark leadership at once makes SLM look unfocused and forces buyers to disprove too many claims.

The market is not waiting for another generic “AI memory.” SLM needs one wedge it can own and demonstrate in under three minutes.

## Current adoption truth

Snapshot on 2026-07-14:

- GitHub: 197 stars and 30 forks.
- npm: 5,362 rolling-month downloads.
- PyPI: 3,925 downloads over the comparable observed window.

These are attention/distribution signals, not unique people, active users, retained users, or production deployments. Repeat installs, CI, mirrors, bots, and upgrades affect package counts. Do not say “millions use competing products” or imply SLM has thousands of active users without auditable telemetry.

## Positioning alternatives

### Alternative A — benchmark champion

Story: “Best LoCoMo score, powered by information geometry.” It is easy to communicate, but the current score is not release-current, competitor protocols are not comparable, and a larger vendor can replace the headline within a week. This is high credibility risk and low defensibility.

### Alternative B — universal memory/cache/compression platform

Story: “One product replaces your memory, cache, compressor, mesh, graph, and agent infrastructure.” The addressable market sounds large, but the buying problem is vague and the shipped subsystems have unequal maturity. Sales cycles and support burden rise before a narrow success case exists.

### Alternative C — private cross-agent continuity with proof

Story: “Your coding agents share one private, source-attributed memory—across sessions and tools.” The demo is visual, the pain is frequent, local-first is meaningful, and the same proof naturally opens commercial licensing, reliability audits, and team deployment.

## L99 recommendation

Choose **Alternative C**. Use the information-geometric work as technical depth, cache/compression as secondary modules, and benchmarks as proof discipline. Do not lead with them.

Canonical line after the contract passes:

> SuperLocalMemory gives Codex, Claude Code, Cursor, and MCP agents one private, source-attributed memory that survives sessions and tool changes.

Qualixar category line:

> Built as AI Reliability Engineering: memory behavior is tested, traced, and recoverable—not assumed.

## Ideal buyers

### Primary — coding-agent and developer-tool teams

Small AI product teams building agents, IDE extensions, or developer tools need persistent project/user state but do not want to build ingestion, retrieval, provenance, deletion, and client adapters. They can buy a commercial license and integration support quickly.

### Secondary — privacy-sensitive engineering teams

Security, legal, healthcare, finance, and internal-platform teams want local/self-hosted memory, clear data paths, retention, deletion, backup, and operational support. They require more evidence and should enter after the integrity release.

### Later — enterprise platform standardization

“Microsoft/AWS grade” is an operating standard, not a feature list. Multi-tenancy, identity, SLOs, audit, recovery, signed supply chain, supported upgrades, and incident response must exist before targeting central enterprise platforms. Selling this too early creates support liabilities a solo founder cannot absorb.

## Immediate paid offer

Do not wait for SaaS billing or a cloud control plane. Sell a fixed-scope **Memory Integrity Sprint** to three founding design partners.

Suggested founding offer:

- Price: `$1,500` outside India or `₹99,000` in India, paid upfront.
- Scope: one team, one repository/workflow, two agent/IDE clients.
- Deliverables: current-state memory audit, SLM integration, source/scope policy, cross-client handoff test, recall quality baseline, injection/security check, backup/restore drill, and a signed findings report.
- Included: 30 days of defect support for the agreed integration and an evaluation commercial license.
- Excluded: custom cloud hosting, unlimited adapters, on-call SRE, and open-ended model tuning.
- Conversion: credit the sprint fee toward the first annual commercial license if purchased within 30 days.

Price alternatives:

- `$499`/`₹39,000` would reduce friction but underprices senior architecture work and attracts support-heavy hobby use.
- `$5,000+` is appropriate after two proof-backed case studies, but is too high for the first paid learning loop.

The recommended founding price is high enough to validate business value and low enough for a startup engineering lead to approve without enterprise procurement.

## Product-led proof assets

### 1. The cross-agent relay

A reproducible demo must show:

1. Claude Code records a decision with source and project scope.
2. Codex opens a new session and recalls it without transcript stuffing.
3. Cursor changes the decision; SLM shows supersession and both source versions.
4. Codex asks an adversarial question and SLM abstains or returns bounded evidence.
5. The user forgets the fact; every index and client stops returning it.

Record terminal commands, versions, timings, retrieved tokens, and the database state. Ship the script so viewers can rerun it.

### 2. Agent Memory Reliability Score

Create a standalone, provider-neutral command or repository that tests a memory endpoint for:

- read-your-write;
- update/supersession;
- contradiction handling;
- scope and tenant isolation;
- forget propagation;
- prompt-injection resistance;
- restart and upgrade survival;
- source attribution;
- context tokens and p95 latency.

Output a shareable JSON/HTML scorecard with raw cases. Competitors can implement the adapter. This turns SLM’s category—AI Reliability Engineering—into an open measurement loop instead of a slogan.

### 3. Benchmark forensics

Publish the normalized competitor audit as a technical article: “Why there is no 100% LoCoMo leaderboard.” Show QA versus retrieval, 1,540 versus 1,986 questions, context flooding, judge tolerance, missing artifacts, and SLM’s own corrected claims. Naming SLM’s mistakes first is the credibility move.

### 4. Memory Failure Museum

Maintain small, runnable failures: wrong-profile leak, stale update, poisoned instruction, partial enrichment, false cache hit, deleted fact still in graph, daemon identity collision. Each issue contains the failing test, fix, and regression artifact. Every resolved case becomes useful launch content and contributor bait.

## Launch sequence

### Stage 0 — truth reset

Correct the live website, README, wiki, CLI, package metadata, paper mapping, license text, and release state. Publish a short changelog titled “What we corrected before asking you to trust V3.7.” Do this before promotion.

### Stage 1 — build in public

For each P0 fix, publish one concise engineering note with the failing runtime contract and the passing regression. Avoid daily progress noise. The unit of content is evidence: failure → design rule → executable proof.

### Stage 2 — recruit design partners

Direct outreach beats broad launch traffic at this stage. Build a list of 30 teams that ship coding agents, MCP tools, or private internal agents. Send a brief founder-led note tied to their current architecture, offer the fixed-scope sprint, and ask for a 20-minute qualification call. The goal is one paid partner before the public relaunch.

### Stage 3 — V3.7 proof release

Release source, signed packages, evidence page, cross-agent relay, benchmark raw outputs, integration matrix, threat model, migration report, and known limitations together. The release post should lead with the handoff demo and link to the deeper math.

### Stage 4 — channel distribution

Use one canonical article/video and adapt it without changing facts:

- GitHub: evidence-first README and reproducible demo.
- Hacker News: “Show HN: one local memory shared by Codex, Claude Code, and Cursor.”
- LinkedIn: founder architecture narrative and design-partner result.
- X: short failure/proof clips and benchmark cards with protocol labels.
- YouTube AI-research channel: full technical walkthrough; never cross-post into the personal/Vedanta channel.
- Relevant Reddit and Discord communities: submit the open reliability suite, not promotional copy.
- Direct newsletters/podcasts: pitch benchmark methodology and cross-agent continuity, not “please cover my product.”

## Content system

One technical artifact should produce:

1. A long-form engineering post.
2. A five-minute demo.
3. Three short clips: cross-agent relay, adversarial memory, forget propagation.
4. One benchmark/protocol graphic.
5. One code-linked LinkedIn post.
6. One X thread with raw artifact links.
7. One contributor issue labeled with a runnable failure.

No naked numbers, paid-sounding superlatives, fictional enterprise use, or anonymous testimonials. Varun is the brand; write as a senior engineer exposing the real tradeoff.

## Distribution loops

- Every reliability scorecard includes “run this against your memory system” and an adapter link.
- Every client integration has a small official verification badge generated from the release matrix.
- Every bug report can contribute a sanitized Failure Museum case and receive attribution.
- Every design-partner report yields an approved, measured case study.
- Every benchmark result links raw outputs and invites reproduction under the same container.

These loops make sharing useful to the sharer. That is more durable than giveaways or exaggerated score cards.

## Metrics that matter

### Revenue

- Primary monthly objective: at least one paid Memory Integrity Sprint.
- Pipeline: qualified teams contacted, replies, calls, proposals, paid conversions, and time-to-close.

### Activation

- Install completed without manual recovery.
- First remembered fact.
- First successful recall in the same client.
- First successful handoff into a second client.
- First update/forget verified across clients.

### Reliability

- Enrichment completion and retry rate.
- Cross-scope false positive/false negative rate.
- p50/p95/p99 recall latency and retrieved tokens.
- Confidence calibration and abstention accuracy.
- Upgrade, recovery, and cross-client contract success.

### Retention and attention

Measure 7- and 30-day active installations only through explicit privacy-preserving opt-in or user-exported diagnostics. Track stars, downloads, impressions, and video views as secondary attention signals; never translate them into “users.”

## Stop-doing list

- Stop adding headline features until the P0/P1 contracts close.
- Stop publishing cross-vendor score tables without protocol normalization.
- Stop calling preprints peer reviewed.
- Stop counting templates or MCP-compatible names as verified integrations.
- Stop using “universal” to mean “there is probably an adapter.”
- Stop building a hosted enterprise platform before one team pays for the local product and support.

## Launch go/no-go

Go only when the `3.7.0` evidence page can prove the cross-agent relay, full benchmark protocol, signed supply chain, safe upgrade, injection boundary, scope isolation, and deletion propagation. If one of those is red, publish the progress and known limitation, but do not call it the integrity relaunch.

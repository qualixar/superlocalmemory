# SuperLocalMemory V3 release-readiness audit

Audit date: 2026-07-14

Audited source: `main@631c6af`

Published package version: `3.6.23`
Decision: **NO-GO for a benchmark-led or enterprise-grade relaunch**

This evidence pack separates shipped truth, benchmark truth, public claims, remediation, and go-to-market. That separation is deliberate: code behavior, benchmark outcomes, and sales copy have different proof requirements and must not inherit credibility from one another.

## Canonical documents

1. [Executive decision](00-EXECUTIVE-DECISION.md) — release verdict, hard decisions, and the recommended sequence.
2. [Architecture and shipped-truth audit](01-ARCHITECTURE-SHIPPED-TRUTH.md) — ingestion, retrieval, cache, compression, mesh, integrations, security, packages, CI, and runtime behavior.
3. [Benchmark and competitor audit](02-BENCHMARK-COMPETITOR-AUDIT.md) — protocol-normalized evidence and mechanics worth adopting.
4. [Website and public-claims audit](03-PUBLIC-CLAIMS-AUDIT.md) — claim-by-claim decisions and replacement language.
5. [Gap register](04-GAP-REGISTER.md) — severity, evidence, release gate, and acceptance criteria.
6. [Next-release plan](05-NEXT-RELEASE-PLAN.md) — corrective patch, integrity release, test matrix, and publication gates.
7. [Launch, positioning, and revenue plan](06-LAUNCH-GTM-PLAN.md) — market wedge, proof assets, channels, offer, and metrics.
8. [Claude code-path verification](07-CLAUDE-INDEPENDENT-VERIFICATION.md) — second-engine spot-check of ten decision-changing findings.
9. [V3.7 master program](08-V3.7-MASTER-PLAN.md) — approved single-public-release execution structure and closure policy.
10. [V3.7 defect ledger](09-V3.7-DEFECT-LEDGER.yaml) — machine-readable source of truth for every registered defect.
11. [V3.7 verification matrix](10-V3.7-VERIFICATION-MATRIX.md) — proof suites, artifact requirements, and release gates.
12. [V3.7 implementation Wave 1](11-V3.7-IMPLEMENTATION-WAVE-1.md) — daemon ownership and test-isolation changes, executable evidence, and remaining blockers.
13. [V3.7 implementation Wave 2](12-V3.7-IMPLEMENTATION-WAVE-2.md) — canonical data-root migration, two-HOME installed-artifact proof, discovered defects, and the installer approval gate.
14. [V3.7 implementation Wave 2B](13-V3.7-IMPLEMENTATION-WAVE-2B-INSTALLERS.md) — rebuilt npm, Unix/macOS, and Windows installer contracts; artifact lifecycle proof; consent boundary; discovered P0/P1 defects; and the unresolved DMG release decision.

## Evidence vocabulary

| Class | Meaning |
|---|---|
| `RUNTIME-PROVEN` | Executed against the current source or published artifact and observed directly. |
| `CODE-PROVEN` | Determined from an executable path and its wiring, but not fully exercised in the target production topology. |
| `ARTIFACT-RECOMPUTED` | Recomputed from a committed or public per-question result artifact; not a rerun of proprietary APIs. |
| `DOC-ONLY` | Present in documentation, a paper, a handoff, or a vendor claim but not independently reproduced. |
| `MISSING` | Required evidence does not exist or was not found. |
| `ABSTAIN` | The available evidence does not support a factual conclusion. |

## Scope and limits

The audit covered the full current SLM repository, published PyPI and npm artifacts, default and slow test selections, isolated runtime probes, the live website, release automation, historical handoffs and backups, and a source-locked cohort of leading memory systems. “Every memory system in the world” is not a finite or auditable scope; the competitor corpus is therefore risk-based and includes the systems with the strongest adoption signals, public code, benchmark claims, or architectural relevance.

The initial audit was read-only. After approval, implementation Waves 1 and 2 changed product and test code for daemon ownership, process isolation, diagnostic resilience, and canonical data-root behavior. Wave 2 has local source plus built wheel/sdist evidence, but it is not a frozen or published V3.7 release candidate; the implementation reports record the exact proof and remaining blockers.

## Release rule

A claim may ship only when its evidence is linked, current for the released commit, and generated under a declared protocol. A passing unit suite is not evidence that an end-to-end product contract works. Reading is not verification; run the contract.

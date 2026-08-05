# V4 release remediation plan

Scope: product candidate `071f03cb`, paper baseline `0503c4e2`, live Mode B data.

This plan implements the approved artifact-bound release route. The live database is
not upgraded until the branch passes the release gates and a fresh backup has been
verified.

## Runtime blockers

1. Make embedding-model migration fail closed. A partial migration must retain the
   old signature and remain retryable; activation occurs only after every target row
   has a verified embedding and model marker. The long-term shadow-index design is
   retained as an explicit follow-up unless implemented and fault-tested here.
2. Resolve cache and learning databases from the active data root during GDPR profile
   erasure. Completeness must fail closed when the target store cannot be resolved or
   verified.
3. Replace mode-derived EU AI Act legal verdicts with deployment-posture reporting.
   Intended purpose and deployment context determine legal classification. Compatibility
   fields remain present but cannot claim legal compliance.
4. Make enterprise preset enforcement truthful. PII redaction and retention are either
   wired to real runtime controls with tests or removed from the enforced preset until
   they are.
5. Derive the dashboard egress/privacy state from runtime configuration. Never display
   `LOCAL ONLY` for cloud, connector, backup, or mesh egress.
6. Add typed destructive confirmation and route operation remediation through it.
7. Ship the same twelve Codex skills from Python and repository/plugin installation
   paths.

## Artifact and paper gates

1. Parameterize experiment source paths and retain raw exp2b and governed-latency
   artifacts from the frozen release candidate.
2. Build wheel, sdist, and npm tarball; record SHA-256 hashes and clean-install evidence.
3. Run the full selected test suite, mode matrix, browser E2E, daemon restart/crash,
   resource soak with idle recovery, CLI, MCP, and Python SDK tests.
4. Reconcile the 34-page preprint and 80-page report from one claims ledger.
5. Preserve the published V3 LoCoMo 74.8% result as protocol-scoped carried-forward
   evidence. Use the published `+12.7 pp` ablation value or explain rounding.

## Authoritative research boundary

- EU AI Act legal risk classification is based on intended purpose and use context,
  not SLM mode or data locality. Source: European Commission, "Navigating the AI Act",
  checked 2026-08-05.
- The published V3 result is arXiv:2603.14588, Table 3: Mode A Retrieval 74.8%,
  10 conversations / 1,276 scored questions, local retrieval followed by GPT-4.1-mini
  answer construction and judging.

## TDD and release sequence

For every runtime blocker: add a focused failing regression test, record the RED commit,
apply the minimal fix, rerun focused and neighboring suites, record the GREEN commit,
then run GitNexus change detection before the commit. No live-data mutation is permitted
until the frozen artifact gates pass.

## Verified product-candidate gates

- Independent code and Python release reviews: no remaining blockers.
- Focused migration, retention, GDPR, deployment, and daemon gate: 158 passed.
- Full selected Python suite: 8,730 passed, 36 skipped, 328 deselected; exit 0.
- npm UI suite: 70 passed; plugin and Copilot-plugin checks in sync.
- Wheel, sdist, and npm tarball built successfully. Clean no-dependency wheel install
  reported version 4.0.0; wheel and npm tarball each contained all 12 Codex skills.
- GitNexus staged impact: medium risk, 11 files, 39 symbols, 4 execution flows.

# Codex Cloud Security Baseline

## Environment scope
Use one reusable Cloud environment per core product repository unless an approved task genuinely requires a second repository.

## Default posture
- Runtime internet: off.
- Start skill: unset until explicitly reviewed and approved.
- Production secrets: none.
- Deployment credentials: none.
- Organization-admin or cloud-admin credentials: none.
- Direct writes to the protected default branch: forbidden.
- Agent merge capability: forbidden.
- Automatic scans, reviews, monitoring, and scheduled tasks: disabled.

## Setup
Use deterministic repository-native install commands and pinned or locked dependencies where available. Do not embed credentials, private URLs, tokens, or production configuration in setup scripts.

## Temporary network access
If a task genuinely requires egress, approval must name the purpose and the smallest practical destination allowlist. Remove the exception after the task.

## GitHub publication
A Cloud worker may edit and test inside its isolated workspace only after implementation approval. Publishing a branch or draft PR requires a separate approval. Merge, release, and deploy remain human-only.

## Verification
Use the repository's existing tests, type checks, linters, and independent gates. Never weaken a gate solely to make work pass.

## Review mode
Code Review and security review are on-demand only. Repository security scans are one-time and explicitly requested. Commit-change monitoring and automatic review remain disabled.

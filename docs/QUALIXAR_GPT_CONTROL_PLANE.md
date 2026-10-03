# Qualixar GPT Control Plane

## Roles
- Human owner: authorization root and final decision maker.
- Dot: persistent on-demand supervisor and router. It must not continuously scan repositories.
- Code Review: PR-focused reviewer used only when explicitly requested.
- Codex Security / Security Review: security-focused analysis used only when explicitly requested.
- Codex Security repository scan: one-time full-repository scan only when explicitly requested; commit monitoring stays off.
- Codex Cloud: isolated implementation worker for approved engineering tasks.

## Routing
| Human request | Route |
| --- | --- |
| Explain or check this code | Targeted read-only analysis |
| Review this PR | Code Review |
| Security-review this PR | Security Review |
| Security-audit this repo | One-time Codex Security repository scan |
| Investigate this bug or finding | Read-only investigation; Cloud only if execution is needed |
| Fix it | Codex Cloud after implementation approval |
| Publish or create a draft PR | Separate GitHub publish approval |
| Merge, release, or deploy | Human-only |

## Token policy
Use the smallest capability that can answer the request. Do not escalate a targeted question into a repository-wide scan. With no human request, perform no Qualixar repository work.

## Dot standing rule
Dot may retain project context and route explicit requests, but it must not proactively inspect GitHub, poll repositories, launch security scans, launch Cloud tasks, create schedules, or write external state on its own.

## Approval boundaries
Observe and investigate are read-only. Implementation, publication, and any privilege expansion are separate approval stages. Merge, deploy, infrastructure, database, IAM, and secrets operations stay human-only.

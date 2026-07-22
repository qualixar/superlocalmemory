# GDPR Compliance

SuperLocalMemory V3.8.0 ships built-in controls that support GDPR compliance
programs. Full documentation: [docs/compliance.md](../docs/compliance.md).

These are engineering controls. Compliance depends on deployment configuration,
use case, and operator responsibility. SLM is not a legal certification.

## Controls summary

| Control | Where | CLI / Dashboard |
|---------|-------|-----------------|
| Data export (Art. 15/20) | Full profile data export as checksummed JSONL | `slm evidence export` · Dashboard → Governance → Data Privacy → Export |
| Erasure (Art. 17) | Profile deletion removes data from 30+ scoped tables | `slm profile delete <name>` · Dashboard → Governance → Data Privacy → Erase |
| Erasure audit log | Erasure is logged to the tamper-proof audit chain before any data is deleted | Dashboard → Governance → Audit |
| Retention rules (Art. 5) | Time-based expiry policies per workspace | `slm config set retention.default_policy gdpr-30d` |
| Audit trail (Art. 5(2)) | Hash-chained record of every store, recall, mutation, and erasure | `slm diagnostics export` · Dashboard → Governance → Audit |
| PII redaction | Configurable redaction before memory content crosses trust boundaries | `slm config set privacy.pii_redaction true` |

## Retention policies

```bash
slm config set retention.default_policy gdpr-30d      # 30-day GDPR policy
slm config set retention.default_policy hipaa-7y       # 7-year HIPAA retention
slm config set retention.default_policy indefinite     # No automatic expiry (personal default)
```

## Erasure completeness

Erasing a profile removes data from memory, entity, graph, mesh, audit,
session, backup-destination, and retention-rule tables. Erasure is logged to
the tamper-proof `audit_chain.db` before the delete operations begin, satisfying
Art. 5(2) accountability requirements.

After erasure, verify with:

```bash
slm diagnostics export /tmp/post-erase-report.json
```

## Notes for operators

- RBAC must be configured (`require_login = true`) to enforce access boundaries
  in multi-user deployments.
- PII redaction is heuristic, not exhaustive. Verify coverage for your content.
- The tamper-proof audit chain is stored separately from the main database and
  is excluded from erasure operations by design.

## Related pages

- [[RBAC and Teams]]
- [[Getting Started]]

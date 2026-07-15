# Compliance
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SuperLocalMemory exposes local storage, erasure, export, provenance, retention, access-policy, and audit controls. These controls can support a compliance program, but the software does not certify a deployment or replace legal and security review.

---

## EU AI Act

The EU AI Act (Regulation 2024/1689) establishes requirements for AI systems operating in the European Union.

### Mode A: local core processing

In Mode A, core storage and retrieval can run without sending memory content to a cloud model provider. Optional model and dependency downloads, connectors, cloud backup, proxies, and other enabled integrations may still use the network.

| Control area | Available technical control | Operator responsibility |
|--------------|-----------------------------|-------------------------|
| **Data location** | Configurable local data root for core memory state | Review optional networked features and operating-system access |
| **Erasure** | `slm forget`, profile deletion, and local export tooling | Include backups, exports, caches, indexes, and provider records in the deletion plan |
| **Transparency** | Local audit and provenance records | Validate coverage and retention for the released version |
| **Risk assessment** | Local modes reduce some external data flows | Classify the actual use case and surrounding AI system |

### Mode B: local model enrichment

Mode B uses an operator-managed Ollama endpoint. Its data path depends on where that endpoint runs and which optional integrations are enabled.

### Mode C: Shared Responsibility

Mode C sends recall queries to a cloud LLM provider. In this mode:

- **Your data** (stored memories) remains local
- **Queries** are sent to the cloud provider
- The cloud provider's compliance posture applies to those queries
- A Data Processing Agreement (DPA) with your provider is recommended

## GDPR

The General Data Protection Regulation applies to personal data of EU residents.

### Right to Erasure (Article 17)

```bash
# Delete specific memories
slm forget "John's phone number"
slm forget --id 42

# Delete all memories before a date
slm forget --before "2025-01-01"

# Delete an entire profile
slm profile delete client-eu
```

Deletion removes the selected local records. Operators must separately account for configured backups, exports, caches, derived indexes, snapshots, and any provider-side records.

### Data Portability (Article 20)

```bash
# Export all memories for a profile
slm profile export work > work-data.json

# The export is standard JSON — readable by any system
```

### Data Minimization (Article 5)

Automatic observations are admitted by explicit capture rules. Once accepted,
raw evidence and a queryable projection can be durable before enrichment runs;
entropy and consolidation behavior therefore cannot be described as an
unconditional pre-storage data-minimization guarantee.

### Purpose Limitation

Profiles are intended to scope data by purpose. Treat them as a product-level policy boundary, not a substitute for operating-system or tenant isolation, until the release's cross-scope security matrix is verified.

## Retention Policies

Named retention policies automate data lifecycle management.

### Built-in policies

| Policy | Retention period | Use case |
|--------|:----------------:|----------|
| `indefinite` | Forever | Default. No automatic deletion. |
| `gdpr-30d` | 30 days | GDPR-compliant short retention |
| `hipaa-7y` | 7 years | HIPAA medical records requirement |

### Apply a policy

```bash
slm retention set gdpr-30d
```

This applies to the active profile. Memories older than the retention period are automatically archived and then deleted.

### Custom policies

```bash
slm retention set custom --days 90
```

### Per-profile policies

Different profiles can have different policies:

```bash
slm profile switch client-eu
slm retention set gdpr-30d

slm profile switch internal
slm retention set indefinite
```

## Access Control

SuperLocalMemory uses Attribute-Based Access Control (ABAC) per profile.

- Each profile is an isolated access boundary
- No cross-profile data access is possible
- Profile switching requires the `slm profile switch` command
- The audit trail logs all profile switches

## Audit Trail

Every memory operation is logged in a tamper-evident hash chain.

### View the audit trail

```bash
slm audit                    # Recent entries
slm audit --limit 100        # Last 100 entries
slm audit --action store     # Only store operations
slm audit --action delete    # Only delete operations
```

### Verify integrity

```bash
slm audit --verify
```

This checks the hash chain for tampering. Each entry contains a hash of the previous entry, creating a blockchain-like chain. Any modification to a past entry breaks the chain and is detected.

### What gets logged

| Action | What is recorded |
|--------|-----------------|
| `store` | Memory ID, timestamp, profile, content hash |
| `recall` | Query, timestamp, profile, result count |
| `delete` | Memory ID, timestamp, profile, reason |
| `profile_switch` | From profile, to profile, timestamp |
| `mode_change` | From mode, to mode, timestamp |
| `migration` | Source version, target version, timestamp |

## HIPAA

For healthcare evaluations, validate the complete deployment with qualified legal and security reviewers. A starting technical checklist is:

1. Inventory and restrict every configured network and backup path
2. Apply the `hipaa-7y` retention policy
3. Use per-patient or per-case profiles for isolation
4. Enable audit trail verification for compliance audits

```bash
slm profile create patient-12345
slm profile switch patient-12345
slm retention set hipaa-7y
```

## SOC 2

SuperLocalMemory supports SOC 2 requirements through:

- **Access controls:** Profile-based isolation with ABAC
- **Audit logging:** Hash-chained, tamper-evident audit trail
- **Data encryption:** SQLite database can be encrypted at rest using OS-level encryption (FileVault, BitLocker, LUKS)
- **Change management:** All configuration changes are logged

## Compliance Checklist

| Requirement | Mode A | Mode B | Mode C |
|-------------|:------:|:------:|:------:|
| Data stays on device | Yes | Yes | Partial (queries sent to cloud) |
| Core path without cloud model provider | Available | Available with local Ollama | No |
| Right to erasure | Yes | Yes | Yes (local); cloud logs depend on provider |
| Audit trail | Yes | Yes | Yes |
| Retention policies | Yes | Yes | Yes |
| Profile isolation | Yes | Yes | Yes |
| Tamper detection | Yes | Yes | Yes |

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

# Encryption at Rest — Assessment & Posture (C4)

SuperLocalMemory is a **local-first** product: all data lives in SQLite files
under the per-user data directory (`~/.superlocalmemory/`). This note records
the encryption-at-rest posture and the concrete controls in place.

## Threat model

The data at risk is memory content, the tamper-evident audit chain, learning
signals, and (with RBAC enabled) user records. The relevant threats for a
local-first deployment are:

1. **Disk theft / cold storage** — laptop or server disk removed and read
   offline.
2. **Same-host other-user access** — another OS account on a shared machine
   reading the DB files.
3. **Backup leakage** — DB copied into an unprotected backup.

## Controls in place

| Control | Status | Notes |
|---|---|---|
| Full-disk encryption (macOS FileVault / LUKS / BitLocker) | **Primary control** | Defends threat (1). Verified FileVault ON on the reference machine. This is the recommended encryption-at-rest mechanism for a local-first app. |
| Data directory `0700` | ✅ enforced | `~/.superlocalmemory/` is owner-only; other users cannot traverse in. |
| DB files `0600` | ✅ enforced (C4) | `harden_db_perms()` (core/security_primitives.py) sets `0600` on every DB file + its `-wal`/`-shm` sidecars at open. Wired into `DatabaseManager`, the audit chain, and the pending store. Closes threat (2) even if the directory perms are later loosened. Historically the files shipped `0644` (world-readable). |
| Secret redaction before persistence | ✅ always on | `redact_secrets()` strips API keys/tokens from content. |
| PII redaction before persistence | ✅ opt-in (C4) | `SLM_PII_REDACTION=1` / `config.pii_redaction` scrubs email/phone/SSN/card/IP at ingest so identifiers never reach disk. |

## Why not application-level DB encryption by default

SQLCipher (page-level AES on the SQLite file) is the usual "encrypt the DB
file" answer, but for this product it is **not** the default because:

* It requires a non-stdlib driver (`pysqlcipher3`) and a SQLCipher build,
  breaking the zero-dependency `pip install superlocalmemory` promise and the
  editable-venv install used here.
* The key must live *somewhere* on the same machine the daemon runs on; without
  a hardware keystore this mostly re-implements what FileVault/LUKS already do
  at the block layer, with worse performance and more moving parts.
* Full-disk encryption already defends the disk-theft threat, and `0600` +
  `0700` defend the same-host threat.

## Recommendation for high-security / shared-host deployments

For a shared server hosting company memory where full-disk encryption is not
available or not trusted, run the daemon under a **dedicated OS user** (so
`0600`/`0700` fully isolate the data) and optionally point it at a **SQLCipher**
build. The DB-open path is centralized (`DatabaseManager`, audit chain, pending
store), so swapping in an encrypted driver is a contained change; the schema and
queries are unaffected. This is documented as an opt-in, not shipped by default.

## Backups

Backups produced by the backup subsystem inherit the source file perms and
should be written to an encrypted destination. Cloud-backup credentials are
already written `0600` (`infra/cloud_backup.py`).

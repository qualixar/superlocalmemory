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

Production backups use the legacy `BackupManager` independent per-file
`sqlite3.backup()` snapshots — `memory.db` primary plus any present managed
stores (`learning.db`, `audit_chain.db`, `code_graph.db`, `pending.db`,
`audit.db`). `lance/` is not included on the legacy path (only the coherent
`BackupCoordinator` primitive handles it as a companion). Companion-copy
failures are logged at `warning` as non-critical. Dashboard **Export**
(`POST /api/backup/export`, `server/routes/backup.py:export_backup`) produces
a single gzip of the latest `memory.db` snapshot only — not a whole-root
export.

Destination permissions: backup files are created by `sqlite3` and
`gzip.open`/`NamedTemporaryFile` and therefore **follow the process umask**,
not source-file `0600` inheritance. Do **not** assume `0600` on the backup
directory or its files. Place legacy backup destinations on an
encrypted/private volume, keep the directory `0700`, and **verify** resulting
snapshot files are owner-only (`0600`) after each backup/restore. There is no
wired whole-root backup/restore route in this release; for offline whole-root
copy/restore, `slm serve stop` first so WAL/SHM checkpoint, then copy the
complete data-root store set (all present `*.db`, `-wal`/`-shm` sidecars,
`lance/` if present). Restore is the inverse offline copy.

Credentials at rest:
- Cloud-backup credentials: OS keychain preferred; fallback is owner-only
  plaintext `~/.superlocalmemory/.credentials.json` via `_atomic_write_creds()`
  (`0600`, parent `0700`) — see `infra/cloud_backup.py`.
- Provider / reranker keys persisted in `config.json`: plaintext, atomic
  `0600` write (`core/config.py:SLMConfig.save()`) — prefer env
  (`OPENAI_API_KEY` / `SLM_CROSS_ENCODER_API_KEY`) to avoid disk persistence.
- Feedback HMAC key: `.feedback-hash-key` (`0600` beside `learning.db`,
  32 bytes); produces 16-hex (64-bit) pseudonym, not encryption.

## Migrations (V4.0.0)

`M038_learning_feedback_channel` (eager, `learning` DB) adds the `channel`
column for `pattern_miner` and is applied eagerly at startup.
`M039_scene_fact_members` (deferred, `memory` DB) builds the normalized
`scene_fact_members` projection with composite **profile-scoped**
membership: `PRIMARY KEY (scene_id, fact_id)`, foreign keys to both
`memory_scenes(profile_id, scene_id)` and `atomic_facts(profile_id, fact_id)`,
covering indexes `idx_scene_fact_members_lookup (profile_id, fact_id,
scene_id)` and `idx_scene_fact_members_order (scene_id, position)`, plus
triggers on `memory_scenes` insert/update keeping `fact_ids_json`
synchronized. Deferred until engine-owned tables exist; no manual
`slm db migrate` normally required. `slm db migrate` is forward-only
(`status`/`--dry-run`/apply) — no rollback. Downgrade requires a verified
pre-upgrade complete backup (stop daemon, whole-root copy including sidecars).

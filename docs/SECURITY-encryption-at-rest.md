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
| Full-disk encryption (macOS FileVault / LUKS / BitLocker) | **Primary control** | Defends threat (1). This is the recommended encryption-at-rest mechanism for a local-first app. |
| Data directory `0700` | ✅ enforced | `~/.superlocalmemory/` is owner-only; other users cannot traverse in. |
| DB files `0600` | ✅ enforced (C4) | `harden_db_perms()` (core/security_primitives.py) sets `0600` on every DB file + its `-wal`/`-shm` sidecars at open. Wired into `DatabaseManager`, the audit chain, and the pending store. Closes threat (2) even if the directory perms are later loosened. Historically the files shipped `0644` (world-readable). |
| Credentials you save | **Stored as written by default; a placeholder with `trust.secret_screen` on** | By default SuperLocalMemory keeps the keys, tokens and passwords you save, so an agent can recall them later; nothing is stripped on save, import or enrichment. With `trust.secret_screen` on (off by default), anything shaped like a credential is replaced by `[REDACTED:TYPE:LAST4]` before it is stored, on every door that saves text, and the original is kept nowhere. Full-disk encryption and the file permissions above are what protect what is stored. |
| Credential redaction on egress | ✅ always on | Every request that can carry memory text off this machine goes through one gate (`core/outbound_http.py`), which redacts credentials for any host that is not loopback: LLM providers, cloud embedders, the remote reranker, the Jev answer check, a LAN Ollama, mesh peers. Context injected into an agent session is redacted the same way. A test fails the build if new code opens an outbound request outside the gate. |
| PII redaction before persistence | ✅ opt-in (C4) | `SLM_PII_REDACTION=1` / `config.pii_redaction` scrubs email/phone/SSN/card/IP at ingest so identifiers never reach disk. |

## Why not application-level DB encryption by default

SQLCipher (page-level AES on the SQLite file) is the usual "encrypt the DB
file" answer, but for this product it is **not** the default because:

* It requires a non-stdlib driver (`pysqlcipher3`) and a SQLCipher build,
  breaking the zero-dependency `pip install superlocalmemory` promise 
* The key must live *somewhere* on the same machine the daemon runs on; without
  a hardware keystore this mostly re-implements what FileVault/LUKS already do
  at the block layer, with worse performance and more moving parts.
* Full-disk encryption already defends the disk-theft threat, and `0600` +
  `0700` keep other accounts out of the files themselves (for the running
  service, see Shared Macs below).

## Shared Macs and other multi-user computers

File permissions keep other user accounts out of SLM's files, but the SLM
service listens on this computer's local address and, in a personal install,
trusts every request that comes from this computer — including requests from
programs run by another account on the same Mac. On a computer shared by
several people with separate accounts, another account can therefore read your
memories through the service and change its settings. A web page on another
site cannot: the service refuses requests that are not addressed
to this computer.

A check of which account each request comes from is not in this release. Until
it exists:

- Prefer a Mac you do not share for SLM.
- On a shared Mac, turn on company mode (**Team → Access policy**, see
  [rbac-teams.md](rbac-teams.md#login-gate)): reading and saving memories then
  needs a signed-in user. The machine owner's administration, including the
  answer-check settings, stays reachable without signing in.

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
whole-root restore route for cloud backups; the copies SLM takes before an
update restore with `slm db restore`. For an offline whole-root copy/restore, `slm serve stop` first so WAL/SHM checkpoint, then copy the
complete data-root store set (all present `*.db`, `-wal`/`-shm` sidecars,
`lance/` if present). Restore is the inverse offline copy.

Credentials at rest:
- Cloud-backup credentials: OS keychain preferred; fallback is owner-only
  plaintext `~/.superlocalmemory/.credentials.json` via `_atomic_write_creds()`
  (`0600`, parent `0700`) — see `infra/cloud_backup.py`.
- Provider / reranker keys persisted in `config.json`: plaintext, atomic
  `0600` write  — prefer env
  (`OPENAI_API_KEY` / `SLM_CROSS_ENCODER_API_KEY`) to avoid disk persistence.
- Feedback HMAC key: `.feedback-hash-key` (`0600` beside `learning.db`,
  32 bytes); produces 16-hex (64-bit) pseudonym, not encryption.

## Migrations

Schema migrations are additive and apply automatically at startup. `slm db migrate`
(`--status`, `--dry-run`) inspects them and is forward-only: there is no
rollback. To go back, use a restore point (`slm db restore`, see
[Restore points](restore-points.md)) or a verified pre-upgrade complete backup
(stop the daemon, then copy the whole data root including sidecars).

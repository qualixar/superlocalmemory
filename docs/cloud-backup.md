# Cloud Backup — Google Drive & GitHub

SuperLocalMemory v3.4.10+ can automatically back up your memory databases to
**Google Drive** and **GitHub**. Credentials are stored in your OS keychain
(macOS Keychain, Windows Credential Locker, or Linux Secret Service) when
available. On systems without a keychain (headless Linux, containers without
Secret Service) they fall back to an owner-only plaintext
`~/.superlocalmemory/.credentials.json` (`0600`, parent `0700`, atomic write
via `_atomic_write_creds` in `src/superlocalmemory/infra/cloud_backup.py`) —
not encrypted. Protect that file and the data root with volume encryption and
owner-only modes; prefer a keychain-capable host when possible.

## GitHub Backup (Recommended)

GitHub backup works out of the box. No additional setup needed beyond a Personal Access Token.

### Setup (2 minutes)

1. Open the SLM Dashboard: `http://localhost:8765`
2. Click the **account widget** in the sidebar (bottom), then click the **GitHub icon**
3. You'll see the "Connect GitHub" form:

   - **Personal Access Token**: Click [Create one here](https://github.com/settings/tokens/new?scopes=repo&description=SLM+Backup) — this opens GitHub with the `repo` scope pre-selected. Click "Generate token" and copy it.
   - **Repository Name**: Default is `slm-backup`. Change if you want.

4. Click **Connect**

That's it. SLM will:
- Verify your token
- Create a **private** repository when the named repository does not exist
- Reuse an existing repository without changing or verifying its visibility;
  confirm that an existing backup repository is private before connecting
- Initialize it with a README
- Show your GitHub avatar and username in the sidebar

### How It Works

- Each GitHub backup is an independent per-file `memory-*.db` snapshot plus
  companion `learning-*.db` etc. when present, published as a **GitHub
  Release** with those files as assets. Not a coherent cross-store epoch;
  companion upload failures are non-critical for the primary `memory.db`.
- The included stores are the `MANAGED_DATABASES` set
  (`src/superlocalmemory/infra/backup.py`): `memory.db`, `learning.db`,
  `audit_chain.db`, `code_graph.db`, `pending.db`, `audit.db` — only those
  present are sent. Canonical M018 ingestion operations and raw evidence live
  in `memory.db`; `pending.db` is a legacy offline compatibility spool where
  present. `lance/` is not included on the legacy path.
- Only the last **5 releases** are kept — older ones are automatically deleted
  to prevent storage bloat
- Backups run in the background — the dashboard never freezes
- Snapshot files follow process `umask`; verify `0600`/`0700` on the data root
  and any manual copy destination

### Restoring from GitHub

1. `slm serve stop` — stop the daemon so WAL/SHM checkpoint before overwriting
   live files
2. Go to your `slm-backup` repo on GitHub
3. Click **Releases** in the sidebar
4. Download the `.db` files (plus `-wal`/`-shm` sidecars and `lance/` if you
   saved a whole-root copy) from the latest release
5. Copy them into `~/.superlocalmemory/` (offline replace) and verify they are
   owner-only (`0600`/`0700`) on an encrypted/private volume
6. Run `slm restart` (there is no wired in-place `restore` route for whole-root
   sets in this release)

---

## Google Drive Backup

Google Drive backup requires a one-time OAuth client setup through Google Cloud Console. This is a Google requirement for any application that accesses Drive on behalf of users.

### Why Is This Needed?

Google requires every application to register an "OAuth client" before it can access your Drive. This is a security measure — it ensures you know exactly which application is accessing your data. For GitHub, a simple Personal Access Token is enough, but Google's security model is stricter.

### Setup (5 minutes)

#### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the project dropdown (top bar) → **New Project**
3. Name it anything (e.g., `slm-backup`) → **Create**
4. Select the new project from the dropdown

#### Step 2: Enable APIs

1. Go to **APIs & Services** → **Library**
2. Search for and enable:
   - **Google Drive API**
   - **People API** (for showing your email/name)

#### Step 3: Configure OAuth Consent Screen

1. Go to **APIs & Services** → **OAuth consent screen**
2. Select **External** → **Create**
3. Fill in:
   - **App name**: `SuperLocalMemory` (or anything)
   - **User support email**: your Gmail
   - **Developer contact email**: your Gmail
4. Click **Save and Continue** through the remaining steps
5. Go to **Test users** → **Add users** → add your Gmail address

#### Step 4: Create OAuth Client

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth client ID**
3. Application type: **Web application**
4. Name: `SLM Dashboard` (or anything)
5. Under **Authorized redirect URIs**, add:
   ```
   http://localhost:8765/api/backup/oauth/google/callback
   ```
6. Click **Create**
7. Copy the **Client ID** and **Client Secret** (you'll need both)

#### Step 5: Connect in SLM

1. Open the SLM Dashboard: `http://localhost:8765`
2. Click the **Google icon** in the sidebar account widget
3. Paste your **Client ID** and **Client Secret**
4. Click **Save & Connect Google Drive**
5. Google's login page opens — sign in and click **Allow**
6. You'll see "Google Drive Connected!" — close the popup

### How It Works

- Backups are uploaded as independent per-file snapshots to a `SLM-Backup`
  folder in your Google Drive; files are **replaced in-place** (no duplicates)
- Included: `MANAGED_DATABASES` (`memory.db`, `learning.db`, `audit_chain.db`,
  `code_graph.db`, `pending.db`, `audit.db`) — only present files are sent;
  `lance/` is not included on the legacy path. Not a coherent epoch.
  Companion upload failures log as non-critical and do not fail the primary.
- OAuth credentials are stored in your OS keychain when available; otherwise
  fallback to owner-only plaintext `.credentials.json` (`0600`, see above)
- Backups run in the background; resulting snapshot files follow process
  `umask` — verify `0600`/`0700` and keep the destination private/encrypted

### Restoring from Google Drive

1. `slm serve stop`
2. Open Google Drive → `SLM-Backup` folder
3. Download all `.db` files (plus sidecars/`lance/` if present for a whole-root
   copy)
4. Copy them to `~/.superlocalmemory/` and verify owner-only modes on an
   encrypted/private volume
5. Run `slm restart` (no wired whole-root restore route; offline copy is the
   supported path)

---

## Sync & Schedule

### Manual Sync

Click **Sync Now** (cloud upload icon) in the sidebar account widget, or go to **Settings** → **Cloud Backup** → **Sync Now**.

### Auto-Backup

SLM automatically creates local backups on a schedule (default: weekly). When cloud destinations are connected, backups are also pushed to the cloud after each auto-backup.

Configure the schedule in **Settings** → **Backup Configuration**:
- **Interval**: Daily or Weekly
- **Max backups**: How many local backups to keep (default: 10)

### Export

Click the **download icon** in the sidebar to export a compressed `.gz` backup
file. This is `POST /api/backup/export` — it creates a single gzipped
`memory-*.db` snapshot (`memory.db`-only) via `BackupManager` and streams it as
a temporary `*.db.gz` (removed after the response). Not a coherent whole-root
export; for a whole-root copy stop the daemon and copy the data-root store set
offline.

---

## What Gets Backed Up

| Database | Contents | Typical Size | Backup inclusion |
|---|---|---|---|
| `memory.db` | Facts, M018 operations/raw evidence, entities, graph edges, embeddings, sessions | Deployment-specific | Always if present (primary) |
| `learning.db` | Learning signals, behavioral patterns, ranker data | 0.5 — 5 MB | If present via `_backup_all_dbs` (non-critical companion) |
| `audit_chain.db` | Audit trail, compliance provenance | 0.5 — 2 MB | If present (companion, warning on failure) |
| `code_graph.db` | Code knowledge graph (if used) | 0.1 — 10 MB | If present (companion) |
| `pending.db` | Legacy offline spool awaiting canonical M018 replay (when present) | Deployment-specific | If present (companion) |
| `audit.db` | Legacy audit (pre-v3.4) | — | If present (companion) |

Production backups use `BackupManager`: independent per-file SQLite
`sqlite3.backup()` snapshots, one file at a time — **not** a coherent
cross-store epoch/manifest/atomic set. `sqlite3.backup()` creates a hot
consistent snapshot of *that* file even while the daemon runs, but there is
no cross-store epoch or atomic rollback across stores on the legacy path.
`lance/` is not included on the legacy path (the coherent `BackupCoordinator`
primitive captures it as an out-of-manifest companion, but it is not wired to
these routes). Destination files follow process `umask`; verify `0600`/`0700`
after copy. For offline whole-root backup/restore: `slm serve stop` first so
WAL/SHM checkpoint, then copy the complete data-root store set.

`MANAGED_DATABASES` registry: `src/superlocalmemory/infra/backup.py`.

---

## Security

- **New GitHub backup repositories are created private.** Existing repositories
  are reused without a visibility check or enforcement, so verify the selected
  repository is private before uploading any memory database.
- **Credentials — OS keychain preferred** — macOS Keychain, Windows Credential
  Locker, or Linux Secret Service when available. **Fallback:** owner-only
  plaintext `~/.superlocalmemory/.credentials.json` (`0600`, parent `0700`,
  atomic `_atomic_write_creds`) on systems without a keychain (headless Linux,
  containers) — not encrypted. Keep that file on an encrypted/private volume
  and verify `0600`/`0700` after writes. Provider/reranker keys persisted in
  `~/.superlocalmemory/config.json` are likewise plaintext protected only by
  atomic `0600` (`core/config.py:SLMConfig.save()`); prefer env
  (`OPENAI_API_KEY` / `SLM_CROSS_ENCODER_API_KEY`) to avoid disk persistence.
- **Google OAuth tokens** are refresh tokens — they can be revoked from your [Google Account Security page](https://myaccount.google.com/permissions)
- **GitHub PATs** can be revoked from [GitHub Settings → Tokens](https://github.com/settings/tokens)
- **Backup destination permissions:** snapshot files follow process `umask`,
  not `0600` inheritance — verify owner-only modes. Use an encrypted/private
  destination for backups/exports (see `docs/SECURITY-encryption-at-rest.md`).
  No wired whole-root restore route — offline `slm serve stop` + copy is the
  supported restore path.
- **No zero-loss claim for the legacy path:** per-file copies have no coherent
  epoch; offline whole-root copy while stopped is the consistent set.

---

## Troubleshooting

### "Sync failed" in the sidebar
Check the destination status in **Settings** → **Cloud Backup**. Common causes:
- GitHub: PAT expired or revoked → reconnect with a new token
- Google: OAuth token expired → click "Connect Google Drive" again to re-authorize

### Google Drive shows "Connection Failed"
- Make sure you added yourself as a **test user** in the OAuth consent screen
- Verify the redirect URI matches exactly: `http://localhost:8765/api/backup/oauth/google/callback`
- Check that the SLM daemon is running on port 8765

### Dashboard freezes during sync
This was fixed in v3.4.10 — syncs now run in a background thread. If you use
the Python path, activate the SLM virtual environment and run
`python -m pip install --upgrade superlocalmemory`.

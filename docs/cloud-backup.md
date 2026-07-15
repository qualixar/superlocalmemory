# Cloud Backup — Google Drive & GitHub

SuperLocalMemory v3.4.10+ can automatically back up your memory databases to **Google Drive** and **GitHub**. All credentials are stored in your OS keychain (macOS Keychain, Windows Credential Locker, or Linux Secret Service) — never in plaintext.

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
- Create a **private** repository (always private — your data is never public)
- Initialize it with a README
- Show your GitHub avatar and username in the sidebar

### How It Works

- Each backup creates a **GitHub Release** with your database files as assets
- The configured database set is included. Canonical M018 ingestion operations and raw evidence live in `memory.db`; `pending.db` is a legacy offline compatibility spool where present.
- Only the last **5 releases** are kept — older ones are automatically deleted to prevent storage bloat
- Backups run in the background — the dashboard never freezes

### Restoring from GitHub

1. Go to your `slm-backup` repo on GitHub
2. Click **Releases** in the sidebar
3. Download the `.db` files from the latest release
4. Copy them to `~/.superlocalmemory/`
5. Run `slm restart`

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

- Backups are uploaded to a `SLM-Backup` folder in your Google Drive
- Files are **replaced in-place** (no duplicates, no storage bloat)
- ALL databases are backed up, not just memory.db
- Your OAuth credentials are stored in your OS keychain
- Backups run in the background

### Restoring from Google Drive

1. Open Google Drive → `SLM-Backup` folder
2. Download all `.db` files
3. Copy them to `~/.superlocalmemory/`
4. Run `slm restart`

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

Click the **download icon** in the sidebar to export a compressed `.gz` backup file you can store anywhere.

---

## What Gets Backed Up

| Database | Contents | Typical Size |
|---|---|---|
| `memory.db` | Facts, M018 operations/raw evidence, entities, graph edges, embeddings, sessions | Deployment-specific |
| `learning.db` | Learning signals, behavioral patterns, ranker data | 0.5 — 5 MB |
| `audit_chain.db` | Audit trail, compliance provenance | 0.5 — 2 MB |
| `code_graph.db` | Code knowledge graph (if used) | 0.1 — 10 MB |
| `pending.db` | Legacy offline spool awaiting canonical M018 replay (when present) | Deployment-specific |

All databases are backed up using SQLite's `sqlite3.backup()` API, which creates a consistent, atomic snapshot even while the daemon is running.

---

## Security

- **GitHub repos are always private** — hardcoded, cannot be changed
- **Credentials stored in OS keychain** — macOS Keychain, Windows Credential Locker, or Linux Secret Service
- **Fallback**: On systems without a keychain (headless Linux), credentials are stored in `~/.superlocalmemory/.credentials.json` with `chmod 0600` (owner-only)
- **Google OAuth tokens** are refresh tokens — they can be revoked from your [Google Account Security page](https://myaccount.google.com/permissions)
- **GitHub PATs** can be revoked from [GitHub Settings → Tokens](https://github.com/settings/tokens)

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

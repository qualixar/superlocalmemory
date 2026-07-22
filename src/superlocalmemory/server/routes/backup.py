# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3.4.10 "Fortress" - Backup Routes

Routes:
  Local:  /api/backup/status, /api/backup/create, /api/backup/configure, /api/backup/list
  Cloud:  /api/backup/destinations, /api/backup/connect/github, /api/backup/connect/gdrive,
          /api/backup/disconnect/{id}, /api/backup/sync, /api/backup/export
"""
import logging
import gzip
import hashlib
import shutil
import urllib.parse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional

from .helpers import BackupConfigRequest, DB_PATH, MEMORY_DIR

logger = logging.getLogger("superlocalmemory.routes.backup")
router = APIRouter()


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log full traceback server-side; return a generic message to the client."""
    logger.exception("backup route error")
    return HTTPException(status_code=500, detail=detail)


# Feature flags
BACKUP_AVAILABLE = False
CLOUD_AVAILABLE = False
try:
    from superlocalmemory.infra.backup import BackupManager
    BACKUP_AVAILABLE = True
except ImportError:
    pass

try:
    from superlocalmemory.infra.cloud_backup import (
        get_destinations, add_destination, remove_destination,
        connect_github, connect_google_drive,
        sync_all_destinations, update_sync_status,
    )
    CLOUD_AVAILABLE = True
except ImportError:
    pass


def _get_backup_manager() -> "BackupManager":
    """Get V3 backup manager instance."""
    return BackupManager(db_path=DB_PATH, base_dir=MEMORY_DIR)


# ---- Request models -------------------------------------------------------

class GitHubConnectRequest(BaseModel):
    pat: str = Field(..., min_length=1)
    repo_name: str = Field(default="slm-backup")

class GDriveConnectRequest(BaseModel):
    auth_code: str = Field(..., min_length=1)
    redirect_uri: str = Field(default="http://localhost:8765/api/backup/oauth/callback")

class GDriveClientConfig(BaseModel):
    client_id: str = Field(..., min_length=1)
    client_secret: str = Field(..., min_length=1)


# ---- Local backup routes (existing) ---------------------------------------

@router.get("/api/backup/status")
async def backup_status():
    """Get auto-backup system status + cloud destinations."""
    if not BACKUP_AVAILABLE:
        return {"status": "not_implemented", "message": "Backup module not available"}
    try:
        manager = _get_backup_manager()
        status = manager.get_status()
        if CLOUD_AVAILABLE:
            status["cloud_destinations"] = get_destinations(DB_PATH)
        else:
            status["cloud_destinations"] = []
        return status
    except Exception:
        raise _internal_error("Backup status error")


@router.post("/api/backup/create")
async def backup_create():
    """Create a manual backup immediately."""
    if not BACKUP_AVAILABLE:
        return {"success": False, "message": "Backup module not available"}
    try:
        manager = _get_backup_manager()
        filename = manager.create_backup(label='manual')
        if filename:
            return {
                "success": True, "filename": str(filename),
                "message": f"Backup created: {filename}",
                "status": manager.get_status(),
            }
        return {"success": False, "message": "Backup failed"}
    except Exception:
        raise _internal_error("Backup create error")


@router.post("/api/backup/configure")
async def backup_configure(request: BackupConfigRequest):
    """Update auto-backup configuration."""
    if not BACKUP_AVAILABLE:
        return {"success": False, "message": "Backup module not available"}
    try:
        manager = _get_backup_manager()
        result = manager.configure(
            interval_hours=request.interval_hours,
            max_backups=request.max_backups,
            enabled=request.enabled,
        )
        return {"success": True, "message": "Backup configuration updated", "status": result}
    except Exception:
        raise _internal_error("Backup configure error")


@router.get("/api/backup/list")
async def backup_list(request: Request):
    """List all available backups."""
    if not BACKUP_AVAILABLE:
        return {"backups": [], "count": 0, "message": "Backup module not available"}
    # This GET is not covered by the mutation middleware. The local machine
    # owner (loopback) is trusted; a remote caller must present a credential;
    # non-loopback uncredentialed callers fail closed. Using the mutation-actor
    # boundary (not require_write_actor) so the same-origin dashboard — whose
    # fetch wrapper only attaches the install token to mutating requests — can
    # still read its own backup list.
    from superlocalmemory.server.write_identity import require_http_mutation_actor
    require_http_mutation_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                                actor_kind="backup-list")
    try:
        manager = _get_backup_manager()
        backups = manager.list_backups()
        return {"backups": backups, "count": len(backups)}
    except Exception:
        raise _internal_error("Backup list error")


# ---- Cloud destination routes (v3.4.10) -----------------------------------

@router.get("/api/backup/destinations")
async def list_destinations():
    """List all configured cloud backup destinations."""
    if not CLOUD_AVAILABLE:
        return {"destinations": [], "cloud_available": False}
    return {"destinations": get_destinations(DB_PATH), "cloud_available": True}


@router.post("/api/backup/connect/github")
async def connect_github_route(request: GitHubConnectRequest):
    """Connect GitHub as a backup destination using PAT."""
    if not CLOUD_AVAILABLE:
        raise HTTPException(status_code=501, detail="Cloud backup module not available")
    result = connect_github(request.pat, request.repo_name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/api/backup/connect/gdrive/config")
async def configure_gdrive_client(request: GDriveClientConfig):
    """Store Google OAuth client credentials (one-time setup)."""
    if not CLOUD_AVAILABLE:
        raise HTTPException(status_code=501, detail="Cloud backup module not available")
    from superlocalmemory.infra.cloud_backup import _store_credential
    _store_credential("gdrive_client_id", request.client_id)
    _store_credential("gdrive_client_secret", request.client_secret)
    return {"success": True, "message": "Google OAuth client configured"}


@router.post("/api/backup/connect/gdrive")
async def connect_gdrive_route(request: GDriveConnectRequest):
    """Complete Google Drive OAuth2 flow with authorization code."""
    if not CLOUD_AVAILABLE:
        raise HTTPException(status_code=501, detail="Cloud backup module not available")
    result = connect_google_drive(request.auth_code, request.redirect_uri)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/api/backup/disconnect/{dest_id}")
async def disconnect_destination(dest_id: str):
    """Remove a cloud backup destination."""
    if not CLOUD_AVAILABLE:
        raise HTTPException(status_code=501, detail="Cloud backup module not available")
    ok = remove_destination(dest_id, DB_PATH)
    if not ok:
        raise HTTPException(status_code=404, detail="Destination not found")
    return {"success": True, "message": "Destination disconnected"}


@router.post("/api/backup/sync")
async def sync_cloud():
    """Manually trigger sync to all cloud destinations.

    Runs the upload in a background thread so it doesn't block the
    dashboard. Returns immediately with status 'syncing'. The actual
    upload status is reflected in the destination's last_sync_status.
    """
    import asyncio
    import threading

    if not CLOUD_AVAILABLE:
        raise HTTPException(status_code=501, detail="Cloud backup module not available")
    if not BACKUP_AVAILABLE:
        raise HTTPException(status_code=501, detail="Backup module not available")

    # Create backup synchronously (fast — SQLite .backup is ~2s)
    manager = _get_backup_manager()
    filename = manager.create_backup(label="cloud-sync")
    if not filename:
        raise HTTPException(status_code=500, detail="Failed to create backup")

    # Run the cloud upload in a background thread (non-blocking)
    def _sync_background():
        try:
            sync_all_destinations(DB_PATH)
        except Exception as exc:
            logger.error("Background cloud sync failed: %s", exc)

    thread = threading.Thread(target=_sync_background, daemon=True)
    thread.start()

    return {
        "success": True,
        "backup": filename,
        "sync": {"status": "syncing", "message": "Upload started in background. Check destination status for progress."},
    }


# ---- Export / Download route (v3.4.10) ------------------------------------

@router.get("/api/backup/export")
async def export_backup(request: Request):
    """Create and download a compressed backup archive."""
    if not BACKUP_AVAILABLE:
        raise HTTPException(status_code=501, detail="Backup module not available")
    # Full-database download. This GET is not covered by the mutation
    # middleware and is triggered by a top-level navigation (window.location),
    # which cannot carry a custom credential header — so it uses the
    # loopback-trusted mutation-actor boundary: the local owner may export,
    # a non-loopback caller without a credential fails closed.
    from superlocalmemory.server.write_identity import require_http_mutation_actor
    require_http_mutation_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                                actor_kind="backup-export")

    manager = _get_backup_manager()
    filename = manager.create_backup(label="export")
    if not filename:
        raise HTTPException(status_code=500, detail="Failed to create backup")

    backup_path = MEMORY_DIR / "backups" / filename
    if not backup_path.exists():
        raise HTTPException(status_code=500, detail="Backup file not found")

    # Compress for download
    gz_path = backup_path.with_suffix(".db.gz")
    with open(backup_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return FileResponse(
        path=str(gz_path),
        media_type="application/gzip",
        filename=gz_path.name,
    )


# ---- OAuth SSO Flows (v3.4.10) -------------------------------------------
# These routes handle the browser popup flow:
# 1. /start → redirect to provider's login page
# 2. Provider redirects back to /callback with auth code
# 3. /callback exchanges code for tokens, stores in keychain, shows success page

_OAUTH_SUCCESS_HTML = """<!DOCTYPE html>
<html><head><title>Connected!</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0;
  display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
.card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(0,212,170,0.3);
  border-radius: 16px; padding: 40px; text-align: center; max-width: 400px; }}
.icon {{ font-size: 48px; margin-bottom: 16px; }}
h2 {{ color: #00D4AA; margin: 0 0 8px; }}
p {{ color: #999; margin: 0 0 20px; }}
.btn {{ background: #00D4AA; color: #0a0a0f; border: none; padding: 10px 24px;
  border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; }}
.btn:hover {{ background: #00b894; }}
</style></head>
<body><div class="card">
<div class="icon">{icon}</div>
<h2>{title}</h2>
<p>{message}</p>
<button class="btn" onclick="window.close()">Close Window</button>
</div></body></html>"""

_OAUTH_ERROR_HTML = """<!DOCTYPE html>
<html><head><title>Connection Failed</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0;
  display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
.card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,71,87,0.3);
  border-radius: 16px; padding: 40px; text-align: center; max-width: 400px; }}
.icon {{ font-size: 48px; margin-bottom: 16px; }}
h2 {{ color: #ff4757; margin: 0 0 8px; }}
p {{ color: #999; margin: 0 0 20px; font-size: 13px; }}
.btn {{ background: #333; color: #e0e0e0; border: none; padding: 10px 24px;
  border-radius: 8px; cursor: pointer; font-size: 14px; }}
</style></head>
<body><div class="card">
<div class="icon">{icon}</div>
<h2>Connection Failed</h2>
<p>{error}</p>
<button class="btn" onclick="window.close()">Close</button>
</div></body></html>"""


# S8-SEC-04: OAuth success / error pages render user-influenced strings
# (``error``, ``error_description``, provider ``message`` / ``title``).
# ``str.format`` doesn't escape HTML, so a hostile OAuth callback can
# inject ``<script>`` into these pages. These helpers HTML-escape every
# interpolated value before emitting the template.
import html as _html


def _oauth_error_page(icon: str, error: str) -> str:
    return _OAUTH_ERROR_HTML.format(
        icon=_html.escape(str(icon), quote=True),
        error=_html.escape(str(error), quote=True),
    )


def _oauth_success_page(icon: str, title: str, message: str) -> str:
    return _OAUTH_SUCCESS_HTML.format(
        icon=_html.escape(str(icon), quote=True),
        title=_html.escape(str(title), quote=True),
        message=_html.escape(str(message), quote=True),
    )


# ---- Google OAuth SSO Flow ------------------------------------------------

@router.get("/api/backup/oauth/google/start")
async def google_oauth_start(request: Request):
    """Start Google OAuth2 flow — redirects to Google's login page."""
    if not CLOUD_AVAILABLE:
        return HTMLResponse(_oauth_error_page(icon="&#x26A0;", error="Cloud backup module not available"))

    from superlocalmemory.infra.cloud_backup import _get_credential

    client_id = _get_credential("gdrive_client_id")
    if not client_id:
        return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Set Up Google Drive</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0;
  display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; padding: 20px; }
.card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px; padding: 32px; max-width: 520px; width: 100%; }
h2 { color: #e0e0e0; margin: 0 0 4px; font-size: 20px; }
.sub { color: #999; font-size: 13px; margin-bottom: 20px; }
.step { padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.step-num { display: inline-block; width: 24px; height: 24px; border-radius: 50%;
  background: rgba(0,212,170,0.15); color: #00D4AA; text-align: center; line-height: 24px;
  font-size: 12px; font-weight: 600; margin-right: 8px; }
.step-text { color: #ccc; font-size: 13px; }
code { background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-size: 12px; }
.fields { margin-top: 20px; }
label { display: block; color: #bbb; font-size: 12px; margin-bottom: 4px; }
input { width: 100%; box-sizing: border-box; padding: 8px 10px; border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04);
  color: #e0e0e0; font-size: 13px; margin-bottom: 12px; }
input:focus { outline: none; border-color: #00D4AA; }
.btn { background: #00D4AA; color: #0a0a0f; border: none; padding: 10px 24px;
  border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; width: 100%; }
.btn:hover { background: #00b894; }
.btn:disabled { opacity: 0.5; }
#status { margin-top: 8px; font-size: 12px; }
a { color: #00D4AA; }
</style></head>
<body><div class="card">
<h2>Connect Google Drive</h2>
<p class="sub">Google Drive backup requires a one-time OAuth setup. Follow these steps:</p>

<div class="step"><span class="step-num">1</span>
<span class="step-text">Go to <a href="https://console.cloud.google.com/apis/credentials" target="_blank">Google Cloud Console</a> and create a project</span></div>

<div class="step"><span class="step-num">2</span>
<span class="step-text">Enable <strong>Google Drive API</strong> and <strong>People API</strong></span></div>

<div class="step"><span class="step-num">3</span>
<span class="step-text">Go to <strong>OAuth consent screen</strong> &rarr; External &rarr; add your email as test user</span></div>

<div class="step"><span class="step-num">4</span>
<span class="step-text">Go to <strong>Credentials</strong> &rarr; Create OAuth Client &rarr; <strong>Web application</strong></span></div>

<div class="step"><span class="step-num">5</span>
<span class="step-text">Add redirect URI: <code>http://localhost:8765/api/backup/oauth/google/callback</code></span></div>

<div class="step" style="border:none"><span class="step-num">6</span>
<span class="step-text">Paste Client ID and Secret below:</span></div>

<div class="fields">
<label>Client ID</label>
<input type="text" id="cid" placeholder="xxxx.apps.googleusercontent.com">
<label>Client Secret</label>
<input type="password" id="csec" placeholder="GOCSPX-xxxx">
<button class="btn" id="saveBtn" onclick="saveAndConnect()">Save &amp; Connect Google Drive</button>
<div id="status"></div>
</div>

<p style="color:#555;font-size:11px;margin-top:16px;">
Your credentials are stored in your OS keychain (macOS Keychain / Windows Credential Locker) &mdash; never in plaintext.
Full guide: <a href="https://github.com/qualixar/superlocalmemory/wiki/Cloud-Backup#google-drive-backup" target="_blank">Cloud Backup Wiki</a>
</p>
</div>
<script>
async function saveAndConnect() {
  var cid = document.getElementById('cid').value.trim();
  var csec = document.getElementById('csec').value.trim();
  if (!cid || !csec) { document.getElementById('status').innerHTML = '<span style="color:#ff4757">Both fields required</span>'; return; }
  document.getElementById('saveBtn').disabled = true;
  document.getElementById('status').innerHTML = '<span style="color:#999">Saving...</span>';
  try {
    var resp = await fetch('/api/backup/connect/gdrive/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({client_id: cid, client_secret: csec})
    });
    if (resp.ok) {
      window.location.href = '/api/backup/oauth/google/start';
    } else {
      document.getElementById('status').innerHTML = '<span style="color:#ff4757">Failed to save</span>';
      document.getElementById('saveBtn').disabled = false;
    }
  } catch(e) {
    document.getElementById('status').innerHTML = '<span style="color:#ff4757">Error: ' + e.message + '</span>';
    document.getElementById('saveBtn').disabled = false;
  }
}
</script></body></html>""")

    # Build the Google OAuth URL
    base_url = str(request.base_url).rstrip("/")
    redirect_uri = f"{base_url}/api/backup/oauth/google/callback"

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/userinfo.email",
        "access_type": "offline",
        "prompt": "consent",
    })

    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@router.get("/api/backup/oauth/google/callback")
async def google_oauth_callback(request: Request, code: str = "", error: str = ""):
    """Google OAuth2 callback — exchanges code for tokens."""
    if error:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error=f"Google denied access: {error}"))

    if not code:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error="No authorization code received"))

    base_url = str(request.base_url).rstrip("/")
    redirect_uri = f"{base_url}/api/backup/oauth/google/callback"

    result = connect_google_drive(code, redirect_uri)

    if "error" in result:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error=result["error"]))

    return HTMLResponse(_oauth_success_page(
        icon="&#x2601;&#xFE0F;",
        title="Google Drive Connected!",
        message=f"Signed in as {result.get('email', 'unknown')}. Your memories will be backed up automatically."
    ))


# ---- GitHub OAuth SSO Flow ------------------------------------------------
# Uses GitHub Device Flow (no OAuth App needed — works with PATs too)
# But for best UX, we use GitHub OAuth Web Flow if a GitHub App is configured,
# otherwise fall back to Device Flow with a nice UI.

@router.get("/api/backup/oauth/github/start")
async def github_oauth_start(request: Request):
    """Start GitHub OAuth flow."""
    if not CLOUD_AVAILABLE:
        return HTMLResponse(_oauth_error_page(icon="&#x26A0;", error="Cloud backup module not available"))

    from superlocalmemory.infra.cloud_backup import _get_credential

    # Check if GitHub OAuth App is configured (client_id)
    gh_client_id = _get_credential("github_client_id")

    if gh_client_id:
        # Full OAuth Web Flow — browser redirects to GitHub login
        base_url = str(request.base_url).rstrip("/")
        redirect_uri = f"{base_url}/api/backup/oauth/github/callback"

        params = urllib.parse.urlencode({
            "client_id": gh_client_id,
            "redirect_uri": redirect_uri,
            "scope": "repo",
            "state": hashlib.sha256(base_url.encode()).hexdigest()[:16],
        })
        return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")

    # No OAuth App — show a friendly PAT entry form (still in the popup)
    return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Connect GitHub</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0;
  display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
.card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px; padding: 32px; max-width: 440px; width: 100%; }
h2 { color: #e0e0e0; margin: 0 0 4px; font-size: 20px; }
.sub { color: #999; font-size: 13px; margin-bottom: 20px; }
label { display: block; color: #bbb; font-size: 13px; margin-bottom: 4px; font-weight: 500; }
input { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04);
  color: #e0e0e0; font-size: 14px; margin-bottom: 16px; }
input:focus { outline: none; border-color: #00D4AA; }
.btn { background: #00D4AA; color: #0a0a0f; border: none; padding: 10px 24px;
  border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; width: 100%; }
.btn:hover { background: #00b894; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
a { color: #00D4AA; }
.hint { color: #666; font-size: 12px; margin-top: 12px; }
#status { margin-top: 12px; font-size: 13px; }
</style></head>
<body><div class="card">
<h2>Connect GitHub</h2>
<p class="sub">Back up your memories to a private GitHub repository.</p>

<label>Personal Access Token</label>
<input type="password" id="pat" placeholder="ghp_xxxxxxxxxxxxxxxxxxxx">

<label>Repository Name</label>
<input type="text" id="repo" value="slm-backup" placeholder="slm-backup">

<button class="btn" id="connectBtn" onclick="doConnect()">Connect</button>
<div id="status"></div>

<p class="hint">
  Need a token? <a href="https://github.com/settings/tokens/new?scopes=repo&description=SLM+Backup" target="_blank">Create one here</a> (select <code>repo</code> scope).
  Your token is stored securely in your OS keychain.
</p>
</div>
<script>
async function doConnect() {
  var pat = document.getElementById('pat').value.trim();
  var repo = document.getElementById('repo').value.trim();
  if (!pat) { document.getElementById('status').innerHTML = '<span style="color:#ff4757">Token required</span>'; return; }

  var btn = document.getElementById('connectBtn');
  btn.disabled = true; btn.textContent = 'Connecting...';
  document.getElementById('status').innerHTML = '<span style="color:#999">Verifying token and creating repo...</span>';

  try {
    var resp = await fetch('/api/backup/connect/github', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pat: pat, repo_name: repo || 'slm-backup'})
    });
    var data = await resp.json();
    if (resp.ok) {
      document.body.innerHTML = '<div class="card" style="text-align:center;background:rgba(255,255,255,0.05);border:1px solid rgba(0,212,170,0.3);border-radius:16px;padding:40px;max-width:400px;">' +
        '<div style="font-size:48px;margin-bottom:16px;">&#x2705;</div>' +
        '<h2 style="color:#00D4AA;margin:0 0 8px;">GitHub Connected!</h2>' +
        '<p style="color:#999;margin:0 0 20px;">Repository: ' + (data.repo || repo) + '</p>' +
        '<button class="btn" style="background:#00D4AA;color:#0a0a0f;border:none;padding:10px 24px;border-radius:8px;cursor:pointer;font-weight:600;" onclick="window.close()">Close Window</button></div>';
    } else {
      document.getElementById('status').innerHTML = '<span style="color:#ff4757">' + (data.detail || 'Connection failed') + '</span>';
      btn.disabled = false; btn.textContent = 'Connect';
    }
  } catch(e) {
    document.getElementById('status').innerHTML = '<span style="color:#ff4757">Connection failed</span>';
    btn.disabled = false; btn.textContent = 'Connect';
  }
}
</script></body></html>""")


@router.get("/api/backup/oauth/github/callback")
async def github_oauth_callback(request: Request, code: str = "", error: str = ""):
    """GitHub OAuth callback — exchanges code for access token."""
    if error:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error=f"GitHub denied access: {error}"))

    if not code:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error="No authorization code received"))

    from superlocalmemory.infra.cloud_backup import _get_credential, _store_credential
    import httpx

    gh_client_id = _get_credential("github_client_id")
    gh_client_secret = _get_credential("github_client_secret")

    if not gh_client_id or not gh_client_secret:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error="GitHub OAuth App not configured"))

    try:
        # Exchange code for access token
        resp = httpx.post(
            "https://github.com/login/oauth/access_token",
            json={"client_id": gh_client_id, "client_secret": gh_client_secret, "code": code},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        data = resp.json()
        access_token = data.get("access_token")
        if not access_token:
            return HTMLResponse(_oauth_error_page(icon="&#x274C;", error=data.get("error_description", "Failed to get access token")))

        # Use the token to connect
        result = connect_github(access_token, "slm-backup")
        if "error" in result:
            return HTMLResponse(_oauth_error_page(icon="&#x274C;", error=result["error"]))

        return HTMLResponse(_oauth_success_page(
            icon="&#x2705;",
            title="GitHub Connected!",
            message=f"Repository: {result.get('repo', 'slm-backup')}. Your memories will be backed up automatically."
        ))

    except Exception as exc:
        return HTMLResponse(_oauth_error_page(icon="&#x274C;", error=str(exc)))

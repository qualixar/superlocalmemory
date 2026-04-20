# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.4.10 "Fortress" — Cloud Backup Infrastructure.

Pushes local SQLite backups to configured cloud destinations:
  - Google Drive (OAuth2 + Drive API v3)
  - GitHub (PAT + Releases API)

All credentials stored in OS keychain via `keyring` library.
All operations are non-blocking and failure-tolerant.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, UTC, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("superlocalmemory.cloud_backup")

MEMORY_DIR = Path.home() / ".superlocalmemory"
DB_PATH = MEMORY_DIR / "memory.db"
KEYRING_SERVICE = "superlocalmemory"

# ---------------------------------------------------------------------------
# Credential management (OS keychain)
# ---------------------------------------------------------------------------


def _get_credential_store() -> Path:
    """Fallback encrypted credential file for systems without a keychain."""
    return MEMORY_DIR / ".credentials.json"


def _store_credential(key: str, value: str) -> bool:
    """Store a credential in the OS keychain (macOS/Windows/Linux).

    Falls back to an encrypted local file on headless Linux or if
    the keyring backend is unavailable.
    """
    # Try OS keychain first (macOS Keychain, Windows Credential Locker, Linux SecretService)
    try:
        import keyring
        from keyring.errors import NoKeyringError
        keyring.set_password(KEYRING_SERVICE, key, value)
        return True
    except (ImportError, NoKeyringError):
        pass  # No keyring backend — use fallback
    except Exception as exc:
        logger.debug("Keyring store failed, using fallback: %s", exc)

    # Fallback: local file with restricted permissions (0600)
    try:
        store_path = _get_credential_store()
        existing = {}
        if store_path.exists():
            existing = json.loads(store_path.read_text())
        existing[key] = value
        store_path.write_text(json.dumps(existing))
        store_path.chmod(0o600)  # Owner read/write only
        return True
    except Exception as exc:
        logger.warning("Failed to store credential '%s': %s", key, exc)
        return False


def _get_credential(key: str) -> str | None:
    """Retrieve a credential from the OS keychain or fallback store."""
    # Try OS keychain first
    try:
        import keyring
        from keyring.errors import NoKeyringError
        val = keyring.get_password(KEYRING_SERVICE, key)
        if val is not None:
            return val
    except (ImportError, NoKeyringError):
        pass
    except Exception:
        pass

    # Fallback: local file
    try:
        store_path = _get_credential_store()
        if store_path.exists():
            data = json.loads(store_path.read_text())
            return data.get(key)
    except Exception:
        pass

    return None


def _delete_credential(key: str) -> bool:
    """Delete a credential from the OS keychain and fallback store."""
    deleted = False

    try:
        import keyring
        from keyring.errors import NoKeyringError
        keyring.delete_password(KEYRING_SERVICE, key)
        deleted = True
    except (ImportError, NoKeyringError):
        pass
    except Exception:
        pass

    # Also clean from fallback
    try:
        store_path = _get_credential_store()
        if store_path.exists():
            data = json.loads(store_path.read_text())
            if key in data:
                del data[key]
                store_path.write_text(json.dumps(data))
                store_path.chmod(0o600)
                deleted = True
    except Exception:
        pass

    return deleted


# ---------------------------------------------------------------------------
# Destination registry (backed by SQLite)
# ---------------------------------------------------------------------------


def get_destinations(db_path: Path | None = None) -> list[dict[str, Any]]:
    """List all configured backup destinations."""
    path = db_path or DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM backup_destinations ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def add_destination(
    destination_type: str,
    display_name: str,
    config: dict[str, Any],
    credentials_ref: str = "",
    db_path: Path | None = None,
) -> str:
    """Register a new backup destination. Returns destination ID."""
    from superlocalmemory.storage.models import _new_id

    dest_id = _new_id()
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO backup_destinations "
            "(id, destination_type, display_name, credentials_ref, config, "
            "created_at, enabled) VALUES (?, ?, ?, ?, ?, ?, 1)",
            (dest_id, destination_type, display_name, credentials_ref,
             json.dumps(config), datetime.now(UTC).isoformat()),
        )
        conn.commit()
        logger.info("Added backup destination: %s (%s)", display_name, destination_type)
        return dest_id
    finally:
        conn.close()


def remove_destination(dest_id: str, db_path: Path | None = None) -> bool:
    """Remove a backup destination and its credentials."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT credentials_ref FROM backup_destinations WHERE id = ?",
            (dest_id,),
        ).fetchone()
        if row and row[0]:
            _delete_credential(row[0])
        conn.execute("DELETE FROM backup_destinations WHERE id = ?", (dest_id,))
        conn.commit()
        return True
    except Exception as exc:
        logger.error("Failed to remove destination %s: %s", dest_id, exc)
        return False
    finally:
        conn.close()


def update_sync_status(
    dest_id: str,
    status: str,
    error: str = "",
    db_path: Path | None = None,
) -> None:
    """Update the sync status of a destination."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "UPDATE backup_destinations SET last_sync_at = ?, "
            "last_sync_status = ?, last_sync_error = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), status, error, dest_id),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Google Drive integration
# ---------------------------------------------------------------------------


def connect_google_drive(auth_code: str, redirect_uri: str) -> dict[str, Any]:
    """Complete Google Drive OAuth2 flow and register as a destination.

    Args:
        auth_code: Authorization code from OAuth2 consent screen redirect.
        redirect_uri: The redirect URI used in the OAuth2 flow.

    Returns:
        Dict with destination_id, user_email, and status.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return {"error": "google-auth-oauthlib not installed. Run: pip install google-auth-oauthlib google-api-python-client"}

    # OAuth2 client config (public client — no secret needed for installed apps)
    client_config = {
        "installed": {
            "client_id": _get_credential("gdrive_client_id") or "",
            "client_secret": _get_credential("gdrive_client_secret") or "",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    if not client_config["installed"]["client_id"]:
        return {"error": "Google OAuth client not configured. Set client_id via dashboard."}

    try:
        flow = Flow.from_client_config(
            client_config,
            scopes=[
                "openid",
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/userinfo.email",
            ],
            redirect_uri=redirect_uri,
        )
        flow.fetch_token(code=auth_code)
        creds = flow.credentials

        # Store refresh token in keychain
        cred_key = "gdrive_refresh_token"
        _store_credential(cred_key, creds.refresh_token or "")
        _store_credential("gdrive_access_token", creds.token or "")

        # Get user email for display
        from googleapiclient.discovery import build
        service = build("oauth2", "v2", credentials=creds)
        user_info = service.userinfo().get().execute()
        email = user_info.get("email", "unknown")

        # Register destination
        dest_id = add_destination(
            destination_type="google_drive",
            display_name=f"Google Drive ({email})",
            config={"email": email, "folder": "SLM-Backup"},
            credentials_ref=cred_key,
        )

        return {"destination_id": dest_id, "email": email, "status": "connected"}

    except Exception as exc:
        logger.error("Google Drive connection failed: %s", exc)
        return {"error": str(exc)}


def _get_drive_service() -> Any | None:
    """Build an authenticated Google Drive service."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        refresh_token = _get_credential("gdrive_refresh_token")
        client_id = _get_credential("gdrive_client_id")
        client_secret = _get_credential("gdrive_client_secret")

        if not all([refresh_token, client_id, client_secret]):
            return None

        creds = Credentials(
            token=_get_credential("gdrive_access_token"),
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
        )
        return build("drive", "v3", credentials=creds)
    except Exception as exc:
        logger.warning("Failed to build Drive service: %s", exc)
        return None


def sync_to_google_drive(backup_path: Path, dest_config: dict) -> bool:
    """Upload a backup file to Google Drive."""
    service = _get_drive_service()
    if service is None:
        logger.warning("Google Drive not connected")
        return False

    try:
        from googleapiclient.http import MediaFileUpload

        folder_name = dest_config.get("folder", "SLM-Backup")

        # Find or create the SLM-Backup folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
        folders = results.get("files", [])

        if folders:
            folder_id = folders[0]["id"]
        else:
            folder_meta = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = service.files().create(body=folder_meta, fields="id").execute()
            folder_id = folder["id"]

        # Upload the backup file
        file_meta = {"name": backup_path.name, "parents": [folder_id]}

        # Check if file already exists (update instead of create duplicate)
        existing = service.files().list(
            q=f"name='{backup_path.name}' and '{folder_id}' in parents and trashed=false",
            spaces="drive",
            fields="files(id)",
        ).execute().get("files", [])

        media = MediaFileUpload(
            str(backup_path),
            mimetype="application/x-sqlite3",
            resumable=True,
        )

        if existing:
            service.files().update(
                fileId=existing[0]["id"],
                media_body=media,
            ).execute()
        else:
            service.files().create(
                body=file_meta,
                media_body=media,
                fields="id",
            ).execute()

        logger.info("Uploaded %s to Google Drive/%s", backup_path.name, folder_name)
        return True

    except Exception as exc:
        logger.error("Google Drive upload failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# GitHub integration
# ---------------------------------------------------------------------------


def connect_github(pat: str, repo_name: str = "slm-backup") -> dict[str, Any]:
    """Register GitHub as a backup destination using a PAT.

    Args:
        pat: Personal Access Token with 'repo' scope.
        repo_name: Name for the backup repo (created if doesn't exist).

    Returns:
        Dict with destination_id, username, repo, and status.
    """
    import httpx

    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        # Verify PAT and get username
        resp = httpx.get("https://api.github.com/user", headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"error": f"Invalid GitHub token (HTTP {resp.status_code})"}

        username = resp.json()["login"]
        full_repo = f"{username}/{repo_name}"

        # Check if repo exists, create if not
        resp = httpx.get(f"https://api.github.com/repos/{full_repo}", headers=headers, timeout=15)
        repo_created = False
        if resp.status_code == 404:
            resp = httpx.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json={
                    "name": repo_name,
                    "private": True,
                    "description": "SuperLocalMemory automated backup — managed by SLM",
                    "auto_init": True,  # Creates initial commit with README
                },
                timeout=15,
            )
            if resp.status_code not in (200, 201):
                return {"error": f"Failed to create repo: {resp.text}"}
            repo_created = True

        # Ensure repo has at least one commit (for Releases API to work)
        if not repo_created:
            commits_resp = httpx.get(
                f"https://api.github.com/repos/{full_repo}/commits",
                headers=headers,
                params={"per_page": 1},
                timeout=15,
            )
            if commits_resp.status_code != 200 or not commits_resp.json():
                # Empty repo — create initial file via Contents API
                import base64
                readme = base64.b64encode(
                    b"# SLM Backup\nAutomated SuperLocalMemory backup.\nEach release = one backup snapshot.\n"
                ).decode()
                httpx.put(
                    f"https://api.github.com/repos/{full_repo}/contents/README.md",
                    headers=headers,
                    json={"message": "init: SLM backup repo", "content": readme},
                    timeout=15,
                )

        # Store PAT in keychain
        cred_key = "github_pat"
        _store_credential(cred_key, pat)

        # Register destination
        dest_id = add_destination(
            destination_type="github",
            display_name=f"GitHub ({full_repo})",
            config={"username": username, "repo": repo_name, "full_repo": full_repo},
            credentials_ref=cred_key,
        )

        return {"destination_id": dest_id, "username": username, "repo": full_repo, "status": "connected"}

    except Exception as exc:
        logger.error("GitHub connection failed: %s", exc)
        return {"error": str(exc)}


def sync_to_github(backup_files: list[Path] | Path, dest_config: dict) -> bool:
    """Upload backup files as GitHub Release assets.

    Creates ONE release per sync, with ALL database backups as assets.
    Uses Releases API to avoid Git LFS (100MB file limit per asset,
    2GB per release — plenty for SLM databases).

    Args:
        backup_files: Single path or list of paths to upload.
        dest_config: Destination config with full_repo key.
    """
    import httpx

    # Accept both single path and list
    if isinstance(backup_files, Path):
        backup_files = [backup_files]

    pat = _get_credential("github_pat")
    if not pat:
        logger.warning("GitHub PAT not found in keychain")
        return False

    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github.v3+json",
    }

    full_repo = dest_config.get("full_repo", "")
    if not full_repo:
        return False

    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tag_name = f"backup-{timestamp}"
        total_mb = sum(f.stat().st_size for f in backup_files) / (1024 * 1024)
        file_list = ", ".join(f"{f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)" for f in backup_files)

        # Create ONE release for all files
        release_resp = httpx.post(
            f"https://api.github.com/repos/{full_repo}/releases",
            headers=headers,
            json={
                "tag_name": tag_name,
                "name": f"SLM Backup {timestamp}",
                "body": f"Automated backup — {len(backup_files)} databases ({total_mb:.1f} MB total)\n\n{file_list}",
                "draft": False,
                "prerelease": False,
            },
            timeout=30,
        )

        if release_resp.status_code not in (200, 201):
            logger.error("GitHub release creation failed: %s", release_resp.text[:200])
            return False

        upload_url = release_resp.json()["upload_url"].split("{")[0]
        uploaded = 0

        # Upload each database as a separate asset on the same release
        for backup_path in backup_files:
            try:
                with open(backup_path, "rb") as f:
                    upload_resp = httpx.post(
                        upload_url,
                        params={"name": backup_path.name},
                        headers={
                            "Authorization": f"token {pat}",
                            "Content-Type": "application/octet-stream",
                        },
                        content=f.read(),
                        timeout=600,
                    )
                if upload_resp.status_code in (200, 201):
                    uploaded += 1
                    logger.info("Uploaded %s to release %s", backup_path.name, tag_name)
                else:
                    logger.warning("Failed to upload %s: %s", backup_path.name, upload_resp.text[:100])
            except Exception as exc:
                logger.warning("Failed to upload %s: %s", backup_path.name, exc)

        logger.info("GitHub sync: %d/%d files uploaded to release %s", uploaded, len(backup_files), tag_name)

        # Cleanup: keep only last MAX_GITHUB_RELEASES releases to prevent repo bloat
        _cleanup_old_releases(full_repo, headers)

        return uploaded > 0

    except Exception as exc:
        logger.error("GitHub sync failed: %s", exc)
        return False


MAX_GITHUB_RELEASES = 5  # Keep last 5 backups, delete older ones


def _cleanup_old_releases(full_repo: str, headers: dict) -> None:
    """Delete old GitHub releases to prevent repo storage from exploding.

    Keeps the most recent MAX_GITHUB_RELEASES releases and deletes the rest.
    """
    import httpx

    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{full_repo}/releases",
            headers=headers,
            params={"per_page": 100},
            timeout=15,
        )
        if resp.status_code != 200:
            return

        releases = resp.json()
        if len(releases) <= MAX_GITHUB_RELEASES:
            return

        # Sort by creation date (newest first), delete everything after MAX
        releases.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        to_delete = releases[MAX_GITHUB_RELEASES:]

        for release in to_delete:
            release_id = release["id"]
            tag = release.get("tag_name", "")

            # Delete release
            httpx.delete(
                f"https://api.github.com/repos/{full_repo}/releases/{release_id}",
                headers=headers,
                timeout=15,
            )
            # Delete the tag too (releases leave orphan tags)
            httpx.delete(
                f"https://api.github.com/repos/{full_repo}/git/refs/tags/{tag}",
                headers=headers,
                timeout=15,
            )
            logger.info("Cleaned up old release: %s", tag)

        logger.info("GitHub cleanup: removed %d old releases, kept %d", len(to_delete), MAX_GITHUB_RELEASES)

    except Exception as exc:
        logger.warning("GitHub release cleanup failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Sync orchestrator
# ---------------------------------------------------------------------------


def _find_latest_backup_set(backup_dir: Path) -> list[Path]:
    """Find the latest complete backup set (all DBs from the same timestamp).

    Returns list of backup files (memory + all companions) from the most
    recent backup run.
    """
    # Find latest memory backup to get the timestamp
    memory_backups = sorted(
        backup_dir.glob("memory-*.db"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not memory_backups:
        return []

    latest = memory_backups[0]
    # Extract timestamp+suffix from "memory-20260414-212803-cloud-sync.db"
    ts_suffix = latest.name.replace("memory-", "").replace(".db", "")

    # Find all companion DB backups with the same timestamp
    backup_set = [latest]
    for f in backup_dir.iterdir():
        if f == latest or not f.name.endswith(".db"):
            continue
        if ts_suffix in f.name:
            backup_set.append(f)

    return backup_set


def sync_all_destinations(db_path: Path | None = None) -> dict[str, Any]:
    """Sync latest backup set (ALL databases) to all enabled cloud destinations."""
    path = db_path or DB_PATH
    results: dict[str, Any] = {}

    destinations = get_destinations(path)
    if not destinations:
        return {"synced": 0, "message": "No cloud destinations configured"}

    backup_dir = path.parent / "backups"
    if not backup_dir.exists():
        return {"synced": 0, "message": "No backups directory"}

    backup_set = _find_latest_backup_set(backup_dir)
    if not backup_set:
        return {"synced": 0, "message": "No backup files found"}

    synced = 0
    db_names = [f.name for f in backup_set]
    total_size_mb = sum(f.stat().st_size for f in backup_set) / (1024 * 1024)

    for dest in destinations:
        if not dest.get("enabled"):
            continue

        dest_id = dest["id"]
        dest_type = dest["destination_type"]
        config = json.loads(dest.get("config", "{}"))

        try:
            if dest_type == "google_drive":
                ok = all(sync_to_google_drive(f, config) for f in backup_set)
            elif dest_type == "github":
                ok = sync_to_github(backup_set, config)
            else:
                ok = False

            status = "success" if ok else "failed"
            update_sync_status(dest_id, status, db_path=path)
            if ok:
                synced += 1
            results[dest_id] = {"type": dest_type, "status": status}

        except Exception as exc:
            update_sync_status(dest_id, "failed", str(exc), db_path=path)
            results[dest_id] = {"type": dest_type, "status": "failed", "error": str(exc)}

    return {"synced": synced, "total": len(destinations), "results": results}

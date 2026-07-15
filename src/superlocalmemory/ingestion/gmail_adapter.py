# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Gmail ingestion adapter — 3 tiers of Gmail access.

Tier 1: File import (.mbox from Google Takeout) — zero setup
Tier 1.5: IMAP polling — no GCP, just email/password
Tier 2: Gmail API with OAuth polling — needs GCP OAuth client, no Pub/Sub
Tier 3: Gmail API with Pub/Sub push — full GCP (future)

OPT-IN only. Enabled via: slm adapters enable gmail

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from superlocalmemory.ingestion.base_adapter import BaseAdapter, AdapterConfig, IngestItem
from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger("superlocalmemory.ingestion.gmail")


def _adapters_config_path() -> Path:
    return state_path("adapters.json")


def _import_dir() -> Path:
    return state_path("import")


class GmailAdapter(BaseAdapter):
    """Gmail ingestion with automatic tier detection."""

    source_type = "gmail"

    def __init__(self, config: AdapterConfig | None = None, tier: str = "auto"):
        super().__init__(config)
        self._tier = tier
        self._mbox_path: str | None = None
        self._mbox_processed = False
        self._history_id: str | None = None
        self._poll_interval = 300  # 5 min for API polling

    def run(self) -> None:
        """Detect tier and run."""
        self._detect_tier()
        logger.info("Gmail adapter starting (tier=%s)", self._tier)
        super().run()

    def fetch_items(self) -> list[IngestItem]:
        """Fetch items based on active tier."""
        if self._tier == "mbox":
            return self._fetch_mbox()
        elif self._tier == "imap":
            return self._fetch_imap()
        elif self._tier == "oauth":
            return self._fetch_oauth()
        return []

    def wait_for_next_cycle(self) -> None:
        """Tier 1 (mbox): run once then stop. Others: poll interval."""
        if self._tier == "mbox" and self._mbox_processed:
            logger.info("MBOX import complete, adapter stopping")
            self.stop()
            return
        self._stop_event.wait(self._poll_interval)

    # -- Tier detection --

    def _detect_tier(self) -> None:
        """Auto-detect the best available tier."""
        if self._tier != "auto":
            return

        adapters_path = _adapters_config_path()
        cfg = {}
        if adapters_path.exists():
            cfg = json.loads(adapters_path.read_text()).get("gmail", {})

        # Check for explicit tier
        if cfg.get("tier") == "mbox" or cfg.get("mbox_path"):
            self._tier = "mbox"
            self._mbox_path = cfg.get("mbox_path", "")
            return

        if cfg.get("tier") == "imap":
            self._tier = "imap"
            return

        # Check for OAuth credentials
        from superlocalmemory.ingestion.credentials import has_credential
        if has_credential("gmail", "refresh_token"):
            self._tier = "oauth"
            return

        # Default: look for mbox file
        mbox_dir = _import_dir()
        mbox_files = list(mbox_dir.glob("*.mbox")) if mbox_dir.exists() else []
        if mbox_files:
            self._tier = "mbox"
            self._mbox_path = str(mbox_files[0])
            return

        logger.warning("No Gmail credentials or MBOX file found. "
                       "Place .mbox in ~/.superlocalmemory/import/ or run setup.")
        self._tier = "mbox"  # Will return empty if no file

    # -- Tier 1: MBOX file import --

    def _fetch_mbox(self) -> list[IngestItem]:
        """Parse .mbox file from Google Takeout."""
        if self._mbox_processed or not self._mbox_path:
            return []

        path = Path(self._mbox_path)
        if not path.exists():
            logger.warning("MBOX file not found: %s", path)
            self._mbox_processed = True
            return []

        import mailbox
        items = []
        mbox = mailbox.mbox(str(path))
        total = len(mbox)
        logger.info("Parsing MBOX: %d messages", total)

        for i, message in enumerate(mbox):
            if self._stop_event.is_set():
                break

            try:
                msg_id = message.get("Message-ID", f"mbox-{i}")
                subject = message.get("Subject", "(no subject)")
                from_addr = message.get("From", "unknown")
                date = message.get("Date", "")

                # Extract plain text body
                body = ""
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode("utf-8", errors="replace")
                                break
                else:
                    payload = message.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")

                # Truncate body
                body = body[:3000] if body else ""

                content = f"Email: {subject}\nFrom: {from_addr}\nDate: {date}\n\n{body}"

                items.append(IngestItem(
                    content=content,
                    dedup_key=str(msg_id).strip("<>"),
                    metadata={
                        "subject": subject,
                        "from": from_addr,
                        "date": date,
                        "source": "mbox_import",
                    },
                ))

                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info("MBOX progress: %d/%d messages", i + 1, total)

            except Exception as exc:
                logger.debug("Failed to parse message %d: %s", i, exc)

        self._mbox_processed = True
        logger.info("MBOX import: %d messages extracted", len(items))
        return items

    # -- Tier 1.5: IMAP polling --

    def _fetch_imap(self) -> list[IngestItem]:
        """Poll via IMAP. Requires email + password credentials."""
        try:
            import imaplib
            from superlocalmemory.ingestion.credentials import load_credential

            host = load_credential("gmail", "imap_host") or "imap.gmail.com"
            email = load_credential("gmail", "email")
            password = load_credential("gmail", "password")

            if not email or not password:
                logger.warning("IMAP credentials not found. Run: slm adapters enable gmail --setup")
                return []

            conn = imaplib.IMAP4_SSL(host)
            conn.login(email, password)
            conn.select("INBOX")

            # Fetch last 20 unseen messages
            _, msg_nums = conn.search(None, "UNSEEN")
            items = []

            for num in msg_nums[0].split()[-20:]:
                if self._stop_event.is_set():
                    break
                try:
                    _, data = conn.fetch(num, "(RFC822)")
                    import email as email_lib
                    msg = email_lib.message_from_bytes(data[0][1])
                    msg_id = msg.get("Message-ID", f"imap-{num.decode()}")
                    subject = msg.get("Subject", "(no subject)")
                    from_addr = msg.get("From", "unknown")

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body = payload.decode("utf-8", errors="replace")
                                    break
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="replace")

                    body = body[:3000] if body else ""
                    content = f"Email: {subject}\nFrom: {from_addr}\n\n{body}"

                    items.append(IngestItem(
                        content=content,
                        dedup_key=str(msg_id).strip("<>"),
                        metadata={"subject": subject, "from": from_addr, "source": "imap"},
                    ))
                except Exception as exc:
                    logger.debug("IMAP fetch error: %s", exc)

            conn.logout()
            return items

        except Exception as exc:
            logger.warning("IMAP polling failed: %s", exc)
            return []

    # -- Tier 2: OAuth API polling --

    def _fetch_oauth(self) -> list[IngestItem]:
        """Poll Gmail API with OAuth. Requires google-api-python-client."""
        try:
            from superlocalmemory.ingestion.credentials import load_credential

            refresh_token = load_credential("gmail", "refresh_token")
            client_id = load_credential("gmail", "client_id")
            client_secret = load_credential("gmail", "client_secret")

            if not all([refresh_token, client_id, client_secret]):
                logger.warning("Gmail OAuth credentials incomplete. Run setup.")
                return []

            # Build credentials
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                token_uri="https://oauth2.googleapis.com/token",
            )

            service = build("gmail", "v1", credentials=creds)

            # Get history since last sync
            if self._history_id:
                results = service.users().history().list(
                    userId="me",
                    startHistoryId=self._history_id,
                    historyTypes=["messageAdded"],
                ).execute()
                history = results.get("history", [])
                msg_ids = []
                for h in history:
                    for added in h.get("messagesAdded", []):
                        msg_ids.append(added["message"]["id"])
            else:
                # Initial: get last 20 messages
                results = service.users().messages().list(
                    userId="me", maxResults=20,
                ).execute()
                msg_ids = [m["id"] for m in results.get("messages", [])]

            # Update history ID for next cycle
            profile = service.users().getProfile(userId="me").execute()
            self._history_id = profile.get("historyId")

            items = []
            for msg_id in msg_ids:
                if self._stop_event.is_set():
                    break
                try:
                    msg = service.users().messages().get(
                        userId="me", id=msg_id, format="full",
                    ).execute()
                    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                    subject = headers.get("Subject", "(no subject)")
                    from_addr = headers.get("From", "unknown")
                    date = headers.get("Date", "")

                    # Extract body from payload
                    body = self._extract_gmail_body(msg.get("payload", {}))
                    body = body[:3000] if body else ""
                    content = f"Email: {subject}\nFrom: {from_addr}\nDate: {date}\n\n{body}"

                    items.append(IngestItem(
                        content=content,
                        dedup_key=msg_id,
                        metadata={"subject": subject, "from": from_addr, "date": date, "source": "oauth"},
                    ))
                except Exception as exc:
                    logger.debug("Gmail API fetch error for %s: %s", msg_id, exc)

            return items

        except ImportError:
            logger.warning("Gmail OAuth requires: pip install 'superlocalmemory[ingestion]'")
            return []
        except Exception as exc:
            logger.warning("Gmail OAuth polling failed: %s", exc)
            return []

    @staticmethod
    def _extract_gmail_body(payload: dict) -> str:
        """Extract plain text body from Gmail API payload."""
        import base64

        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            # Recurse into nested parts
            if "parts" in part:
                result = GmailAdapter._extract_gmail_body(part)
                if result:
                    return result

        return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(message)s")

    adapters_path = _adapters_config_path()
    tier = "auto"
    if adapters_path.exists():
        cfg = json.loads(adapters_path.read_text()).get("gmail", {})
        tier = cfg.get("tier", "auto")

    adapter = GmailAdapter(tier=tier)
    adapter.run()

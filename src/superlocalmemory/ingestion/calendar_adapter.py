# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Google Calendar ingestion adapter — 2 tiers.

Tier 1: ICS file import — zero setup
Tier 2: Google Calendar API with OAuth polling — shares Gmail OAuth credentials

OPT-IN only. Enabled via: slm adapters enable calendar

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.ingestion.base_adapter import BaseAdapter, AdapterConfig, IngestItem
from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger("superlocalmemory.ingestion.calendar")


def _adapters_config_path() -> Path:
    return state_path("adapters.json")


def _import_dir() -> Path:
    return state_path("import")


class CalendarAdapter(BaseAdapter):
    """Google Calendar ingestion with automatic tier detection."""

    source_type = "calendar"

    def __init__(self, config: AdapterConfig | None = None, tier: str = "auto"):
        super().__init__(config)
        self._tier = tier
        self._ics_path: str | None = None
        self._ics_processed = False
        self._sync_token: str | None = None
        self._poll_interval = 900  # 15 min

    def run(self) -> None:
        self._detect_tier()
        logger.info("Calendar adapter starting (tier=%s)", self._tier)
        super().run()

    def fetch_items(self) -> list[IngestItem]:
        if self._tier == "ics":
            return self._fetch_ics()
        elif self._tier == "oauth":
            return self._fetch_oauth()
        return []

    def wait_for_next_cycle(self) -> None:
        if self._tier == "ics" and self._ics_processed:
            logger.info("ICS import complete, adapter stopping")
            self.stop()
            return
        self._stop_event.wait(self._poll_interval)

    def _detect_tier(self) -> None:
        if self._tier != "auto":
            return

        adapters_path = _adapters_config_path()
        cfg = {}
        if adapters_path.exists():
            cfg = json.loads(adapters_path.read_text()).get("calendar", {})

        if cfg.get("tier") == "ics" or cfg.get("ics_path"):
            self._tier = "ics"
            self._ics_path = cfg.get("ics_path", "")
            return

        # Check for OAuth (shares Gmail credentials)
        from superlocalmemory.ingestion.credentials import has_credential
        if has_credential("gmail", "refresh_token"):
            self._tier = "oauth"
            return

        # Look for ICS files
        import_dir = _import_dir()
        ics_files = list(import_dir.glob("*.ics")) if import_dir.exists() else []
        if ics_files:
            self._tier = "ics"
            self._ics_path = str(ics_files[0])
            return

        self._tier = "ics"

    # -- Tier 1: ICS file import --

    def _fetch_ics(self) -> list[IngestItem]:
        if self._ics_processed or not self._ics_path:
            return []

        path = Path(self._ics_path)
        if not path.exists():
            logger.warning("ICS file not found: %s", path)
            self._ics_processed = True
            return []

        items = []

        try:
            # Try icalendar library first
            from icalendar import Calendar
            cal = Calendar.from_ical(path.read_bytes())
            events = [c for c in cal.walk() if c.name == "VEVENT"]
            logger.info("Parsing ICS: %d events", len(events))

            for event in events:
                if self._stop_event.is_set():
                    break
                try:
                    summary = str(event.get("SUMMARY", "(no title)"))
                    dtstart = event.get("DTSTART")
                    dtend = event.get("DTEND")
                    description = str(event.get("DESCRIPTION", ""))
                    location = str(event.get("LOCATION", ""))
                    uid = str(event.get("UID", ""))

                    start_str = dtstart.dt.isoformat() if dtstart else ""
                    end_str = dtend.dt.isoformat() if dtend else ""

                    # Extract attendees
                    attendees = []
                    att_list = event.get("ATTENDEE", [])
                    if not isinstance(att_list, list):
                        att_list = [att_list]
                    for att in att_list:
                        email = str(att).replace("mailto:", "").strip()
                        if email and "@" in email:
                            attendees.append(email.split("@")[0])  # Use name part

                    content = (
                        f"Calendar Event: {summary}\n"
                        f"When: {start_str} to {end_str}\n"
                    )
                    if location:
                        content += f"Where: {location}\n"
                    if attendees:
                        content += f"Attendees: {', '.join(attendees)}\n"
                    if description:
                        content += f"\n{description[:2000]}"

                    dedup_key = f"{uid}-{start_str}" if uid else f"ics-{summary}-{start_str}"

                    items.append(IngestItem(
                        content=content,
                        dedup_key=dedup_key,
                        metadata={
                            "summary": summary,
                            "start": start_str,
                            "end": end_str,
                            "attendees": attendees,
                            "source": "ics_import",
                        },
                    ))

                    # Entity propagation for attendees
                    for attendee in attendees:
                        items.append(IngestItem(
                            content=f"{attendee} attended meeting: {summary} on {start_str}",
                            dedup_key=f"attendee-{attendee}-{dedup_key}",
                            metadata={
                                "entity_name": attendee,
                                "event": summary,
                                "source": "entity_propagation",
                            },
                        ))

                except Exception as exc:
                    logger.debug("Failed to parse event: %s", exc)

        except ImportError:
            # Fallback: basic ICS parsing without icalendar library
            logger.info("icalendar not installed, using basic parser")
            items = self._parse_ics_basic(path)

        self._ics_processed = True
        logger.info("ICS import: %d items", len(items))
        return items

    def _parse_ics_basic(self, path: Path) -> list[IngestItem]:
        """Basic ICS parser without icalendar library."""
        content = path.read_text(encoding="utf-8", errors="replace")
        items = []
        events = content.split("BEGIN:VEVENT")

        for i, block in enumerate(events[1:]):  # Skip first (before any VEVENT)
            try:
                lines = block.split("\n")
                props = {}
                for line in lines:
                    if ":" in line and not line.startswith(" "):
                        key, _, val = line.partition(":")
                        key = key.split(";")[0].strip()
                        props[key] = val.strip()

                summary = props.get("SUMMARY", "(no title)")
                dtstart = props.get("DTSTART", "")
                uid = props.get("UID", f"basic-{i}")

                content_text = f"Calendar Event: {summary}\nWhen: {dtstart}"
                items.append(IngestItem(
                    content=content_text,
                    dedup_key=f"{uid}-{dtstart}",
                    metadata={"summary": summary, "start": dtstart, "source": "ics_basic"},
                ))
            except Exception:
                pass

        return items

    # -- Tier 2: OAuth API polling --

    def _fetch_oauth(self) -> list[IngestItem]:
        """Poll Google Calendar API with OAuth."""
        try:
            from superlocalmemory.ingestion.credentials import load_credential
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            refresh_token = load_credential("gmail", "refresh_token")
            client_id = load_credential("gmail", "client_id")
            client_secret = load_credential("gmail", "client_secret")

            if not all([refresh_token, client_id, client_secret]):
                logger.warning("Calendar OAuth credentials incomplete")
                return []

            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                token_uri="https://oauth2.googleapis.com/token",
            )

            service = build("calendar", "v3", credentials=creds)

            # Incremental sync
            kwargs = {"calendarId": "primary", "singleEvents": True, "maxResults": 50}
            if self._sync_token:
                kwargs["syncToken"] = self._sync_token
            else:
                # Initial sync: last 30 days
                from datetime import timedelta
                time_min = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                kwargs["timeMin"] = time_min

            try:
                results = service.events().list(**kwargs).execute()
            except Exception as api_err:
                if "410" in str(api_err):
                    # Sync token expired — full re-sync
                    logger.info("Calendar sync token expired, doing full re-sync")
                    self._sync_token = None
                    return self._fetch_oauth()
                raise

            self._sync_token = results.get("nextSyncToken")
            events = results.get("items", [])

            items = []
            for event in events:
                if self._stop_event.is_set():
                    break
                try:
                    summary = event.get("summary", "(no title)")
                    start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
                    end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date", ""))
                    description = event.get("description", "")
                    location = event.get("location", "")
                    event_id = event.get("id", "")
                    updated = event.get("updated", "")

                    attendees = []
                    for att in event.get("attendees", []):
                        name = att.get("displayName") or att.get("email", "").split("@")[0]
                        if name:
                            attendees.append(name)

                    content = f"Calendar Event: {summary}\nWhen: {start} to {end}\n"
                    if location:
                        content += f"Where: {location}\n"
                    if attendees:
                        content += f"Attendees: {', '.join(attendees)}\n"
                    if description:
                        content += f"\n{description[:2000]}"

                    dedup_key = f"{event_id}-{updated}"

                    items.append(IngestItem(
                        content=content,
                        dedup_key=dedup_key,
                        metadata={
                            "summary": summary, "start": start,
                            "attendees": attendees, "source": "oauth",
                        },
                    ))

                    # Entity propagation
                    for attendee in attendees:
                        items.append(IngestItem(
                            content=f"{attendee} attended: {summary} on {start}",
                            dedup_key=f"cal-attendee-{attendee}-{event_id}",
                            metadata={"entity_name": attendee, "event": summary, "source": "entity_propagation"},
                        ))

                except Exception as exc:
                    logger.debug("Calendar event parse error: %s", exc)

            return items

        except ImportError:
            logger.warning("Calendar OAuth requires: pip install 'superlocalmemory[ingestion]'")
            return []
        except Exception as exc:
            logger.warning("Calendar OAuth failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(message)s")

    adapters_path = _adapters_config_path()
    tier = "auto"
    if adapters_path.exists():
        cfg = json.loads(adapters_path.read_text()).get("calendar", {})
        tier = cfg.get("tier", "auto")

    adapter = CalendarAdapter(tier=tier)
    adapter.run()

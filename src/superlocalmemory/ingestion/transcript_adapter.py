# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Transcript ingestion adapter — watches for .srt/.vtt/.txt files.

Uses watchdog (cross-platform file watcher) to detect new transcript files.
Parses them, extracts speaker diarization, propagates entities, and POSTs
to the daemon's /ingest endpoint.

OPT-IN only. Enabled via: slm adapters enable transcript

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from superlocalmemory.ingestion.base_adapter import BaseAdapter, AdapterConfig, IngestItem
from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger("superlocalmemory.ingestion.transcript")

_WATCH_EXTENSIONS = {".srt", ".vtt", ".txt"}


def _adapters_config_path() -> Path:
    return state_path("adapters.json")


class TranscriptAdapter(BaseAdapter):
    """Watches a directory for transcript files and ingests them."""

    source_type = "transcript"

    def __init__(self, watch_dir: str | Path, config: AdapterConfig | None = None):
        super().__init__(config)
        self._watch_dir = Path(watch_dir)
        self._pending_files: list[Path] = []
        self._observer = None

    def run(self) -> None:
        """Start file watcher then enter the base adapter loop."""
        if not self._watch_dir.exists():
            logger.error("Watch directory does not exist: %s", self._watch_dir)
            return

        # Start watchdog observer
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _Handler(FileSystemEventHandler):
                def __init__(self, adapter: TranscriptAdapter):
                    self._adapter = adapter

                def on_created(self, event):
                    if event.is_directory:
                        return
                    path = Path(event.src_path)
                    if path.suffix.lower() in _WATCH_EXTENSIONS:
                        self._adapter._pending_files.append(path)

            self._observer = Observer()
            self._observer.schedule(_Handler(self), str(self._watch_dir), recursive=False)
            self._observer.start()
            logger.info("Watching for transcripts in: %s", self._watch_dir)
        except ImportError:
            logger.warning("watchdog not installed — polling mode only")

        # Also scan for existing files on first run
        for path in self._watch_dir.iterdir():
            if path.suffix.lower() in _WATCH_EXTENSIONS and path.is_file():
                self._pending_files.append(path)

        super().run()

        # Cleanup
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def fetch_items(self) -> list[IngestItem]:
        """Return pending transcript files as IngestItems."""
        if not self._pending_files:
            return []

        items = []
        batch = list(self._pending_files)
        self._pending_files.clear()

        for filepath in batch:
            try:
                from superlocalmemory.ingestion.parsers import (
                    parse_transcript_file, content_hash,
                )
                combined_text, speakers = parse_transcript_file(filepath)
                dedup = content_hash(filepath)

                # Main transcript ingestion
                items.append(IngestItem(
                    content=f"Meeting transcript ({filepath.name}):\n{combined_text}",
                    dedup_key=dedup,
                    metadata={
                        "filename": filepath.name,
                        "speakers": speakers,
                        "source": "file_watcher",
                    },
                ))

                # Entity propagation: each speaker gets a timeline entry
                for speaker in speakers:
                    items.append(IngestItem(
                        content=f"{speaker} participated in meeting: {filepath.stem}. "
                                f"Transcript file: {filepath.name}",
                        dedup_key=f"speaker-{speaker}-{dedup}",
                        metadata={
                            "entity_name": speaker,
                            "meeting_file": filepath.name,
                            "source": "entity_propagation",
                        },
                    ))

            except Exception as exc:
                logger.warning("Failed to parse %s: %s", filepath, exc)

        return items

    def wait_for_next_cycle(self) -> None:
        """Wait 30s for new files (watchdog handles detection)."""
        self._stop_event.wait(30)


# ---------------------------------------------------------------------------
# CLI entry point: python -m superlocalmemory.ingestion.transcript_adapter
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(message)s")

    # Load config
    watch_dir = ""
    adapters_path = _adapters_config_path()
    if adapters_path.exists():
        cfg = json.loads(adapters_path.read_text())
        watch_dir = cfg.get("transcript", {}).get("watch_dir", "")

    if not watch_dir:
        print("No watch_dir configured. Set it in ~/.superlocalmemory/adapters.json")
        print('  {"transcript": {"enabled": true, "watch_dir": "/path/to/transcripts"}}')
        sys.exit(1)

    adapter = TranscriptAdapter(watch_dir=watch_dir)
    adapter.run()

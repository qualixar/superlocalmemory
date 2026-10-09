# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Run one installed SLM version against a throwaway data directory.

Every call pins ``SLM_DATA_DIR`` and ``HOME`` to scratch paths, so nothing here
can touch a real ``~/.superlocalmemory``.  The daemon is started and stopped
through the version's own ``slm serve`` command.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import corpus


#: Old versions ignore SLM_DATA_DIR and read ``$HOME/.superlocalmemory``; the data
#: directory is always placed there so every version agrees on where it is.
DATA_SUBDIR = ".superlocalmemory"


def free_port(start: int = 8811) -> int:
    """First TCP port at or above ``start`` that nothing is listening on."""
    for port in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


@dataclass
class Instance:
    """One installed version, one data directory, one throwaway HOME."""

    bin_dir: Path
    data_dir: Path
    home: Path
    port: int = field(default_factory=free_port)

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            SLM_DATA_DIR=str(self.data_dir), HOME=str(self.home),
            SLM_DAEMON_PORT=str(self.port), HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
        )
        return env

    def run(self, *args: str, timeout: int = 300) -> tuple[int, str, str, float]:
        """Run ``slm <args>``; returns (returncode, stdout, stderr, seconds)."""
        started = time.monotonic()
        try:
            p = subprocess.run(
                [str(self.bin_dir / "slm"), *args], env=self.env(), cwd=self.home,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return 124, "", f"timeout after {timeout}s: {exc}", time.monotonic() - started
        return p.returncode, p.stdout, p.stderr, time.monotonic() - started

    def live_port(self) -> int:
        """Port the daemon reports in ``daemon.port`` (old versions ignore SLM_DAEMON_PORT)."""
        try:
            return int((self.data_dir / "daemon.port").read_text().strip())
        except (OSError, ValueError):
            return self.port

    def health(self) -> dict | None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.live_port()}/health", timeout=3) as r:
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001 - not up yet
            return None

    def serve_start(self, wait: int = 180) -> tuple[bool, float, str]:
        """Start the daemon and wait for /health.  Returns (up, seconds, note)."""
        started = time.monotonic()
        rc, out, err, _ = self.run("serve", "start", timeout=wait)
        deadline = started + wait
        while time.monotonic() < deadline:
            if self.health() is not None:
                return True, time.monotonic() - started, ""
            time.sleep(1)
        return False, time.monotonic() - started, (err or out)[-400:]

    def serve_stop(self) -> None:
        self.run("serve", "stop", timeout=60)
        for _ in range(30):
            if self.health() is None:
                break
            time.sleep(1)

    def recall_ids(self, query: str, limit: int = 5) -> tuple[list[str], str, str]:
        """Top corpus ids for ``query``, the channel-status string and any error."""
        rc, out, err, _ = self.run("recall", query, "--json", "--limit", str(limit), timeout=120)
        try:
            data = json.loads(out[out.index("{"):])
        except (ValueError, json.JSONDecodeError):
            return [], "", f"rc={rc} unparsable: {(err or out)[-300:]}"
        body = data.get("data", data)
        rows = body.get("results", []) or []
        ids = [r for r in (corpus.ref_of(x.get("content", "")) for x in rows) if r]
        return ids, json.dumps(body.get("channel_status", {}), sort_keys=True), ""


def make_work_dir(prefix: str = "slm-upg-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("SLM_UPG_TMP")))


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _open_copy(db: Path) -> tuple[sqlite3.Connection, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="slm-upg-ro-"))
    for suffix in ("", "-wal", "-shm"):
        if Path(str(db) + suffix).exists():
            shutil.copy2(str(db) + suffix, tmp / (db.name + suffix))
    return sqlite3.connect(tmp / db.name), tmp


def table_counts(db: Path) -> dict[str, int]:
    """Row count of every user table in ``db`` (read from a private copy)."""
    if not db.exists():
        return {}
    conn, tmp = _open_copy(db)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        out: dict[str, int] = {}
        for name in names:
            try:
                out[name] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            except sqlite3.DatabaseError:
                out[name] = -1  # virtual table whose module is not loaded
        return out
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def contents_by_ref(db: Path) -> dict[str, list[str]]:
    """Map corpus id -> sorted distinct text values that carry it, any table/column."""
    found: dict[str, set[str]] = {}
    if not db.exists():
        return {}
    conn, tmp = _open_copy(db)
    try:
        for name in table_counts_names(conn):
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')]
            for col in cols:
                try:
                    cur = conn.execute(
                        f'SELECT DISTINCT "{col}" FROM "{name}" WHERE typeof("{col}")=\'text\' '
                        f'AND "{col}" LIKE \'%QX-%\'')
                    for (val,) in cur:
                        ref = corpus.ref_of(val)
                        if ref and len(val) < 4000:
                            found.setdefault(ref, set()).add(val)
                except sqlite3.DatabaseError:
                    continue
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return {k: sorted(v) for k, v in sorted(found.items())}


def table_counts_names(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE '%_fts%' AND sql NOT LIKE 'CREATE VIRTUAL%'")]


def schema_version(learning_db: Path) -> int | None:
    """Value in ``slm_schema_version`` (None when the table or file is absent)."""
    if not learning_db.exists():
        return None
    conn, tmp = _open_copy(learning_db)
    try:
        row = conn.execute("SELECT version FROM slm_schema_version WHERE id = 1").fetchone()
        return int(row[0]) if row else None
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)

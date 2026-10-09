# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Run one installed SLM version against a throwaway data directory.

Every call pins ``SLM_DATA_DIR`` and ``HOME`` to scratch paths, so nothing here
can touch a real ``~/.superlocalmemory``.  The daemon is started and stopped
through the version's own ``slm serve`` command.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import signal
import socket
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import corpus
from _verdicts import parse_recall


#: Old versions ignore SLM_DATA_DIR and read ``$HOME/.superlocalmemory``; the data
#: directory is always placed there so every version agrees on where it is.
DATA_SUBDIR = ".superlocalmemory"


#: Ports the product's own daemon uses; a fixture run must never start or probe there.
PROTECTED_PORTS = frozenset({8765, 8767})
#: Range the scratch daemons are placed in.
PORT_LO, PORT_HI = 8840, 8899


def port_listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(lo: int = PORT_LO, hi: int = PORT_HI) -> int:
    """A random port in ``lo..hi`` that nothing listens on (never a protected one)."""
    ports = [p for p in range(lo, hi + 1) if p not in PROTECTED_PORTS]
    random.shuffle(ports)
    for port in ports:
        if not port_listening(port):
            return port
    raise RuntimeError(f"no free port in {lo}-{hi}")


def is_legacy(version: str) -> bool:
    """Versions before 4.0: ``slm serve stop`` kills every SLM daemon on the machine."""
    head = version.split(".")[0]
    return head.isdigit() and int(head) < 4


def package_version(python: Path, package: str = "superlocalmemory") -> str:
    code = f"import importlib.metadata as m; print(m.version({package!r}))"
    p = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True, timeout=60)
    return p.stdout.strip() if p.returncode == 0 else ""


def require_version(python: Path, expected: str, package: str = "superlocalmemory") -> None:
    """Raise unless the interpreter's ``package`` is exactly ``expected``."""
    have = package_version(python, package)
    if have != expected:
        raise RuntimeError(f"{python}: expected {package} {expected}, found {have or 'none'}")


def _proc_text(pid: int, name: str) -> bytes:
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except OSError:
        return b""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return not _proc_text(pid, "stat").split(b") ")[-1].startswith(b"Z")


@dataclass
class Instance:
    """One installed version, one data directory, one throwaway HOME."""

    bin_dir: Path
    data_dir: Path
    home: Path
    port: int = field(default_factory=free_port)
    version: str = ""
    stop_wait: int = 30  # seconds to wait for a graceful `serve stop`

    def __post_init__(self) -> None:
        if self.port in PROTECTED_PORTS:
            raise ValueError(f"port {self.port} belongs to the product daemon; refusing")

    @property
    def legacy(self) -> bool:
        return is_legacy(self.version)

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLM_TEST_")}
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

    def scratch_pid(self) -> int | None:
        """Pid recorded in this instance's own ``daemon.pid`` (None when absent or unreadable)."""
        try:
            pid = int((self.data_dir / "daemon.pid").read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if pid > 1 else None

    def owns(self, pid: int) -> bool:
        """True only when ``pid`` is an SLM process started with this instance's HOME or data dir."""
        cmd = _proc_text(pid, "cmdline").replace(b"\0", b" ")
        if b"superlocalmemory" not in cmd:
            return False
        want = {f"HOME={self.home}".encode(), f"SLM_DATA_DIR={self.data_dir}".encode()}
        return bool(want & set(_proc_text(pid, "environ").split(b"\0")))

    def scratch_alive(self) -> bool:
        pid = self.scratch_pid()
        return pid is not None and _pid_alive(pid) and self.owns(pid)

    def stop_scratch(self, grace: int = 20) -> bool:
        """Signal only the scratch daemon; True once none of ours is running."""
        pid = self.scratch_pid()
        if pid is None or not _pid_alive(pid) or not self.owns(pid):
            return True
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
            try:
                os.kill(pid, sig)
            except OSError:
                break
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and _pid_alive(pid):
                time.sleep(0.2)
            if not _pid_alive(pid):
                break
        return not _pid_alive(pid)

    def live_port(self) -> int:
        """Port the daemon reports in ``daemon.port`` (old versions ignore SLM_DAEMON_PORT)."""
        try:
            return int((self.data_dir / "daemon.port").read_text().strip())
        except (OSError, ValueError):
            return self.port

    def health(self) -> dict | None:
        """GET /health, trusted only while the scratch daemon's own pid is alive."""
        if not self.scratch_alive():
            return None
        url = f"http://127.0.0.1:{self.live_port()}/health"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return json.loads(resp.read().decode())
        except (OSError, ValueError):
            return None

    def _refuse_if_default_port_taken(self) -> None:
        if self.legacy:
            busy = sorted(p for p in PROTECTED_PORTS if port_listening(p))
            if busy:
                raise RuntimeError(f"port(s) {busy} already listening; an old version would collide with it")

    def serve_start(self, wait: int = 180) -> tuple[bool, float, str]:
        """Start the daemon and wait for /health.  Returns (up, seconds, note)."""
        self._refuse_if_default_port_taken()
        started = time.monotonic()
        rc, out, err, _ = self.run("serve", "start", timeout=wait)
        deadline = started + wait
        while time.monotonic() < deadline:
            if self.health() is not None:
                return True, time.monotonic() - started, ""
            time.sleep(1)
        return False, time.monotonic() - started, (err or out)[-400:]

    def serve_stop(self) -> bool:
        """Stop the daemon; True once it has verifiably exited.

        Versions before 4.0 are stopped by pid only (their ``slm serve stop`` is machine-wide).
        """
        if not self.legacy:
            self.run("serve", "stop", timeout=60)
            for _ in range(self.stop_wait):
                if not self.scratch_alive():
                    break
                time.sleep(1)
        return self.stop_scratch()

    @contextlib.contextmanager
    def session(self, wait: int = 180):
        """Start the daemon, yield ``(up, seconds, note)``, and always stop it again."""
        try:
            yield self.serve_start(wait)
        finally:
            if not self.serve_stop():
                raise RuntimeError(f"daemon (pid {self.scratch_pid()}) is still running after stop")

    def recall_ids(self, query: str, limit: int = 5) -> tuple[list[str], str, str]:
        """Top corpus ids for ``query``, the channel-status string and any error."""
        rc, out, err, _ = self.run("recall", query, "--json", "--limit", str(limit), timeout=120)
        rows, status, error = parse_recall(out)
        if error:
            return [], "", f"rc={rc} {error}: {(err or out)[-300:]}"
        ids = [r for r in (corpus.ref_of(x.get("content", "")) for x in rows) if r]
        return ids, status, ""


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


def table_columns(db: Path) -> dict[str, list[str]]:
    """Every user table in ``db`` mapped to its column names, in order."""
    if not db.exists():
        return {}
    conn, tmp = _open_copy(db)
    try:
        out = {}
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            out[name] = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')]
        return out
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def schema_signature(db: Path) -> list[tuple]:
    """Sorted (type, name, sql) of every schema object: changes whenever the schema does."""
    if not db.exists():
        return []
    conn, tmp = _open_copy(db)
    try:
        return sorted(conn.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall(),
                      key=lambda r: (r[0], r[1]))
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def base_table_counts(db: Path) -> dict[str, int]:
    """Row counts of the real tables of ``db`` (full-text shadow tables left out)."""
    return {k: v for k, v in table_counts(db).items() if "_fts" not in k}

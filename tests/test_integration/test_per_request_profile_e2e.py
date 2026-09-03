# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""End-to-end acceptance for per-request profile routing over a REAL daemon.

Requirement-doc acceptance scenarios exercised here (spec section 6):

1. Two clients (doris/zhihui) interleave remember/recall against a real
   unified-daemon subprocess — each hits its own namespace, cross-invisible.
2. ``/status`` keeps reporting the same ``profile`` and
   ``profile_generation`` throughout — per-request routing never moves the
   global active-pointer.
3. Two threads, each pinned to a different ``profile_id``, concurrently
   write N=50 memories: the database count grouped by profile is exactly
   50/50 — zero crossover, zero orphans (total rows == 100).
6. The MCP stdio lane: an ``mslm mcp`` child restricted to
   ``SLM_MCP_TOOLS=remember,recall`` carries ``profile_id`` over a full
   JSON-RPC round trip against the same daemon.

Orchestration follows the established two-process pattern of
``tests/test_integration/test_embedding_fallback_two_process.py``: a REAL
daemon subprocess on a kernel-assigned ephemeral port with an isolated
``SLM_DATA_DIR``. The production daemon on 8765 is never touched — the
ephemeral port is reserved outside the public set, all child environments
are constructed (never inherited), and teardown proves machine state was
restored: the daemon process group is gone, its lifecycle files are
removed, the port is bindable again, and every foreign daemon PID that was
alive before the suite is still alive after it.

The doris/zhihui profiles are pre-created as ``profiles`` table rows via
the same raw-SQL mechanism the server tests use
(``tests/test_server/test_per_request_profile.py``): routing must never
implicitly create a profile.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Ports owned by public/production daemons on this machine. Never bind, never
# connect — enforced by reserving outside this set (and the root conftest's
# audit hook denies them in-process as a second belt).
PRODUCTION_PORTS = {8765, 8767}
PROFILES = ("doris", "zhihui")
N_CONCURRENT_WRITES = 50

# Unique-per-run tokens keep every marker lexical hit attributable to THIS
# run even though one daemon serves the whole module.
RUN_TAG = uuid.uuid4().hex[:8]

_PROXY_VARS = (
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY",
)
_PASSTHROUGH_VARS = ("PATH", "LANG", "LC_ALL", "TMPDIR", "TERM")


def _reserve_private_port() -> int:
    """Ask the kernel for a loopback port, never the public daemon ports."""
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
        if port not in PRODUCTION_PORTS:
            return port
    raise AssertionError("could not reserve an isolated daemon port")


def _child_env(data_root: Path, port: int, home: Path, cache_root: Path) -> dict:
    """A constructed (never inherited) environment for SLM child processes.

    Everything identity-bearing is pinned inside the fixture-owned root:
    SLM_DATA_DIR (databases, locks, descriptor), HOME, and every model cache
    (offline so a cold cache can never trigger a network fetch). Proxy
    variables are stripped so loopback HTTP cannot be middle-boxed.
    """
    env = {name: os.environ[name] for name in _PASSTHROUGH_VARS if name in os.environ}
    env.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(SRC_ROOT),
            "SLM_DATA_DIR": str(data_root),
            "SLM_DAEMON_PORT": str(port),
            "OMP_NUM_THREADS": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(cache_root / "huggingface"),
            "SENTENCE_TRANSFORMERS_HOME": str(cache_root / "sentence-transformers"),
            "XDG_CACHE_HOME": str(cache_root),
            "CI": "1",
            "SLM_NON_INTERACTIVE": "1",
            "SLM_TEST_ISOLATION": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for var in _PROXY_VARS:
        env.pop(var, None)
    return env


def _foreign_daemon_pids() -> set[int]:
    """PIDs of unified daemons that do not belong to this test (production)."""
    try:
        import psutil
    except Exception:  # pragma: no cover — psutil is a test dependency
        return set()
    mine = os.getpid()
    pids: set[int] = set()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if (
            proc.info["pid"] != mine
            and "superlocalmemory.server.unified_daemon" in cmdline
        ):
            pids.add(proc.info["pid"])
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class _RpcClient:
    """Newline-delimited JSON-RPC driver for one ``mslm mcp`` stdio child."""

    def __init__(self, proc: subprocess.Popen, stderr_path: Path) -> None:
        self._proc = proc
        self._stderr_path = stderr_path
        self._responses: dict[object, dict] = {}
        self._next_id = 0
        self._reader = threading.Thread(target=self._read_forever, daemon=True)
        self._reader.start()

    def _read_forever(self) -> None:
        try:
            for raw in self._proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if "id" in message:
                    self._responses[message["id"]] = message
        except Exception:  # reader must outlive the child silently
            pass

    def _stderr_tail(self) -> str:
        try:
            return self._stderr_path.read_text(
                encoding="utf-8", errors="replace",
            )[-1500:]
        except OSError:
            return "(stderr log unavailable)"

    def call(self, method: str, params: dict | None = None, *, timeout: float = 120.0) -> dict:
        self._next_id += 1
        request_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if request_id in self._responses:
                return self._responses[request_id]
            if self._proc.poll() is not None:
                raise AssertionError(
                    f"mcp child exited rc={self._proc.returncode} during {method}; "
                    f"stderr tail:\n{self._stderr_tail()}"
                )
            time.sleep(0.05)
        raise AssertionError(
            f"timed out waiting for {method} response; stderr tail:\n"
            f"{self._stderr_tail()}"
        )

    def notify(self, method: str) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def close(self) -> None:
        """Close stdin (FastMCP exits on EOF), then escalate if needed."""
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)


class RealDaemon:
    """One real unified-daemon subprocess in a fixture-owned namespace."""

    def __init__(self, proc: subprocess.Popen, port: int, data_root: Path,
                 stdout_log: Path, env: dict) -> None:
        self.proc = proc
        self.port = port
        self.data_root = data_root
        self.stdout_log = stdout_log
        self.env = env

    # -- identity ---------------------------------------------------------

    @property
    def descriptor_path(self) -> Path:
        return self.data_root / "daemon.json"

    def descriptor(self) -> dict:
        return json.loads(self.descriptor_path.read_text(encoding="utf-8"))

    # -- HTTP -------------------------------------------------------------

    def request(self, method: str, path: str, body: dict | None = None,
                params: dict | None = None, timeout: float = 90.0) -> tuple[int, dict]:
        """Authenticated loopback request using the daemon's own capability.

        Mirrors ``superlocalmemory.cli.daemon.daemon_request``: the
        descriptor in OUR data root carries the capability/instance headers
        the write routes require.
        """
        descriptor = self.descriptor()
        url = f"http://127.0.0.1:{self.port}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        headers["X-SLM-Daemon-Capability"] = descriptor["capability"]
        headers["X-SLM-Target-Instance"] = descriptor["instance_id"]
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, json.loads(payload)
            except ValueError:
                return exc.code, {"raw": payload}

    def status(self) -> dict:
        code, payload = self.request("GET", "/status")
        assert code == 200, payload
        return payload

    def remember(self, content: str, profile_id: str, idempotency_key: str) -> dict:
        body = {"content": content, "idempotency_key": idempotency_key}
        if profile_id:
            body["profile_id"] = profile_id
        code, payload = self.request("POST", "/remember", body)
        assert code == 200, payload
        assert payload.get("ok") is True, payload
        return payload

    def recall(self, query: str, profile_id: str) -> dict:
        code, payload = self.request(
            "GET", "/recall", params={"q": query, "profile_id": profile_id},
        )
        assert code == 200, payload
        return payload

    # -- lifecycle --------------------------------------------------------

    def _log_tail(self) -> str:
        chunks = []
        for path in (self.stdout_log, self.data_root / "logs" / "daemon.log"):
            try:
                chunks.append(
                    f"--- {path} ---\n"
                    + path.read_text(encoding="utf-8", errors="replace")[-2500:]
                )
            except OSError:
                continue
        return "\n".join(chunks) or "(no daemon logs available)"

    def wait_ready(self, timeout: float = 300.0) -> None:
        """Wait until /status answers 200 (engine serving requests).

        /health is deliberately NOT the readiness bar here: it embeds a
        channel-health probe that can trail engine readiness by minutes on a
        cold, offline-model sandbox. /status is the daemon's own
        non-blocking "serving" answer.
        """
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"daemon exited rc={self.proc.returncode} during startup;\n"
                    f"{self._log_tail()}"
                )
            try:
                code, _ = self.request("GET", "/status", timeout=5)
                if code == 200:
                    return
            except Exception as exc:  # not listening yet / descriptor missing
                last_error = exc
            time.sleep(0.5)
        raise AssertionError(
            f"daemon not ready within {timeout}s (last error: {last_error!r});\n"
            f"{self._log_tail()}"
        )

    def wait_health_fast(self, timeout: float = 300.0) -> None:
        """Wait until /health answers 200 within daemon_request's 2s bar.

        The MCP tool lane preflights /health with a 2-second timeout before
        every daemon call; only a fast answer proves the child will route
        through this daemon instead of reporting DAEMON_UNAVAILABLE.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=2,
                ) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(2.0)
        raise AssertionError(
            "/health never answered within the 2s daemon_request preflight "
            f"bar after {timeout}s;\n" + self._log_tail()
        )

    def precreate_profiles(self, profiles: tuple[str, ...]) -> None:
        """Insert profiles table rows the way the server tests do.

        A routed write must find its profile already present (routing never
        implicitly creates one), so doris/zhihui are seeded by hand before
        any client talks to the daemon. WAL + busy_timeout lets this short
        write land while the daemon holds the database.
        """
        conn = sqlite3.connect(self.data_root / "memory.db", timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            for profile_id in profiles:
                conn.execute(
                    "INSERT OR IGNORE INTO profiles (profile_id, name) "
                    "VALUES (?, ?)",
                    (profile_id, f"E2E Profile {profile_id}"),
                )
            conn.commit()
        finally:
            conn.close()
        # Prove the daemon sees the rows: a routed recall of a seeded
        # profile must be a normal 200, not the unknown_profile 404.
        for profile_id in profiles:
            code, payload = self.request(
                "GET", "/recall",
                params={"q": "seeded-profile-probe", "profile_id": profile_id},
            )
            assert code == 200, payload

    def fact_rows(self, where: str, args: tuple) -> list[tuple]:
        conn = sqlite3.connect(self.data_root / "memory.db", timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            return conn.execute(
                f"SELECT profile_id, COUNT(*) FROM atomic_facts "
                f"WHERE {where} GROUP BY profile_id",
                args,
            ).fetchall()
        finally:
            conn.close()

    def _group_members(self) -> list[str]:
        """Live processes still in the daemon's process group."""
        listing = subprocess.run(
            ["ps", "-eo", "pid,pgid,args"],
            capture_output=True, text=True, timeout=30,
        ).stdout.splitlines()
        return [
            line for line in listing
            if line.split() and line.split()[1] == str(self.proc.pid)
        ]

    def stop(self, foreign_before: set[int]) -> None:
        """Stop the daemon and PROVE machine state was restored."""
        # 1. Graceful stop via the daemon's own capability-bound route.
        try:
            self.request("POST", "/stop", body={}, timeout=10)
        except Exception:
            pass  # escalate below
        graceful = True
        try:
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            graceful = False
            if os.name == "posix":
                os.killpg(self.proc.pid, signal.SIGTERM)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(self.proc.pid, signal.SIGKILL)
                else:
                    self.proc.kill()
                self.proc.wait(timeout=20)
        # 2. No member of the daemon's process group survives (workers
        #    included — they were spawned into the same session). Workers
        #    self-terminate on a ~10s parent-watchdog poll after the daemon
        #    exits, so grant that grace, then force, then assert.
        if os.name == "posix":
            deadline = time.monotonic() + 30
            leaked = self._group_members()
            while leaked and time.monotonic() < deadline:
                time.sleep(1.0)
                leaked = self._group_members()
            if leaked:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                time.sleep(2.0)
                leaked = self._group_members()
            assert leaked == [], (
                f"daemon process group {self.proc.pid} leaked members: {leaked}"
            )

        # 3. Graceful stop removes exactly the ephemeral lifecycle identity.
        if graceful:
            for name in ("daemon.json", "daemon.pid", "daemon.port"):
                assert not (self.data_root / name).exists(), (
                    f"stale lifecycle state survived stop: {name}"
                )

        # 4. The ephemeral port is bindable again.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.port))

        # 5. The production daemon was never touched: every foreign daemon
        #    PID observed before this suite is still alive.
        still_alive = {pid for pid in foreign_before if _alive(pid)}
        assert still_alive == foreign_before, (
            f"foreign daemons changed during the suite: "
            f"before={sorted(foreign_before)} after={sorted(still_alive)}"
        )


@pytest.fixture(scope="module")
def real_daemon(tmp_path_factory):
    """The REAL daemon subprocess shared by every acceptance scenario below."""
    root = tmp_path_factory.mktemp("prp-e2e")
    data_root = root / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    env = _child_env(data_root, port, root / "home", root / "cache")

    foreign_before = _foreign_daemon_pids()

    stdout_log = root / "daemon-stdout.log"
    with stdout_log.open("wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
             "--start", f"--port={port}"],
            stdout=log_file, stderr=log_file, env=env, cwd=str(REPO_ROOT),
            start_new_session=os.name == "posix",
        )
    daemon = RealDaemon(proc, port, data_root, stdout_log, env)
    try:
        daemon.wait_ready()
        daemon.precreate_profiles(PROFILES)
        yield daemon
    finally:
        daemon.stop(foreign_before)


def _contents(payload: dict) -> list[str]:
    return [str(item.get("content", "")) for item in payload.get("results", [])]


class TestAcceptance1InterleavedTwoClients:
    """Scenario 1: two clients, interleaved, each hits only its namespace."""

    def test_interleaved_remember_recall_is_cross_invisible(self, real_daemon):
        doris_token = f"PrpE2e{RUN_TAG}DorisReef"
        zhihui_token = f"PrpE2e{RUN_TAG}ZhihuiLark"
        status_snapshots = [real_daemon.status()]

        # Interleaved writes: doris, zhihui, doris, zhihui — the ordering
        # that silently re-routed clients under the global pointer before.
        real_daemon.remember(
            f"{doris_token} maintains the harbor pilot rota and files the "
            "tide-window ledger for the northern approach.",
            profile_id="doris", idempotency_key=f"{RUN_TAG}-inter-d1",
        )
        status_snapshots.append(real_daemon.status())
        real_daemon.remember(
            f"{zhihui_token} tunes the lantern festival drones and keeps the "
            "flight permits for the river parade.",
            profile_id="zhihui", idempotency_key=f"{RUN_TAG}-inter-z1",
        )
        status_snapshots.append(real_daemon.status())
        real_daemon.remember(
            f"{doris_token} also audits the breaklight buoys every spring tide.",
            profile_id="doris", idempotency_key=f"{RUN_TAG}-inter-d2",
        )
        status_snapshots.append(real_daemon.status())
        real_daemon.remember(
            f"{zhihui_token} rehearses the drone swarm over the old mint.",
            profile_id="zhihui", idempotency_key=f"{RUN_TAG}-inter-z2",
        )
        status_snapshots.append(real_daemon.status())

        # Each client's recall hits its own namespace.
        doris_hits = _contents(real_daemon.recall(
            f"{doris_token} harbor pilot rota", profile_id="doris",
        ))
        assert doris_hits, "doris recall must hit the doris namespace"
        assert any(doris_token in content for content in doris_hits)
        assert not any(zhihui_token in content for content in doris_hits), (
            "zhihui memory leaked into the doris namespace"
        )

        zhihui_hits = _contents(real_daemon.recall(
            f"{zhihui_token} drone flight permits", profile_id="zhihui",
        ))
        assert zhihui_hits, "zhihui recall must hit the zhihui namespace"
        assert any(zhihui_token in content for content in zhihui_hits)
        assert not any(doris_token in content for content in zhihui_hits), (
            "doris memory leaked into the zhihui namespace"
        )

        # Cross-visibility: the other client's unique token is invisible.
        # (Token absence, not an empty result list: by the time later
        # scenarios share this daemon, a zhihui-routed recall may legitimately
        # surface zhihui's own facts — a doris token there is the leak.)
        doris_via_zhihui = _contents(real_daemon.recall(
            f"{doris_token} harbor pilot rota", profile_id="zhihui",
        ))
        assert not any(doris_token in c for c in doris_via_zhihui), (
            "doris facts must be invisible to a zhihui-routed recall"
        )
        zhihui_via_doris = _contents(real_daemon.recall(
            f"{zhihui_token} drone flight permits", profile_id="doris",
        ))
        assert not any(zhihui_token in c for c in zhihui_via_doris), (
            "zhihui facts must be invisible to a doris-routed recall"
        )

        # The routed writes never landed anywhere else.
        rows = real_daemon.fact_rows(
            "content LIKE ? OR content LIKE ?",
            (f"%{doris_token}%", f"%{zhihui_token}%"),
        )
        assert dict(rows) == {"doris": 2, "zhihui": 2}, rows


class TestAcceptance2GlobalPointerFrozen:
    """Scenario 2: profile + profile_generation never move while routing."""

    def test_status_pointer_and_generation_unchanged_throughout(self, real_daemon):
        before = real_daemon.status()
        active = before["profile"]
        assert active not in PROFILES, (
            "the fresh daemon's active profile must differ from the routed "
            "ones, or the freeze assertions below would pass vacuously"
        )

        snapshots = [before]
        real_daemon.remember(
            f"PrpE2e{RUN_TAG}QuartzDune buffers the quarterly readiness "
            "review for the on-call rotation.",
            profile_id="doris", idempotency_key=f"{RUN_TAG}-freeze-d1",
        )
        snapshots.append(real_daemon.status())
        real_daemon.remember(
            f"PrpE2e{RUN_TAG}QuartzDune archives the ferry manifest archive.",
            profile_id="zhihui", idempotency_key=f"{RUN_TAG}-freeze-z1",
        )
        snapshots.append(real_daemon.status())
        assert real_daemon.recall(
            f"PrpE2e{RUN_TAG}QuartzDune readiness review", profile_id="doris",
        )["results"], "routed recall must succeed for the freeze scenario"
        snapshots.append(real_daemon.status())
        assert real_daemon.recall(
            f"PrpE2e{RUN_TAG}QuartzDune ferry manifest", profile_id="zhihui",
        )["results"], "routed recall must succeed for the freeze scenario"
        snapshots.append(real_daemon.status())

        for index, snapshot in enumerate(snapshots):
            assert snapshot["profile"] == active, (
                f"active pointer moved at snapshot {index}: "
                f"{active!r} -> {snapshot['profile']!r}"
            )
            assert snapshot["profile_generation"] == before["profile_generation"], (
                f"profile_generation moved at snapshot {index}: "
                f"{before['profile_generation']!r} -> "
                f"{snapshot['profile_generation']!r}"
            )


class TestAcceptance3ConcurrentWriters:
    """Scenario 3: two threads, two profiles, N=50 each — count, no crossover."""

    def test_concurrent_writes_group_count_50_50_zero_crossover(self, real_daemon):
        doris_marker = f"PrpE2e{RUN_TAG}ConcDoris"
        zhihui_marker = f"PrpE2e{RUN_TAG}ConcZhihui"
        before = real_daemon.status()
        failures: list[str] = []
        barrier = threading.Barrier(2)

        def writer(profile_id: str, marker: str) -> None:
            barrier.wait()  # maximize true interleaving of the two writers
            for index in range(N_CONCURRENT_WRITES):
                try:
                    payload = real_daemon.remember(
                        f"{marker} shard {index:02d} journals the "
                        f"{'harbor' if profile_id == 'doris' else 'lantern'} "
                        f"rotation duty {RUN_TAG}-{index:02d}.",
                        profile_id=profile_id,
                        idempotency_key=f"{RUN_TAG}-conc-{profile_id}-{index:02d}",
                    )
                    if payload.get("profile") != profile_id:
                        failures.append(
                            f"{profile_id}[{index}] echoed profile "
                            f"{payload.get('profile')!r}"
                        )
                except AssertionError as exc:
                    failures.append(f"{profile_id}[{index}]: {exc}")

        threads = [
            threading.Thread(target=writer, args=("doris", doris_marker)),
            threading.Thread(target=writer, args=("zhihui", zhihui_marker)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=600)
        assert not any(thread.is_alive() for thread in threads), (
            "a concurrent writer thread did not finish"
        )
        assert failures == [], f"{len(failures)} writes failed: {failures[:5]}"

        # Grouped count: exactly 50/50 for the two markers.
        rows = real_daemon.fact_rows(
            "content LIKE ? OR content LIKE ?",
            (f"%{doris_marker}%", f"%{zhihui_marker}%"),
        )
        assert dict(rows) == {
            "doris": N_CONCURRENT_WRITES, "zhihui": N_CONCURRENT_WRITES,
        }, f"concurrent writes miscounted: {rows}"

        # Zero crossover: no row for one marker sits in the other profile
        # (or anywhere else), and zero orphans: total rows == 100.
        crossover, total = 0, 0
        conn = sqlite3.connect(real_daemon.data_root / "memory.db", timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            for marker, owner in (
                (doris_marker, "doris"), (zhihui_marker, "zhihui"),
            ):
                crossover += conn.execute(
                    "SELECT COUNT(*) FROM atomic_facts "
                    "WHERE content LIKE ? AND profile_id != ?",
                    (f"%{marker}%", owner),
                ).fetchone()[0]
                total += conn.execute(
                    "SELECT COUNT(*) FROM atomic_facts WHERE content LIKE ?",
                    (f"%{marker}%",),
                ).fetchone()[0]
        finally:
            conn.close()
        assert crossover == 0, f"{crossover} rows crossed profile boundaries"
        assert total == 2 * N_CONCURRENT_WRITES, (
            f"expected {2 * N_CONCURRENT_WRITES} rows, found {total} "
            "(missing rows = lost writes, extra = orphans)"
        )

        after = real_daemon.status()
        assert after["profile"] == before["profile"]
        assert after["profile_generation"] == before["profile_generation"]


class TestAcceptance6McpStdioLane:
    """Scenario 6: profile_id survives a full MCP stdio JSON-RPC round trip."""

    def test_mcp_stdio_remember_recall_carries_profile_id(self, real_daemon):
        # daemon_request preflights /health with a 2s timeout; only a fast
        # answer lets the child route through THIS daemon (a slow one makes
        # the tool return DAEMON_UNAVAILABLE instead of silently falling
        # back, so success below proves daemon routing).
        real_daemon.wait_health_fast()

        token = f"PrpE2e{RUN_TAG}McpSable"
        env = dict(real_daemon.env)
        env.update(
            {
                "SLM_MCP_TOOLS": "remember,recall",
                # Skip the ensure_daemon warmup thread: the descriptor in
                # our isolated root already names the test daemon, and the
                # test must never auto-start anything else.
                "SLM_DISABLE_WARMUP_SIDE_EFFECTS": "1",
            }
        )
        stderr_path = real_daemon.data_root.parent / "mcp-stderr.log"
        with stderr_path.open("wb") as stderr_log:
            mcp = subprocess.Popen(
                [sys.executable, "-m", "superlocalmemory.cli.main", "mcp"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr_log, env=env, cwd=str(REPO_ROOT),
            )
        client = _RpcClient(mcp, stderr_path)
        try:
            # Full handshake a real IDE performs over stdio.
            initialized = client.call("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "prp-e2e", "version": "1.0"},
            })
            assert "result" in initialized, initialized
            client.notify("notifications/initialized")

            # R5 allowlist: SLM_MCP_TOOLS=remember,recall exposes exactly
            # the minimal routed pair and nothing else.
            listed = client.call("tools/list", {})
            tool_names = {tool["name"] for tool in listed["result"]["tools"]}
            assert tool_names == {"remember", "recall"}, tool_names

            status_before = real_daemon.status()

            def tool_payload(result: dict) -> dict:
                assert result.get("result", {}).get("isError") is not True, result
                return json.loads(result["result"]["content"][0]["text"])

            remembered = tool_payload(client.call("tools/call", {
                "name": "remember",
                "arguments": {
                    "content": (
                        f"{token} charts the ferry wake schedule across the "
                        "strait for the night crew."
                    ),
                    "profile_id": "doris",
                    "idempotency_key": f"{RUN_TAG}-mcp-doris-1",
                },
            }))
            assert remembered["success"] is True, remembered
            assert remembered.get("fact_ids"), remembered

            # The write landed in doris — and nowhere else.
            rows = real_daemon.fact_rows(
                "content LIKE ?", (f"%{token}%",),
            )
            assert dict(rows) == {"doris": 1}, rows

            recalled = tool_payload(client.call("tools/call", {
                "name": "recall",
                "arguments": {
                    "query": f"{token} ferry wake schedule",
                    "profile_id": "doris",
                },
            }))
            assert recalled["success"] is True, recalled
            assert any(
                token in str(item.get("content", ""))
                for item in recalled.get("results", [])
            ), f"mcp recall must hit the doris namespace: {recalled}"

            # Cross-invisibility through the same lane: no doris-token fact
            # may surface from a zhihui-routed recall.
            cross = tool_payload(client.call("tools/call", {
                "name": "recall",
                "arguments": {
                    "query": f"{token} ferry wake schedule",
                    "profile_id": "zhihui",
                },
            }))
            assert not any(
                token in str(item.get("content", ""))
                for item in cross.get("results", [])
            ), f"mcp recall leaked doris facts into zhihui: {cross}"

            # The global pointer survived the whole stdio session.
            status_after = real_daemon.status()
            assert status_after["profile"] == status_before["profile"]
            assert (
                status_after["profile_generation"]
                == status_before["profile_generation"]
            )
        finally:
            client.close()
            assert mcp.poll() is not None, "mcp child must exit after stdin EOF"

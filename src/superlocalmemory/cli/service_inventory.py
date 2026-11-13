"""Discover service definitions whose recorded daemon namespace proves ownership.

A legacy name alone grants no authority. Definitions must record an absolute
canonical root, a valid port and the SLM daemon command. Earlier ports of the
same root remain manageable when the operator changes the configured port.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from superlocalmemory.cli.deployment import service_identity
from superlocalmemory.cli.service_live import verified_live
from superlocalmemory.infra.data_root import canonical_data_root
from superlocalmemory.infra.service_files import ServiceFiles
from superlocalmemory.infra.service_mutation import check_lock, serialized

logger = logging.getLogger(__name__)

LEGACY_NAMES = {"macos": "com.qualixar.superlocalmemory", "linux": "superlocalmemory",
                "windows": "SuperLocalMemory"}
_MODULE = "superlocalmemory.server.unified_daemon"


@dataclass(frozen=True)
class ServiceDefinition:
    name: str
    path: Path
    port: int
    content: str
    revision: tuple
    manager_definition: str = ""



def write(path: Path, content: str) -> bool:
    try:
        with ServiceFiles(path.parent, create=True) as files:
            files.write(path.name, content)
        return True
    except OSError as exc:
        # Issue #319: this used to vanish, leaving only "Check logs." Say
        # which file and which guard refused it.
        logger.error("Could not write service definition %s: %s", path, exc)
        return False


def _port(environment: dict, argv: list) -> int | None:
    root = environment.get("SLM_DATA_DIR", "")
    raw = environment.get("SLM_DAEMON_PORT", "")
    if not isinstance(root, str) or not Path(root).is_absolute():
        return None
    if os.path.normcase(str(Path(root).resolve())) != str(canonical_data_root()):
        return None
    if not isinstance(raw, str) or not re.fullmatch(r"[0-9]{1,5}", raw):
        return None
    port = int(raw)
    if not 1 <= port <= 65535 or not isinstance(argv, list) or len(argv) != 5:
        return None
    if not isinstance(argv[0], str) or not Path(argv[0]).is_absolute():
        return None
    return port if argv[1:] == ["-m", _MODULE, "--start", f"--port={port}"] else None


def _candidate(name: str, platform: str) -> bool:
    prefixes = {"com.qualixar.superlocalmemory", service_identity().rsplit(".", 1)[0]}
    return name == LEGACY_NAMES[platform] or any(
        re.fullmatch(re.escape(prefix) + r"\.[0-9a-f]{16}", name) for prefix in prefixes
    )


def _valid_name(name: str, platform: str, port: int) -> bool:
    if name == LEGACY_NAMES[platform]:
        return True
    digest = hashlib.sha256(f"{canonical_data_root()}:{port}".encode()).hexdigest()[:16]
    return _candidate(name, platform) and name.rsplit(".", 1)[-1] == digest


def _macos_port(content: str, name: str) -> int | None:
    import plistlib

    payload = plistlib.loads(content.encode())
    if "Program" in payload or payload.get("Label") != name:
        return None
    return _port(payload.get("EnvironmentVariables", {}), payload.get("ProgramArguments", []))


def _linux_port(content: str, name: str) -> int | None:
    environment, commands = {}, []
    section = ""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        if section != "[Service]" or "=" not in line or line.startswith(("#", ";")):
            continue
        key, value = line.split("=", 1)
        if key not in {"Type", "ExecStart", "Restart", "RestartSec",
                       "StandardOutput", "StandardError", "Environment"}:
            return None  # execution/namespace overrides make ownership unprovable
        if key == "Type" and value != "simple":
            return None
        if key == "Environment":
            if not value:
                environment = {}
            for pair in shlex.split(value):
                env_key, env_value = pair.split("=", 1)
                environment[env_key] = env_value
        if key == "ExecStart":
            commands.append(shlex.split(value))
    return _port(environment, commands[0]) if len(commands) == 1 else None


def _windows_port(content: str) -> int | None:
    environment, commands = {}, []
    for line in content.splitlines():
        if line == 'Set WshShell = CreateObject("WScript.Shell")' or not line.strip():
            continue
        assignment = re.fullmatch(
            r'WshShell.Environment\("Process"\)\("([A-Z_]+)"\) = "((?:[^"]|"")*)"', line
        )
        run = re.fullmatch(r'WshShell.Run "(.*)", 0, False', line)
        if assignment:
            if commands:
                return None  # ownership must describe the environment at launch time
            environment[assignment[1]] = assignment[2].replace('""', '"')
        elif run:
            commands.append(shlex.split(run[1].replace('""', '"')))
        else:
            return None
    return _port(environment, commands[0]) if len(commands) == 1 else None


def _task_wrapper(xml: str) -> Path | None:
    tree = ET.fromstring(xml)
    for element in tree.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    actions = tree.find("Actions")
    if actions is None or len(actions) != 1 or actions[0].tag != "Exec":
        return None
    command = actions[0].findtext("Command", "")
    if command.lower() != "wscript.exe":
        return None
    argument = actions[0].findtext("Arguments", "").strip()
    match = re.fullmatch(r'"([^"]+)"|([^\s"]+)', argument)
    return Path(match[1] or match[2]) if match else None


def _windows_definition(name: str) -> ServiceDefinition | None:
    result = subprocess.run(["schtasks", "/Query", "/TN", name, "/XML"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        return None
    path = _task_wrapper(result.stdout)
    root = canonical_data_root()
    if path is None or path.parent != root:
        return None  # never read the wrapper of another installation
    with ServiceFiles(root) as files:
        content, revision = files.snapshot(path.name)
    port = _windows_port(content)
    if port is None or not _valid_name(name, "windows", port):
        return None
    return ServiceDefinition(name, path, port, content, revision, result.stdout)


def _linux_effective(path: Path) -> bool:
    # Ask systemd about all override search paths, including runtime/global drop-ins.
    dropins = path.parent / f"{path.name}.d"
    if dropins.exists() or dropins.is_symlink():
        return False
    result = _run(["systemctl", "--user", "show", path.stem,
                   "--property=FragmentPath", "--property=DropInPaths"])
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return (result.returncode == 0 and properties.get("FragmentPath") == str(path)
            and properties.get("DropInPaths") == "")


def _file_definition(path: Path, platform: str) -> ServiceDefinition | None:
    try:
        with ServiceFiles(path.parent) as files:
            content, revision = files.snapshot(path.name)
        parser = _macos_port if platform == "macos" else _linux_port
        port = parser(content, path.stem)
        if port is None or not _valid_name(path.stem, platform, port):
            return None
        if platform == "linux" and not _linux_effective(path):
            return None
        return ServiceDefinition(path.stem, path, port, content, revision)
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError):
        return None


def definitions(platform: str) -> list[ServiceDefinition]:
    rows = []
    if platform == "windows":
        root = canonical_data_root()
        names = {LEGACY_NAMES[platform]}
        names.update(p.stem.removeprefix("start-daemon-") for p in root.glob("start-daemon-*.vbs"))
        for name in sorted(names):
            if not _candidate(name, platform):
                continue
            try:
                row = _windows_definition(name)
                if row is not None:
                    rows.append(row)
            except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError):
                continue
        return rows
    base = (Path.home().resolve() / "Library" / "LaunchAgents" if platform == "macos"
            else Path.home().resolve() / ".config" / "systemd" / "user")
    suffix = ".plist" if platform == "macos" else ".service"
    for path in sorted(base.glob(f"*{suffix}")):
        if not _candidate(path.stem, platform):
            continue
        row = _file_definition(path, platform)
        if row is not None:
            rows.append(row)
    return rows


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=10)


@serialized
def remove(platform: str, row: ServiceDefinition) -> bool:
    """Bind authorization to one open file and verify before every OS mutation."""
    try:
        with ServiceFiles(row.path.parent) as files:
            with files.verified(row.path.name, row.content, row.revision) as bound:
                if not _bound_owned(platform, row):
                    return False
                commands = _removal_commands(platform, row)
                for argv, required in commands:
                    loaded = verified_live(platform, row)
                    # The manager query itself can race an older installer.
                    check_lock()
                    bound.check()
                    if loaded and _run(argv).returncode and required:
                        return False
                    bound.check()
                check_lock()
                bound.remove()
        if platform == "linux" and _run(["systemctl", "--user", "daemon-reload"]).returncode:
            return False
        return True
    except (OSError, ValueError, TypeError, AttributeError, ET.ParseError,
            subprocess.SubprocessError):
        return False


def _bound_owned(platform: str, row: ServiceDefinition) -> bool:
    # Parse the bytes of the already-open object, never reopen a Windows leaf
    # held without write/delete sharing. Live manager identity is checked next.
    port = (_windows_port(row.content) if platform == "windows" else
            (_macos_port if platform == "macos" else _linux_port)(row.content, row.name))
    return port == row.port and _valid_name(row.name, platform, row.port)


def _removal_commands(platform: str, row: ServiceDefinition) -> list[tuple[list[str], bool]]:
    if platform == "macos":
        return [(["launchctl", "unload", str(row.path)], True)]
    if platform == "linux":
        return [(["systemctl", "--user", action, row.name], True) for action in ("stop", "disable")]
    return [(["schtasks", "/End", "/TN", row.name], False),
            (["schtasks", "/Delete", "/TN", row.name, "/F"], True)]


def _prepare_windows(path: Path, rows: list[ServiceDefinition]) -> bool:
    current = _run(["schtasks", "/Query", "/TN", service_identity(), "/XML"])
    if current.returncode == 0 and not any(row.name == service_identity() for row in rows):
        return False
    if (path.exists() or path.is_symlink()) and not any(row.path == path for row in rows):
        # A killed/failed task creation can leave an owned wrapper without a task.
        with ServiceFiles(path.parent) as files:
            with files.verified(path.name) as bound:
                port = _windows_port(bound.content)
                if port is None or not _valid_name(service_identity(), "windows", port):
                    return False
                check_lock()
                bound.remove()
    return True


@serialized
def prepare_install(platform: str, path: Path) -> bool:
    try:
        if path.parent.resolve() != path.parent.absolute():
            return False
        rows = definitions(platform)
        if platform == "windows":
            if not _prepare_windows(path, rows):
                return False
        elif path.exists() or path.is_symlink():
            if not any(row.path == path for row in rows):
                return False  # never overwrite an unprovable current-name definition
        return all(remove(platform, row) for row in rows)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


@serialized
def uninstall(platform: str) -> bool:
    return all(remove(platform, row) for row in definitions(platform))


def status(platform: str) -> dict:
    rows = definitions(platform)
    active = False
    if platform == "linux":
        for row in rows:
            try:
                state = _run(["systemctl", "--user", "is-active", row.name])
                active = active or state.stdout.strip() == "active"
            except (OSError, subprocess.SubprocessError):
                continue
    return {"installed": bool(rows), "active": active,
            "service_names": [row.name for row in rows],
            "service_ports": [row.port for row in rows],
            "legacy": any(row.name == LEGACY_NAMES[platform] for row in rows),
            "details": ", ".join(row.name for row in rows) if rows else "Not installed"}

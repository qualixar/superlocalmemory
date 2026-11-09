"""Fail-closed ownership checks of the effective OS job, not only its file."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.cli.service_inventory import ServiceDefinition


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=10, **kwargs)


def _macos(row: ServiceDefinition) -> bool:
    import plistlib  # macOS path only: the other platforms never load it

    from superlocalmemory.cli.service_inventory import _port

    result = _run(["launchctl", "list", row.name])
    if result.returncode in {3, 113}:  # ESRCH / launchctl's absent-service status
        return False
    if result.returncode:
        raise OSError("Cannot inspect loaded LaunchAgent")
    try:
        payload = plistlib.loads(result.stdout.encode())
    except plistlib.InvalidFileException:
        # Legacy launchctl list emits an OpenStep plist; plutil parses it,
        # without relying on the unsupported launchctl print text format.
        converted = _run(["plutil", "-convert", "json", "-o", "-", "--", "-"],
                         input=result.stdout)
        if converted.returncode:
            raise OSError("Cannot parse loaded LaunchAgent")
        payload = json.loads(converted.stdout)
    expected = plistlib.loads(row.content.encode())
    argv = payload.get("ProgramArguments", [])
    if (payload.get("Label") != row.name
            or argv != expected.get("ProgramArguments")
            or payload.get("Program", argv[0] if argv else None) != (argv[0] if argv else None)
            or _port(payload.get("EnvironmentVariables", {}), argv) != row.port):
        raise OSError("Loaded LaunchAgent belongs to another definition")
    return True


def _linux(row: ServiceDefinition) -> bool:
    from superlocalmemory.cli.service_inventory import _linux_effective, _port

    if not _linux_effective(row.path):
        raise OSError("Effective systemd fragment or overrides changed")
    result = _run(["systemctl", "--user", "show", row.name,
                   "--property=Environment", "--property=ExecStart"])
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    pairs = shlex.split(properties.get("Environment", ""))
    environment = dict(pair.split("=", 1) for pair in pairs)
    command = properties.get("ExecStart", "")
    expected = next(line.split("=", 1)[1] for line in row.content.splitlines()
                    if line.startswith("ExecStart="))
    argv = re.fullmatch(r"\{ path=(.*?) ; argv\[\]=(.*?) ; [^{}]* \}", command)
    if (result.returncode or argv is None
            or shlex.split(argv[2]) != shlex.split(expected)
            or _port(environment, shlex.split(argv[2])) != row.port
            or argv[1] != shlex.split(argv[2])[0]):
        raise OSError("Loaded systemd unit belongs to another definition")
    return True


def _windows(row: ServiceDefinition) -> bool:
    from superlocalmemory.cli.service_inventory import _task_wrapper

    result = _run(["schtasks", "/Query", "/TN", row.name, "/XML"])
    if (result.returncode or result.stdout != row.manager_definition
            or _task_wrapper(result.stdout) != row.path):
        raise OSError("Scheduled task changed since ownership validation")
    return True


def verified_live(platform: str, row: ServiceDefinition) -> bool:
    return {"macos": _macos, "linux": _linux, "windows": _windows}[platform](row)

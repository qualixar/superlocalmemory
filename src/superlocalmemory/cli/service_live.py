"""Fail-closed ownership checks of the effective OS job, not only its file."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.cli.service_inventory import ServiceDefinition


class ManagerQueryFailed(OSError):
    """The service manager could not be asked, or its answer could not be read."""


class JobMismatch(OSError):
    """The job the service manager holds is not the installed definition."""


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=10, **kwargs)


def launchctl_list(name: str) -> dict | None:
    """The loaded job's ``launchctl list`` record, or None when it is not loaded."""
    import plistlib  # macOS path only: the other platforms never load it

    from superlocalmemory.cli.launchctl_record import parse_list_output

    result = _run(["launchctl", "list", name])
    if result.returncode in {3, 113}:  # ESRCH / launchctl's absent-service status
        return None
    if result.returncode:
        raise ManagerQueryFailed("Cannot inspect loaded LaunchAgent")
    try:
        payload = plistlib.loads(result.stdout.encode())
    except plistlib.InvalidFileException:
        # launchctl list prints legacy text that is not valid OpenStep (array
        # items and the closing brace end in ';'), so plutil rejects it.
        try:
            payload = parse_list_output(result.stdout)
        except ValueError as exc:
            raise ManagerQueryFailed("Cannot parse loaded LaunchAgent") from exc
    if not isinstance(payload, dict):
        raise ManagerQueryFailed("Cannot parse loaded LaunchAgent")
    return payload


def launchctl_environment(name: str) -> dict[str, str]:
    """The environment launchd starts the loaded job with.

    ``launchctl list`` never reports it; ``launchctl print`` is the only place.
    """
    from superlocalmemory.cli.launchctl_record import print_environment

    result = _run(["launchctl", "print", f"gui/{os.getuid()}/{name}"])
    if result.returncode:
        raise ManagerQueryFailed("Cannot inspect loaded LaunchAgent environment")
    try:
        return print_environment(result.stdout)
    except ValueError as exc:
        raise ManagerQueryFailed("Cannot parse loaded LaunchAgent environment") from exc


def macos_owned_record(row: ServiceDefinition) -> dict | None:
    """The loaded job's record when it is exactly ``row``; None when not loaded.

    Raises ``JobMismatch`` when launchd holds a different job under this label
    (another installation, or a definition edited or replaced without a
    reload) and ``ManagerQueryFailed`` when launchd cannot be read.
    """
    import plistlib  # macOS path only: the other platforms never load it

    from superlocalmemory.cli.service_inventory import _port

    payload = launchctl_list(row.name)
    if payload is None:
        return None
    expected = plistlib.loads(row.content.encode())
    argv = payload.get("ProgramArguments", [])
    if payload.get("Label") != row.name or argv != expected.get("ProgramArguments"):
        raise JobMismatch("Loaded LaunchAgent belongs to another definition")
    environment = payload.get("EnvironmentVariables")
    if environment is None:
        environment = launchctl_environment(row.name)
    if (payload.get("Program", argv[0] if argv else None) != (argv[0] if argv else None)
            or _port(environment, argv) != row.port):
        raise JobMismatch("Loaded LaunchAgent belongs to another definition")
    return payload


def _macos(row: ServiceDefinition) -> bool:
    return macos_owned_record(row) is not None


def _linux(row: ServiceDefinition) -> bool:
    from superlocalmemory.cli.service_inventory import _linux_effective, _port

    if not _linux_effective(row.path):
        probe = _run(["systemctl", "--user", "show", row.name, "--property=LoadState"])
        if probe.returncode:
            raise ManagerQueryFailed("Cannot inspect systemd user unit")
        raise JobMismatch("Effective systemd fragment or overrides changed")
    result = _run(["systemctl", "--user", "show", row.name,
                   "--property=Environment", "--property=ExecStart"])
    if result.returncode:
        raise ManagerQueryFailed("Cannot inspect systemd user unit")
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    pairs = shlex.split(properties.get("Environment", ""))
    environment = dict(pair.split("=", 1) for pair in pairs)
    command = properties.get("ExecStart", "")
    expected = next(line.split("=", 1)[1] for line in row.content.splitlines()
                    if line.startswith("ExecStart="))
    argv = re.fullmatch(r"\{ path=(.*?) ; argv\[\]=(.*?) ; [^{}]* \}", command)
    if (argv is None
            or shlex.split(argv[2]) != shlex.split(expected)
            or _port(environment, shlex.split(argv[2])) != row.port
            or argv[1] != shlex.split(argv[2])[0]):
        raise JobMismatch("Loaded systemd unit belongs to another definition")
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

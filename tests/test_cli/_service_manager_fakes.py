# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""In-memory launchd and systemd that answer in the managers' real formats.

``launchctl list <label>`` really prints legacy text that ``plutil`` rejects
and that never includes the job's environment; ``launchctl print`` is where
the environment lives. Earlier fakes answered ``list`` with the plist XML of
the file itself, which hid both facts. ``fmt="xml"`` keeps that older answer
for tests that must isolate a check from the text parser.

Nothing here reaches a real service manager: the fake replaces
``subprocess.run`` and refuses any command it does not model.
"""

from __future__ import annotations

import plistlib
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace


def _quote(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass
class LaunchdJob:
    plist: dict
    path: Path
    pid: int | None = None


@dataclass
class FakeLaunchd:
    """launchd for one GUI domain, answering in its real text formats."""

    fmt: str = "legacy"
    jobs: dict[str, LaunchdJob] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    bootstrap_rc: int = 0
    kickstart_rc: int = 0
    start_pid: int | None = 4242  # None: the job's process never starts
    query_rc: int = 0  # non-zero: launchd cannot be asked at all
    next_pid: int = 5000

    def load(self, path: Path, pid: int | None = None, plist: dict | None = None) -> None:
        payload = plist if plist is not None else plistlib.loads(path.read_bytes())
        self.jobs[payload["Label"]] = LaunchdJob(payload, path, pid)

    @staticmethod
    def _result(code: int = 0, stdout: str = "", stderr: str = ""):
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

    def _list_text(self, job: LaunchdJob) -> str:
        argv = job.plist["ProgramArguments"]
        lines = ["{", f'\t"Label" = {_quote(job.plist["Label"])};',
                 '\t"LimitLoadToSessionType" = "Aqua";', '\t"OnDemand" = true;',
                 '\t"LastExitStatus" = 0;']
        if job.pid is not None:
            lines.append(f'\t"PID" = {job.pid};')
        lines.append(f'\t"Program" = {_quote(argv[0])};')
        lines.append('\t"ProgramArguments" = (')
        lines.extend(f"\t\t{_quote(item)};" for item in argv)
        lines.extend(["\t);", "};", ""])
        return "\n".join(lines)

    def _list_xml(self, job: LaunchdJob) -> str:
        record = {key: job.plist[key] for key in
                  ("Label", "ProgramArguments", "EnvironmentVariables") if key in job.plist}
        if job.pid is not None:
            record["PID"] = job.pid
        return plistlib.dumps(record).decode()

    def _print_text(self, job: LaunchdJob) -> str:
        environment = job.plist.get("EnvironmentVariables", {})
        lines = [f"gui/501/{job.plist['Label']} = {{", "\tactive count = 1",
                 f"\tpath = {job.path}", "\ttype = LaunchAgent",
                 f"\tstate = {'running' if job.pid else 'not running'}", "",
                 "\tinherited environment = {", "\t\tSSH_AUTH_SOCK => /tmp/agent.sock",
                 "\t}", "", "\tdefault environment = {", "\t\tPATH => /usr/bin:/bin", "\t}",
                 "", "\tenvironment = {"]
        lines.extend(f"\t\t{key} => {value}" for key, value in environment.items())
        lines.extend(["\t\tXPC_SERVICE_NAME => " + job.plist["Label"], "\t}", ""])
        if job.pid:
            lines.append(f"\tpid = {job.pid}")
        lines.extend(["}", ""])
        return "\n".join(lines)

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "plutil":
            # The real plutil rejects launchctl's legacy text.
            return self._result(1, stderr="Property List error: Unexpected character {")
        if argv[0] != "launchctl":
            raise AssertionError(f"unexpected command in a launchd test: {argv}")
        if self.query_rc:
            return self._result(self.query_rc, stderr="Could not reach the domain")
        verb = argv[1]
        if verb == "list":
            job = self.jobs.get(argv[2])
            if job is None:
                return self._result(113, stderr="Could not find service")
            text = self._list_text(job) if self.fmt == "legacy" else self._list_xml(job)
            return self._result(stdout=text)
        if verb == "print":
            job = self.jobs.get(argv[2].rsplit("/", 1)[-1])
            if job is None:
                return self._result(113, stderr="Could not find service")
            return self._result(stdout=self._print_text(job))
        if verb == "bootstrap":
            if self.bootstrap_rc:
                return self._result(self.bootstrap_rc, stderr="Bootstrap failed")
            self.load(Path(argv[3]))
            return self._result()
        if verb == "kickstart":
            job = self.jobs.get(argv[-1].rsplit("/", 1)[-1])
            if job is None or self.kickstart_rc:
                return self._result(self.kickstart_rc or 113, stderr="Could not kickstart")
            if job.pid is None:
                job.pid = self.start_pid
            return self._result()
        if verb in {"unload", "bootout"}:
            label = Path(argv[-1]).stem
            self.jobs.pop(label, None)
            return self._result()
        raise AssertionError(f"unmodelled launchctl call: {argv}")


@dataclass
class FakeSystemd:
    """A systemd user manager that holds units as last loaded (daemon-reload)."""

    units: dict[str, str] = field(default_factory=dict)  # name -> loaded unit text
    pids: dict[str, int] = field(default_factory=dict)
    home: Path = Path("/nonexistent")
    calls: list[list[str]] = field(default_factory=list)
    bus_rc: int = 0  # non-zero: no user bus
    start_pid: int | None = 4343

    @staticmethod
    def _result(code: int = 0, stdout: str = "", stderr: str = ""):
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

    def path(self, name: str) -> Path:
        return self.home / ".config" / "systemd" / "user" / f"{name}.service"

    def _show(self, name: str, argv: list[str]) -> str:
        loaded = self.units.get(name)
        values = {"LoadState": "loaded" if loaded else "not-found",
                  "FragmentPath": str(self.path(name)) if loaded else "",
                  "DropInPaths": "", "MainPID": str(self.pids.get(name, 0))}
        if loaded:
            env, command = [], ""
            for line in loaded.splitlines():
                if line.startswith("Environment="):
                    env.extend(shlex.split(line.split("=", 1)[1]))
                if line.startswith("ExecStart="):
                    command = line.split("=", 1)[1]
            values["Environment"] = shlex.join(env)
            values["ExecStart"] = (f"{{ path={shlex.split(command)[0]} ; argv[]={command} ; "
                                   "ignore_errors=no ; pid=0 ; status=0/0 }")
        wanted = [arg.split("=", 1)[1] for arg in argv if arg.startswith("--property=")]
        return "".join(f"{key}={values.get(key, '')}\n" for key in wanted)

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] != ["systemctl", "--user"]:
            raise AssertionError(f"unexpected command in a systemd test: {argv}")
        if self.bus_rc:
            return self._result(self.bus_rc, stderr="Failed to connect to bus")
        verb, name = argv[2], argv[3]
        if verb == "show":
            return self._result(stdout=self._show(name, argv))
        if verb == "start":
            if name not in self.units:
                return self._result(5, stderr="Unit not found")
            if self.start_pid is not None:
                self.pids[name] = self.start_pid
            return self._result()
        if verb == "is-active":
            return self._result(stdout="active" if self.pids.get(name) else "inactive")
        raise AssertionError(f"unmodelled systemctl call: {argv}")

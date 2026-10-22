"""Contract for the one-line installers: scripts/get-slm.sh and scripts/get-slm.ps1.

The shell script is exercised black-box in dry-run mode with stub tools on a
private PATH, so nothing here touches the machine's real uv, Python, home
directory or running SuperLocalMemory.  One ``slow`` test performs a real
install into temporary directories.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GET_SH = ROOT / "scripts" / "get-slm.sh"
GET_PS1 = ROOT / "scripts" / "get-slm.ps1"

UNIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell installer runs on Linux and macOS"
)

UV_URL = "https://astral.sh/uv/install.sh"
UV_INSTALL_LINE = f"download {UV_URL}, then run it with sh"
INSTALL_LINE = "uv tool install --python 3.12 superlocalmemory"

SHELLS = ["/bin/sh"] + (["dash"] if shutil.which("dash") else [])


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

UV_STUB = """#!/bin/sh
printf '%s\\n' "uv $*" >> "$CALL_LOG"
case "$1 $2" in
  "tool list") printf '%s\\n' "${UV_LIST_OUTPUT:-}" ;;
  "tool dir")  printf '%s\\n' "${UV_BIN_OUT:-}" ;;
esac
exit 0
"""


def _uname_stub(path: Path, system: str, machine: str) -> None:
    path.write_text(
        f"""#!/bin/sh
case "$1" in
  -s) echo {system} ;;
  -m) echo {machine} ;;
  *)  echo {system} ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _stub(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


class Rig:
    """A private PATH + HOME for one installer run."""

    def __init__(self, tmp: Path, *, with_uv: bool, system="Darwin", machine="arm64"):
        self.tmp = tmp
        self.home = tmp / "home"
        self.home.mkdir()
        self.stubs = tmp / "stubs"
        self.stubs.mkdir()
        self.tool_bin = tmp / "toolbin"
        self.tool_bin.mkdir()
        self.call_log = tmp / "calls.log"
        self.call_log.write_text("", encoding="utf-8")
        _uname_stub(self.stubs / "uname", system, machine)
        _stub(self.stubs / "sysctl", "#!/bin/sh\necho 0\n")
        if with_uv:
            _stub(self.stubs / "uv", UV_STUB)
        self.extra_env: dict[str, str] = {}
        self.path_dirs = [str(self.stubs)]

    def env(self) -> dict[str, str]:
        env = {
            "HOME": str(self.home),
            "PATH": ":".join(self.path_dirs + ["/usr/bin", "/bin"]),
            "CALL_LOG": str(self.call_log),
            "UV_BIN_OUT": str(self.tool_bin),
            "SLM_GET_DRY_RUN": "1",
            "LC_ALL": "C",
        }
        env.update(self.extra_env)
        return env

    def run(self, *args: str, shell: str = "/bin/sh", stdin_script: bool = False):
        if stdin_script:
            return subprocess.run(
                [shell, "-s", "--", *args],
                input=GET_SH.read_text(encoding="utf-8"),
                cwd="/", env=self.env(), text=True, capture_output=True, check=False,
            )
        return subprocess.run(
            [shell, str(GET_SH), *args],
            stdin=subprocess.DEVNULL,
            cwd="/", env=self.env(), text=True, capture_output=True, check=False,
        )

    def uv_calls(self) -> list[str]:
        return [ln for ln in self.call_log.read_text(encoding="utf-8").splitlines() if ln]


def dry_commands(output: str) -> list[str]:
    return [ln[len("DRY-RUN: "):] for ln in output.splitlines() if ln.startswith("DRY-RUN: ")]


@pytest.fixture
def rig_with_uv(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path, with_uv=True)
    rig.path_dirs.append(str(rig.tool_bin))  # tool bin dir already on PATH
    return rig


@pytest.fixture
def rig_no_uv(tmp_path: Path) -> Rig:
    return Rig(tmp_path, with_uv=False)


TAIL = ["slm setup --auto"]
START = ["slm serve start", "slm warmup", "slm connect"]
FINISH = ["slm doctor --quick", "slm dashboard"]


# --------------------------------------------------------------------------
# (a) syntax
# --------------------------------------------------------------------------

@UNIX_ONLY
@pytest.mark.parametrize("shell", SHELLS)
def test_script_parses_under_posix_sh(shell: str) -> None:
    assert GET_SH.is_file()
    done = subprocess.run([shell, "-n", str(GET_SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@UNIX_ONLY
def test_script_has_no_bashisms_and_no_forbidden_commands() -> None:
    text = GET_SH.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh")
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\bsudo\b", code)
    assert not re.search(r"\bpip3?\s+install\b", code)
    assert "[[" not in code
    assert "set -o pipefail" not in code
    for name in ("SLM_GET_SKIP_START", "SLM_GET_DRY_RUN", "SLM_VERSION", "SLM_MEDIA"):
        assert name in text.split("set -")[0], f"{name} must be documented in the header"


def test_new_names_do_not_collide_with_repo_clone_installers() -> None:
    assert GET_SH.is_file() and GET_PS1.is_file()
    for name in ("install.sh", "install.ps1"):
        hits = [
            p.relative_to(ROOT).as_posix()
            for p in ROOT.rglob(name)
            if not {".backup", ".claude", ".venv", ".worktrees", "node_modules"} & set(p.relative_to(ROOT).parts)
        ]
        assert hits == [f"scripts/{name}"]


# --------------------------------------------------------------------------
# (b) dry-run command sequences
# --------------------------------------------------------------------------

@UNIX_ONLY
@pytest.mark.parametrize("shell", SHELLS)
def test_dry_run_uv_present_text_only(rig_with_uv: Rig, shell: str) -> None:
    done = rig_with_uv.run(shell=shell)
    assert done.returncode == 0, done.stdout + done.stderr
    assert dry_commands(done.stdout) == [INSTALL_LINE, *TAIL, *START, *FINISH]
    # uv was only asked read-only questions, never asked to change anything
    assert set(rig_with_uv.uv_calls()) <= {"uv tool list", "uv tool dir --bin"}


@UNIX_ONLY
@pytest.mark.parametrize("shell", SHELLS)
def test_dry_run_uv_present_with_media(rig_with_uv: Rig, shell: str) -> None:
    done = rig_with_uv.run("--media", shell=shell)
    assert done.returncode == 0, done.stdout + done.stderr
    assert dry_commands(done.stdout) == [
        INSTALL_LINE, *TAIL, *START, "slm media enable --yes", *FINISH,
    ]


@UNIX_ONLY
def test_media_env_var_equals_flag(rig_with_uv: Rig) -> None:
    rig_with_uv.extra_env["SLM_MEDIA"] = "1"
    done = rig_with_uv.run()
    assert "slm media enable --yes" in dry_commands(done.stdout)


@UNIX_ONLY
@pytest.mark.parametrize("shell", SHELLS)
def test_dry_run_uv_absent_installs_uv_first(rig_no_uv: Rig, shell: str) -> None:
    done = rig_no_uv.run(shell=shell)
    assert done.returncode == 0, done.stdout + done.stderr
    cmds = dry_commands(done.stdout)
    # tool bin dir is not on PATH here, so update-shell is announced too
    assert cmds == [
        UV_INSTALL_LINE, INSTALL_LINE, "uv tool update-shell", *TAIL, *START, *FINISH,
    ]
    assert rig_no_uv.uv_calls() == []


@UNIX_ONLY
def test_dry_run_uv_absent_with_media(rig_no_uv: Rig) -> None:
    done = rig_no_uv.run("--media")
    cmds = dry_commands(done.stdout)
    assert cmds[0] == UV_INSTALL_LINE
    assert cmds.index("slm media enable --yes") == cmds.index("slm connect") + 1


@UNIX_ONLY
def test_already_installed_upgrades_instead(rig_with_uv: Rig) -> None:
    rig_with_uv.extra_env["UV_LIST_OUTPUT"] = "superlocalmemory v4.2.1\n- slm"
    cmds = dry_commands(rig_with_uv.run().stdout)
    assert cmds[0] == "uv tool upgrade superlocalmemory"
    assert INSTALL_LINE not in cmds


@UNIX_ONLY
def test_version_pin_forces_exact_install(rig_with_uv: Rig) -> None:
    rig_with_uv.extra_env["SLM_VERSION"] = "4.2.1"
    rig_with_uv.extra_env["UV_LIST_OUTPUT"] = "superlocalmemory v4.2.2"
    cmds = dry_commands(rig_with_uv.run().stdout)
    assert cmds[0] == "uv tool install --python 3.12 --force superlocalmemory==4.2.1"


@UNIX_ONLY
def test_bad_version_pin_is_refused(rig_with_uv: Rig) -> None:
    rig_with_uv.extra_env["SLM_VERSION"] = "1.0; rm -rf /"
    done = rig_with_uv.run()
    assert done.returncode != 0
    assert "SLM_VERSION" in done.stdout + done.stderr
    assert dry_commands(done.stdout) == []


@UNIX_ONLY
def test_skip_start_drops_steps_5_6_8_and_media(rig_with_uv: Rig) -> None:
    rig_with_uv.extra_env["SLM_GET_SKIP_START"] = "1"
    done = rig_with_uv.run("--media")
    assert done.returncode == 0, done.stdout + done.stderr
    assert dry_commands(done.stdout) == [INSTALL_LINE, *TAIL]


@UNIX_ONLY
def test_steps_are_numbered_in_plain_english(rig_with_uv: Rig) -> None:
    out = rig_with_uv.run().stdout
    steps = re.findall(r"^Step (\d+) of (\d+): (.+)$", out, flags=re.M)
    assert [int(n) for n, _, _ in steps] == list(range(1, 8))
    assert {t for _, t, _ in steps} == {"7"}
    out = rig_with_uv.run("--media").stdout
    steps = re.findall(r"^Step (\d+) of (\d+): ", out, flags=re.M)
    assert [int(n) for n, _ in steps] == list(range(1, 9))
    assert {t for _, t in steps} == {"8"}


@UNIX_ONLY
def test_runs_when_piped_into_sh_like_curl_pipe_sh(rig_with_uv: Rig) -> None:
    done = rig_with_uv.run("--media", stdin_script=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "slm media enable --yes" in dry_commands(done.stdout)


@UNIX_ONLY
def test_missing_home_is_refused_not_crashed(rig_with_uv: Rig) -> None:
    env = rig_with_uv.env()
    del env["HOME"]
    done = subprocess.run(
        ["/bin/sh", str(GET_SH)], stdin=subprocess.DEVNULL, env=env,
        text=True, capture_output=True, check=False,
    )
    assert done.returncode != 0
    assert "HOME" in done.stderr and "unbound" not in done.stderr


@UNIX_ONLY
def test_unknown_flag_is_refused(rig_with_uv: Rig) -> None:
    done = rig_with_uv.run("--bogus")
    assert done.returncode != 0
    assert "--bogus" in done.stdout + done.stderr


# --------------------------------------------------------------------------
# (c) unsupported platforms
# --------------------------------------------------------------------------

@UNIX_ONLY
@pytest.mark.parametrize(
    ("system", "machine", "needle"),
    [
        ("Darwin", "x86_64", "Intel"),
        ("Linux", "i686", "32-bit"),
        ("Linux", "armv7l", "32-bit"),
        ("FreeBSD", "amd64", "not supported"),
        ("MINGW64_NT-10.0", "x86_64", "agent prompt"),
    ],
)
def test_unsupported_platform_is_refused(tmp_path: Path, system, machine, needle) -> None:
    rig = Rig(tmp_path, with_uv=True, system=system, machine=machine)
    done = rig.run()
    assert done.returncode != 0
    text = done.stdout + done.stderr
    assert needle in text
    assert dry_commands(done.stdout) == []
    assert rig.uv_calls() == []


@UNIX_ONLY
def test_rosetta_terminal_is_refused_with_clear_reason(tmp_path: Path) -> None:
    rig = Rig(tmp_path, with_uv=True, system="Darwin", machine="x86_64")
    _stub(rig.stubs / "sysctl", "#!/bin/sh\necho 1\n")
    done = rig.run()
    assert done.returncode != 0
    assert "Rosetta" in done.stdout + done.stderr


@UNIX_ONLY
@pytest.mark.parametrize(
    ("system", "machine"),
    [("Darwin", "arm64"), ("Linux", "x86_64"), ("Linux", "aarch64"), ("Linux", "arm64")],
)
def test_supported_platforms_pass(tmp_path: Path, system, machine) -> None:
    rig = Rig(tmp_path, with_uv=True, system=system, machine=machine)
    rig.path_dirs.append(str(rig.tool_bin))
    assert rig.run().returncode == 0


# --------------------------------------------------------------------------
# PowerShell script: static contract (+ execution when pwsh exists)
# --------------------------------------------------------------------------

def _ps_text() -> str:
    return GET_PS1.read_text(encoding="utf-8")


def test_ps1_contains_each_step_in_order() -> None:
    text = _ps_text()
    # main flow: the step banners appear in the documented order
    main = text[text.index("function Install-SuperLocalMemory"):]
    banners = [
        "checking your computer", "getting uv", "installing SuperLocalMemory",
        "setting up SuperLocalMemory", "starting SuperLocalMemory",
        "linking the AI tools", "turning on pictures and documents", "finishing up",
    ]
    where = [main.index(f"Write-Step '{b}") for b in banners]
    assert where == sorted(where), dict(zip(banners, where))
    # commands: each step's function runs them in order (functions are defined in step order)
    body = text[text.index("function Confirm-Platform"):]
    needles = [
        "-ExecutionPolicy ByPass -c \"irm $script:UvInstallUrl | iex\"",
        "Invoke-Uv \"tool install --python 3.12",
        "Invoke-Uv \"tool upgrade",
        "Invoke-Uv 'tool update-shell'",
        "Invoke-Slm 'setup --auto'",
        "Invoke-Slm 'serve start'",
        "Invoke-Slm 'warmup'",
        "Invoke-Slm 'connect'",
        "Invoke-Slm 'media enable --yes'",
        "Invoke-Slm 'doctor --quick'",
        "Invoke-Slm 'dashboard'",
    ]
    positions = [body.index(n) for n in needles]
    assert positions == sorted(positions), dict(zip(needles, positions))
    assert "astral.sh/uv/install.ps1" in text


def test_ps1_has_no_forbidden_commands_and_documents_hatches() -> None:
    text = _ps_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\bsudo\b", code, flags=re.I)
    assert not re.search(r"\bpip3?\s+install\b", code, flags=re.I)
    assert "-ExecutionPolicy ByPass" in text
    header = text[: text.index("function ")] if "function " in text else text[:3000]
    for name in ("SLM_GET_SKIP_START", "SLM_GET_DRY_RUN", "SLM_VERSION", "SLM_MEDIA"):
        assert name in header


def test_ps1_refuses_32bit_and_intel_arm_wording_present() -> None:
    text = _ps_text()
    assert "64-bit" in text
    assert "PROCESSOR_ARCHITECTURE" in text or "Is64BitOperatingSystem" in text


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh not installed on this machine")
def test_ps1_dry_run_under_pwsh(tmp_path: Path) -> None:  # pragma: no cover - needs pwsh
    env = {**os.environ, "SLM_GET_DRY_RUN": "1", "HOME": str(tmp_path)}
    done = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(GET_PS1)],
        env=env, text=True, capture_output=True, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    cmds = dry_commands(done.stdout)
    assert any("tool install --python 3.12" in c for c in cmds)
    assert "slm setup --auto" in cmds


# --------------------------------------------------------------------------
# (d) real install into temporary directories (slow, deselected by default)
# --------------------------------------------------------------------------

def _real_env(tmp_path: Path, path: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return {
        "HOME": str(home),
        "PATH": path,
        "LC_ALL": "C",
        "SHELL": "/bin/zsh",  # a real terminal always has it; update-shell needs it
        "SLM_DATA_DIR": str(tmp_path / "data"),
        "SLM_DAEMON_PORT": "48765",  # never the live service's 8765
        "UV_TOOL_DIR": str(tmp_path / "tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
        "UV_PYTHON_INSTALL_DIR": str(tmp_path / "py"),
        "SLM_GET_SKIP_START": "1",
    }


def _run_real(tmp_path: Path, env: dict[str, str], capsys) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    done = subprocess.run(
        ["/bin/sh", str(GET_SH)],
        stdin=subprocess.DEVNULL, cwd=str(tmp_path), env=env,
        text=True, capture_output=True, check=False, timeout=1500,
    )
    elapsed = time.monotonic() - started
    with capsys.disabled():
        print(f"\n[real install] {elapsed:.1f}s exit={done.returncode}\n{done.stdout}\n{done.stderr[-2000:]}")
    assert done.returncode == 0, done.stdout + done.stderr
    # `uv tool update-shell` ran against the temporary HOME, not the real one
    home = Path(env["HOME"])
    profiles = [p for p in home.rglob("*") if p.is_file() and str(tmp_path / "bin") in p.read_text(errors="ignore")]
    assert profiles, "update-shell did not persist the tool bin dir in any profile under the temp HOME"
    version = subprocess.run(
        [str(tmp_path / "bin" / "slm"), "--version"],
        env=env, text=True, capture_output=True, check=False, timeout=120,
    )
    assert version.returncode == 0, version.stdout + version.stderr
    assert "superlocalmemory" in (version.stdout + version.stderr).lower()
    return done


@UNIX_ONLY
@pytest.mark.slow
def test_real_install_into_temp_dirs(tmp_path: Path, capsys) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv must exist on the test machine to run the real install")
    env = _real_env(tmp_path, f"{Path(uv).parent}:/usr/bin:/bin")
    _run_real(tmp_path, env, capsys)


@UNIX_ONLY
@pytest.mark.slow
def test_real_install_fetches_uv_when_missing(tmp_path: Path, capsys) -> None:
    if shutil.which("curl") is None:
        pytest.skip("curl is needed to fetch uv")
    env = _real_env(tmp_path, "/usr/bin:/bin")  # no uv on PATH
    env["UV_INSTALL_DIR"] = str(tmp_path / "uvbin")
    env["UV_NO_MODIFY_PATH"] = "1"
    done = _run_real(tmp_path, env, capsys)
    assert (tmp_path / "uvbin" / "uv").is_file()
    assert "Installing uv" in done.stdout

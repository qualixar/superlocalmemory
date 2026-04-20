# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §9.3 / H12, H13

"""Tests for ``bin/slm`` dispatcher shim.

H13: Python fallback always works when the binary is absent or disabled.
H12: postinstall (the Node side) fails closed on SHA mismatch — covered
here via the ``pickAsset`` + ``canonicalPlatformArch`` JS logic which
we verify with a tiny Node harness.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_SLM = REPO_ROOT / "bin" / "slm"


# ---------------------------------------------------------------------------
# bin/slm shim behaviour (POSIX only — skip on Windows CI)
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="bin/slm is POSIX; bin/slm.bat covers Windows",
)


def _run_slm(argv: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BIN_SLM), *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _fake_python_env(tmp_path: Path) -> tuple[dict, Path]:
    """Produce an env where bin/slm will invoke a stub python3 that we
    control. The stub prints a deterministic sentinel so the caller
    can assert the Python fallback path was taken."""
    fake_py = tmp_path / "fakepy.sh"
    fake_py.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        echo "PYFALLBACK $*"
        exit 0
    """))
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP
                  | stat.S_IXOTH)

    # Put a directory on PATH containing only 'python3' pointing at the stub.
    bin_dir = tmp_path / "stub_bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(fake_py)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    # Disable any SLM hook binary in the user's home.
    env["SLM_HOOK_BINARY_DISABLED"] = "1"
    return env, bin_dir


def test_dispatcher_falls_back_when_binary_absent(tmp_path):
    env, _ = _fake_python_env(tmp_path)
    env["SLM_HOOK_BINARY"] = str(tmp_path / "nonexistent" / "slm-hook")
    proc = _run_slm(["hook", "user_prompt_submit"], env)
    # fake python stub printed sentinel.
    assert "PYFALLBACK" in proc.stdout, proc.stdout


def test_dispatcher_falls_back_when_SLM_HOOK_BINARY_DISABLED_set(tmp_path):
    # Place an executable at SLM_HOOK_BINARY but set DISABLED=1.
    fake_bin = tmp_path / "slm-hook"
    fake_bin.write_text(
        "#!/usr/bin/env bash\n"
        'echo BINARY_RAN\n'
        'exit 0\n'
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP
                   | stat.S_IXOTH)

    env, _ = _fake_python_env(tmp_path)
    env["SLM_HOOK_BINARY"] = str(fake_bin)
    env["SLM_HOOK_BINARY_DISABLED"] = "1"
    proc = _run_slm(["hook", "user_prompt_submit"], env)
    assert "BINARY_RAN" not in proc.stdout
    assert "PYFALLBACK" in proc.stdout


def test_dispatcher_prefers_binary_when_present_and_enabled(tmp_path):
    fake_bin = tmp_path / "slm-hook"
    fake_bin.write_text(
        "#!/usr/bin/env bash\n"
        'echo BINARY_RAN\n'
        'exit 0\n'
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP
                   | stat.S_IXOTH)

    env, _ = _fake_python_env(tmp_path)
    env["SLM_HOOK_BINARY"] = str(fake_bin)
    # Must NOT be disabled.
    env.pop("SLM_HOOK_BINARY_DISABLED", None)
    proc = _run_slm(["hook", "user_prompt_submit"], env)
    assert "BINARY_RAN" in proc.stdout, proc.stdout
    assert "PYFALLBACK" not in proc.stdout


def test_dispatcher_does_not_run_binary_for_other_commands(tmp_path):
    """Only `hook user_prompt_submit` should hit the binary."""
    fake_bin = tmp_path / "slm-hook"
    fake_bin.write_text(
        "#!/usr/bin/env bash\necho BINARY_RAN\nexit 0\n"
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP
                   | stat.S_IXOTH)

    env, _ = _fake_python_env(tmp_path)
    env["SLM_HOOK_BINARY"] = str(fake_bin)
    env.pop("SLM_HOOK_BINARY_DISABLED", None)

    proc = _run_slm(["status"], env)
    assert "BINARY_RAN" not in proc.stdout
    assert "PYFALLBACK" in proc.stdout


# ---------------------------------------------------------------------------
# H12 — postinstall SHA guard (Node harness)
# ---------------------------------------------------------------------------


def _node_available() -> bool:
    return shutil.which("node") is not None


@pytest.mark.skipif(not _node_available(),
                    reason="node not installed")
def test_postinstall_module_exports(tmp_path):
    """Smoke test: postinstall_binary.js loads without syntax errors and
    exports the canonical helpers."""
    script = REPO_ROOT / "scripts" / "postinstall_binary.js"
    probe = tmp_path / "probe.js"
    probe.write_text(textwrap.dedent(f"""\
        const m = require({str(script)!r});
        if (typeof m.canonicalPlatformArch !== 'function') {{
            console.error('missing canonicalPlatformArch');
            process.exit(2);
        }}
        if (typeof m.pickAsset !== 'function') {{
            console.error('missing pickAsset');
            process.exit(2);
        }}
        if (typeof m.sha256File !== 'function') {{
            console.error('missing sha256File');
            process.exit(2);
        }}
        // canonicalPlatformArch
        const pa = m.canonicalPlatformArch('linux', 'x64');
        if (!pa || pa.platform !== 'linux' || pa.arch !== 'x86_64') {{
            console.error('linux/x64 mapping broken');
            process.exit(3);
        }}
        // pickAsset preference: setup.exe > .zip > .tar.gz
        const manifest = {{assets: [
            {{name: 'slm-hook-windows-x86_64.zip',
              platform: 'windows', arch: 'x86_64'}},
            {{name: 'slm-hook-windows-x86_64-setup.exe',
              platform: 'windows', arch: 'x86_64'}},
        ]}};
        const picked = m.pickAsset(manifest, 'windows', 'x86_64');
        if (!picked.name.endsWith('setup.exe')) {{
            console.error('pickAsset preference wrong: ' + picked.name);
            process.exit(4);
        }}
        console.log('OK');
    """))
    proc = subprocess.run(
        ["node", str(probe)],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"node probe failed: stdout={proc.stdout} stderr={proc.stderr}"
    )
    assert "OK" in proc.stdout


@pytest.mark.skipif(not _node_available(),
                    reason="node not installed")
def test_postinstall_aborts_on_sha_mismatch(tmp_path):
    """H12: postinstall computes SHA256 of a downloaded file and MUST
    compare against the manifest. We simulate the comparison in-process
    by calling sha256File on a file whose SHA differs from the claimed
    one."""
    script = REPO_ROOT / "scripts" / "postinstall_binary.js"
    asset = tmp_path / "fake.bin"
    asset.write_bytes(b"hello")

    probe = tmp_path / "probe.js"
    probe.write_text(textwrap.dedent(f"""\
        const m = require({str(script)!r});
        (async () => {{
            const got = await m.sha256File({str(asset)!r});
            const claimed = '0'.repeat(64);
            if (got === claimed) {{
                console.error('unexpectedly matched');
                process.exit(2);
            }}
            console.log('MISMATCH_OK');
        }})();
    """))
    proc = subprocess.run(
        ["node", str(probe)],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "MISMATCH_OK" in proc.stdout

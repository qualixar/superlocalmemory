"""PR-245 recheck #1: concurrent legacy adoption must never stop another root."""

import plistlib
import sys
from types import SimpleNamespace

import pytest

from superlocalmemory.cli import service_installer as service
from tests.test_cli.test_legacy_service_migration import (
    sandbox as _sandbox_fixture,
)
from tests.test_cli.test_legacy_service_migration import (
    write_definition,
)

sandbox = _sandbox_fixture

WINDOWS_HOST = sys.platform == "win32"
_LOCKED_ON_WINDOWS = (
    "replays another installer rewriting the definition mid-operation; on a "
    "Windows host the bound definition is held with write and delete sharing "
    "denied, so that rewrite cannot happen (see test_service_definition_handles)"
)


def skip_simulated_posix_on_windows(platform):
    """macOS/Linux cases fake the service manager but use the host
    filesystem, whose rename and inode rules differ on Windows."""
    if WINDOWS_HOST and platform != "windows":
        pytest.skip("simulates a POSIX service manager on the host filesystem")


def destructive(argv, platform, name):
    actions = {"macos": {"unload"}, "linux": {"stop", "disable"},
               "windows": {"/End", "/Delete"}}[platform]
    return name in " ".join(argv) and bool(actions.intersection(argv))


@pytest.mark.skipif(WINDOWS_HOST, reason=_LOCKED_ON_WINDOWS)
@pytest.mark.parametrize("platform", ["macos", "linux", "windows"])
@pytest.mark.parametrize("operation", ["install", "uninstall"])
def test_manager_query_swap_never_mutates_foreign_service(
    sandbox, monkeypatch, platform, operation,
):
    name, path = write_definition(sandbox, platform)
    foreign = sandbox.root / "foreign"
    foreign.mkdir()
    original = path.read_text()
    replacement = original.replace(str(sandbox.root), str(foreign))
    run = service.subprocess.run
    queried, swapped, foreign_mutations = 0, False, []

    def swap(argv, **kwargs):
        nonlocal queried, swapped
        result = run(argv, **kwargs)
        query = {"macos": "list", "linux": "show", "windows": "/Query"}[platform]
        if query in argv and name in argv:
            queried += 1
            if platform == "macos" or queried == 2:
                # Return the old ownership snapshot while the effective job and
                # definition have both been replaced by the other installer.
                path.write_text(replacement)
                sandbox.live[name] = replacement
                swapped = True
        if swapped and destructive(argv, platform, name):
            foreign_mutations.append(argv)
        return result

    monkeypatch.setattr(service.subprocess, "run", swap)
    assert getattr(service, f"{operation}_{platform}")() is False
    assert swapped
    assert foreign_mutations == []
    assert path.read_text() == replacement
    if platform == "macos":
        environment = plistlib.loads(path.read_bytes())["EnvironmentVariables"]
        assert environment["SLM_DATA_DIR"] == str(foreign)


@pytest.mark.skipif(WINDOWS_HOST, reason="macOS launchctl race on a POSIX filesystem")
def test_exact_reviewer_launchctl_list_interleaving(sandbox, monkeypatch):
    _, path = write_definition(sandbox, "macos")
    foreign = sandbox.root / "B"
    foreign.mkdir()
    owned_content = path.read_text()
    payload = plistlib.loads(path.read_bytes())
    payload["EnvironmentVariables"]["SLM_DATA_DIR"] = str(foreign)
    replacement = plistlib.dumps(payload)
    loaded_root, unloaded_roots = str(sandbox.root), []

    def manager(argv, **kwargs):
        nonlocal loaded_root
        if "list" in argv:
            path.write_bytes(replacement)
            loaded_root = str(foreign)
        if "unload" in argv:
            unloaded_roots.append(loaded_root)
        return SimpleNamespace(returncode=0, stdout=owned_content, stderr="")

    monkeypatch.setattr(service.subprocess, "run", manager)
    assert service.uninstall_macos() is False
    assert unloaded_roots == []
    assert path.read_bytes() == replacement


@pytest.mark.parametrize("platform", ["macos", "linux", "windows"])
def test_foreign_effective_job_with_unchanged_owned_file_is_preserved(
    sandbox, monkeypatch, platform,
):
    skip_simulated_posix_on_windows(platform)
    from tests.test_cli.test_legacy_service_migration import task_xml

    name, path = write_definition(sandbox, platform)
    original = path.read_bytes()
    foreign = sandbox.root / "foreign"
    foreign.mkdir()
    sandbox.live[name] = path.read_text().replace(str(sandbox.root), str(foreign))
    if platform == "windows":
        # Switch task only after discovery, while leaving the bound wrapper A.
        run = service.subprocess.run
        queried = 0

        def replace_task(argv, **kwargs):
            nonlocal queried
            if "/Query" in argv and name in argv:
                queried += 1
                if queried == 2:
                    sandbox.tasks[name] = task_xml(foreign / "start-daemon.vbs")
            return run(argv, **kwargs)

        monkeypatch.setattr(service.subprocess, "run", replace_task)
    assert getattr(service, f"uninstall_{platform}")() is False
    assert not any(destructive(argv, platform, name) for argv in sandbox.calls)
    assert path.read_bytes() == original


@pytest.mark.parametrize("platform", ["linux", "windows"])
@pytest.mark.parametrize("swap", ["file", "job"])
def test_swap_after_first_destructive_step_prevents_the_next(
    sandbox, monkeypatch, platform, swap,
):
    skip_simulated_posix_on_windows(platform)
    from tests.test_cli.test_legacy_service_migration import task_xml

    name, path = write_definition(sandbox, platform)
    original = path.read_text()
    foreign = sandbox.root / "foreign"
    foreign.mkdir()
    replacement = original.replace(str(sandbox.root), str(foreign))
    first = {"linux": "stop", "windows": "/End"}[platform]
    second = {"linux": "disable", "windows": "/Delete"}[platform]
    run = service.subprocess.run

    def replace_after_stop(argv, **kwargs):
        result = run(argv, **kwargs)
        if first in argv and name in argv:
            if swap == "file":
                path.write_text(replacement)
            sandbox.live[name] = replacement
            if platform == "windows":
                sandbox.tasks[name] = task_xml(foreign / "start-daemon.vbs")
        return result

    monkeypatch.setattr(service.subprocess, "run", replace_after_stop)
    assert getattr(service, f"uninstall_{platform}")() is False
    assert any(first in argv for argv in sandbox.calls)
    assert not any(second in argv for argv in sandbox.calls)
    assert path.read_text() == (replacement if swap == "file" else original)


@pytest.mark.parametrize("platform", ["macos", "linux", "windows"])
def test_identical_bytes_replaced_by_new_inode_are_not_the_verified_object(
    sandbox, platform,
):
    skip_simulated_posix_on_windows(platform)
    from superlocalmemory.cli import service_inventory

    _, path = write_definition(sandbox, platform)
    row = service_inventory.definitions(platform)[0]
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    sandbox.calls.clear()
    assert service_inventory.remove(platform, row) is False
    assert path.read_text() == row.content
    assert not any(destructive(argv, platform, row.name) for argv in sandbox.calls)


def test_shared_lock_blocks_other_root_install_and_uninstall(sandbox, monkeypatch):
    import os
    import subprocess
    import sys

    from superlocalmemory.infra.service_mutation import mutation

    # A separate Python process must contend on the same permanent inode,
    # irrespective of its selected data root.
    script = (
        "from superlocalmemory.cli import service_installer as s; "
        "assert s.install_macos() is False; assert s.uninstall_macos() is False"
    )
    with mutation():
        # sandbox's fake subprocess.run is for managers, use Popen directly.
        env = {**os.environ, "SLM_DATA_DIR": str(sandbox.root / "B")}
        process = subprocess.Popen([sys.executable, "-c", script], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
    assert not sandbox.calls


def test_windows_bound_wrapper_is_never_opened_twice(sandbox, monkeypatch):
    from superlocalmemory.infra import service_files

    _, path = write_definition(sandbox, "windows")
    original = service_files._definition_fd
    opens = 0

    def once(files, name):
        nonlocal opens
        if files.path / name == path:
            opens += 1
            # Discovery is one open; the removal handle is the second. Native
            # Windows denies another DELETE open while that handle is pinned.
            assert opens <= 2, "Reopened a Windows wrapper held without delete sharing"
        return original(files, name)

    monkeypatch.setattr(service_files, "_definition_fd", once)
    assert service.uninstall_windows()
    assert opens == 2
    assert not path.exists()


def test_orphan_wrapper_swap_cannot_delete_foreign_bytes(sandbox, monkeypatch):
    from superlocalmemory.cli import service_inventory

    path = sandbox.root / f"start-daemon-{service.service_identity()}.vbs"
    path.write_text(service._windows_vbs_content())
    foreign = sandbox.root / "foreign"
    foreign.mkdir()
    replacement = path.read_text().replace(str(sandbox.root), str(foreign))
    parser = service_inventory._windows_port

    def swap_after_ownership(content):
        port = parser(content)
        path.write_text(replacement)
        return port

    monkeypatch.setattr(service_inventory, "_windows_port", swap_after_ownership)
    assert service.install_windows() is False
    assert path.read_text() == replacement
    assert not any("/Create" in argv for argv in sandbox.calls)

"""Archive-level contracts for the candidate wheel and sdist."""

from __future__ import annotations

import email.parser
import tarfile
import zipfile
from pathlib import Path

import pytest

from ._harness import (
    BuiltArtifacts,
    forbidden_wheel_names,
    sdist_names,
    unsafe_archive_names,
    wheel_names,
)

pytestmark = pytest.mark.slow


def _sdist_prefix(artifacts: BuiltArtifacts) -> str:
    names = sdist_names(artifacts.sdist)
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    assert len(roots) == 1, f"sdist must have one root directory: {roots}"
    return roots.pop()


def _wheel_metadata(artifacts: BuiltArtifacts):
    with zipfile.ZipFile(artifacts.wheel) as archive:
        metadata_names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1
        raw = archive.read(metadata_names[0]).decode("utf-8")
    return email.parser.Parser().parsestr(raw)


def test_archives_have_safe_member_names_and_types(
    built_artifacts: BuiltArtifacts,
) -> None:
    wheel_members = wheel_names(built_artifacts.wheel)
    sdist_members = sdist_names(built_artifacts.sdist)
    assert unsafe_archive_names(wheel_members) == []
    assert unsafe_archive_names(sdist_members) == []

    with tarfile.open(built_artifacts.sdist, "r:gz") as archive:
        special = [
            member.name
            for member in archive.getmembers()
            if member.issym() or member.islnk() or member.isdev()
        ]
    assert special == [], f"sdist contains links or device nodes: {special}"


def test_wheel_contains_every_python_module_and_ui_asset(
    built_artifacts: BuiltArtifacts,
) -> None:
    names = wheel_names(built_artifacts.wheel)
    source_root = built_artifacts.snapshot / "src" / "superlocalmemory"
    expected_python = {
        path.relative_to(source_root.parent).as_posix()
        for path in source_root.rglob("*.py")
    }
    expected_ui = {
        path.relative_to(source_root.parent).as_posix()
        for path in (source_root / "ui").rglob("*")
        if path.is_file()
    }
    packaged_python = {
        name
        for name in names
        if name.startswith("superlocalmemory/") and name.endswith(".py")
    }
    assert expected_python == packaged_python
    assert expected_ui <= names
    assert "superlocalmemory/infra/daemon_identity.py" in names


def test_sdist_contains_every_python_module_and_ui_asset(
    built_artifacts: BuiltArtifacts,
) -> None:
    names = sdist_names(built_artifacts.sdist)
    prefix = _sdist_prefix(built_artifacts)
    source_root = built_artifacts.snapshot / "src" / "superlocalmemory"
    expected_python = {
        f"{prefix}/src/{path.relative_to(source_root.parent).as_posix()}"
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix == ".py"
    }
    expected_ui = {
        f"{prefix}/src/{path.relative_to(source_root.parent).as_posix()}"
        for path in (source_root / "ui").rglob("*")
        if path.is_file()
    }
    packaged_python = {
        name
        for name in names
        if name.startswith(f"{prefix}/src/superlocalmemory/")
        and name.endswith(".py")
    }
    assert expected_python == packaged_python
    assert expected_ui <= names


def test_wheel_contains_root_and_optimize_legal_notices(
    built_artifacts: BuiltArtifacts,
) -> None:
    names = wheel_names(built_artifacts.wheel)
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    assert any(name.endswith(".dist-info/licenses/NOTICE") for name in names)
    assert "superlocalmemory/optimize/NOTICE" in names


def test_wheel_contains_packaged_codex_skill_assets(
    built_artifacts: BuiltArtifacts,
) -> None:
    """Every pyproject data-file reference must survive the release snapshot."""
    names = wheel_names(built_artifacts.wheel)
    expected_suffixes = {
        "data/share/superlocalmemory/codex/skills/"
        f"{skill}/SKILL.md"
        for skill in (
            "slm-cache", "slm-compress", "slm-graph", "slm-recall",
            "slm-remember", "slm-session", "slm-status",
        )
    }
    assert all(any(name.endswith(suffix) for name in names) for suffix in expected_suffixes)


def test_wheel_contains_the_claude_code_skills_agents_and_command(
    built_artifacts: BuiltArtifacts,
) -> None:
    """``slm upgrade-hosts`` installs these into a Claude Code that has no plugin, from the wheel."""
    from superlocalmemory.hooks.claude_code_assets import planned_files

    names = wheel_names(built_artifacts.wheel)
    root = Path(__file__).resolve().parents[2] / "plugin-src"
    wanted_bytes = planned_files(root)
    wanted = set(wanted_bytes)
    assert len(wanted) >= 21
    prefix = "data/share/superlocalmemory/claude-code/"
    for rel in wanted:
        assert any(name.endswith(prefix + rel) for name in names), rel
    # The sub-agents in the wheel are the generated, full-tool-name ones, never plugin-src's bare ones.
    agents = sorted(rel for rel in wanted if rel.startswith("agents/"))
    assert len(agents) == 4
    with zipfile.ZipFile(built_artifacts.wheel) as archive:
        for rel in agents:
            member = next(n for n in archive.namelist() if n.endswith(prefix + rel))
            assert archive.read(member) == wanted_bytes[rel], rel
            for line in archive.read(member).decode().split("\n---\n", 1)[0].splitlines():
                if line.startswith("tools:"):
                    slm = [t.strip() for t in line[6:].split(",") if not t.strip()[:1].isupper()]
                    assert slm and all(t.startswith("mcp__superlocalmemory__") for t in slm), (rel, line)


def test_wheel_contains_portable_kit_agents_rules(
    built_artifacts: BuiltArtifacts,
) -> None:
    """``slm connect codex`` must retain its AGENTS.md source after pip install."""
    names = wheel_names(built_artifacts.wheel)
    assert any(
        name.endswith("data/share/superlocalmemory/portable-kit/rules/AGENTS.md")
        for name in names
    )


def test_sdist_contains_root_and_optimize_legal_notices(
    built_artifacts: BuiltArtifacts,
) -> None:
    names = sdist_names(built_artifacts.sdist)
    prefix = _sdist_prefix(built_artifacts)
    assert f"{prefix}/LICENSE" in names
    assert f"{prefix}/NOTICE" in names
    assert f"{prefix}/src/superlocalmemory/optimize/NOTICE" in names


def test_wheel_excludes_local_state_tests_and_credentials(
    built_artifacts: BuiltArtifacts,
) -> None:
    names = wheel_names(built_artifacts.wheel)
    assert forbidden_wheel_names(names) == []


def test_wheel_metadata_and_console_entry_point(
    built_artifacts: BuiltArtifacts,
) -> None:
    metadata = _wheel_metadata(built_artifacts)
    normalized_filename = built_artifacts.wheel.name.replace("-", "_").lower()
    assert metadata["Name"].lower() == "superlocalmemory"
    assert metadata["Version"].replace("-", "_").lower() in normalized_filename
    assert set(metadata["Requires-Python"].split(",")) == {">=3.12", "<3.15"}

    with zipfile.ZipFile(built_artifacts.wheel) as archive:
        entry_names = [n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt")]
        assert len(entry_names) == 1
        entries = archive.read(entry_names[0]).decode("utf-8")
    assert "slm = superlocalmemory.cli.main:main" in entries

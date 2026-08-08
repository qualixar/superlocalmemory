"""Release contract for the intentionally narrowed V4 platform matrix."""

from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_v4_platform_boundary_is_explicit_in_package_and_install_docs() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    classifiers = set(project["classifiers"])
    dependencies = set(project["dependencies"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    getting_started = (ROOT / "docs" / "getting-started.md").read_text(
        encoding="utf-8"
    )

    assert "Operating System :: OS Independent" not in classifiers
    assert "cryptography==50.0.0" in dependencies

    required_statements = (
        "Apple Silicon macOS",
        "64-bit Windows",
        "64-bit Linux",
        "Intel Mac",
        "32-bit Windows",
    )
    for statement in required_statements:
        assert statement in readme
        assert statement in getting_started

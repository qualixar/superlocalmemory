"""Release guard for existing Cozo graph-store compatibility.

Every computer gets the 0.3.0 pin that existing graph stores were written with, except
64-bit ARM Linux: 0.3.0 has no wheel there (pip cannot install SLM at all), and the only
wheel that exists for it is the 0.7.6 series that 3.8.6 shipped to those computers.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
STORE_COMPATIBLE = "0.3.0"
LINUX_ARM64_WHEEL = "0.7.6"

# (sys_platform, platform_machine) as Python reports them.
ENVIRONMENTS = {
    "darwin-arm64": {"sys_platform": "darwin", "platform_machine": "arm64"},
    "darwin-x86_64": {"sys_platform": "darwin", "platform_machine": "x86_64"},
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "linux-aarch64": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "windows-amd64": {"sys_platform": "win32", "platform_machine": "AMD64"},
}


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _pycozo_pins(requirements: list[str], environment: dict[str, str]) -> list[str]:
    pins = []
    for text in requirements:
        req = Requirement(text)
        if req.name.lower() != "pycozo":
            continue
        if req.marker is None or req.marker.evaluate(environment):
            assert "embedded" in req.extras
            pins.append(str(req.specifier))
    return pins


@pytest.mark.parametrize("surface", ["dependencies", "cozo", "scale"])
@pytest.mark.parametrize("name", sorted(ENVIRONMENTS))
def test_each_computer_gets_exactly_one_store_compatible_cozo_pin(surface, name) -> None:
    project = _project()
    reqs = project["dependencies"] if surface == "dependencies" else project["optional-dependencies"][surface]
    want = LINUX_ARM64_WHEEL if name == "linux-aarch64" else STORE_COMPATIBLE
    assert _pycozo_pins(reqs, ENVIRONMENTS[name]) == [f"=={want}"]

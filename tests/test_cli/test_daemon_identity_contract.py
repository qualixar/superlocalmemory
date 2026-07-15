# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""V3.7 daemon namespace and instance-identity contract."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from superlocalmemory.infra.daemon_identity import (
    DAEMON_PROTOCOL,
    DAEMON_SERVICE,
    build_descriptor,
    canonical_data_root,
    clear_descriptor,
    descriptor_matches_health,
    namespace_id_for,
    read_descriptor,
    write_descriptor,
)


def test_canonical_data_root_uses_documented_alias_precedence(
    tmp_path: Path, monkeypatch,
) -> None:
    first = tmp_path / "data"
    second = tmp_path / "memory"
    third = tmp_path / "home"
    monkeypatch.setenv("SLM_DATA_DIR", str(first))
    monkeypatch.setenv("SL_MEMORY_PATH", str(second))
    monkeypatch.setenv("SLM_HOME", str(third))
    assert canonical_data_root() == first.resolve()

    monkeypatch.delenv("SLM_DATA_DIR")
    assert canonical_data_root() == second.resolve()

    monkeypatch.delenv("SL_MEMORY_PATH")
    assert canonical_data_root() == third.resolve()


def test_namespace_is_stable_for_same_root_and_distinct_across_roots(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "a" / ".." / "a"
    root_b = tmp_path / "b"
    assert namespace_id_for(root_a) == namespace_id_for(root_a.resolve())
    assert namespace_id_for(root_a) != namespace_id_for(root_b)


def test_descriptor_is_atomic_private_and_round_trips(tmp_path: Path) -> None:
    descriptor = build_descriptor(
        data_root=tmp_path,
        port=43123,
        version="3.7.0a1",
        pid=1234,
        instance_id="instance-a",
        capability="capability-a",
        state="starting",
    )
    path = write_descriptor(descriptor, data_root=tmp_path)
    loaded = read_descriptor(data_root=tmp_path)

    assert loaded == descriptor
    assert path == tmp_path / "daemon.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["service"] == DAEMON_SERVICE


def test_health_match_requires_full_namespace_instance_and_capability(
    tmp_path: Path,
) -> None:
    descriptor = build_descriptor(
        data_root=tmp_path,
        port=43123,
        version="3.7.0a1",
        pid=1234,
        instance_id="instance-a",
        capability="capability-a",
        state="ready",
    )
    health = descriptor.public_health_fields()

    assert health["daemon_protocol"] == DAEMON_PROTOCOL
    assert descriptor_matches_health(descriptor, health)

    for key, bad_value in (
        ("service", "foreign-service"),
        ("namespace_id", "foreign-namespace"),
        ("instance_id", "foreign-instance"),
        ("capability_fingerprint", "foreign-capability"),
        ("version", "foreign-version"),
        ("port", 43124),
        ("pid", 9999),
    ):
        poisoned = dict(health)
        poisoned[key] = bad_value
        assert not descriptor_matches_health(descriptor, poisoned), key


def test_malformed_descriptor_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "daemon.json").write_text("not-json")
    assert read_descriptor(data_root=tmp_path) is None


def test_shutdown_only_clears_its_own_instance(tmp_path: Path) -> None:
    descriptor = build_descriptor(
        data_root=tmp_path,
        port=43123,
        version="3.7.0a1",
        pid=os.getpid(),
        instance_id="replacement",
        capability="capability-a",
        state="ready",
    )
    path = write_descriptor(descriptor, data_root=tmp_path)

    assert not clear_descriptor("old-instance", data_root=tmp_path)
    assert path.exists()
    assert clear_descriptor("replacement", data_root=tmp_path)
    assert not path.exists()

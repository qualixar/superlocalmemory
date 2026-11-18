# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M-15: the mapping from origin signals to a class, and the order of classes."""

from __future__ import annotations

import itertools

import pytest

from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.trust import origin_class as oc
from superlocalmemory.trust.origin_class import Signals


def _cls(**kw) -> str:
    return oc.classify(Signals(**kw)).origin_class


# -- the lattice --------------------------------------------------------------------------------

def test_the_order_is_the_one_in_the_blueprint() -> None:
    assert oc.ORDER == ("user_direct", "user_confirmed", "system", "agent_inferred", "tool_output",
                        "mesh_peer", "web", "imported_file", "model_generated")
    assert oc.CLASSES == set(oc.ORDER) | {"unknown"}


@pytest.mark.parametrize("a,b", list(itertools.combinations_with_replacement(oc.ORDER, 2)))
def test_at_least_follows_the_order(a, b) -> None:
    assert oc.at_least(a, b) is (oc.ORDER.index(a) <= oc.ORDER.index(b))


def test_at_least_is_reflexive_and_unknown_is_incomparable() -> None:
    assert all(oc.at_least(c, c) for c in oc.ORDER)
    for c in oc.CLASSES:
        assert not oc.at_least("unknown", c) and not oc.at_least(c, "unknown")


def test_join_takes_the_lowest_class_and_unknown_absorbs() -> None:
    assert oc.join("user_direct", "web") == "web"
    assert oc.join("model_generated", "system") == "model_generated"
    assert oc.join("web", "web") == "web"
    assert oc.join("user_direct", "unknown") == "unknown"
    assert oc.join("unknown", "web") == "unknown"
    assert oc.join_all([]) == "user_direct"
    assert oc.join_all(["system", "tool_output", "agent_inferred"]) == "tool_output"


@pytest.mark.parametrize("a,b", list(itertools.product(sorted(oc.CLASSES), repeat=2)))
def test_join_is_commutative(a, b) -> None:
    assert oc.join(a, b) == oc.join(b, a)


def test_join_is_never_above_either_side() -> None:
    for a, b in itertools.product(oc.ORDER, repeat=2):
        j = oc.join(a, b)
        assert oc.at_least(a, j) and oc.at_least(b, j)


def test_every_class_has_plain_words() -> None:
    assert set(oc.LABELS) == oc.CLASSES
    assert oc.label("user_direct") == "you said this"
    assert oc.label("tool_output") == "a tool's output"
    assert oc.label("web") == "a web page"
    assert oc.label("model_generated") == "written by a model"
    assert oc.label("unknown") == "unknown (saved before 4.2.30)"
    assert oc.label("not-a-class") == oc.label("unknown")


# -- the mapping table --------------------------------------------------------------------------

def test_every_known_channel_has_a_row_and_none_falls_through() -> None:
    for channel in sorted(fo.CHANNELS):
        for transform in sorted(fo.TRANSFORMS):
            got = oc.classify(Signals(channel=channel, transform=transform, role="user"))
            assert got.origin_class in oc.CLASSES and not got.basis.endswith(":no_row"), channel


def test_a_channel_added_later_lands_on_unknown_never_on_a_trusted_class() -> None:
    got = oc.classify(Signals(channel="from-the-future", transform="verbatim"))
    assert got == oc.Classified("unknown", "v1:no_row")


def test_every_row_says_why_and_ids_are_unique() -> None:
    ids = [r.row_id for r in oc.ROWS]
    assert len(ids) == len(set(ids))
    assert all(len(r.why) > 20 for r in oc.ROWS)
    assert "user_confirmed" not in {r.origin_class for r in oc.ROWS}  # a person's approval only


@pytest.mark.parametrize("signals,expected,row", [
    (dict(channel="remember_http", transform="verbatim", role="user"), "user_direct", "remember.person"),
    (dict(channel="python_api", transform="verbatim", role="user"), "user_direct", "remember.person"),
    (dict(channel="remember_http", transform="verbatim", role="assistant", agent_id="claude"),
     "agent_inferred", "remember.agent"),
    (dict(channel="remember_http", transform="verbatim", role="user", agent_id="claude"),
     "agent_inferred", "remember.agent"),
    (dict(channel="remember_http", transform="verbatim", role="user", speaker="Alex"),
     "agent_inferred", "remember.agent"),
    (dict(channel="remember_http", transform="verbatim", role=None), "agent_inferred", "remember.agent"),
    (dict(channel="hook_file_change", transform="system_built"), "tool_output", "hook.file_change"),
    (dict(channel="hook_session_summary", transform="system_built"), "agent_inferred",
     "hook.session_summary"),
    (dict(channel="observe", transform="verbatim"), "agent_inferred", "observe.capture"),
    (dict(channel="cli_observe", transform="verbatim"), "agent_inferred", "observe.capture"),
    (dict(channel="mcp_offline", transform="verbatim"), "agent_inferred", "mcp.offline"),
    (dict(channel="ingest_adapter", transform="verbatim"), "web", "adapter.service"),
    (dict(channel="import", transform="copied"), "imported_file", "import.file"),
    (dict(channel="folder", transform="split"), "imported_file", "import.file"),
    (dict(channel="document", transform="excerpt"), "imported_file", "import.file"),
    (dict(channel="bundle", transform="copied"), "imported_file", "import.file"),
    (dict(channel="media", transform="verbatim"), "model_generated", "media.model"),
    (dict(channel="loop_ledger", transform="system_built"), "system", "loop.ledger"),
    (dict(channel="correction", transform="verbatim"), "agent_inferred", "correction.proposal"),
    (dict(channel="derived", transform="split"), "system", "derived.mechanical"),
    (dict(channel="legacy_replay", transform="verbatim"), "unknown", "origin.lost"),
    (dict(channel="restore", transform="verbatim"), "unknown", "origin.lost"),
    (dict(channel="repair", transform="verbatim"), "unknown", "origin.lost"),
    (dict(channel="unknown", transform="unknown"), "unknown", "origin.lost"),
])
def test_the_mapping_rows(signals, expected, row) -> None:
    got = oc.classify(Signals(**signals))
    assert got.origin_class == expected
    assert got.basis == f"v1:{row}"


@pytest.mark.parametrize("source_type,expected", [
    ("mesh-message", "mesh_peer"), ("web-clip", "web"), ("browser-capture", "web"),
    ("bash-output", "tool_output"), ("shell", "tool_output")])
def test_an_unnamed_channel_is_read_from_its_source_type(source_type, expected) -> None:
    for channel in ("unknown", "derived"):
        assert _cls(channel=channel, transform="verbatim", source_type=source_type) == expected


def test_a_named_channel_is_not_overridden_by_its_source_type() -> None:
    assert _cls(channel="remember_http", transform="verbatim", role="user",
                source_type="web-clip") == "user_direct"


# -- the transform only lowers -------------------------------------------------------------------

def test_a_model_rewrite_caps_at_model_generated() -> None:
    got = oc.classify(Signals(channel="remember_http", transform="model_rewritten", role="user"))
    assert got == oc.Classified("model_generated", "v1:remember.person+model_rewritten")
    assert _cls(channel="import", transform="model_rewritten") == "model_generated"
    assert _cls(channel="derived", transform="model_rewritten") == "model_generated"


def test_an_unknown_transform_is_no_better_than_an_agents_guess() -> None:
    assert _cls(channel="remember_http", transform="unknown", role="user") == "agent_inferred"
    assert _cls(channel="remember_http", transform="unknown", agent_id="a") == "agent_inferred"
    assert _cls(channel="ingest_adapter", transform="unknown") == "web"  # already lower: unchanged


def test_a_transform_never_raises_a_class_and_unknown_stays_unknown() -> None:
    for channel in fo.CHANNELS:
        base = _cls(channel=channel, transform="verbatim", role="user")
        for transform in fo.TRANSFORMS:
            got = _cls(channel=channel, transform=transform, role="user")
            if base == "unknown":
                assert got == "unknown"
            else:
                assert oc.at_least(base, got)

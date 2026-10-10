"""A folder's pages are saved with the scope text is saved with by default."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace


def test_folder_text_uses_the_configured_default_scope(env):
    cfg = SimpleNamespace(pii_redaction=False, scope=SimpleNamespace(default_scope="global"))
    env.host = dataclasses.replace(env.host, config=lambda: cfg)
    env.write("n.md", "a note about the folder")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.runtime.saved and {r["scope"] for r in env.runtime.saved} == {"global"}


def test_folder_text_is_personal_when_nothing_is_configured(env):
    env.write("n.md", "a note about the folder")
    env.scan(env.add_and_confirm())
    assert {r["scope"] for r in env.runtime.saved} == {"personal"}

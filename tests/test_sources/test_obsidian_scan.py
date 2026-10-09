"""Scanning an Obsidian vault end to end, and proof that a plain folder is unchanged."""

from __future__ import annotations

import json

from superlocalmemory import sources
from superlocalmemory.core.metadata_guard import strip_reserved_metadata
from superlocalmemory.sources.store import SourceStore


def vault(env):
    (env.root / ".obsidian").mkdir()
    (env.root / ".obsidian" / "app.json").write_text("{}")


def links(env, sid, rel=None):
    media = env.store()
    try:
        rows = media._read().execute(
            "SELECT from_relpath, target, link_kind, label FROM source_links WHERE source_id = ?", (sid,))
        return {tuple(r) for r in rows if rel is None or r[0] == rel}
    finally:
        media.close()


def test_auto_detect(env):
    assert sources.add_source(env.root, profile_id="default").kind == "folder"
    vault(env)
    assert sources.add_source(env.root, profile_id="default").kind == "obsidian"


def test_note_properties_tags_date_and_front_matter_removed(env):
    vault(env)
    env.write("n.md", "---\ntags: [alpha, '#beta']\naliases: [Nick]\ndate: 2024-03-05\nstatus: draft\n---\n# Title\ntext")
    sid = env.add_and_confirm()
    env.scan(sid)
    row = env.runtime.saved[0]
    assert row["content"] == "# Title\ntext"
    assert row["metadata"]["tags"] == "alpha,beta"
    assert row["session_date"] == "2024-03-05"
    props = row["metadata"]["_slm_properties"]
    assert props == {"aliases": ["Nick"], "properties": {"status": "draft"}}
    assert row["metadata"]["_slm_source"]["relpath"] == "n.md"
    assert "app.json" not in " ".join(r["metadata"]["_slm_source"]["relpath"] for r in env.runtime.saved)


def test_property_values_are_redacted():
    from superlocalmemory.sources.obsidian import NoteFields, prepared_extras

    fields = NoteFields(aliases=["x"], properties={"api": "key sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"})
    assert "abcdefghijklmnopqrstuvwxyz0123" not in json.dumps(prepared_extras(fields, None))


def test_property_values_follow_the_personal_data_setting(env, monkeypatch):
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    vault(env)
    env.write("n.md", "---\nowner: someone.else@example.com\n---\nbody")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert "someone.else@example.com" not in json.dumps(env.runtime.saved[0]["metadata"]["_slm_properties"])


def test_a_credential_in_front_matter_quarantines_the_note(env):
    vault(env)
    env.write("n.md", "---\napi: sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD\n---\nbody")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["n.md"]["state"] == "quarantined" and env.runtime.saved == []


def test_caller_cannot_forge_the_properties_key():
    assert "_slm_properties" not in strip_reserved_metadata({"_slm_properties": {"x": 1}, "ok": 1})


def test_links_recorded_replaced_and_embeds_resolved(env):
    vault(env)
    env.write("img/pic.png", b"\x89PNG not really")
    env.write("a.md", "see [[b]] and ![[pic.png]] and [x](sub/c.md)")
    sid = env.add_and_confirm()
    env.scan(sid)
    got = links(env, sid, "a.md")
    assert ("a.md", "b", "wikilink", None) in got
    assert ("a.md", "img/pic.png", "embed", None) in got
    assert ("a.md", "sub/c.md", "md_link", "x") in got
    import os
    path = env.root / "a.md"
    path.write_text("only [[z]]")
    os.utime(path, ns=(2_000_000_000_000_000_000,) * 2)
    env.scan(sid)
    assert links(env, sid, "a.md") == {("a.md", "z", "wikilink", None)}


def test_deleted_note_loses_its_links(env):
    vault(env)
    env.write("keep.md", "keep")
    path = env.write("gone.md", "[[x]]")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert links(env, sid, "gone.md")
    path.unlink()
    env.scan(sid)
    assert links(env, sid, "gone.md") == set()


def test_canvas_becomes_memory_and_links(env):
    vault(env)
    canvas = {"nodes": [{"id": "1", "type": "text", "text": "idea one"},
                        {"id": "2", "type": "file", "file": "n.md"}],
              "edges": [{"id": "e", "fromNode": "1", "toNode": "2"}]}
    env.write("board.canvas", json.dumps(canvas))
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.runtime.contents() == ["idea one"]
    assert env.files(sid)["board.canvas"]["state"] == "indexed"
    assert ("board.canvas", "n.md", "canvas_edge", "file") in links(env, sid)


def test_malformed_canvas_is_skipped(env):
    vault(env)
    env.write("bad.canvas", "{oops")
    env.write("ok.md", "fine")
    sid = env.add_and_confirm()
    env.scan(sid)
    row = env.files(sid)["bad.canvas"]
    assert row["state"] == "skipped" and row["reason"] == "canvas_unreadable"
    assert env.runtime.contents() == ["fine"]


def test_plain_folder_is_unchanged(env):
    env.write("n.md", "---\ntags: [alpha]\nstatus: x\n---\nbody [[b]]")
    env.write("board.canvas", json.dumps({"nodes": [{"id": "1", "type": "text", "text": "idea"}]}))
    sid = env.add_and_confirm()
    env.scan(sid)
    row = next(r for r in env.runtime.saved if "body" in r["content"])
    assert row["content"].startswith("---\ntags: [alpha]")
    assert "tags" not in row["metadata"] and "_slm_properties" not in row["metadata"]
    assert row["session_date"] == ""
    assert links(env, sid) == set()
    canvas = env.files(sid)["board.canvas"]
    assert canvas["state"] == "skipped" and canvas["reason"] == "canvas_not_supported"


def test_moved_note_keeps_its_links(env):
    import shutil

    vault(env)
    path = env.write("old/a.md", "to [[b]]")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "new").mkdir()
    shutil.move(str(path), str(env.root / "new" / "a.md"))
    env.scan(sid)
    assert links(env, sid) == {("new/a.md", "b", "wikilink", None)}

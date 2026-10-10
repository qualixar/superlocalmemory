"""Link extraction (recorded only), embed resolution by shortest path, and the canvas reader."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.sources.links import CanvasError, NameIndex, extract_links, parse_canvas


def kinds(found):
    return {(l.kind, l.target, l.label) for l in found}


def test_link_table():
    body = ("[[Plain]] [[Target|the label]] [[Page#Heading]] [[Page#^abc123]] ![[pic.png]] ![[Sized.png|300]]\n"
            "[md](notes/other.md) [enc](my%20note.md#part) [web](https://example.com/x) [abs](/etc/passwd) "
            "[out](../../outside.md) [mail](mailto:a@b.c) [frag](#top) ![img](x.png)")
    got = kinds(extract_links(body, "dir/note.md", None))
    assert ("wikilink", "Plain", None) in got
    assert ("wikilink", "Target", "the label") in got
    assert ("heading", "Page#Heading", None) in got
    assert ("block", "Page#^abc123", None) in got
    assert ("embed", "pic.png", None) in got
    assert ("embed", "Sized.png", "300") in got
    assert ("md_link", "dir/notes/other.md", "md") in got
    assert ("md_link", "dir/my note.md", "enc") in got
    targets = {t for _, t, _ in got}
    assert not any("example" in t or "passwd" in t or "outside" in t or "mailto" in t for t in targets)
    assert len([g for g in got if g[0] == "md_link"]) == 2


def test_links_in_code_are_ignored_and_count_is_bounded():
    body = "```\n[[inside fence]]\n```\n`[[inline]]` [[real]]"
    assert {l.target for l in extract_links(body, "a.md", None)} == {"real"}
    many = " ".join(f"[[n{i}]]" for i in range(5000))
    assert len(extract_links(many, "a.md", None)) <= 2000


def test_hostile_targets_are_dropped():
    found = extract_links("[[../../etc/passwd]] [[/abs]] [[ok]]", "a.md", None)
    assert {l.target for l in found} == {"ok"}


def test_embed_resolves_by_shortest_path_then_lexical():
    names = NameIndex(["z/deep/er/pic.png", "b/pic.png", "a/pic.png", "pic.png.bak", "Notes/Idea.md"])
    assert names.resolve("pic.png") == "a/pic.png"
    assert names.resolve("b/pic.png") == "b/pic.png"
    assert names.resolve("Idea") == "Notes/Idea.md"
    assert names.resolve("missing.png") is None
    found = extract_links("![[pic.png]] ![[gone.png]]", "x.md", names)
    assert {(l.kind, l.target) for l in found} == {("embed", "a/pic.png"), ("embed", "gone.png")}


def test_canvas_good():
    data = json.dumps({
        "nodes": [{"id": "1", "type": "text", "text": "first idea"},
                  {"id": "2", "type": "text", "text": "second idea"},
                  {"id": "3", "type": "file", "file": "notes/a.md"},
                  {"id": "4", "type": "file", "file": "../../secret.md"}],
        "edges": [{"id": "e", "fromNode": "1", "toNode": "3", "label": "see"},
                  {"id": "f", "fromNode": "1", "toNode": "2"}]}).encode()
    canvas = parse_canvas(data)
    assert canvas.text == "first idea\n\nsecond idea"
    got = kinds(canvas.links)
    assert ("canvas_edge", "notes/a.md", "file") in got
    assert ("canvas_edge", "node:1->notes/a.md", "see") in got
    assert ("canvas_edge", "node:1->node:2", None) in got
    assert not any("secret" in t for _, t, _ in got)


@pytest.mark.parametrize("data", [b"{not json", b"[]", b'{"nodes": 5}', b"\xff\xfe", b"[" * 100_000,
                                  b'{"nodes": []}' + b" " * (2 * 1024 * 1024 + 1)])
def test_canvas_bad(data):
    with pytest.raises(CanvasError):
        parse_canvas(data)

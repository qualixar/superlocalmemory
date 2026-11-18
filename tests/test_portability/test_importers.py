# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9: each importer reads its format's variants, ignores unknown fields and skips what it should."""

from __future__ import annotations

import io
import zipfile

import pytest

from superlocalmemory.portability import readers, registry
from superlocalmemory.portability.model import MEMORY, TURN, ImportFormatError, to_iso
from superlocalmemory.portability.readers import Doc
from tests.test_portability import fixtures as fx


def docs(*pairs):
    return [Doc(name, data if isinstance(data, bytes) else data.encode()) for name, data in pairs]


def texts(parsed):
    return [i.text for i in parsed.items]


# -- ChatGPT ------------------------------------------------------------------------------------

def test_chatgpt_follows_the_current_branch_and_skips_the_rest() -> None:
    parsed = registry.parse_any(docs(("conversations.json", fx.dumps(fx.chatgpt_list()))))
    assert parsed.format == "chatgpt" and parsed.conversations == 2
    assert texts(parsed) == [
        "I prefer short answers with one example.", "Understood, short answers with an example.",
        "Our deploy day is Thursday.", "Noted: deploys happen on Thursday."]
    assert [i.role for i in parsed.items] == ["user", "assistant", "user", "assistant"]
    assert all(i.kind == TURN and i.session_key == "conv-1" for i in parsed.items)
    assert {i.session_at for i in parsed.items} == {to_iso(1_700_000_000.0)}
    assert parsed.items[0].created_at == to_iso(1_700_000_010.0)
    s = parsed.skipped
    assert s["other_branch"] == 1 and s["system"] == 2 and s["tool"] == 1
    assert s["empty"] == 1 and s["empty_conversation"] == 1


def test_chatgpt_split_files_merge_and_a_repeated_conversation_counts_once() -> None:
    a, b = fx.chatgpt_conversation("c-a"), fx.chatgpt_conversation("c-b", 1_710_000_000.0)
    parsed = registry.parse_any(docs(
        ("conversations-000.json", fx.dumps([a])), ("conversations-001.json", fx.dumps([a, b])),
        ("user.json", fx.dumps({"id": "u"}))))
    assert parsed.conversations == 2 and len(parsed.items) == 8
    assert {i.session_key for i in parsed.items} == {"c-a", "c-b"}


def test_chatgpt_accepts_a_single_conversation_a_wrapper_and_a_zip() -> None:
    one = registry.parse_any(docs(("c.json", fx.dumps(fx.chatgpt_conversation()))))
    wrapped = registry.parse_any(docs(("c.json", fx.dumps({"conversations": fx.chatgpt_list()}))))
    assert len(one.items) == len(wrapped.items) == 4
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("export/conversations.json", fx.dumps(fx.chatgpt_list()))
        z.writestr("export/chat.html", "<html></html>")
    zipped = registry.parse_any(readers.from_bytes("export.zip", buf.getvalue()))
    assert texts(zipped) == texts(registry.parse_any(
        docs(("conversations.json", fx.dumps(fx.chatgpt_list())))))


def test_chatgpt_without_current_node_takes_the_newest_leaf_and_survives_a_loop() -> None:
    conv = fx.chatgpt_conversation()
    del conv["current_node"]
    assert len(registry.parse_any(docs(("c.json", fx.dumps([conv]))) ).items) >= 3
    loop = fx.chatgpt_conversation()
    loop["mapping"]["q1"]["parent"] = "a2"  # a cycle: never an endless walk
    registry.parse_any(docs(("c.json", fx.dumps([loop]))))


def test_chatgpt_ignores_unknown_fields_and_non_text_parts() -> None:
    conv = fx.chatgpt_conversation()
    conv["brand_new_field"] = {"x": [1, 2]}
    conv["mapping"]["q1"]["message"]["content"]["parts"] = [
        {"asset_pointer": "file-service://x"}, "I prefer short answers with one example."]
    conv["mapping"]["q1"]["message"]["extra"] = {"anything": True}
    parsed = registry.parse_any(docs(("c.json", fx.dumps([conv]))))
    assert parsed.items[0].text == "I prefer short answers with one example."


def test_a_very_long_turn_is_cut_at_paragraphs() -> None:
    conv = fx.chatgpt_conversation()
    conv["mapping"]["q1"]["message"]["content"]["parts"] = ["\n\n".join(["word " * 400] * 10)]
    parsed = registry.parse_any(docs(("c.json", fx.dumps([conv]))))
    pieces = [i for i in parsed.items if i.role == "user" and i.ref.startswith("q1")]
    assert len(pieces) > 1 and all(len(p.text) <= 12_000 for p in pieces)


# -- Claude -------------------------------------------------------------------------------------

def test_claude_reads_messages_and_maps_human_to_user() -> None:
    parsed = registry.parse_any(docs(("conversations.json", fx.dumps(fx.claude_list()))))
    assert parsed.format == "claude" and parsed.conversations == 2
    assert texts(parsed) == ["My team meets on Mondays.", "Monday meetings noted."]
    assert [i.role for i in parsed.items] == ["user", "assistant"]
    assert parsed.items[0].session_at == "2026-01-15T10:30:00+00:00"
    assert parsed.skipped["tool"] == 1 and parsed.skipped["empty"] == 1
    assert parsed.skipped["empty_conversation"] == 1


def test_claude_variants_wrapper_single_and_messages_role() -> None:
    conv = fx.claude_list()[0]
    for payload in ({"conversations": [conv]}, conv):
        assert len(registry.parse_any(docs(("c.json", fx.dumps(payload)))).items) == 2
    alt = {"uuid": "x", "name": "n", "messages": [{"role": "user", "content": "plain string"}]}
    parsed = registry.parse_any(docs(("c.json", fx.dumps([alt]))))
    assert texts(parsed) == ["plain string"] and parsed.items[0].role == "user"


def test_claude_memories_json_becomes_memory_entries() -> None:
    memories = [{"conversations_memory": "- Likes plain words.\n- Works in Python.",
                 "project_memories": {"p-1": "Project uses a weekly release.", "p-2": 7},
                 "account_uuid": "a"}]
    parsed = registry.parse_any(docs(("conversations.json", fx.dumps(fx.claude_list())),
                                     ("memories.json", fx.dumps(memories))))
    mem = [i for i in parsed.items if i.kind == MEMORY]
    assert [m.text for m in mem] == ["Likes plain words.", "Works in Python.",
                                     "Project uses a weekly release."]
    assert parsed.skipped["unsupported"] == 1


def test_claude_memory_text_dates_and_entries() -> None:
    parsed = registry.parse_any(docs(("memory.txt", fx.CLAUDE_MEMORY_TXT)))
    assert parsed.format == "claude-memory"
    assert texts(parsed) == [
        "Runs a small research team on reliability.", "Prefers short answers.",
        "A paragraph about tone: direct and calm, continued on a second line."]
    assert parsed.items[0].created_at == "2026-01-10T00:00:00+00:00"


# -- Mem0 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["list", "results", "envelope", "memories", "pages"])
def test_mem0_accepts_every_shape(shape) -> None:
    items = fx.MEM0_ITEMS
    payloads = {"list": items, "results": {"results": items},
                "envelope": {"count": 5, "next": None, "previous": None, "results": items},
                "memories": {"memories": items}}
    if shape == "pages":
        parsed = registry.parse_any(docs(
            ("p1.json", fx.dumps({"count": 5, "next": "x", "results": items[:2]})),
            ("p2.json", fx.dumps({"count": 5, "next": None, "results": items[2:]}))))
    else:
        parsed = registry.parse_any(docs(("m.json", fx.dumps(payloads[shape]))))
    assert parsed.format == "mem0"
    assert texts(parsed) == ["Prefers short answers.", "Deploys on Thursdays."]
    assert parsed.items[0].created_at == "2026-02-01T09:00:00+00:00"
    assert parsed.skipped == {"empty": 1, "expired": 1, "replaced": 1}
    assert all(i.kind == MEMORY and not i.role for i in parsed.items)


def test_mem0_old_text_keys_need_the_format_named_and_duplicates_count_once() -> None:
    old = [{"id": "a", "text": "Old style text."}, {"id": "a", "text": "Old style text."}]
    with pytest.raises(ImportFormatError):
        registry.parse_any(docs(("m.json", fx.dumps(old))))
    parsed = registry.parse_any(docs(("m.json", fx.dumps(old))), "mem0")
    assert texts(parsed) == ["Old style text."] and parsed.skipped == {"duplicate": 1}


# -- Letta --------------------------------------------------------------------------------------

def test_letta_blocks_and_messages_and_the_archival_note() -> None:
    parsed = registry.parse_any(docs(("agent.af", fx.dumps(fx.letta_file()))))
    assert parsed.format == "letta"
    blocks = [i for i in parsed.items if i.kind == MEMORY]
    assert [b.text for b in blocks] == ["The user keeps a weekly review on Fridays.",
                                        "The agent answers in plain words."]
    turns = [i for i in parsed.items if i.kind == TURN]
    assert [(t.role, t.text) for t in turns] == [
        ("user", "Remind me about the review."), ("assistant", "Your review is on Friday.")]
    assert turns[0].session_at == "2026-03-01T08:00:00+00:00" and parsed.conversations == 1
    assert parsed.skipped["system"] == 1 and parsed.skipped["tool"] == 1
    assert parsed.skipped["duplicate"] == 1 and parsed.skipped["empty"] == 1
    assert any("archival" in n.lower() for n in parsed.notes)


def test_letta_accepts_a_bare_array_of_blocks() -> None:
    parsed = registry.parse_any(docs(("blocks.json", fx.dumps(fx.letta_file()["blocks"]))))
    assert parsed.format == "letta" and len(parsed.items) == 2


# -- OKF and Markdown ---------------------------------------------------------------------------

def test_okf_concepts_one_memory_per_section_and_reserved_files_skipped() -> None:
    parsed = registry.parse_any(docs(("kb/release.md", fx.OKF_CONCEPT), ("kb/ops/backup.md", fx.OKF_PLAIN),
                                     ("kb/index.md", fx.OKF_INDEX), ("kb/log.md", "## 2026-01-01\nx")))
    assert parsed.format == "okf"
    assert texts(parsed) == [
        "Release checklist\nRun the tests before every release.",
        "Rollback\nKeep the previous build for one week.", "Backups run nightly at two."]
    assert parsed.skipped["index"] == 2 and all(i.kind == MEMORY for i in parsed.items)


def test_okf_needs_a_type_line() -> None:
    mixed = docs(("a.md", fx.OKF_PLAIN), ("b.md", "---\ntitle: no type\n---\nBody."),
                 ("c.md", fx.OKF_CONCEPT))
    parsed = registry.parse_any(mixed)
    assert parsed.format == "okf" and parsed.skipped["not_okf"] == 1


def test_plain_markdown_sections_keep_their_heading_path() -> None:
    parsed = registry.parse_any(docs(("AGENTS.md", fx.AGENTS_MD)))
    assert parsed.format == "markdown"
    assert texts(parsed) == [
        "Project rules\nUse small functions.", "Project rules > Tests\nRun the suite before commit.",
        "Project rules > Tests > Slow tests\nMark them slow.\n\n```\n# not a heading\n```"]


def test_a_memory_index_is_skipped_and_its_topic_files_import() -> None:
    parsed = registry.parse_any(docs(("MEMORY.md", fx.MEMORY_INDEX), ("feedback.md", fx.TOPIC_MD)))
    assert parsed.format == "markdown"
    assert texts(parsed) == ["Keep replies short and direct."] and parsed.skipped == {"index": 1}


def test_a_long_section_is_cut_and_front_matter_is_not_text() -> None:
    body = "# Notes\n" + "\n\n".join(["sentence " * 150] * 8)
    parsed = registry.parse_any(docs(("n.md", "---\nauthor: x\n---\n" + body)))
    assert len(parsed.items) > 1 and all(len(i.text) <= 4_100 for i in parsed.items)
    assert not any("author" in t for t in texts(parsed))


# -- refusal, forcing, fingerprints, intake -----------------------------------------------------

def test_a_file_in_no_known_format_is_refused_naming_every_format() -> None:
    with pytest.raises(ImportFormatError) as err:
        registry.parse_any(docs(("data.json", '{"hello": "world"}')))
    message = str(err.value)
    for name in ("ChatGPT", "Claude", "Mem0", "Letta", "OKF", "Markdown"):
        assert name in message
    with pytest.raises(ImportFormatError, match="not a format I know"):
        registry.parse_any(docs(("a.md", "# x")), "notion")


def test_a_recognised_file_with_nothing_to_import_says_so() -> None:
    with pytest.raises(ImportFormatError, match="no conversations"):
        registry.parse_any(docs(("c.json", fx.dumps([fx.chatgpt_empty()]))))


def test_the_format_can_be_named() -> None:
    parsed = registry.parse_any(docs(("a.md", "- one\n- two")), "claude-memory")
    assert texts(parsed) == ["one", "two"]


def test_the_same_item_has_the_same_fingerprint_and_a_changed_one_does_not() -> None:
    a = registry.parse_any(docs(("c.json", fx.dumps(fx.chatgpt_list())))).items
    b = registry.parse_any(docs(("c.json", fx.dumps(fx.chatgpt_list())))).items
    assert [i.fingerprint("chatgpt") for i in a] == [i.fingerprint("chatgpt") for i in b]
    assert len({i.fingerprint("chatgpt") for i in a}) == len(a)
    assert a[0].fingerprint("chatgpt") != a[0].fingerprint("claude")


@pytest.fixture(autouse=True)
def _own_data_elsewhere(tmp_path_factory, monkeypatch):
    """The data folder is a sibling of the test's files, so neither contains the other."""
    home = tmp_path_factory.mktemp("slm-home")
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    return home


def test_intake_limits(tmp_path) -> None:
    with pytest.raises(readers.IntakeRefused, match="larger than"):
        readers.from_bytes("a.json", b"x" * 20, limit=10)
    with pytest.raises(readers.IntakeRefused, match="empty"):
        readers.from_bytes("a.json", b"")
    with pytest.raises(readers.IntakeRefused, match="could not be opened"):
        readers.from_bytes("a.zip", b"PK\x03\x04not a zip")
    folder = tmp_path / "kb"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.md").write_text("# A\nbody text here")
    (folder / "sub" / "b.md").write_text("# B\nmore body text")
    (folder / "pic.png").write_bytes(b"\x89PNG")
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "c.md").write_text("# C\nnot read")
    got = readers.from_path(folder)
    assert sorted(d.name for d in got) == ["a.md", "sub/b.md"]
    with pytest.raises(readers.IntakeRefused, match="not found"):
        readers.from_path(tmp_path / "missing")


def test_intake_refuses_links_zip_bombs_and_own_data(tmp_path, monkeypatch, _own_data_elsewhere) -> None:
    real = tmp_path / "real.md"
    real.write_text("# A\nbody")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    with pytest.raises(readers.IntakeRefused, match="link"):
        readers.from_path(link)
    from superlocalmemory.portability import constants as C
    monkeypatch.setattr(C, "MAX_UNPACKED_BYTES", 100)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.json", "[" + "0," * 500 + "0]")
    with pytest.raises(readers.IntakeRefused, match="unpacks to more"):
        readers.from_bytes("x.zip", buf.getvalue())
    with pytest.raises(readers.IntakeRefused, match="own data"):
        readers.from_path(_own_data_elsewhere)


def test_times_in_seconds_milliseconds_and_text_are_all_read() -> None:
    assert to_iso(1_700_000_000) == to_iso(1_700_000_000_000) == to_iso("1700000000") \
        == "2023-11-14T22:13:20+00:00"
    assert to_iso("2026-01-15T10:30:00.000Z") == to_iso("2026-01-15T11:30:00+01:00") \
        == "2026-01-15T10:30:00+00:00"
    assert to_iso("2026-01-15") == "2026-01-15T00:00:00+00:00"  # a bare date is midnight UTC
    for nothing in (None, "", "yesterday", True, float("nan")):
        assert to_iso(nothing) == ""

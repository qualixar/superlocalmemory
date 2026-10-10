"""The minimal front-matter parser: the supported subset, and hostile input that must stay inert."""

from __future__ import annotations

import time

from superlocalmemory.sources.obsidian import note_fields, split_front_matter


def test_scalars_inline_and_block_lists():
    text = ("---\ntitle: My note\ntags: [alpha, \"#beta\", 'gamma']\naliases:\n  - First\n  - Second\n"
            "date: 2024-03-05\n---\nBody here\n")
    props, body = split_front_matter(text)
    assert body == "Body here\n"
    assert props["title"] == "My note"
    assert props["tags"] == ["alpha", "#beta", "gamma"]
    assert props["aliases"] == ["First", "Second"]
    assert props["date"] == "2024-03-05"


def test_no_front_matter_is_plain_text():
    for text in ("Just text\n---\nmore\n---\n", "\n---\na: b\n---\nx", "---\nnot a mapping\n---\nx",
                 "---\na: b\nnever closed\n"):
        props, body = split_front_matter(text)
        assert props is None and body == text


def test_bom_and_crlf_are_accepted():
    props, body = split_front_matter("\ufeff---\r\ntags: a\r\n---\r\nBody")
    assert props == {"tags": "a"} and body == "Body"


def test_closing_dots_and_empty_block():
    assert split_front_matter("---\na: 1\n...\nx")[1] == "x"
    props, body = split_front_matter("---\n---\nx")
    assert props == {} and body == "x"


def test_mapping_to_fields():
    props, _ = split_front_matter(
        "---\ntags: [\"#one\", two, one]\ntag: three four\naliases: [A, B]\ncreated: 2023-01-02T10:00:00\n"
        "status: draft\nowner: someone\n---\nx")
    f = note_fields(props)
    assert f.tags == ["one", "two", "three", "four"]
    assert f.aliases == ["A", "B"]
    assert f.session_date == "2023-01-02"
    assert f.properties == {"status": "draft", "owner": "someone"}


def test_date_wins_over_created_and_bad_dates_are_ignored():
    f = note_fields({"date": "2024-02-30", "created": "2022-05-06"})
    assert f.session_date == "2022-05-06"
    assert note_fields({"date": "yesterday"}).session_date == ""


def test_limits():
    props = {"tags": [f"t{i}" for i in range(50)] + ["x" * 100], "aliases": ["a"] * 100}
    props.update({f"k{i}": "v" * 500 for i in range(60)})
    f = note_fields(props)
    assert len(f.tags) == 20 and all(len(t) <= 64 for t in f.tags)
    assert len(f.properties) == 30 and all(len(v) <= 200 for v in f.properties.values())
    assert len(f.aliases) <= 50


def test_hostile_one_megabyte_value_is_not_front_matter_and_fast():
    text = "---\nkey: " + "A" * (1 << 20) + "\n---\nbody"
    start = time.monotonic()
    props, body = split_front_matter(text)
    assert props is None and body == text
    assert time.monotonic() - start < 1.0


def test_hostile_anchors_and_tags_stay_text():
    text = ("---\na: &x value\nb: *x\nc: !!python/object/apply:os.system [\"echo hi\"]\n"
            "d: !!python/object:os.system\nlist: &l [1, 2]\nmerge: <<: *x\n---\nbody")
    props, body = split_front_matter(text)
    assert body == "body"
    assert props["a"] == "&x value" and props["b"] == "*x"
    assert props["c"].startswith("!!python/object/apply")
    f = note_fields(props)
    assert all(isinstance(v, str) for v in f.properties.values())


def test_many_keys_and_deep_nesting_are_bounded():
    lines = "".join(f"k{i}: v\n" for i in range(1000))
    props, _ = split_front_matter("---\n" + lines + "---\nx")
    assert len(note_fields(props or {}).properties) == 30
    nested = "---\na:\n" + "".join("  " * i + "b:\n" for i in range(1, 300)) + "---\nx"
    split_front_matter(nested)  # does not raise, does not recurse


def test_block_scalar_indicators_are_kept_as_text():
    props, _ = split_front_matter("---\nsummary: |\n  line one\n  line two\nnext: ok\n---\nx")
    assert props["next"] == "ok" and props["summary"] == "|"

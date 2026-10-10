"""Which model makes picture vectors, and which model makes the picture query."""

from __future__ import annotations

import pytest

from superlocalmemory.runtimes import space_plan as sp

SEP = ("google/embeddinggemma-2", "", 768)
NOMIC = "nomic-ai/nomic-embed-text-v1.5"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(sp.MODE_ENV, raising=False)


def plan(text=NOMIC, dim=768, **kw):
    return sp.resolve_space_plan(text, dim, separate_model=SEP, **kw)


def test_default_is_separate_and_nothing_changes():
    p = plan()
    assert sp.DEFAULT_SPACE_MODE == "separate"
    assert (p.mode, p.image_model, p.dim, p.text_model, p.query_from_text) == (
        "separate", SEP[0], 768, "", False)
    assert p.reason == "default"


def test_env_selects_paired(monkeypatch):
    monkeypatch.setenv(sp.MODE_ENV, "paired")
    p = plan()
    assert p.mode == "paired" and p.query_from_text is True
    assert p.image_model == "nomic-ai/nomic-embed-vision-v1.5" and p.text_model == NOMIC and p.dim == 768


def test_explicit_request_beats_env(monkeypatch):
    monkeypatch.setenv(sp.MODE_ENV, "paired")
    assert plan(requested="separate").mode == "separate"


def test_invalid_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(sp.MODE_ENV, "bogus")
    assert plan().mode == "separate"


def test_unknown_text_model_falls_back_with_a_reason():
    p = plan("some/other-model", requested="paired")
    assert p.mode == "separate" and "no paired image model" in p.reason and p.image_model == SEP[0]


def test_dim_mismatch_falls_back():
    p = plan(NOMIC, 384, requested="paired")
    assert p.mode == "separate" and "width" in p.reason


def test_org_prefix_is_not_required():
    assert plan("nomic-embed-text-v1.5", requested="paired").mode == "paired"


def test_single_is_refused():
    with pytest.raises(ValueError, match="not available in this build"):
        plan(requested="single")


def test_single_from_env_is_refused(monkeypatch):
    monkeypatch.setenv(sp.MODE_ENV, "single")
    with pytest.raises(ValueError, match="not available in this build"):
        plan()


def test_signature_has_the_five_fields():
    assert plan(requested="paired").signature() == {
        "mode": "paired", "image_model": "nomic-ai/nomic-embed-vision-v1.5", "image_revision": "",
        "dim": 768, "text_model": NOMIC}


def test_compatible_matrix():
    paired, separate = plan(requested="paired"), plan(requested="separate")
    assert sp.compatible(paired, None) and sp.compatible(separate, None)
    assert sp.compatible(paired, paired.signature())
    assert sp.compatible(separate, separate.signature())
    assert not sp.compatible(paired, separate.signature())
    assert not sp.compatible(separate, paired.signature())
    for key, other in (("text_model", "x/y"), ("image_model", "x/y"), ("image_revision", "r9"), ("dim", 384)):
        assert not sp.compatible(paired, {**paired.signature(), key: other})


def test_current_plan_reads_the_live_text_space(tmp_path, monkeypatch):
    import json
    import sqlite3

    (tmp_path / "config.json").write_text(json.dumps({"embedding_signature": "some/other::384"}))
    monkeypatch.setenv(sp.MODE_ENV, "paired")
    p = sp.current_space_plan(tmp_path)
    assert p.mode == "separate" and "no paired image model" in p.reason

    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute("CREATE TABLE embedding_space (id INTEGER PRIMARY KEY, live_signature TEXT, live_config TEXT)")
    conn.execute("INSERT INTO embedding_space VALUES (1, ?, '{}')", (f"{NOMIC}::768",))
    conn.commit()
    conn.close()
    assert sp.current_space_plan(tmp_path).mode == "paired"


def test_current_plan_with_nothing_on_disk_uses_the_shipped_text_model(tmp_path, monkeypatch):
    monkeypatch.setenv(sp.MODE_ENV, "paired")
    assert sp.current_space_plan(tmp_path).mode == "paired"


# -- picture evidence floor -------------------------------------------------------

FLOOR_ENV = "SLM_MEDIA_PAIRED_MIN_SCORE"


def test_paired_plan_carries_its_own_floor_and_separate_has_none(monkeypatch):
    monkeypatch.delenv(FLOOR_ENV, raising=False)
    assert sp.PAIRED_MIN_SCORE == 0.05
    assert plan(requested="paired").min_score == 0.05
    assert plan(requested="separate").min_score is None
    assert plan(requested="paired", text="other/model").min_score is None  # fell back to separate


def test_floor_env_override_valid_and_invalid(monkeypatch):
    monkeypatch.setenv(FLOOR_ENV, "0.12")
    assert plan(requested="paired").min_score == 0.12
    monkeypatch.setenv(FLOOR_ENV, "0")
    assert plan(requested="paired").min_score == 0.0
    for bad in ("", "abc", "-0.1", "1.5", "nan"):
        monkeypatch.setenv(FLOOR_ENV, bad)
        assert plan(requested="paired").min_score == sp.PAIRED_MIN_SCORE
    monkeypatch.setenv(FLOOR_ENV, "0.9")
    assert plan(requested="separate").min_score is None


def test_floor_is_not_part_of_the_space_identity(monkeypatch):
    monkeypatch.delenv(FLOOR_ENV, raising=False)
    a = plan(requested="paired").signature()
    monkeypatch.setenv(FLOOR_ENV, "0.2")
    assert plan(requested="paired").signature() == a
    assert "min_score" not in a

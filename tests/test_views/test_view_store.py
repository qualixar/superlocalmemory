# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saved views: validation, profile scope and the stored record (issue #113)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.views import ViewError, ViewStore, model, recall_arguments, shape_run

from ._store import learning_db


@pytest.fixture()
def store(tmp_path) -> ViewStore:
    return ViewStore(learning_db(tmp_path))


def _code(fn, *args, **kwargs) -> str:
    with pytest.raises(ViewError) as info:
        fn(*args, **kwargs)
    return info.value.code


class TestCreateListShow:
    def test_round_trip(self, store) -> None:
        made = store.create("default", name="  Work   log ", query=" what did I ship ",
                            filters={"kind": "decision", "window": "7d"}, limit=5)
        assert made.name == "Work log" and made.query == "what did I ship"
        assert made.filter_map == {"kind": "decision", "window": "7d"}
        assert store.get("default", "work LOG") == made
        assert store.list("default") == (made,)

    def test_blank_filters_are_dropped_and_defaults_apply(self, store) -> None:
        made = store.create("default", name="a", query="q", filters={"kind": " ", "window": ""})
        assert made.filters == () and made.limit == model.DEFAULT_LIMIT

    def test_list_order_is_by_name_and_stable(self, store) -> None:
        for name in ("beta", "Alpha", "gamma"):
            store.create("default", name=name, query="q")
        assert [v.name for v in store.list("default")] == ["Alpha", "beta", "gamma"]
        assert store.list("default") == store.list("default")

    def test_a_duplicate_name_is_refused_case_insensitively(self, store) -> None:
        store.create("default", name="Work", query="q")
        assert _code(store.create, "default", name="WORK", query="other") == model.VIEW_EXISTS

    def test_the_profile_cap(self, store, monkeypatch) -> None:
        monkeypatch.setattr("superlocalmemory.views.store.MAX_VIEWS_PER_PROFILE", 2)
        store.create("default", name="a", query="q")
        store.create("default", name="b", query="q")
        assert _code(store.create, "default", name="c", query="q") == model.TOO_MANY_VIEWS
        store.create("other", name="c", query="q")       # per profile, not global


class TestValidation:
    @pytest.mark.parametrize("name", ["", "   ", "x" * 81, "bad\x00name", "rtl‮name", 7])
    def test_bad_names(self, store, name) -> None:
        assert _code(store.create, "default", name=name, query="q") == model.INVALID_NAME

    @pytest.mark.parametrize("query", ["", "  ", "q" * 1001, "a\x07b", None])
    def test_bad_queries(self, store, query) -> None:
        assert _code(store.create, "default", name="n", query=query) == model.INVALID_QUERY

    @pytest.mark.parametrize("limit", [0, 51, -1, 2.5, "5", True])
    def test_bad_limits(self, store, limit) -> None:
        assert _code(store.create, "default", name="n", query="q", limit=limit) \
            == model.INVALID_LIMIT

    def test_an_unknown_filter_is_refused_not_dropped(self, store) -> None:
        assert _code(store.create, "default", name="n", query="q",
                     filters={"project": "x"}) == model.UNKNOWN_FILTER

    @pytest.mark.parametrize("filters", [{"kind": "nonsense"}, {"window": "fortnight"},
                                         {"as_of": "yesterday-ish"}, {"kind": 3}, ["kind"]])
    def test_unreadable_filters(self, store, filters) -> None:
        assert _code(store.create, "default", name="n", query="q",
                     filters=filters) == model.INVALID_FILTER

    def test_as_of_is_normalized(self, store) -> None:
        made = store.create("default", name="n", query="q",
                            filters={"as_of": "2026-01-01T00:00:00Z"})
        assert made.filter_map["as_of"].startswith("2026-01-01T00:00:00")

    def test_nothing_is_written_on_a_refusal(self, store) -> None:
        _code(store.create, "default", name="n", query="q", filters={"kind": "nonsense"})
        assert store.list("default") == ()


class TestProfileScope:
    def test_another_profile_never_sees_or_touches_a_view(self, store) -> None:
        store.create("alice", name="Secret plan", query="acquisition")
        assert store.list("bob") == ()
        assert _code(store.get, "bob", "Secret plan") == model.VIEW_NOT_FOUND
        assert _code(store.rename, "bob", "Secret plan", "mine") == model.VIEW_NOT_FOUND
        assert _code(store.delete, "bob", "Secret plan") == model.VIEW_NOT_FOUND
        assert [v.name for v in store.list("alice")] == ["Secret plan"]

    def test_the_same_name_in_two_profiles_is_two_views(self, store) -> None:
        a = store.create("alice", name="Work", query="qa")
        b = store.create("bob", name="Work", query="qb")
        assert a.view_id != b.view_id
        store.delete("alice", "Work")
        assert store.get("bob", "Work") == b

    def test_erase_and_export_are_per_profile(self, store) -> None:
        store.create("alice", name="a1", query="q")
        store.create("alice", name="a2", query="q")
        store.create("bob", name="b1", query="q")
        assert [v["name"] for v in store.export_profile("alice")] == ["a1", "a2"]
        assert store.erase_profile("alice") == 2
        assert store.list("alice") == () and len(store.list("bob")) == 1

    @pytest.mark.parametrize("profile", ["", "  ", None, 3])
    def test_a_profile_is_required(self, store, profile) -> None:
        with pytest.raises(ValueError):
            store.list(profile)


class TestRenameDelete:
    def test_rename_keeps_identity_and_query(self, store) -> None:
        made = store.create("default", name="old", query="q", filters={"window": "30d"})
        renamed = store.rename("default", "OLD", "new")
        assert (renamed.view_id, renamed.query, renamed.filters) == \
               (made.view_id, made.query, made.filters)
        assert store.get("default", "new").name == "new"
        assert _code(store.get, "default", "old") == model.VIEW_NOT_FOUND

    def test_rename_to_a_taken_name_is_refused(self, store) -> None:
        store.create("default", name="a", query="q")
        store.create("default", name="b", query="q")
        assert _code(store.rename, "default", "a", "B") == model.VIEW_EXISTS

    def test_rename_changing_only_case_is_allowed(self, store) -> None:
        store.create("default", name="work", query="q")
        assert store.rename("default", "work", "Work").name == "Work"

    def test_delete(self, store) -> None:
        store.create("default", name="a", query="q")
        assert store.delete("default", "a").name == "a"
        assert store.list("default") == ()


class TestUnavailable:
    def test_a_store_without_the_table_says_so(self, tmp_path) -> None:
        path = tmp_path / "learning.db"
        sqlite3.connect(str(path)).close()
        store = ViewStore(path)
        assert _code(store.list, "default") == model.VIEWS_UNAVAILABLE
        assert store.erase_profile("default") == 0

    def test_a_missing_file_is_not_created(self, tmp_path) -> None:
        store = ViewStore(tmp_path / "learning.db")
        assert _code(store.list, "default") == model.VIEWS_UNAVAILABLE
        assert not (tmp_path / "learning.db").exists()


class TestRunner:
    def test_only_set_filters_reach_recall(self, store) -> None:
        bare = store.create("default", name="a", query="q", limit=3)
        assert recall_arguments(bare) == {"query": "q", "limit": 3}
        full = store.create("default", name="b", query="q",
                            filters={"kind": "decision", "window": "7d"})
        assert recall_arguments(full) == {"query": "q", "limit": 10,
                                          "kind": "decision", "window": "7d"}

    def test_shape_keeps_recalls_order_and_ids(self, store) -> None:
        view = store.create("default", name="a", query="q")
        response = {"results": [
            {"fact_id": "f2", "memory_id": "m2", "content": "two", "score": 0.9,
             "channel_scores": {"x": 1}},
            {"fact_id": "f1", "memory_id": "m1", "content": "one", "score": 0.4},
        ], "no_confident_match": False, "query_type": "semantic", "profile": "default",
           "unrelated": "dropped"}
        run = shape_run(view, response)
        assert run["result_ids"] == ["f2", "f1"]
        assert [r["rank"] for r in run["results"]] == [1, 2]
        assert run["results"][0]["memory_id"] == "m2"
        assert "channel_scores" not in run["results"][0] and "unrelated" not in run
        assert run["no_confident_match"] is False and run["view"]["name"] == "a"


class TestTheIndexIsTheLastWord:
    def test_a_race_past_the_check_is_still_a_clean_refusal(self, store, monkeypatch) -> None:
        """If a name check is ever bypassed, the unique index refuses, as a
        ``view_exists`` refusal rather than a raw database error."""
        store.create("default", name="Work", query="q")
        monkeypatch.setattr(ViewStore, "_find", staticmethod(lambda *a, **k: None))
        assert _code(store.create, "default", name="work", query="q") == model.VIEW_EXISTS

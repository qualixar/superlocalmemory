# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One rule for "is this the same tag" (4.1.22)."""

from __future__ import annotations

import unicodedata

import pytest

from superlocalmemory.core.tag_identity import (
    MAX_TAG_CHARS,
    parse_tag_values,
    tag_key,
    tag_keys,
)

# -- tag_key -----------------------------------------------------------------

@pytest.mark.parametrize("value", [
    "token-optimization", "Token-Optimization", " token-optimization ",
    "TOKEN-OPTIMIZATION", "token-optimization\n",
])
def test_case_and_whitespace_around_a_label_agree(value) -> None:
    assert tag_key(value) == "token-optimization"


def test_internal_whitespace_collapses_to_one_space() -> None:
    assert tag_key("token   optimization") == "token optimization"
    assert tag_key("token\toptimization") == "token optimization"
    assert tag_key("token\n optimization") == "token optimization"


def test_punctuation_is_kept() -> None:
    assert tag_key("v4.1.21") == "v4.1.21"
    assert tag_key("token-optimization") == "token-optimization"
    assert tag_key("#149") == "#149"


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n", 42, [], {"a": 1}])
def test_values_that_name_no_tag(value) -> None:
    assert tag_key(value) is None


def test_decomposed_unicode_matches_composed() -> None:
    composed = "café"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert tag_key(decomposed) == tag_key(composed.upper())


def test_long_label_is_bounded() -> None:
    assert len(tag_key("x" * 5000)) <= MAX_TAG_CHARS


# -- parse_tag_values ----------------------------------------------------

def test_csv_string_splits_on_comma() -> None:
    assert parse_tag_values("project-x,db") == ["project-x", "db"]


def test_csv_string_trims_each_item() -> None:
    assert parse_tag_values("project-x, db , token-optimization") == [
        "project-x", "db", "token-optimization",
    ]


def test_json_array_string_is_parsed() -> None:
    assert parse_tag_values('["a", "b", "c"]') == ["a", "b", "c"]


def test_json_array_string_with_whitespace() -> None:
    assert parse_tag_values('  ["a", "b"]  ') == ["a", "b"]


def test_malformed_json_array_falls_back_to_comma_split() -> None:
    # Starts with "[" but is not valid JSON: treated as a plain CSV string
    # rather than raising or silently discarding everything.
    assert parse_tag_values("[not json, still a tag") == ["[not json", "still a tag"]


def test_real_list_is_used_directly() -> None:
    assert parse_tag_values(["a", "b"]) == ["a", "b"]


def test_real_tuple_and_set_are_accepted() -> None:
    assert parse_tag_values(("a", "b")) == ["a", "b"]
    assert parse_tag_values({"a"}) == ["a"]


def test_a_label_with_a_comma_needs_the_list_form() -> None:
    # Documented limitation: a CSV string cannot express a comma-bearing
    # label. The list form carries it through untouched.
    assert parse_tag_values("a, b, c") != ["a, b, c"]
    assert parse_tag_values(["a, b, c"]) == ["a, b, c"]


def test_empty_items_are_dropped() -> None:
    assert parse_tag_values("a,,b, ,c") == ["a", "b", "c"]


def test_none_and_unrecognised_types_give_empty() -> None:
    assert parse_tag_values(None) == []
    assert parse_tag_values(42) == []
    assert parse_tag_values({"a": 1}) == []


def test_non_string_items_in_a_list_are_dropped_not_raised() -> None:
    assert parse_tag_values(["a", 42, None, "b"]) == ["a", "b"]


def test_display_casing_is_preserved() -> None:
    assert parse_tag_values("Token-Optimization,DB") == ["Token-Optimization", "DB"]


# -- tag_keys -------------------------------------------------------------

def test_tag_keys_deduplicates_by_canonical_identity() -> None:
    assert tag_keys("Token-Optimization, token-optimization, TOKEN-OPTIMIZATION") == [
        "token-optimization",
    ]


def test_tag_keys_keeps_first_seen_order() -> None:
    assert tag_keys("b,a,c,a,b") == ["b", "a", "c"]


def test_tag_keys_from_json_array_string() -> None:
    assert tag_keys('["a", "A", "b"]') == ["a", "b"]


def test_tag_keys_from_real_list() -> None:
    assert tag_keys(["a", "A ", " a"]) == ["a"]


def test_tag_keys_never_invents_a_tag_from_free_text() -> None:
    # A plain sentence is split on its commas like any other string - it is
    # never scanned for words that look tag-like.
    assert tag_keys("the quick brown fox") == ["the quick brown fox"]


def test_tag_keys_empty_for_nothing_asked() -> None:
    assert tag_keys(None) == []
    assert tag_keys("") == []
    assert tag_keys([]) == []

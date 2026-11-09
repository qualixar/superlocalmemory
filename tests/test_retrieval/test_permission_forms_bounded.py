# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The permission-question patterns stay fast because they only see single spaces.

A static audit flagged ``_MODAL_GENERIC``, ``_ALLOWED_GENERIC`` and
``_ALLOWED_IT``: on raw text, "may we" + 290 spaces + "a?x" takes over a second,
because ``\\s+``, the lazy ``[^?]+?`` and the trailing ``\\s*`` can split a
whitespace run in many ways. The only code that runs them is
``permission_action``, which first collapses every whitespace run to one space
and drops leading and trailing whitespace. Each ``\\s`` then sits alone between
two non-space characters, so every split is fixed and the match is linear.
Rule memories (up to 600 characters) are never matched by these three patterns.

These tests pin the facts that make that true, so a refactor that breaks one of
them fails here instead of in a user's recall.
"""

from __future__ import annotations

import ast
import inspect
import random
import re
import sys
import time

from superlocalmemory.retrieval import answer_question_forms as forms

_PATTERNS = ("_MODAL_GENERIC", "_ALLOWED_GENERIC", "_ALLOWED_IT")


def test_split_and_regex_agree_on_what_whitespace_is() -> None:
    # str.split() and re's \s must use the same set, or a character could survive
    # the collapse and still be matched by \s.
    space = re.compile(r"\s")
    for code in range(sys.maxunicode + 1):
        char = chr(code)
        assert (space.match(char) is not None) == char.isspace(), hex(code)


def test_collapsed_questions_have_only_single_inner_spaces() -> None:
    rnd = random.Random(2614)
    pieces = ("may", "we", "is it ok to", "a", "?", " ", "  ", "\t", "\n", "\r\n", "\u3000",
              "\x1c", "\u2028", "\xa0", "0")
    for _ in range(20_000):
        text = forms._strip("".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 30))))
        assert not re.search(r"\s\s|^\s|\s$", text), repr(text)
        assert not re.search(r"\s", text.replace(" ", "")), repr(text)


def test_the_patterns_are_matched_only_after_the_collapse() -> None:
    tree = ast.parse(inspect.getsource(forms))
    users = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and any(isinstance(n, ast.Name) and n.id in _PATTERNS for n in ast.walk(node))}
    assert users == {"permission_action"}
    source = inspect.getsource(forms.permission_action)
    assert source.index("text = _strip(question)") < source.index("pattern.match(text)")


_CRAFTED = (
    "may we" + " " * 290 + "a?x",
    "may we" + " " * 290 + "a",
    " " * 299 + "0",
    " allowed to" + " " * 288 + "0",
    "are we allowed to" + " \t" * 140 + "?b",
    "is it ok for the agent to" + "\u3000" * 270 + "a?",
    "may we " + "a " * 145 + "?x",
    "may we " + "a\t" * 146 + "?",
)


def test_crafted_questions_at_the_length_cap_take_microseconds() -> None:
    memories = ["Never publish " + " " * 580 + "x", "Do not " + "a " * 296, "never " * 100]
    for question in _CRAFTED:
        assert len(question) <= forms._MAX_QUESTION_CHARS
        start = time.perf_counter()
        forms.permission_action(question)
        forms.rule_support(question, memories)
        assert time.perf_counter() - start < 0.05, repr(question[:30])


def test_collapsed_questions_get_the_same_action_as_before() -> None:
    assert forms.permission_action("May  the agent\tpublish   without approval ?") == (
        "publish without approval")
    assert forms.permission_action("is it ok to" + " " * 280 + "ship it?") == "ship it"
    assert forms.permission_action("may we" + " " * 290 + "a?x") is None

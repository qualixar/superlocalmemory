# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm recall`` / ``slm list`` ``--tag`` and ``--tags-match`` (4.1.22)."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.cli.commands import _tags_qs


def test_each_tag_is_its_own_parameter() -> None:
    qs = _tags_qs(Namespace(tags=["Decision Record", "a,b"], tags_match="any"))
    assert qs == "&tags=Decision%20Record&tags=a%2Cb&tags_match=any"


def test_no_tag_sends_nothing_and_all_is_not_sent() -> None:
    assert _tags_qs(Namespace(tags=None, tags_match="all")) == ""
    assert _tags_qs(Namespace(tags=["  "], tags_match="any")) == ""
    assert _tags_qs(Namespace(tags=["x"], tags_match="all")) == "&tags=x"


def test_the_flags_parse() -> None:
    import argparse

    from superlocalmemory.cli.main import _add_tag_args

    parser = argparse.ArgumentParser()
    _add_tag_args(parser)
    args = parser.parse_args(["--tag", "alpha", "--tag", "beta", "--tags-match", "any"])
    assert args.tags == ["alpha", "beta"] and args.tags_match == "any"
    assert parser.parse_args([]).tags is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--tags-match", "some"])


def test_recall_and_list_parsers_carry_the_flags() -> None:
    import inspect

    import superlocalmemory.cli.main as cli_main

    src = inspect.getsource(cli_main)
    assert "_add_tag_args(recall_p)" in src and "_add_tag_args(list_p)" in src

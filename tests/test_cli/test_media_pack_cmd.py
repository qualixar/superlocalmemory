# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm media pack``: sizes are shown first and nothing is downloaded without a yes (WS-8)."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import media_cmd, media_pack_cmd

PACKS = {"packs": [
    {"pack": "captions", "label": "Describe pictures", "what": "Writes a few words.", "state": "off",
     "download_words": "Describe pictures: a 0.5 GB download", "model": "florence-community/Florence-2-base",
     "licence": "MIT", "progress": 0.0,
     "words": "One click to download the picture description model (0.5 GB)."},
    {"pack": "documents", "label": "Read scanned documents better", "what": "A larger model.", "state": "on",
     "download_words": "Read scanned documents better: a 1.9 GB download", "model": "PaddlePaddle/PaddleOCR-VL-1.6",
     "licence": "Apache-2.0", "progress": 1.0},
    {"pack": "regions", "label": "Remember where text was found", "what": "Keeps the place.", "state": "off",
     "download_words": "No download", "model": "", "licence": "", "progress": 1.0}]}


@pytest.fixture()
def daemon(monkeypatch):
    seen = []
    state = {"reply": PACKS, "post": {"pack": {**PACKS["packs"][0], "state": "downloading", "progress": 0.1}}}

    def fake(method, path, body=None, **_kw):
        seen.append((method, path, body))
        return state["post"] if method == "POST" else state["reply"]

    monkeypatch.setattr(media_pack_cmd, "daemon_request", fake)
    monkeypatch.setattr(media_cmd, "is_daemon_running", lambda: True)
    monkeypatch.setattr(media_cmd, "_is_tty", lambda: False)
    return seen, state


def _args(**kw):
    base = {"json": False, "media_command": "pack", "pack_command": "status", "pack": None, "yes": False,
            "remove_files": False}
    base.update(kw)
    return Namespace(**base)


def test_status_lists_each_part_in_plain_words(daemon, capsys):
    media_cmd.cmd_media(_args())
    out = capsys.readouterr().out
    assert "captions: One click to download the picture description model (0.5 GB)." in out
    assert "documents: on." in out  # an older daemon that sends no words: the state is still shown


def test_status_shows_the_small_computer_warning_once(daemon, capsys):
    _, state = daemon
    warning = "This computer has 8.0 GB of memory; pictures and documents may be slow."
    state["reply"] = {"packs": [{**p, "warning": warning} for p in PACKS["packs"]]}
    media_cmd.cmd_media(_args())
    assert capsys.readouterr().out.count(warning) == 1


def test_status_for_one_part_and_as_json(daemon, capsys):
    media_cmd.cmd_media(_args(pack="documents", json=True))
    data = json.loads(capsys.readouterr().out)["data"]
    assert [p["pack"] for p in data["packs"]] == ["documents"]


def test_enable_shows_the_size_first_and_needs_a_yes(daemon, capsys):
    seen, _ = daemon
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(pack_command="enable", pack="captions"))
    out = capsys.readouterr().out
    assert exc.value.code == 2 and "0.5 GB download" in out and "Nothing is downloaded until you say yes" in out
    assert not any(m == "POST" for m, *_ in seen)


def test_enable_with_yes_posts_and_says_it_runs_in_the_background(daemon, capsys):
    seen, _ = daemon
    media_cmd.cmd_media(_args(pack_command="enable", pack="captions", yes=True))
    assert ("POST", "/api/v3/features/media/packs/enable",
            {"pack": "captions", "yes": True, "source": "cli"}) in seen
    out = capsys.readouterr().out
    assert "0.5 GB" in out and "slm media pack status captions" in out


def test_a_part_whose_files_are_here_needs_no_download_question(daemon, capsys):
    seen, state = daemon
    state["post"] = {"pack": {**PACKS["packs"][1]}}
    media_cmd.cmd_media(_args(pack_command="enable", pack="documents"))
    assert "no download" in capsys.readouterr().out and any(m == "POST" for m, *_ in seen)


def test_a_refusal_from_the_daemon_is_said_plainly(daemon, monkeypatch, capsys):
    from superlocalmemory.cli.daemon import DaemonConflict

    def refuse(method, path, body=None, **kw):
        if method == "POST":
            raise DaemonConflict("Turn on Images & documents first (slm media enable); this adds to it.")
        return PACKS

    monkeypatch.setattr(media_pack_cmd, "daemon_request", refuse)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(pack_command="enable", pack="captions", yes=True))
    assert exc.value.code == media_pack_cmd.EXIT_REFUSED
    assert "Turn on Images & documents first" in capsys.readouterr().out


def test_disable_posts_remove_files(daemon, capsys):
    seen, _ = daemon
    media_cmd.cmd_media(_args(pack_command="disable", pack="captions", remove_files=True))
    assert ("POST", "/api/v3/features/media/packs/disable", {"pack": "captions", "remove_files": True}) in seen
    assert "captions is off" in capsys.readouterr().out


def test_daemon_down_exits_3(daemon, monkeypatch, capsys):
    monkeypatch.setattr(media_pack_cmd, "daemon_request", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args())
    assert exc.value.code == 3 and "slm restart" in capsys.readouterr().out


def test_the_parser_knows_the_three_parts_and_refuses_others():
    parser = argparse.ArgumentParser()
    media_cmd.register_media_parser(parser.add_subparsers(dest="command"))
    ok = parser.parse_args(["media", "pack", "enable", "captions", "--yes"])
    assert (ok.media_command, ok.pack_command, ok.pack, ok.yes) == ("pack", "enable", "captions", True)
    assert parser.parse_args(["media", "pack", "status"]).pack is None
    assert parser.parse_args(["media", "pack", "disable", "regions"]).pack == "regions"
    with pytest.raises(SystemExit):
        parser.parse_args(["media", "pack", "enable", "everything"])

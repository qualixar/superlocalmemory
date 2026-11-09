# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLMConfig.save() and `slm config set` share one lock (4.2.14).

Codex review: save() read config.json, then replaced it, without the lock
update_config() holds. A CLI change committed between the two was undone by
save() writing its earlier copy of the file.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import superlocalmemory.core.config as config_module
from superlocalmemory.core.config import SLMConfig

_FLAG = "x_forward_compat_flag"  # a key no dataclass models: save() keeps the disk value


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_cli_write_during_save_is_not_undone(tmp_path: Path, monkeypatch) -> None:
    from superlocalmemory.server.config_file import update_config

    path = _write(tmp_path / "config.json", {"mode": "a", _FLAG: "old"})
    config = SLMConfig.load(path)
    original = config_module._restrict_config_to_owner
    writers: list[threading.Thread] = []

    def cli_set(data: dict) -> None:
        data[_FLAG] = "new"

    def between_read_and_replace(target: Path) -> None:
        # save() has read the file and is about to replace it: a CLI write now
        # must wait for the replace, not land in between and be overwritten.
        writer = threading.Thread(target=update_config, args=(path, cli_set), daemon=True)
        writers.append(writer)
        writer.start()
        writer.join(timeout=0.5)
        original(target)

    monkeypatch.setattr(config_module, "_restrict_config_to_owner", between_read_and_replace)
    config.save(path)
    monkeypatch.setattr(config_module, "_restrict_config_to_owner", original)
    writers[0].join(timeout=10)
    assert not writers[0].is_alive()
    assert json.loads(path.read_text(encoding="utf-8"))[_FLAG] == "new"


def _save_with_answer_check_keys(path: str, ready, done) -> None:
    from dataclasses import replace

    from superlocalmemory.core.config import SLMConfig as _Config

    ready.set()
    config = _Config.load(Path(path))
    # A changed answer-check setting with no answer_check.json yet makes save()
    # create that file and strip the old copies out of config.json through
    # update_config: the same lock, on a new file handle.
    config.retrieval = replace(
        config.retrieval,
        sufficiency_jev_consent=not config.retrieval.sufficiency_jev_consent,
    )
    config.save(Path(path))
    done.set()


def test_save_that_tidies_answer_check_copies_does_not_deadlock(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", {
        "mode": "a",
        "retrieval": {"sufficiency_judge": "auto", "sufficiency_jev_consent": False},
    })
    context = multiprocessing.get_context("spawn")
    ready, done = context.Event(), context.Event()
    child = context.Process(target=_save_with_answer_check_keys,
                            args=(str(path), ready, done))
    child.start()
    try:
        assert ready.wait(60), "child process never started"
        finished = done.wait(10)
    finally:
        if child.is_alive() and not done.is_set():
            child.terminate()
        child.join(timeout=10)
    assert finished, "save() deadlocked on the config lock"
    assert child.exitcode == 0
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert not any(k.startswith("sufficiency_") for k in on_disk["retrieval"])
    assert (tmp_path / "answer_check.json").is_file()


def _cli_writer(path: str, rounds: int, start) -> None:
    import sys

    from superlocalmemory.server.config_file import update_config

    start.wait(60)
    lost: list[tuple[int, object]] = []

    def set_next(data: dict, value: int) -> None:
        # Under the lock the file must still hold this writer's previous
        # value: anything older means a full save replaced it with a stale copy.
        if data.get(_FLAG) != value - 1:
            lost.append((value - 1, data.get(_FLAG)))
        data[_FLAG] = value

    for value in range(rounds):
        update_config(Path(path), lambda data, v=value: set_next(data, v))
        time.sleep(0.001)
    if lost:
        print(f"lost updates: {lost[:5]}", file=sys.stderr)
        sys.exit(3)


def _full_saver(path: str, start, stop) -> None:
    from superlocalmemory.core.config import SLMConfig as _Config

    start.wait(60)
    while not stop.is_set():
        _Config.load(Path(path)).save(Path(path))
    for _ in range(3):  # saves after the last CLI write must keep it too
        _Config.load(Path(path)).save(Path(path))


def test_concurrent_cli_writes_and_full_saves_lose_nothing(tmp_path: Path) -> None:
    rounds = 150
    path = _write(tmp_path / "config.json", {"mode": "a", _FLAG: -1})
    context = multiprocessing.get_context("spawn")
    start, stop = context.Event(), context.Event()
    writer = context.Process(target=_cli_writer, args=(str(path), rounds, start))
    saver = context.Process(target=_full_saver, args=(str(path), start, stop))
    writer.start()
    saver.start()
    start.set()
    try:
        while writer.is_alive():
            # A reader must always see a whole JSON object, never a torn file.
            assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
            writer.join(timeout=0.01)
    finally:
        stop.set()
        writer.join(timeout=60)
        saver.join(timeout=60)
    assert writer.exitcode == 0
    assert saver.exitcode == 0
    assert json.loads(path.read_text(encoding="utf-8"))[_FLAG] == rounds - 1

"""Images and documents are never refused for memory: 16 GB+ is automatic, below it is one click (4.2.30)."""

from __future__ import annotations

import importlib
import logging

import pytest

media_env = importlib.import_module(
    "superlocalmemory.runtimes.media_env"
)  # the package also exports a function of that name
GIB = 1024 ** 3
MEDIA_MIN_RAM_BYTES, media_policy = media_env.MEDIA_MIN_RAM_BYTES, media_env.media_policy
AUTO, ONE_CLICK = media_env.AUTO, media_env.ONE_CLICK


@pytest.fixture(autouse=True)
def _override_unset(monkeypatch):
    monkeypatch.delenv("SLM_MEDIA_ALLOW_LOW_RAM", raising=False)
    monkeypatch.setattr(media_env, "_OVERRIDE_LOGGED", False)


def test_a_small_computer_is_warned_in_the_exact_words_and_never_refused():
    policy = media_policy(8 * GIB, {})
    assert policy.mode == ONE_CLICK
    assert policy.warning == (
        "This computer has 8.0 GB of memory; pictures and documents may be slow and the models are "
        "released when memory runs low.")
    assert "4.5 GB" in media_policy(int(4.5 * GIB), {}).warning


def test_the_threshold_is_15_gib_so_machines_sold_as_16_gb_are_automatic():
    assert MEDIA_MIN_RAM_BYTES == int(15 * GIB)
    assert media_policy(int(15.6 * GIB), {}).mode == AUTO  # a Linux VM sold as 16 GB
    assert media_policy(MEDIA_MIN_RAM_BYTES, {}).mode == AUTO
    assert media_policy(MEDIA_MIN_RAM_BYTES - 1, {}).mode == ONE_CLICK
    assert media_policy(64 * GIB, {}) == media_env.MediaPolicy(AUTO, 64 * GIB, "")


def test_an_unknown_amount_is_one_click_without_a_warning():
    assert media_policy(0, {}) == media_env.MediaPolicy(ONE_CLICK, 0, "")


def test_the_old_refusal_never_refuses_any_more():
    assert media_env.media_ram_refusal(4 * GIB, {}) == ""
    assert media_env.media_ram_refusal(0, {}) == ""
    assert media_env.media_ram_refusal(64 * GIB, {}) == ""


def test_the_developer_override_makes_a_small_machine_automatic_and_warns_once(caplog):
    env = {"SLM_MEDIA_ALLOW_LOW_RAM": "1"}
    with caplog.at_level(logging.WARNING, logger=media_env.logger.name):
        assert media_policy(4 * GIB, env).mode == AUTO
        assert media_policy(4 * GIB, env).mode == AUTO
    warnings = [r for r in caplog.records if "SLM_MEDIA_ALLOW_LOW_RAM" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_the_literal_one_overrides(value):
    assert media_policy(4 * GIB, {"SLM_MEDIA_ALLOW_LOW_RAM": value}).mode == ONE_CLICK


def test_it_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_ALLOW_LOW_RAM", "1")
    assert media_policy(4 * GIB).mode == AUTO
    monkeypatch.delenv("SLM_MEDIA_ALLOW_LOW_RAM")
    assert media_policy(4 * GIB).mode == ONE_CLICK

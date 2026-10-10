"""Images and documents are refused on computers with less than 16 GB of memory."""

from __future__ import annotations

import importlib
import logging

import pytest


media_env = importlib.import_module("superlocalmemory.runtimes.media_env")  # the package also exports a function of that name
GIB = 1024 ** 3
MEDIA_MIN_RAM_BYTES, media_ram_refusal = media_env.MEDIA_MIN_RAM_BYTES, media_env.media_ram_refusal


@pytest.fixture(autouse=True)
def _override_unset(monkeypatch):
    monkeypatch.delenv("SLM_MEDIA_ALLOW_LOW_RAM", raising=False)
    monkeypatch.setattr(media_env, "_OVERRIDE_LOGGED", False)


def test_the_refusal_names_the_machines_memory_in_the_exact_words():
    assert media_ram_refusal(8 * GIB, {}) == (
        "Images and documents need a computer with at least 16 GB of memory; this one has 8.0 GB. "
        "Your text memories keep working.")
    assert "4.5 GB" in media_ram_refusal(int(4.5 * GIB), {})


def test_the_threshold_is_15_gib_so_machines_sold_as_16_gb_pass():
    assert MEDIA_MIN_RAM_BYTES == int(15 * GIB)
    assert media_ram_refusal(int(15.6 * GIB), {}) == ""  # a Linux VM sold as 16 GB
    assert media_ram_refusal(MEDIA_MIN_RAM_BYTES, {}) == ""
    assert media_ram_refusal(MEDIA_MIN_RAM_BYTES - 1, {}) != ""
    assert media_ram_refusal(64 * GIB, {}) == ""


def test_an_unknown_amount_is_allowed():
    assert media_ram_refusal(0, {}) == ""


def test_the_developer_override_allows_a_small_machine_and_warns_once(caplog):
    env = {"SLM_MEDIA_ALLOW_LOW_RAM": "1"}
    with caplog.at_level(logging.WARNING, logger=media_env.logger.name):
        assert media_ram_refusal(4 * GIB, env) == ""
        assert media_ram_refusal(4 * GIB, env) == ""
    warnings = [r for r in caplog.records if "SLM_MEDIA_ALLOW_LOW_RAM" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_the_literal_one_overrides(value):
    assert media_ram_refusal(4 * GIB, {"SLM_MEDIA_ALLOW_LOW_RAM": value}) != ""


def test_it_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_ALLOW_LOW_RAM", "1")
    assert media_ram_refusal(4 * GIB) == ""
    monkeypatch.delenv("SLM_MEDIA_ALLOW_LOW_RAM")
    assert media_ram_refusal(4 * GIB) != ""

"""The memory warning shown wherever images and documents can be turned on."""

from __future__ import annotations

from superlocalmemory.runtimes.media_env import MEDIA_RAM_WARN_BYTES, ram_warning_text

GIB = 1024 ** 3


def test_warning_names_the_machines_memory_and_says_it_does_not_block():
    text = ram_warning_text(4 * GIB)
    assert "4.0 GB" in text and "8 GB" in text
    assert "still" in text.lower()


def test_no_warning_at_or_above_the_threshold():
    assert ram_warning_text(MEDIA_RAM_WARN_BYTES) == ""
    assert ram_warning_text(16 * GIB) == ""


def test_an_unknown_amount_gives_no_warning():
    assert ram_warning_text(0) == ""

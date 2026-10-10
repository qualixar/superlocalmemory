# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The words saving an image adds to its memory itself, kept in one place.

Saving writes them and recall strips them, so a memory whose only text is a
label can be told from one that says something.
"""

from __future__ import annotations

#: Leads the text read from a picture.
TEXT_MARKER = "[Text in image]\n"
#: The whole content of an image that had neither words nor readable text.
NO_TEXT = "[Image without text]"
#: Follows a document's title, so a one-word title is still a saveable memory.
DOCUMENT = "[PDF document]"

#: Every label recall removes before judging whether a memory has text of its own.
LABELS: tuple[str, ...] = (TEXT_MARKER.strip(), NO_TEXT, DOCUMENT)

__all__ = ["DOCUMENT", "LABELS", "NO_TEXT", "TEXT_MARKER"]

# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The portability pack (WS-9, D-19): bring memory in from other apps, and take it back out.

Six importers read exports from ChatGPT, Claude, Mem0, Letta, OKF and plain Markdown into
:class:`~superlocalmemory.portability.model.Item` lists; one shared pipeline previews, runs
(in the background, in batches, through the normal save path) and rolls back an import. Off
by default behind the flag ``portability.import``.
"""

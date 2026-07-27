# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""CLI memory mutations must use the daemon-owned canonical writer."""

from pathlib import Path


def _function_block(source: str, name: str) -> str:
    start = source.index(f"def {name}")
    next_def = source.find("\ndef ", start + 1)
    return source[start: next_def if next_def != -1 else None]


def test_cli_mutations_never_construct_a_local_memory_writer() -> None:
    source = Path("src/superlocalmemory/cli/commands.py").read_text(
        encoding="utf-8"
    )

    forget = _function_block(source, "cmd_forget")
    delete = _function_block(source, "cmd_delete")
    update = _function_block(source, "cmd_update")

    assert 'daemon_request("GET", f"/api/memories?limit=200&offset={offset}")' in forget
    assert "delete_from_daemon" in forget
    assert 'daemon_request("DELETE", path)' in delete
    assert 'daemon_request("PATCH", path, {"content": new_content})' in update
    for block in (forget, delete, update):
        assert "MemoryEngine(" not in block
        assert "engine._db.delete_fact" not in block
        assert "UPDATE atomic_facts SET content" not in block

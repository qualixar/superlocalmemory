# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""CLI memory mutations must use the canonical authorized service."""

from pathlib import Path


def _function_block(source: str, name: str) -> str:
    start = source.index(f"def {name}")
    next_def = source.find("\ndef ", start + 1)
    return source[start: next_def if next_def != -1 else None]


def test_cli_forget_delete_update_use_authorized_mutation_service() -> None:
    source = Path("src/superlocalmemory/cli/commands.py").read_text(
        encoding="utf-8"
    )

    forget = _function_block(source, "cmd_forget")
    delete = _function_block(source, "cmd_delete")
    update = _function_block(source, "cmd_update")

    assert "delete_fact_authorized" in forget
    assert "delete_fact_authorized" in delete
    assert "update_fact_authorized" in update
    for block in (forget, delete, update):
        assert "engine._db.delete_fact" not in block
        assert "UPDATE atomic_facts SET content" not in block

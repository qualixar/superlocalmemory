#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Write plugin-src/skills/slm-cli/SKILL.md from the MCP tool registry.

    python scripts/build_cli_skill.py          # write it
    python scripts/build_cli_skill.py --check  # exit 1 when the file has drifted

Run before ``node scripts/build-all-plugins.mjs``, which ships the skill to
every host. Imports the tools only; it starts no daemon and opens no store.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Write plugin-src/skills/slm-cli/SKILL.md from the MCP tool registry.")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the file has drifted; write nothing")
    check = parser.parse_args(argv).check  # --help and a typo exit here, before any write
    # Registering tools reads settings; keep this run away from the user's own data.
    with tempfile.TemporaryDirectory(prefix="slm-cli-skill-", ignore_cleanup_errors=True) as scratch:
        os.environ["SLM_DATA_DIR"] = scratch
        os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
        sys.path.insert(0, str(ROOT / "src"))
        from superlocalmemory.cli.cli_skill import REGENERATE, SKILL_PATH, render_from_registry

        text = render_from_registry()
    target = ROOT / SKILL_PATH
    if check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != text:
            print(f"{SKILL_PATH} has drifted from the tool registry. Run: {REGENERATE}")
            return 1
        print(f"{SKILL_PATH}: in sync.")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"wrote {SKILL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

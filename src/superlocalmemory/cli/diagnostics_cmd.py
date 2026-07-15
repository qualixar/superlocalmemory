# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Explicit manual export for local aggregate diagnostics."""

from __future__ import annotations

from argparse import Namespace


def cmd_diagnostics(args: Namespace) -> None:
    action = getattr(args, "diagnostics_command", None)
    if action != "export":
        raise SystemExit("choose a diagnostics subcommand: export")

    from superlocalmemory.cli.json_output import json_print
    from superlocalmemory.infra.local_diagnostics import default_diagnostics

    default_diagnostics().export_json(args.destination)
    data = {"exported": True, "reporting": "manual_export_only"}
    if bool(getattr(args, "json", False)):
        json_print("diagnostics", data=data)
        return
    print("Local aggregate diagnostics exported by explicit request.")
    print("No automatic reporting was enabled.")


__all__ = ["cmd_diagnostics"]

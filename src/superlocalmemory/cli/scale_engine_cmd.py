"""CLI surface for the staged Scale Engine lifecycle."""
from __future__ import annotations

import json
from argparse import Namespace


def cmd_db_scale(args: Namespace) -> int:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.scale_engine import ScaleEngineError, ScaleEngineManager

    manager = ScaleEngineManager(SLMConfig.load())
    action = args.scale_action
    try:
        if action == "status":
            result = manager.status()
        elif action == "prepare":
            result = manager.prepare()
        elif action == "verify":
            if not args.stage_id:
                raise ScaleEngineError("verify requires --stage-id (see `slm db scale status`)")
            result = manager.verify(args.stage_id)
        elif action == "promote":
            if not args.stage_id:
                raise ScaleEngineError("promote requires --stage-id (see `slm db scale status`)")
            result = manager.promote(args.stage_id)
        elif action == "rollback":
            if not args.backup_id:
                raise ScaleEngineError("rollback requires --backup-id (see `slm db scale status`)")
            result = manager.rollback(args.backup_id)
        else:
            raise ScaleEngineError(f"unknown Scale Engine action: {action}")
    except ScaleEngineError as exc:
        print(f"Scale Engine: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

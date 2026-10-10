# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which model makes picture vectors, and which model makes the picture query.

``paired``: the picture space is the person's own text space. A vision model
trained into that text model's space embeds the images, and a recall asks with
the text vector it already computed, so no picture worker runs for questions.
``separate``: the picture model embeds both the images and the questions.

Pure: no model is loaded here, nothing heavy is imported.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from superlocalmemory.runtimes import media_models

SpaceMode = Literal["paired", "separate"]
DEFAULT_SPACE_MODE: SpaceMode = "separate"
MODE_ENV = "SLM_MEDIA_SPACE_MODE"
_MODES = ("paired", "separate")

#: Picture evidence floor in the paired space. Provisional until calibrated on the
#: evaluation dev split: measured text-to-image cosines there sit near 0.08, far
#: below the separate model's floor.
PAIRED_MIN_SCORE = 0.05
MIN_SCORE_ENV = "SLM_MEDIA_PAIRED_MIN_SCORE"

#: text model id -> (vision model id, revision, width). Revisions are pinned with the lock files.
PAIRED_VISION: Mapping[str, tuple[str, str, int]] = MappingProxyType({
    "nomic-ai/nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-vision-v1.5", "", 768),
})
_ORG = "nomic-ai/"


@dataclass(frozen=True)
class SpacePlan:
    mode: SpaceMode
    image_model: str
    image_revision: str
    dim: int
    text_model: str  # "" for separate
    query_from_text: bool  # True only for paired
    reason: str
    min_score: float | None = None  # None: the caller keeps its configured floor

    def signature(self) -> dict[str, str | int]:
        return {"mode": self.mode, "image_model": self.image_model,
                "image_revision": self.image_revision, "dim": self.dim, "text_model": self.text_model}


def _paired_floor() -> float:
    """``SLM_MEDIA_PAIRED_MIN_SCORE`` when it is a number in [0, 1], else the constant."""
    raw = os.environ.get(MIN_SCORE_ENV, "")
    try:
        value = float(raw)
    except ValueError:
        return PAIRED_MIN_SCORE
    return value if 0.0 <= value <= 1.0 else PAIRED_MIN_SCORE


def _pair_key(text_model: str) -> str | None:
    for key in (text_model, _ORG + text_model if "/" not in text_model else ""):
        if key in PAIRED_VISION:
            return key
    return None


def _mode(requested: str | None) -> str:
    if requested is None:
        requested = os.environ.get(MODE_ENV, "")
    if requested == "single":
        raise ValueError("space mode 'single' is not available in this build")
    return requested if requested in _MODES else DEFAULT_SPACE_MODE


def resolve_space_plan(text_model: str, text_dim: int, *, requested: str | None = None,
                       separate_model: tuple[str, str, int]) -> SpacePlan:
    """The plan for the live text space. ``requested`` None reads ``SLM_MEDIA_SPACE_MODE``."""
    mode = _mode(requested)
    model, revision, dim = separate_model
    if mode == "paired":
        key = _pair_key(text_model)
        if key is None:
            reason = "text model has no paired image model"
        elif PAIRED_VISION[key][2] != int(text_dim):
            reason = "text vector width does not match the paired image model"
        else:
            vision, vision_rev, vision_dim = PAIRED_VISION[key]
            return SpacePlan("paired", vision, vision_rev, vision_dim, key, True,
                             "text model has a paired image model", _paired_floor())
        return SpacePlan("separate", model, revision, dim, "", False, reason, media_models.min_score_for(model))
    reason = "default" if requested is None and not os.environ.get(MODE_ENV) else "requested"
    return SpacePlan("separate", model, revision, dim, "", False, reason, media_models.min_score_for(model))


def compatible(plan: SpacePlan, stored_signature: Mapping[str, Any] | None) -> bool:
    """True when nothing is stored yet or every part of the signature matches."""
    if stored_signature is None:
        return True
    want = plan.signature()
    return all(str(stored_signature.get(k)) == str(v) for k, v in want.items())


def _live_text_space(data_root: Path) -> tuple[str, int]:
    """``(model, width)`` of the text space, read the way the daemon binds it, without loading anything."""
    signature = ""
    db_path = data_root / "memory.db"
    if db_path.is_file():
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
            try:
                row = conn.execute("SELECT live_signature FROM embedding_space WHERE id = 1").fetchone()
            finally:
                conn.close()
            signature = str(row[0]) if row and row[0] else ""
        except sqlite3.Error:
            signature = ""
    if not signature:
        try:
            data = json.loads((data_root / "config.json").read_text(encoding="utf-8"))
            signature = str(data.get("embedding_signature") or "")
        except (OSError, ValueError, AttributeError):
            signature = ""
    model, _sep, dim = signature.partition("::")
    if model and dim.isdigit():
        return model, int(dim)
    from superlocalmemory.core.config import EmbeddingConfig

    shipped = EmbeddingConfig()
    return shipped.model_name, shipped.dimension


def current_space_plan(data_root: str | Path | None = None, *, requested: str | None = None) -> SpacePlan:
    """The plan for this data folder's live text space and the shipped picture model."""
    from superlocalmemory.runtimes.media_env import MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION

    from superlocalmemory.infra.data_root import canonical_data_root

    root = Path(data_root) if data_root is not None else canonical_data_root()
    text_model, text_dim = _live_text_space(root)
    return resolve_space_plan(text_model, text_dim, requested=requested,
                              separate_model=(MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION, 768))


__all__ = ["DEFAULT_SPACE_MODE", "MIN_SCORE_ENV", "MODE_ENV", "PAIRED_MIN_SCORE", "PAIRED_VISION", "SpaceMode", "SpacePlan",
           "compatible", "current_space_plan", "resolve_space_plan"]

# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Narrowing a recall by project, by the agent that saved a memory, or by what
a memory is about.

``project`` and ``agent`` come from what was recorded when each memory was
saved (the ``project`` and ``agent_id`` an agent passes to remember). ``agent``
is compared without regard to case or surrounding spaces; ``project`` by
``core.project_identity.project_key`` (4.1.21, GitHub #150), so a bare name
matches the same project saved as a full path. ``about`` is a name - a
person, a project, a tool - looked up the way recall looks up names: exact
name, alias, or a spelling close enough to merge automatically, read-only.

The candidates were already checked for visibility (a shared or global
memory may belong to another profile), so matching does not re-filter by
profile. ``saved_by``, ``about``, ``kind`` and ``tags`` are hard filters: only
memories that match are kept, even if that leaves none, because the caller
asked for exactly that. ``project`` filters too, but recall falls back to
unfiltered results - and says so - when nothing it found was saved under the
project (``retrieval.project_scope``), unless ``project_strict`` asks it to
keep only that project's memories even if that leaves none. ``prefer_project``
never filters; it only ranks that project's memories higher. Reads only;
``matching_fact_ids`` never raises (a failure keeps nothing rather than
everything).

``tags`` (4.1.22): exact label matching, composed with every other facet
as AND. A label's identity is ``core.tag_identity.tag_key`` - Unicode NFC,
trimmed, internal whitespace collapsed, casefolded, punctuation kept - never
a raw string compare, so "Token-Optimization" and "token-optimization " are
the same tag. ``tags_match`` is ``"all"`` (every requested label must be on
the memory - the default) or ``"any"`` (at least one). Stored tags are read
from ``memories.metadata_json -> '$.tags'`` and parsed the same way
regardless of whether that value is a comma-separated string, a string that
is itself a JSON array, or (an MCP caller) a real list -
``core.tag_identity.parse_tag_values`` is the one place that distinction is
handled. Unlike ``kind``, no "richer report" lives in this module - the
richer ``tag_scope`` note/reason the response shows is built once, in
``retrieval.tag_scope``, from the same candidates this module already
admitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from superlocalmemory.storage.json_columns import load_json_column

logger = logging.getLogger(__name__)

_MAX_VALUE = 200


def _clean(value: object, limit: int = _MAX_VALUE) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:limit] if text else None


def _clean_project(value: object) -> str | None:
    # A path's identity is its LAST part, so a long path must not be cut to
    # the 200 characters a name gets (core.project_identity).
    from superlocalmemory.core.project_identity import MAX_PROJECT_CHARS

    return _clean(value, MAX_PROJECT_CHARS)


def _clean_tags(value: object) -> tuple[str, ...]:
    """``value`` (a list, or a comma-separated string) to a de-duplicated
    tuple of DISPLAY labels, in first-seen order, one per distinct
    ``tag_key``. A label containing a comma needs the list form - a CSV
    string cannot express it (``core.tag_identity.parse_tag_values``)."""
    from superlocalmemory.core.tag_identity import parse_tag_values, tag_key

    seen: dict[str, None] = {}
    out: list[str] = []
    for item in parse_tag_values(value):
        key = tag_key(item)
        if key is not None and key not in seen:
            seen[key] = None
            out.append(item)
    return tuple(out)


def _clean_tags_match(value: object) -> str:
    text = str(value).strip().lower() if isinstance(value, str) and value.strip() else ""
    return text if text in ("all", "any") else "all"


@dataclass(frozen=True, slots=True)
class Facets:
    project: str | None = None
    agent: str | None = None
    about: str | None = None
    #: 4.1.19 WP8: the DISPLAYED kind (storage.memory_kinds.kind_fields), so a
    #: legacy row matches the kind it is mapped to, not its raw stored value.
    #: Already validated/normalized by the caller (core.kind_query.resolve_kind)
    #: before it ever reaches here — this module only matches, never parses.
    kind: str | None = None
    #: 4.1.21 (#150): rank this project's memories higher; never a filter.
    prefer_project: str | None = None
    #: 4.1.22: exact DISPLAY labels asked for, de-duplicated by
    #: ``core.tag_identity.tag_key`` — never the raw caller input verbatim.
    #: Empty means no tag filter.
    tags: tuple[str, ...] = ()
    #: "all" (every requested label must be on the memory) or "any" (at
    #: least one). Anything else given by a caller collapses to "all".
    tags_match: str = "all"
    #: 4.1.22: ``project`` keeps only that project's memories even when that
    #: leaves none, like every other filter. Off: the 4.1.21 fall-back.
    project_strict: bool = False

    @classmethod
    def of(cls, project: object = None, agent: object = None, about: object = None,
          kind: object = None, prefer_project: object = None,
          tags: object = None, tags_match: object = None,
          project_strict: object = False) -> "Facets":
        clean = _clean_project(project)
        return cls(clean, _clean(agent), _clean(about), _clean(kind),
                   _clean_project(prefer_project), _clean_tags(tags),
                   _clean_tags_match(tags_match),
                   bool(project_strict) and clean is not None)

    @property
    def narrows(self) -> bool:
        """True when any facet filters (everything but ``prefer_project``)."""
        return not (self.project is None and self.agent is None and self.about is None
                    and self.kind is None and not self.tags)

    @property
    def empty(self) -> bool:
        """Nothing was asked for at all, so recall need not be told."""
        return not self.narrows and self.prefer_project is None

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in (("project", self.project), ("agent", self.agent),
                                  ("about", self.about), ("kind", self.kind),
                                  ("prefer_project", self.prefer_project),
                                  ("tags", ",".join(self.tags) if self.tags else None),
                                  ("tags_match", self.tags_match if self.tags else None))
                if v is not None} | ({"project_strict": "true"} if self.project_strict else {})


def _chunks(items: list[str], size: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _by_project(db: Any, fact_ids: list[str], wanted: str) -> set[str]:
    from superlocalmemory.core.project_identity import project_key
    from superlocalmemory.retrieval.project_scope import project_keys

    key = project_key(wanted)
    if key is None:
        return set()
    return {fid for fid, k in project_keys(db, fact_ids).items() if k == key}


def _by_memory_metadata(db: Any, fact_ids: list[str], profile_id: str,
                        key: str, wanted: str) -> set[str]:
    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT f.fact_id FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
            f"WHERE f.fact_id IN ({','.join('?' * len(chunk))}) "
            f"AND lower(trim(json_extract(m.metadata_json, '$.{key}'))) = ?",
            (*chunk, wanted.lower()),
        )
        keep.update(str(dict(r)["fact_id"]) for r in rows)
    return keep


def entity_ids_named(db: Any, names: Iterable[str], profile_id: str,
                     resolver: Any = None) -> set[str]:
    """The known entities these names refer to, read-only.

    Exact name, alias, or a spelling close enough to merge automatically (the
    resolver's ``lookup``); without a resolver, the exact name only. Never
    creates an entity or saves an alias. Raises what the store raises.
    """
    wanted = [n for n in names if isinstance(n, str) and n.strip()]
    if not wanted:
        return set()
    lookup = getattr(resolver, "lookup", None)
    if lookup is not None:
        return {str(e) for e in lookup(wanted, profile_id).values()}
    found: set[str] = set()
    for name in wanted:
        entity = db.get_entity_by_name(name, profile_id)
        if entity is not None:
            found.add(str(entity.entity_id))
    return found


def _about(db: Any, fact_ids: list[str], profile_id: str, name: str, resolver: Any) -> set[str]:
    entity_ids = entity_ids_named(db, [name], profile_id, resolver)
    if not entity_ids:
        return set()
    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT fact_id, canonical_entities_json FROM atomic_facts "
            f"WHERE fact_id IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            d = dict(row)
            try:
                mentioned = set(load_json_column(d.get("canonical_entities_json"), [],
                                                 column="atomic_facts.canonical_entities_json"))
            except (TypeError, ValueError):
                continue
            if mentioned & entity_ids:
                keep.add(str(d["fact_id"]))
    return keep


#: Fallback only for a caller that does not pass the configured value.
#: storage.memory_kinds.kind_fields defaults to this too, so the two never
#: disagree when neither side is told otherwise — but every real caller
#: SHOULD pass the live ``SLMConfig.memory_kinds.display_min_confidence``
#: (4.1.19 M3: this used to be silently hard-coded everywhere).
_DEFAULT_DISPLAY_MIN_CONFIDENCE = 0.20


def _by_kind(db: Any, fact_ids: list[str], wanted: str, *,
            display_min_confidence: float = _DEFAULT_DISPLAY_MIN_CONFIDENCE) -> set[str]:
    from superlocalmemory.storage.memory_kinds import kind_fields

    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT fact_id, memory_kind, memory_kind_source, memory_kind_confidence, "
            "fact_type FROM atomic_facts "
            f"WHERE fact_id IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            d = dict(row)
            fields = kind_fields(d, display_min_confidence=display_min_confidence)
            if fields["memory_kind"] == wanted:
                keep.add(str(d["fact_id"]))
    return keep


def _by_tags(db: Any, fact_ids: list[str], wanted: tuple[str, ...], match: str) -> set[str]:
    """``fact_ids`` whose parent memory's tags satisfy ``wanted``/``match``.

    Reads each candidate's raw stored tag value once (chunked, like the
    other metadata filters here) and parses it with
    ``core.tag_identity.tag_keys`` — handling a CSV string, a JSON-array
    string, or (already) a real list identically. A memory with no tags at
    all never matches, whatever ``wanted`` is.
    """
    from superlocalmemory.core.tag_identity import tag_key, tag_keys

    wanted_keys = frozenset(k for k in (tag_key(t) for t in wanted) if k is not None)
    if not wanted_keys:
        return set()
    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT f.fact_id AS fact_id, "
            "json_extract(m.metadata_json, '$.tags') AS tags "
            "FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
            f"WHERE f.fact_id IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            d = dict(row)
            have = frozenset(tag_keys(d.get("tags")))
            if not have:
                continue
            ok = bool(wanted_keys & have) if match == "any" else wanted_keys <= have
            if ok:
                keep.add(str(d["fact_id"]))
    return keep


def matching_fact_ids(db: Any, fact_ids: Iterable[str], profile_id: str,
                      facets: Facets, resolver: Any = None, *,
                      display_min_confidence: float = _DEFAULT_DISPLAY_MIN_CONFIDENCE,
                      ) -> set[str]:
    """The subset of ``fact_ids`` that matches every requested facet.

    ``display_min_confidence`` (4.1.19 M3): the kind facet's confidence
    threshold for an unconfirmed (model-suggested) kind. Callers that have a
    live ``SLMConfig`` should pass ``config.memory_kinds.display_min_confidence``
    so this agrees with every other kind-aware surface; the default here
    exists only for a caller with no config in hand and matches
    ``storage.memory_kinds.kind_fields``'s own default.
    """
    remaining = list(dict.fromkeys(str(f) for f in fact_ids))
    if not facets.narrows or not remaining:
        return set(remaining)

    def keep(matched: set[str]) -> list[str]:
        # Each filter is asked ONCE for the whole list. It used to sit inside
        # the comprehension's condition, so it was re-run for every id: N
        # queries over N rows per facet. Measured on a 21,739-fact store, one
        # kind-filtered recall called the kind reader 380 times and decoded
        # 170,000 kind rows -- 3 to 20 s of a recall spent on its filter.
        return [f for f in remaining if f in matched]

    try:
        if facets.project is not None:
            remaining = keep(_by_project(db, remaining, facets.project))
        if facets.agent is not None and remaining:
            remaining = keep(_by_memory_metadata(
                db, remaining, profile_id, "agent_id", facets.agent))
        if facets.about is not None and remaining:
            remaining = keep(_about(db, remaining, profile_id, facets.about, resolver))
        if facets.kind is not None and remaining:
            remaining = keep(_by_kind(
                db, remaining, facets.kind,
                display_min_confidence=display_min_confidence))
        if facets.tags and remaining:
            remaining = keep(_by_tags(db, remaining, facets.tags, facets.tags_match))
    except Exception as exc:  # noqa: BLE001 - a filter that cannot run keeps nothing
        logger.warning("Recall facet filter failed (%s); no results kept", type(exc).__name__)
        return set()
    return set(remaining)


def _project_counts(db: Any, profile_id: str, limit: int) -> list[dict]:
    """Projects grouped by ``project_key``, so "slm" and "/x/slm" are one."""
    from superlocalmemory.core.project_identity import project_key

    rows = db.execute(
        "SELECT json_extract(metadata_json, '$.project') AS v, COUNT(*) AS n "
        "FROM memories WHERE profile_id = ? AND json_valid(metadata_json) "
        "GROUP BY v HAVING v IS NOT NULL AND trim(v) != ''", (profile_id,))
    counts: dict[str, int] = {}
    for row in rows:
        d = dict(row)
        key = project_key(d["v"])
        if key is not None:
            counts[key] = counts.get(key, 0) + int(d["n"])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:int(limit)]
    return [{"name": name, "memories": n} for name, n in ranked]


def list_facets(db: Any, profile_id: str, *, limit: int = 50) -> dict[str, list[dict]]:
    """The projects and agents this profile's memories were saved under, with counts."""
    out: dict[str, list[dict]] = {"projects": [], "agents": []}
    try:
        out["projects"] = _project_counts(db, profile_id, limit)
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = db.execute(
            "SELECT lower(trim(json_extract(metadata_json, '$.agent_id'))) AS v, COUNT(*) AS n "
            "FROM memories WHERE profile_id = ? GROUP BY v HAVING v IS NOT NULL AND v != '' "
            "ORDER BY n DESC, v LIMIT ?", (profile_id, int(limit)))
        out["agents"] = [{"name": str(dict(r)["v"]), "memories": int(dict(r)["n"])}
                         for r in rows]
    except Exception:  # noqa: BLE001
        pass
    return out


__all__ = ["Facets", "entity_ids_named", "list_facets", "matching_fact_ids"]

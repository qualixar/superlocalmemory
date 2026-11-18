# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Markdown projection of compiled pages and the way back (WS-6, D-17, ``compiler.wiki``).

``build`` writes every compiled page to a folder as Markdown (``wiki_blocks``). It
reads only verified sentences (``pages.show_page``), never overwrites a file that has
been edited since SLM wrote it, and in shadow only reports what it would write. ``diff``
and ``status`` read the folder and say what changed. ``sync_plan`` reads the folder
back and finds edited blocks **by hash**; ``queue_edits`` turns each into a ProposalSet
of kind ``wiki_edit`` with the slot version it was written against. Nothing here changes
a memory:

* an edit is a proposal; admission of its words is the correction ledger's
  (``compiler/wiki_edit``), with the secret screen and origin ``external_edit``;
* a block whose text was deleted, and a block that vanished from a file, are counted
  and ignored: a deleted line is never a forget;
* a file whose hash is the one SLM recorded when it wrote it is its own write and is
  skipped without being read;
* erasing a fact scrubs the pages that cite it and strips its blocks from the files
  (``scrub_fact``, called by ``compiler/erasure``), so the folder never keeps what
  the store erased.

Files live under ``SLM_WIKI_DIR``, the ``dir`` in ``<data folder>/wiki.json`` (``slm wiki dir``)
or ``<data folder>/wiki``, one folder per profile.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.compiler import page_templates as tpl
from superlocalmemory.compiler import pages, proposals, wiki_blocks, wiki_edit
from superlocalmemory.compiler.page_text import block_hash, content_hash, renumber
from superlocalmemory.compiler.versions import select_rows as _select

logger = logging.getLogger(__name__)

ORIGIN = "wiki"
ENV_DIR = "SLM_WIKI_DIR"
CONFIG_FILE = "wiki.json"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _configured_dir() -> str | None:
    """``dir`` from ``<data folder>/wiki.json`` (``slm wiki dir``), if set and readable."""
    from superlocalmemory.infra.data_root import canonical_data_root

    try:
        value = json.loads((canonical_data_root() / CONFIG_FILE).read_text(encoding="utf-8"))
        return str(value["dir"]) if isinstance(value, dict) and value.get("dir") else None
    except (OSError, ValueError, KeyError):
        return None


def wiki_root(base: str | Path | None = None) -> Path:
    """The folder pages are projected to: ``base``, ``SLM_WIKI_DIR``, ``dir`` in ``wiki.json``
    or ``<data folder>/wiki``."""
    if base:
        return Path(base).expanduser()
    chosen = os.environ.get(ENV_DIR) or _configured_dir()
    if chosen:
        return Path(chosen).expanduser()
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root() / "wiki"


def set_wiki_dir(path: str | Path | None) -> Path:
    """Choose the folder (or, with ``None``, go back to the default). A person's setting: the
    daemon never takes a folder from a request. Returns the folder now in force."""
    from superlocalmemory.infra.data_root import canonical_data_root

    config = canonical_data_root() / CONFIG_FILE
    if path is None:
        config.unlink(missing_ok=True)
        return wiki_root()
    target = Path(path).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise ValueError("That path is a file, not a folder.")
    _write_atomic(config, json.dumps({"dir": str(target)}) + "\n")
    return target


def profile_dir(root: Path, profile_id: str) -> Path:
    return root / (_SAFE.sub("_", profile_id).strip("._") or "profile")


def rel_path(page: Mapping[str, Any]) -> str:
    slug = _SAFE.sub("-", str(page["title"]).lower()).strip("-.")[:60] or "page"
    return f"{page['kind']}/{slug}-{page['page_id'][:8]}.md"


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".slm-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _recorded(conn: Any, profile_id: str) -> dict[str, dict]:
    try:
        return {r["rel_path"]: r for r in _select(
            conn, "SELECT rel_path, page_id, file_hash FROM wiki_files WHERE profile_id = ?",
            (profile_id,))}
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return {}
        raise


def _record(conn: Any, profile_id: str, rel: str, page_id: str | None, digest: str) -> None:
    conn.execute(
        "INSERT INTO wiki_files (profile_id, rel_path, page_id, file_hash, written_at) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(profile_id, rel_path) DO UPDATE SET "
        "page_id = excluded.page_id, file_hash = excluded.file_hash, "
        "written_at = excluded.written_at", (profile_id, rel, page_id, digest, _now()))


def slot_markers(conn: Any, profile_id: str, fact_ids: Sequence[str]) -> dict[str, str]:
    """``{fact_id: "<slot_id>:<version>"}`` for facts that fill a slot (a private legacy
    slot is not a claim about anything and is skipped)."""
    ids = list(dict.fromkeys(fact_ids))
    if not ids:
        return {}
    rows = _select(
        conn, "SELECT r.fact_id AS fact_id, s.slot_id AS slot_id, s.version AS version "
        "FROM slot_revisions r JOIN memory_slots s ON s.slot_id = r.slot_id WHERE s.profile_id = ? "
        "AND s.key_source != 'legacy' AND r.fact_id IN (" + ",".join("?" for _ in ids) + ") "
        "ORDER BY s.created_at DESC, s.slot_id DESC", (profile_id, *ids))
    return {r["fact_id"]: f"{r['slot_id']}:{r['version']}" for r in rows}


# -- build, diff, status ---------------------------------------------------------------------------

def _projected(read: Any, profile_id: str) -> list[tuple[dict, str]]:
    """Every page with something to show, as ``(view, file text)``."""
    out = []
    for row in pages.list_pages(read, profile_id):
        view = pages.show_page(read, profile_id, row["page_id"])
        if view is None or not view["sentences"]:
            continue
        cited = [f for s in view["sentences"] for f in s["facts"]]
        out.append((view, wiki_blocks.render_page(view, slot_markers(read, profile_id, cited))))
    return out


def classify(root: Path, profile_id: str, view: Mapping[str, Any], text: str,
             recorded: Mapping[str, dict]) -> tuple[str, str, Path]:
    """``(state, rel, path)``: ``new``, ``changed``, ``same``, ``edited`` (a person changed the
    file since SLM wrote it: never overwritten) or ``adopted`` (the file already holds this text)."""
    rel = rel_path(view)
    path = profile_dir(root, profile_id) / rel
    disk = _read(path)
    if disk is None:
        return "new", rel, path
    on_disk, new = file_hash(disk), file_hash(text.encode("utf-8"))
    rec = recorded.get(rel)
    if on_disk == new:
        return ("same" if rec and rec["file_hash"] == new else "adopted"), rel, path
    if rec and rec["file_hash"] == on_disk:
        return "changed", rel, path
    return "edited", rel, path


def build(read: Any, write: Any, profile_id: str, root: Path, *, dry_run: bool = False) -> dict:
    """Project every page. ``dry_run`` (shadow) writes nothing and reports what would happen."""
    recorded, report = _recorded(read, profile_id), {"written": [], "unchanged": [],
                                                      "kept_edited": [], "removed": []}
    expected: set[str] = set()
    for view, text in _projected(read, profile_id):
        state, rel, path = classify(root, profile_id, view, text, recorded)
        expected.add(rel)
        if state == "edited":
            report["kept_edited"].append(rel)
        elif state in ("same",):
            report["unchanged"].append(rel)
        else:
            report["written"].append(rel)
            if not dry_run:
                if state != "adopted":
                    _write_atomic(path, text)
                _record(write, profile_id, rel, view["page_id"], file_hash(text.encode("utf-8")))
    for rel, rec in recorded.items():
        if rel in expected:
            continue
        path = profile_dir(root, profile_id) / rel
        disk = _read(path)
        if disk is not None and file_hash(disk) != rec["file_hash"]:
            report["kept_edited"].append(rel)
            continue
        report["removed"].append(rel)
        if not dry_run:
            path.unlink(missing_ok=True)
            write.execute("DELETE FROM wiki_files WHERE profile_id = ? AND rel_path = ?",
                          (profile_id, rel))
    return {"dir": str(profile_dir(root, profile_id)), "dry_run": dry_run, **report}


def diff(read: Any, profile_id: str, root: Path) -> dict:
    """What a build would do, and which files hold edits waiting for ``sync``. Reads only."""
    recorded = _recorded(read, profile_id)
    items = []
    seen: set[str] = set()
    for view, text in _projected(read, profile_id):
        state, rel, _path = classify(root, profile_id, view, text, recorded)
        seen.add(rel)
        if state != "same":
            items.append({"path": rel, "state": state})
    for rel, rec in recorded.items():
        if rel not in seen:
            disk = _read(profile_dir(root, profile_id) / rel)
            items.append({"path": rel, "state": "stale" if disk is None
                          or file_hash(disk) == rec["file_hash"] else "edited"})
    return {"dir": str(profile_dir(root, profile_id)), "items": sorted(items, key=lambda i: i["path"])}


def status(read: Any, profile_id: str, root: Path) -> dict:
    """Counts for ``slm wiki status`` and the dashboard card."""
    changes = diff(read, profile_id, root)["items"]
    counts: dict[str, int] = {}
    for item in changes:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    try:
        waiting = _select(read, "SELECT COUNT(DISTINCT s.set_id) AS n FROM proposal_sets s JOIN "
                          "proposals p ON p.set_id = s.set_id WHERE s.profile_id = ? AND "
                          "s.state = 'proposed' AND p.kind = 'wiki_edit'", (profile_id,))[0]["n"]
    except Exception:  # noqa: BLE001 - a store before M080
        waiting = 0
    return {**pages.status(read, profile_id), "dir": str(profile_dir(root, profile_id)),
            "files": len(_recorded(read, profile_id)), "differences": counts,
            "edits_waiting": int(waiting)}


# -- reading edits back ----------------------------------------------------------------------------

def _stored_sentences(read: Any, profile_id: str, page_id: str | None) -> list[dict] | None:
    if not page_id:
        return None
    rows = _select(read, "SELECT body_json FROM compiled_pages WHERE page_id = ? AND profile_id = ?",
                   (page_id, profile_id))
    return json.loads(rows[0]["body_json"]) if rows else None


def _edit_from(read: Any, profile_id: str, rel: str, page_id: str, block: wiki_blocks.Block,
               stored: Sequence[dict]) -> tuple[dict | None, str]:
    """``(edit, "")`` for an edited block that means a change to one memory, else ``(None, why)``."""
    base = next((s for s in stored if block_hash(s["text"]) == block.hash
                 and tuple(s["facts"]) == block.fact_ids), None)
    if base is None:
        return None, "the page was rebuilt since this file was written; run slm wiki build"
    if len(block.fact_ids) != 1:
        return None, "this line cites several memories; edit one of them"
    fact = tpl.live_facts(read, profile_id, list(block.fact_ids)).get(block.fact_ids[0])
    if fact is None:
        return None, "the memory it cites is gone or no longer current"
    new = wiki_edit.edit_text(base, block.text, fact["content"])
    if new is None or new == fact["content"]:
        return None, ("this line is computed from slot values; edit the memory it cites"
                      if base["type"] == "template" and base.get("template") not in
                      ("slot_value", "story_step") else "the edit does not fit the line")
    if len(new) > wiki_edit.TEXT_LIMIT:
        return None, "the edit is too long"
    return {"path": rel, "page_id": page_id, "fact_id": block.fact_ids[0], "block_hash": block.hash,
            "base": {"slot_id": block.slot_id, "slot_version": block.slot_version},
            "new_content": new,
            "edit_key": wiki_edit.edit_key(profile_id, block.fact_ids[0], block.hash, new)}, ""


def sync_plan(read: Any, profile_id: str, root: Path) -> dict:
    """Read the folder and list the edits it holds. Reads only; proposes nothing yet."""
    folder = profile_dir(root, profile_id)
    recorded = _recorded(read, profile_id)
    plan = {"edits": [], "deleted_lines": 0, "skipped": [], "own_writes": 0, "files": 0,
            "already_proposed": 0}
    if not folder.is_dir():
        return plan
    seen = {r["edit_key"] for r in _select(read, "SELECT edit_key FROM wiki_edits WHERE "
                                           "profile_id = ?", (profile_id,))}
    for path in sorted(folder.rglob("*.md")):
        rel = path.relative_to(folder).as_posix()
        data = _read(path)
        if data is None:
            continue
        plan["files"] += 1
        if recorded.get(rel, {}).get("file_hash") == file_hash(data):
            plan["own_writes"] += 1  # SLM's own write, by hash: nothing to read
            continue
        parsed = wiki_blocks.parse_file(data.decode("utf-8", "replace"))
        stored = _stored_sentences(read, profile_id, parsed.page_id)
        if stored is None:
            plan["skipped"].append({"path": rel, "why": "this file belongs to a page that no "
                                                        "longer exists; run slm wiki build"})
            continue
        present = {b.hash for b in parsed.blocks}
        plan["deleted_lines"] += sum(1 for s in stored if block_hash(s["text"]) not in present)
        for block in parsed.blocks:
            if wiki_blocks.emptied(block):
                plan["deleted_lines"] += 1
            elif wiki_blocks.edited(block):
                edit, why = _edit_from(read, profile_id, rel, parsed.page_id, block, stored)
                if edit is None:
                    plan["skipped"].append({"path": rel, "why": why})
                elif edit["edit_key"] in seen:
                    plan["already_proposed"] += 1
                else:
                    plan["edits"].append(edit)
        if parsed.malformed:
            plan["skipped"].append({"path": rel, "why": f"{parsed.malformed} damaged marker(s) "
                                                        "were left alone"})
    return plan


Admit = Callable[[dict], Mapping[str, Any]]


class SkipEdit(Exception):
    """``admit`` raises it for an edit that cannot be admitted now (with the reason in words)."""


def queue_edits(db: Any, profile_id: str, edits: Sequence[dict], admit: Admit) -> dict:
    """One ProposalSet per edit (so each is decided on its own). ``admit`` carries the edit's
    words through normal admission, off the writer lock (it may embed): it returns the
    correction ``case_id`` and ``successor_fact_id`` it created, or raises :class:`SkipEdit`.
    Nothing is applied. Returns ``{"queued": [...], "skipped": [...]}``."""
    queued, skipped = [], []
    for edit in edits:
        try:
            got = admit(edit)
        except SkipEdit as why:
            skipped.append({"path": edit["path"], "why": str(why)})
            continue
        payload = {"fact_id": edit["fact_id"], "case_id": got["case_id"],
                   "successor_fact_id": got["successor_fact_id"], "page_id": edit["page_id"],
                   "block_hash": edit["block_hash"], "base": edit["base"], "path": edit["path"]}
        with db.raw_connection() as conn:
            set_id = proposals.create_set(
                conn, profile_id=profile_id, origin=ORIGIN, title="Edit from the Markdown wiki",
                proposals=[{"kind": wiki_edit.KIND, "payload": payload}])
            conn.execute("INSERT OR IGNORE INTO wiki_edits (edit_key, profile_id, fact_id, set_id, "
                         "seen_at) VALUES (?, ?, ?, ?, ?)",
                         (edit["edit_key"], profile_id, edit["fact_id"], set_id, _now()))
        queued.append({"set_id": set_id, "fact_id": edit["fact_id"], "path": edit["path"]})
    return {"queued": queued, "skipped": skipped}


# -- erasure -------------------------------------------------------------------------------------

def _strip_blocks(text: str, fact_id: str) -> str:
    pattern = re.compile(r"<!-- slm:block fact=([^ >]*) [^>\n]*?-->\n.*?\n?<!-- /slm:block -->\n?\n?",
                         re.S)
    return pattern.sub(lambda m: "" if fact_id in m.group(1).split(",") else m.group(0), text)


def scrub_pages(conn: Any, profile_id: str, fact_id: str) -> list[str]:
    """Remove the sentences that cite ``fact_id`` from the pages that hold them (I-11),
    keeping the rest. Returns the page ids touched; a page left empty is deleted."""
    touched = []
    for row in _select(conn, "SELECT p.page_id, p.title, p.body_json, p.withheld_json FROM "
                       "compiled_pages p WHERE p.profile_id = ? AND p.page_id IN (SELECT page_id "
                       "FROM page_citations WHERE fact_id = ?)", (profile_id, fact_id)):
        kept = [s for s in json.loads(row["body_json"]) if fact_id not in s["facts"]]
        pid = row["page_id"]
        touched.append(pid)
        if not kept:
            conn.execute("DELETE FROM compiled_pages WHERE page_id = ?", (pid,))
            conn.execute("DELETE FROM page_citations WHERE page_id = ?", (pid,))
            continue
        kept = renumber(kept)
        conn.execute("UPDATE compiled_pages SET body_json = ?, content_hash = ?, shown_count = ? "
                     "WHERE page_id = ?", (json.dumps(kept, sort_keys=True),
                                           content_hash(row["title"], kept), len(kept), pid))
        conn.execute("DELETE FROM page_citations WHERE page_id = ?", (pid,))
        for s in kept:
            for cited in s["facts"]:
                conn.execute("INSERT OR IGNORE INTO page_citations (page_id, seq, fact_id) "
                             "VALUES (?, ?, ?)", (pid, s["seq"], cited))
    return touched


def scrub_fact(conn: Any, profile_id: str, fact_id: str, *, root: Path | None = None) -> dict:
    """Erasure of ``fact_id``: scrub the pages and rewrite the projected files that name it.

    A file is rewritten without the blocks that cite the fact (the rest, and any edit a
    person made elsewhere in it, is kept); a file left with no block is deleted. Never
    raises: a file that cannot be written is logged, and erasure goes on."""
    out = {"pages": [], "files": 0}
    try:
        out["pages"] = scrub_pages(conn, profile_id, fact_id)
    except Exception as exc:  # noqa: BLE001 - tables absent on older schemas
        if "no such table" not in str(exc).lower():
            raise
        return out
    try:
        folder = profile_dir(root or wiki_root(), profile_id)
        recorded = _recorded(conn, profile_id)
    except Exception:  # noqa: BLE001
        return out
    for path in sorted(folder.rglob("*.md")) if folder.is_dir() else []:
        try:
            data = _read(path)
            if data is None or fact_id.encode() not in data:
                continue
            rel = path.relative_to(folder).as_posix()
            rec, old = recorded.get(rel), file_hash(data)
            text = _strip_blocks(data.decode("utf-8", "replace"), fact_id)
            out["files"] += 1
            if "<!-- slm:block " not in text:
                path.unlink(missing_ok=True)
                conn.execute("DELETE FROM wiki_files WHERE profile_id = ? AND rel_path = ?",
                             (profile_id, rel))
                continue
            _write_atomic(path, text)
            if rec and rec["file_hash"] == old:  # SLM's own file stays its own write
                _record(conn, profile_id, rel, rec["page_id"], file_hash(text.encode("utf-8")))
        except Exception:  # noqa: BLE001
            logger.warning("wiki file could not be rewritten after an erasure", exc_info=True)
    return out


__all__ = ["CONFIG_FILE", "ENV_DIR", "ORIGIN", "build", "classify", "diff", "file_hash", "profile_dir",
           "SkipEdit", "queue_edits", "rel_path", "scrub_fact", "scrub_pages", "slot_markers", "status",
           "set_wiki_dir", "sync_plan", "wiki_root"]

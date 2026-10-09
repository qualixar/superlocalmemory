"""An identical copy touched in the same pass its owner changes becomes the owner, not a dangling borrower."""

from __future__ import annotations

import os
from types import SimpleNamespace

PDF = b"%PDF-1.4\n" + b"0" * 100


def bump(path, s=1):
    st = os.stat(path); os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + s * 10**9))


def test_copy_touched_in_same_pass_as_owner_change_takes_over(env, monkeypatch):
    docs, made = {}, []
    def submit(inp, **kw):
        data = inp.data
        if data in docs:
            return SimpleNamespace(status="duplicate", document_id=docs[data], job_id=None, reason="")
        made.append(1); docs[data] = f"d{len(made)}"
        return SimpleNamespace(status="processing", document_id=docs[data], job_id="j", reason="")
    def remove(document_id, profile_id, **kw):
        for k in [k for k, v in docs.items() if v == document_id]:
            del docs[k]
        return True
    monkeypatch.setattr("superlocalmemory.documents.submit_document", submit)
    monkeypatch.setattr("superlocalmemory.documents.remove_document", remove)
    a = env.write("a.pdf", PDF); b = env.write("b.pdf", PDF)
    sid = env.add_and_confirm(); env.scan(sid)
    a.write_bytes(PDF + b"changed"); bump(a, 1)
    bump(b, 1)  # b touched, same bytes (git checkout, sync tool)
    env.scan(sid); env.scan(sid); env.scan(sid)
    f = env.files(sid)["b.pdf"]
    print("b:", f["state"], f["reason"], f["document_id"], f["memory_ids_json"], "live docs:", docs)
    assert f["reason"] is None and f["document_id"] and f["document_id"] in docs.values()

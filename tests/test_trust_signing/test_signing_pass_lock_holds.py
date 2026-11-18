# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7 writer-lock audit: the signing pass lets go of the writer lock between its batches.

It used to take one hold for up to five batches of 200 signatures; a save arriving meanwhile
waited for all of them. One hold per batch bounds that wait to one.
"""

from __future__ import annotations

from superlocalmemory.trust import signatures as sg
from superlocalmemory.trust import signing_key as sk
from tests.test_trust_signing.conftest import SIGNING, save


def test_signing_takes_one_lock_hold_per_batch(db, signer, monkeypatch) -> None:
    monkeypatch.setenv(SIGNING, "on")
    sk.install(signer)
    monkeypatch.setattr(sg, "BATCH", 2)
    sk.install(None)
    for i in range(5):
        save(db, f"Saved by a hook, item {i} of the list {i * 3}.")
    sk.install(signer)
    holds: list[int] = []
    real = db.raw_connection

    def counting():
        holds.append(len(holds))
        return real()

    monkeypatch.setattr(db, "raw_connection", counting)
    batches: list[int] = []
    real_sign = sg.sign_missing

    def sign(conn, who, *, limit=sg.BATCH):
        batches.append(len(holds))  # the lock hold this batch ran in
        return real_sign(conn, who, limit=limit)

    monkeypatch.setattr(sg, "sign_missing", sign)
    assert sg.maintenance_step(db) == 5
    assert len(batches) == len(set(batches)) == 3  # 2 + 2 + 1, each in its own hold

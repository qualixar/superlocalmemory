"""A picture the user saves is never merged into a folder-owned one."""

from __future__ import annotations

from tests.test_sources.test_real_picture import media_rows, pics, png  # noqa: F401


def test_user_picture_survives_the_folder_copy_being_deleted_and_purged(pics, tmp_path):
    from superlocalmemory.media.ingest import MediaInput, remember_media

    env = pics
    path = env.write("p.png", png("a"))
    env.write("keep.canvas", "{}")
    sid = env.add_and_confirm()
    env.scan(sid)
    mine = tmp_path / "mine.png"
    mine.write_bytes(png("a"))
    rec = remember_media(MediaInput(path=mine, file_name="mine.png"), profile_id="default",
                         actor_id="user", runtime=env.runtime, config=None)
    assert rec.status == "stored" and rec.media_id and rec.memory_id
    path.unlink()
    env.scan(sid)
    env.host.purge_after_s = -1.0
    env.scan(sid)
    archived = {r[0] for r in env.db.execute("SELECT memory_id FROM atomic_facts WHERE lifecycle='archived'")}
    assert rec.memory_id not in archived
    assert media_rows(env)[rec.media_id] == "active"

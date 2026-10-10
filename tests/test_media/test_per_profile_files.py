"""Stored originals belong to one profile: the same bytes saved by two profiles are two files."""

from __future__ import annotations

from tests.test_media.test_ingest import env, png, root, save, store  # noqa: F401  (fixtures)
from superlocalmemory.media import erasure, files


def _paths(env, profile):
    row = env.store.find_by_sha(profile, _plain(env, profile))
    return row, env.root / "media" / row["original_relpath"]


def _plain(env, profile):
    import hashlib

    return hashlib.sha256(env.data).hexdigest()


def test_same_bytes_in_two_profiles_make_two_files(env):
    env.data = png("same")
    assert save(env, env.data, profile_id="A").status == "stored"
    assert save(env, env.data, profile_id="B").status == "stored"
    (row_a, path_a), (row_b, path_b) = _paths(env, "A"), _paths(env, "B")
    assert path_a != path_b and path_a.is_file() and path_b.is_file()
    assert row_a["source_sha256"] == row_b["source_sha256"] == _plain(env, "A")


def test_the_same_bytes_in_one_profile_stay_one_file(env):
    env.data = png("same")
    first = save(env, env.data, profile_id="A")
    again = save(env, env.data, profile_id="A")
    assert again.status == "duplicate" and again.media_id == first.media_id
    assert len(list((env.root / "media").rglob("*.png"))) == 1


def test_erasing_a_removes_only_a_file(env):
    env.data = png("same")
    save(env, env.data, profile_id="A")
    save(env, env.data, profile_id="B")
    (_, path_a), (_, path_b) = _paths(env, "A"), _paths(env, "B")
    erasure.erase_profile(env.root, "A")
    assert not path_a.exists() and path_b.is_file()
    assert env.store.find_by_sha("B", _plain(env, "B")) is not None


def test_the_address_depends_on_the_profile(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert files.content_address("A", p) != files.content_address("B", p)
    assert files.content_address("A", p) == files.content_address("A", p)


def test_an_old_row_keeps_its_recorded_path(env):
    """A row written under the earlier plain-hash address still reads and erases by its recorded path."""
    import hashlib

    data = png("old")
    sha = hashlib.sha256(data).hexdigest()
    old = env.root / "media" / sha[:2] / f"{sha}.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"x")
    item = env.store.insert_item(profile_id="A", kind="image", source_sha256=sha, stored_sha256=sha, mime="image/png",
                                 bytes=1, origin="tool", original_relpath=f"{sha[:2]}/{sha}.png")
    erasure.erase_items(env.store, env.root, [item])
    assert not old.exists()

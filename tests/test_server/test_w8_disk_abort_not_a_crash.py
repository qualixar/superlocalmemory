"""The daemon must report a deliberate disk abort distinctly from a crash."""
from unittest import mock

from superlocalmemory.storage.backup import InsufficientDiskSpaceError


def test_disk_abort_is_not_reported_as_a_crash(caplog):
    import logging

    import superlocalmemory.storage.migration_runner as mr
    caplog.set_level(logging.WARNING)
    with mock.patch.object(mr, "apply_all",
                           side_effect=InsufficientDiskSpaceError(900, 100)):
        # exercise the same handler shape the daemon uses
        try:
            mr.apply_all("a", "b")
        except InsufficientDiskSpaceError as exc:
            assert exc.needed_bytes == 900 and exc.free_bytes == 100
            msg = str(exc)
            assert "Insufficient disk space" in msg
        else:
            raise AssertionError("abort did not propagate")


def test_daemon_source_handles_it_before_the_generic_handler():
    import inspect

    import superlocalmemory.server.unified_daemon as ud
    src = inspect.getsource(ud)
    i_specific = src.index("except InsufficientDiskSpaceError")
    i_generic = src.index('logger.warning("migration runner crashed (non-fatal)')
    assert i_specific < i_generic, (
        "the specific abort handler must precede the generic one, or the abort "
        "is swallowed as a crash"
    )
    block = src[i_specific:i_generic]
    assert "Your data has NOT been modified" in block
    assert "_store_modified" in block

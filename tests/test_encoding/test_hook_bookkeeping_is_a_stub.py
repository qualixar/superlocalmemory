"""The hooks' own bookkeeping is never an answer to a recall question.

Measured on the author's store (5,443 facts in 7 days): 291 bare "session ended"
rows, and about 1,056 hook-bookkeeping memories in all, and ``is_stub`` only
knew the bare ``session ended <date> <time>`` shape. Stop has written
``[<project>] session ended <date> <time>`` (with ``| files: ...``) since 4.1.21,
and the Write/Edit hook writes ``File changed: <name>``. The fixtures below are
produced by running those hooks, so a later change to either format breaks
this file instead of quietly un-flagging the rows.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.encoding.non_answer import is_stub
from superlocalmemory.hooks.hook_handlers import handle_hook


def _posted_contents(mock_post) -> list[str]:
    return [c.args[1]['content'] for c in mock_post.call_args_list
            if c.args and c.args[0] == '/remember']


def _run_checkpoint(path: str) -> list[str]:
    with patch('superlocalmemory.hooks.hook_handlers._daemon_post') as post, \
            patch('sys.stdin', io.StringIO(json.dumps({'tool_input': {'file_path': path}}))), \
            patch('sys.stdin.isatty', return_value=False):
        with pytest.raises(SystemExit):
            handle_hook('checkpoint')
    return _posted_contents(post)


def _run_stop(project_dir: str, monkeypatch) -> list[str]:
    monkeypatch.setenv('CLAUDE_PROJECT_DIR', project_dir)
    with patch('superlocalmemory.hooks.hook_handlers._daemon_post') as post, \
            patch('superlocalmemory.hooks.hook_handlers._maybe_consolidate'), \
            patch('superlocalmemory.hooks.hook_handlers._captures_session_summary',
                  return_value=True), \
            patch('superlocalmemory.hooks.hook_handlers.subprocess.run',
                  return_value=MagicMock(stdout='', returncode=0)):
        with pytest.raises(SystemExit):
            handle_hook('stop')
    return _posted_contents(post)


@pytest.fixture
def hook_outputs(monkeypatch, tmp_path):
    """What the real Write/Edit and Stop hooks emit for one short session."""
    import os
    import tempfile
    from superlocalmemory.hooks import hook_handlers as hh

    for name in os.listdir(tempfile.gettempdir()):
        if name.startswith('slm-obs-'):
            os.remove(os.path.join(tempfile.gettempdir(), name))
    changed = [c for p in (f'{tmp_path}/NOTICE', f'{tmp_path}/formats.md')
               for c in _run_checkpoint(p)]
    with_files = _run_stop('/work/Agentic_official', monkeypatch)
    bare = _run_stop('/work/Agentic_official', monkeypatch)  # activity log now cleaned
    assert hh is not None
    return {'changed': changed, 'with_files': with_files, 'bare': bare}


def test_the_stop_hook_summary_is_a_stub_bare_and_with_files(hook_outputs):
    assert hook_outputs['with_files'], 'the Stop hook wrote nothing'
    assert hook_outputs['bare'], 'the Stop hook wrote nothing'
    assert 'files: ' in hook_outputs['with_files'][0]
    assert 'files: ' not in hook_outputs['bare'][0]
    for text in hook_outputs['with_files'] + hook_outputs['bare']:
        assert text.startswith('[Agentic_official] session ended ')
        assert is_stub(text), text


def test_the_file_change_hook_line_is_a_stub(hook_outputs):
    assert [t.split(':')[0] for t in hook_outputs['changed']] == ['File changed'] * 2
    for text in hook_outputs['changed']:
        assert is_stub(text), text


@pytest.mark.parametrize('text', [
    '[Agentic_official] session ended 2026-10-13 11:01',
    '[Agentic_official] session ended 2026-10-13 09:54 | files: NOTICE, formats.md',
    '[superlocalmemory] session ended 2026-10-04 18:20 | files: a.py',
    # ingestion derives a copy of the line without its "[project]" prefix
    'session ended 2026-10-13 09:54 | files: NOTICE, formats.md',
    'File changed: NOTICE',
    'File changed: polish-and-review.md',
    'files: index.html, polish-and-review.md',
    '[claude] session ended (stop) at 2026-10-13 11:01 in Agentic_official.',
    '[codex] session ended (exit) at 2026-10-13 11:01 in my project.'
    ' | transcript: /tmp/x.jsonl | session_id: 9f8e7d6c-1234',
    # 4.2.14: ingestion derives this one without its "[agent]" prefix too
    # (7 such rows were injected at one session start on a real store)
    'session ended (exit) at 2026-11-07 13:49 in config.',
    # a derived copy reads "the session ended <date> <time>" (live store, 9 Nov 2026)
    'the session ended 2026-11-04 16:39',
])
def test_known_hook_shapes_are_stubs(text):
    assert is_stub(text)


@pytest.mark.parametrize('text', [
    # a person's sentence that merely mentions these things stays an answer
    'The session ended with a decision to use PostgreSQL.',
    'My session ended early because the build broke, then I fixed it',
    'File changed: the loader now retries three times before it gives up',
    'File changed: NOTICE, and I need to tell legal about the new wording',
    'files: I keep all my tax files in the blue folder',
    '[Agentic_official] session ended 2026-10-13 11:01 | branch: fix/4.2.2'
    ' | recent: 0545bc9a fix: calibration reads the workspace',
    '[claude] session ended (stop) at 2026-10-13 11:01 in Agentic_official.'
    ' | last_action: Shipped the 4.2.1 release and tagged it.',
    'Remember the file changed notice for the landlord',
    'session ended',
])
def test_sentences_and_real_summaries_stay_answer_material(text):
    assert not is_stub(text)

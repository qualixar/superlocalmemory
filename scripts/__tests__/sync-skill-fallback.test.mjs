/**
 * sync-skill-fallback.test.mjs — the no-MCP fragment lands once, in place, and stays put.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { block, demote, withFragment, START, END } from '../sync-skill-fallback.mjs';

const FOOTER = '\n---\n\n*SuperLocalMemory v9.9.9 · Qualixar · AGPL-3.0-or-later*\n';

describe('sync-skill-fallback', () => {
  test('goes before the version footer', () => {
    const out = withFragment(`# Skill\n\nBody.\n${FOOTER}`, 'Use the shell.\n');
    assert.equal(out, `# Skill\n\nBody.\n\n${block('Use the shell.')}\n${FOOTER}`);
  });

  test('is appended when a skill has no footer', () => {
    assert.equal(withFragment('# Skill\n', 'Use the shell.'), `# Skill\n\n${block('Use the shell.')}\n`);
  });

  test('replaces an older copy in place and is idempotent', () => {
    const once = withFragment(`# Skill\n${FOOTER}`, 'old words');
    const twice = withFragment(once, 'new words');
    assert.ok(twice.includes(`${START}\nnew words\n${END}`));
    assert.ok(!twice.includes('old words'));
    assert.equal(withFragment(twice, 'new words'), twice);
    assert.equal(twice.split(START).length, 2);
  });

  test('goes inside the SuperLocalMemory section of a rules document', () => {
    const rules = '<!-- BEGIN SuperLocalMemory v9 -->\n### Recall\nx\n\n<!-- END SuperLocalMemory v9 -->\n\nfooter\n';
    const out = withFragment(rules, demote('## Shell\nuse it'));
    assert.ok(out.indexOf('### Shell') < out.indexOf('<!-- END SuperLocalMemory'));
    assert.ok(out.indexOf('### Shell') > out.indexOf('### Recall'));
  });

  test('refuses an unterminated block', () => {
    assert.throws(() => withFragment(`${START}\nhalf`, 'x'), /unterminated/);
  });
});

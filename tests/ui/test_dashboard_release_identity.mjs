/** The packaged dashboard must identify and brand the shipped V4 release. */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '../..');
const html = readFileSync(
  join(root, 'src/superlocalmemory/ui/index.html'),
  'utf8',
);
const daemon = readFileSync(
  join(root, 'src/superlocalmemory/server/unified_daemon.py'),
  'utf8',
);
const standaloneUi = readFileSync(
  join(root, 'src/superlocalmemory/server/ui.py'),
  'utf8',
);

describe('dashboard release identity', () => {
  it('identifies the dashboard as V4 and declares an absolute SVG favicon', () => {
    assert.match(html, /<title>SuperLocalMemory V4 — Dashboard<\/title>/);
    assert.match(html, /href="\/static\/favicon\.svg"/);
    assert.doesNotMatch(html, /<title>SuperLocalMemory V3 — Dashboard<\/title>/);
  });

  it('provides the conventional favicon route without duplicating the asset', () => {
    assert.match(daemon, /@application\.get\("\/favicon\.ico", include_in_schema=False\)/);
    assert.match(daemon, /RedirectResponse\(url="\/static\/favicon\.svg", status_code=307\)/);
  });

  it('keeps the unified and standalone dashboard fallbacks on the V4 identity', () => {
    for (const server of [daemon, standaloneUi]) {
      assert.match(server, /SuperLocalMemory V4/);
      assert.doesNotMatch(server, /<title>SuperLocalMemory V3<\/title>/);
      assert.match(server, /@application\.get\("\/favicon\.ico", include_in_schema=False\)/);
    }
  });
});

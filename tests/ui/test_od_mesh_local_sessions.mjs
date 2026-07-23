/** Mesh dashboard must not hide live local agent sessions. */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/od-mesh.js'),
  'utf8',
);

describe('mesh session visibility', function () {
  it('loads every mesh participant and labels local sessions truthfully', function () {
    assert.match(source, /mesh\/peers\?view=all/);
    assert.match(source, /local session/);
    assert.match(source, /remote peer/);
    assert.doesNotMatch(source, /mesh\/peers\?view=remote/);
  });
});

/**
 * Graph rendering must remain responsive on large existing memory stores.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/od-graph.js'),
  'utf8',
);

describe('knowledge graph performance budget', function () {
  it('bounds force simulation while retaining the requested render budget', function () {
    assert.match(source, /PHYSICS_MAX_NODES = 160/);
    assert.match(source, /NODES\.filter\(visible\)\.slice\(0, PHYSICS_MAX_NODES\)/);
    assert.match(source, /PRE_SETTLE_TICKS = 24/);
    assert.match(source, /SETTLE_MAX_FRAMES = 180/);
    assert.match(source, /max_nodes=' \+ MAX_NODES/);
  });
});

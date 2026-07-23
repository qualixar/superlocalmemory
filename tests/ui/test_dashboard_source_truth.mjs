/** Dashboard provenance must never fall back to invented percentages. */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/dashboard.js'),
  'utf8',
);

describe('dashboard ingestion source truth', function () {
  it('renders daemon provenance and contains no representative seed split', function () {
    assert.match(source, /stats\.ingestion_sources/);
    assert.doesNotMatch(source, /MCP agents \(Claude, Cursor\)', 62/);
    assert.doesNotMatch(source, /representative percentages/);
    assert.doesNotMatch(source, /syntheticSeries/);
    assert.doesNotMatch(source, /Math\.random/);
    assert.doesNotMatch(source, /plausible trailing/);
  });
});

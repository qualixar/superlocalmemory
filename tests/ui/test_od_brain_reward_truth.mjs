/** Brain reward views must use action_outcomes reward telemetry only. */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/od-brain.js'),
  'utf8',
);

describe('Brain reward truth', function () {
  it('renders reward telemetry and never labels tool events as reward', function () {
    assert.match(source, /beh\.reward_telemetry/);
    assert.match(source, /reward\.timeline/);
    assert.match(source, /Average settled reward/);
    assert.doesNotMatch(source, /buildReward\\(behavioral, dateMap\\)/);
    assert.doesNotMatch(source, /reward signal · last 26 weeks/);
    assert.doesNotMatch(source, /heuristic reward attribution/);
    assert.match(source, /beh\.cross_project_patterns/);
    assert.match(source, /r\.action_type/);
    assert.match(source, /r\.timestamp/);
    assert.doesNotMatch(source, /p\.is_transferable/);
    assert.match(source, /learning\.ranker_phase/);
    assert.match(source, /ranker\.gates/);
    assert.match(source, /ranker\.signals/);
    assert.match(source, /stats\.models_active_verified/);
    assert.doesNotMatch(source, /ML_GATE/);
  });
});

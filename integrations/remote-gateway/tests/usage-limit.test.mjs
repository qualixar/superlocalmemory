import test from 'node:test';
import assert from 'node:assert/strict';
import { secondsUntilUtcMidnight } from '../src/usage-limit.ts';

test('daily limit resets at the next UTC midnight', () => {
  assert.equal(secondsUntilUtcMidnight(Date.UTC(2026, 9, 8, 23, 59, 30)), 30);
  assert.equal(secondsUntilUtcMidnight(Date.UTC(2026, 9, 8, 0, 0, 0)), 86400);
  assert.equal(secondsUntilUtcMidnight(Date.UTC(2026, 9, 8, 12, 0, 0, 500)), 43200);
});

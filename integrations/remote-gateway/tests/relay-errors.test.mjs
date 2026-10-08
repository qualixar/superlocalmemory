import test from 'node:test';
import assert from 'node:assert/strict';
import { publicRelayCode } from '../src/relay-errors.ts';

test('each relay transport failure keeps its own public code', () => {
  for (const code of ['connector_offline','connector_closed','connector_error','connector_replaced','stale_connector','relay_timeout','relay_busy','origin_timeout','request_cancelled','connection_revoked','connector_unavailable']) {
    assert.equal(publicRelayCode({ error: code }), code);
  }
});

test('unknown or malformed relay bodies never echo arbitrary text', () => {
  assert.equal(publicRelayCode({ error: 'SECRET token abc' }), 'origin_unavailable');
  assert.equal(publicRelayCode({ error: 42 }), 'origin_unavailable');
  assert.equal(publicRelayCode('connector_closed'), 'origin_unavailable');
  assert.equal(publicRelayCode(null), 'origin_unavailable');
});

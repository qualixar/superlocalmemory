import test from 'node:test';
import assert from 'node:assert/strict';
import { createHmac } from 'node:crypto';
import { GRANT_HEADER, GRANT_SCOPE_ORDER, signGrant, importGrantKey, wrapGrantKey, unwrapGrantKey } from '../src/grant.ts';
const KEY = Uint8Array.from({ length: 32 }, (_, i) => i + 1);
const VECTOR = 'v1.eyJ2IjoxLCJraWQiOjEsImNpZCI6ImMwZmZlZTAwYzBmZmVlMDBjMGZmZWUwMGMwZmZlZTAwIiwiYWlkIjoiYXV0aC0xIiwidmVyIjoxLCJhcHAiOiJjbGllbnQtMSIsInNjcCI6WyJzbG06cmVhZCIsInNsbTptZXNoIl0sImZ2IjpmYWxzZSwiZmlkIjoiZnJhbWUtMSIsImdlbiI6MywiZGwiOjE3MDAwMDAwMDAwMDB9.OhBbGApiFGlFV_pL3osXXRfjieznBJJr5_fn_fXkhH0';
const PAYLOAD = '{"v":1,"kid":1,"cid":"c0ffee00c0ffee00c0ffee00c0ffee00","aid":"auth-1","ver":1,"app":"client-1","scp":["slm:read","slm:mesh"],"fv":false,"fid":"frame-1","gen":3,"dl":1700000000000}';
const input = { aid: 'auth-1', ver: 1, app: 'client-1', scp: ['slm:read', 'slm:mesh'], fv: false };
const frame = { cid: 'c0ffee00c0ffee00c0ffee00c0ffee00', fid: 'frame-1', gen: 3, dl: 1700000000000 };
const WRAP = 'ab'.repeat(32);
test('the header name and scope order are fixed', () => { assert.equal(GRANT_HEADER, 'x-slm-grant'); assert.deepEqual([...GRANT_SCOPE_ORDER], ['slm:read', 'slm:write', 'slm:session', 'slm:mesh', 'slm:media']); });
test('the shared test vector signs to the exact header both sides assert', async () => { assert.equal(await signGrant(await importGrantKey(KEY), 1, input, frame), VECTOR); });
test('the vector payload is the exact compact JSON and the mac covers the payload segment', () => {
  const [version, payload, mac] = VECTOR.split('.');
  assert.equal(version, 'v1'); assert.equal(Buffer.from(payload, 'base64url').toString(), PAYLOAD);
  assert.equal(createHmac('sha256', KEY).update('slm-grant-v1.' + payload).digest('base64url'), mac);
});
test('scopes are written in canonical order without duplicates whatever the input order', async () => {
  const header = await signGrant(await importGrantKey(KEY), 1, { ...input, scp: ['slm:media', 'slm:read', 'slm:mesh', 'slm:read', 'slm:write'] }, frame);
  assert.deepEqual(JSON.parse(Buffer.from(header.split('.')[1], 'base64url').toString()).scp, ['slm:read', 'slm:write', 'slm:mesh', 'slm:media']);
});
test('output is unpadded base64url within the verifier shape', async () => {
  const header = await signGrant(await importGrantKey(KEY), 7, { ...input, app: 'Zażółć ?>~ app' }, frame);
  assert.match(header, /^v1\.[A-Za-z0-9_-]{1,8000}\.[A-Za-z0-9_-]{43}$/);
  assert.equal(JSON.parse(Buffer.from(header.split('.')[1], 'base64url').toString()).app, 'Zażółć ?>~ app');
});
test('every bound field changes the mac', async () => {
  const key = await importGrantKey(KEY); const mac = async (i = input, f = frame, kid = 1) => (await signGrant(key, kid, i, f)).split('.')[2];
  const base = await mac();
  for (const other of [await mac(input, { ...frame, fid: 'frame-2' }), await mac(input, { ...frame, gen: 4 }), await mac(input, { ...frame, dl: 1 }), await mac(input, { ...frame, cid: 'other' }), await mac({ ...input, ver: 2 }), await mac(input, frame, 2)]) assert.notEqual(other, base);
});
test('a key must be exactly 32 bytes', async () => { await assert.rejects(importGrantKey(new Uint8Array(31))); await assert.rejects(importGrantKey(new Uint8Array(33))); });
test('a wrapped key round trips and is stored as version, iv and ciphertext', async () => {
  const stored = await wrapGrantKey(KEY, WRAP, 4);
  assert.deepEqual(Object.keys(stored).sort(), ['ct', 'iv', 'v', 'version']); assert.equal(stored.v, 1); assert.equal(stored.version, 4);
  assert.match(stored.iv, /^[A-Za-z0-9_-]{16}$/); assert.ok(!JSON.stringify(stored).includes(Buffer.from(KEY).toString('base64url')));
  assert.deepEqual([...await unwrapGrantKey(stored, WRAP)], [...KEY]);
});
test('each wrap uses a fresh iv', async () => { assert.notEqual((await wrapGrantKey(KEY, WRAP, 1)).iv, (await wrapGrantKey(KEY, WRAP, 1)).iv); });
test('a wrong wrap key, a tampered record or a moved version fails to unwrap', async () => {
  const stored = await wrapGrantKey(KEY, WRAP, 2);
  await assert.rejects(unwrapGrantKey(stored, 'cd'.repeat(32)));
  await assert.rejects(unwrapGrantKey({ ...stored, version: 3 }, WRAP));
  const flipped = Buffer.from(stored.ct, 'base64url'); flipped[0] ^= 1;
  await assert.rejects(unwrapGrantKey({ ...stored, ct: flipped.toString('base64url') }, WRAP));
});
test('a malformed wrap secret or record is refused', async () => {
  await assert.rejects(wrapGrantKey(KEY, 'short', 1)); await assert.rejects(unwrapGrantKey({ v: 2, version: 1, iv: 'x', ct: 'y' }, WRAP));
});

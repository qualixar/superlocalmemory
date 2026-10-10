import { evictDurableObject, runInDurableObject } from 'cloudflare:test';
import { env } from 'cloudflare:workers';
import { afterEach, describe, expect, test } from 'vitest';
import type { RelayDO } from '../../src/relay-do.ts';
import type { RelayFrame } from '../../src/relay-protocol.ts';
import { decodeRelayFrame, encodeRelayFrame } from '../../src/relay-protocol.ts';
type RequestFrame = Extract<RelayFrame, { kind: 'request' }>;
const sockets: WebSocket[] = [];
afterEach(() => { for (const ws of sockets.splice(0)) { try { ws.close(); } catch {} } });
const grant = { aid: 'auth-1', ver: 3, app: 'client-1', scp: ['slm:read', 'slm:mesh'] as const, fv: false };
async function setup(namespace: DurableObjectNamespace<RelayDO> = env.RELAYS) {
  const token = crypto.randomUUID().replaceAll('-', '') + crypto.randomUUID().replaceAll('-', '');
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token))), x => x.toString(16).padStart(2, '0')).join('');
  const stub = namespace.getByName(crypto.randomUUID());
  const binding = { ownerId: 'owner-a', connectionId: 'c0ffee00c0ffee00c0ffee00c0ffee00', installationId: 'installation-a', profileId: 'profile-a', deviceDigest: digest, deviceExpiresAt: Date.now() + 600000 };
  await stub.configureBinding(binding); return { stub, token, binding };
}
/** Connects a laptop socket and queues every request frame it receives. */
async function laptop(stub: DurableObjectStub<RelayDO>, token: string, features: string | null = 'grant-v1') {
  const response = await stub.fetch(new Request('https://private.invalid/connector', { headers: { Upgrade: 'websocket', Authorization: 'Bearer ' + token, ...(features === null ? {} : { 'x-slm-connector-features': features }) } }));
  expect(response.status).toBe(101); const ws = response.webSocket!; sockets.push(ws);
  const ready = new Promise<{ generation: number }>(resolve => ws.addEventListener('message', e => resolve(JSON.parse(String(e.data))), { once: true })); ws.accept();
  const { generation } = await ready; const queue: RequestFrame[] = []; const waiters: Array<(f: RequestFrame) => void> = [];
  ws.addEventListener('message', e => { const d = decodeRelayFrame(String(e.data)); if (!d.ok || d.frame.kind !== 'request') return; const w = waiters.shift(); if (w) w(d.frame); else queue.push(d.frame); });
  const next = () => queue.length ? Promise.resolve(queue.shift()!) : new Promise<RequestFrame>(resolve => waiters.push(resolve));
  const reply = (f: RequestFrame, body = 'ok') => { const r = encodeRelayFrame({ v: 1, kind: 'response', id: f.id, generation: f.generation, status: 200, headers: [['Content-Type', 'application/json']], bodyBase64: btoa(body) }); if (!r.ok) throw new Error('bad fixture'); ws.send(r.text); };
  return { ws, generation, next, reply };
}
function frame(generation: number, headers: Array<[string, string]> = [['Content-Type', 'application/json']], timeout = 2000): RequestFrame {
  return { v: 1, kind: 'request', id: crypto.randomUUID(), generation, deadlineAt: Date.now() + timeout, headers, bodyBase64: btoa('{"jsonrpc":"2.0","id":1,"method":"tools/list"}') };
}
const header = (f: RequestFrame) => f.headers.filter(p => p[0].toLowerCase() === 'x-slm-grant');
function b64uToBytes(text: string) { return Uint8Array.from(atob(text.replaceAll('-', '+').replaceAll('_', '/')), c => c.charCodeAt(0)); }
/** Independent verifier: recomputes the mac over the received payload segment. */
async function verify(value: string, rawKey: string) {
  const [version, payload, mac] = value.split('.'); expect(version).toBe('v1');
  const key = await crypto.subtle.importKey('raw', b64uToBytes(rawKey), { name: 'HMAC', hash: 'SHA-256' }, false, ['verify']);
  expect(await crypto.subtle.verify('HMAC', key, b64uToBytes(mac), new TextEncoder().encode('slm-grant-v1.' + payload))).toBe(true);
  return JSON.parse(new TextDecoder().decode(b64uToBytes(payload))) as Record<string, unknown>;
}
describe('per-request grant', () => {
  test('a relay holding a key signs each forwarded frame, bound to the wire id the laptop sees', async () => {
    const { stub, token, binding } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const rotated = await stub.rotateGrantKey('owner-a'); expect(rotated.version).toBe(1); expect(rotated.key).toMatch(/^[A-Za-z0-9_-]{43}$/);
    const f = frame(generation); const result = stub.forward(f, {}, { grant }); const seen = await next();
    expect(header(seen)).toHaveLength(1); expect(seen.id).not.toBe(f.id);
    const claims = await verify(header(seen)[0][1], rotated.key);
    expect(claims).toEqual({ v: 1, kid: 1, cid: binding.connectionId, aid: 'auth-1', ver: 3, app: 'client-1', scp: ['slm:read', 'slm:mesh'], fv: false, fid: seen.id, gen: generation, dl: f.deadlineAt });
    expect(Object.keys(claims)).toEqual(['v', 'kid', 'cid', 'aid', 'ver', 'app', 'scp', 'fv', 'fid', 'gen', 'dl']);
    reply(seen); expect((await result).status).toBe(200);
  });
  test('a laptop that did not say it reads grants gets none, even when a key exists', async () => {
    const { stub, token } = await setup(); await stub.rotateGrantKey('owner-a'); const { generation, next, reply } = await laptop(stub, token, null);
    const result = stub.forward(frame(generation), {}, { grant }); const seen = await next(); expect(header(seen)).toHaveLength(0); reply(seen); expect((await result).status).toBe(200);
  });
  test('only the exact grant-v1 feature value counts', async () => {
    for (const value of ['grant-v2', 'grant-v1,x', 'GRANT-V1', 'grant-v1;x', '']) {
      const { stub, token } = await setup(); await stub.rotateGrantKey('owner-a'); const { generation, next, reply } = await laptop(stub, token, value);
      const result = stub.forward(frame(generation), {}, { grant }); const seen = await next(); expect(header(seen), value).toHaveLength(0); reply(seen); await result;
    }
  });
  test('a laptop that sends the feature value gets signed frames, also after a restart', async () => {
    const { stub, token } = await setup(); const rotated = await stub.rotateGrantKey('owner-a'); const { generation, next, reply } = await laptop(stub, token, 'grant-v1');
    await evictDurableObject(stub);
    const result = stub.forward(frame(generation), {}, { grant }); const seen = await next(); await verify(header(seen)[0][1], rotated.key); reply(seen); await result;
  });
  test('a rotation that lands while the first key is being unwrapped is not overwritten', async () => {
    const { stub, token } = await setup(); await stub.rotateGrantKey('owner-a'); const { generation } = await laptop(stub, token);
    await evictDurableObject(stub);
    const outcome = await runInDurableObject(stub, async instance => {
      const subtle = crypto.subtle; const realDecrypt = subtle.decrypt.bind(subtle); let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
      (subtle as { decrypt: unknown }).decrypt = async (...args: Parameters<typeof realDecrypt>) => { await gate; return realDecrypt(...args); };
      try {
        const reading = (instance as unknown as { signingKey(): Promise<{ version: number }|null> }).signingKey();
        const second = await instance.rotateGrantKey('owner-a'); release(); const seen = await reading;
        const after = await (instance as unknown as { signingKey(): Promise<{ version: number }|null> }).signingKey();
        return { rotated: second.version, seen: seen?.version, after: after?.version };
      } finally { (subtle as { decrypt: unknown }).decrypt = realDecrypt; release(); }
    });
    expect(generation).toBeGreaterThan(0); expect(outcome).toEqual({ rotated: 2, seen: 2, after: 2 });
  });
  test('every forward gets its own frame id and its own mac', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token); await stub.rotateGrantKey('owner-a');
    const a = stub.forwardCurrent({ v: 1, kind: 'request', id: 'a', deadlineAt: Date.now() + 2000, headers: [], bodyBase64: '' }, {}, { grant }); const fa = await next();
    const b = stub.forwardCurrent({ v: 1, kind: 'request', id: 'b', deadlineAt: Date.now() + 2000, headers: [], bodyBase64: '' }, {}, { grant }); const fb = await next();
    expect(header(fa)[0][1]).not.toBe(header(fb)[0][1]); reply(fa); reply(fb); await a; await b; expect(generation).toBeGreaterThan(0);
  });
  test('without a key no grant header is added, which keeps older laptops unaffected', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const result = stub.forward(frame(generation), {}, { grant }); const seen = await next(); expect(header(seen)).toHaveLength(0); reply(seen); expect((await result).status).toBe(200);
  });
  test('a call that carries no grant context is forwarded without a header even when a key exists', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token); await stub.rotateGrantKey('owner-a');
    const result = stub.forward(frame(generation)); const seen = await next(); expect(header(seen)).toHaveLength(0); reply(seen); await result;
  });
  test('a grant header supplied by the caller is dropped, whether or not a key exists', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const forged: Array<[string, string]> = [['Content-Type', 'application/json'], ['X-SLM-Grant', 'v1.forged.forged']];
    let result = stub.forward(frame(generation, forged), {}, { grant }); let seen = await next(); expect(header(seen)).toHaveLength(0); reply(seen); await result;
    const rotated = await stub.rotateGrantKey('owner-a');
    result = stub.forward(frame(generation, forged)); seen = await next(); expect(header(seen)).toHaveLength(0); reply(seen); await result;
    result = stub.forward(frame(generation, forged), {}, { grant }); seen = await next(); expect(header(seen)).toHaveLength(1);
    expect(header(seen)[0][1]).not.toContain('forged'); await verify(header(seen)[0][1], rotated.key); reply(seen); await result;
  });
  test('rotation starts at version one, increments, changes the key and signs with the newest', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const first = await stub.rotateGrantKey('owner-a'); const second = await stub.rotateGrantKey('owner-a');
    expect([first.version, second.version]).toEqual([1, 2]); expect(second.key).not.toBe(first.key);
    const result = stub.forward(frame(generation), {}, { grant }); const seen = await next();
    expect((await verify(header(seen)[0][1], second.key)).kid).toBe(2); reply(seen); await result;
  });
  test('the key survives a restart and is stored wrapped, never in clear', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const rotated = await stub.rotateGrantKey('owner-a'); await evictDurableObject(stub);
    const stored = await runInDurableObject(stub, async (_i, state) => state.storage.get<Record<string, unknown>>('grant-key'));
    expect(Object.keys(stored!).sort()).toEqual(['ct', 'iv', 'v', 'version']); expect(JSON.stringify(stored)).not.toContain(rotated.key);
    const result = stub.forward(frame(generation), {}, { grant }); const seen = await next(); await verify(header(seen)[0][1], rotated.key); reply(seen); await result;
    expect((await stub.rotateGrantKey('owner-a')).version).toBe(2);
  });
  test('revocation deletes the key', async () => {
    const { stub } = await setup(); await stub.rotateGrantKey('owner-a'); await stub.revoke();
    expect(await runInDurableObject(stub, async (_i, state) => state.storage.get('grant-key'))).toBeUndefined();
    await evictDurableObject(stub); expect(await runInDurableObject(stub, async (_i, state) => state.storage.get('grant-key'))).toBeUndefined();
  });
  test('rotation is refused once revoked, for another owner, before binding and without a wrap key', async () => {
    const { stub } = await setup(); await expect((async () => { await stub.rotateGrantKey('owner-b'); })()).rejects.toThrow('owner_mismatch');
    await expect((async () => { await env.RELAYS.getByName(crypto.randomUUID()).rotateGrantKey('owner-a'); })()).rejects.toThrow('connection_unconfigured');
    const bare = await setup(env.RELAYS_NOWRAP); await expect((async () => { await bare.stub.rotateGrantKey('owner-a'); })()).rejects.toThrow('grant_unavailable');
    const { token } = bare; const { generation, next, reply } = await laptop(bare.stub, token);
    const result = bare.stub.forward(frame(generation), {}, { grant }); const seen = await next(); expect(header(seen)).toHaveLength(0); reply(seen); await result;
    await stub.revoke(); await expect((async () => { await stub.rotateGrantKey('owner-a'); })()).rejects.toThrow('connection_revoked');
  });
  test('a refused rotation keeps the earlier key and version', async () => {
    const { stub } = await setup(); await stub.rotateGrantKey('owner-a');
    await expect((async () => { await stub.rotateGrantKey('owner-b'); })()).rejects.toThrow('owner_mismatch');
    expect((await stub.rotateGrantKey('owner-a')).version).toBe(2);
  });
});
describe('concurrent waits', () => {
  const wait = { grant, wait: true };
  test('a third concurrent wait is refused while an ordinary call still passes', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const w1 = stub.forward(frame(generation), {}, wait); const f1 = await next(); const w2 = stub.forward(frame(generation), {}, wait); const f2 = await next();
    const third = await stub.forward(frame(generation), {}, wait); expect(third.status).toBe(429); expect(await third.json()).toEqual({ error: 'relay_busy' });
    const normal = stub.forward(frame(generation), {}, { grant }); const f3 = await next(); reply(f3); expect((await normal).status).toBe(200);
    reply(f1); reply(f2); await w1; await w2;
  });
  test('a slot is released when the laptop replies', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const w1 = stub.forward(frame(generation), {}, wait); const f1 = await next(); const w2 = stub.forward(frame(generation), {}, wait); const f2 = await next();
    reply(f1); expect((await w1).status).toBe(200);
    const w3 = stub.forward(frame(generation), {}, wait); const f3 = await next(); reply(f3); reply(f2); expect((await w3).status).toBe(200); await w2;
  });
  test('a slot is released when a wait times out', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token);
    const w1 = stub.forward(frame(generation, undefined, 150), {}, wait); await next(); const w2 = stub.forward(frame(generation), {}, wait); const f2 = await next();
    expect((await stub.forward(frame(generation), {}, wait)).status).toBe(429);
    expect((await w1).status).toBe(504);
    const w3 = stub.forward(frame(generation), {}, wait); const f3 = await next(); reply(f3); reply(f2); expect((await w3).status).toBe(200); await w2;
  });
  test('calls that are not waits never use a wait slot, and the total cap of eight is unchanged', async () => {
    const { stub, token } = await setup(); const { generation, next, reply } = await laptop(stub, token); const open: Array<[Promise<Response>, RequestFrame]> = [];
    for (let i = 0; i < 8; i++) { const p = stub.forward(frame(generation), {}, { grant }); open.push([p, await next()]); }
    expect((await stub.forward(frame(generation), {}, { grant })).status).toBe(429);
    for (const [p, f] of open) { reply(f); await p; }
  });
});

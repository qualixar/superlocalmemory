import { env } from 'cloudflare:workers';
import { createExecutionContext as context, waitOnExecutionContext } from 'cloudflare:test';
import { afterEach, expect, test } from 'vitest';
import { resourceGateway } from '../../src/worker-resource.ts';
import type { ResourceEnv } from '../../src/worker-resource.ts';
import { decodeRelayFrame, encodeRelayFrame } from '../../src/relay-protocol.ts';
import type { RelayFrame } from '../../src/relay-protocol.ts';
type RequestFrame = Extract<RelayFrame, { kind: 'request' }>;
const sockets: WebSocket[] = [];
afterEach(() => { for (const s of sockets.splice(0)) { try { s.close(); } catch {} } });
const audience = 'https://mcp.superlocalmemory.com/mcp';
function bytes(text: string) { return Uint8Array.from(atob(text.replaceAll('-', '+').replaceAll('_', '/')), c => c.charCodeAt(0)); }
async function setup(withKey: boolean) {
  const id = crypto.randomUUID().replaceAll('-', ''); const token = 'synthetic-device-token-'.repeat(3);
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token))), b => b.toString(16).padStart(2, '0')).join('');
  const registry = env.REGISTRIES.getByName(id);
  await registry.configure({ connectionId: id, ownerId: 'owner-a', installationId: 'installation-a', profileId: 'profile-a', origin: { kind: 'relay', installationId: 'installation-a', profileId: 'profile-a' }, allowedTools: ['recall'], allowCorrection: false, allowSharedRead: false, allowGlobalRead: false, policyVersion: 1, revokedAt: null });
  await registry.addAuthorization({ authorizationId: 'authorization-a', audience, ownerId: 'owner-a', clientId: 'client-a', connectionId: id, consentedTools: ['recall', 'mesh_inbox', 'mesh_wait'], consentedScopes: ['slm:read', 'slm:mesh'], consentedCorrection: false, consentedSharedRead: false, consentedGlobalRead: false, authorizationVersion: 4, revokedAt: null });
  await registry.setEntitlement('owner-a', Date.now() + 60000, 0);
  const relay = env.RELAYS.getByName(id); await relay.configureBinding({ ownerId: 'owner-a', connectionId: id, installationId: 'installation-a', profileId: 'profile-a', deviceDigest: digest, deviceExpiresAt: Date.now() + 60000 });
  const key = withKey ? (await relay.rotateGrantKey('owner-a')).key : null;
  const upgrade = await relay.fetch(new Request('https://private.invalid/connector', { headers: { Upgrade: 'websocket', Authorization: 'Bearer ' + token, 'x-slm-connector-features': 'grant-v1' } }));
  const socket = upgrade.webSocket!; sockets.push(socket); const ready = new Promise<string>(r => socket.addEventListener('message', e => r(String(e.data)), { once: true })); socket.accept(); await ready;
  const frames: RequestFrame[] = []; const waiting: Array<() => void> = [];
  socket.addEventListener('message', e => { const d = decodeRelayFrame(String(e.data)); if (d.ok && d.frame.kind === 'request') { frames.push(d.frame); waiting.shift()?.(); } });
  const arrived = async (n: number) => { while (frames.length < n) await new Promise<void>(r => { waiting.push(r); }); };
  const reply = (f: RequestFrame) => { const r = encodeRelayFrame({ v: 1, kind: 'response', id: f.id, generation: f.generation, status: 200, headers: [['content-type', 'application/json']], bodyBase64: btoa(JSON.stringify({ jsonrpc: '2.0', id: JSON.parse(atob(f.bodyBase64)).id, result: { content: [] } })) }); if (r.ok) socket.send(r.text); };
  const auth = { async validateToken(resource: string, token: string) { if (token !== 'synthetic-mesh') return null; return { props: { ownerId: 'owner-a', authorizationId: 'authorization-a', connectionId: id }, audience: resource, scope: ['slm:read', 'slm:mesh'], expiresAt: Math.floor(Date.now() / 1000) + 60, userId: 'owner-a', clientId: 'client-a' }; } };
  return { id, key, frames, arrived, reply, fixtureEnv: { ...env, AUTH_SERVER: auth } as ResourceEnv };
}
async function call(e: ResourceEnv, name: string, args: Record<string, unknown> = {}, headers: Record<string, string> = {}, rpcId = 1) {
  const ctx = context(); const r = await resourceGateway.fetch(new Request('https://mcp.superlocalmemory.com/mcp', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', Authorization: 'Bearer synthetic-mesh', ...headers }, body: JSON.stringify({ jsonrpc: '2.0', id: rpcId, method: 'tools/call', params: { name, arguments: args } }) }), e, ctx); await waitOnExecutionContext(ctx); return r;
}
async function claims(value: string, key: string) {
  const [, payload, mac] = value.split('.'); const k = await crypto.subtle.importKey('raw', bytes(key), { name: 'HMAC', hash: 'SHA-256' }, false, ['verify']);
  expect(await crypto.subtle.verify('HMAC', k, bytes(mac), new TextEncoder().encode('slm-grant-v1.' + payload))).toBe(true); return JSON.parse(new TextDecoder().decode(bytes(payload)));
}
test('an admitted call reaches the laptop with a grant built from the verified token and registry, not from the request', async () => {
  const s = await setup(true); const pending = call(s.fixtureEnv, 'mesh_inbox'); await s.arrived(1); const f = s.frames[0];
  const grant = f.headers.find(p => p[0] === 'x-slm-grant')![1];
  expect(await claims(grant, s.key!)).toEqual({ v: 1, kid: 1, cid: s.id, aid: 'authorization-a', ver: 4, app: 'client-a', scp: ['slm:read', 'slm:mesh'], fv: false, fid: f.id, gen: f.generation, dl: f.deadlineAt });
  s.reply(f); expect((await pending).status).toBe(200);
});
test('every tool call is granted, including ordinary ones', async () => {
  const s = await setup(true); const pending = call(s.fixtureEnv, 'recall', { query: 'x' }); await s.arrived(1); expect(s.frames[0].headers.some(p => p[0] === 'x-slm-grant')).toBe(true); s.reply(s.frames[0]); await pending;
});
test('without a grant key the laptop sees no grant header', async () => {
  const s = await setup(false); const pending = call(s.fixtureEnv, 'recall', { query: 'x' }); await s.arrived(1); expect(s.frames[0].headers.some(p => p[0] === 'x-slm-grant')).toBe(false); s.reply(s.frames[0]); await pending;
});
test('a client that sends x-slm headers is refused before anything reaches the laptop', async () => {
  const s = await setup(true);
  for (const name of ['x-slm-grant', 'X-SLM-Peer', 'x-slm-whatever']) { const r = await call(s.fixtureEnv, 'recall', { query: 'x' }, { [name]: 'forged' }); expect(r.status).toBe(400); expect(await r.json()).toMatchObject({ error: { code: -32020, message: 'HEADER_MISMATCH' } }); }
  expect(s.frames).toHaveLength(0);
});
test('mesh_wait takes a wait slot: a third concurrent wait is refused, ordinary calls are not', async () => {
  const s = await setup(true); const w1 = call(s.fixtureEnv, 'mesh_wait', { timeout_s: 5 }, {}, 1); await s.arrived(1); const w2 = call(s.fixtureEnv, 'mesh_wait', { timeout_s: 5 }, {}, 2); await s.arrived(2);
  const third = await call(s.fixtureEnv, 'mesh_wait', { timeout_s: 5 }, {}, 3); expect(third.status).toBe(429); expect(await third.json()).toEqual({ error: 'relay_busy' });
  const ordinary = call(s.fixtureEnv, 'recall', { query: 'x' }, {}, 4); await s.arrived(3); s.reply(s.frames[2]); expect((await ordinary).status).toBe(200);
  s.reply(s.frames[0]); s.reply(s.frames[1]); await w1; await w2;
});
test('a mesh_wait with an out of range timeout is refused at admission', async () => {
  const s = await setup(true); const r = await call(s.fixtureEnv, 'mesh_wait', { timeout_s: 99 }); expect(r.status).toBe(403); expect(await r.json()).toEqual({ error: 'ARGUMENT_DENIED' }); expect(s.frames).toHaveLength(0);
});

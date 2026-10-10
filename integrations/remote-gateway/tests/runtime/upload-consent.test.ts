import { env } from 'cloudflare:workers';
import { afterEach, describe, expect, test } from 'vitest';
import type { AuthorizationGrant, ConnectionGrant, Scope } from '../../src/contracts.ts';
import type { RelayDO } from '../../src/relay-do.ts';
import type { RelayFrame } from '../../src/relay-protocol.ts';
import { decodeRelayFrame, encodeRelayFrame } from '../../src/relay-protocol.ts';

const audience = 'https://mcp.superlocalmemory.com/mcp';
const NONCE = 'n'.repeat(22);

describe('the registry says whether an upload link may still be used', () => {
  const connection: ConnectionGrant = { connectionId: 'connection-u', ownerId: 'owner-a', installationId: 'installation-a', profileId: 'profile-a', origin: { kind: 'relay', installationId: 'installation-a', profileId: 'profile-a' }, allowedTools: ['recall'], allowCorrection: false, allowSharedRead: false, allowGlobalRead: false, policyVersion: 1, revokedAt: null };
  const grant = (id: string, tools: string[], scopes: Scope[]): AuthorizationGrant => ({ authorizationId: id, audience, ownerId: 'owner-a', clientId: 'client-' + id, connectionId: 'connection-u', consentedTools: tools, consentedScopes: scopes, consentedCorrection: false, consentedSharedRead: false, consentedGlobalRead: false, authorizationVersion: 1, revokedAt: null });
  const full = (id: string) => grant(id, ['recall', 'remember_media', 'media_upload_link'], ['slm:read', 'slm:write', 'slm:media']);
  async function registry(entitled = true) {
    const stub = env.REGISTRIES.getByName(crypto.randomUUID()); await stub.configure(connection);
    if (entitled) await stub.setEntitlement('owner-a', Date.now() + 86400000, 0);
    return stub;
  }

  test('yes only while an active app holds the write and media consent for the upload tool', async () => {
    const stub = await registry(); expect(await stub.uploadsAllowed()).toBe(false);
    await stub.addAuthorization(full('app-a')); expect(await stub.uploadsAllowed()).toBe(true);
  });

  test.each([
    ['no media_upload_link in the consent', ['recall', 'remember_media'], ['slm:read', 'slm:write', 'slm:media'] as Scope[]],
    ['no write scope', ['recall', 'media_upload_link'], ['slm:read', 'slm:media'] as Scope[]],
    ['no media scope', ['recall', 'media_upload_link'], ['slm:read', 'slm:write'] as Scope[]],
  ])('no with %s', async (_label, tools, scopes) => {
    const stub = await registry();
    // media_upload_link needs both scopes to be consentable at all; give the consent only what the case lists.
    await stub.addAuthorization(grant('app-a', tools, scopes)); expect(await stub.uploadsAllowed()).toBe(false);
  });

  test('revoking the app, the connection, or ending access turns it off at once', async () => {
    const stub = await registry(); await stub.addAuthorization(full('app-a')); await stub.addAuthorization(full('app-b'));
    await stub.revokeAuthorization('owner-a', 'app-a', 1); expect(await stub.uploadsAllowed()).toBe(true);
    await stub.revokeAuthorization('owner-a', 'app-b', 1); expect(await stub.uploadsAllowed()).toBe(false);
    const other = await registry(); await other.addAuthorization(full('app-a'));
    await other.revokeConnection('owner-a', 1); expect(await other.uploadsAllowed()).toBe(false);
    const lapsed = await registry(false); await lapsed.addAuthorization(full('app-a')); expect(await lapsed.uploadsAllowed()).toBe(false);
  });

  test('an unconfigured registry says no', async () => {
    expect(await env.REGISTRIES.getByName(crypto.randomUUID()).uploadsAllowed()).toBe(false);
  });
});

describe('only a connector that said upload-v1 is sent upload frames', () => {
  const sockets: WebSocket[] = [];
  afterEach(() => { for (const ws of sockets.splice(0)) { try { ws.close(); } catch {} } });
  async function connect(features: string | null) {
    const token = crypto.randomUUID().replaceAll('-', '') + crypto.randomUUID().replaceAll('-', '');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token))), x => x.toString(16).padStart(2, '0')).join('');
    const stub: DurableObjectStub<RelayDO> = env.RELAYS.getByName(crypto.randomUUID());
    await stub.configureBinding({ ownerId: 'owner-a', connectionId: 'c0ffee00c0ffee00c0ffee00c0ffee00', installationId: 'installation-a', profileId: 'profile-a', deviceDigest: digest, deviceExpiresAt: Date.now() + 600000 });
    const response = await stub.fetch(new Request('https://private.invalid/connector', { headers: { Upgrade: 'websocket', Authorization: 'Bearer ' + token, ...(features === null ? {} : { 'x-slm-connector-features': features }) } }));
    expect(response.status).toBe(101); const ws = response.webSocket!; sockets.push(ws);
    const ready = new Promise<{ generation: number }>(resolve => ws.addEventListener('message', e => resolve(JSON.parse(String(e.data))), { once: true })); ws.accept();
    const { generation } = await ready; const seen: RelayFrame[] = [];
    ws.addEventListener('message', e => { const d = decodeRelayFrame(String(e.data)); if (d.ok && d.frame.kind === 'request') { seen.push(d.frame); const r = encodeRelayFrame({ v: 1, kind: 'response', id: d.frame.id, generation: d.frame.generation, status: 200, headers: [['content-type', 'application/json']], bodyBase64: btoa('{"ok":true}') }); if (r.ok) ws.send(r.text); } });
    return { stub, generation, seen };
  }
  const upload = (generation: number): Extract<RelayFrame, { kind: 'request' }> => ({ v: 1, kind: 'request', id: crypto.randomUUID(), generation, deadlineAt: Date.now() + 2000, headers: [['content-type', 'application/octet-stream'], ['x-slm-upload', `info ${'T'.repeat(43)} 0 0 ${NONCE}`]], bodyBase64: '' });
  const mcp = (generation: number): Extract<RelayFrame, { kind: 'request' }> => ({ v: 1, kind: 'request', id: crypto.randomUUID(), generation, deadlineAt: Date.now() + 2000, headers: [['content-type', 'application/json']], bodyBase64: btoa('{"jsonrpc":"2.0","id":1,"method":"tools/list"}') });

  test.each([['grant-v1,upload-v1']])('%s receives upload frames', async features => {
    const { stub, generation, seen } = await connect(features);
    const r = await stub.forward(upload(generation)); expect(r.status).toBe(200); expect(seen.length).toBe(1);
  });

  test.each([['grant-v1'], ['upload-v1'], ['grant-v1,upload-v1,x'], ['UPLOAD-V1'], [null]])('%s is refused upload frames, which never reach it, but still gets ordinary calls', async features => {
    const { stub, generation, seen } = await connect(features);
    const r = await stub.forward(upload(generation));
    expect(r.status).toBe(503); expect(await r.json()).toEqual({ error: 'upload_unsupported' }); expect(seen).toEqual([]);
    expect((await stub.forward(mcp(generation))).status).toBe(200); expect(seen.length).toBe(1);
  });
});

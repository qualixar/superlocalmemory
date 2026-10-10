import { env } from 'cloudflare:workers';
import { describe, expect, test } from 'vitest';
import { enrolled } from './native-fixture.ts';
const OWNER = '16027584';
async function provisioned() { const f = await enrolled(); expect((await f.provision()).status).toBe(200); return f; }
async function addApp(f: Awaited<ReturnType<typeof enrolled>>, id: string, scopes: string[], tools: string[]) {
  const registered = await f.call('/oauth/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ client_name: 'Grant App', redirect_uris: ['https://backend.composio.dev/api/v1/auth-apps/add'], token_endpoint_auth_method: 'none' }) });
  const client = await registered.json() as { client_id: string };
  await env.REGISTRIES.getByName(f.connectionId).addAuthorization({ ownerId: OWNER, connectionId: f.connectionId, authorizationId: id, clientId: client.client_id, audience: 'https://mcp.superlocalmemory.com/mcp', consentedTools: tools, consentedScopes: scopes as never, consentedCorrection: false, consentedSharedRead: false, consentedGlobalRead: false, authorizationVersion: 1, revokedAt: null });
}
type Row = { permissions: Record<string, boolean> } & Record<string, unknown>;
describe('grant key', () => {
  test('is refused before the computer is linked', async () => {
    const f = await enrolled(); const r = await f.ownerCall('/owner/grant-key', {}); expect(r.status).toBe(403); expect(await r.json()).toEqual({ error: 'connection_unavailable' });
  });
  test('is minted on request, returned once with its version, and replaced on every call', async () => {
    const f = await provisioned();
    const first = await f.ownerCall('/owner/grant-key', {}); expect(first.status).toBe(200); expect(first.headers.get('cache-control')).toBe('no-store');
    const a = await first.json() as { version: number; key: string; connection_id: string };
    expect(Object.keys(a).sort()).toEqual(['connection_id', 'key', 'version']); expect(a.version).toBe(1); expect(a.key).toMatch(/^[A-Za-z0-9_-]{43}$/); expect(a.connection_id).toBe(f.connectionId);
    const b = await (await f.ownerCall('/owner/grant-key', {})).json() as { version: number; key: string };
    expect(b.version).toBe(2); expect(b.key).not.toBe(a.key);
  });
  test('needs exactly an empty JSON object as body', async () => {
    const f = await provisioned();
    for (const body of [undefined, [], { version: 2 }, { x: 1 }, 'text', 5, null]) { const r = await f.ownerCall('/owner/grant-key', body); expect(r.status, JSON.stringify(body)).toBe(400); expect(await r.json()).toEqual({ error: 'invalid_request' }); }
    expect((await (await f.ownerCall('/owner/grant-key', {})).json() as { version: number }).version).toBe(1);
  });
  test('is refused for a cancelled connection', async () => {
    const f = await provisioned(); expect((await f.ownerCall('/owner/revoke')).status).toBe(200);
    const r = await f.ownerCall('/owner/grant-key', {}); expect([401, 403]).toContain(r.status);
  });
  test('reports an unconfigured wrap secret as grant_unavailable', async () => {
    const f = await provisioned(); const owned = await env.OWNERS.getByName(OWNER).getConnection(OWNER, f.connectionId);
    await env.RELAYS_NOWRAP.getByName(f.connectionId).configureBinding({ ownerId: OWNER, connectionId: f.connectionId, installationId: f.installationId, profileId: f.profileId, deviceDigest: owned!.deviceDigest, deviceExpiresAt: owned!.deviceExpiresAtMs });
    (f.configuration as unknown as { RELAYS: unknown }).RELAYS = env.RELAYS_NOWRAP;
    const r = await f.ownerCall('/owner/grant-key', {}); expect(r.status).toBe(503); expect(await r.json()).toEqual({ error: 'grant_unavailable' });
  });
  test('is a POST owner operation like the others', async () => {
    const f = await provisioned(); const r = await f.call('/owner/grant-key'); expect(r.status).toBe(405);
  });
});
describe('connected apps listing shapes', () => {
  const v1 = { read: true, save: true, session: false };
  test('keeps the older shape unless the body is exactly version two', async () => {
    const f = await provisioned(); await addApp(f, 'app-a', ['slm:read', 'slm:write', 'slm:mesh'], ['recall', 'remember', 'mesh_peers']);
    for (const body of [undefined, {}, { version: 1 }, { version: 2, extra: true }, { version: '2' }, [2]]) {
      const rows = (await (await f.ownerCall('/owner/apps', body)).json() as { apps: Row[] }).apps;
      expect(rows[0].permissions, JSON.stringify(body)).toEqual(v1); expect(Object.keys(rows[0]).sort()).toEqual(['authorization_id', 'client_host', 'connected_at_ms', 'last_used_at_ms', 'name', 'permissions', 'version']);
    }
  });
  test('adds the two new permissions for version two and changes nothing else', async () => {
    const f = await provisioned(); await addApp(f, 'app-a', ['slm:read', 'slm:write', 'slm:mesh'], ['recall', 'remember', 'mesh_peers']); await addApp(f, 'app-b', ['slm:read', 'slm:media'], ['recall', 'get_media']);
    const older = (await (await f.ownerCall('/owner/apps')).json() as { apps: Row[] }).apps; const newer = (await (await f.ownerCall('/owner/apps', { version: 2 })).json() as { apps: Row[] }).apps;
    const byId = (rows: Row[]) => Object.fromEntries(rows.map(r => [r.authorization_id as string, r]));
    expect(byId(newer)['app-a'].permissions).toEqual({ ...v1, mesh: true, media: false }); expect(byId(newer)['app-b'].permissions).toEqual({ read: true, save: false, session: false, mesh: false, media: true });
    for (const row of newer) { const { permissions: _p, ...rest } = row; const { permissions: _q, ...before } = byId(older)[row.authorization_id as string]; expect(rest).toEqual(before); }
  });
});

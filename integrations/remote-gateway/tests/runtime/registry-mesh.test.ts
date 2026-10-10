import { env } from 'cloudflare:workers';
import { evictDurableObject } from 'cloudflare:test';
import { afterEach, describe, expect, test, vi } from 'vitest';
import type { AuthorizationGrant, ConnectionGrant, RequestEnvelope, Scope, VerifiedActor } from '../../src/contracts.ts';
afterEach(() => { vi.useRealTimers(); });
const audience = 'https://mcp.superlocalmemory.com/mcp';
const connection: ConnectionGrant = { connectionId: 'connection-a', ownerId: 'owner-a', installationId: 'installation-a', profileId: 'profile-a', origin: { kind: 'relay', installationId: 'installation-a', profileId: 'profile-a' }, allowedTools: ['recall', 'get_status'], allowCorrection: false, allowSharedRead: false, allowGlobalRead: false, policyVersion: 1, revokedAt: null };
const scopes: Scope[] = ['slm:read', 'slm:mesh'];
function grantFor(authorizationId: string): AuthorizationGrant { return { authorizationId, audience, ownerId: 'owner-a', clientId: 'client-' + authorizationId, connectionId: 'connection-a', consentedTools: ['recall', 'mesh_peers', 'mesh_send', 'mesh_inbox', 'mesh_wait'], consentedScopes: scopes, consentedCorrection: false, consentedSharedRead: false, consentedGlobalRead: false, authorizationVersion: 1, revokedAt: null }; }
const actorFor = (authorizationId: string): VerifiedActor => ({ ownerId: 'owner-a', authorizationId, connectionId: 'connection-a', clientId: 'client-' + authorizationId, audience, credentialKind: 'oauth', scopes });
const call = (toolName: string, args: Record<string, unknown> = {}): RequestEnvelope => ({ era: 'legacy', rpcMethod: 'tools/call', toolName, arguments: args, originalBody: new Uint8Array() });
async function setup(namespace: DurableObjectNamespace<import('../../src/registry-do.ts').RegistryDO>, ids = ['app-a']) {
  const stub = namespace.getByName(crypto.randomUUID()); await stub.configure(connection);
  for (const id of ids) await stub.addAuthorization(grantFor(id));
  await stub.setEntitlement('owner-a', Date.now() + 3 * 86400000, 0); return stub;
}
describe('mesh allowance', () => {
  test('inbox and wait polls do not consume the daily tool-call allowance', async () => {
    const stub = await setup(env.REGISTRIES);
    for (const tool of ['mesh_inbox', 'mesh_wait', 'mesh_inbox', 'mesh_wait', 'mesh_inbox']) expect((await stub.admit(actorFor('app-a'), audience, call(tool))).allowed).toBe(true);
    for (let i = 0; i < 3; i++) expect((await stub.admit(actorFor('app-a'), audience, call('recall', { query: 'x' }))).allowed).toBe(true);
    expect(await stub.admit(actorFor('app-a'), audience, call('recall', { query: 'x' }))).toEqual({ allowed: false, code: 'DAILY_LIMIT_REACHED', httpStatus: 429 });
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_inbox'))).allowed).toBe(true);
  });
  test('the poll budget refuses at its limit, counts inbox and wait together, and leaves other tools alone', async () => {
    const stub = await setup(env.REGISTRIES_TUNED);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_inbox'))).allowed).toBe(true);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_wait', { timeout_s: 5 }))).allowed).toBe(true);
    expect(await stub.admit(actorFor('app-a'), audience, call('mesh_inbox'))).toEqual({ allowed: false, code: 'DAILY_LIMIT_REACHED', httpStatus: 429 });
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_wait'))).allowed).toBe(false);
    expect((await stub.admit(actorFor('app-a'), audience, call('recall', { query: 'x' }))).allowed).toBe(true);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).allowed).toBe(true);
    expect((await stub.admit(actorFor('app-a'), audience, { ...call('mesh_inbox'), rpcMethod: 'tools/list', toolName: undefined })).allowed).toBe(true);
  });
  test('the poll budget is per connection, survives restart and resets the next day', async () => {
    const stub = await setup(env.REGISTRIES_TUNED, ['app-a', 'app-b']);
    await stub.admit(actorFor('app-a'), audience, call('mesh_inbox')); await stub.admit(actorFor('app-b'), audience, call('mesh_inbox'));
    await evictDurableObject(stub);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_inbox'))).allowed).toBe(false);
    vi.useFakeTimers({ toFake: ['Date'] }); vi.setSystemTime(Date.now() + 86400000 + 1000);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_inbox'))).allowed).toBe(true);
  });
  test('mesh_send counts against the daily allowance', async () => {
    const stub = await setup(env.REGISTRIES);
    for (let i = 0; i < 3; i++) expect((await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).allowed).toBe(true);
    expect(await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).toEqual({ allowed: false, code: 'DAILY_LIMIT_REACHED', httpStatus: 429 });
  });
  test('the 201st send of a day is refused for that app only, and the cap survives restart', async () => {
    const stub = await setup(env.REGISTRIES_TUNED, ['app-a', 'app-b']);
    for (let i = 0; i < 200; i++) { const r = await stub.admit(actorFor('app-a'), audience, call('mesh_send')); if (!r.allowed) throw new Error('refused early at ' + i + ': ' + r.code); }
    expect(await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).toEqual({ allowed: false, code: 'MESH_SEND_LIMIT', httpStatus: 429 });
    await evictDurableObject(stub);
    expect(await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).toMatchObject({ allowed: false, code: 'MESH_SEND_LIMIT' });
    expect((await stub.admit(actorFor('app-b'), audience, call('mesh_send'))).allowed).toBe(true);
    expect((await stub.admit(actorFor('app-a'), audience, call('recall', { query: 'x' }))).allowed).toBe(true);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_peers'))).allowed).toBe(true);
  }, 30000);
  test('a refused send is not counted against the day', async () => {
    const stub = await setup(env.REGISTRIES, ['app-a']);
    for (let i = 0; i < 3; i++) await stub.admit(actorFor('app-a'), audience, call('mesh_send'));
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).allowed).toBe(false);
    vi.useFakeTimers({ toFake: ['Date'] }); vi.setSystemTime(Date.now() + 86400000 + 1000);
    expect((await stub.admit(actorFor('app-a'), audience, call('mesh_send'))).allowed).toBe(true);
  });
  test('mesh and media tool names are accepted in stored grants and an unknown tool still is not', async () => {
    const stub = env.REGISTRIES.getByName(crypto.randomUUID()); await stub.configure({ ...connection, allowedTools: ['recall', 'mesh_peers', 'remember_media'] });
    await stub.addAuthorization({ ...grantFor('app-a'), consentedTools: ['mesh_state', 'get_media', 'remember_document'], consentedScopes: ['slm:read', 'slm:write', 'slm:mesh', 'slm:media'] });
    await expect((async () => { await stub.addAuthorization({ ...grantFor('app-b'), consentedTools: ['delete_everything'] }); })()).rejects.toThrow('invalid_authorization');
    await expect((async () => { await stub.addAuthorization({ ...grantFor('app-c'), consentedScopes: ['slm:admin' as Scope] }); })()).rejects.toThrow('invalid_authorization');
  });
});

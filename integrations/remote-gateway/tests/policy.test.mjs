import test from 'node:test';
import assert from 'node:assert/strict';
import { authorizeRequest } from '../src/request-policy.ts';
const resource = 'https://slm-mcp.example.com/mcp';
function fixture() {
  const actor = { ownerId: 'owner-a', authorizationId: 'grant-a', connectionId: 'connection-a', clientId: 'client-a', audience: resource, credentialKind: 'oauth', scopes: ['slm:read', 'slm:write', 'slm:session'] };
  const authorization = { ...actor, consentedTools: ['recall', 'remember', 'session_init'], consentedScopes: [...actor.scopes], consentedCorrection: false, consentedSharedRead: false, consentedGlobalRead: false, authorizationVersion: 1, revokedAt: null };
  const connection = { connectionId: actor.connectionId, ownerId: actor.ownerId, installationId: 'installation-a', profileId: 'profile-a', exactAgentPath: '/mcp', upstreamUrl: 'https://slm-origin-a.example.com/mcp', originCredentialRef: 'secret-ref-a', allowedTools: [...authorization.consentedTools], allowCorrection: false, allowSharedRead: false, allowGlobalRead: false, policyVersion: 1, revokedAt: null };
  const request = { era: 'legacy', rpcMethod: 'tools/call', toolName: 'recall', arguments: {}, originalBody: new Uint8Array() };
  return { actor, authorization, connection, request };
}
function run(f = fixture()) { return authorizeRequest(f.actor, f.authorization, f.connection, resource, f.request); }
function deny(f, code) { assert.deepEqual(run(f), { allowed: false, code, httpStatus: code === 'INVALID_PRINCIPAL' ? 401 : 403 }); }
test('owner can recall through their own consented connection', () => { const r = run(); assert.equal(r.allowed, true); assert.deepEqual(r.grant.allowedScopes, ['slm:read', 'slm:write', 'slm:session']); });
for (const [name, mutate, code] of [
 ['foreign owner', f => { f.actor.ownerId = 'owner-b'; }, 'BINDING_MISMATCH'],
 ['same owner other client', f => { f.actor.clientId = 'client-b'; }, 'BINDING_MISMATCH'],
 ['connection substitution', f => { f.actor.connectionId = 'connection-b'; }, 'BINDING_MISMATCH'],
 ['foreign grant', f => { f.actor.authorizationId = 'grant-b'; }, 'BINDING_MISMATCH'],
 ['foreign connection owner', f => { f.connection.ownerId = 'owner-b'; }, 'BINDING_MISMATCH'],
 ['wrong audience', f => { f.actor.audience = 'https://foreign.example/mcp'; }, 'INVALID_PRINCIPAL'],
 ['revoked grant', f => { f.authorization.revokedAt = '2026-10-06'; }, 'REVOKED'],
 ['revoked connection', f => { f.connection.revokedAt = '2026-10-06'; }, 'REVOKED'],
 ['token read scope absent', f => { f.actor.scopes = ['slm:write']; }, 'INSUFFICIENT_SCOPE'],
 ['consent read scope absent', f => { f.authorization.consentedScopes = ['slm:write']; }, 'INSUFFICIENT_SCOPE'],
 ['connection removed tool', f => { f.connection.allowedTools = []; }, 'TOOL_DENIED'],
 ['new tool not consented', f => { f.connection.allowedTools.push('fetch'); f.request.toolName = 'fetch'; }, 'TOOL_DENIED'],
 ['admin tool even if grant lists it', f => { f.connection.allowedTools.push('delete_memory'); f.authorization.consentedTools.push('delete_memory'); f.request.toolName = 'delete_memory'; }, 'TOOL_DENIED'],
 ['correction not consented', f => { f.connection.allowCorrection = true; f.request.toolName = 'remember'; f.request.arguments = { replaces: 'fact-a' }; }, 'CORRECTION_DENIED'],
 ['shared read not consented', f => { f.connection.allowSharedRead = true; f.request.arguments = { include_shared: true }; }, 'SHARING_DENIED'],
 ['global read not consented', f => { f.connection.allowGlobalRead = true; f.request.arguments = { include_global: true }; }, 'SHARING_DENIED'],
 ['foreign profile argument', f => { f.request.arguments = { profile_id: 'profile-b' }; }, 'PROFILE_DENIED'],
 ['nonpersonal save', f => { f.request.toolName = 'remember'; f.request.arguments = { scope: 'global' }; }, 'SHARING_DENIED'],
 ['modern initialize', f => { f.request.era = 'modern-2026-07-28'; f.request.rpcMethod = 'initialize'; }, 'METHOD_DENIED'],
 ['legacy discovery', f => { f.request.rpcMethod = 'server/discover'; }, 'METHOD_DENIED'],
 ['unknown method', f => { f.request.rpcMethod = 'resources/read'; }, 'METHOD_DENIED'],
]) test(name, () => { const f = fixture(); mutate(f); deny(f, code); });
test('remember needs write scope', () => { const f = fixture(); f.request.toolName = 'remember'; f.actor.scopes = ['slm:read']; deny(f, 'INSUFFICIENT_SCOPE'); });
test('session needs session scope', () => { const f = fixture(); f.request.toolName = 'session_init'; f.actor.scopes = ['slm:read', 'slm:write']; deny(f, 'INSUFFICIENT_SCOPE'); });
test('explicit correction allowed', () => { const f = fixture(); f.authorization.consentedCorrection = true; f.connection.allowCorrection = true; f.request.toolName = 'remember'; f.request.arguments = { replaces: 'fact-a', profile_id: 'profile-a', scope: 'personal' }; assert.equal(run(f).allowed, true); });
test('empty profile anchor allowed for bound origin', () => { const f = fixture(); f.request.arguments = { profile_id: '' }; assert.equal(run(f).allowed, true); });
test('intersection cannot expand or mutate consent', () => { const f = fixture(); f.actor.scopes.push('unknown'); f.connection.allowedTools.push('fetch'); const before = JSON.stringify(f); const r = run(f); assert.equal(r.allowed, true); assert.deepEqual(r.grant.allowedTools, ['recall', 'remember', 'session_init']); assert.equal(JSON.stringify(f), before); });
for (const [era, method] of [['legacy','initialize'], ['legacy','notifications/initialized'], ['legacy','ping'], ['legacy','tools/list'], ['modern-2026-07-28','server/discover'], ['modern-2026-07-28','tools/list']]) test(`${era} ${method}`, () => { const f = fixture(); f.request.era = era; f.request.rpcMethod = method; delete f.request.toolName; assert.equal(run(f).allowed, true); });
test('unknown era cannot fall back', () => { const f = fixture(); f.request.era = 'future'; deny(f, 'METHOD_DENIED'); });
test('missing grant or connection fails closed', () => { for (const key of ['authorization','connection']) { const f = fixture(); f[key] = null; deny(f, 'BINDING_MISMATCH'); } });
test('consent audience must match resource', () => { const f = fixture(); f.authorization.audience = 'https://other.example/mcp'; deny(f, 'INVALID_PRINCIPAL'); });
test('sharing string cannot bypass permission', () => { const f = fixture(); f.request.arguments = { include_shared: 'true' }; deny(f, 'SHARING_DENIED'); });
test('null profile cannot fall through', () => { const f = fixture(); f.request.arguments = { profile_id: null }; deny(f, 'PROFILE_DENIED'); });

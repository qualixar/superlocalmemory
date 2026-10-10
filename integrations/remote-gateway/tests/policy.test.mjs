import test from 'node:test';
import assert from 'node:assert/strict';
import { authorizeRequest } from '../src/request-policy.ts';
const resource = 'https://slm-mcp.example.com/mcp';
function fixture() {
  const actor = { ownerId: 'owner-a', authorizationId: 'grant-a', connectionId: 'connection-a', clientId: 'client-a', audience: resource, credentialKind: 'oauth', scopes: ['slm:read', 'slm:write', 'slm:session'] };
  const authorization = { ...actor, consentedTools: ['recall', 'remember', 'session_init'], consentedScopes: [...actor.scopes], consentedCorrection: false, consentedSharedRead: false, consentedGlobalRead: false, authorizationVersion: 1, revokedAt: null };
  const connection = { connectionId: actor.connectionId, ownerId: actor.ownerId, installationId: 'installation-a', profileId: 'profile-a', origin: {kind:'https',url:'https://slm-origin-a.example.com/mcp',credentialRef:'secret-ref-a',exactAgentPath:'/mcp'}, allowedTools: [...authorization.consentedTools], allowCorrection: false, allowSharedRead: false, allowGlobalRead: false, policyVersion: 1, revokedAt: null };
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

test('consent cannot select another connection', () => { const f = fixture(); f.authorization.connectionId = 'connection-b'; deny(f, 'BINDING_MISMATCH'); });
test('blank verified identity is invalid', () => { const f = fixture(); f.actor.ownerId = ''; deny(f, 'INVALID_PRINCIPAL'); });
test('unknown token scope is not returned as effective scope', () => { const f = fixture(); f.actor.scopes.push('admin'); f.authorization.consentedScopes.push('admin'); assert.deepEqual(run(f).grant.allowedScopes, ['slm:read', 'slm:write', 'slm:session']); });
test('tool discovery omits tools without actual operation scopes', () => { const f = fixture(); f.actor.scopes = ['slm:read']; f.request.rpcMethod = 'tools/list'; const r = run(f); assert.equal(r.allowed, true); assert.deepEqual(r.grant.allowedTools, ['recall']); });
for (const [argument, consentKey, policyKey] of [['include_shared','consentedSharedRead','allowSharedRead'], ['include_global','consentedGlobalRead','allowGlobalRead']]) {
  test(`${argument} requires connection policy too`, () => { const f = fixture(); f.authorization[consentKey] = true; f.request.arguments = { [argument]: true }; deny(f, 'SHARING_DENIED'); });
  test(`${argument} explicit two-sided consent allowed`, () => { const f = fixture(); f.authorization[consentKey] = true; f.connection[policyKey] = true; f.request.arguments = { [argument]: true }; assert.equal(run(f).allowed, true); });
  test(`${argument} explicit false stays personal`, () => { const f = fixture(); f.request.arguments = { [argument]: false }; assert.equal(run(f).allowed, true); });
}
test('correction also requires connection policy', () => { const f = fixture(); f.authorization.consentedCorrection = true; f.request.toolName = 'remember'; f.request.arguments = { replaces: 'fact-a' }; deny(f, 'CORRECTION_DENIED'); });
test('grant snapshot is detached from input permissions', () => { const f = fixture(); const r = run(f); f.connection.allowedTools.length = 0; f.authorization.consentedTools.length = 0; assert.deepEqual(r.grant.connection.allowedTools, ['recall', 'remember', 'session_init']); assert.deepEqual(r.grant.authorization.consentedTools, ['recall', 'remember', 'session_init']); });
test('missing tool cannot be called', () => { const f = fixture(); delete f.request.toolName; deny(f, 'TOOL_DENIED'); });
test('prototype property is not a tool', () => { const f = fixture(); f.request.toolName = '__proto__'; f.authorization.consentedTools.push('__proto__'); f.connection.allowedTools.push('__proto__'); deny(f, 'TOOL_DENIED'); });

test('effective origin transport is a frozen independent binding',()=>{const f=fixture();f.connection.origin={kind:'relay',installationId:f.connection.installationId,profileId:f.connection.profileId};const r=run(f);assert.equal(r.allowed,true);assert.ok(Object.isFrozen(r.grant.connection.origin));f.connection.origin.profileId='other';assert.equal(r.grant.connection.origin.profileId,'profile-a');});

// Mesh and media tools: gated by their own scopes and by the laptop, not by the connection's fixed tool list.
import { TOOL_SCOPES, LAPTOP_GATED_TOOLS, toolsForScopes } from '../src/request-policy.ts';
function gated(tool, scopes, args = {}, extra = {}) {
  const f = fixture(); f.actor.scopes = ['slm:read', ...scopes]; f.authorization.consentedScopes = [...f.actor.scopes];
  f.authorization.consentedTools = [tool]; f.connection.allowedTools = ['recall']; f.request.toolName = tool; f.request.arguments = args; Object.assign(f.request, extra); return f;
}
test('tool scope table lists every tool with all the scopes it needs', () => {
  assert.deepEqual([...TOOL_SCOPES.get('recall')], ['slm:read']); assert.deepEqual([...TOOL_SCOPES.get('remember')], ['slm:write']); assert.deepEqual([...TOOL_SCOPES.get('session_init')], ['slm:session']);
  for (const t of ['mesh_peers', 'mesh_send', 'mesh_inbox', 'mesh_wait', 'mesh_state']) assert.deepEqual([...TOOL_SCOPES.get(t)], ['slm:mesh']);
  for (const t of ['get_media', 'media_status']) assert.deepEqual([...TOOL_SCOPES.get(t)], ['slm:media']);
  for (const t of ['remember_media', 'remember_document', 'media_upload_link']) assert.deepEqual([...TOOL_SCOPES.get(t)], ['slm:write', 'slm:media']);
  assert.equal(TOOL_SCOPES.size, 9 + 5 + 2 + 3);
});
test('only mesh and media tools are laptop gated', () => { assert.deepEqual([...LAPTOP_GATED_TOOLS].sort(), ['get_media', 'media_status', 'media_upload_link', 'mesh_inbox', 'mesh_peers', 'mesh_send', 'mesh_state', 'mesh_wait', 'remember_document', 'remember_media']); });
test('a mesh tool needs the mesh scope on the token and in the consent', () => {
  for (const tool of ['mesh_peers', 'mesh_send', 'mesh_inbox', 'mesh_wait', 'mesh_state']) {
    const args = tool === 'mesh_state' ? { key: 'k' } : {};
    assert.equal(run(gated(tool, ['slm:mesh'], args)).allowed, true, tool);
    assert.deepEqual(run(gated(tool, [], args)), { allowed: false, code: 'INSUFFICIENT_SCOPE', httpStatus: 403 }, tool);
    const consentOnly = gated(tool, ['slm:mesh'], args); consentOnly.authorization.consentedScopes = ['slm:read']; assert.equal(run(consentOnly).allowed, false);
    const tokenOnly = gated(tool, ['slm:mesh'], args); tokenOnly.actor.scopes = ['slm:read']; assert.equal(run(tokenOnly).allowed, false);
  }
});
test('media tools need the media scope and remember_* also needs write', () => {
  assert.equal(run(gated('get_media', ['slm:media'])).allowed, true); assert.equal(run(gated('media_status', ['slm:media'])).allowed, true);
  assert.equal(run(gated('get_media', ['slm:mesh'])).code, 'INSUFFICIENT_SCOPE');
  for (const tool of ['remember_media', 'remember_document', 'media_upload_link']) {
    assert.equal(run(gated(tool, ['slm:write', 'slm:media'])).allowed, true, tool);
    assert.equal(run(gated(tool, ['slm:media'])).code, 'INSUFFICIENT_SCOPE', tool);
    assert.equal(run(gated(tool, ['slm:write'])).code, 'INSUFFICIENT_SCOPE', tool);
  }
});
test('laptop gated tools skip the connection tool list but an unlisted ordinary tool is still denied', () => {
  const f = gated('mesh_peers', ['slm:mesh']); assert.equal(f.connection.allowedTools.includes('mesh_peers'), false); const r = run(f); assert.equal(r.allowed, true); assert.ok(r.grant.allowedTools.includes('mesh_peers'));
  const g = gated('mesh_peers', ['slm:mesh']); g.authorization.consentedTools = []; assert.equal(run(g).code, 'TOOL_DENIED');
  const h = gated('search', []); h.actor.scopes = ['slm:read']; assert.equal(run(h).code, 'TOOL_DENIED');
});
test('effective scopes carry the mesh and media scopes only when both token and consent have them', () => {
  const f = gated('mesh_peers', ['slm:mesh', 'slm:media']); assert.deepEqual(run(f).grant.allowedScopes, ['slm:read', 'slm:mesh', 'slm:media']);
});
for (const [name, tool, scopes, args] of [
  ['mesh_state set', 'mesh_state', ['slm:mesh'], { action: 'set', key: 'k' }], ['mesh_state delete', 'mesh_state', ['slm:mesh'], { action: 'delete', key: 'k' }], ['mesh_state null action', 'mesh_state', ['slm:mesh'], { action: null, key: 'k' }],
  ['mesh_state empty key', 'mesh_state', ['slm:mesh'], { key: '' }], ['mesh_state missing key', 'mesh_state', ['slm:mesh'], {}], ['mesh_state long key', 'mesh_state', ['slm:mesh'], { key: 'k'.repeat(257) }], ['mesh_state numeric key', 'mesh_state', ['slm:mesh'], { key: 5 }],
  ['mesh_wait zero', 'mesh_wait', ['slm:mesh'], { timeout_s: 0 }], ['mesh_wait 21', 'mesh_wait', ['slm:mesh'], { timeout_s: 21 }], ['mesh_wait string', 'mesh_wait', ['slm:mesh'], { timeout_s: '5' }], ['mesh_wait fraction', 'mesh_wait', ['slm:mesh'], { timeout_s: 2.5 }], ['mesh_wait null', 'mesh_wait', ['slm:mesh'], { timeout_s: null }],
  ['remember_media path', 'remember_media', ['slm:write', 'slm:media'], { path: '/etc/passwd' }], ['remember_document path', 'remember_document', ['slm:write', 'slm:media'], { path: 'a.pdf' }], ['remember_media numeric path', 'remember_media', ['slm:write', 'slm:media'], { path: 3 }],
]) test('argument guard: ' + name, () => { assert.deepEqual(run(gated(tool, scopes, args)), { allowed: false, code: 'ARGUMENT_DENIED', httpStatus: 403 }); });
for (const [name, tool, scopes, args] of [
  ['mesh_state get', 'mesh_state', ['slm:mesh'], { action: 'get', key: 'k' }], ['mesh_state default action', 'mesh_state', ['slm:mesh'], { key: 'k'.repeat(256) }],
  ['mesh_wait 1', 'mesh_wait', ['slm:mesh'], { timeout_s: 1 }], ['mesh_wait 20', 'mesh_wait', ['slm:mesh'], { timeout_s: 20 }], ['mesh_wait default', 'mesh_wait', ['slm:mesh'], {}],
  ['remember_media no path', 'remember_media', ['slm:write', 'slm:media'], {}], ['remember_media null path', 'remember_media', ['slm:write', 'slm:media'], { path: null }], ['remember_document empty path', 'remember_document', ['slm:write', 'slm:media'], { path: '' }],
]) test('argument guard allows: ' + name, () => { assert.equal(run(gated(tool, scopes, args)).allowed, true); });
test('the shared profile and sharing checks still apply to mesh tools', () => { assert.equal(run(gated('mesh_peers', ['slm:mesh'], { profile_id: 'profile-b' })).code, 'PROFILE_DENIED'); assert.equal(run(gated('mesh_peers', ['slm:mesh'], { include_global: true })).code, 'SHARING_DENIED'); });
test('the consent tool list comes from the same table', () => {
  assert.deepEqual(toolsForScopes(['slm:read']), ['recall', 'search', 'fetch', 'get_status']);
  assert.deepEqual(toolsForScopes(['slm:read', 'slm:write']), ['recall', 'search', 'fetch', 'get_status', 'remember']);
  assert.deepEqual(toolsForScopes(['slm:read', 'slm:session']), ['recall', 'search', 'fetch', 'get_status', 'session_init', 'close_session', 'report_feedback', 'report_outcome']);
  assert.deepEqual(toolsForScopes(['slm:read', 'slm:mesh']), ['recall', 'search', 'fetch', 'get_status', 'mesh_peers', 'mesh_send', 'mesh_inbox', 'mesh_wait', 'mesh_state']);
  assert.deepEqual(toolsForScopes(['slm:read', 'slm:media']), ['recall', 'search', 'fetch', 'get_status', 'get_media', 'media_status']);
  assert.ok(toolsForScopes(['slm:read', 'slm:write', 'slm:media']).includes('remember_media'));
  assert.ok(toolsForScopes(['slm:read', 'slm:write', 'slm:media']).includes('media_upload_link'));
  assert.ok(!toolsForScopes(['slm:read', 'slm:media']).includes('media_upload_link'));
  assert.ok(!toolsForScopes(['slm:read', 'slm:write']).includes('media_upload_link'));
  assert.ok(!toolsForScopes(['slm:read', 'slm:media']).includes('remember_media'));
});

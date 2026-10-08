import {env} from 'cloudflare:workers';
import {evictDurableObject} from 'cloudflare:test';
import {describe,expect,test} from 'vitest';
import type {AuthorizationGrant,ConnectionGrant,VerifiedActor,RequestEnvelope} from '../../src/contracts.ts';

const connection:ConnectionGrant={connectionId:'connection-a',ownerId:'owner-a',installationId:'installation-a',profileId:'profile-a',origin:{kind:'relay',installationId:'installation-a',profileId:'profile-a'},allowedTools:['recall','remember','get_status'],allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null};
const grant:AuthorizationGrant={authorizationId:'authorization-a',audience:'https://mcp.superlocalmemory.com/mcp',ownerId:'owner-a',clientId:'client-a',connectionId:'connection-a',consentedTools:['recall','remember'],consentedScopes:['slm:read','slm:write'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null};
const actor:VerifiedActor={ownerId:'owner-a',authorizationId:'authorization-a',connectionId:'connection-a',clientId:'client-a',audience:grant.audience,credentialKind:'oauth',scopes:['slm:read','slm:write']};
const request:RequestEnvelope={era:'legacy',rpcMethod:'tools/call',toolName:'recall',arguments:{query:'synthetic'},originalBody:new Uint8Array()};
async function rejected(operation:()=>Promise<unknown>,message:string){await expect((async()=>{await operation();})()).rejects.toThrow(message);}
async function setup(){const stub=env.REGISTRIES.getByName(crypto.randomUUID());await stub.configure(connection);await stub.addAuthorization(grant);await stub.setEntitlement('owner-a',Date.now()+60000,0);return stub;}

test('registry admits a current bound actor with the intersection of scopes',async()=>{const stub=await setup();const result=await stub.admit(actor,grant.audience,request);expect(result.allowed).toBe(true);if(result.allowed)expect(result.grant.allowedTools).toEqual(['recall','remember']);});
test('registry refuses unknown or cross-owner actor',async()=>{const stub=await setup();expect((await stub.admit({...actor,ownerId:'other'},grant.audience,request)).allowed).toBe(false);expect((await stub.admit({...actor,authorizationId:'other'},grant.audience,request)).allowed).toBe(false);});
test('registry defaults entitlement off and expires it independently of local core',async()=>{const stub=env.REGISTRIES.getByName(crypto.randomUUID());await stub.configure(connection);await stub.addAuthorization(grant);expect(await stub.admit(actor,grant.audience,request)).toMatchObject({allowed:false,code:'ENTITLEMENT_REQUIRED'});await stub.setEntitlement('owner-a',Date.now()-1,0);expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(false);});
test('grant identity cannot be reused to widen consent',async()=>{const stub=await setup();await rejected(async()=>await stub.addAuthorization({...grant,consentedSharedRead:true}),'authorization_conflict');expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);});
test('connection ownership and profile binding cannot be replaced',async()=>{const stub=await setup();await rejected(async()=>await stub.configure({...connection,profileId:'other',origin:{kind:'relay',installationId:connection.installationId,profileId:'other'}}),'connection_conflict');await rejected(async()=>await stub.configure({...connection,ownerId:'other'}),'connection_conflict');});
test('revoking authorization commits before acknowledgement and survives eviction',async()=>{const stub=await setup();expect(await stub.revokeAuthorization('owner-a','authorization-a',1)).toBe(2);await evictDurableObject(stub);expect(await stub.admit(actor,grant.audience,request)).toMatchObject({allowed:false,code:'REVOKED'});await rejected(async()=>await stub.addAuthorization(grant),'authorization_conflict');});
test('revoking connection is terminal and survives restart',async()=>{const stub=await setup();expect(await stub.revokeConnection('owner-a',1)).toBe(2);await evictDurableObject(stub);expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(false);await rejected(async()=>await stub.configure(connection),'connection_revoked');});
test('revocation enforces owner and compare-and-swap version',async()=>{const stub=await setup();await rejected(async()=>await stub.revokeConnection('other',1),'owner_mismatch');await rejected(async()=>await stub.revokeAuthorization('owner-a','authorization-a',0),'version_conflict');expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);});
test('token scopes and profile arguments are rechecked at admission',async()=>{const stub=await setup();expect((await stub.admit({...actor,scopes:['slm:read']},grant.audience,{...request,toolName:'remember',arguments:{content:'synthetic'}})).allowed).toBe(false);expect((await stub.admit(actor,grant.audience,{...request,arguments:{profile_id:'other'}})).allowed).toBe(false);});
test('concurrent immutable grants never admit a widened payload',async()=>{const stub=await setup();const results=await Promise.allSettled([stub.addAuthorization(grant),stub.addAuthorization({...grant,consentedCorrection:true})]);expect(results[0].status).toBe('fulfilled');expect(results[1].status).toBe('rejected');});
test('registry has no public administrative HTTP route',async()=>{const stub=await setup();expect((await stub.fetch(new Request('https://private.invalid/admin'))).status).toBe(404);});
test('malformed control-plane data is rejected before durable mutation',async()=>{const stub=env.REGISTRIES.getByName(crypto.randomUUID());await rejected(async()=>await stub.configure({...connection,policyVersion:0}),'invalid_connection');await rejected(async()=>await stub.configure({...connection,allowedTools:['admin']}),'invalid_connection');await stub.configure(connection);await rejected(async()=>await stub.addAuthorization({...grant,consentedScopes:['invalid'] as never}),'invalid_authorization');});

// Connect Free: the daily cap counts real tool calls exactly (Durable Object, not an
// approximate edge counter). Test binding DAILY_TOOL_CALL_LIMIT=3.
describe('daily tool-call cap',()=>{
 test('tool calls beyond the daily limit are refused with a stable code',async()=>{
  const stub=await setup();
  for(let i=0;i<3;i++)expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);
  expect(await stub.admit(actor,grant.audience,request)).toEqual({allowed:false,code:'DAILY_LIMIT_REACHED',httpStatus:429});
 });
 test('discovery and handshake messages never consume the quota',async()=>{
  const stub=await setup();
  for(const rpcMethod of ['initialize','notifications/initialized','tools/list','tools/list','ping'])expect((await stub.admit(actor,grant.audience,{...request,rpcMethod,toolName:undefined,arguments:undefined})).allowed).toBe(true);
  for(let i=0;i<3;i++)expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);
 });
 test('the count survives eviction',async()=>{
  const stub=await setup();
  for(let i=0;i<2;i++)expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);
  await evictDurableObject(stub);
  expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);
  expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(false);
 });
 test('refused requests do not consume the quota',async()=>{
  const stub=await setup();
  for(let i=0;i<5;i++)expect((await stub.admit({...actor,ownerId:'foreign'},grant.audience,request)).allowed).toBe(false);
  for(let i=0;i<3;i++)expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);
 });
});

// Connected apps: the owner sees which apps hold an active grant and when they last
// used it. Times are stamped by the registry, never supplied by a caller.
describe('connected apps list',()=>{
 test('active grants are listed with a server-stamped connection time',async()=>{
  const before=Date.now();const stub=await setup();
  const apps=await stub.listAuthorizations('owner-a');
  expect(apps).toHaveLength(1);
  expect(apps[0]).toMatchObject({authorizationId:'authorization-a',clientId:'client-a',authorizationVersion:1,lastUsedAt:null});
  expect([...apps[0].consentedScopes].sort()).toEqual([...grant.consentedScopes].sort());
  expect(apps[0].createdAt).toBeGreaterThanOrEqual(before);
 });
 test('a tool call records last use and survives eviction',async()=>{
  const stub=await setup();expect((await stub.admit(actor,grant.audience,request)).allowed).toBe(true);
  await evictDurableObject(stub);
  const [app]=await stub.listAuthorizations('owner-a');
  expect(typeof app.lastUsedAt).toBe('number');expect(typeof app.createdAt).toBe('number');
 });
 test('discovery alone does not count as use',async()=>{
  const stub=await setup();await stub.admit(actor,grant.audience,{...request,rpcMethod:'tools/list',toolName:undefined,arguments:undefined});
  expect((await stub.listAuthorizations('owner-a'))[0].lastUsedAt).toBeNull();
 });
 test('a revoked app leaves the list and another owner cannot read it',async()=>{
  const stub=await setup();await stub.revokeAuthorization('owner-a','authorization-a',1);
  expect(await stub.listAuthorizations('owner-a')).toEqual([]);
  await expect((async()=>{await stub.listAuthorizations('other');})()).rejects.toThrow('owner_mismatch');
 });
});

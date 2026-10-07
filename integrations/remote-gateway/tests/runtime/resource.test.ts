import {env,createExecutionContext} from 'cloudflare:workers';
import {expect,test} from 'vitest';
import {createExecutionContext as context,waitOnExecutionContext} from 'cloudflare:test';
import {resourceGateway} from '../../src/worker-resource.ts';
import type {ResourceEnv} from '../../src/worker-resource.ts';
import type {AuthorizationGrant,ConnectionGrant} from '../../src/contracts.ts';

async function setup(){
 const id=crypto.randomUUID();const connection:ConnectionGrant={connectionId:id,ownerId:'owner-a',installationId:'installation-a',profileId:'profile-a',exactAgentPath:'/mcp',upstreamUrl:'http://127.0.0.1:8765/mcp',originCredentialRef:'keychain:installation-a',allowedTools:['recall','remember'],allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null};
 const authorization:AuthorizationGrant={authorizationId:'authorization-a',audience:'https://mcp.superlocalmemory.com/mcp',ownerId:'owner-a',clientId:'client-a',connectionId:id,consentedTools:['recall','remember'],consentedScopes:['slm:read','slm:write'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null};
 const registry=env.REGISTRIES.getByName(id);await registry.configure(connection);await registry.addAuthorization(authorization);await registry.setEntitlement('owner-a',Date.now()+60000,0);
 const auth={async validateToken(resource:string,token:string){if(!['synthetic-read','synthetic-write','synthetic-wrong-audience'].includes(token))return null;return {props:{ownerId:'owner-a',authorizationId:'authorization-a',connectionId:id},audience:token==='synthetic-wrong-audience'?'https://evil.example/mcp':resource,scope:token==='synthetic-read'?['slm:read']:['slm:read','slm:write'],expiresAt:Math.floor(Date.now()/1000)+60,userId:'owner-a',clientId:'client-a'};}};
 const fixtureEnv={...env,AUTH_SERVER:auth} as ResourceEnv;
 return {id,registry,fixtureEnv};
}
async function call(fixtureEnv:ResourceEnv,token:string|undefined,body:unknown={jsonrpc:'2.0',id:1,method:'tools/list'},headers:Record<string,string>={}){
 const ctx=context();const request=new Request('https://mcp.superlocalmemory.com/mcp',{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream',...(token?{Authorization:'Bearer '+token}:{}),...headers},body:JSON.stringify(body)});
 const response=await resourceGateway.fetch(request,fixtureEnv,ctx);await waitOnExecutionContext(ctx);return response;
}
test('resource SDK publishes discovery and challenges absent tokens',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,undefined);expect(response.status).toBe(401);expect(response.headers.get('WWW-Authenticate')).toContain('resource_metadata');const metadata=await resourceGateway.fetch(new Request('https://mcp.superlocalmemory.com/.well-known/oauth-protected-resource/mcp'),fixtureEnv,context());expect(metadata.status).toBe(200);expect(await metadata.json()).toMatchObject({resource:'https://mcp.superlocalmemory.com/mcp'});});
test('wrong audience and invalid tokens never reach relay',async()=>{const {fixtureEnv}=await setup();expect((await call(fixtureEnv,'synthetic-wrong-audience')).status).toBe(401);expect((await call(fixtureEnv,'unknown')).status).toBe(401);});
test('read token cannot save even if connection permits save',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,'synthetic-read',{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'remember',arguments:{content:'synthetic'}}});expect(response.status).toBe(403);});
test('authenticated admission cannot manufacture an online connector',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,'synthetic-read');expect(response.status).toBe(503);expect(await response.json()).toMatchObject({error:'connector_unavailable'});});
test('revocation denies valid OAuth token after durable acknowledgement',async()=>{const {fixtureEnv,registry}=await setup();await registry.revokeAuthorization('owner-a','authorization-a',1);expect((await call(fixtureEnv,'synthetic-write')).status).toBe(403);});
test('host and Origin boundaries precede remote tool execution',async()=>{const {fixtureEnv}=await setup();expect((await call(fixtureEnv,'synthetic-write',undefined,{Origin:'https://evil.example'})).status).toBe(403);const response=await resourceGateway.fetch(new Request('https://evil.example/mcp',{method:'POST',headers:{Authorization:'Bearer synthetic-write','Content-Type':'application/json'},body:'{}'}),fixtureEnv,context());expect(response.status).toBe(403);});
test('modern header mismatch returns the standard RPC error',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,'synthetic-write',{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'recall',arguments:{query:'synthetic'},_meta:{'io.modelcontextprotocol/protocolVersion':'2026-07-28'}}},{'MCP-Protocol-Version':'2026-07-28','Mcp-Method':'tools/call','Mcp-Name':'remember'});expect(response.status).toBe(400);expect(await response.json()).toMatchObject({error:{code:-32020}});});

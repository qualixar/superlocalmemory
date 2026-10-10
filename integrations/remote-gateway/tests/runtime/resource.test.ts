import {env} from 'cloudflare:workers';
import {expect,test} from 'vitest';
import {createExecutionContext as context,waitOnExecutionContext} from 'cloudflare:test';
import {resourceGateway} from '../../src/worker-resource.ts';
import type {ResourceEnv} from '../../src/worker-resource.ts';
import type {AuthorizationGrant,ConnectionGrant} from '../../src/contracts.ts';
import {decodeRelayFrame,encodeRelayFrame} from '../../src/relay-protocol.ts';

async function setup(){
 const id=crypto.randomUUID();const connection:ConnectionGrant={connectionId:id,ownerId:'owner-a',installationId:'installation-a',profileId:'profile-a',origin:{kind:'relay',installationId:'installation-a',profileId:'profile-a'},allowedTools:['recall','remember'],allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null};
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
// Clients request exactly what the resource advertises. Advertising read alone left ChatGPT and
// Grok Bot read-only: the consent page only offers saving and session tools when they are requested,
// and leaves both unticked. slm:connect is the owner's own scope and is never advertised here.
test('the resource advertises every scope an app may be granted, so the owner chooses on the consent page',async()=>{const {fixtureEnv}=await setup();const scopes=['slm:read','slm:write','slm:session','slm:mesh','slm:media'];const metadata=await resourceGateway.fetch(new Request('https://mcp.superlocalmemory.com/.well-known/oauth-protected-resource/mcp'),fixtureEnv,context());expect((await metadata.json()).scopes_supported).toEqual(scopes);const challenge=(await call(fixtureEnv,undefined)).headers.get('WWW-Authenticate')??'';expect(challenge).toContain('scope="'+scopes.join(' ')+'"');expect(challenge).not.toContain('slm:connect');});
test('wrong audience and invalid tokens never reach relay',async()=>{const {fixtureEnv}=await setup();expect((await call(fixtureEnv,'synthetic-wrong-audience')).status).toBe(401);expect((await call(fixtureEnv,'unknown')).status).toBe(401);});
test('read token cannot save even if connection permits save',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,'synthetic-read',{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'remember',arguments:{content:'synthetic'}}});expect(response.status).toBe(403);});
test('authenticated admission cannot manufacture an online connector',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,'synthetic-read');expect(response.status).toBe(503);expect(await response.json()).toMatchObject({error:'connector_unavailable'});});
test('revocation denies valid OAuth token after durable acknowledgement',async()=>{const {fixtureEnv,registry}=await setup();await registry.revokeAuthorization('owner-a','authorization-a',1);expect((await call(fixtureEnv,'synthetic-write')).status).toBe(403);});
test('host and Origin boundaries precede remote tool execution',async()=>{const {fixtureEnv}=await setup();expect((await call(fixtureEnv,'synthetic-write',undefined,{Origin:'https://evil.example'})).status).toBe(403);const response=await resourceGateway.fetch(new Request('https://evil.example/mcp',{method:'POST',headers:{Authorization:'Bearer synthetic-write','Content-Type':'application/json'},body:'{}'}),fixtureEnv,context());expect(response.status).toBe(403);});
test('modern header mismatch returns the standard RPC error',async()=>{const {fixtureEnv}=await setup();const response=await call(fixtureEnv,'synthetic-write',{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'recall',arguments:{query:'synthetic'},_meta:{'io.modelcontextprotocol/protocolVersion':'2026-07-28'}}},{'MCP-Protocol-Version':'2026-07-28','Mcp-Method':'tools/call','Mcp-Name':'remember'});expect(response.status).toBe(400);expect(await response.json()).toMatchObject({error:{code:-32020}});});

test('authorized HTTP remember and recall cross registry and live relay socket',async()=>{
 const {id,fixtureEnv}=await setup();const token='synthetic-device-token-'.repeat(3);
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token))),byte=>byte.toString(16).padStart(2,'0')).join('');
 const relay=env.RELAYS.getByName(id);await relay.configureBinding({ownerId:'owner-a',connectionId:id,installationId:'installation-a',profileId:'profile-a',deviceDigest:digest,deviceExpiresAt:Date.now()+60000});
 const upgrade=await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}));
 const socket=upgrade.webSocket!;const ready=new Promise<string>(resolve=>socket.addEventListener('message',event=>resolve(String(event.data)),{once:true}));socket.accept();await ready;
 const memories:string[]=[];
 socket.addEventListener('message',event=>{const decoded=decodeRelayFrame(String(event.data));if(!decoded.ok||decoded.frame.kind!=='request')return;const frame=decoded.frame;
  const request=JSON.parse(atob(frame.bodyBase64));let result;
  if(request.params.name==='remember'){memories.push(request.params.arguments.content);result={content:[{type:'text',text:'saved'}],structuredContent:{success:true}};}
  else result={content:[{type:'text',text:memories.join('\n')}],structuredContent:{results:[...memories]}};
  const response=encodeRelayFrame({v:1,kind:'response',id:frame.id,generation:frame.generation,status:200,headers:[['content-type','application/json']],bodyBase64:btoa(JSON.stringify({jsonrpc:'2.0',id:request.id,result}))});if(response.ok)socket.send(response.text);
 });
 try{const saved=await call(fixtureEnv,'synthetic-write',{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'remember',arguments:{content:'synthetic scoped memory'}}});expect(saved.status).toBe(200);
  const recalled=await call(fixtureEnv,'synthetic-read',{jsonrpc:'2.0',id:2,method:'tools/call',params:{name:'recall',arguments:{query:'synthetic'}}});expect(recalled.status).toBe(200);expect(await recalled.json()).toMatchObject({result:{structuredContent:{results:['synthetic scoped memory']}}});expect(recalled.headers.get('cache-control')).toBe('no-store');
 }finally{socket.close();}
});

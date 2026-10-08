import {env,createExecutionContext,waitOnExecutionContext} from 'cloudflare:test';import {expect,test,vi} from 'vitest';
import {generateKeyPair,exportJWK,SignJWT} from 'jose';
import {authFetch,type AuthWorkerEnv} from '../../src/worker-auth.ts';import {tokenHash} from '../../src/device-proof.ts';
const issuer='https://auth.superlocalmemory.com';
const cookies=(response:Response)=>response.headers.getSetCookie().map(x=>x.split(';')[0]).join('; ');
function handle(page:string){return /name="handle" value="([^"]+)"/.exec(page)![1];}
async function enrolled(){
 const configuration={...env,GITHUB_CLIENT_ID:'synthetic-client',GITHUB_CLIENT_SECRET:'synthetic-secret',DEVICE_WRAP_KEY:'a'.repeat(64)} as AuthWorkerEnv;
 const pair=await generateKeyPair('ES256',{extractable:true});const jwk=await exportJWK(pair.publicKey);const connectionId=crypto.randomUUID().replaceAll('-','');const profileId='synthetic';const installationId='i-'+connectionId;const verifier='v'.repeat(43);const redirect='http://127.0.0.1:18767/api/v3/connections/callback';
 async function call(path:string,init?:RequestInit){const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+path,init),configuration,ctx);await waitOnExecutionContext(ctx);return response;}
 const registration=await call('/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'SuperLocalMemory Desktop',redirect_uris:[redirect],token_endpoint_auth_method:'none',grant_types:['authorization_code','refresh_token'],response_types:['code']})});expect(registration.status).toBe(201);const client=await registration.json() as {client_id:string};
 const params=new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:redirect,scope:'slm:connect',state:'s'.repeat(43),code_challenge:await tokenHash(verifier),code_challenge_method:'S256',resource:issuer+'/owner'});
 const started=await call('/bootstrap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({connectionId,installationId,profileId,host:'muse',permissions:{read:true,write:true,correction:false,session:false},deviceJwk:jwk,expiresAtMs:Date.now()+600000,authorizationUrl:issuer+'/authorize?'+params})});expect(started.status).toBe(201);
 const consent=await call('/owner-login?connection_id='+connectionId);expect(consent.status).toBe(200);
 const signIn=await call('/consent',{method:'POST',headers:{Origin:issuer,Cookie:cookies(consent),'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({handle:handle(await consent.text()),decision:'allow'})});expect(signIn.status).toBe(302);
 const upstream=new URL(signIn.headers.get('Location')!);
 vi.stubGlobal('fetch',async(input:RequestInfo|URL)=>{if(String(input)==='https://github.com/login/oauth/access_token')return Response.json({access_token:'synthetic-provider-token'});if(String(input)==='https://api.github.com/user')return Response.json({id:16027584});throw new Error('unexpected outbound request');});
 try{
  const returned=await call('/github/callback?'+new URLSearchParams({code:'synthetic-code',state:upstream.searchParams.get('state')!}),{headers:{Cookie:cookies(signIn)}});expect(returned.status).toBe(302);const callback=new URL(returned.headers.get('Location')!);
  const tokenResponse=await call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({grant_type:'authorization_code',client_id:client.client_id,redirect_uri:redirect,code:callback.searchParams.get('code')!,code_verifier:verifier,resource:issuer+'/owner'})});expect(tokenResponse.status).toBe(200);const issued=await tokenResponse.json() as {access_token:string;scope:string};expect(issued.scope).toBe('slm:connect');
  async function provision(){const proof=await new SignJWT({htm:'POST',htu:issuer+'/owner/connections',ath:await tokenHash(issued.access_token)}).setProtectedHeader({typ:'dpop+jwt',alg:'ES256',jwk}).setIssuedAt().setJti(crypto.randomUUID()).sign(pair.privateKey);return call('/owner/connections',{method:'POST',headers:{Authorization:'Bearer '+issued.access_token,DPoP:proof}});}
  async function ownerCall(path:string,body?:unknown){const proof=await new SignJWT({htm:'POST',htu:issuer+path,ath:await tokenHash(issued.access_token)}).setProtectedHeader({typ:'dpop+jwt',alg:'ES256',jwk}).setIssuedAt().setJti(crypto.randomUUID()).sign(pair.privateKey);return call(path,{method:'POST',headers:{Authorization:'Bearer '+issued.access_token,DPoP:proof,...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)})});}
  return {configuration,connectionId,profileId,installationId,verifier,client,issued,call,provision,ownerCall};
 }finally{vi.unstubAllGlobals();}
}

test('native PKCE bootstrap issues scoped owner token and idempotent encrypted device delivery',async()=>{
 const f=await enrolled();const first=await f.provision();expect(first.status).toBe(200);const device=await first.json() as {device_token:string};expect(device.device_token).toMatch(/^[a-f0-9]{64}$/);
 const second=await f.provision();expect(second.status).toBe(200);expect(await second.json()).toMatchObject(device);
 const owned=await env.OWNERS.getByName('16027584').getConnection('16027584',f.connectionId);expect(owned!.credentialEnvelope).not.toContain(device.device_token);
});

for(const route of ['bootstrap','oauth'])test(route+' cancellation fences a provision paused before owner insertion',async()=>{
 const f=await enrolled();const real=env.OWNERS.getByName('16027584');let release!:()=>void;let entered!:()=>void;
 const barrier=new Promise<void>(resolve=>release=resolve);const reached=new Promise<void>(resolve=>entered=resolve);
 f.configuration.OWNERS={getByName:()=>new Proxy(real,{get(target,key){if(key==='addConnection')return async(...args:Parameters<typeof real.addConnection>)=>{entered();await barrier;return real.addConnection(...args);};return (...args:unknown[])=>Reflect.get(target,key)(...args);}})} as unknown as typeof env.OWNERS;
 const provisioning=f.provision();await reached;
 try{
  const response=route==='bootstrap'?await f.call('/bootstrap/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({connection_id:f.connectionId,verifier:f.verifier})}):await f.call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({token:f.issued.access_token,client_id:f.client.client_id})});
  expect(response.status).toBe(200);
 }finally{release();}
 expect((await provisioning).status).not.toBe(200);expect(await real.getConnection('16027584',f.connectionId)).toBeNull();
 expect((await env.RELAYS.getByName(f.connectionId).fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+'a'.repeat(64)}}))).status).not.toBe(101);
});

test('native OAuth revocation closes an established relay and denies an existing MCP authorization',async()=>{
 const {resourceGateway}=await import('../../src/worker-resource.ts');const {decodeRelayFrame,encodeRelayFrame}=await import('../../src/relay-protocol.ts');
 const f=await enrolled();const delivered=await f.provision();expect(delivered.status).toBe(200);const device=await delivered.json() as {device_token:string};
 const relay=env.RELAYS.getByName(f.connectionId);const registry=env.REGISTRIES.getByName(f.connectionId);const audience='https://mcp.superlocalmemory.com/mcp';
 await registry.addAuthorization({ownerId:'16027584',connectionId:f.connectionId,authorizationId:'existing-mcp',clientId:'synthetic-mcp',audience,consentedTools:['recall'],consentedScopes:['slm:read'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 const configuration={...env,AUTH_SERVER:{async validateToken(){return {userId:'16027584',clientId:'synthetic-mcp',props:{ownerId:'16027584',connectionId:f.connectionId,authorizationId:'existing-mcp'},scope:['slm:read'],expiresAt:Math.floor(Date.now()/1000)+60,audience};}}};
 const upgrade=await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+device.device_token}}));expect(upgrade.status).toBe(101);const socket=upgrade.webSocket!;
 const ready=new Promise<void>(resolve=>socket.addEventListener('message',()=>resolve(),{once:true}));socket.accept();await ready;
 socket.addEventListener('message',event=>{const parsed=decodeRelayFrame(String(event.data));if(!parsed.ok||parsed.frame.kind!=='request')return;const frame=parsed.frame;const rpc=JSON.parse(atob(frame.bodyBase64));const encoded=encodeRelayFrame({v:1,kind:'response',id:frame.id,generation:frame.generation,status:200,headers:[['content-type','application/json']],bodyBase64:btoa(JSON.stringify({jsonrpc:'2.0',id:rpc.id,result:{content:[{type:'text',text:'synthetic memory'}]}}))});if(encoded.ok)socket.send(encoded.text);});
 async function recall(){const context=createExecutionContext();const response=await resourceGateway.fetch(new Request(audience,{method:'POST',headers:{Authorization:'Bearer synthetic-mcp','Content-Type':'application/json',Accept:'application/json'},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'recall',arguments:{query:'synthetic'}}})}),configuration as never,context);await waitOnExecutionContext(context);return response;}
 try{expect((await recall()).status).toBe(200);const closed=new Promise<void>(resolve=>socket.addEventListener('close',()=>resolve(),{once:true}));
  const revoked=await f.call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({token:f.issued.access_token,client_id:f.client.client_id})});expect(revoked.status).toBe(200);await closed;
  expect((await recall()).status).toBe(403);expect((await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+device.device_token}}))).status).toBe(403);
 }finally{socket.close();}
});

async function connectedApp(f:Awaited<ReturnType<typeof enrolled>>,clientName:string,authorizationId:string){
 const registered=await f.call('/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:clientName,redirect_uris:['https://backend.composio.dev/api/v1/auth-apps/add'],token_endpoint_auth_method:'none'})});
 const client=await registered.json() as {client_id:string};
 await env.REGISTRIES.getByName(f.connectionId).addAuthorization({ownerId:'16027584',connectionId:f.connectionId,authorizationId,clientId:client.client_id,audience:'https://mcp.superlocalmemory.com/mcp',consentedTools:['recall','search','fetch','get_status','remember'],consentedScopes:['slm:read','slm:write'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 return client.client_id;
}

test('owner lists connected apps by name and removing one cuts access at once',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);
 const clientId=await connectedApp(f,'Composio','app-composio');
 const listed=await f.ownerCall('/owner/apps');expect(listed.status).toBe(200);
 const body=await listed.json() as {apps:Array<Record<string,unknown>>};
 expect(body.apps).toHaveLength(1);
 expect(body.apps[0]).toMatchObject({authorization_id:'app-composio',name:'Composio',client_host:'backend.composio.dev',permissions:{read:true,save:true,session:false},version:1,last_used_at_ms:null});
 expect(typeof body.apps[0].connected_at_ms).toBe('number');
 const removed=await f.ownerCall('/owner/apps/revoke',{authorization_id:'app-composio',expected_version:1});
 expect(removed.status).toBe(200);expect(await removed.json()).toEqual({revoked:true,version:2});
 const actor={ownerId:'16027584',connectionId:f.connectionId,authorizationId:'app-composio',clientId,audience:'https://mcp.superlocalmemory.com/mcp',credentialKind:'oauth' as const,scopes:['slm:read' as const]};
 expect(await env.REGISTRIES.getByName(f.connectionId).admit(actor,'https://mcp.superlocalmemory.com/mcp',{era:'legacy',rpcMethod:'tools/call',toolName:'recall',arguments:{query:'x'},originalBody:new Uint8Array()})).toMatchObject({allowed:false});
 expect((await (await f.ownerCall('/owner/apps')).json() as {apps:unknown[]}).apps).toEqual([]);
});

test('removing an app needs a well-formed request and the current version',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);await connectedApp(f,'Muse','app-muse');
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'app-muse',expected_version:7})).status).toBe(409);
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'app-muse'})).status).toBe(400);
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'../../etc',expected_version:1})).status).toBe(400);
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'not-there',expected_version:1})).status).toBe(404);
});

test('third-party app names come back as bounded plain text',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);
 await connectedApp(f,'Evil\u0000\u001b[31m'+'x'.repeat(500),'app-hostile');
 const body=await (await f.ownerCall('/owner/apps')).json() as {apps:Array<{name:string}>};
 expect(body.apps[0].name.length).toBeLessThanOrEqual(80);expect(body.apps[0].name).not.toMatch(/[\x00-\x1f\x7f]/);expect(body.apps[0].name.startsWith('Evil')).toBe(true);
});

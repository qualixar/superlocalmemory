import {env,createExecutionContext,waitOnExecutionContext} from 'cloudflare:test';
import {expect,test,vi} from 'vitest';
import {authFetch,type AuthWorkerEnv} from '../../src/worker-auth.ts';
import {authorizationServer} from '../../src/authorization-server.ts';
import {indexedToken} from '../../src/issued-token-protocol.ts';
import {validateIndexedToken} from '../../src/issuer-protocol.ts';
import {tokenHash} from '../../src/device-proof.ts';
const issuer='https://auth.superlocalmemory.com';
function cookies(response:Response){return response.headers.getSetCookie().map(x=>x.split(';')[0]).join('; ');}
function handle(page:string){const result=/name="handle" value="([^"]+)"/.exec(page);if(!result)throw new Error('missing handle');return result[1];}
test('real local OAuth consent, mocked GitHub identity, connection selection and PKCE token exchange',async()=>{
 const configuration={...env,GITHUB_CLIENT_ID:'synthetic-client',GITHUB_CLIENT_SECRET:'synthetic-secret'} as AuthWorkerEnv;
 const connectionId=crypto.randomUUID().replaceAll('-','');const installationId='installation-'+connectionId;const ownerId='16027584';const profileId='synthetic';const jkt='a'.repeat(43);
 const owner=env.OWNERS.getByName(ownerId);await owner.bind(ownerId,installationId,profileId,'native-'+connectionId,jkt);
 await owner.addConnection(ownerId,installationId,profileId,'native-'+connectionId,{connectionId,installationId,profileId,host:'muse',permissions:{read:true,write:true,correction:false,session:false},credentialEnvelope:'encrypted-synthetic',deviceDigest:'a'.repeat(64),deviceJkt:jkt,deviceExpiresAtMs:Date.now()+60000,generation:1,revokedAt:null,cleanupPending:false});
 const registry=env.REGISTRIES.getByName(connectionId);await registry.configure({connectionId,ownerId,installationId,profileId,origin:{kind:'relay',installationId,profileId},allowedTools:['recall','remember'],allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null});await registry.setEntitlement(ownerId,Date.now()+60000,0);
 async function call(path:string,init?:RequestInit){const ctx=createExecutionContext();const response=await authFetch(new Request(path.startsWith('https:')?path:issuer+path,init),configuration,ctx);await waitOnExecutionContext(ctx);return response;}
 const registration=await call('/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'Synthetic MCP client',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none',grant_types:['authorization_code','refresh_token'],response_types:['code']})});
 expect(registration.status).toBe(201);const client=await registration.json() as {client_id:string};const verifier='v'.repeat(43);
 const parameters=new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',scope:'slm:read slm:write slm:session slm:connect',resource:'https://mcp.superlocalmemory.com/mcp',state:'synthetic-client-state',code_challenge:await tokenHash(verifier),code_challenge_method:'S256'});
 const consent=await call('/authorize?'+parameters);expect(consent.status).toBe(200);const consentHandle=handle(await consent.text());
 const signIn=await call('/consent',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded',Origin:issuer,Cookie:cookies(consent)},body:new URLSearchParams({handle:consentHandle,decision:'allow'})});
 expect(signIn.status).toBe(302);const upstream=new URL(signIn.headers.get('Location')!);
 vi.stubGlobal('fetch',async(input:RequestInfo|URL)=>{const url=String(input);if(url==='https://github.com/login/oauth/access_token')return Response.json({access_token:'synthetic-provider-token',token_type:'bearer'});if(url==='https://api.github.com/user')return Response.json({id:16027584});throw new Error('unexpected outbound request');});
 try{
  const selection=await call('/github/callback?'+new URLSearchParams({code:'synthetic-upstream-code',state:upstream.searchParams.get('state')!}),{headers:{Cookie:cookies(signIn)}});
  expect(selection.status).toBe(200);const selectionHandle=handle(await selection.text());
  const completed=await call('/select',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded',Origin:issuer,Cookie:cookies(selection)},body:new URLSearchParams({handle:selectionHandle,connection_id:connectionId,decision:'allow'})});
  expect(completed.status).toBe(302);const returned=new URL(completed.headers.get('Location')!);expect(returned.searchParams.get('state')).toBe('synthetic-client-state');
  const tokenResponse=await call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({grant_type:'authorization_code',client_id:client.client_id,code:returned.searchParams.get('code')!,redirect_uri:'https://client.example/callback',code_verifier:verifier,resource:'https://mcp.superlocalmemory.com/mcp'})});
  expect(tokenResponse.status).toBe(200);const issued=await tokenResponse.json() as {access_token:string;scope:string};
  expect(issued.scope).toBe('slm:read');expect(await validateIndexedToken(issuer+'/owner',issued.access_token,configuration)).toBeNull();const validated=await authorizationServer.validateToken('https://mcp.superlocalmemory.com/mcp',issued.access_token,configuration);expect(validated?.userId).toBe(ownerId);expect(validated?.scope).toEqual(['slm:read']);
  expect(await indexedToken(issued.access_token,configuration)).toMatchObject({ownerId,connectionId,tokenKind:'access',revoked:false});
  expect((await validateIndexedToken('https://mcp.superlocalmemory.com/mcp',issued.access_token,configuration))?.userId).toBe(ownerId);
  const revoked=await call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({client_id:client.client_id,token:issued.access_token})});expect(revoked.status).toBe(200);
  expect(await validateIndexedToken('https://mcp.superlocalmemory.com/mcp',issued.access_token,configuration)).toBeNull();
 }finally{vi.unstubAllGlobals();}
});

// This browser regression uses a real authorization code but only a mocked identity provider.
async function browserSelect(callback:string){
 const settings={...env,GITHUB_CLIENT_ID:'synthetic-client',GITHUB_CLIENT_SECRET:'synthetic-secret'} as AuthWorkerEnv;
 const ctx=createExecutionContext();const api=authorizationServer.getOAuthApi(settings);
 const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'browser-fixture',redirect_uris:[callback],token_endpoint_auth_method:'none'})}),settings,ctx);const client=await registered.json() as {client_id:string};
 const request=await api.parseAuthRequest(new Request(issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:callback,resource:'https://mcp.superlocalmemory.com/mcp',scope:'slm:read',state:'browser-fixture-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'})));
 const ownerId='16027584';const connectionId=crypto.randomUUID().replaceAll('-','');const installation='i-'+connectionId;const native='n-'+connectionId;const jkt='a'.repeat(43);const owner=env.OWNERS.getByName(ownerId);await owner.bind(ownerId,installation,'profile',native,jkt);await owner.addConnection(ownerId,installation,'profile',native,{connectionId,installationId:installation,profileId:'profile',host:'composio',permissions:{read:true,write:false,correction:false,session:false},credentialEnvelope:'encrypted',deviceDigest:'a'.repeat(64),deviceJkt:jkt,deviceExpiresAtMs:Date.now()+60000,generation:1,revokedAt:null,cleanupPending:false});await env.REGISTRIES.getByName(connectionId).configure({connectionId,ownerId,installationId:installation,profileId:'profile',origin:{kind:'relay',installationId:installation,profileId:'profile'},allowedTools:['recall','search','fetch','get_status'],allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null});
 const consent=await api.beginConsent(request);await env.OAUTH_KV.put('slm-consent:'+consent.handle,JSON.stringify({request,ownerId}));
 const response=await authFetch(new Request(issuer+'/select',{method:'POST',headers:{Origin:issuer,Accept:'text/html','Content-Type':'application/x-www-form-urlencoded',Cookie:consent.headers.getSetCookie().map(x=>x.split(';')[0]).join('; ')},body:new URLSearchParams({handle:consent.handle,connection_id:connectionId,decision:'allow'})}),settings,ctx);
 await waitOnExecutionContext(ctx);return response;
}
test('browser client handoff ends the POST before an external callback redirect chain',async()=>{
 const response=await browserSelect('https://client.example/callback');
 expect(response.status).toBe(200);expect(response.headers.get('Content-Type')).toContain('text/html');const page=await response.text();expect(page).toContain('http-equiv="refresh"');expect(page).toContain('https://client.example/callback?');expect(page).toContain('Continue to your AI');
});
// The consent page already admits these loopback callbacks; the code is minted before the
// handoff renders, so rejecting them here would orphan a live grant behind a 500.
test.each(['http://localhost:33418/callback','http://[::1]:33418/callback','http://127.0.0.1:33418/callback'])('browser handoff returns loopback client %s instead of failing after code issue',async(callback)=>{
 const response=await browserSelect(callback);
 expect(response.status).toBe(200);const page=await response.text();expect(page).toContain('http-equiv="refresh"');expect(page).toContain(callback+'?');
});

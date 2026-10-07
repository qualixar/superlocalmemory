import {env,createExecutionContext,waitOnExecutionContext} from 'cloudflare:test';
import {describe,it,expect} from 'vitest';
import {authFetch,type AuthWorkerEnv} from '../../src/worker-auth.ts';
const issuer='https://auth.superlocalmemory.com';
const configuration=()=>({...env,GITHUB_CLIENT_ID:'synthetic-client',GITHUB_CLIENT_SECRET:'synthetic-secret'}) as AuthWorkerEnv;
describe('public auth Worker boundaries',()=>{
 it('serves maintained metadata only on exact issuer host',async()=>{
  const ctx=createExecutionContext();expect((await authFetch(new Request(issuer+'/.well-known/oauth-authorization-server'),configuration(),ctx)).status).toBe(200);
  expect((await authFetch(new Request('https://evil.example/.well-known/oauth-authorization-server'),configuration(),ctx)).status).toBe(403);await waitOnExecutionContext(ctx);
 });
 it('rejects consent without exact Origin or maintained browser cookie',async()=>{
  for(const origin of [undefined,'https://evil.example',issuer]){
   const ctx=createExecutionContext();const headers=new Headers({'Content-Type':'application/x-www-form-urlencoded'});if(origin)headers.set('Origin',origin);
   const response=await authFetch(new Request(issuer+'/consent',{method:'POST',headers,body:'handle=synthetic&decision=allow'}),configuration(),ctx);
   expect(response.status).toBeGreaterThanOrEqual(400);await waitOnExecutionContext(ctx);
  }
 });
 it('public bootstrap ids never confer identity',async()=>{
  const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+'/owner-login?connection_id='+'a'.repeat(32)),configuration(),ctx);
  expect(response.status).toBe(404);await waitOnExecutionContext(ctx);
 });
 it('invalid callback returns bounded failure without provider details',async()=>{
  const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+'/github/callback?state=synthetic&code=synthetic'),configuration(),ctx);
  expect(response.status).toBeGreaterThanOrEqual(400);expect(await response.text()).not.toContain('synthetic-secret');await waitOnExecutionContext(ctx);
 });
});

it('anonymous admission denial prevents registration and bootstrap allocation',async()=>{
 let sharedCalls=0;
 const settings={...configuration(),ANON_SETUP:{async limit(){return {success:false};}},ANON_GLOBAL:{async limit(){sharedCalls++;return {success:true};}},OAUTH_KV:new Proxy({}, {get(){throw new Error('storage must not be consulted');}}),BOOTSTRAPS:new Proxy({}, {get(){throw new Error('DO must not be allocated');}})} as AuthWorkerEnv;
 for(const path of ['/oauth/register','/bootstrap']){
  const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+path,{method:'POST',headers:{'Content-Type':'application/json','CF-Connecting-IP':'192.0.2.1'},body:'{}'}),settings,ctx);
  expect(response.status).toBe(429);expect(sharedCalls).toBe(0);expect(response.headers.get('Retry-After')).toBe('60');await waitOnExecutionContext(ctx);
 }
});

it('missing anonymous admission bindings fail closed',async()=>{
 const settings={...configuration(),ANON_SETUP:undefined,ANON_GLOBAL:undefined} as unknown as AuthWorkerEnv;
 const ctx=createExecutionContext();expect((await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}),settings,ctx)).status).toBe(503);await waitOnExecutionContext(ctx);
});
it('temporary diagnostic mode preserves real confidential registration and saves only safe shapes',async()=>{
 const ctx=createExecutionContext();const settings={...configuration(),DCR_DIAGNOSTICS:'1'};
 const response=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'private-test-name',redirect_uris:['https://client.example/callback?state=private-test-state'],token_endpoint_auth_method:'client_secret_post',grant_types:['authorization_code','refresh_token'],response_types:['code']})}),settings,ctx);
 expect(response.status).toBe(201);const body=await response.json() as {client_id:string;client_secret:string};expect(body.client_id).toBeTruthy();expect(body.client_secret).toBeTruthy();await waitOnExecutionContext(ctx);
 const keys=await env.OAUTH_KV.list({prefix:'slm-dcr-diagnostic:'});expect(keys.keys.length).toBeGreaterThan(0);
 const record=await env.OAUTH_KV.get(keys.keys[keys.keys.length-1].name);expect(record).not.toContain(body.client_secret);expect(record).not.toContain('private-test-name');expect(record).not.toContain('private-test-state');expect(JSON.parse(record!)).toMatchObject({status:201,method:'client_secret_post'});
});
it('MCP consent excludes native scope when client asks for the whole advertised scope list',async()=>{
 const ctx=createExecutionContext();const settings=configuration();
 const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'scope-interoperability-fixture',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none'})}),settings,ctx);
 const client=await registered.json() as {client_id:string};
 const uri=issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',resource:'https://mcp.superlocalmemory.com/mcp',scope:'slm:read slm:write slm:session slm:connect',state:'synthetic-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'});
 const response=await authFetch(new Request(uri),settings,ctx);expect(response.status).toBe(200);const page=await response.text();expect(page).toContain('slm:read');expect(page).not.toContain('slm:connect');await waitOnExecutionContext(ctx);
});

it('consent page permits the real GitHub redirect and uses nonce-bound SLM styling',async()=>{
 const ctx=createExecutionContext();const settings=configuration();const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'fixture',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none'})}),settings,ctx);const client=await registered.json() as {client_id:string};
 const response=await authFetch(new Request(issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',resource:'https://mcp.superlocalmemory.com/mcp',scope:'slm:read',state:'synthetic-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'})),settings,ctx);
 const policy=response.headers.get('Content-Security-Policy')!;expect(policy).toContain("form-action 'self' https://github.com https://client.example");expect(policy).toMatch(/style-src 'nonce-[^']+'/);expect(policy).not.toContain('unsafe-inline');const page=await response.text();expect(page).toContain('Sign in with GitHub');expect(page).toMatch(/<style nonce="[^"]+">/);await waitOnExecutionContext(ctx);
});
it('expired or consumed browser consent shows recovery instructions instead of raw JSON',async()=>{
 const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+'/consent',{method:'POST',headers:{Origin:issuer,Accept:'text/html','Content-Type':'application/x-www-form-urlencoded'},body:'handle=synthetic&decision=allow'}),configuration(),ctx);
 expect(response.status).toBe(400);expect(response.headers.get('Content-Type')).toContain('text/html');const page=await response.text();expect(page).toContain('Return to your SLM dashboard');expect(page).toContain('consent_unavailable');await waitOnExecutionContext(ctx);
});

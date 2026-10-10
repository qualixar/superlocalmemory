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
it('client registration keeps no diagnostic records, even when the old diagnostic switch is set',async()=>{
 const ctx=createExecutionContext();const settings={...configuration(),DCR_DIAGNOSTICS:'1'} as unknown as AuthWorkerEnv;
 const response=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'private-test-name',redirect_uris:['https://backend.composio.dev/api/v1/auth-apps/add'],token_endpoint_auth_method:'client_secret_basic'})}),settings,ctx);
 expect(response.status).toBe(201);await waitOnExecutionContext(ctx);
 const keys=await env.OAUTH_KV.list({prefix:'slm-dcr-diagnostic:'});expect(keys.keys.length).toBe(0);
});
it('MCP consent excludes native scope when client asks for the whole advertised scope list',async()=>{
 const ctx=createExecutionContext();const settings=configuration();
 const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'scope-interoperability-fixture',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none'})}),settings,ctx);
 const client=await registered.json() as {client_id:string};
 const uri=issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',resource:'https://mcp.superlocalmemory.com/mcp',scope:'slm:read slm:write slm:session slm:connect',state:'synthetic-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'});
 const response=await authFetch(new Request(uri),settings,ctx);expect(response.status).toBe(200);const page=await response.text();expect(page).toContain('slm:read');expect(page).not.toContain('slm:connect');expect(page).toContain('Granted: slm:read<');expect(page).toContain('Only if selected above: slm:write, slm:session<');await waitOnExecutionContext(ctx);
});

it('omitting resource never reaches native owner enrollment',async()=>{
 const ctx=createExecutionContext();const settings=configuration();
 const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'owner-probe',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none'})}),settings,ctx);const client=await registered.json() as {client_id:string};
 const response=await authFetch(new Request(issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',scope:'slm:connect',state:'synthetic-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'})),settings,ctx);
 expect(response.status).toBeGreaterThanOrEqual(400);expect(await response.text()).not.toContain('Sign in with GitHub');await waitOnExecutionContext(ctx);
});
it('consent page permits the real GitHub redirect and uses nonce-bound SLM styling',async()=>{
 const ctx=createExecutionContext();const settings=configuration();const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'fixture',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none'})}),settings,ctx);const client=await registered.json() as {client_id:string};
 const response=await authFetch(new Request(issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',resource:'https://mcp.superlocalmemory.com/mcp',scope:'slm:read',state:'synthetic-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'})),settings,ctx);
 expect(response.headers.get('Referrer-Policy')).toBe('strict-origin');const policy=response.headers.get('Content-Security-Policy')!;expect(policy).toContain("form-action 'self' https://github.com https://client.example");expect(policy).toMatch(/style-src 'nonce-[^']+'/);expect(policy).not.toContain('unsafe-inline');const page=await response.text();expect(page).toContain('Sign in with GitHub');expect(page).toContain('name="referrer" content="strict-origin"');expect(page).toMatch(/<style nonce="[^"]+">/);await waitOnExecutionContext(ctx);
});
it('a sign-in that already finished says so when its page is submitted again',async()=>{
 const settings=configuration();await settings.OAUTH_KV.put('slm-consent-done:finished-handle','1',{expirationTtl:3600});
 for(const path of ['/consent','/select']){
  const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+path,{method:'POST',headers:{Origin:issuer,Accept:'text/html','Content-Type':'application/x-www-form-urlencoded'},body:'handle=finished-handle&decision=allow&connection_id='+'a'.repeat(32)}),settings,ctx);
  expect(response.status).toBe(200);const page=await response.text();expect(page).toContain('This sign-in already finished');expect(page).toContain('close this page');expect(page).not.toContain('no longer valid');
  const api=await authFetch(new Request(issuer+path,{method:'POST',headers:{Origin:issuer,'Content-Type':'application/x-www-form-urlencoded'},body:'handle=finished-handle&decision=allow'}),settings,ctx);
  expect(api.status).toBe(409);expect(await api.json()).toEqual({error:'sign_in_already_complete'});await waitOnExecutionContext(ctx);
 }
});
it('expired or consumed browser consent shows recovery instructions instead of raw JSON',async()=>{
 const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+'/consent',{method:'POST',headers:{Origin:issuer,Accept:'text/html','Content-Type':'application/x-www-form-urlencoded'},body:'handle=synthetic&decision=allow'}),configuration(),ctx);
 expect(response.status).toBe(302);const location=new URL(response.headers.get('Location')!);expect(location.pathname).toBe('/sign-in/error');expect(location.searchParams.get('reason')).toBe('consent_unavailable');const recovery=await authFetch(new Request(location),configuration(),ctx);expect(recovery.headers.get('Content-Type')).toContain('text/html');const page=await recovery.text();expect(page).toContain('Return to your SLM dashboard');expect(page).toContain('Restart sign-in');await waitOnExecutionContext(ctx);
});


it('refreshable recovery URL contains no callback code or state',async()=>{
 const ctx=createExecutionContext();const result=await authFetch(new Request(issuer+'/github/callback?code=synthetic-secret-code&state=synthetic-secret-state',{headers:{Accept:'text/html'}}),configuration(),ctx);expect(result.status).toBe(302);const location=result.headers.get('Location')!;expect(location).not.toContain('synthetic-secret');expect(new URL(location).pathname).toBe('/sign-in/error');const page=await authFetch(new Request(location),configuration(),ctx);expect(await page.text()).toContain('Restart sign-in');await waitOnExecutionContext(ctx);
});

it('consent page offers the two optional boxes, unticked, only when the app asks for them',async()=>{
 const settings=configuration();
 async function page(scope:string){
  const ctx=createExecutionContext();
  const registered=await authFetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'mesh-fixture',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none'})}),settings,ctx);
  const client=await registered.json() as {client_id:string};
  const response=await authFetch(new Request(issuer+'/authorize?'+new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:'https://client.example/callback',resource:'https://mcp.superlocalmemory.com/mcp',scope,state:'synthetic-state',code_challenge:'a'.repeat(43),code_challenge_method:'S256'})),settings,ctx);
  expect(response.status).toBe(200);await waitOnExecutionContext(ctx);return response.text();
 }
 const all=await page('slm:read slm:write slm:session slm:mesh slm:media');
 expect(all).toContain('<input type="checkbox" name="mesh" value="yes"> Allow talking to your other bots');
 expect(all).toContain('<input type="checkbox" name="media" value="yes"> Allow images and documents');
 expect(all).not.toMatch(/name="(mesh|media|write|session)"[^>]*checked/);
 expect(all).toContain('Only if selected above: slm:write, slm:session, slm:mesh, slm:media<');
 const some=await page('slm:read slm:mesh');
 expect(some).toContain('name="mesh"');expect(some).not.toContain('name="media"');expect(some).not.toContain('name="write"');
 const plain=await page('slm:read slm:write slm:session');
 expect(plain).not.toContain('name="mesh"');expect(plain).not.toContain('name="media"');expect(plain).toContain('Only if selected above: slm:write, slm:session<');
});

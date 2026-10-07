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

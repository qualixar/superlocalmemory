import {env,createExecutionContext,waitOnExecutionContext} from 'cloudflare:test';
import {describe,it,expect} from 'vitest';
import {authorizationServer,type AuthorizationEnv} from '../../src/authorization-server.ts';
const issuer='https://auth.superlocalmemory.com';
const contextEnv=()=>({...env,OAUTH_KV:env.OAUTH_KV}) as AuthorizationEnv;
describe('maintained production authorization protocol',()=>{
 it('publishes issuer, S256 and DCR endpoints',async()=>{
  const ctx=createExecutionContext();const response=await authorizationServer.fetch(new Request(issuer+'/.well-known/oauth-authorization-server'),contextEnv(),ctx);
  expect(response.status).toBe(200);const body=await response.json() as Record<string,unknown>;
  expect(body.issuer).toBe(issuer);expect(body.registration_endpoint).toBe(issuer+'/oauth/register');
  expect(body.code_challenge_methods_supported).toEqual(['S256']);await waitOnExecutionContext(ctx);
 });
 it('registers an HTTPS public client without issuing memory access',async()=>{
  const ctx=createExecutionContext();const response=await authorizationServer.fetch(new Request(issuer+'/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'SLM synthetic client',redirect_uris:['https://client.example/callback'],token_endpoint_auth_method:'none',grant_types:['authorization_code','refresh_token'],response_types:['code']})}),contextEnv(),ctx);
  expect(response.status).toBe(201);const body=await response.json() as Record<string,unknown>;
  expect(typeof body.client_id).toBe('string');expect(body.access_token).toBeUndefined();await waitOnExecutionContext(ctx);
 });
 it('does not issue a token for an invented authorization code',async()=>{
  const ctx=createExecutionContext();const response=await authorizationServer.fetch(new Request(issuer+'/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({grant_type:'authorization_code',code:'synthetic-invalid',client_id:'synthetic-client',redirect_uri:'https://client.example/callback',code_verifier:'a'.repeat(43),resource:'https://mcp.superlocalmemory.com/mcp'})}),contextEnv(),ctx);
  expect(response.status).toBeGreaterThanOrEqual(400);await waitOnExecutionContext(ctx);
 });
});

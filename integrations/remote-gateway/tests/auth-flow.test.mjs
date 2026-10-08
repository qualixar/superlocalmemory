import test from 'node:test';
import assert from 'node:assert/strict';
import { githubAuthorizationUrl, exchangeGithubCode, renderConsentPage } from '../src/auth-flow.ts';
test('GitHub authorization URL binds exact callback, S256 and library state',()=>{
 const url=new URL(githubAuthorizationUrl('client','safe-state','a'.repeat(43)));
 assert.equal(url.origin,'https://github.com');assert.equal(url.searchParams.get('redirect_uri'),'https://auth.superlocalmemory.com/github/callback');assert.equal(url.searchParams.get('code_challenge_method'),'S256');assert.equal(url.searchParams.get('state'),'safe-state');
});
test('GitHub code exchange uses exact endpoint and denies redirect',async()=>{
 let call;const result=await exchangeGithubCode('client','synthetic-secret','synthetic-code','a'.repeat(43),async(url,options)=>{call={url,options};return Response.json({access_token:'synthetic-token',token_type:'bearer'});});
 assert.equal(result,'synthetic-token');assert.equal(call.url,'https://github.com/login/oauth/access_token');assert.equal(call.options.redirect,'manual');
 assert.equal(call.options.body.get('redirect_uri'),'https://auth.superlocalmemory.com/github/callback');
});
test('provider errors are bounded categories',async()=>{
 await assert.rejects(exchangeGithubCode('client','synthetic','synthetic','a'.repeat(43),async()=>Response.json({error:'SYNTHETIC_SECRET'})),/identity_exchange_failed/);
});
test('consent rendering escapes client-controlled text',()=>{
 const html=renderConsentPage('handle',{clientName:'<script>bad</script>',redirectHostname:'client.example',redirectIsLoopback:false,scope:['slm:read']});
 assert(!html.includes('<script>'));assert(html.includes('&lt;script&gt;'));assert(html.includes('SuperLocalMemory'));
});

test('dashboard recovery receipt only routes to an authenticated loopback port, never grants access',async()=>{
 const {dashboardReturnCookie,dashboardReturnUrl}=await import('../src/auth-flow.ts');const key='a'.repeat(64);
 const cookie=await dashboardReturnCookie('http://127.0.0.1:18767/api/v3/connections/callback',key);assert.ok(cookie.includes('HttpOnly'));assert.ok(cookie.includes('Secure'));const request=new Request('https://auth.superlocalmemory.com/sign-in/error',{headers:{Cookie:cookie.split(';')[0]}});
 assert.equal(await dashboardReturnUrl(request,key),'http://127.0.0.1:18767/#mcp-pane');assert.equal(await dashboardReturnUrl(request,'b'.repeat(64)),undefined);assert.equal(await dashboardReturnCookie('https://evil.example/callback',key),null);assert.equal(await dashboardReturnUrl(new Request(request.url,{headers:{Cookie:cookie.split(';')[0]+'tampered'}}),key),undefined);
});

test('provider failures do not falsely blame expired sign-in links',async()=>{const {renderAuthFailure}=await import('../src/auth-flow.ts');const page=renderAuthFailure('identity_exchange_unavailable');assert.match(page,/could not complete sign-in/);assert.doesNotMatch(page,/link expired or was already used/);});

test('known GitHub exchange failures map to fixed safe categories without server descriptions',async()=>{
 await assert.rejects(exchangeGithubCode('client','synthetic','code','a'.repeat(43),async()=>Response.json({error:'incorrect_client_credentials',error_description:'SYNTHETIC_SECRET'})),/^Error: identity_client_configuration$/);
 await assert.rejects(exchangeGithubCode('client','synthetic','code','a'.repeat(43),async()=>Response.json({error:'bad_verification_code',error_description:'SYNTHETIC_SECRET'})),/^Error: identity_code_rejected$/);
});


test('GitHub redirect is rejected even if its body resembles a token response',async()=>{await assert.rejects(exchangeGithubCode('client','synthetic','code','a'.repeat(43),async()=>new Response(JSON.stringify({access_token:'synthetic-token'}),{status:302,headers:{Location:'https://evil.example/'}})),/identity_exchange_failed/);});

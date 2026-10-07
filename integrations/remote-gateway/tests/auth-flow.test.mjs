import test from 'node:test';
import assert from 'node:assert/strict';
import { githubAuthorizationUrl, exchangeGithubCode, renderConsentPage } from '../src/auth-flow.ts';
test('GitHub authorization URL binds exact callback, S256 and library state',()=>{
 const url=new URL(githubAuthorizationUrl('client','safe-state','a'.repeat(43)));
 assert.equal(url.origin,'https://github.com');assert.equal(url.searchParams.get('redirect_uri'),'https://auth.superlocalmemory.com/github/callback');assert.equal(url.searchParams.get('code_challenge_method'),'S256');assert.equal(url.searchParams.get('state'),'safe-state');
});
test('GitHub code exchange uses exact endpoint and denies redirect',async()=>{
 let call;const result=await exchangeGithubCode('client','synthetic-secret','synthetic-code','a'.repeat(43),async(url,options)=>{call={url,options};return Response.json({access_token:'synthetic-token',token_type:'bearer'});});
 assert.equal(result,'synthetic-token');assert.equal(call.url,'https://github.com/login/oauth/access_token');assert.equal(call.options.redirect,'error');
 assert.equal(call.options.body.get('redirect_uri'),'https://auth.superlocalmemory.com/github/callback');
});
test('provider errors are bounded categories',async()=>{
 await assert.rejects(exchangeGithubCode('client','synthetic','synthetic','a'.repeat(43),async()=>Response.json({error:'SYNTHETIC_SECRET'})),/identity_exchange_failed/);
});
test('consent rendering escapes client-controlled text',()=>{
 const html=renderConsentPage('handle',{clientName:'<script>bad</script>',redirectHostname:'client.example',redirectIsLoopback:false,scope:['slm:read']});
 assert(!html.includes('<script>'));assert(html.includes('&lt;script&gt;'));assert(html.includes('SuperLocalMemory'));
});

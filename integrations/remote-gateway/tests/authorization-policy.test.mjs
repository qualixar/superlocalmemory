import test from 'node:test';
import assert from 'node:assert/strict';
import { validateAuthorizationRequest, selectedScopes, verifyGithubIdentity } from '../src/authorization-policy.ts';
const request={issuer:'https://auth.superlocalmemory.com',resource:'https://mcp.superlocalmemory.com/mcp',responseType:'code',clientId:'client',redirectUri:'https://client.example/callback',scope:['slm:read'],state:'state',codeChallenge:'a'.repeat(43),codeChallengeMethod:'S256'};
test('requires S256 and exact configured resource',()=>{
 assert.equal(validateAuthorizationRequest(request),true);
 for(const change of [{codeChallengeMethod:'plain'},{codeChallenge:undefined},{resource:'https://evil.example'},{scope:['slm:connect']},{scope:[]},{scope:['slm:read','invented']}])assert.equal(validateAuthorizationRequest({...request,...change}),false);
});
test('native owner scope cannot become memory scope',()=>{
 assert.equal(validateAuthorizationRequest({...request,resource:'https://auth.superlocalmemory.com/owner',scope:['slm:connect']}),true);
 assert.equal(validateAuthorizationRequest({...request,resource:'https://auth.superlocalmemory.com/owner',scope:['slm:connect','slm:read']}),false);
});
test('connection ceilings reduce scopes',()=>{
 assert.deepEqual(selectedScopes(['slm:read','slm:write','slm:session'],{read:true,write:false,session:true}),['slm:read','slm:session']);
 assert.deepEqual(selectedScopes(['slm:write'],{read:true,write:true,session:true}),[]);
});
test('GitHub numeric account pilot identity is verified through /user',async()=>{
 const calls=[];const fetcher=async(url,options)=>{calls.push({url,options});return Response.json({id:16027584,login:'varun369'});};
 assert.equal(await verifyGithubIdentity('synthetic-token',fetcher), '16027584');
 assert.equal(calls[0].url,'https://api.github.com/user');
 assert.equal(calls[0].options.redirect,'error');
});
test('wrong GitHub identity and malformed/failing replies denied',async()=>{
 for(const body of [{id:2},{id:'16027584'},{id:16027584.5}])await assert.rejects(verifyGithubIdentity('synthetic-token',async()=>Response.json(body)),/identity_denied/);
 await assert.rejects(verifyGithubIdentity('synthetic-token',async()=>new Response('no',{status:401})),/identity_unavailable/);
});

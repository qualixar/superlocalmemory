import test from 'node:test';
import assert from 'node:assert/strict';
import { validateAuthorizationRequest, selectedScopes, ticked, verifyGithubIdentity } from '../src/authorization-policy.ts';
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
 assert.equal(calls[0].options.redirect,'manual');
});
test('wrong GitHub identity and malformed/failing replies denied',async()=>{
 for(const body of [{id:0},{id:'16027584'},{id:16027584.5}])await assert.rejects(verifyGithubIdentity('synthetic-token',async()=>Response.json(body)),/identity_denied/);
 await assert.rejects(verifyGithubIdentity('synthetic-token',async()=>new Response('no',{status:401})),/identity_unavailable/);
});

test('end-user signup accepts any verified numeric GitHub account without founder allowlist',async()=>{assert.equal(await verifyGithubIdentity('synthetic-token',async()=>Response.json({id:42,login:'synthetic-end-user'})),'42');});

test('memory authorization narrows all-advertised scopes without granting native enrollment',async()=>{
 const {memoryAuthorizationRequest}=await import('../src/authorization-policy.ts');
 const original={...request,scope:['slm:read','slm:write','slm:session','slm:connect']};
 const narrowed=memoryAuthorizationRequest(original);
 assert.deepEqual(narrowed.scope,['slm:read','slm:write','slm:session']);assert.equal(validateAuthorizationRequest(narrowed),true);
 assert.deepEqual(original.scope,['slm:read','slm:write','slm:session','slm:connect']);
 for(const change of [{scope:['slm:connect']},{scope:['slm:read','invented']},{scope:['slm:read','slm:connect','slm:connect']},{resource:'https://auth.superlocalmemory.com/owner'},{codeChallengeMethod:'plain'}])assert.equal(memoryAuthorizationRequest({...original,...change}),null);
});

test('mesh and media scopes are valid memory scopes and read is still required',()=>{
 for(const scope of [['slm:read','slm:mesh'],['slm:read','slm:media'],['slm:read','slm:write','slm:session','slm:mesh','slm:media']])assert.equal(validateAuthorizationRequest({...request,scope}),true);
 for(const scope of [['slm:mesh'],['slm:media','slm:write']])assert.equal(validateAuthorizationRequest({...request,scope}),false);
 assert.equal(validateAuthorizationRequest({...request,scope:['slm:read','slm:mesh','slm:connect']}),false);
});
test('mesh and media scopes pass through when requested and read stays required',()=>{
 const ceiling={read:true,write:false,session:false};
 assert.deepEqual(selectedScopes(['slm:read','slm:mesh','slm:media'],ceiling),['slm:read','slm:mesh','slm:media']);
 assert.deepEqual(selectedScopes(['slm:read','slm:write','slm:media'],ceiling),['slm:read','slm:media']);
 assert.deepEqual(selectedScopes(['slm:mesh','slm:media'],{read:true,write:true,session:true}),[]);
 assert.deepEqual(selectedScopes(['slm:read','slm:mesh'],{read:false,write:true,session:true}),[]);
 assert.deepEqual(selectedScopes(['slm:read'],ceiling),['slm:read']);
});
test('only ticked boxes add scopes on the consent page and read is always kept',()=>{
 const fields=values=>name=>values[name]??null;
 assert.deepEqual(ticked(['slm:read','slm:write','slm:session','slm:mesh','slm:media'],fields({})),['slm:read']);
 assert.deepEqual(ticked(['slm:read','slm:write','slm:session','slm:mesh','slm:media'],fields({mesh:'yes',media:'yes',write:'yes'})),['slm:read','slm:write','slm:mesh','slm:media']);
 assert.deepEqual(ticked(['slm:read','slm:mesh'],fields({media:'yes',mesh:'no'})),['slm:read']);
 assert.deepEqual(ticked(['slm:read','slm:mesh'],fields({mesh:'yes',media:'yes'})),['slm:read','slm:mesh']);
});

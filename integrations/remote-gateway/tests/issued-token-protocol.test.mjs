import test from 'node:test';import assert from 'node:assert/strict';
import {tokenDigest,recordIssuedTokens} from '../src/issued-token-protocol.ts';
test('token digest is stable SHA256 without raw token contents',async()=>{const value=await tokenDigest('synthetic-token');assert.match(value,/^[a-f0-9]{64}$/);assert.equal(await tokenDigest('synthetic-token'),value);});
test('issuance records both access and refresh references before response',async()=>{
 const stored=[];const env={TOKEN_INDEX:{getByName(){return {record:async refs=>stored.push(...refs)};}}};
 const api={unwrapToken:async()=>({expiresAt:Math.floor(Date.now()/1000)+3600,userId:'16027584',audience:'https://mcp.superlocalmemory.com/mcp',grant:{clientId:'client',props:{ownerId:'16027584',authorizationId:'grant',connectionId:'connection'}}})};
 await recordIssuedTokens({access_token:'synthetic-access',refresh_token:'synthetic-refresh'},env,api);
 assert.equal(stored.length,2);assert.deepEqual(stored.map(x=>x.tokenKind),['access','refresh']);assert(!JSON.stringify(stored).includes('synthetic-access'));
});
test('ledger failure prevents successful issuance acknowledgement',async()=>{
 const env={TOKEN_INDEX:{getByName(){return {record:async()=>{throw new Error('unavailable');}};}}};
 const api={unwrapToken:async()=>({expiresAt:Math.floor(Date.now()/1000)+3600,userId:'16027584',audience:'https://mcp.superlocalmemory.com/mcp',grant:{clientId:'client',props:{ownerId:'16027584',authorizationId:'grant',connectionId:'connection'}}})};
 await assert.rejects(recordIssuedTokens({access_token:'synthetic-access'},env,api));
});

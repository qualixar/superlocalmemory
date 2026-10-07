import {env} from 'cloudflare:workers';
import {evictDurableObject} from 'cloudflare:test';
import {expect,test} from 'vitest';
const ref={tokenDigest:'a'.repeat(64),tokenKind:'access' as const,authorizationId:'auth-a',ownerId:'owner-a',clientId:'https://client.example/mcp.json',connectionId:'connection-a',audience:'https://mcp.superlocalmemory.com/mcp',validUntil:new Date(Date.now()+60000).toISOString()};
async function setup(){const stub=env.TOKEN_INDEX.getByName(crypto.randomUUID());await stub.record([ref]);return stub;}
async function rejected(operation:()=>Promise<unknown>,code:string){await expect((async()=>{await operation();})()).rejects.toThrow(code);}
test('token index stores metadata only and supports CIMD client identity',async()=>{const stub=await setup();const found=await stub.lookup(ref.tokenDigest);expect(found).toMatchObject({...ref,revoked:false});expect(JSON.stringify(found)).not.toContain('access_token');});
test('unknown token digest has no authority',async()=>{const stub=await setup();expect(await stub.lookup('b'.repeat(64))).toBeNull();});
test('revocation is durable before success and cannot be overwritten',async()=>{const stub=await setup();expect(await stub.revoke(ref.tokenDigest,ref.clientId)).toBe(true);await evictDurableObject(stub);expect(await stub.lookup(ref.tokenDigest)).toMatchObject({revoked:true});await rejected(()=>stub.record([ref]),'token_revoked');});
test('wrong client cannot revoke a known token',async()=>{const stub=await setup();expect(await stub.revoke(ref.tokenDigest,'other-client')).toBe(false);expect(await stub.lookup(ref.tokenDigest)).toMatchObject({revoked:false});});
test('same digest cannot acquire another owner or authorization',async()=>{const stub=await setup();await rejected(()=>stub.record([{...ref,ownerId:'other'}]),'token_conflict');});
test('expired references are refused for new issuance',async()=>{const stub=env.TOKEN_INDEX.getByName(crypto.randomUUID());await rejected(()=>stub.record([{...ref,validUntil:'2000-01-01T00:00:00.000Z'}]),'invalid_token_reference');});
test('malformed references and unknown token kinds fail closed',async()=>{const stub=env.TOKEN_INDEX.getByName(crypto.randomUUID());await rejected(()=>stub.record([{...ref,tokenKind:'device' as never}]),'invalid_token_reference');await rejected(()=>stub.record([{...ref,tokenDigest:'plain-secret'}]),'invalid_token_reference');});
test('token index exposes no public HTTP administration',async()=>{const stub=await setup();expect((await stub.fetch(new Request('https://private.invalid/token'))).status).toBe(404);});

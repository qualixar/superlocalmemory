import {env,createExecutionContext} from 'cloudflare:test';import {expect,test} from 'vitest';
import {generateKeyPair,exportJWK,calculateJwkThumbprint,SignJWT} from 'jose';
import {connectFetch,type ConnectEnv} from '../../src/worker-connect.ts';import {tokenHash} from '../../src/device-proof.ts';import {tokenDigest} from '../../src/issued-token-protocol.ts';
const endpoint='https://connect.superlocalmemory.com/connector';
test('public connector refuses bearer alone before relay',async()=>{const response=await connectFetch(new Request(endpoint,{headers:{Upgrade:'websocket',Authorization:'Bearer '+ 'a'.repeat(64)}}),env as ConnectEnv,createExecutionContext());expect(response.status).toBe(401);});
test('public connector needs registered proof key and consumes jti durably',async()=>{
 const pair=await generateKeyPair('ES256',{extractable:true});const jwk=await exportJWK(pair.publicKey);const jkt=await calculateJwkThumbprint(jwk);const token='b'.repeat(64);const digest=await tokenDigest(token);const connectionId=crypto.randomUUID().replaceAll('-','');const installationId='i-'+connectionId;const profileId='synthetic';const ownerId='16027584';
 await env.DEVICES.getByName(digest).configure({ownerId,connectionId,installationId,profileId,deviceDigest:digest,deviceJkt:jkt,expiresAtMs:Date.now()+60000});
 const owner=env.OWNERS.getByName(ownerId);await owner.bind(ownerId,installationId,profileId,'native-'+connectionId,jkt);await owner.addConnection(ownerId,installationId,profileId,'native-'+connectionId,{connectionId,installationId,profileId,host:'muse',permissions:{read:true,write:false,correction:false,session:false},credentialEnvelope:'encrypted-synthetic',deviceDigest:digest,deviceJkt:jkt,deviceExpiresAtMs:Date.now()+60000,generation:1,revokedAt:null,cleanupPending:false});
 const proof=await new SignJWT({htm:'GET',htu:endpoint,ath:await tokenHash(token)}).setProtectedHeader({typ:'dpop+jwt',alg:'ES256',jwk}).setIssuedAt().setJti(crypto.randomUUID()).sign(pair.privateKey);
 const request=()=>new Request(endpoint,{headers:{Upgrade:'websocket',Authorization:'Bearer '+token,DPoP:proof}});
 const first=await connectFetch(request(),env as ConnectEnv,createExecutionContext());expect(first.status).toBe(503); // no configured relay; never fake connected
 expect((await connectFetch(request(),env as ConnectEnv,createExecutionContext())).status).toBe(401);
});
test('connector infrastructure failure is retryable rather than an authentication rejection',async()=>{
 const broken={...env,DEVICES:{getByName(){throw new Error('synthetic storage outage');}}} as unknown as ConnectEnv;
 const response=await connectFetch(new Request(endpoint,{headers:{Upgrade:'websocket',Authorization:'Bearer '+'a'.repeat(64),DPoP:'synthetic-proof'}}),broken,createExecutionContext());expect(response.status).toBe(503);expect(await response.json()).toMatchObject({error:'connector_unavailable'});
});

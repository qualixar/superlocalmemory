import test from 'node:test';import assert from 'node:assert/strict';
import {queueProviderCleanup,retryProviderCleanup} from '../src/provider-cleanup.ts';
test('failed provider cleanup is persisted without credential bodies and retried',async()=>{
 const stored=new Map();const env={OAUTH_KV:{async put(key,value){stored.set(key,value);},async list(){return {keys:[...stored.keys()].map(name=>({name}))};},async get(key){return stored.get(key);},async delete(key){stored.delete(key);}}};
 await queueProviderCleanup(env,{userId:'owner',grantId:'grant'});assert.equal(stored.size,1);assert.doesNotMatch([...stored.values()].join(''),/token|PRIVATE KEY/);
 let attempts=0;const api={async revokeGrant(grant,user){assert.equal(grant,'grant');assert.equal(user,'owner');attempts++;}};
 await retryProviderCleanup(env,api);assert.equal(attempts,1);assert.equal(stored.size,0);
});
test('retry failure leaves durable cleanup evidence',async()=>{
 const stored=new Map();const env={OAUTH_KV:{async put(key,value){stored.set(key,value);},async list(){return {keys:[...stored.keys()].map(name=>({name}))};},async get(key){return stored.get(key);},async delete(key){stored.delete(key);}}};
 await queueProviderCleanup(env,{userId:'owner',grantId:'grant'});
 await retryProviderCleanup(env,{async revokeGrant(){throw new Error('synthetic outage');}});
 assert.equal(stored.size,1);
});

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {pruneOtherOwnerConnections} from '../src/prune-owner-connections.ts';

const fixture = () => {
 const ownerId='1234';
 const entries=[
  {connectionId:'a'.repeat(32),ownerId,profileId:'default',host:'chatgpt',revokedAt:null,generation:2,deviceDigest:'a'.repeat(64)},
  {connectionId:'b'.repeat(32),ownerId,profileId:'default',host:'chatgpt',revokedAt:null,generation:1,deviceDigest:'b'.repeat(64)},
  {connectionId:'c'.repeat(32),ownerId,profileId:'default',host:'chatgpt',revokedAt:null,generation:1,deviceDigest:'c'.repeat(64)},
  {connectionId:'d'.repeat(32),ownerId,profileId:'default',host:'claude_web',revokedAt:null,generation:1,deviceDigest:'d'.repeat(64)},
  {connectionId:'e'.repeat(32),ownerId,profileId:'bughunting',host:'chatgpt',revokedAt:null,generation:1,deviceDigest:'e'.repeat(64)},
  {connectionId:'f'.repeat(32),ownerId,profileId:'default',host:'chatgpt',revokedAt:'2026-10-23',generation:2,deviceDigest:'f'.repeat(64)}
 ];
 const steps=[];
 const owner={
   async list(id){assert.equal(id,ownerId);return structuredClone(entries);},
   async getConnection(id,key){assert.equal(id,ownerId);return structuredClone(entries.find(x=>x.connectionId===key)??null);},
   async cancelConnection(id,key){steps.push(['cancel',key]);},
   async revoke(id,key,generation){steps.push(['revoke',key,generation]);const row=entries.find(x=>x.connectionId===key);if(row)row.revokedAt='now';return generation+1;},
   async markCleanup(id,key,generation){steps.push(['cleanup',key,generation]);return true;}
 };
 const env={
   REGISTRIES:{getByName(id){return {async revokeConnection(){steps.push(['registry',id]);}}}},
   DEVICES:{getByName(id){return {async revoke(){steps.push(['device',id]);}}}},
   RELAYS:{getByName(id){return {async revoke(){steps.push(['relay',id]);}}}}
 };
 return {owner,ownerId,env,steps,entries};
};
test('explicit owner-scoped cleanup revokes only other same-profile same-host connections',async()=>{
 const f=fixture();const ids=await pruneOtherOwnerConnections({owner:f.owner,ownerId:f.ownerId,selectedId:'a'.repeat(32),env:f.env});
 assert.deepEqual(ids,['b'.repeat(32),'c'.repeat(32)]);
 assert.equal(f.steps.filter(x=>x[0]==='revoke').length,2);
 assert.ok(!f.steps.some(x=>['a','d','e','f'].includes(x[1]?.[0])));
 assert.equal(f.entries.filter(x=>x.revokedAt===null&&x.profileId==='default'&&x.host==='chatgpt').length,1);
});
test('rejects stale selected connection before any revocations',async()=>{
 const f=fixture();f.entries[0].revokedAt='old';
 await assert.rejects(()=>pruneOtherOwnerConnections({owner:f.owner,ownerId:f.ownerId,selectedId:'a'.repeat(32),env:f.env}),/selected_connection_unavailable/);
 assert.equal(f.steps.length,0);
});
test('rejects unknown selection before any revocations',async()=>{
 const f=fixture();
 await assert.rejects(()=>pruneOtherOwnerConnections({owner:f.owner,ownerId:f.ownerId,selectedId:'9'.repeat(32),env:f.env}),/selected_connection_unavailable/);
 assert.equal(f.steps.length,0);
});
test('does not mark cleanup complete if remote revocation fails',async()=>{
 const f=fixture();
 f.env.REGISTRIES.getByName=()=>({async revokeConnection(){throw Error('relay outage')}});
 await assert.rejects(()=>pruneOtherOwnerConnections({owner:f.owner,ownerId:f.ownerId,selectedId:'a'.repeat(32),env:f.env}),/relay outage/);
 assert.equal(f.steps.some(x=>x[0]==='cleanup'),false);
});
test('validates selected identifier to prevent cross-account lookup',async()=>{
 const f=fixture();
 await assert.rejects(()=>pruneOtherOwnerConnections({owner:f.owner,ownerId:f.ownerId,selectedId:'../../not-a-uuid',env:f.env}),/invalid_selected_connection/);
 assert.equal(f.steps.length,0);
});
test('never revokes the device the kept connection still uses',async()=>{
 const f=fixture();f.entries[1].deviceDigest=f.entries[0].deviceDigest;
 await pruneOtherOwnerConnections({owner:f.owner,ownerId:f.ownerId,selectedId:'a'.repeat(32),env:f.env});
 const devices=f.steps.filter(x=>x[0]==='device').map(x=>x[1]);
 assert.deepEqual(devices,['c'.repeat(64)]);
 assert.equal(f.steps.filter(x=>x[0]==='revoke').length,2);
});

import { evictDurableObject } from 'cloudflare:test';
import { env } from 'cloudflare:workers';
import { afterEach, describe, expect, test } from 'vitest';
import type { RelayDO } from '../../src/relay-do.ts';
import type { RelayFrame } from '../../src/relay-protocol.ts';
import { decodeRelayFrame, encodeRelayFrame } from '../../src/relay-protocol.ts';
const sockets: WebSocket[] = [];
afterEach(() => { for (const ws of sockets.splice(0)) { try { ws.close(); } catch {} } });
async function setup() {
  const token=crypto.randomUUID().replaceAll('-','')+crypto.randomUUID().replaceAll('-','');
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token))),x=>x.toString(16).padStart(2,'0')).join('');
  const stub=env.RELAYS.getByName(crypto.randomUUID());
  const binding={ownerId:'owner-a',connectionId:'connection-a',installationId:'installation-a',profileId:'profile-a',deviceDigest:digest,deviceExpiresAt:Date.now()+60000};
  await stub.configureBinding(binding);return {stub,token,binding};
}
function nextMessage(ws:WebSocket):Promise<string> { return new Promise(resolve=>ws.addEventListener('message',e=>resolve(String(e.data)),{once:true})); }
async function connect(stub:DurableObjectStub<RelayDO>,token:string) {
  const response=await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}));
  expect(response.status).toBe(101);const ws=response.webSocket!;sockets.push(ws);const ready=nextMessage(ws);ws.accept();const value:unknown=JSON.parse(await ready);if(!value||typeof value!=='object'||!('generation' in value)||typeof value.generation!=='number')throw new Error('invalid ready fixture');return {ws,generation:value.generation};
}
function frame(generation:number,id:string=crypto.randomUUID(),timeout=2000) { return {v:1 as const,kind:'request' as const,id,generation,deadlineAt:Date.now()+timeout,headers:[['Content-Type','application/json']] as const,bodyBase64:btoa('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')}; }
type RequestFrame=Extract<RelayFrame,{kind:'request'}>;
function decodeRequest(text:string):RequestFrame {const d=decodeRelayFrame(text);if(!d.ok||d.frame.kind!=='request')throw new Error('expected request fixture');return d.frame;}
function respond(ws:WebSocket,request:RequestFrame,body='synthetic-result',generation=request.generation) { const result=encodeRelayFrame({v:1,kind:'response',id:request.id,generation,status:200,headers:[['Content-Type','application/json']],bodyBase64:btoa(body)});if(!result.ok)throw new Error('bad fixture frame');ws.send(result.text); }
describe('real workerd relay runtime',()=>{
 test('requires bound device authentication before accepting socket',async()=>{const {stub}=await setup();const r=await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+'x'.repeat(64)}}));expect(r.status).toBe(401);});
 test('offline origin is unavailable rather than successful empty recall',async()=>{const {stub}=await setup();expect((await stub.forward(frame(1))).status).toBe(503);});
 test('dispatches exact bytes over hibernatable socket and returns correlated result',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const message=nextMessage(ws);const result=stub.forward(frame(generation));const decoded=decodeRelayFrame(await message);expect(decoded.ok).toBe(true);if(!decoded.ok||decoded.frame.kind!=='request')throw new Error('expected request fixture');respond(ws,decoded.frame);const r=await result;expect(r.status).toBe(200);expect(await r.text()).toBe('synthetic-result');});
 test('survives actual eviction while retaining socket and binding',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);await evictDurableObject(stub);const message=nextMessage(ws);const result=stub.forward(frame(generation));respond(ws,decodeRequest(await message),'after-eviction');expect(await (await result).text()).toBe('after-eviction');});
 test('new connector generation prevents stale request dispatch',async()=>{const {stub,token}=await setup();const first=await connect(stub,token);const second=await connect(stub,token);expect(second.generation).toBeGreaterThan(first.generation);expect((await stub.forward(frame(first.generation))).status).toBe(409);});
 test('revocation blocks new admission and reconnect',async()=>{const {stub,token}=await setup();const {generation}=await connect(stub,token);await stub.revoke();expect((await stub.forward(frame(generation))).status).toBe(403);expect((await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}))).status).toBe(403);});
 test('timeout sends cancellation and returns unavailable status',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const request=nextMessage(ws);const result=stub.forward(frame(generation,undefined,100));await request;const cancel=nextMessage(ws);expect((await result).status).toBe(504);const d=decodeRelayFrame(await cancel);expect(d.ok&&d.frame.kind).toBe('cancel');});
 test('binding owner/profile cannot be overwritten',async()=>{const {stub,binding}=await setup();const operation=(async()=>{await stub.configureBinding({...binding,ownerId:'foreign'});})();await expect(operation).rejects.toThrow('binding_conflict');});
 test('unknown response cannot settle another request',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const message=nextMessage(ws);const result=stub.forward(frame(generation));const req=decodeRequest(await message);respond(ws,{...req,id:'foreign-request'},'wrong');respond(ws,req,'correct');expect(await (await result).text()).toBe('correct');});
});

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

test('expired device credentials rejected',async()=>{const {stub,token,binding}=await setup();await stub.configureBinding({...binding,deviceExpiresAt:Date.now()-1});expect((await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}))).status).toBe(401);});
test('duplicate in-flight caller ID rejected',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const message=nextMessage(ws);const f=frame(generation);const first=stub.forward(f);const req=decodeRequest(await message);expect((await stub.forward(f)).status).toBe(409);respond(ws,req);expect((await first).status).toBe(200);});
test('wrong response generation cannot settle a request',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const message=nextMessage(ws);const result=stub.forward(frame(generation));const req=decodeRequest(await message);respond(ws,req,'wrong',generation+1);respond(ws,req,'correct');expect(await (await result).text()).toBe('correct');});
test('disconnect during in-flight request produces honest unavailable result',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const message=nextMessage(ws);const result=stub.forward(frame(generation));await message;await stub.revoke();expect((await result).status).toBe(403);});
test('credential rotation retires accepted old socket before acknowledgement',async()=>{const {stub,token,binding}=await setup();const {generation}=await connect(stub,token);await stub.configureBinding({...binding,deviceDigest:'0'.repeat(64)});expect((await stub.forward(frame(generation))).status).toBe(503);expect((await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}))).status).toBe(401);});
test('empty notification response preserves HTTP204 semantics',async()=>{const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const message=nextMessage(ws);const result=stub.forward(frame(generation));const req=decodeRequest(await message);const f=encodeRelayFrame({v:1,kind:'response',id:req.id,generation,status:204,headers:[],bodyBase64:''});if(!f.ok)throw new Error('bad fixture');ws.send(f.text);const r=await result;expect(r.status).toBe(204);expect(await r.text()).toBe('');});
test('synthetic MCP remember and recall traverse the complete relay socket path',async()=>{
 const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);const facts:string[]=[];
 ws.addEventListener('message',e=>{
  const d=decodeRelayFrame(String(e.data));if(!d.ok||d.frame.kind!=='request')return;
  const msg:unknown=JSON.parse(atob(d.frame.bodyBase64));
  if(!msg||typeof msg!=='object'||!('params' in msg)||!msg.params||typeof msg.params!=='object'||!('name' in msg.params)||!('id' in msg))throw new Error('invalid MCP fixture');
  let result:unknown;
  if(msg.params.name==='remember'&&'arguments' in msg.params&&msg.params.arguments&&typeof msg.params.arguments==='object'&&'content' in msg.params.arguments&&typeof msg.params.arguments.content==='string'){facts.push(msg.params.arguments.content);result={success:true,fact_ids:['synthetic-fact']};}
  else result={results:[...facts],count:facts.length};
  respond(ws,d.frame,JSON.stringify({jsonrpc:'2.0',id:msg.id,result}));
 });
 const first={...frame(generation),bodyBase64:btoa(JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'remember',arguments:{content:'synthetic local decision',kind:'decision'}}}))};
 expect(await (await stub.forward(first)).json()).toEqual({jsonrpc:'2.0',id:1,result:{success:true,fact_ids:['synthetic-fact']}});
 const second={...frame(generation),bodyBase64:btoa(JSON.stringify({jsonrpc:'2.0',id:2,method:'tools/call',params:{name:'recall',arguments:{query:'decision'}}}))};
 expect(await (await stub.forward(second)).json()).toEqual({jsonrpc:'2.0',id:2,result:{results:['synthetic local decision'],count:1}});
});

test('rejected binding edit does not terminate a healthy relay instance',async()=>{const {stub,token,binding}=await setup();const {ws,generation}=await connect(stub,token);await expect((async()=>{await stub.configureBinding({...binding,profileId:'foreign'});})()).rejects.toThrow('binding_conflict');const message=nextMessage(ws);const result=stub.forward(frame(generation));respond(ws,decodeRequest(await message),'still-connected');expect(await (await result).text()).toBe('still-connected');});

test('revocation before binding survives eviction as a terminal state',async()=>{const stub=env.RELAYS.getByName(crypto.randomUUID());await stub.revoke();await evictDurableObject(stub);expect((await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket'}}))).status).toBe(403);expect((await stub.forward(frame(1))).status).toBe(403);});

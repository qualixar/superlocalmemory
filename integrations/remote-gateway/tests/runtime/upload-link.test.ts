import {env} from 'cloudflare:workers';
import {expect,test} from 'vitest';
import {createExecutionContext as context,waitOnExecutionContext} from 'cloudflare:test';
import {resourceGateway} from '../../src/worker-resource.ts';
import type {ResourceEnv} from '../../src/worker-resource.ts';
import {decodeRelayFrame,encodeRelayFrame} from '../../src/relay-protocol.ts';

const TOKEN='U'.repeat(43);
type Seen={op:string;index:number;total:number;bytes:number};

/** The owner's registry for this connection: live access and an app consented to make upload links. */
async function grantUploads(id:string){
 const registry=env.REGISTRIES.getByName(id);
 await registry.configure({connectionId:id,ownerId:'owner-a',installationId:'installation-a',profileId:'profile-a',origin:{kind:'relay',installationId:'installation-a',profileId:'profile-a'},allowedTools:['recall'],allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null});
 await registry.addAuthorization({authorizationId:'app-a',audience:'https://mcp.superlocalmemory.com/mcp',ownerId:'owner-a',clientId:'client-a',connectionId:id,consentedTools:['recall','media_upload_link'],consentedScopes:['slm:read','slm:write','slm:media'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 await registry.setEntitlement('owner-a',Date.now()+86400000,0);
}
/** A real RelayDO with a real connector socket standing in for the laptop. */
async function laptop(script:(seen:Seen)=>Record<string,unknown>|Response,{consent=true}:{consent?:boolean}={}){
 const id=crypto.randomUUID().replaceAll('-','');const token='synthetic-device-token-'.repeat(3);
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token))),b=>b.toString(16).padStart(2,'0')).join('');
 const relay=env.RELAYS.getByName(id);
 await relay.configureBinding({ownerId:'owner-a',connectionId:id,installationId:'installation-a',profileId:'profile-a',deviceDigest:digest,deviceExpiresAt:Date.now()+60000});
 const upgrade=await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token,'x-slm-connector-features':'grant-v1,upload-v1'}}));
 const socket=upgrade.webSocket!;const ready=new Promise<string>(resolve=>socket.addEventListener('message',e=>resolve(String(e.data)),{once:true}));socket.accept();await ready;
 const seen:Seen[]=[];
 if(consent)await grantUploads(id);
 socket.addEventListener('message',event=>{
  const decoded=decodeRelayFrame(String(event.data));if(!decoded.ok||decoded.frame.kind!=='request')return;const frame=decoded.frame;
  const header=frame.headers.find(h=>h[0]==='x-slm-upload')?.[1]??'';const [op,,index,total]=header.split(' ');
  const bytes=atob(frame.bodyBase64).length;const entry={op:op!,index:Number(index),total:Number(total),bytes};seen.push(entry);
  const out=script(entry);const status=out instanceof Response?out.status:200;
  const body=out instanceof Response?'{"error":"connector_offline"}':JSON.stringify(out);
  const reply=encodeRelayFrame({v:1,kind:'response',id:frame.id,generation:frame.generation,status,headers:[['content-type','application/json']],bodyBase64:btoa(body)});if(reply.ok)socket.send(reply.text);
 });
 return {id,seen,close:()=>socket.close()};
}
const fixtureEnv=()=>({...env,AUTH_SERVER:{async validateToken(){return null;}}} as ResourceEnv);
async function fetchUpload(id:string,method:string,body?:Uint8Array,headers:Record<string,string>={}){
 const ctx=context();
 const response=await resourceGateway.fetch(new Request(`https://mcp.superlocalmemory.com/u/${id}/${TOKEN}`,{method,headers:{...(body?{'content-length':String(body.length),origin:'https://mcp.superlocalmemory.com'}:{}),...headers},body}),fixtureEnv(),ctx);
 await waitOnExecutionContext(ctx);return response;
}
const okScript=(seen:Seen)=>seen.op==='info'?{ok:true,kind:'image',max_bytes:25*1024*1024,expires_at:1}:seen.op==='chunk'?{ok:true,received:seen.bytes}:{ok:true,done:true,message:'Saved to your memory.'};

test('GET serves the picker through a real relay and socket, with the strict policy',async()=>{
 const {id,seen,close}=await laptop(okScript);
 try{const r=await fetchUpload(id,'GET');expect(r.status).toBe(200);expect(r.headers.get('content-security-policy')).toContain("default-src 'none'");
  expect(r.headers.get('referrer-policy')).toBe('no-referrer');expect(await r.text()).toContain('up to 25 MB');expect(seen.map(s=>s.op)).toEqual(['info']);
 }finally{close();}
});

test('POST relays a 1.5 MB file as 700000-byte chunks and returns the laptop result',async()=>{
 const {id,seen,close}=await laptop(okScript);
 try{const bytes=new Uint8Array(1_500_001).fill(7);const r=await fetchUpload(id,'POST',bytes);
  expect(await r.json()).toEqual({ok:true,done:true,message:'Saved to your memory.'});
  expect(seen.map(s=>[s.op,s.bytes])).toEqual([['info',0],['chunk',700000],['chunk',700000],['chunk',100001],['finish',0]]);
 }finally{close();}
});

test('an offline laptop gets the plain asleep page, not a raw code',async()=>{
 const id=crypto.randomUUID().replaceAll('-','');await grantUploads(id);
 const r=await fetchUpload(id,'GET');expect(r.status).toBe(503);const text=await r.text();expect(text).toMatch(/computer/i);expect(text).not.toContain('connector_');
 const post=await fetchUpload(id,'POST',new Uint8Array(10));expect(post.status).toBe(503);expect(((await post.json()) as {ok:boolean}).ok).toBe(false);
});

test('a refusal from the laptop reaches the page as plain text',async()=>{
 const {id,close}=await laptop(()=>({ok:false,code:'used',message:'This upload link has already been used. Ask the app for a new one.'}));
 try{const r=await fetchUpload(id,'GET');expect(r.status).toBe(410);expect(await r.text()).toContain('already been used');}finally{close();}
});

test('a public MCP call cannot carry the upload marker to the laptop',async()=>{
 const {seen,close}=await laptop(okScript);
 try{const ctx=context();const r=await resourceGateway.fetch(new Request('https://mcp.superlocalmemory.com/mcp',{method:'POST',headers:{'content-type':'application/json',authorization:'Bearer synthetic','x-slm-upload':`chunk ${TOKEN} 0 5`},body:'{}'}),fixtureEnv(),ctx);
  await waitOnExecutionContext(ctx);expect(r.status).toBe(401);expect(seen).toEqual([]);}finally{close();}
});

test('consent taken away on the gateway stops the link before the laptop is asked',async()=>{
 const {id,seen,close}=await laptop(okScript,{consent:false});
 try{const page=await fetchUpload(id,'GET');expect(page.status).toBe(410);expect(await page.text()).toMatch(/no longer works/i);
  const post=await fetchUpload(id,'POST',new Uint8Array(10));expect(post.status).toBe(410);expect(seen).toEqual([]);
  await grantUploads(id);expect((await fetchUpload(id,'GET')).status).toBe(200);
  await env.REGISTRIES.getByName(id).revokeAuthorization('owner-a','app-a',1);expect((await fetchUpload(id,'GET')).status).toBe(410);
 }finally{close();}
});

test('a laptop connector that cannot take uploads is told apart from an offline one',async()=>{
 const id=crypto.randomUUID().replaceAll('-','');const token='synthetic-device-token-'.repeat(3);
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token))),b=>b.toString(16).padStart(2,'0')).join('');
 const relay=env.RELAYS.getByName(id);await relay.configureBinding({ownerId:'owner-a',connectionId:id,installationId:'installation-a',profileId:'profile-a',deviceDigest:digest,deviceExpiresAt:Date.now()+60000});
 const upgrade=await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token,'x-slm-connector-features':'grant-v1'}}));
 const socket=upgrade.webSocket!;const ready=new Promise<string>(resolve=>socket.addEventListener('message',e=>resolve(String(e.data)),{once:true}));socket.accept();await ready;
 try{await grantUploads(id);const page=await fetchUpload(id,'GET');expect(page.status).toBe(503);expect(await page.text()).toMatch(/needs an update/i);}finally{socket.close();}
});

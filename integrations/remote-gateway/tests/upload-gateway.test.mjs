import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeRelayFrame,MAX_REQUEST_BYTES} from '../src/relay-protocol.ts';
let api;try{api=await import('../src/upload-gateway.ts');}catch{api=null;}
const CONN='a'.repeat(32);const TOKEN='B'.repeat(43);const URL_=`https://mcp.superlocalmemory.com/u/${CONN}/${TOKEN}`;
const MAX=25*1024*1024;

function laptop(script,{consent=()=>true}={}){
  const frames=[];
  const relay={async forwardCurrent(frame,options,context){
    frames.push(frame);
    const header=frame.headers.find(h=>h[0]==='x-slm-upload')[1];
    const [op,,index,total,nonce]=header.split(' ');
    const out=await script({op,index:Number(index),total:Number(total),nonce,frame,body:Buffer.from(frame.bodyBase64,'base64')});
    if(out instanceof Response)return out;
    return Response.json(out,{status:200});
  }};
  const names=[],limits=[],checks=[],asked=[];
  const env={RELAYS:{getByName(name){names.push(name);return relay;}},
    REGISTRIES:{getByName(name){return {async uploadsAllowed(aid){checks.push(name);asked.push(aid);const v=consent(aid);if(v instanceof Error)throw v;return v;}};}},
    UPLOAD_IP:{async limit(){return {success:true};}},UPLOAD_CONN:{async limit({key}){limits.push(key);return {success:true};}}};
  return {env,frames,names,limits,checks,asked};
}
const live=(extra={})=>laptop(({op,body})=>{
  if(op==='info')return {ok:true,kind:'image',max_bytes:MAX,expires_at:1};
  if(op==='chunk')return {ok:true,received:body.length};
  return {ok:true,done:true,message:'Saved to your memory.'};
});
const req=(method='GET',init={})=>new Request(URL_,{method,...init});
const post=(bytes,headers={})=>req('POST',{body:bytes,headers:{'content-length':String(bytes.length),'content-type':'application/octet-stream',origin:'https://mcp.superlocalmemory.com',...headers}});
const png=n=>{const b=new Uint8Array(n).map((_,i)=>i%251);b.set([0x89,0x50,0x4e,0x47].slice(0,n),0);return b;};

test('paths outside /u/ are not ours',async()=>{
  assert.ok(api);const {env}=live();
  assert.equal(await api.handleUpload(new Request('https://mcp.superlocalmemory.com/mcp',{method:'POST'}),env),null);
  assert.equal(await api.handleUpload(new Request('https://mcp.superlocalmemory.com/users'),env),null);
});

test('a malformed link is a plain 404 and reaches no laptop',async()=>{
  assert.ok(api);const {env,frames,names}=live();
  for(const path of ['/u/','/u/x','/u/'+CONN+'/short','/u/'+CONN.toUpperCase()+'/'+TOKEN,'/u/'+CONN+'/'+TOKEN+'/more']){
    const r=await api.handleUpload(new Request('https://mcp.superlocalmemory.com'+path),env);
    assert.equal(r.status,404,path);assert.match(await r.text(),/not valid/i);
  }
  assert.equal(frames.length,0);assert.equal(names.length,0);
});

test('only GET and POST are served, and no CORS is ever offered',async()=>{
  assert.ok(api);const {env}=live();
  for(const method of ['PUT','DELETE','PATCH','OPTIONS','HEAD']){
    const r=await api.handleUpload(req(method),env);
    assert.equal(r.status,405,method);assert.equal(r.headers.get('access-control-allow-origin'),null);assert.equal(r.headers.get('allow'),'GET, POST');
  }
});

test('GET asks the laptop about the link, then serves the picker for that kind',async()=>{
  assert.ok(api);const {env,frames,names}=live();
  const r=await api.handleUpload(req(),env);
  assert.equal(r.status,200);assert.match(r.headers.get('content-type'),/text\/html/);
  assert.match(await r.text(),/up to 25 MB/);
  assert.deepEqual(names,[CONN]);
  assert.equal(frames.length,1);
  const f=frames[0];
  assert.equal(f.kind,'request');assert.equal(f.bodyBase64,'');
  assert.match(f.headers[1][1],new RegExp(`^info ${TOKEN} 0 0 [A-Za-z0-9_-]{22}$`));assert.equal(f.headers[0][0],'content-type');
  assert.ok(f.deadlineAt-Date.now()<=25000&&f.deadlineAt>Date.now());
  assert.equal(encodeRelayFrame({...f,generation:1}).ok,true);
});

test('GET reports an offline or asleep computer in plain words',async()=>{
  assert.ok(api);
  for(const [code,status] of [['connector_offline',503],['connector_asleep',503],['relay_timeout',504]]){
    const {env}=laptop(()=>Response.json({error:code},{status}));
    const r=await api.handleUpload(req(),env);
    assert.equal(r.status,status);const text=await r.text();
    assert.doesNotMatch(text,new RegExp(code));assert.match(text,/computer/i);
  }
});

test('GET shows the laptop\'s refusal for a dead link',async()=>{
  assert.ok(api);
  for(const [code,status] of [['invalid_link',404],['expired',410],['used',410],['not_allowed',403]]){
    const {env}=laptop(()=>({ok:false,code,message:`Link says <b>${code}</b>`}));
    const r=await api.handleUpload(req(),env);
    assert.equal(r.status,status);const text=await r.text();
    assert.match(text,new RegExp('&lt;b&gt;'+code));assert.doesNotMatch(text,/<b>/);
  }
});

test('POST relays a file as ordered chunks of at most 700000 bytes, then finishes',async()=>{
  assert.ok(api);const {env,frames}=live();
  const bytes=png(1_500_001);
  const r=await api.handleUpload(post(bytes),env);
  assert.equal(r.status,200);assert.deepEqual(await r.json(),{ok:true,done:true,message:'Saved to your memory.'});
  const ops=frames.map(f=>f.headers[1][1].split(' '));
  assert.deepEqual(ops.map(o=>o[0]),['info','chunk','chunk','chunk','finish']);
  assert.deepEqual(ops.filter(o=>o[0]==='chunk').map(o=>[Number(o[2]),Number(o[3])]),[[0,1500001],[1,1500001],[2,1500001]]);
  assert.deepEqual(ops[4].slice(2,4),['0','1500001']);
  const sent=Buffer.concat(frames.filter(f=>f.headers[1][1].startsWith('chunk')).map(f=>Buffer.from(f.bodyBase64,'base64')));
  assert.deepEqual(new Uint8Array(sent),bytes);
  for(const f of frames){assert.ok(Buffer.from(f.bodyBase64,'base64').length<=700000);assert.ok(Buffer.from(f.bodyBase64,'base64').length<=MAX_REQUEST_BYTES);assert.equal(encodeRelayFrame({...f,generation:1}).ok,true);assert.equal(f.headers.length,2);}
  assert.equal(new Set(frames.map(f=>f.id)).size,frames.length);
});

test('exactly one chunk when the file is small, and exactly the boundary size',async()=>{
  assert.ok(api);
  for(const [size,chunks] of [[1,1],[700000,1],[700001,2],[1400000,2]]){
    const {env,frames}=live();
    const r=await api.handleUpload(post(png(size)),env);
    assert.equal(r.status,200,String(size));
    assert.equal(frames.filter(f=>f.headers[1][1].startsWith('chunk')).length,chunks,String(size));
  }
});

test('POST needs a declared length, a body, and one within the laptop\'s limit',async()=>{
  assert.ok(api);
  const {env,frames}=live();
  const noLength=new Request(URL_,{method:'POST',body:png(10),headers:{origin:'https://mcp.superlocalmemory.com'}});
  noLength.headers.delete('content-length');
  assert.equal((await api.handleUpload(noLength,env)).status,411);
  assert.equal((await api.handleUpload(post(new Uint8Array(0)),env)).status,400);
  for(const bad of ['abc','-5','1.5','1e9','']){
    const r=new Request(URL_,{method:'POST',body:png(10),headers:{'content-length':bad,origin:'https://mcp.superlocalmemory.com'}});
    assert.ok([400,411].includes((await api.handleUpload(r,env)).status),bad);
  }
  const huge=post(png(10),{'content-length':String(2**31)});
  assert.equal((await api.handleUpload(huge,env)).status,413);
  assert.equal(frames.length,0);
  const tooBig=laptop(({op})=>op==='info'?{ok:true,kind:'image',max_bytes:100,expires_at:1}:{ok:true});
  const r=await api.handleUpload(post(png(101)),tooBig.env);
  assert.equal(r.status,413);assert.match((await r.json()).message,/too large/i);
  assert.equal(tooBig.frames.length,1);
});

test('a body shorter or longer than it declared stops the upload before finish',async()=>{
  assert.ok(api);
  for(const [declared,actual] of [[1000,500],[1000,1500],[800000,799999]]){
    const {env,frames}=live();
    const r=await api.handleUpload(post(png(actual),{'content-length':String(declared)}),env);
    assert.equal(r.status,400,`${declared}/${actual}`);assert.equal((await r.json()).code,'size_mismatch');
    assert.ok(!frames.some(f=>f.headers[1][1].startsWith('finish')));
  }
});

test('POST refuses a foreign Origin before anything is relayed',async()=>{
  assert.ok(api);const {env,frames}=live();
  for(const origin of ['https://evil.example','null','http://mcp.superlocalmemory.com','https://mcp.superlocalmemory.com.evil.example']){
    const r=await api.handleUpload(post(png(10),{origin}),env);
    assert.equal(r.status,403,origin);
  }
  assert.equal(frames.length,0);
  assert.equal((await api.handleUpload(post(png(10),{origin:'https://mcp.superlocalmemory.com'}),env)).status,200);
});

test('a chunk the laptop refuses ends the upload with its plain reason and sends nothing more',async()=>{
  assert.ok(api);
  const {env,frames}=laptop(({op,index})=>op==='info'?{ok:true,kind:'image',max_bytes:MAX,expires_at:1}:op==='chunk'&&index===1?{ok:false,code:'wrong_type',message:'That is not the kind of file this link is for.'}:{ok:true,received:1});
  const r=await api.handleUpload(post(png(2_000_000)),env);
  assert.equal(r.status,400);assert.deepEqual(await r.json(),{ok:false,code:'wrong_type',message:'That is not the kind of file this link is for.'});
  assert.deepEqual(frames.map(f=>f.headers[1][1].split(' ')[0]),['info','chunk','chunk']);
});

test('a laptop that stops answering mid-upload gives a plain message',async()=>{
  assert.ok(api);
  const {env}=laptop(({op,index})=>op==='info'?{ok:true,kind:'image',max_bytes:MAX,expires_at:1}:index===1?Response.json({error:'connector_closed'},{status:503}):{ok:true,received:1});
  const r=await api.handleUpload(post(png(2_000_000)),env);
  const body=await r.json();assert.equal(r.status,503);assert.equal(body.ok,false);assert.doesNotMatch(body.message,/connector_closed/);
});

test('finish is asked again while the laptop is still saving, then gives up politely',async()=>{
  assert.ok(api);
  let finishes=0;
  const soon=laptop(({op})=>{if(op==='info')return {ok:true,kind:'image',max_bytes:MAX,expires_at:1};if(op==='chunk')return {ok:true,received:1};finishes++;return finishes<3?{ok:true,done:false}:{ok:true,done:true,message:'Saved to your memory.'};});
  assert.deepEqual(await (await api.handleUpload(post(png(100)),soon.env)).json(),{ok:true,done:true,message:'Saved to your memory.'});
  assert.equal(finishes,3);
  let never=0;
  const slow=laptop(({op})=>{if(op==='info')return {ok:true,kind:'image',max_bytes:MAX,expires_at:1};if(op==='chunk')return {ok:true,received:1};never++;return {ok:true,done:false};});
  const r=await api.handleUpload(post(png(100)),slow.env);
  assert.equal(r.status,202);const body=await r.json();assert.equal(body.ok,true);assert.equal(body.done,false);assert.match(body.message,/still/i);
  assert.equal(never,api.FINISH_TRIES);
});

test('both rate limits apply per address and per connection; a missing limiter fails closed',async()=>{
  assert.ok(api);
  const keys=[];
  const base=live();
  const env={...base.env,UPLOAD_IP:{async limit({key}){keys.push(['ip',key]);return {success:false};}}};
  const r=await api.handleUpload(req('GET',{headers:{'cf-connecting-ip':'203.0.113.9'}}),env);
  assert.equal(r.status,429);assert.ok(Number(r.headers.get('retry-after'))>=1);
  assert.match(keys[0][1],/^[a-f0-9]{64}$/);assert.doesNotMatch(keys[0][1],/203/);
  assert.equal(base.frames.length,0);
  const conn={...base.env,UPLOAD_CONN:{async limit({key}){keys.push(['conn',key]);return {success:false};}}};
  assert.equal((await api.handleUpload(req(),conn)).status,429);
  assert.equal(keys.at(-1)[0],'conn');assert.match(keys.at(-1)[1],new RegExp(`^${CONN}:[a-f0-9]{64}$`));
  for(const missing of [{UPLOAD_IP:undefined},{UPLOAD_CONN:undefined},{UPLOAD_IP:{async limit(){throw new Error('down');}}}]){
    assert.equal((await api.handleUpload(req(),{...base.env,...missing})).status,503);
  }
  assert.equal(base.frames.length,0);
});

test('nothing from the laptop or the link reaches a response unescaped, and nothing is cached',async()=>{
  assert.ok(api);
  const {env}=laptop(()=>({ok:false,code:'<x>',message:'<script>alert(1)</script>'}));
  const page=await api.handleUpload(req(),env);
  assert.doesNotMatch(await page.text(),/<script>alert/);
  assert.equal(page.headers.get('cache-control'),'no-store');
  const up=await api.handleUpload(post(png(10)),env);
  assert.equal(up.headers.get('cache-control'),'no-store');assert.equal((await up.json()).code,'error');
});


test('every frame of one upload carries the same fresh nonce; the next upload gets another',async()=>{
  assert.ok(api);
  const nonces=[];
  const first=laptop(({op,nonce})=>{if(op!=='info')nonces.push(nonce);return okReply(op);});
  await api.handleUpload(post(png(1_500_001)),first.env);
  assert.equal(nonces.length,4);assert.equal(new Set(nonces).size,1);assert.match(nonces[0],/^[A-Za-z0-9_-]{22}$/);
  const second=laptop(({op,nonce})=>{if(op!=='info')nonces.push(nonce);return okReply(op);});
  await api.handleUpload(post(png(10)),second.env);
  assert.notEqual(nonces.at(-1),nonces[0]);
});
function okReply(op){return op==='info'?{ok:true,kind:'image',max_bytes:MAX,expires_at:1}:op==='chunk'?{ok:true,received:1}:{ok:true,done:true,message:'Saved to your memory.'};}

test('the per-connection limit counts under the connection and a token hash, never the bare connection or the token',async()=>{
  assert.ok(api);
  const l=live();
  await api.handleUpload(req(),l.env);
  assert.equal(l.limits.length,1);
  assert.match(l.limits[0],new RegExp(`^${CONN}:[a-f0-9]{64}$`));assert.ok(!l.limits[0].includes(TOKEN));
});

test('a link whose consent was taken away is refused before any frame is sent',async()=>{
  assert.ok(api);
  for(const consent of [()=>false,()=>new Error('registry down')]){
    const l=laptop(okReply2,{consent});
    const page=await api.handleUpload(req(),l.env);
    assert.equal(page.status,410);assert.match(await page.text(),/no longer works/i);
    const up=await api.handleUpload(post(png(10)),l.env);
    assert.equal(up.status,410);const body=await up.json();assert.equal(body.ok,false);assert.match(body.message,/no longer works/i);
    assert.equal(l.frames.length,0);
    assert.deepEqual(l.checks,[CONN,CONN]);
  }
});
function okReply2({op}){return okReply(op);}

test('consent is checked again before the file is finished: revoking mid-upload stops the save',async()=>{
  assert.ok(api);
  let allowed=true;
  const l=laptop(({op})=>{if(op==='chunk')allowed=false;return okReply(op);},{consent:()=>allowed});
  const r=await api.handleUpload(post(png(100)),l.env);
  assert.equal(r.status,410);assert.match((await r.json()).message,/no longer works/i);
  assert.ok(!l.frames.some(f=>f.headers[1][1].startsWith('finish')));
});

test('a laptop whose connector cannot take uploads gets a plain "update" message',async()=>{
  assert.ok(api);
  const l=laptop(()=>Response.json({error:'upload_unsupported'},{status:503}));
  const page=await api.handleUpload(req(),l.env);
  assert.equal(page.status,503);assert.match(await page.text(),/update/i);
});

// -- an upload link belongs to the app that asked for it (audit F6) ---------------------------------------------

const bound=(aid,extra={})=>({op})=>op==='info'?{ok:true,kind:'image',max_bytes:MAX,expires_at:1,authorization_id:aid}:okReply(op);

test('the picker checks the consent of the app that issued the link, not any app',async()=>{
  assert.ok(api);
  const l=laptop(bound('app-a'),{consent:aid=>aid!=='app-a'});   // app A revoked, app B still consented
  const r=await api.handleUpload(req(),l.env);
  assert.equal(r.status,410);assert.match(await r.text(),/no longer works/i);
  assert.ok(l.asked.includes('app-a'));
});

test('an upload on a revoked app\'s link is refused before any chunk is sent',async()=>{
  assert.ok(api);
  const l=laptop(bound('app-a'),{consent:aid=>aid!=='app-a'});
  const r=await api.handleUpload(post(png(100)),l.env);
  assert.equal(r.status,410);assert.match((await r.json()).message,/no longer works/i);
  assert.ok(!l.frames.some(f=>/^(chunk|finish)/.test(f.headers[1][1])));
});

test('a link of a still-consented app works end to end and is checked for that app at the finish',async()=>{
  assert.ok(api);
  const l=laptop(({op,body})=>op==='info'?{ok:true,kind:'image',max_bytes:MAX,expires_at:1,authorization_id:'app-b'}:okReply(op),{consent:()=>true});
  const r=await api.handleUpload(post(png(100)),l.env);
  assert.equal(r.status,200);assert.equal((await r.json()).done,true);
  assert.deepEqual(l.asked,[undefined,'app-b','app-b']);
});

test('revoking only the issuing app mid-upload stops its save while another app stays consented',async()=>{
  assert.ok(api);
  let appARevoked=false;
  const l=laptop(({op})=>{if(op==='chunk')appARevoked=true;return bound('app-a')({op});},{consent:aid=>!(appARevoked&&aid==='app-a')});
  const r=await api.handleUpload(post(png(100)),l.env);
  assert.equal(r.status,410);
  assert.ok(!l.frames.some(f=>f.headers[1][1].startsWith('finish')));
});

test('a laptop that names no app gets the old any-app check (older SuperLocalMemory)',async()=>{
  assert.ok(api);
  const l=live();
  await api.handleUpload(post(png(100)),l.env);
  assert.ok(l.asked.length>=2&&l.asked.every(a=>a===undefined));
});

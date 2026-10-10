import test from 'node:test';
import assert from 'node:assert/strict';
let api;try{api=await import('../src/upload-protocol.ts');}catch{api=null;}
const TOKEN='A'.repeat(43);const CONN='a'.repeat(32);const NONCE='n'.repeat(22);

test('the upload path is exactly /u/<32 hex>/<43 url-safe characters>',()=>{
  assert.ok(api);
  assert.deepEqual(api.parseUploadPath(`/u/${CONN}/${TOKEN}`),{connection:CONN,token:TOKEN});
  for(const bad of ['/u/','/u/'+CONN,`/u/${CONN}/${TOKEN}/`,`/u/${CONN}/${TOKEN}x`,`/u/${CONN.toUpperCase()}/${TOKEN}`,`/u/${CONN}/${'A'.repeat(42)}`,`/u/${'a'.repeat(31)}/${TOKEN}`,`/u/${CONN}/${TOKEN.slice(1)}.`,`/x/${CONN}/${TOKEN}`,`/u/${CONN}/../${TOKEN}`,`/u/${CONN}/%41${'A'.repeat(41)}`])assert.equal(api.parseUploadPath(bad),null,bad);
});

test('the frame header is "<op> <token> <index> <total>" and refuses anything else',()=>{
  assert.ok(api);
  assert.equal(api.uploadHeader('chunk',TOKEN,3,2500000,NONCE),`chunk ${TOKEN} 3 2500000 ${NONCE}`);
  assert.equal(api.uploadHeader('info',TOKEN,0,0,NONCE),`info ${TOKEN} 0 0 ${NONCE}`);
  assert.throws(()=>api.uploadHeader('other',TOKEN,0,0,NONCE));
  assert.throws(()=>api.uploadHeader('chunk','short',0,0,NONCE));
  assert.throws(()=>api.uploadHeader('chunk',TOKEN,-1,0,NONCE));
  assert.throws(()=>api.uploadHeader('chunk',TOKEN,0.5,0,NONCE));
  assert.throws(()=>api.uploadHeader('chunk',TOKEN,0,2**40,NONCE));
  for(const bad of ['','short','n'.repeat(21),'n'.repeat(23),'n'.repeat(21)+'!','n'.repeat(21)+' ',undefined])assert.throws(()=>api.uploadHeader('chunk',TOKEN,0,5,bad),String(bad));
});

test('every upload gets its own unguessable nonce of 22 url-safe characters',()=>{
  assert.ok(api);
  const seen=new Set();
  for(let i=0;i<200;i++){const nonce=api.newNonce();assert.match(nonce,/^[A-Za-z0-9_-]{22}$/);seen.add(nonce);}
  assert.equal(seen.size,200);
});

test('the per-connection limit is keyed on the connection and a hash of the token, never the token',async()=>{
  assert.ok(api);
  const a=await api.limitKey(CONN,TOKEN),b=await api.limitKey(CONN,'B'.repeat(43)),c=await api.limitKey('c'.repeat(32),TOKEN);
  assert.match(a,/^[a-f0-9]{32}:[a-f0-9]{64}$/);
  assert.ok(a.startsWith(CONN+':'));assert.ok(!a.includes(TOKEN));
  assert.notEqual(a,b);assert.notEqual(a,c);assert.equal(a,await api.limitKey(CONN,TOKEN));
});

test('the chunk size stays below the relay request cap with room to spare',()=>{
  assert.ok(api);
  assert.equal(api.CHUNK_BYTES,700000);
  assert.ok(api.CHUNK_BYTES<=700*1000);
});

test('a laptop reply is rebuilt from known keys only, with plain bounded text',()=>{
  assert.ok(api);
  assert.deepEqual(api.cleanReply({ok:true,kind:'image',max_bytes:26214400,expires_at:5,extra:'x'}),{ok:true,kind:'image',maxBytes:26214400,received:undefined,done:undefined,message:undefined});
  assert.deepEqual(api.cleanReply({ok:true,done:true,message:'Saved to your memory.',evil:1}),{ok:true,kind:undefined,maxBytes:undefined,received:undefined,done:true,message:'Saved to your memory.'});
  const refused=api.cleanReply({ok:false,code:'expired',message:'x'.repeat(1000)});
  assert.equal(refused.ok,false);assert.equal(refused.code,'expired');assert.equal(refused.message.length,300);
  assert.equal(api.cleanReply({ok:false,code:'<script>',message:'m'}).code,'error');
  for(const junk of [null,3,'text',[],{},{ok:'yes'},{ok:true,kind:'video'},{ok:true,max_bytes:-1},{ok:true,done:'true'}]){
    const out=api.cleanReply(junk);
    assert.ok(out.ok===false||(out.kind===undefined||['image','document'].includes(out.kind)));
  }
  assert.equal(api.cleanReply(null).ok,false);
  assert.equal(api.cleanReply({ok:true,kind:'video'}).kind,undefined);
});

test('refusal codes map to honest HTTP statuses',()=>{
  assert.ok(api);
  const status=api.statusFor;
  assert.equal(status('invalid_link'),404);assert.equal(status('expired'),410);assert.equal(status('used'),410);
  assert.equal(status('not_allowed'),403);assert.equal(status('too_large'),413);assert.equal(status('daily_limit'),429);
  assert.equal(status('anything-else'),400);
});

test('relay failures become plain sentences and never raw codes',()=>{
  assert.ok(api);
  for(const code of ['connector_offline','connector_asleep','connector_unavailable','connector_closed','relay_timeout','relay_busy','connection_revoked','origin_unavailable','upload_unsupported','weird'])
    assert.match(api.relayProblem(code).message,/\S/);
  assert.match(api.relayProblem('connector_asleep').message,/asleep|not running/i);
  assert.match(api.relayProblem('upload_unsupported').message,/update/i);
  assert.equal(api.relayProblem('upload_unsupported').status,503);
  assert.equal(api.relayProblem('connector_asleep').status,503);
  assert.equal(api.relayProblem('relay_busy').status,429);
  assert.doesNotMatch(api.relayProblem('weird').message,/weird/);
});

test('base64 of bytes round trips for sizes around the chunk boundary',()=>{
  assert.ok(api);
  for(const size of [0,1,2,3,699999,700000,700001]){
    const bytes=new Uint8Array(size).map((_,i)=>i*31%251);
    assert.deepEqual(new Uint8Array(Buffer.from(api.toBase64(bytes),'base64')),bytes);
  }
});

test('a body is cut into chunks of at most 700000 bytes and counted',async()=>{
  assert.ok(api);
  const total=1500001;const data=new Uint8Array(total).map((_,i)=>i%251);
  const stream=new ReadableStream({start(controller){for(let i=0;i<total;i+=65536)controller.enqueue(data.subarray(i,Math.min(total,i+65536)));controller.close();}});
  const sizes=[];let seen=0;
  for await(const piece of api.chunksOf(stream,api.CHUNK_BYTES,total)){sizes.push(piece.length);assert.deepEqual(piece,data.subarray(seen,seen+piece.length));seen+=piece.length;}
  assert.deepEqual(sizes,[700000,700000,100001]);
});

test('a body longer than it said is cut off before the extra byte is passed on',async()=>{
  assert.ok(api);
  const stream=new ReadableStream({start(controller){controller.enqueue(new Uint8Array(100));controller.enqueue(new Uint8Array(100));controller.close();}});
  const got=[];
  await assert.rejects(async()=>{for await(const piece of api.chunksOf(stream,60,150))got.push(piece.length);},{code:'size_mismatch'});
  assert.ok(got.reduce((a,b)=>a+b,0)<=150);
});

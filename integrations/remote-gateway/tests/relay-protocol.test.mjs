import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeRelayFrame, encodeRelayFrame, MAX_FRAME_BYTES, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES } from '../src/relay-protocol.ts';
function request() { return { v: 1, kind: 'request', id: 'request-a', generation: 2, deadlineAt: 123456, headers: [['Content-Type','application/json'], ['Mcp-Method','tools/call']], bodyBase64: Buffer.from('{"jsonrpc":"2.0","method":"tools/call"}').toString('base64') }; }
function decode(f) { return decodeRelayFrame(JSON.stringify(f)); }
function reject(f, code = 'INVALID_FRAME') { assert.deepEqual(decode(f), { ok: false, code }); }
test('request bytes and verified routing headers survive roundtrip', () => { const f = request(); f.bodyBase64 = Buffer.from(' {"text":"नमस्ते 👋","spacing":  2}\n').toString('base64'); const encoded = encodeRelayFrame(f); assert.equal(encoded.ok, true); const r = decodeRelayFrame(encoded.text); assert.equal(r.ok, true); assert.deepEqual(r.frame, f); assert.equal(Buffer.from(r.frame.bodyBase64,'base64').toString(), ' {"text":"नमस्ते 👋","spacing":  2}\n'); });
test('response preserves status and binary byte sequence', () => { const f = { v: 1, kind: 'response', id: 'request-a', generation: 2, status: 403, headers: [['Content-Type','application/json']], bodyBase64: Buffer.from([0,1,2,255]).toString('base64') }; const r = decode(f); assert.equal(r.ok, true); assert.deepEqual(r.frame, f); });
test('cancellation contains correlation identity only', () => { const f = { v: 1, kind: 'cancel', id: 'request-a', generation: 2 }; assert.deepEqual(decode(f), { ok: true, frame: f }); });
for (const [label, patch] of [['unsupported version',{v:2}],['unknown kind',{kind:'admin'}],['missing ID',{id:undefined}],['invalid ID',{id:'../foreign'}],['zero generation',{generation:0}],['fraction generation',{generation:1.5}],['expired-shaped deadline',{deadlineAt:-1}],['unknown routing field',{upstreamUrl:'https://attacker.example'}],['owner override',{ownerId:'other'}],['malformed headers',{headers:{}}],['bearer forwarded',{headers:[['Authorization','synthetic-only']]}],['cookie forwarded',{headers:[['Cookie','synthetic-only']]}],['forwarded Host',{headers:[['Host','foreign.example']]}],['header injection',{headers:[['Accept','text/html\r\nX: y']]}],['duplicate headers',{headers:[['Accept','application/json'],['accept','text/event-stream']]}],['malformed base64',{bodyBase64:'%%%%'}],['noncanonical pad bits',{bodyBase64:'AB=='}]]) test(label, () => reject({...request(), ...patch}));
test('JSON arrays and primitive inputs refused', () => { for (const value of [[],null,3,'hello']) reject(value); });
test('malformed JSON refused without raw error text', () => { assert.deepEqual(decodeRelayFrame('{'), {ok:false,code:'INVALID_FRAME'}); });
test('invalid UTF8 wire bytes refused', () => { assert.deepEqual(decodeRelayFrame(new Uint8Array([255])), {ok:false,code:'INVALID_UTF8'}); });
test('duplicate JSON fields rejected by canonical private wire contract', () => { const text = JSON.stringify(request()).replace('"generation":2','"generation":1,"generation":2'); assert.deepEqual(decodeRelayFrame(text), {ok:false,code:'NON_CANONICAL_FRAME'}); });
test('wire size cap enforced before JSON parsing', () => { assert.deepEqual(decodeRelayFrame('x'.repeat(MAX_FRAME_BYTES+1)), {ok:false,code:'FRAME_TOO_LARGE'}); });
test('request body cap enforced on decoded bytes', () => reject({...request(),bodyBase64:Buffer.alloc(MAX_REQUEST_BYTES+1).toString('base64')}, 'BODY_TOO_LARGE'));
test('response body cap enforced on decoded bytes', () => reject({v:1,kind:'response',id:'response-a',generation:1,status:200,headers:[],bodyBase64:Buffer.alloc(MAX_RESPONSE_BYTES+1).toString('base64')}, 'BODY_TOO_LARGE'));
test('empty bodies and headers permitted for transport semantics', () => { assert.equal(decode({...request(),bodyBase64:'',headers:[]}).ok,true); });
test('response cannot forward Set-Cookie', () => reject({v:1,kind:'response',id:'response-a',generation:1,status:200,headers:[['Set-Cookie','synthetic-only']],bodyBase64:''}));
test('invalid HTTP status refused', () => { for (const status of [101,600,200.5]) reject({v:1,kind:'response',id:'response-a',generation:1,status,headers:[],bodyBase64:''}); });
test('returned header snapshot frozen and detached', () => { const f=request(); const r=decode(f); assert.equal(r.ok,true); f.headers[0][1]='changed'; assert.equal(r.frame.headers[0][1],'application/json'); assert.equal(Object.isFrozen(r.frame),true); assert.equal(Object.isFrozen(r.frame.headers[0]),true); });

test('request maximum body boundary accepted', () => { assert.equal(decode({...request(),bodyBase64:Buffer.alloc(MAX_REQUEST_BYTES).toString('base64')}).ok,true); });
test('response maximum body boundary accepted', () => { assert.equal(decode({v:1,kind:'response',id:'response-a',generation:1,status:200,headers:[],bodyBase64:Buffer.alloc(MAX_RESPONSE_BYTES).toString('base64')}).ok,true); });
test('request cannot carry response-only Retry-After', () => reject({...request(),headers:[['Retry-After','1']]}));
test('response cannot carry request routing headers', () => reject({v:1,kind:'response',id:'response-a',generation:1,status:200,headers:[['Mcp-Method','tools/call']],bodyBase64:''}));
test('response cannot contain a request deadline', () => reject({v:1,kind:'response',id:'response-a',generation:1,status:200,headers:[],bodyBase64:'',deadlineAt:123}));
test('cancellation cannot carry payload or headers', () => reject({v:1,kind:'cancel',id:'response-a',generation:1,bodyBase64:''}));
test('excess header count and value length refused', () => { reject({...request(),headers:Array.from({length:33},()=>['Accept','application/json'])}); reject({...request(),headers:[['Accept','a'.repeat(8193)]]}); });
test('deep invalid JSON shape fails closed without stringify recursion', () => { const nested='['.repeat(20000)+'0'+']'.repeat(20000); const text=JSON.stringify(request()).replace('"headers":[["Content-Type","application/json"],["Mcp-Method","tools/call"]]','"headers":'+nested); assert.deepEqual(decodeRelayFrame(text),{ok:false,code:'INVALID_FRAME'}); });
test('malformed runtime input returns a safe error', () => { assert.deepEqual(decodeRelayFrame(null),{ok:false,code:'INVALID_FRAME'}); });
test('encoder refuses malformed runtime frame', () => { assert.deepEqual(encodeRelayFrame({...request(),v:2}),{ok:false,code:'INVALID_FRAME'}); const cycle={};cycle.self=cycle;assert.deepEqual(encodeRelayFrame(cycle),{ok:false,code:'INVALID_FRAME'}); });

test('UTF8 BOM byte and string representations both refused', () => { const text='\ufeff'+JSON.stringify(request()); assert.deepEqual(decodeRelayFrame(text),{ok:false,code:'INVALID_FRAME'}); assert.deepEqual(decodeRelayFrame(new TextEncoder().encode(text)),{ok:false,code:'INVALID_FRAME'}); });
test('schema-approved parameter header roundtrip supported', () => { const f={...request(),headers:[['Mcp-Param-query','project-a']]};const options={requestParamHeaders:['Mcp-Param-query']};const encoded=encodeRelayFrame(f,options);assert.equal(encoded.ok,true);assert.deepEqual(decodeRelayFrame(encoded.text,options).frame,f); });
for (const [label,header,options] of [
 ['unapproved parameter',['Mcp-Param-query','x'],{}],
 ['invalid suffix',['Mcp-Param-','x'],{requestParamHeaders:['Mcp-Param-']}],
 ['credential disguised as schema',['Authorization','x'],{requestParamHeaders:['Authorization']}],
 ['parameter injection',['Mcp-Param-query','x\r\ny'],{requestParamHeaders:['Mcp-Param-query']}],
 ['parameter excessive length',['Mcp-Param-query','x'.repeat(8193)],{requestParamHeaders:['Mcp-Param-query']}],
]) test(label,()=>assert.deepEqual(decodeRelayFrame(JSON.stringify({...request(),headers:[header]}),options),{ok:false,code:'INVALID_FRAME'}));
test('parameter case-duplicates rejected',()=>{const f={...request(),headers:[['Mcp-Param-query','x'],['mcp-param-query','y']]};assert.deepEqual(decodeRelayFrame(JSON.stringify(f),{requestParamHeaders:['Mcp-Param-query']}),{ok:false,code:'INVALID_FRAME'});});
test('schema param never permitted on response',()=>{const f={v:1,kind:'response',id:'r',generation:1,status:200,headers:[['Mcp-Param-query','x']],bodyBase64:''};assert.deepEqual(decodeRelayFrame(JSON.stringify(f),{requestParamHeaders:['Mcp-Param-query']}),{ok:false,code:'INVALID_FRAME'});});

test('gateway relay budget and the laptop companion budget never drift apart', async () => {
  const { RELAY_DEADLINE_MS } = await import('../src/relay-protocol.ts');
  const { readFile } = await import('node:fs/promises');
  const python = await readFile(new URL('../../../src/superlocalmemory/remote_connections/session.py', import.meta.url), 'utf8');
  const laptop = Number(/^RELAY_DEADLINE_MS = (\d+)$/m.exec(python)?.[1]);
  assert.equal(RELAY_DEADLINE_MS, 25000);
  assert.equal(laptop, RELAY_DEADLINE_MS);
});

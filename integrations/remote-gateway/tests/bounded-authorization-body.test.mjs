import test from 'node:test';import assert from 'node:assert/strict';
import {readAuthorizationBody} from '../src/authorization-body.ts';
test('chunked body exceeds byte cap before unlimited buffering',async()=>{
 let cancelled=false;
 const stream=new ReadableStream({start(c){c.enqueue(new Uint8Array(16384));c.enqueue(new Uint8Array(1));},cancel(){cancelled=true;}});
 await assert.rejects(readAuthorizationBody(new Request('https://auth.example',{method:'POST',body:stream,duplex:'half'})),/body_too_large/);assert(cancelled);
});
test('stalled body times out and cancels its reader',async()=>{
 let cancelled=false;
 const stream=new ReadableStream({cancel(){cancelled=true;}});
 await assert.rejects(readAuthorizationBody(new Request('https://auth.example',{method:'POST',body:stream,duplex:'half'}),{timeoutMs:10}),/body_timeout/);assert(cancelled);
});
test('strict UTF8 rejects malformed byte sequences',async()=>{
 await assert.rejects(readAuthorizationBody(new Request('https://auth.example',{method:'POST',body:new Uint8Array([255])})),/invalid_body/);
});

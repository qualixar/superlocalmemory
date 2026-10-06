import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import WebSocket,{WebSocketServer} from 'ws';
import {ConnectorSupervisor} from '../src/connector-supervisor.ts';
import {createNodeConnector} from '../src/node-connector.mjs';

test('Node adapter is exported without opening remote access',()=>assert.equal(typeof createNodeConnector,'function'));
test('invalid local configuration terminates a connecting real ws without process crash',async()=>{
  const server=createServer();server.listen(0,'127.0.0.1');await once(server,'listening');
  const states=[];
  const supervisor=new ConnectorSupervisor({enabled:true,loadEnrollment:async()=>({endpoint:'wss://connect.superlocalmemory.com/connector',deviceToken:'d'.repeat(40),expiresAt:Date.now()+60000,origin:'http://evil.invalid/mcp',originHeaders:{}}),dial:config=>new WebSocket(`ws://127.0.0.1:${server.address().port}`,config.options),onState:s=>states.push(s)});
  try{await supervisor.start();await new Promise(r=>setTimeout(r,10));assert.equal(states.at(-1),'configuration_error');}
  finally{supervisor.stop();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
});
test('real outbound socket forwards to a fixed local HTTP endpoint',async()=>{
  let localAuth;
  const local=createServer((req,res)=>{localAuth=req.headers['x-install-token'];let body='';req.on('data',chunk=>{body+=chunk;});req.on('end',()=>{assert.equal(req.url,'/mcp/');res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({jsonrpc:'2.0',id:7,result:{received:JSON.parse(body).method}}));});});
  local.listen(0,'127.0.0.1');await once(local,'listening');
  const wss=new WebSocketServer({host:'127.0.0.1',port:0});await once(wss,'listening');
  let supervisor;
  try {
    const incoming=once(wss,'connection');
    supervisor=new ConnectorSupervisor({enabled:true,loadEnrollment:async()=>({endpoint:'wss://connect.superlocalmemory.com/connector',deviceToken:'d'.repeat(40),expiresAt:Date.now()+60000,origin:`http://127.0.0.1:${local.address().port}/mcp/`,originHeaders:{'X-Install-Token':'synthetic-origin-token'}}),dial:config=>new WebSocket(`ws://127.0.0.1:${wss.address().port}`,config.options),onState(){}});
    await supervisor.start();const [socket,req]=await incoming;
    assert.equal(req.headers.authorization,'Bearer '+'d'.repeat(40));
    socket.send('{"v":1,"kind":"ready","generation":4}');
    const response=once(socket,'message');socket.send(JSON.stringify({v:1,kind:'request',id:'real-wire',generation:4,deadlineAt:Date.now()+2000,headers:[['content-type','application/json']],bodyBase64:Buffer.from('{"jsonrpc":"2.0","id":7,"method":"ping"}').toString('base64')}));
    const [bytes]=await response;const frame=JSON.parse(bytes.toString());
    assert.equal(frame.id,'real-wire');assert.equal(frame.generation,4);assert.equal(frame.status,200);
    assert.deepEqual(JSON.parse(Buffer.from(frame.bodyBase64,'base64').toString()),{jsonrpc:'2.0',id:7,result:{received:'ping'}});
    assert.equal(localAuth,'synthetic-origin-token');
    assert.ok(!JSON.stringify(frame).includes('synthetic-origin-token'));
  } finally {
    supervisor?.stop();for(const socket of wss.clients)socket.terminate();
    await new Promise(resolve=>wss.close(resolve));
    local.closeAllConnections();await new Promise(resolve=>local.close(resolve));
  }
});

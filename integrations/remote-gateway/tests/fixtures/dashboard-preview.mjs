// Loopback-only visual fixture; never a production SLM/API server.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root=new URL('../../../../src/superlocalmemory/ui/',import.meta.url);
const allowed=new Set(['css/design-system.css','css/neural-glass.css','css/od-bridge.css','js/od-connections.js','js/od-mcp.js','vendor/inter-ui/variable/InterVariable.woff2','vendor/inter-ui/variable/InterVariable-Italic.woff2']);
const html='<!doctype html><html data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/static/css/design-system.css"><link rel="stylesheet" href="/static/css/neural-glass.css"><link rel="stylesheet" href="/static/css/od-bridge.css"></head><body><p style="padding:8px 26px;font-size:12px">Synthetic dashboard preview — no live cloud connection</p><div id="mcp-pane"></div><script src="/static/js/od-connections.js"></script><script src="/static/js/od-mcp.js"></script></body></html>';
const server=http.createServer(async(req,res)=>{
 try{const path=new URL(req.url,'http://localhost').pathname;let body;let type='application/json';
 if(path==='/'){body=html;type='text/html';}
 else if(path==='/api/v3/connections/status')body=JSON.stringify({installation_id:'synthetic-install',available:true,current_profile:'synthetic-profile',hosts:['muse','chatgpt'],connections:[]});
 else if(path==='/api/v3/mcp/profiles')body=JSON.stringify({current:'core',profiles:{core:{count:2,tools:['recall','remember'],description:'Synthetic tool profile'}},total_tools:2,aliases:{}});
 else if(path.startsWith('/static/')&&allowed.has(path.slice(8))){body=await readFile(new URL(path.slice(8),root));type=path.endsWith('.css')?'text/css':path.endsWith('.js')?'text/javascript':'font/woff2';}
 else{res.writeHead(404);res.end();return;}
 res.writeHead(200,{'Content-Type':type==='font/woff2'?type:type+'; charset=utf-8','Cache-Control':'no-store'});res.end(body);
 }catch{res.writeHead(500);res.end('fixture error');}
});
server.listen(0,'127.0.0.1',()=>console.log('http://127.0.0.1:'+server.address().port));

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
const source=readFileSync(new URL('../../../src/superlocalmemory/ui/js/od-connections.js',import.meta.url),'utf8');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function setup(status={available:true,current_profile:'profile-a',hosts:['muse','chatgpt'],connections:[]},reply={state:'pending',connection_id:'synthetic-connection'}){
 status=Object.assign({installation_id:'synthetic-install'},status);
 const dom=new JSDOM('<!doctype html><div id="pane"></div>',{url:'http://127.0.0.1:8765',runScripts:'outside-only',pretendToBeVisual:true});const calls=[];let fail=false;dom.window.open=()=>null;
 dom.window.slmFetch=async(path,init={})=>{calls.push({path,init});if(init.method==='POST'&&fail)throw new Error('synthetic transport error');return new Response(JSON.stringify(init.method==='POST'?reply:status),{status:200});};dom.window.eval(source);const card=dom.window.odCreateAiConnectionsCard();dom.window.document.getElementById('pane').append(card);await tick();await tick();return {dom,card,calls,setFail:v=>{fail=v;}};
}
function click(card,text){const btn=[...card.querySelectorAll('button')].find(x=>x.textContent===text);assert.ok(btn,'button '+text+' exists');btn.click();}
async function submit(f,optIn=true){click(f.card,'Add AI connection');const checkbox=f.card.querySelector('[data-remote-opt-in]');assert.ok(checkbox);checkbox.checked=optIn;f.card.querySelector('form').dispatchEvent(new f.dom.window.Event('submit',{bubbles:true,cancelable:true}));await tick();await tick();}
test('existing dashboard card exposes connection initiation without infrastructure fields',async()=>{const f=await setup();try{assert.match(f.card.textContent,/Connect your AI/);assert.ok(f.card.querySelector('button'));assert.equal(f.card.querySelector('input[type=password]'),null);assert.doesNotMatch(f.card.textContent,/API token|DNS record|terminal command/);}finally{f.dom.window.close();}});
test('rendering never enables remote access automatically',async()=>{const f=await setup();try{assert.equal(f.calls.filter(x=>x.init.method==='POST').length,0);assert.equal(f.calls[0].path,'/api/v3/connections/status');}finally{f.dom.window.close();}});
test('explicit UI consent and chosen host initiate through local authenticated fetch helper',async()=>{const f=await setup();try{await submit(f);const call=f.calls.find(x=>x.init.method==='POST');assert.ok(call);assert.equal(call.path,'/api/v3/connections/initiate');const body=JSON.parse(call.init.body);assert.equal(body.remote_opt_in,true);assert.equal(body.profile_id,'profile-a');assert.equal(body.host,'muse');assert.deepEqual(body.permissions,{read:true,write:false,correction:false,session:false});assert.ok(call.init.headers['Idempotency-Key']);assert.equal(call.init.credentials,'same-origin');}finally{f.dom.window.close();}});
test('absent consent blocks initiation',async()=>{const f=await setup();try{await submit(f,false);assert.equal(f.calls.filter(x=>x.init.method==='POST').length,0);}finally{f.dom.window.close();}});
test('unavailable backend disables initiation honestly',async()=>{const f=await setup({available:false,current_profile:'profile-a',hosts:['muse'],connections:[]});try{assert.ok([...f.card.querySelectorAll('button')].find(x=>x.textContent==='Add AI connection').disabled);assert.match(f.card.textContent,/unavailable/i);}finally{f.dom.window.close();}});
test('pending receipt never appears as connected',async()=>{const f=await setup();try{await submit(f);assert.match(f.card.textContent,/Waiting|requested|pending/i);assert.doesNotMatch(f.card.textContent,/Connected/);}finally{f.dom.window.close();}});
test('unsafe authorization links are never inserted',async()=>{for(const url of ['javascript:alert(1)','https://evil.example/authorize','https://name:pass@auth.superlocalmemory.com/authorize']){const f=await setup(undefined,{state:'pending',connection_id:'synthetic-connection',authorization_url:url});try{await submit(f);assert.equal(f.card.querySelector('a'),null);}finally{f.dom.window.close();}}});
test('backend connection-bound sign-in receipt renders a safe link',async()=>{const url='https://auth.superlocalmemory.com/owner-login?connection_id=synthetic-connection';const f=await setup(undefined,{state:'pending',connection_id:'synthetic-connection',authorization_url:url});try{await submit(f);const link=f.card.querySelector('a');assert.ok(link);assert.equal(link.href,url);assert.equal(link.rel,'noopener noreferrer');assert.equal(link.target,'_blank');}finally{f.dom.window.close();}});
test('sign-in links reject foreign identity and ambiguous query parameters',async()=>{for(const suffix of ['?connection_id=foreign','?connection_id=synthetic-connection&connection_id=foreign','?connection_id=synthetic-connection&state=extra','?connection_id=synthetic-connection#fragment']){const f=await setup(undefined,{state:'pending',connection_id:'synthetic-connection',authorization_url:'https://auth.superlocalmemory.com/owner-login'+suffix});try{await submit(f);assert.equal(f.card.querySelector('a'),null);}finally{f.dom.window.close();}}});
test('same uncertain request retry retains idempotency key',async()=>{const f=await setup();try{f.setFail(true);await submit(f);const form=f.card.querySelector('form');assert.ok(form);f.setFail(false);form.dispatchEvent(new f.dom.window.Event('submit',{bubbles:true,cancelable:true}));await tick();await tick();const posts=f.calls.filter(x=>x.init.method==='POST');assert.equal(posts.length,2);assert.equal(posts[0].init.headers['Idempotency-Key'],posts[1].init.headers['Idempotency-Key']);}finally{f.dom.window.close();}});
test('unverified connected status is not trusted',async()=>{const f=await setup({available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',state:'connected',verified:false}]});try{assert.doesNotMatch(f.card.textContent,/Connected/);}finally{f.dom.window.close();}});

test('pending connection can be cancelled with its observed profile and version',async()=>{
 const status={available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',state:'pending',connection_id:'a'.repeat(32),version:3,verified:false}]};
 const f=await setup(status);try{
 const original=f.dom.window.slmFetch;
 f.dom.window.slmFetch=async(path,init={})=>{if(path.endsWith('/cancel')){f.calls.push({path,init});status.connections[0]={...status.connections[0],state:'cancelled',version:4,cleanup_pending:true};return new Response(JSON.stringify(status.connections[0]),{status:200});}return original(path,init);};
 click(f.card,'Cancel connection');await tick();await tick();
 const request=f.calls.find(x=>x.path.endsWith('/cancel'));assert.ok(request);
 assert.equal(request.path,'/api/v3/connections/'+'a'.repeat(32)+'/cancel');
 assert.deepEqual(JSON.parse(request.init.body),{profile_id:'profile-a',expected_version:3});
 assert.match(f.card.textContent,/cleanup pending/i);assert.doesNotMatch(f.card.textContent,/Connected/);
 assert.equal([...f.card.querySelectorAll('button')].some(x=>x.textContent==='Cancel connection'),false);
 }finally{f.dom.window.close();}
});
test('failed cancellation remains unconfirmed and can be retried',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',state:'pending',connection_id:'a'.repeat(32),version:3,verified:false}]});try{
 const original=f.dom.window.slmFetch;f.dom.window.slmFetch=(path,init)=>path.endsWith('/cancel')?Promise.resolve(new Response('{}',{status:503})):original(path,init);
 click(f.card,'Cancel connection');await tick();await tick();
 assert.match(f.card.textContent,/Cancellation could not be confirmed/);
 assert.equal([...f.card.querySelectorAll('button')].find(x=>x.textContent==='Cancel connection').disabled,false);
 }finally{f.dom.window.close();}
});
test('uncertain initiation can be cancelled and retried with a fresh intent key',async()=>{
 const status={available:true,current_profile:'profile-a',hosts:['muse'],connections:[]};
 const f=await setup(status);try{
 f.setFail(true);await submit(f);const first=f.calls.find(x=>x.init.method==='POST');
 status.connections.push({host:'muse',state:'pending',connection_id:'a'.repeat(32),version:2,verified:false,intent_key:first.init.headers['Idempotency-Key']});
 click(f.card,'Refresh status');await tick();await tick();
 const original=f.dom.window.slmFetch;f.dom.window.slmFetch=async(path,init={})=>{if(path.endsWith('/cancel')){status.connections[0]={...status.connections[0],state:'cancelled',version:3,cleanup_pending:true};return new Response(JSON.stringify(status.connections[0]),{status:200});}return original(path,init);};
 click(f.card,'Cancel connection');await tick();await tick();
 f.card.remove();f.card=f.dom.window.odCreateAiConnectionsCard();f.dom.window.document.getElementById('pane').append(f.card);await tick();await tick();
 f.setFail(false);await submit(f);
 const posts=f.calls.filter(x=>x.path==='/api/v3/connections/initiate'&&x.init.method==='POST');
 assert.equal(posts.length,2);assert.notEqual(posts[0].init.headers['Idempotency-Key'],posts[1].init.headers['Idempotency-Key']);
 }finally{f.dom.window.close();}
});

test('existing MCP pane mounts initiation card alongside current profile tools',async()=>{const f=await setup();try{f.dom.window.eval(readFileSync(new URL('../../../src/superlocalmemory/ui/js/od-mcp.js',import.meta.url),'utf8'));const pane=f.dom.window.document.createElement('div');f.dom.window.document.body.append(pane);f.dom.window.odRenderMcp(pane);await tick();await tick();assert.ok(pane.querySelector('#od-ai-connections'));assert.match(pane.textContent,/MCP & Integrations/);}finally{f.dom.window.close();}});
test('MCP profile API failure retains independent initiation card',async()=>{const f=await setup();try{const original=f.dom.window.slmFetch;f.dom.window.slmFetch=(path,init)=>path==='/api/v3/mcp/profiles'?Promise.resolve(new Response('{}',{status:503})):original(path,init);f.dom.window.eval(readFileSync(new URL('../../../src/superlocalmemory/ui/js/od-mcp.js',import.meta.url),'utf8'));const pane=f.dom.window.document.createElement('div');f.dom.window.document.body.append(pane);f.dom.window.odRenderMcp(pane);await tick();await tick();assert.ok(pane.querySelector('#od-ai-connections'));}finally{f.dom.window.close();}});

test('invalid acknowledgement remains unconfirmed',async()=>{const f=await setup(undefined,{});try{await submit(f);assert.match(f.card.textContent,/could not be confirmed/);}finally{f.dom.window.close();}});
test('acknowledged first host allows deliberate second host with fresh key',async()=>{const f=await setup();try{await submit(f);f.card.querySelector('select').value='chatgpt';f.card.querySelector('form').dispatchEvent(new f.dom.window.Event('submit',{bubbles:true,cancelable:true}));await tick();await tick();const posts=f.calls.filter(x=>x.init.method==='POST');assert.equal(posts.length,2);assert.notEqual(posts[0].init.headers['Idempotency-Key'],posts[1].init.headers['Idempotency-Key']);}finally{f.dom.window.close();}});
test('secret-bearing sign-in URL is not embedded',async()=>{const f=await setup(undefined,{state:'pending',connection_id:'synthetic-connection',authorization_url:'https://auth.superlocalmemory.com/authorize?access_token=synthetic-only'});try{await submit(f);assert.equal(f.card.querySelector('a'),null);}finally{f.dom.window.close();}});

test('uncertain initiation survives card recreation with same key',async()=>{const f=await setup();try{f.setFail(true);await submit(f);f.card.remove();f.card=f.dom.window.odCreateAiConnectionsCard();f.dom.window.document.getElementById('pane').append(f.card);await tick();await tick();f.setFail(false);await submit(f);const posts=f.calls.filter(x=>x.init.method==='POST');assert.equal(posts.length,2);assert.equal(posts[0].init.headers['Idempotency-Key'],posts[1].init.headers['Idempotency-Key']);}finally{f.dom.window.close();}});
test('uncertain persisted intent rejects changed permissions after recreation',async()=>{const f=await setup();try{f.setFail(true);await submit(f);f.card.remove();f.card=f.dom.window.odCreateAiConnectionsCard();f.dom.window.document.getElementById('pane').append(f.card);await tick();await tick();click(f.card,'Add AI connection');f.card.querySelector('[data-remote-opt-in]').checked=true;f.card.querySelector('[data-permission=write]').checked=true;f.card.querySelector('form').dispatchEvent(new f.dom.window.Event('submit',{bubbles:true,cancelable:true}));await tick();assert.equal(f.calls.filter(x=>x.init.method==='POST').length,1);}finally{f.dom.window.close();}});

test('restored non-default host and save permission can be retried without reconstruction',async()=>{const f=await setup();try{f.setFail(true);click(f.card,'Add AI connection');f.card.querySelector('select').value='chatgpt';f.card.querySelector('[data-permission=write]').checked=true;f.card.querySelector('[data-remote-opt-in]').checked=true;f.card.querySelector('form').dispatchEvent(new f.dom.window.Event('submit',{bubbles:true,cancelable:true}));await tick();await tick();f.card.remove();f.card=f.dom.window.odCreateAiConnectionsCard();f.dom.window.document.getElementById('pane').append(f.card);await tick();await tick();assert.equal(f.card.querySelector('select').value,'chatgpt');assert.equal(f.card.querySelector('[data-permission=write]').checked,true);assert.equal(f.card.querySelector('[data-remote-opt-in]').checked,false);f.setFail(false);await submit(f);const posts=f.calls.filter(x=>x.init.method==='POST');assert.equal(posts.length,2);assert.equal(posts[0].init.headers['Idempotency-Key'],posts[1].init.headers['Idempotency-Key']);}finally{f.dom.window.close();}});

test('verified device readiness exposes only canonical URL and disconnect action',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',connection_id:'a'.repeat(32),state:'ready_for_client',verified:true,version:2,mcp_url:'https://mcp.superlocalmemory.com/mcp'}]});
 try{assert.match(f.card.textContent,/Ready for AI client/);assert.equal(f.card.querySelector('input[readonly]')?.value,'https://mcp.superlocalmemory.com/mcp');assert.ok([...f.card.querySelectorAll('button')].some(x=>x.textContent==='Cancel connection'));}finally{f.dom.window.close();}
});
test('unverified or noncanonical endpoint never exposes URL',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',connection_id:'a'.repeat(32),state:'ready_for_client',verified:true,version:2,mcp_url:'https://evil.example/mcp'}]});
 try{assert.equal(f.card.querySelector('input[readonly]'),null);}finally{f.dom.window.close();}
});

test('Composio is a supported dashboard destination',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['composio'],connections:[]});
 try{assert.equal(f.card.querySelector('select').value,'composio');assert.equal([...f.card.querySelectorAll('button')].find(x=>x.textContent==='Add AI connection').disabled,false);}finally{f.dom.window.close();}
});
test('sign-in starts from the user gesture and navigates only to verified receipt',async()=>{
 const url='https://auth.superlocalmemory.com/owner-login?connection_id=synthetic-connection';const f=await setup(undefined,{state:'pending',connection_id:'synthetic-connection',authorization_url:url});const opened=[];const navigated=[];const popup={opener:f.dom.window,location:{replace:value=>navigated.push(value)},close(){}};
 f.dom.window.open=(...args)=>{opened.push(args);return popup;};
 try{await submit(f);assert.equal(opened[0][0],'about:blank');assert.equal(popup.opener,null);assert.deepEqual(navigated,[url]);}finally{f.dom.window.close();}
});
test('pending status polls until canonical verified readiness without another click',async()=>{
 const status={available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',connection_id:'a'.repeat(32),state:'pending',verified:false,version:2}]};
 const f=await setup(status);const scheduled=[];f.dom.window.setTimeout=fn=>{scheduled.push(fn);return 1;};
 try{click(f.card,'Refresh status');await tick();await tick();assert.equal(scheduled.length,1);
 status.connections[0]={...status.connections[0],state:'ready_for_client',verified:true,mcp_url:'https://mcp.superlocalmemory.com/mcp'};
 scheduled.shift()();await tick();await tick();assert.match(f.card.textContent,/Ready for AI client/);assert.equal(scheduled.length,0);
 }finally{f.dom.window.close();}
});

test('guided setup shows visible client choices and a single ordered journey',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['composio','chatgpt','muse','claude_web','other_mcp'],connections:[]});
 try{for(const client of ['Composio','ChatGPT Web','Musebot','Claude Web','Other MCP client'])assert.ok(f.card.querySelector('[data-client="'+({Composio:'composio','ChatGPT Web':'chatgpt',Musebot:'muse','Claude Web':'claude_web','Other MCP client':'other_mcp'}[client])+'"]'));
 assert.match(f.card.textContent,/Choose your AI/);assert.match(f.card.textContent,/Link this computer/);assert.match(f.card.textContent,/Verify connection/);assert.match(f.card.textContent,/Connect your AI/);assert.equal(f.card.querySelector('form').hidden,false);
 }finally{f.dom.window.close();}
});
test('Muse selection explains its adapter and generic client selection remains bounded',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['muse','other_mcp'],connections:[]});
 try{f.card.querySelector('[data-client=muse]').click();assert.match(f.card.textContent,/private adapter/i);f.card.querySelector('[data-client=other_mcp]').click();assert.equal(f.card.querySelector('select').value,'other_mcp');assert.match(f.card.textContent,/compatible.*MCP/i);}finally{f.dom.window.close();}
});
test('unchanged pending polls preserve the same cancel button and chosen host',async()=>{
 const state={available:true,current_profile:'profile-a',hosts:['muse','composio'],connections:[{host:'muse',connection_id:'a'.repeat(32),state:'pending',verified:false,version:3,link_status:'sign_in_required'}]};const f=await setup(state);
 try{const button=[...f.card.querySelectorAll('button')].find(x=>x.textContent==='Cancel connection');f.card.querySelector('select').value='composio';click(f.card,'Refresh status');await tick();await tick();assert.equal([...f.card.querySelectorAll('button')].find(x=>x.textContent==='Cancel connection'),button);assert.equal(f.card.querySelector('select').value,'composio');assert.match(f.card.textContent,/Waiting for GitHub sign-in/);}finally{f.dom.window.close();}
});
test('ready Composio recipe includes correct OAuth metadata URL',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['composio'],connections:[{host:'composio',connection_id:'a'.repeat(32),state:'ready_for_client',verified:true,version:2,mcp_url:'https://mcp.superlocalmemory.com/mcp'}]});
 try{assert.match(f.card.textContent,/OAuth/);assert.equal(f.card.querySelector('input[aria-label="OAuth metadata URL"]')?.value,'https://auth.superlocalmemory.com/.well-known/oauth-authorization-server');}finally{f.dom.window.close();}
});

test('successful cancellation restores client choices and submit without manual refresh',async()=>{
 const state={available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',connection_id:'a'.repeat(32),state:'pending',verified:false,version:3}]};const f=await setup(state);
 try{const original=f.dom.window.slmFetch;f.dom.window.slmFetch=async(path,init)=>{if(path.endsWith('/cancel')){state.connections[0]={...state.connections[0],state:'cancelled',verified:false,cleanup_pending:false,version:4};return new Response(JSON.stringify(state.connections[0]));}return original(path,init);};click(f.card,'Cancel connection');await tick();await tick();assert.equal(f.card.querySelector('[type=submit]').disabled,false);assert.equal(f.card.querySelector('[data-client=muse]').disabled,false);}finally{f.dom.window.close();}
});
test('profile switch clears the old sign-in link and resets consent',async()=>{
 const state={available:true,current_profile:'profile-a',hosts:['muse','composio'],connections:[]};const f=await setup(state,{state:'pending',connection_id:'a'.repeat(32),authorization_url:'https://auth.superlocalmemory.com/owner-login?connection_id='+'a'.repeat(32)});
 try{await submit(f);assert.ok(f.card.querySelector('a'));state.current_profile='profile-b';f.dom.window.slmFetch=async()=>new Response(JSON.stringify({...state,installation_id:'synthetic-install'}));click(f.card,'Refresh status');await tick();await tick();assert.equal(f.card.querySelector('a'),null);assert.equal(f.card.querySelector('[data-remote-opt-in]').checked,false);assert.equal(f.card.querySelector('form').hidden,false);}finally{f.dom.window.close();}
});

test('transport verification highlights step three without claiming ready',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['muse'],connections:[{host:'muse',connection_id:'a'.repeat(32),state:'pending',verified:false,version:3,transport_state:'connecting'}]});
 try{assert.equal(f.card.querySelector('[aria-current=step]').textContent,'Verify connection');assert.match(f.card.textContent,/Verifying computer connection/);assert.equal(f.card.querySelector('input[readonly]'),null);}finally{f.dom.window.close();}
});

test('cancelled attempts stay in collapsed history rather than appearing as duplicate active connections',async()=>{
 const f=await setup({available:true,current_profile:'profile-a',hosts:['composio'],connections:[{host:'composio',connection_id:'a'.repeat(32),state:'cancelled',verified:false,version:5,cleanup_pending:false}]});
 try{const history=f.card.querySelector('details');assert.ok(history);assert.equal(history.open,false);assert.equal(history.querySelector('summary').textContent,'Past connections (1)');assert.match(history.textContent,/Composio: Cancelled/);}finally{f.dom.window.close();}
});

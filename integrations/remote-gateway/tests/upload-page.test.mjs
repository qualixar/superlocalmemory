import test from 'node:test';
import assert from 'node:assert/strict';
let api;try{api=await import('../src/upload-page.ts');}catch{api=null;}

async function page(kind='image',max=26214400){const r=api.uploadPage(kind,max);return {r,html:await r.text(),csp:r.headers.get('content-security-policy')??''};}
const directive=(csp,name)=>(csp.split(';').map(x=>x.trim()).find(x=>x.startsWith(name+' '))??'');

test('the page is locked down: no third parties, no forms, no framing, no referrer, no caching',async()=>{
  assert.ok(api);
  const {r,csp}=await page();
  assert.equal(r.status,200);
  assert.match(r.headers.get('content-type'),/^text\/html; charset=utf-8$/i);
  assert.equal(directive(csp,'default-src'),"default-src 'none'");
  assert.match(directive(csp,'script-src'),/^script-src 'nonce-[A-Za-z0-9+/=_-]{16,}'$/);
  assert.match(directive(csp,'style-src'),/^style-src 'nonce-/);
  assert.equal(directive(csp,'connect-src'),"connect-src 'self'");
  assert.equal(directive(csp,'form-action'),"form-action 'none'");
  assert.equal(directive(csp,'base-uri'),"base-uri 'none'");
  assert.equal(directive(csp,'frame-ancestors'),"frame-ancestors 'none'");
  assert.doesNotMatch(csp,/unsafe-inline|unsafe-eval|\*|https?:/);
  assert.equal(r.headers.get('referrer-policy'),'no-referrer');
  assert.equal(r.headers.get('cache-control'),'no-store');
  assert.equal(r.headers.get('x-content-type-options'),'nosniff');
  assert.match(r.headers.get('x-robots-tag'),/noindex/);
  assert.equal(r.headers.get('access-control-allow-origin'),null);
});

test('the nonce changes on every response and matches the one inline script and style',async()=>{
  assert.ok(api);
  const a=await page(),b=await page();
  const nonce=x=>/script-src 'nonce-([^']+)'/.exec(x.csp)[1];
  assert.notEqual(nonce(a),nonce(b));
  assert.equal(a.html.split('<script').length-1,1);
  assert.equal(a.html.split('<style').length-1,1);
  assert.ok(a.html.includes(`<script nonce="${nonce(a)}">`));
  assert.ok(a.html.includes(`<style nonce="${nonce(a)}">`));
});

test('the page loads nothing from anywhere and has no inline handlers or links',async()=>{
  assert.ok(api);
  const {html}=await page('document',104857600);
  assert.doesNotMatch(html,/\b(src|href|action)\s*=/i);
  assert.doesNotMatch(html,/\son[a-z]+\s*=/i);
  assert.doesNotMatch(html,/https?:\/\//i);
  assert.doesNotMatch(html,/<(iframe|object|embed|link|img|form|base)\b/i);
});

test('the picker and the size hint match the kind',async()=>{
  assert.ok(api);
  const image=await page('image',26214400);
  assert.match(image.html,/accept="image\/png,image\/jpeg,image\/webp"/);
  assert.match(image.html,/up to 25 MB/);
  const doc=await page('document',104857600);
  assert.match(doc.html,/accept="application\/pdf"/);
  assert.match(doc.html,/up to 100 MB/);
  assert.match(doc.html,/PDF/);
});

test('the script uploads the raw file to this same address and shows replies as text only',async()=>{
  assert.ok(api);
  const {html}=await page();
  assert.match(html,/fetch\(location\.pathname/);
  assert.match(html,/method:\s*'POST'/);
  assert.match(html,/credentials:\s*'omit'/);
  assert.match(html,/referrerPolicy:\s*'no-referrer'/);
  assert.doesNotMatch(html,/innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(/);
  assert.match(html,/textContent/);
});

test('a kind or size outside the known set cannot reach the markup',async()=>{
  assert.ok(api);
  const {html}=await page('<script>alert(1)</script>',-5);
  assert.doesNotMatch(html,/alert\(1\)/);
  assert.match(html,/accept="image\/png/);
  const {html:odd}=await page('image',NaN);
  assert.doesNotMatch(odd,/NaN|undefined/);
});

test('a message page escapes everything and carries the status',async()=>{
  assert.ok(api);
  const r=api.messagePage(410,'<img src=x onerror=alert(1)> & "quoted"');
  const html=await r.text();
  assert.equal(r.status,410);
  assert.doesNotMatch(html,/<img/);
  assert.match(html,/&lt;img src=x onerror=alert\(1\)&gt; &amp; &quot;quoted&quot;/);
  assert.equal(r.headers.get('referrer-policy'),'no-referrer');
  assert.equal(r.headers.get('cache-control'),'no-store');
  assert.doesNotMatch(r.headers.get('content-security-policy'),/script-src/);
});

test('a JSON reply carries no cache, no CORS and a fixed shape',async()=>{
  assert.ok(api);
  const r=api.jsonReply(413,{ok:false,code:'too_large',message:'Too big.'});
  assert.equal(r.status,413);
  assert.equal(r.headers.get('cache-control'),'no-store');
  assert.equal(r.headers.get('access-control-allow-origin'),null);
  assert.equal(r.headers.get('x-content-type-options'),'nosniff');
  assert.deepEqual(await r.json(),{ok:false,code:'too_large',message:'Too big.'});
});

async function mount(replyFor,{kind='image',max=26214400}={}){
  const {JSDOM}=await import('jsdom');
  const html=await api.uploadPage(kind,max).text();
  const calls=[];
  const dom=new JSDOM(html,{runScripts:'dangerously',url:'https://mcp.superlocalmemory.com/u/'+'a'.repeat(32)+'/'+'B'.repeat(43),beforeParse(window){
    window.fetch=async(url,init)=>{calls.push({url,init});return replyFor(init);};
  }});
  const {document}=dom.window;
  const pick=(size)=>{const file=new dom.window.File([new Uint8Array(size)],'x.png');Object.defineProperty(document.getElementById('file'),'files',{value:[file],configurable:true});document.getElementById('file').dispatchEvent(new dom.window.Event('change'));};
  return {document,calls,pick,click:()=>document.getElementById('save').click(),settle:()=>new Promise(r=>setTimeout(r,20))};
}

test('the page script posts the raw file to its own address and shows the laptop message as text',async()=>{
  assert.ok(api);
  const m=await mount(async()=>({json:async()=>({ok:true,done:true,message:'Saved to your memory.'})}));
  assert.equal(m.document.getElementById('save').disabled,true);
  m.pick(1000);assert.equal(m.document.getElementById('save').disabled,false);
  m.click();await m.settle();
  assert.equal(m.calls.length,1);
  assert.equal(m.calls[0].url,'/u/'+'a'.repeat(32)+'/'+'B'.repeat(43));
  assert.equal(m.calls[0].init.method,'POST');assert.equal(m.calls[0].init.credentials,'omit');assert.equal(m.calls[0].init.referrerPolicy,'no-referrer');
  assert.equal(m.calls[0].init.body.size,1000);
  assert.equal(m.document.getElementById('status').textContent,'Saved to your memory.');
  assert.equal(m.document.getElementById('save').hidden,true);
});

test('a refusal message is shown as text, never as markup, and the person can try again',async()=>{
  assert.ok(api);
  const m=await mount(async()=>({json:async()=>({ok:false,message:'<img src=x onerror=alert(1)>'})}));
  m.pick(10);m.click();await m.settle();
  const status=m.document.getElementById('status');
  assert.equal(status.textContent,'<img src=x onerror=alert(1)>');assert.equal(status.querySelector('img'),null);
  assert.equal(m.document.getElementById('save').disabled,false);assert.equal(m.document.getElementById('file').disabled,false);
});

test('a file over the limit never leaves the page',async()=>{
  assert.ok(api);
  const m=await mount(async()=>({json:async()=>({ok:true})}),{max:1024*1024});
  m.pick(2*1024*1024);
  assert.equal(m.document.getElementById('save').disabled,true);
  assert.match(m.document.getElementById('status').textContent,/too large.*1 MB/);
  m.click();await m.settle();assert.equal(m.calls.length,0);
});

test('a dropped connection says so and lets the person press Save again',async()=>{
  assert.ok(api);
  const m=await mount(async()=>{throw new Error('network');});
  m.pick(10);m.click();await m.settle();
  assert.match(m.document.getElementById('status').textContent,/interrupted/);
  assert.equal(m.document.getElementById('save').disabled,false);
});

test('the picker says plainly: works once, expires in 10 minutes, do not share it',async()=>{
  assert.ok(api);
  const {html}=await page('image');
  assert.match(html,/works once/i);assert.match(html,/10 minutes/i);assert.match(html,/not share it with anyone/i);
});

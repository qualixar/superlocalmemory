import {test} from 'node:test';
import assert from 'node:assert/strict';
import {renderConnectionSelector} from '../src/owner-selection.ts';
const c=(id,expires,profile='default',host='chatgpt')=>({connectionId:id,deviceExpiresAtMs:expires,profileId:profile,host,revokedAt:null});
test('distinguishes identical profile labels with nonsecret short IDs and expiry, recent first',()=>{
 const page=renderConnectionSelector([c('a'.repeat(32),20000),c('b'.repeat(32),50000),c('c'.repeat(32),40000)],'safe-handle',['slm:read']);
 assert.equal((page.match(/default \(chatgpt\)/g)||[]).length,3);
 assert.ok(page.indexOf('bbbbbbbb')<page.indexOf('cccccccc'));
 assert.ok(page.indexOf('cccccccc')<page.indexOf('aaaaaaaa'));
 assert.match(page,/name="connection_id"/);
});
test('deletion is opt-in, never prechecked, with explicit scope and warning',()=>{
 const page=renderConnectionSelector([c('b'.repeat(32),50000),c('a'.repeat(32),40000)],'safe-handle',['slm:read']);
 assert.match(page,/name="remove_duplicates" value="yes"/);
 assert.doesNotMatch(page,/\bchecked\b/);
 assert.match(page,/other connections with the same local profile and app/i);
});
test('escapes names and consent handle from external data',()=>{
 const page=renderConnectionSelector([c('a'.repeat(32),30000,'<script>alert(1)</script>','chatgpt')],'<img src=x>',['slm:read']);
 assert.doesNotMatch(page,/<script>|<img/);
 assert.match(page,/&lt;script&gt;/);
 assert.match(page,/&lt;img src=x&gt;/);
});

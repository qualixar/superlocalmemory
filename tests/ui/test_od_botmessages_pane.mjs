/**
 * Bot messages pane (js/od-botmessages.js): list, mute, rename, retire.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, writes, flushPromises, TOKEN } from './features_helpers.mjs';

const XSS = '<img src=x onerror="window.__pwn=1">';
const msg = (id, peer, kind, content, extra = {}) => ({
    id, from: { peer_id: peer, app: 'app-' + peer, kind }, to: 'owner', sent_at: '2026-10-09T10:00:00Z',
    expires_at: null, hop: 0, trust: kind === 'web' ? 'untrusted-peer' : 'local-peer', content, refs: [], ...extra });

function setup({ messages, routes = [] } = {}) {
    const state = { messages: messages || [msg(3, 'p-web', 'web', 'hello from web'), msg(2, 'p-loc', 'local', 'hi local')] };
    const base = [['GET', '/api/v3/mesh/messages', () => ({ json: { messages: state.messages } })]];
    const h = makeEnv([...routes, ...base], { modules: ['od-features.js', 'od-botmessages.js'] });
    h.state = state;
    h.pane = h.document.getElementById('pane');
    h.open = async () => { h.window.odRenderBotMessages(h.pane); await flushPromises(); };
    h.btn = (label, scope) => [...(scope || h.pane).querySelectorAll('button')].find(b => b.textContent.trim() === label);
    h.row = peer => h.pane.querySelector(`[data-peer="${peer}"]`);
    return h;
}

describe('Bot messages pane', () => {
    it('requests the newest 50 and renders them in the order given', async () => {
        const h = setup();
        await h.open();
        assert.ok(h.calls.some(c => c.url === '/api/v3/mesh/messages?limit=50&peer='));
        const items = [...h.pane.querySelectorAll('.od-msg')];
        assert.equal(items.length, 2);
        assert.match(items[0].textContent, /hello from web/);
        assert.match(items[1].textContent, /hi local/);
    });

    it('web-origin messages get the untrusted style and label; local ones do not', async () => {
        const h = setup();
        await h.open();
        const [web, loc] = h.pane.querySelectorAll('.od-msg');
        assert.ok(web.classList.contains('untrusted'));
        assert.match(web.textContent, /untrusted/i);
        assert.ok(!loc.classList.contains('untrusted'));
    });

    it('message bodies, app names and peer ids are text, never markup', async () => {
        const evil = msg(1, 'p-x', 'web', 'body ' + XSS);
        evil.from.app = 'app ' + XSS;
        const h = setup({ messages: [evil] });
        await h.open();
        assert.equal(h.pane.querySelectorAll('img').length, 0);
        assert.ok(h.pane.textContent.includes('body ' + XSS));
        assert.ok(h.pane.textContent.includes('app ' + XSS));
        assert.equal(h.window.__pwn, undefined);
    });

    it('mute and unmute POST {muted} with the write credential', async () => {
        const h = setup({ routes: [['POST', '/api/v3/mesh/peers/p-web/mute', () => ({ json: { ok: true, muted: true } })]] });
        await h.open();
        h.btn('Mute', h.row('p-web')).click();
        await flushPromises();
        h.btn('Unmute', h.row('p-web')).click();
        await flushPromises();
        const w = writes(h);
        assert.equal(w[0].url, '/api/v3/mesh/peers/p-web/mute');
        assert.deepEqual(JSON.parse(w[0].body), { muted: true });
        assert.deepEqual(JSON.parse(w[1].body), { muted: false });
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
    });

    it('rename: validates 1-64 printable characters before any request', async () => {
        const h = setup({ routes: [['PATCH', '/api/v3/mesh/peers/p-web', () => ({ json: { ok: true, display_name: 'Ops bot' } })]] });
        await h.open();
        const row = h.row('p-web');
        const input = row.querySelector('input[type=text]');
        for (const bad of ['', '   ', 'x'.repeat(65), 'bell\u0007', 'tab\there']) {
            input.value = bad;
            h.btn('Rename', row).click();
            await flushPromises();
        }
        assert.equal(writes(h).length, 0);
        assert.match(row.textContent, /1 to 64/);
        input.value = 'Ops bot';
        h.btn('Rename', row).click();
        await flushPromises();
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].method, 'PATCH');
        assert.deepEqual(JSON.parse(w[0].body), { display_name: 'Ops bot' });
        assert.match(row.textContent, /Ops bot/);
    });

    it('retire confirms, DELETEs, then reloads the list; declining sends nothing', async () => {
        const h = setup({ routes: [['DELETE', '/api/v3/mesh/peers/p-web', () => ({ json: { ok: true } })]] });
        await h.open();
        h.confirmAnswer = false;
        h.btn('Retire', h.row('p-web')).click();
        await flushPromises();
        assert.equal(writes(h).length, 0);
        h.confirmAnswer = true;
        const before = h.calls.filter(c => c.url.startsWith('/api/v3/mesh/messages')).length;
        h.btn('Retire', h.row('p-web')).click();
        await flushPromises();
        assert.equal(h.confirms.length, 2);
        assert.equal(writes(h)[0].method, 'DELETE');
        assert.ok(h.calls.filter(c => c.url.startsWith('/api/v3/mesh/messages')).length > before);
    });

    it('filtering by peer requests that peer', async () => {
        const h = setup();
        await h.open();
        const sel = h.pane.querySelector('select');
        sel.value = 'p-loc';
        sel.dispatchEvent(new h.window.Event('change', { bubbles: true }));
        await flushPromises();
        assert.ok(h.calls.some(c => c.url === '/api/v3/mesh/messages?limit=50&peer=p-loc'));
    });

    it('a failed write shows the server detail as text', async () => {
        const h = setup({ routes: [['POST', '/api/v3/mesh/peers/p-web/mute', () => ({
            status: 422, json: { detail: 'nope ' + XSS } })]] });
        await h.open();
        h.btn('Mute', h.row('p-web')).click();
        await flushPromises();
        assert.ok(h.pane.textContent.includes('nope ' + XSS));
        assert.equal(h.pane.querySelectorAll('img').length, 0);
    });

    it('a 404 route says not available in this build', async () => {
        const h = makeEnv([], { modules: ['od-features.js', 'od-botmessages.js'] });
        const pane = h.document.getElementById('pane');
        h.window.odRenderBotMessages(pane);
        await flushPromises();
        assert.match(pane.textContent, /not available in this build/);
    });

    it('an empty list says so', async () => {
        const h = setup({ messages: [] });
        await h.open();
        assert.match(h.pane.textContent, /No messages yet/);
    });
});

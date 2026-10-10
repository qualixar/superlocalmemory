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

function setup({ messages, peers, routes = [] } = {}) {
    const state = { messages: messages || [msg(3, 'p-web', 'web', 'hello from web'), msg(2, 'p-loc', 'local', 'hi local')] };
    state.peers = peers || [];
    const base = [['GET', '/api/v3/mesh/messages', () => ({ json: { messages: state.messages } })],
        ['GET', '/api/v3/mesh/peers', () => ({ json: { peers: state.peers } })]];
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
        assert.match(h.pane.textContent, /Bots you connect can leave each other messages here/);
    });

    it('with no messages: one helpful line, an Open Connected apps button, and no Peers heading', async () => {
        const h = setup({ messages: [] });
        const seen = [];
        h.window.slmNavigate = p => seen.push(p);
        await h.open();
        assert.match(h.pane.textContent, /Bots you connect can leave each other messages here\. Allow it per app in Connected apps\./);
        assert.ok(![...h.pane.querySelectorAll('h3')].some(x => x.textContent === 'Peers'));
        h.btn('Open Connected apps').click();
        assert.deepEqual(seen, ['apps-pane']);
    });

    it('with no messages: a numbered "How it works" that matches the approval page and the CLI', async () => {
        const h = setup({ messages: [] });
        await h.open();
        assert.ok([...h.pane.querySelectorAll('h3, h4')].some(x => x.textContent === 'How it works'));
        const steps = [...h.pane.querySelectorAll('ol.od-botmsg-steps > li')].map(li => li.textContent);
        assert.equal(steps.length, 3);
        assert.equal(steps[0], 'Connect an app in Connected apps.');
        assert.equal(steps[1], 'Tick "Allow talking to your other bots" when you approve it.');
        assert.match(steps[2], /^On this computer, allow it for that connection: slm remote keys allow web-<connection id> mesh/);
        assert.match(steps[2], /slm remote keys list/);
        const code = [...h.pane.querySelectorAll('ol.od-botmsg-steps code')].map(c => c.textContent);
        assert.deepEqual(code, ['slm remote keys allow web-<connection id> mesh', 'slm remote keys list']);
        assert.ok(h.btn('Open Connected apps'), 'the button stays');
    });

    it('the steps show only when there are no messages, and not for a filtered peer with none', async () => {
        const h = setup();
        await h.open();
        assert.equal(h.pane.querySelectorAll('ol.od-botmsg-steps').length, 0);
        const peerOnly = setup({ messages: [], peers: [{ peer_id: 'p-x', app: 'x', kind: 'local' }] });
        await peerOnly.open();
        const filter = peerOnly.pane.querySelector('select');
        filter.value = 'p-x';
        filter.dispatchEvent(new peerOnly.window.Event('change', { bubbles: true }));
        await flushPromises();
        assert.match(peerOnly.pane.textContent, /No messages from this peer yet/);
        assert.equal(peerOnly.pane.querySelectorAll('ol.od-botmsg-steps').length, 0);
    });

    it('peers show display name, then app, then a short id, with a kind chip; heading appears with peers', async () => {
        const a = msg(3, 'p-web', 'web', 'x'); a.from.display_name = 'Kitchen bot';
        const b = msg(2, 'p-loc', 'local', 'y');
        const c = msg(1, '0123456789abcdef', 'local', 'z'); delete c.from.app;
        const h = setup({ messages: [a, b, c] });
        await h.open();
        assert.ok([...h.pane.querySelectorAll('h3')].some(x => x.textContent === 'Peers'));
        assert.match(h.row('p-web').textContent, /Kitchen bot/);
        assert.match(h.row('p-web').textContent, /web/);
        assert.match(h.row('p-loc').textContent, /app-p-loc/);
        assert.match(h.row('p-loc').textContent, /this computer/);
        assert.match(h.row('0123456789abcdef').querySelector('strong').textContent, /^01234567$/);
    });

    it('a hostile peer display name is text, never markup', async () => {
        const m = msg(1, 'p-x', 'web', 'hi'); m.from.display_name = XSS;
        const h = setup({ messages: [m] });
        await h.open();
        assert.equal(h.pane.querySelectorAll('img').length, 0);
        assert.ok(h.row('p-x').textContent.includes(XSS));
        assert.equal(h.window.__pwn, undefined);
    });

    const peer = (id, extra = {}) => ({ peer_id: id, display_name: '', kind: 'local', app: '', muted: false,
        last_seen: '2026-10-09T10:00:00Z', status: 'active', ...extra });

    it('lists a connected peer that has sent nothing, with its name, kind chip and muted mark', async () => {
        const h = setup({ messages: [], peers: [
            peer('p-quiet', { display_name: 'Quiet bot', kind: 'web', muted: true }), peer('p-two', { app: 'cursor' })] });
        await h.open();
        assert.ok(h.calls.some(c => c.url === '/api/v3/mesh/peers'));
        assert.ok([...h.pane.querySelectorAll('h3')].some(x => x.textContent === 'Peers'));
        assert.match(h.row('p-quiet').textContent, /Quiet bot/);
        assert.match(h.row('p-quiet').textContent, /web/);
        assert.match(h.row('p-quiet').textContent, /muted/);
        assert.match(h.row('p-two').textContent, /cursor/);
        assert.match(h.row('p-two').textContent, /this computer/);
        assert.ok(!/muted/.test(h.row('p-two').textContent.replace(/Unmute|Mute/g, '')));
    });

    it('peer names fall back from display name to app to the first 8 characters of the id', async () => {
        const h = setup({ messages: [], peers: [peer('aaaaaaaa-1111', { display_name: 'Named', app: 'x' }),
            peer('bbbbbbbb-2222', { app: 'AppOnly' }), peer('cccccccc-3333')] });
        await h.open();
        const name = id => h.row(id).querySelector('strong').textContent;
        assert.equal(name('aaaaaaaa-1111'), 'Named');
        assert.equal(name('bbbbbbbb-2222'), 'AppOnly');
        assert.equal(name('cccccccc-3333'), 'cccccccc');
    });

    it('a peer in both lists is shown once, and a hostile peer name is text', async () => {
        const h = setup({ messages: [msg(1, 'p-web', 'web', 'hi')], peers: [peer('p-web', { display_name: XSS, kind: 'web' })] });
        await h.open();
        assert.equal(h.pane.querySelectorAll('[data-peer="p-web"]').length, 1);
        assert.equal(h.pane.querySelectorAll('img').length, 0);
        assert.ok(h.row('p-web').textContent.includes(XSS));
        assert.equal(h.window.__pwn, undefined);
    });

    it('no peers and no messages keeps the empty state; a failed peer list does not break messages', async () => {
        const h = setup({ messages: [] });
        await h.open();
        assert.match(h.pane.textContent, /Bots you connect can leave each other messages here/);
        assert.ok(![...h.pane.querySelectorAll('h3')].some(x => x.textContent === 'Peers'));
        const g = setup({ routes: [['GET', '/api/v3/mesh/peers', () => ({ status: 500, json: { detail: 'boom' } })]] });
        await g.open();
        assert.equal(g.pane.querySelectorAll('.od-msg').length, 2);
    });
});

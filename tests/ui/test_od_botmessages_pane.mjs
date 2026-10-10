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

    it('mute and unmute POST {muted} with the write credential, through one toggle button', async () => {
        const h = setup({ routes: [['POST', '/api/v3/mesh/peers/p-web/mute', () => ({ json: { ok: true } })]] });
        await h.open();
        const row = h.row('p-web');
        assert.equal(h.btn('Unmute', row), undefined);
        h.btn('Mute', row).click();
        await flushPromises();
        assert.equal(h.btn('Mute', row), undefined);
        h.btn('Unmute', row).click();
        await flushPromises();
        const w = writes(h);
        assert.equal(w[0].url, '/api/v3/mesh/peers/p-web/mute');
        assert.deepEqual(JSON.parse(w[0].body), { muted: true });
        assert.deepEqual(JSON.parse(w[1].body), { muted: false });
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
        assert.ok(h.btn('Mute', row));
    });

    it('rename: no input until Rename is clicked, then validates 1-64 printable characters before any request', async () => {
        const h = setup({ routes: [['PATCH', '/api/v3/mesh/peers/p-web', () => ({ json: { ok: true, display_name: 'Ops bot' } })]] });
        await h.open();
        const row = h.row('p-web');
        assert.equal(row.querySelector('input'), null);
        h.btn('Rename', row).click();
        const input = row.querySelector('input[type=text]');
        assert.ok(input);
        for (const bad of ['', '   ', 'x'.repeat(65), 'bell\u0007', 'tab\there']) {
            input.value = bad;
            h.btn('Save', row).click();
            await flushPromises();
        }
        assert.equal(writes(h).length, 0);
        assert.match(row.textContent, /1 to 64/);
        assert.ok(row.querySelector('input'), 'stays in edit mode after a validation error');
        input.value = 'Ops bot';
        h.btn('Save', row).click();
        await flushPromises();
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].method, 'PATCH');
        assert.deepEqual(JSON.parse(w[0].body), { display_name: 'Ops bot' });
        assert.equal(row.querySelector('input'), null);
        assert.equal(row.querySelector('strong').textContent, 'Ops bot');
        assert.ok(h.btn('Rename', row));
        assert.ok([...h.pane.querySelectorAll('.od-msg strong')].some(x => x.textContent === 'Ops bot'));
        assert.ok([...h.pane.querySelectorAll('option')].some(x => x.textContent === 'Ops bot'));
    });

    it('rename inline: starts with the current name, Enter saves, Escape and Cancel restore without a request', async () => {
        const h = setup({ routes: [['PATCH', '/api/v3/mesh/peers/p-web', () => ({ json: { ok: true, display_name: 'Via enter' } })]] });
        await h.open();
        const row = h.row('p-web');
        const key = (input, k) => input.dispatchEvent(new h.window.KeyboardEvent('keydown', { key: k, bubbles: true }));
        h.btn('Rename', row).click();
        let input = row.querySelector('input[type=text]');
        assert.equal(input.value, 'app-p-web');
        assert.equal(h.btn('Rename', row), undefined);
        input.value = 'Typed';
        key(input, 'Escape');
        assert.equal(row.querySelector('input'), null);
        assert.equal(row.querySelector('strong').textContent, 'app-p-web');
        h.btn('Rename', row).click();
        h.btn('Cancel', row).click();
        assert.equal(row.querySelector('input'), null);
        assert.equal(writes(h).length, 0);
        h.btn('Rename', row).click();
        input = row.querySelector('input[type=text]');
        input.value = 'Via enter';
        key(input, 'Enter');
        await flushPromises();
        assert.equal(writes(h).length, 1);
        assert.equal(row.querySelector('strong').textContent, 'Via enter');
    });

    it('a failed rename keeps the field open and says why', async () => {
        const h = setup({ routes: [['PATCH', '/api/v3/mesh/peers/p-web', () => ({ status: 422, json: { detail: 'taken' } })]] });
        await h.open();
        const row = h.row('p-web');
        h.btn('Rename', row).click();
        row.querySelector('input').value = 'Dup';
        h.btn('Save', row).click();
        await flushPromises();
        assert.ok(row.querySelector('input'));
        assert.match(row.textContent, /taken/);
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
        assert.ok(h.btn('Mute', h.row('p-web')), 'a failed mute leaves the button as it was');
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

    it('a retire button is a quiet danger style, not a plain button', async () => {
        const h = setup();
        await h.open();
        assert.ok(h.btn('Retire', h.row('p-web')).classList.contains('od-peer-retire'));
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

    const ago = ms => new Date(Date.now() - ms).toISOString();
    const named = [
        peer('id-hermes-0001', { app: 'hermes', display_name: '' }),
        peer('id-codex-0002', { app: 'codex' }),
        peer('id-web-0003', { app: 'chat', display_name: 'Browser chat', kind: 'web' })];

    it('message headers show display names, never raw ids, and never an id twice', async () => {
        const m = msg(1, 'id-codex-0002', 'local', 'ship it', { to: 'id-hermes-0001', sent_at: ago(120000) });
        const h = setup({ messages: [m], peers: named });
        await h.open();
        const head = h.pane.querySelector('.od-msg-head');
        assert.match(head.textContent, /codex/);
        assert.match(head.textContent, /hermes/);
        assert.match(head.textContent, /→/);
        assert.ok(!/id-codex|id-hermes/.test(head.textContent));
        assert.equal(head.querySelector('strong').textContent, 'codex');
    });

    it('receiver falls back to everyone for a broadcast and to a short id for an unknown peer', async () => {
        const a = msg(2, 'id-codex-0002', 'local', 'all hands', { to: null });
        const b = msg(1, 'id-codex-0002', 'local', 'who', { to: 'zzzzzzzz-9999-aaaa' });
        const h = setup({ messages: [a, b], peers: named });
        await h.open();
        const heads = [...h.pane.querySelectorAll('.od-msg-head')].map(x => x.textContent);
        assert.match(heads[0], /everyone/);
        assert.match(heads[1], /zzzzzzzz/);
        assert.ok(!/zzzzzzzz-9999/.test(heads[1]));
    });

    it('a sender chip says this computer or web app', async () => {
        const w = msg(2, 'id-web-0003', 'web', 'hi', { to: 'id-codex-0002' });
        const l = msg(1, 'id-codex-0002', 'local', 'yo', { to: 'id-web-0003' });
        const h = setup({ messages: [w, l], peers: named });
        await h.open();
        const [web, loc] = [...h.pane.querySelectorAll('.od-msg-head')];
        assert.match(web.textContent, /Browser chat/);
        assert.match(web.querySelector('.od-kind').textContent, /web app/);
        assert.match(loc.querySelector('.od-kind').textContent, /this computer/);
    });

    it('sent time is relative, with the full local time in the title', async () => {
        const cases = [[5000, 'just now'], [120000, '2 min ago'], [3 * 3600000, '3 hours ago'], [2 * 86400000, '2 days ago']];
        for (const [ms, text] of cases) {
            const iso = ago(ms);
            const h = setup({ messages: [msg(1, 'id-codex-0002', 'local', 'x', { sent_at: iso })], peers: named });
            await h.open();
            const t = h.pane.querySelector('.od-msg time');
            assert.equal(t.textContent, text);
            assert.equal(t.getAttribute('title'), new Date(iso).toLocaleString());
            assert.ok(!h.pane.querySelector('.od-msg').textContent.includes(iso));
        }
    });

    it('long fractional timestamps with an offset still parse, and a junk time degrades to its text', async () => {
        const base = new Date(Date.now() - 120000).toISOString().replace('Z', '');
        const h = setup({ messages: [msg(2, 'id-codex-0002', 'local', 'x', { sent_at: base + '948+00:00' }),
            msg(1, 'id-codex-0002', 'local', 'y', { sent_at: 'not a time' })], peers: named });
        await h.open();
        const times = [...h.pane.querySelectorAll('.od-msg time')].map(t => t.textContent);
        assert.equal(times[0], '2 min ago');
        assert.equal(times[1], 'not a time');
    });

    it('the body is the main text and the sender name is bold', async () => {
        const h = setup({ messages: [msg(1, 'id-codex-0002', 'local', 'the body')], peers: named });
        await h.open();
        const card = h.pane.querySelector('.od-msg');
        assert.equal(card.querySelector('.od-msg-body').textContent, 'the body');
        assert.equal(card.querySelector('.od-msg-head strong').textContent, 'codex');
    });

    it('the untrusted-text notice stays at the top, above the filter and messages', async () => {
        const h = setup({ peers: named });
        await h.open();
        const kids = [...h.pane.children];
        assert.match(kids[0].textContent, /untrusted text/);
        assert.ok(kids[0].compareDocumentPosition(h.pane.querySelector('.od-msg')) & 4);
    });

    it('each peer has exactly one of Mute or Unmute, matching what the API says', async () => {
        const h = setup({ messages: [], peers: [peer('p-a', { app: 'a' }), peer('p-b', { app: 'b', muted: true })] });
        await h.open();
        const a = h.row('p-a'); const b = h.row('p-b');
        assert.ok(h.btn('Mute', a)); assert.equal(h.btn('Unmute', a), undefined);
        assert.ok(h.btn('Unmute', b)); assert.equal(h.btn('Mute', b), undefined);
        assert.match(b.textContent, /muted/);
        assert.ok(!/muted/.test(a.textContent.replace(/Mute/g, '')));
    });

    it('toggling mute updates the muted tag and label', async () => {
        const h = setup({ messages: [], peers: [peer('p-a', { app: 'a' })],
            routes: [['POST', '/api/v3/mesh/peers/p-a/mute', () => ({ json: { ok: true } })]] });
        await h.open();
        const a = h.row('p-a');
        h.btn('Mute', a).click();
        await flushPromises();
        assert.ok(a.querySelector('.od-peer-muted'));
        h.btn('Unmute', a).click();
        await flushPromises();
        assert.equal(a.querySelector('.od-peer-muted'), null);
    });

    it('a peer card shows name, kind chip, a short id and last seen', async () => {
        const h = setup({ messages: [], peers: [peer('0123456789abcdef-long', { app: 'a', kind: 'web', last_seen: ago(180000) })] });
        await h.open();
        const r = h.row('0123456789abcdef-long');
        assert.match(r.querySelector('.od-kind').textContent, /web app/);
        assert.match(r.textContent, /0123456789ab/);
        assert.ok(!r.textContent.includes('0123456789abcdef-long'));
        const seen = r.querySelector('time');
        assert.match(r.textContent, /Last seen 3 min ago/);
        assert.ok(seen.getAttribute('title'));
    });

    it('retire confirms with the peer name and a consequence, then removes the card', async () => {
        const h = setup({ messages: [], peers: [peer('p-a', { app: 'a' })],
            routes: [['DELETE', '/api/v3/mesh/peers/p-a', () => { h.state.peers = []; return { json: { ok: true } }; }]] });
        await h.open();
        h.btn('Retire', h.row('p-a')).click();
        await flushPromises();
        assert.equal(h.confirms.length, 1);
        assert.equal(h.confirms[0].confirmLabel, 'Retire');
        assert.equal(h.row('p-a'), null);
    });

    it('the filter is a labelled, design-system select', async () => {
        const h = setup({ peers: named });
        await h.open();
        const sel = h.pane.querySelector('select');
        assert.ok(sel.classList.contains('od-select'));
        const label = h.pane.querySelector(`label[for="${sel.id}"]`);
        assert.ok(sel.id);
        assert.equal(label.textContent, 'Show messages from');
        assert.equal(sel.options[0].textContent, 'All peers');
        assert.ok([...sel.options].some(o => o.textContent === 'codex'));
    });

    it('Refresh keeps the chosen filter and picks up new names and mute state', async () => {
        const h = setup({ peers: [peer('p-a', { app: 'a' })] });
        await h.open();
        const sel = h.pane.querySelector('select');
        sel.value = 'p-a';
        sel.dispatchEvent(new h.window.Event('change', { bubbles: true }));
        await flushPromises();
        h.state.peers = [peer('p-a', { app: 'a', display_name: 'Fresh', muted: true })];
        h.btn('Refresh').click();
        await flushPromises();
        assert.equal(h.pane.querySelector('select').value, 'p-a');
        assert.equal(h.row('p-a').querySelector('strong').textContent, 'Fresh');
        assert.ok(h.btn('Unmute', h.row('p-a')));
    });
});

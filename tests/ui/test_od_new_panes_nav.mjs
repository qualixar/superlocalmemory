/**
 * Sidebar entries and activation for the Documents & Images pane.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');
const shellSource = readFileSync(join(UI, 'js/od-shell.js'), 'utf8');
const html = readFileSync(join(UI, 'index.html'), 'utf8');

function shell(paneIds) {
    const panes = paneIds.map(id => `<div id="${id}" class="tab-pane"></div>`).join('');
    const dom = new JSDOM(`<!doctype html><html><body>
      <div class="scrim" id="scrim"></div><button id="menuBtn"></button><aside id="sidebar"></aside>
      <main id="main-content"><div id="dashboard-pane" class="tab-pane active"></div>${panes}</main>
      <span id="topbar-crumb"></span><span id="topbar-heading"></span>
      <div class="topbar"><button data-theme-icon></button></div></body></html>`,
        { runScripts: 'dangerously', url: 'http://localhost:8765/' });
    const w = dom.window;
    w.scrollTo = () => {}; w.HTMLElement.prototype.scrollTo = () => {}; w.HTMLElement.prototype.scrollIntoView = () => {};
    w.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    const s = w.document.createElement('script');
    s.textContent = shellSource;
    w.document.head.appendChild(s);
    return w;
}

const ENTRIES = [
    { pane: 'media-pane', label: 'Documents & Images', crumb: 'Memory', render: 'odRenderMedia' },
];

describe('new panes in the sidebar', () => {
    for (const e of ENTRIES) {
        it(`${e.label}: entry with a "new" tag, and activating it renders the pane`, () => {
            const w = shell(ENTRIES.map(x => x.pane));
            let rendered = 0;
            w[e.render] = () => { rendered += 1; };
            w.slmShell({ active: 'dashboard-pane' });
            const link = w.document.querySelector(`[data-tab="${e.pane}"]`);
            assert.ok(link, 'sidebar entry exists');
            assert.equal(link.querySelector('.tag').textContent, 'new');
            assert.ok(link.textContent.includes(e.label));
            link.click();
            assert.equal(rendered, 1);
            assert.equal(w.document.getElementById('topbar-crumb').textContent, e.crumb);
        });

        it(`${e.label}: index.html has the pane container and hidden tab button`, () => {
            assert.match(html, new RegExp(`id="${e.pane}"`));
            assert.match(html, new RegExp(`id="${e.pane.replace('-pane', '-tab')}"`));
        });
    }
});

/**
 * Shared helpers for the dashboard tests of images/documents and bot messages:
 * a URL-routed fetch stub that keeps core.js's write-credential patch around it,
 * a request log, and a controllable poll timer.
 */
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

export const TOKEN = 'tok-test-123';

export function features(media = {}) {
    return { media: Object.assign({ enabled: false, requested: false, env_state: 'not_installed',
        progress: 0, step: '', restart_required: false, precheck: {} }, media),
        mesh: { apps_with_mesh: 0 } };
}

/** routes: [[METHOD, matcher(string|RegExp), (call) => ({status, json}) ]] */
export function makeEnv(routes, { ids = ['pane'], modules = [] } = {}) {
    const calls = [];
    const fetchFn = async function (url, init) {
        init = init || {};
        const method = String(init.method || 'GET').toUpperCase();
        const headers = new Map();
        if (init.headers) new Headers(init.headers).forEach((v, k) => headers.set(k.toLowerCase(), v));
        if (url === '/internal/token') {
            return { ok: true, status: 200, json: async () => ({ token: TOKEN }) };
        }
        const call = { method, url: String(url), headers, body: init.body ? String(init.body) : '' };
        calls.push(call);
        const hit = routes.find(([m, p]) => m === method &&
            (typeof p === 'string' ? call.url === p || call.url.startsWith(p + '?') : p.test(call.url)));
        const res = hit ? hit[2](call) : { status: 404, json: { detail: 'Not found' } };
        const status = res.status || 200;
        return { ok: status >= 200 && status < 300, status, json: async () => res.json };
    };
    const h = buildHarness(ids, fetchFn);
    h.window.Headers = globalThis.Headers;
    h.window.confirmDestructive = function (opts) { h.confirms.push(opts); return Promise.resolve(h.confirmAnswer); };
    h.confirms = [];
    h.confirmAnswer = true;
    h.calls = calls;
    h.timers = [];
    const realSet = h.window.setTimeout.bind(h.window);
    h.window.setTimeout = function (fn, ms) {
        if (ms === 2000) { h.timers.push(fn); return h.timers.length; }
        return realSet(fn, ms);
    };
    h.tick = async function () {            // run the queued 2 s poll callbacks once
        const run = h.timers.splice(0);
        run.forEach(fn => fn());
        await flushPromises();
    };
    modules.forEach(m => evalModule(h.window, m));
    return h;
}

export const writes = h => h.calls.filter(c => c.method !== 'GET');
export { flushPromises };

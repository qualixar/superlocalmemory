/* od-welcome.js — the first-run Welcome card at the top of the Dashboard pane.
 *
 * Shown only on a new store (fewer than 20 memories) that the person has not dismissed.
 * Three rows, each with a status and at most one button: the memory works, which AI
 * agents are connected (plus a prompt to paste into another agent), and the optional
 * pictures and documents. The turn-on flow itself belongs to the Documents & Images
 * pane (od-features.js); this card only opens it. Text memory is never offered the
 * picture model.
 *
 * Reads (all GET, nothing changes):
 *   /api/stats         daemon answers + total memories
 *   /api/v3/local-apps connected agents (configured_hosts, live apps)
 *   /api/v3/features   pictures and documents state
 *
 * Exposes window.odWelcome { mount(host), PROMPT, THRESHOLD, KEY }.
 * Every server string is set with textContent, never parsed as HTML.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  var KEY = 'slm.welcome.dismissed';
  var THRESHOLD = 20;
  var MEDIA_PANE = 'media-pane';
  var PROMPT = 'Install SuperLocalMemory on this computer for me. Follow '
    + 'https://github.com/qualixar/superlocalmemory/blob/main/SETUP_FOR_AGENTS.md step by step and show me '
    + 'what each step printed. Ask me only one question, before turning on anything optional: whether I also '
    + 'want to find my pictures, screenshots and PDFs by describing them. Connect SuperLocalMemory to the '
    + 'agent I am using. When you finish, run slm doctor and tell me in plain words what works.';

  var _closedThisVisit = false;   // Done must work even when storage is blocked

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = String(text);
    return e;
  }

  function button(label, cls, onClick) {
    var b = el('button', cls || 'btn sm', label);
    b.type = 'button';
    b.addEventListener('click', function () { onClick(b); });
    return b;
  }

  // Resolves {ok, status, data}; a dropped connection is status 0, never a throw.
  function api(url) {
    return fetch(url).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { ok: !!r.ok, status: r.status, data: data || {} };
      });
    }, function () { return { ok: false, status: 0, data: {} }; });
  }

  // -- dismissal (storage can be missing or throw) -----------------------------

  function dismissed() {
    if (_closedThisVisit) return true;
    try { return window.localStorage.getItem(KEY) === '1'; } catch (e) { return false; }
  }

  function remember() {
    _closedThisVisit = true;
    try { window.localStorage.setItem(KEY, '1'); } catch (e) { /* closed for this visit only */ }
  }

  // -- row building ---------------------------------------------------------------

  function status(kind, text) {
    return el('span', 'badge ' + kind + ' od-welcome-status', text);
  }

  function row(name, title, badge) {
    var r = el('div', 'od-welcome-row');
    r.setAttribute('data-row', name);
    var head = el('div', 'od-welcome-row-head');
    head.appendChild(el('h4', null, title));
    head.appendChild(badge);
    r.appendChild(head);
    return r;
  }

  function line(text) { return el('p', 'muted', text); }

  function goToMedia() {
    if (typeof window.slmNavigate === 'function') window.slmNavigate(MEDIA_PANE);
  }

  // -- row 1: the daemon answers -----------------------------------------------------

  function memoryProblem(res) {
    if (res.status === 0) return 'Could not reach the local service. It may be stopped.';
    if (res.status === 404) return 'This version of SuperLocalMemory does not report its status.';
    return 'The memory service did not answer as expected (error ' + res.status + ').';
  }

  function memoryRow(res) {
    if (res.ok) {
      var ok = row('memory', 'Your memory is working', status('ok', '✓ Working'));
      ok.appendChild(line('SuperLocalMemory is running on this computer.'));
      return ok;
    }
    var bad = row('memory', 'Your memory is working', status('warn', 'Not answering'));
    bad.appendChild(line(memoryProblem(res)));
    return bad;
  }

  // -- row 2: agents and the paste-in prompt ----------------------------------------------

  function agentNames(data) {
    var seen = {}, out = [];
    (data.configured_hosts || []).forEach(function (h) {
      seen[h.id] = true;
      out.push(String(h.name || h.id) + (Number(h.live) > 0 ? ' (open now)' : ''));
    });
    (data.apps || []).forEach(function (a) {
      if (a && a.host && !seen[a.host]) {
        seen[a.host] = true;
        out.push(String(a.name || a.host) + ' (open now)');
      }
    });
    return out;
  }

  function selectText(node) {
    var sel = window.getSelection && window.getSelection();
    if (!sel || !document.createRange) return;
    var range = document.createRange();
    range.selectNodeContents(node);
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function copyPrompt(box, note) {
    function manual() {
      selectText(box);
      note.textContent = 'Press Ctrl+C (⌘C on a Mac) to copy.';
    }
    var clip = window.navigator && window.navigator.clipboard;
    if (!clip || typeof clip.writeText !== 'function') { manual(); return; }
    try {
      clip.writeText(PROMPT).then(function () { note.textContent = 'Copied.'; }, manual);
    } catch (e) { manual(); }
  }

  function promptBlock(r) {
    r.appendChild(line('To connect another agent, paste this into it:'));
    var box = el('pre', 'od-welcome-prompt', PROMPT);
    var note = el('p', 'muted od-welcome-note');
    note.setAttribute('role', 'status');
    r.appendChild(box);
    var actions = el('div', 'od-welcome-actions');
    actions.appendChild(button('Copy', 'btn sm primary', function () { copyPrompt(box, note); }));
    actions.appendChild(note);
    r.appendChild(actions);
  }

  function agentsRow(res) {
    var names = res.ok ? agentNames(res.data) : [];
    var badge = names.length ? status('ok', '✓ ' + names.length + ' connected') : status('neutral', 'None yet');
    var r = row('agents', 'Your AI agents', badge);
    if (!res.ok) {
      r.appendChild(line(res.status === 0 ? 'Could not read your connected agents: the local service did not answer.'
        : res.status === 404 ? 'The list of connected agents is not available in this build.'
          : 'Could not read your connected agents (error ' + res.status + ').'));
    } else if (!names.length) {
      r.appendChild(line('No agent is connected yet.'));
    } else {
      var list = el('ul', 'od-welcome-list');
      names.forEach(function (n) { list.appendChild(el('li', null, n)); });
      r.appendChild(list);
    }
    promptBlock(r);
    return r;
  }

  // -- row 3: pictures and documents ---------------------------------------------------------

  var MEDIA_SENTENCE = 'Find your pictures, screenshots and PDFs by describing them. '
    + 'Adds a second local model: about 1.5 GB. It sets up by itself on a computer with 16 GB of memory or more.';

  function mediaProblem(res) {
    if (res.status === 404) return 'Pictures and documents are not available in this build.';
    if (res.status === 0) return 'Could not read the pictures and documents status: the local service did not answer.';
    return 'Could not read the pictures and documents status (error ' + res.status + ').';
  }

  function mediaIsOn(m) {
    if (window.odFeatures && typeof window.odFeatures.isOn === 'function') return window.odFeatures.isOn(m);
    return !!(m.enabled && m.env_state === 'ready' && !m.restart_required);
  }

  function mediaOff(r, m) {
    r.appendChild(line(MEDIA_SENTENCE));
    var blocked = m.env_state === 'unsupported';
    if (m.ram_warning) r.appendChild(el('p', null, m.ram_warning));  // a small computer is warned, never refused
    else if (m.env_state === 'unsupported' && m.step) r.appendChild(el('p', null, m.step));
    var go = button('Set up pictures and documents', 'btn sm primary', goToMedia);
    go.disabled = blocked;
    r.appendChild(go);
  }

  function mediaInProgress(r, m) {
    var text = m.enabled && m.env_state === 'installing'
      ? 'Pictures and documents are being set up. This can take several minutes.'
      : m.enabled && m.env_state === 'failed' ? (m.step || m.error || 'Setup did not finish.')
        : 'Almost done. Open Documents & Images to finish.';
    r.appendChild(line(text));
    r.appendChild(button('Open Documents & Images', 'btn sm primary', goToMedia));
  }

  function mediaRow(res) {
    var title = 'Pictures and documents (optional)';
    if (!res.ok || !res.data.media) {
      var bad = row('media', title, status('neutral', 'Unavailable'));
      bad.appendChild(line(mediaProblem(res.ok ? { status: 404 } : res)));
      return bad;
    }
    var m = res.data.media;
    if (mediaIsOn(m)) return row('media', title, status('ok', '✓ On'));
    var r = row('media', title, status('neutral', m.enabled ? 'Setting up' : 'Off'));
    if (m.enabled) mediaInProgress(r, m); else mediaOff(r, m);
    return r;
  }

  // -- card -------------------------------------------------------------------------------------

  function draw(host, token, stats, agents, features) {
    var card = el('div', 'card card-pad od-welcome');
    card.appendChild(el('h3', null, 'Welcome to SuperLocalMemory'));
    card.appendChild(memoryRow(stats));
    card.appendChild(agentsRow(agents));
    card.appendChild(mediaRow(features));
    var foot = el('div', 'od-welcome-foot');
    foot.appendChild(button('Done', 'btn sm primary', function () {
      host.__odWelcomeToken = token + 1;
      remember();
      host.textContent = '';
    }));
    card.appendChild(foot);
    host.textContent = '';
    host.appendChild(card);
  }

  function memoryCount(res) {
    var ov = res.data && res.data.overview;
    return Number(ov && ov.total_memories) || 0;
  }

  // Resolves once drawn (or once it is clear nothing is to be drawn). A card the person
  // dismissed, or a store past the threshold, costs no further request.
  function mount(host) {
    host.textContent = '';
    var token = (host.__odWelcomeToken || 0) + 1;
    host.__odWelcomeToken = token;
    if (dismissed()) return Promise.resolve();
    return api('/api/stats').then(function (stats) {
      if (host.__odWelcomeToken !== token) return null;
      if (stats.ok && memoryCount(stats) >= THRESHOLD) return null;
      return Promise.all([api('/api/v3/local-apps'), api('/api/v3/features')]).then(function (both) {
        if (host.__odWelcomeToken !== token || dismissed()) return;
        draw(host, token, stats, both[0], both[1]);
      });
    });
  }

  function autoMount() {
    var host = document.getElementById('od-welcome');
    if (host) mount(host);
  }

  window.odWelcome = { mount: mount, PROMPT: PROMPT, THRESHOLD: THRESHOLD, KEY: KEY };
  autoMount();
})();

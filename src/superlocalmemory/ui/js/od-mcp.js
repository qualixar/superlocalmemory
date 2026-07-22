// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// od-mcp.js — MCP & Integrations dashboard pane for SuperLocalMemory.
//
// Exposes: window.odRenderMcp(pane)
// Called by the od() shell after it clears the target pane.
//
// Endpoint: GET /api/v3/mcp/profiles
//   {
//     current: "core",
//     profiles: {
//       core:  { count: 14, tools: [...], description: "..." },
//       code:  { count: 21, tools: [...], description: "..." },
//       full:  { count: 39, tools: [...], description: "..." },
//       power: { count: 51, tools: [...], description: "..." },
//       mesh:  { count:  8, tools: [...], description: "..." },
//     },
//     aliases: { "code21": "code", ... },
//     total_tools: <int>
//   }
//
// CSP-safe: no eval(), no on*= attributes, no innerHTML for API data.
//   All user-controlled strings → textContent (XSS-safe).
//   Only static SVG strings use innerHTML (auditable, no API data).
// Immutable style: EL() always creates new nodes.

/* global window, document, slmFetch */

(function () {
  'use strict';

  // =========================================================================
  // Tiny DOM factory — mirrors od-brain.js EL() pattern
  // =========================================================================

  function EL(tag, props, kids) {
    var e = document.createElement(tag);
    var p = props || {};
    Object.keys(p).forEach(function (k) {
      if (k === 'className') { e.className = String(p[k]); }
      else if (k === 'text') { e.textContent = p[k] == null ? '' : String(p[k]); }
      else { e.setAttribute(k, String(p[k])); }
    });
    (kids || []).forEach(function (c) { if (c != null) e.appendChild(c); });
    return e;
  }

  // Inject a child node and return parent (fluent helper)
  function app(parent, child) { parent.appendChild(child); return parent; }

  // CSS injection — scoped styles for this pane only
  var _stylesId = 'od-mcp-styles';
  function injectStyles() {
    if (document.getElementById(_stylesId)) return;
    var s = document.createElement('style');
    s.id = _stylesId;
    s.textContent = [
      '.od-mcp-intro{font-size:13px;color:var(--fg-2);margin:0 0 16px;line-height:1.55}',
      '.od-mcp-chip{display:inline-flex;align-items:center;',
      'font-size:11.5px;font-family:var(--mono,monospace);',
      'background:var(--card-2);border:1px solid var(--border);',
      'border-radius:6px;padding:2px 8px;margin:3px;color:var(--fg-1)}',
      '.od-mcp-chip-wrap{display:flex;flex-wrap:wrap;margin:-3px;margin-top:8px}',
      '.od-mcp-profile-row{display:flex;align-items:flex-start;gap:12px;',
      'padding:12px 0;border-bottom:1px solid var(--border)}',
      '.od-mcp-profile-row:last-child{border-bottom:0}',
      '.od-mcp-profile-name{font-weight:650;font-size:14px;min-width:58px}',
      '.od-mcp-profile-desc{font-size:12.5px;color:var(--fg-2);',
      'flex:1;line-height:1.45;padding-top:1px}',
      '.od-mcp-details{margin-top:6px}',
      '.od-mcp-details summary{font-size:11.5px;color:var(--violet);',
      'cursor:pointer;list-style:none;display:inline-flex;align-items:center;gap:4px}',
      '.od-mcp-details summary::-webkit-details-marker{display:none}',
      '.od-mcp-code{font-family:var(--mono,monospace);font-size:12px;',
      'background:var(--card-2);border:1px solid var(--border);border-radius:8px;',
      'padding:10px 14px;margin:8px 0;overflow-x:auto;white-space:pre;',
      'color:var(--fg-1)}',
      '.od-mcp-step{display:flex;gap:10px;margin:8px 0;align-items:flex-start}',
      '.od-mcp-step-num{width:22px;height:22px;border-radius:50%;',
      'background:var(--violet);color:#fff;font-size:11px;font-weight:700;',
      'display:grid;place-items:center;flex-shrink:0;margin-top:1px}',
      '.od-mcp-step-body{font-size:13px;color:var(--fg-1);line-height:1.5;flex:1}',
      '.od-mcp-highlight{color:var(--violet);font-weight:600}',
      '.od-mcp-warn{font-size:12px;color:var(--warn);margin-top:8px;',
      'display:flex;align-items:flex-start;gap:6px}',
      '.od-mcp-err{padding:20px;text-align:center;color:var(--fg-3);font-size:13px}',
      '.od-mcp-loading{padding:24px;text-align:center;color:var(--fg-3);font-size:13px}',
    ].join('');
    document.head.appendChild(s);
  }

  // =========================================================================
  // Fetch
  // =========================================================================

  function apiFetch(path) {
    var fetcher = typeof slmFetch === 'function' ? slmFetch : fetch;
    return fetcher(path, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      });
  }

  // =========================================================================
  // Helpers
  // =========================================================================

  // Map profile name → badge class
  var _BADGE_CLASS = {
    core: 'badge',
    code: 'badge ok',
    full: 'badge warn',
    power: 'badge danger',
    mesh: 'badge',
  };

  function profileBadge(name, count) {
    var b = EL('span', { text: name + ' · ' + count + ' tools' });
    b.className = _BADGE_CLASS[name] || 'badge';
    return b;
  }

  function toolChips(tools) {
    var wrap = EL('div');
    wrap.className = 'od-mcp-chip-wrap';
    (tools || []).forEach(function (t) {
      var chip = EL('span', { text: t });
      chip.className = 'od-mcp-chip';
      wrap.appendChild(chip);
    });
    return wrap;
  }

  // Static "plug" SVG icon (no API data, safe for innerHTML)
  var PLUG_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" ' +
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 22V12M5 12V2l3 3 4-4 4 4 3-3v10"/>' +
    '<path d="M5 12a7 7 0 0 0 14 0"/></svg>';

  var SETTINGS_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" ' +
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="3"/>' +
    '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06' +
    'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09' +
    'A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06' +
    'A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09' +
    'A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06' +
    'A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09' +
    'a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06' +
    'A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09' +
    'a1.65 1.65 0 0 0-1.51 1z"/></svg>';

  // =========================================================================
  // Build: intro paragraph
  // =========================================================================

  function buildIntro() {
    var p = EL('p');
    p.className = 'od-mcp-intro';
    p.textContent =
      'MCP tools are what your AI agents (Claude Code, Codex, Cursor, Windsurf, …) ' +
      'can call. Your active profile controls how many tools are exposed over the ' +
      'MCP connection. Fewer tools keeps context lean; more tools gives agents ' +
      'deeper capabilities.';
    return p;
  }

  // =========================================================================
  // Build: current profile card
  // =========================================================================

  function buildCurrentCard(data) {
    var current = data.current || 'core';
    var info = (data.profiles || {})[current] || {};
    var count = info.count || 0;
    var tools = info.tools || [];
    var desc = info.description || '';

    var wrap = EL('div');
    wrap.className = 'card';
    wrap.style.marginBottom = '16px';

    // card-head
    var head = EL('div');
    head.className = 'card-head';
    var ic = EL('span');
    ic.style.cssText = 'color:var(--violet);width:18px;height:18px;display:flex;flex-shrink:0';
    ic.innerHTML = PLUG_SVG;  // static SVG, no API data
    head.appendChild(ic);
    head.appendChild(EL('h3', { text: 'Active MCP profile' }));
    head.appendChild(profileBadge(current, count));
    wrap.appendChild(head);

    // card-pad
    var body = EL('div');
    body.className = 'card-pad';

    var descRow = EL('p', { text: desc });
    descRow.style.cssText = 'font-size:13px;color:var(--fg-2);margin:0 0 6px';
    body.appendChild(descRow);

    var envNote = EL('p');
    envNote.className = 'muted';
    envNote.style.fontSize = '12px';
    // Build using textContent + appendChild to keep CSP-safe
    var envLabel = document.createTextNode('Set via ');
    var envCode = EL('code', { text: 'SLM_MCP_PROFILE=' + current });
    envCode.style.cssText = 'font-family:var(--mono,monospace);' +
      'background:var(--card-2);padding:1px 5px;border-radius:4px;font-size:11.5px';
    var envSuffix = document.createTextNode(' in your IDE\'s MCP config.');
    envNote.appendChild(envLabel);
    envNote.appendChild(envCode);
    envNote.appendChild(envSuffix);
    body.appendChild(envNote);

    if (tools.length > 0) {
      var toolLabel = EL('div', { text: count + ' tools in this profile:' });
      toolLabel.style.cssText = 'font-size:11.5px;color:var(--fg-3);margin-top:10px;margin-bottom:2px';
      body.appendChild(toolLabel);
      body.appendChild(toolChips(tools));
    }

    wrap.appendChild(body);
    return wrap;
  }

  // =========================================================================
  // Build: available profiles comparison card
  // =========================================================================

  var _PROFILE_ORDER = ['core', 'code', 'full', 'power', 'mesh'];

  function buildProfilesCard(data) {
    var profiles = data.profiles || {};
    var current = data.current || 'core';

    var wrap = EL('div');
    wrap.className = 'card';
    wrap.style.marginBottom = '16px';

    // card-head
    var head = EL('div');
    head.className = 'card-head';
    var ic = EL('span');
    ic.style.cssText = 'color:var(--violet);width:18px;height:18px;display:flex;flex-shrink:0';
    ic.innerHTML = SETTINGS_SVG;  // static SVG, no API data
    head.appendChild(ic);
    head.appendChild(EL('h3', { text: 'Available profiles' }));
    var totalBadge = EL('span', { text: String(data.total_tools || 0) + ' total tools' });
    totalBadge.className = 'badge';
    head.appendChild(totalBadge);
    wrap.appendChild(head);

    // card-pad: one row per profile
    var body = EL('div');
    body.className = 'card-pad';

    _PROFILE_ORDER.forEach(function (name) {
      var info = profiles[name];
      if (!info) return;

      var row = EL('div');
      row.className = 'od-mcp-profile-row';

      // Name + active indicator
      var nameWrap = EL('div');
      nameWrap.style.minWidth = '76px';
      var nameEl = EL('span', { text: name });
      nameEl.className = 'od-mcp-profile-name';
      nameWrap.appendChild(nameEl);
      if (name === current) {
        var activeTag = EL('span', { text: 'active' });
        activeTag.className = 'badge ok';
        activeTag.style.cssText = 'font-size:10px;margin-left:4px;vertical-align:middle';
        nameWrap.appendChild(activeTag);
      }
      row.appendChild(nameWrap);

      // Description + collapsible tool list
      var rightCol = EL('div');
      rightCol.style.flex = '1';

      var descEl = EL('div', { text: info.description || '' });
      descEl.className = 'od-mcp-profile-desc';
      rightCol.appendChild(descEl);

      // Count badge
      var countBadge = EL('span', { text: String(info.count) + ' tools' });
      countBadge.className = 'badge';
      countBadge.style.cssText = 'font-size:10.5px;margin:4px 0 2px;display:inline-flex';
      rightCol.appendChild(countBadge);

      // Expandable tool list — native <details> (CSP-safe, no JS needed)
      if (info.tools && info.tools.length > 0) {
        var det = document.createElement('details');
        det.className = 'od-mcp-details';
        var summ = document.createElement('summary');
        // Arrow text nodes: collapsed = ▶, expanded via CSS :open won't work here,
        // use a text node that we update via the toggle event (CSP-safe)
        var summText = document.createTextNode('Show tools ▾');
        summ.appendChild(summText);
        det.appendChild(summ);
        det.appendChild(toolChips(info.tools));
        det.addEventListener('toggle', function () {
          summText.nodeValue = det.open ? 'Hide tools ▴' : 'Show tools ▾';
        });
        rightCol.appendChild(det);
      }

      row.appendChild(rightCol);
      body.appendChild(row);
    });

    wrap.appendChild(body);
    return wrap;
  }

  // =========================================================================
  // Build: how-to-change card
  // =========================================================================

  function buildHowToCard() {
    var wrap = EL('div');
    wrap.className = 'card';
    wrap.style.marginBottom = '16px';

    var head = EL('div');
    head.className = 'card-head';
    head.appendChild(EL('h3', { text: 'How to change your profile' }));
    wrap.appendChild(head);

    var body = EL('div');
    body.className = 'card-pad';

    // Step 1
    var step1 = EL('div');
    step1.className = 'od-mcp-step';
    var num1 = EL('div', { text: '1' });
    num1.className = 'od-mcp-step-num';
    var body1 = EL('div');
    body1.className = 'od-mcp-step-body';
    var b1text1 = document.createTextNode('Open your IDE\'s MCP server config (usually ');
    var b1code = EL('code', { text: '.mcp.json' });
    b1code.style.cssText = 'font-family:var(--mono,monospace);' +
      'background:var(--card-2);padding:1px 5px;border-radius:4px;font-size:11.5px';
    var b1text2 = document.createTextNode(' or the MCP settings panel).');
    body1.appendChild(b1text1);
    body1.appendChild(b1code);
    body1.appendChild(b1text2);
    step1.appendChild(num1);
    step1.appendChild(body1);
    body.appendChild(step1);

    // Step 2 with example snippets for all profiles
    var step2 = EL('div');
    step2.className = 'od-mcp-step';
    var num2 = EL('div', { text: '2' });
    num2.className = 'od-mcp-step-num';
    var body2 = EL('div');
    body2.className = 'od-mcp-step-body';
    body2.appendChild(document.createTextNode(
      'Add or update the environment variable for the superlocalmemory server entry:'
    ));

    var codeBlock = EL('div');
    codeBlock.className = 'od-mcp-code';
    // Use textContent — no API data, just static example strings
    codeBlock.textContent =
      '"env": {\n' +
      '  "SLM_MCP_PROFILE": "core"    // 14 tools — minimal\n' +
      '  // "SLM_MCP_PROFILE": "code"  // 24 tools — + code graph + loops\n' +
      '  // "SLM_MCP_PROFILE": "full"  // 42 tools — + mesh\n' +
      '  // "SLM_MCP_PROFILE": "power" // 54 tools — + governance\n' +
      '  // "SLM_MCP_PROFILE": "mesh"  //  8 tools — mesh only\n' +
      '}';
    body2.appendChild(codeBlock);
    step2.appendChild(num2);
    step2.appendChild(body2);
    body.appendChild(step2);

    // Step 3
    var step3 = EL('div');
    step3.className = 'od-mcp-step';
    var num3 = EL('div', { text: '3' });
    num3.className = 'od-mcp-step-num';
    var body3 = EL('div', {
      text: 'Restart the IDE\'s MCP client (close and reopen the project, ' +
            'or use the IDE\'s "Reload MCP" command). The daemon itself does not ' +
            'need to restart.'
    });
    body3.className = 'od-mcp-step-body';
    step3.appendChild(num3);
    step3.appendChild(body3);
    body.appendChild(step3);

    // Honest caveat
    var warn = EL('div');
    warn.className = 'od-mcp-warn';
    var warnIcon = EL('span', { text: '⚠️' });
    warnIcon.setAttribute('aria-hidden', 'true');
    var warnText = EL('span');
    warnText.textContent =
      'Profile selection is per-IDE MCP config, not a live daemon toggle. ' +
      'Different IDEs (Claude Code, Cursor, Windsurf) each have their own ' +
      'MCP config and can use different profiles simultaneously.';
    warn.appendChild(warnIcon);
    warn.appendChild(warnText);
    body.appendChild(warn);

    // Docs link
    var docsRow = EL('p');
    docsRow.style.cssText = 'font-size:12.5px;margin-top:10px;color:var(--fg-2)';
    var docsText = document.createTextNode('Full documentation: ');
    var docsLink = EL('a', {
      href: 'https://qualixar.com/docs/slm/mcp-profiles',
      target: '_blank',
      rel: 'noopener noreferrer',
      text: 'qualixar.com/docs/slm/mcp-profiles'
    });
    docsLink.style.color = 'var(--violet)';
    docsRow.appendChild(docsText);
    docsRow.appendChild(docsLink);
    body.appendChild(docsRow);

    wrap.appendChild(body);
    return wrap;
  }

  // =========================================================================
  // Loading and error state helpers (self-contained — no containerId needed)
  // =========================================================================

  function buildLoading() {
    var d = EL('div');
    d.className = 'od-mcp-loading';
    d.textContent = 'Loading MCP profile data…';
    return d;
  }

  function buildError(message, onRetry) {
    var d = EL('div');
    d.className = 'od-mcp-err';
    var icon = EL('div', { text: '⚠️' });
    icon.style.fontSize = '24px';
    d.appendChild(icon);
    var msg = EL('p', { text: message || 'Could not load MCP profiles.' });
    msg.style.margin = '8px 0';
    d.appendChild(msg);
    if (typeof onRetry === 'function') {
      var btn = EL('button', { type: 'button', text: 'Retry' });
      btn.className = 'btn ghost sm';
      btn.addEventListener('click', onRetry);
      d.appendChild(btn);
    }
    return d;
  }

  // =========================================================================
  // Main render
  // =========================================================================

  window.odRenderMcp = function (pane) {
    if (!pane) return;
    injectStyles();

    // Show loading state immediately
    pane.textContent = '';
    pane.appendChild(buildLoading());

    apiFetch('/api/v3/mcp/profiles')
      .then(function (data) {
        pane.textContent = '';

        // Root wrapper
        var root = EL('div', { id: 'od-mcp-root' });
        root.style.cssText = 'padding:26px;max-width:860px';

        // Page heading
        var pageHead = EL('div');
        pageHead.className = 'page-head';
        pageHead.appendChild(EL('h2', { text: 'MCP & Integrations' }));
        root.appendChild(pageHead);

        root.appendChild(buildIntro());
        root.appendChild(buildCurrentCard(data));
        root.appendChild(buildProfilesCard(data));
        root.appendChild(buildHowToCard());

        pane.appendChild(root);
      })
      .catch(function (err) {
        pane.textContent = '';
        pane.appendChild(buildError(
          'Could not load MCP profile data. Is the daemon running?',
          function () { window.odRenderMcp(pane); }
        ));
      });
  };

  // Auto-run at DOMContentLoaded if the pane already exists in DOM
  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('mcp-pane');
    if (pane) window.odRenderMcp(pane);
  });

}());

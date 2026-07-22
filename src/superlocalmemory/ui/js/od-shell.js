/* SuperLocalMemory — OD Shell v1.0
 * CSP-safe single-page port of assets/shell.js (Open Design 022c4af9).
 *
 * Hard constraints obeyed:
 *   - NO inline <script> blocks or on*= event handlers anywhere
 *   - All events wired via addEventListener or data-act-* delegation
 *   - Syncs data-theme (OD) + data-bs-theme (Bootstrap) + .ng-dark body class
 *   - Sidebar nav uses data-tab="<paneId>" for single-page switching
 *   - Replaces ng-shell.js entirely; all ng-shell functionality ported here
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  /* ================================================================
     Icons (lucide-style 24×24 stroke path data)
     Same set as OD assets/shell.js plus Bootstrap-icon compat shapes.
  ================================================================ */
  var I = {
    dashboard:  '<path d="M3 3h8v8H3zM13 3h8v5h-8zM13 12h8v9h-8zM3 15h8v6H3z"/>',
    brain:      '<path d="M12 5a3 3 0 0 0-6 0 3 3 0 0 0-3 3 3 3 0 0 0 1.5 2.6A3 3 0 0 0 6 17a3 3 0 0 0 6 0zM12 5a3 3 0 0 1 6 0 3 3 0 0 1 3 3 3 3 0 0 1-1.5 2.6A3 3 0 0 1 18 17a3 3 0 0 1-6 0zM12 5v14"/>',
    graph:      '<circle cx="5" cy="6" r="2.5"/><circle cx="19" cy="7" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="17" cy="18" r="2.5"/><path d="M7 7l9 9M7.2 6.6L16.5 6.9M7.5 16.6l8.2.6M6 15.5l1-6.5"/>',
    memories:   '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    entity:     '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 12l9-5M12 12v10M12 12L3 7"/>',
    skill:      '<path d="M12 3l1.9 4.6L18 9l-3.5 3 1 5-3.5-2.5L8.5 17l1-5L6 9l4.1-1.4z"/>',
    health:     '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
    operations: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>',
    optimize:   '<path d="M13 2L4.1 12.3a1 1 0 0 0 .8 1.7H11l-1 8 8.9-10.3a1 1 0 0 0-.8-1.7H12z"/>',
    mesh:       '<circle cx="12" cy="12" r="2.5"/><circle cx="5" cy="5" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path d="M10 10L6.5 6.5M14 10l3.5-3.5M10 14l-3.5 3.5M14 14l3.5 3.5"/>',
    settings:   '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"/>',
    search:     '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    sun:        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/>',
    moon:       '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    menu:       '<path d="M4 6h16M4 12h16M4 18h16"/>',
    lock:       '<rect x="4" y="10" width="16" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    shield:     '<path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6z"/><path d="M9 12l2 2 4-4"/>',
    plug:       '<path d="M9 2v6M15 2v6M7 8h10v3a5 5 0 0 1-10 0zM12 16v6"/>',
    github:     '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>',
    star:       '<path d="M12 2l3 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.9 21l1.2-6.8-5-4.9 6.9-1z"/>',
    pkg:        '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 12l9-5M12 12v10M12 12L3 7"/>',
    clock:      '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    link:       '<path d="M9 15l6-6M10 6l1-1a4 4 0 0 1 6 6l-1 1M14 18l-1 1a4 4 0 0 1-6-6l1-1"/>',
    cloud:      '<path d="M17.5 19a4.5 4.5 0 1 0-1.4-8.8A6 6 0 0 0 4.5 12 3.5 3.5 0 0 0 6 18.9z"/>',
    filter:     '<path d="M3 4h18l-7 8.5V20l-4 1v-8.5z"/>',
    send:       '<path d="M22 2L11 13M22 2l-7 20-4-9-9-4z"/>',
    download:   '<path d="M12 3v12M7 11l5 5 5-5M4 21h16"/>',
    plus:       '<path d="M12 5v14M5 12h14"/>',
    refresh:    '<path d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5"/>',
  };

  function svg(name, cls) {
    return '<svg class="' + (cls || '') + '" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
      (I[name] || '') + '</svg>';
  }
  window.slmIcon = svg;

  /* ================================================================
     Nav model: OD groups → real pane IDs (single-page, no page links)
  ================================================================ */
  var NAV = [
    { g: 'Overview', items: [
      { k: 'dashboard-pane', t: 'Dashboard',   i: 'dashboard', crumb: 'Overview' },
      { k: 'brain-pane',     t: 'Brain',        i: 'brain',     crumb: 'Overview', tag: 'live', hero: true },
    ]},
    { g: 'Memory', items: [
      { k: 'graph-pane',     t: 'Knowledge Graph',  i: 'graph',    crumb: 'Memory', hero: true },
      { k: 'memories-pane',  t: 'Memories',          i: 'memories', crumb: 'Memory' },
      { k: 'entities-pane',  t: 'Entity Explorer',   i: 'entity',   crumb: 'Memory' },
    ]},
    { g: 'Intelligence', items: [
      { k: 'skills-pane',    t: 'Skill Evolution',  i: 'skill',    crumb: 'Intelligence' },
    ]},
    { g: 'System', items: [
      { k: 'health-pane',     t: 'Health',      i: 'health',     crumb: 'System' },
      { k: 'operations-pane', t: 'Governance',  i: 'shield',     crumb: 'System' },
      { k: 'optimize-pane',   t: 'Optimize',    i: 'optimize',   crumb: 'System' },
    ]},
    { g: 'Integrations', items: [
      { k: 'mcp-pane',       t: 'MCP & Tools', i: 'plug',     crumb: 'Integrations' },
      { k: 'mesh-pane',      t: 'Mesh Peers',  i: 'mesh',     crumb: 'Integrations' },
    ]},
    { g: 'Config', items: [
      { k: 'settings-pane',  t: 'Settings',      i: 'settings', crumb: 'Config' },
      { k: 'backup-pane',    t: 'Cloud Backup',  i: 'cloud',    crumb: 'Config' },
    ]},
  ];

  // Flat lookup: paneId → nav item
  var NAV_MAP = {};
  NAV.forEach(function (grp) {
    grp.items.forEach(function (it) { NAV_MAP[it.k] = it; });
  });

  /* ================================================================
     Brand links
  ================================================================ */
  var LINKS = {
    repo:     'https://github.com/qualixar/superlocalmemory',
    npm:      'https://www.npmjs.com/package/superlocalmemory',
    pip:      'https://pypi.org/project/superlocalmemory/',
    qualixar: 'https://qualixar.com',
    author:   'https://varunpratap.com',
    license:  'https://github.com/qualixar/superlocalmemory/blob/main/LICENSE',
    stars:    '2,431',
  };

  /* ================================================================
     Theme: syncs data-theme (OD) + data-bs-theme (Bootstrap) + ng-dark
  ================================================================ */
  var THEME_KEY = 'slm-theme';

  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    document.documentElement.setAttribute('data-bs-theme', t);
    if (t === 'dark') {
      document.body.classList.add('ng-dark');
    } else {
      document.body.classList.remove('ng-dark');
    }
    try { localStorage.setItem(THEME_KEY, t); } catch (e) {}

    // Update all [data-theme-icon] placeholders (sun in dark, moon in light)
    document.querySelectorAll('[data-theme-icon]').forEach(function (el) {
      el.innerHTML = svg(t === 'dark' ? 'sun' : 'moon');
    });

    // Sync inline-styled graph/timeline backgrounds (Bootstrap inline styles override vars)
    ['graph-container', 'memory-timeline-chart'].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.style.setProperty('background', t === 'dark' ? '#0f1012' : '#ffffff', 'important');
      el.style.setProperty('background-color', t === 'dark' ? '#0f1012' : '#ffffff', 'important');
      el.style.setProperty('border-color', t === 'dark' ? 'rgba(255,255,255,0.06)' : '#e5e7eb', 'important');
    });
  }

  // Apply theme immediately (before DOMContentLoaded) to prevent FOUC.
  // Both data-theme (OD CSS) and data-bs-theme (Bootstrap) must be set here.
  (function initThemeEarly() {
    var t = 'dark';
    try { t = localStorage.getItem(THEME_KEY) || 'dark'; } catch (e) {}
    document.documentElement.setAttribute('data-theme', t);
    document.documentElement.setAttribute('data-bs-theme', t);
  }());

  // Public API — override the core.js definition (which only sets data-bs-theme)
  window.toggleDarkMode = function () {
    var cur = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  };
  window.slmToggleTheme = window.toggleDarkMode;

  /* ================================================================
     Build sidebar HTML
  ================================================================ */
  function buildSidebarHTML(activePane) {
    var html =
      '<div class="brand">' +
        '<div class="logo"><img src="static/assets/slm-icon.svg" alt="SLM" width="26" height="26"></div>' +
        '<div><div class="name">SuperLocalMemory</div><div class="sub">by Qualixar</div></div>' +
      '</div>' +
      '<nav class="nav">';

    NAV.forEach(function (grp) {
      html += '<div class="nav-group-label">' + grp.g + '</div>';
      grp.items.forEach(function (it) {
        var on   = it.k === activePane ? ' active' : '';
        var hero = it.hero ? ' hero' : '';
        var tag  = it.tag ? '<span class="tag">' + it.tag + '</span>' : '';
        html +=
          '<a class="nav-link' + on + hero + '" data-tab="' + it.k + '" ' +
          'role="button" tabindex="0" aria-controls="' + it.k + '">' +
          svg(it.i) + '<span>' + it.t + '</span>' + tag +
          '</a>';
      });
    });

    html +=
      '</nav>' +
      '<div class="sidebar-foot" id="od-sidebar-foot">' +
        '<span class="dot pulse"></span>' +
        '<span id="od-version">Local daemon</span>' +
      '</div>' +
      '<div id="od-profile-container" style="padding:0 12px 12px;display:flex;gap:4px;align-items:center"></div>';

    return html;
  }

  /* ================================================================
     Tab activation — single-page SPA switching
  ================================================================ */
  function activateTab(paneId) {
    // Hide all Bootstrap tab panes
    document.querySelectorAll('.tab-pane').forEach(function (p) {
      p.classList.remove('show', 'active');
    });

    // Show target pane
    var pane = document.getElementById(paneId);
    if (pane) pane.classList.add('show', 'active');

    // Update sidebar active state
    document.querySelectorAll('.nav-link[data-tab]').forEach(function (link) {
      link.classList.toggle('active', link.getAttribute('data-tab') === paneId);
    });

    // Update topbar breadcrumb and heading
    var item = NAV_MAP[paneId];
    if (item) {
      var crumbEl = document.getElementById('topbar-crumb');
      var headEl  = document.getElementById('topbar-heading');
      if (crumbEl) crumbEl.textContent = item.crumb || '';
      if (headEl)  headEl.textContent  = item.t;
    }

    // Dispatch Bootstrap shown.bs.tab event for backward compatibility.
    // brain.js, dashboard.js, etc. listen on the hidden tab buttons.
    var tabBtn = document.getElementById(paneId.replace('-pane', '-tab'));
    if (tabBtn) {
      try {
        tabBtn.dispatchEvent(new Event('shown.bs.tab', { bubbles: true }));
      } catch (e) {}
    }

    // Update URL hash without triggering a scroll
    try { history.replaceState(null, '', '#' + paneId); } catch (e) {}

    // Scroll both the window and the content area to top
    window.scrollTo({ top: 0, behavior: 'instant' });
    var contentEl = document.getElementById('main-content');
    if (contentEl) contentEl.scrollTo({ top: 0, behavior: 'instant' });

    // Fire lazy data loaders
    triggerTabLoad(paneId);
    // Deferred retry for async-heavy tabs that need extra time
    setTimeout(function () { triggerTabLoad(paneId); }, 500);

    // Scroll the active sidebar item into view (mobile)
    var activeLink = document.querySelector('.nav-link[data-tab="' + paneId + '"]');
    if (activeLink) {
      activeLink.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  /* ================================================================
     Lazy data loaders — same mapping as ng-shell.js triggerTabLoad
  ================================================================ */
  function triggerTabLoad(tabId) {
    var pane = document.getElementById(tabId);
    // Open Design takeover: if the pane's OD render module is loaded, it OWNS
    // the pane — clear the legacy markup and let the OD module render the
    // approved design wired to live data. Fall back to the legacy loader(s)
    // only when the OD module for that pane isn't present yet.
    function od(fnName) {
      if (pane && typeof window[fnName] === 'function') {
        try { pane.innerHTML = ''; window[fnName](pane); return true; }
        catch (e) { console.error('OD render failed for ' + tabId + ':', e); }
      }
      return false;
    }
    switch (tabId) {
      case 'brain-pane':
        if (od('odRenderBrain')) break;
        if (typeof loadBrain === 'function') loadBrain();
        break;
      case 'graph-pane':
        if (od('odRenderGraph')) break;
        if (typeof loadGraph === 'function') loadGraph();
        if (typeof initMemoryChat === 'function' && !document.getElementById('chat-panel')) {
          initMemoryChat();
        }
        if (typeof loadClusters === 'function') loadClusters();
        break;
      case 'entities-pane':
        if (od('odRenderEntities')) break;
        if (typeof loadEntityExplorer === 'function') loadEntityExplorer();
        break;
      case 'memories-pane':
        if (od('odRenderMemories')) break;
        if (typeof loadMemories === 'function') loadMemories();
        if (typeof loadTimeline === 'function') loadTimeline();
        break;
      case 'health-pane':
        if (od('odRenderHealth')) break;
        if (typeof loadHealthMonitor === 'function') loadHealthMonitor();
        if (typeof initEventStream === 'function') initEventStream();
        if (typeof loadEventStats === 'function') loadEventStats();
        if (typeof loadAgents === 'function') loadAgents();
        if (typeof loadIDEStatus === 'function') loadIDEStatus();
        if (typeof loadMathHealth === 'function') loadMathHealth();
        break;
      case 'operations-pane':
        if (od('odRenderOperations')) break;
        if (typeof loadIngestionStatus === 'function') loadIngestionStatus();
        if (typeof loadLifecycle === 'function') loadLifecycle();
        if (typeof loadTrustDashboard === 'function') loadTrustDashboard();
        if (typeof loadCompliance === 'function') loadCompliance();
        break;
      case 'skills-pane':
        if (od('odRenderSkills')) break;
        if (typeof loadSkillEvolution === 'function') loadSkillEvolution();
        break;
      case 'mesh-pane':
        if (od('odRenderMesh')) break;
        if (typeof loadMeshPeers === 'function') loadMeshPeers();
        break;
      case 'settings-pane':
        if (od('odRenderSettings')) break;
        if (typeof loadSettings === 'function') loadSettings();
        if (typeof loadModeSettings === 'function') loadModeSettings();
        if (typeof loadAutoSettings === 'function') loadAutoSettings();
        if (typeof updateModeUI === 'function') updateModeUI();
        break;
      case 'optimize-pane':
        if (od('odRenderOptimize')) break;
        if (typeof initOptimizeTab === 'function') initOptimizeTab();
        break;
      case 'mcp-pane':
        od('odRenderMcp');
        break;
      case 'backup-pane':
        od('odRenderBackup');
        break;
    }
  }

  /* ================================================================
     Hash routing: handles both pane IDs and section anchors within panes
  ================================================================ */
  function handleHash() {
    var hash = window.location.hash.replace('#', '');
    if (!hash) return;
    var el = document.getElementById(hash);
    if (!el) return;

    // Direct pane navigation
    if (el.classList && el.classList.contains('tab-pane')) {
      activateTab(hash);
      return;
    }

    // Section within a pane (e.g. #health-section-events)
    var parentPane = el.closest && el.closest('.tab-pane');
    if (parentPane && parentPane.id) activateTab(parentPane.id);
    try { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) {}
  }

  /* ================================================================
     Shell mount: called at DOMContentLoaded
  ================================================================ */
  window.slmShell = function (opts) {
    opts = opts || {};
    var active = opts.active || 'dashboard-pane';

    // 1. Fill sidebar
    var sb = document.getElementById('sidebar');
    if (sb) sb.innerHTML = buildSidebarHTML(active);

    // 2. Wire sidebar nav-link clicks and keyboard
    document.querySelectorAll('.nav-link[data-tab]').forEach(function (link) {
      link.addEventListener('click', function (e) {
        e.preventDefault();
        activateTab(this.getAttribute('data-tab'));
      });
      link.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          activateTab(this.getAttribute('data-tab'));
        }
      });
    });

    // 3. Mobile sidebar: hamburger button + scrim
    var menuBtn  = document.getElementById('menuBtn');
    var scrim    = document.getElementById('scrim');
    var sidebarEl = document.getElementById('sidebar');
    if (menuBtn && sidebarEl && scrim) {
      menuBtn.addEventListener('click', function () {
        sidebarEl.classList.toggle('open');
        scrim.classList.toggle('on');
      });
      scrim.addEventListener('click', function () {
        sidebarEl.classList.remove('open');
        scrim.classList.remove('on');
      });
    }

    // 4. Apply current theme (fills icons, syncs classes)
    var t = 'dark';
    try { t = localStorage.getItem(THEME_KEY) || 'dark'; } catch (e) {}
    applyTheme(t);

    // 5. Wire theme toggle buttons via addEventListener (NOT onclick — CSP safe)
    document.querySelectorAll('[data-theme-icon]').forEach(function (btn) {
      // Prevent double-binding if slmShell() is somehow called twice
      if (btn.dataset.odThemeWired) return;
      btn.dataset.odThemeWired = '1';
      btn.addEventListener('click', function () { window.toggleDarkMode(); });
    });

    // 6. Fill data-ic SVG placeholders (icons in pane content, topbar, etc.)
    document.querySelectorAll('[data-ic]').forEach(function (el) {
      el.innerHTML = svg(el.getAttribute('data-ic'), el.getAttribute('data-ic-cls') || '');
    });

    // 7. Inject "Star us on GitHub" CTA into topbar (before the theme button)
    var topbar = document.querySelector('.topbar');
    if (topbar && !topbar.querySelector('.star-cta')) {
      var themeBtn = topbar.querySelector('[data-theme-icon]');
      var starLink = document.createElement('a');
      starLink.className   = 'star-cta';
      starLink.href        = LINKS.repo;
      starLink.target      = '_blank';
      starLink.rel         = 'noopener';
      starLink.title       = 'Star SuperLocalMemory on GitHub';
      starLink.innerHTML   =
        svg('github', 'gh') +
        "<span class='lbl'>Star us on GitHub</span>" +
        "<span class='star-count'>" + svg('star') + LINKS.stars + '</span>';
      if (themeBtn) topbar.insertBefore(starLink, themeBtn);
      else topbar.appendChild(starLink);
    }

    // 8. Inject shared footer into .content (OD site-foot style)
    var content = document.getElementById('main-content');
    if (content && !content.querySelector('.site-foot')) {
      content.insertAdjacentHTML('beforeend',
        '<footer class="site-foot">' +
          '<div class="foot-brand">' +
            '<span class="foot-logo">' +
              '<img src="static/assets/slm-icon.svg" alt="" width="22" height="22">' +
            '</span>' +
            '<div><b>SuperLocalMemory</b><span>by Qualixar · local-first AI memory</span></div>' +
          '</div>' +
          '<nav class="foot-links">' +
            '<a href="' + LINKS.author  + '" target="_blank" rel="noopener">Built by Varun Pratap Bhardwaj</a>' +
            '<a href="' + LINKS.qualixar + '" target="_blank" rel="noopener">qualixar.com</a>' +
            '<a href="' + LINKS.repo    + '" target="_blank" rel="noopener">' + svg('github') + 'GitHub</a>' +
            '<a href="' + LINKS.npm     + '" target="_blank" rel="noopener">' + svg('pkg')    + 'npm</a>' +
            '<a href="' + LINKS.pip     + '" target="_blank" rel="noopener">' + svg('pkg')    + 'pip</a>' +
            '<a href="' + LINKS.license + '" target="_blank" rel="noopener">License · AGPL-3.0</a>' +
          '</nav>' +
        '</footer>');
    }

    // 9. Move profile selector from ng-premount into sidebar footer
    var profileContainer = document.getElementById('od-profile-container');
    var profileSelect    = document.getElementById('profile-select');
    var addProfileBtn    = document.getElementById('add-profile-btn');
    if (profileContainer && profileSelect && profileSelect.parentNode !== profileContainer) {
      profileSelect.style.cssText = 'flex:1;font-size:0.8125rem;max-width:150px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:var(--on-sidebar);padding:3px 6px;';
      profileContainer.appendChild(profileSelect);
      profileSelect.style.display = '';
      if (addProfileBtn) {
        addProfileBtn.className  = '';
        addProfileBtn.style.cssText = 'padding:4px 8px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:var(--on-sidebar);cursor:pointer;';
        profileContainer.appendChild(addProfileBtn);
        addProfileBtn.style.display = '';
      }
    }

    // 10. Fetch daemon version and update sidebar foot
    fetch('/health', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        var verEl = document.getElementById('od-version');
        if (verEl && data && data.version) verEl.textContent = 'v' + data.version;
      })
      .catch(function () {
        var dashVer = document.getElementById('dashboard-version');
        var verEl   = document.getElementById('od-version');
        if (verEl && dashVer && dashVer.textContent && dashVer.textContent !== '...') {
          verEl.textContent = 'v' + dashVer.textContent;
        }
      });

    // 11. Privacy blur (screen recording mode)
    window.togglePrivacyBlur = function () {
      document.body.classList.toggle('ng-privacy-blur');
      var icon = document.getElementById('ng-privacy-icon');
      var isBlurred = document.body.classList.contains('ng-privacy-blur');
      if (icon) icon.className = isBlurred ? 'bi bi-eye' : 'bi bi-eye-slash';
    };
    if (window.location.search.indexOf('blur=1') >= 0) {
      document.body.classList.add('ng-privacy-blur');
    }

    // 12. Activate the initial pane (fires triggerTabLoad for dashboard etc.)
    activateTab(active);
  };

  /* ================================================================
     Chart helpers — ported from OD assets/shell.js
  ================================================================ */

  // Sparkline: array of numbers → inline SVG string
  window.slmSpark = function (data, opts) {
    opts = opts || {};
    var w = opts.w || 200, h = opts.h || 60, pad = 3;
    var min = Math.min.apply(null, data);
    var max = Math.max.apply(null, data);
    var rng = (max - min) || 1;
    var stepX = (w - pad * 2) / (data.length - 1);
    var pts = data.map(function (v, i) {
      return [
        pad + i * stepX,
        h - pad - ((v - min) / rng) * (h - pad * 2),
      ];
    });
    var line = pts.map(function (p, i) {
      return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1);
    }).join(' ');
    var area =
      line +
      ' L' + pts[pts.length - 1][0].toFixed(1) + ' ' + h +
      ' L' + pts[0][0].toFixed(1) + ' ' + h + ' Z';
    var id     = 'g' + Math.random().toString(36).slice(2, 7);
    var stroke = opts.color || 'var(--violet)';
    return (
      '<svg class="spark-lg" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">' +
        '<defs><linearGradient id="' + id + '" x1="0" y1="0" x2="0" y2="1">' +
          '<stop offset="0" stop-color="' + stroke + '" stop-opacity="0.28"/>' +
          '<stop offset="1" stop-color="' + stroke + '" stop-opacity="0"/>' +
        '</linearGradient></defs>' +
        '<path d="' + area + '" fill="url(#' + id + ')"/>' +
        '<path d="' + line + '" fill="none" stroke="' + stroke + '" stroke-width="2" vector-effect="non-scaling-stroke"/>' +
      '</svg>'
    );
  };

  // GitHub-style contribution heatmap
  window.slmHeatmap = function (el, weeks) {
    weeks = weeks || 26;
    var html = '';
    for (var w = 0; w < weeks; w++) {
      for (var d = 0; d < 7; d++) {
        var recency = w / weeks;
        var base = Math.random() * 0.5 + recency * 0.6;
        var l = base < 0.35 ? 0 : base < 0.55 ? 1 : base < 0.72 ? 2 : base < 0.88 ? 3 : 4;
        html += '<i data-l="' + l + '" title="reward signal"></i>';
      }
    }
    el.innerHTML = html;
  };

  // Bar chart helper
  window.slmBars = function (el, data) {
    var maxVal = Math.max.apply(null, data) || 1;
    el.innerHTML = data.map(function (v) {
      return '<i style="height:' + Math.max(4, (v / maxVal) * 100) + '%" title="' + v + '"></i>';
    }).join('');
  };

  /* ================================================================
     DOMContentLoaded bootstrap
  ================================================================ */
  document.addEventListener('DOMContentLoaded', function () {
    // Determine initial active pane from URL hash (if valid tab pane)
    var active = 'dashboard-pane';
    var hash = window.location.hash.replace('#', '');
    if (hash) {
      var el = document.getElementById(hash);
      if (el && el.classList && el.classList.contains('tab-pane')) {
        active = hash;
      } else {
        // Stale section anchor from a previous session — strip it so
        // we always land on Dashboard, not a blank screen.
        try { history.replaceState(null, '', window.location.pathname); } catch (e) {}
      }
    }

    window.slmShell({ active: active });

    // Wire subsequent hash changes (user-triggered navigation, not initial load)
    window.addEventListener('hashchange', handleHash);
  });

}());

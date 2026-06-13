// Neural Glass Shell — Dashboard V2 "Neural Glass"
// Injects sidebar, restructures DOM, handles navigation
// Progressive enhancement: if this fails, Bootstrap tabs still work

(function() {
  'use strict';

  // ── Sidebar Navigation Config ──────────────────────────────
  var NAV_SECTIONS = [
    {
      label: 'Memory',
      items: [
        { id: 'dashboard-pane', icon: 'bi-speedometer2', text: 'Dashboard' },
        { id: 'brain-pane', icon: 'bi-lightbulb', text: 'Brain' },
        { id: 'graph-pane', icon: 'bi-diagram-3', text: 'Knowledge Graph' },
        { id: 'memories-pane', icon: 'bi-list-ul', text: 'Memories' }
      ]
    },
    {
      label: 'System',
      items: [
        { id: 'health-pane', icon: 'bi-heart-pulse', text: 'Health' },
        { id: 'operations-pane', icon: 'bi-diagram-2', text: 'Operations' },
        { id: 'entities-pane', icon: 'bi-person-badge', text: 'Entity Explorer' },
        { id: 'skills-pane', icon: 'bi-lightning-charge', text: 'Skill Evolution' },
        { id: 'mesh-pane', icon: 'bi-share', text: 'Mesh Peers' }
      ]
    },
    {
      label: 'Config',
      items: [
        { id: 'settings-pane', icon: 'bi-gear', text: 'Settings' },
        { id: 'optimize-pane', icon: 'bi-speedometer', text: 'Optimize' }
      ]
    }
  ];

  // ── Build Sidebar HTML ─────────────────────────────────────
  function buildSidebar() {
    var sidebar = document.createElement('aside');
    sidebar.className = 'ng-sidebar';
    sidebar.id = 'ng-sidebar';
    sidebar.setAttribute('role', 'navigation');
    sidebar.setAttribute('aria-label', 'Main navigation');

    // Header
    var header = document.createElement('div');
    header.className = 'ng-sidebar-header';
    header.innerHTML =
      '<div class="ng-sidebar-brand">' +
        '<div class="ng-sidebar-brand-icon">' +
          '<i class="bi bi-diamond-fill" style="font-size:0.875rem"></i>' +
        '</div>' +
        '<div>' +
          '<div class="ng-sidebar-brand-text">SuperLocalMemory</div>' +
          '<div class="ng-sidebar-brand-version" id="ng-version">\u2026</div>' +
        '</div>' +
      '</div>';
    sidebar.appendChild(header);

    // Nav
    var nav = document.createElement('nav');
    nav.className = 'ng-sidebar-nav';

    NAV_SECTIONS.forEach(function(section) {
      var sDiv = document.createElement('div');
      sDiv.className = 'ng-sidebar-section';

      var label = document.createElement('div');
      label.className = 'ng-sidebar-section-label';
      label.textContent = section.label;
      sDiv.appendChild(label);

      section.items.forEach(function(item) {
        var a = document.createElement('a');
        a.className = 'ng-sidebar-item';
        a.setAttribute('data-target', item.id);
        a.setAttribute('role', 'tab');
        a.setAttribute('aria-controls', item.id);
        a.setAttribute('tabindex', '0');
        if (item.id === 'dashboard-pane') a.classList.add('active');

        a.innerHTML =
          '<i class="bi ' + item.icon + ' ng-sidebar-icon"></i>' +
          '<span>' + item.text + '</span>' +
          (item.badge ? '<span class="ng-sidebar-badge">' + item.badge + '</span>' : '');

        a.addEventListener('click', function(e) {
          e.preventDefault();
          activateTab(item.id);
        });

        a.addEventListener('keydown', function(e) {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            activateTab(item.id);
          }
        });

        sDiv.appendChild(a);
      });

      nav.appendChild(sDiv);
    });

    sidebar.appendChild(nav);

    // Footer
    var footer = document.createElement('div');
    footer.className = 'ng-sidebar-footer';

    // Move profile selector from navbar to sidebar footer
    var profileSelect = document.getElementById('profile-select');
    var addProfileBtn = document.getElementById('add-profile-btn');

    footer.innerHTML =
      // v3.4.10: Cloud Backup Account Widget
      '<div id="ng-account-widget" style="margin-bottom:10px;padding:8px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);cursor:pointer;" onclick="document.querySelector(\'[data-target=settings]\')?.click()">' +
        '<div style="display:flex;align-items:center;gap:8px;">' +
          '<span id="ng-account-avatar" style="width:28px;height:28px;border-radius:50%;background:rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;overflow:hidden;">' +
            '<i class="bi bi-cloud-slash" style="font-size:13px;opacity:0.4;"></i>' +
          '</span>' +
          '<div style="flex:1;min-width:0;">' +
            '<div id="ng-account-name" style="font-size:12px;color:#e0e0e0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Not connected</div>' +
            '<div id="ng-account-status" style="font-size:10px;color:#666;">No cloud backup</div>' +
          '</div>' +
          '<span id="ng-account-dot" style="width:7px;height:7px;border-radius:50%;background:#444;flex-shrink:0;" title="No cloud backup"></span>' +
        '</div>' +
        '<div id="ng-account-actions" style="display:none;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05);">' +
          '<div style="display:flex;gap:4px;">' +
            '<button class="ng-btn" onclick="event.stopPropagation();connectGoogleDrive()" title="Connect Google Drive" style="flex:1;justify-content:center;font-size:11px;padding:4px;">' +
              '<i class="bi bi-google" style="color:#4285f4;"></i>' +
            '</button>' +
            '<button class="ng-btn" onclick="event.stopPropagation();connectGitHub()" title="Connect GitHub" style="flex:1;justify-content:center;font-size:11px;padding:4px;">' +
              '<i class="bi bi-github"></i>' +
            '</button>' +
            '<button class="ng-btn" onclick="event.stopPropagation();syncCloudNow()" title="Sync Now" style="flex:1;justify-content:center;font-size:11px;padding:4px;">' +
              '<i class="bi bi-cloud-upload" style="color:#00D4AA;"></i>' +
            '</button>' +
            '<button class="ng-btn" onclick="event.stopPropagation();exportBackup()" title="Export Backup" style="flex:1;justify-content:center;font-size:11px;padding:4px;">' +
              '<i class="bi bi-download" style="color:#f39c12;"></i>' +
            '</button>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div style="margin-bottom:8px">' +
        '<div class="ng-sidebar-section-label" style="padding:0 0 4px">Profile</div>' +
        '<div style="display:flex;gap:4px;align-items:center" id="ng-profile-container"></div>' +
      '</div>' +
      '<div style="display:flex;gap:4px;align-items:center">' +
        '<button class="ng-btn" id="ng-refresh-btn" title="Refresh" style="flex:1;justify-content:center">' +
          '<i class="bi bi-arrow-clockwise"></i>' +
        '</button>' +
        '<button class="ng-btn" id="ng-theme-toggle" title="Toggle theme" onclick="toggleDarkMode()" style="flex:1;justify-content:center">' +
          '<i class="bi bi-sun-fill" id="ng-theme-icon"></i>' +
        '</button>' +
        '<button class="ng-btn" id="ng-privacy-btn" title="Privacy blur (for screen recording)" onclick="togglePrivacyBlur()" style="flex:1;justify-content:center">' +
          '<i class="bi bi-eye-slash" id="ng-privacy-icon"></i>' +
        '</button>' +
        '<a href="https://github.com/qualixar/superlocalmemory" target="_blank" class="ng-btn" title="Star on GitHub" style="flex:1;justify-content:center">' +
          '<i class="bi bi-github"></i>' +
        '</a>' +
      '</div>';

    sidebar.appendChild(footer);

    return sidebar;
  }

  // ── Restructure DOM ────────────────────────────────────────
  function restructureDOM() {
    var body = document.body;
    // Target the MAIN container-fluid (direct child of body), not the one inside navbar
    var containers = document.querySelectorAll('body > .container-fluid');
    var container = containers.length > 0 ? containers[containers.length - 1] : null;
    if (!container) {
      // Fallback: find the container that holds the tab-content
      container = document.querySelector('.tab-content')?.closest('.container-fluid');
    }
    if (!container) return;

    // Create shell wrapper
    var shell = document.createElement('div');
    shell.className = 'ng-shell';

    // Build and insert sidebar
    var sidebar = buildSidebar();
    shell.appendChild(sidebar);

    // Create content wrapper
    var content = document.createElement('main');
    content.className = 'ng-content';
    var inner = document.createElement('div');
    inner.className = 'ng-content-inner';

    // Move container content into inner
    while (container.firstChild) {
      inner.appendChild(container.firstChild);
    }
    content.appendChild(inner);
    shell.appendChild(content);

    // Move footer into content
    var existingFooter = document.querySelector('footer');
    if (existingFooter) {
      content.appendChild(existingFooter);
    }

    // Replace container with shell
    container.parentNode.replaceChild(shell, container);

    // Move dashboard-only elements INTO the dashboard-pane so they scroll
    // with it. stats-container starts ``display:none`` in the HTML to kill
    // the pre-glass flash; reveal it once it's in its final home.
    var dashboardPane = document.getElementById('dashboard-pane');
    if (dashboardPane) {
      ['stats-container', 'privacy-notice', 'feedback-progress'].forEach(function(id) {
        var el = document.getElementById(id);
        if (el) {
          dashboardPane.insertBefore(el, dashboardPane.firstChild);
          if (id === 'stats-container') el.style.display = '';
        }
      });
    }

    // Move profile selector to sidebar footer
    var profileContainer = document.getElementById('ng-profile-container');
    var profileSelect = document.getElementById('profile-select');
    var addProfileBtn = document.getElementById('add-profile-btn');
    if (profileContainer && profileSelect) {
      profileSelect.classList.remove('profile-select');
      profileSelect.style.cssText = 'flex:1;font-size:0.8125rem;';
      profileContainer.appendChild(profileSelect);
      if (addProfileBtn) {
        addProfileBtn.className = 'ng-btn';
        addProfileBtn.style.cssText = 'padding:4px 8px;';
        profileContainer.appendChild(addProfileBtn);
      }
    }

    // Wire refresh button
    var refreshBtn = document.getElementById('ng-refresh-btn');
    if (refreshBtn && typeof refreshDashboard === 'function') {
      refreshBtn.addEventListener('click', refreshDashboard);
    }

    // Sidebar version: always the live daemon version from /health.
    // Don't depend on dashboard.js timing — that used to leave "v3.4.4"
    // hardcoded in the sidebar when dashboard.js hadn't run yet.
    fetch('/health', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        var ngVer = document.getElementById('ng-version');
        if (ngVer && data && data.version) {
          ngVer.textContent = 'v' + data.version;
        }
      })
      .catch(function () {
        // Fallback: if /health is unreachable, fall back to whatever
        // dashboard.js populates into #dashboard-version.
        var dashVer = document.getElementById('dashboard-version');
        var ngVer = document.getElementById('ng-version');
        if (dashVer && ngVer && dashVer.textContent && dashVer.textContent !== '...') {
          ngVer.textContent = 'v' + dashVer.textContent;
        }
      });
  }

  // ── Tab Activation ─────────────────────────────────────────
  function activateTab(targetId) {
    // Deactivate all sidebar items
    document.querySelectorAll('.ng-sidebar-item').forEach(function(item) {
      item.classList.remove('active');
      item.setAttribute('aria-selected', 'false');
    });

    // Activate clicked sidebar item
    var activeItem = document.querySelector('.ng-sidebar-item[data-target="' + targetId + '"]');
    if (activeItem) {
      activeItem.classList.add('active');
      activeItem.setAttribute('aria-selected', 'true');
    }

    // Deactivate all tab panes
    document.querySelectorAll('.tab-pane').forEach(function(pane) {
      pane.classList.remove('show', 'active');
    });

    // Activate target pane
    var targetPane = document.getElementById(targetId);
    if (targetPane) {
      targetPane.classList.add('show', 'active');
    }

    // Dispatch Bootstrap tab event for backward compat.
    //
    // Previous code assigned ``event.target = tabButton``; that property
    // is readonly on DOM Event objects, so in strict mode the assignment
    // throws TypeError, the surrounding try/catch silently swallowed it,
    // and dispatchEvent() was NEVER reached. brain.js's
    // ``shown.bs.tab`` listener never fired when navigating between tabs
    // via the sidebar, so the Brain pane stayed stuck on "Loading Brain..."
    // forever (and any other listener that relied on the event didn't
    // get a chance). The DOM automatically sets ``event.target`` to the
    // element on which ``dispatchEvent`` is called, so we don't need to
    // assign it manually.
    var tabButton = document.getElementById(targetId.replace('-pane', '-tab'));
    if (tabButton) {
      try {
        var tabEvent = new Event('shown.bs.tab', { bubbles: true });
        tabButton.dispatchEvent(tabEvent);
      } catch (e) {
        // Ignore if event dispatch fails (very old browsers)
      }
    }

    // Update URL hash via replaceState (avoids auto-scroll to hash target)
    history.replaceState(null, '', '#' + targetId);

    // Scroll content to top on tab switch
    window.scrollTo({ top: 0, behavior: 'instant' });
    var contentEl = document.querySelector('.ng-content');
    if (contentEl) contentEl.scrollTo({ top: 0, behavior: 'instant' });

    // Dashboard-only elements are inside dashboard-pane, so they auto-hide/show with the tab

    // Trigger data loading for lazy-loaded tabs (immediate + deferred for async-heavy tabs)
    triggerTabLoad(targetId);
    // Deferred retry for tabs that need API data to populate
    setTimeout(function() { triggerTabLoad(targetId); }, 500);

    // Scroll sidebar item into view
    if (activeItem) {
      activeItem.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }

  // ── Lazy Load Tab Data ─────────────────────────────────────
  function triggerTabLoad(tabId) {
    switch(tabId) {
      case 'brain-pane':
        if (typeof loadBrain === 'function') loadBrain();
        break;
      case 'graph-pane':
        if (typeof loadGraph === 'function') loadGraph();
        // v3.4.4: Initialize chat panel if not already present
        if (typeof initMemoryChat === 'function' && !document.getElementById('chat-panel')) {
          initMemoryChat();
        }
        // Domain 3 (v3.4.21): Clusters still folded into graph. Entity
        // Explorer has its own standalone sidebar entry now.
        if (typeof loadClusters === 'function') loadClusters();
        break;
      case 'entities-pane':
        if (typeof loadEntityExplorer === 'function') loadEntityExplorer();
        break;
      case 'memories-pane':
        if (typeof loadMemories === 'function') loadMemories();
        // Domain 4 (v3.4.21): Recall Lab + Timeline now live inside
        // memories-pane. Fire the timeline loader alongside so users
        // don't need a separate tab click to see the chart.
        if (typeof loadTimeline === 'function') loadTimeline();
        break;
      case 'health-pane':
        // v3.4.21 / v3.4.21 (taxonomy): Health = runtime health ONLY
        // (Daemon, Events, Agents, IDEs, Math). Governance concerns live
        // in operations-pane. IDEs folded in v3.4.21 per Varun — they're
        // connected-client state, same category as Agents.
        if (typeof loadHealthMonitor === 'function') loadHealthMonitor();
        if (typeof initEventStream === 'function') initEventStream();
        if (typeof loadEventStats === 'function') loadEventStats();
        if (typeof loadAgents === 'function') loadAgents();
        if (typeof loadIDEStatus === 'function') loadIDEStatus();
        if (typeof loadMathHealth === 'function') loadMathHealth();
        break;
      case 'operations-pane':
        // v3.4.21: data governance (Ingestion, Lifecycle, Trust, Compliance).
        if (typeof loadIngestionStatus === 'function') loadIngestionStatus();
        if (typeof loadLifecycle === 'function') loadLifecycle();
        if (typeof loadTrustDashboard === 'function') loadTrustDashboard();
        if (typeof loadCompliance === 'function') loadCompliance();
        break;
      case 'skills-pane':
        if (typeof loadSkillEvolution === 'function') loadSkillEvolution();
        break;
      case 'mesh-pane':
        if (typeof loadMeshPeers === 'function') loadMeshPeers();
        break;
      case 'settings-pane':
        if (typeof loadSettings === 'function') loadSettings();
        if (typeof loadModeSettings === 'function') loadModeSettings();
        if (typeof loadAutoSettings === 'function') loadAutoSettings();
        if (typeof updateModeUI === 'function') updateModeUI();
        break;
      case 'optimize-pane':
        if (typeof initOptimizeTab === 'function') initOptimizeTab();
        break;
    }
  }

  // ── Hash-based Routing ─────────────────────────────────────
  function handleHash() {
    var hash = window.location.hash.replace('#', '');
    if (!hash) return;
    var el = document.getElementById(hash);
    if (!el) return;
    // Only activate real tab-panes. Previously this tried to activate
    // ANY element whose id matched the hash, which broke when Domain 5
    // sub-nav anchors like ``#health-section-events`` fired — it would
    // strip ``show active`` off every real pane and leave the app blank.
    if (el.classList && el.classList.contains('tab-pane')) {
      activateTab(hash);
      return;
    }
    // Not a tab-pane — find the pane that contains this element (e.g.
    // a section inside health-pane) and activate that first, then scroll.
    var parent = el.closest && el.closest('.tab-pane');
    if (parent && parent.id) activateTab(parent.id);
    try {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      // pre-smooth-scroll browsers: no-op
    }
  }

  // ── Theme Application ───────────────────────────────────────
  function applyNgTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
    // ng-dark class ONLY in dark mode — light mode uses Bootstrap defaults
    if (theme === 'dark') {
      document.body.classList.add('ng-dark');
    } else {
      document.body.classList.remove('ng-dark');
    }
    // Update both icons (original + sidebar)
    var icon = document.getElementById('theme-icon');
    if (icon) icon.className = theme === 'dark' ? 'bi bi-moon-stars-fill' : 'bi bi-sun-fill';
    var ngIcon = document.getElementById('ng-theme-icon');
    if (ngIcon) ngIcon.className = theme === 'dark' ? 'bi bi-moon-stars-fill' : 'bi bi-sun-fill';

    // Fix inline styles that conflict with dark/light mode (setProperty needed to override inline styles)
    var themedElements = ['graph-container', 'memory-timeline-chart'];
    themedElements.forEach(function(id) {
      var el = document.getElementById(id);
      if (!el) return;
      if (theme === 'dark') {
        el.style.setProperty('background', '#0f1012', 'important');
        el.style.setProperty('background-color', '#0f1012', 'important');
        el.style.setProperty('border-color', 'rgba(255,255,255,0.06)', 'important');
      } else {
        el.style.setProperty('background', '#ffffff', 'important');
        el.style.setProperty('background-color', '#ffffff', 'important');
        el.style.setProperty('border-color', '#e5e7eb', 'important');
      }
    });

    // Also fix graph label colors for readability
    var graphLabels = document.querySelectorAll('.node-label');
    graphLabels.forEach(function(l) {
      l.style.fill = theme === 'dark' ? '#ccc' : '#333';
    });
  }

  // ── Initialize ─────────────────────────────────────────────
  function init() {
    // Respect saved theme or auto-detect
    var savedTheme = localStorage.getItem('slm-theme');
    if (!savedTheme) {
      savedTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    applyNgTheme(savedTheme);

    // Override toggleDarkMode — syncs Bootstrap theme + ng-dark class
    window.toggleDarkMode = function() {
      var current = document.documentElement.getAttribute('data-bs-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      localStorage.setItem('slm-theme', next);
      applyNgTheme(next);
    };

    // Restructure DOM
    restructureDOM();

    // v3.4.21 fix: a browser-restored URL fragment (e.g. a stale
    // ``#health-section-events`` from a previous session) used to yank
    // the user off Dashboard on refresh. Refresh should always land on
    // the pane that has ``show active`` in the HTML — Dashboard. Only
    // respond to hash changes the user triggers explicitly AFTER load.
    if (window.location.hash) {
      try {
        history.replaceState(null, '', window.location.pathname);
      } catch (e) {
        // Older browsers without replaceState: leave the hash alone
        // rather than crashing init.
      }
    }

    // Listen for subsequent (user-triggered) hash changes
    window.addEventListener('hashchange', handleHash);

    // toggleDarkMode already overridden above with theme toggle support

    // Privacy blur toggle for screen recording
    window.togglePrivacyBlur = function() {
      document.body.classList.toggle('ng-privacy-blur');
      var icon = document.getElementById('ng-privacy-icon');
      var isBlurred = document.body.classList.contains('ng-privacy-blur');
      if (icon) icon.className = isBlurred ? 'bi bi-eye' : 'bi bi-eye-slash';
    };

    // Auto-enable blur if URL has ?blur=1
    if (window.location.search.indexOf('blur=1') >= 0) {
      document.body.classList.add('ng-privacy-blur');
    }
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

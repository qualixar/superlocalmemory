// Neural Glass — Ingestion Status Tab (v3.4.4)
// Full configuration UI for non-technical users
// API: /api/adapters (GET, POST enable/disable/start/stop)

(function() {
  'use strict';

  var ADAPTER_INFO = {
    gmail: {
      icon: 'bi-envelope-fill',
      title: 'Email (Gmail)',
      description: 'Automatically ingest your emails into memory. SLM extracts key facts, decisions, and action items from your inbox.',
      howItWorks: 'Connects to Gmail via Google OAuth. Only reads — never sends or deletes emails. New emails are processed as they arrive.',
      setup: 'Requires a Google Cloud project with Gmail API enabled. SLM will guide you through OAuth setup.',
      privacy: 'All processing happens locally on your machine. Email content never leaves your device.',
      color: '#ea4335'
    },
    calendar: {
      icon: 'bi-calendar-event-fill',
      title: 'Calendar Events',
      description: 'Remember all your meetings, deadlines, and events. SLM builds a timeline of your schedule and links events to related memories.',
      howItWorks: 'Syncs with Google Calendar via OAuth. Polls for new events hourly and processes updates in real-time via webhooks.',
      setup: 'Requires Google Calendar API access. Uses the same Google Cloud project as Gmail.',
      privacy: 'Event data stays local. Only event titles, times, and descriptions are stored — not attendee details.',
      color: '#4285f4'
    },
    transcript: {
      icon: 'bi-mic-fill',
      title: 'Meeting Transcripts',
      description: 'Turn meeting transcripts into searchable memory. Extract decisions, action items, and key discussions from recorded meetings.',
      howItWorks: 'Watches a folder for .srt, .vtt, or .txt transcript files. Processes new files automatically when they appear.',
      setup: 'Point SLM to your transcripts folder (e.g., where Otter.ai or Circleback saves files).',
      privacy: 'Transcripts are processed locally. Speaker names and content stay on your machine.',
      color: '#34a853'
    }
  };

  window.loadIngestionStatus = function() {
    fetchAdapters();
  };

  function fetchAdapters() {
    fetch('/api/adapters')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        renderIngestionTab(data.adapters || []);
      })
      .catch(function() {
        renderIngestionTab([]);
      });
  }

  function renderIngestionTab(adapters) {
    // Build adapter map for easy lookup
    var adapterMap = {};
    adapters.forEach(function(a) { adapterMap[a.name] = a; });

    var el = document.getElementById('ingestion-overview');
    if (!el) return;

    // Count stats
    var enabledCount = adapters.filter(function(a) { return a.enabled; }).length;
    var runningCount = adapters.filter(function(a) { return a.running; }).length;

    // Overview cards
    el.innerHTML =
      '<div class="row g-3 mb-4">' +
        overviewCard('Available', adapters.length + ' adapters', 'bi-plug') +
        overviewCard('Enabled', enabledCount + ' of ' + adapters.length, 'bi-toggle-on') +
        overviewCard('Running', runningCount + ' active', 'bi-play-circle') +
        overviewCard('Privacy', 'All local', 'bi-shield-lock') +
      '</div>' +
      '<div style="background:var(--ng-accent-muted);border:1px solid rgba(124,106,239,0.2);border-radius:var(--ng-radius-md);padding:12px 16px;margin-bottom:24px;font-size:0.8125rem">' +
        '<strong>How Ingestion Works:</strong> Enable an adapter below → configure it → start it. ' +
        'SLM will automatically process new items and add them to your memory. ' +
        'All data stays on your machine. Disable anytime.' +
      '</div>';

    // Adapter cards
    var cardsEl = document.getElementById('ingestion-adapters');
    if (!cardsEl) return;

    var html = '';
    ['gmail', 'calendar', 'transcript'].forEach(function(name) {
      var info = ADAPTER_INFO[name];
      var adapter = adapterMap[name] || { name: name, enabled: false, running: false };
      html += renderAdapterCard(adapter, info);
    });
    cardsEl.innerHTML = html;

    // Ingestion log section
    var logEl = document.getElementById('ingestion-log');
    if (logEl) {
      if (runningCount === 0 && enabledCount === 0) {
        logEl.innerHTML =
          '<div style="text-align:center;padding:20px;color:var(--ng-text-tertiary);font-size:0.8125rem">' +
            'No adapters active. Enable one above to start building your memory from external sources.' +
          '</div>';
      } else {
        logEl.innerHTML =
          '<div style="text-align:center;padding:20px;color:var(--ng-text-tertiary);font-size:0.8125rem">' +
            'Ingestion log will show here as items are processed.' +
          '</div>';
      }
    }
  }

  function renderAdapterCard(adapter, info) {
    var isEnabled = adapter.enabled;
    var isRunning = adapter.running;
    var statusText = isRunning ? 'Running' : (isEnabled ? 'Enabled (Stopped)' : 'Disabled');
    var statusClass = isRunning ? 'ng-badge-success' : (isEnabled ? 'ng-badge-warning' : 'ng-badge-neutral');

    return '<div class="ng-glass" style="padding:24px;margin-bottom:16px">' +
      // Header
      '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px">' +
        '<div style="display:flex;align-items:center;gap:12px">' +
          '<div style="width:48px;height:48px;border-radius:var(--ng-radius-lg);background:' + info.color + '20;display:flex;align-items:center;justify-content:center">' +
            '<i class="bi ' + info.icon + '" style="font-size:1.5rem;color:' + info.color + '"></i>' +
          '</div>' +
          '<div>' +
            '<div style="font-size:1.125rem;font-weight:590">' + info.title + '</div>' +
            '<span class="ng-badge ' + statusClass + '">' + statusText + '</span>' +
          '</div>' +
        '</div>' +
        // Action buttons
        '<div style="display:flex;gap:8px">' +
          (isEnabled ?
            (isRunning ?
              '<button class="ng-btn" data-act-click="adapter-action" data-adapter="' + adapter.name + '" data-adapter-action="stop">' +
                '<i class="bi bi-stop-circle"></i> Stop</button>' :
              '<button class="ng-btn ng-btn-accent" data-act-click="adapter-action" data-adapter="' + adapter.name + '" data-adapter-action="start">' +
                '<i class="bi bi-play-circle"></i> Start</button>'
            ) +
            '<button class="ng-btn" data-act-click="adapter-action" data-adapter="' + adapter.name + '" data-adapter-action="disable" style="color:var(--ng-status-error)">' +
              '<i class="bi bi-x-circle"></i> Disable</button>' :
            '<button class="ng-btn ng-btn-accent" data-act-click="adapter-action" data-adapter="' + adapter.name + '" data-adapter-action="enable">' +
              '<i class="bi bi-power"></i> Enable</button>'
          ) +
        '</div>' +
      '</div>' +
      // Description
      '<p style="color:var(--ng-text-secondary);font-size:0.875rem;margin-bottom:12px">' + info.description + '</p>' +
      // Details grid
      '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;font-size:0.8125rem">' +
        detailBox('How it works', info.howItWorks, 'bi-gear') +
        detailBox('Setup', info.setup, 'bi-tools') +
        detailBox('Privacy', info.privacy, 'bi-shield-check') +
      '</div>' +
      // Running details
      (isRunning && adapter.pid ?
        '<div style="margin-top:12px;padding:8px 12px;background:var(--ng-status-success-bg);border-radius:var(--ng-radius-sm);font-size:0.8125rem">' +
          '<i class="bi bi-check-circle" style="color:var(--ng-status-success)"></i> ' +
          'Running as PID ' + adapter.pid +
        '</div>' : '') +
    '</div>';
  }

  function detailBox(title, text, icon) {
    return '<div style="padding:10px;background:var(--ng-bg-glass);border-radius:var(--ng-radius-sm);border:1px solid var(--ng-border-subtle)">' +
      '<div style="font-weight:590;margin-bottom:4px;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--ng-text-tertiary)">' +
        '<i class="bi ' + icon + '"></i> ' + title + '</div>' +
      '<div style="color:var(--ng-text-secondary)">' + text + '</div>' +
    '</div>';
  }

  // Adapter actions (called from buttons)
  window.adapterAction = function(name, action) {
    var btn = event.target.closest('button');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Working...';
    }

    fetch('/api/adapters/' + action, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok === false) {
        alert(data.error || data.message || 'Action failed');
      }
      // Refresh the tab
      fetchAdapters();
    })
    .catch(function(err) {
      alert('Error: ' + err.message);
      fetchAdapters();
    });
  };

  function overviewCard(label, value, icon) {
    return '<div class="col-md-3 col-6">' +
      '<div class="ng-glass" style="padding:16px;text-align:center">' +
        '<i class="bi ' + icon + '" style="font-size:1.25rem;color:var(--ng-accent);display:block;margin-bottom:8px"></i>' +
        '<div class="ng-stat-value" style="font-size:1.5rem">' + value + '</div>' +
        '<div class="ng-stat-label">' + label + '</div>' +
      '</div>' +
    '</div>';
  }
})();

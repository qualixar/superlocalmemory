// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// B2 (3.7.9): CSP-safe event delegation. Replaces inline on*= handlers
// (which force script-src 'unsafe-inline') with data-act-* attributes
// dispatched by document-level listeners. Registry keys are an allowlist
// by construction — no reflective window[name] invocation.
(function () {
  'use strict';

  // action-key -> handler(el, event). Handlers call existing globals.
  const ACTIONS = {
    // --- bespoke ---
    'scroll-into': (el) => { const t = document.getElementById(el.dataset.target); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); },
    'sigma-clear-search': () => { sigmaClearSearch(); const i = document.getElementById('sigma-search-input'); if (i) i.value = ''; },
    'export-all': (el, e) => { if (e) e.preventDefault(); exportAll(el.dataset.format); },
    'export-search-results': (el, e) => { if (e) e.preventDefault(); exportSearchResults(); },
    'set-memory-filter': (el) => setMemoryFilter(el.dataset.filter),
    'community-source': (el) => handleCommunitySourceToggle(el.dataset.source),
    // --- value (this.value) ---
    'handle-resolution-change': (el) => handleResolutionChange(el.value),
    'sigma-filter-by-category': (el) => sigmaFilterByCategory(el.value),
    'sigma-search': (el) => sigmaSearch(el.value),
    // --- no-arg ---
    'backup-learning-db': () => backupLearningDb(),
    'clear-event-stream': () => clearEventStream(),
    'compact-dry-run': () => compactDryRun(),
    'compact-execute': () => compactExecute(),
    'connect-git-hub': () => connectGitHub(),
    'connect-google-drive': () => connectGoogleDrive(),
    'copy-memory-to-clipboard': () => copyMemoryToClipboard(),
    'create-backup-now': () => createBackupNow(),
    'create-profile': () => createProfile(),
    'create-retention-policy': () => createRetentionPolicy(),
    'evaluate-tiers-now': () => evaluateTiersNow(),
    'export-backup': () => exportBackup(),
    'export-memory-as-markdown': () => exportMemoryAsMarkdown(),
    'filter-events': () => filterEvents(),
    'load-agents': () => loadAgents(),
    'load-clusters': () => loadClusters(),
    'load-compliance': () => loadCompliance(),
    'load-entity-explorer': () => loadEntityExplorer(),
    'load-graph': () => loadGraph(),
    'load-health-monitor': () => loadHealthMonitor(),
    'load-ingestion-status': () => loadIngestionStatus(),
    'load-mesh-peers': () => loadMeshPeers(),
    'load-skill-evolution': () => loadSkillEvolution(),
    'refresh-dashboard': () => refreshDashboard(),
    'reset-learning-data': () => resetLearningData(),
    'run-community-detection': () => runCommunityDetection(),
    'save-backup-config': () => saveBackupConfig(),
    'search-memories': () => searchMemories(),
    'sigma-reset-view': () => sigmaResetView(),
    'sigma-zoom-in': () => sigmaZoomIn(),
    'sigma-zoom-out': () => sigmaZoomOut(),
    'sync-cloud-now': () => syncCloudNow(),
    'toggle-dark-mode': () => toggleDarkMode(),
    // --- phase 2: handlers on JS-generated markup (dynamic args via data-*) ---
    'toggle-privacy-blur': () => togglePrivacyBlur(),
    'open-settings-tab': () => { const t = document.querySelector('[data-target=settings]'); if (t) t.click(); },
    'show-chat-panel': () => showChatPanel(),
    'show-detail-panel': () => showDetailPanel(),
    'send-chat-from-input': () => sendChatFromInput(),
    'cancel-chat': () => cancelChat(),
    'send-chat-query': (el) => sendChatQuery(el.dataset.query),
    'chat-submit-on-enter': (el, e) => { if (e && e.key === 'Enter') sendChatFromInput(); },
    'citation-click': (el, e) => { if (e) e.preventDefault(); _onCitationClick(el.dataset.factId); },
    'filter-entities': (el) => filterEntities(el.value),
    'navigate-entity-page': (el) => navigateEntityPage(parseInt(el.dataset.page, 10)),
    'show-entity-detail': (el) => showEntityDetail(el.dataset.entity),
    'recompile-entity': (el) => recompileEntity(el.dataset.entity),
    'close-entity-detail': () => { const p = document.getElementById('entity-detail-panel'); if (p) p.style.display = 'none'; },
    'trigger-evolution': () => triggerEvolution(),
    'enable-evolution': () => enableEvolution(),
    'save-evolution-config': () => saveEvolutionConfig(),
    'sigma-highlight-node': (el) => sigmaHighlightNode(el.dataset.nodeId),
    'open-memory-detail': (el) => openMemoryDetail({ id: el.dataset.memId, content: el.dataset.memContent, category: el.dataset.memCategory, importance: parseFloat(el.dataset.memImportance) }, 'graph'),
    'sigma-filter-community': (el) => sigmaFilterByCommunity(parseInt(el.dataset.communityId, 10), el.dataset.communitySrc),
    'odg-select': (el) => odgSelect(el.dataset.odgId),  // od-graph.js node inspector + citation clicks
    'load-memories-page': (el) => loadMemories(parseInt(el.dataset.page, 10)),
    'disconnect-destination': (el) => disconnectDestination(el.dataset.destId),
    'adapter-action': (el) => adapterAction(el.dataset.adapter, el.dataset.adapterAction),
  };

  const ATTR = { click: 'actClick', change: 'actChange', input: 'actInput', keydown: 'actKeydown' };

  function makeHandler(type) {
    const key = ATTR[type];
    return function (event) {
      let el = event.target;
      while (el && el.nodeType === 1) {
        if (el.dataset && el.dataset[key]) {
          const fn = ACTIONS[el.dataset[key]];
          if (typeof fn === 'function') fn(el, event);
          return;
        }
        el = el.parentElement;
      }
    };
  }

  ['click', 'change', 'input', 'keydown'].forEach((t) => document.addEventListener(t, makeHandler(t)));
})();

// od-compliance-ext.js — Compliance tab extensions v1.0
// Extends the OD Operations compliance pane with:
//   Task A: Entity-level GDPR erasure (POST /api/compliance/gdpr/erase-entity)
//   Task B: Erasure Receipts panel (GET /api/compliance/receipts + verify)
//
// This module hooks into odRenderOperations by patching window.odRenderOperations
// to inject the new panels into the compliance tab after the base render.
//
// Security: all user-supplied strings go through esc() before innerHTML insertion.
//           Confirm guard on entity erase (typed confirmation + confirmDestructive).
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

/* global window, document, fetch, Promise */
(function () {
  'use strict';

  // ─── Helpers ──────────────────────────────────────────────────────────────

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtTime(ts) {
    if (!ts) return '—';
    try {
      var d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
      return d.toLocaleString();
    } catch (e) { return String(ts); }
  }

  // ─── Task A: Entity Erase Card ────────────────────────────────────────────

  var ENTITY_ERASE_ID = 'od-entity-erase-card';

  function buildEntityEraseCard() {
    var card = document.createElement('div');
    card.id = ENTITY_ERASE_ID;
    card.className = 'card';
    card.style.marginTop = '16px';
    // Using textContent / controlled HTML (only static strings + esc'd values below)
    card.innerHTML =
      '<div class="card-head">' +
        '<h3>Entity-level GDPR erasure</h3>' +
      '</div>' +
      '<div class="card-pad">' +
        '<p class="muted" style="font-size:13px" id="od-entity-erase-summary">' +
          'Erase all facts mentioning a specific named entity (GDPR Art. 17). ' +
          'Enter the entity name exactly; you will need to confirm it. This is IRREVERSIBLE.' +
        '</p>' +
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px">' +
          '<input id="od-entity-erase-input" type="text" maxlength="200" ' +
            'placeholder="Entity name…" ' +
            'style="flex:1;min-width:180px;padding:7px 10px;border-radius:8px;' +
              'border:1px solid var(--border);background:var(--card);color:var(--fg);font-size:13px">' +
          '<button id="od-entity-erase-btn" class="btn sm" ' +
            'style="border-color:#ff6b6b;color:#ff6b6b">Erase entity…</button>' +
        '</div>' +
      '</div>';
    return card;
  }

  function wireEntityEraseCard() {
    var btn     = document.getElementById('od-entity-erase-btn');
    var input   = document.getElementById('od-entity-erase-input');
    var summary = document.getElementById('od-entity-erase-summary');
    if (!btn || !input) return;

    btn.addEventListener('click', function () {
      var entityName = (input.value || '').trim();
      if (!entityName) {
        if (summary) summary.textContent = 'Enter an entity name first.';
        return;
      }
      // Step 1: typed-confirmation prompt (same guard as profile erase)
      var typed = window.prompt(
        'This PERMANENTLY erases all facts mentioning entity "' + entityName + '".\n' +
        'This cannot be undone.\n\nType the entity name to confirm:'
      );
      if (typed === null) return; // cancelled
      if (typed !== entityName) {
        if (summary) summary.textContent = 'Erase cancelled — name did not match.';
        return;
      }
      // Step 2: confirmDestructive modal
      window.confirmDestructive({
        title: 'Erase entity',
        target: entityName,
        consequence: 'All facts mentioning this entity will be permanently erased.',
        confirmLabel: 'Erase entity',
      }).then(function (confirmed) {
        if (!confirmed) return;
        btn.disabled = true;
        btn.textContent = 'Erasing…';
        if (summary) summary.textContent = 'Erasing entity "' + esc(entityName) + '"…';
        fetch('/api/compliance/gdpr/erase-entity', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ entity_name: entityName, confirm: entityName }),
        })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (!summary) return;
            if (d && d.success) {
              var parts = [];
              if (d.facts != null)   parts.push('facts erased: ' + d.facts);
              if (d.entity != null)  parts.push('entity: ' + esc(String(d.entity)));
              summary.textContent = 'Entity erased — ' + (parts.join(', ') || 'done') + '.';
              input.value = '';
              // Reload receipts panel to reflect new receipt
              loadReceiptsPanel();
            } else {
              summary.textContent = 'Erase failed: ' + esc(String((d && d.error) || 'unknown'));
            }
          })
          .catch(function () {
            if (summary) summary.textContent = 'Erase request failed. Check console.';
          })
          .finally(function () {
            btn.disabled = false;
            btn.textContent = 'Erase entity…';
          });
      });
    });
  }

  // ─── Task B: Erasure Receipts Panel ───────────────────────────────────────

  var RECEIPTS_ID = 'od-receipts-card';

  function buildReceiptsCard() {
    var card = document.createElement('div');
    card.id = RECEIPTS_ID;
    card.className = 'card';
    card.style.marginTop = '16px';
    card.innerHTML =
      '<div class="card-head">' +
        '<h3>Erasure receipts</h3>' +
        '<div class="spacer"></div>' +
        '<button class="btn sm" id="od-receipts-refresh-btn">Refresh</button>' +
      '</div>' +
      '<div class="card-pad">' +
        '<table class="tbl" id="od-receipts-tbl">' +
          '<thead><tr>' +
            '<th>ID</th>' +
            '<th>Type</th>' +
            '<th>Subject</th>' +
            '<th>State</th>' +
            '<th>Facts</th>' +
            '<th>Requested</th>' +
            '<th></th>' +
          '</tr></thead>' +
          '<tbody id="od-receipts-body">' +
            '<tr><td colspan="7" class="dim" style="text-align:center;padding:20px">Loading…</td></tr>' +
          '</tbody>' +
        '</table>' +
      '</div>';
    return card;
  }

  function loadReceiptsPanel() {
    var tbody = document.getElementById('od-receipts-body');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="7" class="dim" style="text-align:center;padding:16px">Loading…</td></tr>';
    fetch('/api/compliance/receipts?limit=50')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.available) {
          tbody.innerHTML = '<tr><td colspan="7" class="dim" style="text-align:center;padding:16px">Receipts unavailable.</td></tr>';
          return;
        }
        var receipts = d.receipts || [];
        if (receipts.length === 0) {
          tbody.innerHTML = '<tr><td colspan="7" class="dim" style="text-align:center;padding:16px">No erasure receipts yet.</td></tr>';
          return;
        }
        tbody.innerHTML = receipts.map(function (r) {
          var shortId = esc(String(r.erasure_id || '').slice(0, 8)) + '…';
          var stateClass = r.state === 'COMPLETE' ? 'ok' : 'danger';
          var allOk = r.all_erased ? '<span class="badge ok">all erased</span>' : '<span class="badge warn">partial</span>';
          return (
            '<tr>' +
              '<td class="mono dim" style="font-size:11px" title="' + esc(r.erasure_id || '') + '">' + shortId + '</td>' +
              '<td><span class="badge neutral">' + esc(r.subject_type || '') + '</span></td>' +
              '<td class="dim" style="font-size:12px;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' +
                esc(r.subject_id || '') +
              '</td>' +
              '<td><span class="badge ' + stateClass + '">' + esc(r.state || '') + '</span></td>' +
              '<td class="num dim">' + esc(String(r.fact_count || 0)) + '</td>' +
              '<td class="dim" style="font-size:11px">' + esc(fmtTime(r.requested_at)) + '</td>' +
              '<td><button class="btn sm ghost" data-verify-id="' + esc(r.erasure_id || '') + '">Verify</button></td>' +
            '</tr>'
          );
        }).join('');
        // Wire verify buttons
        tbody.querySelectorAll('[data-verify-id]').forEach(function (btn) {
          btn.addEventListener('click', function () {
            var eid = btn.getAttribute('data-verify-id');
            btn.disabled = true;
            btn.textContent = '…';
            fetch('/api/compliance/receipts/' + encodeURIComponent(eid) + '/verify')
              .then(function (r) { return r.ok ? r.json() : null; })
              .then(function (d) {
                if (d && d.verified === true) {
                  btn.textContent = 'Verified ✓';
                  btn.style.color = 'var(--ok)';
                } else if (d && d.verified === false) {
                  btn.textContent = 'Tampered!';
                  btn.style.color = 'var(--danger)';
                } else {
                  btn.textContent = 'Error';
                }
                btn.disabled = false;
              })
              .catch(function () { btn.textContent = 'Error'; btn.disabled = false; });
          });
        });
      })
      .catch(function () {
        var tb = document.getElementById('od-receipts-body');
        if (tb) tb.innerHTML = '<tr><td colspan="7" class="dim" style="text-align:center;padding:16px">Failed to load receipts.</td></tr>';
      });
  }

  function wireReceiptsPanel() {
    var refreshBtn = document.getElementById('od-receipts-refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', loadReceiptsPanel);
    }
    loadReceiptsPanel();
  }

  // ─── Injection Hook ───────────────────────────────────────────────────────

  /**
   * Inject the two new panels into the compliance tab section.
   * Called after odRenderOperations() has built and populated the DOM.
   */
  function injectCompliancePanels(container) {
    // Find the compliance tabpane
    var pane = container && container.querySelector('[data-od-pane="compliance"]');
    if (!pane) return;

    // Remove any stale injected panels from a previous render
    var oldErase = pane.querySelector('#' + ENTITY_ERASE_ID);
    if (oldErase) oldErase.remove();
    var oldReceipts = pane.querySelector('#' + RECEIPTS_ID);
    if (oldReceipts) oldReceipts.remove();

    // Append the two new cards at the bottom of the compliance tab
    pane.appendChild(buildEntityEraseCard());
    pane.appendChild(buildReceiptsCard());

    // Wire interactions
    wireEntityEraseCard();
    wireReceiptsPanel();
  }

  // ─── Patch odRenderOperations ─────────────────────────────────────────────

  /**
   * Wait for od-operations.js to define window.odRenderOperations, then wrap it
   * so our panels are injected after every render. Uses polling with a cap —
   * no unbounded loops.
   */
  function patchOdRenderOperations() {
    var base = window.odRenderOperations;
    if (typeof base !== 'function') return false;
    window.odRenderOperations = function (container) {
      base(container);
      // od-operations.js renders asynchronously (Promise.all inside).
      // Give it time to populate the DOM before injecting our panels.
      setTimeout(function () { injectCompliancePanels(container); }, 600);
    };
    return true;
  }

  // Retry for up to 3s in case od-operations.js hasn't defined the function yet.
  var _attempts = 0;
  function tryPatch() {
    if (patchOdRenderOperations()) return;
    _attempts += 1;
    if (_attempts < 30) setTimeout(tryPatch, 100);
  }

  // Also inject into the currently active operations pane on DOMContentLoaded
  document.addEventListener('DOMContentLoaded', function () {
    tryPatch();
    // If the operations pane is already in the DOM and populated, inject now
    var pane = document.getElementById('operations-pane');
    if (pane && pane.querySelector('[data-od-pane="compliance"]')) {
      injectCompliancePanels(pane);
    }
  });

}());

// Extracted od-memories.js feature scope; initialized once with the pane helpers.
(function () {
  'use strict';
  var ctx;
  var _esc, _fmt, _ago, _imp, _scoreCls, _catCls, _normSearch, _sharedToStr, _getMutToken, _injectCSS, render, _scaffold, _allScaffold, _tlScaffold, _loading, _recallStarter, _switchTab, _loadCatCounts, _rebuildCatBar, _loadMem, _renderTable, _renderPag, _doSort, _doSearch;
  window.SLMMemoryDrawer = function (helpers) {
    ctx = helpers;
    ({ _esc, _fmt, _ago, _imp, _scoreCls, _catCls, _normSearch, _sharedToStr, _getMutToken, _injectCSS, render, _scaffold, _allScaffold, _tlScaffold, _loading, _recallStarter, _switchTab, _loadCatCounts, _rebuildCatBar, _loadMem, _renderTable, _renderPag, _doSort, _doSearch } = ctx);
    return { _ensureDrawer:_ensureDrawer, _openDrawer:_openDrawer, _closeDrawer:_closeDrawer, _readWriteResponse:_readWriteResponse, _showWriteFailure:_showWriteFailure, _loadFacts:_loadFacts, _loadCodeLinks:_loadCodeLinks, _startEdit:_startEdit, _startDel:_startDel };
  };
  function _ensureDrawer() {
    if (document.getElementById('od-mem-drawer')) return;
    var scrim = document.createElement('div');
    scrim.id = 'od-mem-drawer-scrim';
    scrim.className = 'od-drawer-scrim';
    scrim.dataset.odAct = 'close-drawer';
    document.body.appendChild(scrim);
    var drawer = document.createElement('aside');
    drawer.id = 'od-mem-drawer';
    drawer.className = 'od-drawer';
    document.body.appendChild(drawer);
  }

  var _drawerOpener = null;
  function _openDrawer(idx) {
    var mem = ctx.state.memories[idx];
    if (!mem) return;
    _drawerOpener = document.activeElement;
    _ensureDrawer();
    var drawer = document.getElementById('od-mem-drawer');
    var scrim  = document.getElementById('od-mem-drawer-scrim');
    if (!drawer || !scrim) return;

    var imp      = _imp(mem.importance);
    var kind     = mem.memory_kind || mem.category || 'semantic';
    var catLabel = mem.memory_kind_label || mem.category || 'semantic';
    var scorePct = Math.round((parseFloat(mem.importance) || 0) * 100);

    drawer.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:flex-start">' +
        '<div>' +
          '<div style="font-size:12px;color:var(--fg-2)">Memory</div>' +
          '<h2 style="font-size:19px;margin-top:2px">' +
            _esc('#' + (mem.id || '').substring(0, 12)) + '</h2>' +
        '</div>' +
        '<button class="btn icon ghost" aria-label="Close memory" data-od-act="close-drawer">✕</button>' +
      '</div>' +
      '<div class="od-prov" style="font-size:14px;line-height:1.7">' +
        _esc(mem.content || '') +
      '</div>' +
      '<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">' +
        '<span class="badge ' + _esc(_catCls(kind)) + '">' + _esc(catLabel) + '</span>' +
        '<span class="badge neutral">' + _esc(mem.project_name || 'no project') + '</span>' +
        '<span class="badge ' + (mem.scope === 'global' ? 'success' : mem.scope === 'shared' ? 'warn' : 'neutral') +
          '">' + _esc(mem.scope || 'personal') + '</span>' +
      '</div>' +
      // Scope & sharing control (C2) — set personal/shared/global from the UI
      '<h3 style="font-size:13px;margin:20px 0 8px">Scope &amp; sharing</h3>' +
      '<div class="od-prov" style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">' +
        '<select id="od-scope-sel" style="padding:6px 8px;border-radius:6px;background:var(--card-2);color:var(--fg);border:1px solid var(--border);font-size:13px">' +
          ['personal', 'shared', 'global'].map(function (s) {
            return '<option value="' + s + '"' + ((mem.scope || 'personal') === s ? ' selected' : '') + '>' + s + '</option>';
          }).join('') +
        '</select>' +
        '<input id="od-scope-shared" placeholder="share with profiles (comma-sep)" ' +
          'value="' + _esc(_sharedToStr(mem.shared_with)) + '" ' +
          'style="flex:1;min-width:140px;padding:6px 8px;border-radius:6px;background:var(--card-2);color:var(--fg);border:1px solid var(--border);font-size:13px">' +
        '<button data-od-act="set-scope" data-mid="' + _esc(mem.id) + '" ' +
          'style="padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);cursor:pointer;font-size:13px">Apply</button>' +
      '</div>' +
      '<h3 style="font-size:13px;margin:20px 0 8px">Why this was remembered</h3>' +
      '<div class="od-prov">' +
        '<div style="font-size:13px;color:var(--fg-2);line-height:1.6">' +
          'Importance: <b>' + imp + '/10</b>. ' +
          'Score: <b>' + scorePct + '%</b>. ' +
          'Recalled <b>' + _esc(String(mem.access_count || 0)) + '</b> time' +
          (mem.access_count === 1 ? '' : 's') + '. ' +
          'Stored ' + _esc(_ago(mem.created_at)) + '.' +
        '</div>' +
        // TODO: /api/memories/{id}/score-breakdown — Semantic/BM25/EntityGraph/Temporal
        //       decomposition is not yet exposed by the daemon; show when available.
      '</div>' +
      '<div id="od-drawer-truth"></div>' +
      '<h3 style="font-size:13px;margin:20px 0 8px">Atomic facts</h3>' +
      '<div id="od-drawer-facts" style="color:var(--fg-2);font-size:13px">Loading facts…</div>' +
      '<div id="od-drawer-code"></div>' +
      '<div style="display:flex;gap:8px;margin-top:20px;padding-top:12px;border-top:1px solid var(--border)">' +
        '<button data-od-act="edit-mem" data-mid="' + _esc(mem.id) + '" style="padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);cursor:pointer;font-size:13px">Edit</button>' +
        '<button data-od-act="del-mem" data-mid="' + _esc(mem.id) + '" style="padding:6px 14px;border:1px solid var(--danger);border-radius:6px;background:transparent;color:var(--danger);cursor:pointer;font-size:13px">Delete</button>' +
      '</div><div id="od-mem-act" style="margin-top:10px"></div>';

    drawer.classList.add('open');
    scrim.classList.add('on');
    // A row in this pane is an ATOMIC FACT, not a source memory: /api/memories
    // reports total 3603 on a store holding 830 memories and 3603 facts, and
    // every row id resolves through /api/facts/{id}. The row carries its parent
    // separately as memory_id.
    //
    // "Atomic facts" therefore needs the PARENT id — it lists the other facts
    // extracted from the same source memory. Passing mem.id asked
    // /api/memories/{fact_id}/facts, which has no children to return, so this
    // section read "No atomic facts recorded for this memory" for every memory
    // on every store. Not an empty state: the wrong identifier.
    if (mem.memory_id) {
      _loadFacts(mem.memory_id);
    } else {
      // No parent row: this record was stored directly as a fact rather than
      // extracted from a longer memory (198 of 3,608 on a real store). Saying
      // "no atomic facts recorded" there is wrong in a way that matters — it
      // reads as data missing, when the fact IS the record.
      var fx = document.getElementById('od-drawer-facts');
      if (fx) fx.textContent = 'Stored directly — this record is itself the atomic fact.';
    }
    // Code links hang off the FACT, so this one takes the row id.
    if (mem.id) _loadCodeLinks(mem.id);
    // 4.2.30: when it was true, open disputes, a changed source file (od-truth.js).
    if (mem.id && window.SLMTruth && window.SLMTruth.detail) {
      window.SLMTruth.detail(mem.id, document.getElementById('od-drawer-truth'));
    }
    drawer.querySelector('[data-od-act="close-drawer"]').focus();
  }

  function _closeDrawer() {
    var d = document.getElementById('od-mem-drawer');
    var s = document.getElementById('od-mem-drawer-scrim');
    if (d) d.classList.remove('open');
    if (s) s.classList.remove('on');
    if (_drawerOpener && _drawerOpener.isConnected) _drawerOpener.focus();
  }

  function _readWriteResponse(response) {
    return response.json().catch(function () { return {}; }).then(function (body) {
      if (response.ok && body.success !== false) return body;
      var detail = body.detail;
      var message = typeof detail === 'string' ? detail : detail && detail.message;
      throw new Error(message || body.message || body.error || ('Request failed (' + response.status + ')'));
    });
  }

  function _showWriteFailure(error, original) {
    var drawer = document.getElementById('od-mem-drawer');
    var existing = document.getElementById('od-memory-write-error');
    if (existing) existing.remove();
    var status = document.createElement('div');
    status.id = 'od-memory-write-error'; status.setAttribute('role', 'status');
    status.textContent = error.message + ' Your changes are still here.';
    var retry = document.createElement('button');
    retry.type = 'button'; retry.className = 'btn'; retry.textContent = 'Retry';
    retry.dataset.odAct = 'retry-memory-write';
    retry.addEventListener('click', function () { status.remove(); original.click(); });
    status.appendChild(retry);
    if (drawer) drawer.appendChild(status);
  }

  function _loadFacts(memId) {
    var el = document.getElementById('od-drawer-facts');
    if (!el) return;
    fetch('/api/memories/' + encodeURIComponent(memId) + '/facts')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var facts = d.facts || [];
        if (facts.length === 0) {
          el.textContent = 'No atomic facts recorded for this memory.'; return;
        }
        el.innerHTML = facts.map(function (f) {
          return '<div class="od-prov" style="margin-bottom:8px">' +
            '<span class="badge neutral" style="font-size:10px">' +
              _esc(f.fact_type || 'fact') +
            '</span>' +
            '<div style="font-size:13px;line-height:1.6;margin-top:4px">' +
              _esc((f.content || '').substring(0, 200)) +
            '</div></div>';
        }).join('');
      })
      .catch(function () { if (el) el.textContent = 'Could not load facts.'; });
  }

  /* Code this memory refers to (4.0.8).
   *
   * Renders into the memory drawer — the panel actually used to inspect a
   * memory. 4.0.7 put this in js/fact-detail.js, which only binds to
   * `.fact-result-item` rows in the search-results view, so the feature was
   * invisible to anyone browsing Memories.
   *
   * KEYED ON THE ROW'S OWN ID, which is an ATOMIC FACT id. The Memories pane
   * lists atomic facts — /api/memories returns total 3603 on a store with 830
   * memories and 3603 facts, and each row id resolves through /api/facts/{id}.
   * (That mismatch is also why the drawer's "Atomic facts" block always reads
   * "No atomic facts recorded": it asks /api/memories/{fact_id}/facts, which
   * has no children to return. Left alone here — separate defect, separate fix.)
   *
   * Reads /api/facts/{id}, which serves links from code_graph.db. Recall never
   * opens that database, so nothing here can affect retrieval.
   *
   * textContent throughout: every value is a qualified name or path read out of
   * an indexed repository, i.e. untrusted input as far as this panel goes.
   */
  function _loadCodeLinks(factId) {
    var host = document.getElementById('od-drawer-code');
    if (!host || !factId) return;
    fetch('/api/facts/' + encodeURIComponent(factId))
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        var links = (d && d.code_links) || [];
        if (!links.length) return;   // silent when there is nothing to show

        var wrap = document.createElement('div');
        wrap.style.cssText = 'margin-top:6px;padding-top:6px;border-top:1px dashed var(--border)';

        var label = document.createElement('div');
        label.style.cssText = 'font-size:11px;color:var(--fg-2);margin-bottom:4px';
        label.textContent = 'Code referenced (' + links.length + ')';
        wrap.appendChild(label);

        links.slice(0, 6).forEach(function (l) {
          var row = document.createElement('div');
          row.style.cssText = 'font-size:12px;line-height:1.5';

          var code = document.createElement('code');
          code.style.cssText = 'color:var(--violet)';
          code.textContent = l.qualified_name || l.name || '?';
          row.appendChild(code);

          if (l.kind) {
            var kind = document.createElement('span');
            kind.style.cssText = 'color:var(--fg-2);font-size:10px;margin-left:6px';
            kind.textContent = l.kind;
            row.appendChild(kind);
          }
          // A stale link points at code that has moved or gone. Saying so is the
          // entire value of tracking staleness.
          if (l.is_stale) {
            var stale = document.createElement('span');
            stale.className = 'badge';
            stale.style.cssText = 'font-size:9px;margin-left:6px';
            stale.textContent = 'code changed';
            row.appendChild(stale);
          }
          wrap.appendChild(row);
        });

        if (links.length > 6) {
          var more = document.createElement('div');
          more.style.cssText = 'font-size:11px;color:var(--fg-2);margin-top:2px';
          more.textContent = '… and ' + (links.length - 6) + ' more';
          wrap.appendChild(more);
        }
        host.appendChild(wrap);
      })
      .catch(function () { /* display extra: never surface an error here */ });
  }

  function _startEdit(mid) { // show inline edit form in drawer
    var m = ctx.state.memories.filter(function (x) { return x.id === mid; })[0];
    var el = document.getElementById('od-mem-act'); if (!el || !m) return;
    el.innerHTML = '<textarea id="od-ea" rows="4" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);font-size:13px">' + _esc(m.content) + '</textarea><button data-od-act="save-edit" data-mid="' + _esc(mid) + '" style="display:block;margin-top:6px;padding:6px 14px;border:none;border-radius:6px;background:var(--violet);color:#fff;cursor:pointer">Save</button>';
  }
  function _startDel(mid) { // show delete confirmation in drawer
    var el = document.getElementById('od-mem-act'); if (!el) return;
    el.innerHTML = '<p style="font-size:13px;color:var(--fg-2);margin-bottom:10px">Permanently delete this memory? This cannot be undone.</p><button data-od-act="confirm-del" data-mid="' + _esc(mid) + '" style="padding:6px 14px;border-radius:6px;background:var(--danger);color:#fff;border:none;cursor:pointer;margin-right:8px">Confirm delete</button><button data-od-act="cancel-act" style="padding:6px 14px;border-radius:6px;background:var(--card-2);border:1px solid var(--border);color:var(--fg);cursor:pointer">Cancel</button>';
  }

  // ── Event delegation ─────────────────────────────────────────────────────────

}());

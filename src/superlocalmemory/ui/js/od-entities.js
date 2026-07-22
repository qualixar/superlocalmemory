// od-entities.js — Entity Explorer
// OD design port for SuperLocalMemory dashboard.
//
// Render target  : document.getElementById('entities-pane')
// Public API     : window.odRenderEntities(container)
// Auto-init      : fires on DOMContentLoaded if #entities-pane exists
//
// CSP-safe: no inline on* handlers; all events wired via delegation (data-od-act).
// XSS-safe: every API string passes through _esc() before DOM insertion.
//           nosemgrep: innerHTML — all dynamic values escaped via _esc()
//
// Endpoints verified against live daemon at http://127.0.0.1:8765
//   GET /api/entity/list?limit=N&offset=N
//       → { entities[], total, limit, offset }
//       entity fields: entity_id, name, type, fact_count, first_seen, last_seen,
//                      summary_preview, has_compiled_truth, confidence, last_compiled_at
//   GET /api/entity/{name}
//       → { entity_name, entity_type, compiled_truth, knowledge_summary,
//            timeline[], source_fact_ids[], last_compiled_at, confidence }
//   POST /api/entity/{name}/recompile  (write — not called here, shown disabled)
//
// CRIT fixes applied:
//   1. XSS — _esc() wraps every interpolated string
//   2. timeline empty-state — timeline:[] in real data; shows graceful placeholder
//   3. Tab/filter state — all 5 type filters wired, no filter leaves an invisible state

(function () {
  'use strict';

  // ── Utilities ──────────────────────────────────────────────────────────────

  // CRIT-1: Local escapeHtml to prevent XSS if core.js loads after this module.
  function _esc(s) {
    if (typeof escapeHtml === 'function') return escapeHtml(s);
    var d = document.createElement('div');
    d.textContent = String(s == null ? '' : s);
    return d.innerHTML;
  }

  function _fmt(iso) {
    if (!iso) return '—';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso) : d.toLocaleDateString();
  }

  function _ago(iso) {
    if (!iso) return '';
    var s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 0) s = 0;
    if (s < 60)    return Math.floor(s) + 's ago';
    if (s < 3600)  return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
  }

  // Map entity type → CSS variable color name (no hex — design-system tokens only)
  function _typeVar(type) {
    var MAP = {
      person:       '--violet',
      concept:      '--warn',
      organization: '--cyan',
      skill:        '--ok',
      event:        '--ok',
      place:        '--cyan',
    };
    return 'var(' + (MAP[String(type).toLowerCase()] || '--fg-2') + ')';
  }

  // Map entity type → badge class
  function _typeCls(type) {
    var MAP = {
      person:       'violet',
      concept:      'warn',
      organization: 'cyan',
      skill:        'ok',
      event:        'ok',
    };
    return MAP[String(type).toLowerCase()] || 'neutral';
  }

  // Map entity type → Bootstrap Icon class
  function _typeIcon(type) {
    var MAP = {
      person:       'bi-person',
      concept:      'bi-lightbulb',
      organization: 'bi-building',
      skill:        'bi-lightning-charge',
      event:        'bi-calendar-event',
      place:        'bi-geo-alt',
    };
    return MAP[String(type).toLowerCase()] || 'bi-circle';
  }

  // ── CSS (injected once) ────────────────────────────────────────────────────

  var _cssInjected = false;
  function _injectCSS() {
    if (_cssInjected) return;
    _cssInjected = true;
    var s = document.createElement('style');
    s.dataset.odModule = 'entities';
    s.textContent =
      '.od-tl{position:relative;padding-left:8px;}' +
      '.od-tl-item{display:flex;gap:14px;padding:0 0 20px 14px;position:relative;' +
        'border-left:2px solid var(--border);margin-left:5px;}' +
      '.od-tl-item:last-child{border-left-color:transparent;padding-bottom:0;}' +
      '.od-tl-dot{position:absolute;left:-7px;top:2px;width:12px;height:12px;' +
        'border-radius:50%;border:2px solid var(--card);}' +
      '.od-ent-row{display:flex;align-items:center;gap:12px;padding:12px 16px;' +
        'border-bottom:1px solid var(--border);cursor:pointer;transition:background .12s;}' +
      '.od-ent-row:last-child{border-bottom:0;}' +
      '.od-ent-row:hover{background:var(--card-2);}' +
      '.od-ent-row.active{background:var(--violet-soft);}' +
      '.od-ent-av{width:32px;height:32px;border-radius:9px;display:grid;place-items:center;' +
        'font-weight:700;font-size:13px;color:#fff;flex-shrink:0;}';
    document.head.appendChild(s);
  }

  // ── Module state ───────────────────────────────────────────────────────────

  var _st = {
    rootId:    null,
    entities:  [],
    total:     0,
    page:      0,
    pageSize:  50,
    typeFilter: 'all',
    search:    '',
    selected:  null, // entity name
  };

  // ── Main entry ─────────────────────────────────────────────────────────────

  function render(container) {
    if (!container) return;
    _injectCSS();
    var id = 'od-ent-' + Math.random().toString(36).slice(2, 8);
    _st = Object.assign({}, _st, { rootId: id, page: 0, typeFilter: 'all', selected: null });
    container.innerHTML = _scaffold(id);
    _wire(container, id);
    _loadList(id);
  }

  // ── Scaffold ───────────────────────────────────────────────────────────────

  function _scaffold(id) {
    return (
      '<div id="' + id + '">' +
        '<div class="page-head">' +
          '<h2>Canonical entities</h2>' +
          '<p style="color:var(--fg-2);margin-top:5px">' +
            'People, orgs, concepts and events resolved from your facts — ' +
            'each with a temporal timeline of how its knowledge changed over time.' +
          '</p>' +
        '</div>' +

        '<div style="display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start">' +

          // Left panel: list
          '<div class="card" style="position:sticky;top:86px">' +
            '<div class="card-head">' +
              '<h3>Entities <span style="font-size:12px;font-weight:400;' +
                'color:var(--fg-2)" id="' + id + '-count"></span></h3>' +
            '</div>' +

            // Search
            '<div style="padding:12px 16px;border-bottom:1px solid var(--border)">' +
              '<label class="search" style="width:100%" for="' + id + '-search">' +
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
                  'stroke-width="1.8" width="15" height="15">' +
                  '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>' +
                '<input id="' + id + '-search" type="text" placeholder="Search entities…" ' +
                  'data-od-act="ent-search">' +
              '</label>' +
            '</div>' +

            // Type filter chips
            '<div style="display:flex;gap:6px;padding:10px 12px;flex-wrap:wrap;' +
              'border-bottom:1px solid var(--border)">' +
              _typeChips(id, 'all') +
            '</div>' +

            // Entity list
            '<div id="' + id + '-list" style="max-height:62vh;overflow-y:auto">' +
              '<div style="padding:40px;text-align:center;color:var(--fg-2);font-size:13px">' +
                'Loading entities…' +
              '</div>' +
            '</div>' +

            // Pagination
            '<div id="' + id + '-pg" style="padding:10px 16px;border-top:1px solid var(--border)"></div>' +
          '</div>' +

          // Right panel: detail
          '<div id="' + id + '-detail">' +
            '<div style="padding:60px;text-align:center;color:var(--fg-2)">' +
              '<i class="bi bi-person-badge" style="font-size:3rem;opacity:0.3;display:block;margin-bottom:12px"></i>' +
              '<div>Select an entity to explore its knowledge and temporal timeline</div>' +
            '</div>' +
          '</div>' +

        '</div>' +
      '</div>'
    );
  }

  // Match design exactly: All / Person / Org / Concept / Event (5 chips).
  // data-type carries the API type string; label is display-only.
  // TODO: Live data returns type="concept" for most entities; Person (5) exists,
  //       but "organization" does not appear yet — clicking Org shows empty state
  //       until the daemon's entity-extraction model assigns that type.
  var TYPE_CHIPS = [
    { val: 'all',          label: 'All' },
    { val: 'person',       label: 'Person' },
    { val: 'organization', label: 'Org' },
    { val: 'concept',      label: 'Concept' },
    { val: 'event',        label: 'Event' },
  ];

  function _typeChips(id, active) {
    return TYPE_CHIPS.map(function (tc) {
      return '<div class="chip' + (tc.val === active ? ' on' : '') + '" ' +
        'data-od-act="type-filter" data-type="' + _esc(tc.val) + '">' +
        _esc(tc.label) +
      '</div>';
    }).join('');
  }

  // ── Entity list fetch & render ─────────────────────────────────────────────

  function _loadList(id) {
    var offset = _st.page * _st.pageSize;
    var url = '/api/entity/list?limit=' + _st.pageSize + '&offset=' + offset;
    var listEl = document.getElementById(id + '-list');
    if (listEl) {
      listEl.innerHTML = '<div style="padding:40px;text-align:center;' +
        'color:var(--fg-2);font-size:13px">Loading entities…</div>';
    }

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        _st = Object.assign({}, _st, {
          entities: d.entities || [],
          total: d.total || 0,
        });
        _renderList(id);
        _renderPag(id, d.total || 0, offset);
        // Auto-select first entity on initial load — matches design default state
        if (!_st.selected && d.entities && d.entities.length > 0) {
          _selectEntity(id, d.entities[0].name);
        }
      })
      .catch(function (err) {
        var el = document.getElementById(id + '-list');
        if (el) {
          el.innerHTML = '<div style="padding:24px;text-align:center;' +
            'color:var(--danger);font-size:13px">Failed: ' + _esc(err.message) + '</div>';
        }
      });
  }

  function _renderList(id) {
    var q      = _st.search.toLowerCase();
    var tf     = _st.typeFilter;
    var all    = _st.entities;

    var filtered = all.filter(function (e) {
      var typeOk = tf === 'all' || e.type === tf;
      var searchOk = !q ||
        (e.name || '').toLowerCase().indexOf(q) >= 0 ||
        (e.type || '').toLowerCase().indexOf(q) >= 0 ||
        (e.summary_preview || '').toLowerCase().indexOf(q) >= 0;
      return typeOk && searchOk;
    });

    var cntEl = document.getElementById(id + '-count');
    if (cntEl) cntEl.textContent = filtered.length + ' of ' + _st.total.toLocaleString();

    var listEl = document.getElementById(id + '-list');
    if (!listEl) return;

    if (filtered.length === 0) {
      listEl.innerHTML = '<div style="padding:40px;text-align:center;' +
        'color:var(--fg-2);font-size:13px">' +
        (q || tf !== 'all' ? 'No entities match this filter.' :
          'No entities found. Entity extraction runs during consolidation.') +
        '</div>';
      return;
    }

    listEl.innerHTML = filtered.map(function (e) {
      var av     = (e.name || '?').charAt(0).toUpperCase();
      var color  = _typeVar(e.type);
      var isAct  = _st.selected === e.name;
      return '<div class="od-ent-row' + (isAct ? ' active' : '') + '" ' +
          'data-od-act="select-entity" data-ename="' +
          _esc(e.name).replace(/"/g, '&quot;') + '">' +
        '<div class="od-ent-av" style="background:' + color + '">' + _esc(av) + '</div>' +
        '<div style="flex:1;min-width:0">' +
          '<div style="font-weight:600;white-space:nowrap;overflow:hidden;' +
            'text-overflow:ellipsis">' + _esc(e.name) + '</div>' +
          '<div style="font-size:11.5px;color:var(--fg-3)">' +
            _esc(e.type) + ' · ' + _esc(String(e.fact_count || 0)) + ' facts' +
          '</div>' +
        '</div>' +
      '</div>';
    }).join('');
  }

  function _renderPag(id, total, offset) {
    var el = document.getElementById(id + '-pg');
    if (!el) return;
    var pg   = Math.floor(offset / _st.pageSize);
    var last = Math.max(0, Math.ceil(total / _st.pageSize) - 1);
    if (last <= 0) { el.innerHTML = ''; return; }

    el.innerHTML =
      '<div style="display:flex;justify-content:center;gap:8px">' +
        '<button class="btn sm" ' + (pg <= 0 ? 'disabled' :
          'data-od-act="list-pg" data-page="' + (pg - 1) + '"') + '>← Prev</button>' +
        '<span style="padding:6px;font-size:12px;color:var(--fg-2)">' +
          (pg + 1) + '/' + (last + 1) + '</span>' +
        '<button class="btn sm" ' + (pg >= last ? 'disabled' :
          'data-od-act="list-pg" data-page="' + (pg + 1) + '"') + '>Next →</button>' +
      '</div>';
  }

  // ── Entity detail fetch & render ───────────────────────────────────────────

  function _selectEntity(id, name) {
    _st = Object.assign({}, _st, { selected: name });
    // Refresh active state in list
    _renderList(id);

    var detEl = document.getElementById(id + '-detail');
    if (!detEl) return;
    detEl.innerHTML = '<div style="padding:60px;text-align:center;color:var(--fg-2)">' +
      '<div style="font-size:13px">Loading ' + _esc(name) + '…</div></div>';

    fetch('/api/entity/' + encodeURIComponent(name))
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) { _renderDetail(id, d); })
      .catch(function (err) {
        if (detEl) {
          detEl.innerHTML = '<div style="padding:24px;color:var(--danger);font-size:13px">' +
            'Could not load entity: ' + _esc(err.message) + '</div>';
        }
      });
  }

  function _renderDetail(id, d) {
    var detEl = document.getElementById(id + '-detail');
    if (!detEl) return;

    // Find entity in list for metadata not in detail endpoint
    var listEnt    = _st.entities.find(function (e) { return e.name === d.entity_name; }) || {};
    var type       = d.entity_type || listEnt.type || 'concept';
    var color      = _typeVar(type);
    var badgeCls   = _typeCls(type);
    var factsCount = listEnt.fact_count || (d.source_fact_ids || []).length;
    var av         = (d.entity_name || '?').charAt(0).toUpperCase();
    // summary: prefer full knowledge_summary; truncate to ~280 chars for inline display
    // (matches design's 1–2-sentence paragraph density; full text in compiled-truth card)
    var summaryRaw = d.knowledge_summary || listEnt.summary_preview || '';
    var summary    = summaryRaw.length > 280 ? summaryRaw.substring(0, 280) + '…' : summaryRaw;

    detEl.innerHTML =
      // Header card: avatar · name · type+facts badges · inline summary · dates row
      '<div class="card">' +
        '<div class="card-pad">' +
          '<div style="display:flex;align-items:center;gap:14px">' +
            '<div class="od-ent-av" style="width:48px;height:48px;border-radius:11px;' +
              'font-size:18px;background:' + color + '">' + _esc(av) + '</div>' +
            '<div>' +
              '<h2 style="font-size:20px">' + _esc(d.entity_name || '') + '</h2>' +
              '<div style="margin-top:4px">' +
                '<span class="badge ' + _esc(badgeCls) + '">' + _esc(type) + '</span> ' +
                '<span class="badge neutral">' + _esc(String(factsCount)) + ' facts</span>' +
              '</div>' +
            '</div>' +
          '</div>' +
          // Inline summary (matches design's <p class="muted"> placement)
          (summary
            ? '<p class="muted" style="margin-top:14px;line-height:1.6">' + _esc(summary) + '</p>'
            : '') +
          // Dates row
          '<div style="display:flex;gap:26px;margin-top:16px;flex-wrap:wrap">' +
            _dateField('First seen', listEnt.first_seen) +
            _dateField('Last seen',  listEnt.last_seen)  +
            _dateField('Compiled',   d.last_compiled_at) +
          '</div>' +
        '</div>' +
      '</div>' +

      // Compiled truth (extra card — not in design mock but valuable live data)
      (d.compiled_truth ? _summaryCard('Compiled Truth', d.compiled_truth) : '') +

      // Temporal timeline
      '<div class="card" style="margin-top:16px">' +
        '<div class="card-head">' +
          '<h3>Temporal timeline</h3>' +
          '<span class="sub">how this entity\'s facts changed · 3-date model</span>' +
        '</div>' +
        '<div class="card-pad">' +
          _renderTimeline(d) +
        '</div>' +
      '</div>';
  }

  function _dateField(label, iso) {
    return '<div>' +
      '<div class="dim" style="font-size:11px">' + _esc(label) + '</div>' +
      '<div class="mono">' + _esc(_fmt(iso)) + '</div>' +
    '</div>';
  }

  function _summaryCard(title, text) {
    return '<div class="card" style="margin-top:16px">' +
      '<div class="card-head"><h3 style="font-size:13.5px">' + _esc(title) + '</h3></div>' +
      '<div class="card-pad" style="font-size:13.5px;color:var(--fg-2);line-height:1.7;' +
        'white-space:pre-wrap">' + _esc(text) + '</div>' +
    '</div>';
  }

  // CRIT-2: timeline is [] in real data — graceful empty state, no crash
  function _renderTimeline(d) {
    var events = d.timeline || [];
    var facts  = d.source_fact_ids || [];

    if (events.length === 0) {
      // Build a synthetic timeline from available metadata
      var listEnt = _st.entities.find(function (e) { return e.name === d.entity_name; }) || {};
      var synth   = [];
      if (listEnt.first_seen) {
        synth.push({
          ts:    listEnt.first_seen,
          kind:  'observation',
          text:  'Entity first observed in memory store.',
        });
      }
      if (listEnt.last_seen && listEnt.last_seen !== listEnt.first_seen) {
        synth.push({
          ts:    listEnt.last_seen,
          kind:  'referenced',
          text:  'Most recent memory mentioning this entity.',
        });
      }
      if (d.last_compiled_at) {
        synth.push({
          ts:    d.last_compiled_at,
          kind:  'compiled',
          text:  'Knowledge summary last compiled.',
        });
      }

      if (synth.length === 0) {
        return '<div style="color:var(--fg-3);font-size:13px;padding:8px 0">' +
          'No temporal events recorded yet. ' +
          facts.length + ' source fact' + (facts.length === 1 ? '' : 's') + ' indexed.' +
          // TODO: /api/entity/{name}/timeline — per-fact temporal history not yet exposed
          '</div>';
      }

      return _buildTlHTML(synth);
    }

    // Real timeline data (when daemon exposes it)
    var items = events.map(function (ev) {
      return {
        ts:   ev.timestamp || ev.date || '',
        kind: ev.event_type || ev.type || 'observation',
        text: ev.description || ev.content || '',
      };
    });
    return _buildTlHTML(items);
  }

  function _buildTlHTML(items) {
    var COLORS = {
      observation: 'var(--cyan)',
      referenced:  'var(--violet)',
      interval:    'var(--warn)',
      compiled:    'var(--ok)',
    };
    return '<div class="od-tl">' +
      items.map(function (item) {
        var col = COLORS[item.kind] || 'var(--fg-3)';
        return '<div class="od-tl-item">' +
          '<div class="od-tl-dot" style="background:' + col + '"></div>' +
          '<div>' +
            '<div style="display:flex;gap:8px;align-items:center">' +
              '<span class="mono dim" style="font-size:12px">' +
                _esc(_fmt(item.ts)) +
              '</span>' +
              '<span class="badge neutral" style="font-size:10px">' +
                _esc(item.kind) +
              '</span>' +
            '</div>' +
            '<div style="margin-top:3px;font-size:13.5px">' +
              _esc(item.text) +
            '</div>' +
          '</div>' +
        '</div>';
      }).join('') +
    '</div>';
  }

  // ── Event delegation ──────────────────────────────────────────────────────

  function _wire(container, id) {
    container.addEventListener('click', function (e) {
      var el = e.target.closest('[data-od-act]');
      if (!el) return;
      var act = el.dataset.odAct;

      if (act === 'type-filter') {
        _st = Object.assign({}, _st, { typeFilter: el.dataset.type, search: '' });
        // Clear the search input DOM value so it stays in sync with state
        var searchEl = document.getElementById(id + '-search');
        if (searchEl) searchEl.value = '';
        _refreshTypeChips(id, el.dataset.type);
        _renderList(id);
        return;
      }
      if (act === 'select-entity') {
        var name = el.dataset.ename;
        if (name) _selectEntity(id, name);
        return;
      }
      if (act === 'list-pg') {
        var pg = parseInt(el.dataset.page, 10);
        if (!isNaN(pg)) {
          _st = Object.assign({}, _st, { page: pg });
          _loadList(id);
        }
        return;
      }
    });

    container.addEventListener('input', function (e) {
      if (e.target.dataset.odAct === 'ent-search') {
        _st = Object.assign({}, _st, { search: e.target.value.trim() });
        _renderList(id);
      }
    });
  }

  function _refreshTypeChips(id, active) {
    var root = document.getElementById(id);
    if (!root) return;
    var wrap = root.querySelector('[data-od-act="type-filter"]');
    if (!wrap) return;
    var parent = wrap.parentNode;
    if (!parent) return;
    parent.innerHTML = _typeChips(id, active);
  }

  // ── Public API + auto-init ─────────────────────────────────────────────────

  window.odRenderEntities = render;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      var pane = document.getElementById('entities-pane');
      if (pane) render(pane);
    });
  } else {
    var _pane = document.getElementById('entities-pane');
    if (_pane) render(_pane);
  }

}());

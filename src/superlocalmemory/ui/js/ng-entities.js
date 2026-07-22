// Neural Glass — Entity Explorer Tab
// Browse entities, their knowledge summaries, and trigger recompilation (v3.4.3)
// API: /api/entity/list, /api/entity/{name}, /api/entity/{name}/recompile

(function() {
  'use strict';

  var allEntities = [];
  var currentPage = 0;
  var PAGE_SIZE = 50;

  window.loadEntityExplorer = function() {
    fetchEntityList();
  };

  function fetchEntityList(offset) {
    offset = offset || 0;
    fetch('/api/entity/list?limit=' + PAGE_SIZE + '&offset=' + offset)
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(data) {
        allEntities = data.entities || [];
        renderEntityList(allEntities, data.total || 0, offset);
      })
      .catch(function(err) {
        console.error('[entities] fetchEntityList failed:', err);
        renderEntityList([], 0, 0);
      });
  }

  function renderEntityList(entities, total, offset) {
    var el = document.getElementById('entities-list');
    if (!el) return;

    if (total === 0) {
      el.innerHTML =
        '<div class="text-center" style="padding:40px;color:var(--ng-text-tertiary)">' +
          '<i class="bi bi-person-badge" style="font-size:3rem;display:block;margin-bottom:12px;opacity:0.3"></i>' +
          '<div style="font-size:1rem;margin-bottom:4px">No entities found</div>' +
          '<div style="font-size:0.8125rem">Entity extraction runs during memory consolidation (every 6 hours).<br>' +
          'Store more memories to build your entity graph.</div>' +
        '</div>';
      return;
    }

    // Search filter
    var html =
      '<div style="margin-bottom:16px">' +
        '<input type="text" class="form-control" id="entity-search-input" placeholder="Search ' + total + ' entities..." ' +
          'data-act-input="filter-entities" style="max-width:500px">' +
      '</div>';

    // Stats summary
    html +=
      '<div class="row g-3 mb-4">' +
        statCard('Total Entities', total, 'bi-people') +
        statCard('Showing', entities.length + ' of ' + total, 'bi-eye') +
        statCard('Top Type', getTopType(entities), 'bi-tag') +
        statCard('Avg Facts', getAvgFacts(entities), 'bi-collection') +
      '</div>';

    // Entity grid
    html += '<div class="row g-3" id="entity-grid">';
    entities.forEach(function(e) {
      html += renderEntityCard(e);
    });
    html += '</div>';

    // Pagination
    if (total > PAGE_SIZE) {
      var pages = Math.ceil(total / PAGE_SIZE);
      var currentP = Math.floor(offset / PAGE_SIZE);
      html += '<div style="display:flex;justify-content:center;gap:8px;margin-top:24px">';
      for (var i = 0; i < Math.min(pages, 10); i++) {
        var isActive = i === currentP;
        html += '<button class="ng-btn' + (isActive ? ' ng-btn-accent' : '') + '" ' +
          'data-act-click="navigate-entity-page" data-page="' + i + '" style="min-width:36px">' + (i + 1) + '</button>';
      }
      if (pages > 10) html += '<span style="padding:8px;color:var(--ng-text-tertiary)">...</span>';
      html += '</div>';
    }

    el.innerHTML = html;
  }

  function renderEntityCard(entity) {
    var typeColor = getTypeColor(entity.type);
    var summaryText = entity.summary_preview || 'No summary yet — click to view details';
    var hasTruth = entity.has_compiled_truth;
    var typeIcon = getTypeIcon(entity.type);

    // Skill entities get a distinctive accent border
    var borderAccent = entity.type === 'skill'
      ? 'border-left:3px solid ' + typeColor + ';'
      : '';

    return '<div class="col-md-6 col-lg-4">' +
      '<div class="ng-glass ng-hover-border" style="padding:16px;cursor:pointer;transition:border-color 0.2s;' + borderAccent + '" ' +
        'data-act-click="show-entity-detail" data-entity="' + escapeHtml(entity.name).replace(/"/g, '&quot;') + '">' +
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">' +
          '<div style="font-weight:590;font-size:0.9375rem;color:var(--ng-text-primary)">' +
            '<i class="bi ' + typeIcon + '" style="color:' + typeColor + ';margin-right:4px"></i>' +
            escapeHtml(entity.name) +
          '</div>' +
          '<span class="ng-badge ng-badge-accent">' + escapeHtml(entity.type) + '</span>' +
        '</div>' +
        '<div style="font-size:0.8125rem;color:var(--ng-text-secondary);margin-bottom:8px;' +
          'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">' +
          escapeHtml(summaryText) +
        '</div>' +
        '<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.75rem;color:var(--ng-text-tertiary)">' +
          '<span>' + entity.fact_count + ' facts</span>' +
          '<span>' + (entity.last_seen ? timeAgo(entity.last_seen) : '') + '</span>' +
        '</div>' +
      '</div>' +
    '</div>';
  }

  function getTypeIcon(type) {
    var icons = {
      person: 'bi-person',
      concept: 'bi-lightbulb',
      organization: 'bi-building',
      place: 'bi-geo-alt',
      event: 'bi-calendar-event',
      skill: 'bi-lightning-charge'
    };
    return icons[type] || 'bi-circle';
  }

  // Show entity detail panel
  window.showEntityDetail = function(entityName) {
    var panel = document.getElementById('entity-detail-panel');
    if (!panel) return;

    panel.style.display = 'block';
    panel.innerHTML =
      '<div class="ng-glass-elevated" style="padding:24px">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">' +
          '<h5 style="margin:0"><i class="bi bi-person-badge"></i> ' + escapeHtml(entityName) + '</h5>' +
          '<div style="display:flex;gap:8px">' +
            '<button class="ng-btn" data-act-click="recompile-entity" data-entity="' + escapeHtml(entityName).replace(/"/g, '&quot;') + '">' +
              '<i class="bi bi-arrow-repeat"></i> Recompile</button>' +
            '<button class="ng-btn" data-act-click="close-entity-detail">' +
              '<i class="bi bi-x-lg"></i></button>' +
          '</div>' +
        '</div>' +
        '<div id="entity-detail-content"><div class="text-center" style="padding:16px"><div class="spinner-border"></div></div></div>' +
      '</div>';

    // Fetch detail
    fetch('/api/entity/' + encodeURIComponent(entityName))
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var content = document.getElementById('entity-detail-content');
        if (!content) return;

        var html = '';

        // Entity type and confidence
        html += '<div style="margin-bottom:16px">' +
          '<span class="ng-badge ng-badge-accent">' + escapeHtml(data.entity_type || 'unknown') + '</span>' +
          ' <span class="ng-badge ng-badge-neutral">Confidence: ' + ((data.confidence || 0.5) * 100).toFixed(0) + '%</span>' +
          (data.last_compiled_at ? ' <span class="ng-badge ng-badge-neutral">Compiled: ' + timeAgo(data.last_compiled_at) + '</span>' : '') +
        '</div>';

        // Knowledge summary
        if (data.knowledge_summary) {
          html += '<div style="margin-bottom:16px">' +
            '<h6 style="font-size:0.8125rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--ng-text-tertiary);margin-bottom:8px">Knowledge Summary</h6>' +
            '<div style="font-size:0.875rem;color:var(--ng-text-secondary);line-height:1.6;white-space:pre-wrap;background:var(--ng-bg-glass);padding:12px;border-radius:var(--ng-radius-md)">' +
              escapeHtml(data.knowledge_summary) +
            '</div>' +
          '</div>';
        }

        // Compiled truth
        if (data.compiled_truth) {
          html += '<div style="margin-bottom:16px">' +
            '<h6 style="font-size:0.8125rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--ng-text-tertiary);margin-bottom:8px">Compiled Truth</h6>' +
            '<div style="font-size:0.875rem;color:var(--ng-text-primary);line-height:1.6;background:var(--ng-bg-glass);padding:12px;border-radius:var(--ng-radius-md)">' +
              escapeHtml(data.compiled_truth) +
            '</div>' +
          '</div>';
        }

        // Source facts
        if (data.source_fact_ids && data.source_fact_ids.length > 0) {
          html += '<div>' +
            '<h6 style="font-size:0.8125rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--ng-text-tertiary);margin-bottom:8px">' +
              'Source Facts (' + data.source_fact_ids.length + ')</h6>' +
            '<div style="font-size:0.75rem;color:var(--ng-text-quaternary)">' +
              data.source_fact_ids.slice(0, 10).map(function(id) { return '<code>' + id.substring(0, 12) + '</code>'; }).join(' ') +
              (data.source_fact_ids.length > 10 ? ' + ' + (data.source_fact_ids.length - 10) + ' more' : '') +
            '</div>' +
          '</div>';
        }

        content.innerHTML = html;
      })
      .catch(function(err) {
        var content = document.getElementById('entity-detail-content');
        if (content) {
          content.innerHTML = '<div style="color:var(--ng-text-tertiary);padding:16px">Could not load entity details: ' + err.message + '</div>';
        }
      });

    // Scroll to detail panel
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  // Recompile entity
  window.recompileEntity = function(entityName) {
    fetch('/api/entity/' + encodeURIComponent(entityName) + '/recompile', { method: 'POST' })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.ok) {
          alert('Entity "' + entityName + '" recompiled successfully.');
          window.showEntityDetail(entityName); // Refresh detail view
        } else {
          alert('Recompilation failed: ' + (data.reason || 'unknown'));
        }
      })
      .catch(function(err) {
        alert('Error: ' + err.message);
      });
  };

  // Filter entities by search text
  window.filterEntities = function(query) {
    if (!query) {
      fetchEntityList();
      return;
    }
    var q = query.toLowerCase();
    var filtered = allEntities.filter(function(e) {
      return e.name.toLowerCase().indexOf(q) >= 0 ||
             (e.type && e.type.toLowerCase().indexOf(q) >= 0) ||
             (e.summary_preview && e.summary_preview.toLowerCase().indexOf(q) >= 0);
    });
    var grid = document.getElementById('entity-grid');
    if (!grid) return;
    grid.innerHTML = filtered.length > 0
      ? filtered.map(renderEntityCard).join('')
      : '<div class="col-12" style="text-align:center;padding:24px;color:var(--ng-text-tertiary)">No entities match "' + escapeHtml(query) + '"</div>';
  };

  // Pagination
  window.navigateEntityPage = function(page) {
    currentPage = page;
    fetchEntityList(page * PAGE_SIZE);
  };

  // Helpers
  function statCard(label, value, icon) {
    return '<div class="col-md-3 col-6"><div class="ng-glass" style="padding:12px;text-align:center">' +
      '<i class="bi ' + icon + '" style="color:var(--ng-accent);font-size:1.125rem;display:block;margin-bottom:4px"></i>' +
      '<div style="font-size:1.25rem;font-weight:590">' + value + '</div>' +
      '<div class="ng-stat-label">' + label + '</div>' +
    '</div></div>';
  }

  function getTopType(entities) {
    var counts = {};
    entities.forEach(function(e) { counts[e.type] = (counts[e.type] || 0) + 1; });
    var top = Object.entries(counts).sort(function(a, b) { return b[1] - a[1]; })[0];
    return top ? top[0] : 'N/A';
  }

  function getAvgFacts(entities) {
    if (entities.length === 0) return '0';
    var sum = entities.reduce(function(s, e) { return s + (e.fact_count || 0); }, 0);
    return (sum / entities.length).toFixed(1);
  }

  function getTypeColor(type) {
    var colors = {
      person: '#3b82f6', concept: '#10b981', organization: '#f59e0b',
      location: '#ef4444', skill: '#8b5cf6', event: '#ec4899'
    };
    return colors[type] || '#7C6AEF';
  }

  function escapeHtml(s) { var d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
  function escapeAttr(s) { return (s || '').replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

  function timeAgo(iso) {
    if (!iso) return '';
    var d = (Date.now() - new Date(iso).getTime()) / 1000;
    if (d < 0) d = 0;
    if (d < 60) return Math.floor(d) + 's ago';
    if (d < 3600) return Math.floor(d / 60) + 'm ago';
    if (d < 86400) return Math.floor(d / 3600) + 'h ago';
    return Math.floor(d / 86400) + 'd ago';
  }
})();

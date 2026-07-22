// Neural Glass — Skill Evolution Tab
// Browse skill performance, evolution history, and health status (v3.4.10)
// API: /api/behavioral/assertions (category=skill_performance, skill_correlation)
//      /api/behavioral/tool-events (tool_name=Skill)
//      /api/entity/list (type filter for skill entities)

(function() {
  'use strict';

  window.loadSkillEvolution = function() {
    fetchEvolutionEngine();
    fetchSkillOverview();
    fetchSkillPerformance();
    fetchSkillLineage();
  };

  function fetchEvolutionEngine() {
    var container = document.getElementById('skills-overview-cards');
    if (!container) return;

    fetch('/api/evolution/status')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var enabled = data.enabled || false;
        var backend = data.backend || 'none';
        var stats = data.stats || {};

        var statusColor = enabled ? '#10b981' : '#888';
        var statusText = enabled ? 'Enabled (' + backend + ')' : 'Disabled';
        var statusIcon = enabled ? 'bi-lightning-charge-fill' : 'bi-lightning-charge';

        var html = '<div class="card" style="padding:16px;margin-bottom:16px;border-left:3px solid ' + statusColor + '">' +
          '<div style="display:flex;justify-content:space-between;align-items:center">' +
            '<div>' +
              '<div style="font-weight:600;font-size:1rem;margin-bottom:4px">' +
                '<i class="bi ' + statusIcon + '" style="color:' + statusColor + ';margin-right:6px"></i>' +
                'Evolution Engine: ' + statusText +
              '</div>' +
              '<div style="font-size:0.8125rem;color:var(--bs-body-color)">' +
                (enabled
                  ? 'Backend: <strong>' + escapeHtml(backend) + '</strong> | ' +
                    'Evolved: ' + (stats.promoted || 0) + ' | ' +
                    'Rejected: ' + (stats.rejected || 0) + ' | ' +
                    'Budget: ' + (stats.cycle_budget_remaining || 3) + ' remaining this cycle'
                  : 'Enable via CLI: <code>slm config set evolution.enabled true</code> or via <code>slm setup</code>') +
              '</div>' +
            '</div>' +
            '<div style="display:flex;gap:8px">' +
              (enabled
                ? '<button class="btn btn-sm btn-outline-success" data-act-click="trigger-evolution"><i class="bi bi-play-fill"></i> Run Now</button>'
                : '<button class="btn btn-sm btn-outline-primary" data-act-click="enable-evolution"><i class="bi bi-power"></i> Enable</button>') +
            '</div>' +
          '</div>';

        // Per-step evolution model config (v3.7.9) — shown when enabled.
        if (enabled) {
          var cfg = data.config || {};
          var modelOpts = function(sel) {
            var cur = sel || 'auto';
            if (cur === '') cur = 'auto';
            return ['auto', 'haiku', 'sonnet', 'ollama'].map(function(m) {
              return '<option value="' + m + '"' + (m === cur ? ' selected' : '') + '>' + m + '</option>';
            }).join('');
          };
          html += '<div style="margin-top:12px;border-top:1px solid var(--bs-border-color);padding-top:12px">' +
            '<div style="font-size:0.8125rem;font-weight:600;margin-bottom:8px">Evolution models ' +
              '<span style="font-weight:400;color:var(--ng-text-tertiary,#888)">(lowest-cost by default)</span></div>' +
            '<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end">' +
              '<label style="font-size:0.75rem">Generate<br><select id="evo-mutation-model" class="form-select form-select-sm">' + modelOpts(cfg.mutation_model) + '</select></label>' +
              '<label style="font-size:0.75rem">Verify<br><select id="evo-verify-model" class="form-select form-select-sm">' + modelOpts(cfg.verify_model) + '</select></label>' +
              '<label style="font-size:0.75rem">Confirm<br><select id="evo-confirm-model" class="form-select form-select-sm">' + modelOpts(cfg.confirm_model) + '</select></label>' +
              '<label style="font-size:0.75rem">Max/cycle<br><input id="evo-max-cycle" type="number" min="1" max="50" value="' + (cfg.max_per_cycle || 3) + '" class="form-control form-control-sm" style="width:84px"></label>' +
              '<button class="btn btn-sm btn-primary" data-act-click="save-evolution-config"><i class="bi bi-save"></i> Save</button>' +
            '</div>' +
            '<div style="font-size:0.7rem;color:var(--ng-text-tertiary,#888);margin-top:8px">' +
              '⚠ Evolution makes background LLM calls during consolidation (capped 10/cycle, 3 cycles/day). ' +
              '“auto” picks the cheapest capable model for your backend; the verifier stays independent of the generator.' +
            '</div>' +
          '</div>';
        }

        // Evolution history (if any)
        if (data.recent && data.recent.length > 0) {
          html += '<div style="margin-top:12px;border-top:1px solid var(--bs-border-color);padding-top:12px">' +
            '<div style="font-size:0.8125rem;font-weight:600;margin-bottom:8px">Recent Evolution</div>';
          data.recent.forEach(function(r) {
            var sColor = r.status === 'promoted' ? '#10b981' : r.status === 'rejected' ? '#ef4444' : '#f59e0b';
            html += '<div style="display:flex;justify-content:space-between;font-size:0.75rem;padding:4px 0;border-bottom:1px solid var(--bs-border-color)">' +
              '<span><i class="bi bi-lightning-charge" style="color:#8b5cf6"></i> ' + escapeHtml(r.skill_name) + '</span>' +
              '<span style="color:' + sColor + '">' + escapeHtml(r.status) + ' (' + escapeHtml(r.evolution_type) + ')</span>' +
            '</div>';
          });
          html += '</div>';
        }

        html += '</div>';

        // Insert before the overview cards
        var overviewInner = document.getElementById('skills-overview-inner');
        if (overviewInner) {
          overviewInner.insertAdjacentHTML('beforebegin', html);
        } else {
          container.insertAdjacentHTML('afterbegin', html);
        }
      })
      .catch(function() {
        // API not available yet — skip silently
      });
  }

  window.enableEvolution = function() {
    fetch('/api/evolution/enable', { method: 'POST' })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.ok) {
          loadSkillEvolution();
        } else {
          alert('Could not enable: ' + (data.error || 'unknown'));
        }
      })
      .catch(function(err) { alert('Error: ' + err.message); });
  };

  window.triggerEvolution = function() {
    fetch('/api/evolution/run', { method: 'POST' })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        alert('Evolution cycle: ' + (data.evolved || 0) + ' evolved, ' +
              (data.rejected || 0) + ' rejected, ' + (data.candidates || 0) + ' candidates');
        loadSkillEvolution();
      })
      .catch(function(err) { alert('Error: ' + err.message); });
  };

  window.saveEvolutionConfig = function() {
    var g = function(id) { var e = document.getElementById(id); return e ? e.value : undefined; };
    var body = {
      mutation_model: g('evo-mutation-model'),
      verify_model: g('evo-verify-model'),
      confirm_model: g('evo-confirm-model'),
    };
    var mx = g('evo-max-cycle');
    if (mx) body.max_evolutions_per_cycle = parseInt(mx, 10);
    fetch('/api/evolution/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.ok) {
          loadSkillEvolution();
        } else {
          alert('Could not save evolution config: ' + (data.error || 'unknown'));
        }
      })
      .catch(function(err) { alert('Error: ' + err.message); });
  };

  function fetchSkillOverview() {
    var el = document.getElementById('skills-overview-cards');
    if (!el) return;

    // Compatibility notice + ECC credit + docs links
    var noticeHtml =
      '<div class="card" style="padding:12px 16px;margin-bottom:16px;border-left:3px solid #8b5cf6">' +
        '<div style="font-size:0.8125rem;color:var(--bs-body-color)">' +
          '<i class="bi bi-info-circle" style="color:#8b5cf6;margin-right:6px"></i>' +
          '<strong>Skill Evolution</strong> currently tracks <strong>Claude Code</strong> skills. ' +
          'The <code>/api/v3/tool-event</code> endpoint accepts events from any IDE client. ' +
          'Enhanced observation support available with ' +
          '<a href="https://github.com/affaan-m/everything-claude-code" target="_blank" style="color:#8b5cf6">Everything Claude Code (ECC)</a> ' +
          'via <code>slm ingest --source ecc</code>.' +
        '</div>' +
        '<div style="font-size:0.75rem;color:var(--bs-secondary-color);margin-top:8px">' +
          '<a href="https://superlocalmemory.com/skill-evolution" target="_blank" style="color:#8b5cf6;margin-right:12px"><i class="bi bi-globe"></i> Learn more</a>' +
          '<a href="https://github.com/qualixar/superlocalmemory/blob/main/docs/skill-evolution.md" target="_blank" style="color:#8b5cf6"><i class="bi bi-book"></i> Documentation</a>' +
        '</div>' +
      '</div>';
    el.innerHTML = noticeHtml + '<div id="skills-overview-inner"></div>';
    el = document.getElementById('skills-overview-inner');

    // Fetch tool events for Skill calls + assertions for skill_performance
    Promise.all([
      fetch('/api/behavioral/tool-events?tool_name=Skill&limit=500').then(function(r) { return r.json(); }),
      fetch('/api/behavioral/assertions?category=skill_performance&limit=50').then(function(r) { return r.json(); }),
      fetch('/api/behavioral/assertions?category=skill_correlation&limit=20').then(function(r) { return r.json(); }),
    ]).then(function(results) {
      var events = results[0].events || [];
      var perfAssertions = results[1].assertions || [];
      var corrAssertions = results[2].assertions || [];

      // Count unique skills from events
      var skillNames = {};
      events.forEach(function(e) {
        var name = extractSkillName(e);
        if (name) skillNames[name] = (skillNames[name] || 0) + 1;
      });

      var html = '<div class="row g-3 mb-4">' +
        overviewCard('Total Skill Events', events.length, 'bi-lightning-charge', 'var(--ng-accent)') +
        overviewCard('Unique Skills', Object.keys(skillNames).length, 'bi-grid-3x3', '#8b5cf6') +
        overviewCard('Performance Assertions', perfAssertions.length, 'bi-graph-up', '#10b981') +
        overviewCard('Skill Correlations', corrAssertions.length, 'bi-link-45deg', '#f59e0b') +
      '</div>';

      el.innerHTML = html;
    }).catch(function() {
      el.innerHTML = '<div class="alert alert-warning">Could not load skill overview</div>';
    });
  }

  function fetchSkillPerformance() {
    var el = document.getElementById('skills-list');
    if (!el) return;

    Promise.all([
      fetch('/api/behavioral/assertions?category=skill_performance&limit=50').then(function(r) { return r.json(); }),
      fetch('/api/behavioral/assertions?category=skill_correlation&limit=20').then(function(r) { return r.json(); }),
      fetch('/api/behavioral/tool-events?tool_name=Skill&limit=500').then(function(r) { return r.json(); }),
    ]).then(function(results) {
      var perfAssertions = results[0].assertions || [];
      var corrAssertions = results[1].assertions || [];
      var events = results[2].events || [];

      var html = '';

      // Section 1: Skill Performance
      html += '<h5 style="margin-bottom:16px"><i class="bi bi-lightning-charge" style="color:#8b5cf6"></i> Skill Performance</h5>';

      if (perfAssertions.length === 0 && events.length === 0) {
        html += '<div class="card" style="padding:24px;text-align:center;color:var(--bs-secondary-color)">' +
          '<i class="bi bi-lightning-charge" style="font-size:2.5rem;display:block;margin-bottom:12px;opacity:0.3"></i>' +
          '<div style="font-size:1rem;margin-bottom:4px;color:var(--bs-body-color)">No skill performance data yet</div>' +
          '<div style="font-size:0.8125rem">' +
            'Skill tracking starts automatically after the enriched hook captures data.<br>' +
            'Use skills in your sessions — performance assertions will appear after consolidation.' +
          '</div>' +
        '</div>';
      } else if (perfAssertions.length > 0) {
        html += '<div class="row g-3">';
        perfAssertions.forEach(function(a) {
          html += renderSkillCard(a);
        });
        html += '</div>';
      } else {
        // We have events but no assertions yet (need consolidation)
        html += '<div class="card" style="padding:16px;margin-bottom:16px">' +
          '<div style="font-size:0.875rem;color:var(--bs-body-color)">' +
            '<i class="bi bi-info-circle" style="color:#8b5cf6;margin-right:6px"></i>' +
            events.length + ' skill events collected. Run consolidation to generate performance assertions.' +
          '</div>' +
        '</div>';

        // Show raw event summary
        var skillCounts = {};
        events.forEach(function(e) {
          var name = extractSkillName(e);
          if (name) skillCounts[name] = (skillCounts[name] || 0) + 1;
        });

        html += '<div class="row g-3">';
        Object.keys(skillCounts).sort(function(a, b) {
          return skillCounts[b] - skillCounts[a];
        }).forEach(function(name) {
          html += '<div class="col-md-6 col-lg-4"><div class="card" style="padding:16px;border-left:3px solid #8b5cf6">' +
            '<div style="display:flex;justify-content:space-between;align-items:center">' +
              '<div style="font-weight:600;font-size:0.9375rem">' +
                '<i class="bi bi-lightning-charge" style="color:#8b5cf6;margin-right:4px"></i>' +
                escapeHtml(name) +
              '</div>' +
              '<span class="badge" style="background:#8b5cf620;color:#8b5cf6;font-size:0.75rem">' + skillCounts[name] + ' events</span>' +
            '</div>' +
          '</div></div>';
        });
        html += '</div>';
      }

      // Section 2: Skill Correlations
      if (corrAssertions.length > 0) {
        html += '<h5 style="margin-top:32px;margin-bottom:16px"><i class="bi bi-link-45deg" style="color:#f59e0b"></i> Skill Correlations</h5>';
        html += '<div class="row g-3">';
        corrAssertions.forEach(function(a) {
          html += '<div class="col-md-6"><div class="card" style="padding:12px">' +
            '<div style="font-size:0.875rem">' +
              '<strong>' + escapeHtml(a.trigger_condition || '') + '</strong>' +
            '</div>' +
            '<div style="font-size:0.8125rem;color:var(--bs-body-color);margin-top:4px">' +
              escapeHtml(a.action || '') +
            '</div>' +
            '<div style="font-size:0.75rem;color:var(--bs-secondary-color);margin-top:4px">' +
              'Confidence: ' + ((a.confidence || 0) * 100).toFixed(0) + '%' +
            '</div>' +
          '</div></div>';
        });
        html += '</div>';
      }

      el.innerHTML = html;
    }).catch(function(err) {
      el.innerHTML = '<div class="text-center" style="padding:24px;color:var(--ng-text-tertiary)">' +
        'Error loading skill data: ' + err.message + '</div>';
    });
  }

  function renderSkillCard(assertion) {
    var conf = assertion.confidence || 0;
    var confPct = (conf * 100).toFixed(0);
    var confColor = conf >= 0.7 ? '#10b981' : conf >= 0.5 ? '#f59e0b' : '#ef4444';

    // Extract skill name from trigger_condition
    var skillName = (assertion.trigger_condition || '').replace('when considering skill ', '');

    return '<div class="col-md-6 col-lg-4">' +
      '<div class="card" style="padding:16px;border-left:3px solid #8b5cf6;cursor:pointer">' +
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">' +
          '<div style="font-weight:600;font-size:0.9375rem">' +
            '<i class="bi bi-lightning-charge" style="color:#8b5cf6;margin-right:4px"></i>' +
            escapeHtml(skillName) +
          '</div>' +
          '<span class="badge" style="background:' + confColor + ';color:#fff;font-size:0.75rem">' +
            confPct + '%' +
          '</span>' +
        '</div>' +
        '<div style="font-size:0.8125rem;color:var(--bs-body-color);margin-bottom:8px">' +
          escapeHtml(assertion.action || 'No performance data yet') +
        '</div>' +
        '<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.75rem;color:var(--bs-secondary-color)">' +
          '<span>Evidence: ' + (assertion.evidence_count || 0) + ' invocations</span>' +
          '<span>Reinforced: ' + (assertion.reinforcement_count || 0) + 'x</span>' +
        '</div>' +
      '</div>' +
    '</div>';
  }

  function extractSkillName(event) {
    var input = event.input_summary || '';
    var output = event.output_summary || '';

    // Try input_summary (enriched hook format)
    if (input) {
      try {
        var inp = JSON.parse(input);
        if (inp.skill) return inp.skill;
      } catch(e) {}
    }

    // Try output_summary (ECC ingestion format)
    if (output) {
      try {
        var out = JSON.parse(output);
        if (out.commandName) return out.commandName;
      } catch(e) {}
    }

    return null;
  }

  function overviewCard(label, value, icon, color) {
    return '<div class="col-md-3 col-6"><div class="card" style="padding:12px;text-align:center">' +
      '<i class="bi ' + icon + '" style="color:' + color + ';font-size:1.125rem;display:block;margin-bottom:4px"></i>' +
      '<div style="font-size:1.25rem;font-weight:600">' + value + '</div>' +
      '<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--bs-secondary-color)">' + label + '</div>' +
    '</div></div>';
  }

  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
  }

  // ── Skill Lineage (Version DAG) ────────────────────────────

  var LINEAGE_COLORS = {
    promoted: '#22c55e',
    rejected: '#ef4444',
    pending:  '#eab308',
    original: '#6b7280'
  };

  function fetchSkillLineage() {
    var container = document.getElementById('skill-lineage-container');
    if (!container) return;

    fetch('/api/evolution/lineage')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var lineage = data.lineage || [];
        if (lineage.length === 0) {
          container.innerHTML =
            '<div class="card" style="padding:24px;text-align:center;color:var(--bs-secondary-color)">' +
              '<i class="bi bi-diagram-3" style="font-size:2.5rem;display:block;margin-bottom:12px;opacity:0.3"></i>' +
              '<div style="font-size:1rem;margin-bottom:4px;color:var(--bs-body-color)">No skill lineage data yet</div>' +
              '<div style="font-size:0.8125rem">' +
                'Lineage appears after skills evolve. Run an evolution cycle to generate skill versions.' +
              '</div>' +
            '</div>';
          return;
        }
        var html = '<h5 style="margin-bottom:16px"><i class="bi bi-diagram-3" style="color:#8b5cf6"></i> Skill Lineage</h5>';
        html += '<div class="card" style="padding:16px;margin-bottom:16px">';
        html += '<div id="lineage-dag-wrapper" style="max-height:400px;overflow:auto;position:relative"></div>';
        html += '</div>';
        html += '<div id="lineage-table-wrapper"></div>';
        container.innerHTML = html;
        renderLineageDAG(lineage);
        renderLineageTable(lineage);
      })
      .catch(function() {
        container.innerHTML = '';
      });
  }

  function renderLineageDAG(lineage) {
    var wrapper = document.getElementById('lineage-dag-wrapper');
    if (!wrapper) return;

    // Build adjacency: id -> node, parent_skill_id -> children
    var nodeMap = {};
    var childrenMap = {};
    var roots = [];

    lineage.forEach(function(item) {
      nodeMap[item.id] = item;
      if (!childrenMap[item.id]) childrenMap[item.id] = [];
    });

    lineage.forEach(function(item) {
      var pid = item.parent_skill_id;
      if (pid && nodeMap[pid]) {
        if (!childrenMap[pid]) childrenMap[pid] = [];
        childrenMap[pid].push(item.id);
      } else {
        roots.push(item.id);
      }
    });

    // If no explicit roots found, treat all nodes as roots
    if (roots.length === 0) {
      lineage.forEach(function(item) { roots.push(item.id); });
    }

    // Assign layers via BFS (Sugiyama layer assignment)
    var layers = {};
    var queue = [];
    var visited = {};
    roots.forEach(function(rid) {
      layers[rid] = 0;
      queue.push(rid);
      visited[rid] = true;
    });
    var maxLayer = 0;
    while (queue.length > 0) {
      var nid = queue.shift();
      var children = childrenMap[nid] || [];
      children.forEach(function(cid) {
        if (!visited[cid]) {
          visited[cid] = true;
          layers[cid] = (layers[nid] || 0) + 1;
          if (layers[cid] > maxLayer) maxLayer = layers[cid];
          queue.push(cid);
        }
      });
    }

    // Handle nodes not reached by BFS (disconnected)
    lineage.forEach(function(item) {
      if (layers[item.id] === undefined) {
        layers[item.id] = 0;
      }
    });

    // Group nodes by layer
    var layerGroups = {};
    Object.keys(layers).forEach(function(nid) {
      var l = layers[nid];
      if (!layerGroups[l]) layerGroups[l] = [];
      layerGroups[l].push(nid);
    });

    // Layout constants
    var nodeW = 140;
    var nodeH = 44;
    var layerGap = 80;
    var nodeGap = 24;
    var padX = 20;
    var padY = 20;

    // Compute max width needed
    var maxNodesInLayer = 0;
    for (var l = 0; l <= maxLayer; l++) {
      var count = (layerGroups[l] || []).length;
      if (count > maxNodesInLayer) maxNodesInLayer = count;
    }

    var svgW = Math.max(300, padX * 2 + maxNodesInLayer * (nodeW + nodeGap) - nodeGap);
    var svgH = padY * 2 + (maxLayer + 1) * (nodeH + layerGap) - layerGap;

    // Compute positions
    var positions = {};
    for (var ly = 0; ly <= maxLayer; ly++) {
      var group = layerGroups[ly] || [];
      var totalW = group.length * nodeW + (group.length - 1) * nodeGap;
      var startX = (svgW - totalW) / 2;
      var yPos = padY + ly * (nodeH + layerGap);
      group.forEach(function(nid, idx) {
        positions[nid] = {
          x: startX + idx * (nodeW + nodeGap),
          y: yPos
        };
      });
    }

    // Build SVG
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + svgW + '" height="' + svgH + '" ' +
      'style="display:block;margin:0 auto;font-family:system-ui,-apple-system,sans-serif">';

    // Arrowhead marker
    svg += '<defs><marker id="lineage-arrow" viewBox="0 0 10 7" refX="10" refY="3.5" ' +
      'markerWidth="8" markerHeight="6" orient="auto-start-reverse">' +
      '<path d="M 0 0 L 10 3.5 L 0 7 z" fill="#888"/></marker></defs>';

    // Draw edges
    lineage.forEach(function(item) {
      var pid = item.parent_skill_id;
      if (pid && positions[pid] && positions[item.id]) {
        var from = positions[pid];
        var to = positions[item.id];
        var x1 = from.x + nodeW / 2;
        var y1 = from.y + nodeH;
        var x2 = to.x + nodeW / 2;
        var y2 = to.y;
        // Curved path
        var midY = (y1 + y2) / 2;
        svg += '<path d="M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + midY + ', ' + x2 + ' ' + midY + ', ' + x2 + ' ' + y2 + '" ' +
          'fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#lineage-arrow)"/>';
        // Edge label
        var etype = item.evolution_type || '';
        if (etype) {
          var lx = (x1 + x2) / 2;
          var labelY = midY - 4;
          svg += '<text x="' + lx + '" y="' + labelY + '" text-anchor="middle" ' +
            'fill="#888" font-size="10" font-weight="500">' + escapeHtml(etype) + '</text>';
        }
      }
    });

    // Draw nodes
    lineage.forEach(function(item) {
      var pos = positions[item.id];
      if (!pos) return;
      var status = (item.status || 'original').toLowerCase();
      var fillColor = LINEAGE_COLORS[status] || LINEAGE_COLORS.original;
      var label = (item.skill_name || 'unknown');
      if (label.length > 16) label = label.substring(0, 14) + '..';

      svg += '<g class="lineage-node" data-id="' + item.id + '" style="cursor:pointer">';
      svg += '<rect x="' + pos.x + '" y="' + pos.y + '" width="' + nodeW + '" height="' + nodeH + '" ' +
        'rx="8" ry="8" fill="' + fillColor + '" fill-opacity="0.15" stroke="' + fillColor + '" stroke-width="2"/>';
      // Node label (skill name)
      svg += '<text x="' + (pos.x + nodeW / 2) + '" y="' + (pos.y + 18) + '" text-anchor="middle" ' +
        'fill="' + fillColor + '" font-size="12" font-weight="600">' + escapeHtml(label) + '</text>';
      // Status sub-label
      svg += '<text x="' + (pos.x + nodeW / 2) + '" y="' + (pos.y + 34) + '" text-anchor="middle" ' +
        'fill="' + fillColor + '" font-size="10" opacity="0.7">' + escapeHtml(status) + '</text>';
      svg += '</g>';
    });

    svg += '</svg>';
    wrapper.innerHTML = svg;

    // Click handler: highlight row in table
    wrapper.querySelectorAll('.lineage-node').forEach(function(g) {
      g.addEventListener('click', function() {
        var id = g.getAttribute('data-id');
        highlightLineageRow(id);
      });
    });
  }

  function renderLineageTable(lineage) {
    var wrapper = document.getElementById('lineage-table-wrapper');
    if (!wrapper) return;

    var html = '<div class="card" style="padding:16px">' +
      '<div style="font-weight:600;font-size:0.9375rem;margin-bottom:12px">' +
        '<i class="bi bi-table" style="color:#8b5cf6;margin-right:6px"></i>Lineage Details' +
      '</div>' +
      '<div class="table-responsive"><table class="table table-sm table-hover" style="font-size:0.8125rem;margin-bottom:0">' +
      '<thead><tr>' +
        '<th>Skill Name</th>' +
        '<th>Type</th>' +
        '<th>Parent</th>' +
        '<th>Status</th>' +
        '<th>Verified</th>' +
        '<th>Created</th>' +
      '</tr></thead><tbody>';

    // Build a quick lookup for parent names
    var nameMap = {};
    lineage.forEach(function(item) {
      nameMap[item.id] = item.skill_name || item.id;
    });

    lineage.forEach(function(item) {
      var status = (item.status || 'original').toLowerCase();
      var color = LINEAGE_COLORS[status] || LINEAGE_COLORS.original;
      var parentName = item.parent_skill_id ? (nameMap[item.parent_skill_id] || item.parent_skill_id) : '-';
      var verified = item.blind_verified ? '<i class="bi bi-check-circle-fill" style="color:#22c55e"></i>' : '<i class="bi bi-dash-circle" style="color:var(--bs-secondary-color)"></i>';
      var created = item.created_at ? new Date(item.created_at).toLocaleDateString() : '-';

      html += '<tr class="lineage-table-row" data-id="' + item.id + '" style="cursor:pointer">' +
        '<td style="font-weight:500">' + escapeHtml(item.skill_name || '') + '</td>' +
        '<td><span class="badge" style="background:' + color + '20;color:' + color + ';font-size:0.6875rem">' +
          escapeHtml(item.evolution_type || 'ORIGINAL') + '</span></td>' +
        '<td>' + escapeHtml(parentName) + '</td>' +
        '<td><span style="color:' + color + ';font-weight:500">' + escapeHtml(status) + '</span></td>' +
        '<td>' + verified + '</td>' +
        '<td style="color:var(--bs-secondary-color)">' + created + '</td>' +
      '</tr>';
    });

    html += '</tbody></table></div></div>';
    wrapper.innerHTML = html;

    // Click rows to highlight in DAG
    wrapper.querySelectorAll('.lineage-table-row').forEach(function(row) {
      row.addEventListener('click', function() {
        var id = row.getAttribute('data-id');
        highlightLineageNode(id);
        highlightLineageRow(id);
      });
    });
  }

  function highlightLineageRow(id) {
    // Clear previous highlights
    document.querySelectorAll('.lineage-table-row').forEach(function(row) {
      row.style.background = '';
    });
    var target = document.querySelector('.lineage-table-row[data-id="' + id + '"]');
    if (target) {
      target.style.background = 'rgba(139, 92, 246, 0.12)';
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function highlightLineageNode(id) {
    // Reset all nodes
    document.querySelectorAll('.lineage-node rect').forEach(function(rect) {
      rect.setAttribute('stroke-width', '2');
    });
    // Highlight selected
    var node = document.querySelector('.lineage-node[data-id="' + id + '"] rect');
    if (node) {
      node.setAttribute('stroke-width', '4');
      node.closest('.lineage-node').parentElement.closest('svg')
        .closest('#lineage-dag-wrapper')
        .scrollTop = 0; // Scroll DAG to top if needed
    }
  }
})();

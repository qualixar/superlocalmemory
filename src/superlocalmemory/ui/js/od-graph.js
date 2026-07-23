/* ============================================================
   od-graph.js — Knowledge Graph (Open Design port, live data)
   Faithful port of open-design/graph.html: force-directed canvas
   engine, 3-tier model (community > entity > episode), node inspector,
   Ask-Memory chat.

   Live data (NO invented values):
     GET  /api/graph?max_nodes=&min_importance=   nodes/links/clusters
     GET  /api/entity/list?limit=                 tier-2 entities
     POST /api/search                             Ask-Memory answers
   Exposes window.odRenderGraph(container); the shell calls it on tab open.
   ============================================================ */
(function () {
  'use strict';

  var GRAPH_URL = '/api/graph';
  var RECALL_URL = '/api/search';   /* POST {query, limit} -> {results:[...]} */
  var MAX_NODES = 120;              /* default budget; slider range 20-2000 */
  /* Keep force-layout work bounded even when a user asks to render "All". */
  var PHYSICS_MAX_NODES = 160;
  var PRE_SETTLE_TICKS = 24;

  /* ---- 16-colour categorical palette (Tableau-inspired, accessible) ---- */
  var CAT_PALETTE = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
    '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
    '#52B2BF', '#D4A5A5', '#7EB0D5', '#FD7F6F', '#B2E061', '#FFBE7D'
  ];
  /* fixed colours per entity type -- consistent across renders */
  var ETYPE_COLORS = {
    person:   '#4E79A7',
    org:      '#F28E2B',
    concept:  '#59A14F',
    event:    '#E15759',
    location: '#76B7B2',
    tool:     '#EDC948',
    project:  '#B07AA1',
    skill:    '#FF9DA7'
  };
  var EPISODE_COLOR = '#8899AA';   /* neutral-desaturated -- episodes are context */

  /* module state (re-render safe) */
  var mount = null, cv = null, ctx = null, stage = null;
  var raf = null, running = false, fitFrames = 0;
  /* settle-freeze: stop the rAF loop once kinetic energy drops below SETTLE_KE */
  var frames = 0, _ke = 0;
  var SETTLE_MIN = 30, SETTLE_KE = 0.2, SETTLE_MAX_FRAMES = 180;
  var NODES = [], LINKS = [], idx = {};
  var tierVisible = { 1: true, 2: true, 3: true };  /* independent tier toggles */
  var scale = 1, ox = 0, oy = 0, dpr = Math.max(1, window.devicePixelRatio || 1);
  var W = 0, H = 0, selected = null, hover = null;
  var dragNode = null, panning = false, last = null;
  var PAL = {}, themeObs = null;

  /* bound handlers kept so re-render can detach them */
  var onMove = null, onUp = null;

  /* ---------------- markup (from graph.html <main>) ---------------- */
  function scaffold() {
    return '' +
    '<div class="graph-shell">' +
      '<div class="graph-stage" id="odg-stage">' +
        '<canvas id="odg-cv"></canvas>' +
        '<div class="graph-overlay">' +
          '<div class="graph-tools">' +
            '<div class="seg" id="odg-tier">' +
              '<button data-tier="1" class="active">Communities</button>' +
              '<button data-tier="2" class="active">Entities</button>' +
              '<button data-tier="3" class="active">Episodes</button>' +
            '</div>' +
            '<div class="seg" id="odg-layout">' +
              '<button data-lay="fcose" class="active">Force</button>' +
              '<button data-lay="concentric">By importance</button>' +
            '</div>' +
            '<label class="chip" style="gap:8px" title="How many nodes to show">' +
              '<span data-ic="filter"></span> Nodes ' +
              '<input id="odg-budget" type="range" min="20" max="2000" step="20" value="' + MAX_NODES + '" style="width:96px;accent-color:var(--violet)">' +
              '<span class="cnt" id="odg-budgetv">' + MAX_NODES + '</span>' +
            '</label>' +
            '<button id="odg-showall" class="chip" style="padding:2px 10px;font-size:12px;cursor:pointer" title="Show all nodes (up to 2000)">All</button>' +
            '<span class="badge neutral" id="odg-count">-- nodes</span>' +
          '</div>' +
          '<div class="graph-legend card glass" id="odg-legend"></div>' +
          '<div class="graph-zoom card">' +
            '<button id="odg-zin" aria-label="Zoom in">+</button>' +
            '<button id="odg-zout" aria-label="Zoom out">-</button>' +
            '<button id="odg-zfit" aria-label="Fit" title="Fit to view">&#10562;</button>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<aside class="inspector">' +
        '<div class="inspector-scroll" id="odg-insp">' +
          '<div class="inspector-empty"><div style="font-size:34px;margin-bottom:8px">&#9671;</div>' +
          'Loading your knowledge graph…</div>' +
        '</div>' +
        '<div class="ask">' +
          '<div class="ask-head"><span data-ic="brain"></span> Ask your memory</div>' +
          '<div class="ask-log" id="odg-log">' +
            '<div class="msg a">Ask a question and I’ll answer from your own memory — with the facts I used as citations.</div>' +
          '</div>' +
          '<div class="ask-input">' +
            '<input id="odg-ask" placeholder="Ask over your memory…" autocomplete="off">' +
            '<button id="odg-send" data-ic="send" aria-label="Send"></button>' +
          '</div>' +
        '</div>' +
      '</aside>' +
    '</div>';
  }

  function q(sel) { return mount ? mount.querySelector(sel) : null; }

  /* ---- deterministic colour helpers ---- */
  function hashStr(s) {
    var h = 5381, i;
    for (i = 0; i < s.length; i++) h = (((h << 5) + h) + s.charCodeAt(i)) | 0;
    return (h >>> 0);   /* unsigned 32-bit -- never negative */
  }
  function commColor(cid) {
    return CAT_PALETTE[hashStr(String(cid)) % CAT_PALETTE.length];
  }

  /* ---------------- data -> tiered graph (real fields only) ---------------- */
  function shortWords(s, n) {
    return String(s || '').trim().split(/\s+/).slice(0, n || 4).join(' ');
  }
  function catColor(cat) {
    if (cat === 'episodic' || cat === 'temporal') return 'episode';
    if (cat === 'semantic' || cat === 'opinion') return 'entity';
    return 'community';
  }

  function buildGraph(graph, entityResp) {
    NODES = []; LINKS = []; idx = {};
    function add(n) {
      n.x = (Math.random() - 0.5) * 600; n.y = (Math.random() - 0.5) * 400;
      n.vx = 0; n.vy = 0;
      if (!n.r) n.r = 6 + (n.imp || 5) * 1.7;   /* fallback radius */
      NODES.push(n); idx[n.id] = n;
    }
    var rawNodes = (graph && graph.nodes) || [];
    var rawLinks = (graph && graph.links) || [];

    /* tier 1 -- communities from real community_id grouping */
    var comm = {};
    rawNodes.forEach(function (n) {
      var cid = (n.community_id === 0 || n.community_id) ? String(n.community_id) : null;
      if (!cid) return;
      if (!comm[cid]) comm[cid] = { count: 0, imp: 0 };
      comm[cid].count++;
      comm[cid].imp = Math.max(comm[cid].imp, Math.round((n.importance || 0.5) * 10));
    });
    Object.keys(comm).forEach(function (cid) {
      add({ id: 'comm:' + cid, label: 'Community · ' + comm[cid].count, tier: 1,
            type: 'community', imp: Math.min(10, 5 + Math.round(comm[cid].count / 4)),
            r: 14 + Math.min(16, comm[cid].count * 0.9),   /* radius scales with member count */
            members: comm[cid].count });
    });

    /* tier 2 -- entities from /api/entity/list */
    var entByName = {};
    var ents = (entityResp && entityResp.entities) || [];
    ents.forEach(function (e) {
      var id = 'ent:' + (e.entity_id || e.name);
      var imp = Math.max(3, Math.min(10, Math.round((e.confidence || 0.5) * 10) + 2));
      add({ id: id, label: e.name, tier: 2, type: 'entity',
            etype: e.type || 'concept', facts: e.fact_count || 0,
            imp: imp, r: 6 + imp * 2.2,                /* radius scales with importance */
            summary: e.summary_preview || '', lastSeen: (e.last_seen || '').slice(0, 10) });
      if (e.name) entByName[String(e.name).toLowerCase()] = id;
    });

    /* tier 3 -- episodes = real memory nodes */
    rawNodes.forEach(function (n) {
      var id = String(n.id);
      var content = n.content || n.content_preview || '';
      var imp = Math.max(2, Math.round((n.importance || 0.5) * 10));
      add({ id: id, label: shortWords(content, 4) + '…', tier: 3, type: 'episode',
            ftype: n.category || 'memory', imp: imp, r: 4 + imp * 1.4,   /* smallest tier */
            comm: (n.community_id === 0 || n.community_id) ? 'comm:' + n.community_id : null,
            content: content, created: (n.created_at || '').slice(0, 10),
            project: n.project_name || '', entities: n.entities || [],
            catKey: catColor(n.category) });
      /* episode -> community (real grouping) */
      if (idx['comm:' + n.community_id]) LINKS.push({ s: 'comm:' + n.community_id, t: id, rt: 'entity', w: 0.4 });
      /* episode -> entity (real: node.entities names) */
      (n.entities || []).forEach(function (nm) {
        var eid = entByName[String(nm).toLowerCase()];
        if (eid) LINKS.push({ s: eid, t: id, rt: 'entity', w: 0.3 });
      });
    });

    /* real relationship edges -- keep the strongest to avoid hairball */
    var rel = [];
    rawLinks.forEach(function (l) {
      var s = String(l.source), t = String(l.target);
      if (idx[s] && idx[t]) rel.push({ s: s, t: t, rt: l.relationship_type || 'related', w: l.weight || 0.5 });
    });
    rel.sort(function (a, b) { return b.w - a.w; });
    rel.slice(0, Math.round(NODES.length * 1.4)).forEach(function (l) { LINKS.push(l); });
  }

  /* ---------------- palette + canvas sizing ---------------- */
  function readPalette() {
    var cs = getComputedStyle(document.documentElement);
    function v(k) { return cs.getPropertyValue(k).trim(); }
    PAL = { fg: v('--fg'), fg2: v('--fg-2'), border: v('--border-strong'), card: v('--card'),
            danger: v('--danger'), cyan: v('--cyan') };
  }

  /* categorical node colour -- deterministic, theme-independent */
  function nodeColor(n) {
    if (n.type === 'episode') return EPISODE_COLOR;
    if (n.type === 'community') return commColor(n.id.replace('comm:', ''));
    /* entity: fixed type colour if known, else hash the type name */
    return ETYPE_COLORS[n.etype] || commColor(n.etype || 'unknown');
  }

  function resize() {
    if (!stage || !cv) return;
    W = stage.clientWidth; H = stage.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function visible(n) { return !!tierVisible[n.tier]; }
  function visLink(l) { return idx[l.s] && idx[l.t] && visible(idx[l.s]) && visible(idx[l.t]); }

  /* ---------------- physics (ported from design) ---------------- */
  function tick() {
    /* Force repulsion is O(n^2).  Render every requested node, but only
       simulate a stable bounded subset so the dashboard remains interactive
       for existing installs with very large graphs. */
    var vis = NODES.filter(visible).slice(0, PHYSICS_MAX_NODES), simulated = {}, i, j;
    vis.forEach(function (n) { simulated[n.id] = true; });
    for (i = 0; i < vis.length; i++) {
      var a = vis[i];
      for (j = i + 1; j < vis.length; j++) {
        var b = vis[j];
        var dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
        var f = 3600 / d2, fx = dx / d * f, fy = dy / d * f;   /* repulsion */
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    LINKS.forEach(function (l) {
      if (!visLink(l) || !simulated[l.s] || !simulated[l.t]) return;
      var a = idx[l.s], b = idx[l.t];
      var dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      var rest = (a.type === 'community' || b.type === 'community') ? 150 : 100;
      var f = (d - rest) * 0.02, fx = dx / d * f, fy = dy / d * f;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    });
    var mv2 = 0;
    vis.forEach(function (n) {
      if (n === dragNode) return;
      n.vx += -n.x * 0.002; n.vy += -n.y * 0.002;   /* gravity toward centre */
      n.vx *= 0.86; n.vy *= 0.86; n.x += n.vx; n.y += n.vy;
      if (!isFinite(n.x) || !isFinite(n.y)) { n.x = (Math.random() - 0.5) * 200; n.y = (Math.random() - 0.5) * 200; n.vx = 0; n.vy = 0; }
      var v2 = n.vx * n.vx + n.vy * n.vy;
      if (v2 > mv2) mv2 = v2;
    });
    _ke = mv2;   /* max per-node kinetic energy */
  }
  function W2S(x, y) { return [x * scale + ox + W / 2, y * scale + oy + H / 2]; }
  function S2W(px, py) { return [(px - ox - W / 2) / scale, (py - oy - H / 2) / scale]; }

  function draw() {
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    LINKS.forEach(function (l) {
      if (!visLink(l)) return;
      var a = W2S(idx[l.s].x, idx[l.s].y), b = W2S(idx[l.t].x, idx[l.t].y);
      var contra = l.rt === 'contradiction' || l.rt === 'supersedes';
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      ctx.strokeStyle = contra ? PAL.danger : PAL.border;
      ctx.globalAlpha = contra ? 0.75 : 0.45; ctx.lineWidth = contra ? 1.6 : 1;
      ctx.setLineDash(contra ? [4, 3] : []); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    });
    NODES.forEach(function (n) {
      if (!visible(n)) return;
      var p = W2S(n.x, n.y), r = n.r * Math.sqrt(scale);
      if (n === selected) { ctx.beginPath(); ctx.arc(p[0], p[1], r + 6, 0, 7); ctx.strokeStyle = PAL.cyan; ctx.lineWidth = 2; ctx.stroke(); }
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7);
      ctx.fillStyle = nodeColor(n); ctx.globalAlpha = (n === hover || n === selected) ? 1 : 0.92; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5; ctx.strokeStyle = PAL.card; ctx.stroke();
      if (n.tier < 3 || n === hover || n === selected || scale > 1.6) {
        ctx.fillStyle = PAL.fg;
        ctx.font = (n.tier === 1 ? '650 ' : '500 ') + (n.tier === 1 ? 13 : 11.5) + 'px -apple-system,system-ui,sans-serif';
        ctx.textAlign = 'center'; ctx.fillText(n.label, p[0], p[1] + r + 13);
      }
    });
  }

  function loop() {
    if (!running) return;
    tick();
    if (fitFrames < 90) { fit(); fitFrames++; }   /* keep settling cloud in view */
    draw();
    frames++;
    if ((frames > SETTLE_MIN && _ke < SETTLE_KE) || frames > SETTLE_MAX_FRAMES) {
      running = false; raf = null; return;
    }
    raf = requestAnimationFrame(loop);
  }

  /* wake: restart physics after a perturbation (drag, toggle, budget change) */
  function wake() { frames = 0; if (!running) { running = true; loop(); } }
  /* redraw: single repaint when sim is frozen (pan/zoom/hover) */
  function redraw() { if (!running) draw(); }

  /* ---------------- interactions ---------------- */
  function nodeAt(px, py) {
    var w = S2W(px, py), best = null, bd = 1e9;
    NODES.forEach(function (n) {
      if (!visible(n)) return;
      var dx = n.x - w[0], dy = n.y - w[1], d = Math.sqrt(dx * dx + dy * dy);
      if (d < n.r + 6 && d < bd) { bd = d; best = n; }
    });
    return best;
  }
  function fit() {
    var vis = NODES.filter(visible); if (!vis.length) return;
    var xs = vis.map(function (n) { return n.x; }), ys = vis.map(function (n) { return n.y; });
    var minx = Math.min.apply(0, xs), maxx = Math.max.apply(0, xs), miny = Math.min.apply(0, ys), maxy = Math.max.apply(0, ys);
    var w = maxx - minx + 160, h = maxy - miny + 160;
    scale = Math.max(0.35, Math.min(1.4, Math.min(W / w, H / h) || 1));
    ox = -((minx + maxx) / 2) * scale; oy = -((miny + maxy) / 2) * scale;
  }
  function concentric() {
    var vis = NODES.filter(visible).slice().sort(function (a, b) { return b.imp - a.imp; });
    vis.forEach(function (n, i) { var ring = 10 - n.imp, ang = i * 2.399, rad = ring * 46 + 30; n.x = Math.cos(ang) * rad; n.y = Math.sin(ang) * rad; n.vx = 0; n.vy = 0; });
  }
  function updateCount() {
    var c = q('#odg-count'); if (!c) return;
    c.textContent = NODES.filter(visible).length + ' nodes · ' + LINKS.filter(visLink).length + ' edges';
  }

  /* ---- dynamic legend: reflects what is actually on screen ---- */
  function updateLegend() {
    var el = q('#odg-legend'); if (!el) return;
    var lines = [];
    /* communities sorted by member count -- up to 8 entries */
    if (tierVisible[1]) {
      NODES.filter(function (n) { return n.type === 'community'; })
        .sort(function (a, b) { return b.members - a.members; })
        .slice(0, 8).forEach(function (cn) {
          var cid = cn.id.replace('comm:', '');
          lines.push('<div class="k"><b style="background:' + commColor(cid) + '"></b> Cluster ' + cid +
            ' <span style="opacity:0.55;font-size:10px">(' + cn.members + ')</span></div>');
        });
    }
    /* entity types -- up to 4 most common */
    if (tierVisible[2]) {
      var etypes = {};
      NODES.filter(function (n) { return n.type === 'entity'; })
        .forEach(function (n) { etypes[n.etype] = (etypes[n.etype] || 0) + 1; });
      var etArr = Object.keys(etypes).sort(function (a, b) { return etypes[b] - etypes[a]; }).slice(0, 4);
      if (etArr.length) {
        if (lines.length) lines.push('<hr style="border:none;border-top:1px solid var(--border);margin:5px 0">');
        etArr.forEach(function (et) {
          var col = ETYPE_COLORS[et] || commColor(et || 'unknown');
          lines.push('<div class="k"><b style="background:' + col + '"></b> ' + esc(et || 'unknown') +
            ' <span style="opacity:0.55;font-size:10px">(' + etypes[et] + ')</span></div>');
        });
      }
    }
    /* episodes -- single entry */
    if (tierVisible[3]) {
      if (lines.length) lines.push('<hr style="border:none;border-top:1px solid var(--border);margin:5px 0">');
      lines.push('<div class="k"><b style="background:' + EPISODE_COLOR + '"></b> Episode</div>');
    }
    if (!lines.length) lines.push('<div class="k" style="opacity:0.5">No tiers visible</div>');
    el.innerHTML = lines.join('');
  }

  /* ---------------- inspector ---------------- */
  var ETNAME = { person: 'Person', org: 'Organization', concept: 'Concept', event: 'Event' };
  function bar(v, max) {
    return '<span style="display:inline-block;vertical-align:middle;width:70px;height:6px;border-radius:99px;background:var(--card-2);overflow:hidden;margin-right:7px"><span style="display:block;height:100%;width:' + (v / max * 100) + '%;background:linear-gradient(90deg,var(--cyan),var(--violet))"></span></span>';
  }
  function field(k, v) { return '<div class="field"><div class="k">' + k + '</div><div class="v">' + v + '</div></div>'; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function escAttr(s) { return esc(s).replace(/"/g, '&quot;'); }

  function selectById(id) {
    var n = idx[id];
    if (n) {
      /* if the node's tier is hidden, reveal it and sync the button */
      if (!tierVisible[n.tier]) {
        tierVisible[n.tier] = true;
        var btn = mount ? mount.querySelector('#odg-tier button[data-tier="' + n.tier + '"]') : null;
        if (btn) btn.classList.add('active');
        updateCount(); updateLegend();
      }
      selected = n; renderInsp(n); redraw();
    }
  }
  window.odgSelect = selectById;

  function renderInsp(n) {
    var el = q('#odg-insp'); if (!el) return;
    var col = nodeColor(n);
    var badge = '<span class="badge ' + (n.type === 'episode' ? 'cyan' : n.type === 'entity' ? 'violet' : 'warn') + '">' + n.type + '</span>';
    var h = '<div style="display:flex;align-items:center;gap:11px;margin-bottom:14px">' +
      '<span class="avatar" style="background:' + col + '">' + esc((n.label[0] || '#')) + '</span>' +
      '<div><div style="font-weight:650;font-size:15px;line-height:1.2">' + esc(n.label.replace('…', '')) + '</div><div style="margin-top:4px">' + badge + '</div></div></div>';
    if (n.type === 'episode') {
      h += field('Content', '<span style="line-height:1.55">' + esc(n.content) + '</span>');
      h += field('Fact type', '<span class="badge neutral">' + esc(n.ftype) + '</span>');
      h += field('Importance', bar(n.imp, 10) + ' <b>' + n.imp + '</b>/10');
      if (n.project) h += field('Project', '<span class="mono">' + esc(n.project) + '</span>');
      if (n.created) h += field('Created', '<span class="mono">' + esc(n.created) + '</span>');
    } else if (n.type === 'entity') {
      h += field('Entity type', '<span class="badge neutral">' + esc(ETNAME[n.etype] || n.etype) + '</span>');
      h += field('Facts about this', '<b>' + n.facts + '</b> atomic facts');
      if (n.summary) h += field('Summary', '<span style="line-height:1.5">' + esc(n.summary) + '</span>');
      if (n.lastSeen) h += field('Last seen', '<span class="mono">' + esc(n.lastSeen) + '</span>');
      h += field('Connections', LINKS.filter(function (l) { return l.s === n.id || l.t === n.id; }).length + ' edges');
    } else {
      h += field('Members', '<b>' + (n.members || 0) + '</b> memories');
      h += field('Importance', bar(n.imp, 10) + ' <b>' + n.imp + '</b>/10');
    }
    var nb = LINKS.filter(function (l) { return l.s === n.id || l.t === n.id; }).slice(0, 6).map(function (l) {
      var o = idx[l.s === n.id ? l.t : l.s]; if (!o) return '';
      return '<div class="list-row" style="padding:8px 0;cursor:pointer" data-act-click="odg-select" data-odg-id="' + escAttr(o.id) + '">' +
        '<span style="width:9px;height:9px;border-radius:50%;background:' + nodeColor(o) + '"></span>' +
        '<span style="flex:1;font-size:12.5px">' + esc(o.label.replace('…', '')) + '</span>' +
        '<span class="badge neutral" style="font-size:10px">' + esc(l.rt) + '</span></div>';
    }).join('');
    if (nb) h += '<div class="field"><div class="k" style="margin-bottom:4px">Relationships</div>' + nb + '</div>';
    el.innerHTML = h;
  }

  /* ---------------- Ask-Memory (real /api/search) ---------------- */
  function push(cls, html) {
    var log = q('#odg-log'); if (!log) return null;
    var d = document.createElement('div'); d.className = 'msg ' + cls; d.innerHTML = html;
    log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
  }
  function sendAsk() {
    var input = q('#odg-ask'); if (!input) return;
    var query = input.value.trim(); if (!query) return; input.value = '';
    push('q', esc(query));
    var t = push('a', '<span class="typing"><i></i><i></i><i></i></span>');
    fetch(RECALL_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: query, limit: 5 }) })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (data) {
        var items = data.results || data.memories || data.facts || (Array.isArray(data) ? data : []);
        if (!items.length) { t.innerHTML = 'No stored memory matched that yet.'; return; }
        var top = items[0];
        var ans = esc(top.content || top.text || top.summary || ('Found ' + items.length + ' related memories.'));
        var cites = items.slice(0, 4).map(function (it) {
          var id = String(it.id || it.fact_id || '');
          var lbl = esc(shortWords(it.content || it.text || it.summary || 'fact', 4));
          return '<span class="cite"' + (idx[id] ? ' data-act-click="odg-select" data-odg-id="' + escAttr(id) + '"' : '') + '>◈ ' + lbl + '</span>';
        }).join('');
        t.innerHTML = ans + (cites ? '<div style="margin-top:8px">' + cites + '</div>' : '');
      })
      .catch(function () { t.innerHTML = 'Could not reach recall right now. Try again in a moment.'; });
  }

  /* ---------------- wiring ---------------- */
  function wireControls() {
    stage = q('#odg-stage'); cv = q('#odg-cv'); ctx = cv.getContext('2d');

    /* tier buttons -- independent toggles: each click flips visibility + re-renders */
    mount.querySelectorAll('#odg-tier button').forEach(function (b) {
      b.onclick = function () {
        b.classList.toggle('active');
        tierVisible[+b.dataset.tier] = b.classList.contains('active');
        updateCount(); updateLegend(); fit(); wake();
      };
    });

    /* layout buttons -- switch layout AND restart sim / static redraw */
    mount.querySelectorAll('#odg-layout button').forEach(function (b) {
      b.onclick = function () {
        mount.querySelectorAll('#odg-layout button').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        if (b.dataset.lay === 'concentric') {
          concentric(); fit(); redraw();          /* static radial layout -- no physics */
        } else {
          fitFrames = 0; frames = 0; wake();      /* force -- restart physics from current pos */
        }
      };
    });

    var budget = q('#odg-budget');
    if (budget) budget.oninput = function () {
      q('#odg-budgetv').textContent = budget.value; MAX_NODES = +budget.value; load();
    };

    /* "All" button -- jump to 2000 nodes and reload */
    var showall = q('#odg-showall');
    if (showall) showall.onclick = function () {
      var bgt = q('#odg-budget');
      if (bgt) { bgt.value = '2000'; q('#odg-budgetv').textContent = '2000'; }
      MAX_NODES = 2000; load();
    };

    q('#odg-zin').onclick = function () { scale = Math.min(4, scale * 1.2); redraw(); };
    q('#odg-zout').onclick = function () { scale = Math.max(0.3, scale * 0.83); redraw(); };
    q('#odg-zfit').onclick = function () { fit(); redraw(); };
    q('#odg-send').onclick = sendAsk;
    q('#odg-ask').addEventListener('keydown', function (e) { if (e.key === 'Enter') sendAsk(); });

    cv.addEventListener('mousedown', function (e) {
      var r = cv.getBoundingClientRect(), n = nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (n) { dragNode = n; selected = n; renderInsp(n); wake(); } else { panning = true; }
      last = [e.clientX, e.clientY];
    });
    onMove = function (e) {
      var r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      if (dragNode) { var w = S2W(mx, my); dragNode.x = w[0]; dragNode.y = w[1]; dragNode.vx = 0; dragNode.vy = 0; wake(); }
      else if (panning && last) { ox += e.clientX - last[0]; oy += e.clientY - last[1]; last = [e.clientX, e.clientY]; redraw(); }
      else { var wasHover = hover; hover = nodeAt(mx, my); cv.style.cursor = hover ? 'pointer' : 'grab'; if (hover !== wasHover) redraw(); }
    };
    onUp = function () { dragNode = null; panning = false; last = null; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    stage.addEventListener('wheel', function (e) {
      e.preventDefault();
      var r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, w = S2W(mx, my);
      var f = e.deltaY < 0 ? 1.12 : 0.89; scale = Math.max(0.3, Math.min(4, scale * f));
      var ns = W2S(w[0], w[1]); ox += mx - ns[0]; oy += my - ns[1];
      redraw();
    }, { passive: false });
  }

  function setInspectorEmpty() {
    var el = q('#odg-insp');
    if (el) el.innerHTML = '<div class="inspector-empty"><div style="font-size:34px;margin-bottom:8px">&#9671;</div>Click any node to inspect it.<br>Drag to reposition · scroll to zoom.</div>';
  }
  function setInspectorMsg(msg) {
    var el = q('#odg-insp');
    if (el) el.innerHTML = '<div class="inspector-empty"><div style="font-size:34px;margin-bottom:8px">&#9671;</div>' + esc(msg) + '</div>';
  }

  /* drawLoading: paint a placeholder on the canvas while /api/graph is in-flight.
     Prevents the blank-canvas flicker on tab switch (graph fetch takes ~200-800 ms). */
  function drawLoading() {
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.fillStyle = PAL.fg2 || '#8a8a8a';
    ctx.font = '13px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Loading knowledge graph…', W / 2, H / 2);
    ctx.restore();
  }

  function load() {
    drawLoading();   /* paint immediately -- prevents blank-canvas flicker */
    /* scale entity fetch with node budget so the entity tier stays representative */
    var entityLimit = Math.max(40, Math.round(MAX_NODES * 0.5));
    var url = GRAPH_URL + '?max_nodes=' + MAX_NODES;
    var entityUrl = '/api/entity/list?limit=' + entityLimit;
    Promise.all([
      fetch(url).then(function (r) { if (!r.ok) throw new Error('graph ' + r.status); return r.json(); }),
      fetch(entityUrl).then(function (r) { return r.ok ? r.json() : { entities: [] }; }).catch(function () { return { entities: [] }; })
    ]).then(function (res) {
      buildGraph(res[0], res[1]);
      if (!NODES.length) {
        setInspectorMsg('No memories in this scope yet. Store something to grow the graph.');
        running = false; draw(); updateCount(); updateLegend(); return;
      }
      readPalette(); resize(); updateCount(); updateLegend(); selected = null; setInspectorEmpty();
      for (var i = 0; i < PRE_SETTLE_TICKS; i++) tick();   /* bounded pre-settle keeps navigation responsive */
      fitFrames = 0; frames = 0;
      if (!running) { running = true; loop(); } else { wake(); }
    }).catch(function (err) {
      setInspectorMsg('Could not load the graph (' + ((err && err.message) || 'error') + ').');
      running = false;
    });
  }

  function teardown() {
    running = false;
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    if (onMove) { window.removeEventListener('mousemove', onMove); onMove = null; }
    if (onUp) { window.removeEventListener('mouseup', onUp); onUp = null; }
    if (themeObs) { themeObs.disconnect(); themeObs = null; }
  }

  /* ---------------- public entry ---------------- */
  window.odRenderGraph = function (container) {
    teardown();                          /* re-render safe */
    tierVisible = { 1: true, 2: true, 3: true };   /* reset toggles on each mount */
    mount = container;
    container.classList.add('graph-content');
    container.style.padding = '0';       /* full-bleed hero */
    container.innerHTML = scaffold();
    readPalette();
    wireControls();
    resize();
    window.addEventListener('resize', resize);
    themeObs = new MutationObserver(readPalette);
    themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    load();
  };
})();

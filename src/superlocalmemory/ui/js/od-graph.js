/* ============================================================
   od-graph.js — Knowledge Graph (Open Design port, live data)
   Faithful port of open-design/graph.html: force-directed canvas
   engine, 3-tier model (community > entity > episode), node inspector,
   Ask-Memory chat.

   Live data (NO invented values):
     GET  /api/graph?max_nodes=&min_importance=   nodes/links/clusters
     GET  /api/entity/list?limit=                 tier-2 entities
     POST /api/recall                             Ask-Memory answers
   Exposes window.odRenderGraph(container); the shell calls it on tab open.
   ============================================================ */
(function () {
  'use strict';

  var GRAPH_URL = '/api/graph';
  var ENTITY_URL = '/api/entity/list?limit=40';
  var RECALL_URL = '/api/search';   // POST {query, limit} -> {results:[...]}  (there is no /api/recall)
  var MAX_NODES = 40;           // "how many nodes to keep" budget (slider 40–600)

  /* module state (re-render safe) */
  var mount = null, cv = null, ctx = null, stage = null;
  var raf = null, running = false, fitFrames = 0;
  // Freeze the force sim once it settles so the graph stops drifting/rotating.
  // frames counts sim steps since the last wake; _ke is the total kinetic
  // energy of the last tick. Below SETTLE_KE after SETTLE_MIN steps → freeze
  // (stop the rAF loop entirely; interactions repaint on demand via redraw()).
  var frames = 0, _ke = 0;
  // _ke is the MAX per-node kinetic energy (node-count independent). Freeze
  // once the fastest node is nearly still (< ~0.45 px/frame → _ke < 0.2) after
  // a warmup, or unconditionally after a hard cap so it can never drift forever.
  var SETTLE_MIN = 60, SETTLE_KE = 0.2, SETTLE_MAX_FRAMES = 600;
  var NODES = [], LINKS = [], idx = {};
  var scale = 1, ox = 0, oy = 0, dpr = Math.max(1, window.devicePixelRatio || 1);
  var W = 0, H = 0, maxTier = 3, selected = null, hover = null;
  var dragNode = null, panning = false, last = null;
  var PAL = {}, themeObs = null;

  /* bound handlers kept so re-render can detach the old ones */
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
              '<button data-tier="1">Communities</button>' +
              '<button data-tier="2">Entities</button>' +
              '<button data-tier="3" class="active">Episodes</button>' +
            '</div>' +
            '<div class="seg" id="odg-layout">' +
              '<button data-lay="fcose" class="active">Force</button>' +
              '<button data-lay="concentric">By importance</button>' +
            '</div>' +
            '<label class="chip" style="gap:8px" title="How many nodes to keep">' +
              '<span data-ic="filter"></span> Nodes ' +
              '<input id="odg-budget" type="range" min="40" max="600" step="20" value="' + MAX_NODES + '" style="width:96px;accent-color:var(--violet)">' +
              '<span class="cnt" id="odg-budgetv">' + MAX_NODES + '</span>' +
            '</label>' +
            '<span class="badge neutral" id="odg-count">— nodes</span>' +
          '</div>' +
          '<div class="graph-legend card glass">' +
            '<div class="k"><b style="background:var(--node-episode)"></b> Episode</div>' +
            '<div class="k"><b style="background:var(--node-entity)"></b> Entity</div>' +
            '<div class="k"><b style="background:var(--node-community)"></b> Community</div>' +
          '</div>' +
          '<div class="graph-zoom card">' +
            '<button id="odg-zin" aria-label="Zoom in">+</button>' +
            '<button id="odg-zout" aria-label="Zoom out">−</button>' +
            '<button id="odg-zfit" aria-label="Fit" title="Fit to view">⤢</button>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<aside class="inspector">' +
        '<div class="inspector-scroll" id="odg-insp">' +
          '<div class="inspector-empty"><div style="font-size:34px;margin-bottom:8px">◇</div>' +
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

  /* ---------------- data → tiered graph (real fields only) ---------------- */
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
      n.vx = 0; n.vy = 0; n.r = 6 + n.imp * 1.7;
      NODES.push(n); idx[n.id] = n;
    }
    var rawNodes = (graph && graph.nodes) || [];
    var rawLinks = (graph && graph.links) || [];

    /* tier 1 — communities from real community_id grouping */
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
            members: comm[cid].count });
    });

    /* tier 2 — entities from /api/entity/list */
    var entByName = {};
    var ents = (entityResp && entityResp.entities) || [];
    ents.forEach(function (e) {
      var id = 'ent:' + (e.entity_id || e.name);
      add({ id: id, label: e.name, tier: 2, type: 'entity',
            etype: e.type || 'concept', facts: e.fact_count || 0,
            imp: Math.max(3, Math.min(10, Math.round((e.confidence || 0.5) * 10) + 2)),
            summary: e.summary_preview || '', lastSeen: (e.last_seen || '').slice(0, 10) });
      if (e.name) entByName[String(e.name).toLowerCase()] = id;
    });

    /* tier 3 — episodes = real memory nodes */
    rawNodes.forEach(function (n) {
      var id = String(n.id);
      var content = n.content || n.content_preview || '';
      add({ id: id, label: shortWords(content, 4) + '…', tier: 3, type: 'episode',
            ftype: n.category || 'memory', imp: Math.max(2, Math.round((n.importance || 0.5) * 10)),
            comm: (n.community_id === 0 || n.community_id) ? 'comm:' + n.community_id : null,
            content: content, created: (n.created_at || '').slice(0, 10),
            project: n.project_name || '', entities: n.entities || [],
            catKey: catColor(n.category) });
      // episode -> community (real grouping)
      if (idx['comm:' + n.community_id]) LINKS.push({ s: 'comm:' + n.community_id, t: id, rt: 'entity', w: 0.4 });
      // episode -> entity (real: node.entities names)
      (n.entities || []).forEach(function (nm) {
        var eid = entByName[String(nm).toLowerCase()];
        if (eid) LINKS.push({ s: eid, t: id, rt: 'entity', w: 0.3 });
      });
    });

    /* real relationship edges — keep the strongest so the layout stays readable
       (the raw graph is very densely connected; showing every edge is a hairball) */
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
    PAL = { episode: v('--node-episode'), entity: v('--node-entity'), community: v('--node-community'),
            fg: v('--fg'), fg2: v('--fg-2'), border: v('--border-strong'), card: v('--card'),
            danger: v('--danger'), cyan: v('--cyan') };
  }
  function nodeColor(n) { return n.type === 'episode' ? (PAL[n.catKey] || PAL.episode) : PAL[n.type]; }
  function resize() {
    if (!stage || !cv) return;
    W = stage.clientWidth; H = stage.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function visible(n) { return n.tier <= maxTier; }
  function visLink(l) { return idx[l.s] && idx[l.t] && visible(idx[l.s]) && visible(idx[l.t]); }

  /* ---------------- physics (ported from design) ---------------- */
  function tick() {
    var vis = NODES.filter(visible), i, j;
    for (i = 0; i < vis.length; i++) {
      var a = vis[i];
      for (j = i + 1; j < vis.length; j++) {
        var b = vis[j];
        var dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
        var f = 3600 / d2, fx = dx / d * f, fy = dy / d * f;   // repulsion (design-stable)
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    LINKS.forEach(function (l) {
      if (!visLink(l)) return;
      var a = idx[l.s], b = idx[l.t];
      var dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      var rest = (a.type === 'community' || b.type === 'community') ? 150 : 100;
      var f = (d - rest) * 0.02, fx = dx / d * f, fy = dy / d * f;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    });
    var mv2 = 0;
    vis.forEach(function (n) {
      if (n === dragNode) return;
      n.vx += -n.x * 0.002; n.vy += -n.y * 0.002;   // gravity (design-stable)
      n.vx *= 0.86; n.vy *= 0.86; n.x += n.vx; n.y += n.vy;
      if (!isFinite(n.x) || !isFinite(n.y)) { n.x = (Math.random() - 0.5) * 200; n.y = (Math.random() - 0.5) * 200; n.vx = 0; n.vy = 0; }
      var v2 = n.vx * n.vx + n.vy * n.vy;
      if (v2 > mv2) mv2 = v2;
    });
    _ke = mv2;   // max per-node kinetic energy
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
    if (fitFrames < 90) { fit(); fitFrames++; }   // keep the settling cloud in view — can't drift off
    draw();
    frames++;
    // Freeze once the layout has converged: stop the rAF loop so nodes hold
    // still (no perpetual drift/rotation). A perturbation (drag / new data /
    // budget change) calls wake() to re-run physics until it re-settles.
    if ((frames > SETTLE_MIN && _ke < SETTLE_KE) || frames > SETTLE_MAX_FRAMES) {
      running = false; raf = null; return;
    }
    raf = requestAnimationFrame(loop);
  }

  // Re-run physics after a perturbation (drag, filter, budget). Does not re-fit.
  function wake() {
    frames = 0;
    if (!running) { running = true; loop(); }
  }

  // Repaint once when the sim is frozen (pan / zoom / hover change the view but
  // not node positions, so no physics is needed — just a redraw).
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
    // clamp: never shrink the core to a dot (outliers), never zoom in so far every label shows
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

  /* ---------------- inspector ---------------- */
  var ETNAME = { person: 'Person', org: 'Organization', concept: 'Concept', event: 'Event' };
  function bar(v, max) {
    return '<span style="display:inline-block;vertical-align:middle;width:70px;height:6px;border-radius:99px;background:var(--card-2);overflow:hidden;margin-right:7px"><span style="display:block;height:100%;width:' + (v / max * 100) + '%;background:linear-gradient(90deg,var(--cyan),var(--violet))"></span></span>';
  }
  function field(k, v) { return '<div class="field"><div class="k">' + k + '</div><div class="v">' + v + '</div></div>'; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function escAttr(s) { return esc(s).replace(/"/g, '&quot;'); }  // double-quote-safe for HTML attribute values
  function selectById(id) { var n = idx[id]; if (n) { maxTier = Math.max(maxTier, n.tier); selected = n; renderInsp(n); } }
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

  /* ---------------- Ask-Memory (real /api/recall) ---------------- */
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
    mount.querySelectorAll('#odg-tier button').forEach(function (b) {
      b.onclick = function () {
        mount.querySelectorAll('#odg-tier button').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active'); maxTier = +b.dataset.tier; updateCount(); setTimeout(fit, 400);
      };
    });
    mount.querySelectorAll('#odg-layout button').forEach(function (b) {
      b.onclick = function () {
        mount.querySelectorAll('#odg-layout button').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        if (b.dataset.lay === 'concentric') concentric();
        else NODES.forEach(function (n) { n.x = (Math.random() - 0.5) * 500; n.y = (Math.random() - 0.5) * 380; });
      };
    });
    var budget = q('#odg-budget');
    if (budget) budget.oninput = function () {
      q('#odg-budgetv').textContent = budget.value; MAX_NODES = +budget.value; load();
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
    var el = q('#odg-insp'); if (el) el.innerHTML = '<div class="inspector-empty"><div style="font-size:34px;margin-bottom:8px">◇</div>Click any node to inspect it.<br>Drag to reposition · scroll to zoom.</div>';
  }
  function setInspectorMsg(msg) {
    var el = q('#odg-insp'); if (el) el.innerHTML = '<div class="inspector-empty"><div style="font-size:34px;margin-bottom:8px">◇</div>' + esc(msg) + '</div>';
  }

  function load() {
    var url = GRAPH_URL + '?max_nodes=' + MAX_NODES;
    Promise.all([
      fetch(url).then(function (r) { if (!r.ok) throw new Error('graph ' + r.status); return r.json(); }),
      fetch(ENTITY_URL).then(function (r) { return r.ok ? r.json() : { entities: [] }; }).catch(function () { return { entities: [] }; })
    ]).then(function (res) {
      buildGraph(res[0], res[1]);
      if (!NODES.length) { setInspectorMsg('No memories in this scope yet. Store something to grow the graph.'); running = false; draw(); updateCount(); return; }
      readPalette(); resize(); updateCount(); selected = null; setInspectorEmpty();
      for (var i = 0; i < 120; i++) tick();   // pre-settle so it doesn't open as a clump
      fitFrames = 0;                           // the loop auto-fits while the sim converges
      frames = 0;                              // fresh settle window before the freeze kicks in
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
    teardown();                       // re-render safe
    mount = container;
    container.classList.add('graph-content');
    container.style.padding = '0';    // full-bleed hero (design's .content.graph-content)
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

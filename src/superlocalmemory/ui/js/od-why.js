// od-why.js: the memory detail's "Why was this shown?" section (4.2.30, WS-5, section 8.10.3 h).
//
// Mounts into the memory drawer (od-memory-drawer.js calls SLMWhy.mount once per opened
// memory). Nothing is fetched until the person asks: the button reads
// GET /api/trust/why/{fact_id}, the same report `slm why` prints, and shows it in plain words:
// where the memory came from, whether the instruction firewall holds it back (and who
// approved it), whether its signature still matches, and the last times it was shown to an
// agent with the reason for each. Read-only. Every string from the server is set with
// textContent, never parsed as HTML.
(function () {
  'use strict';

  var RECENT = 5;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }

  function when(iso) {
    if (!iso) return 'date unknown';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso).slice(0, 16).replace('T', ' ') : d.toLocaleString();
  }

  function part(title, text, tone) {
    var box = el('div', 'truth-item');
    box.appendChild(el('p', 'dash-card-lead', title));
    box.appendChild(el('p', 'dash-card-meta' + (tone ? ' ' + tone : ''), text));
    return box;
  }

  function approvals(list) {
    return (list || []).map(function (a) {
      var how = a.decided_by ? a.state + ' by ' + a.decided_by + ' via ' + a.decided_channel : a.state;
      return 'It asked to be: ' + a.wanted + ' (' + how + ').';
    });
  }

  function render(container, d) {
    container.textContent = '';
    var o = d.origin || {};
    var box = part('Where it came from', o.words || '');
    box.setAttribute('data-origin', o['class'] || '');
    if (o.rule && o.rule.why) {
      box.appendChild(el('p', 'dash-card-meta', 'Why it is counted that way: ' + o.rule.why));
    }
    container.appendChild(box);

    var fw = d.firewall || {};
    var fwBox = part('Instruction firewall', fw.words || '', fw.below_system && fw.acting ? 'tone-warn' : '');
    approvals(fw.approvals).forEach(function (t) { fwBox.appendChild(el('p', 'dash-card-meta', t)); });
    container.appendChild(fwBox);

    var sig = d.signature || {};
    container.appendChild(part('Signature', sig.words || '', sig.state === 'modified' ? 'tone-warn' : ''));

    // WS-8: where on its picture or page the text was read ("from page 3, top left"); only present
    // when the "keep the place" setting is on and the memory came from a picture or a scanned page.
    var places = (d.source_regions || []).map(function (r) { return r && r.where; }).filter(Boolean);
    if (places.length) {
      var placeBox = part('Where the text was read', places.join('; ') + '.');
      placeBox.setAttribute('data-regions', String(places.length));
      container.appendChild(placeBox);
    }

    var dl = d.deliveries || {};
    var recent = (dl.recent || []).slice(0, RECENT);
    var head = dl.considered
      ? 'Shown to your agents ' + dl.shown + ' time' + (dl.shown === 1 ? '' : 's') +
        ' (considered ' + dl.considered + ')'
      : 'Shown to your agents';
    var shown = part(head, dl.considered ? '' : 'No agent has been shown this memory yet.');
    recent.forEach(function (i) {
      var row = el('p', 'dash-card-meta');
      row.appendChild(el('span', 'truth-when', when(i.at) + ': '));
      row.appendChild(document.createTextNode((i.why && i.why.words) || ''));
      shown.appendChild(row);
    });
    container.appendChild(shown);
  }

  function load(factId, container) {
    container.textContent = 'Loading...';
    return fetch('/api/trust/why/' + encodeURIComponent(factId), { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        if (!d || d.success === false) throw new Error('not available');
        render(container, d);
      })
      .catch(function () {
        container.textContent = '';
        container.appendChild(el('p', 'dash-card-meta tone-warn',
          'SuperLocalMemory could not read why this memory was shown. Try again, or run slm why in a terminal.'));
      });
  }

  function mount(factId, container) {
    if (!container || !factId) return Promise.resolve();
    container.textContent = '';
    var title = el('h3', 'truth-detail-title', 'Why was this shown?');
    title.style.fontSize = '13px'; // the theme's `.ng-dark h3` (1.25rem) outranks the class rule
    container.appendChild(title);
    var body = el('div', 'truth-why-body');
    var btn = el('button', 'btn sm ghost', 'Show where it came from and when agents saw it');
    btn.type = 'button';
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      btn.disabled = true;
      load(factId, body).then(function () { btn.hidden = true; });
    });
    container.appendChild(btn);
    container.appendChild(body);
    return Promise.resolve();
  }

  window.SLMWhy = { mount: mount, load: load, _render: render };
})();

// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// od-import.js: the Dashboard home "Bring your memories from another app" card (4.2.30, WS-9, D-19).
//
// For people who are not technical: choose a file you exported from ChatGPT, Claude, Mem0, Letta or a
// folder of notes; the card says in words what is in it ("We found 214 conversations from ChatGPT,
// March 2024 to October 2026. 12 empty ones will be skipped."), nothing is saved until you press
// Import, you watch it work, and Undo takes the whole thing back. It uses the routes `slm import` uses:
//   GET  /api/import/runs                the switch, the limits and recent imports
//   POST /api/import/preview             what a file holds (nothing is saved)
//   POST /api/import/run                 start it in the background
//   GET  /api/import/runs/{id}           progress
//   POST /api/import/runs/{id}/cancel    stop; what was saved is kept
//   POST /api/import/runs/{id}/undo      take it all back
//   POST /api/import/switch              turn importing on or off (this computer)
// The daemon records who did it (the dashboard) and lists it under "Recent decisions by your agents".
// No raw ids are ever shown, and every server string is set with textContent, never as HTML.
(function () {
  'use strict';

  var STATE = {
    queued: 'Getting ready', running: 'Importing', done: 'Done', cancelled: 'Stopped',
    failed: 'Stopped on a problem', rolling_back: 'Taking it back', rolled_back: 'Taken back'
  };
  var POLL_MS = 1500;
  var view = { enabled: false, kill: false, limit: 64, runs: [], preview: null, run: null,
               busy: false, error: '', file: '', timer: null, undoing: false };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function button(label, cls, onClick) {
    var b = el('button', 'btn sm ' + (cls || 'ghost'), label);
    b.type = 'button';
    b.addEventListener('click', function (e) { e.preventDefault(); onClick(b); });
    return b;
  }
  function detailOf(json, fallback) {
    var d = json && (json.detail || json.error);
    if (d && typeof d === 'object') d = d.message || d.code;
    return d ? String(d) : fallback;
  }
  function when(iso) {
    if (!iso) return '';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? '' : d.toLocaleDateString();
  }

  function call(method, url, body) {
    var init = { method: method, credentials: 'same-origin' };
    if (body !== undefined) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    return fetch(url, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok || !j.success) {
          var e = new Error(detailOf(j, 'That did not work. Please try again.'));
          e.status = r.status;
          throw e;
        }
        return j;
      });
    });
  }

  function say(message) { view.error = message || ''; render(); }

  // ---- reading a chosen file ------------------------------------------------------------------
  function readBase64(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () {
        var text = String(reader.result || '');
        resolve(text.slice(text.indexOf(',') + 1));
      };
      reader.onerror = function () { reject(new Error('That file could not be read.')); };
      reader.readAsDataURL(file);
    });
  }

  function chosen(file) {
    if (!file) return;
    view.error = '';
    if (file.size > view.limit * 1024 * 1024) {
      view.preview = null;
      say('That file is bigger than ' + view.limit + ' MB, which is too big to send from this page. ' +
          'Ask your agent to run: slm import run <your file>');
      return;
    }
    view.busy = true; view.file = file.name; view.preview = null; render();
    readBase64(file).then(function (b64) {
      return call('POST', '/api/import/preview', { filename: file.name, data_base64: b64 });
    }).then(function (j) {
      view.preview = j; view.busy = false; view.error = ''; render();
    }).catch(function (e) {
      view.busy = false; view.preview = null; say(e.message);
    });
  }

  // ---- doing it -------------------------------------------------------------------------------
  function start() {
    if (!view.preview || view.busy) return;
    view.busy = true; render();
    call('POST', '/api/import/run', { upload_id: view.preview.upload_id, filename: view.file })
      .then(function (j) {
        view.run = j.run; view.preview = null; view.busy = false; view.undoing = false;
        poll();
      }).catch(function (e) { view.busy = false; say(e.message); });
  }

  function poll() {
    clearTimeout(view.timer);
    if (!view.run) return;
    call('GET', '/api/import/runs/' + encodeURIComponent(view.run.run_id)).then(function (j) {
      view.run = j.run; render();
      if (['queued', 'running', 'rolling_back'].indexOf(j.run.state) >= 0) {
        view.timer = setTimeout(poll, POLL_MS);
      } else { load(); }
    }).catch(function (e) { say(e.message); });
  }

  function stopIt() {
    if (!view.run) return;
    call('POST', '/api/import/runs/' + encodeURIComponent(view.run.run_id) + '/cancel', {})
      .then(function (j) { view.run = j.run; render(); poll(); })
      .catch(function (e) { say(e.message); });
  }

  function undo(runId) {
    view.busy = true; view.undoing = true; render();
    call('POST', '/api/import/runs/' + encodeURIComponent(runId) + '/undo', {}).then(function (j) {
      view.run = j.run; view.busy = false; poll();
    }).catch(function (e) { view.busy = false; view.undoing = false; say(e.message); });
  }

  function turn(state) {
    view.busy = true; render();
    call('POST', '/api/import/switch', { state: state }).then(function () {
      view.busy = false; return load();
    }).catch(function (e) { view.busy = false; say(e.message); });
  }

  function reset() {
    clearTimeout(view.timer);
    view.preview = null; view.run = null; view.file = ''; view.error = ''; view.undoing = false;
    render(); load();
  }

  // ---- drawing --------------------------------------------------------------------------------
  function plural(n, one, many) { return n.toLocaleString() + ' ' + (n === 1 ? one : many); }

  function drawOff(body) {
    if (view.kill) {
      body.appendChild(el('p', 'dash-card-meta tone-warn',
        'Importing is paused because the SuperLocalMemory kill switch is on. ' +
        'Turn the kill switch off first, then importing can be switched on.'));
      return;
    }
    body.appendChild(el('p', 'dash-card-lead',
      'Move your memory over from ChatGPT, Claude, Mem0, Letta, or a folder of notes.'));
    body.appendChild(el('p', 'dash-card-meta',
      'Importing is turned off. When you turn it on you can choose a file, see what is in it, ' +
      'and undo the whole import afterwards.'));
    var actions = el('div', 'truth-actions');
    var b = button('Turn on importing', 'primary', function () { turn('on'); });
    b.disabled = view.busy;
    actions.appendChild(b);
    body.appendChild(actions);
  }

  function drawPick(body) {
    body.appendChild(el('p', 'dash-card-lead',
      'Move your memory over from ChatGPT, Claude, Mem0, Letta, or a folder of notes.'));
    body.appendChild(el('p', 'dash-card-meta',
      'Choose the file you exported (a .json or .zip file, an agent file, or a Markdown or text ' +
      'file). Nothing is saved until you press Import, and you can undo it afterwards. ' +
      'What you already remember is never changed.'));
    var input = el('input'); input.type = 'file'; input.id = 'dash-import-file';
    input.accept = '.json,.zip,.af,.md,.markdown,.txt';
    input.setAttribute('aria-label', 'Choose a file to import');
    input.disabled = view.busy;
    input.addEventListener('change', function () { chosen(input.files && input.files[0]); });
    var actions = el('div', 'truth-actions');
    actions.appendChild(input);
    body.appendChild(actions);
  }

  function drawPreview(body) {
    var p = view.preview;
    body.appendChild(el('p', 'dash-card-lead', p.summary));
    if (p.would_save === 0) {
      body.appendChild(el('p', 'dash-card-meta tone-warn',
        'Everything in this file is already in your memory, so there is nothing new to import.'));
    } else {
      body.appendChild(el('p', 'dash-card-meta',
        plural(p.would_save, 'piece', 'pieces') + ' will be added. They are marked as imported, ' +
        'so they never replace something you already told your agents. If one disagrees with ' +
        'something you said, it waits as a question for you.'));
    }
    var actions = el('div', 'truth-actions');
    var go = button('Import', 'primary', start);
    go.disabled = view.busy || p.would_save === 0;
    actions.appendChild(go);
    actions.appendChild(button('Choose a different file', 'ghost', reset));
    body.appendChild(actions);
  }

  function drawRun(body) {
    var r = view.run;
    var label = STATE[r.state] || r.state;
    var pct = Math.round((r.progress || 0) * 100);
    if (r.state === 'rolling_back') label = 'Taking it back';
    body.appendChild(el('p', 'dash-card-lead', label + (r.state === 'running' ? ' ... ' + pct + '%' : '')));
    if (r.state === 'running' || r.state === 'queued') {
      var bar = el('progress'); bar.max = 100; bar.value = pct;
      bar.setAttribute('aria-label', 'Import progress');
      body.appendChild(bar);
      body.appendChild(el('p', 'dash-card-meta',
        plural(r.saved_items, 'piece', 'pieces') + ' saved so far. You can leave this page; ' +
        'it keeps going.'));
      var stop = el('div', 'truth-actions');
      stop.appendChild(button('Stop', 'ghost', stopIt));
      body.appendChild(stop);
      return;
    }
    if (r.state === 'rolling_back') {
      var b2 = el('progress'); b2.setAttribute('aria-label', 'Taking the import back');
      body.appendChild(b2);
      body.appendChild(el('p', 'dash-card-meta', 'This can take a minute for a big import.'));
      return;
    }
    drawFinished(body, r);
  }

  function drawFinished(body, r) {
    if (r.state === 'done' || r.state === 'cancelled') {
      body.appendChild(el('p', 'dash-card-meta',
        plural(r.saved_items, 'piece', 'pieces') + ' from ' + (r.source_label || 'your file') +
        ' are now in your memory.'));
    }
    if (r.state === 'rolled_back') {
      body.appendChild(el('p', 'dash-card-meta tone-ok',
        'Undone. Your memory is exactly as it was before this import.'));
    }
    if (r.note) body.appendChild(el('p', 'dash-card-meta tone-warn', r.note));
    var left = (r.skipped_words || []).filter(function (s) { return s.count > 0; });
    if (left.length && r.state !== 'rolled_back') {
      body.appendChild(el('p', 'dash-card-meta', 'Left out: ' + left.map(function (s) {
        return s.count.toLocaleString() + ' ' + s.words;
      }).join(', ') + '.'));
    }
    if (r.disputes_opened) {
      body.appendChild(el('p', 'dash-card-meta',
        plural(r.disputes_opened, 'thing in this import disagrees', 'things in this import disagree') +
        ' with what you already had. ' + (r.disputes_opened === 1 ? 'It is' : 'They are') +
        ' waiting under "Needs your answer"; your earlier memory stays as it is.'));
    }
    var actions = el('div', 'truth-actions');
    if (r.undoable && r.state !== 'rolled_back') {
      var u = button('Undo this import', 'ghost', function () { undo(r.run_id); });
      u.disabled = view.busy;
      actions.appendChild(u);
    }
    actions.appendChild(button('Import another file', 'ghost', reset));
    body.appendChild(actions);
  }

  function drawRecent(body) {
    var shown = view.runs.filter(function (r) { return !view.run || r.run_id !== view.run.run_id; })
      .slice(0, 3);
    if (!shown.length) return;
    body.appendChild(el('p', 'dash-card-meta', 'Earlier imports'));
    shown.forEach(function (r) {
      var row = el('div', 'truth-item');
      row.appendChild(el('p', 'dash-card-lead',
        plural(r.saved_items, 'piece', 'pieces') + ' from ' + (r.source_label || 'a file')));
      row.appendChild(el('p', 'dash-card-meta',
        (STATE[r.state] || r.state) + (when(r.created_at) ? ' · ' + when(r.created_at) : '')));
      if (r.undoable && r.state !== 'rolled_back') {
        var actions = el('div', 'truth-actions');
        actions.appendChild(button('Undo', 'ghost', function () {
          view.run = r; undo(r.run_id);
        }));
        row.appendChild(actions);
      }
      body.appendChild(row);
    });
  }

  function render() {
    var card = $('dash-import-card'), body = $('dash-import-body');
    if (!card || !body) return;
    body.textContent = '';
    if (!view.enabled) drawOff(body);
    else if (view.run) drawRun(body);
    else if (view.preview) drawPreview(body);
    else drawPick(body);
    if (view.busy && !view.run) body.appendChild(el('p', 'dash-card-meta', 'One moment ...'));
    if (view.error) body.appendChild(el('p', 'dash-card-meta tone-warn truth-error', view.error));
    if (view.enabled && !view.preview) drawRecent(body);
  }

  function load() {
    if (!$('dash-import-card')) return Promise.resolve();
    return fetch('/api/import/runs?limit=5', { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (j) {
        view.enabled = !!j.enabled; view.kill = !!j.kill_switch;
        view.limit = (j.limits && j.limits.max_upload_mb) || 64;
        view.runs = j.runs || [];
        var live = view.runs.filter(function (r) {
          return ['queued', 'running', 'rolling_back'].indexOf(r.state) >= 0;
        })[0];
        if (live && !view.run) { view.run = live; poll(); }
        render();
      }, function () {
        view.enabled = false; render();
      });
  }

  window.SLMImport = { reload: load, _render: render, _state: view, _choose: chosen };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(load, 500); });
}());

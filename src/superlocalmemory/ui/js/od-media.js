// od-media.js — Documents & Images pane: upload, thumbnails, documents, lint.
// Public: window.odRenderMedia(pane)
// XSS-safe: file names, OCR previews, titles and server messages are untrusted and
// are set with textContent only. Writes go through the page's fetch, which core.js
// wraps with the local write credential.
// Routes: POST /api/v3/media/remember   POST /api/v3/documents
//         GET /api/v3/media/{id}/thumb  GET /api/v3/jobs/{id}
//         GET /api/v3/documents         GET /api/v3/documents/lint
//         DELETE /api/v3/documents/{id}
(function () {
  'use strict';

  var MB = 1024 * 1024;
  var IMAGE_LIMIT = 25 * MB;
  var PDF_LIMIT = 100 * MB;
  var POLL_MS = 2000;
  var ID_RE = /^[0-9a-f]{32}$/;
  var FINAL_JOB = { done: 1, failed: 1, cancelled: 1 };

  function F() { return window.odFeatures; }
  function el(tag, cls, text) { return F().el(tag, cls, text); }

  // ------------------------------------------------------------------ files
  function kindOf(file) {
    var type = String(file.type || '').toLowerCase();
    var name = String(file.name || '').toLowerCase();
    if (type.indexOf('image/') === 0) return 'image';
    if (type === 'application/pdf' || /\.pdf$/.test(name)) return 'pdf';
    return '';
  }

  // Returns the reason a file is refused, or '' when it may be sent.
  function refusal(file) {
    var kind = kindOf(file);
    if (!kind) return 'Choose an image or a PDF.';
    if (kind === 'image' && file.size > IMAGE_LIMIT) return 'Too large: images can be up to 25 MB.';
    if (kind === 'pdf' && file.size > PDF_LIMIT) return 'Too large: PDFs can be up to 100 MB.';
    return '';
  }

  function readBase64(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { resolve(String(r.result).replace(/^data:[^,]*,/, '')); };
      r.onerror = function () { reject(new Error('Could not read the file.')); };
      r.readAsDataURL(file);
    });
  }

  // ---------------------------------------------------------------- results
  function newResult(ui, name) {
    var li = el('li');
    li.appendChild(el('strong', null, name));
    var status = el('span', 'muted', ' Sending…');
    li.appendChild(status);
    ui.results.appendChild(li);
    return { li: li, status: status };
  }

  function setStatus(row, text) { row.status.textContent = ' ' + text; }

  function addThumb(ui, mediaId, name) {
    if (!ID_RE.test(String(mediaId || ''))) return;
    var fig = el('figure');
    var img = el('img');
    img.setAttribute('src', '/api/v3/media/' + mediaId + '/thumb');
    img.setAttribute('alt', name);
    img.setAttribute('loading', 'lazy');
    img.addEventListener('error', function () { img.style.visibility = 'hidden'; });
    fig.appendChild(img);
    fig.appendChild(el('figcaption', null, name));
    ui.grid.appendChild(fig);
  }

  function showImageReceipt(ui, row, name, r) {
    if (r.status === 'refused') return setStatus(row, 'Not saved: ' + (r.reason || 'refused'));
    if (r.status === 'warming') return setStatus(row, 'The image reader is starting up. Try again in a moment.');
    setStatus(row, r.status === 'duplicate' ? 'Already saved.' : 'Saved.');
    addThumb(ui, r.media_id || r.duplicate_of, name);
    if (r.extracted_text_preview) row.li.appendChild(el('p', 'muted', r.extracted_text_preview));
  }

  function postOutcome(res, row, fallback) {
    if (res.status === 404) { setStatus(row, 'Not available in this build.'); return false; }
    if (res.ok || (res.data && res.data.status)) return true;
    setStatus(row, F().failText(res, fallback));
    return false;
  }

  function sendImage(ui, file, row, b64) {
    return F().api('POST', '/api/v3/media/remember', { base64: b64 }).then(function (res) {
      if (postOutcome(res, row, 'Could not save the image.')) showImageReceipt(ui, row, file.name, res.data);
    });
  }

  function sendPdf(ui, file, row, b64) {
    var body = { base64: b64, file_name: String(file.name || '').slice(0, 255) };
    return F().api('POST', '/api/v3/documents', body).then(function (res) {
      if (!postOutcome(res, row, 'Could not save the PDF.')) return;
      var r = res.data;
      if (r.status === 'refused') return setStatus(row, 'Not saved: ' + (r.reason || 'refused'));
      if (r.status === 'duplicate') return setStatus(row, 'Already saved.');
      setStatus(row, 'Queued.');
      if (r.job_id) pollJob(ui, row, r.job_id);
    });
  }

  function uploadFile(ui, file) {
    var row = newResult(ui, file.name || 'file');
    var why = refusal(file);
    if (why) { setStatus(row, why); return Promise.resolve(); }
    return readBase64(file).then(function (b64) {
      return kindOf(file) === 'image' ? sendImage(ui, file, row, b64) : sendPdf(ui, file, row, b64);
    }).catch(function (e) { setStatus(row, e && e.message ? e.message : 'Could not send the file.'); });
  }

  function onPick(ui, input) {
    var files = Array.prototype.slice.call(input.files || []);
    return files.reduce(function (p, f) {
      return p.then(function () { return uploadFile(ui, f); });
    }, Promise.resolve());
  }

  // ------------------------------------------------------------------- jobs
  function pollJob(ui, row, jobId) {
    window.setTimeout(function () {
      if (!ui.root.isConnected) return;
      F().api('GET', '/api/v3/jobs/' + jobId).then(function (res) {
        var j = res.data || {};
        if (!res.ok) return setStatus(row, F().failText(res, 'Could not read progress.'));
        if (!FINAL_JOB[j.state]) {
          setStatus(row, 'Reading page ' + (j.done || 0) + ' of ' + (j.total || '?') + '…');
          return pollJob(ui, row, jobId);
        }
        setStatus(row, j.state === 'done' ? 'Done.' : 'Did not finish (' + (j.error || j.state) + ').');
        loadDocuments(ui);
        loadLint(ui);
      });
    }, POLL_MS);
  }

  // -------------------------------------------------------------- documents
  function pagesLine(d) {
    var parts = [(d.page_count || 0) + ' pages'];
    if (d.pages_ocr) parts.push(d.pages_ocr + ' read as images');
    if (d.pages_empty) parts.push(d.pages_empty + ' empty');
    return parts.join(', ');
  }

  function removeDocument(ui, d) {
    F().confirmThen({
      title: 'Remove document', target: String(d.title || 'Document').slice(0, 80),
      consequence: 'Its saved pages are removed from your memory.', confirmLabel: 'Remove',
    }, function () {
      F().api('DELETE', '/api/v3/documents/' + d.document_id).then(function (res) {
        if (res.ok) return loadDocuments(ui);
        ui.docs.appendChild(el('p', 'muted', F().failText(res, 'Could not remove the document.')));
      });
    });
  }

  function documentRow(ui, d) {
    var li = el('li');
    li.appendChild(el('strong', null, d.title || 'Untitled'));
    li.appendChild(el('span', 'muted', ' ' + (d.state || '') + ' · ' + pagesLine(d)));
    var names = (d.entities || []).map(function (e) { return e.name; }).join(', ');
    if (names) li.appendChild(el('div', 'muted', names));
    if (ID_RE.test(String(d.document_id || ''))) {
      li.appendChild(F().button('Remove', 'btn sm', function () { removeDocument(ui, d); }));
    }
    return li;
  }

  function loadDocuments(ui) {
    return F().api('GET', '/api/v3/documents?limit=50').then(function (res) {
      ui.docs.textContent = '';
      if (res.status === 404) return F().notAvailable(ui.docs, 'The document list');
      if (!res.ok) return ui.docs.appendChild(el('p', 'muted', F().failText(res, 'Could not load documents.')));
      var docs = res.data.documents || [];
      if (!docs.length) return ui.docs.appendChild(el('p', 'muted', 'No documents yet.'));
      var ul = el('ul', 'od-media-list');
      docs.forEach(function (d) { ul.appendChild(documentRow(ui, d)); });
      ui.docs.appendChild(ul);
    });
  }

  function lintLine(d) {
    var parts = [];
    parts.push((d.empty_pages || []).length + ' empty pages');
    parts.push((d.no_entities || []).length + ' pages without people or places');
    parts.push((d.duplicate_pages || []).length + ' repeated pages');
    parts.push(d.contradicted == null ? 'contradictions not checked'
      : d.contradicted.length + ' pages that disagree with newer memories');
    return 'Check: ' + parts.join(', ') + '.';
  }

  function loadLint(ui) {
    return F().api('GET', '/api/v3/documents/lint').then(function (res) {
      ui.lint.textContent = '';
      if (res.status === 404) return F().notAvailable(ui.lint, 'The document check');
      if (res.ok) ui.lint.appendChild(el('p', 'muted', lintLine(res.data)));
    });
  }

  // ------------------------------------------------------------------- pane
  function buildBody(host) {
    host.textContent = '';
    var ui = { root: host, results: el('ul', 'od-media-list'), grid: el('div', 'od-media-grid'),
               docs: el('div'), lint: el('div') };
    var input = el('input');
    input.type = 'file';
    input.setAttribute('accept', 'image/*,application/pdf,.pdf');
    input.multiple = true;
    input.addEventListener('change', function () { onPick(ui, input); });
    var up = el('div', 'card card-pad od-media-section');
    up.appendChild(el('h3', null, 'Add images or PDFs'));
    up.appendChild(el('p', 'muted', 'Images up to 25 MB and PDFs up to 100 MB. They stay on this computer.'));
    [input, ui.results, ui.grid].forEach(function (n) { up.appendChild(n); });
    var docs = el('div', 'od-media-section');
    docs.appendChild(el('h3', null, 'Documents'));
    docs.appendChild(ui.lint);
    docs.appendChild(ui.docs);
    host.appendChild(up);
    host.appendChild(docs);
    loadDocuments(ui);
    loadLint(ui);
  }

  function odRenderMedia(pane) {
    pane.textContent = '';
    var head = el('div', 'page-head');
    head.appendChild(el('h2', null, 'Documents & Images'));
    head.appendChild(el('p', 'muted', 'Remember pictures and read PDFs on this computer.'));
    var cardHost = el('div');
    var body = el('div');
    [head, cardHost, body].forEach(function (n) { pane.appendChild(n); });
    F().mountMediaCard(cardHost, {
      onReady: function () { if (!body.firstChild) buildBody(body); },
      onNotReady: function () { body.textContent = ''; },
    });
  }

  window.odRenderMedia = odRenderMedia;
})();

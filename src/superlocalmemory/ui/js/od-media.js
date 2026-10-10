// od-media.js — Documents & Images pane: upload, thumbnails, documents, lint.
// Public: window.odRenderMedia(pane)
// XSS-safe: file names, OCR previews, titles and server messages are untrusted and
// are set with textContent only. Writes go through the page's fetch, which core.js
// wraps with the local write credential.
// The Folders section (od-sources.js) is rendered below the documents list.
// Routes: POST /api/v3/media/upload?kind=image|pdf (the file itself as the body, streamed)
//         GET /api/v3/media (saved images)  GET /api/v3/media/{id}/thumb  GET /api/v3/jobs/{id}
//         GET /api/v3/documents         GET /api/v3/documents/lint
//         DELETE /api/v3/documents/{id}  POST /api/v3/documents/{id}/retry (a failed document, from its stored copy)
(function () {
  'use strict';

  var MB = 1024 * 1024;
  var IMAGE_LIMIT = 25 * MB;
  var PDF_LIMIT = 100 * MB;
  var POLL_MS = 2000;
  var UPLOAD_TIMEOUT_MS = 10 * 60 * 1000;
  var ID_RE = /^[0-9a-f]{32}$/;
  var FINAL_JOB = { done: 1, failed: 1, cancelled: 1 };
  var PAGE = 60;

  // What a job's short failure code means, in plain words. The codes come from the PDF
  // reader (documents/parse_proc.py, runtimes/pdf_parse.py) and the document pipeline
  // (documents/pipeline.py); any other code gets the fallback sentence.
  var JOB_FAILURES = {
    encrypted: 'This PDF is password-protected. Remove the password and add it again.',
    too_many_pages: 'This PDF has more pages than SuperLocalMemory reads (500). Split it and add the parts.',
    page_timeout: 'One page took too long to read. Try again, or re-save the PDF and add it again.',
    time_limit: 'Reading this PDF took too long. Try again, or split it into smaller parts.',
    memory_limit: 'This PDF needed more memory than is available. Close other apps and try again.',
    parse_ended: 'The PDF reader stopped early. Try again.',
    save_failed: 'The pages were read but could not be saved. Try again.',
    image_tools: 'The picture tools are not ready. Check Images and documents in settings, then try again.',
    failed: 'This PDF could not be read. It may be damaged. Try again, or re-save it and add it again.'
  };
  var JOB_FAILURE_FALLBACK = 'This PDF could not be finished. Try adding it again.';

  function failureText(code) {
    var known = Object.prototype.hasOwnProperty.call(JOB_FAILURES, code) ? JOB_FAILURES[code] : '';
    return known || JOB_FAILURE_FALLBACK;
  }

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

  // The file goes up as the raw request body, so the sizes above are the real limits
  // (a JSON/base64 body would stop an image near 9 MB and a PDF near 25 MB). Resolves
  // {ok, status, data} like odFeatures.api; a dropped connection is status 0.
  function sendFile(kind, file) {
    var url = '/api/v3/media/upload?kind=' + kind;
    if (kind === 'pdf') url += '&file_name=' + encodeURIComponent(String(file.name || '').slice(0, 255));
    var init = { method: 'POST', body: file, timeoutMs: UPLOAD_TIMEOUT_MS,
      headers: { 'Content-Type': file.type || 'application/octet-stream' } };
    return fetch(url, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { ok: !!r.ok, status: r.status, data: data || {} };
      });
    }, function () { return { ok: false, status: 0, data: {} }; });
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
    if (ui.shown[mediaId]) return;
    ui.shown[mediaId] = true;
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

  // ----------------------------------------------------------- saved images
  function savedName(item) {
    var day = String(item.created_at || '').slice(0, 10);
    return day ? 'Saved ' + day : 'Saved image';
  }

  function showMore(ui, cursor) {
    if (ui.more) { ui.more.remove(); ui.more = null; }
    if (!cursor) return;
    ui.more = F().button('Show more', 'btn sm', function () { loadImages(ui, cursor); });
    ui.gridWrap.appendChild(ui.more);
  }

  function loadImages(ui, cursor) {
    var url = '/api/v3/media?limit=' + PAGE + (cursor ? '&cursor=' + encodeURIComponent(cursor) : '');
    return F().api('GET', url).then(function (res) {
      if (ui.note) { ui.note.remove(); ui.note = null; }
      if (res.status === 404) return;
      if (!res.ok) {
        ui.note = el('p', 'muted', 'Saved images could not be loaded.');
        return ui.gridWrap.appendChild(ui.note);
      }
      (res.data.items || []).forEach(function (it) {
        if (it && it.has_thumb) addThumb(ui, it.media_id, savedName(it));
      });
      showMore(ui, typeof res.data.next_cursor === 'string' ? res.data.next_cursor : '');
    });
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

  function sendImage(ui, file, row) {
    return sendFile('image', file).then(function (res) {
      if (postOutcome(res, row, 'Could not save the image.')) showImageReceipt(ui, row, file.name, res.data);
    });
  }

  function sendPdf(ui, file, row) {
    return sendFile('pdf', file).then(function (res) {
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
    var send = kindOf(file) === 'image' ? sendImage : sendPdf;
    return send(ui, file, row).catch(function (e) {
      setStatus(row, e && e.message ? e.message : 'Could not send the file.');
    });
  }

  // One at a time, in order; each file gets its own result line.
  function uploadAll(ui, list) {
    var files = Array.prototype.slice.call(list || []);
    return files.reduce(function (p, f) {
      return p.then(function () { return uploadFile(ui, f); });
    }, Promise.resolve());
  }

  function onPick(ui, input) { return uploadAll(ui, input.files); }

  // The dashed area people drop files on, with a real button for those who would rather
  // choose. Both end in the same upload as the native input does.
  function dropZone(ui, input) {
    var zone = el('div', 'od-dropzone');
    zone.appendChild(el('p', null, 'Drop pictures or PDFs here, or choose files'));
    zone.appendChild(F().button('Choose files', 'btn sm primary', function () { input.click(); }));
    zone.appendChild(el('p', 'muted', 'Images up to 25 MB and PDFs up to 100 MB. They stay on this computer.'));
    zone.addEventListener('dragover', function (e) { e.preventDefault(); zone.classList.add('is-over'); });
    zone.addEventListener('dragleave', function () { zone.classList.remove('is-over'); });
    zone.addEventListener('drop', function (e) {
      e.preventDefault();
      zone.classList.remove('is-over');
      uploadAll(ui, e.dataTransfer && e.dataTransfer.files);
    });
    return zone;
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
        setStatus(row, j.state === 'done' ? 'Done.'
          : j.state === 'cancelled' ? 'Cancelled.' : 'Did not finish. ' + failureText(String(j.error || '')));
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

  // A failed document is read again from the copy already saved: nothing is dropped again.
  function retryDocument(ui, d, li, button) {
    button.disabled = true;
    var row = { status: el('span', 'muted', ' Trying again…') };
    li.appendChild(row.status);
    F().api('POST', '/api/v3/documents/' + d.document_id + '/retry').then(function (res) {
      if (!res.ok) {
        button.disabled = false;
        return setStatus(row, F().failText(res, 'Could not try again.'));
      }
      var jobId = String((res.data && res.data.job_id) || '');
      if (res.data && res.data.status === 'processing' && ID_RE.test(jobId)) {
        setStatus(row, 'Reading…');
        return pollJob(ui, row, jobId);
      }
      loadDocuments(ui);
    });
  }

  function documentRow(ui, d) {
    var li = el('li');
    li.appendChild(el('strong', null, d.title || 'Untitled'));
    li.appendChild(el('span', 'muted', ' ' + (d.state || '') + ' · ' + pagesLine(d)));
    var names = (d.entities || []).map(function (e) { return e.name; }).join(', ');
    if (names) li.appendChild(el('div', 'muted', names));
    if (ID_RE.test(String(d.document_id || ''))) {
      if (d.state === 'failed') {
        li.appendChild(F().button('Try again', 'btn sm primary', function (b) { retryDocument(ui, d, li, b); }));
      }
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
               docs: el('div'), lint: el('div'), shown: {}, more: null, note: null,
               gridWrap: el('div') };
    ui.gridWrap.appendChild(ui.grid);
    var input = el('input');
    input.type = 'file';
    input.hidden = true;                      // the drop zone's button opens it
    input.setAttribute('accept', 'image/*,application/pdf,.pdf');
    input.multiple = true;
    input.addEventListener('change', function () { onPick(ui, input); });
    var up = el('div', 'card card-pad od-media-section');
    up.appendChild(el('h3', null, 'Add images or PDFs'));
    up.appendChild(dropZone(ui, input));
    [input, ui.results, ui.gridWrap].forEach(function (n) { up.appendChild(n); });
    var docs = el('div', 'od-media-section');
    docs.appendChild(el('h3', null, 'Documents'));
    docs.appendChild(ui.lint);
    docs.appendChild(ui.docs);
    if (window.odMediaFind) window.odMediaFind.mount(host);
    host.appendChild(up);
    host.appendChild(docs);
    loadDocuments(ui);
    loadLint(ui);
    loadImages(ui, '');
  }

  var GB = 1024; // MB per GB in the memory line, the same 1024 the MB limits above use

  function gb(mb) { return (mb / GB).toFixed(1) + ' GB'; }

  // One muted line: what the picture model uses right now. Drawn from the features read
  // the pane already makes, so it adds no polling.
  function ramText(ram) {
    if (!ram) return '';
    if (ram.worker_rss_mb == null) return 'Picture model: not running';
    return 'Picture model: ' + gb(ram.worker_rss_mb) + ' in use'
      + (ram.worker_cap_mb ? ', limit ' + gb(ram.worker_cap_mb) : '');
  }

  function odRenderMedia(pane) {
    pane.textContent = '';
    var head = el('div', 'page-head');
    head.appendChild(el('h2', null, 'Documents & Images'));
    head.appendChild(el('p', 'muted', 'Remember pictures and read PDFs on this computer.'));
    var cardHost = el('div');
    var ramLine = el('p', 'muted');
    ramLine.setAttribute('data-od-ram', '');
    ramLine.hidden = true;
    var body = el('div');
    var folders = el('div', 'od-media-section');
    [head, cardHost, ramLine, body, folders].forEach(function (n) { pane.appendChild(n); });
    // Folders have their own switch, so they show whether or not images are on; drawn once.
    if (typeof window.odRenderSources === 'function') window.odRenderSources(folders);
    F().mountMediaCard(cardHost, {
      onReady: function (m) {
        var text = ramText(m && m.ram);
        ramLine.textContent = text;
        ramLine.hidden = !text;
        if (!body.firstChild) buildBody(body);
      },
      onNotReady: function () { body.textContent = ''; ramLine.hidden = true; },
    });
  }

  window.odRenderMedia = odRenderMedia;
})();

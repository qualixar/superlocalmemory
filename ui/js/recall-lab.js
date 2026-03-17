// SuperLocalMemory V3 — Recall Lab with Dual-Display + Pagination
// Part of Qualixar | https://superlocalmemory.com

var recallLabState = {
    allResults: [],
    page: 0,
    perPage: 10,
    query: '',
    synthesis: '',
};

document.getElementById('recall-lab-search')?.addEventListener('click', function() {
    var query = document.getElementById('recall-lab-query').value.trim();
    if (!query) return;

    recallLabState.query = query;
    recallLabState.page = 0;
    var perPageEl = document.getElementById('recall-lab-per-page');
    recallLabState.perPage = perPageEl ? parseInt(perPageEl.value) : 10;
    var fetchLimit = Math.max(recallLabState.perPage * 5, 50);

    var resultsDiv = document.getElementById('recall-lab-results');
    var metaDiv = document.getElementById('recall-lab-meta');
    resultsDiv.textContent = '';
    var spinner = document.createElement('div');
    spinner.className = 'text-center py-4';
    var spinnerInner = document.createElement('div');
    spinnerInner.className = 'spinner-border text-primary';
    spinner.appendChild(spinnerInner);
    resultsDiv.appendChild(spinner);

    fetch('/api/v3/recall/trace', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query, limit: fetchLimit, synthesize: true})
    }).then(function(r) { return r.json(); }).then(function(data) {
        if (data.error) {
            resultsDiv.textContent = '';
            var errDiv = document.createElement('div');
            errDiv.className = 'alert alert-danger';
            errDiv.textContent = data.error;
            resultsDiv.appendChild(errDiv);
            return;
        }

        metaDiv.textContent = '';
        appendMetaField(metaDiv, 'Query type: ', data.query_type || 'unknown');
        metaDiv.appendChild(document.createTextNode(' | '));
        appendMetaField(metaDiv, 'Results: ', String((data.results || []).length));
        metaDiv.appendChild(document.createTextNode(' | '));
        appendMetaField(metaDiv, 'Time: ', (data.retrieval_time_ms || 0).toFixed(0) + 'ms');

        recallLabState.allResults = data.results || [];
        recallLabState.synthesis = data.synthesis || '';

        if (recallLabState.allResults.length === 0) {
            resultsDiv.textContent = '';
            var infoDiv = document.createElement('div');
            infoDiv.className = 'alert alert-info';
            infoDiv.textContent = 'No results found.';
            resultsDiv.appendChild(infoDiv);
            return;
        }

        renderRecallPage();
    }).catch(function(e) {
        resultsDiv.textContent = '';
        var errDiv = document.createElement('div');
        errDiv.className = 'alert alert-danger';
        errDiv.textContent = 'Error: ' + e.message;
        resultsDiv.appendChild(errDiv);
    });
});

function renderRecallPage() {
    var resultsDiv = document.getElementById('recall-lab-results');
    resultsDiv.textContent = '';

    var results = recallLabState.allResults;
    var start = recallLabState.page * recallLabState.perPage;
    var end = Math.min(start + recallLabState.perPage, results.length);
    var pageResults = results.slice(start, end);
    var totalPages = Math.ceil(results.length / recallLabState.perPage);

    // Synthesis banner (Mode B/C only)
    if (recallLabState.synthesis && recallLabState.page === 0) {
        var synBanner = document.createElement('div');
        synBanner.className = 'alert alert-light border-start border-4 border-primary mb-3';
        var synTitle = document.createElement('strong');
        synTitle.textContent = 'AI Summary';
        synBanner.appendChild(synTitle);
        synBanner.appendChild(document.createElement('br'));
        var synText = document.createElement('span');
        synText.textContent = recallLabState.synthesis;
        synBanner.appendChild(synText);
        resultsDiv.appendChild(synBanner);
    }

    var listGroup = document.createElement('div');
    listGroup.className = 'list-group';

    pageResults.forEach(function(r, i) {
        var globalIndex = start + i;
        var channels = r.channel_scores || {};
        var maxChannel = Math.max(channels.semantic || 0, channels.bm25 || 0, channels.entity_graph || 0, channels.temporal || 0) || 1;
        var hasSource = r.source_content && r.source_content.length > 0;
        var displayText = hasSource ? r.source_content : r.content;

        var item = document.createElement('div');
        item.className = 'list-group-item';

        // Score badge row
        var scoreRow = document.createElement('div');
        scoreRow.className = 'd-flex justify-content-between align-items-center mb-1';
        var numLabel = document.createElement('strong');
        numLabel.textContent = '#' + (globalIndex + 1);
        scoreRow.appendChild(numLabel);
        var scoreBadges = document.createElement('div');
        scoreBadges.innerHTML = '<span class="badge bg-primary me-1">Score: ' + r.score + '</span>' +
            '<span class="badge bg-secondary me-1">Trust: ' + r.trust_score + '</span>' +
            '<span class="badge bg-outline-info" style="border:1px solid #0dcaf0;color:#0dcaf0;">Conf: ' + r.confidence + '</span>';
        scoreRow.appendChild(scoreBadges);
        item.appendChild(scoreRow);

        // Original memory text (primary display)
        var contentDiv = document.createElement('div');
        contentDiv.className = 'mb-2';
        contentDiv.style.cssText = 'white-space:pre-wrap; line-height:1.5;';
        var truncated = displayText.length > 500 ? displayText.substring(0, 500) + '...' : displayText;
        contentDiv.textContent = truncated;
        item.appendChild(contentDiv);

        // Expandable atomic fact section (only if source differs from content)
        if (hasSource && r.content !== r.source_content) {
            var expandBtn = document.createElement('button');
            expandBtn.className = 'btn btn-sm btn-outline-secondary mb-2';
            expandBtn.textContent = 'Show matched fact + channels';
            var factSection = document.createElement('div');
            factSection.style.display = 'none';
            factSection.className = 'border-top pt-2 mt-1';

            var factLabel = document.createElement('small');
            factLabel.className = 'text-muted d-block mb-1';
            factLabel.textContent = 'Matched atomic fact:';
            factSection.appendChild(factLabel);

            var factContent = document.createElement('div');
            factContent.className = 'small bg-light p-2 rounded mb-2';
            factContent.textContent = r.content;
            factSection.appendChild(factContent);

            // Channel bars
            factSection.appendChild(buildChannelBar('Semantic', channels.semantic || 0, maxChannel, 'primary'));
            factSection.appendChild(buildChannelBar('BM25', channels.bm25 || 0, maxChannel, 'success'));
            factSection.appendChild(buildChannelBar('Entity', channels.entity_graph || 0, maxChannel, 'info'));
            factSection.appendChild(buildChannelBar('Temporal', channels.temporal || 0, maxChannel, 'warning'));

            expandBtn.addEventListener('click', function() {
                var visible = factSection.style.display !== 'none';
                factSection.style.display = visible ? 'none' : 'block';
                expandBtn.textContent = visible ? 'Show matched fact + channels' : 'Hide matched fact';
            });

            item.appendChild(expandBtn);
            item.appendChild(factSection);
        } else {
            // No source_content — show channel bars inline
            var barsDiv = document.createElement('div');
            barsDiv.className = 'mt-1';
            barsDiv.appendChild(buildChannelBar('Semantic', channels.semantic || 0, maxChannel, 'primary'));
            barsDiv.appendChild(buildChannelBar('BM25', channels.bm25 || 0, maxChannel, 'success'));
            barsDiv.appendChild(buildChannelBar('Entity', channels.entity_graph || 0, maxChannel, 'info'));
            barsDiv.appendChild(buildChannelBar('Temporal', channels.temporal || 0, maxChannel, 'warning'));
            item.appendChild(barsDiv);
        }

        // Click for detail modal
        item.style.cursor = 'pointer';
        (function(result) {
            contentDiv.addEventListener('click', function() {
                if (typeof openMemoryDetail === 'function') {
                    openMemoryDetail({
                        id: result.fact_id,
                        memory_id: result.memory_id,
                        content: result.source_content || result.content,
                        score: result.score,
                        importance: Math.round((result.confidence || 0.5) * 10),
                        category: 'recall',
                        created_at: null,
                        trust_score: result.trust_score,
                        channel_scores: result.channel_scores
                    }, 'recall'); // source='recall': show View Original, hide Expand Neighbors
                }
            });
        })(r);

        listGroup.appendChild(item);
    });
    resultsDiv.appendChild(listGroup);

    // Pagination
    if (totalPages > 1) {
        var nav = document.createElement('nav');
        nav.className = 'mt-3';
        var ul = document.createElement('ul');
        ul.className = 'pagination justify-content-center';

        var prevLi = document.createElement('li');
        prevLi.className = 'page-item' + (recallLabState.page === 0 ? ' disabled' : '');
        var prevA = document.createElement('a');
        prevA.className = 'page-link';
        prevA.href = '#';
        prevA.textContent = 'Previous';
        prevA.addEventListener('click', function(e) {
            e.preventDefault();
            if (recallLabState.page > 0) { recallLabState.page--; renderRecallPage(); }
        });
        prevLi.appendChild(prevA);
        ul.appendChild(prevLi);

        for (var p = 0; p < totalPages; p++) {
            var li = document.createElement('li');
            li.className = 'page-item' + (p === recallLabState.page ? ' active' : '');
            var a = document.createElement('a');
            a.className = 'page-link';
            a.href = '#';
            a.textContent = String(p + 1);
            (function(pageNum) {
                a.addEventListener('click', function(e) {
                    e.preventDefault();
                    recallLabState.page = pageNum;
                    renderRecallPage();
                });
            })(p);
            li.appendChild(a);
            ul.appendChild(li);
        }

        var nextLi = document.createElement('li');
        nextLi.className = 'page-item' + (recallLabState.page >= totalPages - 1 ? ' disabled' : '');
        var nextA = document.createElement('a');
        nextA.className = 'page-link';
        nextA.href = '#';
        nextA.textContent = 'Next';
        nextA.addEventListener('click', function(e) {
            e.preventDefault();
            if (recallLabState.page < totalPages - 1) { recallLabState.page++; renderRecallPage(); }
        });
        nextLi.appendChild(nextA);
        ul.appendChild(nextLi);
        nav.appendChild(ul);
        resultsDiv.appendChild(nav);

        var info = document.createElement('div');
        info.className = 'text-center text-muted small';
        info.textContent = 'Showing ' + (start + 1) + '-' + end + ' of ' + results.length + ' results';
        resultsDiv.appendChild(info);
    }
}

function appendMetaField(parent, label, value) {
    parent.appendChild(document.createTextNode(label));
    var strong = document.createElement('strong');
    strong.textContent = value;
    parent.appendChild(strong);
}

function buildChannelBar(name, score, max, color) {
    var pct = max > 0 ? Math.round((score / max) * 100) : 0;
    var row = document.createElement('div');
    row.className = 'd-flex align-items-center mb-1';
    var label = document.createElement('span');
    label.className = 'me-2';
    label.style.width = '70px';
    label.style.fontSize = '0.75rem';
    label.textContent = name;
    row.appendChild(label);
    var progressWrap = document.createElement('div');
    progressWrap.className = 'progress flex-grow-1';
    progressWrap.style.height = '14px';
    var bar = document.createElement('div');
    bar.className = 'progress-bar bg-' + color;
    bar.style.width = pct + '%';
    bar.textContent = score.toFixed(3);
    progressWrap.appendChild(bar);
    row.appendChild(progressWrap);
    return row;
}

document.getElementById('recall-lab-query')?.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') document.getElementById('recall-lab-search')?.click();
});

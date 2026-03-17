// SuperLocalMemory V2 - Memories List + Sorting
// Depends on: core.js, modal.js (openMemoryDetail)
//
// Security: All dynamic values are escaped via escapeHtml() before DOM insertion.
// The innerHTML usage in renderMemoriesTable is safe because every interpolated
// value passes through escapeHtml(). Data comes from our own local SQLite DB only.
// nosemgrep: innerHTML-xss — all dynamic values escaped via escapeHtml()

async function loadMemories() {
    var category = document.getElementById('filter-category').value;
    var project = document.getElementById('filter-project').value;
    var url = '/api/memories?limit=50';
    if (category) url += '&category=' + encodeURIComponent(category);
    if (project) url += '&project_name=' + encodeURIComponent(project);

    showLoading('memories-list', 'Loading memories...');
    try {
        var response = await fetch(url);
        var data = await response.json();
        lastSearchResults = null;
        var exportBtn = document.getElementById('export-search-btn');
        if (exportBtn) exportBtn.style.display = 'none';
        renderMemoriesTable(data.memories, false);
    } catch (error) {
        console.error('Error loading memories:', error);
        showEmpty('memories-list', 'exclamation-triangle', 'Failed to load memories');
    }
}

function renderMemoriesTable(memories, showScores) {
    var container = document.getElementById('memories-list');
    if (!memories || memories.length === 0) {
        showEmpty('memories-list', 'journal-x', 'No memories found. Try a different search or filter.');
        return;
    }

    window._slmMemories = memories;
    var scoreHeader = showScores ? '<th>Score</th>' : '';

    var rows = '';
    memories.forEach(function(mem, idx) {
        var content = mem.summary || mem.content || '';
        var contentPreview = content.length > 100 ? content.substring(0, 100) + '...' : content;
        var importance = mem.importance || 5;
        var importanceClass = importance >= 8 ? 'success' : importance >= 5 ? 'warning' : 'secondary';

        var scoreCell = '';
        if (showScores) {
            var score = mem.score || 0;
            var pct = Math.round(score * 100);
            var barColor = pct >= 70 ? '#43e97b' : pct >= 40 ? '#f9c74f' : '#f94144';
            scoreCell = '<td><span class="score-label">' + escapeHtml(String(pct)) + '%</span>'
                + '<div class="score-bar-container"><div class="score-bar">'
                + '<div class="score-bar-fill" style="width:' + pct + '%;background:' + barColor + '"></div>'
                + '</div></div></td>';
        }

        var memId = mem.memory_id || mem.id;
        var expandBtnHtml = '<button class="btn btn-sm btn-outline-secondary expand-facts-btn ms-1" data-memory-id="' + escapeHtml(String(memId)) + '" title="View atomic facts">&#9660;</button>';

        rows += '<tr data-mem-idx="' + idx + '">'
            + '<td>' + escapeHtml(String(mem.id)) + '</td>'
            + '<td><span class="badge bg-primary">' + escapeHtml(mem.category || 'None') + '</span></td>'
            + '<td><small>' + escapeHtml(mem.project_name || '-') + '</small></td>'
            + '<td class="memory-content" title="' + escapeHtml(content) + '">' + escapeHtml(contentPreview) + expandBtnHtml + '</td>'
            + scoreCell
            + '<td><span class="badge bg-' + importanceClass + ' badge-importance">' + escapeHtml(String(importance)) + '</span></td>'
            + '<td>' + escapeHtml(String(mem.cluster_id || '-')) + '</td>'
            + '<td><small>' + escapeHtml(formatDate(mem.created_at)) + '</small></td>'
            + '</tr>';
    });

    var tableHtml = '<table class="table table-hover memory-table"><thead><tr>'
        + '<th class="sortable" data-sort="id">ID</th>'
        + '<th class="sortable" data-sort="category">Category</th>'
        + '<th class="sortable" data-sort="project">Project</th>'
        + '<th>Content</th>'
        + scoreHeader
        + '<th class="sortable" data-sort="importance">Importance</th>'
        + '<th>Cluster</th>'
        + '<th class="sortable" data-sort="created">Created</th>'
        + '</tr></thead><tbody>' + rows + '</tbody></table>';

    // All values above escaped via escapeHtml() — safe for trusted local data
    container.textContent = '';
    container.insertAdjacentHTML('beforeend', tableHtml);

    var table = container.querySelector('table');
    if (table) {
        table.addEventListener('click', function(e) {
            var th = e.target.closest('th.sortable');
            if (th) { handleSort(th); return; }

            // Expand facts button
            var expandBtn = e.target.closest('.expand-facts-btn');
            if (expandBtn) {
                e.stopPropagation();
                var memId = expandBtn.getAttribute('data-memory-id');
                toggleFactsExpansion(expandBtn, memId);
                return;
            }

            var row = e.target.closest('tr[data-mem-idx]');
            if (row && !e.target.closest('.expand-facts-btn')) {
                var idx = parseInt(row.getAttribute('data-mem-idx'), 10);
                if (window._slmMemories && window._slmMemories[idx]) {
                    openMemoryDetail(window._slmMemories[idx]);
                }
            }
        });
    }
}

// ============================================================================
// Column Sorting
// ============================================================================

var currentSort = { column: null, direction: 'asc' };

function handleSort(th) {
    var col = th.getAttribute('data-sort');
    if (!col) return;

    if (currentSort.column === col) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = col;
        currentSort.direction = 'asc';
    }

    document.querySelectorAll('#memories-list th.sortable').forEach(function(h) {
        h.classList.remove('sort-asc', 'sort-desc');
    });
    th.classList.add('sort-' + currentSort.direction);

    if (!window._slmMemories) return;
    var memories = window._slmMemories.slice();
    var dir = currentSort.direction === 'asc' ? 1 : -1;

    memories.sort(function(a, b) {
        var av, bv;
        switch (col) {
            case 'id': return ((a.id || 0) - (b.id || 0)) * dir;
            case 'importance': return ((a.importance || 0) - (b.importance || 0)) * dir;
            case 'category':
                av = (a.category || '').toLowerCase(); bv = (b.category || '').toLowerCase();
                return av < bv ? -dir : av > bv ? dir : 0;
            case 'project':
                av = (a.project_name || '').toLowerCase(); bv = (b.project_name || '').toLowerCase();
                return av < bv ? -dir : av > bv ? dir : 0;
            case 'created':
                av = a.created_at || ''; bv = b.created_at || '';
                return av < bv ? -dir : av > bv ? dir : 0;
            case 'score': return ((a.score || 0) - (b.score || 0)) * dir;
            default: return 0;
        }
    });

    window._slmMemories = memories;
    var showScores = memories.length > 0 && typeof memories[0].score === 'number';
    renderMemoriesTable(memories, showScores);
}

// ============================================================================
// Navigation from Graph (v2.6.5)
// ============================================================================

// Scroll to a specific memory by ID (called from graph double-click)
function scrollToMemory(memoryId) {
    const container = document.getElementById('memories-list');
    if (!container) return;

    // Find the memory in current list
    if (!window._slmMemories) {
        console.warn('No memories loaded yet. Loading all memories...');
        loadMemories().then(() => {
            setTimeout(() => scrollToMemoryInTable(memoryId), 500);
        });
        return;
    }

    scrollToMemoryInTable(memoryId);
}

async function toggleFactsExpansion(btn, memoryId) {
    var row = btn.closest('tr');
    if (!row) return;
    var existingExpansion = row.nextElementSibling;
    if (existingExpansion && existingExpansion.classList.contains('facts-expansion-row')) {
        existingExpansion.remove();
        btn.innerHTML = '&#9660;';
        return;
    }

    btn.innerHTML = '&#8987;';
    try {
        var resp = await fetch('/api/memories/' + encodeURIComponent(memoryId) + '/facts');
        var data = await resp.json();
        var expRow = document.createElement('tr');
        expRow.className = 'facts-expansion-row';
        var expCell = document.createElement('td');
        expCell.colSpan = 8;
        expCell.style.cssText = 'background:#f8f9fa; padding:8px 16px;';

        if (data.facts && data.facts.length > 0) {
            var label = document.createElement('small');
            label.className = 'text-muted';
            label.textContent = data.fact_count + ' atomic facts:';
            expCell.appendChild(label);

            data.facts.forEach(function(f, i) {
                var fDiv = document.createElement('div');
                fDiv.className = 'small py-1' + (i < data.facts.length - 1 ? ' border-bottom' : '');
                var badge = document.createElement('span');
                badge.className = 'badge bg-secondary me-1';
                badge.style.fontSize = '0.65rem';
                badge.textContent = f.fact_type;
                fDiv.appendChild(badge);
                fDiv.appendChild(document.createTextNode(f.content.substring(0, 200)));
                expCell.appendChild(fDiv);
            });
        } else {
            expCell.textContent = 'No atomic facts found for this memory.';
        }

        expRow.appendChild(expCell);
        row.parentNode.insertBefore(expRow, row.nextSibling);
        btn.innerHTML = '&#9650;';
    } catch (err) {
        btn.innerHTML = '&#9660;';
        console.error('Failed to load facts:', err);
    }
}

function scrollToMemoryInTable(memoryId) {
    const memId = String(memoryId);

    // Find the memory index
    let idx = -1;
    if (window._slmMemories) {
        idx = window._slmMemories.findIndex(m => String(m.id) === memId);
    }

    if (idx === -1) {
        console.warn(`Memory #${memoryId} not found in current list`);
        return;
    }

    // Find the table row
    const row = document.querySelector(`tr[data-mem-idx="${idx}"]`);
    if (!row) {
        console.warn(`Row for memory #${memoryId} not found in DOM`);
        return;
    }

    // Scroll to row
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Highlight temporarily
    row.style.backgroundColor = '#fff3cd';
    setTimeout(() => {
        row.style.backgroundColor = '';
    }, 2000);
}

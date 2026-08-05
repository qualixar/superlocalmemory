// SuperLocalMemory V2 - Memory Detail Modal + Copy/Export
// Depends on: core.js
//
// Security: All dynamic values escaped via escapeHtml(). Data from local DB only.
// nosemgrep: innerHTML-xss — all dynamic values escaped

var currentMemoryDetail = null;

function openMemoryDetail(mem, source) {
    // source: 'graph', 'recall', 'memories', or undefined
    currentMemoryDetail = mem;
    var fromGraph = source === 'graph';
    var fromRecall = source === 'recall';
    var body = document.getElementById('memory-detail-body');
    if (!mem) {
        body.textContent = 'No memory data';
        return;
    }

    // Store last focused element (for keyboard nav return)
    if (!window.lastFocusedElement) {
        window.lastFocusedElement = document.activeElement;
    }

    var content = mem.content || mem.summary || '(no content)';
    var tags = mem.tags || '';
    var importance = mem.importance || 5;
    var importanceClass = importance >= 8 ? 'success' : importance >= 5 ? 'warning' : 'secondary';

    // Build detail using DOM nodes for safety
    body.textContent = '';

    var contentDiv = document.createElement('div');
    contentDiv.className = 'memory-detail-content';
    contentDiv.textContent = content;
    body.appendChild(contentDiv);

    body.appendChild(document.createElement('hr'));

    var dl = document.createElement('dl');
    dl.className = 'memory-detail-meta row';

    // v3.4.31: disambiguate Fact ID (the atomic unit) from Memory ID (the parent).
    var factId = mem.fact_id || mem.id || '';
    var memoryId = mem.memory_id || mem.id || '';

    // Left column
    var col1 = document.createElement('div');
    col1.className = 'col-md-6';
    addDetailRow(col1, 'Memory ID', String(memoryId || '-'));
    if (factId && factId !== memoryId) {
        addDetailRow(col1, 'Fact ID', String(factId));
    }
    addDetailBadgeRow(col1, 'Category', mem.category || mem.fact_type || 'None', 'bg-primary');
    addDetailRow(col1, 'Project', mem.project_name || '-');
    addDetailTagsRow(col1, 'Tags', tags);
    dl.appendChild(col1);

    // Right column
    var col2 = document.createElement('div');
    col2.className = 'col-md-6';
    addDetailBadgeRow(col2, 'Importance', importance + '/10', 'bg-' + importanceClass);
    addDetailRow(col2, 'Cluster', String(mem.cluster_id || '-'));
    addDetailRow(col2, 'Created', formatDateFull(mem.created_at));
    if (mem.updated_at) addDetailRow(col2, 'Updated', formatDateFull(mem.updated_at));

    if (typeof mem.score === 'number') {
        var pct = Math.round(mem.score * 100);
        addDetailRow(col2, 'Relevance Score', pct + '%');
    }
    dl.appendChild(col2);

    body.appendChild(dl);

    // v3.4.31: hydrate with full memory + fact list from /api/memories/{id}/detail
    if (memoryId) {
        fetch('/api/memories/' + encodeURIComponent(memoryId) + '/detail')
            .then(function(r) { return r.ok ? r.json() : null; })
            .then(function(data) {
                if (!data || !data.memory) return;
                var hydration = document.getElementById('memory-detail-hydration');
                if (hydration) hydration.remove();
                var block = document.createElement('div');
                block.id = 'memory-detail-hydration';
                block.className = 'mt-3';
                var h = document.createElement('h6');
                h.innerHTML = '<i class="bi bi-diagram-3"></i> Atomic facts extracted from this memory (' + (data.fact_count || 0) + ')';
                block.appendChild(h);
                var list = document.createElement('div');
                list.className = 'list-group list-group-flush';
                (data.facts || []).forEach(function(f) {
                    var row = document.createElement('div');
                    row.className = 'list-group-item list-group-item-action small fact-result-item';
                    row.setAttribute('data-fact-id', f.fact_id);
                    row.style.cursor = 'pointer';
                    var badge = '<span class="badge bg-secondary me-2">' + (f.fact_type || '-') + '</span>';
                    var confText = ' · confidence ' + (f.confidence || 0).toFixed(2);
                    row.innerHTML = badge + escapeHtml(String(f.content || '')) + '<small class="text-muted">' + escapeHtml(confText) + '</small>';
                    list.appendChild(row);
                });
                block.appendChild(list);
                body.appendChild(block);
            })
            .catch(function(err) { console.warn('Hydration error:', err); });
    }

    // Context-aware action buttons
    if (mem.id) {
        body.appendChild(document.createElement('hr'));

        var actionsDiv = document.createElement('div');
        actionsDiv.className = 'memory-detail-graph-actions';
        actionsDiv.style.cssText = 'display:flex; gap:10px; flex-wrap:wrap;';

        // "View Original Memory" — shown on Recall Lab + Memories, hidden on Graph
        // (On Graph the node IS the memory; on Recall Lab we have a fact, not the original)
        if (!fromGraph) {
            var viewBtn = document.createElement('button');
            viewBtn.className = 'btn btn-primary btn-sm';
            viewBtn.innerHTML = '<i class="bi bi-journal-text"></i> View Original Memory';
            viewBtn.onclick = function() {
                var mid = mem.memory_id || mem.id;
                viewBtn.disabled = true;
                viewBtn.textContent = 'Loading...';
                fetch('/api/memories/' + encodeURIComponent(mid) + '/facts')
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.ok && data.original_content) {
                            contentDiv.textContent = '';
                            var origLabel = document.createElement('small');
                            origLabel.className = 'text-muted d-block mb-1';
                            origLabel.textContent = 'Original memory (' + (data.fact_count || 0) + ' atomic facts extracted):';
                            contentDiv.appendChild(origLabel);
                            var origText = document.createElement('div');
                            origText.style.cssText = 'white-space:pre-wrap;background:#f8f9fa;padding:10px;border-radius:6px;margin-bottom:8px;';
                            origText.textContent = data.original_content;
                            contentDiv.appendChild(origText);
                            if (data.facts && data.facts.length > 0) {
                                var toggle = document.createElement('button');
                                toggle.className = 'btn btn-sm btn-outline-secondary mb-2';
                                toggle.textContent = 'Show atomic facts (' + data.facts.length + ')';
                                var factsDiv = document.createElement('div');
                                factsDiv.style.display = 'none';
                                data.facts.forEach(function(f) {
                                    var fDiv = document.createElement('div');
                                    fDiv.className = 'small py-1 border-bottom';
                                    var badge = document.createElement('span');
                                    badge.className = 'badge bg-secondary me-1';
                                    badge.style.fontSize = '0.6rem';
                                    badge.textContent = f.fact_type;
                                    fDiv.appendChild(badge);
                                    fDiv.appendChild(document.createTextNode(f.content));
                                    factsDiv.appendChild(fDiv);
                                });
                                toggle.onclick = function() {
                                    var hidden = factsDiv.style.display === 'none';
                                    factsDiv.style.display = hidden ? 'block' : 'none';
                                    toggle.textContent = hidden ? 'Hide atomic facts' : 'Show atomic facts (' + data.facts.length + ')';
                                };
                                contentDiv.appendChild(toggle);
                                contentDiv.appendChild(factsDiv);
                            }
                            viewBtn.textContent = 'Showing original';
                        } else {
                            viewBtn.textContent = 'Not available';
                        }
                    }).catch(function() {
                        viewBtn.textContent = 'Failed to load';
                        viewBtn.disabled = false;
                    });
            };
            actionsDiv.appendChild(viewBtn);
        }

        // "Expand Neighbors" — shown on Graph, hidden elsewhere (no graph context)
        if (fromGraph) {
            var expandBtn = document.createElement('button');
            expandBtn.className = 'btn btn-outline-secondary btn-sm';
            expandBtn.innerHTML = '<i class="bi bi-diagram-3"></i> Expand Neighbors';
            expandBtn.onclick = function() {
                modal.hide();
                setTimeout(function() {
                    if (typeof expandNeighbors === 'function') expandNeighbors(mem.id);
                }, 300);
            };
            actionsDiv.appendChild(expandBtn);
        }

        // "Filter to Cluster" — always available if cluster exists
        if (mem.cluster_id) {
            var filterBtn = document.createElement('button');
            filterBtn.className = 'btn btn-outline-info btn-sm';
            var filterIcon = document.createElement('i');
            filterIcon.className = 'bi bi-funnel';
            filterBtn.appendChild(filterIcon);
            filterBtn.appendChild(document.createTextNode(' Filter to Cluster ' + mem.cluster_id));
            filterBtn.onclick = function() {
                modal.hide();
                // Switch to Graph tab
                const graphTab = document.querySelector('a[href="#graph"]');
                if (graphTab) graphTab.click();
                // Apply cluster filter after a delay
                setTimeout(function() {
                    if (typeof filterState !== 'undefined' && typeof filterByCluster === 'function' && typeof renderGraph === 'function') {
                        filterState.cluster_id = mem.cluster_id;
                        const filtered = filterByCluster(originalGraphData, mem.cluster_id);
                        renderGraph(filtered);
                        // Update URL
                        const url = new URL(window.location);
                        url.searchParams.set('cluster_id', mem.cluster_id);
                        window.history.replaceState({}, '', url);
                    }
                }, 500);
            };
            actionsDiv.appendChild(filterBtn);
        }

        // Edit button — always available
        var editBtn = document.createElement('button');
        editBtn.className = 'btn btn-outline-warning btn-sm';
        editBtn.innerHTML = '<i class="bi bi-pencil"></i> Edit';
        editBtn.onclick = function() {
            var currentText = contentDiv.textContent;
            var textarea = document.createElement('textarea');
            textarea.className = 'form-control mb-2';
            textarea.rows = 4;
            textarea.value = currentText;
            contentDiv.textContent = '';
            contentDiv.appendChild(textarea);
            var saveBtn = document.createElement('button');
            saveBtn.className = 'btn btn-sm btn-success me-1';
            saveBtn.textContent = 'Save';
            saveBtn.onclick = function() {
                var newContent = textarea.value.trim();
                if (!newContent) return;
                fetch('/api/memories/' + encodeURIComponent(mem.id), {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({content: newContent})
                }).then(function(r) { return r.json(); }).then(function(d) {
                    if (d.success) {
                        contentDiv.textContent = newContent;
                        mem.content = newContent;
                        if (typeof showToast === 'function') showToast('Memory updated');
                    }
                });
            };
            var cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-sm btn-secondary';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.onclick = function() { contentDiv.textContent = currentText; };
            contentDiv.appendChild(saveBtn);
            contentDiv.appendChild(cancelBtn);
        };
        actionsDiv.appendChild(editBtn);

        // S9-DASH-08: Forget (soft-archive) — non-destructive; row stays
        // in atomic_facts with archive_status='archived' and gets a
        // payload copy in memory_archive for future restore.
        var forgetBtn = document.createElement('button');
        forgetBtn.className = 'btn btn-outline-warning btn-sm';
        forgetBtn.innerHTML = '<i class="bi bi-archive"></i> Forget';
        forgetBtn.title = 'Archive this memory — hidden from recall but recoverable';
        forgetBtn.onclick = async function() {
            var confirmed = await confirmDestructive({
                title: 'Forget memory',
                target: mem.content ? mem.content.slice(0, 80) : 'Memory #' + mem.id,
                consequence: 'Archived — hidden from recall but recoverable.',
            });
            if (!confirmed) return;
            forgetBtn.disabled = true;
            fetch('/api/memories/' + encodeURIComponent(mem.id) + '/forget',
                {method: 'POST'})
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    if (d.success) {
                        modal.hide();
                        if (typeof showToast === 'function') {
                            showToast('Memory archived');
                        }
                        if (typeof loadMemories === 'function') {
                            setTimeout(loadMemories, 300);
                        }
                    } else {
                        forgetBtn.disabled = false;
                    }
                }).catch(function() { forgetBtn.disabled = false; });
        };
        actionsDiv.appendChild(forgetBtn);

        // S9-DASH-08: Merge — this fact is a duplicate of another;
        // keep the target, archive this one. Writes memory_merge_log.
        var mergeBtn = document.createElement('button');
        mergeBtn.className = 'btn btn-outline-info btn-sm';
        mergeBtn.innerHTML = '<i class="bi bi-union"></i> Merge into...';
        mergeBtn.title = 'Mark this as a duplicate of another fact';
        mergeBtn.onclick = function() {
            var target = prompt(
                'Merge this memory INTO which fact_id?\n\n'
                + '(Paste the target fact_id — this memory will be '
                + 'archived and audit-logged to memory_merge_log.)'
            );
            if (!target || !target.trim()) return;
            mergeBtn.disabled = true;
            fetch('/api/memories/' + encodeURIComponent(mem.id) + '/merge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({into: target.trim()}),
            }).then(function(r) { return r.json(); })
            .then(function(d) {
                if (d.success) {
                    modal.hide();
                    if (typeof showToast === 'function') {
                        showToast('Merged into ' + d.into);
                    }
                    if (typeof loadMemories === 'function') {
                        setTimeout(loadMemories, 300);
                    }
                } else {
                    mergeBtn.disabled = false;
                    if (typeof showToast === 'function') {
                        showToast('Merge failed: '
                            + (d && (d.error || d.detail) || 'unknown'));
                    }
                }
            }).catch(function() { mergeBtn.disabled = false; });
        };
        actionsDiv.appendChild(mergeBtn);

        // Task F: Set Scope — PATCH /api/memories/{fact_id}/scope
        // Lets the user change personal → shared → global visibility.
        var scopeBtn = document.createElement('button');
        scopeBtn.className = 'btn btn-outline-secondary btn-sm';
        scopeBtn.innerHTML = '<i class="bi bi-globe"></i> Set Scope…';
        scopeBtn.title = 'Change memory visibility: personal, shared, or global';
        (function() {
            var scopeFormEl = null;
            scopeBtn.addEventListener('click', function() {
                // Toggle the inline scope form
                if (scopeFormEl) {
                    scopeFormEl.remove();
                    scopeFormEl = null;
                    return;
                }
                scopeFormEl = document.createElement('div');
                scopeFormEl.className = 'mt-2 p-2 border rounded';
                scopeFormEl.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center;width:100%';
                var scopeSel = document.createElement('select');
                scopeSel.className = 'form-select form-select-sm';
                scopeSel.style.width = 'auto';
                ['personal', 'shared', 'global'].forEach(function(s) {
                    var opt = document.createElement('option');
                    opt.value = s;
                    opt.textContent = s;
                    if (s === (mem.scope || 'personal')) opt.selected = true;
                    scopeSel.appendChild(opt);
                });
                var sharedInput = document.createElement('input');
                sharedInput.className = 'form-control form-control-sm';
                sharedInput.placeholder = 'shared_with (profile1,profile2)';
                sharedInput.style.flex = '1';
                sharedInput.style.display = scopeSel.value === 'shared' ? '' : 'none';
                if (mem.shared_with) {
                    try {
                        var sw = typeof mem.shared_with === 'string'
                            ? JSON.parse(mem.shared_with) : mem.shared_with;
                        sharedInput.value = Array.isArray(sw) ? sw.join(',') : String(sw);
                    } catch(e) { sharedInput.value = String(mem.shared_with || ''); }
                }
                scopeSel.addEventListener('change', function() {
                    sharedInput.style.display = scopeSel.value === 'shared' ? '' : 'none';
                });
                var saveBtn = document.createElement('button');
                saveBtn.className = 'btn btn-sm btn-primary';
                saveBtn.textContent = 'Save';
                saveBtn.addEventListener('click', function() {
                    var scope = scopeSel.value;
                    var sharedWith = scope === 'shared' ? sharedInput.value : '';
                    saveBtn.disabled = true;
                    saveBtn.textContent = 'Saving…';
                    fetch('/api/memories/' + encodeURIComponent(mem.fact_id || mem.id) + '/scope', {
                        method: 'PATCH',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({scope: scope, shared_with: sharedWith}),
                    }).then(function(r) { return r.json(); })
                    .then(function(d) {
                        if (d.success) {
                            mem.scope = scope;
                            if (typeof showToast === 'function') showToast('Scope set to ' + scope);
                            scopeFormEl.remove(); scopeFormEl = null;
                            if (typeof loadMemories === 'function') setTimeout(loadMemories, 300);
                        } else {
                            if (typeof showToast === 'function') showToast('Scope update failed: ' + (d.detail || d.error || 'unknown'));
                            saveBtn.disabled = false; saveBtn.textContent = 'Save';
                        }
                    }).catch(function() {
                        if (typeof showToast === 'function') showToast('Network error setting scope.');
                        saveBtn.disabled = false; saveBtn.textContent = 'Save';
                    });
                });
                scopeFormEl.appendChild(scopeSel);
                scopeFormEl.appendChild(sharedInput);
                scopeFormEl.appendChild(saveBtn);
                actionsDiv.insertAdjacentElement('afterend', scopeFormEl);
            });
        }());
        actionsDiv.appendChild(scopeBtn);

        // Task G: Pin — POST /api/tiers/pin — keep this fact in the active tier forever
        var factIdForTier = mem.fact_id || mem.id;
        var pinBtn = document.createElement('button');
        pinBtn.className = 'btn btn-outline-success btn-sm';
        pinBtn.innerHTML = '<i class="bi bi-pin-fill"></i> Pin';
        pinBtn.title = 'Pin to active tier — this fact will not be demoted by lifecycle';
        pinBtn.addEventListener('click', function() {
            pinBtn.disabled = true;
            fetch('/api/tiers/pin', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({fact_id: factIdForTier, reason: 'pinned from dashboard'}),
            }).then(function(r) { return r.json(); })
            .then(function(d) {
                pinBtn.disabled = false;
                if (typeof showToast === 'function') {
                    showToast(d && d.success ? 'Fact pinned to active tier.' : 'Pin failed: ' + (d && (d.detail || d.error) || 'unknown'));
                }
            }).catch(function() {
                pinBtn.disabled = false;
                if (typeof showToast === 'function') showToast('Network error pinning fact.');
            });
        });
        actionsDiv.appendChild(pinBtn);

        // Task G: Unpin — POST /api/tiers/unpin — allows normal tier demotion again
        var unpinBtn = document.createElement('button');
        unpinBtn.className = 'btn btn-outline-warning btn-sm';
        unpinBtn.innerHTML = '<i class="bi bi-pin-angle"></i> Unpin';
        unpinBtn.title = 'Unpin — allow normal lifecycle tier demotion';
        unpinBtn.addEventListener('click', function() {
            unpinBtn.disabled = true;
            fetch('/api/tiers/unpin', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({fact_id: factIdForTier, reason: ''}),
            }).then(function(r) { return r.json(); })
            .then(function(d) {
                unpinBtn.disabled = false;
                if (typeof showToast === 'function') {
                    showToast(d && d.success ? 'Fact unpinned — will age normally.' : 'Unpin failed: ' + (d && (d.detail || d.error) || 'unknown'));
                }
            }).catch(function() {
                unpinBtn.disabled = false;
                if (typeof showToast === 'function') showToast('Network error unpinning fact.');
            });
        });
        actionsDiv.appendChild(unpinBtn);

        // Delete button — always available (hard delete, irreversible)
        var deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-outline-danger btn-sm';
        deleteBtn.innerHTML = '<i class="bi bi-trash"></i> Delete';
        deleteBtn.title = 'Permanently delete (cannot be undone) — prefer Forget';
        deleteBtn.onclick = async function() {
            var confirmed = await confirmDestructive({
                title: 'Delete memory',
                target: mem.content ? mem.content.slice(0, 80) : 'Memory #' + mem.id,
                consequence: 'Permanently deleted — this cannot be undone.',
                confirmLabel: 'Delete',
            });
            if (!confirmed) return;
            fetch('/api/memories/' + encodeURIComponent(mem.id), {method: 'DELETE'})
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    if (d.success) {
                        modal.hide();
                        if (typeof showToast === 'function') showToast('Memory deleted');
                        if (typeof loadMemories === 'function') setTimeout(loadMemories, 300);
                    }
                });
        };
        actionsDiv.appendChild(deleteBtn);

        body.appendChild(actionsDiv);
    }

    // v2.7.4: Add feedback buttons to modal body
    if (typeof createFeedbackButtons === 'function' && mem && mem.id) {
        var feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'mt-3 pt-2 border-top';
        var feedbackLabel = document.createElement('small');
        feedbackLabel.className = 'text-muted d-block mb-1';
        feedbackLabel.textContent = 'Was this memory useful?';
        feedbackDiv.appendChild(feedbackLabel);
        feedbackDiv.appendChild(createFeedbackButtons(mem.id));
        body.appendChild(feedbackDiv);
    }

    var modalEl = document.getElementById('memoryDetailModal');
    var modal = new bootstrap.Modal(modalEl);

    // v2.7.4: Start dwell time tracking
    if (typeof startDwellTracking === 'function' && mem && mem.id) {
        startDwellTracking(mem.id);
    }

    // Focus first interactive element when modal opens
    modalEl.addEventListener('shown.bs.modal', function() {
        const firstButton = modalEl.querySelector('button, a[href]');
        if (firstButton) {
            firstButton.focus();
        }
    }, { once: true });

    // Return focus when modal closes + stop dwell tracking
    modalEl.addEventListener('hidden.bs.modal', function() {
        // v2.7.4: Stop dwell time tracking
        if (typeof stopDwellTracking === 'function') {
            stopDwellTracking();
        }
        if (window.lastFocusedElement && typeof window.lastFocusedElement.focus === 'function') {
            window.lastFocusedElement.focus();
            window.lastFocusedElement = null;
        }
    }, { once: true });

    modal.show();
}

function addDetailRow(parent, label, value) {
    var dt = document.createElement('dt');
    dt.textContent = label;
    parent.appendChild(dt);
    var dd = document.createElement('dd');
    dd.textContent = value;
    parent.appendChild(dd);
}

function addDetailBadgeRow(parent, label, value, badgeClass) {
    var dt = document.createElement('dt');
    dt.textContent = label;
    parent.appendChild(dt);
    var dd = document.createElement('dd');
    var badge = document.createElement('span');
    badge.className = 'badge ' + badgeClass;
    badge.textContent = value;
    dd.appendChild(badge);
    parent.appendChild(dd);
}

function addDetailTagsRow(parent, label, tags) {
    var dt = document.createElement('dt');
    dt.textContent = label;
    parent.appendChild(dt);
    var dd = document.createElement('dd');
    var tagList = typeof tags === 'string' ? tags.split(',') : (tags || []);
    if (tagList.length === 0 || (tagList.length === 1 && !tagList[0].trim())) {
        dd.className = 'text-muted';
        dd.textContent = 'None';
    } else {
        tagList.forEach(function(t) {
            var tag = t.trim();
            if (tag) {
                var badge = document.createElement('span');
                badge.className = 'badge bg-secondary me-1';
                badge.textContent = tag;
                dd.appendChild(badge);
            }
        });
    }
    parent.appendChild(dd);
}

/**
 * Show a shared confirmation modal for destructive dashboard actions.
 * Creates the modal element on first call; reuses it on subsequent calls.
 *
 * @param {object} opts
 * @param {string} opts.title          Short header, e.g. "Delete profile"
 * @param {string} opts.target         Exact item being acted on, e.g. "my-project"
 * @param {string} opts.consequence    What happens, e.g. "Memories moved to default profile"
 * @param {string} [opts.confirmLabel] Confirm button text (default: "Confirm")
 * @param {string} [opts.confirmationText] Exact text required to unlock confirmation
 * @returns {Promise<boolean>} Resolves true when confirmed, false when cancelled
 */
var activeDestructiveConfirmation = null;

function confirmDestructive(opts) {
    return new Promise(function(resolve) {
        if (activeDestructiveConfirmation) {
            activeDestructiveConfirmation.cancel();
        }
        var settled = false;
        var MODAL_ID = 'slm-confirm-destructive-modal';
        var modalEl = document.getElementById(MODAL_ID);

        if (!modalEl) {
            modalEl = document.createElement('div');
            modalEl.id = MODAL_ID;
            modalEl.className = 'modal fade';
            modalEl.setAttribute('tabindex', '-1');
            modalEl.setAttribute('aria-modal', 'true');
            modalEl.setAttribute('role', 'dialog');
            modalEl.innerHTML =
                '<div class="modal-dialog modal-dialog-centered">' +
                '<div class="modal-content">' +
                '<div class="modal-header border-0 pb-1">' +
                '<h5 class="modal-title slm-cd-title text-danger"></h5>' +
                '<button type="button" class="btn-close"' +
                ' data-slm-cd-action="cancel" aria-label="Close"></button>' +
                '</div>' +
                '<div class="modal-body pt-1">' +
                '<p class="slm-cd-target fw-semibold mb-1"></p>' +
                '<p class="slm-cd-consequence text-muted small mb-2"></p>' +
                '<label class="form-label small mb-1" for="slm-cd-challenge">' +
                'Type <code class="slm-cd-confirmation-text"></code> to continue</label>' +
                '<input id="slm-cd-challenge" class="form-control form-control-sm slm-cd-challenge"' +
                ' type="text" autocomplete="off" spellcheck="false">' +
                '</div>' +
                '<div class="modal-footer border-0 pt-0">' +
                '<button type="button" class="btn btn-secondary btn-sm"' +
                ' data-slm-cd-action="cancel">Cancel</button>' +
                '<button type="button" class="btn btn-danger btn-sm"' +
                ' data-slm-cd-action="confirm">Confirm</button>' +
                '</div>' +
                '</div>' +
                '</div>';
            document.body.appendChild(modalEl);
        }

        var titleEl = modalEl.querySelector('.slm-cd-title');
        var targetEl = modalEl.querySelector('.slm-cd-target');
        var consequenceEl = modalEl.querySelector('.slm-cd-consequence');
        var confirmationTextEl = modalEl.querySelector('.slm-cd-confirmation-text');
        var challengeInput = modalEl.querySelector('.slm-cd-challenge');
        var confirmBtn = modalEl.querySelector('[data-slm-cd-action="confirm"]');
        var confirmationText = opts.confirmationText || opts.target || 'CONFIRM';

        if (titleEl) titleEl.textContent = opts.title || 'Confirm action';
        if (targetEl) targetEl.textContent = opts.target || '';
        if (consequenceEl) consequenceEl.textContent = opts.consequence || '';
        if (confirmationTextEl) confirmationTextEl.textContent = confirmationText;
        if (challengeInput) challengeInput.value = '';
        if (confirmBtn) {
            confirmBtn.textContent = opts.confirmLabel || 'Confirm';
            confirmBtn.disabled = true;
        }

        var bsModal = null;
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            bsModal = bootstrap.Modal.getOrCreateInstance(modalEl);
        }

        function settle(result, hideModal) {
            if (settled) return;
            settled = true;
            if (activeDestructiveConfirmation === confirmationSession) {
                activeDestructiveConfirmation = null;
            }
            modalEl.removeEventListener('click', onAction);
            if (challengeInput) challengeInput.removeEventListener('input', onChallengeInput);
            if (bsModal) {
                modalEl.removeEventListener('hidden.bs.modal', onHide);
                modalEl.removeEventListener('shown.bs.modal', onShown);
                if (hideModal !== false) bsModal.hide();
            }
            resolve(result);
        }

        function onAction(e) {
            var actionEl = e.target.closest('[data-slm-cd-action]');
            if (!actionEl) return;
            if (actionEl.getAttribute('data-slm-cd-action') === 'confirm' &&
                    (!challengeInput || challengeInput.value !== confirmationText)) return;
            settle(actionEl.getAttribute('data-slm-cd-action') === 'confirm');
        }

        function onChallengeInput() {
            if (confirmBtn) confirmBtn.disabled = challengeInput.value !== confirmationText;
        }

        function onShown() {
            if (challengeInput) challengeInput.focus();
        }

        function onHide() { settle(false); }

        var confirmationSession = {
            cancel: function() { settle(false, false); }
        };
        activeDestructiveConfirmation = confirmationSession;

        modalEl.addEventListener('click', onAction);
        if (challengeInput) challengeInput.addEventListener('input', onChallengeInput);
        if (bsModal) {
            modalEl.addEventListener('hidden.bs.modal', onHide, { once: true });
            modalEl.addEventListener('shown.bs.modal', onShown, { once: true });
            bsModal.show();
        }
    });
}

function copyMemoryToClipboard() {
    if (!currentMemoryDetail) return;
    var text = currentMemoryDetail.content || currentMemoryDetail.summary || '';
    navigator.clipboard.writeText(text).then(function() {
        showToast('Copied to clipboard');
    }).catch(function() {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('Copied to clipboard');
    });
}

function exportMemoryAsMarkdown() {
    if (!currentMemoryDetail) return;
    var mem = currentMemoryDetail;
    var md = '# Memory #' + (mem.id || 'unknown') + '\n\n';
    md += '**Category:** ' + (mem.category || 'None') + '  \n';
    md += '**Project:** ' + (mem.project_name || '-') + '  \n';
    md += '**Importance:** ' + (mem.importance || 5) + '/10  \n';
    md += '**Tags:** ' + (mem.tags || 'None') + '  \n';
    md += '**Created:** ' + (mem.created_at || '-') + '  \n';
    if (mem.cluster_id) md += '**Cluster:** ' + mem.cluster_id + '  \n';
    md += '\n---\n\n';
    md += mem.content || mem.summary || '(no content)';
    md += '\n\n---\n*Exported from SuperLocalMemory V2*\n';

    downloadFile('memory-' + (mem.id || 'export') + '.md', md, 'text/markdown');
}

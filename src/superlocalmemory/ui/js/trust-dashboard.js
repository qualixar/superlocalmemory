// SuperLocalMemory V3 — Trust Dashboard
// Loads and displays Bayesian trust scores per agent and per fact.
//
// v3.4.21 (Operations pane): /api/v3/trust/dashboard returns
// thousands of rows in one shot (3,900+ agents on Varun's live DB).
// Client-side pagination keeps the table bounded; the user chooses
// sort order + page size; all data stays real (no mock fallback).

(function trustDashboard() {
    'use strict';

    var STATE = {
        agents: [],
        page: 0,
        pageSize: 25,
        sort: 'score-desc',
    };

    function _sortedAgents() {
        var arr = STATE.agents.slice();
        switch (STATE.sort) {
            case 'score-asc':
                arr.sort(function (a, b) {
                    return (a.trust_score || 0) - (b.trust_score || 0);
                });
                break;
            case 'evidence-desc':
                arr.sort(function (a, b) {
                    return (b.evidence_count || 0) - (a.evidence_count || 0);
                });
                break;
            case 'updated-desc':
                arr.sort(function (a, b) {
                    return String(b.last_updated || '').localeCompare(
                        String(a.last_updated || '')
                    );
                });
                break;
            case 'score-desc':
            default:
                arr.sort(function (a, b) {
                    return (b.trust_score || 0) - (a.trust_score || 0);
                });
                break;
        }
        return arr;
    }

    function _renderPage() {
        var tbody = document.getElementById('trust-agents-body');
        if (!tbody) return;

        var agents = _sortedAgents();
        var total = agents.length;
        var pageSize = Math.max(1, STATE.pageSize);
        var maxPage = Math.max(0, Math.ceil(total / pageSize) - 1);
        if (STATE.page > maxPage) STATE.page = maxPage;
        if (STATE.page < 0) STATE.page = 0;
        var start = STATE.page * pageSize;
        var end = Math.min(total, start + pageSize);
        var slice = agents.slice(start, end);

        tbody.textContent = '';
        slice.forEach(function (a) {
            var tr = document.createElement('tr');
            var score = a.trust_score || 0;
            var badge = score >= 0.7 ? 'success'
                : score >= 0.3 ? 'warning' : 'danger';
            var label = score >= 0.7 ? 'Trusted'
                : score >= 0.3 ? 'Neutral' : 'Low trust';

            var tdTarget = document.createElement('td');
            tdTarget.textContent = a.target_id || '';
            tr.appendChild(tdTarget);

            var tdType = document.createElement('td');
            var spanType = document.createElement('span');
            spanType.className = 'badge bg-secondary';
            spanType.textContent = a.target_type || '';
            tdType.appendChild(spanType);
            tr.appendChild(tdType);

            var tdScore = document.createElement('td');
            var progress = document.createElement('div');
            progress.className = 'progress';
            progress.style.height = '20px';
            var bar = document.createElement('div');
            bar.className = 'progress-bar bg-' + badge;
            bar.style.width = Math.round(score * 100) + '%';
            bar.textContent = score.toFixed(3);
            progress.appendChild(bar);
            tdScore.appendChild(progress);
            tr.appendChild(tdScore);

            var tdEvidence = document.createElement('td');
            tdEvidence.textContent = a.evidence_count || 0;
            tr.appendChild(tdEvidence);

            var tdStatus = document.createElement('td');
            var spanStatus = document.createElement('span');
            spanStatus.className = 'badge bg-' + badge;
            spanStatus.textContent = label;
            tdStatus.appendChild(spanStatus);
            tr.appendChild(tdStatus);

            tbody.appendChild(tr);
        });

        // Pagination controls state
        var info = document.getElementById('trust-page-info');
        if (info) {
            if (total === 0) {
                info.textContent = 'No rows';
            } else {
                info.textContent = (start + 1) + '\u2013' + end
                    + ' of ' + total.toLocaleString();
            }
        }
        var detail = document.getElementById('trust-page-detail');
        if (detail) {
            detail.textContent = 'Page ' + (STATE.page + 1)
                + ' of ' + Math.max(1, maxPage + 1);
        }
        var prev = document.getElementById('trust-prev-btn');
        var next = document.getElementById('trust-next-btn');
        if (prev) prev.disabled = STATE.page <= 0;
        if (next) next.disabled = STATE.page >= maxPage;
    }

    var TRUST_PANE_ID = 'operations-pane';

    async function loadTrustDashboard() {
        try {
            var resp = await fetch('/api/v3/trust/dashboard');
            if (!resp.ok) {
                // CRIT-1: pass lexically-scoped loadTrustDashboard (inside IIFE)
                showPaneError(TRUST_PANE_ID, paneErrorMessage(resp.status), loadTrustDashboard, true);
                return;
            }
            var data = await resp.json();

            clearPaneError(TRUST_PANE_ID, true);

            var agents = data.agents || [];
            STATE.agents = agents;
            STATE.page = 0;

            var el;
            el = document.getElementById('trust-agent-count');
            if (el) el.textContent = agents.length.toLocaleString();

            var avg = agents.length > 0
                ? (agents.reduce(function (s, a) {
                    return s + (a.trust_score || 0);
                }, 0) / agents.length).toFixed(3)
                : '\u2014';
            el = document.getElementById('trust-avg-score');
            if (el) el.textContent = avg;

            el = document.getElementById('trust-burst-count');
            if (el) el.textContent = (data.alerts || []).length;

            _renderPage();
        } catch (e) {
            showPaneError(TRUST_PANE_ID, paneErrorMessage(0), loadTrustDashboard, true);
            if (window.console && window.console.debug) {
                window.console.debug('Trust dashboard error:', e);
            }
        }
    }

    function _wireControls() {
        var sort = document.getElementById('trust-sort');
        if (sort) {
            sort.addEventListener('change', function () {
                STATE.sort = sort.value;
                STATE.page = 0;
                _renderPage();
            });
        }
        var pageSize = document.getElementById('trust-page-size');
        if (pageSize) {
            pageSize.addEventListener('change', function () {
                STATE.pageSize = parseInt(pageSize.value, 10) || 25;
                STATE.page = 0;
                _renderPage();
            });
        }
        var prev = document.getElementById('trust-prev-btn');
        if (prev) {
            prev.addEventListener('click', function () {
                if (STATE.page > 0) {
                    STATE.page -= 1;
                    _renderPage();
                }
            });
        }
        var next = document.getElementById('trust-next-btn');
        if (next) {
            next.addEventListener('click', function () {
                STATE.page += 1;
                _renderPage();
            });
        }
    }

    function _boot() {
        _wireControls();
        // Re-load when the old Bootstrap trust-tab fires or when
        // ng-shell activates operations-pane.
        var trustTab = document.getElementById('trust-tab');
        if (trustTab) {
            trustTab.addEventListener('shown.bs.tab', loadTrustDashboard);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _boot);
    } else {
        _boot();
    }

    // Public surface
    window.loadTrustDashboard = loadTrustDashboard;
})();

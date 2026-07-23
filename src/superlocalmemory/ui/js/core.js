// SuperLocalMemory V2 - Core Utilities
// Shared functions used by all other modules.
// Security: All dynamic text MUST pass through escapeHtml() before DOM insertion.
// Data originates from our own trusted local SQLite database (localhost only).

// ============================================================================
// v3.4.23 — slmFetch(): fetch with 15s abort timeout
// ----------------------------------------------------------------------------
// Bare fetch() never resolves when the daemon dies mid-request (socket kept
// open, Promise pending). That leaves dashboard spinners running forever and
// stacks up orphan fetches that make hard-refresh hang. slmFetch wraps every
// request in an AbortController with a 15 s ceiling, so a dead daemon
// surfaces as a normal rejection and the UI can show a clear error.
// ============================================================================

window.SLM_FETCH_TIMEOUT_MS = 15000;
window.SLM_INSTALL_TOKEN_KEY = 'slm_install_token';

// B2 (3.7.9): the install token is kept in a private closure, never in
// sessionStorage — a stored token is trivially readable by any injected
// script (sessionStorage.getItem), which was the theft leg of the XSS chain.
window.slmInstallToken = (function () {
    var _cache = null;  // in-memory only; not on window, not in storage
    return async function (forceRefresh) {
        if (!forceRefresh && _cache) return _cache;
        var response = await window.__slmOriginalFetch(
            '/internal/token',
            {credentials: 'same-origin'}
        );
        if (!response.ok) return '';
        var payload = await response.json();
        var token = payload && payload.token ? payload.token : '';
        if (token) _cache = token;
        return token;
    };
})();

// Global fetch patch: apply the abort timeout to every same-origin request
// automatically. 17 UI modules call bare fetch() — patching here avoids
// touching each one and guarantees no future callsite can regress to an
// un-timed fetch that holds the spinner forever. External URLs are passed
// through unchanged. Callers that already supply
// `signal` keep their own behavior. `init.timeoutMs` lets callers override
// the default per-request.
(function patchFetch() {
    if (window.__slmFetchPatched) return;
    window.__slmFetchPatched = true;
    var _origFetch = window.fetch.bind(window);
    window.__slmOriginalFetch = _origFetch;
    window.fetch = function (input, init) {
        init = Object.assign({}, init || {});
        var urlStr = typeof input === 'string'
            ? input
            : (input && (input.url || input.href)) || '';
        // Resolve every URL before deciding whether it is local. In particular,
        // protocol-relative URLs such as //host/path inherit the current scheme
        // but are not same-origin and must never receive the install token.
        var isSameOrigin = false;
        var requestUrl = null;
        try {
            requestUrl = new URL(urlStr, window.location.href);
            isSameOrigin = requestUrl.origin === window.location.origin;
        } catch (e) {
            // A malformed URL is handled by native fetch. Treat it as external
            // here so it cannot obtain local credentials while failing.
        }
        var method = String(
            init.method || (input && input.method) || 'GET'
        ).toUpperCase();
        var mutating = ['POST', 'PUT', 'PATCH', 'DELETE'].indexOf(method) !== -1;
        var invalidatesCache = init.slmInvalidatesCache !== false;
        var requiresWriteAuth = init.slmRequiresWriteAuth !== false;
        delete init.slmInvalidatesCache;
        delete init.slmRequiresWriteAuth;

        function invalidateAfterMutation(request) {
            if (!isSameOrigin || !mutating || !invalidatesCache) return request;
            return request.then(function (response) {
                if (
                    response && response.ok &&
                    typeof window.slmInvalidatePanes === 'function'
                ) {
                    window.slmInvalidatePanes();
                }
                if (
                    response && response.ok &&
                    typeof window.slmInvalidateDashboardCache === 'function'
                ) {
                    window.slmInvalidateDashboardCache();
                }
                return response;
            });
        }

        function send() {
            if (!isSameOrigin || init.signal) {
                return invalidateAfterMutation(_origFetch(input, init));
            }
            var controller = new AbortController();
            var timeoutMs = init.timeoutMs || window.SLM_FETCH_TIMEOUT_MS;
            var timer = setTimeout(function () { controller.abort(); }, timeoutMs);
            init.signal = controller.signal;
            return invalidateAfterMutation(
                _origFetch(input, init).finally(function () { clearTimeout(timer); })
            );
        }

        if (
            isSameOrigin &&
            mutating &&
            requiresWriteAuth &&
            (!requestUrl || requestUrl.pathname !== '/internal/token')
        ) {
            return window.slmInstallToken(false).then(function (token) {
                if (!token) throw new Error('local write credential unavailable');
                var headers = new Headers(
                    init.headers || (input && input.headers) || {}
                );
                if (!headers.has('X-Install-Token')) {
                    headers.set('X-Install-Token', token);
                }
                init.headers = headers;
                init.credentials = init.credentials || 'same-origin';
                return send();
            });
        }
        return send();
    };
})();

// Thin named wrapper for callsites that want explicit timeout control.
// Equivalent to the patched fetch above but accepts `init.timeoutMs`.
async function slmFetch(input, init) {
    return fetch(input, init || {});
}

// ============================================================================
// v3.4.23 — version fingerprint + auto-reload on daemon upgrade
// ----------------------------------------------------------------------------
// index.html ships with <meta name="slm-version" content="__SLM_VERSION__">
// that the server fills in at serve time. After page load we ask the daemon
// for its current version via /api/version; on mismatch we clear localStorage
// and hard-reload once, so a stale tab never lingers after `slm restart` or
// a package upgrade. Guarded by sessionStorage to avoid reload loops.
// ============================================================================

async function checkVersionFingerprint() {
    try {
        var metaEl = document.querySelector('meta[name="slm-version"]');
        var pageVersion = metaEl ? metaEl.getAttribute('content') : null;
        if (!pageVersion || pageVersion === '__SLM_VERSION__') return;
        var resp = await slmFetch('/api/version', { timeoutMs: 5000 });
        if (!resp.ok) return;
        var data = await resp.json();
        var serverVersion = data && data.version;
        if (!serverVersion || serverVersion === pageVersion) return;
        try {
            if (sessionStorage.getItem('slm-version-reload-done') === serverVersion) {
                console.warn('[slm] version mismatch persists after reload:',
                    pageVersion, '!=', serverVersion);
                return;
            }
            sessionStorage.setItem('slm-version-reload-done', serverVersion);
        } catch (e) {
            // sessionStorage blocked (private mode, quota, etc.) — fall through
            // to reload. Worst case: we reload twice instead of once, still
            // safe because server version converges on second attempt.
        }
        try {
            // Preserve theme; drop everything else that might be stale.
            var theme = localStorage.getItem('slm-theme');
            localStorage.clear();
            if (theme) localStorage.setItem('slm-theme', theme);
        } catch (e) { /* localStorage may be blocked */ }
        console.info('[slm] daemon upgraded', pageVersion, '->', serverVersion,
            '— reloading');
        location.reload();
    } catch (err) {
        // Network error or daemon down: don't reload, just log.
        console.debug('[slm] version check skipped:', err && err.message);
    }
}

// ============================================================================
// Dark Mode
// ============================================================================

function initDarkMode() {
    var saved = localStorage.getItem('slm-theme');
    var theme;
    if (saved) {
        theme = saved;
    } else {
        theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    applyTheme(theme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
    var icon = document.getElementById('theme-icon');
    if (icon) {
        icon.className = theme === 'dark' ? 'bi bi-moon-stars-fill' : 'bi bi-sun-fill';
    }
}

function toggleDarkMode() {
    var current = document.documentElement.getAttribute('data-bs-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    localStorage.setItem('slm-theme', next);
    applyTheme(next);
}

// ============================================================================
// Animated Counter
// ============================================================================

function animateCounter(elementId, target) {
    var el = document.getElementById(elementId);
    if (!el) return;
    var duration = 600;
    var startTime = null;

    function step(timestamp) {
        if (!startTime) startTime = timestamp;
        var progress = Math.min((timestamp - startTime) / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(eased * target).toLocaleString();
        if (progress < 1) {
            requestAnimationFrame(step);
        } else {
            el.textContent = target.toLocaleString();
        }
    }

    if (target === 0) {
        el.textContent = '0';
    } else {
        requestAnimationFrame(step);
    }
}

// ============================================================================
// HTML Escaping — all dynamic text MUST pass through this before DOM insertion
// ============================================================================

function escapeHtml(text) {
    if (!text) return '';
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(String(text)));
    return div.innerHTML;
}

// ============================================================================
// Loading / Empty State helpers
// ============================================================================

function showLoading(containerId, message) {
    var el = document.getElementById(containerId);
    if (!el) return;
    el.textContent = '';
    var wrapper = document.createElement('div');
    wrapper.className = 'loading';
    var spinner = document.createElement('div');
    spinner.className = 'spinner-border text-primary';
    spinner.setAttribute('role', 'status');
    var msg = document.createElement('div');
    msg.textContent = message || 'Loading...';
    wrapper.appendChild(spinner);
    wrapper.appendChild(msg);
    el.appendChild(wrapper);
}

function showEmpty(containerId, icon, message) {
    var el = document.getElementById(containerId);
    if (!el) return;
    el.textContent = '';
    var wrapper = document.createElement('div');
    wrapper.className = 'empty-state';
    var iconEl = document.createElement('i');
    iconEl.className = 'bi bi-' + icon + ' d-block';
    var p = document.createElement('p');
    p.textContent = message;
    wrapper.appendChild(iconEl);
    wrapper.appendChild(p);
    el.appendChild(wrapper);
}

// ============================================================================
// Pane error state — WP-12
// showPaneError / clearPaneError / paneErrorMessage
// ============================================================================

/**
 * Map an HTTP status (or 0 for network failure) to a user-readable message.
 * @param {number} status  0 = network/abort, ≥400 = HTTP error
 * @returns {string}
 */
function paneErrorMessage(status) {
    if (!status || status === 0) {
        return 'Service unavailable — check network connection';
    }
    if (status >= 500) {
        return 'Server error ' + status + ' — daemon may be down';
    }
    return 'Request failed ' + status;
}

/**
 * Render an inline error banner inside a pane container.
 *
 * slotMode=false (container panes — math-health, ide-status):
 *   Clears the container and appends the error div directly.
 *
 * slotMode=true (field-scatter panes — dashboard, trust, optimize):
 *   Inserts/replaces a #<containerId>-error-slot div at the top of the
 *   container, leaving existing field values visible.
 *
 * @param {string}        containerId  - getElementById target
 * @param {string}        message      - user-facing message (set via textContent — XSS-safe)
 * @param {Function|null} onRetry      - callback for Retry button; null = no button
 * @param {boolean}       slotMode     - true for field-scatter panes
 */
function showPaneError(containerId, message, onRetry, slotMode) {
    var el = document.getElementById(containerId);
    if (!el) return;

    // Build the error div
    var errDiv = document.createElement('div');
    errDiv.className = 'pane-error';
    errDiv.setAttribute('role', 'alert');

    var icon = document.createElement('i');
    icon.className = 'bi bi-exclamation-triangle';
    errDiv.appendChild(icon);

    var p = document.createElement('p');
    p.textContent = message; // textContent — no XSS risk
    errDiv.appendChild(p);

    if (typeof onRetry === 'function') {
        var btn = document.createElement('button');
        btn.className = 'btn btn-sm btn-outline-danger pane-error-retry';
        btn.textContent = 'Retry';
        // Capture onRetry lexically to avoid closure issues in IIFEs (CRIT-1)
        (function(cb) {
            btn.addEventListener('click', function() { cb(); });
        }(onRetry));
        errDiv.appendChild(btn);
    }

    if (slotMode) {
        // Insert/replace error slot at top of container
        var slotId = containerId + '-error-slot';
        errDiv.id = slotId;
        var existing = document.getElementById(slotId);
        if (existing) {
            existing.parentNode.replaceChild(errDiv, existing);
        } else {
            el.insertBefore(errDiv, el.firstChild);
        }
    } else {
        // Replace container contents
        el.textContent = '';
        el.appendChild(errDiv);
    }
}

/**
 * Remove the error state previously set by showPaneError.
 *
 * @param {string}  containerId - getElementById target
 * @param {boolean} slotMode    - must match the slotMode used in showPaneError
 */
function clearPaneError(containerId, slotMode) {
    if (slotMode) {
        var slot = document.getElementById(containerId + '-error-slot');
        if (slot && slot.parentNode) slot.parentNode.removeChild(slot);
    } else {
        var el = document.getElementById(containerId);
        if (!el) return;
        var err = el.querySelector('.pane-error');
        if (err) el.removeChild(err);
    }
}

// ============================================================================
// Safe HTML builder — tagged template for escaped interpolation
// ============================================================================

function safeHtml(templateParts) {
    var args = Array.prototype.slice.call(arguments, 1);
    var result = '';
    for (var i = 0; i < templateParts.length; i++) {
        result += templateParts[i];
        if (i < args.length) {
            result += escapeHtml(String(args[i]));
        }
    }
    return result;
}

// ============================================================================
// File Download helper
// ============================================================================

function downloadFile(filename, content, mimeType) {
    var blob = new Blob([content], { type: mimeType });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================================
// Toast notification
// ============================================================================

function showToast(message) {
    var toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#333;color:#fff;padding:10px 20px;border-radius:8px;font-size:0.9rem;z-index:9999;opacity:0;transition:opacity 0.3s;';
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(function() { toast.style.opacity = '1'; });
    setTimeout(function() {
        toast.style.opacity = '0';
        setTimeout(function() {
            if (toast.parentNode) document.body.removeChild(toast);
        }, 300);
    }, 2000);
}

// ============================================================================
// Date Formatters
// ============================================================================

function formatDate(dateString) {
    if (!dateString) return '-';
    var date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatDateFull(dateString) {
    if (!dateString) return '-';
    var date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// ============================================================================
// Stats (loaded on startup)
// ============================================================================

async function loadStats() {
    try {
        var response = await slmFetch('/api/stats');
        var data = await response.json();
        var ov = data.overview || {};
        var memories = ov.total_memories || 0;
        var facts = ov.total_facts || 0;
        var ratio = ov.facts_per_memory || 0;

        animateCounter('stat-memories', memories);
        animateCounter('stat-facts', facts);
        animateCounter('stat-nodes', ov.graph_nodes || 0);
        animateCounter('stat-edges', ov.graph_edges || 0);

        var mSub = document.getElementById('stat-memories-sub');
        if (mSub) {
            mSub.textContent = memories > 0
                ? 'what you stored'
                : '\u00a0';
        }
        var fSub = document.getElementById('stat-facts-sub');
        if (fSub) {
            fSub.textContent = memories > 0
                ? 'avg ' + ratio + ' per memory'
                : '\u00a0';
        }
        populateFilters(data.categories || [], data.projects || []);
    } catch (error) {
        console.error('Error loading stats:', error);
        animateCounter('stat-memories', 0);
        animateCounter('stat-facts', 0);
        animateCounter('stat-nodes', 0);
        animateCounter('stat-edges', 0);
    }
}

// Refresh entire dashboard — called by the refresh button in the header
function refreshDashboard() {
    loadProfiles();
    loadStats();
    if (typeof loadGraph === 'function') loadGraph();
    if (typeof loadMemories === 'function') loadMemories();
    if (typeof loadEventStats === 'function') loadEventStats();
    if (typeof loadAgents === 'function') loadAgents();
}

function populateFilters(categories, projects) {
    var categorySelect = document.getElementById('filter-category');
    var projectSelect = document.getElementById('filter-project');
    // Clear existing options beyond the first placeholder to prevent duplicates on refresh
    if (categorySelect) while (categorySelect.options.length > 1) categorySelect.remove(1);
    if (projectSelect) while (projectSelect.options.length > 1) projectSelect.remove(1);
    // Guard appendChild: in the OD dashboard #filter-category / #filter-project
    // live in a folded pane and are frequently absent. Without these guards a
    // null.appendChild threw, tripping the loadStats() catch and zeroing every
    // dashboard counter after a profile create/switch.
    if (categorySelect) {
        categories.forEach(function(cat) {
            if (cat.category) {
                var option = document.createElement('option');
                option.value = cat.category;
                option.textContent = cat.category + ' (' + cat.count + ')';
                categorySelect.appendChild(option);
            }
        });
    }
    if (projectSelect) {
        projects.forEach(function(proj) {
            if (proj.project_name) {
                var option = document.createElement('option');
                option.value = proj.project_name;
                option.textContent = proj.project_name + ' (' + proj.count + ')';
                projectSelect.appendChild(option);
            }
        });
    }
}

// ============================================================================
// Application Init (DOMContentLoaded)
// ============================================================================

window.addEventListener('DOMContentLoaded', function() {
    initDarkMode();
    // v3.4.23: version check runs first and non-blocking. If a mismatch is
    // detected it triggers location.reload(), so the rest of init on the
    // stale page becomes a no-op.
    checkVersionFingerprint();
    loadProfiles();
    loadStats();
    loadGraph();
    // DASH-V1/V2/V3 fix: populate the landing-page cards (Operating Mode,
    // LLM Provider, Memories facts, Version) on first paint. loadDashboard()
    // was only bound to visibilitychange/focus/hashchange, none of which fire
    // on a fresh load, so those cards were stuck on their static "Loading…".
    if (typeof loadDashboard === 'function') loadDashboard();

    // v2.5 — Event Bus + Agent Registry (graceful if functions don't exist)
    if (typeof initEventStream === 'function') initEventStream();
    if (typeof loadEventStats === 'function') loadEventStats();
    if (typeof loadAgents === 'function') loadAgents();

    // v3.4.10 — Account widget (cloud backup status in header)
    if (typeof loadCloudDestinations === 'function') loadCloudDestinations();
});

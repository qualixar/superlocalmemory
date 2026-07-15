/**
 * SuperLocalMemory V2 - Feedback Module (v2.7.4)
 * Copyright (c) 2026 Varun Pratap Bhardwaj
 * Licensed under GNU Affero General Public License v3.0 or later
 *
 * Collects implicit and explicit feedback signals from dashboard
 * interactions. All data stays 100% local.
 *
 * Signals:
 *   thumbs_up    - User clicks thumbs up on a memory card
 *   thumbs_down  - User clicks thumbs down on a memory card
 *   pin          - User pins/bookmarks a memory
 *   dwell_time   - Time spent viewing memory detail modal
 *   search_click - User clicks a search result (positive for clicked)
 */

// Module state
var feedbackState = {
    lastSearchQuery: '',
    modalOpenTime: null,
    modalMemoryId: null,
    searchResultIds: [],
};

/**
 * Record explicit feedback (thumbs up/down, pin).
 * Sends POST /api/feedback to backend.
 */
function recordFeedback(memoryId, feedbackType, query) {
    if (!memoryId || !feedbackType) return;

    fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            memory_id: memoryId,
            feedback_type: feedbackType,
            query: query || feedbackState.lastSearchQuery || '',
        }),
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.success) {
            showFeedbackToast(feedbackType, memoryId);
            refreshFeedbackStats();
        }
    })
    .catch(function() {
        // Silent failure — feedback should never break UI
    });
}

/**
 * Start tracking dwell time when a memory detail modal opens.
 */
function startDwellTracking(memoryId) {
    feedbackState.modalOpenTime = Date.now();
    feedbackState.modalMemoryId = memoryId;
}

/**
 * Stop tracking and record dwell time when modal closes.
 */
function stopDwellTracking() {
    if (!feedbackState.modalOpenTime || !feedbackState.modalMemoryId) return;

    var dwellMs = Date.now() - feedbackState.modalOpenTime;
    var dwellSeconds = dwellMs / 1000;
    var memId = feedbackState.modalMemoryId;

    feedbackState.modalOpenTime = null;
    feedbackState.modalMemoryId = null;

    // Only record if meaningful (>1s)
    if (dwellSeconds < 1.0) return;

    fetch('/api/feedback/dwell', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            memory_id: memId,
            dwell_time: dwellSeconds,
            query: feedbackState.lastSearchQuery || '',
        }),
    })
    .then(function(r) { return r.json(); })
    .catch(function() {
        // Silent failure
    });
}

/**
 * Track which search results are displayed.
 */
function trackSearchResults(query, resultIds) {
    feedbackState.lastSearchQuery = query;
    feedbackState.searchResultIds = resultIds || [];
}

/**
 * Create feedback buttons (thumbs up/down + pin) for a memory card.
 * Returns a DOM element containing the buttons.
 * Uses safe DOM methods (no innerHTML with user data).
 */
function createFeedbackButtons(memoryId, query) {
    var container = document.createElement('div');
    container.className = 'feedback-buttons d-flex gap-1 align-items-center';
    container.setAttribute('role', 'group');
    container.setAttribute('aria-label', 'Feedback for memory ' + memoryId);

    // Thumbs up
    var upBtn = document.createElement('button');
    upBtn.className = 'btn btn-outline-success btn-sm feedback-btn';
    var upIcon = document.createElement('i');
    upIcon.className = 'bi bi-hand-thumbs-up';
    upBtn.appendChild(upIcon);
    upBtn.title = 'This memory was useful';
    upBtn.setAttribute('aria-label', 'Mark as useful');
    upBtn.onclick = function(e) {
        e.stopPropagation();
        recordFeedback(memoryId, 'thumbs_up', query);
        upBtn.classList.remove('btn-outline-success');
        upBtn.classList.add('btn-success');
        upBtn.disabled = true;
        downBtn.disabled = true;
    };

    // Thumbs down
    var downBtn = document.createElement('button');
    downBtn.className = 'btn btn-outline-danger btn-sm feedback-btn';
    var downIcon = document.createElement('i');
    downIcon.className = 'bi bi-hand-thumbs-down';
    downBtn.appendChild(downIcon);
    downBtn.title = 'Not useful';
    downBtn.setAttribute('aria-label', 'Mark as not useful');
    downBtn.onclick = function(e) {
        e.stopPropagation();
        recordFeedback(memoryId, 'thumbs_down', query);
        downBtn.classList.remove('btn-outline-danger');
        downBtn.classList.add('btn-danger');
        upBtn.disabled = true;
        downBtn.disabled = true;
    };

    // Pin/Bookmark
    var pinBtn = document.createElement('button');
    pinBtn.className = 'btn btn-outline-warning btn-sm feedback-btn';
    var pinIcon = document.createElement('i');
    pinIcon.className = 'bi bi-pin-angle';
    pinBtn.appendChild(pinIcon);
    pinBtn.title = 'Pin this memory';
    pinBtn.setAttribute('aria-label', 'Pin memory');
    pinBtn.onclick = function(e) {
        e.stopPropagation();
        recordFeedback(memoryId, 'pin', query);
        pinBtn.classList.remove('btn-outline-warning');
        pinBtn.classList.add('btn-warning');
        pinBtn.disabled = true;
    };

    container.appendChild(upBtn);
    container.appendChild(downBtn);
    container.appendChild(pinBtn);
    return container;
}

/**
 * Show a small toast notification after feedback.
 */
function showFeedbackToast(type, memoryId) {
    var messages = {
        thumbs_up: 'Thanks! This helps improve future results.',
        thumbs_down: 'Noted. Results will improve over time.',
        pin: 'Memory pinned!',
    };
    var msg = messages[type] || 'Feedback recorded';

    // Use existing showToast if available
    if (typeof showToast === 'function') {
        showToast(msg, 'success');
    }
}

/**
 * Fetch and render feedback stats (progress bar, signal count).
 */
function refreshFeedbackStats() {
    fetch('/api/feedback/stats')
    .then(function(r) { return r.json(); })
    .then(function(data) {
        renderFeedbackProgress(data);
    })
    .catch(function() {});
}

/**
 * Render the feedback progress bar using safe DOM methods.
 */
function renderFeedbackProgress(stats) {
    var container = document.getElementById('feedback-progress');
    if (!container) return;

    var total = stats.total_signals || 0;
    var phase = stats.ranking_phase || 'baseline';
    var progress = stats.progress || 0;
    var target = stats.target || 200;

    var phaseLabels = {
        baseline: 'Baseline (collecting data)',
        rule_based: 'Rule-Based (learning your preferences)',
        ml_model: 'ML-Powered (fully personalized)',
    };
    var phaseColors = {
        baseline: 'bg-secondary',
        rule_based: 'bg-info',
        ml_model: 'bg-success',
    };

    // Clear container
    while (container.firstChild) container.removeChild(container.firstChild);

    // Header row
    var headerRow = document.createElement('div');
    headerRow.className = 'd-flex justify-content-between align-items-center mb-1';

    var label = document.createElement('small');
    label.className = 'text-muted';
    var icon = document.createElement('i');
    icon.className = 'bi bi-graph-up me-1';
    label.appendChild(icon);
    label.appendChild(document.createTextNode('Learning Progress'));

    var badge = document.createElement('span');
    badge.className = 'badge ' + (phaseColors[phase] || 'bg-secondary');
    badge.textContent = phaseLabels[phase] || phase;

    headerRow.appendChild(label);
    headerRow.appendChild(badge);
    container.appendChild(headerRow);

    // Progress bar
    var progressOuter = document.createElement('div');
    progressOuter.className = 'progress';
    progressOuter.style.height = '8px';

    var progressBar = document.createElement('div');
    progressBar.className = 'progress-bar ' + (phaseColors[phase] || 'bg-secondary');
    progressBar.setAttribute('role', 'progressbar');
    progressBar.style.width = progress + '%';
    progressBar.setAttribute('aria-valuenow', String(total));
    progressBar.setAttribute('aria-valuemin', '0');
    progressBar.setAttribute('aria-valuemax', String(target));

    progressOuter.appendChild(progressBar);
    container.appendChild(progressOuter);

    // Count text
    var countText = document.createElement('small');
    countText.className = 'text-muted';
    countText.textContent = total + '/' + target + ' signals';
    container.appendChild(countText);
}

/**
 * Create the privacy notice banner using safe DOM methods.
 */
function createPrivacyNotice() {
    var container = document.getElementById('privacy-notice');
    if (!container) return;

    var alert = document.createElement('div');
    alert.className = 'alert alert-light border d-flex align-items-center py-2 mb-3';
    alert.setAttribute('role', 'alert');

    var lockIcon = document.createElement('i');
    lockIcon.className = 'bi bi-shield-lock me-2 text-success';

    var text = document.createElement('small');
    text.appendChild(document.createTextNode('Learning is 100% local. Your behavioral data never leaves this machine. '));

    var link = document.createElement('a');
    link.href = '#';
    link.className = 'ms-1';
    link.textContent = 'Learn more';
    link.onclick = function(e) {
        e.preventDefault();
        showPrivacyDetails();
    };
    text.appendChild(link);

    alert.appendChild(lockIcon);
    alert.appendChild(text);
    container.appendChild(alert);
}

/**
 * Show privacy details in an alert (safe, no raw HTML injection).
 */
function showPrivacyDetails() {
    var info = 'What data is collected:\n' +
        '- Which memories you find useful (thumbs up/down)\n' +
        '- Time spent viewing memory details\n' +
        '- Search patterns (queries are hashed, never stored as raw text)\n\n' +
        'Where it is stored:\n' +
        '~/.superlocalmemory/learning.db (local SQLite file)\n\n' +
        'How to delete it:\n' +
        'Use "Reset Learning Data" in Settings, or run:\n' +
        'rm ~/.superlocalmemory/learning.db';
    alert(info);
}

/**
 * Reset all learning data.
 */
function resetLearningData() {
    if (!confirm('Reset all learning data? Your memories will be preserved.')) return;

    fetch('/api/learning/reset', {method: 'POST'})
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.success) {
            if (typeof showToast === 'function') showToast('Learning data reset', 'success');
            refreshFeedbackStats();
        }
    })
    .catch(function() {});
}

// Initialize: load stats on page ready
document.addEventListener('DOMContentLoaded', function() {
    createPrivacyNotice();
    refreshFeedbackStats();
});

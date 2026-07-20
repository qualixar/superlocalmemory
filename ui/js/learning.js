// SuperLocalMemory V2 - Learning System Dashboard (v2.7)
// Copyright (c) 2026 Varun Pratap Bhardwaj - GNU Affero General Public License v3.0 or later
// NOTE: All dynamic values pass through escapeHtml() from core.js before DOM insertion.

var _learningData = null;

async function loadLearning() {
    try {
        var response = await fetch('/api/learning/status');
        var data = await response.json();
        _learningData = data;
        renderLearningStatus(data);
    } catch (error) {
        console.error('Error loading learning status:', error);
        var el = document.getElementById('learning-phase');
        if (el) el.textContent = 'Unavailable';
        var detail = document.getElementById('learning-phase-detail');
        if (detail) detail.textContent = 'Learning system not available';
    }
}

function renderLearningStatus(data) {
    renderPhase(data);
    renderFeedbackCount(data.stats);
    renderEngagementHealth(data.engagement);
    renderProgressBar(data.stats ? data.stats.feedback_count : 0);
    renderWhatWeLearned(data);
    renderTechPreferences(data.tech_preferences || []);
    renderWorkflowPatterns(data.workflow_patterns || []);
    renderSourceQuality(data.source_scores || {});
    renderPrivacyStats(data.stats);
}

function renderWhatWeLearned(data) {
    var container = document.getElementById('what-we-learned-content');
    var profileBadge = document.getElementById('learned-profile-badge');
    if (!container) return;
    container.textContent = '';

    // Show active profile
    if (profileBadge && data.stats && data.stats.active_profile) {
        profileBadge.textContent = data.stats.active_profile;
    }

    var insights = [];

    // Collect tech preference insights
    var techPrefs = data.tech_preferences || [];
    var highConfTech = techPrefs.filter(function(p) { return p.confidence >= 0.6; });
    if (highConfTech.length > 0) {
        var techNames = highConfTech.map(function(p) { return p.value; }).slice(0, 5);
        insights.push({
            icon: 'bi-cpu',
            color: 'text-primary',
            text: 'You prefer: ' + techNames.join(', '),
            detail: highConfTech.length + ' tech preferences learned',
        });
    }

    // Collect workflow insights
    var workflows = data.workflow_patterns || [];
    var sequences = workflows.filter(function(w) { return w.type === 'sequence'; });
    var temporal = workflows.filter(function(w) { return w.type === 'temporal'; });
    if (sequences.length > 0) {
        insights.push({
            icon: 'bi-diagram-3',
            color: 'text-info',
            text: sequences.length + ' workflow sequence' + (sequences.length > 1 ? 's' : '') + ' detected',
            detail: 'Common patterns in how you work',
        });
    }
    if (temporal.length > 0) {
        insights.push({
            icon: 'bi-clock',
            color: 'text-warning',
            text: temporal.length + ' time-based pattern' + (temporal.length > 1 ? 's' : '') + ' found',
            detail: 'When you tend to work on what',
        });
    }

    // Source quality insights
    var sources = data.source_scores || {};
    var sourceCount = Object.keys(sources).length;
    if (sourceCount > 0) {
        var bestSource = Object.entries(sources).sort(function(a, b) { return b[1] - a[1]; })[0];
        insights.push({
            icon: 'bi-trophy',
            color: 'text-success',
            text: 'Best source: ' + bestSource[0] + ' (' + Math.round(bestSource[1] * 100) + '% quality)',
            detail: sourceCount + ' sources tracked',
        });
    }

    // Feedback volume insight
    var feedbackCount = (data.stats && data.stats.feedback_count) || 0;
    if (feedbackCount > 0) {
        var phase = data.ranking_phase || 'baseline';
        var phaseLabel = {baseline: 'collecting data', rule_based: 'learning your preferences', ml_model: 'fully personalized'}[phase] || phase;
        insights.push({
            icon: 'bi-graph-up',
            color: 'text-success',
            text: feedbackCount + ' feedback signals collected',
            detail: 'Currently ' + phaseLabel,
        });
    }

    // Render insights
    if (insights.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        var emptyIcon = document.createElement('i');
        emptyIcon.className = 'bi bi-lightbulb';
        emptyIcon.style.fontSize = '2rem';
        empty.appendChild(emptyIcon);
        var emptyText = document.createElement('p');
        emptyText.className = 'mt-2 mb-0 small';
        emptyText.textContent = 'Start using recall and giving feedback. SuperLocalMemory will learn your preferences automatically.';
        empty.appendChild(emptyText);
        container.appendChild(empty);
        return;
    }

    for (var i = 0; i < insights.length; i++) {
        var insight = insights[i];
        var row = document.createElement('div');
        row.className = 'd-flex align-items-start mb-2 p-2 rounded';
        row.style.backgroundColor = 'var(--bs-body-bg)';

        var icon = document.createElement('i');
        icon.className = 'bi ' + insight.icon + ' ' + insight.color + ' me-3';
        icon.style.fontSize = '1.3rem';
        icon.style.marginTop = '2px';

        var textDiv = document.createElement('div');
        var mainText = document.createElement('div');
        mainText.className = 'fw-semibold';
        mainText.textContent = insight.text;
        var detailText = document.createElement('small');
        detailText.className = 'text-muted';
        detailText.textContent = insight.detail;
        textDiv.appendChild(mainText);
        textDiv.appendChild(detailText);

        row.appendChild(icon);
        row.appendChild(textDiv);
        container.appendChild(row);
    }
}

function renderPhase(data) {
    var phaseEl = document.getElementById('learning-phase');
    var phaseDetail = document.getElementById('learning-phase-detail');
    if (!phaseEl) return;

    if (!data.ranking_phase) {
        phaseEl.textContent = 'Not Available';
        phaseEl.style.color = 'var(--bs-secondary)';
        if (phaseDetail) phaseDetail.textContent = 'Install: pip3 install lightgbm scipy';
        return;
    }

    var labels = { 'baseline': 'Baseline', 'rule_based': 'Rule-Based', 'ml_model': 'ML Model' };
    var colors = { 'baseline': 'var(--bs-secondary)', 'rule_based': 'var(--bs-primary)', 'ml_model': 'var(--bs-warning)' };
    phaseEl.textContent = labels[data.ranking_phase] || data.ranking_phase;
    phaseEl.style.color = colors[data.ranking_phase] || 'var(--bs-primary)';

    var fc = (data.stats && data.stats.feedback_count) || 0;
    if (phaseDetail) {
        if (data.ranking_phase === 'baseline') phaseDetail.textContent = 'Need 20+ signals. Currently: ' + fc;
        else if (data.ranking_phase === 'rule_based') phaseDetail.textContent = 'Active! Need 200+ for ML. Currently: ' + fc;
        else phaseDetail.textContent = 'Full ML ranking with ' + fc + ' signals';
    }
}

function renderFeedbackCount(stats) {
    var el = document.getElementById('learning-feedback-count');
    var detail = document.getElementById('learning-feedback-detail');
    if (el && stats) el.textContent = stats.feedback_count || 0;
    if (detail && stats) detail.textContent = (stats.unique_queries || 0) + ' unique queries';
}

function renderEngagementHealth(engagement) {
    var el = document.getElementById('learning-health');
    var detail = document.getElementById('learning-health-detail');
    if (!el) return;
    if (!engagement) { el.textContent = '--'; return; }

    var status = engagement.health_status || 'UNKNOWN';
    var colors = { 'HEALTHY': 'var(--bs-success)', 'DECLINING': 'var(--bs-warning)', 'AT_RISK': 'var(--bs-danger)', 'INACTIVE': 'var(--bs-secondary)' };
    el.textContent = status;
    el.style.color = colors[status] || 'var(--bs-secondary)';
    if (detail) {
        var p = [];
        if (engagement.days_active !== undefined) p.push(engagement.days_active + ' days active');
        if (engagement.memories_per_day !== undefined) p.push(engagement.memories_per_day.toFixed(1) + ' mem/day');
        detail.textContent = p.join(' | ') || 'No data';
    }
}

function renderProgressBar(count) {
    var bar = document.getElementById('learning-progress');
    if (!bar) return;
    var pct = 0, cls = 'bg-secondary', lbl = '';

    if (count >= 200) { pct = 100; cls = 'bg-warning'; lbl = 'ML Model Active'; }
    else if (count >= 20) { pct = 10 + ((count - 20) / 180) * 60; cls = 'bg-primary'; lbl = 'Rule-Based (' + count + '/200)'; }
    else if (count > 0) { pct = (count / 20) * 10; lbl = 'Baseline (' + count + '/20)'; }
    else { lbl = 'No feedback yet'; }

    bar.style.width = Math.min(pct, 100) + '%';
    bar.className = 'progress-bar ' + cls;
    bar.textContent = lbl;
}

function renderTechPreferences(patterns) {
    var container = document.getElementById('learning-tech-prefs');
    if (!container) return;
    container.textContent = '';

    if (!patterns || patterns.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'No patterns yet. Use recall + feedback to start learning.';
        container.appendChild(empty);
        return;
    }

    for (var i = 0; i < patterns.length; i++) {
        var p = patterns[i];
        var confPct = Math.round((p.confidence || 0) * 100);
        var barColor = confPct >= 80 ? 'bg-success' : (confPct >= 60 ? 'bg-primary' : 'bg-secondary');

        var row = document.createElement('div');
        row.className = 'd-flex align-items-center mb-2';

        var label = document.createElement('div');
        label.style.minWidth = '120px';
        var labelSpan = document.createElement('span');
        labelSpan.className = 'text-muted small';
        labelSpan.textContent = p.key || '';
        label.appendChild(labelSpan);

        var barWrap = document.createElement('div');
        barWrap.className = 'flex-grow-1 mx-2';
        var progress = document.createElement('div');
        progress.className = 'progress';
        progress.style.height = '18px';
        progress.style.borderRadius = '9px';
        var barEl = document.createElement('div');
        barEl.className = 'progress-bar ' + barColor;
        barEl.style.width = confPct + '%';
        barEl.style.borderRadius = '9px';
        barEl.style.fontSize = '0.7rem';
        barEl.textContent = (p.value || '') + ' (' + confPct + '%)';
        progress.appendChild(barEl);
        barWrap.appendChild(progress);

        var evidence = document.createElement('small');
        evidence.className = 'text-muted';
        evidence.style.minWidth = '60px';
        evidence.style.textAlign = 'right';
        evidence.textContent = (p.evidence || 0) + ' ev.';

        row.appendChild(label);
        row.appendChild(barWrap);
        row.appendChild(evidence);
        container.appendChild(row);
    }
}

function renderWorkflowPatterns(workflows) {
    var container = document.getElementById('learning-workflows');
    if (!container) return;
    container.textContent = '';

    if (!workflows || workflows.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'Sequences detected after 30+ memories.';
        container.appendChild(empty);
        return;
    }

    for (var i = 0; i < workflows.length; i++) {
        var w = workflows[i];
        var card = document.createElement('div');
        card.className = 'mb-2 p-2 border rounded';

        if (w.type === 'sequence') {
            var seq = [];
            try { seq = JSON.parse(w.value); } catch(e) { seq = [String(w.value)]; }
            var flowDiv = document.createElement('div');
            flowDiv.className = 'd-flex align-items-center flex-wrap gap-1';
            for (var j = 0; j < seq.length; j++) {
                if (j > 0) {
                    var arrow = document.createElement('i');
                    arrow.className = 'bi bi-arrow-right text-muted small';
                    flowDiv.appendChild(arrow);
                }
                var badge = document.createElement('span');
                badge.className = 'badge bg-primary bg-opacity-75';
                badge.textContent = seq[j];
                flowDiv.appendChild(badge);
            }
            card.appendChild(flowDiv);
        } else if (w.type === 'temporal') {
            // Parse temporal pattern: show "Morning: code (26%)" format
            var parsed = {};
            try { parsed = JSON.parse(w.value); } catch(e) { parsed = {}; }
            var timeBadge = document.createElement('span');
            timeBadge.className = 'badge bg-info bg-opacity-75 me-2';
            timeBadge.textContent = (w.key || '').charAt(0).toUpperCase() + (w.key || '').slice(1);
            card.appendChild(timeBadge);

            var activity = document.createElement('span');
            activity.className = 'fw-bold';
            activity.textContent = parsed.dominant_activity || w.value;
            card.appendChild(activity);

            var evCount = document.createElement('small');
            evCount.className = 'text-muted ms-2';
            evCount.textContent = '(' + (parsed.evidence_count || 0) + ' memories)';
            card.appendChild(evCount);

            // Show distribution as mini bar
            if (parsed.distribution) {
                var distDiv = document.createElement('div');
                distDiv.className = 'd-flex flex-wrap gap-1 mt-1';
                var sortedActivities = Object.entries(parsed.distribution).sort(function(a, b) { return b[1] - a[1]; });
                var total = sortedActivities.reduce(function(s, e) { return s + e[1]; }, 0);
                for (var k = 0; k < sortedActivities.length; k++) {
                    var actName = sortedActivities[k][0];
                    var actCount = sortedActivities[k][1];
                    var actPct = Math.round((actCount / total) * 100);
                    var actBadge = document.createElement('span');
                    actBadge.className = 'badge bg-light text-dark';
                    actBadge.style.fontSize = '0.65rem';
                    actBadge.textContent = actName + ' ' + actPct + '%';
                    distDiv.appendChild(actBadge);
                }
                card.appendChild(distDiv);
            }
        } else {
            var typeBadge = document.createElement('span');
            typeBadge.className = 'badge bg-info bg-opacity-75';
            typeBadge.textContent = w.key || w.type;
            card.appendChild(typeBadge);
            var valSpan = document.createElement('span');
            valSpan.className = 'small ms-1';
            valSpan.textContent = w.value || '';
            card.appendChild(valSpan);
        }

        var confSmall = document.createElement('small');
        confSmall.className = 'text-muted d-block';
        confSmall.textContent = 'Confidence: ' + Math.round((w.confidence || 0) * 100) + '%';
        card.appendChild(confSmall);
        container.appendChild(card);
    }
}

function renderSourceQuality(scores) {
    var container = document.getElementById('learning-sources');
    if (!container) return;
    container.textContent = '';

    var sources = Object.keys(scores);
    if (sources.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'Source quality computed after feedback signals.';
        container.appendChild(empty);
        return;
    }

    sources.sort(function(a, b) { return scores[b] - scores[a]; });

    for (var i = 0; i < sources.length; i++) {
        var src = sources[i];
        var pct = Math.round(scores[src] * 100);
        var barColor = pct >= 60 ? 'bg-success' : (pct >= 40 ? 'bg-warning' : 'bg-danger');

        var row = document.createElement('div');
        row.className = 'd-flex align-items-center mb-2';

        var label = document.createElement('div');
        label.style.minWidth = '140px';
        var code = document.createElement('code');
        code.className = 'small';
        code.textContent = src;
        label.appendChild(code);

        var barWrap = document.createElement('div');
        barWrap.className = 'flex-grow-1 mx-2';
        var progress = document.createElement('div');
        progress.className = 'progress';
        progress.style.height = '16px';
        progress.style.borderRadius = '8px';
        var barEl = document.createElement('div');
        barEl.className = 'progress-bar ' + barColor;
        barEl.style.width = pct + '%';
        barEl.style.borderRadius = '8px';
        barEl.style.fontSize = '0.65rem';
        barEl.textContent = pct + '%';
        progress.appendChild(barEl);
        barWrap.appendChild(progress);

        row.appendChild(label);
        row.appendChild(barWrap);
        container.appendChild(row);
    }
}

function renderPrivacyStats(stats) {
    if (!stats) return;
    var el;
    el = document.getElementById('learning-db-size');
    if (el) el.textContent = (stats.db_size_kb || 0) + ' KB';
    el = document.getElementById('learning-pattern-count');
    if (el) el.textContent = stats.transferable_patterns || 0;
    el = document.getElementById('learning-model-count');
    if (el) el.textContent = stats.models_trained || 0;
    el = document.getElementById('learning-source-count');
    if (el) el.textContent = stats.tracked_sources || 0;
}

async function resetLearning() {
    if (!confirm('Delete all learning data? Memories will be preserved.')) return;
    try {
        var response = await fetch('/api/learning/reset', { method: 'POST' });
        var data = await response.json();
        if (data.success) { alert('Learning data reset. Memories preserved.'); loadLearning(); }
        else alert('Reset failed: ' + (data.error || 'Unknown error'));
    } catch (error) { alert('Reset failed: ' + error.message); }
}

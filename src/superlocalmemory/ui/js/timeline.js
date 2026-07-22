// SuperLocalMemory V2 - Timeline View (D3.js bar chart)
// Depends on: core.js

async function loadTimeline() {
    showLoading('timeline-chart', 'Loading timeline...');
    try {
        var response = await fetch('/api/v3/timeline/?range=30d&group_by=date&limit=1000');
        if (!response.ok) throw new Error('HTTP ' + response.status);
        var data = await response.json();
        renderTimeline(data.events || data.timeline);
    } catch (error) {
        console.error('Error loading timeline:', error);
        showEmpty('timeline-chart', 'clock-history', 'Failed to load timeline');
    }
}

// DASH-V5 (3.7.9): the /timeline endpoint returns individual events (with a
// `timestamp` but no `count`/`date`), while this chart needs per-day buckets.
// Reading a non-existent `count` produced a [0, NaN] y-domain and 300+ rects
// with height="NaN". Aggregate events into per-day counts client-side (and
// still accept a pre-aggregated {date,count} shape if the API ever provides it).
function _bucketTimeline(timeline) {
    var first = timeline[0] || {};
    if (first.count !== undefined && (first.date || first.period)) {
        return timeline
            .map(function(d) { return { date: d.date || d.period, count: +d.count || 0 }; })
            .filter(function(d) { return d.date; });
    }
    var byDay = {};
    timeline.forEach(function(e) {
        var ts = e.timestamp || e.date || e.period || '';
        var day = String(ts).slice(0, 10);
        if (day) byDay[day] = (byDay[day] || 0) + 1;
    });
    return Object.keys(byDay).sort().map(function(day) {
        return { date: day, count: byDay[day] };
    });
}

function renderTimeline(timeline) {
    var container = document.getElementById('timeline-chart');
    if (!timeline || timeline.length === 0) {
        showEmpty('timeline-chart', 'clock-history', 'No timeline data for the last 30 days.');
        return;
    }
    var buckets = _bucketTimeline(timeline);
    if (buckets.length === 0) {
        showEmpty('timeline-chart', 'clock-history', 'No timeline data for the last 30 days.');
        return;
    }
    var margin = { top: 20, right: 20, bottom: 50, left: 50 };
    var width = Math.max(10, (container.clientWidth || 600) - margin.left - margin.right);
    var height = 300 - margin.top - margin.bottom;
    container.textContent = '';
    var svg = d3.select('#timeline-chart').append('svg').attr('width', width + margin.left + margin.right).attr('height', height + margin.top + margin.bottom).append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
    var x = d3.scaleBand().range([0, width]).domain(buckets.map(function(d) { return d.date; })).padding(0.1);
    var maxCount = d3.max(buckets, function(d) { return d.count; }) || 1;
    var y = d3.scaleLinear().range([height, 0]).domain([0, maxCount]);
    svg.append('g').attr('transform', 'translate(0,' + height + ')').call(d3.axisBottom(x)).selectAll('text').attr('transform', 'rotate(-45)').style('text-anchor', 'end');
    svg.append('g').call(d3.axisLeft(y).ticks(Math.min(maxCount, 8)));
    svg.selectAll('.bar').data(buckets).enter().append('rect').attr('class', 'bar').attr('x', function(d) { return x(d.date); }).attr('y', function(d) { return y(d.count); }).attr('width', x.bandwidth()).attr('height', function(d) { return Math.max(0, height - y(d.count)); }).attr('fill', '#667eea').attr('rx', 3);
}

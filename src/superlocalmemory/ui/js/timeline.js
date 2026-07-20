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

function renderTimeline(timeline) {
    var container = document.getElementById('timeline-chart');
    if (!timeline || timeline.length === 0) {
        showEmpty('timeline-chart', 'clock-history', 'No timeline data for the last 30 days.');
        return;
    }
    var margin = { top: 20, right: 20, bottom: 50, left: 50 };
    var width = container.clientWidth - margin.left - margin.right;
    var height = 300 - margin.top - margin.bottom;
    container.textContent = '';
    var svg = d3.select('#timeline-chart').append('svg').attr('width', width + margin.left + margin.right).attr('height', height + margin.top + margin.bottom).append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
    var x = d3.scaleBand().range([0, width]).domain(timeline.map(function(d) { return d.date || d.period; })).padding(0.1);
    var y = d3.scaleLinear().range([height, 0]).domain([0, d3.max(timeline, function(d) { return d.count; })]);
    svg.append('g').attr('transform', 'translate(0,' + height + ')').call(d3.axisBottom(x)).selectAll('text').attr('transform', 'rotate(-45)').style('text-anchor', 'end');
    svg.append('g').call(d3.axisLeft(y));
    svg.selectAll('.bar').data(timeline).enter().append('rect').attr('class', 'bar').attr('x', function(d) { return x(d.date || d.period); }).attr('y', function(d) { return y(d.count); }).attr('width', x.bandwidth()).attr('height', function(d) { return height - y(d.count); }).attr('fill', '#667eea').attr('rx', 3);
}

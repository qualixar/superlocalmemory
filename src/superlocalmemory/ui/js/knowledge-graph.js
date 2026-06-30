// SuperLocalMemory v3.4.1 — Sigma.js WebGL Knowledge Graph
// Copyright (c) 2026 Varun Pratap Bhardwaj — AGPL-3.0-or-later
// Replaces Cytoscape.js as default renderer. Cytoscape kept as fallback toggle.

// ============================================================================
// GLOBAL STATE (mirrors graph-core.js contract for interoperability)
// ============================================================================

var sigmaInstance = null;
var sigmaGraph = null;          // graphology Graph instance
var sigmaState = {
    hoveredNode: null,
    selectedNode: null,
    searchQuery: '',
    suggestions: new Set(),
    highlightedNodes: new Set(), // For event bus highlights (Phase 3)
};

// Define shared globals (previously in graph-core.js, removed in v3.4.1)
if (typeof graphData === 'undefined') var graphData = { nodes: [], links: [] };
if (typeof originalGraphData === 'undefined') var originalGraphData = { nodes: [], links: [] };
if (typeof filterState === 'undefined') var filterState = { cluster_id: null, entity: null };
if (typeof isInitialLoad === 'undefined') var isInitialLoad = true;

// ============================================================================
// CLUSTER COLORS (same palette as graph-core.js for consistency)
// ============================================================================

const SIGMA_CLUSTER_COLORS = [
    '#667eea', '#764ba2', '#43e97b', '#38f9d7',
    '#4facfe', '#00f2fe', '#f093fb', '#f5576c',
    '#fa709a', '#fee140', '#30cfd0', '#330867'
];

function getSigmaClusterColor(communityId) {
    if (communityId === null || communityId === undefined || communityId === 0) return '#999999';
    return SIGMA_CLUSTER_COLORS[Math.abs(communityId) % SIGMA_CLUSTER_COLORS.length];
}

// Color by fact_type when communities are not available
var CATEGORY_COLORS = {
    'semantic':  '#667eea',  // Indigo — knowledge facts
    'episodic':  '#43e97b',  // Green — session events
    'opinion':   '#f093fb',  // Pink — decisions & opinions
    'temporal':  '#4facfe',  // Blue — time-referenced facts
};

function getNodeColor(node) {
    // Priority 1: community_id from graph intelligence
    if (node.community_id && node.community_id !== 0) {
        return getSigmaClusterColor(node.community_id);
    }
    // Priority 2: fact_type category coloring
    if (node.category && CATEGORY_COLORS[node.category]) {
        return CATEGORY_COLORS[node.category];
    }
    return '#667eea'; // Default indigo
}

// ============================================================================
// RENDERER CHECK — Only activate if user chose Sigma
// ============================================================================

function isSigmaRenderer() {
    // v3.4.1: Sigma.js is the ONLY renderer. Cytoscape removed.
    return true;
}

// ============================================================================
// GRAPH DATA TRANSFORMATION (API response → graphology format)
// ============================================================================

function transformDataForSigma(data) {
    // Import graphology from ESM module (loaded in index.html)
    if (typeof graphology === 'undefined') {
        console.error('[Sigma] graphology not loaded');
        return null;
    }
    var graph = new graphology.Graph({ multi: false, type: 'undirected' });

    var nodes = data.nodes || [];
    var links = data.links || [];
    var nodeCount = nodes.length;

    // Compute degree for sizing
    var degreeMap = {};
    links.forEach(function(link) {
        var s = String(link.source);
        var t = String(link.target);
        degreeMap[s] = (degreeMap[s] || 0) + 1;
        degreeMap[t] = (degreeMap[t] || 0) + 1;
    });

    // Add nodes with random initial positions (ForceAtlas2 will refine)
    var spread = Math.sqrt(nodeCount) * 40;
    nodes.forEach(function(node, i) {
        var id = String(node.id);
        var degree = degreeMap[id] || 0;
        var importance = node.importance || 0.5;
        var communityId = node.community_id || 0;

        // Golden angle distribution for initial positions
        var angle = i * (Math.PI * (3 - Math.sqrt(5)));
        var radius = spread * Math.sqrt((i + 1) / nodeCount);

        // Node size: blend of degree and importance
        var size = Math.max(3, Math.min(20, 3 + degree * 1.5 + importance * 8));

        // Label: first 4 words of content
        var contentPreview = node.content_preview || node.content || '';
        var label = contentPreview.split(/\s+/).slice(0, 4).join(' ') || node.category || 'Memory';

        try {
            graph.addNode(id, {
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius,
                size: size,
                color: getNodeColor(node),
                label: label,
                // SLM-specific data (for detail panel)
                slm_content: node.content || '',
                slm_content_preview: contentPreview,
                slm_category: node.category || '',
                slm_project_name: node.project_name || '',
                slm_importance: importance,
                slm_community_id: communityId,
                slm_pagerank: node.pagerank_score || 0,
                slm_degree_centrality: node.degree_centrality || 0,
                slm_created_at: node.created_at || '',
                slm_entities: node.entities || [],
            });
        } catch (e) {
            // Skip duplicate node IDs silently
            if (!e.message.includes('already exist')) {
                console.warn('[Sigma] Node add error:', e.message);
            }
        }
    });

    // Add edges
    links.forEach(function(link) {
        var sourceId = String(link.source);
        var targetId = String(link.target);
        try {
            if (graph.hasNode(sourceId) && graph.hasNode(targetId)) {
                graph.addEdge(sourceId, targetId, {
                    weight: link.weight || 0.5,
                    color: getEdgeColor(link.relationship_type),
                    size: Math.max(0.5, (link.weight || 0.5) * 2),
                    slm_type: link.relationship_type || 'entity',
                });
            }
        } catch (e) {
            // Skip duplicate edges silently
        }
    });

    return graph;
}

function getEdgeColor(type) {
    var colors = {
        'entity': '#cccccc',
        'temporal': '#4facfe',
        'semantic': '#667eea',
        'causal': '#43e97b',
        'contradiction': '#f5576c',
        'supersedes': '#f093fb',
    };
    return colors[type] || '#cccccc';
}

// ============================================================================
// LAYOUT — ForceAtlas2 (synchronous for <500 nodes, batched for larger)
// ============================================================================

function runSigmaLayout(graph) {
    if (typeof graphologyLibrary === 'undefined') {
        console.warn('[Sigma] graphologyLibrary not loaded, skipping layout');
        return;
    }
    var nodeCount = graph.order;
    var settings = graphologyLibrary.layoutForceAtlas2.inferSettings(graph);
    settings.barnesHutOptimize = nodeCount > 500;
    settings.slowDown = nodeCount > 1000 ? 5 : 1;

    var iterations = nodeCount > 5000 ? 30 : nodeCount > 2000 ? 50 : nodeCount > 500 ? 100 : 200;
    graphologyLibrary.layoutForceAtlas2.assign(graph, {
        iterations: iterations,
        settings: settings,
    });
    console.log('[Sigma] ForceAtlas2 done:', nodeCount, 'nodes,', iterations, 'iterations');
}

// ============================================================================
// RENDER — Main entry point for Sigma.js graph
// ============================================================================

function renderSigmaGraph(data) {
    if (typeof Sigma === 'undefined') {
        console.error('[Sigma] Sigma.js not loaded — check CDN');
        return;
    }

    var container = document.getElementById('graph-container');
    if (!container) return;

    // CRITICAL: Don't render if container is hidden (Bootstrap tab not visible)
    // Sigma.js needs real pixel dimensions to create WebGL canvases.
    if (container.offsetWidth === 0 || container.offsetHeight === 0) {
        console.log('[Sigma] Container hidden (0 dimensions), deferring render');
        // Store data for deferred render when tab becomes visible
        window._sigmaPendingData = data;
        return;
    }

    // Destroy previous instance
    if (sigmaInstance) {
        try { sigmaInstance.kill(); } catch (e) { /* ok */ }
        sigmaInstance = null;
        sigmaGraph = null;
    }

    // Clear container
    container.textContent = '';

    var nodes = data.nodes || [];
    if (nodes.length === 0) {
        var emptyMsg = document.createElement('div');
        emptyMsg.style.cssText = 'text-align:center; padding:50px; color:#666;';
        emptyMsg.textContent = 'No memories found. Try adjusting filters.';
        container.appendChild(emptyMsg);
        return;
    }

    // Transform and layout
    sigmaGraph = transformDataForSigma(data);
    if (!sigmaGraph) return;

    runSigmaLayout(sigmaGraph);

    // Create Sigma renderer
    // allowInvalidContainer: Bootstrap tabs have display:none on inactive panes,
    // so the container has 0 width on first render. Sigma will resize on refresh().
    sigmaInstance = new Sigma(sigmaGraph, container, {
        allowInvalidContainer: true,
        // Rendering
        renderLabels: true,
        labelDensity: 0.5,
        labelRenderedSizeThreshold: 8,
        labelFont: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        labelSize: 11,
        labelColor: { color: '#333333' },
        // Performance — scale with node count
        hideEdgesOnMove: nodes.length > 300,
        hideLabelsOnMove: nodes.length > 1000,
        enableEdgeEvents: false,
        labelGridCellSize: nodes.length > 1000 ? 200 : 100,
        // Appearance
        defaultNodeColor: '#999999',
        defaultEdgeColor: '#cccccc',
        stagePadding: 40,
    });

    // Force resize + camera fit after Bootstrap tab transition completes
    requestAnimationFrame(function() {
        setTimeout(function() {
            if (sigmaInstance) {
                sigmaInstance.refresh();
                // Auto-fit camera to show all nodes in viewport
                sigmaInstance.getCamera().animatedReset({ duration: 300 });
            }
        }, 200);
    });

    // Node/Edge reducers for hover + search + community highlighting
    sigmaInstance.setSetting('nodeReducer', function(node, data) {
        var res = Object.assign({}, data);
        var state = sigmaState;

        // v3.4.1: Apply frontend Louvain community color when in live mode
        if (communitySource === 'live' && sigmaGraph.hasNode(node)) {
            var louvainComm = sigmaGraph.getNodeAttribute(node, 'community');
            if (louvainComm !== undefined && louvainComm !== null) {
                res.color = SIGMA_CLUSTER_COLORS[louvainComm % SIGMA_CLUSTER_COLORS.length];
            }
        }

        // Search highlighting
        if (state.searchQuery && state.suggestions.size > 0) {
            if (!state.suggestions.has(node)) {
                res.color = '#e0e0e0';
                res.label = '';
                res.zIndex = 0;
            } else {
                res.highlighted = true;
                res.zIndex = 1;
            }
        }

        // Hover highlighting
        if (state.hoveredNode) {
            if (node === state.hoveredNode) {
                res.highlighted = true;
                res.zIndex = 2;
            } else if (sigmaGraph.hasNode(state.hoveredNode) && sigmaGraph.neighbors(state.hoveredNode).indexOf(node) !== -1) {
                res.highlighted = true;
                res.zIndex = 1;
            } else if (!state.searchQuery) {
                res.color = '#e0e0e0';
                res.label = '';
                res.zIndex = 0;
            }
        }

        // Selected node
        if (state.selectedNode === node) {
            res.highlighted = true;
            res.zIndex = 3;
        }

        // Event bus highlights (Phase 3)
        if (state.highlightedNodes.has(node)) {
            res.highlighted = true;
            res.color = '#ff6b6b';
            res.zIndex = 2;
        }

        return res;
    });

    sigmaInstance.setSetting('edgeReducer', function(edge, data) {
        var res = Object.assign({}, data);
        var state = sigmaState;

        if (state.hoveredNode && sigmaGraph.hasNode(state.hoveredNode)) {
            var extremities = sigmaGraph.extremities(edge);
            if (extremities.indexOf(state.hoveredNode) === -1) {
                res.hidden = true;
            } else {
                res.color = '#667eea';
                res.size = Math.max(res.size, 2);
            }
        }

        if (state.searchQuery && state.suggestions.size > 0) {
            var ext = sigmaGraph.extremities(edge);
            if (!state.suggestions.has(ext[0]) && !state.suggestions.has(ext[1])) {
                res.hidden = true;
            }
        }

        return res;
    });

    // Event handlers
    sigmaInstance.on('enterNode', function(event) {
        sigmaState.hoveredNode = event.node;
        sigmaInstance.refresh();
        showSigmaTooltip(event.node, event.event);
    });

    sigmaInstance.on('leaveNode', function() {
        sigmaState.hoveredNode = null;
        sigmaInstance.refresh();
        hideSigmaTooltip();
    });

    sigmaInstance.on('clickNode', function(event) {
        sigmaState.selectedNode = event.node;
        sigmaInstance.refresh();
        openSigmaNodeDetail(event.node);
    });

    sigmaInstance.on('doubleClickNode', function(event) {
        // Double-click → ask chat about this node
        var attrs = sigmaGraph.getNodeAttributes(event.node);
        var label = attrs.label || '';
        if (window.SLMEventBus && label) {
            SLMEventBus.publish('slm:chat:queryAbout', {
                query: 'Tell me everything about: ' + label,
            });
        }
    });

    sigmaInstance.on('clickStage', function() {
        sigmaState.selectedNode = null;
        sigmaState.highlightedNodes.clear();
        sigmaInstance.refresh();
        hideSigmaTooltip();
    });

    // Update stats
    if (typeof updateGraphStats === 'function') {
        updateGraphStats(data);
    }

    // Update panels
    updateSigmaStatsPanel(data);

    // v3.4.1: Run frontend Louvain by default (Live mode), fallback to backend
    if (communitySource === 'live') {
        var resolution = parseFloat((document.getElementById('community-resolution') || {}).value) || 1.0;
        detectCommunitiesInBrowser(resolution);
    } else {
        loadCommunities();
    }

    console.log('[Sigma] Rendered', sigmaGraph.order, 'nodes,', sigmaGraph.size, 'edges');
}

// ============================================================================
// TOOLTIP
// ============================================================================

function showSigmaTooltip(nodeId, mouseEvent) {
    if (!sigmaGraph || !sigmaGraph.hasNode(nodeId)) return;
    var attrs = sigmaGraph.getNodeAttributes(nodeId);
    var tooltip = document.getElementById('sigma-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'sigma-tooltip';
        tooltip.style.cssText = 'position:fixed;z-index:9999;background:#fff;border:1px solid #dee2e6;'
            + 'border-radius:8px;padding:10px 14px;box-shadow:0 4px 12px rgba(0,0,0,0.15);'
            + 'max-width:300px;pointer-events:none;font-size:13px;';
        document.body.appendChild(tooltip);
    }
    var preview = (attrs.slm_content_preview || '').substring(0, 100);
    tooltip.innerHTML = '<strong>' + escapeHtml(attrs.label || '') + '</strong>'
        + '<div class="text-muted small mt-1">' + escapeHtml(preview) + '</div>'
        + '<div class="mt-1"><span class="badge bg-primary me-1">' + escapeHtml(attrs.slm_category || '') + '</span>'
        + '<span class="badge bg-secondary">Trust: ' + (attrs.slm_importance || 0).toFixed(2) + '</span></div>';
    tooltip.style.display = 'block';

    if (mouseEvent && mouseEvent.original) {
        tooltip.style.left = (mouseEvent.original.clientX + 15) + 'px';
        tooltip.style.top = (mouseEvent.original.clientY + 15) + 'px';
    }
}

function hideSigmaTooltip() {
    var tooltip = document.getElementById('sigma-tooltip');
    if (tooltip) tooltip.style.display = 'none';
}

// ============================================================================
// NODE DETAIL PANEL (reuses existing openMemoryDetail from modal.js)
// ============================================================================

function openSigmaNodeDetail(nodeId) {
    if (!sigmaGraph || !sigmaGraph.hasNode(nodeId)) return;
    var attrs = sigmaGraph.getNodeAttributes(nodeId);

    // Populate right panel instead of opening modal
    var panel = document.getElementById('sigma-detail-content');
    if (panel) {
        var neighbors = sigmaGraph.neighbors(nodeId);
        var neighborList = neighbors.slice(0, 10).map(function(nid) {
            var na = sigmaGraph.getNodeAttributes(nid);
            return '<div class="border-bottom py-1 cursor-pointer" onclick="sigmaHighlightNode(\'' + nid + '\')">'
                + '<small class="text-primary">' + escapeHtml((na.label || '').substring(0, 40)) + '</small>'
                + '</div>';
        }).join('');

        panel.innerHTML = ''
            + '<div class="mb-2">'
            + '<span class="badge" style="background:' + attrs.color + '">' + escapeHtml(attrs.slm_category || 'memory') + '</span>'
            + ' <span class="badge bg-secondary">Trust: ' + (attrs.slm_importance || 0).toFixed(2) + '</span>'
            + '</div>'
            + '<div class="mb-2" style="line-height:1.5;">' + escapeHtml(attrs.slm_content || attrs.slm_content_preview || '') + '</div>'
            + '<div class="text-muted small mb-2">'
            + '<i class="bi bi-clock"></i> ' + escapeHtml(attrs.slm_created_at || 'Unknown')
            + ' &bull; <i class="bi bi-diagram-3"></i> ' + neighbors.length + ' connections'
            + ' &bull; PageRank: ' + (attrs.slm_pagerank || 0).toFixed(4)
            + '</div>'
            + '<hr class="my-2">'
            + '<h6 class="mb-1">Connected (' + neighbors.length + ')</h6>'
            + '<div style="max-height:250px;overflow-y:auto;">' + (neighborList || '<span class="text-muted">No connections</span>') + '</div>'
            + '<hr class="my-2">'
            + '<button class="btn btn-sm btn-outline-primary w-100" onclick="openMemoryDetail({id:\'' + nodeId + '\',content:\'' + escapeHtml((attrs.slm_content || '').substring(0, 80).replace(/'/g, "\\'")) + '\',category:\'' + (attrs.slm_category || '') + '\',importance:' + (attrs.slm_importance || 0.5) + '},\'graph\')"><i class="bi bi-box-arrow-up-right"></i> Full Detail</button>';
    }
}

// Stats panel update
function updateSigmaStatsPanel(data) {
    var panel = document.getElementById('sigma-stats-panel');
    if (!panel) return;
    var nodes = data.nodes || [];
    var links = data.links || [];
    var categories = {};
    nodes.forEach(function(n) {
        var cat = n.category || 'unknown';
        categories[cat] = (categories[cat] || 0) + 1;
    });
    var catHtml = Object.keys(categories).map(function(k) {
        return '<div>' + k + ': <strong>' + categories[k] + '</strong></div>';
    }).join('');
    panel.innerHTML = '<div>Nodes: <strong>' + nodes.length + '</strong></div>'
        + '<div>Edges: <strong>' + links.length + '</strong></div>'
        + '<hr class="my-1">' + catHtml;
}

// Category filter
function sigmaFilterByCategory(category) {
    if (!sigmaGraph || !sigmaInstance) return;
    sigmaState.searchQuery = '';
    sigmaState.suggestions.clear();

    if (category) {
        sigmaGraph.forEachNode(function(nodeId, attrs) {
            if (attrs.slm_category === category) {
                sigmaState.suggestions.add(nodeId);
            }
        });
        sigmaState.searchQuery = '__filter__'; // trigger reducer
    }
    sigmaInstance.refresh();
}

// ============================================================================
// SEARCH (filter nodes by label match)
// ============================================================================

function sigmaSearch(query) {
    if (!sigmaGraph || !sigmaInstance) return;
    sigmaState.searchQuery = query.trim().toLowerCase();
    sigmaState.suggestions.clear();

    if (sigmaState.searchQuery) {
        sigmaGraph.forEachNode(function(nodeId, attrs) {
            var label = (attrs.label || '').toLowerCase();
            var content = (attrs.slm_content_preview || '').toLowerCase();
            if (label.indexOf(sigmaState.searchQuery) !== -1 ||
                content.indexOf(sigmaState.searchQuery) !== -1) {
                sigmaState.suggestions.add(nodeId);
            }
        });
    }

    sigmaInstance.refresh();

    // If exactly one match, focus camera on it
    if (sigmaState.suggestions.size === 1) {
        var matchedNode = sigmaState.suggestions.values().next().value;
        var pos = sigmaGraph.getNodeAttributes(matchedNode);
        sigmaInstance.getCamera().animate({ x: pos.x, y: pos.y, ratio: 0.3 }, { duration: 500 });
    }

    return sigmaState.suggestions.size;
}

function sigmaClearSearch() {
    sigmaState.searchQuery = '';
    sigmaState.suggestions.clear();
    if (sigmaInstance) sigmaInstance.refresh();
}

// ============================================================================
// CAMERA CONTROLS
// ============================================================================

function sigmaZoomIn() {
    if (sigmaInstance) sigmaInstance.getCamera().animatedZoom({ duration: 300 });
}

function sigmaZoomOut() {
    if (sigmaInstance) sigmaInstance.getCamera().animatedUnzoom({ duration: 300 });
}

function sigmaResetView() {
    if (sigmaInstance) sigmaInstance.getCamera().animatedReset({ duration: 500 });
}

// ============================================================================
// HIGHLIGHT NODE (for event bus — Phase 3)
// ============================================================================

function sigmaHighlightNode(factId) {
    if (!sigmaGraph || !sigmaInstance) return;
    var nodeId = String(factId);
    if (!sigmaGraph.hasNode(nodeId)) return;

    sigmaState.highlightedNodes.clear();
    sigmaState.highlightedNodes.add(nodeId);
    sigmaInstance.refresh();

    // Focus camera
    var pos = sigmaGraph.getNodeAttributes(nodeId);
    sigmaInstance.getCamera().animate({ x: pos.x, y: pos.y, ratio: 0.4 }, { duration: 400 });
}

// ============================================================================
// INTEGRATION: Override loadGraph() to route to Sigma when active
// ============================================================================

// Store original loadGraph (from graph-core.js) before overriding
var _originalLoadGraph = (typeof loadGraph === 'function') ? loadGraph : null;

// This runs AFTER graph-core.js is loaded (script order in index.html)
function loadGraphSigma() {
    if (!isSigmaRenderer()) {
        // Delegate to Cytoscape
        if (_originalLoadGraph) _originalLoadGraph();
        return;
    }

    var maxNodes = 100;
    var maxNodesEl = document.getElementById('graph-max-nodes');
    if (maxNodesEl) maxNodes = parseInt(maxNodesEl.value) || 100;

    var minImportance = 1;
    var minImpEl = document.getElementById('graph-min-importance');
    if (minImpEl) minImportance = parseInt(minImpEl.value) || 1;

    // Apply cluster filter for larger fetch
    var fetchLimit = (typeof filterState !== 'undefined' && filterState.cluster_id) ? 200 : maxNodes;

    if (typeof showLoadingSpinner === 'function') showLoadingSpinner();

    fetch('/api/graph?max_nodes=' + fetchLimit + '&min_importance=' + minImportance)
        .then(function(r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        })
        .then(function(data) {
            // Store in shared globals
            if (typeof window.graphData !== 'undefined') window.graphData = data;
            if (typeof window.originalGraphData !== 'undefined') {
                window.originalGraphData = JSON.parse(JSON.stringify(data));
            }

            // Apply filters if set
            if (typeof filterState !== 'undefined') {
                if (filterState.cluster_id && typeof filterByCluster === 'function') {
                    data = filterByCluster(window.originalGraphData || data, filterState.cluster_id);
                }
                if (filterState.entity && typeof filterByEntity === 'function') {
                    data = filterByEntity(window.originalGraphData || data, filterState.entity);
                }
            }

            renderSigmaGraph(data);
            if (typeof hideLoadingSpinner === 'function') hideLoadingSpinner();
            if (typeof updateFilterBadge === 'function') updateFilterBadge();
        })
        .catch(function(error) {
            console.error('[Sigma] Load error:', error);
            if (typeof showError === 'function') showError('Failed to load graph: ' + error.message);
            if (typeof hideLoadingSpinner === 'function') hideLoadingSpinner();
        });
}

// ============================================================================
// RENDERER TOGGLE
// ============================================================================

// ============================================================================
// COMMUNITY DETECTION (v3.4.1: Frontend Louvain + Backend Leiden/LP)
// ============================================================================

var communitySource = 'live'; // 'live' (frontend Louvain) or 'backend' (API)
var communityResolutionTimer = null;
var liveCommunityMap = null; // { communityId: [nodeIds] }

// ── Frontend Louvain (in-browser, real-time) ─────────────────────

function detectCommunitiesInBrowser(resolution) {
    if (!sigmaGraph || sigmaGraph.order === 0) return;
    resolution = resolution || 1.0;

    var t0 = performance.now();
    try {
        // graphology-library UMD exposes community methods
        var louvainFn = null;
        if (window.graphologyLibrary) {
            var cl = window.graphologyLibrary.communitiesLouvain;
            louvainFn = cl ? (cl.assign || cl) : null;
        }
        if (!louvainFn) {
            console.warn('[knowledge-graph] graphology-communities-louvain not available, using backend communities');
            handleCommunitySourceToggle('backend');
            return;
        }

        // Louvain assign mutates the graph, adding 'community' attribute
        louvainFn(sigmaGraph, { resolution: resolution });

        // Build community map for panel rendering
        liveCommunityMap = {};
        sigmaGraph.forEachNode(function(nodeId, attrs) {
            var commId = attrs.community;
            if (commId === undefined || commId === null) commId = 0;
            if (!liveCommunityMap[commId]) liveCommunityMap[commId] = [];
            liveCommunityMap[commId].push(nodeId);
        });

        // Refresh Sigma to pick up community colors via nodeReducer
        if (sigmaInstance) sigmaInstance.refresh();

        var dt = Math.round(performance.now() - t0);
        var commCount = Object.keys(liveCommunityMap).length;
        console.log('[knowledge-graph] Frontend Louvain: ' + commCount + ' communities at resolution ' + resolution.toFixed(1) + ' (' + dt + 'ms)');

        // Render the community list panel
        renderLiveCommunityPanel(liveCommunityMap);

        // Warn about edge cases
        var nodeCount = sigmaGraph.order;
        if (commCount === 1) {
            showToast('All nodes in one community. Try increasing resolution.', 'info');
        } else if (commCount >= nodeCount * 0.9) {
            showToast('Nearly every node is its own community. Lower resolution.', 'warning');
        } else if (commCount > 100) {
            showToast('Very fine resolution: ' + commCount + ' communities.', 'info');
        }
    } catch (e) {
        console.warn('[knowledge-graph] Frontend Louvain failed, falling back to backend:', e.message);
        handleCommunitySourceToggle('backend');
    }
}

function renderLiveCommunityPanel(communityMap) {
    var panel = document.getElementById('community-list-panel');
    if (!panel || !communityMap) return;

    var entries = Object.keys(communityMap).map(function(cid) {
        return { community_id: parseInt(cid, 10), members: communityMap[cid] };
    }).sort(function(a, b) { return b.members.length - a.members.length; });

    if (entries.length === 0) {
        panel.innerHTML = '<div class="text-muted small">No communities detected.</div>';
        return;
    }

    var html = '';
    entries.forEach(function(c) {
        var color = SIGMA_CLUSTER_COLORS[c.community_id % SIGMA_CLUSTER_COLORS.length];
        // Generate simple label from most common entity
        var label = generateLiveLabel(c.community_id, c.members);
        html += '<div class="d-flex align-items-center mb-1 cursor-pointer" '
            + 'onclick="sigmaFilterByCommunity(' + c.community_id + ', \'' + communitySource + '\')" '
            + 'title="' + c.members.length + ' memories">'
            + '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + color + ';flex-shrink:0;"></span>'
            + '<span class="ms-1 text-truncate" style="font-size:0.75rem;">' + escapeHtml(label) + '</span>'
            + '<span class="badge bg-light text-dark ms-auto" style="font-size:0.65rem;">' + c.members.length + '</span>'
            + '</div>';
    });
    html += '<button class="btn btn-sm btn-outline-secondary w-100 mt-1" onclick="sigmaClearSearch()">'
        + '<i class="bi bi-x-circle"></i> Clear Filter</button>';
    panel.innerHTML = html;
}

function generateLiveLabel(communityId, memberNodeIds) {
    // Use most common entity names from node attributes
    if (!sigmaGraph || !memberNodeIds || memberNodeIds.length === 0) return 'Community ' + communityId;
    var entityFreq = {};
    memberNodeIds.slice(0, 20).forEach(function(nid) {
        try {
            var entities = sigmaGraph.getNodeAttribute(nid, 'slm_entities');
            if (entities && Array.isArray(entities)) {
                entities.forEach(function(e) {
                    entityFreq[e] = (entityFreq[e] || 0) + 1;
                });
            }
        } catch (_) { /* skip */ }
    });
    var sorted = Object.keys(entityFreq).sort(function(a, b) { return entityFreq[b] - entityFreq[a]; });
    return sorted.length > 0 ? sorted.slice(0, 3).join(', ') : 'Community ' + communityId;
}

// ── Backend Communities (API, consolidated) ──────────────────────

function loadCommunities() {
    fetch('/api/v3/graph/communities')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var panel = document.getElementById('community-list-panel');
            if (!panel) return;

            var communities = data.communities || [];
            if (communities.length === 0) {
                panel.innerHTML = '<div class="text-muted small">No communities detected yet.</div>'
                    + '<button class="btn btn-sm btn-outline-primary w-100 mt-1" onclick="runCommunityDetection()">'
                    + '<i class="bi bi-cpu"></i> Detect Communities</button>';
                return;
            }

            var html = '';
            communities.forEach(function(c) {
                var color = c.color || SIGMA_CLUSTER_COLORS[c.community_id % SIGMA_CLUSTER_COLORS.length];
                html += '<div class="d-flex align-items-center mb-1 cursor-pointer" '
                    + 'onclick="sigmaFilterByCommunity(' + c.community_id + ', \'backend\')" '
                    + 'title="' + (c.top_entities || []).join(', ') + '">'
                    + '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + color + ';flex-shrink:0;"></span>'
                    + '<span class="ms-1 text-truncate" style="font-size:0.75rem;">' + escapeHtml(c.label) + '</span>'
                    + '<span class="badge bg-light text-dark ms-auto" style="font-size:0.65rem;">' + c.member_count + '</span>'
                    + '</div>';
            });
            html += '<button class="btn btn-sm btn-outline-secondary w-100 mt-1" onclick="sigmaClearSearch()">'
                + '<i class="bi bi-x-circle"></i> Clear Filter</button>';
            html += '<button class="btn btn-sm btn-outline-primary w-100 mt-1" onclick="runCommunityDetection()">'
                + '<i class="bi bi-arrow-clockwise"></i> Refresh</button>';
            panel.innerHTML = html;
        })
        .catch(function() { /* silent */ });
}

function runCommunityDetection() {
    var panel = document.getElementById('community-list-panel');
    if (panel) panel.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm text-primary"></div> Detecting...</div>';

    fetch('/api/v3/graph/run-communities', { method: 'POST' })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.success) {
                loadCommunities();
                loadGraphSigma();
            } else {
                if (panel) panel.innerHTML = '<div class="text-danger small">Failed: ' + (data.error || 'Unknown') + '</div>';
            }
        })
        .catch(function(e) {
            if (panel) panel.innerHTML = '<div class="text-danger small">Error: ' + e.message + '</div>';
        });
}

// ── Source Toggle + Resolution Slider ────────────────────────────

function handleCommunitySourceToggle(source) {
    communitySource = source;
    var slider = document.getElementById('community-resolution-container');

    if (source === 'live') {
        if (slider) slider.style.display = 'block';
        // Set radio button state
        var liveRadio = document.getElementById('community-live');
        if (liveRadio) liveRadio.checked = true;
        var resolution = parseFloat(document.getElementById('community-resolution').value) || 1.0;
        detectCommunitiesInBrowser(resolution);
    } else {
        if (slider) slider.style.display = 'none';
        var backendRadio = document.getElementById('community-backend');
        if (backendRadio) backendRadio.checked = true;
        loadCommunities();
    }
}

function handleResolutionChange(value) {
    var resolution = parseFloat(value);
    var display = document.getElementById('community-resolution-value');
    if (display) display.textContent = resolution.toFixed(1);

    // Debounce 300ms
    if (communityResolutionTimer) clearTimeout(communityResolutionTimer);
    communityResolutionTimer = setTimeout(function() {
        detectCommunitiesInBrowser(resolution);
    }, 300);
}

// ── Community Filter ─────────────────────────────────────────────

function sigmaFilterByCommunity(communityId, source) {
    if (!sigmaGraph || !sigmaInstance) return;
    sigmaState.searchQuery = '__community__';
    sigmaState.suggestions.clear();

    if (source === 'live' && liveCommunityMap && liveCommunityMap[communityId]) {
        // Filter by frontend Louvain assignment
        liveCommunityMap[communityId].forEach(function(nid) {
            sigmaState.suggestions.add(nid);
        });
    } else {
        // Filter by backend community_id stored in node attributes
        sigmaGraph.forEachNode(function(nodeId, attrs) {
            if (attrs.slm_community_id === communityId) {
                sigmaState.suggestions.add(nodeId);
            }
        });
    }

    sigmaInstance.refresh();
}

function updateRendererUI() {
    // v3.4.1: Sigma.js is the only renderer. Panels always visible.
    var engineName = document.getElementById('graph-engine-name');
    if (engineName) engineName.textContent = 'Sigma.js WebGL';
}

// ============================================================================
// INIT — Override loadGraph + set up UI on page load
// ============================================================================

// Set global functions — these are called by modal.js, clusters.js, core.js
if (typeof window !== 'undefined') {
    window.loadGraph = loadGraphSigma;
    window.renderGraph = renderSigmaGraph; // modal.js/clusters.js call renderGraph(data)
}

// escapeHtml — needed by tooltip and detail panel (was in graph-core.js)
if (typeof escapeHtml === 'undefined') {
    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }
    window.escapeHtml = escapeHtml;
}

// getClusterColor — called by graph-ui.js for badge colors
if (typeof getClusterColor === 'undefined') {
    window.getClusterColor = getSigmaClusterColor;
}

// Initialize renderer UI on DOM ready + auto-scroll on tab switch
document.addEventListener('DOMContentLoaded', function() {
    updateRendererUI();

    // When Knowledge Graph tab becomes visible, render or refresh Sigma
    var graphTab = document.getElementById('graph-tab');
    if (graphTab) {
        graphTab.addEventListener('shown.bs.tab', function() {
            // Tab is now visible — container has real dimensions
            setTimeout(function() {
                if (sigmaInstance) {
                    // Already rendered — just refresh + center
                    sigmaInstance.refresh();
                    sigmaInstance.getCamera().animatedReset({ duration: 300 });
                } else if (window._sigmaPendingData) {
                    // Deferred render — data was fetched while tab was hidden
                    console.log('[Sigma] Rendering deferred data');
                    renderSigmaGraph(window._sigmaPendingData);
                    window._sigmaPendingData = null;
                } else {
                    // First time — trigger load
                    loadGraphSigma();
                }
                var container = document.getElementById('graph-container');
                if (container) {
                    container.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100); // Wait for Bootstrap transition to complete
        });
    }
});

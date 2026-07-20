// SuperLocalMemory V2.6.5 - Interactive Knowledge Graph - Filtering Module
// Copyright (c) 2026 Varun Pratap Bhardwaj — GNU Affero General Public License v3.0 or later
// Part of modular graph visualization system (split from monolithic graph-cytoscape.js)

// ============================================================================
// FILTER LOGIC
// ============================================================================

function filterByCluster(data, cluster_id) {
    console.log('[filterByCluster] Filtering for cluster_id:', cluster_id, 'Type:', typeof cluster_id);
    console.log('[filterByCluster] Total nodes in data:', data.nodes.length);

    // Convert to integer for comparison
    const targetClusterId = parseInt(cluster_id);
    let debugCount = 0;

    const filteredNodes = data.nodes.filter(n => {
        // Convert node cluster_id to integer for comparison
        const nodeClusterId = n.cluster_id ? parseInt(n.cluster_id) : null;

        // Log first 3 nodes to debug
        if (debugCount < 3) {
            console.log('[filterByCluster] Sample node:', {
                id: n.id,
                cluster_id: n.cluster_id,
                type: typeof n.cluster_id,
                parsed: nodeClusterId,
                match: nodeClusterId === targetClusterId
            });
            debugCount++;
        }

        // Use strict equality after type conversion
        return nodeClusterId === targetClusterId;
    });

    console.log('[filterByCluster] Filtered to', filteredNodes.length, 'nodes');

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = data.links.filter(l =>
        nodeIds.has(l.source) && nodeIds.has(l.target)
    );

    console.log('[filterByCluster] Found', filteredLinks.length, 'edges between filtered nodes');

    return { nodes: filteredNodes, links: filteredLinks, clusters: data.clusters };
}

function filterByEntity(data, entity) {
    const filteredNodes = data.nodes.filter(n => {
        if (n.entities && Array.isArray(n.entities)) {
            return n.entities.includes(entity);
        }
        if (n.tags && n.tags.includes(entity)) {
            return true;
        }
        return false;
    });
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = data.links.filter(l =>
        nodeIds.has(l.source) && nodeIds.has(l.target)
    );
    return { nodes: filteredNodes, links: filteredLinks, clusters: data.clusters };
}

function clearGraphFilters() {
    console.log('[clearGraphFilters] Clearing all filters and reloading full graph');

    // Clear filter state
    filterState = { cluster_id: null, entity: null };

    // CRITICAL: Clear URL completely - remove ?cluster_id=X from address bar
    const cleanUrl = window.location.origin + window.location.pathname;
    window.history.replaceState({}, '', cleanUrl);
    console.log('[clearGraphFilters] URL cleaned to:', cleanUrl);

    // CRITICAL: Clear saved layout positions so nodes don't stay in corner
    // When switching from filtered to full graph, we want fresh layout
    localStorage.removeItem('slm_graph_layout');
    console.log('[clearGraphFilters] Cleared saved layout positions');

    // Reload full graph from API (respects dropdown settings)
    loadGraph();
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

function setupGraphEventListeners() {
    // Load graph when Graph tab is clicked
    const graphTab = document.querySelector('a[href="#graph"]');
    if (graphTab) {
        // CRITICAL FIX: Add click handler to clear filter even when already on KG tab
        // Bootstrap's shown.bs.tab only fires when SWITCHING TO tab, not when clicking same tab!
        graphTab.addEventListener('click', function(event) {
            console.log('[Event] Knowledge Graph tab CLICKED');

            // Check if we're viewing a filtered graph
            const hasFilter = filterState.cluster_id || filterState.entity;

            if (hasFilter && cy) {
                console.log('[Event] Click detected on KG tab while filter active - clearing filter');

                // Clear filter state
                filterState = { cluster_id: null, entity: null };

                // Clear URL completely
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, '', cleanUrl);
                console.log('[Event] URL cleaned to:', cleanUrl);

                // CRITICAL: Clear saved layout positions so nodes don't stay in corner
                localStorage.removeItem('slm_graph_layout');
                console.log('[Event] Cleared saved layout positions');

                // Reload with full graph
                loadGraph();
            }
        });

        // Also keep shown.bs.tab for when switching FROM another tab TO KG tab
        graphTab.addEventListener('shown.bs.tab', function(event) {
            console.log('[Event] Knowledge Graph tab SHOWN (tab switch)');

            if (cy || sigmaInstance) {
                // Graph already exists - user is returning to KG tab from another tab
                // Clear filter and reload to show full graph
                console.log('[Event] Returning to KG tab from another tab - clearing filter');

                // Clear filter state
                filterState = { cluster_id: null, entity: null };

                // Clear URL completely
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, '', cleanUrl);
                console.log('[Event] URL cleaned to:', cleanUrl);

                // CRITICAL: Clear saved layout positions so nodes don't stay in corner
                localStorage.removeItem('slm_graph_layout');
                console.log('[Event] Cleared saved layout positions');

                // Reload with full graph (respects dropdown settings)
                loadGraph();
            } else {
                // First load - check if cluster filter is in URL (from cluster badge click)
                const urlParams = new URLSearchParams(window.location.search);
                const clusterIdParam = urlParams.get('cluster_id');

                if (clusterIdParam) {
                    console.log('[Event] First load with cluster filter:', clusterIdParam);
                    // Load with filter (will be applied in loadGraph)
                } else {
                    console.log('[Event] First load, no filter');
                }

                loadGraph();
            }
        });
    }

    // Reload graph when dropdown settings change
    const maxNodesSelect = document.getElementById('graph-max-nodes');
    if (maxNodesSelect) {
        maxNodesSelect.addEventListener('change', function() {
            console.log('[Event] Max nodes changed to:', this.value);

            // Clear any active filter when user manually changes settings
            if (filterState.cluster_id || filterState.entity) {
                console.log('[Event] Clearing filter due to dropdown change');
                filterState = { cluster_id: null, entity: null };

                // Clear URL
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, '', cleanUrl);
            }

            // Clear saved layout when changing node count (different # of nodes = different layout)
            localStorage.removeItem('slm_graph_layout');
            console.log('[Event] Cleared saved layout due to settings change');

            loadGraph();
        });
    }

    const minImportanceSelect = document.getElementById('graph-min-importance');
    if (minImportanceSelect) {
        minImportanceSelect.addEventListener('change', function() {
            console.log('[Event] Min importance changed to:', this.value);

            // Clear any active filter when user manually changes settings
            if (filterState.cluster_id || filterState.entity) {
                console.log('[Event] Clearing filter due to dropdown change');
                filterState = { cluster_id: null, entity: null };

                // Clear URL
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, '', cleanUrl);
            }

            // Clear saved layout when changing importance (different nodes = different layout)
            localStorage.removeItem('slm_graph_layout');
            console.log('[Event] Cleared saved layout due to settings change');

            loadGraph();
        });
    }

    console.log('[Init] Graph event listeners setup complete');
}

// ============================================================================
// INITIALIZATION
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupGraphEventListeners);
} else {
    setupGraphEventListeners();
}

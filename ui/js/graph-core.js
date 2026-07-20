// SuperLocalMemory V2.6.5 - Interactive Knowledge Graph - Core Rendering Module
// Copyright (c) 2026 Varun Pratap Bhardwaj — GNU Affero General Public License v3.0 or later
// Part of modular graph visualization system (split from monolithic graph-cytoscape.js)

// ============================================================================
// GLOBAL STATE
// ============================================================================

var cy = null; // Cytoscape.js instance (global)
var graphData = { nodes: [], links: [] }; // Raw data from API
var originalGraphData = { nodes: [], links: [] }; // Unfiltered data (for reset)
var currentLayout = 'fcose'; // Default layout
var filterState = { cluster_id: null, entity: null }; // Current filters
var isInitialLoad = true; // Track if this is the first graph load
var focusedNodeIndex = 0; // Keyboard navigation: currently focused node
var keyboardNavigationEnabled = false; // Track if keyboard nav is active
var lastFocusedElement = null; // Store last focused element for modal return

// ============================================================================
// CLUSTER COLORS
// ============================================================================

const CLUSTER_COLORS = [
    '#667eea', '#764ba2', '#43e97b', '#38f9d7',
    '#4facfe', '#00f2fe', '#f093fb', '#f5576c',
    '#fa709a', '#fee140', '#30cfd0', '#330867'
];

function getClusterColor(cluster_id) {
    if (!cluster_id) return '#999';
    return CLUSTER_COLORS[cluster_id % CLUSTER_COLORS.length];
}

// ============================================================================
// HTML ESCAPE UTILITY
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// GRAPH LOADING
// ============================================================================

async function loadGraph() {
    var maxNodes = document.getElementById('graph-max-nodes').value;
    var minImportance = document.getElementById('graph-min-importance')?.value || 1;

    // Get cluster filter from URL params ONLY on initial load
    // After that, use filterState (which tab event handler controls)
    if (isInitialLoad) {
        const urlParams = new URLSearchParams(window.location.search);
        const clusterIdParam = urlParams.get('cluster_id');
        if (clusterIdParam) {
            filterState.cluster_id = parseInt(clusterIdParam);
            console.log('[loadGraph] Initial load with cluster filter from URL:', filterState.cluster_id);
        }
        isInitialLoad = false;
    }

    // CRITICAL FIX: When filtering by cluster, fetch MORE nodes to ensure all cluster members are included
    // Otherwise only top N memories are fetched and cluster filter fails
    const fetchLimit = filterState.cluster_id ? 200 : maxNodes;
    console.log('[loadGraph] Fetching with limit:', fetchLimit, 'Cluster filter:', filterState.cluster_id);

    try {
        showLoadingSpinner();
        const response = await fetch(`/api/graph?max_nodes=${fetchLimit}&min_importance=${minImportance}`);
        graphData = await response.json();
        originalGraphData = JSON.parse(JSON.stringify(graphData)); // Deep copy

        // Apply filters if set
        if (filterState.cluster_id) {
            graphData = filterByCluster(originalGraphData, filterState.cluster_id);
        }
        if (filterState.entity) {
            graphData = filterByEntity(originalGraphData, filterState.entity);
        }

        renderGraph(graphData);
        hideLoadingSpinner();
    } catch (error) {
        console.error('Error loading graph:', error);
        showError('Failed to load graph. Please try again.');
        hideLoadingSpinner();
    }
}

// ============================================================================
// DATA TRANSFORMATION
// ============================================================================

function transformDataForCytoscape(data) {
    const elements = [];

    // Add nodes
    data.nodes.forEach(node => {
        const contentPreview = node.content_preview || node.summary || node.content || '';
        // Label: first 4 words of content (readable on node), fallback to category
        const contentWords = contentPreview.split(/\s+/).slice(0, 4).join(' ');
        const label = contentWords || node.category || `Memory #${node.id}`;
        const preview = contentPreview.substring(0, 50) + (contentPreview.length > 50 ? '...' : '');

        elements.push({
            group: 'nodes',
            data: {
                id: String(node.id),
                label: label,
                content: node.content || '',
                summary: node.summary || '',
                content_preview: preview,
                category: node.category || '',
                project_name: node.project_name || '',
                cluster_id: node.cluster_id || 0,
                importance: node.importance || 5,
                tags: node.tags || '',
                entities: node.entities || [],
                created_at: node.created_at || '',
                // For rendering
                weight: (node.importance || 5) * 5 // Size multiplier
            }
        });
    });

    // Add edges
    data.links.forEach(link => {
        const sourceId = String(typeof link.source === 'object' ? link.source.id : link.source);
        const targetId = String(typeof link.target === 'object' ? link.target.id : link.target);

        elements.push({
            group: 'edges',
            data: {
                id: `${sourceId}-${targetId}`,
                source: sourceId,
                target: targetId,
                weight: link.weight || 0.5,
                relationship_type: link.relationship_type || '',
                shared_entities: link.shared_entities || []
            }
        });
    });

    return elements;
}

// ============================================================================
// GRAPH RENDERING
// ============================================================================

function renderGraph(data) {
    const container = document.getElementById('graph-container');
    if (!container) {
        console.error('Graph container not found');
        return;
    }

    // Clear existing graph
    if (cy) {
        cy.destroy();
    }

    // Check node count for performance tier
    const nodeCount = data.nodes.length;
    let layout = determineBestLayout(nodeCount);

    // Transform data
    const elements = transformDataForCytoscape(data);

    if (elements.length === 0) {
        const emptyMsg = document.createElement('div');
        emptyMsg.style.cssText = 'text-align:center; padding:50px; color:#666;';
        emptyMsg.textContent = 'No memories found. Try adjusting filters.';
        container.textContent = ''; // Clear safely
        container.appendChild(emptyMsg);
        return;
    }

    // Clear container (remove spinner/old content) - safe approach
    container.textContent = '';
    console.log('[renderGraph] Container cleared, initializing Cytoscape with', elements.length, 'elements');

    // Initialize Cytoscape WITHOUT running layout yet
    try {
        cy = cytoscape({
            container: container,
            elements: elements,
            style: getCytoscapeStyles(),
            layout: { name: 'preset' }, // Don't run layout yet
            minZoom: 0.2,
            maxZoom: 3,
            wheelSensitivity: 0.05,  // Reduced from 0.2 - less sensitive zoom
            autoungrabify: false,
            autounselectify: false,
            // Mobile touch support
            touchTapThreshold: 8,  // Pixels of movement allowed for tap (vs drag)
            desktopTapThreshold: 4,  // Desktop click threshold
            pixelRatio: 'auto'  // Retina/HiDPI support
        });
        console.log('[renderGraph] Cytoscape initialized successfully, nodes:', cy.nodes().length);
    } catch (error) {
        console.error('[renderGraph] Cytoscape initialization failed:', error);
        showError('Failed to render graph: ' + error.message);
        return;
    }

    // Add interactions
    addCytoscapeInteractions();

    // Add navigator (mini-map) if available
    if (cy.navigator && nodeCount > 50) {
        cy.navigator({
            container: false, // Will be added to DOM separately if needed
            viewLiveFramerate: 0,
            dblClickDelay: 200,
            removeCustomContainer: false,
            rerenderDelay: 100
        });
    }

    // Update UI
    updateFilterBadge();
    updateGraphStats(data);

    // Announce graph load to screen readers
    updateScreenReaderStatus(`Graph loaded with ${data.nodes.length} memories and ${data.links.length} connections`);

    // Check if we should restore saved layout or run fresh layout
    const hasSavedLayout = localStorage.getItem('slm_graph_layout');

    if (hasSavedLayout) {
        // Restore saved positions, then fit
        restoreSavedLayout();
        console.log('[renderGraph] Restored saved layout positions');
        cy.fit(null, 80);
    } else {
        // No saved layout - run layout algorithm with fit
        console.log('[renderGraph] Running fresh layout:', layout);
        const layoutConfig = getLayoutConfig(layout);
        const graphLayout = cy.layout(layoutConfig);

        // CRITICAL: Wait for layout to finish, THEN force fit
        graphLayout.on('layoutstop', function() {
            console.log('[renderGraph] Layout completed, forcing fit to viewport');
            cy.fit(null, 80); // Force center with 80px padding
        });

        graphLayout.run();
    }
}

// ============================================================================
// LAYOUT ALGORITHMS
// ============================================================================

function determineBestLayout(nodeCount) {
    if (nodeCount <= 500) {
        return 'fcose'; // Full interactive graph
    } else if (nodeCount <= 2000) {
        return 'cose'; // Faster force-directed
    } else {
        return 'circle'; // Focus mode (circular for large graphs)
    }
}

function getLayoutConfig(layoutName) {
    const configs = {
        'fcose': {
            name: 'fcose',
            quality: 'default',
            randomize: false,
            animate: false,  // Disabled for stability
            fit: true,
            padding: 80,  // Increased padding to keep within bounds
            nodeSeparation: 100,
            idealEdgeLength: 100,
            edgeElasticity: 0.45,
            nestingFactor: 0.1,
            gravity: 0.25,
            numIter: 2500,
            tile: true,
            tilingPaddingVertical: 10,
            tilingPaddingHorizontal: 10,
            gravityRangeCompound: 1.5,
            gravityCompound: 1.0,
            gravityRange: 3.8
        },
        'cose': {
            name: 'cose',
            animate: false,  // Disabled for stability
            fit: true,
            padding: 80,  // Increased padding
            nodeRepulsion: 8000,
            idealEdgeLength: 100,
            edgeElasticity: 100,
            nestingFactor: 5,
            gravity: 80,
            numIter: 1000,
            randomize: false
        },
        'circle': {
            name: 'circle',
            animate: true,
            animationDuration: 1000,
            fit: true,
            padding: 50,
            sort: function(a, b) {
                return b.data('importance') - a.data('importance');
            }
        },
        'grid': {
            name: 'grid',
            animate: true,
            animationDuration: 1000,
            fit: true,
            padding: 50,
            sort: function(a, b) {
                return b.data('importance') - a.data('importance');
            }
        },
        'breadthfirst': {
            name: 'breadthfirst',
            animate: true,
            animationDuration: 1000,
            fit: true,
            padding: 50,
            directed: false,
            circle: false,
            spacingFactor: 1.5,
            sort: function(a, b) {
                return b.data('importance') - a.data('importance');
            }
        },
        'concentric': {
            name: 'concentric',
            animate: true,
            animationDuration: 1000,
            fit: true,
            padding: 50,
            concentric: function(node) {
                return node.data('importance');
            },
            levelWidth: function() {
                return 2;
            }
        }
    };

    return configs[layoutName] || configs['fcose'];
}

// ============================================================================
// CYTOSCAPE STYLES
// ============================================================================

function getCytoscapeStyles() {
    return [
        {
            selector: 'node',
            style: {
                'background-color': function(ele) {
                    return getClusterColor(ele.data('cluster_id'));
                },
                'width': function(ele) {
                    return Math.max(20, ele.data('weight'));
                },
                'height': function(ele) {
                    return Math.max(20, ele.data('weight'));
                },
                'label': 'data(label)',
                'font-size': '10px',
                'text-valign': 'center',
                'text-halign': 'center',
                'color': '#333',
                'text-outline-width': 2,
                'text-outline-color': '#fff',
                'border-width': function(ele) {
                    // Trust score → border thickness (if available)
                    return 2;
                },
                'border-color': '#555'
            }
        },
        {
            selector: 'node:selected',
            style: {
                'border-width': 4,
                'border-color': '#667eea',
                'background-color': '#667eea'
            }
        },
        {
            selector: 'node.highlighted',
            style: {
                'border-width': 4,
                'border-color': '#ff6b6b',
                'box-shadow': '0 0 20px #ff6b6b'
            }
        },
        {
            selector: 'node.dimmed',
            style: {
                'opacity': 0.3
            }
        },
        {
            selector: 'node.keyboard-focused',
            style: {
                'border-width': 5,
                'border-color': '#0066ff',
                'border-style': 'solid',
                'box-shadow': '0 0 15px #0066ff'
            }
        },
        {
            selector: 'edge',
            style: {
                'width': function(ele) {
                    return Math.max(1, ele.data('weight') * 3);
                },
                'line-color': '#ccc',
                'line-style': function(ele) {
                    return ele.data('weight') > 0.3 ? 'solid' : 'dashed';
                },
                'curve-style': 'bezier',
                'target-arrow-shape': 'none',
                'opacity': 0.6
            }
        },
        {
            selector: 'edge.highlighted',
            style: {
                'line-color': '#667eea',
                'width': 3,
                'opacity': 1
            }
        },
        {
            selector: 'edge.dimmed',
            style: {
                'opacity': 0.1
            }
        }
    ];
}

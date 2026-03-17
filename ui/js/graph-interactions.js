// SuperLocalMemory V2.6.5 - Interactive Knowledge Graph - Interactions Module
// Copyright (c) 2026 Varun Pratap Bhardwaj — MIT License
// Part of modular graph visualization system (split from monolithic graph-cytoscape.js)

// ============================================================================
// CYTOSCAPE INTERACTIONS
// ============================================================================

function addCytoscapeInteractions() {
    if (!cy) return;

    // Hover: Show tooltip
    cy.on('mouseover', 'node', function(evt) {
        const node = evt.target;
        const pos = evt.renderedPosition;
        showTooltip(node, pos.x, pos.y);

        // Highlight connected nodes
        node.addClass('highlighted');
        node.connectedEdges().addClass('highlighted');
        node.neighborhood('node').addClass('highlighted');

        // Dim others
        cy.nodes().not(node).not(node.neighborhood('node')).addClass('dimmed');
        cy.edges().not(node.connectedEdges()).addClass('dimmed');
    });

    cy.on('mouseout', 'node', function(evt) {
        hideTooltip();

        // Remove highlighting
        cy.elements().removeClass('highlighted').removeClass('dimmed');
    });

    // Single click: Open modal preview (source='graph' for context-aware buttons)
    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        openMemoryModal(node);
    });

    // Double click: Navigate to Memories tab
    cy.on('dbltap', 'node', function(evt) {
        const node = evt.target;
        navigateToMemoryTab(node.data('id'));
    });

    // Enable node dragging
    cy.on('drag', 'node', function(evt) {
        // Node position is automatically updated by Cytoscape
    });

    // Save layout when drag ends
    cy.on('dragfree', 'node', function(evt) {
        saveLayoutPositions();
    });

    // Pan & zoom events (for performance monitoring)
    let panZoomTimeout;
    cy.on('pan zoom', function() {
        clearTimeout(panZoomTimeout);
        panZoomTimeout = setTimeout(() => {
            saveLayoutPositions();
        }, 1000);
    });

    // Add keyboard navigation
    setupKeyboardNavigation();
}

// ============================================================================
// TOOLTIP
// ============================================================================

let tooltipTimeout;
function showTooltip(node, x, y) {
    clearTimeout(tooltipTimeout);
    tooltipTimeout = setTimeout(() => {
        let tooltip = document.getElementById('graph-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'graph-tooltip';
            tooltip.style.cssText = 'position:fixed; background:#333; color:#fff; padding:10px; border-radius:6px; font-size:12px; max-width:300px; z-index:10000; pointer-events:none; box-shadow: 0 4px 12px rgba(0,0,0,0.3);';
            document.body.appendChild(tooltip);
        }

        // Build tooltip content safely (no innerHTML)
        tooltip.textContent = ''; // Clear
        const data = node.data();

        const title = document.createElement('strong');
        title.textContent = data.label;
        tooltip.appendChild(title);

        tooltip.appendChild(document.createElement('br'));

        const meta = document.createElement('span');
        meta.style.color = '#aaa';
        meta.textContent = `Cluster ${data.cluster_id} • Importance ${data.importance}`;
        tooltip.appendChild(meta);

        tooltip.appendChild(document.createElement('br'));

        const preview = document.createElement('span');
        preview.style.cssText = 'font-size:11px; color:#ccc;';
        preview.textContent = data.content_preview;
        tooltip.appendChild(preview);

        tooltip.style.display = 'block';
        tooltip.style.left = (x + 20) + 'px';
        tooltip.style.top = (y - 20) + 'px';
    }, 200);
}

function hideTooltip() {
    clearTimeout(tooltipTimeout);
    const tooltip = document.getElementById('graph-tooltip');
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

// ============================================================================
// MODAL INTEGRATION
// ============================================================================

function openMemoryModal(node) {
    const memoryData = {
        id: node.data('id'),
        content: node.data('content'),
        summary: node.data('summary'),
        category: node.data('category'),
        project_name: node.data('project_name'),
        cluster_id: node.data('cluster_id'),
        importance: node.data('importance'),
        tags: node.data('tags'),
        created_at: node.data('created_at')
    };

    if (typeof openMemoryDetail === 'function') {
        openMemoryDetail(memoryData, 'graph'); // source='graph': show Expand Neighbors, hide View Original
    }
}

function navigateToMemoryTab(memoryId) {
    // Switch to Memories tab
    const memoriesTab = document.querySelector('a[href="#memories"]');
    if (memoriesTab) {
        memoriesTab.click();
    }

    // Scroll to memory after a short delay (for tab to load)
    setTimeout(() => {
        if (typeof scrollToMemory === 'function') {
            scrollToMemory(memoryId);
        } else {
            console.warn('scrollToMemory function not found in memories.js');
        }
    }, 300);
}

// ============================================================================
// KEYBOARD NAVIGATION
// ============================================================================

function setupKeyboardNavigation() {
    if (!cy) return;

    const container = document.getElementById('graph-container');
    if (!container) return;

    // Make container focusable
    container.setAttribute('tabindex', '0');

    // Focus handler - enable keyboard nav when container is focused
    container.addEventListener('focus', function() {
        keyboardNavigationEnabled = true;
        if (cy.nodes().length > 0) {
            focusNodeAtIndex(0);
        }
    });

    container.addEventListener('blur', function() {
        keyboardNavigationEnabled = false;
        cy.nodes().removeClass('keyboard-focused');
    });

    // Keyboard event handler
    container.addEventListener('keydown', function(e) {
        if (!keyboardNavigationEnabled || !cy) return;

        const nodes = cy.nodes();
        if (nodes.length === 0) return;

        const currentNode = nodes[focusedNodeIndex];

        switch(e.key) {
            case 'Tab':
                e.preventDefault();
                if (e.shiftKey) {
                    // Shift+Tab: previous node
                    focusedNodeIndex = (focusedNodeIndex - 1 + nodes.length) % nodes.length;
                } else {
                    // Tab: next node
                    focusedNodeIndex = (focusedNodeIndex + 1) % nodes.length;
                }
                focusNodeAtIndex(focusedNodeIndex);
                announceNode(nodes[focusedNodeIndex]);
                break;

            case 'Enter':
            case ' ':
                e.preventDefault();
                if (currentNode) {
                    lastFocusedElement = container;
                    openMemoryModal(currentNode);
                }
                break;

            case 'ArrowRight':
                e.preventDefault();
                moveToAdjacentNode('right', currentNode);
                break;

            case 'ArrowLeft':
                e.preventDefault();
                moveToAdjacentNode('left', currentNode);
                break;

            case 'ArrowDown':
                e.preventDefault();
                moveToAdjacentNode('down', currentNode);
                break;

            case 'ArrowUp':
                e.preventDefault();
                moveToAdjacentNode('up', currentNode);
                break;

            case 'Escape':
                e.preventDefault();
                if (filterState.cluster_id || filterState.entity) {
                    clearGraphFilters();
                    updateScreenReaderStatus('Filters cleared, showing all memories');
                } else {
                    container.blur();
                    keyboardNavigationEnabled = false;
                }
                break;

            case 'Home':
                e.preventDefault();
                focusedNodeIndex = 0;
                focusNodeAtIndex(0);
                announceNode(nodes[0]);
                break;

            case 'End':
                e.preventDefault();
                focusedNodeIndex = nodes.length - 1;
                focusNodeAtIndex(focusedNodeIndex);
                announceNode(nodes[focusedNodeIndex]);
                break;
        }
    });
}

function focusNodeAtIndex(index) {
    if (!cy) return;

    const nodes = cy.nodes();
    if (index < 0 || index >= nodes.length) return;

    // Remove focus from all nodes
    cy.nodes().removeClass('keyboard-focused');

    // Add focus to target node
    const node = nodes[index];
    node.addClass('keyboard-focused');

    // Center node in viewport with smooth animation
    cy.animate({
        center: { eles: node },
        zoom: Math.max(cy.zoom(), 1.0),
        duration: 300,
        easing: 'ease-in-out'
    });
}

function moveToAdjacentNode(direction, currentNode) {
    if (!currentNode) return;

    const nodes = cy.nodes();
    const currentPos = currentNode.position();
    let bestNode = null;
    let bestScore = Infinity;

    // Find adjacent nodes based on direction
    nodes.forEach((node, index) => {
        if (node.id() === currentNode.id()) return;

        const pos = node.position();
        const dx = pos.x - currentPos.x;
        const dy = pos.y - currentPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        let isCorrectDirection = false;
        let directionScore = 0;

        switch(direction) {
            case 'right':
                isCorrectDirection = dx > 0;
                directionScore = dx;
                break;
            case 'left':
                isCorrectDirection = dx < 0;
                directionScore = -dx;
                break;
            case 'down':
                isCorrectDirection = dy > 0;
                directionScore = dy;
                break;
            case 'up':
                isCorrectDirection = dy < 0;
                directionScore = -dy;
                break;
        }

        // Combine distance with direction preference
        if (isCorrectDirection) {
            const score = distance - (directionScore * 0.5);
            if (score < bestScore) {
                bestScore = score;
                bestNode = node;
                focusedNodeIndex = index;
            }
        }
    });

    if (bestNode) {
        focusNodeAtIndex(focusedNodeIndex);
        announceNode(bestNode);
    }
}

function announceNode(node) {
    if (!node) return;

    const data = node.data();
    const message = `Memory ${data.id}: ${data.label}, Cluster ${data.cluster_id}, Importance ${data.importance} out of 10`;
    updateScreenReaderStatus(message);
}

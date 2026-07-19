// SuperLocalMemory v3.4.1 — Ask My Memory Chat Interface
// Copyright (c) 2026 Varun Pratap Bhardwaj — AGPL-3.0-or-later
// SSE streaming chat grounded in multi-producer memory retrieval

// ============================================================================
// STATE
// ============================================================================

var chatState = {
    messages: [],       // {role: 'user'|'assistant', content: '', citations: []}
    streaming: false,
    abortController: null,
    mode: 'a',          // a=raw results, b=ollama, c=cloud
};

// Quick action presets
var QUICK_ACTIONS = [
    { label: 'What changed this week?', query: 'What changed or was decided in the last 7 days?' },
    { label: 'Key decisions', query: 'What are the most important decisions that were made?' },
    { label: 'Find contradictions', query: 'Are there any contradicting or conflicting memories?' },
    { label: 'Summarize', query: 'Give me a high-level summary of what you know about me and my work.' },
];

// ============================================================================
// INIT — Build chat UI in the right panel
// ============================================================================

function initMemoryChat() {
    var rightPanel = document.getElementById('graph-right-panel');
    if (!rightPanel) return;

    // Replace the detail panel with chat + detail toggle
    rightPanel.innerHTML = ''
        + '<div class="card" style="height:550px; display:flex; flex-direction:column;">'
        + '  <div class="card-header py-2 d-flex align-items-center justify-content-between">'
        + '    <div>'
        + '      <button class="btn btn-sm btn-outline-primary active me-1" id="chat-tab-btn" onclick="showChatPanel()"><i class="bi bi-chat-dots"></i> Ask Memory</button>'
        + '      <button class="btn btn-sm btn-outline-secondary" id="detail-tab-btn" onclick="showDetailPanel()"><i class="bi bi-info-circle"></i> Detail</button>'
        + '    </div>'
        + '    <span class="badge bg-secondary" id="chat-mode-badge" style="font-size:0.7rem;">Mode A</span>'
        + '  </div>'

        // Chat panel
        + '  <div id="chat-panel" style="flex:1; display:flex; flex-direction:column; overflow:hidden;">'
        + '    <div id="chat-messages" style="flex:1; overflow-y:auto; padding:8px; font-size:0.85rem;"></div>'

        // Quick actions
        + '    <div id="chat-quick-actions" class="px-2 py-1 border-top" style="font-size:0.75rem;">'
        + '      ' + QUICK_ACTIONS.map(function(a) {
            return '<button class="btn btn-sm btn-outline-secondary me-1 mb-1" onclick="sendChatQuery(\'' + a.query.replace(/'/g, "\\'") + '\')">' + a.label + '</button>';
        }).join('')
        + '    </div>'

        // Input
        + '    <div class="p-2 border-top">'
        + '      <div class="input-group input-group-sm">'
        + '        <input type="text" class="form-control" id="chat-input" placeholder="Ask your memory..."'
        + '               onkeydown="if(event.key===\'Enter\')sendChatFromInput()">'
        + '        <button class="btn btn-primary" onclick="sendChatFromInput()" id="chat-send-btn">'
        + '          <i class="bi bi-send"></i>'
        + '        </button>'
        + '        <button class="btn btn-outline-danger d-none" onclick="cancelChat()" id="chat-cancel-btn">'
        + '          <i class="bi bi-stop-circle"></i>'
        + '        </button>'
        + '      </div>'
        + '    </div>'
        + '  </div>'

        // Detail panel (hidden by default)
        + '  <div id="detail-panel" style="flex:1; overflow-y:auto; padding:8px; font-size:0.85rem; display:none;">'
        + '    <div id="sigma-detail-content" class="text-muted">Click a node to see its details.</div>'
        + '  </div>'

        + '</div>';

    // Load chat mode from config
    _loadChatMode();
}

// ============================================================================
// PANEL TOGGLE (Chat vs Detail)
// ============================================================================

function showChatPanel() {
    var chatPanel = document.getElementById('chat-panel');
    var detailPanel = document.getElementById('detail-panel');
    var chatBtn = document.getElementById('chat-tab-btn');
    var detailBtn = document.getElementById('detail-tab-btn');
    if (chatPanel) chatPanel.style.display = 'flex';
    if (detailPanel) detailPanel.style.display = 'none';
    if (chatBtn) { chatBtn.classList.add('active'); chatBtn.classList.replace('btn-outline-secondary', 'btn-outline-primary'); }
    if (detailBtn) { detailBtn.classList.remove('active'); detailBtn.classList.replace('btn-outline-primary', 'btn-outline-secondary'); }
}

function showDetailPanel() {
    var chatPanel = document.getElementById('chat-panel');
    var detailPanel = document.getElementById('detail-panel');
    var chatBtn = document.getElementById('chat-tab-btn');
    var detailBtn = document.getElementById('detail-tab-btn');
    if (chatPanel) chatPanel.style.display = 'none';
    if (detailPanel) detailPanel.style.display = 'flex';
    if (detailBtn) { detailBtn.classList.add('active'); detailBtn.classList.replace('btn-outline-secondary', 'btn-outline-primary'); }
    if (chatBtn) { chatBtn.classList.remove('active'); chatBtn.classList.replace('btn-outline-primary', 'btn-outline-secondary'); }
}

// ============================================================================
// SEND MESSAGE
// ============================================================================

function sendChatFromInput() {
    var input = document.getElementById('chat-input');
    if (!input || !input.value.trim()) return;
    sendChatQuery(input.value.trim());
    input.value = '';
}

function sendChatQuery(query) {
    if (chatState.streaming) return; // Don't allow concurrent

    // Add user message
    _addMessage('user', query);
    _renderMessages();

    // Start streaming
    chatState.streaming = true;
    _toggleStreamUI(true);

    var assistantMsg = { role: 'assistant', content: '', citations: [] };
    chatState.messages.push(assistantMsg);

    chatState.abortController = new AbortController();

    fetch('/api/v3/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            mode: chatState.mode,
            limit: 10,
        }),
        signal: chatState.abortController.signal,
    }).then(function(response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        function pump() {
            return reader.read().then(function(result) {
                if (result.done) {
                    _onStreamEnd();
                    return;
                }
                buffer += decoder.decode(result.value, { stream: true });
                var lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line

                var currentEvent = '';
                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i].replace(/\r$/, ''); // strip \r from \r\n streams
                    if (line.startsWith('event: ')) {
                        currentEvent = line.substring(7).trim();
                    } else if (line.startsWith('data: ')) {
                        var data = line.substring(6);
                        _handleSSEEvent(currentEvent, data, assistantMsg);
                    }
                }
                _renderMessages();
                _scrollToBottom();
                return pump();
            });
        }
        return pump();
    }).catch(function(err) {
        if (err.name === 'AbortError') {
            assistantMsg.content += '\n\n[Cancelled]';
        } else {
            assistantMsg.content += '\n\n[Error: ' + err.message + ']';
        }
        _onStreamEnd();
    });
}

function cancelChat() {
    if (chatState.abortController) {
        chatState.abortController.abort();
    }
}

// ============================================================================
// SSE EVENT HANDLING
// ============================================================================

function _handleSSEEvent(eventType, data, assistantMsg) {
    if (eventType === 'token') {
        assistantMsg.content += data;
    } else if (eventType === 'citation') {
        try {
            var citation = JSON.parse(data);
            assistantMsg.citations.push(citation);
        } catch (e) { /* skip */ }
    } else if (eventType === 'error') {
        try {
            var err = JSON.parse(data);
            assistantMsg.content += '\n[Error: ' + (err.message || 'Unknown') + ']';
        } catch (e) {
            assistantMsg.content += '\n[Error]';
        }
    }
    // 'done' event handled by stream end
}

// ============================================================================
// MESSAGE RENDERING
// ============================================================================

function _addMessage(role, content) {
    chatState.messages.push({ role: role, content: content, citations: [] });
}

function _renderMessages() {
    var container = document.getElementById('chat-messages');
    if (!container) return;
    container.innerHTML = '';

    chatState.messages.forEach(function(msg) {
        var div = document.createElement('div');
        div.className = 'mb-2 p-2 rounded ' + (msg.role === 'user'
            ? 'bg-primary bg-opacity-10 text-end'
            : 'bg-light');

        // Render content with basic markdown (bold, newlines)
        var html = (msg.content || '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');

        // Make [MEM-N] citations clickable
        html = html.replace(/\[MEM-(\d+)\]/g, function(match, num) {
            var idx = parseInt(num) - 1;
            var citation = msg.citations[idx];
            if (citation) {
                return '<a href="#" class="badge bg-primary text-decoration-none" '
                    + 'onclick="event.preventDefault(); _onCitationClick(\'' + citation.fact_id + '\')" '
                    + 'title="' + (citation.content_preview || '').replace(/"/g, '&quot;') + '">'
                    + match + '</a>';
            }
            return '<span class="badge bg-secondary">' + match + '</span>';
        });

        div.innerHTML = '<small class="text-muted">' + (msg.role === 'user' ? 'You' : 'Memory') + '</small><br>' + html;
        container.appendChild(div);
    });
}

function _scrollToBottom() {
    var container = document.getElementById('chat-messages');
    if (container) container.scrollTop = container.scrollHeight;
}

// ============================================================================
// CITATION CLICK → HIGHLIGHT IN GRAPH
// ============================================================================

function _onCitationClick(factId) {
    // Use event bus for graph ↔ chat linking
    if (window.SLMEventBus) {
        SLMEventBus.publishDebounced('slm:chat:citationClicked', { factId: factId }, 100);
    }
    // Also show detail panel
    if (typeof openSigmaNodeDetail === 'function') {
        openSigmaNodeDetail(factId);
        showDetailPanel();
    }
}

// ============================================================================
// UI HELPERS
// ============================================================================

function _toggleStreamUI(streaming) {
    var sendBtn = document.getElementById('chat-send-btn');
    var cancelBtn = document.getElementById('chat-cancel-btn');
    var input = document.getElementById('chat-input');
    if (sendBtn) sendBtn.classList.toggle('d-none', streaming);
    if (cancelBtn) cancelBtn.classList.toggle('d-none', !streaming);
    if (input) input.disabled = streaming;
}

function _onStreamEnd() {
    chatState.streaming = false;
    chatState.abortController = null;
    _toggleStreamUI(false);
    _renderMessages();
    _scrollToBottom();
}

function _loadChatMode() {
    // Auto-detect mode from Settings — no user dropdown needed
    fetch('/api/v3/mode').then(function(r) { return r.json(); }).then(function(data) {
        var mode = data.mode || 'a';
        chatState.mode = mode;

        var modeNames = { 'a': 'Mode A · Raw Results', 'b': 'Mode B · Ollama', 'c': 'Mode C · Cloud LLM' };
        var modeColors = { 'a': 'bg-secondary', 'b': 'bg-success', 'c': 'bg-primary' };

        var badge = document.getElementById('chat-mode-badge');
        if (badge) {
            badge.textContent = modeNames[mode] || 'Mode ' + mode.toUpperCase();
            badge.className = 'badge ' + (modeColors[mode] || 'bg-secondary');
            badge.style.fontSize = '0.7rem';
            badge.title = 'Change mode in Settings tab';
        }

        // Show guidance for Mode A users
        if (mode === 'a') {
            var msgs = document.getElementById('chat-messages');
            if (msgs && msgs.children.length === 0) {
                msgs.innerHTML = '<div class="text-muted small p-3 text-center">'
                    + '<i class="bi bi-info-circle"></i> <strong>Mode A</strong> — No LLM connected.<br>'
                    + 'Chat returns raw memory retrieval results.<br>'
                    + 'For AI-powered conversation, switch to <strong>Mode B</strong> (Ollama) or <strong>Mode C</strong> (Cloud) in the <strong>Settings</strong> tab.<br>'
                    + '<br>You can also use the <strong>Recall Lab</strong> tab for full recall search.'
                    + '</div>';
            }
        }
    }).catch(function() { /* keep default 'a' */ });
}

// ============================================================================
// INIT ON DOM READY
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // v3.4.4: Robust init — works with both Bootstrap tabs AND Neural Glass sidebar.
    // Strategy: poll for graph-pane visibility instead of relying on tab events.

    // Method 1: Bootstrap tab event (legacy compat)
    var graphTab = document.getElementById('graph-tab');
    if (graphTab) {
        graphTab.addEventListener('shown.bs.tab', function() {
            if (!document.getElementById('chat-panel')) {
                initMemoryChat();
            }
        });
    }

    // Method 2: MutationObserver on graph-pane class changes (Neural Glass sidebar)
    var graphPane = document.getElementById('graph-pane');
    if (graphPane) {
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(m) {
                if (m.attributeName === 'class' && graphPane.classList.contains('active')) {
                    if (!document.getElementById('chat-panel')) {
                        initMemoryChat();
                    }
                }
            });
        });
        observer.observe(graphPane, { attributes: true, attributeFilter: ['class'] });

        // Method 3: If graph-pane is ALREADY active on page load (e.g. hash navigation)
        if (graphPane.classList.contains('active')) {
            setTimeout(function() {
                if (!document.getElementById('chat-panel')) {
                    initMemoryChat();
                }
            }, 500);
        }
    }
});

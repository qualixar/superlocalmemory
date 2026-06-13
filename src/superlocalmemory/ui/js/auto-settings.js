// SuperLocalMemory V3 — Auto-Capture/Recall Settings
// Wires the auto-capture and auto-recall toggle switches to the V3 API.

async function loadAutoSettings() {
    try {
        var captureResp = await fetch('/api/v3/auto-capture/config');
        var recallResp = await fetch('/api/v3/auto-recall/config');
        var capture = captureResp.ok ? await captureResp.json() : {};
        var recall = recallResp.ok ? await recallResp.json() : {};

        var cc = capture.config || {};
        var rc = recall.config || {};

        var el;
        el = document.getElementById('auto-capture-toggle');
        if (el) el.checked = cc.enabled !== false;
        el = document.getElementById('auto-capture-decisions');
        if (el) el.checked = cc.capture_decisions !== false;
        el = document.getElementById('auto-capture-bugs');
        if (el) el.checked = cc.capture_bugs !== false;
        el = document.getElementById('auto-recall-toggle');
        if (el) el.checked = rc.enabled !== false;
        el = document.getElementById('auto-recall-session');
        if (el) el.checked = rc.on_session_start !== false;
    } catch (e) {
        console.log('Auto settings load error:', e);
    }
}

function saveAutoCaptureConfig() {
    var payload = {
        enabled: document.getElementById('auto-capture-toggle')?.checked,
        capture_decisions: document.getElementById('auto-capture-decisions')?.checked,
        capture_bugs: document.getElementById('auto-capture-bugs')?.checked
    };
    fetch('/api/v3/auto-capture/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).catch(function(e) { console.log('Save auto-capture error:', e); });
}

function saveAutoRecallConfig() {
    var payload = {
        enabled: document.getElementById('auto-recall-toggle')?.checked,
        on_session_start: document.getElementById('auto-recall-session')?.checked
    };
    fetch('/api/v3/auto-recall/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).catch(function(e) { console.log('Save auto-recall error:', e); });
}

// Bind change listeners for auto-capture toggles
document.querySelectorAll('#auto-capture-toggle, #auto-capture-decisions, #auto-capture-bugs').forEach(function(el) {
    if (el) {
        el.addEventListener('change', saveAutoCaptureConfig);
    }
});

// Bind change listeners for auto-recall toggles
document.querySelectorAll('#auto-recall-toggle, #auto-recall-session').forEach(function(el) {
    if (el) {
        el.addEventListener('change', saveAutoRecallConfig);
    }
});

// ============================================================================
// Mode / Provider / Model Configuration (Professional Settings)
// ============================================================================

var PROVIDER_CONFIG = {
    'ollama': {
        name: 'Ollama',
        needsKey: false,
        endpoint: 'http://localhost:11434',
        endpointEditable: true,
        detectModels: true, // auto-detect via /api/v3/ollama/status
    },
    'openrouter': {
        name: 'OpenRouter',
        needsKey: true,
        endpoint: 'https://openrouter.ai/api/v1',
        endpointEditable: false,
    },
    'openai': {
        name: 'OpenAI',
        needsKey: true,
        endpoint: 'https://api.openai.com/v1',
        endpointEditable: true, // editable for Azure OpenAI
    },
    'anthropic': {
        name: 'Anthropic',
        needsKey: true,
        endpoint: 'https://api.anthropic.com',
        endpointEditable: false,
    },
};

var MODEL_OPTIONS = {
    'none': [],
    'ollama': [
        {value: 'llama3.1:8b', label: 'Llama 3.1 8B'},
        {value: 'llama3.2:latest', label: 'Llama 3.2'},
        {value: 'qwen3-vl:8b', label: 'Qwen3 VL 8B'},
        {value: 'mistral:latest', label: 'Mistral'},
    ],
    'openrouter': [
        {value: 'meta-llama/llama-3.1-8b-instruct:free', label: 'Llama 3.1 8B (Free)'},
        {value: 'google/gemini-2.0-flash-001', label: 'Gemini 2.0 Flash'},
        {value: 'anthropic/claude-3.5-haiku', label: 'Claude 3.5 Haiku'},
        {value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini'},
        {value: 'deepseek/deepseek-chat-v3-0324:free', label: 'DeepSeek V3 (Free)'},
    ],
    'openai': [
        {value: 'gpt-4o-mini', label: 'GPT-4o Mini'},
        {value: 'gpt-4o', label: 'GPT-4o'},
        {value: 'gpt-4-turbo', label: 'GPT-4 Turbo'},
    ],
    'anthropic': [
        {value: 'claude-3-5-haiku-latest', label: 'Claude 3.5 Haiku'},
        {value: 'claude-3-5-sonnet-latest', label: 'Claude 3.5 Sonnet'},
        {value: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6'},
    ],
};

async function loadModeSettings() {
    try {
        var resp = await fetch('/api/v3/mode');
        if (!resp.ok) return;
        var data = await resp.json();
        var mode = data.mode || 'a';
        var provider = data.provider || 'none';
        var model = data.model || '';

        // Set radio button
        var radio = document.getElementById('mode-' + mode + '-radio');
        if (radio) radio.checked = true;

        // Set provider dropdown
        var provEl = document.getElementById('settings-provider');
        if (provEl && provider !== 'none') provEl.value = provider;

        // Update banner
        var modeNames = {a: 'Mode A — Local Guardian', b: 'Mode B — Smart Local', c: 'Mode C — Full Power'};
        var bannerMode = document.getElementById('settings-current-mode');
        if (bannerMode) {
            var label = modeNames[mode] || mode;
            if (provider && provider !== 'none') label += ' | ' + provider;
            if (model) label += ' | ' + model;
            bannerMode.textContent = label;
        }

        var bannerDetail = document.getElementById('settings-current-detail');
        if (bannerDetail) {
            if (mode === 'a') bannerDetail.textContent = 'Zero cloud — EU AI Act compliant';
            else if (data.has_key) bannerDetail.textContent = 'API key configured';
            else if (provider === 'ollama') bannerDetail.textContent = 'No API key needed';
            else bannerDetail.textContent = 'API key not set';
        }

        var banner = document.getElementById('settings-current-banner');
        if (banner) {
            banner.className = mode === 'a' ? 'alert alert-success mb-3' :
                               mode === 'b' ? 'alert alert-info mb-3' :
                               'alert alert-warning mb-3';
        }

        // Show provider panel and populate model dropdown
        updateModeUI();

        // After provider UI updates, set the saved model value
        if (model) {
            setTimeout(function() {
                var modelEl = document.getElementById('settings-model');
                if (modelEl) {
                    // Check if option exists, if not add it
                    var found = false;
                    for (var i = 0; i < modelEl.options.length; i++) {
                        if (modelEl.options[i].value === model) { found = true; break; }
                    }
                    if (!found) {
                        var opt = document.createElement('option');
                        opt.value = model;
                        opt.textContent = model + ' (current)';
                        modelEl.insertBefore(opt, modelEl.firstChild);
                    }
                    modelEl.value = model;
                }
            }, 500);
        }
    } catch (e) {
        console.log('Load mode settings error:', e);
    }
}

function updateModeUI() {
    var mode = document.querySelector('input[name="settings-mode-radio"]:checked')?.value || 'a';
    var panel = document.getElementById('settings-provider-panel');
    if (panel) {
        panel.style.display = (mode === 'a') ? 'none' : 'block';
    }
    // Only set provider if it's currently empty (first load or Mode A→B/C)
    var providerEl = document.getElementById('settings-provider');
    if (providerEl && !providerEl.value) {
        if (mode === 'b') providerEl.value = 'ollama';
    }
    if (mode !== 'a') updateProviderUI();
}

function updateProviderUI() {
    var provider = document.getElementById('settings-provider')?.value || 'none';
    var modelSelect = document.getElementById('settings-model');
    var modelHint = document.getElementById('settings-model-hint');

    // Preserve current model before rebuilding dropdown
    var currentModel = modelSelect ? modelSelect.value : '';

    var cfg = PROVIDER_CONFIG[provider] || {};

    // Show/hide API key column
    var keyCol = document.getElementById('settings-key-col');
    if (keyCol) keyCol.style.display = cfg.needsKey ? 'block' : 'none';

    // Show/hide endpoint row
    var endpointRow = document.getElementById('settings-endpoint-row');
    var endpointInput = document.getElementById('settings-endpoint');
    if (endpointRow) {
        endpointRow.style.display = cfg.endpointEditable ? 'block' : 'none';
        if (endpointInput && cfg.endpoint) endpointInput.value = cfg.endpoint;
    }

    // For Ollama: check live status and populate real models
    if (provider === 'ollama') {
        if (modelHint) modelHint.textContent = 'Checking Ollama...';
        fetch('/api/v3/ollama/status').then(function(r) { return r.json(); }).then(function(data) {
            if (modelSelect) {
                modelSelect.textContent = '';
                if (data.running && data.models.length > 0) {
                    data.models.forEach(function(m) {
                        var opt = document.createElement('option');
                        opt.value = m.name;
                        opt.textContent = m.name;
                        modelSelect.appendChild(opt);
                    });
                    if (modelHint) modelHint.textContent = 'Ollama running (' + data.count + ' models)';
                    if (modelHint) modelHint.className = 'text-success small';
                } else {
                    var opt = document.createElement('option');
                    opt.value = '';
                    opt.textContent = 'Ollama not running!';
                    modelSelect.appendChild(opt);
                    if (modelHint) modelHint.textContent = 'Ollama not detected. Run: ollama serve';
                    if (modelHint) modelHint.className = 'text-danger small';
                }
            }
        }).catch(function() {
            if (modelHint) { modelHint.textContent = 'Ollama not reachable'; modelHint.className = 'text-danger small'; }
        });
        return;
    }

    // For other providers: use static model list
    if (modelSelect) {
        modelSelect.textContent = '';
        var options = MODEL_OPTIONS[provider] || [];
        if (options.length === 0) {
            var opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'N/A (Mode A)';
            modelSelect.appendChild(opt);
        } else {
            options.forEach(function(o) {
                var opt = document.createElement('option');
                opt.value = o.value;
                opt.textContent = o.label;
                modelSelect.appendChild(opt);
            });
        }
        // Restore previous model selection if it exists in the new list
        if (currentModel) {
            var found = false;
            for (var i = 0; i < modelSelect.options.length; i++) {
                if (modelSelect.options[i].value === currentModel) { found = true; break; }
            }
            if (found) {
                modelSelect.value = currentModel;
            } else if (currentModel) {
                // Model not in list — add it as custom option so user doesn't lose their choice
                var custom = document.createElement('option');
                custom.value = currentModel;
                custom.textContent = currentModel + ' (saved)';
                modelSelect.insertBefore(custom, modelSelect.firstChild);
                modelSelect.value = currentModel;
            }
        }
    }

    // Update hint
    if (modelHint) {
        var hints = {
            'none': 'No LLM needed in Mode A',
            'openrouter': '200+ models via OpenRouter API',
            'openai': 'OpenAI models (requires API key)',
            'anthropic': 'Anthropic models (requires API key)',
        };
        modelHint.textContent = hints[provider] || '';
        modelHint.className = 'text-muted small';
    }
}

async function testConnection() {
    var provider = document.getElementById('settings-provider')?.value || '';
    var model = document.getElementById('settings-model')?.value || '';
    var apiKey = document.getElementById('settings-api-key')?.value || '';
    var resultEl = document.getElementById('settings-test-result');

    if (!provider) {
        if (resultEl) { resultEl.textContent = 'Select a provider first'; resultEl.className = 'ms-2 small text-danger'; }
        return;
    }

    if (resultEl) { resultEl.textContent = 'Testing...'; resultEl.className = 'ms-2 small text-muted'; }

    try {
        var testBody = {provider: provider, model: model};
        if (apiKey) testBody.api_key = apiKey;
        var resp = await fetch('/api/v3/provider/test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(testBody)
        });
        var data = await resp.json();
        if (data.success) {
            if (resultEl) { resultEl.textContent = 'Connected! ' + (data.message || ''); resultEl.className = 'ms-2 small text-success fw-bold'; }
        } else {
            if (resultEl) { resultEl.textContent = 'Failed: ' + (data.error || 'Unknown'); resultEl.className = 'ms-2 small text-danger'; }
        }
    } catch (e) {
        if (resultEl) { resultEl.textContent = 'Error: ' + e.message; resultEl.className = 'ms-2 small text-danger'; }
    }
}

async function saveAllSettings() {
    var mode = document.querySelector('input[name="settings-mode-radio"]:checked')?.value || 'a';
    var provider = document.getElementById('settings-provider')?.value || 'none';
    if (mode === 'a') provider = 'none';
    var model = document.getElementById('settings-model')?.value || '';
    var apiKey = document.getElementById('settings-api-key')?.value || '';

    var statusEl = document.getElementById('settings-save-status');
    var saveBtn = document.getElementById('settings-save-all');
    if (saveBtn) saveBtn.disabled = true;
    if (statusEl) { statusEl.textContent = 'Saving...'; statusEl.style.display = 'inline'; statusEl.className = 'ms-2 text-muted'; }

    try {
        // V3.4.24: Include embedding params in save payload
        var embParams = getEmbeddingParams();
        var payload = Object.assign({mode: mode, provider: provider, model: model, api_key: apiKey}, embParams);
        var modeResp = await fetch('/api/v3/mode/set', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (modeResp.ok) {
            var modeData = await modeResp.json();
            var msg = 'Configuration saved! Mode: ' + mode.toUpperCase() +
                (provider !== 'none' ? ' | Provider: ' + provider : '');
            if (modeData.needs_reindex) {
                msg += ' | Embeddings will be re-indexed on next use (may take several minutes).';
            }
            if (statusEl) {
                statusEl.textContent = msg;
                statusEl.className = modeData.needs_reindex ? 'ms-2 text-warning fw-bold' : 'ms-2 text-success fw-bold';
            }
            loadModeSettings();
            loadEmbeddingSettings();
        } else {
            if (statusEl) { statusEl.textContent = 'Save failed'; statusEl.className = 'ms-2 text-danger'; }
        }
    } catch (e) {
        if (statusEl) { statusEl.textContent = 'Error: ' + e.message; statusEl.className = 'ms-2 text-danger'; }
    }
    if (saveBtn) saveBtn.disabled = false;

    // Auto-hide status after 5 seconds
    setTimeout(function() {
        if (statusEl) statusEl.style.display = 'none';
    }, 5000);
}

// ============================================================================
// Embedding Configuration (V3.4.24 — Custom OpenAI-compatible endpoints)
// ============================================================================

async function loadEmbeddingSettings() {
    try {
        var resp = await fetch('/api/v3/embedding/config');
        if (!resp.ok) return;
        var data = await resp.json();

        var provEl = document.getElementById('settings-emb-provider');
        if (provEl) {
            provEl.value = data.is_openai_compatible ? 'openai' : 'default';
        }

        if (data.is_openai_compatible) {
            var modelEl = document.getElementById('settings-emb-model');
            if (modelEl) modelEl.value = data.model_name || '';
            var dimEl = document.getElementById('settings-emb-dimension');
            if (dimEl) dimEl.value = data.dimension || '';
            var epEl = document.getElementById('settings-emb-endpoint');
            if (epEl) epEl.value = data.api_endpoint || '';
        }

        updateEmbeddingUI();

        var info = document.getElementById('settings-emb-info');
        if (info) {
            var _name = (data.model_name || 'unknown').replace(/[<>&"']/g, function(c) {
                return {'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c];
            });
            if (data.is_openai_compatible) {
                info.innerHTML = 'Using custom endpoint: <strong>' + _name + '</strong> (' + data.dimension + 'd)';
            } else {
                info.innerHTML = 'Using local <strong>' + _name + '</strong> (' + data.dimension + 'd)';
            }
        }
    } catch (e) {
        console.log('Load embedding settings error:', e);
    }
}

function updateEmbeddingUI() {
    var provider = document.getElementById('settings-emb-provider')?.value || 'default';
    var isCustom = provider === 'openai';

    var modelCol = document.getElementById('settings-emb-model-col');
    var dimCol = document.getElementById('settings-emb-dim-col');
    var endpointRow = document.getElementById('settings-emb-endpoint-row');
    var testRow = document.getElementById('settings-emb-test-row');

    if (modelCol) modelCol.style.display = isCustom ? 'block' : 'none';
    if (dimCol) dimCol.style.display = isCustom ? 'block' : 'none';
    if (endpointRow) endpointRow.style.display = isCustom ? 'flex' : 'none';
    if (testRow) testRow.style.display = isCustom ? 'block' : 'none';

    var info = document.getElementById('settings-emb-info');
    if (info && !isCustom) {
        info.innerHTML = 'Using local <strong>nomic-embed-text-v1.5</strong> (768d)';
    }
}

async function testEmbeddingEndpoint() {
    var endpoint = document.getElementById('settings-emb-endpoint')?.value || '';
    var model = document.getElementById('settings-emb-model')?.value || '';
    var key = document.getElementById('settings-emb-key')?.value || '';
    var resultEl = document.getElementById('settings-emb-test-result');

    if (!endpoint) {
        if (resultEl) { resultEl.textContent = 'Enter an endpoint first'; resultEl.className = 'ms-2 small text-danger'; }
        return;
    }

    if (resultEl) { resultEl.textContent = 'Testing...'; resultEl.className = 'ms-2 small text-muted'; }

    try {
        var resp = await fetch('/api/v3/embedding/test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({api_endpoint: endpoint, model_name: model, api_key: key})
        });
        var data = await resp.json();
        if (data.success) {
            if (resultEl) { resultEl.textContent = data.message; resultEl.className = 'ms-2 small text-success fw-bold'; }
            var dimEl = document.getElementById('settings-emb-dimension');
            if (dimEl && data.dimension) {
                if (!dimEl.value) {
                    dimEl.value = data.dimension;
                } else if (parseInt(dimEl.value) !== data.dimension) {
                    if (resultEl) {
                        resultEl.textContent = 'Connected! Warning: endpoint returns ' + data.dimension + 'd but you entered ' + dimEl.value + 'd';
                        resultEl.className = 'ms-2 small text-warning fw-bold';
                    }
                }
            }
        } else {
            if (resultEl) { resultEl.textContent = 'Failed: ' + (data.error || 'Unknown'); resultEl.className = 'ms-2 small text-danger'; }
        }
    } catch (e) {
        if (resultEl) { resultEl.textContent = 'Error: ' + e.message; resultEl.className = 'ms-2 small text-danger'; }
    }
}

function getEmbeddingParams() {
    var provider = document.getElementById('settings-emb-provider')?.value || 'default';
    if (provider !== 'openai') return {};
    return {
        embedding_provider: 'openai',
        embedding_endpoint: document.getElementById('settings-emb-endpoint')?.value || '',
        embedding_model: document.getElementById('settings-emb-model')?.value || '',
        embedding_dimension: parseInt(document.getElementById('settings-emb-dimension')?.value) || 0,
        embedding_key: document.getElementById('settings-emb-key')?.value || '',
    };
}

// Bind events
document.getElementById('settings-provider')?.addEventListener('change', updateProviderUI);
document.getElementById('settings-save-all')?.addEventListener('click', saveAllSettings);
document.getElementById('settings-test-btn')?.addEventListener('click', testConnection);
document.getElementById('settings-emb-provider')?.addEventListener('change', updateEmbeddingUI);
document.getElementById('settings-emb-test-btn')?.addEventListener('click', testEmbeddingEndpoint);

// Mode radio buttons
document.querySelectorAll('input[name="settings-mode-radio"]').forEach(function(radio) {
    radio.addEventListener('change', updateModeUI);
});

// Load settings when the settings tab is shown
document.getElementById('settings-tab')?.addEventListener('shown.bs.tab', function() {
    loadAutoSettings();
    loadModeSettings();
    loadEmbeddingSettings();
    updateModeUI();
});

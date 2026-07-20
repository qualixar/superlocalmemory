// SuperLocalMemory V3 — Optimize Tab
// Part of Qualixar | https://superlocalmemory.com

(function() {
  'use strict';

  var _pollTimer = null;

  window.initOptimizeTab = function() {
    _loadOptimizeConfig();
    _loadSavings();
    _startSavingsPolling();
  };

  function _startSavingsPolling() {
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = setInterval(_loadSavings, 10000);
  }

  var CFG_CARD = 'optimize-config-card';

  async function _loadOptimizeConfig() {
    try {
      var resp = await fetch('/api/optimize/config');
      if (!resp.ok) {
        showPaneError(CFG_CARD, paneErrorMessage(resp.status), _loadOptimizeConfig, true);
        return;
      }
      var cfg = await resp.json();
      clearPaneError(CFG_CARD, true);
      _setToggle('opt-enabled', cfg.enabled);
      _setToggle('opt-proxy-enabled', cfg.proxy_enabled);
      _setToggle('opt-cache-enabled', cfg.cache_enabled);
      _setToggle('opt-semantic-enabled', cfg.semantic_enabled);
      _setToggle('opt-compress-enabled', cfg.compress_enabled);
      _setSelect('opt-compress-mode', cfg.compress_mode);
      _setToggle('opt-compress-prose', cfg.compress_prose);
      var verEl = document.getElementById('opt-config-version');
      if (verEl) verEl.textContent = cfg.config_version || '-';
    } catch (e) {
      showPaneError(CFG_CARD, paneErrorMessage(0), _loadOptimizeConfig, true);
      console.log('Optimize config load error:', e);
    }
  }

  var SAV_CARD = 'optimize-savings-card';

  async function _loadSavings() {
    try {
      var resp = await fetch('/api/optimize/savings');
      if (!resp.ok) {
        // D-3: no Retry on polling loader \u2014 auto-heals on next poll
        showPaneError(SAV_CARD, paneErrorMessage(resp.status), null, true);
        return;
      }
      var data = await resp.json();
      clearPaneError(SAV_CARD, true);
      var tokensSaved = (data.tokens_saved_input || 0) + (data.tokens_saved_output || 0) + (data.tokens_saved_compress || 0);
      _setText('opt-tokens-saved', tokensSaved.toLocaleString());
      var costSaved = data.cost_saved || {};
      _setText('opt-usd-saved', '$' + (costSaved.usd || 0).toFixed(4));
      _setText('opt-inr-saved', '\u20B9' + (costSaved.inr || 0).toFixed(2));
      _setText('opt-hit-rate', ((data.hit_rate || 0) * 100).toFixed(1) + '%');
      _setText('opt-cache-entries', data.entries || 0);
      _setText('opt-cache-size', _formatBytes(data.cache_bytes || 0));
      if (data.compress_ratio !== null && data.compress_ratio !== undefined) {
        _setText('opt-compression-ratio', data.compress_ratio.toFixed(2));
      }
      _setText('opt-pricing-date', data.pricing_date || '-');
      if (data.is_stale) {
        _setText('opt-stale-warning', 'Pricing data may be outdated');
      }
    } catch (e) {
      // D-3: no Retry on polling loader
      showPaneError(SAV_CARD, paneErrorMessage(0), null, true);
      console.log('Savings load error:', e);
    }
  }

  // Toggle handlers
  document.addEventListener('change', function(e) {
    var id = e.target.id;
    if (!id || !id.startsWith('opt-')) return;
    var val = e.target.checked;

    var fieldMap = {
      'opt-enabled':          'enabled',
      'opt-proxy-enabled':    'proxy_enabled',
      'opt-cache-enabled':    'cache_enabled',
      'opt-semantic-enabled': 'semantic_enabled',
      'opt-compress-enabled': 'compress_enabled',
      'opt-compress-prose':   'compress_prose'
    };

    if (id === 'opt-compress-mode') {
      var mode = e.target.value;
      if (mode === 'aggressive') {
        // M-03: Bootstrap modal replaces browser confirm() for aggressive warning
        var modalEl = document.getElementById('optimizeAggressiveModal');
        if (modalEl && typeof bootstrap !== 'undefined') {
          var modal = bootstrap.Modal.getOrCreateInstance(modalEl);
          modal.show();
          var confirmBtn = document.getElementById('optimize-aggressive-confirm');
          var cancelBtn = document.getElementById('optimize-aggressive-cancel');
          if (confirmBtn) confirmBtn.onclick = function() { modal.hide(); _putConfig({compress_mode: 'aggressive'}); };
          if (cancelBtn) cancelBtn.onclick = function() { modal.hide(); e.target.value = 'safe'; };
          return;
        }
        // Fallback for environments without Bootstrap
        if (!confirm('WARNING: Aggressive mode may reduce output fidelity.\n\nDo NOT use for: code generation, legal text, exact-output tasks, math.\nSafe for: summarization, brainstorming, open-ended chat.\n\nContinue?')) {
          e.target.value = 'safe';
          return;
        }
      }
      _putConfig({compress_mode: mode});
      return;
    }

    var field = fieldMap[id];
    if (field) {
      var body = {};
      body[field] = val;
      _putConfig(body);
      if (id === 'opt-proxy-enabled' || id === 'opt-enabled') {
        var notice = document.getElementById('opt-restart-notice');
        if (notice) notice.classList.remove('d-none');
      }
    }
  });

  async function _putConfig(body) {
    try {
      var resp = await fetch('/api/optimize/config', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      if (!resp.ok) {
        console.warn('Config update failed: HTTP ' + resp.status);
        _loadOptimizeConfig(); // revert toggles to server state
      }
    } catch (e) {
      console.log('Config update error:', e);
      _loadOptimizeConfig(); // revert toggles to server state
    }
  }

  // Copy proxy URL to clipboard
  document.addEventListener('click', function(e) {
    if (e.target && e.target.id === 'opt-copy-url') {
      var url = 'http://localhost:8765';
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).catch(function() {
          _fallbackCopy(url);
        });
      } else {
        _fallbackCopy(url);
      }
      e.target.textContent = 'Copied!';
      setTimeout(function() { e.target.textContent = 'Copy URL'; }, 2000);
    }
  });

  function _fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }

  function _setToggle(id, val) {
    var el = document.getElementById(id);
    if (el) el.checked = !!val;
  }
  function _setSelect(id, val) {
    var el = document.getElementById(id);
    if (el) el.value = val || 'safe';
  }
  function _setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  function _formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    var k = 1024;
    var sizes = ['B', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
  }
})();

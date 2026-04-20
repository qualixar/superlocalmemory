// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
// Part of SuperLocalMemory v3.4.21 — LLD-04 §4.3 (v2)
//
// XSS-safe Brain renderer. Hard rule U7: no unsafe DOM-injection
// sinks (banned by static grep). Every DOM node is built via
// document.createElement + textContent. Tooltips use
// setAttribute('title', string) which the browser auto-escapes.
//
// Design principle (April 18, 2026): ONE honest view. Every real ML
// signal, every adapter, every statistical counter, every interactive
// control is on the page. No hidden developer mode. No toggles. Non-
// technical users see exactly what the system is doing and why.
//
// Primary endpoint: /api/v3/brain (install-token gated, auto-fetched).
// Secondary endpoint: /api/behavioral/status (open on loopback).
// Interactive endpoints:
//   - POST /api/behavioral/report-outcome  (Report Outcome form)
//   - POST /api/learning/reset             (Reset Learning button)

(() => {
  'use strict';

  const TOKEN_STORAGE_KEY = 'slm_install_token';

  // --------------------------------------------------------------------
  // Safe DOM helper — the ONLY way this file creates nodes.
  // --------------------------------------------------------------------
  function EL(tag, props, kids) {
    const el = document.createElement(tag);
    const p = props || {};
    for (const k in p) {
      if (!Object.prototype.hasOwnProperty.call(p, k)) continue;
      const v = p[k];
      if (k === 'className') {
        el.className = v;
      } else if (k === 'text') {
        el.textContent = v == null ? '' : String(v);
      } else if (k.length > 2 && k.slice(0, 2) === 'on') {
        el.addEventListener(k.slice(2).toLowerCase(), v);
      } else {
        el.setAttribute(k, String(v));
      }
    }
    const children = kids || [];
    for (let i = 0; i < children.length; i += 1) {
      const c = children[i];
      if (c != null) el.appendChild(c);
    }
    return el;
  }

  function withTooltip(node, opts) {
    const o = opts || {};
    const src = o.source ? 'Source: ' + o.source + '. ' : '';
    const ml = o.isRealMl === false ? 'Statistical counter, not ML. ' : '';
    const note = o.note || '';
    node.setAttribute('title', src + ml + note);
    return node;
  }

  function badge(kind, label) {
    return EL('span', {
      className: 'brain-honesty-badge ' + kind,
      text: label,
    });
  }

  // --------------------------------------------------------------------
  // Token handling — auto-fetch from local daemon, never prompt user.
  // --------------------------------------------------------------------
  function readToken() {
    try {
      return window.sessionStorage
        ? window.sessionStorage.getItem(TOKEN_STORAGE_KEY)
        : null;
    } catch (e) {
      return null;
    }
  }

  function writeToken(value) {
    try {
      if (window.sessionStorage) {
        window.sessionStorage.setItem(TOKEN_STORAGE_KEY, value);
      }
    } catch (e) {
      // storage disabled — keep in-memory.
    }
  }

  async function fetchTokenFromServer() {
    try {
      const resp = await fetch('/internal/token', {
        credentials: 'same-origin',
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      const tok = data && typeof data.token === 'string'
        ? data.token.trim() : '';
      if (tok) {
        writeToken(tok);
        return tok;
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  async function ensureToken() {
    const cached = readToken();
    if (cached) return cached;
    return fetchTokenFromServer();
  }

  // --------------------------------------------------------------------
  // Network
  // --------------------------------------------------------------------
  async function fetchBrain() {
    let token = await ensureToken();
    const headers = {};
    if (token) headers['X-Install-Token'] = token;
    let resp = await fetch('/api/v3/brain', {
      headers: headers,
      credentials: 'same-origin',
    });
    if (resp.status === 401) {
      token = await fetchTokenFromServer();
      if (token) {
        resp = await fetch('/api/v3/brain', {
          headers: {'X-Install-Token': token},
          credentials: 'same-origin',
        });
      }
    }
    if (!resp.ok) throw new Error('brain_fetch_failed:' + resp.status);
    return resp.json();
  }

  async function fetchBehavioralStatus() {
    try {
      const resp = await fetch('/api/behavioral/status', {
        credentials: 'same-origin',
      });
      if (!resp.ok) return null;
      return resp.json();
    } catch (e) {
      return null;
    }
  }

  async function postReportOutcome(body) {
    const resp = await fetch('/api/behavioral/report-outcome', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      credentials: 'same-origin',
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('report_outcome_failed:' + resp.status);
    return resp.json();
  }

  async function postResetLearning() {
    const resp = await fetch('/api/learning/reset', {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!resp.ok) throw new Error('reset_learning_failed:' + resp.status);
    return resp.json();
  }

  // --------------------------------------------------------------------
  // Formatting helpers
  // --------------------------------------------------------------------
  function fmtBytes(n) {
    const v = Number(n) || 0;
    if (v < 1024) return v + ' B';
    if (v < 1024 * 1024) return (v / 1024).toFixed(1) + ' KB';
    if (v < 1024 * 1024 * 1024) return (v / (1024 * 1024)).toFixed(1) + ' MB';
    return (v / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  function fmtDate(iso) {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return String(iso);
      return d.toLocaleString();
    } catch (e) {
      return String(iso);
    }
  }

  function statRow(label, value) {
    return EL('div', {className: 'brain-stat-row'}, [
      EL('span', {className: 'brain-stat-label', text: label}),
      EL('span', {className: 'brain-stat-value', text: String(value)}),
    ]);
  }

  // --------------------------------------------------------------------
  // Card: Learning (phase, signals, model status)
  // --------------------------------------------------------------------
  // Threshold at which the LightGBM ranker becomes active (LLD-02 §4.10).
  const ML_MODEL_THRESHOLD = 200;

  async function postRetrain(includeSynthetic) {
    const resp = await fetch('/api/learning/retrain', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      credentials: 'same-origin',
      body: JSON.stringify({include_synthetic: !!includeSynthetic}),
    });
    if (!resp.ok) throw new Error('retrain_failed:' + resp.status);
    return resp.json();
  }

  async function postMigrateLegacy() {
    const resp = await fetch('/api/learning/migrate-legacy', {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!resp.ok) throw new Error('migrate_failed:' + resp.status);
    return resp.json();
  }

  function cardLearning(l) {
    const data = l || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: "How I'm getting smarter"}));

    const phase = data.phase || 1;
    const label = data.phase_label || 'Cold start';
    wrap.appendChild(EL('span', {
      className: 'brain-phase-pill active',
      text: 'Phase ' + phase + ' — ' + label,
    }));

    // Progress toward ML activation — the headline thing.
    const signals = data.signals_total || 0;
    const active = !!data.model_active;
    const threshold = ML_MODEL_THRESHOLD;
    const pct = Math.min(100, Math.round((signals / threshold) * 100));

    const progWrap = EL('div', {className: 'brain-progress-wrap'});
    progWrap.appendChild(EL('div', {
      className: 'brain-progress-label',
      text: active
        ? 'LightGBM ranker trained and active'
        : signals + ' / ' + threshold + ' signals collected',
    }));
    const bar = EL('div', {className: 'brain-progress-bar'});
    bar.appendChild(EL('div', {
      className: 'brain-progress-fill' + (active ? ' done' : ''),
      style: 'width:' + (active ? 100 : pct) + '%',
    }));
    progWrap.appendChild(bar);
    wrap.appendChild(progWrap);

    // Plain-English explanation for non-technical users.
    const helpText = active
      ? 'I have learned from your usage and trained a LightGBM model. '
        + "It's actively re-ranking recall results now."
      : (signals >= threshold
        ? 'Enough signals collected. You can train the model now, or '
          + 'wait — it trains automatically on the next cycle.'
        : 'I need ' + (threshold - signals) + ' more real recalls to '
          + 'train the LightGBM ranker. Keep using SLM — no action '
          + 'needed. It activates automatically.');
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: helpText,
    }));

    // Manual training trigger — always present so the user can run a
    // training cycle on demand. Behaviour depends on available data:
    // below threshold with no migrated legacy data → disabled with help;
    // below threshold WITH migrated legacy rows → "Train on all data"
    // including synthetic; at/above threshold → "Train model now".
    const migrated = data.legacy_migrated_count || 0;
    if (!active) {
      const hasLegacy = migrated > 0;
      const canTrain = signals >= threshold || hasLegacy;
      const trainLabel = hasLegacy
        ? 'Train model now (includes ' + migrated + ' legacy rows)'
        : 'Train model now';
      const trainBtn = EL('button', {
        type: 'button',
        className: 'btn btn-sm btn-primary',
        text: trainLabel,
      });
      if (!canTrain) trainBtn.setAttribute('disabled', 'disabled');
      const trainStatus = EL('span', {
        className: 'brain-form-status',
        role: 'status',
        'aria-live': 'polite',
      });
      trainBtn.addEventListener('click', async () => {
        if (!canTrain) return;
        trainStatus.textContent = 'Training…';
        try {
          const r = await postRetrain(hasLegacy);
          if (r && r.trained) {
            trainStatus.textContent = 'Training complete. Refreshing…';
            setTimeout(loadBrain, 1200);
          } else {
            trainStatus.textContent = (r && r.message)
              || 'Not enough training rows yet.';
          }
        } catch (e) {
          trainStatus.textContent = 'Could not start training — try again.';
        }
      });
      wrap.appendChild(trainBtn);
      wrap.appendChild(trainStatus);
    }

    const grid = EL('div', {className: 'brain-stat-grid'});
    grid.appendChild(statRow(
      'Features computed', String(data.features_total || 0)
        + ' / ' + String(data.feature_count_expected || 0),
    ));
    grid.appendChild(statRow(
      'Model version', data.model_version || '—',
    ));
    grid.appendChild(statRow(
      'Model trained', fmtDate(data.model_trained_at),
    ));
    grid.appendChild(statRow(
      'Historic rows migrated', migrated,
    ));
    wrap.appendChild(grid);
    wrap.appendChild(badge('real', 'real ML'));
    return wrap;
  }

  // --------------------------------------------------------------------
  // Card: Legacy migration (only shows when there are un-migrated rows)
  // --------------------------------------------------------------------
  // S9-DASH-03: read a persistent dismissal flag so the card stops
  // nagging once the migration endpoint has reported ``already_done``.
  // Previously this card showed forever while ``pending > 0``, even
  // when the 20 remaining rows were permanently un-migratable stubs
  // (malformed / duplicate). The dismissal is keyed to a migration
  // version so a future re-run with a new sentinel still surfaces.
  function _getMigrationDismissed() {
    try {
      return window.localStorage.getItem('slm_migrate_legacy_done') === '1';
    } catch (e) { return false; }
  }
  function _setMigrationDismissed() {
    try { window.localStorage.setItem('slm_migrate_legacy_done', '1'); }
    catch (e) { /* ignore quota / privacy-mode */ }
  }

  function cardLegacyMigration(l) {
    const data = l || {};
    const pending = data.legacy_migration_pending || 0;
    if (pending <= 0) return null;   // nothing to show when everything migrated
    if (_getMigrationDismissed()) return null;  // user completed it already

    const wrap = EL('section', {className: 'brain-section brain-migration'});
    wrap.appendChild(EL('h4', {text: 'Historic data — ready to migrate'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'I found ' + pending + ' historic feedback rows from before '
        + "v3.4.21. Migrate them into the new learning tables so I can "
        + 'use them to train the LightGBM ranker. Your memories are '
        + 'untouched — this only copies feedback metadata forward.',
    }));

    const btn = EL('button', {
      type: 'button',
      className: 'btn btn-sm btn-primary',
      text: 'Migrate ' + pending + ' rows',
    });
    const status = EL('span', {
      className: 'brain-form-status',
      role: 'status',
      'aria-live': 'polite',
    });

    btn.addEventListener('click', async () => {
      btn.setAttribute('disabled', 'disabled');
      status.textContent = 'Migrating…';
      try {
        const r = await postMigrateLegacy();
        const copied = (r && r.copied != null) ? r.copied : 0;
        if (r && r.already_done) {
          // The migration previously completed; remaining rows are
          // structurally un-migratable. Dismiss the card permanently
          // so we don't nag on every page load.
          status.textContent = 'Already migrated. ' + pending
            + ' rows could not be copied (malformed) — skipping.';
          _setMigrationDismissed();
        } else if (r && r.success !== false) {
          status.textContent = 'Migrated ' + copied + ' rows. '
            + 'Reloading…';
          // Also dismiss in case this run left a residual pending
          // count that won't clear (e.g. sentinel written but a
          // handful of rows failed per-row).
          if (copied === 0) _setMigrationDismissed();
        } else {
          status.textContent = (r && r.error) || 'Migration failed.';
          // Only re-enable on explicit failure so the user can retry.
          btn.removeAttribute('disabled');
        }
        // Reload after a short delay so the refreshed brain state
        // re-evaluates the card visibility. When dismissed, the card
        // will not re-render at all.
        setTimeout(loadBrain, 1000);
      } catch (e) {
        btn.removeAttribute('disabled');
        status.textContent = 'Could not migrate — try again.';
      }
    });
    wrap.appendChild(btn);
    wrap.appendChild(status);
    return wrap;
  }

  // --------------------------------------------------------------------
  // Card: Bandit (adaptive ranking arms)
  // --------------------------------------------------------------------
  function cardBandit(b) {
    const data = b || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Adaptive ranking (bandit)'}));

    const top = data.top_arm_global;
    const topText = top
      ? String(top.arm_id) + '  (plays=' + (top.plays || 0) + ')'
      : '—';

    const grid = EL('div', {className: 'brain-stat-grid'});
    grid.appendChild(statRow('Top arm', topText));
    grid.appendChild(statRow(
      'Strata active',
      String(data.strata_active || 0)
        + ' / ' + String(data.strata_total || 0),
    ));
    grid.appendChild(statRow(
      'Unsettled plays', data.unsettled_plays || 0,
    ));
    grid.appendChild(statRow(
      'Oldest unsettled',
      (data.oldest_unsettled_seconds || 0) + ' s',
    ));
    wrap.appendChild(grid);
    wrap.appendChild(badge('real', 'real ML'));
    return wrap;
  }

  // --------------------------------------------------------------------
  // Card: Usage (recalls, top query types)
  // --------------------------------------------------------------------
  function cardUsage(u) {
    const data = u || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'How you use me'}));
    const node = EL('p', {
      text: 'Recalls in the last 24 hours: '
        + String(data.recalls_last_24h || 0),
    });
    withTooltip(node, {
      source: data.source || 'behavioral_patterns_counters',
      isRealMl: false,
      note: data.disclaimer || '',
    });
    wrap.appendChild(node);

    // Real payload keys: {type, pct}. Not {name, count}.
    const types = data.top_query_types || [];
    if (types.length > 0) {
      wrap.appendChild(EL('h5', {
        className: 'brain-subhead',
        text: 'Top query types',
      }));
      const list = EL('ul', {className: 'brain-list'});
      for (let i = 0; i < Math.min(types.length, 5); i += 1) {
        const t = types[i] || {};
        const name = t.type || t.name || '—';
        const pct = (t.pct != null) ? Number(t.pct).toFixed(1) + '%'
          : (t.count != null ? String(t.count) : '');
        list.appendChild(EL('li', {
          text: String(name) + ' — ' + pct,
        }));
      }
      wrap.appendChild(list);
    }

    // Time-of-day buckets — show when you actually use SLM.
    const buckets = data.top_time_buckets || [];
    if (buckets.length > 0) {
      wrap.appendChild(EL('h5', {
        className: 'brain-subhead',
        text: 'When you use me most',
      }));
      const list = EL('ul', {className: 'brain-list'});
      for (let i = 0; i < Math.min(buckets.length, 5); i += 1) {
        const b = buckets[i] || {};
        const name = b.bucket || b.name || '—';
        const pct = (b.pct != null) ? Number(b.pct).toFixed(1) + '%'
          : (b.count != null ? String(b.count) : '');
        list.appendChild(EL('li', {
          text: String(name) + ' — ' + pct,
        }));
      }
      wrap.appendChild(list);
    }

    wrap.appendChild(badge('counter', 'statistical counter'));
    return wrap;
  }

  // --------------------------------------------------------------------
  // Card: Preferences (topics, entities, tech)
  // --------------------------------------------------------------------
  function cardPreferences(p) {
    const data = p || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'What I know about you'}));

    const cols = EL('div', {className: 'brain-pref-cols'});

    cols.appendChild(prefColumn(
      'Topics', data.topics || [],
      (t) => String(t.name) + ' — '
        + Math.round(Number(t.strength || 0) * 100) + '%',
      'Topic clusters emerge from a larger memory set. Entities and '
        + 'tech on the right are already learned.',
    ));
    cols.appendChild(prefColumn(
      'Entities', (data.entities || []).slice(0, 12),
      (e) => String(e.name) + '  ·  ' + (e.mention_count || 0),
    ));
    cols.appendChild(prefColumn(
      'Tech', (data.tech || []).slice(0, 12),
      (t) => String(t.name) + '  ·  '
        + Math.round(Number(t.frequency || 0) * 100) + '%',
    ));
    wrap.appendChild(cols);

    const redacted = data.redacted_count || 0;
    if (redacted > 0) {
      wrap.appendChild(EL('p', {
        className: 'brain-notice',
        text: String(redacted) + ' values redacted as likely secrets',
      }));
    }
    wrap.appendChild(badge('real', 'real data'));
    return wrap;
  }

  function prefColumn(title, items, fmt, emptyMsg) {
    const col = EL('div', {className: 'brain-pref-col'});
    col.appendChild(EL('h5', {
      className: 'brain-subhead',
      text: title,
    }));
    const list = EL('ul', {className: 'brain-list'});
    if (items.length === 0) {
      list.appendChild(EL('li', {
        className: 'brain-empty',
        text: emptyMsg || 'None yet.',
      }));
    }
    for (let i = 0; i < items.length; i += 1) {
      list.appendChild(EL('li', {text: fmt(items[i])}));
    }
    col.appendChild(list);
    return col;
  }

  // --------------------------------------------------------------------
  // Card: Behavioral Outcomes + learned patterns
  // --------------------------------------------------------------------
  function cardBehavioralOutcomes(bh) {
    const data = bh || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Behavioral outcomes'}));

    const breakdown = data.outcome_breakdown || {};
    const total = data.total_outcomes || 0;
    const patterns = data.patterns || [];

    // Honest framing: the tiles count user-reported outcomes (via the
    // "Teach the system" form below); the pattern list is auto-detected
    // from your memories. Zero outcomes doesn't mean no learning — the
    // patterns below are evidence of learning from memory structure.
    if (total === 0 && patterns.length > 0) {
      wrap.appendChild(EL('p', {
        className: 'brain-help',
        text: "I haven't received any outcome reports yet — those come "
          + 'from the form below. But I have auto-detected '
          + patterns.length + ' patterns from the structure of your '
          + 'memories (shown below). Reporting outcomes will sharpen '
          + 'the ranker further.',
      }));
    } else if (total === 0) {
      wrap.appendChild(EL('p', {
        className: 'brain-help',
        text: "I haven't received any outcome reports yet. Use the form "
          + 'below to tell me what worked; each report improves ranking.',
      }));
    }

    const tiles = EL('div', {className: 'brain-outcome-tiles'});
    tiles.appendChild(outcomeTile(
      'Reports', total, 'neutral',
    ));
    tiles.appendChild(outcomeTile(
      'Success', breakdown.success || 0, 'success',
    ));
    tiles.appendChild(outcomeTile(
      'Failure', breakdown.failure || 0, 'failure',
    ));
    tiles.appendChild(outcomeTile(
      'Partial', breakdown.partial || 0, 'warn',
    ));
    wrap.appendChild(tiles);

    if (patterns.length > 0) {
      wrap.appendChild(EL('h5', {
        className: 'brain-subhead',
        text: 'Auto-detected patterns (' + patterns.length + ')',
      }));
      const list = EL('ul', {className: 'brain-list'});
      for (let i = 0; i < Math.min(patterns.length, 10); i += 1) {
        const p = patterns[i];
        const rate = Math.round(Number(p.success_rate || 0) * 100);
        const li = EL('li');
        li.appendChild(EL('span', {
          text: String(p.pattern_type || 'pattern') + ': '
            + String(p.pattern_key || '—') + '  ·  '
            + rate + '%  ·  ' + (p.evidence_count || 0) + ' evidence  ',
        }));
        // S9-DASH-04: delete button — one click kills a wrong pattern.
        const del = EL('button', {
          type: 'button',
          className: 'brain-pattern-del',
          text: '✕',
          title: 'Delete this pattern',
          style: 'margin-left:6px; font-size:11px; cursor:pointer; '
            + 'background:transparent; border:1px solid #555; '
            + 'color:#c88; border-radius:3px; padding:0 5px;',
        });
        del.addEventListener('click', async () => {
          del.classList.add('slm-anim-click');
          setTimeout(() => del.classList.remove('slm-anim-click'), 200);
          del.setAttribute('disabled', 'disabled');
          del.classList.add('slm-anim-spin');
          del.textContent = '';
          try {
            const resp = await fetch('/api/patterns/delete', {
              method: 'DELETE',
              headers: {'Content-Type': 'application/json'},
              credentials: 'same-origin',
              body: JSON.stringify({
                pattern_type: p.pattern_type || '',
                pattern_key: p.pattern_key || '',
              }),
            });
            const r = await resp.json();
            del.classList.remove('slm-anim-spin');
            if (r && r.success) {
              li.classList.add('slm-anim-success');
              setTimeout(() => {
                li.style.opacity = '0.35';
                li.style.textDecoration = 'line-through';
              }, 400);
              del.textContent = '✓';
            } else {
              li.classList.add('slm-anim-fail');
              setTimeout(() => li.classList.remove('slm-anim-fail'), 700);
              del.textContent = '✕';
              del.removeAttribute('disabled');
            }
          } catch (e) {
            del.classList.remove('slm-anim-spin');
            li.classList.add('slm-anim-fail');
            setTimeout(() => li.classList.remove('slm-anim-fail'), 700);
            del.textContent = '✕';
            del.removeAttribute('disabled');
          }
        });
        li.appendChild(del);
        list.appendChild(li);
      }
      wrap.appendChild(list);
    }
    wrap.appendChild(badge('real', 'real data'));
    return wrap;
  }

  function outcomeTile(label, count, kind) {
    const t = EL('div', {className: 'brain-outcome-tile ' + kind});
    t.appendChild(EL('div', {
      className: 'brain-outcome-count', text: String(count),
    }));
    t.appendChild(EL('div', {
      className: 'brain-outcome-label', text: label,
    }));
    return t;
  }

  // --------------------------------------------------------------------
  // Card: Report outcome form (interactive — teach the system)
  // --------------------------------------------------------------------
  function cardReportOutcome() {
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Teach the system — report an outcome'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'Tell me what worked. Each report improves ranking.',
    }));

    const form = EL('form', {
      className: 'brain-form',
      noValidate: 'true',
    });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      await handleReportSubmit(form);
    });

    const ids = EL('input', {
      type: 'text',
      name: 'memory_ids',
      placeholder: 'Memory IDs, comma-separated',
      className: 'brain-input',
    });
    const outcomeSel = EL('select', {
      name: 'outcome',
      className: 'brain-input',
    }, [
      EL('option', {value: 'success', text: 'Success'}),
      EL('option', {value: 'failure', text: 'Failure'}),
      EL('option', {value: 'partial', text: 'Partial'}),
    ]);
    const actionSel = EL('select', {
      name: 'action_type',
      className: 'brain-input',
    }, [
      EL('option', {value: 'code_written', text: 'Code written'}),
      EL('option', {value: 'decision_made', text: 'Decision made'}),
      EL('option', {value: 'debug_resolved', text: 'Debug resolved'}),
      EL('option', {value: 'architecture_chosen', text: 'Architecture chosen'}),
      EL('option', {value: 'other', text: 'Other'}),
    ]);
    const context = EL('input', {
      type: 'text',
      name: 'context',
      placeholder: 'Context (optional)',
      className: 'brain-input',
    });
    const submit = EL('button', {
      type: 'submit',
      className: 'btn btn-sm btn-primary brain-form-submit',
      text: 'Report',
    });
    const status = EL('span', {
      className: 'brain-form-status',
      role: 'status',
      'aria-live': 'polite',
    });

    form.appendChild(ids);
    form.appendChild(outcomeSel);
    form.appendChild(actionSel);
    form.appendChild(context);
    form.appendChild(submit);
    form.appendChild(status);
    wrap.appendChild(form);
    return wrap;
  }

  async function handleReportSubmit(form) {
    const ids = form.elements.namedItem('memory_ids');
    const outcome = form.elements.namedItem('outcome');
    const actionType = form.elements.namedItem('action_type');
    const context = form.elements.namedItem('context');
    const status = form.querySelector('.brain-form-status');

    const rawIds = (ids && ids.value || '').trim();
    if (!rawIds) {
      if (status) status.textContent = 'Enter at least one memory ID.';
      return;
    }
    const memoryIds = rawIds.split(',')
      .map((s) => s.trim()).filter(Boolean);

    const body = {
      memory_ids: memoryIds,
      outcome: outcome ? outcome.value : 'success',
      action_type: actionType ? actionType.value : 'other',
      context: context ? context.value : '',
    };

    const submitBtn = form.querySelector('.brain-form-submit');
    if (submitBtn) {
      submitBtn.classList.add('slm-anim-click');
      setTimeout(() => submitBtn.classList.remove('slm-anim-click'), 200);
      submitBtn.setAttribute('disabled', 'disabled');
      submitBtn.classList.add('slm-anim-spin');
    }
    if (status) status.textContent = 'Reporting…';
    try {
      await postReportOutcome(body);
      if (submitBtn) {
        submitBtn.classList.remove('slm-anim-spin');
        submitBtn.removeAttribute('disabled');
      }
      form.classList.add('slm-anim-success');
      setTimeout(() => form.classList.remove('slm-anim-success'), 900);
      if (status) status.textContent = 'Recorded. Thanks — I will learn from this.';
      form.reset();
      // Refresh the Brain so the outcome tiles update immediately.
      setTimeout(loadBrain, 600);
    } catch (e) {
      if (submitBtn) {
        submitBtn.classList.remove('slm-anim-spin');
        submitBtn.removeAttribute('disabled');
      }
      form.classList.add('slm-anim-fail');
      setTimeout(() => form.classList.remove('slm-anim-fail'), 700);
      if (status) {
        status.textContent = 'Could not record right now — try again.';
      }
      if (window.console && window.console.debug) {
        window.console.debug('report outcome error:', e && e.message);
      }
    }
  }

  // --------------------------------------------------------------------
  // Card: Cross-platform adapters
  // --------------------------------------------------------------------
  function cardCrossPlatform(cp) {
    const data = cp || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Connected clients'}));

    const grid = EL('div', {className: 'brain-adapter-grid'});
    const order = [
      ['claude_code', 'Claude Code'],
      ['cursor', 'Cursor'],
      ['antigravity', 'Antigravity'],
      ['copilot', 'Copilot'],
      ['mcp', 'MCP clients'],
      ['cli', 'CLI'],
    ];
    for (let i = 0; i < order.length; i += 1) {
      const key = order[i][0];
      const nice = order[i][1];
      grid.appendChild(adapterTile(nice, data[key] || {}));
    }
    wrap.appendChild(grid);
    wrap.appendChild(badge('real', 'real sync state'));
    return wrap;
  }

  function adapterTile(nice, entry) {
    const active = !!entry.active;
    const tile = EL('div', {
      className: 'brain-adapter-tile ' + (active ? 'on' : 'off'),
    });
    tile.appendChild(EL('div', {
      className: 'brain-adapter-name', text: nice,
    }));
    tile.appendChild(EL('div', {
      className: 'brain-adapter-dot '
        + (active ? 'dot-on' : 'dot-off'),
    }));
    tile.appendChild(EL('div', {
      className: 'brain-adapter-detail',
      text: active
        ? (entry.last_sync
          ? 'Synced ' + fmtDate(entry.last_sync)
          : 'Connected')
        : (entry.reason || 'Not connected'),
    }));
    return tile;
  }

  // --------------------------------------------------------------------
  // Card: Cache (prewarm hit-rate, size)
  // --------------------------------------------------------------------
  function cardCache(c) {
    const data = c || {};
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Context cache'}));
    const grid = EL('div', {className: 'brain-stat-grid'});
    grid.appendChild(statRow('Cached entries', data.entry_count || 0));
    grid.appendChild(statRow(
      'Database size', fmtBytes(data.db_size_bytes || 0),
    ));
    wrap.appendChild(grid);
    wrap.appendChild(badge('counter', 'statistical counter'));
    return wrap;
  }

  // --------------------------------------------------------------------
  // SVG chart helper — XSS-safe (createElementNS + setAttribute only).
  // --------------------------------------------------------------------
  const SVG_NS = 'http://www.w3.org/2000/svg';

  function svgEL(tag, attrs, kids) {
    const el = document.createElementNS(SVG_NS, tag);
    const a = attrs || {};
    for (const k in a) {
      if (!Object.prototype.hasOwnProperty.call(a, k)) continue;
      el.setAttribute(k, String(a[k]));
    }
    const children = kids || [];
    for (let i = 0; i < children.length; i += 1) {
      const c = children[i];
      if (c != null) el.appendChild(c);
    }
    return el;
  }

  function renderEvolutionChart(points) {
    const n = (points || []).length;
    const W = 640, H = 160, PAD_L = 32, PAD_R = 12, PAD_T = 12, PAD_B = 22;
    const plotW = W - PAD_L - PAD_R;
    const plotH = H - PAD_T - PAD_B;
    const svg = svgEL('svg', {
      viewBox: '0 0 ' + W + ' ' + H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': 'Daily learning signals over time',
      className: 'brain-evolution-chart',
    });
    // Background.
    svg.appendChild(svgEL('rect', {
      x: 0, y: 0, width: W, height: H,
      fill: 'rgba(255,255,255,0.02)',
    }));

    if (n === 0) {
      svg.appendChild(svgEL('text', {
        x: W / 2, y: H / 2, 'text-anchor': 'middle',
        'dominant-baseline': 'middle', fill: 'rgba(255,255,255,0.55)',
        'font-size': '12',
      }, [document.createTextNode('No signals recorded yet')]));
      return svg;
    }

    let maxV = 0;
    for (let i = 0; i < n; i += 1) {
      const v = +points[i].signals || 0;
      if (v > maxV) maxV = v;
    }
    const yMax = Math.max(1, maxV);

    // Gridlines (4 horizontal).
    for (let g = 0; g <= 4; g += 1) {
      const y = PAD_T + (plotH * g) / 4;
      svg.appendChild(svgEL('line', {
        x1: PAD_L, y1: y, x2: W - PAD_R, y2: y,
        stroke: 'rgba(255,255,255,0.06)', 'stroke-width': '1',
      }));
      const label = Math.round(yMax - (yMax * g) / 4);
      svg.appendChild(svgEL('text', {
        x: PAD_L - 4, y: y + 3, 'text-anchor': 'end',
        fill: 'rgba(255,255,255,0.45)', 'font-size': '10',
      }, [document.createTextNode(String(label))]));
    }

    // Line path.
    const stepX = n > 1 ? plotW / (n - 1) : 0;
    const coords = [];
    for (let i = 0; i < n; i += 1) {
      const x = PAD_L + stepX * i;
      const v = +points[i].signals || 0;
      const y = PAD_T + plotH - (plotH * v) / yMax;
      coords.push([x, y]);
    }
    let d = '';
    for (let i = 0; i < coords.length; i += 1) {
      d += (i === 0 ? 'M' : 'L') + coords[i][0].toFixed(1)
        + ' ' + coords[i][1].toFixed(1);
    }
    svg.appendChild(svgEL('path', {
      d: d, fill: 'none', stroke: '#7b9cff', 'stroke-width': '2',
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }));

    // Area fill below the line.
    if (coords.length > 1) {
      const first = coords[0], last = coords[coords.length - 1];
      const baseY = PAD_T + plotH;
      const areaD = d + 'L' + last[0].toFixed(1) + ' ' + baseY
        + 'L' + first[0].toFixed(1) + ' ' + baseY + 'Z';
      svg.appendChild(svgEL('path', {
        d: areaD, fill: 'rgba(123,156,255,0.18)', stroke: 'none',
      }));
    }

    // Point markers + native tooltips (browser-escaped).
    for (let i = 0; i < coords.length; i += 1) {
      const circle = svgEL('circle', {
        cx: coords[i][0].toFixed(1), cy: coords[i][1].toFixed(1),
        r: '2.5', fill: '#7b9cff',
      });
      const title = svgEL('title', {}, [document.createTextNode(
        points[i].date + ' — ' + (+points[i].signals || 0) + ' signal'
          + (points[i].signals === 1 ? '' : 's'),
      )]);
      circle.appendChild(title);
      svg.appendChild(circle);
    }

    // X-axis labels: first, middle, last date.
    const pickIdx = n === 1 ? [0] : [0, Math.floor((n - 1) / 2), n - 1];
    for (let i = 0; i < pickIdx.length; i += 1) {
      const idx = pickIdx[i];
      const x = PAD_L + stepX * idx;
      svg.appendChild(svgEL('text', {
        x: x, y: H - 6, 'text-anchor': i === 0 ? 'start'
          : (i === pickIdx.length - 1 ? 'end' : 'middle'),
        fill: 'rgba(255,255,255,0.55)', 'font-size': '10',
      }, [document.createTextNode(points[idx].date.slice(5))]));
    }

    return svg;
  }

  // --------------------------------------------------------------------
  // Card: Evolution over time (daily signal trend)
  // --------------------------------------------------------------------
  function cardEvolution(ev) {
    const data = ev || {};
    const points = Array.isArray(data.points) ? data.points : [];
    const total = +data.total_signals || 0;
    const days = +data.days || points.length || 0;

    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Evolution over time'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'Daily learning-signal volume over the last '
        + days + ' days. ' + total + ' total signal'
        + (total === 1 ? '' : 's') + ' in window.',
    }));

    const chartWrap = EL('div', {className: 'brain-chart-wrap'});
    chartWrap.appendChild(renderEvolutionChart(points));
    wrap.appendChild(chartWrap);

    wrap.appendChild(withTooltip(
      badge('counter', 'real — learning_signals'),
      {source: 'learning_signals.created_at (grouped by day)',
       isRealMl: false,
       note: 'Counts are raw signal writes, not ML predictions.'},
    ));
    return wrap;
  }

  // --------------------------------------------------------------------
  // Card: Danger zone (reset learning data)
  // --------------------------------------------------------------------
  function cardDangerZone() {
    const wrap = EL('section', {className: 'brain-section brain-danger'});
    wrap.appendChild(EL('h4', {text: 'Danger zone'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'Deletes learning.db and all learned signals. '
        + 'Your memories are preserved.',
    }));
    const btn = EL('button', {
      type: 'button',
      className: 'btn btn-sm btn-outline-danger brain-danger-btn',
      text: 'Reset learning data',
    });
    const status = EL('span', {
      className: 'brain-form-status',
      role: 'status',
      'aria-live': 'polite',
    });
    btn.addEventListener('click', async () => {
      const ok = window.confirm(
        'Reset all learning data? Memories will be preserved, '
        + 'but learned patterns and ranking signals will be deleted.',
      );
      if (!ok) return;
      status.textContent = 'Resetting…';
      try {
        await postResetLearning();
        status.textContent = 'Done. Starting fresh.';
        setTimeout(loadBrain, 800);
      } catch (e) {
        status.textContent = 'Could not reset — try again.';
        if (window.console && window.console.debug) {
          window.console.debug('reset learning error:', e && e.message);
        }
      }
    });
    wrap.appendChild(btn);
    wrap.appendChild(status);
    return wrap;
  }

  // --------------------------------------------------------------------
  // Root render — one honest view, all sections
  // --------------------------------------------------------------------
  function renderAll(brain, behavioral) {
    const b = brain || {};
    const nodes = [
      cardLearning(b.learning),
    ];
    // Only shows when there are legacy rows pending migration; hidden
    // once everything is migrated so the user isn't nagged.
    const migration = cardLegacyMigration(b.learning);
    if (migration) nodes.push(migration);
    nodes.push(cardBandit(b.bandit));
    nodes.push(cardUsage(b.usage));
    nodes.push(cardPreferences(b.preferences));
    nodes.push(cardBehavioralOutcomes(behavioral));
    nodes.push(cardReportOutcome());
    // S9-DASH-05: live closed-loop tiles — reward, shadow test,
    // evolution cost. Data already exposed via /api/v3/brain; we only
    // needed to render it.
    const rewardCard = cardRewardPreview(b.reward_preview);
    if (rewardCard) nodes.push(rewardCard);
    const shadowCard = cardShadowPreview(b.shadow_preview);
    if (shadowCard) nodes.push(shadowCard);
    const evoCostCard = cardEvolutionCostPreview(b.evolution_cost_preview);
    if (evoCostCard) nodes.push(evoCostCard);
    const oqCard = cardOutcomeQueue(b.outcome_queue);
    if (oqCard) nodes.push(oqCard);
    nodes.push(cardCrossPlatform(b.cross_platform));
    nodes.push(cardCache(b.cache));
    nodes.push(cardEvolution(b.evolution_preview));
    nodes.push(cardDangerZone());
    return nodes;
  }

  // --------------------------------------------------------------------
  // S9-DASH-05: live closed-loop tiles
  // --------------------------------------------------------------------
  function cardRewardPreview(rp) {
    if (!rp) return null;
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Reward signal (last 24h)'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'Settled outcomes landing in action_outcomes — the labels '
        + 'your LightGBM ranker trains on. Neutral (0.5) is the '
        + 'reaper default when no hook signals accumulated.',
    }));
    const grid = EL('div', {className: 'brain-kv-grid'});
    grid.appendChild(kv('Rows (24h)', String(rp.rows_24h || 0)));
    grid.appendChild(kv('Mean reward', Number(rp.mean_reward_24h || 0).toFixed(3)));
    wrap.appendChild(grid);
    wrap.appendChild(badge(rp.is_real ? 'real' : 'stub', rp.source || ''));
    return wrap;
  }

  function cardShadowPreview(sp) {
    if (!sp) return null;
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Shadow A/B test'}));
    const hasCandidate = sp.active_candidate_id !== null
      && sp.active_candidate_id !== undefined;
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: hasCandidate
        ? 'A candidate ranker is being evaluated against the active '
          + 'model. Paired NDCG@10 observations accumulate until Phase '
          + 'A strong-stop (n=100) or Phase B full validation (n=885).'
        : 'No candidate in flight. Next retrain will fire when drift '
          + 'is detected or on the 6-hour cadence.',
    }));
    const grid = EL('div', {className: 'brain-kv-grid'});
    grid.appendChild(kv(
      'Candidate id',
      hasCandidate ? String(sp.active_candidate_id) : 'none',
    ));
    grid.appendChild(kv(
      'Paired observations', String(sp.paired_observations || 0),
    ));
    grid.appendChild(kv(
      'Rollbacks (90d)', String(sp.rollback_count_90d || 0),
    ));
    wrap.appendChild(grid);
    wrap.appendChild(badge(sp.is_real ? 'real' : 'stub', sp.source || ''));
    return wrap;
  }

  function cardEvolutionCostPreview(ec) {
    if (!ec) return null;
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Evolution LLM cost (7d)'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'Skill-evolver LLM spend. Mode A never calls an LLM here; '
        + 'Mode B uses Ollama (no cost); Mode C may spend API tokens.',
    }));
    const grid = EL('div', {className: 'brain-kv-grid'});
    grid.appendChild(kv('Calls (7d)', String(ec.calls_7d || 0)));
    grid.appendChild(kv('Cost (USD)', '$' + Number(ec.cost_usd_7d || 0).toFixed(4)));
    grid.appendChild(kv('Tokens in', String(ec.tokens_in_7d || 0)));
    grid.appendChild(kv('Tokens out', String(ec.tokens_out_7d || 0)));
    wrap.appendChild(grid);
    wrap.appendChild(badge(ec.is_real ? 'real' : 'stub', ec.source || ''));
    return wrap;
  }

  function cardOutcomeQueue(oq) {
    if (!oq) return null;
    const wrap = EL('section', {className: 'brain-section'});
    wrap.appendChild(EL('h4', {text: 'Outcome queue (producer health)'}));
    wrap.appendChild(EL('p', {
      className: 'brain-help',
      text: 'Non-blocking background pipeline that turns every recall '
        + 'into a pending_outcome for the closed-loop learning. Drops '
        + 'and failures are always zero on a healthy install.',
    }));
    const counters = oq.counters || {};
    const grid = EL('div', {className: 'brain-kv-grid'});
    grid.appendChild(kv('Queue depth', String(oq.queue_depth || 0)));
    grid.appendChild(kv('Pending now', String(oq.pending_outcomes_now || 0)));
    grid.appendChild(kv('Enqueued', String(counters.recall_enqueued || 0)));
    grid.appendChild(kv('Persisted', String(counters.recall_persisted || 0)));
    grid.appendChild(kv('Reaped (TTL)', String(counters.recall_reaped || 0)));
    grid.appendChild(kv('Drops', String(counters.recall_dropped_queue_full || 0)));
    grid.appendChild(kv('Persist fails', String(counters.recall_persist_failed || 0)));
    wrap.appendChild(grid);
    wrap.appendChild(badge(oq.is_real ? 'real' : 'stub', oq.source || ''));
    return wrap;
  }

  function kv(label, value) {
    const cell = EL('div', {className: 'brain-kv'});
    cell.appendChild(EL('div', {
      className: 'brain-kv-label', text: label,
    }));
    cell.appendChild(EL('div', {
      className: 'brain-kv-value', text: value,
    }));
    return cell;
  }

  function renderError(err) {
    const root = document.getElementById('brain-content');
    if (!root) return;
    root.replaceChildren();
    root.appendChild(EL('div', {
      className: 'brain-error',
      text: "Couldn't load Brain right now. The daemon may still be "
        + 'warming up — try again in a moment.',
    }));
    const btn = EL('button', {
      className: 'btn btn-sm btn-outline-secondary',
      type: 'button',
      text: 'Retry',
    });
    btn.addEventListener('click', loadBrain);
    root.appendChild(btn);
    if (err && err.message && window.console && window.console.debug) {
      window.console.debug('brain load error:', err.message);
    }
  }

  function renderInto(nodes) {
    const root = document.getElementById('brain-content');
    if (!root) return;
    root.replaceChildren();
    for (let i = 0; i < nodes.length; i += 1) {
      root.appendChild(nodes[i]);
    }
  }

  async function loadBrain() {
    try {
      const [brain, behavioral] = await Promise.all([
        fetchBrain(),
        fetchBehavioralStatus(),
      ]);
      renderInto(renderAll(brain, behavioral));
    } catch (e) {
      renderError(e);
    }
  }

  // --------------------------------------------------------------------
  // Public surface + boot
  // --------------------------------------------------------------------
  window.loadBrain = loadBrain;
  // Kept for backward compatibility with any scripts that still call
  // the old toggleBrainView() — it is a no-op now. The Brain view is
  // always the full view; there is no developer-only mode.
  window.toggleBrainView = function toggleBrainView() {
    loadBrain();
  };

  function initBrainBoot() {
    document.addEventListener('shown.bs.tab', (event) => {
      const t = event && event.target;
      if (t && t.id === 'brain-tab') setTimeout(loadBrain, 0);
    });

    const btn = document.getElementById('brain-tab');
    if (btn) {
      btn.addEventListener('click', () => setTimeout(loadBrain, 0));
    }

    if (window.location.hash === '#brain-pane') {
      setTimeout(loadBrain, 0);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBrainBoot);
  } else {
    initBrainBoot();
  }
})();

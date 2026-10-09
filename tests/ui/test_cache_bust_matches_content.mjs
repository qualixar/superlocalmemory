/**
 * Cache-bust params must match the content they version.
 *
 * WHY THIS EXISTS — it cost two false conclusions in a single session.
 *
 * index.html versions its JS with `?v=` query params. Those were hand-bumped.
 * Twice during the 4.0.6 release I loaded the dashboard, read stale JavaScript,
 * and concluded something was broken that was in fact already fixed:
 *
 *   1. I reported that an agent had claimed a fix it "had not made" — the file
 *      on disk had the fix; the browser was serving the previous `?v=` copy.
 *      (Served 43,512 bytes vs 43,677 on disk.)
 *   2. I reported the Connected-clients pane showing the wrong message for a
 *      stale presence registry — the `registry_status === 'stale'` branch was
 *      correct in source, but od-brain.js had changed at 13:52 while
 *      index.html's `?v=406` was written at 13:06, so the browser reused the
 *      old asset.
 *
 * A human hitting this sees a fix "not working", re-reports it, and someone
 * spends an afternoon debugging code that is already correct. So the version
 * param is now the first 8 hex chars of the file's SHA-256, and this test fails
 * whenever the two drift. Editing a versioned file WITHOUT re-stamping it is
 * the bug this catches.
 *
 * To re-stamp after editing:
 *   python3 - <<'PY'
 *   import hashlib, pathlib, re
 *   ui = pathlib.Path('src/superlocalmemory/ui'); html = (ui/'index.html').read_text()
 *   for name in ('od-brain.js','od-graph.js'):
 *       h = hashlib.sha256((ui/'js'/name).read_bytes()).hexdigest()[:8]
 *       html = re.sub(rf'static/js/{re.escape(name)}\?v=[0-9a-f]+',
 *                     f'static/js/{name}?v={h}', html)
 *   (ui/'index.html').write_text(html)
 *   PY
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { createHash } from 'crypto';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');

//: Files whose cache-bust param is content-derived. Add a file here once its
//: `?v=` is switched from a hand-picked number to a content hash.
const HASHED = ['od-brain.js', 'od-graph.js', 'fact-detail.js', 'od-memories.js',
                'od-optimize.js', 'od-mesh.js', 'od-boundedloops.js',
                // 4.1.20: the nav and Settings change with the Answer Check tab;
                // a hand-numbered ?v= would keep serving the old nav after upgrade.
                'od-shell.js', 'od-settings.js', 'od-answercheck.js', 'od-answercheck-tryit.js',
                // 4.1.20 audit: the Memory activity fix and the Automatic answer-check
                // state lived in files a browser could keep serving from cache.
                'dashboard.js', 'answer-check.js',
                // 4.1.21 (#113): the Summaries and Saved views panes.
                'od-summaries.js', 'od-views.js',
                // 4.1.21: the Ops Health table now explains an unconfirmed deletion;
                // a stale copy would keep showing a bare "Exhausted" row.
                'od-ops-health.js',
                // 4.1.21: the live event stream subscribes to memory.captured;
                // a stale copy would keep listening for an event nobody emits.
                'events.js', 'od-connections.js', 'od-mcp.js',
                // 4.1.22: Settings shows a running embedding re-index; a stale copy
                // would keep answering a started switch with "Saved".
                'od-reindex.js', 'od-decay-preview.js',
                // 4.1.23: the Connected apps pane. The sidebar entry, the pane and the
                // internet-connection flow all moved; a stale copy of any of these
                // would keep showing the flow inside MCP & Tools.
                'od-apps-ui.js', 'od-apps-list.js', 'od-apps.js',
                // 4.1.23: Health → Memory store.
                'od-store-check.js',
                // 4.1.25: images and documents card; the restart function moved to be shared.
                'od-features.js', 'od-operations.js', 'od-media.js'];
//: Stylesheets stamped the same way (static/css/<name>?v=<sha256[:8]>).
const HASHED_CSS = ['od-apps.css'];

describe('cache-bust params match file content', function () {
  const html = readFileSync(join(UI, 'index.html'), 'utf8');

  for (const name of HASHED) {
    it(`${name} is stamped with its own content hash`, function () {
      const bytes = readFileSync(join(UI, 'js', name));
      const expected = createHash('sha256').update(bytes).digest('hex').slice(0, 8);

      const m = html.match(
        new RegExp(`static/js/${name.replace('.', '\\.')}\\?v=([0-9a-f]+)`),
      );
      assert.ok(m, `index.html does not reference static/js/${name} with a ?v= param`);
      assert.equal(m[1], expected,
        `${name} was edited without re-stamping its cache-bust param.\n` +
        `  index.html says ?v=${m[1]}\n` +
        `  content hash is ?v=${expected}\n` +
        'Returning users will be served the OLD file and will see none of your ' +
        'changes — which reads as "the fix does not work". Re-stamp it (see the ' +
        'snippet at the top of this test file).');
    });
  }
});

describe('stylesheet cache-bust params match file content', function () {
  const html = readFileSync(join(UI, 'index.html'), 'utf8');

  for (const name of HASHED_CSS) {
    it(`${name} is stamped with its own content hash`, function () {
      const bytes = readFileSync(join(UI, 'css', name));
      const expected = createHash('sha256').update(bytes).digest('hex').slice(0, 8);
      const m = html.match(
        new RegExp(`static/css/${name.replace('.', '\\.')}\\?v=([0-9a-f]+)`),
      );
      assert.ok(m, `index.html does not reference static/css/${name} with a ?v= param`);
      assert.equal(m[1], expected,
        `${name} was edited without re-stamping its cache-bust param ` +
        `(index.html says ?v=${m[1]}, content hash is ?v=${expected}).`);
    });
  }
});

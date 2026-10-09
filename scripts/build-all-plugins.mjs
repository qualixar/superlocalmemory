/**
 * build-all-plugins.mjs — build every plugin, in the one order that works.
 *
 * build-plugin.mjs writes plugin/CLAUDE.md, the skills and the Cursor-format
 * files (build-cursor-plugin.mjs is a module it imports, not a separate
 * program). The copilot, codex, antigravity and hermes builders read that
 * output, so build-plugin must finish first. Stops at the first failure.
 *
 *   node scripts/build-all-plugins.mjs            # build all
 *   node scripts/build-all-plugins.mjs --check    # verify all are in sync
 *   node scripts/build-all-plugins.mjs --dry-run  # print the order only
 */

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

export const STEPS = Object.freeze([
  'build-plugin.mjs',
  'build-copilot-plugin.mjs',
  'build-codex-plugin.mjs',
  'build-antigravity-plugin.mjs',
  'build-hermes-plugin.mjs',
]);

function main(argv) {
  const dryRun = argv.includes('--dry-run');
  const passthrough = argv.includes('--check') ? ['--check'] : [];
  for (const [i, script] of STEPS.entries()) {
    console.log(`step ${i + 1}: ${script}`);
    if (dryRun) continue;
    const r = spawnSync(process.execPath, [path.join(HERE, script), ...passthrough], {
      stdio: 'inherit',
    });
    if (r.status !== 0) {
      console.error(`build-all-plugins: ${script} failed (exit ${r.status ?? r.signal})`);
      return r.status || 1;
    }
  }
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exit(main(process.argv.slice(2)));
}

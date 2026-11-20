/**
 * build-cursor-plugin.mjs — the Cursor-format files of the plugin tree.
 *
 * Grok Bot installs plugins in Cursor's format. It spawns an MCP server's
 * `command` exactly as written and, on its plugin computer, does not replace
 * `${CLAUDE_PLUGIN_ROOT}`. The Claude Code `.mcp.json` names a script inside the
 * plugin through that variable, so on Grok Bot it can never start.
 *
 * So the Cursor manifest names its own MCP definition, and that definition is a
 * command found on PATH: `uvx`, pinned to this release. No `${...}` appears in
 * any file written here — Cursor treats every `${VAR}` other than its plugin
 * root as a user variable that the listing must declare.
 *
 * Written by build-plugin.mjs as part of the same plan, so `--check` covers
 * these files and the version pin moves with plugin-src/manifest.json.
 *
 * Reference: https://cursor.com/docs/reference/plugins (manifest fields,
 * component discovery, variable expansion, marketplace.json, submission list).
 */

import fs from 'node:fs';
import path from 'node:path';

/** Who is calling, for attribution, and what a shared bot computer needs. */
export const CURSOR_SERVER_ENV = Object.freeze({
  // Attribution. Grok Bot and the Cursor editor both read this manifest and
  // the definition cannot tell them apart, so it names the plugin format.
  SLM_AGENT_ID: 'cursor_plugin',
  // No SLM_MCP_PROFILE: the default tool set (full: memory, mesh, optimize),
  // like every host; `slm tools set <name>` changes it. The memory savings a
  // shared bot computer needs are the worker and reranker settings below.
  // Never wait on a prompt (setup_wizard.is_interactive honours it).
  SLM_NON_INTERACTIVE: '1',
  // CPU-only torch from uv: no CUDA wheel stack on a computer with no GPU.
  // The variable rather than `--torch-backend`, so an older uvx that does not
  // know the option ignores it instead of refusing to start.
  UV_TORCH_BACKEND: 'cpu',
  TOKENIZERS_PARALLELISM: 'false',
  // Lite bot-host profile: the box is shared by every bot on it with
  // 1.8-3.5 GiB free RAM, and the cross-encoder reranker subprocess alone
  // measured ~200 MB resident once warm (plus its own PyTorch import). Turn
  // it off here — opt-in via env, nowhere else — so recall still works (BM25
  // + semantic + the other fusion channels), just without cross-encoder
  // re-ordering of the fused results. Quality cost: reranking measurably
  // improves top-of-list precision in SLM's own benchmarks (see
  // bench-v342-locomo.md referenced in core/config.py); turning it off trades
  // that precision for RAM headroom on a host where OOM would lose the
  // session entirely. Set SLM_RERANKER_ENABLED=true to opt back in once RAM
  // allows. SLM_RERANKER_IDLE_TIMEOUT is set too so a future re-enable (or a
  // host that flips this at the MCP env level) recycles the worker quickly
  // instead of holding it warm for the default 30 minutes.
  SLM_RERANKER_ENABLED: 'false',
  SLM_RERANKER_IDLE_TIMEOUT: '120',
  // Explicit, not just relying on the default: one embedding worker, never a
  // pool, on a computer this memory-tight.
  SLM_MAX_EMBEDDING_WORKERS: '1',
});

const DESCRIPTION =
  'Local-first long-term memory for your agents and bots: remember, recall and '
  + 'session context stored on your own computer, with reversible compression '
  + 'and a cache for large tool outputs.';

/**
 * Every skill every other host gets: the plugin now serves the default tool
 * set, not the 18-tool core one, so no skill teaches a tool it lacks except
 * slm-graph and slm-governance, which say how to turn their tools on.
 */
export const CURSOR_SKILLS = Object.freeze([
  'slm-getting-started-bot',
  'slm-bot-memory',
  'slm-remember',
  'slm-recall',
  'slm-session',
  'slm-scope',
  'slm-mesh',
  'slm-cache',
  'slm-compress',
  'slm-cli',
  'slm-status',
  'slm-profile',
  'slm-loop',
  'slm-media',
  'slm-web-access',
  'slm-graph',
  'slm-governance',
]);

const KEYWORDS = ['memory', 'long-term-memory', 'local-first', 'mcp', 'agents'];
const LOGO_SRC = path.join('assets', 'branding', 'slm-mark.svg');
const LOGO_REL = 'assets/logo.svg';

function sortedJson(obj) {
  const sort = (val) => {
    if (Array.isArray(val)) return val.map(sort);
    if (val !== null && typeof val === 'object') {
      return Object.fromEntries(Object.keys(val).sort().map((k) => [k, sort(val[k])]));
    }
    return val;
  };
  return JSON.stringify(sort(obj), null, 2) + '\n';
}

/** The package spec uvx installs: always this release, never "latest". */
export function packagePin(manifest) {
  return `superlocalmemory==${manifest.version}`;
}

export function renderCursorMcpJson(manifest) {
  return sortedJson({
    mcpServers: {
      superlocalmemory: {
        command: 'uvx',
        args: ['--from', packagePin(manifest), 'slm', 'mcp'],
        env: { ...CURSOR_SERVER_ENV },
      },
    },
  });
}

/** plugin/.cursor-plugin/plugin.json — documented fields only. */
export function renderCursorPluginJson(manifest) {
  return sortedJson({
    author: { name: 'Qualixar' },
    description: DESCRIPTION,
    homepage: 'https://superlocalmemory.com',
    // The Claude Code hooks file is not Cursor's format. Declaring hooks
    // replaces folder discovery, so Cursor never tries to run it.
    hooks: { hooks: {} },
    keywords: KEYWORDS,
    license: 'MIT',
    logo: LOGO_REL,
    mcpServers: './mcp.cursor.json',
    name: manifest.pluginName,
    repository: manifest.repository,
    // A curated 4-6, not the full set every other host gets — see
    // CURSOR_SKILLS. A manifest field replaces folder discovery, so this
    // points Cursor at cursor-skills/ instead of the full skills/ directory.
    skills: './cursor-skills/',
    version: manifest.version,
  });
}

/** Repo-root .cursor-plugin/marketplace.json — the Cursor tree only. */
export function renderCursorMarketplaceJson(manifest) {
  const owner = { name: 'Qualixar' };
  if (manifest.marketplace && manifest.marketplace.ownerEmail) {
    owner.email = manifest.marketplace.ownerEmail;
  }
  return sortedJson({
    metadata: { description: 'SuperLocalMemory: local-first long-term memory for agents and bots.' },
    name: 'qualixar',
    owner,
    plugins: [
      {
        category: 'productivity',
        description: DESCRIPTION,
        homepage: 'https://superlocalmemory.com',
        keywords: KEYWORDS,
        license: 'MIT',
        name: manifest.pluginName,
        repository: manifest.repository,
        source: './cursor-plugin',
        version: manifest.version,
      },
    ],
  });
}

/**
 * Plan entries (absolute path -> content) for the Cursor-format files.
 * @param {string} root repository root
 * @param {object} manifest parsed plugin-src/manifest.json
 * @param {string} pluginRoot absolute plugin/ directory
 * @param {Map<string, string>} [fullSkillsPlan] the already-rendered
 *   plugin/skills/<name>/SKILL.md entries build-plugin.mjs's buildPlan() has
 *   by the time it calls this (skills are planned before cursorPlan runs).
 *   Reusing those entries means the Cursor copy gets the exact same
 *   version-stamped, attribution-normalized content as every other host,
 *   through one rendering path instead of two that could drift apart.
 *   Defaults to an empty Map so existing callers that don't need the
 *   cursor-skills/ subset (tests of the other three files) still work; a
 *   real build always passes the real plan.
 * @returns {Map<string, string>}
 */
// The Cursor-format bundle is a thin MIT launcher for the separately installed
// AGPL engine (owner's decision, 14 Oct 2026): Cursor's Marketplace excludes
// copyleft licences. It lives in its own folder so plugin/ (Claude) stays AGPL.
const MIT_LICENSE = `MIT License

Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This license covers this plugin bundle only. The SuperLocalMemory engine it
starts (installed separately) is licensed under AGPL-3.0-or-later.
`;

export function cursorPlan(root, manifest, pluginRoot, fullSkillsPlan = new Map()) {
  const cursorRoot = path.join(root, 'cursor-plugin');
  const logoPath = path.join(root, LOGO_SRC);
  let logo;
  try {
    logo = fs.readFileSync(logoPath, 'utf8');
  } catch (err) {
    throw new Error(`cursorPlan: cannot read logo ${logoPath}: ${err.message}`);
  }
  const plan = new Map([
    [path.join(cursorRoot, '.cursor-plugin', 'plugin.json'), renderCursorPluginJson(manifest)],
    [path.join(cursorRoot, 'mcp.cursor.json'), renderCursorMcpJson(manifest)],
    [path.join(cursorRoot, ...LOGO_REL.split('/')), logo],
    [path.join(cursorRoot, 'LICENSE'), MIT_LICENSE],
    [path.join(root, '.cursor-plugin', 'marketplace.json'), renderCursorMarketplaceJson(manifest)],
  ]);
  for (const name of CURSOR_SKILLS) {
    const srcKey = path.join(pluginRoot, 'skills', name, 'SKILL.md');
    const content = fullSkillsPlan.get(srcKey);
    if (content === undefined) {
      throw new Error(
        `cursorPlan: CURSOR_SKILLS names '${name}', but ${srcKey} is not in the rendered `
        + 'skills plan (check plugin-src/manifest.json lists it, and plugin-src/skills/'
        + `${name}/SKILL.md exists).`,
      );
    }
    plan.set(path.join(cursorRoot, 'cursor-skills', name, 'SKILL.md'), content);
  }
  return plan;
}

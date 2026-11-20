/**
 * build-cursor-plugin.test.mjs — the Cursor-format files (Grok Bot).
 *
 * Grok Bot spawns an MCP command verbatim and never replaces ${...} on its
 * plugin computer, so the Cursor definition must be a PATH command, pinned to
 * the release being built, with no placeholder anywhere.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  CURSOR_SERVER_ENV,
  CURSOR_SKILLS,
  cursorPlan,
  packagePin,
  renderCursorMarketplaceJson,
  renderCursorMcpJson,
  renderCursorPluginJson,
} from '../build-cursor-plugin.mjs';

/** A minimal fullSkillsPlan with every CURSOR_SKILLS name present, the way
 * build-plugin.mjs's buildPlan() would have rendered them by the time it
 * calls cursorPlan() for real. */
function stubSkillsPlan(pluginRoot) {
  const plan = new Map();
  for (const name of CURSOR_SKILLS) {
    plan.set(path.join(pluginRoot, 'skills', name, 'SKILL.md'), `# ${name}\n`);
  }
  return plan;
}

const MANIFEST = {
  version: '9.8.7',
  pluginName: 'superlocalmemory',
  repository: 'https://github.com/qualixar/superlocalmemory',
  marketplace: { ownerEmail: 'owner@example.com', description: 'x' },
};

// Fields https://cursor.com/docs/reference/plugins documents for plugin.json.
const DOCUMENTED_PLUGIN_FIELDS = new Set([
  'name', 'description', 'version', 'author', 'homepage', 'repository', 'license',
  'keywords', 'logo', 'rules', 'agents', 'skills', 'commands', 'hooks', 'mcpServers',
  'variables',
]);

function allStrings(value, out = []) {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) value.forEach((v) => allStrings(v, out));
  else if (value && typeof value === 'object') Object.values(value).forEach((v) => allStrings(v, out));
  return out;
}

describe('Cursor MCP definition', () => {
  const server = JSON.parse(renderCursorMcpJson(MANIFEST)).mcpServers.superlocalmemory;

  test('is a PATH command pinned to the release being built', () => {
    assert.equal(server.command, 'uvx');
    assert.deepEqual(server.args, ['--from', 'superlocalmemory==9.8.7', 'slm', 'mcp']);
    assert.equal(packagePin(MANIFEST), 'superlocalmemory==9.8.7');
  });

  test('contains no ${...} placeholder in command, args or env', () => {
    for (const s of allStrings(server)) assert.ok(!s.includes('${'), s);
  });

  test('names the plugin format, forces no tool set, non-interactive and CPU torch', () => {
    assert.deepEqual(server.env, { ...CURSOR_SERVER_ENV });
    assert.equal(server.env.SLM_AGENT_ID, 'cursor_plugin');
    assert.equal(server.env.SLM_MCP_PROFILE, undefined);
    assert.equal(server.env.SLM_NON_INTERACTIVE, '1');
    assert.equal(server.env.UV_TORCH_BACKEND, 'cpu');
    assert.ok(!('SLM_DATA_DIR' in server.env), 'never re-point the store');
  });

  test('opts into the lite bot-host profile (reranker off, short idle, one embedding worker)', () => {
    assert.equal(server.env.SLM_RERANKER_ENABLED, 'false');
    assert.equal(server.env.SLM_RERANKER_IDLE_TIMEOUT, '120');
    assert.equal(server.env.SLM_MAX_EMBEDDING_WORKERS, '1');
  });
});

describe('Cursor plugin.json', () => {
  const pj = JSON.parse(renderCursorPluginJson(MANIFEST));

  test('uses only documented fields, a kebab-case name and the release version', () => {
    for (const k of Object.keys(pj)) assert.ok(DOCUMENTED_PLUGIN_FIELDS.has(k), `undocumented field ${k}`);
    assert.match(pj.name, /^[a-z0-9][a-z0-9.-]*$/);
    assert.equal(pj.version, '9.8.7');
  });

  test('every path is relative and stays inside the plugin', () => {
    for (const p of [pj.logo, pj.skills, pj.mcpServers]) {
      assert.ok(!path.isAbsolute(p) && !p.split('/').includes('..'), p);
    }
    assert.equal(pj.mcpServers, './mcp.cursor.json');
  });

  test('declares hooks inline-empty so the Claude Code hooks file is never run', () => {
    assert.deepEqual(pj.hooks, { hooks: {} });
  });

  test('contains no ${...} placeholder', () => {
    for (const s of allStrings(pj)) assert.ok(!s.includes('${'), s);
  });
});

describe('Cursor marketplace.json', () => {
  test('lists only the Cursor-ready tree, from its own MIT folder ./cursor-plugin', () => {
    const mk = JSON.parse(renderCursorMarketplaceJson(MANIFEST));
    assert.equal(mk.plugins.length, 1);
    assert.equal(mk.plugins[0].name, 'superlocalmemory');
    assert.equal(mk.plugins[0].source, './cursor-plugin');
    assert.equal(mk.plugins[0].version, '9.8.7');
    assert.equal(mk.owner.name, 'Qualixar');
  });
});

describe('cursorPlan', () => {
  test('writes the manifest, MCP file, logo, licence, marketplace and skills; fails loudly without the logo source', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-cursor-'));
    const pluginRoot = path.join(tmp, 'plugin');
    assert.throws(() => cursorPlan(tmp, MANIFEST, pluginRoot), /logo/);
    fs.mkdirSync(path.join(tmp, 'assets', 'branding'), { recursive: true });
    fs.writeFileSync(path.join(tmp, 'assets', 'branding', 'slm-mark.svg'), '<svg/>\n');
    const plan = cursorPlan(tmp, MANIFEST, pluginRoot, stubSkillsPlan(pluginRoot));
    const cursorRoot = path.join(tmp, 'cursor-plugin');
    assert.deepEqual([...plan.keys()].sort(), [
      path.join(tmp, '.cursor-plugin', 'marketplace.json'),
      path.join(cursorRoot, '.cursor-plugin', 'plugin.json'),
      path.join(cursorRoot, 'assets', 'logo.svg'),
      path.join(cursorRoot, 'mcp.cursor.json'),
      path.join(cursorRoot, 'LICENSE'),
      ...CURSOR_SKILLS.map((name) => path.join(cursorRoot, 'cursor-skills', name, 'SKILL.md')),
    ].sort());
    assert.equal(plan.get(path.join(cursorRoot, 'assets', 'logo.svg')), '<svg/>\n');
  });

  test('fails loudly when a curated skill is missing from the rendered plan', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-cursor-'));
    const pluginRoot = path.join(tmp, 'plugin');
    fs.mkdirSync(path.join(tmp, 'assets', 'branding'), { recursive: true });
    fs.writeFileSync(path.join(tmp, 'assets', 'branding', 'slm-mark.svg'), '<svg/>\n');
    const incomplete = stubSkillsPlan(pluginRoot);
    incomplete.delete(path.join(pluginRoot, 'skills', CURSOR_SKILLS[0], 'SKILL.md'));
    assert.throws(
      () => cursorPlan(tmp, MANIFEST, pluginRoot, incomplete),
      new RegExp(CURSOR_SKILLS[0]),
    );
  });

  test('CURSOR_SKILLS is every skill, the bot-host ones included', () => {
    // Derived from plugin-src, so a new skill needs no edit here, only a place in CURSOR_SKILLS.
    const skillsDir = new URL('../../plugin-src/skills/', import.meta.url);
    const every = fs.readdirSync(skillsDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory()
        && fs.existsSync(new URL(`${entry.name}/SKILL.md`, skillsDir)))
      .map((entry) => entry.name)
      .sort();
    assert.deepEqual([...CURSOR_SKILLS].sort(), every);
    assert.ok(CURSOR_SKILLS.includes('slm-getting-started-bot'));
    assert.ok(CURSOR_SKILLS.includes('slm-bot-memory'));
  });
});

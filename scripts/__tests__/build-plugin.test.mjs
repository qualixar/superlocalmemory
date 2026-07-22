/**
 * build-plugin.test.mjs — DOC-CORRECT layout TDD test suite (node:test, mkdtemp fixtures)
 *
 * v3.6.14: single plugin/ target, 7 new skills, no ide/pkg/commands targets.
 * Layout: .claude-plugin/marketplace.json (repo root) + plugin/ (plugin root).
 *
 * Run: node --test scripts/__tests__/build-plugin.test.mjs
 */

import { test, describe, before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

// ---------------------------------------------------------------------------
// Module path (relative to this file location: scripts/__tests__/)
// ---------------------------------------------------------------------------
const REPO_ROOT = path.resolve(import.meta.dirname, '../..');
const SCRIPT_PATH = path.resolve(import.meta.dirname, '../build-plugin.mjs');

// Lazy-import the module (it exports pure functions)
let mod;
async function getModule() {
  if (!mod) {
    mod = await import(SCRIPT_PATH);
  }
  return mod;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function makeTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'slm-wp04-'));
}

function sha256(content) {
  return crypto.createHash('sha256').update(content).digest('hex');
}

function writeFile(dir, rel, content) {
  const full = path.join(dir, rel);
  fs.mkdirSync(path.dirname(full), { recursive: true });
  fs.writeFileSync(full, content, 'utf8');
  return full;
}

function readFile(p) {
  return fs.readFileSync(p, 'utf8');
}

// The 7 new skills
const NEW_SKILLS = [
  'slm-cache',
  'slm-compress',
  'slm-graph',
  'slm-recall',
  'slm-remember',
  'slm-session',
  'slm-status',
];

// Minimal fixture manifest — DOC-CORRECT layout
function makeManifest(overrides = {}) {
  return {
    version: '3.6.14',
    pluginName: 'superlocalmemory',
    displayName: 'SuperLocalMemory',
    repository: 'https://github.com/qualixar/superlocalmemory',
    keywords: ['memory', 'mcp', 'agents', 'local-first', 'context-compression'],
    skills: NEW_SKILLS.map(name => ({ name, hasReadme: false })),
    targets: {
      plugin: 'plugin/.claude-plugin/plugin.json',
    },
    marketplace: {
      owner: 'qualixar',
      ownerEmail: 'varun.pratap.bhardwaj@gmail.com',
      repo: 'superlocalmemory',
      description: 'Local-first agent memory + reversible context compression and KV cache, as an MCP server. 20-tool code profile with graph intelligence.',
    },
    ...overrides,
  };
}

// Minimal skill body with frontmatter version (for stamp tests)
const OLD_ATTRIBUTION_BLOCK = `
**Created by:** [Varun Pratap Bhardwaj](https://github.com/varun369) (Solution Architect)
**Project:** SuperLocalMemory V2
**License:** MIT (see [LICENSE](../../LICENSE))
**Repository:** https://github.com/varun369/SuperLocalMemoryV2

*Open source doesn't mean removing credit. Attribution must be preserved per MIT License terms.*
`.trimStart();

function makeSkillBody(name, version = '3.4.23') {
  return [
    '---',
    `name: ${name}`,
    `description: Test skill ${name}`,
    `version: "${version}"`,
    'license: AGPL-3.0-or-later',
    `compatibility: "SuperLocalMemory V2 installed"`,
    'attribution:',
    '  creator: Varun Pratap Bhardwaj',
    '  role: Solution Architect',
    `  project: SuperLocalMemory V2`,
    '---',
    '',
    `# ${name}`,
    '',
    'Some body text. API v2 endpoint. Uses the MIT license approach.',
    '',
    '## Usage',
    '',
    'Run the thing.',
    '',
    OLD_ATTRIBUTION_BLOCK,
  ].join('\n');
}

// Skill body WITHOUT a version line (new-style skills)
function makeSkillBodyNoVersion(name) {
  return [
    '---',
    `name: ${name}`,
    `description: Test skill ${name}`,
    'allowed-tools: Bash',
    '---',
    '',
    `# ${name}`,
    '',
    'Body text.',
  ].join('\n');
}

// Setup fixture with all 7 skills + optional extras
function setupFixture(tmp, overrides = {}) {
  const manifest = makeManifest(overrides);
  for (const skill of manifest.skills) {
    writeFile(tmp, `plugin-src/skills/${skill.name}/SKILL.md`, makeSkillBody(skill.name));
  }
  writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
  writeFile(tmp, 'plugin-src/hooks/hooks.json', '{"hooks":{"SessionStart":[]}}\n');
  writeFile(tmp, 'plugin-src/.mcp.json', '{"mcpServers":{"superlocalmemory":{"command":"${CLAUDE_PLUGIN_DATA}/venv/bin/slm","args":["mcp"],"env":{"SLM_MCP_PROFILE":"code","SLM_DATA_DIR":"${CLAUDE_PLUGIN_DATA}"}}}}\n');
  writeFile(tmp, 'plugin-src/settings.json', '{"permissions":{"allow":[]}}\n');
  writeFile(tmp, 'plugin-src/requirements.txt', 'superlocalmemory>=3.6.14\n');
  writeFile(tmp, 'plugin-src/rules/CLAUDE.md.fragment', '<!-- SLM -->\n');
  return manifest;
}

// ---------------------------------------------------------------------------
// TEST 1 — stampVersion replaces only frontmatter version line
// ---------------------------------------------------------------------------
describe('stampVersion', () => {
  test('replaces only frontmatter version line (body v3.4.22 untouched)', async () => {
    const { stampVersion } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.22');
    const bodyWithVersion = body + '\n\nSome note about (v3.4.22) API\n';
    const result = stampVersion(bodyWithVersion, '3.6.14');
    assert.match(result, /^---\n[\s\S]*?version: "3\.6\.14"/m);
    assert.match(result, /\(v3\.4\.22\)/);
  });

  test('stampVersion is idempotent', async () => {
    const { stampVersion } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const once = stampVersion(body, '3.6.14');
    const twice = stampVersion(once, '3.6.14');
    assert.equal(once, twice);
  });

  test('throws if frontmatter has no version line', async () => {
    const { stampVersion } = await getModule();
    const noVersion = '---\nname: foo\n---\n\n# Foo\n';
    assert.throws(() => stampVersion(noVersion, '3.6.14'), /version/i);
  });
});

// ---------------------------------------------------------------------------
// TEST — renderSkillFile: skills without version line are copied verbatim
// ---------------------------------------------------------------------------
describe('renderSkillFile no-version', () => {
  test('skill without version: line in frontmatter copied verbatim (no stamp error)', async () => {
    const { renderSkillFile } = await getModule();
    const body = makeSkillBodyNoVersion('slm-graph');
    // Must not throw
    const result = renderSkillFile(body, '3.6.14');
    assert.ok(result.includes('slm-graph'), 'skill name preserved');
    assert.ok(!result.includes('version: "3.6.14"'), 'no version stamp injected');
  });
});

// ---------------------------------------------------------------------------
// TEST 2 — renderPluginJson: sorted, author.url, mcpServers/hooks pointers
// ---------------------------------------------------------------------------
describe('renderPluginJson', () => {
  test('deterministic sorted output + author.url + pointers', async () => {
    const { renderPluginJson } = await getModule();
    const manifest = makeManifest();
    const json = renderPluginJson(manifest);
    const parsed = JSON.parse(json);
    assert.equal(parsed.name, 'superlocalmemory');
    assert.equal(parsed.version, '3.6.14');
    assert.equal(parsed.author.name, 'Qualixar');
    // REGRESSION: author.url must NOT be silently dropped
    assert.equal(parsed.author.url, 'https://github.com/qualixar/superlocalmemory');
    // pointers relative to plugin root
    assert.ok(parsed.hooks && parsed.hooks.includes('hooks.json'), 'hooks pointer present');
    assert.ok(parsed.mcpServers && parsed.mcpServers.includes('.mcp.json'), 'mcpServers pointer present');
    // keys must be sorted at top level
    const keys = Object.keys(parsed);
    assert.deepEqual(keys, [...keys].sort());
  });

  test('renderPluginJson is deterministic (call twice same output)', async () => {
    const { renderPluginJson } = await getModule();
    const manifest = makeManifest();
    assert.equal(renderPluginJson(manifest), renderPluginJson(manifest));
  });
});

// ---------------------------------------------------------------------------
// TEST 3 — renderMarketplaceJson: source="./plugin", no version in plugin entry
// ---------------------------------------------------------------------------
describe('renderMarketplaceJson', () => {
  test('source="./plugin", no version in plugin entry, owner.name=Qualixar', async () => {
    const { renderMarketplaceJson } = await getModule();
    const manifest = makeManifest();
    const json = renderMarketplaceJson(manifest);
    const parsed = JSON.parse(json);
    assert.equal(parsed.name, 'qualixar');
    assert.equal(parsed.owner.name, 'Qualixar');
    assert.ok(Array.isArray(parsed.plugins));
    assert.equal(parsed.plugins.length, 1);
    assert.equal(parsed.plugins[0].name, 'superlocalmemory');
    // DOC-CORRECT: source must be "./plugin" not "./"
    assert.equal(parsed.plugins[0].source, './plugin', 'source must be ./plugin');
    // AC-3: NO version key in plugin entry
    assert.equal(parsed.plugins[0].version, undefined, 'plugin entry must have no version key');
  });
});

// ---------------------------------------------------------------------------
// TEST 4 — buildPlan: 7 skills in plugin/skills/, no commands/, marketplace at root
// ---------------------------------------------------------------------------
describe('buildPlan', () => {
  test('builds 7 skills in plugin/skills/ + plugin.json + marketplace.json + agents + extras', async () => {
    const { buildPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    const plan = buildPlan(tmp, manifest);

    // 7 skill SKILL.md files
    const skillFiles = [...plan.keys()].filter(k => k.endsWith('SKILL.md'));
    assert.equal(skillFiles.length, 7, `expected 7 SKILL.md, got ${skillFiles.length}`);

    // All in plugin/skills/
    for (const f of skillFiles) {
      assert.ok(f.includes(`plugin${path.sep}skills${path.sep}`), `SKILL.md must be in plugin/skills/: ${f}`);
    }

    // plugin/.claude-plugin/plugin.json
    const pluginJsonFiles = [...plan.keys()].filter(k => k.endsWith('plugin.json'));
    assert.equal(pluginJsonFiles.length, 1);
    assert.ok(pluginJsonFiles[0].includes(`plugin${path.sep}.claude-plugin`), 'plugin.json must be in plugin/.claude-plugin/');

    // marketplace.json at repo .claude-plugin/ (not plugin/)
    const marketplaceFiles = [...plan.keys()].filter(k => k.endsWith('marketplace.json'));
    assert.equal(marketplaceFiles.length, 1);
    // must be in <root>/.claude-plugin/marketplace.json
    const mf = marketplaceFiles[0];
    assert.ok(mf.startsWith(path.join(tmp, '.claude-plugin')), `marketplace.json must be at repo root .claude-plugin/, got ${mf}`);

    // NO commands/ entries
    const commandFiles = [...plan.keys()].filter(k => k.includes(`${path.sep}commands${path.sep}`));
    assert.equal(commandFiles.length, 0, 'no commands/ dir in output');

    // _GENERATED.md banner
    const banners = [...plan.keys()].filter(k => k.endsWith('_GENERATED.md'));
    assert.equal(banners.length, 1, 'exactly 1 _GENERATED.md banner in plugin root');
  });

  test('manifest.targets.plugin validation: throws on missing plugin key', async () => {
    const { loadManifest } = await getModule();
    const tmp = makeTmp();
    const bad = makeManifest({ targets: {} });
    writeFile(tmp, 'plugin-src/manifest.json', JSON.stringify(bad, null, 2));
    assert.throws(() => loadManifest(tmp), /targets\.plugin/i);
  });
});

// ---------------------------------------------------------------------------
// TEST 5 — derivePluginRoot returns plugin/ given targets.plugin
// ---------------------------------------------------------------------------
describe('derivePluginRoot', () => {
  test('derives plugin/ from targets.plugin="plugin/.claude-plugin/plugin.json"', async () => {
    const { derivePluginRoot } = await getModule();
    const manifest = makeManifest();
    const tmp = '/tmp/testrepo';
    const result = derivePluginRoot(tmp, manifest);
    assert.equal(result, path.resolve(tmp, 'plugin'));
  });
});

// ---------------------------------------------------------------------------
// TEST 6 — build-twice = 0 writes (idempotent applyPlan)
// ---------------------------------------------------------------------------
describe('applyPlan idempotent', () => {
  test('second build writes 0 files', async () => {
    const { buildPlan, applyPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    const plan1 = buildPlan(tmp, manifest);
    const wrote1 = applyPlan(plan1);
    assert.ok(wrote1 > 0, 'first build should write something');

    const plan2 = buildPlan(tmp, manifest);
    const wrote2 = applyPlan(plan2);
    assert.equal(wrote2, 0, 'second build must write 0 files (idempotent)');
  });
});

// ---------------------------------------------------------------------------
// TEST 7 — checkPlan: in-sync, stale, missing, extra
// ---------------------------------------------------------------------------
describe('checkPlan', () => {
  test('checkPlan returns empty arrays when in sync', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    applyPlan(buildPlan(tmp, manifest));
    const { stale, missing, extra } = checkPlan(buildPlan(tmp, manifest), tmp, manifest);
    assert.equal(stale.length, 0);
    assert.equal(missing.length, 0);
    assert.equal(extra.length, 0);
  });

  test('checkPlan detects stale file', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    applyPlan(buildPlan(tmp, manifest));
    // tamper with a generated file
    const skillFile = path.join(tmp, 'plugin', 'skills', 'slm-recall', 'SKILL.md');
    fs.writeFileSync(skillFile, 'STALE CONTENT\n', 'utf8');
    const { stale } = checkPlan(buildPlan(tmp, manifest), tmp, manifest);
    assert.ok(stale.length > 0, 'should detect stale file');
  });

  test('checkPlan detects missing file', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    applyPlan(buildPlan(tmp, manifest));
    fs.unlinkSync(path.join(tmp, 'plugin', 'skills', 'slm-recall', 'SKILL.md'));
    const { missing } = checkPlan(buildPlan(tmp, manifest), tmp, manifest);
    assert.ok(missing.length > 0, 'should detect missing file');
  });

  test('checkPlan detects extra orphan file in plugin/', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    applyPlan(buildPlan(tmp, manifest));
    // add orphan in plugin/skills/
    writeFile(tmp, 'plugin/skills/slm-orphan/SKILL.md', '# orphan\n');
    const { extra } = checkPlan(buildPlan(tmp, manifest), tmp, manifest);
    assert.ok(extra.length > 0, 'should detect extra orphan');
  });
});

// ---------------------------------------------------------------------------
// TEST 8 — pruneOrphans: removes orphan, escape guard holds
// ---------------------------------------------------------------------------
describe('pruneOrphans', () => {
  test('prunes removed skill from plugin/, plugin-src untouched', async () => {
    const { buildPlan, applyPlan, pruneOrphans, deriveManagedRoots } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    applyPlan(buildPlan(tmp, manifest));

    // Simulate skill removed: build without slm-status
    const reducedManifest = makeManifest({
      skills: manifest.skills.filter(s => s.name !== 'slm-status'),
    });
    // Copy remaining skill sources (slm-status source stays — plugin-src untouched)
    const reducedPlan = buildPlan(tmp, reducedManifest);
    const roots = deriveManagedRoots(tmp, reducedManifest);
    assert.ok(reducedPlan.size > 0, 'plan must not be empty');

    pruneOrphans(tmp, reducedPlan, roots);

    // slm-status removed from plugin/
    assert.ok(!fs.existsSync(path.join(tmp, 'plugin', 'skills', 'slm-status')));
    // plugin-src untouched
    assert.ok(fs.existsSync(path.join(tmp, 'plugin-src', 'skills', 'slm-status', 'SKILL.md')));
  });

  test('pruneOrphans: escape guard never exits plugin/ root', async () => {
    const { deriveManagedRoots, pruneOrphans } = await getModule();
    const tmp = makeTmp();
    const manifest = makeManifest();
    // Plant sentinel outside managed roots
    writeFile(tmp, 'ide/configs/settings.json', '{"protected":true}');
    const sentinelPath = path.join(tmp, 'ide', 'configs', 'settings.json');
    const before = fs.readFileSync(sentinelPath, 'utf8');

    // Empty plugin/ (no orphans, roots don't exist yet — prune is a no-op)
    const roots = deriveManagedRoots(tmp, manifest);
    // Build a minimal non-empty plan
    const fakePlan = new Map([[path.join(tmp, 'plugin', 'dummy.txt'), 'x\n']]);
    pruneOrphans(tmp, fakePlan, roots);

    // sentinel untouched
    assert.equal(fs.readFileSync(sentinelPath, 'utf8'), before);
  });
});

// ---------------------------------------------------------------------------
// TEST 9 — SLM_MCP_PROFILE=code in .mcp.json
// ---------------------------------------------------------------------------
describe('.mcp.json profile=code', () => {
  test('plan includes .mcp.json with SLM_MCP_PROFILE=code', async () => {
    const { buildPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    const plan = buildPlan(tmp, manifest);

    const mcpEntry = [...plan.entries()].find(([k]) => k.endsWith('.mcp.json') && k.includes(`plugin${path.sep}.mcp`));
    assert.ok(mcpEntry, 'plan must include plugin/.mcp.json');
    const content = JSON.parse(mcpEntry[1]);
    const server = Object.values(content.mcpServers)[0];
    assert.equal(server.env.SLM_MCP_PROFILE, 'code', 'SLM_MCP_PROFILE must be "code"');
  });
});

// ---------------------------------------------------------------------------
// TEST 10 — renderSkillFile golden snapshot
// ---------------------------------------------------------------------------
describe('renderSkillFile', () => {
  test('stamps version 3.6.14 and normalizes OQ-2 attribution', async () => {
    const { renderSkillFile } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const result = renderSkillFile(body, '3.6.14');
    assert.match(result, /version: "3\.6\.14"/);
    assert.match(result, /SuperLocalMemory v3\.6\.14/);
    assert.match(result, /AGPL-3\.0-or-later/);
    assert.match(result, /qualixar\/superlocalmemory/);
    assert.match(result, /API v2/);
    assert.match(result, /Uses the MIT license approach/);
  });

  test('renderSkillFile is idempotent', async () => {
    const { renderSkillFile } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const once = renderSkillFile(body, '3.6.14');
    const twice = renderSkillFile(once, '3.6.14');
    assert.equal(once, twice);
  });
});

// ---------------------------------------------------------------------------
// TEST 11 — main exits 1 on missing manifest
// ---------------------------------------------------------------------------
describe('main exit codes', () => {
  test('main exits 1 when manifest is missing', async () => {
    const tmp = makeTmp();
    const result = spawnSync(
      process.execPath,
      [SCRIPT_PATH, '--quiet'],
      { cwd: tmp, encoding: 'utf8' }
    );
    assert.equal(result.status, 1, 'should exit 1 on missing manifest');
  });
});

// ---------------------------------------------------------------------------
// TEST 12 — LF + single trailing newline
// ---------------------------------------------------------------------------
describe('output format', () => {
  test('generated SKILL.md uses LF and ends with single newline', async () => {
    const { renderSkillFile } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const result = renderSkillFile(body, '3.6.14');
    assert.ok(!result.includes('\r'), 'must not contain CR');
    assert.ok(result.endsWith('\n'), 'must end with newline');
    assert.ok(!result.endsWith('\n\n'), 'must not end with double newline');
  });
});

// ---------------------------------------------------------------------------
// TEST 13 — REGRESSION: marketplace.json source is "./plugin" not "./"
// ---------------------------------------------------------------------------
describe('REGRESSION: marketplace source="./plugin"', () => {
  test('renderMarketplaceJson source is ./plugin (DOC-CORRECT)', async () => {
    const { renderMarketplaceJson } = await getModule();
    const manifest = makeManifest();
    const parsed = JSON.parse(renderMarketplaceJson(manifest));
    assert.equal(parsed.plugins[0].source, './plugin', 'source must be ./plugin per DOC-CORRECT layout');
  });
});

// ---------------------------------------------------------------------------
// TEST 14 — REGRESSION: no ide/ or src/ targets in plan
// ---------------------------------------------------------------------------
describe('REGRESSION: no multi-skill-target', () => {
  test('plan contains NO ide/skills or src/superlocalmemory/skills entries', async () => {
    const { buildPlan } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);
    const plan = buildPlan(tmp, manifest);

    const ideFiles = [...plan.keys()].filter(k => k.includes('ide/skills') || k.includes('ide\\skills'));
    const srcFiles = [...plan.keys()].filter(k =>
      k.includes('src/superlocalmemory/skills') || k.includes('src\\superlocalmemory\\skills')
    );
    assert.equal(ideFiles.length, 0, 'no ide/skills entries allowed');
    assert.equal(srcFiles.length, 0, 'no src/superlocalmemory/skills entries allowed');
  });
});

// ---------------------------------------------------------------------------
// TEST 15 — REGRESSION: partial-target no cross-delete (now: prune scoped to plugin/)
// ---------------------------------------------------------------------------
describe('REGRESSION: prune scoped to plugin/ only', () => {
  test('pruneOrphans only touches plugin/, not other dirs', async () => {
    const { buildPlan, applyPlan, pruneOrphans, deriveManagedRoots } = await getModule();
    const tmp = makeTmp();
    const manifest = setupFixture(tmp);

    // Plant files in a non-managed dir
    writeFile(tmp, 'ide/configs/settings.json', '{"protected":true}');
    writeFile(tmp, 'some-other-dir/file.txt', 'content\n');

    applyPlan(buildPlan(tmp, manifest));

    const plan = buildPlan(tmp, manifest);
    const roots = deriveManagedRoots(tmp, manifest);
    pruneOrphans(tmp, plan, roots);

    // Non-managed dirs untouched
    assert.ok(fs.existsSync(path.join(tmp, 'ide', 'configs', 'settings.json')));
    assert.ok(fs.existsSync(path.join(tmp, 'some-other-dir', 'file.txt')));
  });
});

// ---------------------------------------------------------------------------
// TEST 16 — Full build on real REPO_ROOT succeeds + --check exits 0
// ---------------------------------------------------------------------------
describe('Integration: real repo build', () => {
  test('plugin builder declares its ESM format by extension', () => {
    assert.equal(path.extname(SCRIPT_PATH), '.mjs');
  });

  test('node build-plugin.mjs succeeds on real repo', () => {
    const result = spawnSync(
      process.execPath,
      [SCRIPT_PATH, '--quiet'],
      { cwd: REPO_ROOT, encoding: 'utf8' }
    );
    if (result.status !== 0) {
      console.error('STDOUT:', result.stdout);
      console.error('STDERR:', result.stderr);
    }
    assert.equal(result.status, 0, `build exited ${result.status}: ${result.stderr}`);
    assert.doesNotMatch(
      result.stderr,
      /MODULE_TYPELESS_PACKAGE_JSON/,
      'plugin builder must use an unambiguous Node module format'
    );
  });

  test('--check exits 0 after build', () => {
    const result = spawnSync(
      process.execPath,
      [SCRIPT_PATH, '--check', '--quiet'],
      { cwd: REPO_ROOT, encoding: 'utf8' }
    );
    assert.equal(result.status, 0, `--check exited ${result.status}: ${result.stdout}${result.stderr}`);
  });

  test('plugin/.claude-plugin/plugin.json matches the source manifest version', () => {
    const p = path.join(REPO_ROOT, 'plugin', '.claude-plugin', 'plugin.json');
    const manifestPath = path.join(REPO_ROOT, 'plugin-src', 'manifest.json');
    assert.ok(fs.existsSync(p), 'plugin/.claude-plugin/plugin.json must exist');
    const parsed = JSON.parse(fs.readFileSync(p, 'utf8'));
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    assert.equal(parsed.version, manifest.version);
    assert.equal(parsed.author.name, 'Qualixar');
    assert.ok(parsed.author.url, 'author.url must be present');
  });

  test('.claude-plugin/marketplace.json has source="./plugin"', () => {
    const p = path.join(REPO_ROOT, '.claude-plugin', 'marketplace.json');
    assert.ok(fs.existsSync(p), '.claude-plugin/marketplace.json must exist');
    const parsed = JSON.parse(fs.readFileSync(p, 'utf8'));
    assert.equal(parsed.plugins[0].source, './plugin');
    assert.equal(parsed.plugins[0].version, undefined, 'no version in marketplace plugin entry');
  });

  test('plugin/.mcp.json has SLM_MCP_PROFILE=code', () => {
    const p = path.join(REPO_ROOT, 'plugin', '.mcp.json');
    assert.ok(fs.existsSync(p), 'plugin/.mcp.json must exist');
    const parsed = JSON.parse(fs.readFileSync(p, 'utf8'));
    const server = Object.values(parsed.mcpServers)[0];
    assert.equal(server.env.SLM_MCP_PROFILE, 'code', 'SLM_MCP_PROFILE must be code');
  });

  test('all 7 skills exist in plugin/skills/', () => {
    const skills = ['slm-cache', 'slm-compress', 'slm-graph', 'slm-recall', 'slm-remember', 'slm-session', 'slm-status'];
    for (const s of skills) {
      const p = path.join(REPO_ROOT, 'plugin', 'skills', s, 'SKILL.md');
      assert.ok(fs.existsSync(p), `plugin/skills/${s}/SKILL.md must exist`);
    }
  });

  test('commands/ ships the slm-loop bounded-loop command', () => {
    const cmd = path.join(REPO_ROOT, 'plugin', 'commands', 'slm-loop.md');
    assert.ok(fs.existsSync(cmd), 'plugin/commands/slm-loop.md must exist');
    const body = fs.readFileSync(cmd, 'utf8');
    assert.ok(/bounded loop/i.test(body), 'command must describe a bounded loop');
  });

  test('slm-loop-runner agent ships in plugin/agents/', () => {
    const agent = path.join(REPO_ROOT, 'plugin', 'agents', 'slm-loop-runner.md');
    assert.ok(fs.existsSync(agent), 'plugin/agents/slm-loop-runner.md must exist');
  });
});

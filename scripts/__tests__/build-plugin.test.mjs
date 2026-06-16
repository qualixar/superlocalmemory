/**
 * build-plugin.test.mjs — WP-04 TDD test suite (node:test, mkdtemp fixtures)
 *
 * 18 LLD §6 tests + 2 regression tests (root-skills-unchanged; partial-target-no-cross-delete)
 * + author.url assertion.
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
const SCRIPT_PATH = path.resolve(import.meta.dirname, '../build-plugin.js');

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

// Minimal fixture manifest for test isolation
function makeManifest(overrides = {}) {
  return {
    version: '3.6.14',
    pluginName: 'superlocalmemory',
    skills: [
      { name: 'slm-recall', hasReadme: false },
      { name: 'slm-remember', hasReadme: false },
      { name: 'slm-status', hasReadme: false },
      { name: 'slm-list-recent', hasReadme: false },
      { name: 'slm-switch-profile', hasReadme: false },
      { name: 'slm-build-graph', hasReadme: false },
      { name: 'slm-show-patterns', hasReadme: false },
      { name: 'slm-optimize', hasReadme: true },
    ],
    targets: {
      ide: 'ide/skills',
      pkg: 'src/superlocalmemory/skills',
      plugin: '.claude-plugin/skills',
    },
    marketplace: {
      owner: 'qualixar',
      repo: 'superlocalmemory',
      description: 'SuperLocalMemory skill plugin for Claude Code',
    },
    ...overrides,
  };
}

// Minimal skill body (with old attribution footer)
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

// ---------------------------------------------------------------------------
// TEST 1 — stampVersion replaces only frontmatter version line
// ---------------------------------------------------------------------------
describe('stampVersion', () => {
  test('replaces only frontmatter version line (body v3.4.22 untouched)', async () => {
    const { stampVersion } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.22');
    // body contains "(v3.4.22)" in some text too
    const bodyWithVersion = body + '\n\nSome note about (v3.4.22) API\n';
    const result = stampVersion(bodyWithVersion, '3.6.14');
    // frontmatter version updated
    assert.match(result, /^---\n[\s\S]*?version: "3\.6\.14"/m);
    // body prose reference preserved
    assert.match(result, /\(v3\.4\.22\)/);
  });

  // TEST 2 — idempotent
  test('stampVersion is idempotent', async () => {
    const { stampVersion } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const once = stampVersion(body, '3.6.14');
    const twice = stampVersion(once, '3.6.14');
    assert.equal(once, twice);
  });

  // TEST 3 — throws if no version line in frontmatter
  test('throws if frontmatter has no version line', async () => {
    const { stampVersion } = await getModule();
    const noVersion = '---\nname: foo\n---\n\n# Foo\n';
    assert.throws(() => stampVersion(noVersion, '3.6.14'), /version/i);
  });
});

// ---------------------------------------------------------------------------
// TEST 4 — renderPluginJson deterministic + sorted + author.url preserved
// ---------------------------------------------------------------------------
describe('renderPluginJson', () => {
  test('deterministic sorted output + author.url survives', async () => {
    const { renderPluginJson } = await getModule();
    const manifest = makeManifest();
    const json = renderPluginJson(manifest);
    const parsed = JSON.parse(json);
    // required field
    assert.equal(parsed.name, 'superlocalmemory');
    assert.equal(parsed.version, '3.6.14');
    // REGRESSION: author.url must NOT be silently dropped
    assert.equal(parsed.author.url, 'https://github.com/qualixar');
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
// TEST 5 — renderMarketplaceJson deterministic + sorted
// ---------------------------------------------------------------------------
describe('renderMarketplaceJson', () => {
  test('deterministic sorted + has required fields', async () => {
    const { renderMarketplaceJson } = await getModule();
    const manifest = makeManifest();
    const json = renderMarketplaceJson(manifest);
    const parsed = JSON.parse(json);
    assert.equal(parsed.name, 'superlocalmemory');
    assert.ok(parsed.owner);
    assert.equal(parsed.owner.name, 'qualixar');
    assert.ok(Array.isArray(parsed.plugins));
    assert.equal(parsed.plugins.length, 1);
    assert.equal(parsed.plugins[0].name, 'superlocalmemory');
    assert.equal(parsed.plugins[0].source, './');
  });
});

// ---------------------------------------------------------------------------
// TEST 6 — buildPlan all target = 8×3 + readmes + banners + 2 json
// ---------------------------------------------------------------------------
describe('buildPlan', () => {
  test('all target builds 8x3 skill files + 2 json + banners + readme', async () => {
    const { buildPlan } = await getModule();
    const tmp = makeTmp();
    // create plugin-src structure
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    // agents, commands, rules (skip .gitkeep)
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall cmd\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    const plan = buildPlan(tmp, manifest, 'all');

    // 8 skills × 3 targets = 24 skill files
    const skillFiles = [...plan.keys()].filter(k => k.endsWith('SKILL.md'));
    assert.equal(skillFiles.length, 24);

    // 1 README × 3 targets = 3
    const readmeFiles = [...plan.keys()].filter(k => k.endsWith('README.md'));
    assert.equal(readmeFiles.length, 3);

    // 2 JSON files
    const jsonFiles = [...plan.keys()].filter(k => k.endsWith('.json'));
    assert.equal(jsonFiles.length, 2);

    // 3 _GENERATED.md banners (one per managed root)
    const banners = [...plan.keys()].filter(k => k.endsWith('_GENERATED.md'));
    assert.equal(banners.length, 3);
  });

  // TEST 7 — target=ide scoped (only ide/skills written)
  test('target=ide scopes to ide only', async () => {
    const { buildPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    const plan = buildPlan(tmp, manifest, 'ide');

    // only ide target paths
    const nonIde = [...plan.keys()].filter(k =>
      !k.includes('ide/skills') && !k.endsWith('_GENERATED.md'));
    // non-ide entries: could have JSON for plugin target but ide-only should have none
    // ide target = only ide/skills
    const ideFiles = [...plan.keys()].filter(k => k.includes('ide/skills'));
    assert.ok(ideFiles.length > 0, 'ide/skills must have entries');

    // must NOT have src/ or .claude-plugin/skills/ entries
    const srcFiles = [...plan.keys()].filter(k => k.includes('src/superlocalmemory/skills'));
    const pluginFiles = [...plan.keys()].filter(k => k.includes('.claude-plugin/skills'));
    assert.equal(srcFiles.length, 0, 'no src files for ide-only target');
    assert.equal(pluginFiles.length, 0, 'no .claude-plugin files for ide-only target');
  });
});

// ---------------------------------------------------------------------------
// TEST 8 — GOLDEN snapshot: rendered SKILL.md has correct version + attribution
// ---------------------------------------------------------------------------
describe('renderSkillFile', () => {
  test('stamps version 3.6.14 and normalizes OQ-2 attribution', async () => {
    const { renderSkillFile } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const result = renderSkillFile(body, '3.6.14');

    // version stamped
    assert.match(result, /version: "3\.6\.14"/);
    // V2 → v3.6.14 in attribution footer
    assert.match(result, /SuperLocalMemory v3\.6\.14/);
    // MIT → AGPL-3.0-or-later in attribution footer
    assert.match(result, /AGPL-3\.0-or-later/);
    // repo updated in attribution footer
    assert.match(result, /qualixar\/superlocalmemory/);
    // body prose NOT mutated ("API v2" should be untouched)
    assert.match(result, /API v2/);
    // "Uses the MIT license approach" in body NOT mutated
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
// TEST 9 — build-twice = 0 writes (idempotent applyPlan)
// ---------------------------------------------------------------------------
describe('applyPlan idempotent', () => {
  test('second build writes 0 files', async () => {
    const { buildPlan, applyPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    const plan1 = buildPlan(tmp, manifest, 'all');
    const wrote1 = applyPlan(plan1);
    assert.ok(wrote1 > 0, 'first build should write something');

    const plan2 = buildPlan(tmp, manifest, 'all');
    const wrote2 = applyPlan(plan2);
    assert.equal(wrote2, 0, 'second build must write 0 files (idempotent)');
  });
});

// ---------------------------------------------------------------------------
// TEST 10 — --check exits 0 when in-sync
// ---------------------------------------------------------------------------
describe('checkPlan', () => {
  test('checkPlan returns empty arrays when in sync', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    applyPlan(buildPlan(tmp, manifest, 'all'));

    const { stale, missing, extra } = checkPlan(buildPlan(tmp, manifest, 'all'), tmp, manifest, 'all');
    assert.equal(stale.length, 0);
    assert.equal(missing.length, 0);
    assert.equal(extra.length, 0);
  });

  // TEST 11 — --check exits 2 when stale
  test('checkPlan detects stale file', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    applyPlan(buildPlan(tmp, manifest, 'all'));
    // tamper with a generated file
    const ideSkill = path.join(tmp, 'ide/skills/slm-recall/SKILL.md');
    fs.writeFileSync(ideSkill, 'STALE CONTENT\n', 'utf8');

    const { stale } = checkPlan(buildPlan(tmp, manifest, 'all'), tmp, manifest, 'all');
    assert.ok(stale.length > 0, 'should detect stale file');
  });

  // TEST 12 — --check detects missing file
  test('checkPlan detects missing file', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    applyPlan(buildPlan(tmp, manifest, 'all'));
    // delete a file
    fs.unlinkSync(path.join(tmp, 'ide/skills/slm-recall/SKILL.md'));

    const { missing } = checkPlan(buildPlan(tmp, manifest, 'all'), tmp, manifest, 'all');
    assert.ok(missing.length > 0, 'should detect missing file');
  });

  // TEST 13 — --check detects extra/orphan
  test('checkPlan detects extra orphan file', async () => {
    const { buildPlan, applyPlan, checkPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    applyPlan(buildPlan(tmp, manifest, 'all'));
    // add orphan file
    writeFile(tmp, 'ide/skills/slm-orphan/SKILL.md', '# orphan\n');

    const { extra } = checkPlan(buildPlan(tmp, manifest, 'all'), tmp, manifest, 'all');
    assert.ok(extra.length > 0, 'should detect extra orphan');
  });
});

// ---------------------------------------------------------------------------
// TEST 14 — prune removes removed skill, escape guard never exits managed roots
// ---------------------------------------------------------------------------
describe('pruneOrphans', () => {
  test('prunes removed skill but guard never exits managed roots', async () => {
    const { buildPlan, applyPlan, pruneOrphans, deriveManagedRoots } = await getModule();
    const tmp = makeTmp();
    const allSkills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                       'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of allSkills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    applyPlan(buildPlan(tmp, manifest, 'all'));

    // Simulate a skill being removed: build plan without slm-status
    const reducedManifest = makeManifest({
      skills: manifest.skills.filter(s => s.name !== 'slm-status'),
    });
    const reducedPlan = buildPlan(tmp, reducedManifest, 'all');
    const roots = deriveManagedRoots(tmp, reducedManifest, 'all');

    // must not prune when plan is empty (guard)
    assert.ok(reducedPlan.size > 0, 'plan must not be empty');

    pruneOrphans(tmp, reducedPlan, roots);

    // slm-status should be removed from all managed dirs
    assert.ok(!fs.existsSync(path.join(tmp, 'ide/skills/slm-status')));
    assert.ok(!fs.existsSync(path.join(tmp, 'src/superlocalmemory/skills/slm-status')));
    assert.ok(!fs.existsSync(path.join(tmp, '.claude-plugin/skills/slm-status')));

    // plugin-src must be untouched (not a managed root)
    assert.ok(fs.existsSync(path.join(tmp, 'plugin-src/skills/slm-status/SKILL.md')));
  });
});

// ---------------------------------------------------------------------------
// TEST 15 — does NOT touch ide/configs (AC-8)
// ---------------------------------------------------------------------------
describe('AC-8 non-touch', () => {
  test('build does not touch ide/configs', async () => {
    const { buildPlan, applyPlan, pruneOrphans, deriveManagedRoots } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    // plant a sentinel in ide/configs
    writeFile(tmp, 'ide/configs/settings.json', '{"protected":true}');
    const sentinelContent = readFile(path.join(tmp, 'ide/configs/settings.json'));

    const manifest = makeManifest();
    const plan = buildPlan(tmp, manifest, 'all');
    applyPlan(plan);
    const roots = deriveManagedRoots(tmp, manifest, 'all');
    pruneOrphans(tmp, plan, roots);

    // sentinel must be unchanged
    const afterContent = readFile(path.join(tmp, 'ide/configs/settings.json'));
    assert.equal(afterContent, sentinelContent);
  });
});

// ---------------------------------------------------------------------------
// TEST 16 — README verbatim no-stamp
// ---------------------------------------------------------------------------
describe('README verbatim', () => {
  test('README.md is copied verbatim without version stamp', async () => {
    const { buildPlan, applyPlan } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    const readmeContent = '# README\n\nversion: "OLD"\n\nSome text.\n';
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', readmeContent);
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();
    applyPlan(buildPlan(tmp, manifest, 'all'));

    // Check one generated README
    const genReadme = path.join(tmp, 'ide/skills/slm-optimize/README.md');
    const content = readFile(genReadme);
    // must be identical to source (no stamp applied)
    assert.equal(content, readmeContent);
  });
});

// ---------------------------------------------------------------------------
// TEST 17 — main exits 1 on missing manifest
// ---------------------------------------------------------------------------
describe('main exit codes', () => {
  test('main exits 1 when manifest is missing', async () => {
    const tmp = makeTmp();
    const result = spawnSync(
      process.execPath,
      [SCRIPT_PATH, '--target', 'all', '--quiet'],
      { cwd: tmp, encoding: 'utf8' }
    );
    assert.equal(result.status, 1, 'should exit 1 on missing manifest');
  });
});

// ---------------------------------------------------------------------------
// TEST 18 — LF + single trailing newline
// ---------------------------------------------------------------------------
describe('output format', () => {
  test('generated SKILL.md uses LF and ends with single newline', async () => {
    const { renderSkillFile, normalizeNewlines } = await getModule();
    const body = makeSkillBody('slm-recall', '3.4.23');
    const result = renderSkillFile(body, '3.6.14');
    // no CRLF
    assert.ok(!result.includes('\r'), 'must not contain CR');
    // ends with exactly one newline
    assert.ok(result.endsWith('\n'), 'must end with newline');
    assert.ok(!result.endsWith('\n\n'), 'must not end with double newline');
  });
});

// ---------------------------------------------------------------------------
// REGRESSION TEST A — root skills/ BYTE-IDENTICAL after full build
// ---------------------------------------------------------------------------
describe('REGRESSION: root skills/ read-only', () => {
  test('full build leaves root skills/*/SKILL.md byte-identical (BUG-2 regression)', async () => {
    // Capture sha256 of all root skills SKILL.md before invoking build
    const rootSkillsDir = path.join(REPO_ROOT, 'skills');
    const skillDirs = fs.readdirSync(rootSkillsDir).filter(d =>
      fs.statSync(path.join(rootSkillsDir, d)).isDirectory()
    );

    const before = {};
    for (const d of skillDirs) {
      const p = path.join(rootSkillsDir, d, 'SKILL.md');
      if (fs.existsSync(p)) {
        before[p] = sha256(fs.readFileSync(p));
      }
    }

    // Run the full build (uses real REPO_ROOT)
    const result = spawnSync(
      process.execPath,
      [SCRIPT_PATH, '--target', 'all', '--quiet'],
      { cwd: REPO_ROOT, encoding: 'utf8' }
    );

    // Build should succeed
    if (result.status !== 0) {
      console.error('STDOUT:', result.stdout);
      console.error('STDERR:', result.stderr);
    }
    assert.equal(result.status, 0, `build exited ${result.status}: ${result.stderr}`);

    // Check all root SKILL.md files are unchanged
    const after = {};
    for (const d of skillDirs) {
      const p = path.join(rootSkillsDir, d, 'SKILL.md');
      if (fs.existsSync(p)) {
        after[p] = sha256(fs.readFileSync(p));
      }
    }

    for (const [p, hash] of Object.entries(before)) {
      assert.equal(after[p], hash, `REGRESSION BUG-2: root ${p} was modified by build`);
    }
  });
});

// ---------------------------------------------------------------------------
// REGRESSION TEST B — partial target (--target ide) leaves src/ and .claude-plugin/ untouched
// ---------------------------------------------------------------------------
describe('REGRESSION: partial target no cross-delete (BUG-1 regression)', () => {
  test('--target ide leaves src/ and .claude-plugin/ skills byte-unchanged', async () => {
    const { buildPlan, applyPlan, pruneOrphans, deriveManagedRoots } = await getModule();
    const tmp = makeTmp();
    const skills = ['slm-recall', 'slm-remember', 'slm-status', 'slm-list-recent',
                    'slm-switch-profile', 'slm-build-graph', 'slm-show-patterns', 'slm-optimize'];
    for (const s of skills) {
      writeFile(tmp, `plugin-src/skills/${s}/SKILL.md`, makeSkillBody(s));
    }
    writeFile(tmp, 'plugin-src/skills/slm-optimize/README.md', '# README\n');
    writeFile(tmp, 'plugin-src/agents/slm-memory-advisor.md', '# Advisor\n');
    writeFile(tmp, 'plugin-src/commands/slm-recall.md', '# recall\n');
    writeFile(tmp, 'plugin-src/rules/AGENTS.md', '# AGENTS\n');
    writeFile(tmp, 'plugin-src/hooks/.gitkeep', '');

    const manifest = makeManifest();

    // First: build ALL to populate src/ and .claude-plugin/
    applyPlan(buildPlan(tmp, manifest, 'all'));

    // Capture hashes of src/ and .claude-plugin/ skills
    const srcRoot = path.join(tmp, 'src/superlocalmemory/skills');
    const pluginRoot = path.join(tmp, '.claude-plugin/skills');

    function captureHashes(dir) {
      const hashes = {};
      if (!fs.existsSync(dir)) return hashes;
      for (const d of fs.readdirSync(dir)) {
        const sub = path.join(dir, d);
        if (!fs.statSync(sub).isDirectory()) continue;
        for (const f of fs.readdirSync(sub)) {
          const fp = path.join(sub, f);
          hashes[fp] = sha256(fs.readFileSync(fp));
        }
      }
      return hashes;
    }

    const srcBefore = captureHashes(srcRoot);
    const pluginBefore = captureHashes(pluginRoot);

    // Now build with --target ide ONLY (prune only ide/skills)
    const idePlan = buildPlan(tmp, manifest, 'ide');
    applyPlan(idePlan);
    const ideRoots = deriveManagedRoots(tmp, manifest, 'ide');
    pruneOrphans(tmp, idePlan, ideRoots);

    // src/ and .claude-plugin/ must be unchanged
    const srcAfter = captureHashes(srcRoot);
    const pluginAfter = captureHashes(pluginRoot);

    for (const [p, h] of Object.entries(srcBefore)) {
      assert.equal(srcAfter[p], h, `REGRESSION BUG-1: src/ file ${p} was modified by --target ide`);
      assert.ok(srcAfter[p] !== undefined, `REGRESSION BUG-1: src/ file ${p} was deleted by --target ide`);
    }
    for (const [p, h] of Object.entries(pluginBefore)) {
      assert.equal(pluginAfter[p], h, `REGRESSION BUG-1: .claude-plugin/ file ${p} was modified by --target ide`);
      assert.ok(pluginAfter[p] !== undefined, `REGRESSION BUG-1: .claude-plugin/ file ${p} was deleted by --target ide`);
    }
  });
});

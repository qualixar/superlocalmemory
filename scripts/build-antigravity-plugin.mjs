#!/usr/bin/env node
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// build-antigravity-plugin.mjs — generate antigravity-plugin/ from plugin-src/.
//
// USAGE
//   node scripts/build-antigravity-plugin.mjs           # write
//   node scripts/build-antigravity-plugin.mjs --check   # verify in sync (exit 2)
//
// WHY
//   There was no Antigravity tree at all, so on that host SLM had no skills, no
//   agents, no commands and no hooks — nothing. Every other surface is generated
//   from plugin-src/; this one was simply missing, which is the least visible way
//   for a distribution channel to be broken.
//
//   Derived from the SAME source as the other three so the four cannot drift.

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';

const ROOT = join(dirname(new URL(import.meta.url).pathname), '..');
const OUT = join(ROOT, 'antigravity-plugin');
const SRC = join(ROOT, 'plugin-src');
const AGENT_ID = 'antigravity';

const manifest = JSON.parse(readFileSync(join(SRC, 'manifest.json'), 'utf8'));
const VERSION = manifest.version;
if (!VERSION) throw new Error('manifest.version missing');

const check = process.argv.includes('--check');
const files = new Map();

const nl = (s) => s.replace(/\n*$/, '\n');
/** Point the memory agent id at this host, as the other builders do. */
const retarget = (s) => s.replace(/SLM_AGENT_ID["']?\s*[:=]\s*["'][^"']+["']/g,
  `SLM_AGENT_ID="${AGENT_ID}"`);
/** Keep every stated version equal to the manifest's. */
function stamp(md) {
  let out = md.replace(/SuperLocalMemory v\d+\.\d+\.\d+/g, `SuperLocalMemory v${VERSION}`);
  const fm = out.match(/^(---\n)([\s\S]*?)(\n---\n)/);
  if (fm && /^version:/m.test(fm[2])) {
    out = out.replace(fm[0], fm[1] + fm[2].replace(/^version:.*$/m, `version: "${VERSION}"`) + fm[3]);
  }
  return out;
}

// skills/, agents/, commands/ — one source, retargeted.
for (const [sub, out] of [['skills', 'skills'], ['agents', 'agents'], ['commands', 'commands']]) {
  const dir = join(SRC, sub);
  if (!existsSync(dir)) continue;
  if (sub === 'skills') {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      const f = join(dir, e.name, 'SKILL.md');
      if (!existsSync(f)) continue;
      files.set(`${out}/${e.name}/SKILL.md`, nl(stamp(retarget(readFileSync(f, 'utf8')))));
    }
  } else {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      if (!e.isFile() || !e.name.endsWith('.md')) continue;
      files.set(`${out}/${e.name}`, nl(stamp(retarget(readFileSync(join(dir, e.name), 'utf8')))));
    }
  }
}

// Keep the legacy hook tree for already-installed packs, but also emit the
// root-level `hooks.json` required by current Antigravity. Its events and tool
// names are host-native; a Codex-shaped `SessionStart` hook is not discoverable
// by Antigravity 2.x.
{
  const codexHooks = join(ROOT, 'codex-plugin', 'hooks', 'hooks.json');
  if (existsSync(codexHooks)) {
    files.set('hooks/hooks.json', nl(readFileSync(codexHooks, 'utf8')));
  }
}

files.set('hooks.json', JSON.stringify({
  'slm-lifecycle': {
    PreInvocation: [
      { type: 'command', command: 'python3 ./scripts/antigravity_hook_adapter.py pre-invocation', timeout: 15 },
    ],
    PreToolUse: [{
      matcher: 'web_search|web_fetch',
      hooks: [{ type: 'command', command: 'python3 ./scripts/antigravity_hook_adapter.py pre-tool', timeout: 5 }],
    }],
    PostToolUse: [{
      matcher: 'view_file|write_to_file|replace_file_content|run_command|search_files|web_fetch|web_search|invoke_subagent|generate_image',
      hooks: [{ type: 'command', command: 'python3 ./scripts/antigravity_hook_adapter.py post-tool', timeout: 5 }],
    }],
    Stop: [{ type: 'command', command: 'python3 ./scripts/antigravity_hook_adapter.py stop', timeout: 10 }],
  },
}, null, 2) + '\n');

// Antigravity communicates with hooks through a JSON protocol. SLM's shared
// lifecycle hooks intentionally emit host-neutral text, so this tiny stdlib
// adapter converts that output to Antigravity's documented response shapes.
// It invokes the installed `slm` executable instead of importing package code
// through an arbitrary Python interpreter.
files.set('scripts/antigravity_hook_adapter.py', `#!/usr/bin/env python3
import json
import hashlib
import os
import subprocess
import sys
import tempfile


def payload():
    try:
        value = json.load(sys.stdin)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def slm(action, data):
    workspace_paths = data.get("workspacePaths")
    env = dict(os.environ)
    if isinstance(workspace_paths, list) and workspace_paths and isinstance(workspace_paths[0], str):
        env["CLAUDE_PROJECT_DIR"] = workspace_paths[0]
    try:
        result = subprocess.run(
            ["slm", "hook", action], input=json.dumps(data), text=True,
            capture_output=True, timeout=8, env=env,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def lifecycle_marker(data):
    conversation_id = data.get("conversationId")
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), "slm-antigravity-" + digest)


def main():
    event = sys.argv[1] if len(sys.argv) == 2 else ""
    data = payload()
    if event == "pre-invocation":
        marker = lifecycle_marker(data)
        if marker and os.path.exists(marker):
            print(json.dumps({"injectSteps": []}))
            return
        context = "\\n\\n".join(part for part in (slm("mandate", data), slm("start", data)) if part)
        if marker:
            try:
                with open(marker, "x", encoding="utf-8"):
                    pass
            except FileExistsError:
                print(json.dumps({"injectSteps": []}))
                return
        print(json.dumps({"injectSteps": [{"ephemeralMessage": context}]} if context else {"injectSteps": []}))
    elif event == "pre-tool":
        tool_call = data.get("toolCall") if isinstance(data.get("toolCall"), dict) else {}
        slm("before_web", {"tool_input": tool_call.get("args", {})})
        print(json.dumps({"decision": "allow"}))
    elif event == "post-tool":
        tool_call = data.get("toolCall") if isinstance(data.get("toolCall"), dict) else {}
        slm("post_tool_outcome", {
            "tool_name": tool_call.get("name", ""),
            "tool_input": tool_call.get("args", {}),
            "tool_response": data.get("toolResponse", data.get("error", "")),
        })
        print("{}")
    elif event == "stop":
        slm("stop", data)
        slm("stop_outcome", data)
        marker = lifecycle_marker(data)
        if marker:
            try:
                os.remove(marker)
            except OSError:
                pass
        print(json.dumps({"decision": "allow"}))
    else:
        print("{}")


if __name__ == "__main__":
    main()
`);

// mcp_config.json — how Antigravity starts the server.
files.set('mcp_config.json', JSON.stringify({
  mcpServers: {
    superlocalmemory: {
      command: 'slm',
      args: ['mcp'],
      // Antigravity is a content-production primary surface. Give it the
      // full, opt-in power profile while keeping the operator's store path
      // untouched and the host identity explicit.
      env: {
        SLM_AGENT_ID: AGENT_ID,
        SLM_MCP_PROFILE: 'power',
        SLM_MCP_ALL_TOOLS: '1',
      },
    },
  },
}, null, 2) + '\n');

// plugin.json — the manifest Antigravity reads to list this plugin.
files.set('plugin.json', JSON.stringify({
  name: 'superlocalmemory',
  version: VERSION,
  description:
    'Local-first agent memory with auditable hybrid retrieval. Remember '
    + 'decisions and recall them by asking, entirely on your machine.',
  author: {
    name: 'Qualixar',
    email: 'varun.pratap.bhardwaj@gmail.com',
    url: 'https://github.com/qualixar',
  },
  homepage: 'https://qualixar.com',
  repository: 'https://github.com/qualixar/superlocalmemory',
  license: 'AGPL-3.0-or-later',
  keywords: ['memory', 'mcp', 'agents', 'local-first', 'context-compression'],
  skills: './skills/',
  agents: './agents/',
  commands: './commands/',
  hooks: './hooks/hooks.json',
  mcpServers: './mcp_config.json',
}, null, 2) + '\n');

files.set('_GENERATED.md',
  '# antigravity-plugin/ — GENERATED\n\n'
  + 'Built by `scripts/build-antigravity-plugin.mjs` from `plugin-src/`. '
  + `Version stamped from \`plugin-src/manifest.json\`.\n\nVersion: **${VERSION}**\n\n`
  + 'Do not edit by hand — regenerate instead.\n\n'
  + [...files.keys()].sort().map((p) => `- \`${p}\``).join('\n') + '\n');

let drift = 0;
for (const [rel, content] of files) {
  const path = join(OUT, rel);
  if (check) {
    if (!existsSync(path) || readFileSync(path, 'utf8') !== content) {
      console.error(`drift: ${rel}`);
      drift += 1;
    }
    continue;
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content);
}
if (check) {
  if (drift) { console.error(`antigravity-plugin: ${drift} file(s) out of sync.`); process.exit(2); }
  console.log('antigravity-plugin: in sync.');
} else {
  console.log(`antigravity-plugin built: ${files.size} files, v${VERSION}.`);
}

// agent-tool-names.mjs — how each host spells an SLM tool in a sub-agent's `tools:` line.
//
// plugin-src/agents/*.md is the single source and names SLM tools bare
// (`recall`, `remember`, ...). Hosts that read that tree directly (Codex,
// Antigravity, Hermes, Copilot) keep the bare names. Hosts that read the built
// plugin/ tree need the full MCP name:
//
//   Claude Code, SLM installed as a plugin:  mcp__plugin_<plugin>_<server>__<tool>
//   Claude Code, SLM added as a plain MCP:   mcp__<server>__<tool>
//
// Claude Code matches a sub-agent's `tools:` entries by full name only
// (https://code.claude.com/docs/en/sub-agents and /en/mcp), so a bare `recall`
// resolves to nothing and the agent never receives the memory tools. The plugin
// form replaces every character outside A-Z a-z 0-9 _ - with `_`.
//
// Entries that do not resolve are ignored while at least one resolves, so the
// Claude Code build lists both spellings and one install route or the other
// matches. Capitalised names (Read, Bash) are host built-ins and never change.

export const SERVER_KEY = 'superlocalmemory';

// A bare SLM tool: lower-case, and not already a full `mcp__...` name.
const BARE = /^(?!mcp__)[a-z][a-z0-9_]*$/;
const PLUGIN_FORM = /^mcp__plugin_/;

const sanitize = (s) => String(s).replace(/[^A-Za-z0-9_-]/g, '_');

/** `mcp__plugin_<plugin>_<server>__` — the prefix of a plugin-bundled server's tools. */
export function pluginToolPrefix(pluginName, serverKey = SERVER_KEY) {
  return `mcp__plugin_${sanitize(pluginName)}_${sanitize(serverKey)}__`;
}

/** `mcp__<server>__` — the prefix of a directly added MCP server's tools. */
export function directToolPrefix(serverKey = SERVER_KEY) {
  return `mcp__${sanitize(serverKey)}__`;
}

/** Apply `fn(entries) -> entries` to the front-matter `tools:` line; no-op when there is none. */
function mapToolsLine(md, fn) {
  const fm = md.match(/^---\n[\s\S]*?\n---\n/);
  if (!fm) return md;
  const head = fm[0].replace(/^tools:(.*)$/m, (_, list) => {
    const entries = list.split(',').map((t) => t.trim()).filter(Boolean);
    return 'tools: ' + [...new Set(fn(entries))].join(', ');
  });
  return head + md.slice(fm[0].length);
}

/** Claude Code: each bare tool becomes its plugin-bundled and direct-MCP full names. */
export function claudeCodeAgentTools(md, pluginName, serverKey = SERVER_KEY) {
  const plugin = pluginToolPrefix(pluginName, serverKey);
  const direct = directToolPrefix(serverKey);
  return mapToolsLine(md, (entries) => entries.flatMap((t) => (
    BARE.test(t) ? [`${direct}${t}`, `${plugin}${t}`] : [t]
  )));
}

/** Hosts that add SLM as a plain MCP server (Muse): the direct full name only. */
export function directAgentTools(md, serverKey = SERVER_KEY) {
  const direct = directToolPrefix(serverKey);
  return mapToolsLine(md, (entries) => entries
    .filter((t) => !PLUGIN_FORM.test(t))
    .map((t) => (BARE.test(t) ? `${direct}${t}` : t)));
}

/** Hosts with their own tool naming (VS Code agent plugin): back to the bare source names. */
export function bareAgentTools(md, serverKey = SERVER_KEY) {
  const direct = directToolPrefix(serverKey);
  return mapToolsLine(md, (entries) => entries
    .filter((t) => !PLUGIN_FORM.test(t))
    .map((t) => (t.startsWith(direct) ? t.slice(direct.length) : t)));
}

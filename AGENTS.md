<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **superlocalmemory** (41723 symbols, 80264 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/superlocalmemory/context` | Codebase overview, check index freshness |
| `gitnexus://repo/superlocalmemory/clusters` | All functional areas |
| `gitnexus://repo/superlocalmemory/processes` | All execution flows |
| `gitnexus://repo/superlocalmemory/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## SuperLocalMemory 4.2.30 commands

```bash
slm flags                        # every 4.2.30 flag and its state; `slm flags set <flag> on|shadow|off`; `slm flags kill on` switches every 4.2.30 subsystem off at once
slm disputes list                # open disputes and changed source files; `settle`, `reopen`, `still-true`
slm audit --list --json          # blind audit questions to label with your own model id (`--answer ITEM_ID LABEL --model ID`)
slm proposals list               # proposed memory changes; `show`, `apply`, `reject`, `rollback`
slm wiki status                  # compiled pages and edits waiting; `diff`, `build`, `rebuild`, `sync`, `dir`
slm import preview FILE          # what a file holds, nothing saved; `run`, `status`, `list`, `cancel`, `undo`
slm trust pending                # memories held back from becoming instructions; `promote`, `withdraw`, `status`
slm why FACT_ID --json           # where a memory came from and when agents were shown it; read-only
slm signing status               # is signing available; `resign` signs again what a lost key signed
slm measurement status --json    # what SLM measures about whether memory helps; `on` and `off` only when the user asks
slm cues list --json             # the cues that bring memories back, with the memory each belongs to
```

Run `slm <command> --help` for the exact options.

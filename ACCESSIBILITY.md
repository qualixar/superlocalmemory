# Accessibility

SuperLocalMemory should be usable by everyone who builds with AI agents, including people who use screen readers, keyboard-only navigation, magnification or reduced motion. This file says what we do today, where we fall short, and how to tell us about a barrier.

## What you interact with

- **The `slm` command line.** Output is plain text. Most commands accept `--json` for structured output that scripts and assistive tools can parse instead of reading formatted tables.
- **MCP tools inside your agent.** Your agent host (Claude Code, Codex, Cursor and others) renders these, so its own accessibility support applies.
- **The local web dashboard** (`slm dashboard`), which runs in your own browser.

## What the dashboard does today

- Interactive controls carry accessible names (`aria-label`) and landmark roles where the visual label alone isn't enough.
- Keyboard focus is visible (`:focus-visible` styles).
- Animations respect your system's reduced-motion setting (`prefers-reduced-motion`).
- Light and dark appearance follow your system setting (`prefers-color-scheme`).
- Your browser's zoom and text-size settings work as usual.

## Known limitations

- The dashboard has not had a formal audit against [WCAG 2.2](https://www.w3.org/TR/WCAG22/) yet, so we don't claim conformance at any level.
- Some data views (graphs and charts) are visual first and don't yet have a full text alternative. The same data is available from the CLI with `--json`.
- Supported environments: the latest versions of Chrome, Edge, Firefox and Safari on macOS (Apple Silicon), Windows and Linux.

## Report a barrier

Open an [issue](https://github.com/qualixar/superlocalmemory/issues/new/choose) and add "accessibility" to the title, or email admin@superlocalmemory.com if you'd rather not post publicly. Tell us the page or command, what you were trying to do, and the assistive technology and browser you use. Accessibility bugs are treated as bugs, not feature requests.

# GitHub Copilot adapter

Read and follow [the canonical repository instructions](/AGENTS.md) before proposing a change. They define architecture, public API, validation, Git discipline, artifacts, and security for every file.

When a target path has a nested `AGENTS.md`, apply it as a local delta. Files under `.github/instructions/` add only the matching `applyTo` file-pattern rules; they do not replace or weaken the root contract.

For extension work, use exactly one canonical guide linked from `/AGENTS.md` and inspect the current implementation and nearest tests before generating code. Keep suggestions bounded to the requested task, preserve unrelated edits, and state which validation remains to run.

Useful starting points are [testing](/docs/dev/testing.md), [architecture health](/docs/dev/architecture_health.md), and [the extension index](/docs/topics/extending.md).

```agent-docs
path: AGENTS.md
path: docs/dev/testing.md
path: docs/dev/architecture_health.md
path: docs/topics/extending.md
```

# Philharmonic (`pro-orchestrator`)

A zero-dependency **Antigravity (`agy`) plugin** that enforces a Pro-tier Central Orchestrator workflow with `flash` and `flash_lite` worker subagents and deterministic **ASD-STE100 (Simplified Technical English)** prompt linting.

---

## Features

- **Strict Tiered Delegation**: Pins the root session as the Central Orchestrator (`pro`) and delegates file discovery, research, and bounded code changes to `flash_lite` (`scout`) and `flash` (`research`, `implementer`) subagents.
- **Deterministic Guard Hooks**: Four lifecycle hooks (`pre-invocation`, `pre-tool`, `post-tool`, `stop`) implemented in [orchestrator_guard.py](.agents/plugins/pro-orchestrator/scripts/orchestrator_guard.py) prevent direct workspace mutations by the orchestrator, block `Model: "inherit"` / `Model: "pro"` subagent spawns, and track active subagents and branch worktrees.
- **ASD-STE100 Prompt Linter Gate**: Every `invoke_subagent` and `send_message` payload is checked by [ste_lint.py](.agents/plugins/pro-orchestrator/scripts/ste_lint.py) before execution to reject ambiguous instructions.
- **Zero External Dependencies**: Uses only the Python 3 standard library.

---

## Prerequisites

- **Google Antigravity** (`agy` CLI or Antigravity IDE)
- **Python 3.8+** (`python3` available on `PATH`)

---

## Quick Start

### 1. Run directly inside this repository

Clone the repository and launch Antigravity inside it:

```bash
git clone https://github.com/leafy7382/philharmonic.git
cd philharmonic
agy
```

Antigravity automatically discovers [AGENTS.md](AGENTS.md) and loads [.agents/plugins/pro-orchestrator/plugin.json](.agents/plugins/pro-orchestrator/plugin.json).

### 2. Install into another project workspace

Copy the plugin and operating protocol into your target repository:

```bash
mkdir -p /path/to/your-project/.agents/plugins
cp -r .agents/plugins/pro-orchestrator /path/to/your-project/.agents/plugins/
cp AGENTS.md /path/to/your-project/AGENTS.md
```

### 3. Install globally for all projects

Symlink or copy the plugin into your user-level Antigravity plugins directory:

```bash
mkdir -p ~/.gemini/config/plugins
ln -s "$(pwd)/.agents/plugins/pro-orchestrator" ~/.gemini/config/plugins/pro-orchestrator
```

---

## Subagent Architecture & Automatic Token Optimization

To minimize both Pro-tier context growth and subagent spawn overhead:
- **Direct Execution Fast-Path**: Single-step commands and reads/edits of $\le 2$ known files execute directly in the orchestrator with zero subagent overhead.
- **Lazy Subagent Registration**: Custom subagents (`scout` and `implementer`) are registered via `define_subagent` only immediately before first invoking them.
- **Automatic Tier Downgrade**: The `pre-tool` hook automatically downgrades read-only subagent calls (`scout` / `research`) with $\le 3$ numbered rules and no mutation verbs to `Model: "flash_lite"`.

| Subagent Type | Model Tier | Write Tools | Workspace Mode | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Disabled | `inherit` | Single-hop lookups, file/symbol extraction, log slicing ($\le 3$ rules) |
| `research` | `flash` (auto `flash_lite` if $\le 3$ rules) | Disabled | `inherit` | Multi-file analysis and documentation lookups ($> 3$ rules) |
| `implementer` | `flash` | Enabled | `branch` / `share` | Bounded code edits and test execution |

---

## ASD-STE100 Prompt Contract

All subagent prompts (`invoke_subagent` and `send_message`) must follow the strict structure below and pass [ste_lint.py](.agents/plugins/pro-orchestrator/scripts/ste_lint.py):

```markdown
TASK: <single-sentence objective>

RULES:
1. Open `<file_path>`.
2. <imperative instruction under 20 words>.
3. Do not change any file outside `<allowed_files>`.
4. Run `<test_command>` to check the result.
5. On failure, write raw output to `<artifactDirectoryPath>/scratch/<task_id>.log`.

REPORT FORMAT:
Return only these four lines:
STATUS: COMPLETED | BLOCKED | FAILED
FILES_CHANGED: <comma-separated paths or NONE>
VERIFICATION: <one sentence stating the command run and exit result>
NOTES: <one sentence stating findings, or file:// log URI if BLOCKED/FAILED>
```

### Enforced Lint Checks

- **Sentence Length**: Maximum 20 words per numbered rule (`[long-sentence]`).
- **No Semicolons**: Split compound clauses into separate numbered rules (`[semicolon]`).
- **No Soft Phrasal Verbs**: Replace phrases like *figure out*, *look into*, or *spin up* with direct verbs (`[phrasal-verb]`).
- **No Hidden Nominalizations**: Use active imperative verbs instead of noun-heavy constructions (`[nominalization]`).
- **No Synonym Rotation**: Use one consistent verb for the same action throughout a prompt (`[synonym-rotation]`).

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/pro-orchestrator/hooks.json) and executed by [orchestrator_guard.py](.agents/plugins/pro-orchestrator/scripts/orchestrator_guard.py):

1. **`pre-invocation`**: Verifies the active session is pinned to a Pro-tier model and deduplicates warnings per invocation.
2. **`pre-tool`**:
   - Auto-allows workspace, worktree, and scratch artifact file/command operations.
   - Blocks `Model: "pro"`, rewrites `Model: "inherit"`, and automatically downgrades simple read-only subagent tasks ($\le 3$ rules) to `Model: "flash_lite"`.
   - Runs the STE-100 linter gate on all `invoke_subagent` and `send_message` prompts.
   - Deduplicates `pending_spawns` tracking across multiple hook registrations.
3. **`post-tool`**: Synchronizes spawned subagent conversation IDs, model tiers, and `branch` workspaces in session state, rolling back failed spawns.
4. **`stop`**: Blocks premature completion if any `Workspace: "branch"` subagent worktrees remain unkilled when the session goes idle.

---

## Repository Structure

```text
.
├── AGENTS.md
├── README.md
└── .agents/
    └── plugins/
        └── pro-orchestrator/
            ├── hooks.json
            ├── plugin.json
            ├── rules/
            │   └── AGENTS.md
            └── scripts/
                ├── orchestrator_guard.py
                └── ste_lint.py
```

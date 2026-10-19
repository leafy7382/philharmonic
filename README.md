# Philharmonic (`pro-orchestrator`)

A zero-dependency **Antigravity (`agy`) plugin** that enforces a Central Orchestrator workflow (any model can run this role) with `flash` and `flash_lite` worker subagents and deterministic **ASD-STE100 (Simplified Technical English)** prompt linting.

---

## Features

- **Strict Tiered Delegation**: Allows any model to run as the Central Orchestrator, delegating file discovery, research, and bounded code changes to `flash_lite` (`scout`) and `flash` (`research`, `implementer`) subagents.
- **Context Separation & Report Enforcement**:
  - `PreInvocation` records the root orchestrator conversation ID and injects a subagent context isolation reminder into worker subagents.
  - `PreToolUse` caps direct root orchestrator file reads at $\le 2$ workspace files per turn and blocks the root orchestrator from reading raw `.log` or `transcript.jsonl` files directly (enforcing delegation to `scout`). `grep_search` and `find_by_name` are also blocked from targeting raw `.log` and `transcript*.jsonl` files directly in the root orchestrator.
  - Shell command chains (`&&`, `||`, `;`) are split and inspected per sub-command so compound commands like `git status && pytest` cannot bypass the git exemption or edit-and-test loop gate.
  - `Stop` validates that worker subagents return strictly the 4-header `REPORT FORMAT` (Option A: up to 10 `- ` bullet lines under `NOTES:`, $\le 1,600$ chars; Option B: mandatory inline error summary and `file://.../*.log` URI when `STATUS` is `BLOCKED` or `FAILED`) before their response enters the parent orchestrator's context window.
- **O(1) Hook Performance & State Accuracy**:
  - Incremental `transcript_offset` cursor for subagent synchronization and $64\text{ KB}$ tail-reading for `Stop` report extraction.
  - Local subagent identity caching (`self_subagent_meta`) across isolated brain directories.
  - Pre-compiled regular expression engine in `ste_lint.py`.
- **Empirical Test Receipts**: Every code change requires a subsequent passing test run recorded in `post-tool` before the `stop` hook allows completion.
- **2-Turn Circuit Breaker**: Tracks consecutive command/subagent failures in `post-tool` and trips after 2 failures, halting retries in `pre-tool` (while allowing scratch log writes) and allowing immediate `STATUS: BLOCKED` termination.
- **Test Tampering Audit**: `pre-tool` and `stop` unified diff auditing block edits to test files that remove test functions, reduce assertion counts, add skip/xfail markers, or insert trivial `assert True` checks.
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

To minimize both orchestrator context growth and subagent spawn overhead:
- **Direct Execution Fast-Path & Fast-Path Budget**: Single-step commands and reads/edits of $\le 2$ known files execute directly in the orchestrator with zero subagent overhead, enforced by a deterministic fast-path budget gate ($\le 2$ workspace files, $\le 2$ non-git commands, and blocking direct code+test loops in the root orchestrator).
  - `grep_search` and `find_by_name` are also blocked from targeting raw `.log` and `transcript*.jsonl` files directly in the root orchestrator.
  - Shell command chains (`&&`, `||`, `;`) are split and inspected per sub-command so compound commands like `git status && pytest` cannot bypass the git exemption or edit-and-test loop gate.
- **Lazy Subagent Registration**: Custom subagents (`scout` and `implementer`) are registered via `define_subagent` only immediately before first invoking them.
- **Automatic Tier Downgrade**: The `pre-tool` hook automatically downgrades read-only subagent calls (`scout` / `research`) with $\le 3$ numbered rules and no mutation verbs to `Model: "flash_lite"`.
- **O(1) Hook Performance & State Accuracy**:
  - Incremental `transcript_offset` cursor for subagent synchronization and $64\text{ KB}$ tail-reading for `Stop` report extraction.
  - Local subagent identity caching (`self_subagent_meta`) across isolated brain directories.
  - Pre-compiled regular expression engine in `ste_lint.py`.

| Subagent Type | Model Tier | Write Tools | Workspace Mode | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Disabled | `inherit` | Single-hop lookups, file/symbol extraction, log slicing ($\le 3$ rules) |
| `research` | `flash` (auto `flash_lite` if $\le 3$ rules) | Scratch-only (`*.md`) | `inherit` | Multi-file analysis, docs lookups, and scratch reports |
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
Return only these four headers (plus up to 10 `- ` bullet lines under NOTES if needed):
STATUS: COMPLETED | BLOCKED | FAILED
FILES_CHANGED: <comma-separated paths or NONE>
VERIFICATION: <one sentence stating the command run and exit result>
NOTES: <summary (and up to 10 `- ` bullets), or inline error cause plus file://.log URI if BLOCKED/FAILED>
```

### Bounded Subagent Report Format (Option A + Option B)

Subagent reports must follow bounded reporting rules:
- **Option A (Bounded `NOTES` Bullets)**: Reports may include up to 10 compact bullet lines (`- ` or `* `) under `NOTES:` (total report length capped at $\le 1,600$ chars).
- **Option B (Inline Diagnosis & Scratch Log URI)**: When `STATUS` is `BLOCKED` or `FAILED`, `NOTES:` must include both an inline error summary (at least 3 words describing the failure) and a `file://.../*.log` URI pointing to raw failure output.

### Scratch Specification Relay

For complex tasks requiring technical requirements, schema definitions, or test scenarios that exceed ASD-STE100 sentence constraints, write the detailed technical specification to `<artifactDirectoryPath>/scratch/spec_<task>.md` and instruct the subagent to open the spec file in Rule 1.

### Enforced Lint Checks

- **Sentence Length**: Maximum 20 words per numbered rule (`[long-sentence]`).
- **No Semicolons**: Split compound clauses into separate numbered rules (`[semicolon]`).
- **No Soft Phrasal Verbs**: Replace phrases like *figure out*, *look into*, or *spin up* with direct verbs (`[phrasal-verb]`).
- **No Hidden Nominalizations**: Use active imperative verbs instead of noun-heavy constructions (`[nominalization]`).
- **No Synonym Rotation**: Use one consistent verb for the same action throughout a prompt (`[synonym-rotation]`).

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/pro-orchestrator/hooks.json) and executed by [orchestrator_guard.py](.agents/plugins/pro-orchestrator/scripts/orchestrator_guard.py):

1. **`pre-invocation`**: Records the root orchestrator conversation ID, injects a subagent context isolation reminder into worker subagents, and deduplicates warnings per invocation.
2. **`pre-tool`**:
   - Auto-allows workspace, worktree, and scratch artifact file/command operations while enforcing the deterministic fast-path budget gate ($\le 2$ workspace file mutations, $\le 2$ direct file reads, $\le 2$ non-git commands, and blocking direct code+test loops in the root orchestrator).
   - Blocks the root orchestrator from reading raw `.log` or `transcript*.jsonl` files directly; `grep_search` and `find_by_name` are also blocked from targeting raw `.log` and `transcript*.jsonl` files directly in the root orchestrator (enforcing delegation to `scout`).
   - Splits and inspects shell command chains (`&&`, `||`, `;`) per sub-command so compound commands like `git status && pytest` cannot bypass the git exemption or edit-and-test loop gate.
   - Gates `research` subagents to writing markdown reports exclusively to `<artifactDirectoryPath>/scratch/*.md` (blocking workspace code mutations).
   - Blocks `Model: "pro"`, rewrites `Model: "inherit"`, and automatically downgrades simple read-only subagent tasks ($\le 3$ rules) to `Model: "flash_lite"`.
   - Runs the STE-100 linter gate on all `invoke_subagent` and `send_message` prompts via a pre-compiled regular expression engine in `ste_lint.py`.
   - Blocks test file edits that remove test functions, reduce assertion counts, add skip/xfail markers, or insert trivial `assert True` checks (Test Tampering Audit).
   - Halts retries when the 2-Turn Circuit Breaker is tripped, while allowing scratch log writes.
   - Uses local subagent identity caching (`self_subagent_meta`) across isolated brain directories and deduplicates `pending_spawns` tracking across multiple hook registrations.
3. **`post-tool`**:
   - Synchronizes spawned subagent conversation IDs, model tiers, and `branch` workspaces in session state via an incremental `transcript_offset` cursor, rolling back failed spawns.
   - Tracks command and subagent failures, tripping the 2-turn circuit breaker after 2 consecutive failures.
   - Records empirical test receipts on passing test commands following code mutations.
4. **`stop`**:
   - Blocks premature completion if any `Workspace: "branch"` subagent worktrees remain unkilled when the session goes idle.
   - Uses $64\text{ KB}$ tail-reading for report extraction and validates that worker subagents return strictly the 4-header `REPORT FORMAT` (Option A: up to 10 `- ` bullet lines under `NOTES:`, $\le 1,600$ chars; Option B: mandatory inline error summary and `file://.../*.log` URI when `STATUS` is `BLOCKED` or `FAILED`) before allowing completion.
   - Blocks completion with missing empirical test receipts when code files were modified without a subsequent passing test run recorded in `post-tool`.
   - Performs git diff auditing to block test tampering.
   - Allows immediate turn termination when the 2-turn circuit breaker is tripped so the agent can report `STATUS: BLOCKED`.

---

## Running Tests

Run the full test suite using `pytest`:

```bash
pytest -q
```

This runs all unit test suites covering both the lifecycle guard hooks (`tests/test_orchestrator_guard.py`) and the ASD-STE100 prompt linter (`tests/test_ste_lint.py`).

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

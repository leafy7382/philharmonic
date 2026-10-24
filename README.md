# Philharmonic (`philharmonic_orchestra`)

An **Antigravity (`agy`) plugin** that enforces a Central Orchestrator workflow using `flash` and `flash_lite` worker subagents, bounded direct-execution budgets, structured subagent prompts/reports, and test verification gates. Built with the Python 3 standard library (zero external dependencies).

---

## Features

- **On-Demand Session Gating**: Inactive by default in normal chats. Activates for the current conversation when you run `/philharmonic` or `orchestrator_guard.py activate`.
- **Tiered Subagent Delegation**: Any model can act as the Central Orchestrator while delegating file discovery, research, and multi-file edits to `flash_lite` (`scout`) and `flash` (`research`, `implementer`) subagents.
- **Direct Fast-Path Budget**: Allows the root orchestrator to handle small tasks directly (up to 2 workspace file reads, 2 workspace file edits, and 2 non-git commands per turn, plus 1 verification test run after a direct edit) while requiring `implementer` delegation for larger changes or iterative edit-and-test loops.
- **Context Isolation**: Blocks the root orchestrator from reading or searching raw `.log` and `transcript*.jsonl` files directly (`read_file`, `grep_search`, `find_by_name`), keeping bulky logs inside `scout` subagents.
- **Structured Prompt & Report Contracts**: Validates subagent prompts (`TASK:`, `RULES:`, `REPORT FORMAT:`) on dispatch and checks that subagents return a bounded 4-header report on completion.
- **Test Verification & Anti-Tampering Gate**: Requires a passing test run after code modifications before completion, and blocks test file edits or diffs that delete tests, reduce assertions, comment out `assert` statements, add skip/xfail markers, or insert `assert True`.
- **2-Turn Circuit Breaker**: Trips after 2 consecutive command or subagent failures, halting further retries and allowing immediate `STATUS: BLOCKED` termination.

---

## Prerequisites

- **Google Antigravity** (`agy` CLI or Antigravity IDE)
- **Python 3.8+** (`python3` on `PATH`)

---

## Quick Start

### 1. Run in this repository

```bash
git clone https://github.com/leafy7382/philharmonic.git
cd philharmonic
agy
```

Activate the workflow in any conversation with:

```text
/philharmonic
```

This loads [.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md](.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md) and enables the session guard for the current conversation. Conversations without `/philharmonic` remain ungated.

### 2. Install into another project workspace

```bash
mkdir -p /path/to/your-project/.agents/plugins
cp -r .agents/plugins/philharmonic_orchestra /path/to/your-project/.agents/plugins/
```

### 3. Install globally for all projects

```bash
mkdir -p ~/.gemini/config/plugins
ln -s "$(pwd)/.agents/plugins/philharmonic_orchestra" ~/.gemini/config/plugins/philharmonic_orchestra
```

---

## Subagent Architecture

Custom subagents (`scout` and `implementer`) are registered lazily via `define_subagent` immediately before first use; the built-in `research` subagent requires no registration.

| Subagent Type | Model Tier | Write Access | Workspace Mode | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Disabled | `inherit` | Single-hop lookups, file/symbol extraction, log slicing (<= 3 rules) |
| `research` | `flash` (auto-downgraded to `flash_lite` if <= 3 read-only rules) | Scratch-only (`<artifactDirectoryPath>/scratch/*.md`) | `inherit` | Multi-file analysis, documentation lookups, and scratch reports |
| `implementer` | `flash` | Enabled | `branch` / `share` / `inherit` | Bounded code edits and test execution |

---

## Subagent Prompt and Report Contract

All subagent prompts (`invoke_subagent` and `send_message`) must use the following structure:

```markdown
TASK: <single-sentence objective>

RULES:
1. Open `<file_path>`.
2. <concise instruction>.
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

### Report Validation Rules

- **Bounded `NOTES`**: Reports may include up to 10 bullet lines (`- ` or `* `) under `NOTES:`, with total report length capped at 1,600 characters.
- **Failure Diagnosis**: When `STATUS` is `BLOCKED` or `FAILED`, `NOTES:` must include both an inline error summary (at least 3 words) and a `file://.../*.log` URI pointing to the raw failure output in `<artifactDirectoryPath>/scratch/`.
- **Scratch Specification Relay**: For tasks with detailed specifications or schemas, write the spec to `<artifactDirectoryPath>/scratch/spec_<task>.md` and reference it in Rule 1 of `RULES:`.

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/philharmonic_orchestra/hooks.json) and implemented in [orchestrator_guard.py](.agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py), with session state persisted via `flock` in `<artifactDirectoryPath>/orchestrator_state.json`:

1. **`pre-invocation`**: Records the root orchestrator conversation ID, caches subagent identity across isolated brain directories, and injects a context-isolation reminder into worker subagents.
2. **`pre-tool`**:
   - Enforces Antigravity native tool names (`read_file`, `write_file`, `edit_file`, `list_dir`, `grep_search`, `find_by_name`, `run_command`), PascalCase `define_subagent` parameters, and flat `invoke_subagent` parameters.
   - Enforces the root orchestrator fast-path budget (<= 2 file reads, <= 2 file edits, <= 2 non-git commands, and 1 verification test run after edits). Compound shell commands (`&&`, `||`, `;`) are split and checked per sub-command.
   - Blocks root orchestrator reads and searches on `.log` and `transcript*.jsonl` files.
   - Restricts `research` subagents to writing markdown files under `<artifactDirectoryPath>/scratch/*.md`.
   - Blocks `Model: "pro"`, rewrites `Model: "inherit"`, and downgrades read-only subagent tasks with <= 3 rules and no mutation verbs to `Model: "flash_lite"`.
   - Validates `invoke_subagent` and `send_message` prompts against the structured prompt contract.
   - Inspects edits to test files and blocks test deletion, assertion reduction, commented-out assertions, skip/xfail additions, or `assert True`.
   - Blocks retries when the 2-turn circuit breaker is tripped (while still allowing scratch log writes).
3. **`post-tool`**:
   - Tracks spawned subagent conversation IDs, model tiers, and `branch` workspaces using an incremental `transcript_offset` cursor, rolling back failed spawns.
   - Records passing test runs after code edits and increments or resets the 2-turn circuit breaker counter.
4. **`stop`**:
   - Blocks completion if any `Workspace: "branch"` subagent worktree has not been terminated via `manage_subagents` (`Action: "kill"`).
   - Reads the tail (64 KB) of subagent transcripts to verify the 4-header `REPORT FORMAT`.
   - Blocks completion if code files were modified without a subsequent passing test run, or if `git diff HEAD -U0` reveals test tampering.
   - Allows immediate termination when the circuit breaker is tripped so the agent can report `STATUS: BLOCKED`.

---

## CLI Commands

Inspect, toggle, or reset the session guard from the terminal:

```bash
# View active subagents, pending spawns, direct budget, test receipts, and circuit breaker state
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status

# Activate guard enforcement for the current session
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate

# Deactivate guard enforcement for the current session
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate

# Reset session guard state to defaults
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py reset

# Validate hooks.json and SKILL.md configuration
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
```

---

## Running Tests

Run the test suite with Python's built-in `unittest` runner (or `pytest`):

```bash
python3 -m unittest discover -s tests
```

---

## Repository Structure

```text
.
├── README.md
├── .agents/
│   └── plugins/
│       └── philharmonic_orchestra/
│           ├── hooks.json
│           ├── plugin.json
│           ├── scripts/
│           │   └── orchestrator_guard.py
│           └── skills/
│               └── philharmonic/
│                   └── SKILL.md
└── tests/
    └── test_orchestrator_guard.py
```

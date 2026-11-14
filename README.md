# Philharmonic (`philharmonic_orchestra`)

An **Antigravity (`agy`) plugin** that enforces a Central Orchestrator workflow using `flash` and `flash_lite` worker subagents, bounded direct-execution budgets, structured subagent prompts/reports, and test verification gates. Built with the Python 3 standard library (zero external dependencies).

---

## Features

- **On-Demand Session Gating**: Inactive by default in normal chats. Activates for the current conversation when you run `/philharmonic` or `orchestrator_guard.py activate`.
- **Tiered Subagent Delegation**: Any model can act as the Central Orchestrator while delegating file discovery, research, and multi-file edits to `flash_lite` (`scout`) and `flash` (`research`, `implementer`) subagents.
- **Direct Fast-Path Budget**: Allows the root orchestrator to handle small tasks directly (up to 400 lines / 16 KB across $\le 8$ workspace files with line/byte estimation and `list_dir` free, 2 workspace file edits, and 2 non-git commands per turn, plus 1 verification test run after a direct edit) while requiring `implementer` delegation for larger changes or iterative edit-and-test loops.
- **System Orchestrator Hybrid Coexistence Mode**: Calling `exit_plan_mode` during plan execution unlocks multi-file mutations and iterative test loops for Antigravity's execution model while keeping `.log` isolation, optional plan `ALLOWED_FILES:` enforcement, test anti-tampering guards, empirical test receipts, and circuit breaker gates active. Resets on `[plan review]` or the next user prompt.
- **Context Isolation**: Blocks the root orchestrator from reading or searching raw `.log` and `transcript*.jsonl` files directly (`read_file`, `grep_search`, `find_by_name`, `list_dir`), keeping bulky logs inside `scout` subagents.
- **Structured Prompt & Report Contracts**: Validates subagent prompts (`TASK:`, `ALLOWED_FILES:`, optional `STAGE:`, `SPEC_FILE:`, `REPLAN_IF:`, `RULES:`, `REPORT FORMAT:`) on dispatch and verifies reports on completion (with compact 2-header shorthand for `scout`).
- **Grounded Spec Relay**: Validates that specifications referenced via `SPEC_FILE:` exist and contain `SIGNATURES`, `PITFALLS` (or `INVARIANTS`), and `VERIFICATION` sections before dispatching `implementer` subagents.
- **Staged Mid-Flight Checkpoints**: Subagents can report intermediate progress via `STATUS: CHECKPOINT` without prematurely verifying empirical test receipts, allowing step-by-step review and dynamic file allowlist expansion.
- **Test Verification & Anti-Tampering Gate**: Requires a passing test run after code modifications before completion, and blocks test file edits or diffs that delete tests, reduce assertions, comment out `assert` statements, add skip/xfail markers, or insert `assert True`.
- **2-Turn Circuit Breaker & 1-Shot Re-Plan**: Trips after 2 consecutive command or subagent failures, halting further retries. Includes a 1-shot re-plan recovery budget allowing the root orchestrator to provide `REPLAN: <revised plan rationale>` to reset the circuit breaker once before hard-locking into `STATUS: BLOCKED`.

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

### 3. Install globally for all projects (Local CLI)

```bash
mkdir -p ~/.gemini/config/plugins
ln -s "$(pwd)/.agents/plugins/philharmonic_orchestra" ~/.gemini/config/plugins/philharmonic_orchestra
```

### 4. Install on Remote Projects (Antigravity Cloud VM)

- **Persistent Custom Skill (recommended)**: Ask Antigravity to register `.agents/plugins/philharmonic_orchestra/skills/philharmonic` via `save_skill` so `philharmonic` persists across all remote conversations and projects.
- **Persistent `~/.gemini` copy**: Since `~/.gemini` (`/shared/user/.gemini`) persists across remote VMs while project directories are ephemeral, copy (do not symlink) the plugin:
  ```bash
  mkdir -p ~/.gemini/config/plugins
  cp -r .agents/plugins/philharmonic_orchestra ~/.gemini/config/plugins/
  ```
- **Remote VM Dual-Schema Support**: Both `SKILL.md` and `orchestrator_guard.py` support local `agy` CLI and Remote VM tool schemas. On Remote VMs, `invoke_subagent` uses `Subagents: [...]` with `Model: "inherit"` and `Workspace: "inherit"` (since the remote VM harness does not initialize `PlatformClient` for tiered model resolution or branched workspaces).

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
ALLOWED_FILES: <comma-separated file paths or NONE>
STAGE: 1/2
SPEC_FILE: <artifactDirectoryPath>/scratch/spec_<task>.md
REPLAN_IF: <condition that triggers early replan>

RULES:
1. Open `<file_path>`.
2. <concise instruction>.
3. Do not change any file outside `ALLOWED_FILES`.
4. Run `<test_command>` to check the result.
5. On failure, write raw output to `<artifactDirectoryPath>/scratch/<task_id>.log`.

REPORT FORMAT:
Return only these four headers (plus up to 10 `- ` bullet lines under NOTES if needed):
STATUS: COMPLETED | CHECKPOINT | NEEDS_REPLAN | BLOCKED | FAILED
FILES_CHANGED: <comma-separated paths or NONE>
VERIFICATION: <one sentence stating the command run and exit result>
NOTES: <summary (and up to 10 `- ` bullets), or inline error cause plus file://.log URI if BLOCKED/FAILED>
```

### Report Validation Rules

- **Deterministic File Allowlist (`ALLOWED_FILES:`)**: When `ALLOWED_FILES:` is specified in `invoke_subagent` or `send_message`, `PreToolUse` deterministically blocks the subagent from mutating any workspace file outside that allowlist (while still allowing `<artifactDirectoryPath>/scratch/*.log` failure logs).
- **Staged Mid-Flight Checkpoints (`STAGE:`)**: Subagents can report intermediate progress via `STATUS: CHECKPOINT` without prematurely verifying empirical test receipts, enabling progressive diff review and dynamic allowlist expansion via `send_message`.
- **Grounded Spec Relay (`SPEC_FILE:`)**: Specs referenced in prompts are verified to exist and contain `SIGNATURES`, `PITFALLS` (or `INVARIANTS`), and `VERIFICATION` before dispatching `implementer` subagents.
- **Replan-Aware Circuit Breaker (`REPLAN_IF:`)**: Subagents halt and report `STATUS: NEEDS_REPLAN` when assumptions fail. Root orchestrators can use `REPLAN: <rationale>` to reset a tripped circuit breaker once per turn.
- **Compact Scout Shorthand**: For read-only `scout` tasks ($\le 3$ rules), a compact 2-header report (`STATUS:` + `NOTES:`) is supported.
- **Bounded `NOTES`**: Reports may include up to 10 bullet lines (`- ` or `* `) under `NOTES:`, with total report length capped at 1,600 characters.
- **Failure Diagnosis**: When `STATUS` is `BLOCKED` or `FAILED`, `NOTES:` must include both an inline error summary (at least 3 words) and a `file://.../*.log` URI pointing to the raw failure output in `<artifactDirectoryPath>/scratch/`. When `STATUS` is `NEEDS_REPLAN`, `NOTES:` must provide an inline explanation of at least 3 words.

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/philharmonic_orchestra/hooks.json) and dispatched via [orchestrator_guard.py](.agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py) (backed by the modular [guard/](.agents/plugins/philharmonic_orchestra/scripts/guard) package: `constants.py`, `identity.py`, `policies.py`, `state_store.py`, `verification.py`, `cli.py`), with session state persisted via `flock` in `<artifactDirectoryPath>/orchestrator_state.json`:

1. **`pre-invocation`**: Records the root orchestrator conversation ID, caches subagent identity across isolated brain directories, handles hybrid coexistence resets (on `[plan review]` or next user input), resets circuit breaker `replan_count`, and injects a context-isolation reminder into worker subagents.
2. **`pre-tool`**:
   - Supports both local `agy` CLI (`read_file`, `write_file`, `edit_file`, `list_dir`, `grep_search`, `find_by_name`, `run_command`, flat `invoke_subagent`, PascalCase `define_subagent`) and Remote VM (`view_file`, `write_to_file`, `replace_file_content`, `lookup_skill`, `lookup_skills`, `Subagents` list `invoke_subagent`, `snake_case` `define_subagent`) tool schemas.
   - Enforces the root orchestrator direct read budget (<= 400 lines / <= 16 KB across <= 8 workspace files with line/byte estimation; `list_dir` free), mutation limits (<= 2 file edits), command limits (<= 2 non-git commands), and 1 verification test run after edits. Git inspection commands (status, diff, log) with safe non-interactive environment variable prefixes (`GIT_EDITOR=true`, `GIT_SEQUENCE_EDITOR=:`, `GIT_PAGER=cat`, `GIT_TERMINAL_PROMPT=0`, `LC_ALL=C`) do not count against command limits and remain allowed when tripped.
   - Handles `exit_plan_mode` and `enter_plan_mode` triggers for System Orchestrator hybrid coexistence mode, lifting direct edit/command/test loop caps while enforcing plan `ALLOWED_FILES:`.
   - Blocks root orchestrator reads, directory listings, and searches on `.log` and `transcript*.jsonl` files.
   - Validates grounded specs (`SPEC_FILE:`) containing required sections (`SIGNATURES`, `PITFALLS`/`INVARIANTS`, `VERIFICATION`) for `implementer`.
   - Tracks `STAGE:` history and allows 1-shot circuit breaker reset with `REPLAN:` header on dispatch.
   - Restricts `research` subagents to writing markdown files under `<artifactDirectoryPath>/scratch/*.md` and enforces per-subagent `ALLOWED_FILES:` workspace write allowlists on `implementer` subagents.
   - Blocks `Model: "pro"`, rewrites `Model: "inherit"`, and downgrades read-only subagent tasks with <= 3 rules and no mutation verbs to `Model: "flash_lite"`.
   - Validates `invoke_subagent` and `send_message` prompts against the structured prompt contract.
   - Inspects edits to test files and blocks test deletion, assertion reduction, commented-out assertions, skip/xfail additions, or `assert True`.
   - Blocks retries when the 2-turn circuit breaker is tripped (while still allowing scratch log writes and 1-shot re-plan dispatch).
3. **`post-tool`**:
   - Tracks spawned subagent conversation IDs, model tiers, and `branch` workspaces using an incremental `transcript_offset` cursor, rolling back failed spawns.
   - Records passing test runs after code edits and increments or resets the 2-turn circuit breaker counter (with Antigravity sandbox boundary errors such as `.git/` read-only errors and sandbox error banners exempted from tripping the breaker).
   - Distinguishes non-final statuses (`CHECKPOINT`, `NEEDS_REPLAN`, `BLOCKED`, `FAILED`) to avoid prematurely marking test receipts as verified.
   - Rolls back hybrid coexistence state if `exit_plan_mode` tool call returns an error.
4. **`stop`**:
   - Blocks completion if any `Workspace: "branch"` subagent worktree has not been terminated via `manage_subagents` (`Action: "kill"`).
   - Validates subagent report formats (supporting compact 2-header format for `scout` and 4-header format with `CHECKPOINT`/`NEEDS_REPLAN`).
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
│           │   ├── guard/
│           │   │   ├── cli.py
│           │   │   ├── constants.py
│           │   │   ├── identity.py
│           │   │   ├── policies.py
│           │   │   ├── state_store.py
│           │   │   └── verification.py
│           │   └── orchestrator_guard.py
│           └── skills/
│               └── philharmonic/
│                   └── SKILL.md
└── tests/
    └── test_orchestrator_guard.py
```

# Philharmonic (`philharmonic_orchestra`)

A zero-dependency **Antigravity (`agy`) plugin** that enforces a Central Orchestrator workflow (any model can run this role) with `flash` and `flash_lite` worker subagents and deterministic structured prompt and report contracts.

---

## Features

- **On-Demand Activation & Session Gating**: Inactive by default in normal chats. Automatically activates for the current conversation when you invoke `/philharmonic` (loading `.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md`) or manually via CLI `activate`/`deactivate`.
- **Strict Tiered Delegation**: Allows any model to run as the Central Orchestrator, delegating file discovery, research, and bounded code changes to `flash_lite` (`scout`) and `flash` (`research`, `implementer`) subagents.
- **Native Antigravity Tool Enforcement**: Enforces Antigravity's native tool schemas (`read_file`, `write_file`, `edit_file`, `list_dir`, `grep_search`, `find_by_name`, `run_command`), PascalCase `define_subagent` arguments (`Name`, `EnableWriteTools`, `EnableMCPTools`, `EnableSubagentTools`, `SystemPrompt`), and flat `invoke_subagent` arguments (`TypeName`, `Role`, `Prompt`, `Model`, `Workspace`).
- **Context Separation & Direct Fast-Path Budget**:
  - `PreInvocation` records the root orchestrator conversation ID and injects a subagent context isolation reminder into worker subagents.
  - `PreToolUse` caps direct root orchestrator file reads at $\le 2$ workspace files per turn and blocks the root orchestrator from reading raw `.log` or `transcript*.jsonl` files directly (enforcing delegation to `scout`). `grep_search` and `find_by_name` are also blocked from targeting raw `.log` and `transcript*.jsonl` files directly in the root orchestrator.
  - Shell command chains (`&&`, `||`, `;`) are split and inspected per sub-command so compound commands like `git status && pytest` cannot bypass the git exemption or edit-and-test loop gate.
  - **Option A Fast-Path Verification**: The root orchestrator may directly read $\le 2$ workspace files, edit $\le 2$ workspace files, and run $\le 2$ non-git commands per turn. After a direct fast-path code edit ($\le 2$ files), the root orchestrator may run 1 verification test command (`pytest`, `unittest`, etc.) to record an empirical test receipt. Iterative edit-and-test loops (editing again after a test run, or running a second test command after an edit) are blocked and must be delegated to `implementer`.
- **Comment-Aware Test-Tampering Detection & Diff Auditing**:
  - `pre-tool` inspects edits to test files, blocking net deletion or commenting-out of test functions or assertions (e.g. `# assert ...`), addition of skip/xfail markers, or insertion of trivial `assert True` checks.
  - `stop` audits `git diff HEAD -U0` across both staged and unstaged test changes before allowing completion.
- **Session-Persistent State & Test Receipts**:
  - Atomic `flock` state persistence in `<artifactDirectoryPath>/orchestrator_state.json`.
  - Session-persistent `test_receipt` and `circuit_breaker` across multi-step invocations.
  - Every code change requires a subsequent passing test run recorded in `post-tool` before the `stop` hook allows completion.
- **2-Turn Circuit Breaker**:
  - Tracks consecutive command and subagent failures in `post-tool` and trips after 2 failures, halting retries in `pre-tool` (while allowing scratch log writes) and allowing immediate `STATUS: BLOCKED` termination.
- **Structured Prompt & Report Schema Gate**:
  - Enforces a structured inter-agent prompt contract (`TASK:`, `RULES:`, `REPORT FORMAT:`) and bounded 4-header report format across subagent invocations.
- **CLI Diagnostic Commands**:
  - Terminal status, reset, and health recovery commands:
    ```bash
    python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status
    python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py reset
    python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
    ```
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

Philharmonic is an on-demand workflow. Activate it in any conversation by typing:
```text
/philharmonic
```
This loads [.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md](.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md) and automatically activates the session guard for the conversation. Regular chat sessions without `/philharmonic` remain ungated.

### 2. Install into another project workspace

Copy the plugin into your target repository:

```bash
mkdir -p /path/to/your-project/.agents/plugins
cp -r .agents/plugins/philharmonic_orchestra /path/to/your-project/.agents/plugins/
```

### 3. Install globally for all projects

Symlink or copy the plugin into your user-level Antigravity plugins directory:

```bash
mkdir -p ~/.gemini/config/plugins
ln -s "$(pwd)/.agents/plugins/philharmonic_orchestra" ~/.gemini/config/plugins/philharmonic_orchestra
```

---

## Subagent Architecture & Automatic Token Optimization

To minimize both orchestrator context growth and subagent spawn overhead:
- **Direct Execution Fast-Path (Option A Single Verification Pass)**: Single-step commands and reads/edits of $\le 2$ known files execute directly in the orchestrator with zero subagent overhead, enforced by a deterministic fast-path budget gate ($\le 2$ workspace files, $\le 2$ non-git commands). The orchestrator is permitted 1 verification test run after a direct edit of $\le 2$ files, while iterative edit-and-test loops are blocked and must be delegated to `implementer`.
  - `grep_search` and `find_by_name` are also blocked from targeting raw `.log` and `transcript*.jsonl` files directly in the root orchestrator.
  - Shell command chains (`&&`, `||`, `;`) are split and inspected per sub-command so compound commands like `git status && pytest` cannot bypass the git exemption or edit-and-test loop gate.
- **Lazy Subagent Registration**: Custom subagents (`scout` and `implementer`) are registered via native `define_subagent` (PascalCase schema) only immediately before first invoking them.
- **Automatic Tier Downgrade**: The `pre-tool` hook automatically downgrades read-only subagent calls (`scout` / `research`) with $\le 3$ numbered rules and no mutation verbs to `Model: "flash_lite"`.
- **O(1) Hook Performance & State Accuracy**:
  - Incremental `transcript_offset` cursor for subagent synchronization and $64\text{ KB}$ tail-reading for `Stop` report extraction.
  - Local subagent identity caching (`self_subagent_meta`) across isolated brain directories.
  - Session state persisted with atomic file locking (`flock`).

| Subagent Type | Model Tier | Write Tools | Workspace Mode | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Disabled | `inherit` | Single-hop lookups, file/symbol extraction, log slicing ($\le 3$ rules) |
| `research` | `flash` (auto `flash_lite` if $\le 3$ rules) | Scratch-only (`*.md`) | `inherit` | Multi-file analysis, docs lookups, and scratch reports |
| `implementer` | `flash` | Enabled | `branch` / `share` / `inherit` | Bounded code edits and test execution |

---

## Subagent Prompt and Report Contract

All subagent prompts (`invoke_subagent` and `send_message`) must follow the structured contract below:

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

### Bounded Subagent Report Format (Option A + Option B)

Subagent reports must follow bounded reporting rules:
- **Option A (Bounded `NOTES` Bullets)**: Reports may include up to 10 compact bullet lines (`- ` or `* `) under `NOTES:` (total report length capped at $\le 1,600$ chars).
- **Option B (Inline Diagnosis & Scratch Log URI)**: When `STATUS` is `BLOCKED` or `FAILED`, `NOTES:` must include both an inline error summary (at least 3 words describing the failure) and a `file://.../*.log` URI pointing to raw failure output.

### Scratch Specification Relay

For complex tasks requiring technical requirements, schema definitions, or test scenarios, write the detailed technical specification to `<artifactDirectoryPath>/scratch/spec_<task>.md` and instruct the subagent to open the spec file in Rule 1.

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/philharmonic_orchestra/hooks.json) and executed by [orchestrator_guard.py](.agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py):

1. **`pre-invocation`**: Records the root orchestrator conversation ID, injects a subagent context isolation reminder into worker subagents, and deduplicates warnings per invocation.
2. **`pre-tool`**:
   - Auto-allows workspace, worktree, and scratch artifact file/command operations while enforcing native tool schemas (`read_file`, `write_file`, `edit_file`, `list_dir`, `grep_search`, `find_by_name`, `run_command`, PascalCase `define_subagent`, flat `invoke_subagent`).
   - Enforces the deterministic fast-path budget gate ($\le 2$ workspace file mutations, $\le 2$ direct file reads, $\le 2$ non-git commands, and Option A single verification test run after edits).
   - Blocks the root orchestrator from reading raw `.log` or `transcript*.jsonl` files directly; `grep_search` and `find_by_name` are also blocked from targeting raw `.log` and `transcript*.jsonl` files directly in the root orchestrator (enforcing delegation to `scout`).
   - Splits and inspects shell command chains (`&&`, `||`, `;`) per sub-command so compound commands like `git status && pytest` cannot bypass the git exemption or edit-and-test loop gate.
   - Gates `research` subagents to writing markdown reports exclusively to `<artifactDirectoryPath>/scratch/*.md` (blocking workspace code mutations).
   - Blocks `Model: "pro"`, rewrites `Model: "inherit"`, and automatically downgrades simple read-only subagent tasks ($\le 3$ rules) to `Model: "flash_lite"`.
   - Validates that `invoke_subagent` and `send_message` prompts are non-empty and adhere to the structured prompt format.
   - Comment-aware test-tampering detection: blocks test file edits that remove test functions, reduce assertion counts, comment out assertions (`# assert ...`), add skip/xfail markers, or insert trivial `assert True` checks.
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
   - Performs git diff auditing (`git diff HEAD -U0`) to block test tampering.
   - Allows immediate turn termination when the 2-turn circuit breaker is tripped so the agent can report `STATUS: BLOCKED`.

---

## CLI Diagnostic Commands

Inspect, verify, or recover the orchestrator guard state from the command line:

```bash
# View active subagents, pending spawns, direct budget, receipts, and circuit breaker
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status

# Activate guard enforcement for the current session
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate

# Deactivate guard enforcement for the current session
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate

# Reset session guard state to clean defaults
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py reset

# Run diagnostic checks on hooks configuration and protocol rules
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
```

---

## Running Tests

Run the full test suite using `pytest`:

```bash
pytest -q
```

This runs the unit test suite covering the lifecycle guard hooks (`tests/test_orchestrator_guard.py`).

---

## Repository Structure

```text
.
├── README.md
└── .agents/
    └── plugins/
        └── philharmonic_orchestra/
            ├── hooks.json
            ├── plugin.json
            ├── scripts/
            │   └── orchestrator_guard.py
            └── skills/
                └── philharmonic/
                    └── SKILL.md
```

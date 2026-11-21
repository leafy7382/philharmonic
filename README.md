# Philharmonic (`philharmonic_orchestra`)

An **Antigravity (`agy`) plugin** enforcing a Central Orchestrator workflow with `flash`/`flash_lite` workers, read-only `verifier` and `reviewer` roles, bounded direct budgets, structured contracts, and test gates. Zero external dependencies (Python 3 stdlib only).

---

## Features

- **On-Demand Session Gating**: Inactive by default; activates per conversation via `/philharmonic` or `orchestrator_guard.py activate`.
- **Pattern Selection Flowchart**: Routes tasks in `SKILL.md` by shape — fast-path for simple/coupled steps, Builder + fresh read-only `verifier` for medium changes, disjoint `implementer` lanes for independent modules, and a parallel `reviewer` team with distinct `REVIEW_LENS:` assignments for large changes.
- **Tiered Subagent Delegation**: Delegates discovery, research, bounded edits, fresh verification, and review to `flash_lite` (`scout`), `flash` (`research`, `implementer`), and read-only `verifier`/`reviewer` (`flash` default; explicit `pro`/`inherit` allowed; never `flash_lite`).
- **Disjoint Lanes & Freshness Gate**: Blocks overlapping `ALLOWED_FILES:` across concurrent `implementer` workers on `Workspace: "inherit"` (allowing isolated `"branch"`/`"share"` worktrees), and tracks `mutation_epoch` so stale `verifier`/`reviewer` threads cannot be reused via `send_message` after code edits.
- **Direct Fast-Path Budget**: Allows <= 400 lines / <= 16 KB reads across <= 8 files (`list_dir` free), <= 2 edits, <= 2 non-git commands per turn, and 1 verification test after direct edits; larger work or edit-test loops require `implementer`.
- **Hybrid Coexistence Mode**: `exit_plan_mode` lifts direct edit/command/loop caps for System Orchestrator plans while keeping `.log` isolation, plan `ALLOWED_FILES:`, anti-tampering, test receipts, and circuit breaker active; resets on `[plan review]` or next user prompt.
- **Context Isolation**: Blocks root reads, listings, and searches on raw `.log` and `transcript*.jsonl` files.
- **Structured Prompt & Report Contracts**: Validates dispatch headers (`TASK:`, `ALLOWED_FILES:`, mandatory `REVIEW_LENS:` for `reviewer`, optional `STAGE:`, `SPEC_FILE:`, `REPLAN_IF:`, `RULES:`, `REPORT FORMAT:`) and completion reports.
- **Grounded Spec Relay**: Verifies `SPEC_FILE:` exists with `SIGNATURES`, `PITFALLS`/`INVARIANTS`, and `VERIFICATION` before dispatching `implementer`.
- **Staged Checkpoints**: Supports `STATUS: CHECKPOINT` for step-by-step diff review and `ALLOWED_FILES:` expansion via `send_message`.
- **Test Verification & Anti-Tampering Gate**: Requires a passing test run after code edits and blocks test deletion, assertion reduction, commented-out `assert`s, skip/xfail markers, or `assert True`.
- **2-Turn Circuit Breaker & 1-Shot Re-Plan**: Trips after 2 consecutive failures; `REPLAN: <rationale>` resets the breaker once per turn before hard-locking into `STATUS: BLOCKED`.

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

Run `/philharmonic` in chat to load [.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md](.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md) and enable the guard for the current conversation.

### 2. Install into another workspace

```bash
mkdir -p /path/to/your-project/.agents/plugins
cp -r .agents/plugins/philharmonic_orchestra /path/to/your-project/.agents/plugins/
```

### 3. Install globally (Local CLI)

```bash
mkdir -p ~/.gemini/config/plugins
ln -s "$(pwd)/.agents/plugins/philharmonic_orchestra" ~/.gemini/config/plugins/philharmonic_orchestra
```

### 4. Install on Remote Projects (Antigravity Cloud VM)

- **Persistent Skill (recommended)**: Register `.agents/plugins/philharmonic_orchestra/skills/philharmonic` via `save_skill`.
- **Persistent `~/.gemini` copy**: Copy (do not symlink) the plugin into `~/.gemini/config/plugins/`:
  ```bash
  mkdir -p ~/.gemini/config/plugins
  cp -r .agents/plugins/philharmonic_orchestra ~/.gemini/config/plugins/
  ```
- **Remote VM Dual-Schema**: `SKILL.md` and `orchestrator_guard.py` support both local `agy` CLI and Remote VM schemas (`Subagents: [...]` with `Model: "inherit"`, `Workspace: "inherit"`, and `snake_case` `define_subagent`).

---

## Subagent Architecture

Custom subagents (`scout`, `implementer`, `verifier`, `reviewer`) are registered lazily via `define_subagent` before first use; `research` is built-in.

| Subagent Type | Model Tier | Write Access | Workspace Mode | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Disabled | `inherit` | Single-hop lookups, symbol extraction, log slicing (<= 3 rules) |
| `research` | `flash` (downgraded to `flash_lite` if <= 3 read-only rules) | Scratch-only (`<artifactDirectoryPath>/scratch/*.md`) | `inherit` | Multi-file analysis, doc lookups, scratch reports |
| `implementer` | `flash` | Enabled | `branch` / `share` / `inherit` | Bounded code edits and test runs (disjoint `ALLOWED_FILES:` on `inherit`) |
| `verifier` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Fresh post-build verification (`VERDICT: PASS \| FAIL`) |
| `reviewer` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Specialized review with distinct `REVIEW_LENS:` (`VERDICT: APPROVE \| REQUEST_CHANGES`) |

---

## Subagent Prompt and Report Contract

All `invoke_subagent` and `send_message` prompts must use:

```markdown
TASK: <single-sentence objective>
ALLOWED_FILES: <comma-separated file paths or NONE>
REVIEW_LENS: <required for reviewer, e.g., security | edge-cases | simplicity | ux>
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
NOTES: <summary (for verifier/reviewer start with VERDICT: PASS|FAIL or APPROVE|REQUEST_CHANGES, plus up to 10 `- ` bullets), or inline error cause plus file://.log URI if BLOCKED/FAILED>
```

### Report Validation Rules

- **`ALLOWED_FILES:` & Disjoint Lanes**: `PreToolUse` blocks subagent edits outside `ALLOWED_FILES:` (allowing `<artifactDirectoryPath>/scratch/*.log`). Concurrent `implementer` lanes on `Workspace: "inherit"` cannot overlap files or ancestor/descendant paths (`"branch"`/`"share"` worktrees may overlap).
- **Mandatory & Distinct `REVIEW_LENS:`**: `reviewer` requires `REVIEW_LENS:`; normalized lenses (`"edge cases"`, `"edge-cases"`, `"edge_cases"`) must be distinct across active `reviewer`s until completion or kill.
- **Verdicts (`verifier`/`reviewer`)**: Require the full 4-header report (2-header shorthand is `scout`-only) with `VERDICT: PASS | FAIL` or `VERDICT: APPROVE | REQUEST_CHANGES` under `NOTES:`.
- **Checkpoints (`STAGE:`) & Spec Relay (`SPEC_FILE:`)**: `STATUS: CHECKPOINT` enables mid-flight review without clearing receipts; `SPEC_FILE:` requires `SIGNATURES`, `PITFALLS`/`INVARIANTS`, and `VERIFICATION` for `implementer`.
- **Re-Plan (`REPLAN_IF:`), Shorthand & Limits**: `REPLAN: <rationale>` resets a tripped breaker once per turn; `scout` (<= 3 rules) supports 2-header shorthand; `NOTES:` allows <= 10 bullets within 1,600 chars total. `BLOCKED`/`FAILED` requires a >= 3-word summary + `file://.../*.log` URI; `NEEDS_REPLAN` requires >= 3 words.

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/philharmonic_orchestra/hooks.json) and dispatched via [orchestrator_guard.py](.agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py) (backed by [guard/](.agents/plugins/philharmonic_orchestra/scripts/guard): `constants.py`, `identity.py`, `policies.py`, `state_store.py`, `verification.py`, `cli.py`, `__init__.py`), using `flock` state at `<artifactDirectoryPath>/orchestrator_state.json`:

1. **`pre-invocation`**: Records root conversation ID, caches subagent identity across brain dirs, resets hybrid coexistence on `[plan review]` or new user input, resets `replan_count`, and injects context-isolation reminders.
2. **`pre-tool`**:
   - Supports local `agy` CLI and Remote VM tool schemas (`Subagents` list, `snake_case`/`PascalCase` `define_subagent`).
   - Enforces root read/edit/command budgets, exempts safe non-interactive git inspection (`status`/`diff`/`log`), handles `exit_plan_mode`/`enter_plan_mode` coexistence, and blocks root access to `.log`/`transcript*.jsonl`.
   - Enforces strict read-only tools on `scout`, `verifier`, `reviewer` (`EnableWriteTools`/`EnableMCPTools`/`EnableSubagentTools` forced `false`; `MUTATION_TOOLS` and mutating shell commands blocked while `read_file` and git inspection remain allowed), scratch-only markdown on `research`, and `ALLOWED_FILES:` on `implementer`.
   - Enforces per-role model policy (`verifier`/`reviewer` default to `"flash"`, allow `"pro"`/`"inherit"`, never `"flash_lite"` or short-prompt downgrade; standard workers block `"pro"`, rewrite `"inherit"`, and downgrade simple read-only tasks to `"flash_lite"`).
   - Validates `SPEC_FILE:`, `STAGE:`/`REPLAN:`, `REVIEW_LENS:` uniqueness, disjoint `implementer` lanes on `inherit`, and the `mutation_epoch` freshness gate (blocks `send_message` to `verifier`/`reviewer` after code edits).
   - Increments `test_receipt["mutation_epoch"]` on code mutations, tracks `status` (`active`/`killed`), blocks test tampering, and halts retries when the 2-turn circuit breaker is tripped.
3. **`post-tool`**:
   - Tracks subagent IDs, model tiers, `status`, `spawned_mutation_epoch`, `review_lens`, and `branch` workspaces via `transcript_offset`, rolling back failed spawns and marking synchronous completions `"completed"`.
   - Records passing test runs, propagates child `implementer` dirty edits and `mutation_epoch` to the parent brain, prevents `READONLY_NON_VERIFICATION_TYPES` (`scout`, `research`, `verifier`, `reviewer`) from clearing dirty receipts, and updates the circuit breaker (exempting sandbox boundary errors).
   - Preserves dirty receipts on `CHECKPOINT`/`NEEDS_REPLAN`/`BLOCKED`/`FAILED` and rolls back failed `exit_plan_mode` transitions.
4. **`stop`**:
   - Requires `manage_subagents` (`kill`) cleanup for `Workspace: "branch"` subagents and validates 4-header reports (`scout` keeps 2-header shorthand), marking clean subagent stops `"completed"` to release lanes/lenses.
   - Blocks completion on missing post-edit test receipts or `git diff HEAD -U0` test tampering (keeping `verifier`/`reviewer` protocol-guided via `SKILL.md`), and allows immediate exit when the circuit breaker is tripped.

---

## CLI Commands

```bash
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py reset
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
```

---

## Running Tests

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
│           │   │   ├── __init__.py
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

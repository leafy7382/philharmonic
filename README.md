# Philharmonic (`philharmonic_orchestra`)

An **Antigravity (`agy`) plugin** enforcing a Central Orchestrator workflow with `flash`/`flash_lite` workers, read-only `verifier` and `reviewer` roles, bounded direct budgets, structured contracts, and test gates. Zero external dependencies (Python 3 stdlib only).

---

## Features

- **On-Demand Session Gating**: Inactive by default; activates per conversation via `/philharmonic` or `orchestrator_guard.py activate`.
- **Pattern Selection Flowchart**: Routes tasks in `SKILL.md` by shape — fast-path for simple/coupled steps, Builder + fresh `verifier` for medium changes, disjoint `implementer` lanes for independent modules, Best of N (`MODE: BEST_OF_N`) in isolated `"branch"`/`"share"` worktrees for high-variance tasks with deterministic tests/evals, and a parallel `reviewer` team (`REVIEW_LENS:`) for large changes.
- **Tiered Subagent Delegation**: Delegates discovery, research, bounded edits, fresh verification, and review to `flash_lite` (`scout`), `flash` (`research`, `implementer`; explicit `"pro"` allowed for `MODE: BEST_OF_N` candidates), and read-only `verifier`/`reviewer` (`flash` default; explicit `pro`/`inherit` allowed; never `flash_lite`).
- **Disjoint Lanes, Best-of-N Worktrees & Freshness Gate**: Blocks overlapping `ALLOWED_FILES:` across concurrent `implementer` workers on `Workspace: "inherit"` (allowing isolated `"branch"`/`"share"` worktrees), isolates `MODE: BEST_OF_N` candidate receipts/failures per `COHORT:`, and blocks `send_message` to stale `verifier`/`reviewer` threads after code edits (`mutation_epoch`).
- **Direct Fast-Path Budget**: Allows <= 400 lines / <= 16 KB reads across <= 8 files (`list_dir` free), <= 2 edits, <= 2 non-git commands per turn, and 1 verification test after direct edits; larger work or edit-test loops require `implementer`.
- **Hybrid Coexistence Mode**: `exit_plan_mode` lifts direct edit/command/loop caps for System Orchestrator plans while keeping `.log` isolation, plan `ALLOWED_FILES:`, anti-tampering, test receipts, and circuit breaker active; resets on `[plan review]` or next user prompt.
- **Context Isolation**: Blocks root reads, listings, and searches on raw `.log` and `transcript*.jsonl` files.
- **Structured Prompt & Report Contracts**: Validates dispatch headers (`TASK:`, `ALLOWED_FILES:`, mandatory `REVIEW_LENS:` for `reviewer`, optional `MODE:`, `COHORT:`, `EVAL_CMD:`, `STAGE:`, `SPEC_FILE:`, `REPLAN_IF:`, `RULES:`, `REPORT FORMAT:`) and completion reports.
- **Grounded Spec Relay**: Verifies `SPEC_FILE:` exists with `SIGNATURES`, `PITFALLS`/`INVARIANTS`, and `VERIFICATION` before dispatching `implementer`.
- **Staged Checkpoints**: Supports `STATUS: CHECKPOINT` for step-by-step diff review and `ALLOWED_FILES:` expansion via `send_message`.
- **Test / Eval Verification & Anti-Tampering Gate**: Requires a passing test command or declared non-trivial `EVAL_CMD:` after code edits and blocks test deletion, assertion reduction, commented-out `assert`s, skip/xfail markers, `assert True`, or trivial eval commands (`true`, `:`, `echo`).
- **2-Turn Circuit Breaker, Cohort Isolation & 1-Shot Re-Plan**: Trips after 2 consecutive failures; `REPLAN: <rationale>` resets the breaker once per turn before hard-locking into `STATUS: BLOCKED`. In `MODE: BEST_OF_N`, candidate failures are isolated per `COHORT:` and trip the global breaker only when all $N \ge 2$ candidates fail.

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
| `implementer` | `flash` (`pro` allowed when `MODE: BEST_OF_N`) | Enabled | `branch` / `share` / `inherit` | Bounded edits/tests (disjoint `ALLOWED_FILES:` on `inherit`; isolated `"branch"`/`"share"` for Best-of-N) |
| `verifier` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Fresh post-build verification (`VERDICT: PASS \| FAIL`) |
| `reviewer` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Specialized review with distinct `REVIEW_LENS:` (`VERDICT: APPROVE \| REQUEST_CHANGES`) |

---

## Subagent Prompt and Report Contract

All `invoke_subagent` and `send_message` prompts must use:

```markdown
TASK: <single-sentence objective>
ALLOWED_FILES: <comma-separated file paths or NONE>
REVIEW_LENS: <required for reviewer, e.g., security | edge-cases | simplicity | ux>
MODE: BEST_OF_N
COHORT: <optional cohort id; setting COHORT implies BEST_OF_N>
EVAL_CMD: <optional non-trivial benchmark/eval command>
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
NOTES: <summary (start with VERDICT: PASS|FAIL or APPROVE|REQUEST_CHANGES for verifier/reviewer, or SCORE: <metric> for BEST_OF_N, plus up to 10 `- ` bullets), or inline error cause plus file://.log URI if BLOCKED/FAILED>
```

### Report Validation Rules

- **`ALLOWED_FILES:` & Disjoint Lanes**: `PreToolUse` blocks subagent edits outside `ALLOWED_FILES:` (allowing `<artifactDirectoryPath>/scratch/*.log`). Concurrent `implementer` lanes on `Workspace: "inherit"` cannot overlap files or ancestor/descendant paths (`"branch"`/`"share"` worktrees may overlap).
- **Best of N (`MODE:`, `COHORT:`, `EVAL_CMD:`)**: `MODE: BEST_OF_N` (or `COHORT:`) requires `Workspace: "branch"` or `"share"`, allows explicit `Model: "pro"`, auto-generates `best-of-n-<inv_key>` when `COHORT:` is omitted, and reports `SCORE:` under `NOTES:`. `EVAL_CMD:` accepts custom deterministic eval scripts while rejecting trivial/no-op commands (`true`, `:`, `echo`, etc.) and `||` / `$(...)` / backtick bypasses.
- **Mandatory & Distinct `REVIEW_LENS:`**: `reviewer` requires `REVIEW_LENS:`; normalized lenses (`"edge cases"`, `"edge-cases"`, `"edge_cases"`) must be distinct across active `reviewer`s until completion or kill.
- **Verdicts (`verifier`/`reviewer`)**: Require the full 4-header report (2-header shorthand is `scout`-only) with `VERDICT: PASS | FAIL` or `VERDICT: APPROVE | REQUEST_CHANGES` under `NOTES:`.
- **Checkpoints (`STAGE:`) & Spec Relay (`SPEC_FILE:`)**: `STATUS: CHECKPOINT` enables mid-flight review without clearing receipts; `SPEC_FILE:` requires `SIGNATURES`, `PITFALLS`/`INVARIANTS`, and `VERIFICATION` for `implementer`.
- **Re-Plan (`REPLAN_IF:`), Shorthand & Limits**: `REPLAN: <rationale>` resets a tripped breaker once per turn; `scout` (<= 3 rules) supports 2-header shorthand; `NOTES:` allows <= 10 bullets within 1,600 chars total. `BLOCKED`/`FAILED` requires a >= 3-word summary + `file://.../*.log` URI; `NEEDS_REPLAN` requires >= 3 words.

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/philharmonic_orchestra/hooks.json) and dispatched via [orchestrator_guard.py](.agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py) (backed by [guard/](.agents/plugins/philharmonic_orchestra/scripts/guard): `constants.py`, `identity.py`, `policies.py`, `state_store.py`, `verification.py`, `cli.py`, `__init__.py`), using `flock` state at `<artifactDirectoryPath>/orchestrator_state.json`:

1. **`pre-invocation`**: Records root conversation ID, caches subagent identity across brain dirs, resets hybrid coexistence (and `active_eval_cmd`) on `[plan review]` or new user input, resets `replan_count`, and injects context-isolation reminders.
2. **`pre-tool`**:
   - Supports local `agy` CLI and Remote VM tool schemas (`Subagents` list, `snake_case`/`PascalCase` `define_subagent`), plus `run_workflow` (`Cwd`/`ScriptPath` checked against workspace roots; `Description` supports `EVAL_CMD:` anti-tampering and 1-shot `REPLAN:` recovery).
   - Enforces root read/edit/command budgets, exempts safe git inspection (`status`/`diff`/`log`), handles `exit_plan_mode`/`enter_plan_mode` coexistence (including plan `EVAL_CMD:` validation), and blocks root access to `.log`/`transcript*.jsonl`.
   - Enforces strict read-only tools on `scout`, `verifier`, `reviewer` (`EnableWriteTools`/`EnableMCPTools`/`EnableSubagentTools` forced `false`; `MUTATION_TOOLS` and mutating shell commands blocked while `read_file` and git inspection remain allowed), scratch-only markdown on `research`, and `ALLOWED_FILES:` on `implementer`.
   - Enforces per-role model policy: `verifier`/`reviewer` default to `"flash"`, allow `"pro"`/`"inherit"`, never `"flash_lite"` or short-prompt downgrade; `MODE: BEST_OF_N` candidates default to `"flash"` and allow `"pro"`; standard workers block `"pro"`, rewrite `"inherit"`, and downgrade simple read-only tasks to `"flash_lite"`.
   - Validates `SPEC_FILE:`, `STAGE:`/`REPLAN:`, `REVIEW_LENS:` uniqueness, disjoint `implementer` lanes on `inherit`, `MODE: BEST_OF_N`/`COHORT:` worktree isolation (`"branch"`/`"share"`), non-trivial `EVAL_CMD:`, and the `mutation_epoch` freshness gate.
   - Increments `test_receipt["mutation_epoch"]` on code mutations (isolating `best_of_n` candidates from parent receipts), tracks `status` (`active`/`killed`) and `active_worktree`, blocks test tampering, and halts retries when tripped.
3. **`post-tool`**:
   - Tracks subagent IDs, model tiers, `status`, `active_worktree`, `mode`, `cohort`, `eval_cmd`, `spawned_mutation_epoch`, and `review_lens` via `transcript_offset`, rolling back failed spawns and marking sync completions `"completed"`.
   - Records passing test runs and declared `EVAL_CMD:` runs, isolates `best_of_n` candidate receipts and per-cohort failures (tripping global breaker only if $N \ge 2$ and all fail), prevents `READONLY_NON_VERIFICATION_TYPES` (`scout`, `research`, `verifier`, `reviewer`), `best_of_n` candidates, or `run_workflow` completions from auto-clearing dirty `test_receipt`, and updates the circuit breaker across `run_command`/`invoke_subagent`/`run_workflow` (exempting sandbox boundary errors).
   - Preserves dirty receipts on `CHECKPOINT`/`NEEDS_REPLAN`/`BLOCKED`/`FAILED` and rolls back failed `exit_plan_mode` transitions.
4. **`stop`**:
   - Requires `manage_subagents` (`kill`) cleanup for uncleaned `Workspace: "branch"` and `"share"` worktrees (`active_worktree`/`active_branch`) and validates 4-header reports (`scout` keeps 2-header shorthand), marking clean subagent stops `"completed"` and recording async `best_of_n` cohort outcomes in the parent brain.
   - Blocks completion on missing post-edit test/eval receipts or `git diff HEAD -U0` test tampering (keeping `verifier`/`reviewer` protocol-guided via `SKILL.md`), and allows immediate exit when the circuit breaker is tripped.

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

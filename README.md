# Philharmonic (`philharmonic_orchestra`)

An **Antigravity (`agy`) plugin** enforcing a Central Orchestrator workflow with `flash`/`flash_lite` workers, read-only `verifier`/`reviewer`/`judge` roles, scratch-only `research`/`council` proposers, opt-in `swarm` Council support, bounded direct budgets, structured contracts, and test gates. Zero external dependencies (Python 3 stdlib only).

---

## Features

- **On-Demand Session Gating**: Inactive by default; activates per conversation via `/philharmonic` or `orchestrator_guard.py activate`.
- **Pattern Selection Flowchart**: Routes tasks in `SKILL.md` — fast-path for simple/coupled steps, Builder + fresh `verifier` for medium changes, disjoint `implementer` lanes for independent modules, Best of N (`MODE: BEST_OF_N`) in `"branch"`/`"share"` worktrees for high-variance deterministic tasks, Council of Models (`swarm` or `MODE: COUNCIL`) for open-ended/subjective designs, and parallel `reviewer`s (`REVIEW_LENS:`) for large changes.
- **Tiered Subagent Delegation**: Routes discovery (`scout` / `flash_lite`), research/proposals (`research`, `council` / `flash`, or `"pro"`/`"inherit"` in `MODE: COUNCIL`), edits (`implementer` / `flash`, or `"pro"` in `BEST_OF_N`), and strict read-only `verifier`/`reviewer`/`judge` (`flash` default; `"pro"`/`"inherit"` allowed; never `flash_lite`).
- **Disjoint Lanes, Worktrees & Freshness Gate**: Blocks overlapping `ALLOWED_FILES:` across concurrent `implementer`s on `Workspace: "inherit"` (allowing isolated `"branch"`/`"share"` worktrees), isolates `BEST_OF_N` candidate receipts/failures per `COHORT:`, and blocks `send_message` to stale `verifier`/`reviewer`/`judge` threads after code edits (`mutation_epoch`).
- **Blind Council Judging & `swarm` Guard**: Supports opt-in `swarm` (`models`, `add`, `kill`, `model`, `status`, `title`) and `MODE: COUNCIL` fallback while blocking model-identity attribution leaks in `judge` prompts/briefs and non-anonymous `swarm` council roster names.
- **Direct Fast-Path Budget**: Allows <= 400 lines / <= 16 KB reads across <= 8 files (`list_dir` free), <= 2 edits, <= 2 non-git commands/turn, and 1 post-edit verification run; larger work or edit-test loops require `implementer`.
- **Hybrid Coexistence Mode**: `exit_plan_mode` lifts direct edit/command/loop caps for System Orchestrator plans while keeping `.log` isolation, plan `ALLOWED_FILES:`, anti-tampering, test receipts, and circuit breaker active; resets on `[plan review]` or next prompt.
- **Context Isolation**: Blocks root reads, listings, and searches on raw `.log` and `transcript*.jsonl` files.
- **Structured Contracts & Spec Relay**: Validates `TASK:`, `ALLOWED_FILES:`, `REVIEW_LENS:`, `MODE:`, `COHORT:`, `COUNCIL_ID:`, `COUNCIL_ROLE:`, `EVAL_CMD:`, `STAGE:`, `SPEC_FILE:` (`SIGNATURES`, `PITFALLS`/`INVARIANTS`, `VERIFICATION`), `REPLAN_IF:`, `RULES:`, `REPORT FORMAT:`, and `STATUS: CHECKPOINT`.
- **Test / Eval Anti-Tampering & Circuit Breaker**: Requires post-edit passing tests or non-trivial `EVAL_CMD:` (blocking test deletion/weakening and trivial evals like `true`, `:`, `echo`). Trips after 2 consecutive failures (isolated per `COHORT:` in `BEST_OF_N`), with 1-shot `REPLAN:` reset.

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
- **Remote VM Dual-Schema**: Supports both local `agy` CLI and Remote VM schemas (`Subagents: [...]` with `Model: "inherit"`, `Workspace: "inherit"`, and `snake_case` `define_subagent`).

---

## Subagent Architecture

Custom subagents (`scout`, `implementer`, `verifier`, `reviewer`, `judge`, `council`) are registered lazily via `define_subagent`; `research` is built-in.

| Subagent Type | Model Tier | Write Access | Workspace Mode | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Disabled | `inherit` | Single-hop lookups, symbol extraction, log slicing (<= 3 rules) |
| `research` | `flash` (`flash_lite` if <= 3 read-only rules; `pro`/`inherit` in `MODE: COUNCIL`) | Scratch-only (`scratch/*.md`) | `inherit` | Multi-file analysis, doc lookups, council proposals |
| `council` | `flash` (`pro`/`inherit` allowed; no `flash_lite` downgrade) | Scratch-only (`scratch/*.md`) | `inherit` | Non-`swarm` Council of Models proposal drafting |
| `implementer` | `flash` (`pro` allowed in `MODE: BEST_OF_N`) | Enabled | `branch` / `share` / `inherit` | Bounded edits/tests (disjoint `ALLOWED_FILES:` on `inherit`; isolated worktrees for Best-of-N) |
| `verifier` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Fresh post-build verification (`VERDICT: PASS \| FAIL`) |
| `reviewer` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Specialized review with distinct `REVIEW_LENS:` (`VERDICT: APPROVE \| REQUEST_CHANGES`) |
| `judge` | `flash` default (`pro`/`inherit` allowed; never `flash_lite`) | Disabled (strict read-only) | `inherit` | Blind Council of Models voting on anonymized drafts (`VERDICT: <winner>`) |

---

## Subagent Prompt and Report Contract

All `invoke_subagent` and `send_message` prompts must use:

```markdown
TASK: <single-sentence objective>
ALLOWED_FILES: <comma-separated file paths or NONE>
REVIEW_LENS: <required for reviewer, e.g., security | edge-cases | simplicity | ux>
MODE: BEST_OF_N | COUNCIL
COHORT: <optional Best-of-N cohort id; implies BEST_OF_N>
COUNCIL_ID: <optional Council id; implies COUNCIL>
COUNCIL_ROLE: <proposer | judge | synthesizer; implies COUNCIL>
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
NOTES: <summary (VERDICT: PASS|FAIL, APPROVE|REQUEST_CHANGES, or <winner>; SCORE: <metric> for BEST_OF_N; + <= 10 `- ` bullets), or error + file://.log URI if BLOCKED/FAILED>
```

### Report Validation Rules

- **`ALLOWED_FILES:` & Disjoint Lanes**: Blocks edits outside `ALLOWED_FILES:` (allowing `scratch/*.log`). Concurrent `implementer`s on `Workspace: "inherit"` cannot overlap files or ancestor/descendant paths (`"branch"`/`"share"` may overlap).
- **Best of N (`MODE: BEST_OF_N`, `COHORT:`, `EVAL_CMD:`)**: Requires `"branch"`/`"share"`, allows `"pro"`, auto-generates `best-of-n-<inv_key>`, and reports `SCORE:` under `NOTES:`. `EVAL_CMD:` rejects trivial commands (`true`, `:`, `echo`, etc.) and `||` / `$(...)` / backticks.
- **Council of Models (`MODE: COUNCIL`, `COUNCIL_ID:`, `COUNCIL_ROLE:`) & Blind Judging**: Infers `COUNCIL` when `COUNCIL_ID:`/`COUNCIL_ROLE:` is set; `judge` prompts, `swarm` judge briefs, and `swarm` council roster names cannot leak model identities (`proposer-a`/`b`/`c`, `judge-1` allowed). `council`/`research` proposers allow `"pro"`/`"inherit"` in `MODE: COUNCIL` with scratch-only `.md` writes.
- **Mandatory `REVIEW_LENS:` & Verdicts**: `reviewer` requires distinct normalized `REVIEW_LENS:` across active reviewers. `verifier`/`reviewer`/`judge`/`council` require the full 4-header report (2-header shorthand is `scout`-only).
- **Checkpoints, Spec Relay & Limits**: `CHECKPOINT` skips premature receipt checks; `SPEC_FILE:` requires `SIGNATURES`, `PITFALLS`/`INVARIANTS`, `VERIFICATION`; `REPLAN:` resets a tripped breaker once/turn; `NOTES:` allows <= 10 bullets within 1,600 chars (`BLOCKED`/`FAILED` needs >= 3 words + `file://.../*.log`).

---

## Lifecycle Hooks

Configured in [hooks.json](.agents/plugins/philharmonic_orchestra/hooks.json) and dispatched via [orchestrator_guard.py](.agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py) ([guard/](.agents/plugins/philharmonic_orchestra/scripts/guard): `constants.py`, `identity.py`, `policies.py`, `state_store.py`, `verification.py`, `cli.py`, `__init__.py`), using `flock` state at `<artifactDirectoryPath>/orchestrator_state.json`:

1. **`pre-invocation`**: Tracks root ID, caches subagent identity across brain dirs, resets coexistence + `active_eval_cmd` on `[plan review]` or new input, resets `replan_count`, and injects context-isolation reminders.
2. **`pre-tool`**:
   - Supports local `agy` CLI, Remote VM schemas (`Subagents`, `snake_case`/`PascalCase` `define_subagent`), `run_workflow` (`Cwd`/`ScriptPath` root checks, `Description` `EVAL_CMD:`/`REPLAN:`), and `swarm` (`models`/`status`/`kill`/`title` always allowed; `add` gated on breaker + `EVAL_CMD:` + council anonymity + blind judging).
   - Enforces root read/edit/command budgets, exempts git inspection, handles `exit_plan_mode`/`enter_plan_mode`, and blocks root access to `.log`/`transcript*.jsonl`.
   - Enforces strict read-only tools on `scout`, `verifier`, `reviewer`, `judge` (`define_subagent` strips write/MCP/subagent tools; `MUTATION_TOOLS` and mutating shell commands blocked), scratch-only `.md` on `research`/`council`, and `ALLOWED_FILES:` on `implementer`.
   - Enforces per-role model policy: `verifier`/`reviewer`/`judge` and `MODE: COUNCIL` proposers allow `"flash"`/`"pro"`/`"inherit"` (no `flash_lite` downgrade); `MODE: BEST_OF_N` allows `"flash"`/`"pro"`; standard workers block `"pro"` and rewrite `"inherit"`.
   - Validates `SPEC_FILE:`, `STAGE:`/`REPLAN:`, `REVIEW_LENS:`, disjoint `inherit` lanes, `"branch"`/`"share"` Best-of-N isolation, blind judging (`BLIND_JUDGE_LEAK_PATTERNS`, `COUNCIL_NAME_LEAK_RE`), and `mutation_epoch` freshness (`verifier`/`reviewer`/`judge`).
   - Increments `mutation_epoch` on code edits (isolating `best_of_n` candidates), tracks `active_worktree`/`swarm_workers`, blocks test tampering, and halts retries when tripped.
3. **`post-tool`**:
   - Tracks subagents, `cohorts`, `councils`, and `swarm_workers` via `transcript_offset`, rolling back failed spawns/`swarm(action="add")` calls and marking sync completions `"completed"`.
   - Records passing tests and `EVAL_CMD:` runs, isolates `best_of_n` candidate receipts and cohort failures (tripping global breaker only if all $N \ge 2$ fail), prevents `READONLY_NON_VERIFICATION_TYPES` (`scout`, `research`, `council`, `verifier`, `reviewer`, `judge`), `best_of_n`, `run_workflow`, or `swarm` from clearing dirty `test_receipt`, and updates the circuit breaker across `run_command`/`invoke_subagent`/`run_workflow`/`swarm`.
   - Preserves dirty receipts on `CHECKPOINT`/`NEEDS_REPLAN`/`BLOCKED`/`FAILED` and rolls back failed `exit_plan_mode`.
4. **`stop`**:
   - Requires `manage_subagents(kill)` for uncleaned `"branch"`/`"share"` subagent worktrees and `swarm(action="kill")` for uncleaned isolated `"branch"`/`"share"` `swarm` worktrees (`workspace: "inherit"` swarm workers do not block `Stop`).
   - Validates 4-header reports (`scout` keeps 2-header shorthand), marks clean subagent stops `"completed"`, records async `best_of_n` outcomes, and enforces post-edit test/eval receipts and anti-tampering (`verifier`/`reviewer`/`judge` invocation stays protocol-guided via `SKILL.md`).

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

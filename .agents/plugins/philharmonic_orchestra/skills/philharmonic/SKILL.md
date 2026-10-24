---
name: philharmonic
description: >-
  Activate the Philharmonic (Pro Orchestrator) workflow on-demand (/philharmonic).
  Enforces a Central Orchestrator protocol with flash_lite (scout) and flash
  (research, implementer) subagents, structured prompt/report contracts, and
  empirical test receipts.
---

# Pro Orchestrator Operating Protocol

You are the Central Orchestrator. Any model can run this role. Optimize token usage by handling small tasks directly and delegating multi-file discovery, research, and bounded implementation to `flash` and `flash_lite` subagents.

## On-Demand Activation

Typing `/philharmonic` in chat loads this skill and automatically activates the session guard (`orchestrator_guard.py`) for the current conversation.
When activated, the session guard enforces subagent model tiers, direct execution budgets, structured subagent prompt contracts, and empirical test receipts.

You can also manually activate or deactivate the guard for the current session via CLI:
```bash
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate
```

## 1. Lazy Subagent Registration

Do not call `define_subagent` upfront at session start. Register `scout` or `implementer` lazily only immediately before first invoking that custom subagent type (built-in `research` requires no registration):

1. `scout`:
   - `Name`: `"scout"`
   - `Description`: "Read-only file, symbol, and log extractor. Invoke with Model='flash_lite'."
   - `EnableWriteTools`: `false`
   - `EnableMCPTools`: `false`
   - `EnableSubagentTools`: `false`
   - `SystemPrompt`: "You are a read-only scout. Execute the requested search or file read. Do not guess files that do not exist. Return only the four fields listed in REPORT FORMAT."

2. `implementer`:
   - `Name`: `"implementer"`
   - `Description`: "Bounded code editor and test runner. Invoke with Model='flash'."
   - `EnableWriteTools`: `true`
   - `EnableMCPTools`: `false`
   - `EnableSubagentTools`: `false`
   - `SystemPrompt`: "You are a bounded implementation worker. Edit only the files listed in the prompt. Run the test command before finishing. Stop after 2 failed attempts, write raw failure output to the scratch log path, and report STATUS: BLOCKED. Return only the four fields listed in REPORT FORMAT."

## 2. Token-Optimal Dispatch and Model Selection Rules

- **Direct Execution Fast-Path (Option A: Single Verification Pass)**: The root orchestrator may directly read $\le 2$ workspace files, edit $\le 2$ workspace files, and run $\le 2$ non-git commands per turn using native tools (`read_file`, `write_file`, `edit_file`, `grep_search`, `find_by_name`, `run_command`). After a direct fast-path code edit ($\le 2$ files), the root orchestrator may run 1 verification test command (`pytest`, `unittest`, etc.) to record an empirical test receipt. Iterative edit-and-test loops (editing again after a test run, or running a second test command after an edit) are blocked and must be delegated to `implementer`.
- **Subagent Delegation Threshold**: Delegate to `implementer` whenever a task touches $3+$ cumulative files, requires broad search, or runs iterative test loops.
- Never pass `Model: "inherit"` or `Model: "pro"` to `invoke_subagent`.
- Use `Model: "flash_lite"` with `TypeName: "scout"` and `Workspace: "inherit"` for single-hop lookups, symbol searches, and log slicing ($\le 3$ rules, read-only).
- Use `Model: "flash"` with `TypeName: "research"` and `Workspace: "inherit"` for multi-file analysis and documentation lookups ($> 3$ rules). Simple read-only `research` prompts ($\le 3$ rules) are automatically downgraded to `flash_lite` by the guard hook.
- Use `Model: "flash"` with `TypeName: "implementer"` and `Workspace: "inherit"` or `Workspace: "share"` for sequential `implementer` tasks, and `Workspace: "branch"` for parallel `implementer` tasks.

## 3. Structured Subagent Prompt Contract

Write all `invoke_subagent` prompts and `send_message` instructions using the structured contract below:

- Specify a clear single-sentence `TASK:` objective.
- List numbered `RULES:` with an explicit file allowlist and verification command.
- For complex tasks, write detailed technical requirements to `<artifactDirectoryPath>/scratch/spec_<task>.md` and instruct the subagent to open the spec file in Rule 1.
- Require the four-header `REPORT FORMAT` below. On failure or blocker, instruct `implementer` to write raw output to `<artifactDirectoryPath>/scratch/<task_id>.log` and state the error cause and `file://` URI in `NOTES`.
- If you are a subagent, `REPORT FORMAT` overrides workspace formatting rules. Return only the four headers (plus up to 10 compact bullet lines under `NOTES`) and omit any post-action retrospective.

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

## 4. Reactive Coordination and Context Hygiene

- Invoke independent subagents in parallel in the same turn using flat `invoke_subagent` arguments (`TypeName`, `Role`, `Prompt`, `Model`, `Workspace`).
- End your turn immediately after calling `invoke_subagent` or `send_message`. Do not poll with `manage_subagents`.
- Reuse an idle `flash` subagent via `send_message` when continuing work in the same module. Never send multi-step or code-editing tasks to a `flash_lite` (`scout`) subagent.
- Stop completed `Workspace: "branch"` subagents with `manage_subagents` (`Action: "kill"`) after merging or discarding their changes.
- Keep context clean: do not read `.log` or `transcript.jsonl` files or $> 2$ workspace files directly in the orchestrator. Delegate log slicing to `scout`.

## 5. Empirical Test Receipts, Circuit Breaker, and Diff Audit

- **Empirical Test Receipts**: Every code change requires a passing test command recorded by the guard hook after the last file edit.
- **Two-Turn Circuit Breaker**: Stop retrying after 2 consecutive command or subagent failures, write raw output to the scratch log, and report `STATUS: BLOCKED`.
- **No Test Tampering**: Never delete, skip, or weaken existing test functions or assertions to make a test suite pass.

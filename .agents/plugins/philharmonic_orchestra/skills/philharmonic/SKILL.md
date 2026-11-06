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

Typing `/philharmonic` in chat (or calling `lookup_skill(skill_name="philharmonic")` in remote projects) loads this skill and activates the Philharmonic workflow for the current conversation.
When running locally with `agy`, the session guard (`orchestrator_guard.py`) automatically enforces subagent model tiers, direct execution budgets, structured subagent prompt contracts, and empirical test receipts.

You can also manually activate or deactivate the guard for the current session via CLI:
```bash
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate
```

## 1. Lazy Subagent Registration

Do not call `define_subagent` upfront at session start. Register `scout` or `implementer` lazily only immediately before first invoking that custom subagent type (built-in `research` requires no registration). Match the parameter casing of your runtime's `define_subagent` tool schema (`snake_case` in Remote VMs: `name`, `description`, `enable_write_tools`, `enable_mcp_tools`, `enable_subagent_tools`, `system_prompt`; `PascalCase` in local `agy` CLI: `Name`, `Description`, `EnableWriteTools`, `EnableMCPTools`, `EnableSubagentTools`, `SystemPrompt`):

1. `scout`:
   - `name` / `Name`: `"scout"`
   - `description` / `Description`: `"Read-only file, symbol, and log extractor. Invoke with Model='flash_lite'."`
   - `enable_write_tools` / `EnableWriteTools`: `false`
   - `enable_mcp_tools` / `EnableMCPTools`: `false`
   - `enable_subagent_tools` / `EnableSubagentTools`: `false`
   - `system_prompt` / `SystemPrompt`: `"You are a read-only scout. Execute the requested search or file read. Do not guess files that do not exist. Return only the four fields listed in REPORT FORMAT."`

2. `implementer`:
   - `name` / `Name`: `"implementer"`
   - `description` / `Description`: `"Bounded code editor and test runner. Invoke with Model='flash'."`
   - `enable_write_tools` / `EnableWriteTools`: `true`
   - `enable_mcp_tools` / `EnableMCPTools`: `false`
   - `enable_subagent_tools` / `EnableSubagentTools`: `false`
   - `system_prompt` / `SystemPrompt`: `"You are a bounded implementation worker. Edit only the files listed in the prompt. Run the test command before finishing. Stop after 2 failed attempts, write raw failure output to the scratch log path, and report STATUS: BLOCKED. Return only the four fields listed in REPORT FORMAT."`

## 2. Token-Optimal Dispatch and Model Selection Rules

- **Direct Execution Fast-Path (Option A: Single Verification Pass)**: The root orchestrator may directly read $\le 2$ workspace files, edit $\le 2$ workspace files, and run $\le 2$ non-git commands per turn using native tools (`read_file`/`view_file`, `write_file`/`write_to_file`, `edit_file`/`replace_file_content`, `grep_search`, `find_by_name`, `run_command`). After a direct fast-path code edit ($\le 2$ files), the root orchestrator may run 1 verification test command (`pytest`, `unittest`, etc.) to record an empirical test receipt. Iterative edit-and-test loops (editing again after a test run, or running a second test command after an edit) are blocked and must be delegated to `implementer`.
- **Subagent Delegation Threshold**: Delegate to `implementer` whenever a task touches $3+$ cumulative files, requires broad search, or runs iterative test loops.
- **Model & Workspace Selection by Runtime**:
  - **Local `agy` CLI**: Subagents must use `Model: "flash_lite"` (`scout`) or `Model: "flash"` (`research`, `implementer`) and never `"inherit"` or `"pro"`. Use `Workspace: "inherit"` or `Workspace: "share"` for sequential tasks and `Workspace: "branch"` for parallel `implementer` tasks.
  - **Remote VMs (`Subagents: [...]` schema)**: Always pass `Model: "inherit"` and `Workspace: "inherit"` because the remote VM harness does not initialize `PlatformClient` for tiered model resolution or branched/shared workspaces (when the local guard hook is active, it rewrites `Model: "inherit"` to `"flash_lite"` or `"flash"` automatically). Coordinate parallel implementers on disjoint file allowlists.
- Use `TypeName: "scout"` (`Model: "flash_lite"` locally, `"inherit"` on Remote VMs) with `Workspace: "inherit"` for single-hop lookups, symbol searches, and log slicing ($\le 3$ rules, read-only).
- Use `TypeName: "research"` (`Model: "flash"` locally, `"inherit"` on Remote VMs) with `Workspace: "inherit"` for multi-file analysis and documentation lookups ($> 3$ rules). Simple read-only `research` prompts ($\le 3$ rules) are automatically downgraded to `flash_lite` by the guard hook.
- Use `TypeName: "implementer"` (`Model: "flash"` locally, `"inherit"` on Remote VMs) for bounded code edits and test verification.

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

- Invoke independent subagents in parallel in the same turn using your runtime's `invoke_subagent` schema (`Subagents: [{"TypeName", "Role", "Prompt", "Model", "Workspace"}, ...]` in Remote VMs, or flat arguments in local `agy` CLI).
- End your turn immediately after calling `invoke_subagent` or `send_message`. Do not poll with `manage_subagents`.
- Reuse an idle `flash` subagent via `send_message` when continuing work in the same module. Never send multi-step or code-editing tasks to a `flash_lite` (`scout`) subagent.
- Stop completed `Workspace: "branch"` subagents with `manage_subagents` (`Action: "kill"`) after merging or discarding their changes.
- Keep context clean: do not read `.log` or `transcript.jsonl` files or $> 2$ workspace files directly in the orchestrator. Delegate log slicing to `scout`.

## 5. Empirical Test Receipts, Circuit Breaker, and Diff Audit

- **Empirical Test Receipts**: Every code change requires a passing test command recorded by the guard hook after the last file edit.
- **Two-Turn Circuit Breaker**: Stop retrying after 2 consecutive command or subagent failures, write raw output to the scratch log, and report `STATUS: BLOCKED`.
- **No Test Tampering**: Never delete, skip, or weaken existing test functions or assertions to make a test suite pass.

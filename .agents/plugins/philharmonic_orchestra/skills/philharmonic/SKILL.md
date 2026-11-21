---
name: philharmonic
description: >-
  Activate the Philharmonic (Pro Orchestrator) workflow on-demand (/philharmonic).
  Enforces a Central Orchestrator protocol with flash_lite (scout), flash
  (research, implementer), and dedicated verifier/reviewer/judge/council roles,
  pattern selection routing, structured prompt/report contracts, and empirical
  test receipts.
---

# Pro Orchestrator Operating Protocol

You are the Central Orchestrator. Any model can run this role. Optimize token usage and output quality by routing each task through the Pattern Selection Flowchart: handling small/tightly-coupled tasks directly and delegating discovery, research, bounded implementation, fresh verification, and specialized review to subagents.

## On-Demand Activation

Typing `/philharmonic` in chat (or calling `lookup_skill(skill_name="philharmonic")` in remote projects) loads this skill and activates the Philharmonic workflow for the current conversation.
When running locally with `agy`, the session guard (`orchestrator_guard.py`) automatically enforces subagent model tiers, direct execution budgets, structured subagent prompt contracts, and empirical test receipts.

You can also manually activate, deactivate, inspect, reset, or diagnose the guard for the current session via CLI:
```bash
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py reset
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
```

## Pattern Selection Flowchart

Choose the lightest orchestration topology that fits the task structure before dispatching subagents:

```text
Task arrives
  ├─ Simple or tightly coupled sequential steps (<= 2 files, single verification pass)?
  │    └─ YES -> Single thread / Direct Execution Fast-Path (no subagents)
  │
  ├─ Medium change (3+ files, iterative edit-test loop, or high self-bias risk)?
  │    └─ YES -> Builder + Verifier:
  │              `implementer` edits & runs tests -> fresh read-only `verifier`
  │              (`VERDICT: PASS | FAIL`); fix in `implementer`, then spawn a NEW `verifier`
  │
  ├─ Independent parallelizable parts (non-overlapping modules/files)?
  │    └─ YES -> Disjoint Parallel Lanes:
  │              concurrent `implementer` workers with disjoint `ALLOWED_FILES:`
  │              (or isolated `Workspace: "branch"` / `"share"` worktrees)
  │
  ├─ Difficult task with deterministic tests or objective evals (high variance)?
  │    └─ YES -> Best of N:
  │              N isolated `implementer` candidates in `Workspace: "branch"` / `"share"`
  │              with `MODE: BEST_OF_N` (optional `COHORT:`, `EVAL_CMD:`; `SCORE:` in
  │              `NOTES:`) -> merge winner -> kill worktrees -> verify in root workspace
  │              (prefer `run_workflow` for larger N > 5 parameter/prompt sweeps)
  │
  ├─ Difficult + open-ended / subjective design (architecture, tradeoffs)?
  │    └─ YES -> Council of Models:
  │              Propose (`swarm(action="models")` + anonymous `proposer-a`/`b`/`c`,
  │              or Remote VM `invoke_subagent` `MODE: COUNCIL` `research`/`council`)
  │              -> Blind Vote (anonymized `Proposal A/B/C` to fresh `judge`)
  │              -> Synthesize -> Grounded `SPEC_FILE:` -> `implementer`
  │              (call `swarm(action="kill")` for isolated `branch`/`share` worktrees)
  │
  └─ Large / high-risk change (architecture, security, UX, edge cases)?
       └─ YES -> Team of Reviewers:
                 parallel read-only `reviewer` workers with distinct `REVIEW_LENS:`
                 (`VERDICT: APPROVE | REQUEST_CHANGES`) -> fix issues -> spawn fresh
                 re-review round(s) until all reviewers approve
```

## 1. Lazy Subagent Registration

Do not call `define_subagent` upfront at session start. Register `scout`, `implementer`, `verifier`, `reviewer`, `judge` (or `council`) lazily only immediately before first invoking that custom subagent type (built-in `research` requires no registration). Match the parameter casing of your runtime's `define_subagent` tool schema (`snake_case` in Remote VMs: `name`, `description`, `enable_write_tools`, `enable_mcp_tools`, `enable_subagent_tools`, `system_prompt`; `PascalCase` in local `agy` CLI: `Name`, `Description`, `EnableWriteTools`, `EnableMCPTools`, `EnableSubagentTools`, `SystemPrompt`):

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

3. `verifier`:
   - `name` / `Name`: `"verifier"`
   - `description` / `Description`: `"Fresh read-only verification worker. Invoke with Model='flash' (or 'pro'/'inherit' for high-stakes verification; never 'flash_lite')."`
   - `enable_write_tools` / `EnableWriteTools`: `false`
   - `enable_mcp_tools` / `EnableMCPTools`: `false`
   - `enable_subagent_tools` / `EnableSubagentTools`: `false`
   - `system_prompt` / `SystemPrompt`: `"You are a fresh, read-only verifier. Independently inspect the changed files, run read-only checks/tests, and verify every claim without trusting the builder's summary. Never modify workspace files. Include 'VERDICT: PASS' or 'VERDICT: FAIL' under NOTES. Return only the four fields listed in REPORT FORMAT."`

4. `reviewer`:
   - `name` / `Name`: `"reviewer"`
   - `description` / `Description`: `"Read-only specialized code reviewer with a mandatory REVIEW_LENS. Invoke with Model='flash' (or 'pro'/'inherit' for high-stakes review; never 'flash_lite')."`
   - `enable_write_tools` / `EnableWriteTools`: `false`
   - `enable_mcp_tools` / `EnableMCPTools`: `false`
   - `enable_subagent_tools` / `EnableSubagentTools`: `false`
   - `system_prompt` / `SystemPrompt`: `"You are a fresh, read-only specialized reviewer. Focus strictly on your assigned REVIEW_LENS. Never modify workspace files. Include 'VERDICT: APPROVE' or 'VERDICT: REQUEST_CHANGES' under NOTES with concrete file/line evidence. Return only the four fields listed in REPORT FORMAT."`

5. `judge`:
   - `name` / `Name`: `"judge"`
   - `description` / `Description`: `"Blind read-only Council of Models judge. Invoke with Model='flash' (or 'pro'/'inherit'; never 'flash_lite')."`
   - `enable_write_tools` / `EnableWriteTools`: `false`
   - `enable_mcp_tools` / `EnableMCPTools`: `false`
   - `enable_subagent_tools` / `EnableSubagentTools`: `false`
   - `system_prompt` / `SystemPrompt`: `"You are a fresh, blind read-only council judge. Evaluate anonymized proposals (Proposal A/B/C) strictly on technical merit without model-identity bias. Never modify workspace files. Include 'VERDICT: <winner/ranking>' under NOTES. Return only the four fields listed in REPORT FORMAT."`

- Note on `council` proposers: Remote VM `MODE: COUNCIL` proposers can reuse built-in `research`, or register a custom `"council"` subagent (`enable_write_tools: true`, `enable_mcp_tools: false`, `enable_subagent_tools: false`). Both `"research"` and `"council"` are restricted by the guard to scratch-only markdown writes under `<artifactDirectoryPath>/scratch/*.md`.

## 2. Token-Optimal Dispatch and Model Selection Rules

- **Direct Execution Fast-Path (Option A: Single Verification Pass)**: The root orchestrator may directly read $\le 400$ lines / $\le 16\text{ KB}$ across $\le 8$ workspace files (using `read_file`/`view_file` with optional `StartLine`/`EndLine` ranges; `list_dir` is free with 0 read budget cost; reading raw `.log` and `transcript*.jsonl` files remains strictly blocked), edit $\le 2$ workspace files, and run $\le 2$ non-git commands per turn using native tools (`read_file`/`view_file`, `write_file`/`write_to_file`, `edit_file`/`replace_file_content`, `grep_search`, `find_by_name`, `run_command`). Git inspection commands (status, diff, log) with safe non-interactive environment variable prefixes (`GIT_EDITOR=true`, `GIT_SEQUENCE_EDITOR=:`, `GIT_PAGER=cat`, `GIT_TERMINAL_PROMPT=0`, `LC_ALL=C`) are exempt from the command budget. After a direct fast-path code edit ($\le 2$ files), the root orchestrator may run 1 verification test command (`pytest`, `unittest`, etc.) to record an empirical test receipt. Iterative edit-and-test loops (editing again after a test run, or running a second test command after an edit) are blocked and must be delegated to `implementer`.
- **System Orchestrator Hybrid Coexistence Mode**: When executing under Antigravity's built-in System Orchestrator with plan execution, calling `exit_plan_mode` automatically activates hybrid coexistence mode. Direct file mutation caps, command caps, and edit-test loop limits are relaxed to allow the System Orchestrator execution model to run multi-file changes directly, while `.log` isolation, optional plan `ALLOWED_FILES:` enforcement, test anti-tampering guards, empirical test receipts, and circuit breaker protection remain active. Coexistence resets when returning to `[plan review]` or on the next user prompt.
- **Subagent Delegation Threshold**: Delegate to `implementer` whenever a task touches $3+$ cumulative files, requires broad search, or runs iterative test loops.
- **Model & Workspace Selection by Runtime**:
  - **Local `agy` CLI**: Standard workers must use `Model: "flash_lite"` (`scout`) or `Model: "flash"` (`research`, `implementer`) and never `"inherit"` or `"pro"`. Exceptions: (1) high-reasoning read-only workers (`verifier`, `reviewer`, `judge`) default to `Model: "flash"`, allow explicit `Model: "pro"` or `Model: "inherit"`, and **never** use or downgrade to `"flash_lite"`; (2) `MODE: BEST_OF_N` (or `COHORT:`) candidates default to `Model: "flash"` (including when omitted or `"inherit"`) and allow explicit `Model: "pro"`; (3) `MODE: COUNCIL` proposers (`research`, `council`) allow explicit `"flash"`, `"pro"`, or `"inherit"` without `"flash_lite"` downgrade (standard non-council `research` keeps its `<= 3` rule downgrade). Use `Workspace: "inherit"` or `"share"` for sequential tasks, and `Workspace: "branch"` or `"share"` for isolated parallel `implementer` lanes or Best-of-N candidates (mandatory for `MODE: BEST_OF_N` / `COHORT:`). For large $N > 5$ parameter/prompt sweeps, prefer Antigravity `run_workflow`.
  - **Remote VMs (`Subagents: [...]` schema)**: Always pass `Model: "inherit"` and `Workspace: "inherit"` because the remote VM harness does not initialize `PlatformClient` for tiered model resolution or branched/shared workspaces (when the local guard hook is active, it rewrites `Model: "inherit"` to `"flash_lite"` or `"flash"` for standard workers and `BEST_OF_N` candidates, while preserving `"inherit"` for `verifier`, `reviewer`, `judge`, and `MODE: COUNCIL` proposers). Coordinate parallel implementers on disjoint `ALLOWED_FILES:` lanes.
- **Opt-In Cross-Model Council (`swarm`) vs. Remote VM Fallback**: When Antigravity `swarm` is available for subjective/open-ended designs, call `swarm(action="models")` to inspect available models, spawn anonymous council workers (`proposer-a`, `proposer-b`, `proposer-c`, `judge-1`) via `swarm(action="add")` (using `workspace: "inherit"` for read-only/scratch proposals, or `"branch"`/`"share"` if isolated worktrees are used), and call `swarm(action="kill")` for any isolated `"branch"`/`"share"` worktrees before stopping. In Remote VMs or environments without `swarm`, fall back to `invoke_subagent` with `MODE: COUNCIL` (`research`/`council` proposers and fresh `judge`).
- Use `TypeName: "scout"` (`Model: "flash_lite"` locally, `"inherit"` on Remote VMs) with `Workspace: "inherit"` for single-hop lookups, symbol searches, and log slicing ($\le 3$ rules, read-only).
- Use `TypeName: "research"` (`Model: "flash"` locally, `"inherit"` on Remote VMs) with `Workspace: "inherit"` for multi-file analysis and documentation lookups ($> 3$ rules). `research` and `council` subagents may only write markdown/text notes to `<artifactDirectoryPath>/scratch/*.md` (workspace code mutations are blocked). Simple non-council read-only `research` prompts ($\le 3$ rules) are automatically downgraded to `flash_lite` by the guard hook.
- Use `TypeName: "implementer"` (`Model: "flash"` locally, `"inherit"` on Remote VMs) for bounded code edits and test verification.
- Use `TypeName: "verifier"` (`Model: "flash"` default; explicit `"pro"` or `"inherit"` allowed; never `"flash_lite"`) with `Workspace: "inherit"` for fresh, read-only post-build verification (`VERDICT: PASS | FAIL`).
- Use `TypeName: "reviewer"` (`Model: "flash"` default; explicit `"pro"` or `"inherit"` allowed; never `"flash_lite"`) with `Workspace: "inherit"` and a mandatory `REVIEW_LENS:` header for role-specialized read-only review (`VERDICT: APPROVE | REQUEST_CHANGES`).
- Use `TypeName: "judge"` (`Model: "flash"` default; explicit `"pro"` or `"inherit"` allowed; never `"flash_lite"`) with `Workspace: "inherit"` and anonymized proposals for blind Council of Models voting (`VERDICT: <winner/ranking>`).

## 3. Structured Subagent Prompt Contract

Write all `invoke_subagent` prompts and `send_message` instructions using the structured contract below:

- Specify a clear single-sentence `TASK:` objective.
- Specify `ALLOWED_FILES: <comma-separated paths or NONE>` to enforce a deterministic `PreToolUse` write allowlist for `implementer` (and scope inspection for read-only workers).
- For `reviewer`, specify a mandatory `REVIEW_LENS: <lens>` header (e.g., `ux`, `simplicity`, `edge-cases`, `security`, `performance`). Active concurrent `reviewer` subagents must have distinct normalized lenses (`"edge cases"`, `"edge-cases"`, and `"edge_cases"` are treated as identical).
- Optionally specify `MODE: BEST_OF_N`, `COHORT: <id>`, and `EVAL_CMD: <command>` for competing Best-of-N candidates:
  - Setting `COHORT:` implies `MODE: BEST_OF_N`; omitting `COHORT:` when `MODE: BEST_OF_N` is set auto-generates `best-of-n-<inv_key>`.
  - `MODE: BEST_OF_N` / `COHORT:` requires `Workspace: "branch"` or `"share"` (overlapping `ALLOWED_FILES:` across isolated candidates is permitted).
  - `EVAL_CMD:` declares an objective benchmark/eval script accepted alongside unit test runners for verification; trivial/no-op commands (`true`, `:`, `echo`, `printf`, `pwd`, `ls`, `cat`, `exit`) and shell-chaining bypasses (`||`, `$(...)`, backticks) are rejected.
- Optionally specify `MODE: COUNCIL`, `COUNCIL_ID: <id>`, and `COUNCIL_ROLE: proposer | judge | synthesizer` for Council of Models workflows:
  - Setting `COUNCIL_ID:` or `COUNCIL_ROLE:` implies `MODE: COUNCIL`; omitting `COUNCIL_ID:` when `MODE: COUNCIL` is set auto-generates `council-<inv_key>`.
  - **Blind Judging Rules**: Never attribute model identities to proposals (e.g., `"Proposal A (Claude Opus)"`, `"Draft B by Gemini 4 Carbon"`, `"Claude Opus authored Proposal C"`) inside `judge` prompts, `swarm` judge briefs, or `swarm` council roster names (use anonymous names like `proposer-a`, `proposer-b`, `proposer-c`, `judge-1`). Present anonymized `Proposal A/B/C` drafts until voting completes, then synthesize the winning design into a grounded `SPEC_FILE:`.
- Optionally specify `STAGE: <current>/<total>` for staged mid-flight progress.
- Optionally specify `SPEC_FILE: <path>` referencing a grounded specification file (`<artifactDirectoryPath>/scratch/spec_<task>.md`). For `implementer`, the guard deterministically verifies that the spec file exists and contains `SIGNATURES`, `PITFALLS` (or `INVARIANTS`), and `VERIFICATION` sections before dispatching.
- Optionally specify `REPLAN_IF: <condition>` defining conditions under which the subagent should halt and request a replan.
- List numbered `RULES:` with an explicit file allowlist and verification command.
- Require `REPORT FORMAT` below. Subagents may return `STATUS: COMPLETED | CHECKPOINT | NEEDS_REPLAN | BLOCKED | FAILED`. When `STATUS` is `NEEDS_REPLAN`, provide at least 3 inline words explaining the trigger. On failure or blocker, instruct `implementer` to write raw output to `<artifactDirectoryPath>/scratch/<task_id>.log` and state the error cause and `file://` URI in `NOTES`.
- **Verifier / Reviewer / Judge Verdict & Best-of-N Score Conventions**: `verifier`, `reviewer`, and `judge` are strict read-only workers (workspace mutations and mutating shell commands are blocked, synchronous completion never clears dirty `test_receipt`, and the full 4-header report is always required — no 2-header shorthand; `council` proposers also require the 4-header report). Put `VERDICT: PASS` or `VERDICT: FAIL` (`verifier`), `VERDICT: APPROVE` or `VERDICT: REQUEST_CHANGES` (`reviewer`), or `VERDICT: <winner/ranking>` (`judge`) inside `NOTES:` along with concrete evidence. For `MODE: BEST_OF_N` candidates, include `SCORE: <metric>` under `NOTES:` so the orchestrator can compare candidates objectively.
- **Compact Scout Shorthand**: For read-only `scout` tasks ($\le 3$ rules), a compact 2-header report (`STATUS: COMPLETED | BLOCKED | FAILED` and `NOTES: <findings>`) is supported.
- If you are a subagent, `REPORT FORMAT` overrides workspace formatting rules. Return only the required headers (plus up to 10 compact bullet lines under `NOTES`) and omit any post-action retrospective. Subagent reports must be `<= 1600` characters total with inline summary text on the `NOTES:` header line (plus up to 10 `- ` bullets).

```markdown
TASK: <single-sentence objective>
ALLOWED_FILES: <comma-separated file paths or NONE>
REVIEW_LENS: <required for reviewer, e.g., security | edge-cases | simplicity | ux>
MODE: BEST_OF_N | COUNCIL
COHORT: <optional Best-of-N cohort id, e.g., solver-v1>
COUNCIL_ID: <optional Council id, e.g., auth-design>
COUNCIL_ROLE: <proposer | judge | synthesizer>
EVAL_CMD: <optional non-trivial deterministic eval command>
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
Return only these four headers (plus up to 10 `- ` bullet lines under NOTES if needed; <= 1600 chars total):
STATUS: COMPLETED | CHECKPOINT | NEEDS_REPLAN | BLOCKED | FAILED
FILES_CHANGED: <comma-separated paths or NONE>
VERIFICATION: <one sentence stating the command run and exit result>
NOTES: <summary (start with VERDICT: PASS|FAIL, APPROVE|REQUEST_CHANGES, or <winner> for verifier/reviewer/judge, or SCORE: <metric> for BEST_OF_N, plus up to 10 `- ` bullets), or inline error cause plus file://.log URI if BLOCKED/FAILED>
```

## 4. Reactive Coordination and Context Hygiene

- Invoke independent subagents in parallel in the same turn using your runtime's `invoke_subagent` schema (`Subagents: [{"TypeName", "Role", "Prompt", "Model", "Workspace"}, ...]` in Remote VMs, or flat arguments in local `agy` CLI). For large Best-of-N sweeps ($N > 5$), prefer Antigravity `run_workflow` (`Cwd`/`ScriptPath` must stay within workspace roots; `Description` supports optional `EVAL_CMD:` with anti-tampering and 1-shot `REPLAN:` recovery when tripped).
- **Disjoint Parallel Lanes (`implementer`)**: Concurrent `implementer` workers sharing `Workspace: "inherit"` (or reactivated via `send_message`) must have non-overlapping `ALLOWED_FILES:` paths (including ancestor/descendant directories such as `web/` vs `web/App.tsx`). Overlapping paths are permitted only when both lanes run in isolated `Workspace: "branch"` or `"share"` worktrees (including `MODE: BEST_OF_N`), or sequentially after the earlier lane finishes (`Stop`) or is killed.
- **Fresh-Thread Rule (`verifier`, `reviewer`, and `judge`)**: Never reuse a `verifier`, `reviewer`, or `judge` via `send_message` after any workspace code edit has occurred (`mutation_epoch` incremented by a root edit or `implementer` edit). Spawn a fresh `verifier`, `reviewer`, or `judge` via `invoke_subagent` for each verification/review/judging round (clarifying `send_message` follow-ups while `mutation_epoch` is unchanged remain allowed). Round-2 reviewers may reuse a lens once round-1 reviewers finish (`Stop`) or are killed.
- End your turn immediately after calling `invoke_subagent` or `send_message`. Do not poll with `manage_subagents`.
- Reuse an idle `flash` `implementer` via `send_message` when continuing work in the same module. Never send multi-step or code-editing tasks to `scout`, `verifier`, `reviewer`, or `judge` (`flash_lite` / strict read-only).
- **Staged Mid-Flight Checkpoints**: For multi-phase tasks, dispatch subagents with `STAGE: 1/N`. Subagents report `STATUS: CHECKPOINT` upon completing an intermediate stage, allowing orchestrator diff inspection and dynamic `ALLOWED_FILES:` allowlist expansion via `send_message` with `STAGE: 2/N` without resetting test receipts prematurely.
- Stop completed or losing `Workspace: "branch"` and `"share"` subagent worktrees with `manage_subagents` (`Action: "kill"`), and clean up isolated `"branch"`/`"share"` `swarm` worktrees with `swarm(action="kill")` before stopping (`Stop` blocks while uncleaned isolated subagent or `swarm` worktrees remain; `workspace: "inherit"` swarm workers do not block `Stop`).
- Keep context clean: do not read `.log` or `transcript.jsonl` files or $> 8$ workspace files directly in the orchestrator. Delegate log slicing to `scout`.

## 5. Empirical Test Receipts, Circuit Breaker, and Diff Audit

- **Empirical Test Receipts & `EVAL_CMD:`**: Every code change requires a passing test command (or declared non-trivial `EVAL_CMD:`) recorded by the guard hook after the last file edit. `MODE: BEST_OF_N` candidates isolate their `test_receipt` and `mutation_epoch` from the root brain — after merging the winning candidate's worktree, run verification in the root workspace before finishing. Likewise, `run_workflow`, `swarm`, `judge`, and `council` completions do **not** auto-clear a dirty `test_receipt` — run a passing test or declared `EVAL_CMD:` afterward.
- **Two-Turn Circuit Breaker, Cohort Isolation & 1-Shot Re-Plan**: Stop retrying after 2 consecutive command, subagent, `run_workflow`, or `swarm(action="add")` failures, write raw output to the scratch log, and report `STATUS: BLOCKED`. In `MODE: BEST_OF_N`, individual candidate failures are tracked per `COHORT:` and do not trip the root orchestrator's global circuit breaker unless $N \ge 2$ and all $N$ candidates fail (`0` succeeded). Tripped orchestrators have a 1-shot re-plan budget: adding a `REPLAN: <revised plan rationale>` header (or inside `run_workflow` `Description` / `swarm` `brief`) resets the circuit breaker once per turn before hard-locking (`swarm` `status`/`models`/`kill`/`title` remain allowed while tripped). Antigravity sandbox boundary errors (such as `.git/` read-only filesystem errors or sandbox error banners) are automatically exempted from incrementing the failure streak.
- **No Test Tampering**: Never delete, skip, or weaken existing test functions or assertions to make a test suite pass, and never use trivial `EVAL_CMD:` commands or shell fallbacks (`||`).

## 6. Post-Turn Evaluation

At the end of substantive orchestrator turns that mutate workspace files, run verification tests, or dispatch `implementer` subagents (pure read-only Q&A, code reviews, and explanations are exempt), include a concise **Post-Turn Evaluation** covering:
- **Potential Issues**: Residual edge cases, runtime constraints, or state caveats.
- **Capability Gaps**: Differences between protocol intent and deterministic hook enforcement.
- **Next Steps**: Concrete follow-up actions or verification steps.


# Philharmonic (`philharmonic_orchestra`)

A plugin for **Google Antigravity (`agy`)** that turns the main session into a lightweight coordinator and delegates research, edits, tests, and code reviews to focused subagents. Built with Python 3.8+ standard library only (zero external dependencies).

Instead of relying on prompt instructions alone, hook scripts enforce role boundaries, closed-world shell allowlists, isolated worktrees, blind judging, and test gates before work finishes.

---

## Quick Start

### 1. Try it in this repo

```bash
git clone https://github.com/leafy7382/philharmonic.git
cd philharmonic
agy
```

Type `/philharmonic` in chat to enable the workflow and hook checks for the current conversation. Everything stays inactive until you activate it.

### 2. Copy into another project

```bash
mkdir -p /path/to/your-project/.agents/plugins
cp -r .agents/plugins/philharmonic_orchestra /path/to/your-project/.agents/plugins/
```

### 3. Install globally (Local CLI)

```bash
mkdir -p ~/.gemini/config/plugins
ln -s "$(pwd)/.agents/plugins/philharmonic_orchestra" ~/.gemini/config/plugins/philharmonic_orchestra
```

### 4. Use on Antigravity Cloud VMs

- **Save the skill** at `.agents/plugins/philharmonic_orchestra/skills/philharmonic` with `save_skill`, or copy (do not symlink) the plugin folder into `~/.gemini/config/plugins/`:
  ```bash
  mkdir -p ~/.gemini/config/plugins
  cp -r .agents/plugins/philharmonic_orchestra ~/.gemini/config/plugins/
  ```
- **Remote VM note**: Both local CLI and Cloud VM schemas are supported. Because Best-of-N runs require `"branch"` or `"share"` worktrees, use local worktrees or `run_workflow` with `EVAL_CMD:` instead of `Workspace: "inherit"`.

---

## How to Use It (Workflow Patterns)

Once `/philharmonic` is active, the main agent picks a workflow based on the size and risk of the task:

| Situation | Pattern | How it runs |
| :--- | :--- | :--- |
| Tiny 1–2 file tweak | **Direct Fast-Path** | Main agent reads up to 400 lines across 8 files, makes up to 2 edits, and runs 1 test check (`exit_plan_mode` relaxes edit/command caps while carrying out an approved plan). |
| Normal feature or bug fix | **Builder + Verifier** | Write a short spec in `scratch/`, assign one `implementer` to specific files, then spawn a fresh `verifier` (`VERDICT: PASS \| FAIL`) after edits land. |
| Independent modules | **Parallel Lanes** | Run multiple `implementer`s at the same time with non-overlapping file lists on `inherit`, or give each its own `"branch"`/`"share"` worktree. |
| Hard optimization / low first-pass odds | **Best of N** | Spawn competing `implementer`s in `"branch"`/`"share"` worktrees with `MODE: BEST_OF_N`, shared `COHORT:`, and an objective `EVAL_CMD:` (each reports `SCORE:`). For > 5 candidates, use `run_workflow`. |
| Architecture / open-ended design | **Council of Models** | Collect independent drafts via `swarm` or `MODE: COUNCIL` (`research`/`council` proposers), let a blind `judge` pick a winner (`proposer-a`/`b`), write a spec, then build. |
| Large diff or risky refactor | **Review Team** | Spawn parallel `reviewer`s, each with a different `REVIEW_LENS:` (e.g., `security-bypass`, `concurrency-and-state`), returning `APPROVE` or `REQUEST_CHANGES`. |

### Typical day-to-day flow

1. Start with `/philharmonic` and ask for what you want to build or fix.
2. Let the main agent gather context (`scout`/`research`) and write a spec under `scratch/`.
3. Let scoped `implementer` workers make the edits and run tests, followed by a fresh `verifier` or parallel `reviewer`s.
4. Clean up any temporary `"branch"`/`"share"` worktrees, then let the main agent stage and commit directly with `git add <files> && git commit -m "..."`.

---

## Roles and Execution Boundaries

Custom agents (`scout`, `implementer`, `verifier`, `reviewer`, `judge`, `council`) are defined lazily as needed; `research` is built in.

| Role | Model Tier | File Access | Workspace | Job |
| :--- | :--- | :--- | :--- | :--- |
| `scout` | `flash_lite` | Read-only | `inherit` | Quick symbol lookups, file slices, and reading `.log`/transcript files |
| `research` | `flash` (`pro`/`inherit` in `MODE: COUNCIL`) | Scratch (`scratch/*.md`) | `inherit` | Multi-file investigation and Council proposals |
| `council` | `flash` / `pro` / `inherit` | Scratch (`scratch/*.md`) | `inherit` | Council draft writing outside `swarm` |
| `implementer` | `flash` (`pro` only in `MODE: BEST_OF_N`) | Scoped by `ALLOWED_FILES:` | `inherit`, `branch`, or `share` | Editing code and running tests |
| `verifier` | `flash` / `pro` / `inherit` | Strict read-only | `inherit` | Independent pass/fail verification after edits |
| `reviewer` | `flash` / `pro` / `inherit` | Strict read-only | `inherit` | Focused review with a unique `REVIEW_LENS:` |
| `judge` | `flash` / `pro` / `inherit` | Strict read-only | `inherit` | Blind voting across anonymized Council drafts |

### What the guard actually blocks and allows

Instead of playing catch-up with shell blacklists, shell and script checks are allowlist-based:

- **Main agent (`run_command`)**: Can run declarative read-only inspection verbs, `git` read commands, non-interactive `git add <paths>` / `git commit -m/-F` (explicit workspace files only, never `.agents/.state`), real test/lint commands, `orchestrator_guard.py doctor|status`, and `scratch/` file commands. Inline `python3 -c`, shell wrappers (`bash -c`, `sh -c`, `eval`), `tar`/`unzip`/`find`/`xargs`, `awk`/`sed`, heredocs/`$()`/backticks, and mutating shell tools (`rm`, `mv`, `cp`, `rsync`, `patch`, `dd`) are denied and must be delegated to an `implementer`.
- **Read-only & scratch subagents**: Locked to the same read-only shell allowlist (plus `scratch/*.md` writes for `research` and `council`). Raw `.log` and `transcript*.jsonl` reads are blocked for the main agent and routed through `scout`.
- **Implementers**: Can edit files and run simple file-targeting shell commands (`sed -i`, `touch`, `cp`, `rm`, `mv`, `chmod`, `dd`) only inside their single-line `ALLOWED_FILES:` list (exact paths, directory prefixes, or simple globs; brace expansion `{a,b}` and POSIX classes `[:...:]` fail closed). Editing plugin files requires listing the exact file paths in `ALLOWED_FILES:`.
- **Workflow boundaries & paths**: `run_workflow` enforces role/scope gates (`scout`/`verifier`/`reviewer`/`judge`/`research`/`council`, `ALLOWED_FILES:`, plan scope), `EVAL_CMD:` safety, and `Cwd`/`ScriptPath` boundaries against allowed roots and guard files rather than AST-emulating script bodies. Lightweight workspace root checks (`..` escapes or outside paths) pair with OS sandbox confinement to prompt or deny out-of-workspace access.
- **Safety nets**: State is stored with file locking and atomic writes in `<artifactDirectoryPath>/orchestrator_state.json`. Verification goes stale whenever code changes, 2 consecutive failures trip a circuit breaker (with a 1-shot `REPLAN:` reset; isolated per candidate cohort in Best-of-N), and uncleaned `"branch"`/`"share"` worktrees block `Stop`. Memory or recursion exhaustion in the guard itself fails closed (`deny`).

---

## Subagent Prompt & Report Contract

Pass headers one per line when calling `invoke_subagent` or `send_message`:

```markdown
TASK: <one-sentence goal>
ALLOWED_FILES: <comma-separated file paths on one line, or NONE>
REVIEW_LENS: <required for reviewer, e.g. security-bypass | edge-cases>
MODE: BEST_OF_N | COUNCIL
COHORT: <cohort id for Best-of-N>
COUNCIL_ID: <council id>
COUNCIL_ROLE: proposer | judge | synthesizer
EVAL_CMD: <non-trivial, non-mutating benchmark command>
STAGE: 1/2
SPEC_FILE: <artifactDirectoryPath>/scratch/spec_<task>.md
REPLAN_IF: <condition to stop early and replan>

RULES:
1. Read `SPEC_FILE` or `<file_path>`.
2. Keep edits inside `ALLOWED_FILES`.
3. Run `<test_command>` and report real output.
4. Write failure logs to `<artifactDirectoryPath>/scratch/<task_id>.log`.

REPORT FORMAT:
STATUS: COMPLETED | CHECKPOINT | NEEDS_REPLAN | BLOCKED | FAILED
FILES_CHANGED: <comma-separated paths or NONE>
VERIFICATION: <command run and result>
NOTES: <summary (VERDICT: ... or SCORE: ...; <= 10 `- ` bullets), or error + file://.log URI if BLOCKED/FAILED>
```

Quick rules that trip people up:
- **`ALLOWED_FILES:` must be on one comma-separated line** — multi-line YAML bullet lists are ignored by the header parser.
- **`SPEC_FILE:` needs three sections**: `SIGNATURES`, `PITFALLS`/`INVARIANTS`, and `VERIFICATION`.
- **Anonymous Councils**: Never mention model/provider names (`claude`, `gpt`, `gemini`, `flash`, `pro`) inside `judge` prompts, `swarm` briefs, or council roster names (`proposer-a`, `judge-1` are fine).
- **Failure reports**: `implementer` `BLOCKED`/`FAILED` reports require both a reason and a `file://.../*.log` link; read-only and scratch roles only need a short explanation. `scout` can reply with just `STATUS:` and `NOTES:`.

---

## CLI & Tests

```bash
# Inspect or manage guard state for the current workspace
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py activate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py deactivate
python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py reset

# Run the unit test suite (107 tests)
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
│           ├── hooks.json                      # Hook bindings (pre-invocation, pre-tool, post-tool, stop)
│           ├── plugin.json
│           ├── scripts/
│           │   ├── orchestrator_guard.py       # Hook entrypoint (fail-closed on memory/recursion errors)
│           │   └── guard/
│           │       ├── __init__.py             # Public guard package exports
│           │       ├── cli.py                  # status | activate | deactivate | reset | doctor
│           │       ├── constants.py            # Shared patterns and limits
│           │       ├── state_store.py          # Locked atomic JSON state persistence and pruning
│           │       ├── identity.py             # Session/subagent binding and lightweight workspace root checks
│           │       ├── path_guards.py          # Allowlist, scratch, simple glob, and protected-path checks
│           │       ├── ast_guard.py            # Test command and test-runner argument checks
│           │       ├── shell_tokens.py         # Shell tokenizer, env unwrapping, and EVAL_CMD checks
│           │       ├── shell_targets.py        # Shell write-target extraction for implementers
│           │       ├── root_allowlist.py       # Declarative closed-world allowlists for main agent & read-only roles
│           │       ├── subagent_policy.py      # Subagent, swarm, council, budget & workflow boundary policies
│           │       ├── verification.py         # Post-tool test receipts, stop gates & diff auditing
│           │       └── policies.py             # Pre-tool policy dispatcher and guard self-protection
│           └── skills/
│               └── philharmonic/
│                   └── SKILL.md                # Coordinator instructions and pattern flowchart
└── tests/
    └── test_orchestrator_guard.py              # Full guard unit test suite
```

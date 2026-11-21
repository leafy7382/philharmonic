# Phase 2 Plan: Best of N Support (Pattern 3)

## Context & Source Reference

* **Source:** Google Cloud (`@googlecloud`) X Article — *"5 agent orchestration patterns backed by research"* by YK Sugi (Senior Developer Relations Engineer, Google Cloud), published November 19, 2026 (`https://x.com/googlecloud/status/2123428849262248255`).
* **Motivation:** Phase 1 added Patterns 0, 1, 2, and 5 (`verifier`, `reviewer`, disjoint `implementer` lanes, `REVIEW_LENS:` deduplication, and `mutation_epoch` freshness). Phase 2 implements **Pattern 3 (Best of N)**: running the same difficult task across $N$ isolated attempts in parallel and selecting the winner via deterministic tests or scored evaluations (`EVAL_CMD:`).
* **Future Phase (Out of Scope for Phase 2):**
  * **Phase 3 — Council of Models (Pattern 4):** Opt-in cross-model proposal and blind-judging workflow via Antigravity `swarm`, feeding anonymized proposals into grounded `SPEC_FILE:` specs.

---

## Confirmed Policy Decisions

1. **`COHORT:` requirement:**
   * Keep `COHORT: <id>` **optional** when `MODE: BEST_OF_N` is set, auto-generating `best-of-n-<inv_key>` when omitted.
   * Setting `COHORT: <id>` alone also implies `mode = "best_of_n"`.
2. **`BEST_OF_N` candidate model ceiling:**
   * Default to `"flash"` when `Model` is omitted or `"inherit"`.
   * Allow explicit `Model: "pro"` for `MODE: BEST_OF_N` candidates (while standard non-`BEST_OF_N` `implementer`, `scout`, and `research` subagents keep existing caps and still deny `"pro"`).

---

## Goals & Design Principles

1. **Support `Workspace: "share"` alongside `"branch"`:** Track `active_worktree` across spawns, `manage_subagents(kill` / `kill_all)`, and `Stop` cleanup enforcement.
2. **Isolate competing candidates mechanically:** Require `"branch"` or `"share"` worktree isolation when `MODE: BEST_OF_N` or `COHORT:` is active, allow overlapping `ALLOWED_FILES:` across isolated candidates, scope failures per cohort instead of tripping the global 2-turn circuit breaker prematurely, and isolate candidate `test_receipt` / `mutation_epoch` changes from parent brain state.
3. **Accept custom objective evaluation commands (`EVAL_CMD:`):** Allow declared benchmark/eval scripts to satisfy empirical verification alongside standard unit test runners while blocking trivial/no-op commands (`true`, `:`, `echo`, etc.) and shell-chaining bypasses.

---

## Step-by-Step Implementation Plan

### Step 1 — Constants & Exports
**Files:** [constants.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/constants.py), [__init__.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/__init__.py)

1. Add `ISOLATED_WORKSPACES = {"branch", "share"}`.
2. Add single-line header regexes (`re.I`):
   * `MODE_HEADER_RE = re.compile(r"(?m)^[ \t]*MODE:[ \t]*([^\r\n]+)$", re.I)`
   * `COHORT_HEADER_RE = re.compile(r"(?m)^[ \t]*COHORT:[ \t]*([^\r\n]+)$", re.I)`
   * `EVAL_CMD_HEADER_RE = re.compile(r"(?m)^[ \t]*EVAL_CMD:[ \t]*([^\r\n]+)$", re.I)`
3. Add `TRIVIAL_EVAL_CMD_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:true|:|echo|printf|pwd|ls|cat|exit)\b", re.I)` to reject no-op eval commands.
4. Re-export all new constants and helper functions in [__init__.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/__init__.py) (`__all__`).

---

### Step 2 — State Schema & Pruning
**File:** [state_store.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/state_store.py)

1. Add `"cohorts": {}` and `"active_eval_cmd": None` to `_default_state()`, with backward-compatible `.setdefault` handling in `_load_state()`.
2. Prune `state["cohorts"]` in `_prune_state()` (cap number of cohorts at `MAX_STATE_INVOCATION_HISTORY` by `created_at` and bound per-cohort `succeeded` / `failed` lists to `MAX_STATE_LIST_ENTRIES`).
3. Standardize candidate metadata fields on `pending_spawns` and `subagents[cid]`: `"active_worktree"`, `"mode"`, `"cohort"`, `"eval_cmd"`.

---

### Step 3 — Subagent Metadata & Parent Cohort Sync
**File:** [identity.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/identity.py)

1. Preserve `active_worktree`, `mode`, `cohort`, and `eval_cmd` across `_normalize_subagent_meta`, `_sync_subagents_from_transcript`, and optional backward-compatible kwargs on `register_subagent(...)`.
2. Set `active_worktree = workspace.strip().lower() in ISOLATED_WORKSPACES` alongside `active_branch = workspace.strip().lower() == "branch"`.
3. Add `_record_cohort_outcome_in_parent_brain(payload, state, child_cid, cohort_id, outcome, error_summary="")` (reusing parent/sibling brain lookup + `state_transaction`) so async `best_of_n` subagents completing in `handle_stop` record their outcome in parent `cohorts[cohort_id]` and trip the parent circuit breaker only if `total_spawned >= 2` and 0 of $N$ candidates succeeded.

---

### Step 4 — Policy Enforcement
**File:** [policies.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/policies.py)

1. Add helpers:
   * `_parse_mode_header(prompt_text)` (normalizes `best_of_n` / `best-of-n` / `best of n` to `"best_of_n"`).
   * `_parse_cohort_header(prompt_text)` (normalizes whitespace/underscores to `-`, lowercased).
   * `_parse_eval_cmd_header(prompt_text)` and `_is_trivial_eval_command(eval_cmd)` (rejects empty, `||` trivial fallback, or all-trivial sub-commands).
   * `_is_declared_eval_command(cmd, state, payload)` (rejects `$(`, `` ` ``, and `||`; matches declared subagent, cohort, or root `active_eval_cmd` at exact or space-delimited argument boundary).
2. In `_validate_and_prepare_subagent_entry` & `_handle_invoke_subagent`:
   * If `MODE: BEST_OF_N` or `COHORT:` is present: set `mode = "best_of_n"` and `cohort_id = parsed_cohort or f"best-of-n-{inv_key}"`.
   * **Require worktree isolation:** deny if `workspace_mode.lower() not in ISOLATED_WORKSPACES`.
   * **Model policy:** allow explicit `Model: "pro"` and `"flash"` untouched (omitted / `"inherit"` defaults to `"flash"` via `DEFAULT_MODEL_BY_TYPE`). Standard non-`BEST_OF_N` workers still deny `"pro"`.
   * **Eval command check:** if `EVAL_CMD:` is present, deny if `_is_trivial_eval_command` returns `True`.
   * **Atomic cohort registration:** register/update `state["cohorts"][cohort_id]` (`total_spawned`, `succeeded`, `failed`, `eval_cmd`, `created_at`) and `state["active_eval_cmd"]` only after every entry in a Remote VM `Subagents` batch passes validation.
3. In `_handle_manage_subagents`: clear both `active_branch = False` and `active_worktree = False` on `kill` and `kill_all`.
4. In `exit_plan_mode` / `enter_plan_mode` and `_check_and_record_direct_budget`: parse optional `EVAL_CMD:` from plan text (deny if trivial), clear `active_eval_cmd` on plan exit/reset, and accept `_is_declared_eval_command` alongside `_is_test_command` for root post-edit verification.
5. In `handle_pre_tool` code mutation receipt tracking: skip `_propagate_receipt_to_parent_brain` when the caller has `mode == "best_of_n"`.

---

### Step 5 — Cohort Circuit Breaker, Receipt Isolation & Worktree Cleanup
**File:** [verification.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/verification.py)

1. In `handle_post_tool`:
   * Capture cohort metadata (`mode`, `cohort_id`) before popping failed `pending_spawns`.
   * **Per-candidate cohort breaker:** record candidate failures in `state["cohorts"][cohort_id]["failed"]` without incrementing global `cb["failure_streak"]`; trip global `cb` only when `total_spawned >= 2`, `len(failed) >= total_spawned`, and `len(succeeded) == 0`. Record synchronous successes in `cohort["succeeded"]` and reset global streak.
   * **Candidate receipt isolation:** skip `_propagate_receipt_to_parent_brain` for `mode == "best_of_n"` callers, and never allow synchronous `best_of_n` candidate completions to clear parent's dirty `test_receipt`.
   * **Declared `EVAL_CMD:` receipt:** treat passing `is_eval_cmd = tool_name == "run_command" and _is_declared_eval_command(cmd, state, payload)` alongside `is_test_cmd`.
2. In `handle_stop`:
   * Check `active_worktree` or `active_branch` on both `subagents` and `pending_spawns` so uncleaned `"branch"` and `"share"` worktrees block premature completion.
   * Record async `best_of_n` candidate completion outcome in parent brain via `_record_cohort_outcome_in_parent_brain`.

---

### Step 6 — Status CLI Output
**File:** [cli.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/cli.py)

* Surface `cohorts` and `activeEvalCmd` in `_cli_status` while keeping all existing keys backward-compatible.

---

### Step 7 — Protocol & Documentation
**Files:** [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md), [README.md](../README.md)

1. Update [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md): add **Difficult + deterministic tests/evals -> Best of N** to the Pattern Selection Flowchart, document `MODE: BEST_OF_N`, optional `COHORT:`, optional `EVAL_CMD:`, `SCORE:` reporting under `NOTES:`, explicit `"pro"` allowance for `BEST_OF_N` candidates, `"branch"`/`"share"` worktree cleanup, and `run_workflow` guidance for larger $N > 5$ sweeps. Keep `name: philharmonic` and `## 1. Lazy Subagent Registration` .. `## 6. Post-Turn Evaluation`.
2. Sync [README.md](../README.md) while keeping total size strictly `< 15,800` bytes.

---

### Step 8 — Unit Tests
**File:** [test_orchestrator_guard.py](../tests/test_orchestrator_guard.py)

Add `test_phase2_*` test suites without altering or weakening existing tests:
1. `Workspace: "share"` worktree tracking, `Stop` cleanup enforcement, and `manage_subagents(kill` / `kill_all)` clearing.
2. `MODE: BEST_OF_N` / `COHORT:` validation: auto-generated `best-of-n-<inv_key>` default, explicit `COHORT:` implies `best_of_n`, `"inherit"` denied, `"branch"`/`"share"` allowed with overlapping `ALLOWED_FILES:`, explicit `Model: "pro"` allowed (omitted/`"inherit"` -> `"flash"`, non-`BEST_OF_N` still denies `"pro"`), and atomic batch rollback.
3. Per-candidate cohort circuit breaker (2/5 failures do not trip global breaker; $N/N$ all-failed trips global breaker; local candidate 2-turn breaker still trips).
4. Candidate receipt isolation (failing/passing `best_of_n` candidates do not dirty, bump `mutation_epoch`, or clear parent `test_receipt`).
5. `EVAL_CMD:` verification & anti-tampering (custom eval script clears `test_receipt` on exit 0; trivial commands `true`, `echo pass`, `:` denied).
6. `_cli_status` surfacing `cohorts` and `activeEvalCmd`.

---

## Verification Plan

1. `python3 -m unittest discover -s tests` -> all 80 existing + new Phase 2 tests pass (0 failures, 0 errors).
2. `python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor` -> `"healthy": true`, `"issues": []`.
3. `wc -c README.md` -> `< 15,800` bytes.

---

## Pitfalls & Re-Plan Triggers

* Preserve existing `Stop` error phrasing (`"Uncleaned branched subagent worktrees remain"`) or ensure existing `active_branch` tests in [test_orchestrator_guard.py](../tests/test_orchestrator_guard.py) continue passing verbatim.
* In `verification.py`, snapshot `pending_spawns` metadata **before** popping failed entries on `invoke_subagent` error so cohort failure accounting has the right `cohort_id` and `mode`.
* Re-plan if any existing test fails due to an unexpected coupling with `active_worktree`, `cohorts`, or `EVAL_CMD:`.

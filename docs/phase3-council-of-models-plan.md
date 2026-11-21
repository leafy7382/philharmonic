# Phase 3 Plan: Council of Models & Cross-Model Tooling (Pattern 4)

## Context & Source Reference

* **Source:** Google Cloud (`@googlecloud`) X Article — *"5 agent orchestration patterns backed by research"* by YK Sugi (Senior Developer Relations Engineer, Google Cloud), published November 19, 2026 (`https://x.com/googlecloud/status/2123428849262248255`).
* **Motivation:** Phase 1 added Patterns 0, 1, 2, and 5 (`verifier`, `reviewer`, disjoint `implementer` lanes, `REVIEW_LENS:` deduplication, and `mutation_epoch` freshness), and Phase 2 added Pattern 3 (**Best of N** with `"share"` worktrees, `COHORT:`, `EVAL_CMD:`, and `run_workflow` guards). Phase 3 completes the five-pattern suite with **Pattern 4 (Council of Models)**: asking multiple diverse models for independent proposals, having fresh judges vote blind (with model identities hidden), and synthesizing the strongest ideas into a grounded `SPEC_FILE:` for `implementer` execution.

---

## Confirmed Policy Decisions

1. **`swarm` cleanup gate in `Stop`:**
   * Block `Stop` **only** on uncleaned **isolated** (`branch`/`share`) swarm worktrees (`status == "active"` and `active_worktree == True`), consistent with `invoke_subagent`. Swarm workers running in `workspace: "inherit"` do not block `Stop`.
2. **Non-`swarm` fallback proposer model ceiling (`invoke_subagent` with `MODE: COUNCIL`):**
   * Allow explicit `Model: "pro"` / `"inherit"` (and `"flash"`, defaulting to `"flash"` when omitted) on `MODE: COUNCIL` proposers (`research` / `council`), without auto-downgrading to `"flash_lite"`.
   * Keep `"council"` proposers restricted to scratch-only `.md` writes (`<artifactDirectoryPath>/scratch/*.md`, workspace code mutations denied, dirty `test_receipt` never cleared). Standard non-council `research` keeps its existing caps and `<= 3` rule downgrade.

---

## Goals & Design Principles

1. **Support opt-in cross-family model diversity via `swarm` and clean Remote VM fallback:** Document and guard `swarm` (`models`, `add`, `kill`, `model`, `status`, `title`) for Council of Models workflows while supporting `invoke_subagent` (`MODE: COUNCIL`) in environments without `swarm`.
2. **Mechanically enforce blind judging:** Reject model-identity attribution leaks inside `judge` prompts/briefs (`BLIND_JUDGE_LEAK_PATTERNS`) and non-anonymous `swarm` council roster names (`COUNCIL_NAME_LEAK_RE`), without false positives on unrelated technical prose (e.g., `"carbon accounting"`, `"Gemini API client"`).
3. **Preserve strict read-only verification and circuit-breaker hygiene:** Treat `judge` as a strict read-only high-reasoning subagent (`flash` default, explicit `"pro"` / `"inherit"` allowed, never `"flash_lite"`, 4-header report required, `mutation_epoch` freshness gated, never clears dirty `test_receipt`), track `swarm` failures/rollbacks, and gate isolated `swarm` worktree cleanup on `Stop`.

---

## Step-by-Step Implementation Plan

### Step 1 — Constants & Exports
**Files:** [constants.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/constants.py), [__init__.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/__init__.py)

1. Update role sets and defaults:
   * `STRICT_READONLY_SUBAGENT_TYPES = {"scout", "verifier", "reviewer", "judge"}`
   * `HIGH_REASONING_READONLY_TYPES = {"verifier", "reviewer", "judge"}`
   * `SCRATCH_ONLY_SUBAGENT_TYPES = {"research", "researcher", "council"}`
   * `READONLY_NON_VERIFICATION_TYPES = {"scout", "research", "researcher", "council", "verifier", "reviewer", "judge"}`
   * `ALLOWED_COUNCIL_PROPOSER_MODELS = {"flash", "pro", "inherit"}`
   * Add `"judge": "flash"` and `"council": "flash"` to `DEFAULT_MODEL_BY_TYPE`.
2. Add single-line header regexes (`[ \t]*([^\r\n]+)$`, `re.I`): `COUNCIL_ID_HEADER_RE` and `COUNCIL_ROLE_HEADER_RE`.
3. Add high-precision blindness regexes (`re.I`):
   * `COUNCIL_NAME_LEAK_RE` matching model family/variant tokens (`gemini`, `claude`, `opus`, `sonnet`, `haiku`, `gpt`, `openai`, `anthropic`, `deepseek`, `llama`, `qwen`, `mistral`, `carbon`, `neon`, `argon`, `hydrogen`, `fable`) in `swarm` council worker roster names.
   * `BLIND_JUDGE_LEAK_PATTERNS` catching explicit model-to-proposal attribution (e.g., `"Proposal A (Claude Opus 5.5)"`, `"Draft B by Gemini 4 Carbon"`, `"Claude Opus authored Proposal C"`) inside judge prompts/briefs.
4. Re-export all new constants and helper functions alphabetically in [__init__.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/__init__.py).

---

### Step 2 — State Schema & Pruning
**File:** [state_store.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/state_store.py)

1. Add `"councils": {}` and `"swarm_workers": {}` to `_default_state()`, preserved via `.setdefault` in `_load_state()`.
2. Bound `state["councils"]` (up to `MAX_STATE_INVOCATION_HISTORY` by `created_at`; cap `proposers`, `judges`, `models` lists at `MAX_STATE_LIST_ENTRIES`) and `state["swarm_workers"]` (drop oldest `"killed"` entries past `MAX_STATE_LIST_ENTRIES`) in `_prune_state()`.

---

### Step 3 — Subagent Metadata & Role Helpers
**File:** [identity.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/identity.py)

1. Preserve `council_id` and `council_role` across `_normalize_subagent_meta`, `_sync_subagents_from_transcript`, and optional backward-compatible kwargs on `register_subagent(...)`.
2. Add `_is_scratch_only_subagent(payload, state) -> bool` checking `_get_subagent_type(payload, state) in SCRATCH_ONLY_SUBAGENT_TYPES` (while keeping `_is_research_subagent` backward-compatible).

---

### Step 4 — Policy Enforcement & `swarm` Hooks
**File:** [policies.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/policies.py)

1. Add helpers: `_parse_council_id_header`, `_parse_council_role_header` (`proposer` | `judge` | `synthesizer`), `_detect_blind_judge_leak`, `_detect_council_name_leak`, and `_expand_swarm_worker_names` (handling `name`, `n > 1`, and `tasks`).
2. `_handle_define_subagent`: strip `EnableWriteTools`, `EnableSubagentTools`, and `EnableMCPTools` for `"judge"` (`STRICT_READONLY_SUBAGENT_TYPES`), and strip `EnableSubagentTools` + `EnableMCPTools` for `"council"`.
3. `_validate_and_prepare_subagent_entry` & `_handle_invoke_subagent`:
   * Extend `_parse_mode_header` to normalize `"council"` / `"council_of_models"` -> `"council"`; infer `mode = "council"` if `COUNCIL_ID:` or `COUNCIL_ROLE:` is set without `MODE:`; default `council_id = f"council-{inv_key}"` when omitted.
   * Allow explicit `"pro"` / `"inherit"` / `"flash"` on `"judge"` and on `MODE: COUNCIL` proposers (`research` / `council`) without `"flash_lite"` downgrade (standard non-council `research` unchanged).
   * Deny judge prompts (`norm_type == "judge"` or `council_role == "judge"`) when `_detect_blind_judge_leak` finds model attribution.
   * Atomically record council membership in `state["councils"][council_id]` after batch validation.
4. `_handle_swarm(payload, args, state)` in `handle_pre_tool`:
   * Gate `swarm(action="add")` behind the tripped circuit breaker unless `REPLAN:` appears in `brief` (1-shot budget), while always allowing `status`, `models`, `kill`, and `title`.
   * Validate `EVAL_CMD:` in `brief` against `_is_trivial_eval_command`; in council mode enforce anonymous roster names (`_detect_council_name_leak`) and blind judge briefs (`_detect_blind_judge_leak`).
   * Track expanded worker names in `state["swarm_workers"]` with `active_worktree = workspace in ISOLATED_WORKSPACES`; clear on `action="kill"` (`status="killed"`, `active_worktree=False`); update `"model"` on `action="model"`.
5. Use `_is_scratch_only_subagent` for `MUTATION_TOOLS` in `handle_pre_tool` so `"council"` proposers share scratch-only `.md` write isolation.

---

### Step 5 — Post-Tool Rollback & `Stop` Worktree Gate
**File:** [verification.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/verification.py)

1. In `handle_post_tool`: include `"swarm"` in circuit-breaker tracking — roll back failed `action="add"` `swarm_workers` / `councils` entries and increment breaker on non-sandbox errors; reset breaker on success without auto-clearing dirty `test_receipt`.
2. In `handle_stop`: include uncleaned isolated `swarm_workers` (`status == "active"` and `active_worktree`) in the uncleaned worktree gate (reminding to merge/discard and call `swarm(action="kill")`), leaving `workspace: "inherit"` swarm workers unblocked. Keep `_is_scout_subagent` strictly `"scout"`-only.

---

### Step 6 — Status CLI Output
**File:** [cli.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/cli.py)

* Surface `"councils"` and `"swarmWorkers"` in `_cli_status` while preserving existing keys.

---

### Step 7 — Protocol & Docs
**Files:** [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md), [README.md](../README.md)

1. Update [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md): complete the Pattern Selection Flowchart with **Difficult + open-ended / subjective -> Council of Models** (`Propose -> Blind Vote -> Synthesize -> Grounded SPEC_FILE:`), authorize opt-in `swarm` usage and Remote VM `invoke_subagent` (`MODE: COUNCIL`) fallback, and add lazy `define_subagent` recipe for `judge` (plus note `council`). Preserve `name: philharmonic` and headings `## 1.`..`## 6.`.
2. Sync [README.md](../README.md) while keeping total size strictly `< 15,800` bytes.

---

### Step 8 — Unit Tests
**File:** [test_orchestrator_guard.py](../tests/test_orchestrator_guard.py)

Add `test_phase3_*` suites without altering existing tests:
1. `judge` `define_subagent` stripping (`PascalCase` + `snake_case`), model policy (`flash` default, `pro`/`inherit` allowed, `flash_lite` rewritten), strict read-only tools/shell, 4-header report, dirty receipt preservation, and `mutation_epoch` freshness gate.
2. Blind judging anti-leakage (denies attributed model names, allows anonymized `Proposal A/B/C` and benign technical prose).
3. `swarm` council hooks (anonymous roster name enforcement, `n`/`tasks` expansion, `kill` cleanup, `Stop` gate on isolated `branch`/`share` worktrees only, `PostToolUse` error rollback + circuit breaker, tripped breaker `REPLAN:` gate).
4. Non-`swarm` `MODE: COUNCIL` proposer fallback (`research`/`council` allowing `"pro"`/`"inherit"` without `flash_lite` downgrade, scratch-only `.md` writes, non-clearing of dirty `test_receipt`) + `_cli_status` surfacing.

---

## Verification Plan

1. `python3 -m unittest discover -s tests` -> all 90 existing + new Phase 3 tests pass (0 failures, 0 errors).
2. `python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor` -> `"healthy": true`, `"issues": []`.
3. `wc -c README.md` -> strictly `< 15,800` bytes.

---

## Pitfalls & Re-Plan Triggers

* Tune `BLIND_JUDGE_LEAK_PATTERNS` carefully so phrases like `"Proposal A: use carbon theme"` or `"Gemini API"` do **not** false-positive, while explicit attribution (`"Proposal A (Claude Opus 5.5)"`, `"Draft B by Gemini 4 Carbon"`, `"Claude Opus authored Proposal C"`) is reliably caught.
* Keep `README.md` strictly `< 15,800` bytes by tightening existing prose before adding Phase 3 summaries (`README.md` is currently `14,430` bytes).
* Re-plan if any existing test fails unexpectedly due to `SCRATCH_ONLY_SUBAGENT_TYPES` or `swarm` `Stop` checks.

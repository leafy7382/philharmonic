# Phase 1 Plan: Evolving Philharmonic with Research-Backed Agent Orchestration Patterns

## Context & Source Reference

* **Source:** Google Cloud (`@googlecloud`) X Article — *"5 agent orchestration patterns backed by research"* by YK Sugi (Senior Developer Relations Engineer, Google Cloud), published November 19, 2026 (`https://x.com/googlecloud/status/2123428849262248255`).
* **Motivation:** Philharmonic currently enforces cost- and hygiene-driven controls (`flash` / `flash_lite` delegation, read/edit/command budgets, `ALLOWED_FILES:` write isolation, grounded `SPEC_FILE:` contracts, empirical test receipts, test anti-tampering, and a 2-turn circuit breaker). Phase 1 integrates the article's quality- and reliability-driven patterns that map directly onto Philharmonic's lifecycle hooks (`PreInvocation`, `PreToolUse`, `PostToolUse`, `Stop`):
  * **Pattern 0 (Decision Flowchart Gate):** Route simple or tightly coupled sequential tasks to a single thread; use multi-agent topologies only when task structure justifies them.
  * **Pattern 1 (Builder + Verifier):** Separate building from verification in a fresh, read-only `verifier` thread to eliminate self-bias.
  * **Pattern 2 (Split Work into Lanes):** Mechanically prevent overlapping `ALLOWED_FILES:` across concurrent `implementer` workers sharing `Workspace: "inherit"`.
  * **Pattern 5 (Team of Reviewers):** Add a dedicated read-only `reviewer` role with mandatory, deduplicated `REVIEW_LENS:` assignments across concurrent reviewers and fresh re-review rounds after fixes.
* **Future Phases (Out of Scope for Phase 1):**
  * **Phase 2 — Best of N (Pattern 3):** Track `Workspace: "share"` alongside `"branch"`, add `COHORT:` / `MODE: BEST_OF_N` per-candidate failure streaks, allow overlapping `ALLOWED_FILES:` across isolated worktrees, and accept declared `EVAL_CMD:` runs.
  * **Phase 3 — Council of Models (Pattern 4):** Opt-in cross-model proposal/blind-judging workflow via Antigravity `swarm` (feeding anonymized proposals into grounded `SPEC_FILE:` specs), with lightweight `swarm` hygiene hooks.

---

## Confirmed Policy Decisions

1. **Model ceiling for `verifier` and `reviewer`:**
   * Default to `"flash"` when `Model` is omitted (or if `"flash_lite"` or an unrecognized value is passed).
   * Allow explicit `Model: "pro"` or `Model: "inherit"` for high-stakes verification and review.
   * Never allow or auto-downgrade `verifier` or `reviewer` to `"flash_lite"`.
2. **Large-change `Stop` gate:**
   * Keep `verifier` and `reviewer` invocation **protocol-guided** via the Decision Flowchart in [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md).
   * Do **not** add a hard `Stop` block when `verifier` or `reviewer` is omitted (`Stop` remains gated on empirical test receipts, test anti-tampering, active `"branch"` cleanup, and subagent report validation).

---

## Goals & Design Principles

1. **Mechanically enforce pattern invariants once invoked:**
   * Fresh, read-only verification (`verifier`) that never downgrades to `flash_lite`, never mutates code, never clears dirty empirical test receipts, and cannot be reused via `send_message` after new code edits.
   * Disjoint parallel lanes (`implementer`) with deterministic `ALLOWED_FILES:` overlap detection across concurrent workers sharing `Workspace: "inherit"`.
   * Role-specialized parallel review (`reviewer`) with mandatory, deduplicated `REVIEW_LENS:` enforcement across concurrent reviewers and fresh-round enforcement after fixes.
2. **Preserve zero-regression compatibility:** Keep existing 4-header report rules (and 2-header `scout` shorthand), Remote VM dual-schema support (`Subagents: [...]` and `snake_case` `define_subagent`), zero external dependencies (Python 3 stdlib only), and existing [test_orchestrator_guard.py](../tests/test_orchestrator_guard.py) expectations intact.
3. **Avoid false-positive friction:** Keep routine/small edits lightweight and track subagent lifecycle states cleanly so sequential follow-up turns and round-2 fresh reviewers are not falsely blocked.

---

## Step-by-Step Implementation Plan

### Step 1 — Role Sets, Model Policy, and Header Regex
**Files:** [constants.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/constants.py), [__init__.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/__init__.py)

1. Keep `ALLOWED_SUBAGENT_MODELS = {"flash", "flash_lite"}` exported for backward compatibility and standard workers (`scout`, `research`, `researcher`, `implementer`).
2. Add explicit role sets and model policy constants:
   * `STRICT_READONLY_SUBAGENT_TYPES = {"scout", "verifier", "reviewer"}`
   * `HIGH_REASONING_READONLY_TYPES = {"verifier", "reviewer"}`
   * `READONLY_NON_VERIFICATION_TYPES = {"scout", "research", "researcher", "verifier", "reviewer"}`
   * `ALLOWED_VERIFIER_REVIEWER_MODELS = {"flash", "pro", "inherit"}`
3. Update `DEFAULT_MODEL_BY_TYPE` with `"verifier": "flash"` and `"reviewer": "flash"`.
4. Add `REVIEW_LENS_HEADER_RE = re.compile(r"(?m)^[ \t]*REVIEW_LENS:[ \t]*([^\r\n]+)$", re.I)`.
5. Re-export any newly shared constants in [__init__.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/__init__.py) if it exposes `constants.py` symbols.

---

### Step 2 — State Schema Extensions
**File:** [state_store.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/state_store.py)

1. Add `"mutation_epoch": 0` to `test_receipt` in `_default_state()` (always read via `.get("mutation_epoch", 0)` for backward compatibility with existing `orchestrator_state.json` files).
2. Standardize three new fields on `pending_spawns` and `subagents[cid]` entries:
   * `"status": "active" | "completed" | "killed"` (default `"active"` on spawn)
   * `"spawned_mutation_epoch": int` (snapshot of `test_receipt["mutation_epoch"]` at dispatch)
   * `"review_lens": Optional[str]` (normalized lens string for `reviewer`)

---

### Step 3 — Subagent Metadata & Parent Completion Sync
**File:** [identity.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/identity.py)

1. In `_sync_subagents_from_transcript` (L277–354) and subagent entry construction (L555–578), preserve `status` (defaulting to `"active"`), `spawned_mutation_epoch`, and `review_lens` when migrating `pending_spawns` entries into `state["subagents"][cid]` and caching `self_subagent_meta`.
2. Add `_mark_subagent_status_in_parent_brain(payload, state, child_cid, status)` using the existing parent/sibling brain state lookup and `flock` write pattern, allowing a subagent's `handle_stop` to mark itself `"completed"` in the parent orchestrator's `orchestrator_state.json`.

---

### Step 4 — Policy Enforcement
**File:** [policies.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/policies.py)

1. **New helpers:**
   * `_parse_review_lens_header(prompt)` & `_normalize_review_lens(lens)`: extracts `REVIEW_LENS:`, strips whitespace, lowercases, and normalizes internal spaces/hyphens/underscores so `"edge cases"`, `"edge-cases"`, and `"edge_cases"` collide cleanly.
   * `_normalize_allowlist_entry(entry)` & `_find_allowed_files_overlap(files_a, files_b)`: resolves relative to `PROJECT_ROOT`, strips trailing `/`, `/*`, `/**`, and returns the conflicting pair if two entries are equal or one is an ancestor directory of the other (e.g., `web/` vs `web/App.tsx`).
   * `_iter_active_subagents(state, batch_entries)`: collects active workers from:
     * (a) earlier entries in the same Remote VM `Subagents` batch,
     * (b) live `pending_spawns` (local CLI parallel calls in the same turn), and
     * (c) `state["subagents"]` entries with `status == "active"`.
2. **`_handle_define_subagent` (L167–189):**
   * Force `EnableSubagentTools` / `enable_subagent_tools` = `False` for all types (existing).
   * Force `EnableWriteTools` / `enable_write_tools` = `False` for all `STRICT_READONLY_SUBAGENT_TYPES` (`scout`, `verifier`, `reviewer`).
   * Force `EnableMCPTools` / `enable_mcp_tools` = `False` for `scout`, `implementer`, `verifier`, and `reviewer`.
3. **`_is_simple_readonly_task` (L192–201):**
   * Return `False` immediately for `HIGH_REASONING_READONLY_TYPES` (`verifier`, `reviewer`) so short prompts ($\le 3$ rules) are never downgraded to `flash_lite`.
4. **`_validate_and_prepare_subagent_entry` (L244–321) & `_handle_invoke_subagent` (L324–403):**
   * **Per-role model policy:**
     * If `type_name in HIGH_REASONING_READONLY_TYPES` (`verifier`, `reviewer`):
       * Allow explicit `"flash"`, `"pro"`, or `"inherit"` without rewriting.
       * If omitted/empty, `"flash_lite"`, or unrecognized, rewrite to `"flash"`.
     * All other types (`scout`, `research`, `researcher`, `implementer`, custom): unchanged (`"pro"` denied; `"inherit"` / omitted rewritten via `DEFAULT_MODEL_BY_TYPE`; simple read-only `scout`/`research` downgraded to `flash_lite`).
   * **Mandatory & distinct `REVIEW_LENS:` (Pattern 5):**
     * If `type_name == "reviewer"`, require non-empty `REVIEW_LENS:`; deny if missing.
     * Check normalized `review_lens` against all active `reviewer` entries from `_iter_active_subagents`; deny if another active `reviewer` already holds that lens.
   * **Disjoint lane collision check (Pattern 2):**
     * If `type_name == "implementer"` and `allowed_files` is non-empty, compare against all active `implementer` entries with non-empty `allowed_files`.
     * Deny if `_find_allowed_files_overlap` finds a conflict **and** either lane runs in `Workspace: "inherit"` (overlapping `ALLOWED_FILES:` remains allowed when both lanes are isolated in `"branch"` / `"share"` worktrees). Omitted `ALLOWED_FILES:` remains backward-compatible.
   * Attach `status: "active"`, `spawned_mutation_epoch: state["test_receipt"].get("mutation_epoch", 0)`, and `review_lens` to every prepared pending spawn entry.
5. **`_handle_send_message` (L406–501):**
   * **Freshness gate (Patterns 1 & 5):** If the target subagent has `type in HIGH_REASONING_READONLY_TYPES` (`verifier`, `reviewer`) and `state["test_receipt"].get("mutation_epoch", 0) > meta.get("spawned_mutation_epoch", 0)`, deny and direct the orchestrator to spawn a fresh `verifier` / `reviewer` via `invoke_subagent`. (Clarifying follow-ups while `mutation_epoch` is unchanged remain allowed.)
   * **Implementer reactivation & expansion:** If the target is an `implementer`, check its effective `allowed_files` (existing or newly supplied `ALLOWED_FILES:`) against all *other* active `implementer`s under the same `inherit` overlap rule before setting `status = "active"`.
6. **`_handle_manage_subagents` (L504–545):**
   * On `kill_all`, set `status = "killed"` on all `state["subagents"]` entries (in addition to clearing `pending_spawns` and `active_branch`).
   * On `kill`, set `status = "killed"` on targeted `ConversationIds`.
7. **Strict read-only enforcement (L1067–1074, L1116–1137):**
   * Generalize `scout`'s `MUTATION_TOOLS` block and mutating shell command block (`>`, `rm`, `mv`, `cp`, `tee`, `touch`, `truncate`, `dd`, `sed -i`, `perl -i`) to all `STRICT_READONLY_SUBAGENT_TYPES` (`scout`, `verifier`, `reviewer`). Keep `read_file` and `git diff` / `status` / `log` allowed.
8. **Bump `mutation_epoch` (L1208–1228):**
   * Whenever a workspace code file is mutated via `MUTATION_TOOLS`, increment `state["test_receipt"]["mutation_epoch"] = state["test_receipt"].get("mutation_epoch", 0) + 1`.

---

### Step 5 — Receipts, Completion Sync, and Report Validation
**File:** [verification.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/verification.py)

1. **`_propagate_receipt_to_parent_brain` (L255–302):** When propagating a dirty mutation (`dirty=True`) from a child `implementer` to the parent brain, also increment the parent brain's `test_receipt["mutation_epoch"]` so root freshness checks catch subagent edits as well as root edits.
2. **`handle_post_tool` (L305–446):**
   * Replace the hardcoded `{"scout", "research", "researcher"}` check (L406–432) with `READONLY_NON_VERIFICATION_TYPES` so synchronous `verifier` / `reviewer` returns **never** clear a dirty `test_receipt`.
   * On synchronous (non-async-ack) `invoke_subagent` completion, set `state["subagents"][cid]["status"] = "completed"`.
3. **`handle_stop` (L590–712):**
   * Keep `_is_scout_subagent` restricted to `"scout"` so `verifier` and `reviewer` require the full 4-header report format.
   * When a subagent passes `Stop` validation cleanly, call `_mark_subagent_status_in_parent_brain(..., "completed")` so finished lanes and round-1 review lenses are released cleanly.

---

### Step 6 — Status CLI Output
**File:** [cli.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/cli.py)

* Expose `status` and `reviewLens` on subagent summaries and `mutationEpoch` inside `testReceipt` in `_cli_status` (L30–45). Keep `_cli_doctor` checks intact.

---

### Step 7 — Protocol & Documentation
**Files:** [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md), [README.md](../README.md)

1. Update [SKILL.md](../.agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md):
   * Add a **Pattern Selection Flowchart** section mapping task structure to topology:
     * Simple / tightly coupled sequential steps $\rightarrow$ single thread / direct fast-path
     * Medium changes $\rightarrow$ Builder + fresh read-only `verifier`
     * Independent parallel parts $\rightarrow$ parallel `implementer` lanes with disjoint `ALLOWED_FILES:`
     * Large changes $\rightarrow$ parallel read-only `reviewer` team with distinct `REVIEW_LENS:` + fresh re-review rounds until all approve
   * Add lazy `define_subagent` templates for `verifier` and `reviewer` (`PascalCase` + `snake_case`).
   * Document model policy for `verifier` / `reviewer` (defaults to `"flash"`, explicit `"pro"` or `"inherit"` allowed for high-stakes verification/review, never `"flash_lite"`).
   * Document the required `REVIEW_LENS:` prompt header for `reviewer`, the `VERDICT: PASS | FAIL` / `VERDICT: APPROVE | REQUEST_CHANGES` convention inside the 4-header report format, fresh-thread rules, and disjoint lane rules.
2. Update [README.md](../README.md):
   * Sync the Subagent Architecture table (`scout`, `research`, `implementer`, `verifier`, `reviewer`), Prompt Contract (`REVIEW_LENS:`), and Lifecycle Hooks overview.

---

### Step 8 — Unit Tests
**File:** [test_orchestrator_guard.py](../tests/test_orchestrator_guard.py)

Add focused unit tests in `TestOrchestratorGuard`:
1. `define_subagent` strips write/MCP/subagent tools for `verifier` and `reviewer` (`PascalCase` + `snake_case`).
2. Model policy:
   * `verifier` / `reviewer`: omitted $\rightarrow$ `"flash"`; explicit `"flash"` $\rightarrow$ `"flash"`; explicit `"pro"` $\rightarrow$ `"pro"`; explicit `"inherit"` $\rightarrow$ `"inherit"`; `"flash_lite"` $\rightarrow$ rewritten to `"flash"`; $\le 3$ read-only rules **not** downgraded to `"flash_lite"`.
   * Regression guard: `"pro"` remains denied and `"inherit"` remains rewritten on `scout` / `research` / `implementer`.
3. Read-only enforcement on `verifier` / `reviewer`: `MUTATION_TOOLS` and mutating shell commands blocked; `read_file` and `git diff` allowed; synchronous completion leaves dirty `test_receipt` untouched.
4. Report validation: `verifier` / `reviewer` require 4 headers (2-header scout shorthand rejected).
5. Freshness gate: `send_message` to `verifier` / `reviewer` allowed before code edits; denied after root edit or `implementer` edit (`mutation_epoch` incremented via parent propagation).
6. Disjoint lane detection:
   * Exact (`src/a.py` vs `src/a.py`) and directory-prefix (`web/` vs `web/App.tsx`) overlap denied across Remote VM `Subagents` batch, local CLI `pending_spawns`, and active `subagents` in `inherit` workspace.
   * Disjoint lanes (`web/`, `api/`, `docs/`) allowed.
   * Overlapping files allowed when isolated in separate `"branch"` or `"share"` workspaces.
   * Sequential reuse allowed after `handle_stop` marks prior `implementer` `"completed"` or `manage_subagents(kill)` marks it `"killed"`.
   * Omitted `ALLOWED_FILES:` stays backward-compatible.
7. `REVIEW_LENS:` enforcement:
   * Missing `REVIEW_LENS:` denied on `reviewer`.
   * Duplicate normalized lenses (`edge cases` vs `edge-cases`) denied across same batch, `pending_spawns`, and active `subagents`.
   * Distinct lenses (`ux`, `simplicity`, `edge-cases`, `security`) allowed.
   * Round-2 fresh `reviewer` reusing `"security"` allowed after round-1 `handle_stop` marks round-1 `"completed"`.

---

## Verification Plan

1. Run the full unit test suite:
   ```bash
   python3 -m unittest discover -s tests
   ```
   Expected: 0 failures and 0 errors across all existing and new tests.
2. Run the Philharmonic doctor check:
   ```bash
   python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor
   ```
   Expected: `"healthy": true` and `"issues": []`.

---

## Pitfalls & Re-Plan Triggers

* **Pitfalls already handled in this plan:**
  * Local CLI parallel spawns sit in `pending_spawns` until `PostToolUse`, while Remote VM spawns arrive in one `Subagents` list — both are covered by `_iter_active_subagents`.
  * Legacy `orchestrator_state.json` entries lack `status` or `mutation_epoch` — default missing `mutation_epoch` to `0` and treat only entries with explicit `status == "active"` as holding a live lane/lens lock so pre-existing test fixtures (`register_subagent` without `handle_stop`) continue to pass.
  * Synchronous `invoke_subagent` completion in [verification.py](../.agents/plugins/philharmonic_orchestra/scripts/guard/verification.py) (L406–432) must exclude `verifier` / `reviewer` from clearing dirty `test_receipt`.
* **Re-plan if:**
  * Existing tests in [test_orchestrator_guard.py](../tests/test_orchestrator_guard.py) (especially around L217, L1517, L2293, L2525, L2837, L4273) assert an undocumented behavior on `subagents` dict shape, `Model: "inherit"` rewriting, or `register_subagent` helper defaults that conflicts with `status: "active"`.
  * Remote VM `Subagents` schema uses a nested `Model` key casing not already handled by `_validate_and_prepare_subagent_entry`.
  * `handle_stop` parent-brain lookup fails when a subagent runs in a worktree whose `parent_conversation_id` is absent from `self_subagent_meta`.

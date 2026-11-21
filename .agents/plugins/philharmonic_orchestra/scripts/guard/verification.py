"""Test integrity checking, git diff auditing, empirical test receipts, circuit breaker, and stop hook."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ASSERTION_RE,
    COMMENT_LINE_RE,
    DOC_EXTENSIONS,
    ISOLATED_WORKSPACES,
    MAX_CONSECUTIVE_FAILURES,
    MAX_REPORT_BULLET_LINES,
    MAX_SUBAGENT_REPORT_CHARS,
    PROJECT_ROOT,
    READONLY_NON_VERIFICATION_TYPES,
    REPORT_BULLET_RE,
    REPORT_LINE_PATTERNS,
    REPORT_LOG_URI_RE,
    SCOUT_COMPACT_LINE_PATTERNS,
    SUBAGENT_SPAWN_RE,
    TAMPER_SKIP_RE,
    TEST_DEF_RE,
    TEST_FILE_RE,
    TRIVIAL_ASSERT_RE,
    UUID_RE,
    _extract_call_id,
    _is_git_only_command,
    _is_sandbox_boundary_error,
    _is_test_command,
    _strip_comments,
)
from guard.identity import (
    _ensure_session_active,
    _is_known_subagent,
    _is_root_orchestrator,
    _is_scout_subagent,
    _mark_subagent_status_in_parent_brain,
    _normalize_subagent_meta,
    _record_cohort_outcome_in_parent_brain,
    _sync_subagents_from_transcript,
    _workspaces_compatible,
)
from guard.policies import (
    _compute_prompt_hash,
    _expand_swarm_worker_names,
    _extract_target_path,
    _is_artifact_or_worktree_path,
    _is_declared_eval_command,
    _parse_cohort_header,
    _parse_mode_header,
)
from guard.state_store import (
    _current_invocation_key,
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)


def _is_test_file(path_str: str) -> bool:
    if not path_str:
        return False
    norm = path_str.replace("\\", "/")
    return bool(TEST_FILE_RE.search(norm))


def _is_code_file(path_str: str) -> bool:
    if not path_str:
        return False
    ext = Path(path_str).suffix.lower()
    return ext not in DOC_EXTENSIONS


def _compare_test_blobs(
    old_text: str, new_text: str, file_name: str, is_diff: bool = False
) -> Optional[str]:
    """Compare old and new test content for tampering (skip markers, trivial asserts, or net removals)."""
    if len(TAMPER_SKIP_RE.findall(new_text)) > len(TAMPER_SKIP_RE.findall(old_text)):
        if is_diff:
            return f"[ORCHESTRATOR GUARD] Diff audit detected test skip/xfail marker added in '{file_name}'."
        return (
            f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
            "adding skip/xfail decorators or disabling tests is prohibited."
        )

    if len(TRIVIAL_ASSERT_RE.findall(new_text)) > len(
        TRIVIAL_ASSERT_RE.findall(old_text)
    ):
        if is_diff:
            return f"[ORCHESTRATOR GUARD] Diff audit detected trivial 'assert True' added in '{file_name}'."
        return (
            f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
            "replacing checks with trivial 'assert True' is prohibited."
        )

    if is_diff or old_text:
        old_clean = _strip_comments(old_text)
        new_clean = _strip_comments(new_text)
        old_tests = len(TEST_DEF_RE.findall(old_clean))
        new_tests = len(TEST_DEF_RE.findall(new_clean))
        if new_tests < old_tests:
            if is_diff:
                return (
                    f"[ORCHESTRATOR GUARD] Diff audit detected net removal of {old_tests - new_tests} "
                    f"test definition(s) in '{file_name}'."
                )
            return (
                f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                f"edit reduces test count from {old_tests} to {new_tests}."
            )

        old_asserts = len(ASSERTION_RE.findall(old_clean))
        new_asserts = len(ASSERTION_RE.findall(new_clean))
        if new_asserts < old_asserts:
            if is_diff:
                return (
                    f"[ORCHESTRATOR GUARD] Diff audit detected net removal of {old_asserts - new_asserts} "
                    f"assertion(s) in '{file_name}'."
                )
            return (
                f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                f"edit reduces assertion count from {old_asserts} to {new_asserts}."
            )

    return None


def _check_test_tampering_in_edit(
    tool_name: str, args: Dict[str, Any], payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Block edits to test files that delete/comment-out tests/assertions or add skip markers."""
    target = _extract_target_path(tool_name, args)
    if not target or _is_artifact_or_worktree_path(target, payload):
        return None
    if not _is_test_file(target):
        return None

    chunks: List[Tuple[str, str]] = []
    if tool_name in {"edit_file", "replace_file_content"}:
        old_chunk = args.get("Target") or args.get("TargetContent") or ""
        new_chunk = args.get("Replacement") or args.get("ReplacementContent") or ""
        chunks.append((old_chunk, new_chunk))
    elif tool_name == "multi_replace_file_content":
        for chunk in (args.get("ReplacementChunks") or []):
            if isinstance(chunk, dict):
                chunks.append(
                    (
                        chunk.get("TargetContent") or "",
                        chunk.get("ReplacementContent") or "",
                    )
                )
    elif tool_name in {"write_file", "write_to_file"}:
        new_code = args.get("Content") or args.get("CodeContent") or ""
        old_code = ""
        try:
            resolved = Path(os.path.expanduser(target)).resolve()
            if resolved.exists():
                old_code = resolved.read_text(encoding="utf-8")
        except OSError:
            old_code = ""
        if args.get("Append") is True and old_code:
            chunks.append((old_code, old_code + "\n" + new_code))
        else:
            chunks.append((old_code, new_code))

    file_name = Path(target).name
    for old_text, new_text in chunks:
        reason = _compare_test_blobs(old_text, new_text, file_name, is_diff=False)
        if reason:
            return {"decision": "deny", "reason": reason}

    return None


def audit_diff_text(diff_text: str) -> Optional[str]:
    """Audit unified git diff text for test tampering across test files."""
    if not diff_text:
        return None

    current_file: Optional[str] = None
    pending_old_file: Optional[str] = None
    removed_lines: List[str] = []
    added_lines: List[str] = []

    def _check_accumulated(
        fname: Optional[str], rem: List[str], add: List[str]
    ) -> Optional[str]:
        if not fname or not _is_test_file(fname):
            return None
        return _compare_test_blobs(
            "\n".join(rem), "\n".join(add), Path(fname).name, is_diff=True
        )

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("--- a/"):
            pending_old_file = raw_line[6:].strip()
            continue
        elif raw_line.startswith("+++ "):
            verdict = _check_accumulated(current_file, removed_lines, added_lines)
            if verdict:
                return verdict
            if raw_line.startswith("+++ b/"):
                current_file = raw_line[6:].strip()
            elif raw_line.strip() == "+++ /dev/null":
                current_file = pending_old_file
            else:
                current_file = None
            pending_old_file = None
            removed_lines = []
            added_lines = []
        elif (
            raw_line.startswith("---")
            or raw_line.startswith("@@")
            or raw_line.startswith("diff --git")
        ):
            continue
        elif raw_line.startswith("-"):
            removed_lines.append(raw_line[1:])
        elif raw_line.startswith("+"):
            added_lines.append(raw_line[1:])

    return _check_accumulated(current_file, removed_lines, added_lines)


def _get_circuit_breaker(
    state: Dict[str, Any], payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    cb_root = state.setdefault("circuit_breaker", {})
    if "failure_streak" not in cb_root:
        cb_root.clear()
        cb_root.update(
            {
                "failure_streak": 0,
                "tripped": False,
                "seen_failures": [],
                "last_error": "",
                "replan_count": 0,
            }
        )
    cb_root.setdefault("replan_count", 0)
    return cb_root


def _get_test_receipt(
    state: Dict[str, Any], payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    receipt_root = state.setdefault("test_receipt", {})
    if "dirty" not in receipt_root:
        prev_epoch = receipt_root.get("mutation_epoch", 0)
        receipt_root.clear()
        receipt_root.update(
            {
                "dirty": False,
                "verified": False,
                "last_mutated_file": "",
                "last_test_cmd": "",
                "mutation_epoch": prev_epoch,
            }
        )
    receipt_root.setdefault("mutation_epoch", 0)
    return receipt_root


def _current_subagent_meta(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    """Return metadata dictionary for the active subagent caller, if any."""
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict) and meta:
        return meta
    cid = (payload.get("conversationId") or "").strip().lower()
    sub_info = (state.get("subagents") or {}).get(cid)
    if isinstance(sub_info, dict) and sub_info:
        return sub_info
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir and cid:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                for sibling_state_path in resolved_art.parent.glob(
                    "*/orchestrator_state.json"
                ):
                    if sibling_state_path.parent == resolved_art:
                        continue
                    try:
                        sib_data = _load_state(sibling_state_path)
                    except (OSError, json.JSONDecodeError):
                        continue
                    sib_subagents = sib_data.get("subagents") or {}
                    if cid in sib_subagents and isinstance(sib_subagents[cid], dict):
                        resolved_meta = _normalize_subagent_meta(sib_subagents[cid])
                        if "parent_conversation_id" not in resolved_meta:
                            resolved_meta["parent_conversation_id"] = (
                                sib_data.get("root_conversation_id")
                                or sibling_state_path.parent.name
                            )
                        state["self_subagent_meta"] = resolved_meta
                        _save_state(payload, state)
                        return resolved_meta
        except OSError:
            pass
    return {}


def _extract_expected_cohort_specs(
    args: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Tuple[Optional[str], Optional[str], str]]]:
    """Extract (entries, [(cohort_id_or_none, role_or_none, prompt_hash), ...]) from invoke_subagent args."""
    entries = (
        args["Subagents"]
        if isinstance(args.get("Subagents"), list) and args["Subagents"]
        else [args]
    )
    specs: List[Tuple[Optional[str], Optional[str], str]] = []
    for item in entries:
        if not isinstance(item, dict):
            specs.append((None, None, ""))
            continue
        prompt_text = str(item.get("Prompt") or "")
        _ = _parse_mode_header(prompt_text)
        c_id = _parse_cohort_header(prompt_text)
        role_val = (item.get("Role") or item.get("TypeName") or "").strip().lower() or None
        prompt_hash = _compute_prompt_hash(prompt_text)
        specs.append((c_id, role_val, prompt_hash))
    return entries, specs


def _pending_matches_spec(
    p: Any,
    exp_cohort: Optional[str],
    exp_role: Optional[str],
    exp_prompt_hash: Optional[str] = None,
    require_prompt_hash: bool = False,
) -> bool:
    """Return True if a pending_spawns record matches the expected cohort, role/type, and optional prompt_hash."""
    if not isinstance(p, dict):
        return False
    if exp_cohort is not None:
        if str(p.get("cohort") or "").strip().lower() != exp_cohort:
            return False
    if exp_role is not None:
        p_role = str(p.get("role") or "").strip().lower()
        p_type = str(p.get("type_name") or p.get("type") or "").strip().lower()
        if exp_role not in {p_role, p_type}:
            return False
    if require_prompt_hash:
        if not exp_prompt_hash or str(p.get("prompt_hash") or "") != exp_prompt_hash:
            return False
    return True


def _match_pending_spawns_two_pass(
    candidates: List[Dict[str, Any]],
    entries: List[Dict[str, Any]],
    expected_specs: List[Tuple[Optional[str], Optional[str], str]],
    *,
    allow_cohort_only_without_role: bool = False,
    allow_type_fallback: bool = False,
    allow_tail_fallback: bool = False,
) -> List[Dict[str, Any]]:
    """Match up to len(entries) active pending spawns via Pass 1 (prompt_hash) and Pass 2 (cohort/role FIFO)."""
    active_candidates = [
        p
        for p in candidates
        if isinstance(p, dict) and p.get("status", "active") == "active"
    ]
    max_matches = len(entries)
    matched_by_idx: Dict[int, Dict[str, Any]] = {}
    used_ids: set = set()

    # Pass 1: exact prompt_hash + cohort/role match
    for idx, (exp_cohort, exp_role, exp_prompt_hash) in enumerate(expected_specs):
        if idx >= max_matches:
            break
        if not exp_prompt_hash:
            continue
        for p in active_candidates:
            if id(p) not in used_ids and _pending_matches_spec(
                p,
                exp_cohort,
                exp_role,
                exp_prompt_hash=exp_prompt_hash,
                require_prompt_hash=True,
            ):
                used_ids.add(id(p))
                matched_by_idx[idx] = p
                break

    # Pass 2: cohort/role FIFO fallback (plus optional call-site fallbacks)
    for idx, (exp_cohort, exp_role, _) in enumerate(expected_specs):
        if idx >= max_matches or len(matched_by_idx) >= max_matches:
            break
        if idx in matched_by_idx:
            continue
        item = (
            entries[idx]
            if idx < len(entries) and isinstance(entries[idx], dict)
            else {}
        )
        chosen: Optional[Dict[str, Any]] = None
        if exp_cohort is not None or exp_role is not None:
            for p in active_candidates:
                if id(p) not in used_ids and _pending_matches_spec(
                    p, exp_cohort, exp_role
                ):
                    chosen = p
                    break
        if chosen is None and allow_cohort_only_without_role:
            has_explicit_role = bool(str(item.get("Role") or "").strip())
            if not has_explicit_role:
                for p in active_candidates:
                    if id(p) not in used_ids and (
                        exp_cohort is None
                        or str(p.get("cohort") or "").strip().lower() == exp_cohort
                    ):
                        chosen = p
                        break
        if chosen is None and allow_type_fallback:
            inv_type = str(item.get("TypeName") or "").strip().lower()
            if inv_type:
                for p in active_candidates:
                    if id(p) not in used_ids and (
                        exp_cohort is None
                        or str(p.get("cohort") or "").strip().lower() == exp_cohort
                    ):
                        p_type = str(
                            p.get("type_name") or p.get("type") or ""
                        ).strip().lower()
                        if p_type == inv_type:
                            chosen = p
                            break
        if chosen is None and allow_tail_fallback:
            for p in reversed(active_candidates):
                if id(p) not in used_ids:
                    chosen = p
                    break
            if chosen is None:
                for p in reversed(candidates):
                    if isinstance(p, dict) and id(p) not in used_ids:
                        chosen = p
                        break
        if chosen is not None:
            used_ids.add(id(chosen))
            matched_by_idx[idx] = chosen

    return [
        matched_by_idx[idx]
        for idx in range(min(max_matches, len(expected_specs)))
        if idx in matched_by_idx
    ]


def _collect_invoke_cohort_candidates(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    args: Dict[str, Any],
    call_id: str,
    output_text: str,
    pre_popped_entries: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[str, str]]:
    """Resolve (cohort_id, candidate_key) pairs for an invoke_subagent call."""
    matched_cids_list: List[str] = []
    for regex in (SUBAGENT_SPAWN_RE, UUID_RE):
        for m in regex.finditer(output_text or ""):
            cid_val = m.group(1).lower()
            if cid_val not in matched_cids_list:
                matched_cids_list.append(cid_val)
    matched_cids_set = set(matched_cids_list)

    if pre_popped_entries:
        results: List[Tuple[str, str]] = []
        multi = len(pre_popped_entries) > 1
        for idx, p in enumerate(pre_popped_entries):
            c_id = p.get("cohort")
            if p.get("mode") == "best_of_n" and c_id:
                cand_key = (
                    p.get("candidate_id")
                    or (matched_cids_list[idx] if idx < len(matched_cids_list) else None)
                    or p.get("call_sig")
                    or (f"{call_id}:{idx}" if call_id and multi else call_id)
                    or ""
                )
                results.append((str(c_id), str(cand_key)))
        if results:
            return results

    subagent_results: List[Tuple[str, str]] = []
    for sub_cid, sub_meta in (state.get("subagents") or {}).items():
        if not isinstance(sub_meta, dict):
            continue
        if sub_cid.lower() in matched_cids_set or (
            call_id and sub_meta.get("call_id") == call_id
        ):
            if sub_meta.get("mode") == "best_of_n" and sub_meta.get("cohort"):
                cand_key = sub_meta.get("candidate_id") or sub_cid.lower()
                subagent_results.append((str(sub_meta["cohort"]), str(cand_key)))
    if subagent_results:
        return subagent_results

    if call_id:
        pending_matches = [
            p
            for p in (state.get("pending_spawns") or [])
            if isinstance(p, dict)
            and p.get("mode") == "best_of_n"
            and p.get("cohort")
            and p.get("call_id") == call_id
        ]
    else:
        entries, expected_specs = _extract_expected_cohort_specs(args)
        active_pending = [
            p
            for p in (state.get("pending_spawns") or [])
            if isinstance(p, dict)
            and p.get("mode") == "best_of_n"
            and p.get("cohort")
            and p.get("status", "active") == "active"
        ]
        pending_matches = _match_pending_spawns_two_pass(
            active_pending,
            entries,
            expected_specs,
            allow_cohort_only_without_role=True,
        )

    if pending_matches:
        multi = len(pending_matches) > 1
        return [
            (
                str(p["cohort"]),
                str(
                    p.get("candidate_id")
                    or (matched_cids_list[idx] if idx < len(matched_cids_list) else None)
                    or p.get("call_sig")
                    or (f"{call_id}:{idx}" if call_id and multi else call_id)
                    or ""
                ),
            )
            for idx, p in enumerate(pending_matches)
        ]

    entries = (
        args["Subagents"]
        if isinstance(args.get("Subagents"), list) and args["Subagents"]
        else [args]
    )
    inv_key = _current_invocation_key(payload, state)
    entry_results: List[Tuple[str, str]] = []
    multi = len(entries) > 1
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        prompt_text = str(item.get("Prompt") or "")
        mode = _parse_mode_header(prompt_text)
        cohort_id = _parse_cohort_header(prompt_text)
        if cohort_id and mode is None:
            mode = "best_of_n"
        if mode == "best_of_n":
            c_id = cohort_id or f"best-of-n-{inv_key}"
            cand_key = (
                (matched_cids_list[idx] if idx < len(matched_cids_list) else None)
                or (f"{call_id}:{idx}" if call_id and multi else call_id)
                or ""
            )
            entry_results.append((c_id, str(cand_key)))
    return entry_results


def _apply_cohort_invoke_outcome(
    state: Dict[str, Any],
    cb: Dict[str, Any],
    cohort_candidates: List[Tuple[str, str]],
    outcome: str,
    error_summary: str = "",
) -> None:
    """Update cohort succeeded/failed tracking and trip global breaker only if 0/N (N>=2) survive."""
    cohorts = state.setdefault("cohorts", {})
    for cohort_id, cand_key in cohort_candidates:
        cohort_entry = cohorts.get(cohort_id)
        if not isinstance(cohort_entry, dict):
            cohort_entry = {
                "total_spawned": 0,
                "succeeded": [],
                "failed": [],
                "created_at": time.time(),
            }
            cohorts[cohort_id] = cohort_entry
        succeeded = cohort_entry.setdefault("succeeded", [])
        failed = cohort_entry.setdefault("failed", [])
        if outcome == "failed":
            key = cand_key or f"fail:{len(failed)}"
            if key not in succeeded and key not in failed:
                failed.append(key)
            total_spawned = int(cohort_entry.get("total_spawned", 0) or 0)
            if (
                total_spawned >= 2
                and len(failed) >= total_spawned
                and len(succeeded) == 0
            ):
                cb["failure_streak"] = max(
                    MAX_CONSECUTIVE_FAILURES, int(cb.get("failure_streak", 0) or 0) + 1
                )
                cb["tripped"] = True
                cb["last_error"] = (
                    error_summary or "All Best-of-N cohort candidates failed"
                )
        else:
            key = cand_key or f"ok:{len(succeeded)}"
            if key not in succeeded:
                succeeded.append(key)
            if key in failed:
                failed.remove(key)
            cb["failure_streak"] = 0
            cb["tripped"] = False


def _propagate_receipt_to_parent_brain(
    payload: Dict[str, Any],
    dirty: bool,
    verified: bool,
    last_mutated_file: Optional[str] = None,
    last_test_cmd: Optional[str] = None,
) -> None:
    """Propagate subagent workspace code mutation / test verification receipts to parent brain state."""
    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir:
        return
    try:
        resolved_art = Path(artifact_dir).resolve()
        if resolved_art.parent.name != "brain":
            return

        current_state = _load_state(payload)
        _is_known_subagent(payload, current_state)
        meta = current_state.get("self_subagent_meta") or {}
        parent_cid = str(meta.get("parent_conversation_id") or "").strip().lower()

        cid = (payload.get("conversationId") or "").strip().lower()
        for sibling_state_path in resolved_art.parent.glob("*/orchestrator_state.json"):
            if sibling_state_path.parent == resolved_art:
                continue

            with state_transaction(sibling_state_path):
                sib_data = _load_state(sibling_state_path)
                sib_subagents = sib_data.get("subagents") or {}
                root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
                dir_name = sibling_state_path.parent.name.lower()

                matches_subagent = bool(cid and cid in sib_subagents)
                matches_parent = bool(parent_cid and (root_cid == parent_cid or dir_name == parent_cid))

                if (matches_subagent or matches_parent) and _workspaces_compatible(
                    payload, sib_data
                ):
                    parent_receipt = _get_test_receipt(sib_data)
                    parent_receipt["dirty"] = dirty
                    parent_receipt["verified"] = verified
                    if dirty:
                        parent_receipt["mutation_epoch"] = (
                            parent_receipt.get("mutation_epoch", 0) + 1
                        )
                    if last_mutated_file is not None:
                        parent_receipt["last_mutated_file"] = last_mutated_file
                    if last_test_cmd is not None:
                        parent_receipt["last_test_cmd"] = last_test_cmd
                    _save_state(sibling_state_path, sib_data)
    except OSError:
        pass


def handle_post_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}
        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        call_id = _extract_call_id(payload)

        if tool_name == "exit_plan_mode" and payload.get("error"):
            state["plan_execute_phase"] = False
            state["plan_execute_consumed"] = False
            state["plan_execute_user_input"] = None
            state["plan_execute_turn_finished"] = False
            state["plan_allowed_files"] = None
            state["active_eval_cmd"] = None

        failed_cohort_entries: List[Dict[str, Any]] = []
        if tool_name == "invoke_subagent" and payload.get("error"):
            pending = state.get("pending_spawns", [])
            recorded = state.get("recorded_spawn_calls", [])
            if call_id and any(p.get("call_id") == call_id for p in pending):
                failed_cohort_entries = [
                    dict(p)
                    for p in pending
                    if isinstance(p, dict)
                    and p.get("call_id") == call_id
                    and p.get("mode") == "best_of_n"
                    and p.get("cohort")
                ]
                state["pending_spawns"] = [p for p in pending if p.get("call_id") != call_id]
                if isinstance(recorded, list):
                    state["recorded_spawn_calls"] = [
                        sig for sig in recorded
                        if not (isinstance(sig, str) and call_id in sig)
                    ]
            else:
                entries, expected_specs = _extract_expected_cohort_specs(args)
                matched_pops = _match_pending_spawns_two_pass(
                    pending,
                    entries,
                    expected_specs,
                    allow_tail_fallback=True,
                )
                if matched_pops:
                    matched_ids = {id(p) for p in matched_pops}
                    state["pending_spawns"] = [
                        p for p in pending if id(p) not in matched_ids
                    ]
                    for popped in matched_pops:
                        if (
                            popped.get("mode") == "best_of_n"
                            and popped.get("cohort")
                        ):
                            failed_cohort_entries.append(dict(popped))
        else:
            _sync_subagents_from_transcript(payload, state)

        cb = _get_circuit_breaker(state, payload)
        receipt = _get_test_receipt(state, payload)
        error_text = str(payload.get("error") or "").strip()

        if tool_name in {"run_command", "invoke_subagent", "run_workflow", "swarm"}:
            cmd = (args.get("CommandLine") or "").strip()
            is_test_cmd = tool_name == "run_command" and _is_test_command(cmd)
            is_eval_cmd = tool_name == "run_command" and _is_declared_eval_command(
                cmd, state, payload
            )
            is_verification_cmd = is_test_cmd or is_eval_cmd
            output_text = str(
                payload.get("output")
                or payload.get("result")
                or tool_call.get("result")
                or tool_call.get("output")
                or ""
            )
            if not error_text and tool_name in {"run_command", "run_workflow", "swarm"}:
                exit_code = (
                    payload.get("exitCode")
                    if payload.get("exitCode") is not None
                    else payload.get("returnCode")
                    if payload.get("returnCode") is not None
                    else tool_call.get("exitCode")
                )
                if isinstance(exit_code, int) and exit_code != 0:
                    error_text = f"Command exited with code {exit_code}"
                else:
                    m_exit = re.search(
                        r"(?m)^Command exited with code ([1-9][0-9]*)\b", output_text
                    )
                    if m_exit:
                        error_text = f"Command exited with code {m_exit.group(1)}"
            if (
                not error_text
                and tool_name == "run_workflow"
                and "Traceback (most recent call last):" in output_text
            ):
                error_text = "Workflow script raised an exception"

            caller_mode = _current_subagent_meta(payload, state).get("mode")
            if error_text:
                if (
                    tool_name == "swarm"
                    and str(args.get("action") or "").strip().lower() == "add"
                ):
                    swarm_workers = state.get("swarm_workers")
                    rolled_back_names: List[str] = []
                    if isinstance(swarm_workers, dict):
                        if call_id:
                            to_remove = [
                                wname
                                for wname, meta in swarm_workers.items()
                                if isinstance(meta, dict)
                                and meta.get("call_id") == call_id
                            ]
                        else:
                            to_remove = [
                                wname
                                for wname in _expand_swarm_worker_names(args)
                                if wname in swarm_workers
                            ]
                        for wname in to_remove:
                            swarm_workers.pop(wname, None)
                            rolled_back_names.append(wname)
                    if rolled_back_names:
                        councils = state.get("councils")
                        if isinstance(councils, dict):
                            for c_entry in councils.values():
                                if not isinstance(c_entry, dict):
                                    continue
                                for role_list_key in ("proposers", "judges"):
                                    role_list = c_entry.get(role_list_key)
                                    if isinstance(role_list, list):
                                        c_entry[role_list_key] = [
                                            item
                                            for item in role_list
                                            if item not in rolled_back_names
                                        ]
                elif (
                    tool_name == "swarm"
                    and str(args.get("action") or "").strip().lower() == "kill"
                ):
                    wname = str(args.get("name") or "").strip()
                    swarm_workers = state.get("swarm_workers")
                    if (
                        wname
                        and isinstance(swarm_workers, dict)
                        and wname in swarm_workers
                        and isinstance(swarm_workers[wname], dict)
                    ):
                        meta = swarm_workers[wname]
                        meta["status"] = "active"
                        meta["active_worktree"] = (
                            str(meta.get("workspace") or "inherit").strip().lower()
                            in ISOLATED_WORKSPACES
                        )
                cohort_candidates = (
                    _collect_invoke_cohort_candidates(
                        payload,
                        state,
                        args,
                        call_id,
                        output_text,
                        pre_popped_entries=failed_cohort_entries,
                    )
                    if tool_name == "invoke_subagent"
                    else []
                )
                if cohort_candidates:
                    _apply_cohort_invoke_outcome(
                        state,
                        cb,
                        cohort_candidates,
                        "failed",
                        error_summary=error_text,
                    )
                else:
                    is_sandbox_err = (
                        tool_name in {"run_command", "run_workflow"}
                        and _is_sandbox_boundary_error(payload, error_text, output_text)
                    )
                    if not is_sandbox_err:
                        seen_failures: List[str] = cb.setdefault("seen_failures", [])
                        dedup_id = f"{tool_name}:{call_id}:{error_text}"
                        if not call_id or dedup_id not in seen_failures:
                            if call_id:
                                seen_failures.append(dedup_id)
                            cb["failure_streak"] = cb.get("failure_streak", 0) + 1
                            cb["last_error"] = error_text
                            if cb["failure_streak"] >= MAX_CONSECUTIVE_FAILURES:
                                cb["tripped"] = True
                if is_verification_cmd:
                    receipt["verified"] = False
                    receipt["last_test_cmd"] = cmd
                    if (
                        not _is_root_orchestrator(payload, state)
                        and caller_mode != "best_of_n"
                    ):
                        _propagate_receipt_to_parent_brain(
                            payload,
                            dirty=bool(receipt.get("dirty", True)),
                            verified=False,
                            last_test_cmd=cmd,
                        )
            else:
                subagent_failed_status = bool(
                    tool_name == "invoke_subagent"
                    and re.search(
                        r"(?m)^\s*STATUS:\s*(?:FAILED|BLOCKED)\b", output_text
                    )
                )
                subagent_non_final = bool(
                    tool_name == "invoke_subagent"
                    and re.search(
                        r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED|NEEDS_REPLAN|CHECKPOINT)\b",
                        output_text,
                    )
                )
                entries = (
                    args["Subagents"]
                    if isinstance(args.get("Subagents"), list) and args["Subagents"]
                    else [args]
                )
                invoked_types = [
                    str((item or {}).get("TypeName") or "").strip().lower()
                    for item in entries
                    if isinstance(item, dict)
                ]
                is_async_spawn_ack = bool(
                    tool_name == "invoke_subagent"
                    and re.search(r"Created the following subagents?\b", output_text)
                    and not re.search(r"(?m)^\s*STATUS:\s*COMPLETED\b", output_text)
                )
                cohort_candidates = (
                    _collect_invoke_cohort_candidates(
                        payload, state, args, call_id, output_text
                    )
                    if tool_name == "invoke_subagent"
                    else []
                )
                if cohort_candidates and subagent_failed_status:
                    _apply_cohort_invoke_outcome(
                        state,
                        cb,
                        cohort_candidates,
                        "failed",
                        error_summary="Best-of-N candidate reported STATUS: FAILED/BLOCKED",
                    )
                elif (
                    cohort_candidates
                    and not subagent_non_final
                    and not is_async_spawn_ack
                ):
                    _apply_cohort_invoke_outcome(
                        state, cb, cohort_candidates, "succeeded"
                    )
                elif (
                    (tool_name == "invoke_subagent" and not subagent_failed_status)
                    or is_verification_cmd
                    or (tool_name == "run_command" and not _is_git_only_command(cmd))
                    or tool_name == "run_workflow"
                    or tool_name == "swarm"
                ):
                    cb["failure_streak"] = 0
                    cb["tripped"] = False

                if tool_name == "invoke_subagent" and not is_async_spawn_ack:
                    matched_cids = {
                        m.group(1).lower()
                        for m in SUBAGENT_SPAWN_RE.finditer(output_text)
                    } | {
                        m.group(1).lower()
                        for m in UUID_RE.finditer(output_text)
                    }
                    for sub_cid, sub_meta in (state.get("subagents") or {}).items():
                        if not isinstance(sub_meta, dict):
                            continue
                        if sub_cid.lower() in matched_cids or (
                            call_id and sub_meta.get("call_id") == call_id
                        ):
                            sub_meta["status"] = "completed"

                    pending_list = state.get("pending_spawns") or []
                    if call_id:
                        for pending_entry in pending_list:
                            if (
                                isinstance(pending_entry, dict)
                                and pending_entry.get("status", "active") == "active"
                                and pending_entry.get("call_id") == call_id
                            ):
                                pending_entry["status"] = "completed"
                    else:
                        _, expected_specs = _extract_expected_cohort_specs(args)
                        matched_completions = _match_pending_spawns_two_pass(
                            pending_list,
                            entries,
                            expected_specs,
                            allow_type_fallback=True,
                        )
                        for chosen_entry in matched_completions:
                            chosen_entry["status"] = "completed"

                is_verification_subagent = (
                    tool_name == "invoke_subagent"
                    and not subagent_non_final
                    and not is_async_spawn_ack
                    and not cohort_candidates
                    and any(
                        t not in READONLY_NON_VERIFICATION_TYPES for t in invoked_types
                    )
                )
                if is_verification_cmd or (is_verification_subagent and receipt.get("dirty")):
                    receipt["dirty"] = False
                    receipt["verified"] = True
                    receipt["last_test_cmd"] = cmd or "invoke_subagent"
                    if (
                        is_verification_cmd
                        and not _is_root_orchestrator(payload, state)
                        and caller_mode != "best_of_n"
                    ):
                        _propagate_receipt_to_parent_brain(
                            payload,
                            dirty=False,
                            verified=True,
                            last_test_cmd=cmd,
                        )

        _save_state(payload, state)
        return {}


def validate_subagent_report(text: str, is_scout: bool = False) -> Optional[str]:
    """Validate that a subagent's final output matches the REPORT FORMAT."""
    stripped = (text or "").strip()
    if not stripped:
        return "Subagent response is empty."
    if len(stripped) > MAX_SUBAGENT_REPORT_CHARS:
        return (
            f"Subagent report length ({len(stripped)} chars) exceeds maximum "
            f"{MAX_SUBAGENT_REPORT_CHARS} chars."
        )
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]

    if (
        is_scout
        and len(lines) >= 2
        and bool(SCOUT_COMPACT_LINE_PATTERNS[0].match(lines[0]))
        and bool(SCOUT_COMPACT_LINE_PATTERNS[1].match(lines[1]))
    ):
        max_lines = 2 + MAX_REPORT_BULLET_LINES
        if len(lines) > max_lines:
            return (
                f"Expected 2 header lines plus at most {MAX_REPORT_BULLET_LINES} bullet lines "
                f"(2-{max_lines} lines total), got {len(lines)}."
            )
        bullet_lines = lines[2:]
        for b_idx, b_line in enumerate(bullet_lines, start=3):
            if not REPORT_BULLET_RE.match(b_line):
                return f"Line {b_idx} after NOTES must be a compact bullet starting with '- ' or '* '."

        status_val = lines[0].split(":", 1)[1].strip().upper().split()[0]
        if status_val in {"BLOCKED", "FAILED"}:
            notes_body = lines[1].split(":", 1)[1].strip()
            combined_notes = " ".join(
                [notes_body] + [b[1:].strip() for b in bullet_lines]
            ).strip()
            non_uri_text = REPORT_LOG_URI_RE.sub("", combined_notes).strip(" -—:,.;()")
            if len(non_uri_text.split()) < 3:
                return (
                    f"When STATUS is {status_val}, NOTES must include an inline error summary "
                    "(at least 3 words)."
                )
        return None

    max_lines = 4 + MAX_REPORT_BULLET_LINES
    if len(lines) < 4 or len(lines) > max_lines:
        return (
            f"Expected 4 header lines plus at most {MAX_REPORT_BULLET_LINES} bullet lines "
            f"(4-{max_lines} lines total), got {len(lines)}."
        )
    expected_prefixes = ("STATUS:", "FILES_CHANGED:", "VERIFICATION:", "NOTES:")
    for idx, (pattern, prefix) in enumerate(
        zip(REPORT_LINE_PATTERNS, expected_prefixes), start=1
    ):
        if not pattern.match(lines[idx - 1]):
            return f"Line {idx} must match '{prefix} <value>'."

    bullet_lines = lines[4:]
    for b_idx, b_line in enumerate(bullet_lines, start=5):
        if not REPORT_BULLET_RE.match(b_line):
            return f"Line {b_idx} after NOTES must be a compact bullet starting with '- ' or '* '."

    status_val = lines[0].split(":", 1)[1].strip().upper().split()[0]
    notes_body = lines[3].split(":", 1)[1].strip()
    combined_notes = " ".join(
        [notes_body] + [b[1:].strip() for b in bullet_lines]
    ).strip()

    if status_val == "NEEDS_REPLAN":
        non_uri_text = REPORT_LOG_URI_RE.sub("", combined_notes).strip(" -—:,.;()")
        if len(non_uri_text.split()) < 3:
            return (
                "When STATUS is NEEDS_REPLAN, NOTES must include an inline explanation "
                "(at least 3 words)."
            )
    elif status_val in {"BLOCKED", "FAILED"}:
        if not REPORT_LOG_URI_RE.search(combined_notes):
            return f"When STATUS is {status_val}, NOTES must include a 'file://.../*.log' scratch log URI."
        non_uri_text = REPORT_LOG_URI_RE.sub("", combined_notes).strip(" -—:,.;()")
        if len(non_uri_text.split()) < 3:
            return (
                f"When STATUS is {status_val}, NOTES must include an inline error summary "
                "(at least 3 words) in addition to the file:// scratch log URI."
            )

    return None


def _extract_last_model_response(payload_or_path: Any) -> Optional[str]:
    """Extract the final model response text from payload or transcriptPath if available."""
    if isinstance(payload_or_path, dict):
        if (
            "lastAssistantMessage" in payload_or_path
            and payload_or_path["lastAssistantMessage"] is not None
        ):
            return str(payload_or_path["lastAssistantMessage"])
        transcript_path = payload_or_path.get("transcriptPath")
    else:
        transcript_path = str(payload_or_path)

    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        file_size = os.path.getsize(transcript_path)
        tail_size = 65536
        offset = max(0, file_size - tail_size)
        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            tail_bytes = fh.read()
        tail_text = tail_bytes.decode("utf-8", errors="replace")
        lines = tail_text.splitlines()
        if offset > 0 and len(lines) > 1:
            lines = lines[1:]
        for raw_line in reversed(lines):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                step = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if step.get("type") == "PLANNER_RESPONSE" and not step.get("tool_calls"):
                content = step.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    except OSError:
        return None
    return None


def _get_workspace_git_diff(payload: Dict[str, Any]) -> str:
    """Return unified git diff from payload or by running git diff HEAD in workspace."""
    if payload.get("gitDiff"):
        return str(payload["gitDiff"])
    for diff_cmd in (["git", "diff", "HEAD", "-U0"], ["git", "diff", "-U0"]):
        try:
            proc = subprocess.run(
                diff_cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout or ""
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def _finalize_subagent_stop_sync(
    payload: Dict[str, Any], state: Dict[str, Any], cb_tripped: bool = False
) -> None:
    """Sync subagent completion and Best-of-N cohort outcome into parent brain state."""
    cid = (payload.get("conversationId") or "").strip().lower()
    if not cid:
        return
    _mark_subagent_status_in_parent_brain(payload, state, cid, "completed")
    sub_meta = _current_subagent_meta(payload, state)
    if sub_meta.get("mode") == "best_of_n" and sub_meta.get("cohort"):
        last_msg = _extract_last_model_response(payload) or ""
        status_m = re.search(r"(?m)^\s*STATUS:\s*([A-Za-z_]+)\b", last_msg)
        report_status = status_m.group(1).upper() if status_m else ""
        if report_status in {"FAILED", "BLOCKED"} or (
            cb_tripped and report_status != "COMPLETED"
        ):
            outcome = "failed"
            cb_root = _get_circuit_breaker(state, payload)
            err_summary = (
                str(cb_root.get("last_error") or "").strip()
                or f"Subagent {cid} finished with STATUS: {report_status or 'BLOCKED'}"
            )
        else:
            outcome = "succeeded"
            err_summary = ""
        sub_reg = (state.get("subagents") or {}).get(cid)
        candidate_key = str(
            sub_meta.get("candidate_id")
            or (sub_reg.get("candidate_id") if isinstance(sub_reg, dict) else None)
            or cid
        )
        _record_cohort_outcome_in_parent_brain(
            payload,
            state,
            candidate_key,
            str(sub_meta["cohort"]),
            outcome,
            error_summary=err_summary,
        )


def handle_stop(payload: Dict[str, Any]) -> Dict[str, Any]:
    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}

        termination_reason = payload.get("terminationReason", "")
        if termination_reason in {
            "USER_CANCELED",
            "ERROR",
            "MAX_INVOCATIONS",
            "MAX_FORCED_INVOCATIONS",
            "MAX_TOKEN_BUDGET_EXCEEDED",
        }:
            return {"decision": "allow"}

        if not payload.get("fullyIdle", True):
            return {"decision": "allow"}

        _sync_subagents_from_transcript(payload, state)

        active_branches = [
            cid
            for cid, meta in state.get("subagents", {}).items()
            if isinstance(meta, dict)
            and (meta.get("active_worktree") or meta.get("active_branch"))
        ]
        active_pending = [
            p.get("role", "implementer")
            for p in state.get("pending_spawns", [])
            if isinstance(p, dict)
            and (p.get("active_worktree") or p.get("active_branch"))
        ]
        active_swarm_worktrees = [
            wname
            for wname, meta in (state.get("swarm_workers") or {}).items()
            if isinstance(meta, dict)
            and meta.get("status") == "active"
            and meta.get("active_worktree")
        ]

        if active_branches or active_pending or active_swarm_worktrees:
            exec_key = f"{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
            reminded = state.setdefault("stop_reminded_executions", [])
            if exec_key not in reminded:
                reminded.append(exec_key)
                _save_state(payload, state)
                if active_branches or active_pending:
                    targets = ", ".join(active_branches or active_pending)
                    return {
                        "decision": "continue",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] Uncleaned branched subagent worktrees remain ({targets}). "
                            "Verify/merge branch changes and call manage_subagents(Action='kill') before finishing."
                        ),
                    }
                return {
                    "decision": "continue",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Uncleaned isolated swarm worktrees remain "
                        f"({', '.join(active_swarm_worktrees)}). "
                        "Merge/discard changes and call swarm(action='kill') before finishing."
                    ),
                }

        if not _is_root_orchestrator(payload, state):
            sent_key = f"sent:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
            reminded = state.setdefault("stop_reminded_executions", [])
            if sent_key not in reminded:
                last_msg = _extract_last_model_response(payload)
                if last_msg is not None:
                    report_err = validate_subagent_report(
                        last_msg, is_scout=_is_scout_subagent(payload, state)
                    )
                    if report_err:
                        report_key = f"report:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
                        if report_key not in reminded:
                            reminded.append(report_key)
                            _save_state(payload, state)
                            return {
                                "decision": "continue",
                                "reason": (
                                    f"[ORCHESTRATOR GUARD] Subagent context-separation violation: {report_err} "
                                    "Return ONLY the exact 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                                    "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                                    "file:// .log URI when BLOCKED or FAILED; scouts may use compact 2-header STATUS + NOTES)."
                                ),
                            }

        cb = _get_circuit_breaker(state, payload)
        if cb.get("tripped"):
            if _is_known_subagent(payload, state):
                _finalize_subagent_stop_sync(payload, state, cb_tripped=True)
            return {"decision": "allow"}

        inv_key = _current_invocation_key(payload, state)
        mutated_in_turn = bool(
            state.get("direct_budget", {}).get(inv_key, {}).get("mutated_files")
        )
        receipt = _get_test_receipt(state, payload)
        diff_text = (
            _get_workspace_git_diff(payload)
            if (payload.get("gitDiff") or receipt.get("dirty") or mutated_in_turn)
            else ""
        )
        if diff_text:
            tamper_finding = audit_diff_text(diff_text)
            if tamper_finding:
                exec_key = f"tamper:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
                reminded = state.setdefault("stop_reminded_executions", [])
                if exec_key not in reminded:
                    reminded.append(exec_key)
                    _save_state(payload, state)
                    return {"decision": "continue", "reason": tamper_finding}

        if receipt.get("dirty") and not receipt.get("verified"):
            exec_key = f"receipt:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
            reminded = state.setdefault("stop_reminded_executions", [])
            if exec_key not in reminded:
                reminded.append(exec_key)
                _save_state(payload, state)
                last_file = Path(receipt.get("last_mutated_file") or "workspace").name
                return {
                    "decision": "continue",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Empirical test receipt missing: code file "
                        f"'{last_file}' was modified without a subsequent passing test run "
                        "recorded in PostToolUse. Run the test suite to verify changes before finishing."
                    ),
                }

        if _is_known_subagent(payload, state):
            _finalize_subagent_stop_sync(payload, state, cb_tripped=False)
        elif payload.get("fullyIdle") is True and state.get("plan_execute_phase"):
            has_active_subagents = any(
                isinstance(p, dict) and p.get("status", "active") == "active"
                for p in (state.get("pending_spawns") or [])
            ) or any(
                isinstance(s, dict) and s.get("status") == "active"
                for s in (state.get("subagents") or {}).values()
            )
            if not has_active_subagents:
                state["plan_execute_turn_finished"] = True
                _save_state(payload, state)

        return {"decision": "allow"}

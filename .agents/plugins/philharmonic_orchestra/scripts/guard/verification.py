"""Test integrity checking, git diff auditing, empirical test receipts, circuit breaker, and stop hook."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ASSERTION_RE, DOC_EXTENSIONS, ISOLATED_WORKSPACES, MAX_CONSECUTIVE_FAILURES,
    MAX_REPORT_BULLET_LINES, MAX_SUBAGENT_REPORT_CHARS, PROJECT_ROOT,
    READONLY_NON_VERIFICATION_TYPES, REPORT_BULLET_RE, REPORT_LINE_PATTERNS,
    REPORT_LOG_URI_RE, SCOUT_COMPACT_LINE_PATTERNS, SUBAGENT_SPAWN_RE,
    TAMPER_SKIP_RE, TEST_DEF_RE, TEST_FILE_RE, TRIVIAL_ASSERT_RE, UUID_RE,
    _count_early_test_returns, _extract_call_id, _is_sandbox_boundary_error, _strip_comments,
)
from guard.identity import (
    _apply_cohort_outcome, _ensure_session_active, _get_subagent_type,
    _is_known_subagent, _is_root_orchestrator, _is_scout_subagent,
    _mark_subagent_status_in_parent_brain, _normalize_subagent_meta,
    _preclaim_subagent_meta, _read_sibling_snapshot,
    _record_cohort_outcome_in_parent_brain, _sync_subagents_from_transcript,
    _workspaces_compatible,
)
from guard.state_store import (
    _current_invocation_key, _get_circuit_breaker, _get_test_receipt,
    _has_other_locked_path, _load_state, _save_state, _state_file_path, state_transaction,
)


def _is_test_file(path_str: str) -> bool:
    return bool(path_str and TEST_FILE_RE.search(path_str.replace("\\", "/")))


def _is_code_file(path_str: str) -> bool:
    return bool(path_str and Path(path_str).suffix.lower() not in DOC_EXTENSIONS)


def _compare_test_blobs(old_text: str, new_text: str, file_name: str, is_diff: bool = False) -> Optional[str]:
    """Compare old and new test content for tampering (skip markers, trivial asserts, or net removals)."""
    def _msg(diff_txt: str, edit_txt: str) -> str:
        return f"[ORCHESTRATOR GUARD] {diff_txt if is_diff else edit_txt}"

    old_skip = _strip_comments(old_text, file_name=file_name, preserve_ignore=True)
    new_skip = _strip_comments(new_text, file_name=file_name, preserve_ignore=True)
    if len(TAMPER_SKIP_RE.findall(new_skip)) > len(TAMPER_SKIP_RE.findall(old_skip)):
        return _msg(
            f"Diff audit detected test skip/xfail marker added in '{file_name}'.",
            f"Test tampering blocked in '{file_name}': adding skip/xfail decorators or disabling tests is prohibited.",
        )

    old_clean = _strip_comments(old_text, file_name=file_name)
    new_clean = _strip_comments(new_text, file_name=file_name)
    if len(TRIVIAL_ASSERT_RE.findall(new_clean)) > len(TRIVIAL_ASSERT_RE.findall(old_clean)):
        return _msg(
            f"Diff audit detected trivial 'assert True' added in '{file_name}'.",
            f"Test tampering blocked in '{file_name}': replacing checks with trivial 'assert True' is prohibited.",
        )
    if _count_early_test_returns(new_text, file_name=file_name) > _count_early_test_returns(old_text, file_name=file_name):
        return _msg(
            f"Diff audit detected early bare 'return' added in '{file_name}'.",
            f"Test tampering blocked in '{file_name}': adding early bare 'return' inside a test function is prohibited.",
        )

    if is_diff or old_text:
        old_tests, new_tests = len(TEST_DEF_RE.findall(old_clean)), len(TEST_DEF_RE.findall(new_clean))
        if new_tests < old_tests:
            return _msg(
                f"Diff audit detected net removal of {old_tests - new_tests} test definition(s) in '{file_name}'.",
                f"Test tampering blocked in '{file_name}': edit reduces test count from {old_tests} to {new_tests}.",
            )
        old_asserts, new_asserts = len(ASSERTION_RE.findall(old_clean)), len(ASSERTION_RE.findall(new_clean))
        if new_asserts < old_asserts:
            return _msg(
                f"Diff audit detected net removal of {old_asserts - new_asserts} assertion(s) in '{file_name}'.",
                f"Test tampering blocked in '{file_name}': edit reduces assertion count from {old_asserts} to {new_asserts}.",
            )
    return None


def _validate_str_keys(d: Dict[str, Any], keys: Tuple[str, ...], ctx: str) -> Optional[Dict[str, Any]]:
    for key in keys:
        if (val := d.get(key)) is not None and not isinstance(val, str):
            return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] Invalid non-string '{key}' {ctx}."}
    return None


def _check_test_tampering_in_edit(tool_name: str, args: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Block edits to test files that delete/comment-out tests/assertions or add skip markers."""
    from guard.path_guards import _extract_target_path, _is_artifact_path

    target = _extract_target_path(tool_name, args)
    if not target or _is_artifact_path(target, payload) or not _is_test_file(target):
        return None
    file_name = Path(target).name
    try:
        raw_p = Path(os.path.expanduser(target))
        resolved_target = (raw_p if raw_p.is_absolute() else (PROJECT_ROOT / raw_p)).resolve()
    except (OSError, ValueError):
        return {"decision": "deny", "reason": "[ORCHESTRATOR GUARD] Invalid test path."}

    chunks: List[Tuple[str, str]] = []
    if tool_name in {"edit_file", "replace_file_content"}:
        if err := _validate_str_keys(args, ("Target", "TargetContent", "Replacement", "ReplacementContent"), "argument in test edit"):
            return err
        chunks.append((args.get("Target") or args.get("TargetContent") or "", args.get("Replacement") or args.get("ReplacementContent") or ""))
    elif tool_name == "multi_replace_file_content":
        raw_chunks = args.get("ReplacementChunks")
        if raw_chunks is not None and not isinstance(raw_chunks, list):
            return {"decision": "deny", "reason": "[ORCHESTRATOR GUARD] Invalid non-list 'ReplacementChunks' argument in test edit."}
        for chunk in (raw_chunks or []):
            if isinstance(chunk, dict):
                if err := _validate_str_keys(chunk, ("TargetContent", "ReplacementContent"), "in ReplacementChunks"):
                    return err
                chunks.append((chunk.get("TargetContent") or "", chunk.get("ReplacementContent") or ""))
    elif tool_name in {"write_file", "write_to_file"}:
        if err := _validate_str_keys(args, ("Content", "CodeContent"), "argument in test write"):
            return err
        new_code = args.get("Content") or args.get("CodeContent") or ""
        try:
            old_code = resolved_target.read_text(encoding="utf-8") if resolved_target.exists() else ""
        except (OSError, ValueError):
            old_code = ""
        chunks.append((old_code, (old_code + "\n" + new_code) if (args.get("Append") is True and old_code) else new_code))

    if tool_name in {"edit_file", "replace_file_content", "multi_replace_file_content"} and chunks:
        try:
            if resolved_target.is_file():
                before_text = resolved_target.read_text(encoding="utf-8")
                after_text = before_text
                all_applied = True
                for old_chunk, new_chunk in chunks:
                    if not old_chunk or old_chunk not in after_text:
                        all_applied = False
                        break
                    count = -1 if (tool_name == "edit_file" or args.get("AllowMultiple") is True) else 1
                    after_text = after_text.replace(old_chunk, new_chunk, count) if count == 1 else after_text.replace(old_chunk, new_chunk)
                if all_applied:
                    reason = _compare_test_blobs(before_text, after_text, file_name, is_diff=False)
                    return {"decision": "deny", "reason": reason} if reason else None
        except (OSError, ValueError):
            pass

    for old_text, new_text in chunks:
        if reason := _compare_test_blobs(old_text, new_text, file_name, is_diff=False):
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
    in_test_hunk = False
    test_def_indent = nested_def_indent = hunk_body_indent = None
    hunk_added_returns = hunk_removed_returns = 0

    def _reset_hunk(in_test: bool = False) -> None:
        nonlocal in_test_hunk, test_def_indent, nested_def_indent, hunk_body_indent, hunk_added_returns, hunk_removed_returns
        in_test_hunk, test_def_indent, nested_def_indent, hunk_body_indent = in_test, None, None, None
        hunk_added_returns = hunk_removed_returns = 0

    def _return_err(fname: Optional[str]) -> Optional[str]:
        if fname and _is_test_file(fname) and hunk_added_returns > hunk_removed_returns:
            return f"[ORCHESTRATOR GUARD] Diff audit detected early bare 'return' added in '{Path(fname).name}'."
        return None

    def _check_accumulated(fname: Optional[str], rem: List[str], add: List[str]) -> Optional[str]:
        if not fname or not _is_test_file(fname):
            return None
        return _return_err(fname) or _compare_test_blobs("\n".join(rem), "\n".join(add), Path(fname).name, is_diff=True)

    def _is_bare_return(body: str) -> bool:
        cleaned = _strip_comments(body, file_name=current_file or "").strip()
        return bool(cleaned and re.search(r"(?:^|[;:]\s*)return(?:\b|$)", cleaned))

    def _update_hunk_def_state(line_body: str) -> None:
        nonlocal in_test_hunk, test_def_indent, nested_def_indent, hunk_body_indent
        stripped = line_body.strip()
        if not stripped or stripped.startswith(("#", "//")):
            return
        line_indent = len(line_body) - len(line_body.lstrip(" \t"))
        if re.match(r"^(?:async\s+)?def\s+test_[A-Za-z0-9_]+\s*\(", stripped):
            in_test_hunk, test_def_indent, nested_def_indent, hunk_body_indent = True, line_indent, None, None
            return
        if not in_test_hunk:
            return
        if test_def_indent is not None and line_indent <= test_def_indent and not stripped.startswith(("@", ")")):
            in_test_hunk, test_def_indent, nested_def_indent, hunk_body_indent = False, None, None, None
            return
        if nested_def_indent is not None and line_indent <= nested_def_indent:
            nested_def_indent = None
        if nested_def_indent is None and re.match(r"^(?:async\s+)?def\s+", stripped):
            if (test_def_indent is not None and line_indent > test_def_indent) or (
                test_def_indent is None
                and re.match(r"^(?:async\s+)?def\s+(?!test_)[A-Za-z0-9_]*", stripped)
                and line_indent > 0
                and (hunk_body_indent is None or line_indent >= hunk_body_indent)
            ):
                nested_def_indent = line_indent
            else:
                in_test_hunk, test_def_indent, nested_def_indent, hunk_body_indent = False, None, None, None
        elif test_def_indent is None and nested_def_indent is None and hunk_body_indent is None and not stripped.startswith(("@", ")")):
            hunk_body_indent = line_indent

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("--- a/"):
            pending_old_file = raw_line[6:].strip()
        elif raw_line.startswith("+++ "):
            if verdict := _check_accumulated(current_file, removed_lines, added_lines):
                return verdict
            current_file = raw_line[6:].strip() if raw_line.startswith("+++ b/") else (pending_old_file if raw_line.strip() == "+++ /dev/null" else None)
            pending_old_file = None
            removed_lines, added_lines = [], []
            _reset_hunk(False)
        elif raw_line.startswith("@@"):
            if err := _return_err(current_file):
                return err
            _reset_hunk(bool(re.search(r"@@.*?\b(?:async\s+)?def\s+test_[A-Za-z0-9_]+", raw_line)))
        elif raw_line.startswith(("---", "diff --git")):
            continue
        elif raw_line.startswith(" "):
            _update_hunk_def_state(raw_line[1:])
        elif raw_line.startswith("-"):
            line_body = raw_line[1:]
            removed_lines.append(line_body)
            if in_test_hunk and nested_def_indent is None and _is_bare_return(line_body):
                hunk_removed_returns += 1
        elif raw_line.startswith("+"):
            line_body = raw_line[1:]
            added_lines.append(line_body)
            _update_hunk_def_state(line_body)
            if in_test_hunk and nested_def_indent is None and _is_bare_return(line_body):
                hunk_added_returns += 1
    return _check_accumulated(current_file, removed_lines, added_lines)


def _current_subagent_meta(payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Return metadata dictionary for the active subagent caller, if any."""
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict) and meta:
        return meta
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    if isinstance(subagents, dict) and isinstance(subagents.get(cid), dict) and subagents[cid]:
        return subagents[cid]
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir and cid:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                for sibling_state_path in resolved_art.parent.glob("*/orchestrator_state.json"):
                    if sibling_state_path.parent == resolved_art:
                        continue
                    sib_data = _read_sibling_snapshot(sibling_state_path)
                    if not isinstance(sib_data, dict) or not _workspaces_compatible(payload, sib_data):
                        continue
                    sib_subs = sib_data.get("subagents")
                    if isinstance(sib_subs, dict) and isinstance(sib_subs.get(cid), dict):
                        resolved_meta = _normalize_subagent_meta(sib_subs[cid])
                        resolved_meta.setdefault("parent_conversation_id", sib_data.get("root_conversation_id") or sibling_state_path.parent.name)
                        state["self_subagent_meta"] = resolved_meta
                        _save_state(payload, state)
                        return resolved_meta
        except (OSError, ValueError):
            pass
    return {}


def _extract_matched_cids(output_text: str) -> List[str]:
    matched: List[str] = []
    for regex in (SUBAGENT_SPAWN_RE, UUID_RE):
        for m in regex.finditer(output_text or ""):
            cid_val = m.group(1).lower()
            if cid_val not in matched:
                matched.append(cid_val)
    return matched


def _invoke_entries(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = args.get("Subagents")
    return raw if isinstance(raw, list) and raw else [args]


def _extract_expected_cohort_specs(args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[Optional[str], Optional[str], str]]]:
    """Extract (entries, [(cohort_id_or_none, role_or_none, prompt_hash), ...]) from invoke_subagent args."""
    from guard.subagent_policy import _compute_prompt_hash, _parse_cohort_header

    entries = _invoke_entries(args)
    specs: List[Tuple[Optional[str], Optional[str], str]] = []
    for item in entries:
        if not isinstance(item, dict):
            specs.append((None, None, ""))
            continue
        prompt_text = str(item.get("Prompt") or "")
        role_val = str(item.get("Role") or item.get("TypeName") or "").strip().lower() or None
        specs.append((_parse_cohort_header(prompt_text), role_val, _compute_prompt_hash(prompt_text)))
    return entries, specs


def _pending_matches_spec(
    p: Any,
    exp_cohort: Optional[str],
    exp_role: Optional[str],
    exp_prompt_hash: Optional[str] = None,
    require_prompt_hash: bool = False,
) -> bool:
    if not isinstance(p, dict):
        return False
    if exp_cohort is not None and str(p.get("cohort") or "").strip().lower() != exp_cohort:
        return False
    if exp_role is not None:
        if exp_role not in {str(p.get("role") or "").strip().lower(), str(p.get("type_name") or p.get("type") or "").strip().lower()}:
            return False
    if require_prompt_hash and (not exp_prompt_hash or str(p.get("prompt_hash") or "") != exp_prompt_hash):
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
    active = [p for p in candidates if isinstance(p, dict) and p.get("status", "active") == "active"]
    max_matches = len(entries)
    matched_by_idx: Dict[int, Dict[str, Any]] = {}
    used_ids: set = set()

    for idx, (exp_cohort, exp_role, exp_hash) in enumerate(expected_specs[:max_matches]):
        if not exp_hash:
            continue
        for p in active:
            if id(p) not in used_ids and _pending_matches_spec(p, exp_cohort, exp_role, exp_hash, require_prompt_hash=True):
                used_ids.add(id(p))
                matched_by_idx[idx] = p
                break

    for idx, (exp_cohort, exp_role, _) in enumerate(expected_specs[:max_matches]):
        if len(matched_by_idx) >= max_matches:
            break
        if idx in matched_by_idx:
            continue
        item = entries[idx] if idx < len(entries) and isinstance(entries[idx], dict) else {}
        chosen: Optional[Dict[str, Any]] = None
        if exp_cohort is not None or exp_role is not None:
            chosen = next((p for p in active if id(p) not in used_ids and _pending_matches_spec(p, exp_cohort, exp_role)), None)
        if chosen is None and allow_cohort_only_without_role and not str(item.get("Role") or "").strip():
            chosen = next(
                (p for p in active if id(p) not in used_ids and (exp_cohort is None or str(p.get("cohort") or "").strip().lower() == exp_cohort)),
                None,
            )
        if chosen is None and allow_type_fallback and (inv_type := str(item.get("TypeName") or "").strip().lower()):
            chosen = next(
                (
                    p for p in active
                    if id(p) not in used_ids
                    and (exp_cohort is None or str(p.get("cohort") or "").strip().lower() == exp_cohort)
                    and str(p.get("type_name") or p.get("type") or "").strip().lower() == inv_type
                ),
                None,
            )
        if chosen is None and allow_tail_fallback:
            chosen = next((p for p in reversed(active) if id(p) not in used_ids), None) or next(
                (p for p in reversed(candidates) if isinstance(p, dict) and id(p) not in used_ids), None
            )
        if chosen is not None:
            used_ids.add(id(chosen))
            matched_by_idx[idx] = chosen

    return [matched_by_idx[idx] for idx in range(min(max_matches, len(expected_specs))) if idx in matched_by_idx]


def _collect_invoke_cohort_candidates(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    args: Dict[str, Any],
    call_id: str,
    output_text: str,
    pre_popped_entries: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[str, str]]:
    """Resolve (cohort_id, candidate_key) pairs for an invoke_subagent call."""
    matched_cids = _extract_matched_cids(output_text)
    matched_cids_set = set(matched_cids)

    def _cand_pairs(records: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        multi = len(records) > 1
        out: List[Tuple[str, str]] = []
        for idx, p in enumerate(records):
            if p.get("mode") == "best_of_n" and (c_id := p.get("cohort")):
                cand_key = (
                    p.get("candidate_id")
                    or (matched_cids[idx] if idx < len(matched_cids) else None)
                    or p.get("call_sig")
                    or (f"{call_id}:{idx}" if call_id and multi else call_id)
                    or ""
                )
                out.append((str(c_id), str(cand_key)))
        return out

    if pre_popped_entries and (pairs := _cand_pairs(pre_popped_entries)):
        return pairs

    subagent_results = [
        (str(meta["cohort"]), str(meta.get("candidate_id") or sub_cid.lower()))
        for sub_cid, meta in (state.get("subagents") or {}).items()
        if isinstance(meta, dict)
        and (sub_cid.lower() in matched_cids_set or (call_id and meta.get("call_id") == call_id))
        and meta.get("mode") == "best_of_n"
        and meta.get("cohort")
    ]
    if subagent_results:
        return subagent_results

    pending_all = [
        p for p in (state.get("pending_spawns") or [])
        if isinstance(p, dict) and p.get("mode") == "best_of_n" and p.get("cohort")
    ]
    if call_id:
        pending_matches = [p for p in pending_all if p.get("call_id") == call_id]
    else:
        entries, expected_specs = _extract_expected_cohort_specs(args)
        pending_matches = _match_pending_spawns_two_pass(
            [p for p in pending_all if p.get("status", "active") == "active"],
            entries,
            expected_specs,
            allow_cohort_only_without_role=True,
        )
    if pending_matches:
        return _cand_pairs(pending_matches)

    from guard.subagent_policy import _parse_cohort_header, _parse_mode_header

    entries = _invoke_entries(args)
    inv_key = _current_invocation_key(payload, state)
    multi = len(entries) > 1
    entry_results: List[Tuple[str, str]] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        prompt_text = str(item.get("Prompt") or "")
        cohort_id = _parse_cohort_header(prompt_text)
        mode = _parse_mode_header(prompt_text) or ("best_of_n" if cohort_id else None)
        if mode == "best_of_n":
            cand_key = (matched_cids[idx] if idx < len(matched_cids) else None) or (f"{call_id}:{idx}" if call_id and multi else call_id) or ""
            entry_results.append((cohort_id or f"best-of-n-{inv_key}", str(cand_key)))
    return entry_results


def _apply_cohort_invoke_outcome(
    state: Dict[str, Any], cb: Dict[str, Any], cohort_candidates: List[Tuple[str, str]], outcome: str, error_summary: str = ""
) -> None:
    """Update cohort succeeded/failed tracking and trip global breaker on 0/N (N>=2) or N<=1 failure."""
    for cohort_id, cand_key in cohort_candidates:
        _apply_cohort_outcome(state, cand_key, cohort_id, outcome, error_summary=error_summary, create_if_missing=True, cb=cb)


def _propagate_receipt_to_parent_brain(
    payload: Dict[str, Any],
    dirty: bool,
    verified: bool,
    last_mutated_file: Optional[str] = None,
    last_test_cmd: Optional[str] = None,
    *,
    bump_epoch: bool = True,
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
        parent_cid = str((current_state.get("self_subagent_meta") or {}).get("parent_conversation_id") or "").strip().lower()
        cid = str(payload.get("conversationId") or "").strip().lower()
        for sibling_state_path in resolved_art.parent.glob("*/orchestrator_state.json"):
            if sibling_state_path.parent == resolved_art or _has_other_locked_path(sibling_state_path):
                continue
            with state_transaction(sibling_state_path):
                sib_data = _load_state(sibling_state_path)
                sib_subagents = sib_data.get("subagents") or {}
                root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
                dir_name = sibling_state_path.parent.name.lower()
                if (bool(cid and cid in sib_subagents) or bool(parent_cid and (root_cid == parent_cid or dir_name == parent_cid))) and _workspaces_compatible(payload, sib_data):
                    parent_receipt = _get_test_receipt(sib_data)
                    parent_receipt["dirty"] = False if verified else (bool(parent_receipt.get("dirty", False)) or bool(dirty))
                    parent_receipt["verified"] = verified
                    if dirty and bump_epoch:
                        parent_receipt["mutation_epoch"] = parent_receipt.get("mutation_epoch", 0) + 1
                    if last_mutated_file is not None:
                        parent_receipt["last_mutated_file"] = last_mutated_file
                    if last_test_cmd is not None:
                        parent_receipt["last_test_cmd"] = last_test_cmd
                    _save_state(sibling_state_path, sib_data)
    except (OSError, ValueError):
        pass


def _sig_matches_call_id(sig: Any, call_id: str) -> bool:
    if not isinstance(sig, str) or not call_id:
        return False
    try:
        parsed = json.loads(sig)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, list) or len(parsed) <= 3:
        return False
    sig_cid = str(parsed[3] or "")
    return sig_cid == call_id or sig_cid.startswith(f"{call_id}:")


def _pop_failed_invoke_spawns(state: Dict[str, Any], args: Dict[str, Any], call_id: str) -> List[Dict[str, Any]]:
    raw_pending = state.get("pending_spawns")
    pending = raw_pending if isinstance(raw_pending, list) else []
    if call_id and any(isinstance(p, dict) and p.get("call_id") == call_id for p in pending):
        failed = [dict(p) for p in pending if isinstance(p, dict) and p.get("call_id") == call_id and p.get("mode") == "best_of_n" and p.get("cohort")]
        state["pending_spawns"] = [p for p in pending if isinstance(p, dict) and p.get("call_id") != call_id]
        if isinstance(recorded := state.get("recorded_spawn_calls"), list):
            state["recorded_spawn_calls"] = [sig for sig in recorded if not _sig_matches_call_id(sig, call_id)]
        return failed
    entries, expected_specs = _extract_expected_cohort_specs(args)
    matched_pops = _match_pending_spawns_two_pass(pending, entries, expected_specs, allow_tail_fallback=True)
    if not matched_pops:
        return []
    matched_ids = {id(p) for p in matched_pops}
    state["pending_spawns"] = [p for p in pending if id(p) not in matched_ids]
    return [dict(p) for p in matched_pops if p.get("mode") == "best_of_n" and p.get("cohort")]


def _extract_tool_error(payload: Dict[str, Any], tool_call: Dict[str, Any], tool_name: str, output_text: str) -> str:
    error_text = str(payload.get("error") or "").strip()
    if not error_text and tool_name in {"run_command", "run_workflow", "swarm"}:
        exit_code = next(
            (v for v in (payload.get("exitCode"), payload.get("returnCode"), tool_call.get("exitCode")) if v is not None),
            None,
        )
        if isinstance(exit_code, int) and exit_code != 0:
            error_text = f"Command exited with code {exit_code}"
        elif m_exit := re.search(r"(?m)^Command exited with code ([1-9][0-9]*)\b", output_text):
            error_text = f"Command exited with code {m_exit.group(1)}"
    if not error_text and tool_name == "run_workflow" and "Traceback (most recent call last):" in output_text:
        return "Workflow script raised an exception"
    return error_text


def _swarm_target_names(swarm_workers: Dict[str, Any], args: Dict[str, Any], call_id: str) -> List[str]:
    if call_id:
        return [w for w, meta in swarm_workers.items() if isinstance(meta, dict) and meta.get("call_id") == call_id]
    from guard.subagent_policy import _expand_swarm_worker_names

    return [w for w in _expand_swarm_worker_names(args) if w in swarm_workers]


def _rollback_swarm_call(state: Dict[str, Any], args: Dict[str, Any], call_id: str) -> None:
    action = str(args.get("action") or "").strip().lower()
    swarm_workers = state.get("swarm_workers")
    if not isinstance(swarm_workers, dict):
        return
    if action == "add":
        rolled_names: List[str] = []
        rolled_models: Dict[str, List[str]] = {}
        for wname in _swarm_target_names(swarm_workers, args, call_id):
            removed = swarm_workers.pop(wname, None)
            prev_worker = removed.get("_prev_worker") if isinstance(removed, dict) else None
            if isinstance(prev_worker, dict):
                restored = dict(prev_worker)
                restored.pop("_prev_worker", None)
                swarm_workers[wname] = restored
            else:
                rolled_names.append(wname)
            if isinstance(removed, dict) and removed.get("council_id") and removed.get("model"):
                rolled_models.setdefault(str(removed["council_id"]), []).append(str(removed["model"]))
        if rolled_names or rolled_models:
            for c_id, c_entry in (state.get("councils") or {}).items():
                if not isinstance(c_entry, dict):
                    continue
                if rolled_names:
                    for key in ("proposers", "judges", "synthesizers"):
                        if isinstance(c_entry.get(key), list):
                            c_entry[key] = [item for item in c_entry[key] if item not in rolled_names]
                if c_id in rolled_models and isinstance(c_entry.get("models"), list):
                    still_used = {
                        str(src.get("model"))
                        for bucket in (swarm_workers, state.get("subagents") or {})
                        for src in bucket.values()
                        if isinstance(src, dict) and src.get("council_id") == c_id and src.get("model")
                    }
                    to_drop = set(rolled_models[c_id]) - still_used
                    if to_drop:
                        c_entry["models"] = [m for m in c_entry["models"] if m not in to_drop]
    elif action == "kill":
        wname = str(args.get("name") or "").strip()
        if wname and isinstance(swarm_workers.get(wname), dict):
            meta = swarm_workers[wname]
            was_active = meta.pop("pre_kill_active", True)
            meta["status"] = "active" if was_active else "killed"
            meta["active_worktree"] = bool(was_active and str(meta.get("workspace") or "inherit").strip().lower() in ISOLATED_WORKSPACES)


def _update_subagents_on_invoke(
    state: Dict[str, Any], args: Dict[str, Any], call_id: str, output_text: str, failed_status: bool, non_final: bool
) -> None:
    matched_cids = set(_extract_matched_cids(output_text))
    for sub_cid, sub_meta in (state.get("subagents") or {}).items():
        if isinstance(sub_meta, dict) and (sub_cid.lower() in matched_cids or (call_id and sub_meta.get("call_id") == call_id)):
            if failed_status:
                sub_meta["status"] = "failed"
            elif non_final:
                sub_meta["status"] = "needs_replan" if re.search(r"(?m)^\s*STATUS:\s*NEEDS_REPLAN\b", output_text) else "checkpoint"
            else:
                sub_meta["status"] = "completed"

    pending_list = state.get("pending_spawns") or []
    if call_id:
        matched_ids = {id(p) for p in pending_list if isinstance(p, dict) and p.get("status", "active") == "active" and p.get("call_id") == call_id}
    else:
        entries, expected_specs = _extract_expected_cohort_specs(args)
        matched_ids = {id(p) for p in _match_pending_spawns_two_pass(pending_list, entries, expected_specs, allow_type_fallback=True)}
    if matched_ids:
        updated_pending: List[Any] = []
        for p in pending_list:
            if id(p) in matched_ids and isinstance(p, dict):
                if p.get("active_worktree") or p.get("active_branch"):
                    p["status"] = "completed"
                    updated_pending.append(p)
            else:
                updated_pending.append(p)
        state["pending_spawns"] = updated_pending


def _record_cb_failure(cb: Dict[str, Any], tool_name: str, call_id: str, error_text: str) -> None:
    seen_failures: List[str] = cb.setdefault("seen_failures", [])
    dedup_id = f"{tool_name}:{call_id}:{error_text}"
    if call_id and dedup_id in seen_failures:
        return
    if call_id:
        seen_failures.append(dedup_id)
    cb["failure_streak"] = cb.get("failure_streak", 0) + 1
    cb["last_error"] = error_text
    if cb["failure_streak"] >= MAX_CONSECUTIVE_FAILURES:
        cb["tripped"] = True


def _has_unsafe_test_cwd(cmd: str, cmd_cwd: str, payload: Dict[str, Any], *, parsed: Optional[Any] = None) -> bool:
    """Return True if cmd_cwd or any cd target in cmd enters /tmp, staging, or guard plugin dirs."""
    from guard.path_guards import _is_guard_plugin_cwd, _is_staging_or_tmp_path, _resolve_cmd_target
    from guard.shell_tokens import _advance_effective_cwd, parse_shell_command
    cur = cmd_cwd
    if _is_guard_plugin_cwd(cur) or _is_staging_or_tmp_path(_resolve_cmd_target(".", cur), payload):
        return True
    p = parsed if parsed is not None else parse_shell_command(cmd)
    if p.has_subshell:
        return True
    for sub in p.subcmds:
        cur, is_cd, cd_unv = _advance_effective_cwd(sub.raw, cur, None, payload, parsed_sub=sub)
        if is_cd and (cd_unv or _is_guard_plugin_cwd(cur) or _is_staging_or_tmp_path(_resolve_cmd_target(".", cur), payload)):
            return True
    return False


def handle_post_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    _preclaim_subagent_meta(payload)
    deferred_propagations: List[Dict[str, Any]] = []
    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}
        raw_tc = payload.get("toolCall")
        tool_call = raw_tc if isinstance(raw_tc, dict) else {}
        tool_name = tool_call.get("name") if isinstance(tool_call.get("name"), str) else ""
        args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
        call_id = _extract_call_id(payload)

        if tool_name == "exit_plan_mode" and payload.get("error"):
            state.update({"plan_execute_phase": False, "plan_execute_consumed": False, "plan_execute_user_input": None, "plan_execute_turn_finished": False, "plan_allowed_files": None, "active_eval_cmd": None})

        failed_cohort_entries = _pop_failed_invoke_spawns(state, args, call_id) if (tool_name == "invoke_subagent" and payload.get("error")) else []
        if not (tool_name == "invoke_subagent" and payload.get("error")):
            _sync_subagents_from_transcript(payload, state)

        cb, receipt = _get_circuit_breaker(state, payload), _get_test_receipt(state, payload)
        if tool_name in {"run_command", "invoke_subagent", "run_workflow", "swarm"}:
            raw_cmd = args.get("CommandLine")
            cmd = raw_cmd.strip() if isinstance(raw_cmd, str) else ""
            raw_cwd = args.get("Cwd")
            cmd_cwd = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else str(PROJECT_ROOT)
            cmd_parsed = None
            is_verification_cmd = False
            if tool_name == "run_command":
                from guard.shell_tokens import parse_shell_command
                cmd_parsed = parse_shell_command(cmd)
                if not _has_unsafe_test_cwd(cmd, cmd_cwd, payload, parsed=cmd_parsed):
                    from guard.ast_guard import _is_test_command
                    from guard.shell_tokens import _is_declared_eval_command
                    is_verification_cmd = _is_test_command(cmd, cur_dir=cmd_cwd, payload=payload, parsed=cmd_parsed) or _is_declared_eval_command(cmd, state, payload, parsed=cmd_parsed)
            output_text = str(payload.get("output") or payload.get("result") or tool_call.get("result") or tool_call.get("output") or "")
            error_text = _extract_tool_error(payload, tool_call, tool_name, output_text)
            caller_mode = _current_subagent_meta(payload, state).get("mode")
            swarm_action = str(args.get("action") or "").strip().lower() if tool_name == "swarm" else ""

            if error_text:
                if tool_name == "swarm":
                    _rollback_swarm_call(state, args, call_id)
                cohort_cands = _collect_invoke_cohort_candidates(payload, state, args, call_id, output_text, pre_popped_entries=failed_cohort_entries) if tool_name == "invoke_subagent" else []
                if cohort_cands:
                    _apply_cohort_invoke_outcome(state, cb, cohort_cands, "failed", error_summary=error_text)
                elif not (tool_name in {"run_command", "run_workflow"} and _is_sandbox_boundary_error(payload, error_text, output_text)):
                    _record_cb_failure(cb, tool_name, call_id, error_text)
                if is_verification_cmd:
                    receipt["verified"], receipt["last_test_cmd"] = False, cmd
                    if not _is_root_orchestrator(payload, state) and caller_mode != "best_of_n":
                        deferred_propagations.append({"dirty": bool(receipt.get("dirty", False)), "verified": False, "last_test_cmd": cmd, "bump_epoch": False})
            else:
                if swarm_action == "add" and isinstance(sw := state.get("swarm_workers"), dict):
                    for wname in _swarm_target_names(sw, args, call_id):
                        if isinstance(sw.get(wname), dict):
                            sw[wname].pop("_prev_worker", None)
                subagent_failed_status = bool(tool_name == "invoke_subagent" and re.search(r"(?m)^\s*STATUS:\s*(?:FAILED|BLOCKED)\b", output_text))
                subagent_non_final = bool(tool_name == "invoke_subagent" and re.search(r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED|NEEDS_REPLAN|CHECKPOINT)\b", output_text))
                invoked_types = [str((it or {}).get("TypeName") or "").strip().lower() for it in _invoke_entries(args) if isinstance(it, dict)]
                is_async_spawn_ack = bool(
                    tool_name == "invoke_subagent"
                    and re.search(r"Created the following subagents?\b", output_text)
                    and not re.search(r"(?m)^\s*STATUS:\s*COMPLETED\b", output_text)
                )
                cohort_cands = _collect_invoke_cohort_candidates(payload, state, args, call_id, output_text) if tool_name == "invoke_subagent" else []
                is_non_git_cmd = False
                if tool_name == "run_command" and not is_verification_cmd:
                    from guard.shell_tokens import _is_git_only_command
                    is_non_git_cmd = not _is_git_only_command(cmd, parsed=cmd_parsed)
                if cohort_cands and subagent_failed_status:
                    _apply_cohort_invoke_outcome(state, cb, cohort_cands, "failed", error_summary="Best-of-N candidate reported STATUS: FAILED/BLOCKED")
                elif cohort_cands and not subagent_non_final and not is_async_spawn_ack:
                    _apply_cohort_invoke_outcome(state, cb, cohort_cands, "succeeded")
                elif (
                    (tool_name == "invoke_subagent" and not subagent_non_final and not is_async_spawn_ack)
                    or is_verification_cmd
                    or is_non_git_cmd
                    or tool_name == "run_workflow"
                    or swarm_action == "add"
                ):
                    cb["failure_streak"], cb["tripped"] = 0, False

                if tool_name == "invoke_subagent" and not is_async_spawn_ack:
                    _update_subagents_on_invoke(state, args, call_id, output_text, subagent_failed_status, subagent_non_final)

                is_verification_subagent = (
                    tool_name == "invoke_subagent" and not subagent_non_final and not is_async_spawn_ack
                    and not cohort_cands and any(t not in READONLY_NON_VERIFICATION_TYPES for t in invoked_types)
                )
                if is_verification_cmd or (is_verification_subagent and receipt.get("dirty")):
                    receipt["dirty"], receipt["verified"], receipt["last_test_cmd"] = False, True, (cmd or "invoke_subagent")
                    if is_verification_cmd and not _is_root_orchestrator(payload, state) and caller_mode != "best_of_n":
                        deferred_propagations.append({"dirty": False, "verified": True, "last_test_cmd": cmd})
        _save_state(payload, state)

    for prop_kwargs in deferred_propagations:
        _propagate_receipt_to_parent_brain(payload, **prop_kwargs)
    return {}


def validate_subagent_report(text: str, is_scout: bool = False, is_readonly_or_scratch: bool = False) -> Optional[str]:
    """Validate that a subagent's final output matches the REPORT FORMAT."""
    stripped = (text or "").strip()
    if not stripped:
        return "Subagent response is empty."
    if len(stripped) > MAX_SUBAGENT_REPORT_CHARS:
        return f"Subagent report length ({len(stripped)} chars) exceeds maximum {MAX_SUBAGENT_REPORT_CHARS} chars."
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]

    is_compact = (
        is_scout
        and len(lines) >= 2
        and bool(SCOUT_COMPACT_LINE_PATTERNS[0].match(lines[0]))
        and bool(SCOUT_COMPACT_LINE_PATTERNS[1].match(lines[1]))
    )
    header_count = 2 if is_compact else 4
    max_lines = header_count + MAX_REPORT_BULLET_LINES
    if len(lines) < header_count or len(lines) > max_lines:
        return f"Expected {header_count} header lines plus at most {MAX_REPORT_BULLET_LINES} bullet lines ({header_count}-{max_lines} lines total), got {len(lines)}."
    if not is_compact:
        for idx, (pattern, prefix) in enumerate(zip(REPORT_LINE_PATTERNS, ("STATUS:", "FILES_CHANGED:", "VERIFICATION:", "NOTES:")), start=1):
            if not pattern.match(lines[idx - 1]):
                return f"Line {idx} must match '{prefix} <value>'."

    bullet_lines = lines[header_count:]
    for b_idx, b_line in enumerate(bullet_lines, start=header_count + 1):
        if not REPORT_BULLET_RE.match(b_line):
            return f"Line {b_idx} after NOTES must be a compact bullet starting with '- ' or '* '."

    status_val = lines[0].split(":", 1)[1].strip().upper().split()[0]
    notes_body = lines[header_count - 1].split(":", 1)[1].strip()
    combined_notes = " ".join([notes_body] + [b[1:].strip() for b in bullet_lines]).strip()
    non_uri_text = REPORT_LOG_URI_RE.sub("", combined_notes).strip(" -—:,.;()")

    if not is_compact and status_val == "NEEDS_REPLAN":
        if len(non_uri_text.split()) < 3:
            return "When STATUS is NEEDS_REPLAN, NOTES must include an inline explanation (at least 3 words)."
    elif status_val in {"BLOCKED", "FAILED"}:
        if not is_compact and not is_readonly_or_scratch and not REPORT_LOG_URI_RE.search(combined_notes):
            return f"When STATUS is {status_val}, NOTES must include a 'file://.../*.log' scratch log URI."
        if len(non_uri_text.split()) < 3:
            suffix = "" if is_compact else ("." if is_readonly_or_scratch else " in addition to the file:// scratch log URI.")
            return f"When STATUS is {status_val}, NOTES must include an inline error summary (at least 3 words){suffix}"
    return None


def _extract_last_model_response(payload_or_path: Any) -> Optional[str]:
    """Extract the final model response text from payload or transcriptPath if available."""
    if isinstance(payload_or_path, dict):
        if payload_or_path.get("lastAssistantMessage") is not None:
            return str(payload_or_path["lastAssistantMessage"])
        transcript_path = payload_or_path.get("transcriptPath")
    else:
        transcript_path = str(payload_or_path)
    if not isinstance(transcript_path, str) or not transcript_path:
        return None
    try:
        if not os.path.exists(transcript_path):
            return None
        file_size = os.path.getsize(transcript_path)
        offset = max(0, file_size - 65536)
        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
        if offset > 0 and len(lines) > 1:
            lines = lines[1:]
        for raw_line in reversed(lines):
            if not (raw_line := raw_line.strip()):
                continue
            try:
                step = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(step, dict) and step.get("type") == "PLANNER_RESPONSE" and not step.get("tool_calls"):
                if isinstance(content := step.get("content"), str) and content.strip():
                    return content
    except (OSError, ValueError):
        return None
    return None


def _get_workspace_git_diff(payload: Dict[str, Any]) -> str:
    """Return unified git diff from payload or by running git diff HEAD in workspace."""
    if payload.get("gitDiff"):
        return str(payload["gitDiff"])
    import subprocess

    for diff_cmd in (["git", "diff", "HEAD", "-U0"], ["git", "diff", "-U0"]):
        try:
            proc = subprocess.run(diff_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=False)
            if proc.returncode == 0:
                return proc.stdout or ""
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def _resolve_subagent_stop_cohort(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    cid: str,
    cb_tripped: bool,
    force_failed: bool,
    abnormal_reason: str,
) -> Tuple[bool, Optional[Tuple[str, str, str, str]]]:
    last_msg = _extract_last_model_response(payload) or ""
    status_m = re.search(r"(?m)^\s*STATUS:\s*([A-Za-z_]+)\b", last_msg)
    report_status = status_m.group(1).upper() if status_m else ""
    if not force_failed and not cb_tripped and report_status in {"CHECKPOINT", "NEEDS_REPLAN"}:
        return True, None
    sub_meta = _current_subagent_meta(payload, state)
    if sub_meta.get("mode") != "best_of_n" or not sub_meta.get("cohort"):
        return False, None
    if force_failed or report_status in {"FAILED", "BLOCKED"} or (cb_tripped and report_status != "COMPLETED"):
        outcome = "failed"
        cb_root = _get_circuit_breaker(state, payload)
        err_summary = str(cb_root.get("last_error") or "").strip() or (
            f"Subagent {cid} terminated abnormally ({abnormal_reason})"
            if force_failed and abnormal_reason
            else f"Subagent {cid} finished with STATUS: {report_status or 'BLOCKED'}"
        )
    else:
        outcome, err_summary = "succeeded", ""
    sub_reg = (state.get("subagents") or {}).get(cid)
    cand_key = str(sub_meta.get("candidate_id") or (sub_reg.get("candidate_id") if isinstance(sub_reg, dict) else None) or cid)
    return False, (cand_key, str(sub_meta["cohort"]), outcome, err_summary)


def _apply_local_subagent_stop_update(
    payload: Dict[str, Any], state: Dict[str, Any], cb_tripped: bool = False, *, force_failed: bool = False, abnormal_reason: str = ""
) -> Tuple[Optional[str], bool]:
    cid = str(payload.get("conversationId") or "").strip().lower()
    if not cid:
        return None, False
    skipped, cohort_info = _resolve_subagent_stop_cohort(payload, state, cid, cb_tripped, force_failed, abnormal_reason)
    if skipped:
        return None, True
    local_changed = False
    for entry in ((state.get("subagents") or {}).get(cid), state.get("self_subagent_meta")):
        if isinstance(entry, dict) and entry.get("status") != "completed":
            entry["status"] = "completed"
            local_changed = True
    cand_key: Optional[str] = None
    if cohort_info is not None:
        cand_key, cohort_id, outcome, err_summary = cohort_info
        if _apply_cohort_outcome(state, cand_key, cohort_id, outcome, err_summary):
            local_changed = True
    if local_changed:
        _save_state(payload, state)
    return cand_key, False


def _finalize_subagent_stop_sync(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    cb_tripped: bool = False,
    *,
    force_failed: bool = False,
    abnormal_reason: str = "",
    update_local: bool = True,
) -> None:
    """Sync subagent completion and Best-of-N cohort outcome into parent brain state."""
    cid = str(payload.get("conversationId") or "").strip().lower()
    if not cid:
        return
    skipped, cohort_info = _resolve_subagent_stop_cohort(payload, state, cid, cb_tripped, force_failed, abnormal_reason)
    if skipped:
        return
    _mark_subagent_status_in_parent_brain(payload, state, cid, "completed", update_local=update_local)
    if cohort_info is not None:
        cand_key, cohort_id, outcome, err_summary = cohort_info
        _record_cohort_outcome_in_parent_brain(
            payload, state, cand_key, cohort_id, outcome, error_summary=err_summary, update_local=update_local
        )


def _record_stop_reminder(state: Dict[str, Any], payload: Dict[str, Any], prefix: str = "") -> bool:
    cid, exec_num = payload.get("conversationId", ""), payload.get("executionNum", 0)
    exec_key = f"{prefix}:{cid}:{exec_num}" if prefix else f"{cid}:{exec_num}"
    reminded = state.setdefault("stop_reminded_executions", [])
    if exec_key in reminded:
        return False
    reminded.append(exec_key)
    _save_state(payload, state)
    return True


def _has_active_subagents(state: Dict[str, Any]) -> bool:
    return (
        any(isinstance(p, dict) and p.get("status", "active") == "active" for p in (state.get("pending_spawns") or []))
        or any(isinstance(s, dict) and s.get("status") == "active" for s in (state.get("subagents") or {}).values())
        or any(isinstance(w, dict) and w.get("status") == "active" and w.get("active_worktree") for w in (state.get("swarm_workers") or {}).values())
    )


def handle_stop(payload: Dict[str, Any]) -> Dict[str, Any]:
    _preclaim_subagent_meta(payload)
    deferred_stop_params: Optional[Dict[str, Any]] = None
    state_snapshot: Dict[str, Any] = {}

    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}
        raw_term = payload.get("terminationReason")
        termination_reason = raw_term if isinstance(raw_term, str) else ""
        if termination_reason in {"USER_CANCELED", "ERROR", "MAX_INVOCATIONS", "MAX_FORCED_INVOCATIONS", "MAX_TOKEN_BUDGET_EXCEEDED"}:
            if _is_known_subagent(payload, state) and str(payload.get("conversationId") or "").strip():
                _apply_local_subagent_stop_update(payload, state, cb_tripped=True, force_failed=True, abnormal_reason=termination_reason)
                deferred_stop_params = {"cb_tripped": True, "force_failed": True, "abnormal_reason": termination_reason}
                state_snapshot = dict(state)
            if deferred_stop_params is None:
                return {"decision": "allow"}
        elif not payload.get("fullyIdle", True):
            return {"decision": "allow"}
        else:
            synced_transcript = _sync_subagents_from_transcript(payload, state)
            state_saved = False

            active_branches = [cid for cid, m in state.get("subagents", {}).items() if isinstance(m, dict) and (m.get("active_worktree") or m.get("active_branch"))]
            active_pending = [p.get("role", "implementer") for p in state.get("pending_spawns", []) if isinstance(p, dict) and (p.get("active_worktree") or p.get("active_branch"))]
            active_swarm = [w for w, m in (state.get("swarm_workers") or {}).items() if isinstance(m, dict) and m.get("status") == "active" and m.get("active_worktree")]
            if (active_branches or active_pending or active_swarm) and _record_stop_reminder(state, payload):
                if active_branches or active_pending:
                    return {"decision": "continue", "reason": f"[ORCHESTRATOR GUARD] Uncleaned branched subagent worktrees remain ({', '.join(active_branches or active_pending)}). Verify/merge branch changes and call manage_subagents(Action='kill') before finishing."}
                return {"decision": "continue", "reason": f"[ORCHESTRATOR GUARD] Uncleaned isolated swarm worktrees remain ({', '.join(active_swarm)}). Merge/discard changes and call swarm(action='kill') before finishing."}

            if not _is_root_orchestrator(payload, state):
                sent_key = f"sent:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
                if sent_key not in state.setdefault("stop_reminded_executions", []) and (last_msg := _extract_last_model_response(payload)) is not None:
                    report_err = validate_subagent_report(last_msg, is_scout=_is_scout_subagent(payload, state), is_readonly_or_scratch=(_get_subagent_type(payload, state) in READONLY_NON_VERIFICATION_TYPES))
                    if report_err and _record_stop_reminder(state, payload, "report"):
                        return {
                            "decision": "continue",
                            "reason": (
                                f"[ORCHESTRATOR GUARD] Subagent context-separation violation: {report_err} "
                                "Return ONLY the exact 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                                "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                                "file:// .log URI when BLOCKED or FAILED; scouts may use compact 2-header STATUS + NOTES)."
                            ),
                        }

            if _get_circuit_breaker(state, payload).get("tripped"):
                if _is_known_subagent(payload, state):
                    if not _apply_local_subagent_stop_update(payload, state, cb_tripped=True)[1]:
                        state_saved, deferred_stop_params, state_snapshot = True, {"cb_tripped": True}, dict(state)
                else:
                    if synced_transcript:
                        _save_state(payload, state)
                    return {"decision": "allow"}
            else:
                inv_key = _current_invocation_key(payload, state)
                mutated_in_turn = bool(state.get("direct_budget", {}).get(inv_key, {}).get("mutated_files"))
                receipt = _get_test_receipt(state, payload)
                cur_epoch, audited_epoch = int(receipt.get("mutation_epoch", 0) or 0), int(receipt.get("audited_epoch", 0) or 0)
                should_audit = bool(payload.get("gitDiff") or receipt.get("dirty") or mutated_in_turn or (cur_epoch > audited_epoch))
                if should_audit and (diff_text := _get_workspace_git_diff(payload)):
                    if (tamper_finding := audit_diff_text(diff_text)) and _record_stop_reminder(state, payload, "tamper"):
                        return {"decision": "continue", "reason": tamper_finding}
                if should_audit and cur_epoch > audited_epoch:
                    receipt["audited_epoch"] = cur_epoch
                    _save_state(payload, state)
                    state_saved = True
                if receipt.get("dirty") and not receipt.get("verified") and _record_stop_reminder(state, payload, "receipt"):
                    last_file = Path(receipt.get("last_mutated_file") or "workspace").name
                    return {"decision": "continue", "reason": f"[ORCHESTRATOR GUARD] Empirical test receipt missing: code file '{last_file}' was modified without a subsequent passing test run recorded in PostToolUse. Run the test suite to verify changes before finishing."}

                if _is_known_subagent(payload, state):
                    if not _apply_local_subagent_stop_update(payload, state, cb_tripped=False)[1]:
                        state_saved, deferred_stop_params, state_snapshot = True, {"cb_tripped": False}, dict(state)
                elif payload.get("fullyIdle") is True and state.get("plan_execute_phase") and not _has_active_subagents(state):
                    state["plan_execute_turn_finished"] = True
                    _save_state(payload, state)
                    state_saved = True

            if synced_transcript and not state_saved:
                _save_state(payload, state)

    if deferred_stop_params is not None:
        _finalize_subagent_stop_sync(payload, state_snapshot, update_local=False, **deferred_stop_params)
    return {"decision": "allow"}

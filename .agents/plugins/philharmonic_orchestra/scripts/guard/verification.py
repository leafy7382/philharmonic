"""Test integrity checking, git diff auditing, empirical test receipts, circuit breaker, and stop hook."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ASSERTION_RE,
    COMMENT_LINE_RE,
    DOC_EXTENSIONS,
    MAX_CONSECUTIVE_FAILURES,
    MAX_REPORT_BULLET_LINES,
    MAX_SUBAGENT_REPORT_CHARS,
    PROJECT_ROOT,
    REPORT_BULLET_RE,
    REPORT_LINE_PATTERNS,
    REPORT_LOG_URI_RE,
    SCOUT_COMPACT_LINE_PATTERNS,
    TAMPER_SKIP_RE,
    TEST_DEF_RE,
    TEST_FILE_RE,
    TRIVIAL_ASSERT_RE,
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
    _sync_subagents_from_transcript,
    _workspaces_compatible,
)
from guard.policies import (
    _extract_target_path,
    _is_artifact_or_worktree_path,
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
        receipt_root.clear()
        receipt_root.update(
            {
                "dirty": False,
                "verified": False,
                "last_mutated_file": "",
                "last_test_cmd": "",
            }
        )
    return receipt_root


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
            state["plan_allowed_files"] = None

        if tool_name == "invoke_subagent" and payload.get("error"):
            pending = state.get("pending_spawns", [])
            recorded = state.get("recorded_spawn_calls", [])
            if call_id and any(p.get("call_id") == call_id for p in pending):
                state["pending_spawns"] = [p for p in pending if p.get("call_id") != call_id]
                if isinstance(recorded, list):
                    state["recorded_spawn_calls"] = [
                        sig for sig in recorded
                        if not (isinstance(sig, str) and call_id in sig)
                    ]
            else:
                pop_count = (
                    len(args["Subagents"])
                    if isinstance(args.get("Subagents"), list) and args["Subagents"]
                    else 1
                )
                for _ in range(min(pop_count, len(pending))):
                    if pending:
                        pending.pop()
        else:
            _sync_subagents_from_transcript(payload, state)

        cb = _get_circuit_breaker(state, payload)
        receipt = _get_test_receipt(state, payload)
        error_text = str(payload.get("error") or "").strip()

        if tool_name in {"run_command", "invoke_subagent"}:
            cmd = (args.get("CommandLine") or "").strip()
            is_test_cmd = tool_name == "run_command" and _is_test_command(cmd)
            output_text = str(
                payload.get("output")
                or payload.get("result")
                or tool_call.get("result")
                or tool_call.get("output")
                or ""
            )
            if not error_text and tool_name == "run_command":
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

            if error_text:
                is_sandbox_err = (
                    tool_name == "run_command"
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
                if is_test_cmd:
                    receipt["verified"] = False
                    receipt["last_test_cmd"] = cmd
                    if not _is_root_orchestrator(payload, state):
                        _propagate_receipt_to_parent_brain(
                            payload,
                            dirty=bool(receipt.get("dirty", True)),
                            verified=False,
                            last_test_cmd=cmd,
                        )
            else:
                if (
                    tool_name == "invoke_subagent"
                    or is_test_cmd
                    or (tool_name == "run_command" and not _is_git_only_command(cmd))
                ):
                    cb["failure_streak"] = 0
                    cb["tripped"] = False
                subagent_non_final = bool(
                    tool_name == "invoke_subagent"
                    and re.search(r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED|NEEDS_REPLAN|CHECKPOINT)\b", output_text)
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
                is_verification_subagent = (
                    tool_name == "invoke_subagent"
                    and not subagent_non_final
                    and not is_async_spawn_ack
                    and any(
                        t not in {"scout", "research", "researcher"} for t in invoked_types
                    )
                )
                if is_test_cmd or (is_verification_subagent and receipt.get("dirty")):
                    receipt["dirty"] = False
                    receipt["verified"] = True
                    receipt["last_test_cmd"] = cmd or "invoke_subagent"
                    if is_test_cmd and not _is_root_orchestrator(payload, state):
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
            if meta.get("active_branch")
        ]
        active_pending = [
            p.get("role", "implementer")
            for p in state.get("pending_spawns", [])
            if p.get("active_branch")
        ]

        if active_branches or active_pending:
            exec_key = f"{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
            reminded = state.setdefault("stop_reminded_executions", [])
            if exec_key not in reminded:
                reminded.append(exec_key)
                _save_state(payload, state)
                targets = ", ".join(active_branches or active_pending)
                return {
                    "decision": "continue",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Uncleaned branched subagent worktrees remain ({targets}). "
                        "Verify/merge branch changes and call manage_subagents(Action='kill') before finishing."
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

        return {"decision": "allow"}

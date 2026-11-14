"""Subagent identity tracking, transcript pairing, role resolution, and context reminder injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from guard.constants import (
    ALLOWED_SYSTEM_PATH_PREFIXES,
    PENDING_SPAWN_TTL_SECONDS,
    PROJECT_ROOT,
    SLASH_COMMAND_RE,
    SUBAGENT_SPAWN_RE,
    UUID_RE,
)
from guard.state_store import (
    _current_invocation_key,
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)


def _allowed_roots(payload: Dict[str, Any]) -> List[Path]:
    roots: List[Path] = [PROJECT_ROOT]
    for wp in payload.get("workspacePaths") or []:
        try:
            roots.append(Path(wp).resolve())
        except OSError:
            pass
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_artifact = Path(artifact_dir).resolve()
            roots.append(resolved_artifact)
            if resolved_artifact.parent.name == "brain":
                roots.append(resolved_artifact.parent)
                worktrees_dir = resolved_artifact.parent.parent / "worktrees"
                roots.append(worktrees_dir)
        except OSError:
            pass
    return roots


def _is_within_allowed_roots(target: Optional[str], roots: List[Path]) -> bool:
    if not target:
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _command_stays_in_workspace(
    command_line: str, cwd: str, roots: List[Path]
) -> bool:
    """Return True unless the command targets paths outside allowed roots."""
    base_dir = Path(os.path.expanduser(cwd)).resolve() if cwd else PROJECT_ROOT
    for match in re.finditer(r"(?:^|[\s=\"'><:])(/[A-Za-z0-9._/\-]*)", command_line):
        abs_path = match.group(1)
        if abs_path.startswith("//"):
            continue
        start_idx, end_idx = match.span(1)
        if (
            start_idx > 0
            and command_line[start_idx - 1] in ("'", '"')
            and end_idx < len(command_line)
            and command_line[end_idx] in ("'", '"', " ")
            and re.fullmatch(r"/[^/]+/[dgip]?", abs_path)
        ):
            continue
        if abs_path == "/tmp" or abs_path.startswith(ALLOWED_SYSTEM_PATH_PREFIXES):
            continue
        if not _is_within_allowed_roots(abs_path, roots):
            return False
    for match in re.finditer(
        r"(?:^|[\s=\"'><:])(\$(?:\{HOME\}|HOME|\{OLDPWD\}|OLDPWD)[A-Za-z0-9._/\-]*)",
        command_line,
    ):
        raw_token = match.group(1)
        home_val = os.environ.get("HOME") or str(Path.home())
        oldpwd_val = os.environ.get("OLDPWD") or ""
        expanded = (
            raw_token.replace("${HOME}", home_val)
            .replace("$HOME", home_val)
            .replace("${OLDPWD}", oldpwd_val)
            .replace("$OLDPWD", oldpwd_val)
        )
        if not _is_within_allowed_roots(expanded, roots):
            return False
    for match in re.finditer(r"(?:^|[\s=\"'])(~[A-Za-z0-9._/\-]*)", command_line):
        tilde_path = os.path.expanduser(match.group(1))
        if not _is_within_allowed_roots(tilde_path, roots):
            return False
    for match in re.finditer(
        r"(?:^|[\s=\"'])([A-Za-z0-9._\-/]*\.\.[A-Za-z0-9._\-/]*)", command_line
    ):
        rel_token = match.group(1)
        candidate = (base_dir / rel_token).resolve()
        if not _is_within_allowed_roots(str(candidate), roots):
            return False
    for match in re.finditer(
        r"(?:^|[\s=\"'><:])([A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*)",
        command_line,
    ):
        token = match.group(1)
        if token.startswith("-") or token.startswith("$"):
            continue
        try:
            candidate = base_dir / token
            if candidate.is_symlink():
                if not _is_within_allowed_roots(str(candidate.resolve()), roots):
                    return False
        except OSError:
            pass
    return True


def _is_parent_orchestrator(cid: str, state: Dict[str, Any]) -> bool:
    """Return True if cid is recorded as the root/parent orchestrator."""
    if not cid or cid in state.get("subagents", {}):
        return False
    if state.get("root_conversation_id") == cid:
        return True
    prefix = f'["{cid}"'
    return any(
        call.startswith(prefix) for call in state.get("recorded_spawn_calls", [])
    )


def _normalize_subagent_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(meta)
    if "type" in normalized and "type_name" not in normalized:
        normalized["type_name"] = normalized["type"]
    elif "type_name" in normalized and "type" not in normalized:
        normalized["type"] = normalized["type_name"]
    return normalized


def _workspaces_compatible(
    caller_payload: Dict[str, Any], sib_data: Dict[str, Any]
) -> bool:
    """Check whether caller payload and sibling state belong to compatible workspaces."""
    caller_wps = caller_payload.get("workspacePaths") or []
    sib_roots = sib_data.get("workspace_roots") or []
    if not caller_wps or not sib_roots:
        return True

    caller_resolved = set()
    for wp in caller_wps:
        try:
            caller_resolved.add(str(Path(wp).resolve()))
        except OSError:
            pass

    sib_resolved = set()
    for wp in sib_roots:
        try:
            sib_resolved.add(str(Path(wp).resolve()))
        except OSError:
            pass

    if not caller_resolved or not sib_resolved:
        return True

    if caller_resolved & sib_resolved:
        return True

    for c_path in caller_resolved:
        c_parts = Path(c_path).parts
        for s_path in sib_resolved:
            s_parts = Path(s_path).parts
            if "worktrees" in c_parts and "worktrees" in s_parts:
                c_idx = c_parts.index("worktrees")
                s_idx = s_parts.index("worktrees")
                if c_parts[: c_idx + 1] == s_parts[: s_idx + 1]:
                    return True
            if "worktrees" in c_parts and str(Path(s_path)) in c_path:
                return True
            if "worktrees" in s_parts and str(Path(c_path)) in s_path:
                return True

    return False


def _is_known_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if this invocation belongs to a spawned worker subagent."""
    if state.get("self_subagent_meta"):
        return True
    if payload.get("isSubagent") is True:
        return True
    cid = (payload.get("conversationId") or "").strip().lower()
    if cid and cid in state.get("subagents", {}):
        return True
    if _is_parent_orchestrator(cid, state):
        return False
    for wp in payload.get("workspacePaths") or []:
        if "/worktrees/" in str(wp).replace("\\", "/"):
            return True
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
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

                    if not _workspaces_compatible(payload, sib_data):
                        continue

                    sibling_subagents = sib_data.get("subagents") or {}
                    if cid and cid in sibling_subagents:
                        meta = _normalize_subagent_meta(sibling_subagents[cid])
                        if "parent_conversation_id" not in meta:
                            meta["parent_conversation_id"] = (
                                sib_data.get("root_conversation_id")
                                or sibling_state_path.parent.name
                            )
                        state["self_subagent_meta"] = meta
                        _save_state(payload, state)
                        return True

                    pending_list = sib_data.get("pending_spawns") or []
                    if pending_list and cid != sib_data.get("root_conversation_id"):
                        with state_transaction(sibling_state_path):
                            fresh_sib_data = _load_state(sibling_state_path)
                            if not _workspaces_compatible(payload, fresh_sib_data):
                                continue
                            fresh_pending = fresh_sib_data.get("pending_spawns") or []
                            if fresh_pending and cid != fresh_sib_data.get("root_conversation_id"):
                                try:
                                    age = time.time() - sibling_state_path.stat().st_mtime
                                    if age < PENDING_SPAWN_TTL_SECONDS:
                                        parent_cid = (
                                            fresh_sib_data.get("root_conversation_id")
                                            or sibling_state_path.parent.name
                                        )
                                        if cid:
                                            claimed = _normalize_subagent_meta(
                                                fresh_pending.pop(0)
                                            )
                                            claimed["parent_conversation_id"] = parent_cid
                                            fresh_sib_data["pending_spawns"] = fresh_pending
                                            fresh_sib_data.setdefault("subagents", {})[cid] = claimed
                                            _save_state(sibling_state_path, fresh_sib_data)
                                        else:
                                            claimed = _normalize_subagent_meta(fresh_pending[0])
                                            claimed["parent_conversation_id"] = parent_cid
                                        state["self_subagent_meta"] = claimed
                                        _save_state(payload, state)
                                        return True
                                except OSError:
                                    pass
        except OSError:
            pass
    return False


def _sync_subagents_from_transcript(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> bool:
    """Scan transcript.jsonl incrementally to pair pending_spawns with conversationIds."""
    transcript_path = payload.get("transcriptPath")
    if not transcript_path or not os.path.exists(transcript_path):
        return False

    pending: List[Dict[str, Any]] = state.get("pending_spawns", [])
    subagents: Dict[str, Dict[str, Any]] = state.setdefault("subagents", {})
    parent_cid = (payload.get("conversationId") or "").lower()
    changed = False

    try:
        file_size = os.path.getsize(transcript_path)
        offset = int(state.get("transcript_offset", 0))
        if offset < 0 or offset > file_size:
            offset = 0

        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            raw_new_offset = fh.tell()

        if not chunk:
            return changed

        if chunk.endswith(b"\n"):
            process_chunk = chunk
            new_offset = raw_new_offset
        else:
            last_nl = chunk.rfind(b"\n")
            if last_nl != -1:
                process_chunk = chunk[: last_nl + 1]
                new_offset = offset + last_nl + 1
            else:
                stripped = chunk.strip()
                if stripped.startswith(b"{") and not stripped.endswith(b"}"):
                    return changed
                process_chunk = chunk
                new_offset = raw_new_offset

        if new_offset != offset:
            state["transcript_offset"] = new_offset
            changed = True

        text = process_chunk.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            if (
                "docs/hooks.md" in raw_line
                or "docs/" in raw_line
                or "view_file" in raw_line
                or "read_file" in raw_line
                or "File Path: `file://" in raw_line
            ):
                continue
            if not (
                "Created the following subagent" in raw_line
                or "invoke_subagent" in raw_line
            ):
                continue
            matches = list(SUBAGENT_SPAWN_RE.finditer(raw_line))
            if not matches and "Created the following subagent" in raw_line:
                matches = list(UUID_RE.finditer(raw_line))
            for match in matches:
                cid = match.group(1).lower()
                if cid == parent_cid or cid in subagents:
                    continue
                if pending:
                    spawn_meta = pending.pop(0)
                    subagents[cid] = spawn_meta
                    changed = True
    except OSError:
        return changed

    return changed


def _is_slash_command_triggered(payload: Dict[str, Any]) -> bool:
    user_input = payload.get("lastUserInput")
    if isinstance(user_input, str):
        if SLASH_COMMAND_RE.search(user_input) or "skills/philharmonic/SKILL.md" in user_input:
            return True

    tool_call = payload.get("toolCall")
    if isinstance(tool_call, dict):
        tool_name = (tool_call.get("name") or "").lower()
        args = tool_call.get("args") or {}
        if tool_name in {"read_file", "view_file"}:
            for key in ("Path", "FilePath", "file_path", "path", "AbsolutePath"):
                val = args.get(key)
                if isinstance(val, str) and val.replace("\\", "/").endswith("skills/philharmonic/SKILL.md"):
                    return True
        elif tool_name == "lookup_skill":
            skill_name = str(args.get("skill_name") or args.get("SkillName") or "").strip().lower()
            if skill_name == "philharmonic":
                return True
        elif tool_name == "lookup_skills":
            skill_names = args.get("skill_names") or args.get("SkillNames") or []
            if isinstance(skill_names, list) and any(
                str(s).strip().lower() == "philharmonic" for s in skill_names
            ):
                return True
    return False


def _ensure_session_active(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    state_path: Optional[Path] = None,
) -> bool:
    if _is_slash_command_triggered(payload):
        state["active"] = True
        state["session_evaluated"] = True
        if state_path is not None:
            _save_state(state_path, state)
        return True

    if state.get("active") is True:
        return True

    if _is_known_subagent(payload, state):
        state["active"] = True
        state["session_evaluated"] = True
        if state_path is not None:
            _save_state(state_path, state)
        return True

    if "lastUserInput" in payload and payload.get("lastUserInput") is not None:
        if not state.get("session_evaluated"):
            state["session_evaluated"] = True
            state["active"] = False
            if state_path is not None:
                _save_state(state_path, state)
        return False

    if state.get("session_evaluated") is True:
        return bool(state.get("active", False))

    return True


def handle_pre_invocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}
        cid = (payload.get("conversationId") or "").strip().lower()

        if not _is_known_subagent(payload, state):
            changed = False
            if cid and not state.get("root_conversation_id"):
                state["root_conversation_id"] = cid
                changed = True

            user_input = str(payload.get("lastUserInput") or "")
            if state.get("plan_execute_phase"):
                if "[plan review]" in user_input.lower() or (
                    bool(user_input.strip()) and state.get("plan_execute_consumed")
                ):
                    state["plan_execute_phase"] = False
                    state["plan_execute_consumed"] = False
                    state["plan_allowed_files"] = None
                    changed = True
                else:
                    state["plan_execute_consumed"] = True
                    changed = True

            wps = [
                str(Path(wp).resolve())
                for wp in (payload.get("workspacePaths") or [])
            ]
            if wps and state.get("workspace_roots") != wps:
                state["workspace_roots"] = wps
                changed = True

            inv_num = payload.get("invocationNum")
            budget_map = state.get("direct_budget")
            if inv_num is not None and inv_num != 0 and inv_num != "0":
                prev_inv = state.get("current_invocation_num")
                if prev_inv != inv_num:
                    state["current_invocation_num"] = inv_num
                    changed = True
                if isinstance(budget_map, dict):
                    if str(inv_num) in budget_map:
                        budget_map.pop(str(inv_num), None)
                        changed = True
                    if "0" in budget_map:
                        budget_map.pop("0", None)
                        changed = True
                if (
                    inv_num == 1
                    or inv_num == "1"
                    or (prev_inv is not None and str(prev_inv) != str(inv_num))
                ):
                    from guard.verification import _get_circuit_breaker

                    cb = _get_circuit_breaker(state, payload)
                    if cb.get("failure_streak") or cb.get("tripped") or cb.get("replan_count"):
                        cb.update(
                            {
                                "failure_streak": 0,
                                "tripped": False,
                                "seen_failures": [],
                                "last_error": "",
                                "replan_count": 0,
                            }
                        )
                        changed = True
            else:
                if isinstance(budget_map, dict) and "0" in budget_map:
                    budget_map.pop("0", None)
                    changed = True
            if changed:
                _save_state(payload, state)
            return {}

        dedup_key = f"{cid}:{_current_invocation_key(payload, state)}"
        reminded: List[str] = state.setdefault("isolation_reminded_invocations", [])
        if dedup_key in reminded:
            return {}
        reminded.append(dedup_key)
        _save_state(payload, state)
        return {
            "injectSteps": [
                {
                    "ephemeralMessage": (
                        "[ORCHESTRATOR GUARD] Subagent context isolation active: ignore root orchestrator "
                        "delegation rules in SKILL.md. Execute only your assigned TASK and RULES, and return "
                        "strictly the 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                        "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                        "file:// .log URI when BLOCKED or FAILED; status may be COMPLETED, CHECKPOINT, "
                        "NEEDS_REPLAN, BLOCKED, or FAILED; scouts may use compact 2-header STATUS + NOTES)."
                    )
                }
            ]
        }


def _get_subagent_type(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
    if not _is_known_subagent(payload, state):
        return ""
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict):
        t = meta.get("type_name") or meta.get("type")
        if t is not None:
            return str(t).strip().lower()
    cid = (payload.get("conversationId") or "").strip().lower()
    sub_info = state.get("subagents", {}).get(cid)
    if isinstance(sub_info, dict):
        t = sub_info.get("type_name") or sub_info.get("type")
        if t is not None:
            return str(t).strip().lower()
    return ""


def _is_research_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a research subagent."""
    return _get_subagent_type(payload, state) in {"research", "researcher"}


def _is_scout_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a scout subagent."""
    return _get_subagent_type(payload, state) == "scout"


def _is_root_orchestrator(
    payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> bool:
    """Return True when the caller is the root orchestrator."""
    if state is None:
        state = _load_state(payload)
    return not _is_known_subagent(payload, state)


def register_subagent(
    payload: Dict[str, Any],
    conversation_id: str,
    type_name: str,
    model: str,
    workspace: str = "inherit",
    role: str = "",
    allowed_files: Optional[List[str]] = None,
) -> None:
    """Helper for explicitly registering a subagent conversationId -> tier mapping."""
    with state_transaction(payload):
        state = _load_state(payload)
        cid = conversation_id.strip().lower()
        record: Dict[str, Any] = {
            "role": role or type_name,
            "type": type_name,
            "model": model,
            "workspace": workspace,
            "active_branch": workspace == "branch",
        }
        if allowed_files is not None:
            record["allowed_files"] = list(allowed_files)
        state.setdefault("subagents", {})[cid] = _normalize_subagent_meta(record)
        _save_state(payload, state)

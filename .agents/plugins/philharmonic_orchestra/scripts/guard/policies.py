"""Tool-call policies, direct execution fast-path budget, and pre-tool validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ALLOWED_FILES_HEADER_RE,
    ALLOWED_SUBAGENT_MODELS,
    DEFAULT_MODEL_BY_TYPE,
    FORBIDDEN_DIRECT_READ_RE,
    MAX_DIRECT_COMMANDS,
    MAX_DIRECT_MUTATION_FILES,
    MAX_DIRECT_READ_FILES,
    MUTATION_TOOLS,
    MUTATION_VERBS_RE,
    NUMBERED_RULE_RE,
    PATH_ARG_KEYS,
    PROJECT_ROOT,
    READ_TOOLS,
    SEARCH_TOOLS,
    _extract_call_id,
    _is_git_only_command,
    _is_test_command,
    _split_shell_commands,
)
from guard.identity import (
    _allowed_roots,
    _command_stays_in_workspace,
    _ensure_session_active,
    _is_known_subagent,
    _is_research_subagent,
    _is_root_orchestrator,
    _is_scout_subagent,
    _is_within_allowed_roots,
    _normalize_subagent_meta,
    _sync_subagents_from_transcript,
)
from guard.state_store import (
    _current_invocation_key,
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)


def _parse_allowed_files_header(prompt_text: str) -> Optional[List[str]]:
    """Parse ALLOWED_FILES: header from a subagent prompt, returning None if omitted."""
    if not prompt_text:
        return None
    match = ALLOWED_FILES_HEADER_RE.search(prompt_text)
    if not match:
        return None
    raw_val = match.group(1).strip()
    tokens = [
        tok.strip().strip("`'\"")
        for tok in raw_val.split(",")
        if tok.strip().strip("`'\"")
    ]
    if len(tokens) == 1 and tokens[0].upper() == "NONE":
        return []
    return tokens


def _get_subagent_allowed_files(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Optional[List[str]]:
    """Return the per-subagent ALLOWED_FILES list if configured, or None if unrestricted."""
    if not _is_known_subagent(payload, state):
        return None
    cid = (payload.get("conversationId") or "").strip().lower()
    sub_info = state.get("subagents", {}).get(cid)
    if isinstance(sub_info, dict) and sub_info.get("allowed_files") is not None:
        return list(sub_info["allowed_files"])
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
                        sib_data = json.loads(
                            sibling_state_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        continue
                    sib_subagents = sib_data.get("subagents") or {}
                    if (
                        cid in sib_subagents
                        and sib_subagents[cid].get("allowed_files") is not None
                    ):
                        updated_meta = _normalize_subagent_meta(sib_subagents[cid])
                        if state.get("self_subagent_meta") != updated_meta:
                            state["self_subagent_meta"] = updated_meta
                            _save_state(payload, state)
                        return list(sib_subagents[cid]["allowed_files"])
        except OSError:
            pass
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict) and meta.get("allowed_files") is not None:
        return list(meta["allowed_files"])
    return None


def _is_target_in_allowed_files(
    target: str, allowed_files: List[str], workspace_paths: Optional[List[str]] = None
) -> bool:
    """Return True if target matches any file, directory, or allowlist entry in allowed_files."""
    if not allowed_files or not target:
        return False
    try:
        raw_target = Path(os.path.expanduser(target))
        target_path = (
            raw_target if raw_target.is_absolute() else (PROJECT_ROOT / raw_target)
        ).resolve()
    except OSError:
        return False

    search_roots = [PROJECT_ROOT] + [
        Path(wp).resolve() for wp in (workspace_paths or [])
    ]

    for raw_entry in allowed_files:
        entry = str(raw_entry).strip().strip("`'\"")
        if not entry or entry.upper() == "NONE":
            continue
        try:
            raw_ep = Path(os.path.expanduser(entry))
            if raw_ep.is_absolute():
                entry_paths = [raw_ep.resolve()]
            else:
                entry_paths = [(root / raw_ep).resolve() for root in search_roots]
        except OSError:
            continue
        for entry_path in entry_paths:
            if target_path == entry_path:
                return True
            try:
                target_path.relative_to(entry_path)
                return True
            except ValueError:
                pass
    return False


def _handle_define_subagent(args: Dict[str, Any]) -> Dict[str, Any]:
    overwrite: Dict[str, Any] = {}
    if args.get("EnableSubagentTools") is True:
        overwrite["EnableSubagentTools"] = False
    if args.get("enable_subagent_tools") is True:
        overwrite["enable_subagent_tools"] = False
    sub_name = str(args.get("Name") or args.get("name") or "").strip().lower()
    if sub_name == "scout":
        if args.get("EnableWriteTools") is True:
            overwrite["EnableWriteTools"] = False
        if args.get("enable_write_tools") is True:
            overwrite["enable_write_tools"] = False
    if sub_name in {"scout", "implementer"}:
        if args.get("EnableMCPTools") is True:
            overwrite["EnableMCPTools"] = False
        if args.get("enable_mcp_tools") is True:
            overwrite["enable_mcp_tools"] = False
    if overwrite:
        return {
            "decision": "allow",
            "overwrite": overwrite,
        }
    return {"decision": "allow"}


def _is_simple_readonly_task(type_name: str, prompt_text: str) -> bool:
    """Return True if a read-only subagent prompt has <= 3 rules and no mutation verbs."""
    if type_name not in {"scout", "research", "researcher"}:
        return False
    rule_count = len(NUMBERED_RULE_RE.findall(prompt_text))
    cleaned_prompt = re.sub(
        r"(?im)^\s*(?:\d+[.)]\s*)?do\s+not\s+.*$", "", prompt_text
    )
    has_mutation = bool(MUTATION_VERBS_RE.search(cleaned_prompt))
    return not has_mutation and rule_count <= 3


def _validate_and_prepare_subagent_entry(
    entry: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Validate a single subagent spec and return (denial, spawn_record, target_model)."""
    type_name = (entry.get("TypeName") or "").strip()
    prompt_text = entry.get("Prompt") or ""
    if not type_name:
        return (
            {
                "decision": "deny",
                "reason": "invoke_subagent requires a non-empty 'TypeName'.",
            },
            None,
            None,
        )
    if not prompt_text or not prompt_text.strip():
        return (
            {
                "decision": "deny",
                "reason": "invoke_subagent requires a non-empty 'Prompt'.",
            },
            None,
            None,
        )

    requested_model = (entry.get("Model") or "inherit").strip()
    role_label = (entry.get("Role") or type_name or "subagent").strip()

    if requested_model == "pro":
        return (
            {
                "decision": "deny",
                "reason": (
                    f"Subagent '{role_label}' requested Model='pro'. "
                    "Subagents must use 'flash' or 'flash_lite'."
                ),
            },
            None,
            None,
        )

    target_model = requested_model
    if target_model not in ALLOWED_SUBAGENT_MODELS:
        target_model = DEFAULT_MODEL_BY_TYPE.get(type_name, "flash")

    if target_model == "flash" and _is_simple_readonly_task(type_name, prompt_text):
        target_model = "flash_lite"

    workspace_mode = (entry.get("Workspace") or "inherit").strip()
    spawn_record: Dict[str, Any] = {
        "role": role_label,
        "type": type_name,
        "type_name": type_name,
        "model": target_model,
        "workspace": workspace_mode,
        "active_branch": workspace_mode == "branch",
    }
    allowed_files = _parse_allowed_files_header(prompt_text)
    if allowed_files is not None:
        spawn_record["allowed_files"] = allowed_files
    return None, spawn_record, target_model


def _handle_invoke_subagent(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if state is None:
        state = _load_state(payload)
    call_id = _extract_call_id(payload)
    recorded_calls: List[str] = state.setdefault("recorded_spawn_calls", [])
    is_list_schema = "Subagents" in args and isinstance(args.get("Subagents"), list)
    if is_list_schema and not args["Subagents"]:
        return {
            "decision": "deny",
            "reason": "invoke_subagent requires a non-empty 'Subagents' list.",
        }

    entries = args["Subagents"] if is_list_schema else [args]
    validated_items: List[Tuple[int, Dict[str, Any], Dict[str, Any], str]] = []
    for idx, raw_item in enumerate(entries):
        item = raw_item if isinstance(raw_item, dict) else {}
        denial, spawn_record, target_model = _validate_and_prepare_subagent_entry(item)
        if denial is not None or spawn_record is None or target_model is None:
            return denial or {
                "decision": "deny",
                "reason": "Invalid subagent entry."
                if is_list_schema
                else "Invalid invoke_subagent arguments.",
            }
        validated_items.append((idx, item, spawn_record, target_model))

    workspace_roots = [
        str(Path(wp).resolve())
        for wp in (payload.get("workspacePaths") or [])
    ]
    if workspace_roots and state.get("workspace_roots") != workspace_roots:
        state["workspace_roots"] = workspace_roots

    updated_list: List[Dict[str, Any]] = []
    any_model_overwritten = False
    state_changed = False
    for idx, item, spawn_record, target_model in validated_items:
        sig_call_id = f"{call_id}:{idx}" if is_list_schema else call_id
        call_sig = json.dumps(
            [
                payload.get("conversationId", ""),
                _current_invocation_key(payload, state),
                payload.get("executionNum", 0),
                sig_call_id,
                spawn_record["type_name"],
                spawn_record["role"],
                spawn_record["workspace"],
            ],
            sort_keys=True,
        )
        if workspace_roots:
            spawn_record["workspace_roots"] = workspace_roots
        spawn_record["call_id"] = call_id
        spawn_record["call_sig"] = call_sig
        spawn_record["spawned_at"] = time.time()

        if call_sig not in recorded_calls:
            recorded_calls.append(call_sig)
            state.setdefault("pending_spawns", []).append(spawn_record)
            state_changed = True
        new_item = dict(item)
        if item.get("Model") != target_model:
            new_item["Model"] = target_model
            any_model_overwritten = True
        updated_list.append(new_item)

    if state_changed:
        _save_state(payload, state)
    if any_model_overwritten:
        return {
            "decision": "allow",
            "overwrite": {"Subagents": updated_list}
            if is_list_schema
            else {"Model": updated_list[0]["Model"]},
        }
    return {"decision": "allow"}


def _handle_send_message(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recipient = (args.get("Recipient") or "").strip().lower()
    message = args.get("Message") or ""

    if not message or not message.strip():
        return {"decision": "deny", "reason": "send_message.Message cannot be empty."}

    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    _save_state(payload, state)

    if not _is_root_orchestrator(payload, state) and re.match(r"^\s*STATUS:", message):
        from guard.verification import validate_subagent_report

        err = validate_subagent_report(message)
        if err:
            return {
                "decision": "deny",
                "reason": f"[ORCHESTRATOR GUARD] Subagent report format error in send_message: {err}",
            }
        sent_key = f"sent:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
        reminded = state.setdefault("stop_reminded_executions", [])
        if sent_key not in reminded:
            reminded.append(sent_key)
            _save_state(payload, state)

    subagent_info = state.get("subagents", {}).get(recipient)
    if subagent_info:
        if subagent_info.get("model") == "flash_lite":
            rule_count = len(NUMBERED_RULE_RE.findall(message))
            has_mutation = bool(MUTATION_VERBS_RE.search(message))
            if has_mutation or rule_count > 3:
                return {
                    "decision": "deny",
                    "reason": (
                        f"Recipient '{recipient}' is a 'flash_lite' ({subagent_info.get('type', 'scout')}) "
                        "subagent. Do not send code mutation or multi-step (>3 rules) tasks to 'flash_lite'. "
                        "Invoke a 'flash' subagent ('research' or 'implementer') instead."
                    ),
                }

    parsed_allowed = _parse_allowed_files_header(message)
    if parsed_allowed is not None:
        if subagent_info:
            subagent_info["allowed_files"] = parsed_allowed
            _save_state(payload, state)
        artifact_dir = payload.get("artifactDirectoryPath")
        if artifact_dir and recipient:
            try:
                resolved_art = Path(artifact_dir).resolve()
                if resolved_art.parent.name == "brain":
                    sib_state_file = (
                        resolved_art.parent / recipient / "orchestrator_state.json"
                    )
                    if sib_state_file.exists():
                        with state_transaction(sib_state_file):
                            sib_state = _load_state(sib_state_file)
                            meta = sib_state.setdefault("self_subagent_meta", {})
                            meta["allowed_files"] = parsed_allowed
                            _save_state(sib_state_file, sib_state)
            except OSError:
                pass

    return {"decision": "allow"}


def _handle_manage_subagents(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = str(args.get("Action") or args.get("action") or "").strip().lower()
    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    subagents = state.get("subagents", {})

    if action == "kill_all":
        for meta in subagents.values():
            meta["active_branch"] = False
        for pending in state.get("pending_spawns", []):
            pending["active_branch"] = False
        state["pending_spawns"] = []
        _save_state(payload, state)
    elif action == "kill":
        raw_ids = (
            args.get("ConversationIds")
            or args.get("conversation_ids")
            or ([args["ConversationId"]] if args.get("ConversationId") else [])
            or ([args["conversation_id"]] if args.get("conversation_id") else [])
        )
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        target_ids = {str(cid).strip().lower() for cid in raw_ids if cid}
        for cid in target_ids:
            if cid in subagents:
                subagents[cid]["active_branch"] = False
        remaining_pending = []
        killed_left = len(target_ids)
        for pending in state.get("pending_spawns", []):
            if pending.get("active_branch") and killed_left > 0:
                pending["active_branch"] = False
                killed_left -= 1
            remaining_pending.append(pending)
        state["pending_spawns"] = remaining_pending
        _save_state(payload, state)

    return {"decision": "allow"}


def _is_artifact_or_worktree_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside the conversation artifact/brain directory or a subagent worktree."""
    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir or not target:
        return False
    try:
        resolved_target = Path(os.path.expanduser(target)).resolve()
        resolved_artifact = Path(artifact_dir).resolve()
    except OSError:
        return False

    exempt_roots = [resolved_artifact]
    if resolved_artifact.parent.name == "brain":
        exempt_roots.append(resolved_artifact.parent)
        exempt_roots.append(resolved_artifact.parent.parent / "worktrees")

    for root in exempt_roots:
        try:
            resolved_target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_allowed_research_scratch_write(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is a .md or .txt file inside a scratch directory under brain/."""
    if not target or not _is_artifact_or_worktree_path(target, payload):
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except OSError:
        return False
    return resolved.suffix.lower() in {".md", ".txt"} and "scratch" in resolved.parts


def _extract_any_path_arg(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Return the first present path argument value from PATH_ARG_KEYS, or None."""
    for key in PATH_ARG_KEYS:
        if key in args and args.get(key) is not None:
            return str(args.get(key))
    return None


def _extract_target_path(tool_name: str, args: Dict[str, Any]) -> str:
    """Extract the target file or directory path for native Antigravity tools."""
    if tool_name in (SEARCH_TOOLS | READ_TOOLS | MUTATION_TOOLS):
        return _extract_any_path_arg(tool_name, args) or ""
    return ""


def _check_and_record_direct_budget(
    payload: Dict[str, Any],
    tool_name: str,
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Enforce direct execution fast-path limits on the root orchestrator."""
    if state is None:
        state = _load_state(payload)
    if not _is_root_orchestrator(payload, state):
        return None

    inv_key = _current_invocation_key(payload, state)
    budget = state.setdefault("direct_budget", {}).setdefault(
        inv_key,
        {
            "mutated_files": [],
            "read_files": [],
            "commands": [],
            "seen_calls": [],
            "test_runs_after_edit": 0,
        },
    )
    call_id = _extract_call_id(payload)

    if tool_name in SEARCH_TOOLS:
        search_target = _extract_target_path(tool_name, args)
        if search_target:
            norm_slashes = search_target.replace("\\", "/")
            if FORBIDDEN_DIRECT_READ_RE.search(norm_slashes):
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
                        f"directly read raw log or transcript file '{Path(search_target).name}'. "
                        "Delegate log slicing to a 'scout' subagent (Model='flash_lite')."
                    ),
                }
        return None

    if tool_name in READ_TOOLS:
        target = _extract_target_path(tool_name, args)
        if not target:
            return None
        norm_slashes = str(target).replace("\\", "/")
        if FORBIDDEN_DIRECT_READ_RE.search(norm_slashes):
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
                    f"directly read raw log or transcript file '{Path(target).name}'. "
                    "Delegate log slicing to a 'scout' subagent (Model='flash_lite')."
                ),
            }
        if _is_artifact_or_worktree_path(target, payload) or "/skills/" in norm_slashes:
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except OSError:
            return None

        read_files: List[str] = budget.setdefault("read_files", [])
        if norm_target not in read_files:
            if len(read_files) >= MAX_DIRECT_READ_FILES:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Direct execution fast-path read limit "
                        f"({MAX_DIRECT_READ_FILES} workspace files) exceeded when reading "
                        f"'{Path(norm_target).name}'. Delegate multi-file reads or discovery to a "
                        "'scout' (Model='flash_lite') or 'research' (Model='flash') subagent."
                    ),
                }
            read_files.append(norm_target)
            _save_state(payload, state)
        return None

    if tool_name in MUTATION_TOOLS:
        target = _extract_target_path(tool_name, args)
        if not target or _is_artifact_or_worktree_path(target, payload):
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except OSError:
            return None

        if budget.get("test_runs_after_edit", 0) >= 1:
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Iterative edit-and-test loop detected in the orchestrator. "
                    "Delegate multi-step code and test execution loops to an 'implementer' subagent (Model='flash')."
                ),
            }

        mutated_files: List[str] = budget.setdefault("mutated_files", [])
        if norm_target not in mutated_files:
            if len(mutated_files) >= MAX_DIRECT_MUTATION_FILES:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Direct execution fast-path limit "
                        f"({MAX_DIRECT_MUTATION_FILES} workspace files) exceeded when editing "
                        f"'{Path(norm_target).name}'. Delegate multi-file changes to an "
                        "'implementer' subagent (Model='flash')."
                    ),
                }
            mutated_files.append(norm_target)
            _save_state(payload, state)
        return None

    if tool_name == "run_command":
        cmd = (args.get("CommandLine") or "").strip()
        if _is_git_only_command(cmd):
            return None

        seen_calls: List[str] = budget.setdefault("seen_calls", [])
        if call_id and call_id in seen_calls:
            return None

        mutated_files = budget.get("mutated_files", [])
        is_test = _is_test_command(cmd)
        if mutated_files and is_test:
            if budget.get("test_runs_after_edit", 0) >= 1:
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Direct workspace edit followed by iterative test execution "
                        "detected in the orchestrator. Delegate code and test execution loops "
                        "to an 'implementer' subagent (Model='flash')."
                    ),
                }
            budget["test_runs_after_edit"] = budget.get("test_runs_after_edit", 0) + 1

        commands: List[str] = budget.setdefault("commands", [])
        if len(commands) >= MAX_DIRECT_COMMANDS:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Direct execution fast-path command limit "
                    f"({MAX_DIRECT_COMMANDS} commands per turn) exceeded. Delegate iterative "
                    "command or test execution to a 'scout', 'research', or 'implementer' subagent."
                ),
            }

        if call_id:
            seen_calls.append(call_id)
        commands.append(cmd)
        _save_state(payload, state)
        return None

    return None


def handle_pre_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {"decision": "allow"}

        from guard.verification import (
            _check_test_tampering_in_edit,
            _get_circuit_breaker,
            _get_test_receipt,
            _is_code_file,
            _propagate_receipt_to_parent_brain,
        )

        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}

        cb = _get_circuit_breaker(state, payload)
        if cb.get("tripped"):
            if tool_name in MUTATION_TOOLS:
                target = _extract_target_path(tool_name, args)
                if not _is_artifact_or_worktree_path(target, payload):
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                            f"{cb.get('failure_streak', 2)} consecutive failures "
                            f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                            "Stop retrying, write failure details to the scratch log if needed, "
                            "and report STATUS: BLOCKED."
                        ),
                    }
            elif tool_name == "run_command":
                cmd = (args.get("CommandLine") or "").strip()
                sub_cmds = _split_shell_commands(cmd)
                is_safe_git_inspection = (
                    _is_git_only_command(cmd)
                    and bool(sub_cmds)
                    and all(
                        bool(re.match(r"^\s*git\s+(status|diff|log)\b", c))
                        for c in sub_cmds
                    )
                )
                if not is_safe_git_inspection:
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                            f"{cb.get('failure_streak', 2)} consecutive failures "
                            f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                            "Stop retrying, write failure details to the scratch log if needed, "
                            "and report STATUS: BLOCKED."
                        ),
                    }
            elif tool_name in {"invoke_subagent", "send_message"}:
                if tool_name == "send_message" and not _is_root_orchestrator(payload, state):
                    msg_text = str(args.get("Message") or "")
                    if re.search(r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED)\b", msg_text):
                        pass
                    else:
                        return {
                            "decision": "deny",
                            "reason": (
                                f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                                f"{cb.get('failure_streak', 2)} consecutive failures "
                                f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                                "Stop retrying, write failure details to the scratch log if needed, "
                                "and report STATUS: BLOCKED."
                            ),
                        }
                else:
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                            f"{cb.get('failure_streak', 2)} consecutive failures "
                            f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                            "Stop retrying, write failure details to the scratch log if needed, "
                            "and report STATUS: BLOCKED."
                        ),
                    }

        if tool_name in SEARCH_TOOLS:
            budget_denial = _check_and_record_direct_budget(payload, tool_name, args, state)
            if budget_denial:
                return budget_denial

        if tool_name == "define_subagent":
            return _handle_define_subagent(args)

        if tool_name == "invoke_subagent":
            return _handle_invoke_subagent(payload, args, state)

        if tool_name == "send_message":
            return _handle_send_message(payload, args, state)

        if tool_name == "manage_subagents":
            return _handle_manage_subagents(payload, args, state)

        if tool_name in MUTATION_TOOLS:
            if _is_scout_subagent(payload, state):
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Scout subagents (Model='flash_lite') are strictly "
                        "read-only. File mutations are prohibited."
                    ),
                }
            if _is_research_subagent(payload, state):
                target = _extract_target_path(tool_name, args)
                if not _is_allowed_research_scratch_write(target, payload):
                    return {
                        "decision": "deny",
                        "reason": (
                            "[ORCHESTRATOR GUARD] Research subagents may only write markdown reports "
                            "to <artifactDirectoryPath>/scratch/*.md. Workspace mutations are prohibited."
                        ),
                    }

            allowed_files = _get_subagent_allowed_files(payload, state)
            target = _extract_target_path(tool_name, args)
            if (
                allowed_files is not None
                and target
                and not _is_artifact_or_worktree_path(target, payload)
            ):
                if not _is_target_in_allowed_files(
                    target, allowed_files, payload.get("workspacePaths")
                ):
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] File allowlist violation: subagent attempted to mutate "
                            f"'{Path(target).name}', which is outside ALLOWED_FILES "
                            f"({', '.join(allowed_files) if allowed_files else 'NONE'})."
                        ),
                    }

            tamper_denial = _check_test_tampering_in_edit(tool_name, args, payload)
            if tamper_denial is not None:
                return tamper_denial

        roots = _allowed_roots(payload)

        if tool_name == "run_command":
            cwd = args.get("Cwd", "") or str(PROJECT_ROOT)
            cmd = args.get("CommandLine", "") or ""
            sub_cmds = _split_shell_commands(cmd)

            if _is_scout_subagent(payload, state):
                has_write_redir = False
                for m in re.finditer(r"(?:^|[^0-9&])>{1,2}\s*([^\s;&|><]+)", cmd):
                    target_sink = m.group(1).strip().strip("`'\"")
                    if target_sink not in {"/dev/null", "&1", "&2"}:
                        has_write_redir = True
                        break
                has_mutating_cmd = False
                for sc in sub_cmds:
                    if re.search(r"\b(?:rm|mv|cp|tee|touch|truncate|dd)\b", sc) or re.search(
                        r"\b(?:sed|perl)\s+-[^\s]*i\b", sc
                    ):
                        has_mutating_cmd = True
                        break
                if has_write_redir or has_mutating_cmd:
                    return {
                        "decision": "deny",
                        "reason": (
                            "[ORCHESTRATOR GUARD] Scout subagents (Model='flash_lite') are strictly "
                            "read-only. Shell file mutations are prohibited."
                        ),
                    }

            allowed_files = _get_subagent_allowed_files(payload, state)
            if allowed_files is not None:
                write_targets: List[str] = []
                for m in re.finditer(r"(?:^|[^0-9&])>{1,2}\s*([^\s;&|><]+)", cmd):
                    target_sink = m.group(1).strip().strip("`'\"")
                    if target_sink not in {"/dev/null", "&1", "&2"}:
                        write_targets.append(target_sink)
                for sc in sub_cmds:
                    m_cmd = re.search(r"\b(tee|rm|touch)\b(?:\s+-[^\s]+)*\s+([^;&|]+)", sc)
                    if m_cmd:
                        rest = m_cmd.group(2).strip()
                        for arg_tok in rest.split():
                            arg_clean = arg_tok.strip().strip("`'\"")
                            if arg_clean and not arg_clean.startswith("-"):
                                write_targets.append(arg_clean)
                    m_sed = re.search(
                        r"\bsed\s+-[^\s]*i[^\s]*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+([^;&|]+)",
                        sc,
                    )
                    if m_sed:
                        rest = m_sed.group(1).strip()
                        for arg_tok in rest.split():
                            arg_clean = arg_tok.strip().strip("`'\"")
                            if arg_clean and not arg_clean.startswith("-"):
                                write_targets.append(arg_clean)

                for wt in write_targets:
                    if (
                        not _is_artifact_or_worktree_path(wt, payload)
                        and wt != "/tmp"
                        and not wt.startswith("/tmp/")
                    ):
                        if not _is_target_in_allowed_files(
                            wt, allowed_files, payload.get("workspacePaths")
                        ):
                            return {
                                "decision": "deny",
                                "reason": (
                                    f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: "
                                    f"'{wt}' is outside ALLOWED_FILES "
                                    f"({', '.join(allowed_files) if allowed_files else 'NONE'})."
                                ),
                            }

            if _is_within_allowed_roots(cwd, roots) and _command_stays_in_workspace(
                cmd, cwd, roots
            ):
                budget_denial = _check_and_record_direct_budget(
                    payload, tool_name, args, state
                )
                if budget_denial:
                    return budget_denial
                overrides = [f"command({cmd})"]
                for r in roots:
                    overrides.append(f"file({r}/**)")
                return {
                    "decision": "allow",
                    "permissionOverrides": overrides,
                }
            return {"decision": "ask"}

        target_path = _extract_any_path_arg(tool_name, args)
        if target_path is not None:
            if _is_within_allowed_roots(target_path, roots):
                budget_denial = _check_and_record_direct_budget(
                    payload, tool_name, args, state
                )
                if budget_denial:
                    return budget_denial
                if tool_name in MUTATION_TOOLS:
                    if not _is_artifact_or_worktree_path(
                        target_path, payload
                    ) and _is_code_file(target_path):
                        receipt = _get_test_receipt(state, payload)
                        receipt["dirty"] = True
                        receipt["verified"] = False
                        try:
                            receipt["last_mutated_file"] = str(
                                Path(os.path.expanduser(target_path)).resolve()
                            )
                        except OSError:
                            receipt["last_mutated_file"] = target_path
                        _save_state(payload, state)
                        if not _is_root_orchestrator(payload, state):
                            _propagate_receipt_to_parent_brain(
                                payload,
                                dirty=True,
                                verified=False,
                                last_mutated_file=receipt["last_mutated_file"],
                            )
                return {"decision": "allow"}
            return {"decision": "ask"}

        return {"decision": "allow"}

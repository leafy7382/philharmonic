"""Tool-call policies, direct execution fast-path budget, and pre-tool validation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from guard.constants import (
    ALLOWED_FILES_HEADER_RE,
    ALLOWED_SUBAGENT_MODELS,
    ALLOWED_VERIFIER_REVIEWER_MODELS,
    COHORT_HEADER_RE,
    DEFAULT_MODEL_BY_TYPE,
    DEFAULT_UNBOUNDED_READ_BYTES,
    DEFAULT_UNBOUNDED_READ_LINES,
    DIR_LIST_TOOLS,
    EVAL_CMD_HEADER_RE,
    FORBIDDEN_DIRECT_READ_RE,
    GROUNDED_SPEC_SECTION_PATTERNS,
    HIGH_REASONING_READONLY_TYPES,
    ISOLATED_WORKSPACES,
    MAX_DIRECT_COMMANDS,
    MAX_DIRECT_MUTATION_FILES,
    MAX_DIRECT_READ_BYTES,
    MAX_DIRECT_READ_FILES,
    MAX_DIRECT_READ_LINES,
    MAX_REPLAN_ATTEMPTS,
    MODE_HEADER_RE,
    MUTATION_TOOLS,
    MUTATION_VERBS_RE,
    NUMBERED_RULE_RE,
    PATH_ARG_KEYS,
    PENDING_SPAWN_TTL_SECONDS,
    PROJECT_ROOT,
    READ_TOOLS,
    REPLAN_HEADER_RE,
    REPLAN_IF_HEADER_RE,
    REVIEW_LENS_HEADER_RE,
    SAFE_GIT_INSPECTION_RE,
    SEARCH_TOOLS,
    SPEC_FILE_HEADER_RE,
    STAGE_HEADER_RE,
    STRICT_READONLY_SUBAGENT_TYPES,
    TRIVIAL_EVAL_CMD_RE,
    _extract_call_id,
    _is_git_only_command,
    _is_test_command,
    _split_shell_commands,
)
from guard.identity import (
    _allowed_roots,
    _command_stays_in_workspace,
    _ensure_session_active,
    _get_subagent_type,
    _is_known_subagent,
    _is_research_subagent,
    _is_root_orchestrator,
    _is_scout_subagent,
    _is_strict_readonly_subagent,
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


def _parse_review_lens_header(prompt_text: str) -> Optional[str]:
    """Parse REVIEW_LENS: header from a subagent prompt, returning None if omitted."""
    if not prompt_text:
        return None
    match = REVIEW_LENS_HEADER_RE.search(prompt_text)
    if not match:
        return None
    return match.group(1).strip().strip("`'\"")


def _normalize_review_lens(lens: str) -> str:
    """Normalize a review lens string so spaces, hyphens, and underscores collide cleanly."""
    cleaned = re.sub(r"[\s_-]+", "-", (lens or "").strip().lower())
    return cleaned.strip("-")


def _parse_mode_header(prompt_text: str) -> Optional[str]:
    """Parse MODE: header from a subagent prompt, normalizing Best-of-N variants."""
    if not prompt_text:
        return None
    match = MODE_HEADER_RE.search(prompt_text)
    if not match:
        return None
    raw_val = match.group(1).strip().strip("`'\"")
    if not raw_val:
        return None
    cleaned = re.sub(r"[\s\-]+", "_", raw_val.lower()).strip("_")
    if cleaned in {"best_of_n", "bestofn"}:
        return "best_of_n"
    return cleaned or None


def _parse_cohort_header(prompt_text: str) -> Optional[str]:
    """Parse COHORT: header from a subagent prompt, returning None if omitted."""
    if not prompt_text:
        return None
    match = COHORT_HEADER_RE.search(prompt_text)
    if not match:
        return None
    raw_val = match.group(1).strip().strip("`'\"")
    if not raw_val or raw_val.upper() == "NONE":
        return None
    cleaned = re.sub(r"[\s_-]+", "-", raw_val.lower()).strip("-")
    return cleaned or None


def _compute_prompt_hash(prompt_text: str) -> str:
    """Return a short deterministic SHA-256 prefix for non-empty subagent prompts."""
    if not prompt_text:
        return ""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def _parse_eval_cmd_header(prompt_text: str) -> Optional[str]:
    """Parse EVAL_CMD: header from a prompt or plan, returning None if omitted."""
    if not prompt_text:
        return None
    match = EVAL_CMD_HEADER_RE.search(prompt_text)
    if not match:
        return None
    raw_val = match.group(1).strip()
    if (
        len(raw_val) >= 2
        and raw_val[0] == raw_val[-1]
        and raw_val[0] in {"`", "'", '"'}
    ):
        raw_val = raw_val[1:-1].strip()
    if not raw_val or raw_val.upper() == "NONE":
        return None
    return raw_val


def _has_unquoted_or_operator(cmd: str) -> bool:
    """Return True if cmd contains an unquoted shell '||' operator."""
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single and not in_double and cmd[i : i + 2] == "||":
            return True
        i += 1
    return False


def _is_trivial_eval_command(eval_cmd: str) -> bool:
    """Return True if eval_cmd is empty, masks failure with '||', or runs only trivial no-ops."""
    if not eval_cmd or not eval_cmd.strip():
        return True
    if _has_unquoted_or_operator(eval_cmd):
        return True
    parts = _split_shell_commands(eval_cmd)
    if not parts:
        return True
    return all(bool(TRIVIAL_EVAL_CMD_RE.match(part)) for part in parts)


def _strip_leading_env_assignments(cmd: str) -> str:
    return re.sub(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", "", cmd).strip()


def _is_declared_eval_command(
    cmd: str, state: Dict[str, Any], payload: Dict[str, Any]
) -> bool:
    """Return True if cmd invokes a non-trivial declared EVAL_CMD without shell-chaining bypasses."""
    if not cmd or "$(" in cmd or "`" in cmd or _has_unquoted_or_operator(cmd):
        return False

    declared_raw: List[str] = []
    cid = (payload.get("conversationId") or "").strip().lower()
    sub_entry = (state.get("subagents") or {}).get(cid)
    if isinstance(sub_entry, dict) and sub_entry.get("eval_cmd"):
        declared_raw.append(str(sub_entry["eval_cmd"]))
    self_meta = state.get("self_subagent_meta")
    if isinstance(self_meta, dict) and self_meta.get("eval_cmd"):
        declared_raw.append(str(self_meta["eval_cmd"]))
    if state.get("active_eval_cmd"):
        declared_raw.append(str(state["active_eval_cmd"]))
    for cohort_entry in (state.get("cohorts") or {}).values():
        if isinstance(cohort_entry, dict) and cohort_entry.get("eval_cmd"):
            declared_raw.append(str(cohort_entry["eval_cmd"]))

    declared = [
        _strip_leading_env_assignments(d)
        for d in declared_raw
        if isinstance(d, str) and d.strip()
    ]
    declared = [d for d in declared if d and not _is_trivial_eval_command(d)]
    if not declared:
        return False

    sub_cmds = _split_shell_commands(cmd)
    if not sub_cmds:
        return False
    for sc in sub_cmds:
        stripped = _strip_leading_env_assignments(sc)
        if not stripped or _is_trivial_eval_command(stripped):
            continue
        for decl in declared:
            if stripped == decl or stripped.startswith(decl + " "):
                return True
    return False


def _normalize_allowlist_entry(entry: str) -> Optional[Path]:
    """Normalize an ALLOWED_FILES entry relative to PROJECT_ROOT, stripping glob tails."""
    cleaned = str(entry or "").strip().strip("`'\"")
    if not cleaned or cleaned.upper() == "NONE":
        return None
    cleaned = cleaned.replace("\\", "/")
    while cleaned.endswith("/**") or cleaned.endswith("/*"):
        if cleaned.endswith("/**"):
            cleaned = cleaned[:-3]
        elif cleaned.endswith("/*"):
            cleaned = cleaned[:-2]
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    if not cleaned:
        return None
    try:
        raw_path = Path(os.path.expanduser(cleaned))
        return (
            raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
        ).resolve()
    except OSError:
        return None


def _find_allowed_files_overlap(
    files_a: List[str], files_b: List[str]
) -> Optional[Tuple[str, str]]:
    """Return the first conflicting pair (entry_a, entry_b) if two allowlists overlap."""
    if not files_a or not files_b:
        return None
    norm_a: List[Tuple[str, Path]] = []
    for item in files_a:
        p = _normalize_allowlist_entry(item)
        if p is not None:
            norm_a.append((item, p))
    norm_b: List[Tuple[str, Path]] = []
    for item in files_b:
        p = _normalize_allowlist_entry(item)
        if p is not None:
            norm_b.append((item, p))
    for raw_a, path_a in norm_a:
        for raw_b, path_b in norm_b:
            if path_a == path_b:
                return (raw_a, raw_b)
            try:
                path_a.relative_to(path_b)
                return (raw_a, raw_b)
            except ValueError:
                pass
            try:
                path_b.relative_to(path_a)
                return (raw_a, raw_b)
            except ValueError:
                pass
    return None


def _iter_active_subagents(
    state: Dict[str, Any],
    batch_entries: Optional[List[Dict[str, Any]]] = None,
    exclude_cid: Optional[str] = None,
    exclude_call_sigs: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Collect active worker records from batch_entries, non-expired pending_spawns, and active subagents."""
    active: List[Dict[str, Any]] = []
    for entry in batch_entries or []:
        if isinstance(entry, dict):
            active.append(entry)

    now = time.time()
    for pending in (state.get("pending_spawns") or []):
        if not isinstance(pending, dict):
            continue
        if exclude_call_sigs and pending.get("call_sig") in exclude_call_sigs:
            continue
        if pending.get("status", "active") != "active":
            continue
        spawned_at = pending.get("spawned_at")
        if spawned_at is not None:
            try:
                if (now - float(spawned_at)) > PENDING_SPAWN_TTL_SECONDS:
                    continue
            except (ValueError, TypeError):
                pass
        active.append(pending)

    norm_exclude = (exclude_cid or "").strip().lower()
    for cid, entry in (state.get("subagents") or {}).items():
        if not isinstance(entry, dict):
            continue
        if norm_exclude and str(cid).strip().lower() == norm_exclude:
            continue
        if entry.get("status") == "active":
            active.append(entry)
    return active


def _handle_define_subagent(args: Dict[str, Any]) -> Dict[str, Any]:
    overwrite: Dict[str, Any] = {}
    if args.get("EnableSubagentTools") is True:
        overwrite["EnableSubagentTools"] = False
    if args.get("enable_subagent_tools") is True:
        overwrite["enable_subagent_tools"] = False
    sub_name = str(args.get("Name") or args.get("name") or "").strip().lower()
    if sub_name in STRICT_READONLY_SUBAGENT_TYPES:
        if args.get("EnableWriteTools") is True:
            overwrite["EnableWriteTools"] = False
        if args.get("enable_write_tools") is True:
            overwrite["enable_write_tools"] = False
    if sub_name in {"scout", "implementer", "verifier", "reviewer"}:
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
    norm_type = (type_name or "").strip().lower()
    if norm_type in HIGH_REASONING_READONLY_TYPES:
        return False
    if norm_type not in {"scout", "research", "researcher"}:
        return False
    rule_count = len(NUMBERED_RULE_RE.findall(prompt_text))
    cleaned_prompt = re.sub(
        r"(?im)^\s*(?:\d+[.)]\s*)?do\s+not\s+.*$", "", prompt_text
    )
    has_mutation = bool(MUTATION_VERBS_RE.search(cleaned_prompt))
    return not has_mutation and rule_count <= 3


def _validate_spec_file_header(
    prompt_text: str, type_name: str, payload: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Validate SPEC_FILE: header. Returns (resolved_spec_path_str, error_reason)."""
    match = SPEC_FILE_HEADER_RE.search(prompt_text)
    if not match:
        return None, None
    raw_val = match.group(1).strip().strip("`'\"")
    if not raw_val or raw_val.upper() == "NONE":
        return None, None

    raw_path = Path(os.path.expanduser(raw_val))
    spec_path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    try:
        resolved_path = spec_path.resolve()
    except OSError:
        resolved_path = spec_path

    if type_name.lower() == "implementer":
        if not resolved_path.is_file():
            return None, (
                f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' does not exist. "
                "Have 'research' or root write the grounded spec before dispatching 'implementer'."
            )
        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, (
                f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' could not be read."
            )
        for pattern in GROUNDED_SPEC_SECTION_PATTERNS:
            if not pattern.search(content):
                return None, (
                    f"[ORCHESTRATOR GUARD] SPEC_FILE '{resolved_path.name}' is missing required "
                    "grounded sections (SIGNATURES, PITFALLS/INVARIANTS, VERIFICATION)."
                )

    return str(resolved_path), None


def _validate_and_prepare_subagent_entry(
    entry: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    batch_entries: Optional[List[Dict[str, Any]]] = None,
    exclude_call_sigs: Optional[Set[str]] = None,
    inv_key: str = "0",
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Validate a single subagent spec and return (denial, spawn_record, target_model)."""
    type_name = (entry.get("TypeName") or "").strip()
    norm_type = type_name.lower()
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

    mode = _parse_mode_header(prompt_text)
    cohort_id = _parse_cohort_header(prompt_text)
    if cohort_id and mode is None:
        mode = "best_of_n"
    if mode == "best_of_n" and not cohort_id:
        cohort_id = f"best-of-n-{inv_key}"

    eval_cmd = _parse_eval_cmd_header(prompt_text)
    if eval_cmd is not None and _is_trivial_eval_command(eval_cmd):
        return (
            {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected: '{eval_cmd}'. "
                    "Provide a real test or evaluation script command."
                ),
            },
            None,
            None,
        )

    raw_model = (entry.get("Model") or "").strip()
    role_label = (entry.get("Role") or type_name or "subagent").strip()
    workspace_mode = (entry.get("Workspace") or "inherit").strip()

    if mode == "best_of_n" and workspace_mode.lower() not in ISOLATED_WORKSPACES:
        return (
            {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] MODE: BEST_OF_N requires isolated Workspace='branch' or "
                    f"Workspace='share' for candidate '{role_label}' (got '{workspace_mode}')."
                ),
            },
            None,
            None,
        )

    if norm_type in HIGH_REASONING_READONLY_TYPES:
        if raw_model in ALLOWED_VERIFIER_REVIEWER_MODELS:
            target_model = raw_model
        else:
            target_model = DEFAULT_MODEL_BY_TYPE.get(norm_type, "flash")
    elif mode == "best_of_n":
        if raw_model in {"pro", "flash"}:
            target_model = raw_model
        else:
            target_model = DEFAULT_MODEL_BY_TYPE.get(norm_type, "flash")
    else:
        requested_model = raw_model or "inherit"
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
            target_model = DEFAULT_MODEL_BY_TYPE.get(norm_type, "flash")

        if target_model == "flash" and _is_simple_readonly_task(norm_type, prompt_text):
            target_model = "flash_lite"

    allowed_files = _parse_allowed_files_header(prompt_text)

    normalized_lens: Optional[str] = None
    if norm_type == "reviewer":
        raw_lens = _parse_review_lens_header(prompt_text)
        normalized_lens = _normalize_review_lens(raw_lens or "")
        if not normalized_lens:
            return (
                {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Reviewer '{role_label}' requires a non-empty "
                        "'REVIEW_LENS: <lens>' header (e.g., 'REVIEW_LENS: security')."
                    ),
                },
                None,
                None,
            )

    active_workers = (
        _iter_active_subagents(
            state,
            batch_entries=batch_entries,
            exclude_call_sigs=exclude_call_sigs,
        )
        if state is not None or batch_entries
        else []
    )

    if norm_type == "reviewer" and normalized_lens:
        for other in active_workers:
            other_type = str(
                other.get("type_name") or other.get("type") or ""
            ).strip().lower()
            if other_type != "reviewer":
                continue
            other_lens = _normalize_review_lens(str(other.get("review_lens") or ""))
            if other_lens and other_lens == normalized_lens:
                return (
                    {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] Duplicate REVIEW_LENS '{normalized_lens}' "
                            f"for reviewer '{role_label}'. Each concurrent reviewer must hold "
                            "a distinct REVIEW_LENS."
                        ),
                    },
                    None,
                    None,
                )

    if norm_type == "implementer" and allowed_files:
        for other in active_workers:
            other_type = str(
                other.get("type_name") or other.get("type") or ""
            ).strip().lower()
            if other_type != "implementer":
                continue
            other_allowed = other.get("allowed_files")
            if not other_allowed:
                continue
            other_ws = str(other.get("workspace") or "inherit").strip().lower()
            if workspace_mode.lower() != "inherit" and other_ws != "inherit":
                continue
            overlap = _find_allowed_files_overlap(allowed_files, other_allowed)
            if overlap is not None:
                return (
                    {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] Overlapping ALLOWED_FILES between concurrent "
                            f"implementers sharing 'inherit' workspace ('{overlap[0]}' vs '{overlap[1]}'). "
                            "Assign disjoint ALLOWED_FILES or isolate lanes in 'branch'/'share' workspaces."
                        ),
                    },
                    None,
                    None,
                )

    mutation_epoch = (
        state.get("test_receipt", {}).get("mutation_epoch", 0)
        if isinstance(state, dict)
        else 0
    )
    spawn_record: Dict[str, Any] = {
        "role": role_label,
        "type": type_name,
        "type_name": type_name,
        "model": target_model,
        "workspace": workspace_mode,
        "active_branch": workspace_mode.lower() == "branch",
        "active_worktree": workspace_mode.lower() in ISOLATED_WORKSPACES,
        "status": "active",
        "spawned_mutation_epoch": mutation_epoch,
    }
    if normalized_lens is not None:
        spawn_record["review_lens"] = normalized_lens
    if allowed_files is not None:
        spawn_record["allowed_files"] = allowed_files
    if mode is not None:
        spawn_record["mode"] = mode
    if cohort_id is not None:
        spawn_record["cohort"] = cohort_id
    if eval_cmd is not None:
        spawn_record["eval_cmd"] = eval_cmd
    prompt_hash = _compute_prompt_hash(prompt_text)
    if prompt_hash:
        spawn_record["prompt_hash"] = prompt_hash

    replan_if_match = REPLAN_IF_HEADER_RE.search(prompt_text)
    if replan_if_match:
        spawn_record["replan_if"] = replan_if_match.group(1).strip()

    stage_match = STAGE_HEADER_RE.search(prompt_text)
    if stage_match:
        stage_val = stage_match.group(1).strip()
        spawn_record["stage"] = stage_val
        spawn_record["stage_history"] = [stage_val]

    spec_file, spec_err = _validate_spec_file_header(prompt_text, type_name)
    if spec_err:
        return {"decision": "deny", "reason": spec_err}, None, None
    if spec_file:
        spawn_record["spec_file"] = spec_file

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
    inv_key = _current_invocation_key(payload, state)
    exec_num = payload.get("executionNum", 0)
    caller_cid = payload.get("conversationId", "")

    entry_call_sigs: List[str] = []
    for idx, raw_item in enumerate(entries):
        item = raw_item if isinstance(raw_item, dict) else {}
        sig_call_id = f"{call_id}:{idx}" if is_list_schema else call_id
        t_name = (item.get("TypeName") or "").strip()
        r_label = (item.get("Role") or t_name or "subagent").strip()
        ws_mode = (item.get("Workspace") or "inherit").strip()
        p_hash = _compute_prompt_hash(str(item.get("Prompt") or ""))
        entry_call_sigs.append(
            json.dumps(
                [caller_cid, inv_key, exec_num, sig_call_id, t_name, r_label, ws_mode, p_hash],
                sort_keys=True,
            )
        )
    existing_retry_sigs = {sig for sig in entry_call_sigs if sig in recorded_calls}

    validated_items: List[Tuple[int, Dict[str, Any], Dict[str, Any], str, str]] = []
    batch_entries: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(entries):
        item = raw_item if isinstance(raw_item, dict) else {}
        call_sig = entry_call_sigs[idx]
        denial, spawn_record, target_model = _validate_and_prepare_subagent_entry(
            item,
            state=state,
            batch_entries=batch_entries,
            exclude_call_sigs=existing_retry_sigs,
            inv_key=inv_key,
        )
        if denial is not None or spawn_record is None or target_model is None:
            return denial or {
                "decision": "deny",
                "reason": "Invalid subagent entry."
                if is_list_schema
                else "Invalid invoke_subagent arguments.",
            }
        batch_entries.append(spawn_record)
        validated_items.append((idx, item, spawn_record, target_model, call_sig))

    workspace_roots = [
        str(Path(wp).resolve())
        for wp in (payload.get("workspacePaths") or [])
    ]
    if workspace_roots and state.get("workspace_roots") != workspace_roots:
        state["workspace_roots"] = workspace_roots

    updated_list: List[Dict[str, Any]] = []
    any_model_overwritten = False
    state_changed = False
    for idx, item, spawn_record, target_model, call_sig in validated_items:
        if workspace_roots:
            spawn_record["workspace_roots"] = workspace_roots
        spawn_record["call_id"] = call_id
        spawn_record["call_sig"] = call_sig
        spawn_record["spawned_at"] = time.time()

        if call_sig not in recorded_calls:
            recorded_calls.append(call_sig)
            if spawn_record.get("mode") == "best_of_n" and spawn_record.get("cohort"):
                c_id = spawn_record["cohort"]
                cohorts = state.setdefault("cohorts", {})
                cohort_entry = cohorts.setdefault(
                    c_id,
                    {
                        "total_spawned": 0,
                        "succeeded": [],
                        "failed": [],
                        "created_at": time.time(),
                    },
                )
                cohort_entry["total_spawned"] = (
                    int(cohort_entry.get("total_spawned", 0) or 0) + 1
                )
                spawn_record["candidate_id"] = (
                    f"{c_id}#{cohort_entry['total_spawned']}"
                )
                if spawn_record.get("eval_cmd"):
                    cohort_entry["eval_cmd"] = spawn_record["eval_cmd"]
            state.setdefault("pending_spawns", []).append(spawn_record)
            if spawn_record.get("eval_cmd"):
                state["active_eval_cmd"] = spawn_record["eval_cmd"]
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


def _lookup_recipient_meta(
    payload: Dict[str, Any], state: Dict[str, Any], recipient: str
) -> Optional[Dict[str, Any]]:
    """Resolve recipient subagent metadata from local state or sibling brain state."""
    if not recipient:
        return None
    sub_info = state.get("subagents", {}).get(recipient)
    if isinstance(sub_info, dict):
        return sub_info
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                sib_state_file = (
                    resolved_art.parent / recipient / "orchestrator_state.json"
                )
                if sib_state_file.exists():
                    sib_state = _load_state(sib_state_file)
                    meta = sib_state.get("self_subagent_meta")
                    if isinstance(meta, dict):
                        return meta
        except OSError:
            pass
    return None


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

        err = validate_subagent_report(message, is_scout=_is_scout_subagent(payload, state))
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
    recipient_meta = subagent_info if isinstance(subagent_info, dict) else _lookup_recipient_meta(payload, state, recipient)
    recipient_type = (
        str(recipient_meta.get("type_name") or recipient_meta.get("type") or "").strip().lower()
        if recipient_meta
        else ""
    )

    if recipient_meta and recipient_type in HIGH_REASONING_READONLY_TYPES:
        current_epoch = state.get("test_receipt", {}).get("mutation_epoch", 0)
        spawned_epoch = recipient_meta.get("spawned_mutation_epoch", 0)
        if current_epoch > spawned_epoch:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Stale {recipient_type} '{recipient}': workspace code was "
                    f"mutated since spawn (epoch {spawned_epoch} -> {current_epoch}). "
                    f"Spawn a fresh '{recipient_type}' via invoke_subagent instead of reusing via send_message."
                ),
            }

    spec_file, spec_err = _validate_spec_file_header(message, recipient_type, payload)
    if spec_err:
        return {"decision": "deny", "reason": spec_err}
    if spec_file and subagent_info:
        subagent_info["spec_file"] = spec_file
        _save_state(payload, state)

    replan_if_match = REPLAN_IF_HEADER_RE.search(message)
    if replan_if_match and subagent_info:
        subagent_info["replan_if"] = replan_if_match.group(1).strip()
        _save_state(payload, state)

    stage_match = STAGE_HEADER_RE.search(message)
    if stage_match and subagent_info:
        stage_val = stage_match.group(1).strip()
        subagent_info["stage"] = stage_val
        subagent_info.setdefault("stage_history", []).append(stage_val)
        _save_state(payload, state)

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
    if recipient_type == "implementer" and recipient_meta:
        effective_allowed = (
            parsed_allowed
            if parsed_allowed is not None
            else recipient_meta.get("allowed_files")
        )
        target_ws = str(recipient_meta.get("workspace") or "inherit").strip().lower()
        if effective_allowed:
            for other in _iter_active_subagents(state, exclude_cid=recipient):
                other_type = str(
                    other.get("type_name") or other.get("type") or ""
                ).strip().lower()
                if other_type != "implementer":
                    continue
                other_allowed = other.get("allowed_files")
                if not other_allowed:
                    continue
                other_ws = str(other.get("workspace") or "inherit").strip().lower()
                if target_ws != "inherit" and other_ws != "inherit":
                    continue
                overlap = _find_allowed_files_overlap(effective_allowed, other_allowed)
                if overlap is not None:
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] Overlapping ALLOWED_FILES when reactivating "
                            f"implementer '{recipient}' in 'inherit' workspace ('{overlap[0]}' vs '{overlap[1]}'). "
                            "Wait for conflicting implementer to complete or assign disjoint ALLOWED_FILES."
                        ),
                    }
        if subagent_info is not None:
            subagent_info["status"] = "active"
            _save_state(payload, state)

    if parsed_allowed is not None:
        if subagent_info:
            subagent_info["allowed_files"] = parsed_allowed
            _save_state(payload, state)

    if parsed_allowed is not None or stage_match is not None or (recipient_type == "implementer" and recipient_meta):
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
                            if parsed_allowed is not None:
                                meta["allowed_files"] = parsed_allowed
                            if stage_match is not None:
                                stage_val = stage_match.group(1).strip()
                                meta["stage"] = stage_val
                                meta.setdefault("stage_history", []).append(stage_val)
                            if recipient_type == "implementer":
                                meta["status"] = "active"
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
            if isinstance(meta, dict):
                meta["active_branch"] = False
                meta["active_worktree"] = False
                meta["status"] = "killed"
        for pending in state.get("pending_spawns", []):
            if isinstance(pending, dict):
                pending["active_branch"] = False
                pending["active_worktree"] = False
                pending["status"] = "killed"
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
            if cid in subagents and isinstance(subagents[cid], dict):
                subagents[cid]["active_branch"] = False
                subagents[cid]["active_worktree"] = False
                subagents[cid]["status"] = "killed"
        remaining_pending = []
        killed_left = len(target_ids)
        for pending in state.get("pending_spawns", []):
            if (pending.get("active_worktree") or pending.get("active_branch")) and killed_left > 0:
                pending["active_branch"] = False
                pending["active_worktree"] = False
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
    if tool_name in (SEARCH_TOOLS | READ_TOOLS | DIR_LIST_TOOLS | MUTATION_TOOLS):
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

    if state.get("plan_execute_phase"):
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
            if target:
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
            return None

        if tool_name in DIR_LIST_TOOLS:
            target = _extract_target_path(tool_name, args)
            if target:
                norm_slashes = str(target).replace("\\", "/")
                if FORBIDDEN_DIRECT_READ_RE.search(norm_slashes):
                    return {
                        "decision": "deny",
                        "reason": (
                            "[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
                            f"directly list or read log or transcript directory/file '{Path(target).name}'. "
                            "Delegate log slicing to a 'scout' subagent (Model='flash_lite')."
                        ),
                    }
            return None

        if tool_name in MUTATION_TOOLS:
            target = _extract_target_path(tool_name, args)
            plan_allowed = state.get("plan_allowed_files")
            if (
                plan_allowed is not None
                and target
                and not _is_artifact_or_worktree_path(target, payload)
            ):
                if not _is_target_in_allowed_files(
                    target, plan_allowed, payload.get("workspacePaths")
                ):
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] File allowlist violation: plan execution attempted to mutate "
                            f"'{Path(target).name}', which is outside plan ALLOWED_FILES "
                            f"({', '.join(plan_allowed) if plan_allowed else 'NONE'})."
                        ),
                    }
            return None

        return None

    inv_key = _current_invocation_key(payload, state)
    budget = state.setdefault("direct_budget", {}).setdefault(
        inv_key,
        {
            "mutated_files": [],
            "read_files": [],
            "read_ranges": [],
            "read_lines": 0,
            "read_bytes": 0,
            "commands": [],
            "seen_calls": [],
            "test_runs_after_edit": 0,
        },
    )
    call_id = _extract_call_id(payload)

    if tool_name in DIR_LIST_TOOLS:
        target = _extract_target_path(tool_name, args)
        if target:
            norm_slashes = str(target).replace("\\", "/")
            if FORBIDDEN_DIRECT_READ_RE.search(norm_slashes):
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
                        f"directly list or read log or transcript directory/file '{Path(target).name}'. "
                        "Delegate log slicing to a 'scout' subagent (Model='flash_lite')."
                    ),
                }
        return None

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

        start_line = (
            args.get("StartLine")
            if args.get("StartLine") is not None
            else args.get("start_line")
            if args.get("start_line") is not None
            else args.get("StartLineNumber")
        )
        end_line = (
            args.get("EndLine")
            if args.get("EndLine") is not None
            else args.get("end_line")
            if args.get("end_line") is not None
            else args.get("EndLineNumber")
        )

        read_ranges: List[str] = budget.setdefault("read_ranges", [])
        read_files: List[str] = budget.setdefault("read_files", [])
        range_key = f"{norm_target}:{start_line or ''}:{end_line or ''}"
        full_key = f"{norm_target}::"
        if range_key in read_ranges or full_key in read_ranges:
            return None

        p = Path(norm_target)
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                all_lines = text.splitlines()
            except OSError:
                all_lines = []
            if start_line is not None or end_line is not None:
                s = max(1, int(start_line or 1))
                e = min(len(all_lines), int(end_line or len(all_lines)))
                if s <= e:
                    sliced = all_lines[s - 1 : e]
                    lines_cost = len(sliced)
                    bytes_cost = sum(len(x.encode("utf-8")) + 1 for x in sliced)
                else:
                    lines_cost = 0
                    bytes_cost = 0
            else:
                lines_cost = min(800, len(all_lines))
                try:
                    bytes_cost = min(46080, p.stat().st_size)
                except OSError:
                    bytes_cost = sum(len(x.encode("utf-8")) + 1 for x in all_lines)
        else:
            if start_line is not None and end_line is not None:
                lines_cost = max(1, int(end_line) - int(start_line) + 1)
                bytes_cost = lines_cost * 40
            else:
                lines_cost = DEFAULT_UNBOUNDED_READ_LINES
                bytes_cost = DEFAULT_UNBOUNDED_READ_BYTES

        current_lines = budget.setdefault("read_lines", 0)
        current_bytes = budget.setdefault("read_bytes", 0)
        file_is_new = norm_target not in read_files
        if (
            current_lines + lines_cost > MAX_DIRECT_READ_LINES
            or current_bytes + bytes_cost > MAX_DIRECT_READ_BYTES
            or (file_is_new and len(read_files) >= MAX_DIRECT_READ_FILES)
        ):
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Direct execution fast-path read budget "
                    f"({MAX_DIRECT_READ_LINES} lines / {MAX_DIRECT_READ_BYTES} bytes / "
                    f"{MAX_DIRECT_READ_FILES} workspace files) exceeded when reading "
                    f"'{Path(norm_target).name}'. Pass StartLine/EndLine for targeted inspection "
                    "or delegate to a 'scout' (Model='flash_lite') or 'research' (Model='flash') subagent."
                ),
            }

        read_ranges.append(range_key)
        if file_is_new:
            read_files.append(norm_target)
        budget["read_lines"] = current_lines + lines_cost
        budget["read_bytes"] = current_bytes + bytes_cost
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
        is_test = _is_test_command(cmd) or _is_declared_eval_command(
            cmd, state, payload
        )
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
            _current_subagent_meta,
            _get_circuit_breaker,
            _get_test_receipt,
            _is_code_file,
            _propagate_receipt_to_parent_brain,
        )

        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}

        if tool_name == "exit_plan_mode" and _is_root_orchestrator(payload, state):
            plan_text = str(args.get("plan") or args.get("Plan") or "")
            plan_eval = _parse_eval_cmd_header(plan_text)
            if plan_eval is not None and _is_trivial_eval_command(plan_eval):
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in plan: '{plan_eval}'. "
                        "Provide a real test or evaluation script command."
                    ),
                }
            state["active"] = True
            state["session_evaluated"] = True
            state["plan_execute_phase"] = True
            state["plan_execute_consumed"] = False
            state["plan_execute_user_input"] = (
                str(payload.get("lastUserInput") or "").strip() or None
            )
            state["plan_execute_turn_finished"] = False
            state["plan_allowed_files"] = _parse_allowed_files_header(plan_text)
            state["active_eval_cmd"] = plan_eval
            _save_state(payload, state)
            return {"decision": "allow"}

        if tool_name == "enter_plan_mode" and _is_root_orchestrator(payload, state):
            state["plan_execute_phase"] = False
            state["plan_execute_consumed"] = False
            state["plan_execute_user_input"] = None
            state["plan_execute_turn_finished"] = False
            state["plan_allowed_files"] = None
            state["active_eval_cmd"] = None
            _save_state(payload, state)
            return {"decision": "allow"}

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
                        bool(SAFE_GIT_INSPECTION_RE.match(c))
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
            elif tool_name in {"invoke_subagent", "send_message", "run_workflow"}:
                if tool_name == "send_message" and not _is_root_orchestrator(payload, state):
                    msg_text = str(args.get("Message") or "")
                    if re.search(r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED|NEEDS_REPLAN)\b", msg_text):
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
                    replan_found = False
                    if tool_name == "invoke_subagent":
                        entries = (
                            args["Subagents"]
                            if isinstance(args.get("Subagents"), list)
                            else [args]
                        )
                        for item in entries:
                            if isinstance(item, dict) and REPLAN_HEADER_RE.search(
                                str(item.get("Prompt") or "")
                            ):
                                replan_found = True
                                break
                    elif tool_name == "send_message":
                        replan_found = bool(
                            REPLAN_HEADER_RE.search(str(args.get("Message") or ""))
                        )
                    elif tool_name == "run_workflow":
                        replan_found = bool(
                            REPLAN_HEADER_RE.search(str(args.get("Description") or ""))
                            or REPLAN_HEADER_RE.search(str(args.get("Script") or ""))
                        )

                    replan_count = cb.get("replan_count", 0)
                    if replan_found and replan_count < MAX_REPLAN_ATTEMPTS:
                        cb["replan_count"] = replan_count + 1
                        cb["tripped"] = False
                        cb["failure_streak"] = 0
                        _save_state(payload, state)
                    elif replan_count < MAX_REPLAN_ATTEMPTS:
                        return {
                            "decision": "deny",
                            "reason": (
                                f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                                f"{cb.get('failure_streak', 2)} consecutive failures "
                                f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                                "Stop retrying, write failure details to the scratch log if needed, "
                                "and report STATUS: BLOCKED, or include 'REPLAN: <revised plan rationale>' "
                                "to use the 1-shot re-plan budget."
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

        roots = _allowed_roots(payload)

        if tool_name in (SEARCH_TOOLS | DIR_LIST_TOOLS):
            budget_denial = _check_and_record_direct_budget(payload, tool_name, args, state)
            if budget_denial:
                return budget_denial
            if tool_name in DIR_LIST_TOOLS:
                target_path = _extract_target_path(tool_name, args) or str(PROJECT_ROOT)
                if _is_within_allowed_roots(target_path, roots):
                    return {"decision": "allow"}
                return {"decision": "ask"}

        if tool_name == "define_subagent":
            return _handle_define_subagent(args)

        if tool_name == "invoke_subagent":
            return _handle_invoke_subagent(payload, args, state)

        if tool_name == "send_message":
            return _handle_send_message(payload, args, state)

        if tool_name == "manage_subagents":
            return _handle_manage_subagents(payload, args, state)

        if tool_name in MUTATION_TOOLS:
            if _is_strict_readonly_subagent(payload, state):
                sub_type = _get_subagent_type(payload, state) or "readonly"
                if sub_type == "scout":
                    reason = (
                        "[ORCHESTRATOR GUARD] Scout subagents (Model='flash_lite') are strictly "
                        "read-only. File mutations are prohibited."
                    )
                else:
                    reason = (
                        f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly "
                        "read-only. File mutations are prohibited."
                    )
                return {
                    "decision": "deny",
                    "reason": reason,
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

            if _is_strict_readonly_subagent(payload, state):
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
                    sub_type = _get_subagent_type(payload, state) or "readonly"
                    if sub_type == "scout":
                        reason = (
                            "[ORCHESTRATOR GUARD] Scout subagents (Model='flash_lite') are strictly "
                            "read-only. Shell file mutations are prohibited."
                        )
                    else:
                        reason = (
                            f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly "
                            "read-only. Shell file mutations are prohibited."
                        )
                    return {
                        "decision": "deny",
                        "reason": reason,
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

        if tool_name == "run_workflow":
            cwd = args.get("Cwd")
            if cwd and not _is_within_allowed_roots(str(cwd), roots):
                return {"decision": "ask"}
            script_path = args.get("ScriptPath")
            if script_path and not _is_within_allowed_roots(str(script_path), roots):
                return {"decision": "ask"}
            eval_cmd = _parse_eval_cmd_header(str(args.get("Description") or ""))
            if eval_cmd is None and args.get("Script"):
                uncommented_script = re.sub(
                    r"(?m)^[ \t]*#+[ \t]*", "", str(args.get("Script") or "")
                )
                eval_cmd = _parse_eval_cmd_header(uncommented_script)
            if eval_cmd is not None and _is_trivial_eval_command(eval_cmd):
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in run_workflow: '{eval_cmd}'. "
                        "Provide a real test or evaluation script command."
                    ),
                }
            if eval_cmd:
                state["active_eval_cmd"] = eval_cmd
                _save_state(payload, state)
            return {"decision": "allow"}

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
                        receipt["mutation_epoch"] = receipt.get("mutation_epoch", 0) + 1
                        try:
                            receipt["last_mutated_file"] = str(
                                Path(os.path.expanduser(target_path)).resolve()
                            )
                        except OSError:
                            receipt["last_mutated_file"] = target_path
                        _save_state(payload, state)
                        if (
                            not _is_root_orchestrator(payload, state)
                            and _current_subagent_meta(payload, state).get("mode")
                            != "best_of_n"
                        ):
                            _propagate_receipt_to_parent_brain(
                                payload,
                                dirty=True,
                                verified=False,
                                last_mutated_file=receipt["last_mutated_file"],
                            )
                return {"decision": "allow"}
            return {"decision": "ask"}

        return {"decision": "allow"}

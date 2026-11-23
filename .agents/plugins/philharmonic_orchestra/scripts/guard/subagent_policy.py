"""Subagent dispatch/messaging/swarm policies, direct execution fast-path budget, and workflow validation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from guard.ast_guard import _is_test_command
from guard.constants import (
    ALLOWED_BEST_OF_N_TIERS,
    ALLOWED_COUNCIL_PROPOSER_TIERS,
    ALLOWED_HIGH_REASONING_TIERS,
    ALLOWED_STANDARD_SUBAGENT_TIERS,
    COHORT_HEADER_RE,
    COUNCIL_ID_HEADER_RE,
    COUNCIL_ROLE_HEADER_RE,
    DEFAULT_TIER_BY_TYPE,
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
    MODE_HEADER_RE,
    MUTATION_TOOLS,
    MUTATION_VERBS_RE,
    NUMBERED_RULE_RE,
    PENDING_SPAWN_TTL_SECONDS,
    PROJECT_ROOT,
    READONLY_NON_VERIFICATION_TYPES,
    READ_TOOLS,
    REPLAN_IF_HEADER_RE,
    REVIEW_LENS_HEADER_RE,
    SCRATCH_ONLY_SUBAGENT_TYPES,
    SEARCH_TOOLS,
    SPEC_FILE_HEADER_RE,
    STAGE_HEADER_RE,
    STRICT_READONLY_SUBAGENT_TYPES,
    _extract_call_id,
    get_blind_judge_leak_patterns,
    get_council_name_leak_re,
    get_model_alias_map,
    get_model_for_tier,
    get_model_tier_map,
    infer_subagent_tier,
    resolve_model_tiers,
    select_target_tier,
)
from guard.identity import (
    _get_subagent_type,
    _is_root_orchestrator,
    _is_scout_subagent,
    _is_scratch_only_subagent,
    _is_strict_readonly_subagent,
    _is_within_allowed_roots,
    _read_sibling_snapshot,
    _sync_subagents_from_transcript,
)
from guard.path_guards import (
    _extract_target_path,
    _find_allowed_files_overlap,
    _get_subagent_allowed_files,
    _is_artifact_or_worktree_path,
    _is_artifact_path,
    _is_guard_plugin_cwd,
    _is_guard_plugin_target,
    _is_guard_state_target,
    _is_target_in_allowed_files,
    _parse_allowed_files_header,
    _resolve_cmd_target,
)
from guard.shell_targets import _validate_shell_allowlist
from guard.shell_tokens import (
    _is_declared_eval_command,
    _is_safe_git_inspection_command,
    _is_trivial_eval_command,
    _is_unsafe_eval_command,
    _split_shell_commands,
)
from guard.state_store import _current_invocation_key, _has_other_locked_path, _load_state, _save_state, state_transaction


def _deny(reason: str) -> Dict[str, Any]:
    return {"decision": "deny", "reason": reason}


def _entry_deny(reason: str) -> Tuple[Dict[str, Any], None, None]:
    return _deny(reason), None, None


def _parse_header(
    text: str,
    pattern: re.Pattern[str],
    *,
    drop_none: bool = True,
    norm_sep: Optional[str] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Extract and optionally normalize a single-line header value."""
    if not text:
        return None
    match = pattern.search(text)
    if not match:
        return None
    raw_val = match.group(1).strip().strip("`'\"")
    if drop_none and (not raw_val or raw_val.upper() == "NONE"):
        return None
    if norm_sep is not None:
        sep_re = r"[\s\-]+" if norm_sep == "_" else r"[\s_-]+"
        if aliases and norm_sep == "_":
            sep_re = r"[\s_-]+"
        cleaned = re.sub(sep_re, norm_sep, raw_val.lower()).strip(norm_sep)
        if aliases:
            cleaned = aliases.get(cleaned, cleaned)
        return cleaned or None
    return raw_val


def _parse_review_lens_header(prompt_text: str) -> Optional[str]:
    return _parse_header(prompt_text, REVIEW_LENS_HEADER_RE, drop_none=False)


def _normalize_review_lens(lens: str) -> str:
    """Normalize a review lens string so spaces, hyphens, and underscores collide cleanly."""
    return re.sub(r"[\s_-]+", "-", (lens or "").strip().lower()).strip("-")


def _parse_mode_header(prompt_text: str) -> Optional[str]:
    return _parse_header(
        prompt_text,
        MODE_HEADER_RE,
        norm_sep="_",
        aliases={"bestofn": "best_of_n", "council_of_models": "council", "councilofmodels": "council"},
    )


def _parse_cohort_header(prompt_text: str) -> Optional[str]:
    return _parse_header(prompt_text, COHORT_HEADER_RE, norm_sep="-")


def _parse_council_id_header(text: str) -> Optional[str]:
    return _parse_header(text, COUNCIL_ID_HEADER_RE, norm_sep="-")


def _parse_council_role_header(text: str) -> Optional[str]:
    return _parse_header(
        text,
        COUNCIL_ROLE_HEADER_RE,
        norm_sep="_",
        aliases={
            "proposal": "proposer",
            "member": "proposer",
            "blind_judge": "judge",
            "voter": "judge",
            "synthesis": "synthesizer",
            "synth": "synthesizer",
        },
    )


def _parse_eval_cmd_header(prompt_text: str) -> Optional[str]:
    """Parse EVAL_CMD: header from a prompt or plan, returning None if omitted."""
    if not prompt_text:
        return None
    match = EVAL_CMD_HEADER_RE.search(prompt_text)
    if not match:
        return None
    raw_val = match.group(1).strip()
    if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in {"`", "'", '"'}:
        raw_val = raw_val[1:-1].strip()
    return None if (not raw_val or raw_val.upper() == "NONE") else raw_val


def _check_mode_header_conflicts(
    mode: Optional[str],
    cohort_id: Optional[str],
    council_id: Optional[str],
    council_role: Optional[str],
    context: str = "",
) -> Optional[str]:
    if (
        (mode is not None and mode != "best_of_n" and cohort_id)
        or (mode is not None and mode != "council" and (council_id or council_role))
        or (cohort_id and (council_id or council_role))
    ):
        loc = f" in {context}" if context else ""
        return (
            f"[ORCHESTRATOR GUARD] Conflicting mode headers{loc}: cannot mix "
            "MODE: BEST_OF_N / COHORT with MODE: COUNCIL / COUNCIL_ID / COUNCIL_ROLE "
            "or attach COHORT/COUNCIL headers to an incompatible MODE."
        )
    return None


def _resolve_scratch_doc_candidates(text: str, artifact_dir: str) -> List[Path]:
    """Return existing scratch .md/.txt/.json file paths referenced in text within artifact_dir/scratch."""
    try:
        resolved_art = Path(artifact_dir).resolve()
        scratch_root = (resolved_art / "scratch").resolve()
    except (OSError, ValueError):
        return []
    seen: Set[Path] = set()
    results: List[Path] = []
    for match in re.finditer(r"(?:file://)?([^\s`'\"<>(),;:]+\.(?:md|txt|json))", text, re.I):
        try:
            tok_path = Path(os.path.expanduser(match.group(1)))
        except (OSError, ValueError):
            continue
        to_check = [tok_path] if tok_path.is_absolute() else [resolved_art / tok_path, PROJECT_ROOT / tok_path]
        for cand in to_check:
            try:
                resolved_cand = cand.resolve()
                resolved_cand.relative_to(scratch_root)
                is_file = resolved_cand.is_file()
            except (OSError, ValueError):
                continue
            if resolved_cand not in seen and is_file:
                seen.add(resolved_cand)
                results.append(resolved_cand)
    return results


def _match_blind_judge_patterns(text: str) -> Optional[str]:
    """Check raw and markdown-stripped text against active blind-judge leak patterns."""
    if not text:
        return None
    normalized = re.sub(r"[*_`]+", "", text)
    patterns = get_blind_judge_leak_patterns()
    for candidate_text in ([text] if normalized == text else [text, normalized]):
        for pattern in patterns:
            if m := pattern.search(candidate_text):
                return m.group(0)
    return None


def _detect_blind_judge_leak(text: str, payload: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return the matched attribution string if text or referenced scratch docs leak model identity to a judge, else None."""
    if not text:
        return None
    if direct_match := _match_blind_judge_patterns(text):
        return direct_match
    artifact_dir = payload.get("artifactDirectoryPath") if isinstance(payload, dict) else None
    if not artifact_dir:
        return None
    for candidate in _resolve_scratch_doc_candidates(text, str(artifact_dir)):
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")[:65536]
        except (OSError, ValueError):
            continue
        if file_match := _match_blind_judge_patterns(content):
            return f"{candidate.name}: {file_match}"
    return None


def _detect_council_name_leak(name: str) -> Optional[str]:
    """Return the matched token if a council worker name reveals model identity, else None."""
    m = get_council_name_leak_re().search(name) if name else None
    return m.group(0) if m else None


def _expand_swarm_worker_names(args: Dict[str, Any]) -> List[str]:
    """Expand swarm worker name(s) from 'name', 'tasks', and 'n' arguments."""
    name = str(args.get("name") or "").strip()
    if not name:
        return []
    tasks, n = args.get("tasks"), args.get("n")
    if isinstance(tasks, list) and len(tasks) > 0:
        return [f"{name}-{i}" for i in range(1, len(tasks) + 1)]
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        return [f"{name}-{i}" for i in range(1, n + 1)]
    return [name]


def _record_council_member(
    state: Dict[str, Any],
    council_id: str,
    member_name: str,
    council_role: Optional[str],
    model: str,
    is_judge_type: bool = False,
) -> None:
    """Record a proposer/judge member and model in state['councils'][council_id]."""
    council_entry = state.setdefault("councils", {}).setdefault(
        council_id, {"proposers": [], "judges": [], "models": [], "created_at": time.time()}
    )
    proposers = council_entry.setdefault("proposers", [])
    judges = council_entry.setdefault("judges", [])
    synthesizers = council_entry.setdefault("synthesizers", [])
    models = council_entry.setdefault("models", [])
    if council_role == "judge" or (council_role is None and is_judge_type):
        bucket = judges
    elif council_role == "synthesizer":
        bucket = synthesizers
    elif council_role == "proposer" or (council_role is None and not is_judge_type):
        bucket = proposers
    else:
        bucket = None
    if bucket is not None and member_name and member_name not in bucket:
        bucket.append(member_name)
    if model and model not in models:
        models.append(model)


def _compute_prompt_hash(prompt_text: str) -> str:
    """Return a short deterministic SHA-256 prefix for non-empty subagent prompts."""
    if not prompt_text:
        return ""
    return hashlib.sha256(str(prompt_text or "").encode("utf-8", errors="surrogatepass")).hexdigest()[:12]


def _iter_active_subagents(
    state: Dict[str, Any],
    batch_entries: Optional[List[Dict[str, Any]]] = None,
    exclude_cid: Optional[str] = None,
    exclude_call_sigs: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Collect active worker records from batch_entries, non-expired pending_spawns, and active subagents."""
    active: List[Dict[str, Any]] = [e for e in (batch_entries or []) if isinstance(e, dict)]
    now = time.time()
    for pending in (state.get("pending_spawns") or []):
        if not isinstance(pending, dict):
            continue
        if exclude_call_sigs and pending.get("call_sig") in exclude_call_sigs:
            continue
        if pending.get("status", "active") != "active":
            continue
        if (spawned_at := pending.get("spawned_at")) is not None:
            try:
                if (now - float(spawned_at)) > PENDING_SPAWN_TTL_SECONDS:
                    continue
            except (ValueError, TypeError):
                pass
        active.append(pending)

    norm_exclude = (exclude_cid or "").strip().lower()
    raw_subs = state.get("subagents")
    if isinstance(raw_subs, dict):
        for cid, entry in raw_subs.items():
            if isinstance(entry, dict) and (not norm_exclude or str(cid).strip().lower() != norm_exclude):
                if entry.get("status") == "active":
                    active.append(entry)
    return active


def _handle_define_subagent(args: Dict[str, Any]) -> Dict[str, Any]:
    overwrite: Dict[str, Any] = {}
    sub_name = str(args.get("Name") or args.get("name") or "").strip().lower()
    flag_rules = (
        ("EnableSubagentTools", "enable_subagent_tools", True),
        ("EnableWriteTools", "enable_write_tools", sub_name in STRICT_READONLY_SUBAGENT_TYPES),
        ("EnableMCPTools", "enable_mcp_tools", sub_name in {"scout", "implementer", "verifier", "reviewer", "judge", "council"}),
    )
    for pascal_key, snake_key, apply in flag_rules:
        if apply:
            for key in (pascal_key, snake_key):
                if args.get(key) is True:
                    overwrite[key] = False
    return {"decision": "allow", "overwrite": overwrite} if overwrite else {"decision": "allow"}


def _is_simple_readonly_task(type_name: str, prompt_text: str) -> bool:
    """Return True if a read-only subagent prompt has <= 3 rules and no mutation verbs."""
    norm_type = (type_name or "").strip().lower()
    if norm_type in HIGH_REASONING_READONLY_TYPES or norm_type not in {"scout", "research", "researcher"}:
        return False
    rule_count = len(NUMBERED_RULE_RE.findall(prompt_text))
    cleaned_prompt = re.sub(r"(?im)^\s*(?:\d+[.)]\s*)?do\s+not\s+.*$", "", prompt_text)
    return not MUTATION_VERBS_RE.search(cleaned_prompt) and rule_count <= 3


def _validate_spec_file_header(
    prompt_text: str, type_name: str, payload: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Validate SPEC_FILE: header. Returns (resolved_spec_path_str, error_reason)."""
    raw_val = _parse_header(prompt_text, SPEC_FILE_HEADER_RE)
    if not raw_val:
        return None, None
    try:
        raw_path = Path(os.path.expanduser(raw_val))
        resolved_path = (raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)).resolve()
    except (OSError, ValueError):
        return None, f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' is not a valid path."

    if type_name.lower() == "implementer":
        try:
            is_file = resolved_path.is_file()
        except (OSError, ValueError):
            is_file = False
        if not is_file:
            return None, (
                f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' does not exist. "
                "Have 'research' or root write the grounded spec before dispatching 'implementer'."
            )
        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None, f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' could not be read."
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
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Validate a single subagent spec and return (denial, spawn_record, target_model)."""
    type_name = str(entry.get("TypeName") or "").strip()
    norm_type = type_name.lower()
    raw_prompt = entry.get("Prompt")
    prompt_text = "" if raw_prompt is None else str(raw_prompt)
    if not type_name:
        return _entry_deny("invoke_subagent requires a non-empty 'TypeName'.")
    if not prompt_text or not prompt_text.strip():
        return _entry_deny("invoke_subagent requires a non-empty 'Prompt'.")
    if "\x00" in prompt_text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in prompt_text):
        return _entry_deny("[ORCHESTRATOR GUARD] Subagent Prompt contains disallowed NUL byte or lone UTF-16 surrogate.")

    parsed_mode = _parse_mode_header(prompt_text)
    cohort_id = _parse_cohort_header(prompt_text)
    council_id = _parse_council_id_header(prompt_text)
    council_role = _parse_council_role_header(prompt_text)
    if err := _check_mode_header_conflicts(parsed_mode, cohort_id, council_id, council_role):
        return _entry_deny(err)

    mode = "best_of_n" if (cohort_id and parsed_mode is None) else parsed_mode
    if mode == "best_of_n":
        if norm_type != "implementer":
            return _entry_deny(f"[ORCHESTRATOR GUARD] MODE: BEST_OF_N is only allowed for 'implementer' subagents (got '{type_name}').")
        cohort_id = cohort_id or f"best-of-n-{inv_key}"

    role_label = str(entry.get("Role") or type_name or "subagent").strip()
    role_lower = role_label.lower()
    if norm_type == "council" and ("judge" in role_lower or "arbiter" in role_lower):
        mode = mode or "council"
        council_role = council_role or "judge"
    if (council_id or council_role) and mode is None:
        mode = "council"
    if mode == "council":
        if norm_type == "implementer":
            return _entry_deny("[ORCHESTRATOR GUARD] MODE: COUNCIL is not allowed for 'implementer' subagents. Use 'research', 'council', or 'judge'.")
        council_id = council_id or f"council-{inv_key}"

    is_blind_judge = (
        norm_type == "judge"
        or council_role == "judge"
        or (mode == "council" and (norm_type == "reviewer" or council_role not in {"proposer", None}))
        or (
            (norm_type in {"judge", "reviewer", "council", "research", "researcher"} or mode == "council")
            and (
                "judge" in role_lower or "arbiter" in role_lower or "synthesizer" in role_lower
                or council_role in {"judge", "arbiter", "synthesizer"}
            )
        )
    )
    if is_blind_judge and (leak := _detect_blind_judge_leak(prompt_text, payload=payload)):
        return _entry_deny(f"[ORCHESTRATOR GUARD] Blind judging violation: judge prompt leaks model identity ('{leak}'). Anonymize proposals before invoking judge.")

    eval_cmd = _parse_eval_cmd_header(prompt_text)
    if eval_cmd is not None and _is_trivial_eval_command(eval_cmd):
        return _entry_deny(f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected: '{eval_cmd}'. Provide a real test or evaluation script command.")

    raw_model = str(entry.get("Model") or "").strip()
    workspace_mode = str(entry.get("Workspace") or "inherit").strip()
    if mode == "best_of_n" and workspace_mode.lower() not in ISOLATED_WORKSPACES:
        return _entry_deny(
            f"[ORCHESTRATOR GUARD] MODE: BEST_OF_N requires isolated Workspace='branch' or "
            f"Workspace='share' for candidate '{role_label}' (got '{workspace_mode}')."
        )

    tier_map, alias_map = get_model_tier_map(), get_model_alias_map()
    default_tier = DEFAULT_TIER_BY_TYPE.get(norm_type, "standard")
    matched_tiers = resolve_model_tiers(raw_model, tier_map=tier_map, alias_map=alias_map)
    if norm_type in HIGH_REASONING_READONLY_TYPES:
        target_tier = select_target_tier(matched_tiers, ALLOWED_HIGH_REASONING_TIERS, default_tier)
    elif mode == "best_of_n":
        target_tier = select_target_tier(matched_tiers, ALLOWED_BEST_OF_N_TIERS, default_tier)
    elif (parsed_mode == "council" and norm_type in SCRATCH_ONLY_SUBAGENT_TYPES) or norm_type == "council":
        target_tier = select_target_tier(matched_tiers, ALLOWED_COUNCIL_PROPOSER_TIERS, default_tier)
    else:
        if "frontier" in matched_tiers and not (matched_tiers & ALLOWED_STANDARD_SUBAGENT_TIERS):
            std_wire, lite_wire = get_model_for_tier("standard", tier_map), get_model_for_tier("lite", tier_map)
            allowed_desc = f"'{std_wire}'" if std_wire == lite_wire else f"'{std_wire}' or '{lite_wire}'"
            return _entry_deny(
                f"Subagent '{role_label}' requested Model='{raw_model}' (frontier tier). Subagents must use {allowed_desc}."
            )
        target_tier = select_target_tier(matched_tiers, ALLOWED_STANDARD_SUBAGENT_TIERS, default_tier)
        if target_tier == "standard" and _is_simple_readonly_task(norm_type, prompt_text):
            target_tier = "lite"
    target_model = get_model_for_tier(target_tier, tier_map)

    allowed_files = _parse_allowed_files_header(prompt_text)
    normalized_lens: Optional[str] = None
    if norm_type == "reviewer":
        normalized_lens = _normalize_review_lens(_parse_review_lens_header(prompt_text) or "")
        if not normalized_lens:
            return _entry_deny(f"[ORCHESTRATOR GUARD] Reviewer '{role_label}' requires a non-empty 'REVIEW_LENS: <lens>' header (e.g., 'REVIEW_LENS: security').")

    active_workers = (
        _iter_active_subagents(state, batch_entries=batch_entries, exclude_call_sigs=exclude_call_sigs)
        if state is not None or batch_entries
        else []
    )
    if norm_type == "reviewer" and normalized_lens:
        for other in active_workers:
            if str(other.get("type_name") or other.get("type") or "").strip().lower() == "reviewer":
                if _normalize_review_lens(str(other.get("review_lens") or "")) == normalized_lens:
                    return _entry_deny(
                        f"[ORCHESTRATOR GUARD] Duplicate REVIEW_LENS '{normalized_lens}' for reviewer '{role_label}'. "
                        "Each concurrent reviewer must hold a distinct REVIEW_LENS."
                    )

    if norm_type == "implementer" and allowed_files:
        for other in active_workers:
            if str(other.get("type_name") or other.get("type") or "").strip().lower() != "implementer":
                continue
            other_allowed = other.get("allowed_files")
            if not other_allowed:
                continue
            other_ws = str(other.get("workspace") or "inherit").strip().lower()
            if workspace_mode.lower() != "inherit" and other_ws != "inherit":
                continue
            if (overlap := _find_allowed_files_overlap(allowed_files, other_allowed)) is not None:
                return _entry_deny(
                    f"[ORCHESTRATOR GUARD] Overlapping ALLOWED_FILES between concurrent implementers sharing 'inherit' "
                    f"workspace ('{overlap[0]}' vs '{overlap[1]}'). Assign disjoint ALLOWED_FILES or isolate lanes in 'branch'/'share' workspaces."
                )

    mutation_epoch = state.get("test_receipt", {}).get("mutation_epoch", 0) if isinstance(state, dict) else 0
    spawn_record: Dict[str, Any] = {
        "role": role_label,
        "type": type_name,
        "type_name": type_name,
        "model": target_model,
        "model_tier": target_tier,
        "workspace": workspace_mode,
        "active_branch": workspace_mode.lower() == "branch",
        "active_worktree": workspace_mode.lower() in ISOLATED_WORKSPACES,
        "status": "active",
        "spawned_mutation_epoch": mutation_epoch,
    }
    for key, val in (
        ("review_lens", normalized_lens),
        ("allowed_files", allowed_files),
        ("mode", mode),
        ("cohort", cohort_id),
        ("council_id", council_id),
        ("council_role", council_role),
        ("eval_cmd", eval_cmd),
    ):
        if val is not None:
            spawn_record[key] = val
    if prompt_hash := _compute_prompt_hash(prompt_text):
        spawn_record["prompt_hash"] = prompt_hash
    if replan_if_match := REPLAN_IF_HEADER_RE.search(prompt_text):
        spawn_record["replan_if"] = replan_if_match.group(1).strip()
    if stage_match := STAGE_HEADER_RE.search(prompt_text):
        stage_val = stage_match.group(1).strip()
        spawn_record["stage"] = stage_val
        spawn_record["stage_history"] = [stage_val]

    spec_file, spec_err = _validate_spec_file_header(prompt_text, type_name)
    if spec_err:
        return _entry_deny(spec_err)
    if spec_file:
        spawn_record["spec_file"] = spec_file
    return None, spawn_record, target_model


def _handle_invoke_subagent(
    payload: Dict[str, Any], args: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if state is None:
        state = _load_state(payload)
    call_id = _extract_call_id(payload)
    recorded_calls: List[str] = state.setdefault("recorded_spawn_calls", [])
    is_list_schema = "Subagents" in args and isinstance(args.get("Subagents"), list)
    if is_list_schema and not args["Subagents"]:
        return _deny("invoke_subagent requires a non-empty 'Subagents' list.")

    entries = args["Subagents"] if is_list_schema else [args]
    inv_key = _current_invocation_key(payload, state)
    exec_num = payload.get("executionNum", 0)
    caller_cid = str(payload.get("conversationId") or "").strip().lower()

    entry_call_sigs: List[str] = []
    for idx, raw_item in enumerate(entries):
        item = raw_item if isinstance(raw_item, dict) else {}
        sig_call_id = f"{call_id}:{idx}" if is_list_schema else call_id
        t_name = str(item.get("TypeName") or "").strip()
        entry_call_sigs.append(
            json.dumps(
                [
                    caller_cid,
                    inv_key,
                    exec_num,
                    sig_call_id,
                    t_name,
                    str(item.get("Role") or t_name or "subagent").strip(),
                    str(item.get("Workspace") or "inherit").strip(),
                    str(item.get("Model") or "").strip(),
                    _compute_prompt_hash(str(item.get("Prompt") or "")),
                ],
                sort_keys=True,
            )
        )
    existing_retry_sigs = {sig for sig in entry_call_sigs if sig in recorded_calls}

    validated_items: List[Tuple[int, Dict[str, Any], Dict[str, Any], str, str]] = []
    batch_entries: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(entries):
        item = raw_item if isinstance(raw_item, dict) else {}
        denial, spawn_record, target_model = _validate_and_prepare_subagent_entry(
            item,
            state=state,
            batch_entries=batch_entries,
            exclude_call_sigs=existing_retry_sigs,
            inv_key=inv_key,
            payload=payload,
        )
        if denial is not None or spawn_record is None or target_model is None:
            return denial or _deny("Invalid subagent entry." if is_list_schema else "Invalid invoke_subagent arguments.")
        batch_entries.append(spawn_record)
        validated_items.append((idx, item, spawn_record, target_model, entry_call_sigs[idx]))

    workspace_roots: List[str] = []
    raw_wps = payload.get("workspacePaths")
    if isinstance(raw_wps, (list, tuple)):
        for wp in raw_wps:
            if isinstance(wp, str) and wp:
                try:
                    workspace_roots.append(str(Path(wp).resolve()))
                except (OSError, ValueError):
                    pass
    if workspace_roots and state.get("workspace_roots") != workspace_roots:
        state["workspace_roots"] = workspace_roots

    updated_list: List[Dict[str, Any]] = []
    any_model_overwritten = state_changed = False
    for _, item, spawn_record, target_model, call_sig in validated_items:
        if workspace_roots:
            spawn_record["workspace_roots"] = workspace_roots
        spawn_record.update({"call_id": call_id, "call_sig": call_sig, "spawned_at": time.time()})

        if call_sig not in recorded_calls:
            recorded_calls.append(call_sig)
            if spawn_record.get("mode") == "best_of_n" and spawn_record.get("cohort"):
                c_id = spawn_record["cohort"]
                cohort_entry = state.setdefault("cohorts", {}).setdefault(
                    c_id, {"total_spawned": 0, "succeeded": [], "failed": [], "created_at": time.time()}
                )
                cohort_entry["total_spawned"] = int(cohort_entry.get("total_spawned", 0) or 0) + 1
                spawn_record["candidate_id"] = f"{c_id}#{cohort_entry['total_spawned']}"
                if spawn_record.get("eval_cmd"):
                    cohort_entry["eval_cmd"] = spawn_record["eval_cmd"]
            if spawn_record.get("mode") == "council" and spawn_record.get("council_id"):
                s_type = str(spawn_record.get("type_name") or spawn_record.get("type") or "").strip().lower()
                _record_council_member(
                    state,
                    str(spawn_record["council_id"]),
                    str(spawn_record.get("role") or s_type or "member"),
                    spawn_record.get("council_role"),
                    str(spawn_record.get("model") or target_model),
                    is_judge_type=(s_type == "judge"),
                )
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
        return {"decision": "allow", "overwrite": {"Subagents": updated_list} if is_list_schema else {"Model": updated_list[0]["Model"]}}
    return {"decision": "allow"}


def _lookup_recipient_meta(payload: Dict[str, Any], state: Dict[str, Any], recipient: str) -> Optional[Dict[str, Any]]:
    """Resolve recipient subagent or swarm worker metadata from local state or sibling brain state."""
    raw_recipient = (recipient or "").strip()
    if not raw_recipient:
        return None
    norm_recipient = raw_recipient.lower()
    for bucket_key in ("subagents", "swarm_workers"):
        bucket = state.get(bucket_key)
        if isinstance(bucket, dict):
            hit = bucket.get(raw_recipient) or bucket.get(norm_recipient)
            if isinstance(hit, dict):
                return hit
            if bucket_key == "swarm_workers":
                for wname, wmeta in bucket.items():
                    if str(wname).strip().lower() == norm_recipient and isinstance(wmeta, dict):
                        return wmeta
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                for cand_name in dict.fromkeys((raw_recipient, norm_recipient)):
                    sib_state_file = resolved_art.parent / cand_name / "orchestrator_state.json"
                    if sib_state_file.exists():
                        sib_state = _read_sibling_snapshot(sib_state_file)
                        meta = sib_state.get("self_subagent_meta") if isinstance(sib_state, dict) else None
                        if isinstance(meta, dict):
                            return meta
        except (OSError, ValueError):
            pass
    return None


def _sync_sibling_recipient_state(
    payload: Dict[str, Any],
    raw_recipient: str,
    recipient: str,
    parsed_allowed: Optional[List[str]],
    stage_match: Optional[re.Match[str]],
    reactivate_impl: bool,
    deferred_actions: Optional[List[Callable[[], None]]],
) -> None:
    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir or not raw_recipient:
        return
    try:
        resolved_art = Path(artifact_dir).resolve()
        if resolved_art.parent.name != "brain":
            return
        sib_state_file = resolved_art.parent / raw_recipient / "orchestrator_state.json"
        if not sib_state_file.exists() and recipient != raw_recipient:
            sib_state_file = resolved_art.parent / recipient / "orchestrator_state.json"
        if not sib_state_file.exists():
            return
        stage_val = stage_match.group(1).strip() if stage_match is not None else None

        def _sync() -> None:
            with state_transaction(sib_state_file):
                sib_state = _load_state(sib_state_file)
                meta = sib_state.setdefault("self_subagent_meta", {})
                if parsed_allowed is not None:
                    meta["allowed_files"] = parsed_allowed
                if stage_val is not None:
                    meta["stage"] = stage_val
                    meta.setdefault("stage_history", []).append(stage_val)
                if reactivate_impl:
                    meta["status"] = "active"
                _save_state(sib_state_file, sib_state)

        if deferred_actions is not None:
            deferred_actions.append(_sync)
        elif not _has_other_locked_path(sib_state_file):
            _sync()
    except (OSError, ValueError):
        pass


def _handle_send_message(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    deferred_actions: Optional[List[Callable[[], None]]] = None,
) -> Dict[str, Any]:
    raw_recipient = args.get("Recipient")
    if raw_recipient is not None and not isinstance(raw_recipient, str):
        return _deny("send_message.Recipient must be a string.")
    raw_message = args.get("Message")
    if not isinstance(raw_message, str) or not raw_message.strip():
        return _deny("send_message.Message cannot be empty.")
    recipient, message = (raw_recipient or "").strip().lower(), raw_message

    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    _save_state(payload, state)

    if not _is_root_orchestrator(payload, state) and re.match(r"^\s*STATUS:", message):
        from guard.verification import validate_subagent_report
        sub_type = _get_subagent_type(payload, state)
        if err := validate_subagent_report(message, is_scout=_is_scout_subagent(payload, state), is_readonly_or_scratch=(sub_type in READONLY_NON_VERIFICATION_TYPES)):
            return _deny(f"[ORCHESTRATOR GUARD] Subagent report format error in send_message: {err}")
        sent_key = f"sent:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
        reminded = state.setdefault("stop_reminded_executions", [])
        if sent_key not in reminded:
            reminded.append(sent_key)
            _save_state(payload, state)

    subagents = state.get("subagents")
    subagent_info = subagents.get(recipient) if isinstance(subagents, dict) else None
    recipient_meta = subagent_info if isinstance(subagent_info, dict) else _lookup_recipient_meta(payload, state, raw_recipient)
    recipient_type = str((recipient_meta or {}).get("type_name") or (recipient_meta or {}).get("type") or "").strip().lower()

    if recipient_meta and recipient_type in HIGH_REASONING_READONLY_TYPES:
        current_epoch = state.get("test_receipt", {}).get("mutation_epoch", 0)
        spawned_epoch = recipient_meta.get("spawned_mutation_epoch", 0)
        if current_epoch > spawned_epoch:
            return _deny(
                f"[ORCHESTRATOR GUARD] Stale {recipient_type} '{recipient}': workspace code was mutated since spawn "
                f"(epoch {spawned_epoch} -> {current_epoch}). Spawn a fresh '{recipient_type}' via invoke_subagent instead of reusing via send_message."
            )

    is_swarm_worker = bool(recipient_meta and recipient_meta in (state.get("swarm_workers") or {}).values())
    recipient_name_or_role = str((recipient_meta or {}).get("name") or (recipient_meta or {}).get("role") or "").lower()
    recipient_council_role, recipient_mode = (recipient_meta or {}).get("council_role"), (recipient_meta or {}).get("mode")
    is_blind_judge_recipient = bool(
        recipient_meta
        and (
            recipient_type == "judge"
            or recipient_council_role in {"judge", "arbiter", "synthesizer"}
            or (recipient_mode == "council" and (recipient_type == "reviewer" or recipient_council_role not in {"proposer", None}))
            or ((is_swarm_worker or recipient_mode == "council") and ("judge" in recipient_name_or_role or "arbiter" in recipient_name_or_role))
        )
    )
    if is_blind_judge_recipient and (leak := _detect_blind_judge_leak(message, payload=payload)):
        return _deny(f"[ORCHESTRATOR GUARD] Blind judging violation: judge prompt leaks model identity ('{leak}'). Anonymize proposals before invoking judge.")

    spec_file, spec_err = _validate_spec_file_header(message, recipient_type, payload)
    if spec_err:
        return _deny(spec_err)

    local_dirty = False
    if spec_file and subagent_info:
        subagent_info["spec_file"] = spec_file
        local_dirty = True
    if (replan_if_match := REPLAN_IF_HEADER_RE.search(message)) and subagent_info:
        subagent_info["replan_if"] = replan_if_match.group(1).strip()
        local_dirty = True
    stage_match = STAGE_HEADER_RE.search(message)
    if stage_match and subagent_info:
        stage_val = stage_match.group(1).strip()
        subagent_info["stage"] = stage_val
        subagent_info.setdefault("stage_history", []).append(stage_val)
        local_dirty = True

    target_meta = subagent_info if isinstance(subagent_info, dict) else (recipient_meta if recipient_type else None)
    if isinstance(target_meta, dict):
        target_tier = str(target_meta.get("model_tier") or "").strip().lower()
        if not target_tier:
            target_tier = infer_subagent_tier(target_meta.get("model"), recipient_type or target_meta.get("type"))
        if target_tier == "lite":
            if MUTATION_VERBS_RE.search(message) or len(NUMBERED_RULE_RE.findall(message)) > 3:
                tier_map = get_model_tier_map()
                lite_model = str(target_meta.get("model") or get_model_for_tier("lite", tier_map))
                std_model = get_model_for_tier("standard", tier_map)
                return _deny(
                    f"Recipient '{recipient}' is a lite-tier ('{lite_model}', {target_meta.get('type', 'scout')}) subagent. "
                    f"Do not send code mutation or multi-step (>3 rules) tasks to lite-tier ('{lite_model}') workers. "
                    f"Invoke a standard-tier ('{std_model}') subagent ('research' or 'implementer') instead."
                )

    parsed_allowed = _parse_allowed_files_header(message)
    if recipient_type == "implementer" and recipient_meta:
        effective_allowed = parsed_allowed if parsed_allowed is not None else recipient_meta.get("allowed_files")
        target_ws = str(recipient_meta.get("workspace") or "inherit").strip().lower()
        if effective_allowed:
            for other in _iter_active_subagents(state, exclude_cid=recipient):
                if str(other.get("type_name") or other.get("type") or "").strip().lower() != "implementer":
                    continue
                other_allowed = other.get("allowed_files")
                other_ws = str(other.get("workspace") or "inherit").strip().lower()
                if not other_allowed or (target_ws != "inherit" and other_ws != "inherit"):
                    continue
                if (overlap := _find_allowed_files_overlap(effective_allowed, other_allowed)) is not None:
                    return _deny(
                        f"[ORCHESTRATOR GUARD] Overlapping ALLOWED_FILES when reactivating implementer '{recipient}' "
                        f"in 'inherit' workspace ('{overlap[0]}' vs '{overlap[1]}'). Wait for conflicting implementer to complete or assign disjoint ALLOWED_FILES."
                    )
        if subagent_info is not None:
            subagent_info["status"] = "active"
            local_dirty = True

    if parsed_allowed is not None and subagent_info:
        subagent_info["allowed_files"] = parsed_allowed
        local_dirty = True
    if local_dirty:
        _save_state(payload, state)

    if parsed_allowed is not None or stage_match is not None or (recipient_type == "implementer" and recipient_meta):
        _sync_sibling_recipient_state(payload, raw_recipient or "", recipient, parsed_allowed, stage_match, recipient_type == "implementer", deferred_actions)
    return {"decision": "allow"}


def _mark_killed(entry: Dict[str, Any]) -> None:
    entry["active_branch"] = False
    entry["active_worktree"] = False
    entry["status"] = "killed"


def _handle_manage_subagents(
    payload: Dict[str, Any], args: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    action = str(args.get("Action") or args.get("action") or "").strip().lower()
    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    subagents = state.get("subagents")
    if not isinstance(subagents, dict):
        subagents = {}
        state["subagents"] = subagents

    if action == "kill_all":
        for meta in subagents.values():
            if isinstance(meta, dict):
                _mark_killed(meta)
        for pending in (state.get("pending_spawns") if isinstance(state.get("pending_spawns"), list) else ()):
            if isinstance(pending, dict):
                _mark_killed(pending)
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
        if not isinstance(raw_ids, list):
            return _deny("[ORCHESTRATOR GUARD] manage_subagents ConversationIds must be a list or string.")
        target_ids = {str(cid).strip().lower() for cid in raw_ids if cid}
        matched_count = 0
        for cid in target_ids:
            if cid in subagents and isinstance(subagents[cid], dict):
                _mark_killed(subagents[cid])
                matched_count += 1
        remaining_pending = []
        killed_left = max(0, len(target_ids) - matched_count)
        for pending in state.get("pending_spawns", []):
            if isinstance(pending, dict) and (pending.get("active_worktree") or pending.get("active_branch")) and killed_left > 0:
                pending["active_branch"] = False
                pending["active_worktree"] = False
                killed_left -= 1
            remaining_pending.append(pending)
        state["pending_spawns"] = remaining_pending
        _save_state(payload, state)
    return {"decision": "allow"}


def _sharded_active_workers(workers: Any, wname: str) -> List[str]:
    if not wname or not isinstance(workers, dict):
        return []
    return [
        k for k, meta in workers.items()
        if isinstance(meta, dict) and meta.get("base_name") == wname and meta.get("sharded") is True and meta.get("status") == "active"
    ]


def _handle_swarm(payload: Dict[str, Any], args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and track swarm tool calls (council anonymity, blind judging, worktrees, EVAL_CMD)."""
    action = str(args.get("action") or "").strip().lower()
    if action in {"status", "models", "title"}:
        return {"decision": "allow"}

    workers = state.get("swarm_workers")
    if action == "kill":
        wname = str(args.get("name") or "").strip()
        if isinstance(workers, dict) and wname in workers and isinstance(workers[wname], dict):
            workers[wname]["pre_kill_active"] = workers[wname].get("status") == "active"
            workers[wname]["status"] = "killed"
            workers[wname]["active_worktree"] = False
            _save_state(payload, state)
        elif sharded := _sharded_active_workers(workers, wname):
            return _deny(
                f"[ORCHESTRATOR GUARD] Swarm group '{wname}' was spawned as sharded workers ({', '.join(sharded)}). "
                "Call swarm(action='kill') with each individual worker name to clean up their worktrees."
            )
        return {"decision": "allow"}

    if action == "model":
        wname = str(args.get("name") or "").strip()
        new_model = str(args.get("model") or "").strip()
        if isinstance(workers, dict) and wname and wname not in workers:
            if sharded := _sharded_active_workers(workers, wname):
                return _deny(
                    f"[ORCHESTRATOR GUARD] Swarm group '{wname}' was spawned as sharded workers ({', '.join(sharded)}). "
                    "Call swarm(action='model') with each individual worker name."
                )
        if isinstance(workers, dict) and wname in workers and isinstance(workers[wname], dict) and new_model:
            workers[wname]["model"] = new_model
            _save_state(payload, state)
        return {"decision": "allow"}

    if action == "add":
        brief = str(args.get("brief") or "")
        name = str(args.get("name") or "").strip()
        model = str(args.get("model") or "").strip()
        workspace = str(args.get("workspace") or "inherit").strip()

        eval_cmd = _parse_eval_cmd_header(brief)
        if eval_cmd is not None and _is_trivial_eval_command(eval_cmd):
            return _deny(f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in swarm: '{eval_cmd}'. Provide a real test or evaluation script command.")

        mode = _parse_mode_header(brief)
        cohort_id = _parse_cohort_header(brief)
        council_id = _parse_council_id_header(brief)
        council_role = _parse_council_role_header(brief)
        if err := _check_mode_header_conflicts(mode, cohort_id, council_id, council_role, context="swarm brief"):
            return _deny(err)
        if (council_id or council_role) and mode is None:
            mode = "council"
        if mode == "council" and not council_id:
            council_id = f"council-{_current_invocation_key(payload, state)}"

        name_lower = name.lower()
        if mode == "council" or council_id or council_role or any(k in name_lower for k in ("judge", "proposer", "arbiter")):
            if name_leak := _detect_council_name_leak(name):
                return _deny(
                    f"[ORCHESTRATOR GUARD] Blind council violation: swarm worker name '{name}' reveals model identity "
                    f"('{name_leak.strip()}'). Use anonymous names like 'proposer-a', 'judge-1'."
                )
        is_council_arbiter = (
            council_role in {"judge", "arbiter", "synthesizer"}
            or "judge" in name_lower
            or "arbiter" in name_lower
            or (mode == "council" and council_role != "proposer")
        )
        if is_council_arbiter and (leak := _detect_blind_judge_leak(brief, payload=payload)):
            return _deny(f"[ORCHESTRATOR GUARD] Blind judging violation: judge prompt leaks model identity ('{leak}'). Anonymize proposals before invoking judge.")

        if eval_cmd:
            state["active_eval_cmd"] = eval_cmd

        worker_names = _expand_swarm_worker_names(args)
        is_sharded = len(worker_names) > 1
        call_id = _extract_call_id(payload)
        now_ts = time.time()
        is_isolated = workspace.lower() in ISOLATED_WORKSPACES
        swarm_workers = state.setdefault("swarm_workers", {})
        for wname in worker_names:
            existing = swarm_workers.get(wname)
            prev_snap = None
            if isinstance(existing, dict):
                prev_snap = dict(existing["_prev_worker"] if isinstance(existing.get("_prev_worker"), dict) else existing)
                prev_snap.pop("_prev_worker", None)
            worker_record: Dict[str, Any] = {
                "name": wname,
                "base_name": name,
                "sharded": is_sharded,
                "model": model,
                "workspace": workspace,
                "active_worktree": is_isolated,
                "status": "active",
                "mode": mode,
                "council_id": council_id,
                "council_role": council_role,
                "call_id": call_id,
                "spawned_at": now_ts,
            }
            if prev_snap is not None:
                worker_record["_prev_worker"] = prev_snap
            swarm_workers[wname] = worker_record

        if mode == "council" and council_id:
            is_judge_name = (council_role == "judge") or (council_role is None and "judge" in name_lower)
            for wname in worker_names:
                _record_council_member(state, council_id, wname, council_role, model, is_judge_type=is_judge_name)

        _save_state(payload, state)
    return {"decision": "allow"}


def _first_present(args: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if (val := args.get(key)) is not None:
            return val
    return None


def _check_forbidden_direct_read(tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if tool_name not in (SEARCH_TOOLS | READ_TOOLS | DIR_LIST_TOOLS):
        return None
    target = _extract_target_path(tool_name, args)
    if target and FORBIDDEN_DIRECT_READ_RE.search(str(target).replace("\\", "/")):
        action_desc = "list or read log or transcript directory/file" if tool_name in DIR_LIST_TOOLS else "read raw log or transcript file"
        return _deny(
            f"[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
            f"directly {action_desc} '{Path(target).name}'. Delegate log slicing to a 'scout' subagent (Model='{get_model_for_tier('lite')}')."
        )
    return None


def _estimate_read_cost(norm_target: str, start_line: Any, end_line: Any) -> Tuple[int, int]:
    p = Path(norm_target)
    try:
        is_file = p.is_file()
    except (OSError, ValueError):
        is_file = False
    if is_file:
        try:
            all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, ValueError):
            all_lines = []
        if start_line is not None or end_line is not None:
            try:
                s = max(1, int(start_line) if start_line is not None else 1)
                e = min(len(all_lines), int(end_line) if end_line is not None else len(all_lines))
                sliced = all_lines[s - 1 : e] if s <= e else []
                return len(sliced), sum(len(x.encode("utf-8")) + 1 for x in sliced)
            except (ValueError, TypeError, OverflowError):
                return (min(800, len(all_lines)) or DEFAULT_UNBOUNDED_READ_LINES, sum(len(x.encode("utf-8")) + 1 for x in all_lines) or DEFAULT_UNBOUNDED_READ_BYTES)
        try:
            return min(800, len(all_lines)), min(46080, p.stat().st_size)
        except (OSError, ValueError):
            return min(800, len(all_lines)), sum(len(x.encode("utf-8")) + 1 for x in all_lines)
    if start_line is not None and end_line is not None:
        try:
            lines_cost = max(1, int(end_line) - int(start_line) + 1)
            return lines_cost, lines_cost * 40
        except (ValueError, TypeError, OverflowError):
            pass
    return DEFAULT_UNBOUNDED_READ_LINES, DEFAULT_UNBOUNDED_READ_BYTES


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
    if denial := _check_forbidden_direct_read(tool_name, args):
        return denial

    if state.get("plan_execute_phase"):
        plan_allowed = state.get("plan_allowed_files")
        if tool_name in MUTATION_TOOLS:
            target = _extract_target_path(tool_name, args)
            if plan_allowed is not None and target and not _is_artifact_path(target, payload) and not _is_target_in_allowed_files(target, plan_allowed, payload.get("workspacePaths")):
                return _deny(
                    f"[ORCHESTRATOR GUARD] File allowlist violation: plan execution attempted to mutate "
                    f"'{Path(target).name}', which is outside plan ALLOWED_FILES ({', '.join(plan_allowed) if plan_allowed else 'NONE'})."
                )
        elif tool_name == "run_command" and plan_allowed is not None:
            raw_cwd = args.get("Cwd")
            cwd = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else str(PROJECT_ROOT)
            cmd = args.get("CommandLine").strip() if isinstance(args.get("CommandLine"), str) else ""
            return _validate_shell_allowlist(cmd, _split_shell_commands(cmd), cwd, plan_allowed, payload, allowlist_label="plan ALLOWED_FILES")
        return None

    if tool_name in (DIR_LIST_TOOLS | SEARCH_TOOLS):
        return None

    inv_key = _current_invocation_key(payload, state)
    budget = state.setdefault("direct_budget", {}).setdefault(
        inv_key,
        {"mutated_files": [], "read_files": [], "read_ranges": [], "read_lines": 0, "read_bytes": 0, "commands": [], "seen_calls": [], "test_runs_after_edit": 0},
    )

    if tool_name in READ_TOOLS:
        target = _extract_target_path(tool_name, args)
        if not target or _is_artifact_or_worktree_path(target, payload) or "/skills/" in str(target).replace("\\", "/"):
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except (OSError, ValueError):
            return None

        start_line = _first_present(args, "StartLine", "start_line", "StartLineNumber")
        end_line = _first_present(args, "EndLine", "end_line", "EndLineNumber")
        read_ranges: List[str] = budget.setdefault("read_ranges", [])
        read_files: List[str] = budget.setdefault("read_files", [])
        range_key = f"{norm_target}:{start_line or ''}:{end_line or ''}"
        if range_key in read_ranges or f"{norm_target}::" in read_ranges:
            return None

        lines_cost, bytes_cost = _estimate_read_cost(norm_target, start_line, end_line)
        current_lines, current_bytes = budget.setdefault("read_lines", 0), budget.setdefault("read_bytes", 0)
        file_is_new = norm_target not in read_files
        if current_lines + lines_cost > MAX_DIRECT_READ_LINES or current_bytes + bytes_cost > MAX_DIRECT_READ_BYTES or (file_is_new and len(read_files) >= MAX_DIRECT_READ_FILES):
            tier_map = get_model_tier_map()
            return _deny(
                f"[ORCHESTRATOR GUARD] Direct execution fast-path read budget "
                f"({MAX_DIRECT_READ_LINES} lines / {MAX_DIRECT_READ_BYTES} bytes / {MAX_DIRECT_READ_FILES} workspace files) exceeded when reading '{Path(norm_target).name}'. "
                f"Pass StartLine/EndLine for targeted inspection or delegate to a 'scout' (Model='{get_model_for_tier('lite', tier_map)}') or 'research' (Model='{get_model_for_tier('standard', tier_map)}') subagent."
            )
        read_ranges.append(range_key)
        if file_is_new:
            read_files.append(norm_target)
        budget["read_lines"], budget["read_bytes"] = current_lines + lines_cost, current_bytes + bytes_cost
        _save_state(payload, state)
        return None

    if tool_name in MUTATION_TOOLS:
        target = _extract_target_path(tool_name, args)
        if not target or _is_artifact_path(target, payload):
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except (OSError, ValueError):
            return None
        if budget.get("test_runs_after_edit", 0) >= 1:
            return _deny(f"[ORCHESTRATOR GUARD] Iterative edit-and-test loop detected in the orchestrator. Delegate multi-step code and test execution loops to an 'implementer' subagent (Model='{get_model_for_tier('standard')}').")
        mutated_files: List[str] = budget.setdefault("mutated_files", [])
        if norm_target not in mutated_files:
            if len(mutated_files) >= MAX_DIRECT_MUTATION_FILES:
                return _deny(
                    f"[ORCHESTRATOR GUARD] Direct execution fast-path limit ({MAX_DIRECT_MUTATION_FILES} workspace files) exceeded "
                    f"when editing '{Path(norm_target).name}'. Delegate multi-file changes to an 'implementer' subagent (Model='{get_model_for_tier('standard')}')."
                )
            mutated_files.append(norm_target)
            _save_state(payload, state)
        return None

    if tool_name == "run_command":
        raw_cmd = args.get("CommandLine")
        cmd = raw_cmd.strip() if isinstance(raw_cmd, str) else ""
        if _is_safe_git_inspection_command(cmd):
            return None
        call_id = _extract_call_id(payload)
        seen_calls: List[str] = budget.setdefault("seen_calls", [])
        if call_id and call_id in seen_calls:
            return None
        if budget.get("mutated_files", []) and (_is_test_command(cmd) or _is_declared_eval_command(cmd, state, payload)):
            if budget.get("test_runs_after_edit", 0) >= 1:
                return _deny(f"[ORCHESTRATOR GUARD] Direct workspace edit followed by iterative test execution detected in the orchestrator. Delegate code and test execution loops to an 'implementer' subagent (Model='{get_model_for_tier('standard')}').")
            budget["test_runs_after_edit"] = budget.get("test_runs_after_edit", 0) + 1
        commands: List[str] = budget.setdefault("commands", [])
        if len(commands) >= MAX_DIRECT_COMMANDS:
            return _deny(
                f"[ORCHESTRATOR GUARD] Direct execution fast-path command limit ({MAX_DIRECT_COMMANDS} commands per turn) exceeded. "
                "Delegate iterative command or test execution to a 'scout', 'research', or 'implementer' subagent."
            )
        if call_id:
            seen_calls.append(call_id)
        commands.append(cmd)
        _save_state(payload, state)
    return None


def _is_unsafe_declared_eval(ev: str) -> bool:
    """Return True if a declared EVAL_CMD contains mutating shell verbs, shell/exec wrappers, or awk/sed."""
    return _is_unsafe_eval_command(ev)


def _handle_run_workflow(
    payload: Dict[str, Any], args: Dict[str, Any], state: Dict[str, Any], roots: List[Path]
) -> Dict[str, Any]:
    """Validate run_workflow calls against subagent role restrictions, allowed roots, guard targets, and EVAL_CMD."""
    if _is_strict_readonly_subagent(payload, state):
        sub_type = _get_subagent_type(payload, state) or "readonly"
        return _deny(f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly read-only. Workflow execution is prohibited.")
    if _is_scratch_only_subagent(payload, state):
        role_word = "Council" if _get_subagent_type(payload, state) == "council" else "Research"
        return _deny(f"[ORCHESTRATOR GUARD] {role_word} subagents may only write markdown reports to <artifactDirectoryPath>/scratch/*.md. Workflow execution is prohibited.")
    if _get_subagent_allowed_files(payload, state) is not None:
        return _deny("[ORCHESTRATOR GUARD] Implementer subagents restricted by ALLOWED_FILES cannot execute unbounded workflows via run_workflow.")
    if _is_root_orchestrator(payload, state) and state.get("plan_execute_phase") and state.get("plan_allowed_files") is not None:
        return _deny("[ORCHESTRATOR GUARD] Root orchestrator restricted by ALLOWED_FILES cannot execute unbounded workflows via run_workflow.")

    desc_eval = _parse_eval_cmd_header(str(args.get("Description") or ""))
    raw_script = str(args.get("Script") or "")
    script_eval = _parse_eval_cmd_header(re.sub(r"(?m)^[ \t]*#+[ \t]*", "", raw_script)) if raw_script else None
    for candidate_eval in (desc_eval, script_eval):
        if candidate_eval is not None and _is_trivial_eval_command(candidate_eval):
            return _deny(f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in run_workflow: '{candidate_eval}'. Provide a real test or evaluation script command.")
        if candidate_eval is not None and _is_unsafe_declared_eval(candidate_eval):
            return _deny(f"[ORCHESTRATOR GUARD] Unsafe mutating or wrapper EVAL_CMD rejected in run_workflow: '{candidate_eval}'.")

    cwd = args.get("Cwd")
    if cwd:
        cwd_str = str(cwd)
        if _is_guard_state_target(cwd_str, payload=payload) or _is_guard_plugin_target(cwd_str) or _is_guard_plugin_cwd(cwd_str):
            return _deny("[ORCHESTRATOR GUARD] Workflow Cwd cannot target guard state or plugin files.")
        if not _is_within_allowed_roots(cwd_str, roots):
            return {"decision": "ask"}

    if script_path := args.get("ScriptPath"):
        sp_str = str(script_path)
        base_cwd = str(cwd) if cwd else str(PROJECT_ROOT)
        sp_resolved_str = _resolve_cmd_target(sp_str, base_cwd)
        if (
            _is_guard_state_target(sp_str, payload=payload, cwd=base_cwd)
            or _is_guard_state_target(sp_resolved_str, payload=payload)
            or _is_guard_plugin_target(sp_str, base_cwd)
            or _is_guard_plugin_target(sp_resolved_str, base_cwd)
        ):
            return _deny("[ORCHESTRATOR GUARD] Workflow ScriptPath cannot target guard state or plugin files.")
        if not _is_within_allowed_roots(sp_resolved_str, roots):
            return {"decision": "ask"}
        try:
            raw_sp = Path(os.path.expanduser(sp_str))
            sp_candidates = [Path(sp_resolved_str)] + ([raw_sp.resolve()] if raw_sp.is_absolute() else [])
            if any(sp_r.name == "orchestrator_guard.py" for sp_r in sp_candidates):
                return _deny("[ORCHESTRATOR GUARD] Workflow ScriptPath cannot target orchestrator_guard.py.")
        except (OSError, ValueError):
            pass

    if eval_cmd := (desc_eval or script_eval):
        state["active_eval_cmd"] = eval_cmd
        _save_state(payload, state)
    return {"decision": "allow"}

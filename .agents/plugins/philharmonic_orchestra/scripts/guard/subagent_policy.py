"""Subagent dispatch/messaging/swarm policies, direct execution fast-path budget, and workflow validation."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from guard.ast_guard import (
    _PY_FRAME_ATTRS,
    _PY_MUTATING_ATTRS,
    _PY_MUTATING_NAMES,
    _SAFE_JSON_IMPORTS,
    _SAFE_WORKFLOW_STDLIB_MODULES,
)
from guard.constants import (
    ALLOWED_COUNCIL_PROPOSER_MODELS,
    ALLOWED_SUBAGENT_MODELS,
    ALLOWED_VERIFIER_REVIEWER_MODELS,
    BLIND_JUDGE_LEAK_PATTERNS,
    COHORT_HEADER_RE,
    COUNCIL_ID_HEADER_RE,
    COUNCIL_NAME_LEAK_RE,
    COUNCIL_ROLE_HEADER_RE,
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
    MODE_HEADER_RE,
    MUTATING_SHELL_CMD_RE,
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
    SHELL_EXEC_WRAPPER_RE,
    SPEC_FILE_HEADER_RE,
    STAGE_HEADER_RE,
    STRICT_READONLY_SUBAGENT_TYPES,
    _extract_call_id,
    _has_nul_or_surrogate_input,
    _is_test_command,
    _preprocess_shell_heredocs_and_comments,
    _split_shell_commands,
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
from guard.root_allowlist import _is_root_allowlisted_command
from guard.shell_targets import _validate_shell_allowlist
from guard.shell_tokens import (
    _is_declared_eval_command,
    _is_safe_git_inspection_command,
    _is_trivial_eval_command,
    _strip_shell_quoted_literals,
)
from guard.state_store import (
    _current_invocation_key,
    _has_other_locked_path,
    _load_state,
    _save_state,
    state_transaction,
)


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
    """Parse MODE: header from a subagent prompt, normalizing Best-of-N and Council variants."""
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
    if cleaned in {"council", "council_of_models", "councilofmodels"}:
        return "council"
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


def _parse_council_id_header(text: str) -> Optional[str]:
    """Parse COUNCIL_ID: header from a prompt or brief, returning None if omitted."""
    if not text:
        return None
    match = COUNCIL_ID_HEADER_RE.search(text)
    if not match:
        return None
    raw_val = match.group(1).strip().strip("`'\"")
    if not raw_val or raw_val.upper() == "NONE":
        return None
    cleaned = re.sub(r"[\s_-]+", "-", raw_val.lower()).strip("-")
    return cleaned or None


def _parse_council_role_header(text: str) -> Optional[str]:
    """Parse COUNCIL_ROLE: header from a prompt or brief, normalizing proposer | judge | synthesizer."""
    if not text:
        return None
    match = COUNCIL_ROLE_HEADER_RE.search(text)
    if not match:
        return None
    raw_val = match.group(1).strip().strip("`'\"").lower()
    if not raw_val or raw_val == "none":
        return None
    cleaned = re.sub(r"[\s_-]+", "_", raw_val).strip("_")
    if cleaned in {"proposer", "proposal", "member"}:
        return "proposer"
    if cleaned in {"judge", "blind_judge", "voter"}:
        return "judge"
    if cleaned in {"synthesizer", "synthesis", "synth"}:
        return "synthesizer"
    return cleaned or None


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
        raw_tok = match.group(1)
        try:
            tok_path = Path(os.path.expanduser(raw_tok))
        except (OSError, ValueError):
            continue
        to_check = (
            [tok_path]
            if tok_path.is_absolute()
            else [resolved_art / tok_path, PROJECT_ROOT / tok_path]
        )
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
    """Check raw and markdown-stripped text against BLIND_JUDGE_LEAK_PATTERNS."""
    if not text:
        return None
    normalized = re.sub(r"[*_`]+", "", text)
    candidates = [text] if normalized == text else [text, normalized]
    for candidate_text in candidates:
        for pattern in BLIND_JUDGE_LEAK_PATTERNS:
            match = pattern.search(candidate_text)
            if match:
                return match.group(0)
    return None


def _detect_blind_judge_leak(
    text: str, payload: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Return the matched attribution string if text or referenced scratch docs leak model identity to a judge, else None."""
    if not text:
        return None
    direct_match = _match_blind_judge_patterns(text)
    if direct_match:
        return direct_match
    artifact_dir = payload.get("artifactDirectoryPath") if isinstance(payload, dict) else None
    if not artifact_dir:
        return None
    for candidate in _resolve_scratch_doc_candidates(text, str(artifact_dir)):
        try:
            with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(65536)
        except (OSError, ValueError):
            continue
        file_match = _match_blind_judge_patterns(content)
        if file_match:
            return f"{candidate.name}: {file_match}"
    return None


def _detect_council_name_leak(name: str) -> Optional[str]:
    """Return the matched token if a council worker name reveals model identity, else None."""
    if not name:
        return None
    match = COUNCIL_NAME_LEAK_RE.search(name)
    if match:
        return match.group(0)
    return None


def _expand_swarm_worker_names(args: Dict[str, Any]) -> List[str]:
    """Expand swarm worker name(s) from 'name', 'tasks', and 'n' arguments."""
    name = str(args.get("name") or "").strip()
    if not name:
        return []
    tasks = args.get("tasks")
    n = args.get("n")
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
    councils = state.setdefault("councils", {})
    council_entry = councils.setdefault(
        council_id,
        {
            "proposers": [],
            "judges": [],
            "models": [],
            "created_at": time.time(),
        },
    )
    proposers = council_entry.setdefault("proposers", [])
    judges = council_entry.setdefault("judges", [])
    synthesizers = council_entry.setdefault("synthesizers", [])
    models = council_entry.setdefault("models", [])
    if council_role == "judge" or (council_role is None and is_judge_type):
        if member_name and member_name not in judges:
            judges.append(member_name)
    elif council_role == "synthesizer":
        if member_name and member_name not in synthesizers:
            synthesizers.append(member_name)
    elif council_role == "proposer" or (council_role is None and not is_judge_type):
        if member_name and member_name not in proposers:
            proposers.append(member_name)
    if model and model not in models:
        models.append(model)


def _compute_prompt_hash(prompt_text: str) -> str:
    """Return a short deterministic SHA-256 prefix for non-empty subagent prompts."""
    if not prompt_text:
        return ""
    return hashlib.sha256(
        str(prompt_text or "").encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:12]


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
    raw_subs = state.get("subagents")
    if isinstance(raw_subs, dict):
        for cid, entry in raw_subs.items():
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
    if sub_name in {"scout", "implementer", "verifier", "reviewer", "judge", "council"}:
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

    try:
        raw_path = Path(os.path.expanduser(raw_val))
        spec_path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
        resolved_path = spec_path.resolve()
    except (OSError, ValueError):
        return None, (
            f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' is not a valid path."
        )

    if type_name.lower() == "implementer":
        try:
            is_spec_file = resolved_path.is_file()
        except (OSError, ValueError):
            is_spec_file = False
        if not is_spec_file:
            return None, (
                f"[ORCHESTRATOR GUARD] SPEC_FILE '{raw_val}' does not exist. "
                "Have 'research' or root write the grounded spec before dispatching 'implementer'."
            )
        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
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
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Validate a single subagent spec and return (denial, spawn_record, target_model)."""
    type_name = str(entry.get("TypeName") or "").strip()
    norm_type = type_name.lower()
    raw_prompt = entry.get("Prompt")
    prompt_text = "" if raw_prompt is None else str(raw_prompt)
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
    if "\x00" in prompt_text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in prompt_text):
        return (
            {
                "decision": "deny",
                "reason": "[ORCHESTRATOR GUARD] Subagent Prompt contains disallowed NUL byte or lone UTF-16 surrogate.",
            },
            None,
            None,
        )

    parsed_mode = _parse_mode_header(prompt_text)
    cohort_id = _parse_cohort_header(prompt_text)
    council_id = _parse_council_id_header(prompt_text)
    council_role = _parse_council_role_header(prompt_text)

    if (
        (parsed_mode is not None and parsed_mode != "best_of_n" and cohort_id)
        or (parsed_mode is not None and parsed_mode != "council" and (council_id or council_role))
        or (cohort_id and (council_id or council_role))
    ):
        return (
            {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Conflicting mode headers: cannot mix "
                    "MODE: BEST_OF_N / COHORT with MODE: COUNCIL / COUNCIL_ID / COUNCIL_ROLE "
                    "or attach COHORT/COUNCIL headers to an incompatible MODE."
                ),
            },
            None,
            None,
        )

    mode = parsed_mode
    if cohort_id and mode is None:
        mode = "best_of_n"
    if mode == "best_of_n":
        if norm_type != "implementer":
            return (
                {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] MODE: BEST_OF_N is only allowed for 'implementer' "
                        f"subagents (got '{type_name}')."
                    ),
                },
                None,
                None,
            )
        if not cohort_id:
            cohort_id = f"best-of-n-{inv_key}"

    role_label_raw = str(entry.get("Role") or type_name or "subagent").strip()
    role_lower = role_label_raw.lower()
    if norm_type == "council" and ("judge" in role_lower or "arbiter" in role_lower):
        mode = mode or "council"
        council_role = council_role or "judge"

    if (council_id or council_role) and mode is None:
        mode = "council"
    if mode == "council":
        if norm_type == "implementer":
            return (
                {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] MODE: COUNCIL is not allowed for 'implementer' "
                        "subagents. Use 'research', 'council', or 'judge'."
                    ),
                },
                None,
                None,
            )
        if not council_id:
            council_id = f"council-{inv_key}"

    is_blind_judge = (
        norm_type == "judge"
        or council_role == "judge"
        or (mode == "council" and norm_type == "reviewer")
        or (mode == "council" and council_role not in {"proposer", None})
        or (
            (
                norm_type in {"judge", "reviewer", "council", "research", "researcher"}
                or mode == "council"
            )
            and (
                "judge" in role_lower
                or "arbiter" in role_lower
                or "synthesizer" in role_lower
                or council_role in {"judge", "arbiter", "synthesizer"}
            )
        )
    )
    if is_blind_judge:
        leak = _detect_blind_judge_leak(prompt_text, payload=payload)
        if leak:
            return (
                {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Blind judging violation: judge prompt leaks model identity "
                        f"('{leak}'). Anonymize proposals before invoking judge."
                    ),
                },
                None,
                None,
            )

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

    raw_model = str(entry.get("Model") or "").strip()
    role_label = str(entry.get("Role") or type_name or "subagent").strip()
    workspace_mode = str(entry.get("Workspace") or "inherit").strip()

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
    elif (
        (parsed_mode == "council" and norm_type in SCRATCH_ONLY_SUBAGENT_TYPES)
        or norm_type == "council"
    ):
        if raw_model in ALLOWED_COUNCIL_PROPOSER_MODELS:
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
    if council_id is not None:
        spawn_record["council_id"] = council_id
    if council_role is not None:
        spawn_record["council_role"] = council_role
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
    caller_cid = str(payload.get("conversationId") or "").strip().lower()

    entry_call_sigs: List[str] = []
    for idx, raw_item in enumerate(entries):
        item = raw_item if isinstance(raw_item, dict) else {}
        sig_call_id = f"{call_id}:{idx}" if is_list_schema else call_id
        t_name = str(item.get("TypeName") or "").strip()
        r_label = str(item.get("Role") or t_name or "subagent").strip()
        ws_mode = str(item.get("Workspace") or "inherit").strip()
        raw_model = str(item.get("Model") or "").strip()
        p_hash = _compute_prompt_hash(str(item.get("Prompt") or ""))
        entry_call_sigs.append(
            json.dumps(
                [caller_cid, inv_key, exec_num, sig_call_id, t_name, r_label, ws_mode, raw_model, p_hash],
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
            payload=payload,
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

    workspace_roots: List[str] = []
    raw_wps = payload.get("workspacePaths")
    if isinstance(raw_wps, (list, tuple)):
        for wp in raw_wps:
            if not isinstance(wp, str) or not wp:
                continue
            try:
                workspace_roots.append(str(Path(wp).resolve()))
            except (OSError, ValueError):
                continue
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
            if spawn_record.get("mode") == "council" and spawn_record.get("council_id"):
                s_type = str(
                    spawn_record.get("type_name") or spawn_record.get("type") or ""
                ).strip().lower()
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
    """Resolve recipient subagent or swarm worker metadata from local state or sibling brain state."""
    raw_recipient = (recipient or "").strip()
    if not raw_recipient:
        return None
    norm_recipient = raw_recipient.lower()
    raw_subs = state.get("subagents")
    subagents = raw_subs if isinstance(raw_subs, dict) else {}
    sub_info = subagents.get(raw_recipient) or subagents.get(norm_recipient)
    if isinstance(sub_info, dict):
        return sub_info
    raw_swarm = state.get("swarm_workers")
    swarm_workers = raw_swarm if isinstance(raw_swarm, dict) else {}
    sw_info = swarm_workers.get(raw_recipient) or swarm_workers.get(norm_recipient)
    if isinstance(sw_info, dict):
        return sw_info
    for wname, wmeta in swarm_workers.items():
        if str(wname).strip().lower() == norm_recipient and isinstance(wmeta, dict):
            return wmeta
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                checked: Set[Path] = set()
                for cand_name in (raw_recipient, norm_recipient):
                    sib_state_file = (
                        resolved_art.parent / cand_name / "orchestrator_state.json"
                    )
                    if sib_state_file in checked:
                        continue
                    checked.add(sib_state_file)
                    if sib_state_file.exists():
                        sib_state = _read_sibling_snapshot(sib_state_file)
                        meta = (
                            sib_state.get("self_subagent_meta")
                            if isinstance(sib_state, dict)
                            else None
                        )
                        if isinstance(meta, dict):
                            return meta
        except (OSError, ValueError):
            pass
    return None


def _handle_send_message(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    deferred_actions: Optional[List[Callable[[], None]]] = None,
) -> Dict[str, Any]:
    raw_recipient = args.get("Recipient")
    if raw_recipient is not None and not isinstance(raw_recipient, str):
        return {"decision": "deny", "reason": "send_message.Recipient must be a string."}
    raw_message = args.get("Message")
    if not isinstance(raw_message, str) or not raw_message.strip():
        return {"decision": "deny", "reason": "send_message.Message cannot be empty."}
    recipient = (raw_recipient or "").strip().lower()
    message = raw_message

    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    _save_state(payload, state)

    if not _is_root_orchestrator(payload, state) and re.match(r"^\s*STATUS:", message):
        from guard.verification import validate_subagent_report

        sub_type = _get_subagent_type(payload, state)
        err = validate_subagent_report(
            message,
            is_scout=_is_scout_subagent(payload, state),
            is_readonly_or_scratch=(sub_type in READONLY_NON_VERIFICATION_TYPES),
        )
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

    subagents = state.get("subagents")
    subagent_info = subagents.get(recipient) if isinstance(subagents, dict) else None
    recipient_meta = (
        subagent_info
        if isinstance(subagent_info, dict)
        else _lookup_recipient_meta(payload, state, raw_recipient)
    )
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

    is_swarm_worker = bool(
        recipient_meta
        and recipient_meta in (state.get("swarm_workers") or {}).values()
    )
    recipient_name_or_role = str(
        (recipient_meta or {}).get("name")
        or (recipient_meta or {}).get("role")
        or ""
    ).lower()
    recipient_council_role = (recipient_meta or {}).get("council_role")
    recipient_mode = (recipient_meta or {}).get("mode")
    is_blind_judge_recipient = bool(
        recipient_meta
        and (
            recipient_type == "judge"
            or recipient_council_role in {"judge", "arbiter", "synthesizer"}
            or (recipient_mode == "council" and recipient_type == "reviewer")
            or (
                recipient_mode == "council"
                and recipient_council_role not in {"proposer", None}
            )
            or (
                (is_swarm_worker or recipient_mode == "council")
                and (
                    "judge" in recipient_name_or_role
                    or "arbiter" in recipient_name_or_role
                )
            )
        )
    )
    if is_blind_judge_recipient:
        leak = _detect_blind_judge_leak(message, payload=payload)
        if leak:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Blind judging violation: judge prompt leaks model identity "
                    f"('{leak}'). Anonymize proposals before invoking judge."
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
        if artifact_dir and raw_recipient:
            try:
                resolved_art = Path(artifact_dir).resolve()
                if resolved_art.parent.name == "brain":
                    sib_state_file = (
                        resolved_art.parent / raw_recipient / "orchestrator_state.json"
                    )
                    if not sib_state_file.exists() and recipient != raw_recipient:
                        sib_state_file = (
                            resolved_art.parent / recipient / "orchestrator_state.json"
                        )
                    if sib_state_file.exists():
                        stage_val = stage_match.group(1).strip() if stage_match is not None else None

                        def _sync_sibling_recipient(
                            target_file: Path = sib_state_file,
                            allowed_val: Optional[List[str]] = parsed_allowed,
                            new_stage: Optional[str] = stage_val,
                            reactivate_impl: bool = (recipient_type == "implementer"),
                        ) -> None:
                            with state_transaction(target_file):
                                sib_state = _load_state(target_file)
                                meta = sib_state.setdefault("self_subagent_meta", {})
                                if allowed_val is not None:
                                    meta["allowed_files"] = allowed_val
                                if new_stage is not None:
                                    meta["stage"] = new_stage
                                    meta.setdefault("stage_history", []).append(new_stage)
                                if reactivate_impl:
                                    meta["status"] = "active"
                                _save_state(target_file, sib_state)

                        if deferred_actions is not None:
                            deferred_actions.append(_sync_sibling_recipient)
                        elif not _has_other_locked_path(sib_state_file):
                            _sync_sibling_recipient()
            except (OSError, ValueError):
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
    subagents = state.get("subagents")
    if not isinstance(subagents, dict):
        subagents = {}
        state["subagents"] = subagents

    if action == "kill_all":
        for meta in subagents.values():
            if isinstance(meta, dict):
                meta["active_branch"] = False
                meta["active_worktree"] = False
                meta["status"] = "killed"
        raw_pending = state.get("pending_spawns")
        if isinstance(raw_pending, list):
            for pending in raw_pending:
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
        if not isinstance(raw_ids, list):
            return {
                "decision": "deny",
                "reason": "[ORCHESTRATOR GUARD] manage_subagents ConversationIds must be a list or string.",
            }
        target_ids = {str(cid).strip().lower() for cid in raw_ids if cid}
        matched_count = 0
        for cid in target_ids:
            if cid in subagents and isinstance(subagents[cid], dict):
                subagents[cid]["active_branch"] = False
                subagents[cid]["active_worktree"] = False
                subagents[cid]["status"] = "killed"
                matched_count += 1
        remaining_pending = []
        killed_left = max(0, len(target_ids) - matched_count)
        for pending in state.get("pending_spawns", []):
            if (
                isinstance(pending, dict)
                and (pending.get("active_worktree") or pending.get("active_branch"))
                and killed_left > 0
            ):
                pending["active_branch"] = False
                pending["active_worktree"] = False
                killed_left -= 1
            remaining_pending.append(pending)
        state["pending_spawns"] = remaining_pending
        _save_state(payload, state)

    return {"decision": "allow"}


def _handle_swarm(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and track swarm tool calls (council anonymity, blind judging, worktrees, EVAL_CMD)."""
    action = str(args.get("action") or "").strip().lower()
    if action in {"status", "models", "title"}:
        return {"decision": "allow"}

    if action == "kill":
        wname = str(args.get("name") or "").strip()
        workers = state.get("swarm_workers")
        if isinstance(workers, dict) and wname in workers and isinstance(workers[wname], dict):
            workers[wname]["pre_kill_active"] = workers[wname].get("status") == "active"
            workers[wname]["status"] = "killed"
            workers[wname]["active_worktree"] = False
            _save_state(payload, state)
            return {"decision": "allow"}
        if wname and isinstance(workers, dict):
            sharded_active = [
                k
                for k, meta in workers.items()
                if isinstance(meta, dict)
                and meta.get("base_name") == wname
                and meta.get("sharded") is True
                and meta.get("status") == "active"
            ]
            if sharded_active:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Swarm group '{wname}' was spawned as sharded workers "
                        f"({', '.join(sharded_active)}). Call swarm(action='kill') with each individual "
                        "worker name to clean up their worktrees."
                    ),
                }
        return {"decision": "allow"}

    if action == "model":
        wname = str(args.get("name") or "").strip()
        new_model = str(args.get("model") or "").strip()
        workers = state.get("swarm_workers")
        if isinstance(workers, dict) and wname and wname not in workers:
            sharded_active = [
                k
                for k, meta in workers.items()
                if isinstance(meta, dict)
                and meta.get("base_name") == wname
                and meta.get("sharded") is True
                and meta.get("status") == "active"
            ]
            if sharded_active:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Swarm group '{wname}' was spawned as sharded workers "
                        f"({', '.join(sharded_active)}). Call swarm(action='model') with each individual "
                        "worker name."
                    ),
                }
        if (
            isinstance(workers, dict)
            and wname in workers
            and isinstance(workers[wname], dict)
            and new_model
        ):
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
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in swarm: '{eval_cmd}'. "
                    "Provide a real test or evaluation script command."
                ),
            }

        mode = _parse_mode_header(brief)
        cohort_id = _parse_cohort_header(brief)
        council_id = _parse_council_id_header(brief)
        council_role = _parse_council_role_header(brief)
        if (
            (mode is not None and mode != "best_of_n" and cohort_id)
            or (mode is not None and mode != "council" and (council_id or council_role))
            or (cohort_id and (council_id or council_role))
        ):
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Conflicting mode headers in swarm brief: cannot mix "
                    "MODE: BEST_OF_N / COHORT with MODE: COUNCIL / COUNCIL_ID / COUNCIL_ROLE "
                    "or attach COHORT/COUNCIL headers to an incompatible MODE."
                ),
            }
        if (council_id or council_role) and mode is None:
            mode = "council"
        if mode == "council" and not council_id:
            council_id = f"council-{_current_invocation_key(payload, state)}"

        if (
            mode == "council"
            or council_id
            or council_role
            or "judge" in name.lower()
            or "proposer" in name.lower()
            or "arbiter" in name.lower()
        ):
            name_leak = _detect_council_name_leak(name)
            if name_leak:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Blind council violation: swarm worker name '{name}' "
                        f"reveals model identity ('{name_leak.strip()}'). "
                        "Use anonymous names like 'proposer-a', 'judge-1'."
                    ),
                }
        is_council_arbiter = (
            council_role in {"judge", "arbiter", "synthesizer"}
            or "judge" in name.lower()
            or "arbiter" in name.lower()
            or (mode == "council" and council_role != "proposer")
        )
        if is_council_arbiter:
            leak = _detect_blind_judge_leak(brief, payload=payload)
            if leak:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Blind judging violation: judge prompt leaks model identity "
                        f"('{leak}'). Anonymize proposals before invoking judge."
                    ),
                }

        if eval_cmd:
            state["active_eval_cmd"] = eval_cmd

        worker_names = _expand_swarm_worker_names(args)
        is_sharded = len(worker_names) > 1
        call_id = _extract_call_id(payload)
        now_ts = time.time()
        is_isolated = workspace.lower() in ISOLATED_WORKSPACES
        swarm_workers = state.setdefault("swarm_workers", {})
        prev_workers: Dict[str, Dict[str, Any]] = {}
        for wname in worker_names:
            existing_worker = swarm_workers.get(wname)
            if isinstance(existing_worker, dict):
                if isinstance(existing_worker.get("_prev_worker"), dict):
                    snap = dict(existing_worker["_prev_worker"])
                else:
                    snap = dict(existing_worker)
                snap.pop("_prev_worker", None)
                prev_workers[wname] = snap
        for wname in worker_names:
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
            if wname in prev_workers:
                worker_record["_prev_worker"] = prev_workers[wname]
            swarm_workers[wname] = worker_record

        if mode == "council" and council_id:
            is_judge_name = (council_role == "judge") or (
                council_role is None and "judge" in name.lower()
            )
            for wname in worker_names:
                _record_council_member(
                    state,
                    council_id,
                    wname,
                    council_role,
                    model,
                    is_judge_type=is_judge_name,
                )

        _save_state(payload, state)
        return {"decision": "allow"}

    return {"decision": "allow"}


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
                and not _is_artifact_path(target, payload)
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

        if tool_name == "run_command":
            plan_allowed = state.get("plan_allowed_files")
            if plan_allowed is not None:
                raw_cwd = args.get("Cwd")
                cwd = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else str(PROJECT_ROOT)
                raw_cmd = args.get("CommandLine")
                cmd = raw_cmd.strip() if isinstance(raw_cmd, str) else ""
                sub_cmds = _split_shell_commands(cmd)
                return _validate_shell_allowlist(
                    cmd, sub_cmds, cwd, plan_allowed, payload, allowlist_label="plan ALLOWED_FILES"
                )
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
        except (OSError, ValueError):
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
        try:
            is_file = p.is_file()
        except (OSError, ValueError):
            is_file = False
        if is_file:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                all_lines = text.splitlines()
            except (OSError, ValueError):
                all_lines = []
            if start_line is not None or end_line is not None:
                try:
                    s = max(1, int(start_line) if start_line is not None else 1)
                    e = min(
                        len(all_lines),
                        int(end_line) if end_line is not None else len(all_lines),
                    )
                    if s <= e:
                        sliced = all_lines[s - 1 : e]
                        lines_cost = len(sliced)
                        bytes_cost = sum(len(x.encode("utf-8")) + 1 for x in sliced)
                    else:
                        lines_cost = 0
                        bytes_cost = 0
                except (ValueError, TypeError, OverflowError):
                    lines_cost = min(800, len(all_lines)) or DEFAULT_UNBOUNDED_READ_LINES
                    bytes_cost = (
                        sum(len(x.encode("utf-8")) + 1 for x in all_lines)
                        or DEFAULT_UNBOUNDED_READ_BYTES
                    )
            else:
                lines_cost = min(800, len(all_lines))
                try:
                    bytes_cost = min(46080, p.stat().st_size)
                except (OSError, ValueError):
                    bytes_cost = sum(len(x.encode("utf-8")) + 1 for x in all_lines)
        else:
            if start_line is not None and end_line is not None:
                try:
                    lines_cost = max(1, int(end_line) - int(start_line) + 1)
                    bytes_cost = lines_cost * 40
                except (ValueError, TypeError, OverflowError):
                    lines_cost = DEFAULT_UNBOUNDED_READ_LINES
                    bytes_cost = DEFAULT_UNBOUNDED_READ_BYTES
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
        if not target or _is_artifact_path(target, payload):
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except (OSError, ValueError):
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
        raw_cmd = args.get("CommandLine")
        cmd = raw_cmd.strip() if isinstance(raw_cmd, str) else ""
        if _is_safe_git_inspection_command(cmd):
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


def _is_unsafe_declared_eval(ev: str) -> bool:
    """Return True if a declared EVAL_CMD contains mutating shell verbs, shell/exec wrappers, or awk/sed."""
    if not ev:
        return False
    return bool(
        MUTATING_SHELL_CMD_RE.search(ev)
        or SHELL_EXEC_WRAPPER_RE.search(ev)
        or re.search(
            r"\b(?:[gmn]?awk|sed|perl|node|ruby|php|lua|(?:ba|da|z|k|a|c|tc|mk|rba)?sh|fish|eval|source)\b",
            ev,
            re.I,
        )
    )


def _literal_str_ast(node: ast.AST, depth: int = 0) -> Optional[str]:
    """Extract constant string or simple literal '+' concatenation from an AST node."""
    if depth > 16:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if len(node.value) <= 65536 else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_str_ast(node.left, depth + 1)
        right = _literal_str_ast(node.right, depth + 1)
        if left is not None and right is not None and len(left) + len(right) <= 65536:
            return left + right
    return None


def _handle_run_workflow(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Dict[str, Any],
    roots: List[Path],
) -> Dict[str, Any]:
    """Validate run_workflow calls against subagent role restrictions, allowed roots, closed-world script allowlist, and EVAL_CMD."""
    if _is_strict_readonly_subagent(payload, state):
        sub_type = _get_subagent_type(payload, state) or "readonly"
        return {
            "decision": "deny",
            "reason": (
                f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly "
                "read-only. Workflow execution is prohibited."
            ),
        }
    if _is_scratch_only_subagent(payload, state):
        sub_type = _get_subagent_type(payload, state)
        role_word = "Council" if sub_type == "council" else "Research"
        return {
            "decision": "deny",
            "reason": (
                f"[ORCHESTRATOR GUARD] {role_word} subagents may only write markdown reports "
                "to <artifactDirectoryPath>/scratch/*.md. Workflow execution is prohibited."
            ),
        }
    if _get_subagent_allowed_files(payload, state) is not None:
        return {
            "decision": "deny",
            "reason": (
                "[ORCHESTRATOR GUARD] Implementer subagents restricted by ALLOWED_FILES "
                "cannot execute unbounded workflows via run_workflow."
            ),
        }
    if (
        _is_root_orchestrator(payload, state)
        and state.get("plan_execute_phase")
        and state.get("plan_allowed_files") is not None
    ):
        return {
            "decision": "deny",
            "reason": (
                "[ORCHESTRATOR GUARD] Root orchestrator restricted by ALLOWED_FILES "
                "cannot execute unbounded workflows via run_workflow."
            ),
        }

    desc_eval = _parse_eval_cmd_header(str(args.get("Description") or ""))
    raw_script = str(args.get("Script") or "")
    script_eval = (
        _parse_eval_cmd_header(re.sub(r"(?m)^[ \t]*#+[ \t]*", "", raw_script))
        if raw_script
        else None
    )
    for candidate_eval in (desc_eval, script_eval):
        if candidate_eval is not None and _is_trivial_eval_command(candidate_eval):
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in run_workflow: '{candidate_eval}'. "
                    "Provide a real test or evaluation script command."
                ),
            }
        if candidate_eval is not None and _is_unsafe_declared_eval(candidate_eval):
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Unsafe mutating or wrapper EVAL_CMD rejected in run_workflow: '{candidate_eval}'."
                ),
            }

    _tamper_re = re.compile(
        r"orchestrator_state|orchestrator_guard|\.agents(?:/|\\|\b)|philharmonic_orchestra|hooks\.json"
        r"|\b(?:from\s+[A-Za-z0-9_.]*\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|cli)\b|import\s+(?:\\\r?\n|[^;\n])*\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|cli)\b)"
        r"|(?:\s|^)-[A-Za-z]*m\s*(?:guard|orchestrator_guard|state_store|cli)\b"
        r"|\.state\b",
        re.I,
    )
    _wf_base_cwd = str(args.get("Cwd") or PROJECT_ROOT)
    _wf_wipe_glob_re = re.compile(
        r"(?:^|[\s'\"`/])(?:\.\[[^\]]+\]gents|\.\[\^[^\]]+\]|\bhooks\.\*)", re.I
    )
    _wf_unsafe_names = _PY_MUTATING_NAMES | {
        "dir", "rmtree", "remove", "unlink", "rmdir", "rename", "replace",
        "chmod", "chown", "system", "popen", "Popen", "check_call", "check_output",
        "attrgetter", "methodcaller", "itemgetter",
    }
    _wf_unsafe_attrs = _PY_MUTATING_ATTRS | _PY_FRAME_ATTRS | {
        "Popen", "check_call", "check_output",
    }

    def _build_wf_state(text: str) -> Dict[str, Any]:
        wf_state = dict(state)
        wf_cohorts = dict(wf_state.get("cohorts") or {})
        sc_eval = _parse_eval_cmd_header(re.sub(r"(?m)^[ \t]*#+[ \t]*", "", text))
        for idx_e, ev in enumerate((desc_eval, sc_eval)):
            if ev and not _is_trivial_eval_command(ev) and not _is_unsafe_declared_eval(ev):
                wf_cohorts[f"__wf_eval_{idx_e}"] = {"eval_cmd": ev}
        wf_state["cohorts"] = wf_cohorts
        return wf_state

    def _workflow_script_violates_allowlist(script_text: str) -> bool:
        if not script_text or not script_text.strip():
            return False
        if _has_nul_or_surrogate_input(script_text):
            return True
        sc_eval = _parse_eval_cmd_header(re.sub(r"(?m)^[ \t]*#+[ \t]*", "", script_text))
        if sc_eval and (_is_trivial_eval_command(sc_eval) or _is_unsafe_declared_eval(sc_eval)):
            return True
        wf_state = _build_wf_state(script_text)
        try:
            tree: Optional[ast.AST] = ast.parse(script_text)
        except (RecursionError, MemoryError):
            return True
        except (SyntaxError, ValueError, TypeError):
            tree = None

        if tree is not None:
            if _tamper_re.search(script_text):
                return True
            try:
                call_func_ids = {
                    id(n.func)
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if _tamper_re.search(node.value):
                            return True
                    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                        lit_s = _literal_str_ast(node)
                        if lit_s and _tamper_re.search(lit_s):
                            return True
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.name or ""
                            if name.startswith("json.") or name.split(".", 1)[0] not in _SAFE_WORKFLOW_STDLIB_MODULES:
                                return True
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        if node.level != 0 or not mod or mod.startswith("json.") or mod.split(".", 1)[0] not in _SAFE_WORKFLOW_STDLIB_MODULES:
                            return True
                        if mod == "json" and any((a.name or "") not in _SAFE_JSON_IMPORTS for a in node.names):
                            return True
                    elif isinstance(node, ast.Name):
                        if node.id.startswith("_") or node.id in _wf_unsafe_names:
                            return True
                        if node.id == "call_tool" and id(node) not in call_func_ids:
                            return True
                    elif isinstance(node, ast.Attribute):
                        if node.attr.startswith("_") or node.attr in _wf_unsafe_attrs:
                            return True
                    elif isinstance(node, ast.Call):
                        fn_name = (
                            node.func.id
                            if isinstance(node.func, ast.Name)
                            else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                        )
                        if fn_name == "call_tool":
                            tool_arg: Optional[ast.AST] = node.args[0] if node.args else None
                            payload_arg: Optional[ast.AST] = node.args[1] if len(node.args) >= 2 else None
                            for kw in node.keywords:
                                if kw.arg in {"name", "tool_name"}:
                                    tool_arg = kw.value
                                elif kw.arg in {"args", "arguments"}:
                                    payload_arg = kw.value
                            tool_val = _literal_str_ast(tool_arg) if tool_arg is not None else None
                            if tool_val is None or tool_val in MUTATION_TOOLS:
                                return True
                            if tool_val == "run_command":
                                cmd_val: Optional[str] = None
                                if isinstance(payload_arg, ast.Dict):
                                    for k_n, v_n in zip(payload_arg.keys, payload_arg.values):
                                        if k_n is not None and _literal_str_ast(k_n) == "CommandLine":
                                            cmd_val = _literal_str_ast(v_n)
                                            break
                                if not cmd_val or not _is_root_allowlisted_command(
                                    cmd_val, _split_shell_commands(cmd_val), _wf_base_cwd, payload, wf_state
                                ):
                                    return True
            except (RecursionError, MemoryError):
                return True
            return False

        if _tamper_re.search(script_text) or _wf_wipe_glob_re.search(script_text):
            return True
        cleaned_wf, h_subst, h_unclosed = _preprocess_shell_heredocs_and_comments(script_text)
        if h_subst or h_unclosed or "<<" in _strip_shell_quoted_literals(cleaned_wf):
            return True
        exec_lines = [ln for ln in script_text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        exec_script = "\n".join(exec_lines)
        if not exec_script.strip():
            return False
        return not _is_root_allowlisted_command(
            exec_script, _split_shell_commands(exec_script), _wf_base_cwd, payload, wf_state
        )

    if raw_script and _workflow_script_violates_allowlist(raw_script):
        return {
            "decision": "deny",
            "reason": (
                "[ORCHESTRATOR GUARD] Workflow Script violates closed-world allowlist "
                "or attempts to access/modify orchestrator guard files or state."
            ),
        }

    cwd = args.get("Cwd")
    if cwd:
        cwd_str = str(cwd)
        if (
            _is_guard_state_target(cwd_str, payload=payload)
            or _is_guard_plugin_target(cwd_str)
            or _is_guard_plugin_cwd(cwd_str)
        ):
            return {
                "decision": "deny",
                "reason": "[ORCHESTRATOR GUARD] Workflow Cwd cannot target guard state or plugin files.",
            }
        if not _is_within_allowed_roots(cwd_str, roots):
            return {"decision": "ask"}

    script_path = args.get("ScriptPath")
    if script_path:
        sp_str = str(script_path)
        base_cwd = str(cwd) if cwd else str(PROJECT_ROOT)
        sp_resolved_str = _resolve_cmd_target(sp_str, base_cwd)
        if (
            _is_guard_state_target(sp_str, payload=payload)
            or _is_guard_state_target(sp_resolved_str, payload=payload)
            or _is_guard_plugin_target(sp_str, base_cwd)
            or _is_guard_plugin_target(sp_resolved_str, base_cwd)
        ):
            return {
                "decision": "deny",
                "reason": "[ORCHESTRATOR GUARD] Workflow ScriptPath cannot target guard state or plugin files.",
            }
        if not _is_within_allowed_roots(sp_resolved_str, roots):
            return {"decision": "ask"}
        try:
            raw_sp = Path(os.path.expanduser(sp_str))
            sp_candidates = [Path(sp_resolved_str)]
            if raw_sp.is_absolute():
                sp_candidates.append(raw_sp.resolve())
            for sp_resolved in sp_candidates:
                if sp_resolved.name == "orchestrator_guard.py":
                    return {
                        "decision": "deny",
                        "reason": "[ORCHESTRATOR GUARD] Workflow ScriptPath cannot target orchestrator_guard.py.",
                    }
                if sp_resolved.is_file():
                    sp_content = sp_resolved.read_text(encoding="utf-8", errors="replace")
                    if _workflow_script_violates_allowlist(sp_content):
                        return {
                            "decision": "deny",
                            "reason": (
                                "[ORCHESTRATOR GUARD] Workflow ScriptPath file violates closed-world "
                                "allowlist or attempts to access/modify orchestrator guard files or state."
                            ),
                        }
        except (OSError, ValueError):
            pass

    eval_cmd = desc_eval or script_eval
    if eval_cmd:
        state["active_eval_cmd"] = eval_cmd
        _save_state(payload, state)
    return {"decision": "allow"}

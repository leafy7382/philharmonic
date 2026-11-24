"""Subagent identity tracking, transcript pairing, role resolution, and context reminder injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import posixpath
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from guard.constants import (
    ALLOWED_SYSTEM_PATH_PREFIXES,
    ISOLATED_WORKSPACES,
    MAX_CONSECUTIVE_FAILURES,
    PENDING_SPAWN_TTL_SECONDS,
    PROJECT_ROOT,
    SCRATCH_ONLY_SUBAGENT_TYPES,
    SLASH_COMMAND_RE,
    STRICT_READONLY_SUBAGENT_TYPES,
    SUBAGENT_SPAWN_RE,
    UUID_RE,
    infer_subagent_tier,
)
from guard.state_store import (
    _current_invocation_key,
    _has_other_locked_path,
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)


def _resolved_workspace_paths(raw_wps: Any) -> List[Path]:
    """Resolve valid string entries from a workspacePaths list/tuple."""
    out: List[Path] = []
    for wp in (raw_wps if isinstance(raw_wps, (list, tuple)) else ()):
        if isinstance(wp, str) and wp:
            try:
                out.append(Path(wp).resolve())
            except (OSError, ValueError):
                pass
    return out


def _brain_dir_from_artifact(artifact_dir: Any) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (resolved_artifact_dir, brain_dir_if_brain_parent) or (None, None)."""
    if not artifact_dir:
        return None, None
    try:
        resolved_art = Path(artifact_dir).resolve()
        return resolved_art, (resolved_art.parent if resolved_art.parent.name == "brain" else None)
    except (OSError, ValueError):
        return None, None


def _first_active_pending_idx(pending: Any) -> Optional[int]:
    """Return index of first active entry in pending_spawns, or None."""
    if not isinstance(pending, list):
        return None
    return next((i for i, p in enumerate(pending) if isinstance(p, dict) and p.get("status", "active") == "active"), None)


def _allowed_roots(payload: Dict[str, Any]) -> List[Path]:
    """Return all workspace, artifact, brain, and worktree roots permitted for this invocation."""
    roots: List[Path] = [PROJECT_ROOT, *_resolved_workspace_paths(payload.get("workspacePaths"))]
    resolved_art, brain_dir = _brain_dir_from_artifact(payload.get("artifactDirectoryPath"))
    if resolved_art is not None:
        roots.append(resolved_art)
        if brain_dir is not None:
            roots.extend((brain_dir, brain_dir.parent / "worktrees"))
    return roots


def _is_within_allowed_roots(target: Optional[str], roots: List[Path]) -> bool:
    """Return True if target resolves inside any root in roots."""
    if not isinstance(target, str) or not target or "\x00" in target or any(0xD800 <= ord(ch) <= 0xDFFF for ch in target):
        return False
    if "$" in target and "'" in target:
        from guard.shell_tokens import _has_nul_or_surrogate_input

        if _has_nul_or_surrogate_input(target):
            return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except (OSError, ValueError, TypeError):
        return False
    return any(resolved == r or r in resolved.parents for r in roots)


def _command_stays_in_workspace(command_line: str, cwd: str, roots: List[Path], *, parsed: Optional[Any] = None) -> bool:
    """Return True unless the command targets obvious paths outside allowed roots."""
    from guard.shell_tokens import _advance_effective_cwd, parse_shell_command

    p = parsed if parsed is not None else parse_shell_command(command_line)
    if p.has_nul_surr or p.has_ansi_c or p.has_subshell:
        return False

    try:
        base_dir = Path(os.path.expanduser(cwd)).resolve() if cwd else PROJECT_ROOT
    except (OSError, ValueError):
        return False

    def _in_bounds(raw_path: str, allow_sys: bool = True) -> bool:
        norm = posixpath.normpath(re.sub(r"^/+", "/", raw_path))
        return (allow_sys and (norm == "/tmp" or norm.startswith(ALLOWED_SYSTEM_PATH_PREFIXES))) or _is_within_allowed_roots(norm, roots)

    def _tok_ok(tok: str, cur_dir: Path, cmd_verb: str) -> bool:
        if tok.startswith("-C") and len(tok) > 2:
            tok = tok[2:]
        elif "=" in tok and not tok.startswith("=") and (tok.startswith("--") or tok.split("=", 1)[1].startswith(("/", "~", "$")) or ".." in tok.split("=", 1)[1]):
            tok = tok.split("=", 1)[1]
        if "://" in tok:
            scheme, rest = tok.split("://", 1)
            tok = ("/" + rest.lstrip("/")) if scheme.lower() == "file" else ""
        if not tok or (tok.startswith("-") and not tok.startswith("/")):
            return True
        if tok.startswith(("~-", "~+")):
            return False
        if tok.startswith("~"):
            try:
                return (exp := os.path.expanduser(tok)) != tok and _in_bounds(exp, allow_sys=False)
            except (OSError, ValueError):
                return False
        for var in ("HOME", "OLDPWD"):
            for pat in (f"${var}", f"${{{var}}}"):
                if pat in tok:
                    val = os.environ.get(var) or (str(Path.home()) if var == "HOME" else "")
                    return bool(val) and _in_bounds(tok.replace(pat, val), allow_sys=False)
        if tok.startswith("/"):
            if cmd_verb in {"awk", "gawk", "mawk", "nawk", "sed"} and re.fullmatch(r"/[^/]+/[dgip]?", tok):
                return True
            return _in_bounds(tok, allow_sys=True)
        try:
            parts = Path(tok).parts
            if ".." in parts:
                return _in_bounds(str((cur_dir / tok).resolve()), allow_sys=False)
            cand, cur = cur_dir / tok, cur_dir
            for part in parts:
                if (cur := cur / part).is_symlink():
                    return _in_bounds(str(cur.resolve()), allow_sys=False)
            return _in_bounds(str(cand.resolve()), allow_sys=False) if (cand.is_symlink() or cand.exists()) else True
        except (OSError, ValueError):
            return False

    cur_dir = str(base_dir)
    for sub in p.subcmds:
        new_cwd, is_cd, cd_unv = _advance_effective_cwd(sub.raw, cur_dir, None, parsed_sub=sub)
        if is_cd:
            if cd_unv or not _in_bounds(new_cwd, allow_sys=True):
                return False
            cur_dir = new_cwd
        else:
            toks = [tok for tok, _, _ in sub.raw_tokens]
            cmd_verb = next(
                (posixpath.basename(t).lower() for t in toks
                 if t not in {"(", "{", "!", "if", "then", "elif", "else", "while", "until", "do"}
                 and not re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", t)),
                "",
            )
            if any(not _tok_ok(tok, Path(cur_dir), cmd_verb) for tok in toks):
                return False
    return True


def _is_parent_orchestrator(cid: str, state: Dict[str, Any]) -> bool:
    """Return True if cid is recorded as the root/parent orchestrator."""
    subagents = state.get("subagents")
    if not cid or (isinstance(subagents, dict) and cid in subagents):
        return False
    if state.get("root_conversation_id") == cid:
        return True
    prefix, recorded = f'["{cid}"', state.get("recorded_spawn_calls")
    return isinstance(recorded, list) and any(isinstance(c, str) and c.startswith(prefix) for c in recorded)


def _normalize_subagent_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize type/type_name, model_tier, optional cohort/council fields, and active_worktree flag."""
    normalized = dict(meta)
    if "type" in normalized and "type_name" not in normalized:
        normalized["type_name"] = normalized["type"]
    elif "type_name" in normalized and "type" not in normalized:
        normalized["type"] = normalized["type_name"]
    if raw_tier := str(normalized.get("model_tier") or "").strip().lower():
        normalized["model_tier"] = raw_tier
    elif inferred := infer_subagent_tier(normalized.get("model"), normalized.get("type_name") or normalized.get("type")):
        normalized["model_tier"] = inferred
    for field_name in ("candidate_id", "council_id", "council_role"):
        if meta.get(field_name) is not None:
            if val := str(meta[field_name]).strip():
                normalized[field_name] = val
            else:
                normalized.pop(field_name, None)
    if "active_worktree" not in normalized:
        ws = str(normalized.get("workspace") or "").strip().lower()
        if ws in ISOLATED_WORKSPACES or normalized.get("active_branch"):
            normalized["active_worktree"] = bool(normalized.get("active_branch") or ws in ISOLATED_WORKSPACES)
    return normalized


def _workspaces_compatible(caller_payload: Dict[str, Any], sib_data: Dict[str, Any]) -> bool:
    """Check whether caller payload and sibling state belong to compatible workspaces."""
    caller_res = {str(p) for p in _resolved_workspace_paths(caller_payload.get("workspacePaths"))}
    sib_res = {str(p) for p in _resolved_workspace_paths(sib_data.get("workspace_roots"))}
    if not caller_res or not sib_res or (caller_res & sib_res):
        return True
    for cp in caller_res:
        c_parts = Path(cp).parts
        for sp in sib_res:
            s_parts = Path(sp).parts
            if ("worktrees" in c_parts and "worktrees" in s_parts and c_parts[: c_parts.index("worktrees") + 1] == s_parts[: s_parts.index("worktrees") + 1]) or ("worktrees" in c_parts and sp in cp) or ("worktrees" in s_parts and cp in sp):
                return True
    return False


def _read_sibling_snapshot(sibling_state_path: Path) -> Optional[Dict[str, Any]]:
    """Read sibling state safely without acquiring LOCK_SH when another state lock is already held."""
    try:
        if _has_other_locked_path(sibling_state_path):
            data = json.loads(sibling_state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        return _load_state(sibling_state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _is_non_subagent_user_session(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True when payload/state clearly identifies an independent root user session."""
    if payload.get("isSubagent") is True:
        return False
    if payload.get("isSubagent") is False or (state.get("session_evaluated") is True and state.get("active") is False):
        return True
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    raw_wps = payload.get("workspacePaths")
    wps_iter = raw_wps if isinstance(raw_wps, (list, tuple)) else ()
    if (
        state.get("session_evaluated") is True
        and not state.get("self_subagent_meta")
        and not state.get("is_subagent_session")
        and not (cid and isinstance(subagents, dict) and cid in subagents)
        and not any(isinstance(wp, str) and "/worktrees/" in wp.replace("\\", "/") for wp in wps_iter)
    ):
        return True
    return "lastUserInput" in payload and payload.get("lastUserInput") is not None and not _is_slash_command_triggered(payload)


def _with_parent_meta(meta: Dict[str, Any], sib_data: Dict[str, Any], sib_path: Path) -> Dict[str, Any]:
    """Normalize sibling subagent metadata and attach parent_conversation_id if absent."""
    norm = _normalize_subagent_meta(meta)
    norm.setdefault("parent_conversation_id", sib_data.get("root_conversation_id") or sib_path.parent.name)
    return norm


def _claim_sibling_subagent(payload: Dict[str, Any], cid: str, can_claim_pending: bool) -> Optional[Dict[str, Any]]:
    """Inspect sibling brain states to resolve or claim subagent metadata for cid."""
    resolved_art, brain_dir = _brain_dir_from_artifact(payload.get("artifactDirectoryPath"))
    if brain_dir is None or resolved_art is None:
        return None
    try:
        for sib_path in brain_dir.glob("*/orchestrator_state.json"):
            if sib_path.parent == resolved_art:
                continue
            sib_data = _read_sibling_snapshot(sib_path)
            if not isinstance(sib_data, dict) or not _workspaces_compatible(payload, sib_data):
                continue
            sib_subs = sib_data.get("subagents")
            if cid and isinstance(sib_subs, dict) and isinstance(sib_subs.get(cid), dict):
                return _with_parent_meta(sib_subs[cid], sib_data, sib_path)
            if (
                not can_claim_pending
                or _first_active_pending_idx(sib_data.get("pending_spawns")) is None
                or cid == sib_data.get("root_conversation_id")
                or _has_other_locked_path(sib_path)
            ):
                continue
            with state_transaction(sib_path):
                fresh = _load_state(sib_path)
                if not _workspaces_compatible(payload, fresh) or _is_parent_orchestrator(cid, fresh):
                    continue
                fresh_subs = fresh.get("subagents")
                if cid and isinstance(fresh_subs, dict) and isinstance(fresh_subs.get(cid), dict):
                    return _with_parent_meta(fresh_subs[cid], fresh, sib_path)
                raw_pend = fresh.get("pending_spawns")
                active_idx = _first_active_pending_idx(raw_pend)
                if active_idx is not None and isinstance(raw_pend, list) and time.time() - sib_path.stat().st_mtime < PENDING_SPAWN_TTL_SECONDS:
                    raw_entry = raw_pend.pop(active_idx) if cid else raw_pend[active_idx]
                    claimed = _with_parent_meta(raw_entry, fresh, sib_path)
                    claimed.setdefault("status", "active")
                    if cid:
                        fresh["pending_spawns"] = raw_pend
                        fresh.setdefault("subagents", {})[cid] = claimed
                        _save_state(sib_path, fresh)
                    return claimed
    except (OSError, ValueError):
        pass
    return None


def _is_known_subagent(payload: Dict[str, Any], state: Dict[str, Any], *, persist_self_meta: bool = True) -> bool:
    """Return True if this invocation belongs to a spawned worker subagent."""
    cid = str(payload.get("conversationId") or "").strip().lower()
    if _is_parent_orchestrator(cid, state):
        return False
    if state.get("self_subagent_meta"):
        return True
    local_subagents = state.get("subagents")
    if cid and isinstance(local_subagents, dict) and isinstance(local_subagents.get(cid), dict):
        return True
    raw_tc = payload.get("toolCall")
    tool_name = raw_tc.get("name") if isinstance(raw_tc, dict) and isinstance(raw_tc.get("name"), str) else ""
    can_claim = not _is_non_subagent_user_session(payload, state) and tool_name not in {"exit_plan_mode", "enter_plan_mode"}
    if (claimed_meta := _claim_sibling_subagent(payload, cid, can_claim)) is not None:
        state["self_subagent_meta"] = claimed_meta
        if persist_self_meta:
            _save_state(payload, state)
        return True
    if payload.get("isSubagent") is True or state.get("is_subagent_session") is True:
        return True
    raw_wps = payload.get("workspacePaths")
    return isinstance(raw_wps, (list, tuple)) and any(isinstance(wp, str) and "/worktrees/" in wp.replace("\\", "/") for wp in raw_wps)


def _preclaim_subagent_meta(payload: Dict[str, Any]) -> None:
    """Resolve and persist sibling subagent metadata before acquiring state_transaction(payload)."""
    raw_tc = payload.get("toolCall")
    raw_name = raw_tc.get("name") if isinstance(raw_tc, dict) else None
    if _is_slash_command_triggered(payload) or (raw_name if isinstance(raw_name, str) else "") in {"exit_plan_mode", "enter_plan_mode"}:
        return
    pre_state = _load_state(payload)
    cid = str(payload.get("conversationId") or "").strip().lower()
    if (
        pre_state.get("self_subagent_meta")
        or _is_non_subagent_user_session(payload, pre_state)
        or _is_parent_orchestrator(cid, pre_state)
        or (cid and isinstance(pre_state.get("subagents"), dict) and cid in pre_state["subagents"])
    ):
        return
    _is_known_subagent(payload, pre_state, persist_self_meta=False)
    if (claimed := pre_state.get("self_subagent_meta")) is not None:
        with state_transaction(payload):
            own_state = _load_state(payload)
            own_state.setdefault("self_subagent_meta", claimed)
            own_state["is_subagent_session"] = True
            _save_state(payload, own_state)


def _sync_subagents_from_transcript(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Scan transcript.jsonl incrementally to pair pending_spawns with conversationIds."""
    transcript_path = payload.get("transcriptPath")
    try:
        if not isinstance(transcript_path, str) or not transcript_path or not os.path.exists(transcript_path):
            return False
    except (OSError, ValueError):
        return False

    pending: List[Dict[str, Any]] = state.get("pending_spawns") if isinstance(state.get("pending_spawns"), list) else []
    subagents: Dict[str, Dict[str, Any]] = state.setdefault("subagents", {}) if isinstance(state.get("subagents"), dict) else {}
    state["subagents"] = subagents
    parent_cid = str(payload.get("conversationId") or "").strip().lower()
    changed = False

    try:
        file_size = os.path.getsize(transcript_path)
        offset = int(state.get("transcript_offset", 0))
        offset = 0 if (offset < 0 or offset > file_size) else offset
        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            raw_new_offset = fh.tell()
        if not chunk:
            return changed
        process_chunk, new_offset = chunk, raw_new_offset
        if not chunk.endswith(b"\n"):
            last_nl = chunk.rfind(b"\n")
            if last_nl != -1:
                process_chunk, new_offset = chunk[: last_nl + 1], offset + last_nl + 1
            elif chunk.strip().startswith(b"{") and not chunk.strip().endswith(b"}"):
                return changed
        if new_offset != offset:
            state["transcript_offset"] = new_offset
            changed = True

        for raw_line in process_chunk.decode("utf-8", errors="replace").splitlines():
            if not raw_line.strip() or any(m in raw_line for m in ("docs/hooks.md", "docs/", "view_file", "read_file", "File Path: `file://")):
                continue
            if "Created the following subagent" not in raw_line and "invoke_subagent" not in raw_line:
                continue
            matches = list(SUBAGENT_SPAWN_RE.finditer(raw_line))
            if not matches and "Created the following subagent" in raw_line:
                matches = list(UUID_RE.finditer(raw_line))
            for match in matches:
                cid = match.group(1).lower()
                if cid == parent_cid or cid in subagents:
                    continue
                if (active_idx := _first_active_pending_idx(pending)) is not None:
                    spawn_meta = pending.pop(active_idx)
                    if isinstance(spawn_meta, dict):
                        spawn_meta = _normalize_subagent_meta(spawn_meta)
                        spawn_meta.setdefault("status", "active")
                    subagents[cid] = spawn_meta
                    changed = True
    except (OSError, ValueError):
        return changed
    return changed


def _is_slash_command_triggered(payload: Dict[str, Any]) -> bool:
    """Return True if the user prompt or tool call activates the /philharmonic workflow."""
    if payload.get("isSubagent") is True:
        return False
    user_input = payload.get("lastUserInput")
    if isinstance(user_input, str):
        if (
            re.search(r"(?m)^\s*(?:ALLOWED_FILES|SPEC_FILE|REVIEW_LENS|ROLE|WORK_DIR|TASK|OBJECTIVE|FILES|CONSTRAINTS|ACCEPTANCE_CRITERIA):\s*\S", user_input)
            and not SLASH_COMMAND_RE.search(user_input)
        ):
            return False
        if SLASH_COMMAND_RE.search(user_input) or "skills/philharmonic/SKILL.md" in user_input:
            return True

    raw_tc = payload.get("toolCall")
    if isinstance(raw_tc, dict) and raw_tc:
        tool_name = str(raw_tc.get("name") or "").lower()
        args = raw_tc.get("args") if isinstance(raw_tc.get("args"), dict) else {}
        if tool_name in {"read_file", "view_file"}:
            return any(
                isinstance(args.get(k), str) and str(args.get(k)).replace("\\", "/").endswith("skills/philharmonic/SKILL.md")
                for k in ("Path", "FilePath", "file_path", "path", "AbsolutePath")
            )
        if tool_name == "lookup_skill":
            return str(args.get("skill_name") or args.get("SkillName") or "").strip().lower() == "philharmonic"
        if tool_name == "lookup_skills":
            names = args.get("skill_names") or args.get("SkillNames") or []
            return isinstance(names, list) and any(str(s).strip().lower() == "philharmonic" for s in names)
    return False


def _bind_root_session_if_unset(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Bind root_conversation_id on state when caller is a non-subagent root session and it is unset."""
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    if (
        cid and not state.get("root_conversation_id") and payload.get("isSubagent") is not True
        and not state.get("self_subagent_meta") and not state.get("is_subagent_session")
        and not (isinstance(subagents, dict) and cid in subagents)
    ):
        state["root_conversation_id"] = cid
        return True
    return False


def _ensure_session_active(payload: Dict[str, Any], state: Dict[str, Any], state_path: Optional[Path] = None) -> bool:
    """Activate, pass through, or mark inactive based on slash command, subagent, or root session state."""
    if _is_slash_command_triggered(payload):
        state["active"] = state["session_evaluated"] = True
        _bind_root_session_if_unset(payload, state)
        if state_path is not None:
            _save_state(state_path, state)
        return True
    if state.get("active") is True:
        return True
    if _is_non_subagent_user_session(payload, state):
        if not state.get("session_evaluated"):
            state["session_evaluated"], state["active"] = True, False
            if state_path is not None:
                _save_state(state_path, state)
        return False
    if _is_known_subagent(payload, state):
        state["active"] = state["session_evaluated"] = True
        cid = str(payload.get("conversationId") or "").strip().lower()
        subagents = state.get("subagents")
        if not (cid and isinstance(subagents, dict) and cid in subagents) and not state.get("root_conversation_id"):
            state["is_subagent_session"] = True
        if state_path is not None:
            _save_state(state_path, state)
        return True
    return bool(state.get("active", False)) if state.get("session_evaluated") is True else True


def handle_pre_invocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle pre-invocation hook: track root/plan state or inject subagent isolation reminder."""
    _preclaim_subagent_meta(payload)
    with state_transaction(payload):
        state_path, state = _state_file_path(payload), _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}
        cid = str(payload.get("conversationId") or "").strip().lower()

        if not _is_known_subagent(payload, state):
            changed = False
            if cid and not state.get("root_conversation_id"):
                state["root_conversation_id"] = cid
                changed = True

            user_input = str(payload.get("lastUserInput") or "")
            if state.get("plan_execute_phase"):
                norm_input = user_input.strip()
                if (
                    "[plan review]" in user_input.lower()
                    or (bool(norm_input) and bool(state.get("plan_execute_turn_finished")))
                    or (bool(norm_input) and bool(state.get("plan_execute_consumed"))
                        and state.get("plan_execute_user_input") is not None and norm_input != state.get("plan_execute_user_input"))
                ):
                    state.update({
                        "plan_execute_phase": False, "plan_execute_consumed": False, "plan_execute_user_input": None,
                        "plan_execute_turn_finished": False, "plan_allowed_files": None, "active_eval_cmd": None,
                    })
                    changed = True
                else:
                    if not state.get("plan_execute_consumed"):
                        state["plan_execute_consumed"] = True
                        changed = True
                    if norm_input and not state.get("plan_execute_user_input"):
                        state["plan_execute_user_input"] = norm_input
                        changed = True

            if (wps := [str(p) for p in _resolved_workspace_paths(payload.get("workspacePaths"))]) and state.get("workspace_roots") != wps:
                state["workspace_roots"] = wps
                changed = True

            inv_num, budget_map = payload.get("invocationNum"), state.get("direct_budget")
            if inv_num is not None and inv_num not in (0, "0"):
                prev_inv = state.get("current_invocation_num")
                if prev_inv != inv_num:
                    state["current_invocation_num"], changed = inv_num, True
                if isinstance(budget_map, dict):
                    for bkey in (str(inv_num), "0"):
                        if budget_map.pop(bkey, None) is not None:
                            changed = True
                if inv_num in (1, "1") or (prev_inv is not None and str(prev_inv) != str(inv_num)):
                    from guard.state_store import _get_circuit_breaker

                    cb = _get_circuit_breaker(state, payload)
                    if cb.get("failure_streak") or cb.get("tripped") or cb.get("replan_count"):
                        cb.update({"failure_streak": 0, "tripped": False, "seen_failures": [], "last_error": "", "replan_count": 0})
                        changed = True
            elif isinstance(budget_map, dict) and budget_map.pop("0", None) is not None:
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
            "injectSteps": [{
                "ephemeralMessage": (
                    "[ORCHESTRATOR GUARD] Subagent context isolation active: ignore root orchestrator "
                    "delegation rules in SKILL.md. Execute only your assigned TASK and RULES, and return "
                    "strictly the 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                    "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                    "file:// .log URI when BLOCKED or FAILED; status may be COMPLETED, CHECKPOINT, "
                    "NEEDS_REPLAN, BLOCKED, or FAILED; scouts may use compact 2-header STATUS + NOTES)."
                )
            }]
        }


def _subagent_meta_for_caller(payload: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return metadata dict for the active subagent caller, or None."""
    if not _is_known_subagent(payload, state):
        return None
    if isinstance(meta := state.get("self_subagent_meta"), dict):
        return meta
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.setdefault("subagents", {}) if isinstance(state.get("subagents"), dict) else {}
    state["subagents"] = subagents
    return subagents[cid] if isinstance(subagents.get(cid), dict) else None


def _get_subagent_type(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Return lowercase subagent type name for the active caller, or empty string."""
    meta = _subagent_meta_for_caller(payload, state)
    t = (meta.get("type_name") or meta.get("type")) if meta is not None else None
    return str(t).strip().lower() if t is not None else ""


def _is_scratch_only_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is restricted to scratch-only writes."""
    return _get_subagent_type(payload, state) in SCRATCH_ONLY_SUBAGENT_TYPES


def _is_scout_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a scout subagent."""
    return _get_subagent_type(payload, state) == "scout"


def _is_strict_readonly_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a strictly read-only subagent."""
    if _get_subagent_type(payload, state) in STRICT_READONLY_SUBAGENT_TYPES:
        return True
    meta = _subagent_meta_for_caller(payload, state)
    return isinstance(meta, dict) and meta.get("council_role") in {"judge", "arbiter"}


def _update_parent_brain_siblings(
    payload: Dict[str, Any], state: Dict[str, Any], match_cids: Tuple[str, ...],
    updater: Callable[[Dict[str, Any], Dict[str, Any]], bool],
) -> None:
    """Apply updater(sib_data, sib_subagents) to matching unlocked sibling brain states."""
    resolved_art, brain_dir = _brain_dir_from_artifact(payload.get("artifactDirectoryPath"))
    if brain_dir is None or resolved_art is None:
        return
    parent_cid = str((state.get("self_subagent_meta") or {}).get("parent_conversation_id") or "").strip().lower()
    cids = [c for c in match_cids if c]
    try:
        for sib_path in brain_dir.glob("*/orchestrator_state.json"):
            if sib_path.parent == resolved_art or _has_other_locked_path(sib_path):
                continue
            with state_transaction(sib_path):
                sib_data = _load_state(sib_path)
                sib_subs = sib_data.get("subagents") or {}
                root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
                if (
                    any(c in sib_subs for c in cids)
                    or (parent_cid and (root_cid == parent_cid or sib_path.parent.name.lower() == parent_cid))
                ) and _workspaces_compatible(payload, sib_data) and updater(sib_data, sib_subs):
                    _save_state(sib_path, sib_data)
    except (OSError, ValueError):
        pass


def _mark_subagent_status_in_parent_brain(
    payload: Dict[str, Any], state: Dict[str, Any], child_cid: str, status: str, *, update_local: bool = True
) -> None:
    """Update a subagent's status in its own state (when update_local) and parent brain state if present."""
    if not (norm_cid := (child_cid or "").strip().lower()):
        return
    if update_local:
        local_changed = False
        subagents = state.get("subagents")
        if isinstance(subagents, dict) and isinstance(subagents.get(norm_cid), dict) and subagents[norm_cid].get("status") != status:
            subagents[norm_cid]["status"], local_changed = status, True
        if isinstance(self_meta := state.get("self_subagent_meta"), dict) and self_meta.get("status") != status:
            self_meta["status"], local_changed = status, True
        if local_changed:
            _save_state(payload, state)

    def _update_sib(sib_data: Dict[str, Any], sib_subs: Dict[str, Any]) -> bool:
        if isinstance(entry := sib_subs.get(norm_cid), dict) and entry.get("status") != status:
            entry["status"] = status
            return True
        return False

    _update_parent_brain_siblings(payload, state, (norm_cid,), _update_sib)


def _apply_cohort_outcome(
    target_state: Dict[str, Any], child_cid: str, cohort_id: str, outcome: str,
    error_summary: str = "", *, create_if_missing: bool = False, cb: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a cohort's succeeded/failed tracking and parent circuit breaker in target_state."""
    cohorts = target_state.setdefault("cohorts", {})
    if not isinstance(cohort_entry := cohorts.get(cohort_id), dict):
        if not create_if_missing:
            return False
        cohort_entry = cohorts[cohort_id] = {"total_spawned": 0, "succeeded": [], "failed": [], "created_at": time.time()}
    succeeded, failed = cohort_entry.setdefault("succeeded", []), cohort_entry.setdefault("failed", [])
    cb_target = cb if cb is not None else target_state.setdefault(
        "circuit_breaker", {"failure_streak": 0, "tripped": False, "seen_failures": [], "last_error": "", "replan_count": 0}
    )
    norm_cid = (child_cid or "").strip().lower()
    if outcome == "succeeded":
        key = norm_cid or f"ok:{len(succeeded)}"
        if key not in succeeded:
            succeeded.append(key)
        if key in failed:
            failed.remove(key)
        cb_target["failure_streak"], cb_target["tripped"] = 0, False
    else:
        key = norm_cid or f"fail:{len(failed)}"
        newly_failed = key not in succeeded and key not in failed
        if newly_failed:
            failed.append(key)
        total_spawned = int(cohort_entry.get("total_spawned", 0) or 0)
        if total_spawned >= 2 and newly_failed and len(failed) >= total_spawned and len(succeeded) == 0:
            cb_target["failure_streak"] = max(MAX_CONSECUTIVE_FAILURES, int(cb_target.get("failure_streak", 0) or 0) + 1)
            cb_target["tripped"] = True
            cb_target["last_error"] = error_summary or "All Best-of-N cohort candidates failed"
        elif total_spawned < 2 and newly_failed:
            cb_target["failure_streak"] = int(cb_target.get("failure_streak", 0) or 0) + 1
            cb_target["last_error"] = error_summary or "Best-of-N candidate failed"
            if cb_target["failure_streak"] >= MAX_CONSECUTIVE_FAILURES:
                cb_target["tripped"] = True
    return True


def _record_cohort_outcome_in_parent_brain(
    payload: Dict[str, Any], state: Dict[str, Any], child_cid: str, cohort_id: str,
    outcome: str, error_summary: str = "", *, update_local: bool = True,
) -> None:
    """Record an async Best-of-N subagent completion outcome in parent brain cohorts."""
    norm_cid, conv_cid = (child_cid or "").strip().lower(), str(payload.get("conversationId") or "").strip().lower()
    if not norm_cid or not cohort_id:
        return
    if update_local and _apply_cohort_outcome(state, norm_cid, cohort_id, outcome, error_summary):
        _save_state(payload, state)

    def _update_sib(sib_data: Dict[str, Any], sib_subs: Dict[str, Any]) -> bool:
        sib_meta = sib_subs.get(conv_cid) if conv_cid else None
        eff = str((sib_meta.get("candidate_id") if isinstance(sib_meta, dict) else None) or norm_cid).strip().lower()
        return _apply_cohort_outcome(sib_data, eff, cohort_id, outcome, error_summary)

    _update_parent_brain_siblings(payload, state, (norm_cid, conv_cid), _update_sib)


def _is_root_orchestrator(payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when the caller is the root orchestrator."""
    return not _is_known_subagent(payload, state if state is not None else _load_state(payload))


def register_subagent(
    payload: Dict[str, Any], conversation_id: str, type_name: str, model: str,
    workspace: str = "inherit", role: str = "", allowed_files: Optional[List[str]] = None,
    status: Optional[str] = None, spawned_mutation_epoch: Optional[int] = None,
    review_lens: Optional[str] = None, active_worktree: Optional[bool] = None,
    mode: Optional[str] = None, cohort: Optional[str] = None, eval_cmd: Optional[str] = None,
    candidate_id: Optional[str] = None, council_id: Optional[str] = None, council_role: Optional[str] = None,
    model_tier: Optional[str] = None,
) -> None:
    """Helper for explicitly registering a subagent conversationId -> tier mapping."""
    with state_transaction(payload):
        state = _load_state(payload)
        cid, norm_ws = conversation_id.strip().lower(), workspace.strip().lower()
        record: Dict[str, Any] = {
            "role": role or type_name, "type": type_name, "model": model, "workspace": workspace,
            "active_branch": norm_ws == "branch",
            "active_worktree": active_worktree if active_worktree is not None else (norm_ws in ISOLATED_WORKSPACES),
        }
        for key, val in (
            ("allowed_files", list(allowed_files) if allowed_files is not None else None),
            ("status", status), ("spawned_mutation_epoch", spawned_mutation_epoch),
            ("review_lens", review_lens), ("mode", mode), ("cohort", cohort),
            ("eval_cmd", eval_cmd), ("candidate_id", candidate_id),
            ("council_id", council_id), ("council_role", council_role),
            ("model_tier", model_tier),
        ):
            if val is not None:
                record[key] = val
        state.setdefault("subagents", {})[cid] = _normalize_subagent_meta(record)
        _save_state(payload, state)

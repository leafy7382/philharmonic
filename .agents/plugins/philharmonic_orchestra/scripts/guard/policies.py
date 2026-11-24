"""Tool-call policies, direct execution fast-path budget, and pre-tool validation."""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from guard.constants import (
    DIR_LIST_TOOLS,
    MAX_REPLAN_ATTEMPTS,
    MUTATION_TOOLS,
    PROJECT_ROOT,
    REPLAN_HEADER_RE,
    SEARCH_TOOLS,
    get_model_for_tier,
)
from guard.identity import (
    _allowed_roots,
    _bind_root_session_if_unset,
    _command_stays_in_workspace,
    _ensure_session_active,
    _get_subagent_type,
    _is_known_subagent,
    _is_root_orchestrator,
    _is_scratch_only_subagent,
    _is_strict_readonly_subagent,
    _is_within_allowed_roots,
    _preclaim_subagent_meta,
)
from guard.state_store import _get_circuit_breaker, _get_test_receipt, _load_state, _save_state, _state_file_path, state_transaction

_GUARD_MOD_RE = re.compile(r"\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|model_tiers|cli)\b")
_GUARD_IMPORT_RE = re.compile(
    r"(?:\borchestrator_guard\b"
    r"|\b(?:from\s+[A-Za-z0-9_.]*\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|model_tiers|cli)\b"
    r"|import\s+(?:\\\r?\n|[^;\n])*\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|model_tiers|cli)\b)"
    r"|(?:^|\s)-[A-Za-z]*m\s*(?:guard|orchestrator_guard|state_store|model_tiers|cli)\b)"
)
_AGENTS_STATE_REF_RE = re.compile(r"(?:^|[\s'\"/(=])\.agents(?:/(?:\.state|\*|\.\S*)|(?=$|[\s'\"/;)|&]))")
_UNQUOTED_REDIR_OUT_RE = re.compile(r"(?:^|[^<])(?:[0-9]*>(?!&\d|\s*/dev/null(?=$|[\s;|&)]))|&>|>>|>\|)")
_PLUGIN_MARKERS = ("hooks.json", "philharmonic_orchestra", ".agents/plugins", ".agents/hooks.json")


def _is_under_root(path_str: str, root: Optional[Path]) -> bool:
    if root is None:
        return False
    try:
        rp = Path(os.path.expanduser(path_str)).resolve()
        return rp == root or root in rp.parents
    except (OSError, ValueError):
        return False


def _exact_impl_plugin_file_allowed(guard_target: str, payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if an implementer subagent has an exact non-glob ALLOWED_FILES entry for guard_target."""
    from guard.path_guards import _get_subagent_allowed_files

    if _is_root_orchestrator(payload, state):
        return False
    sub_type = (_get_subagent_type(payload, state) or "").strip().lower()
    impl_allowed = _get_subagent_allowed_files(payload, state)
    if sub_type != "implementer" or not impl_allowed:
        return False
    try:
        raw_gt = Path(os.path.expanduser(guard_target))
        resolved_gt = (raw_gt if raw_gt.is_absolute() else (PROJECT_ROOT / raw_gt)).resolve()
        if resolved_gt.is_dir():
            return False
        search_roots = [PROJECT_ROOT]
        for wp in (payload.get("workspacePaths") if isinstance(payload.get("workspacePaths"), (list, tuple)) else ()):
            if isinstance(wp, str) and wp:
                try:
                    search_roots.append(Path(wp).resolve())
                except (OSError, ValueError):
                    pass
        for raw_entry in impl_allowed:
            entry = str(raw_entry).strip().strip("`'\"")
            if not entry or entry.upper() == "NONE" or any(ch in entry for ch in "*?["):
                continue
            try:
                raw_ep = Path(os.path.expanduser(entry))
                entry_paths = [raw_ep.resolve()] if raw_ep.is_absolute() else [(root / raw_ep).resolve() for root in search_roots]
            except (OSError, ValueError):
                continue
            if any(ep == resolved_gt and not ep.is_dir() for ep in entry_paths):
                return True
    except (OSError, ValueError):
        return False
    return False


def _is_root_safe_git_stage_commit_chain(
    cmd: str, cleaned: str, sub_cmds: List[str], cwd: str, payload: Dict[str, Any], *, parsed: Optional[Any] = None
) -> bool:
    """Return True if cmd is a clean git add/commit/inspection chain on Root with no state or shell escapes."""
    from guard.path_guards import _is_guard_plugin_cwd, _is_staging_or_tmp_path
    from guard.root_allowlist import _CANONICAL_GIT_BINS, _is_safe_git_stage_or_commit
    from guard.shell_targets import _is_safe_readonly_git_tokens
    from guard.shell_tokens import parse_shell_command

    fb_state_dir = (PROJECT_ROOT / ".agents" / ".state").resolve()
    if _is_under_root(cwd, fb_state_dir) or _is_guard_plugin_cwd(cwd) or _is_staging_or_tmp_path(cwd, payload):
        return False
    p = parsed if parsed is not None else parse_shell_command(cmd)
    eff_cleaned = p.cleaned if parsed is not None else (cleaned or p.cleaned)
    unquoted_cmd = p.unquoted if eff_cleaned == p.cleaned else parse_shell_command(eff_cleaned).unquoted
    if (
        p.has_ansi_c or p.has_subshell or p.redir_targets or _UNQUOTED_REDIR_OUT_RE.search(unquoted_cmd)
        or "{" in unquoted_cmd or "}" in unquoted_cmd or "<<" in unquoted_cmd or "<(" in unquoted_cmd or ">(" in unquoted_cmd or not sub_cmds
    ):
        return False
    iter_subs = p.subcmds if [s.raw for s in p.subcmds] == list(sub_cmds) else tuple(s for sc in sub_cmds for s in parse_shell_command(sc).subcmds)
    for sub in iter_subs:
        if sub.has_in_redir or sub.unwrap_unv or sub.opaque or sub.verb != "git" or sub.raw_verb not in _CANONICAL_GIT_BINS:
            return False
        arg_tokens = list(sub.args)
        if not (_is_safe_git_stage_or_commit(arg_tokens, cwd, payload) or _is_safe_readonly_git_tokens(arg_tokens, cwd, payload)):
            return False
    return True


def _shell_targets_guard_or_state(
    cmd: str, sub_cmds: List[str], cwd: str, payload: Dict[str, Any], state: Dict[str, Any], wts: List[str], has_unverifiable: bool,
    *, parsed: Optional[Any] = None,
) -> bool:
    """Pre-breaker guard-state & plugin self-protection check for run_command."""
    from guard.path_guards import (
        _has_shell_path_expansion, _is_guard_plugin_target, _is_guard_state_target,
        _is_in_allowed_scratch_dir, _resolve_cmd_target,
    )
    from guard.root_allowlist import _is_safe_scratch_shell_subcmd
    from guard.shell_targets import _is_mutating_or_wrapper_shell_subcmd
    from guard.shell_tokens import _cmd_enters_guard_plugin_cwd, parse_shell_command

    p = parsed if parsed is not None else parse_shell_command(cmd)
    if p.has_nul_surr or p.has_ansi_c:
        return True
    same_subs = [s.raw for s in p.subcmds] == list(sub_cmds)
    iter_subs = p.subcmds if same_subs else tuple(s for sc in sub_cmds for s in parse_shell_command(sc).subcmds)
    has_dynamic_verb = any((s.raw_verb and ("$" in s.raw_verb or "`" in s.raw_verb)) or (not s.verb and ("$" in s.raw or "`" in s.raw)) for s in iter_subs)
    is_root = _is_root_orchestrator(payload, state)
    if is_root and (p.has_heredoc_subst or p.has_unclosed or p.has_cmd_subst or has_dynamic_verb):
        return True
    if is_root and _is_root_safe_git_stage_commit_chain(cmd, p.cleaned, sub_cmds, cwd, payload, parsed=p):
        return False
    if any(_is_guard_plugin_target(wt, cwd) or _is_guard_state_target(wt, payload=payload, cwd=cwd) for wt in wts):
        return True

    cmd_norm, cmd_lower = cmd.replace("\\", "/"), cmd.lower()
    mentions_state = bool(re.search(r"orchestrator_state\S*", cmd) or ("orchestrator" in cmd_lower and ("state.json" in cmd_lower or "state.lock" in cmd_lower)))
    if any(_has_shell_path_expansion(wt) for wt in wts):
        if any(k in cmd_lower for k in ("/tmp", "state.json", "state.lock", "orchestrator", ".agents", "/brain/")):
            return True
        if is_root and any(("$" in wt or "`" in wt or "{" in wt) and not _is_in_allowed_scratch_dir(_resolve_cmd_target(wt, cwd), payload, state) for wt in wts):
            return True

    if has_unverifiable and (
        bool(_GUARD_IMPORT_RE.search(cmd))
        or (("import_module" in cmd or "__import__" in cmd) and bool(_GUARD_MOD_RE.search(cmd)))
        or _cmd_enters_guard_plugin_cwd(sub_cmds, cwd, parsed=(p if same_subs else None))
        or any(m in cmd_norm or m in cmd for m in _PLUGIN_MARKERS)
    ):
        return True

    fb_state_dir, agents_dir = (PROJECT_ROOT / ".agents" / ".state").resolve(), (PROJECT_ROOT / ".agents").resolve()
    res_art = brain_root = None
    if art_raw := payload.get("artifactDirectoryPath"):
        try:
            res_art = Path(os.path.expanduser(str(art_raw))).resolve()
            brain_root = res_art.parent if res_art.parent.name == "brain" else None
        except (OSError, ValueError):
            res_art = None

    refs_state_dir = (
        ".agents/.state" in cmd_norm or str(fb_state_dir).replace("\\", "/") in cmd_norm
        or bool(_AGENTS_STATE_REF_RE.search(cmd)) or _is_under_root(cwd, fb_state_dir)
    )
    if not refs_state_dir and res_art is not None:
        art_name = res_art.name
        has_art_ref = len(art_name) >= 6 and art_name in cmd and (mentions_state or "brain" in cmd_lower or "orchestrator" in cmd_lower)
        cands = [str(res_art), str(art_raw)] + ([str(brain_root)] if brain_root else [])
        if "/brain/" in cmd_norm or has_art_ref or any(c and (c in cmd or c.replace("\\", "/") in cmd_norm) for c in cands) or _is_under_root(cwd, res_art) or _is_under_root(cwd, brain_root):
            refs_state_dir = True

    mut_subs: Optional[List[Any]] = None
    if (mentions_state and not has_unverifiable and ">" not in cmd) or (refs_state_dir and not has_unverifiable):
        mut_subs = [s for s in iter_subs if _is_mutating_or_wrapper_shell_subcmd(s.raw, cwd=cwd, payload=payload, parsed_sub=s)]
    if mentions_state and (has_unverifiable or ">" in cmd or bool(mut_subs)):
        return True
    if refs_state_dir:
        if has_unverifiable or (mut_subs and not all(_is_safe_scratch_shell_subcmd(s.raw, cwd, payload, state, parsed_sub=s) for s in mut_subs)):
            return True
        for wt in wts:
            if _has_shell_path_expansion(wt):
                return True
            rwt = _resolve_cmd_target(wt, cwd)
            if _is_under_root(rwt, fb_state_dir) or rwt == str(agents_dir) or ((_is_under_root(rwt, res_art) or _is_under_root(rwt, brain_root)) and not _is_in_allowed_scratch_dir(rwt, payload, state)):
                return True
        if not wts and _UNQUOTED_REDIR_OUT_RE.search(p.unquoted):
            return True
    return False


def _cb_denial(cb: Dict[str, Any], with_replan_hint: bool = False) -> Dict[str, Any]:
    base = (
        f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after {cb.get('failure_streak', 2)} consecutive failures "
        f"(last error: {cb.get('last_error', 'command/subagent failed')}). Stop retrying, write failure details to the scratch log if needed, and report STATUS: BLOCKED"
    )
    return {"decision": "deny", "reason": base + (", or include 'REPLAN: <revised plan rationale>' to use the 1-shot re-plan budget." if with_replan_hint else ".")}


def _readonly_denial(payload: Dict[str, Any], state: Dict[str, Any], *, is_scratch: bool, shell: bool) -> Dict[str, Any]:
    sub_type = _get_subagent_type(payload, state) or ("research" if is_scratch else "readonly")
    if is_scratch:
        role_word = "Council" if sub_type == "council" else "Research"
        return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] {role_word} subagents may only write markdown reports to <artifactDirectoryPath>/scratch/*.md. Workspace mutations are prohibited."}
    kind = "Shell file mutations" if shell else "File mutations"
    reason = f"[ORCHESTRATOR GUARD] Scout subagents (Model='{get_model_for_tier('lite')}') are strictly read-only. {kind} are prohibited." if sub_type == "scout" else f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly read-only. {kind} are prohibited."
    return {"decision": "deny", "reason": reason}


def _record_dirty_receipt(
    state: Dict[str, Any], payload: Dict[str, Any], deferred_actions: List[Callable[[], None]],
    last_file: Optional[str], get_receipt: Callable[..., Dict[str, Any]], get_meta: Callable[..., Dict[str, Any]], prop_fn: Callable[..., None],
) -> None:
    is_root = _is_root_orchestrator(payload, state)
    sub_mode = get_meta(payload, state).get("mode") if not is_root else None
    receipt = get_receipt(state, payload)
    receipt["dirty"] = True
    receipt["verified"] = False
    receipt["mutation_epoch"] = receipt.get("mutation_epoch", 0) + 1
    if last_file:
        receipt["last_mutated_file"] = last_file
    _save_state(payload, state)
    if not is_root and sub_mode != "best_of_n":
        lf = receipt.get("last_mutated_file")
        deferred_actions.append(lambda lf=lf: prop_fn(payload, dirty=True, verified=False, last_mutated_file=lf))


def handle_pre_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    _preclaim_subagent_meta(payload)
    deferred_actions: List[Callable[[], None]] = []
    result: Dict[str, Any] = {"decision": "allow"}

    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {"decision": "allow"}
        if not _is_known_subagent(payload, state) and _bind_root_session_if_unset(payload, state):
            _save_state(payload, state)

        raw_tc = payload.get("toolCall")
        tool_call = raw_tc if isinstance(raw_tc, dict) else {}
        tool_name = str(tool_call.get("name") or "")
        args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}

        if tool_name in {"exit_plan_mode", "enter_plan_mode"} and _is_root_orchestrator(payload, state):
            if tool_name == "exit_plan_mode":
                from guard.path_guards import _parse_allowed_files_header
                from guard.subagent_policy import _parse_eval_cmd_header

                plan_text = str(args.get("plan") or args.get("Plan") or "")
                plan_eval = _parse_eval_cmd_header(plan_text)
                if plan_eval is not None:
                    from guard.shell_tokens import _is_trivial_eval_command

                    if _is_trivial_eval_command(plan_eval):
                        return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] Trivial or no-op EVAL_CMD rejected in plan: '{plan_eval}'. Provide a real test or evaluation script command."}
                state.update({"active": True, "session_evaluated": True})
                _bind_root_session_if_unset(payload, state)
                state.update({"plan_execute_phase": True, "plan_execute_consumed": False, "plan_execute_user_input": str(payload.get("lastUserInput") or "").strip() or None, "plan_execute_turn_finished": False, "plan_allowed_files": _parse_allowed_files_header(plan_text), "active_eval_cmd": plan_eval})
            else:
                state.update({"plan_execute_phase": False, "plan_execute_consumed": False, "plan_execute_user_input": None, "plan_execute_turn_finished": False, "plan_allowed_files": None, "active_eval_cmd": None})
            _save_state(payload, state)
            return {"decision": "allow"}

        _guard_cmd, _guard_sub_cmds, _guard_cwd, _guard_wts, _guard_has_unv, _guard_parsed = "", [], str(PROJECT_ROOT), [], False, None
        if tool_name in MUTATION_TOOLS:
            from guard.path_guards import _extract_target_path, _is_artifact_path, _is_guard_plugin_target, _is_guard_state_target
            gt = _extract_target_path(tool_name, args)
            if _is_guard_state_target(gt, payload=payload):
                return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] Direct modification of guard state file '{Path(gt).name}' is prohibited."}
            if gt and _is_guard_plugin_target(gt) and not _exact_impl_plugin_file_allowed(gt, payload, state):
                return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] Direct modification of guard plugin file '{Path(gt).name}' is prohibited."}
        elif tool_name == "run_command":
            from guard.shell_targets import _extract_shell_write_targets
            from guard.shell_tokens import parse_shell_command
            raw_cmd = args.get("CommandLine")
            _guard_cmd = raw_cmd.strip() if isinstance(raw_cmd, str) else ""
            raw_cwd = args.get("Cwd")
            _guard_cwd = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else str(PROJECT_ROOT)
            _guard_parsed = parse_shell_command(_guard_cmd)
            _guard_sub_cmds = [s.raw for s in _guard_parsed.subcmds]
            _guard_wts, _guard_has_unv = _extract_shell_write_targets(_guard_cmd, _guard_sub_cmds, cwd=_guard_cwd, payload=payload, parsed=_guard_parsed)
            if _shell_targets_guard_or_state(_guard_cmd, _guard_sub_cmds, _guard_cwd, payload, state, _guard_wts, _guard_has_unv, parsed=_guard_parsed):
                return {"decision": "deny", "reason": "[ORCHESTRATOR GUARD] Direct modification of guard state files ('orchestrator_state.json' / 'orchestrator_state.lock') via shell is prohibited."}

        cb = _get_circuit_breaker(state, payload)
        pending_replan_commit = False
        if cb.get("tripped"):
            if tool_name in MUTATION_TOOLS:
                if not _is_artifact_path(_extract_target_path(tool_name, args), payload):
                    return _cb_denial(cb)
            elif tool_name == "run_command":
                from guard.shell_tokens import _is_safe_git_inspection_command
                if not _is_safe_git_inspection_command(_guard_cmd, parsed=_guard_parsed):
                    return _cb_denial(cb)
            elif tool_name in {"invoke_subagent", "send_message", "run_workflow"} or (tool_name == "swarm" and str(args.get("action") or "").strip().lower() == "add"):
                if tool_name == "send_message" and not _is_root_orchestrator(payload, state):
                    if not re.search(r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED|NEEDS_REPLAN)\b", str(args.get("Message") or "")):
                        return _cb_denial(cb)
                else:
                    replan_texts = (
                        [str(it.get("Prompt") or "") for it in (args["Subagents"] if isinstance(args.get("Subagents"), list) else [args]) if isinstance(it, dict)] if tool_name == "invoke_subagent"
                        else ([str(args.get("Message") or "")] if tool_name == "send_message"
                              else ([str(args.get("Description") or ""), str(args.get("Script") or "")] if tool_name == "run_workflow"
                                    else [str(args.get("brief") or "")]))
                    )
                    replan_count = cb.get("replan_count", 0)
                    if any(REPLAN_HEADER_RE.search(t) for t in replan_texts) and replan_count < MAX_REPLAN_ATTEMPTS:
                        pending_replan_commit = True
                    else:
                        return _cb_denial(cb, with_replan_hint=replan_count < MAX_REPLAN_ATTEMPTS)

        if tool_name == "run_command" and _is_root_orchestrator(payload, state) and _guard_cmd:
            from guard.root_allowlist import _is_root_allowlisted_command
            if not _is_root_allowlisted_command(_guard_cmd, _guard_sub_cmds, _guard_cwd, payload, state, parsed=_guard_parsed):
                return {"decision": "deny", "reason": "[ORCHESTRATOR GUARD] Root orchestrator run_command is restricted by the closed-world allowlist (read-only inspection, safe git inspection, verified tests/linters/EVAL_CMD, canonical guard CLI, and scratch/ writes). Delegate mutating or shell-wrapper commands to an implementer subagent."}

        roots = _allowed_roots(payload)
        if tool_name in (SEARCH_TOOLS | DIR_LIST_TOOLS):
            from guard.subagent_policy import _check_and_record_direct_budget
            if budget_denial := _check_and_record_direct_budget(payload, tool_name, args, state):
                return budget_denial
            if tool_name in DIR_LIST_TOOLS:
                from guard.path_guards import _extract_target_path
                target_path = _extract_target_path(tool_name, args) or str(PROJECT_ROOT)
                return {"decision": "allow"} if _is_within_allowed_roots(target_path, roots) else {"decision": "ask"}

        if tool_name == "define_subagent":
            from guard.subagent_policy import _handle_define_subagent
            return _handle_define_subagent(args)

        if tool_name in {"invoke_subagent", "send_message", "manage_subagents", "swarm", "run_workflow"}:
            from guard.subagent_policy import _handle_invoke_subagent, _handle_manage_subagents, _handle_run_workflow, _handle_send_message, _handle_swarm
            if tool_name == "invoke_subagent":
                result = _handle_invoke_subagent(payload, args, state)
            elif tool_name == "send_message":
                result = _handle_send_message(payload, args, state, deferred_actions=deferred_actions)
            elif tool_name == "manage_subagents":
                result = _handle_manage_subagents(payload, args, state)
            elif tool_name == "swarm":
                result = _handle_swarm(payload, args, state)
            else:
                result = _handle_run_workflow(payload, args, state, roots)
            if pending_replan_commit and result.get("decision") == "allow":
                cb["replan_count"] = int(cb.get("replan_count", 0) or 0) + 1
                cb["tripped"], cb["failure_streak"] = False, 0
                _save_state(payload, state)
        elif tool_name == "run_command":
            from guard.path_guards import _get_subagent_allowed_files, _is_artifact_path, _is_tmp_path, _resolve_cmd_target
            from guard.root_allowlist import _is_readonly_subagent_allowlisted_command
            from guard.shell_targets import _validate_shell_allowlist
            from guard.subagent_policy import _check_and_record_direct_budget
            cwd, cmd = _guard_cwd, args.get("CommandLine") if isinstance(args.get("CommandLine"), str) else ""
            cmd_parsed = _guard_parsed if (_guard_parsed is not None and cmd.strip() == _guard_cmd) else parse_shell_command(cmd)
            sub_cmds = _guard_sub_cmds if cmd.strip() == _guard_cmd else [s.raw for s in cmd_parsed.subcmds]
            if _is_strict_readonly_subagent(payload, state) and cmd.strip() and not _is_readonly_subagent_allowlisted_command(cmd, sub_cmds, cwd, payload, state, allow_scratch_writes=False, parsed=cmd_parsed):
                return _readonly_denial(payload, state, is_scratch=False, shell=True)
            if _is_scratch_only_subagent(payload, state) and cmd.strip() and not _is_readonly_subagent_allowlisted_command(cmd, sub_cmds, cwd, payload, state, allow_scratch_writes=True, parsed=cmd_parsed):
                return _readonly_denial(payload, state, is_scratch=True, shell=True)
            if (allowed_files := _get_subagent_allowed_files(payload, state)) is not None:
                if allowlist_denial := _validate_shell_allowlist(cmd, sub_cmds, cwd, allowed_files, payload, allowlist_label="ALLOWED_FILES", parsed=cmd_parsed):
                    return allowlist_denial
            if _is_within_allowed_roots(cwd, roots) and _command_stays_in_workspace(cmd, cwd, roots, parsed=cmd_parsed):
                if budget_denial := _check_and_record_direct_budget(payload, tool_name, args, state, parsed=cmd_parsed):
                    return budget_denial
                write_targets, has_unv_mut = (_guard_wts, _guard_has_unv) if cmd.strip() == _guard_cmd else _extract_shell_write_targets(cmd, sub_cmds, cwd=cwd, payload=payload, parsed=cmd_parsed)
                non_exempt = [_resolve_cmd_target(wt, cwd) for wt in write_targets if not _is_artifact_path(_resolve_cmd_target(wt, cwd), payload) and not _is_tmp_path(_resolve_cmd_target(wt, cwd))]
                is_root_safe_git = (
                    _is_root_orchestrator(payload, state)
                    and bool(cmd.strip())
                    and _is_root_safe_git_stage_commit_chain(cmd, cmd_parsed.cleaned, sub_cmds, cwd, payload, parsed=cmd_parsed)
                )
                if (non_exempt or has_unv_mut) and not is_root_safe_git:
                    from guard.verification import _current_subagent_meta, _propagate_receipt_to_parent_brain
                    _record_dirty_receipt(state, payload, deferred_actions, non_exempt[0] if non_exempt else None, _get_test_receipt, _current_subagent_meta, _propagate_receipt_to_parent_brain)
                result = {"decision": "allow", "permissionOverrides": [f"command({cmd})"] + [f"file({r}/**)" for r in roots]}
            else:
                result = {"decision": "ask"}
        else:
            if tool_name in MUTATION_TOOLS:
                from guard.path_guards import _get_subagent_allowed_files, _is_allowed_research_scratch_write, _is_target_in_allowed_files
                from guard.verification import _check_test_tampering_in_edit
                if _is_strict_readonly_subagent(payload, state):
                    return _readonly_denial(payload, state, is_scratch=False, shell=False)
                target = _extract_target_path(tool_name, args)
                if _is_scratch_only_subagent(payload, state) and not _is_allowed_research_scratch_write(target, payload, state):
                    return _readonly_denial(payload, state, is_scratch=True, shell=False)
                allowed_files = _get_subagent_allowed_files(payload, state)
                if allowed_files is not None and target and not _is_artifact_path(target, payload) and not _is_target_in_allowed_files(target, allowed_files, payload.get("workspacePaths")):
                    return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] File allowlist violation: subagent attempted to mutate '{Path(target).name}', which is outside ALLOWED_FILES ({', '.join(allowed_files) if allowed_files else 'NONE'})."}
                if (tamper_denial := _check_test_tampering_in_edit(tool_name, args, payload)) is not None:
                    return tamper_denial
            target_path = None
            if args:
                from guard.path_guards import _extract_any_path_arg
                target_path = _extract_any_path_arg(tool_name, args)
            if target_path is not None:
                if _is_within_allowed_roots(target_path, roots):
                    from guard.subagent_policy import _check_and_record_direct_budget
                    if budget_denial := _check_and_record_direct_budget(payload, tool_name, args, state):
                        return budget_denial
                    if tool_name in MUTATION_TOOLS and not _is_artifact_path(target_path, payload):
                        from guard.verification import _current_subagent_meta, _is_code_file, _propagate_receipt_to_parent_brain
                        if _is_code_file(target_path):
                            try:
                                lf = str(Path(os.path.expanduser(target_path)).resolve())
                            except (OSError, ValueError):
                                lf = target_path
                            _record_dirty_receipt(state, payload, deferred_actions, lf, _get_test_receipt, _current_subagent_meta, _propagate_receipt_to_parent_brain)
                    result = {"decision": "allow"}
                else:
                    result = {"decision": "ask"}

    for action in deferred_actions:
        action()
    return result


_COMPAT_MODULES = (
    ("guard.shell_tokens", frozenset({"_cmd_enters_guard_plugin_cwd", "_extract_shell_redir_targets", "_has_nul_or_surrogate_input", "_has_unquoted_cmd_subst", "_is_declared_eval_command", "_is_safe_git_inspection_command", "_is_trivial_eval_command", "_preprocess_shell_heredocs_and_comments", "_split_shell_commands", "_split_shell_tokens_raw_with_meta", "_strip_redir_tokens_with_meta", "_strip_shell_quoted_literals", "_unwrap_shell_command_tokens_with_raw"})),
    ("guard.subagent_policy", frozenset({"_check_and_record_direct_budget", "_compute_prompt_hash", "_expand_swarm_worker_names", "_handle_define_subagent", "_handle_invoke_subagent", "_handle_manage_subagents", "_handle_run_workflow", "_handle_send_message", "_handle_swarm", "_parse_cohort_header", "_parse_mode_header", "_parse_eval_cmd_header", "_parse_allowed_files_header"})),
    ("guard.path_guards", frozenset({"_extract_any_path_arg", "_extract_target_path", "_get_subagent_allowed_files", "_has_shell_path_expansion", "_is_allowed_research_scratch_write", "_is_artifact_path", "_is_guard_plugin_cwd", "_is_guard_plugin_target", "_is_guard_state_target", "_is_in_allowed_scratch_dir", "_is_staging_or_tmp_path", "_is_target_in_allowed_files", "_is_tmp_path", "_resolve_cmd_target"})),
    ("guard.root_allowlist", frozenset({"_CANONICAL_GIT_BINS", "_has_unsafe_ansi_c_literal", "_is_readonly_subagent_allowlisted_command", "_is_root_allowlisted_command", "_is_safe_git_stage_or_commit", "_is_safe_scratch_shell_subcmd"})),
    ("guard.shell_targets", frozenset({"_extract_shell_write_targets", "_is_mutating_or_wrapper_shell_subcmd", "_is_safe_readonly_git_tokens", "_validate_shell_allowlist"})),
)


def __getattr__(name: str) -> Any:
    """Lazily resolve legacy compatibility attributes previously imported at module top level."""
    for mod_name, names in _COMPAT_MODULES:
        if name in names:
            import importlib
            val = getattr(importlib.import_module(mod_name), name)
            globals()[name] = val
            return val
    raise AttributeError(f"module 'guard.policies' has no attribute {name!r}")

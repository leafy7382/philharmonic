"""Tool-call policies, direct execution fast-path budget, and pre-tool validation."""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from guard.ast_guard import (
    _PY_FRAME_ATTRS,
    _PY_MUTATING_ATTRS,
    _PY_MUTATING_MODULES,
    _PY_MUTATING_NAMES,
    _PYTEST_OTHER_MUT_FLAGS,
    _PYTEST_PATH_OUTPUT_FLAGS,
    _PYTEST_SKIP_VAL_FLAGS,
    _PYTEST_STAGING_VAL_FLAGS,
    _SAFE_JSON_IMPORTS,
    _SAFE_PURE_STDLIB_MODULES,
    _SAFE_PYTHON_MODULES,
    _SAFE_WORKFLOW_STDLIB_MODULES,
    _SHELL_WRAPPER_VERBS,
    _check_pytest_mutating_args,
    _is_trivial_python_inline,
    _pytest_targets_staging_or_tmp,
    _python_inline_code_is_mutating,
    _unwrap_matching_quotes,
)
from guard.constants import (
    DIR_LIST_TOOLS,
    MAX_REPLAN_ATTEMPTS,
    MUTATING_SHELL_CMD_RE,
    MUTATION_TOOLS,
    PROJECT_ROOT,
    REPLAN_HEADER_RE,
    SEARCH_TOOLS,
    SHELL_EXEC_WRAPPER_RE,
    _PYTHON_MUTATING_OPS_STR,
    _has_nul_or_surrogate_input,
    _preprocess_shell_heredocs_and_comments,
    _split_shell_commands,
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
from guard.path_guards import (
    _POSIX_GLOB_CLASSES,
    _expand_simple_braces,
    _extract_any_path_arg,
    _extract_target_path,
    _find_allowed_files_overlap,
    _get_subagent_allowed_files,
    _guard_state_sentinel,
    _has_shell_path_expansion,
    _is_allowed_research_scratch_write,
    _is_artifact_or_worktree_path,
    _is_artifact_path,
    _is_canonical_guard_script,
    _is_guard_plugin_cwd,
    _is_guard_plugin_target,
    _is_guard_state_target,
    _is_in_allowed_scratch_dir,
    _is_readonly_canonical_guard_invocation,
    _is_staging_or_tmp_path,
    _is_target_in_allowed_files,
    _is_tmp_path,
    _is_worktree_path,
    _normalize_allowlist_entry,
    _parse_allowed_files_header,
    _path_matches_protected_segments,
    _resolve_cmd_target,
    _safe_fnmatchcase,
    _sanitize_decoded_shell_token,
    _segment_matches_literal,
    _target_is_ancestor_of_protected_guard,
)
from guard.root_allowlist import (
    _ROOT_DEV_META_VERBS,
    _ROOT_READONLY_GIT_SUBCMDS,
    _has_unsafe_ansi_c_literal,
    _is_readonly_subagent_allowlisted_command,
    _is_root_allowlisted_command,
    _is_root_safe_git_cmd,
    _is_safe_git_stage_or_commit,
    _is_safe_scratch_shell_subcmd,
)
from guard.shell_targets import (
    _extract_shell_write_targets,
    _validate_shell_allowlist,
)
from guard.shell_tokens import (
    SAFE_READONLY_SHELL_VERBS,
    SAFE_XARGS_READONLY_CMDS,
    _ANSI_C_SIMPLE_ESCAPES,
    _DANGEROUS_ENV_PREFIXES,
    _DANGEROUS_ENV_VARS,
    _LEADING_GROUPING_WORDS,
    _OPAQUE_EXEC_VERBS,
    _REDIR_OP_RE,
    _SAFE_GIT_EDITOR_VALS,
    _SAFE_NPX_RUNNERS,
    _SAFE_PAGER_VALS,
    _STANDALONE_END_WORDS,
    _TOKEN_REDIR_OP_RE,
    _TRAILING_GROUPING_TOKENS,
    _UNVERIFIABLE_WRAPPERS,
    _advance_effective_cwd,
    _cmd_enters_guard_plugin_cwd,
    _consume_short_flag_cluster,
    _decode_ansi_c_escape,
    _extract_shell_redir_targets,
    _has_unquoted_cmd_subst,
    _has_unquoted_or_operator,
    _is_dangerous_env_assignment,
    _is_declared_eval_command,
    _is_safe_git_inspection_command,
    _is_trivial_eval_command,
    _is_trivial_eval_subcmd,
    _split_shell_tokens,
    _split_shell_tokens_raw,
    _split_shell_tokens_raw_with_meta,
    _strip_leading_env_assignments,
    _strip_redir_tokens_with_meta,
    _strip_shell_quoted_literals,
    _tokenize_shell_tail,
    _unquoted_group_delta,
    _unquoted_subshell_opens_closes,
    _unwrap_shell_command_tokens,
    _unwrap_shell_command_tokens_with_raw,
    _unwrap_trivial_eval_prefix,
)
from guard.state_store import (
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)
from guard.subagent_policy import (
    _check_and_record_direct_budget,
    _compute_prompt_hash,
    _detect_blind_judge_leak,
    _detect_council_name_leak,
    _expand_swarm_worker_names,
    _handle_define_subagent,
    _handle_invoke_subagent,
    _handle_manage_subagents,
    _handle_run_workflow,
    _handle_send_message,
    _handle_swarm,
    _is_simple_readonly_task,
    _iter_active_subagents,
    _lookup_recipient_meta,
    _match_blind_judge_patterns,
    _normalize_review_lens,
    _parse_cohort_header,
    _parse_council_id_header,
    _parse_council_role_header,
    _parse_eval_cmd_header,
    _parse_mode_header,
    _parse_review_lens_header,
    _record_council_member,
    _resolve_scratch_doc_candidates,
    _validate_and_prepare_subagent_entry,
    _validate_spec_file_header,
)

_GUARD_MOD_RE = re.compile(
    r"\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|cli)\b"
)
_GUARD_IMPORT_RE = re.compile(
    r"(?:\borchestrator_guard\b"
    r"|\b(?:from\s+[A-Za-z0-9_.]*\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|cli)\b"
    r"|import\s+(?:\\\r?\n|[^;\n])*\b(?:guard|orchestrator_guard|state_store|identity|policies|verification|constants|cli)\b)"
    r"|(?:^|\s)-[A-Za-z]*m\s*(?:guard|orchestrator_guard|state_store|cli)\b)"
)
_AGENTS_STATE_REF_RE = re.compile(
    r"(?:^|[\s'\"/(=])\.agents(?:/(?:\.state|\*|\.\S*)|(?=$|[\s'\"/;)|&]))"
)
_UNQUOTED_REDIR_OUT_RE = re.compile(
    r"(?:^|[^<])(?:[0-9]*>(?!&\d|\s*/dev/null(?=$|[\s;|&)]))|&>|>>|>\|)"
)
_PLUGIN_MARKERS = ("hooks.json", "philharmonic_orchestra", ".agents/plugins", ".agents/hooks.json")


def _is_under_root(path_str: str, root: Optional[Path]) -> bool:
    if root is None:
        return False
    try:
        rp = Path(os.path.expanduser(path_str)).resolve()
        return rp == root or root in rp.parents
    except (OSError, ValueError):
        return False


def _exact_impl_plugin_file_allowed(
    guard_target: str, payload: Dict[str, Any], state: Dict[str, Any]
) -> bool:
    """Return True if an implementer subagent has an exact non-glob ALLOWED_FILES entry for guard_target."""
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
        raw_wps = payload.get("workspacePaths")
        if isinstance(raw_wps, (list, tuple)):
            for wp in raw_wps:
                if isinstance(wp, str) and wp:
                    try:
                        search_roots.append(Path(wp).resolve())
                    except (OSError, ValueError):
                        continue
        for raw_entry in impl_allowed:
            entry = str(raw_entry).strip().strip("`'\"")
            if not entry or entry.upper() == "NONE" or any(ch in entry for ch in "*?["):
                continue
            try:
                raw_ep = Path(os.path.expanduser(entry))
                entry_paths = (
                    [raw_ep.resolve()]
                    if raw_ep.is_absolute()
                    else [(root / raw_ep).resolve() for root in search_roots]
                )
            except (OSError, ValueError):
                continue
            if any(ep == resolved_gt and not ep.is_dir() for ep in entry_paths):
                return True
    except (OSError, ValueError):
        return False
    return False


def _is_root_safe_git_stage_commit_chain(
    cmd: str,
    cleaned: str,
    sub_cmds: List[str],
    cwd: str,
    payload: Dict[str, Any],
) -> bool:
    """Return True if cmd is a clean git add/commit/inspection chain on Root with no state or shell escapes."""
    fb_state_dir = (PROJECT_ROOT / ".agents" / ".state").resolve()
    if _is_under_root(cwd, fb_state_dir) or _is_guard_plugin_cwd(cwd) or _is_staging_or_tmp_path(cwd, payload):
        return False
    unquoted_cmd = _strip_shell_quoted_literals(cleaned)
    if (
        _has_unsafe_ansi_c_literal(cleaned)
        or _extract_shell_redir_targets(cmd)
        or _UNQUOTED_REDIR_OUT_RE.search(unquoted_cmd)
        or "{" in unquoted_cmd
        or "}" in unquoted_cmd
        or "<<" in unquoted_cmd
        or "<(" in unquoted_cmd
        or ">(" in unquoted_cmd
        or not sub_cmds
    ):
        return False
    for sc in sub_cmds:
        raw_meta = _split_shell_tokens_raw_with_meta(sc)
        if any(is_redir and "<" in tok for tok, is_redir, _ in raw_meta):
            return False
        toks, uq = _strip_redir_tokens_with_meta(raw_meta)
        um: Dict[str, bool] = {}
        verb, _, arg_tokens, unwrap_unv = _unwrap_shell_command_tokens_with_raw(toks, uq, out_meta=um)
        if unwrap_unv or um.get("opaque") or verb != "git":
            return False
        if not (
            _is_safe_git_stage_or_commit(arg_tokens, cwd, payload)
            or _is_root_safe_git_cmd(arg_tokens, cwd, payload)
        ):
            return False
    return True


def _shell_targets_guard_or_state(
    cmd: str,
    sub_cmds: List[str],
    cwd: str,
    payload: Dict[str, Any],
    state: Dict[str, Any],
    wts: List[str],
    has_unverifiable: bool,
) -> bool:
    """Pre-breaker guard-state & plugin self-protection check for run_command."""
    if _has_nul_or_surrogate_input(cmd):
        return True
    cleaned, h_subst, h_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    has_dynamic_verb = False
    for sc in sub_cmds:
        raw_meta = _split_shell_tokens_raw_with_meta(sc)
        toks, uq = _strip_redir_tokens_with_meta(raw_meta)
        verb, rc_tok, _, _ = _unwrap_shell_command_tokens_with_raw(toks, uq)
        if (rc_tok and ("$" in rc_tok or "`" in rc_tok)) or (not verb and ("$" in sc or "`" in sc)):
            has_dynamic_verb = True
            break
    is_root = _is_root_orchestrator(payload, state)
    if is_root and (
        h_subst or h_unclosed or _has_unquoted_cmd_subst(cleaned) or has_dynamic_verb
    ):
        return True
    if is_root and _is_root_safe_git_stage_commit_chain(cmd, cleaned, sub_cmds, cwd, payload):
        return False

    for wt in wts:
        res_wt = _resolve_cmd_target(wt, cwd)
        if (
            _is_guard_plugin_target(wt, cwd)
            or _is_guard_plugin_target(res_wt, cwd)
            or _is_guard_state_target(wt, payload=payload)
            or _is_guard_state_target(res_wt, payload=payload)
        ):
            return True

    cmd_norm = cmd.replace("\\", "/")
    cmd_lower = cmd.lower()
    mentions_state = bool(
        re.search(r"orchestrator_state\S*", cmd)
        or ("orchestrator" in cmd_lower and ("state.json" in cmd_lower or "state.lock" in cmd_lower))
    )
    if any(_has_shell_path_expansion(wt) for wt in wts):
        if any(k in cmd_lower for k in ("/tmp", "state.json", "state.lock", "orchestrator", ".agents", "/brain/")):
            return True
        if _is_root_orchestrator(payload, state) and any(
            ("$" in wt or "`" in wt or "{" in wt)
            and not _is_in_allowed_scratch_dir(_resolve_cmd_target(wt, cwd), payload, state)
            for wt in wts
        ):
            return True

    enters_plugin = _cmd_enters_guard_plugin_cwd(sub_cmds, cwd)
    if has_unverifiable and (
        bool(_GUARD_IMPORT_RE.search(cmd))
        or (("import_module" in cmd or "__import__" in cmd) and bool(_GUARD_MOD_RE.search(cmd)))
        or enters_plugin
        or any(m in cmd_norm or m in cmd for m in _PLUGIN_MARKERS)
    ):
        return True

    fb_state_dir = (PROJECT_ROOT / ".agents" / ".state").resolve()
    agents_dir = (PROJECT_ROOT / ".agents").resolve()
    fb_state_str = str(fb_state_dir).replace("\\", "/")
    art_raw = payload.get("artifactDirectoryPath")
    res_art: Optional[Path] = None
    brain_root: Optional[Path] = None
    if art_raw:
        try:
            res_art = Path(os.path.expanduser(str(art_raw))).resolve()
            if res_art.parent.name == "brain":
                brain_root = res_art.parent
        except (OSError, ValueError):
            res_art = None

    refs_state_dir = (
        ".agents/.state" in cmd_norm
        or fb_state_str in cmd_norm
        or bool(_AGENTS_STATE_REF_RE.search(cmd))
        or _is_under_root(cwd, fb_state_dir)
    )
    if not refs_state_dir and res_art is not None:
        art_name = res_art.name
        has_art_ref = len(art_name) >= 6 and art_name in cmd and (
            mentions_state or "brain" in cmd_lower or "orchestrator" in cmd_lower
        )
        cands = [str(res_art), str(art_raw)] + ([str(brain_root)] if brain_root else [])
        if (
            "/brain/" in cmd_norm
            or has_art_ref
            or any(c and (c in cmd or c.replace("\\", "/") in cmd_norm) for c in cands)
            or _is_under_root(cwd, res_art)
            or _is_under_root(cwd, brain_root)
        ):
            refs_state_dir = True

    mut_scs = [sc for sc in sub_cmds if MUTATING_SHELL_CMD_RE.search(sc)]
    if mentions_state and (has_unverifiable or bool(SHELL_EXEC_WRAPPER_RE.search(cmd)) or ">" in cmd or bool(mut_scs)):
        return True
    if refs_state_dir:
        has_unsafe_mut = bool(mut_scs) and not all(
            _is_safe_scratch_shell_subcmd(sc, cwd, payload, state) for sc in mut_scs
        )
        if has_unverifiable or bool(SHELL_EXEC_WRAPPER_RE.search(cmd)) or has_unsafe_mut:
            return True
        for wt in wts:
            if _has_shell_path_expansion(wt):
                return True
            rwt = _resolve_cmd_target(wt, cwd)
            if _is_under_root(rwt, fb_state_dir) or rwt == str(agents_dir):
                return True
            if (_is_under_root(rwt, res_art) or _is_under_root(rwt, brain_root)) and not _is_in_allowed_scratch_dir(
                rwt, payload, state
            ):
                return True
        if not wts and _UNQUOTED_REDIR_OUT_RE.search(_strip_shell_quoted_literals(cleaned)):
            return True
    return False


def _cb_denial(cb: Dict[str, Any], with_replan_hint: bool = False) -> Dict[str, Any]:
    base = (
        f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
        f"{cb.get('failure_streak', 2)} consecutive failures "
        f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
        "Stop retrying, write failure details to the scratch log if needed, "
        "and report STATUS: BLOCKED"
    )
    suffix = ", or include 'REPLAN: <revised plan rationale>' to use the 1-shot re-plan budget." if with_replan_hint else "."
    return {"decision": "deny", "reason": base + suffix}


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

        from guard.verification import (
            _check_test_tampering_in_edit,
            _current_subagent_meta,
            _get_circuit_breaker,
            _get_test_receipt,
            _is_code_file,
            _propagate_receipt_to_parent_brain,
        )

        raw_tool_call = payload.get("toolCall")
        tool_call = raw_tool_call if isinstance(raw_tool_call, dict) else {}
        tool_name = str(tool_call.get("name") or "")
        raw_args = tool_call.get("args")
        args = raw_args if isinstance(raw_args, dict) else {}

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
            _bind_root_session_if_unset(payload, state)
            state["plan_execute_phase"] = True
            state["plan_execute_consumed"] = False
            state["plan_execute_user_input"] = str(payload.get("lastUserInput") or "").strip() or None
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

        _guard_cmd = ""
        _guard_sub_cmds: List[str] = []
        _guard_cwd = str(PROJECT_ROOT)
        _guard_wts: List[str] = []
        _guard_has_unverifiable = False

        if tool_name in MUTATION_TOOLS:
            guard_target = _extract_target_path(tool_name, args)
            if _is_guard_state_target(guard_target, payload=payload):
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Direct modification of guard state file "
                        f"'{Path(guard_target).name}' is prohibited."
                    ),
                }
            if guard_target and _is_guard_plugin_target(guard_target):
                if not _exact_impl_plugin_file_allowed(guard_target, payload, state):
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] Direct modification of guard plugin file "
                            f"'{Path(guard_target).name}' is prohibited."
                        ),
                    }
        elif tool_name == "run_command":
            raw_cmd_val = args.get("CommandLine")
            _guard_cmd = raw_cmd_val.strip() if isinstance(raw_cmd_val, str) else ""
            raw_guard_cwd = args.get("Cwd")
            _guard_cwd = raw_guard_cwd if isinstance(raw_guard_cwd, str) and raw_guard_cwd else str(PROJECT_ROOT)
            _guard_sub_cmds = _split_shell_commands(_guard_cmd)
            _guard_wts, _guard_has_unverifiable = _extract_shell_write_targets(
                _guard_cmd, _guard_sub_cmds, cwd=_guard_cwd, payload=payload
            )
            if _shell_targets_guard_or_state(
                _guard_cmd, _guard_sub_cmds, _guard_cwd, payload, state, _guard_wts, _guard_has_unverifiable
            ):
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Direct modification of guard state files "
                        "('orchestrator_state.json' / 'orchestrator_state.lock') via shell is prohibited."
                    ),
                }

        cb = _get_circuit_breaker(state, payload)
        pending_replan_commit = False
        if cb.get("tripped"):
            if tool_name in MUTATION_TOOLS:
                if not _is_artifact_path(_extract_target_path(tool_name, args), payload):
                    return _cb_denial(cb)
            elif tool_name == "run_command":
                raw_cmd_cb = args.get("CommandLine")
                cmd_cb = raw_cmd_cb.strip() if isinstance(raw_cmd_cb, str) else ""
                if not _is_safe_git_inspection_command(cmd_cb):
                    return _cb_denial(cb)
            elif tool_name in {"invoke_subagent", "send_message", "run_workflow"} or (
                tool_name == "swarm" and str(args.get("action") or "").strip().lower() == "add"
            ):
                if tool_name == "send_message" and not _is_root_orchestrator(payload, state):
                    msg_text = str(args.get("Message") or "")
                    if not re.search(r"(?m)^\s*STATUS:\s*(?:BLOCKED|FAILED|NEEDS_REPLAN)\b", msg_text):
                        return _cb_denial(cb)
                else:
                    replan_found = False
                    if tool_name == "invoke_subagent":
                        entries = args["Subagents"] if isinstance(args.get("Subagents"), list) else [args]
                        replan_found = any(
                            isinstance(it, dict) and REPLAN_HEADER_RE.search(str(it.get("Prompt") or ""))
                            for it in entries
                        )
                    elif tool_name == "send_message":
                        replan_found = bool(REPLAN_HEADER_RE.search(str(args.get("Message") or "")))
                    elif tool_name == "run_workflow":
                        replan_found = bool(
                            REPLAN_HEADER_RE.search(str(args.get("Description") or ""))
                            or REPLAN_HEADER_RE.search(str(args.get("Script") or ""))
                        )
                    elif tool_name == "swarm":
                        replan_found = bool(REPLAN_HEADER_RE.search(str(args.get("brief") or "")))

                    replan_count = cb.get("replan_count", 0)
                    if replan_found and replan_count < MAX_REPLAN_ATTEMPTS:
                        pending_replan_commit = True
                    elif replan_count < MAX_REPLAN_ATTEMPTS:
                        return _cb_denial(cb, with_replan_hint=True)
                    else:
                        return _cb_denial(cb)

        if (
            tool_name == "run_command"
            and _is_root_orchestrator(payload, state)
            and _guard_cmd
            and not _is_root_allowlisted_command(_guard_cmd, _guard_sub_cmds, _guard_cwd, payload, state)
        ):
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Root orchestrator run_command is restricted by the "
                    "closed-world allowlist (read-only inspection, safe git inspection, verified "
                    "tests/linters/EVAL_CMD, canonical guard CLI, and scratch/ writes). "
                    "Delegate mutating or shell-wrapper commands to an implementer subagent."
                ),
            }

        roots = _allowed_roots(payload)

        if tool_name in (SEARCH_TOOLS | DIR_LIST_TOOLS):
            budget_denial = _check_and_record_direct_budget(payload, tool_name, args, state)
            if budget_denial:
                return budget_denial
            if tool_name in DIR_LIST_TOOLS:
                target_path = _extract_target_path(tool_name, args) or str(PROJECT_ROOT)
                return {"decision": "allow"} if _is_within_allowed_roots(target_path, roots) else {"decision": "ask"}

        if tool_name == "define_subagent":
            return _handle_define_subagent(args)

        if tool_name in {"invoke_subagent", "send_message", "manage_subagents", "swarm", "run_workflow"}:
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
                cb["tripped"] = False
                cb["failure_streak"] = 0
                _save_state(payload, state)
        else:
            if tool_name in MUTATION_TOOLS:
                if _is_strict_readonly_subagent(payload, state):
                    sub_type = _get_subagent_type(payload, state) or "readonly"
                    reason = (
                        "[ORCHESTRATOR GUARD] Scout subagents (Model='flash_lite') are strictly "
                        "read-only. File mutations are prohibited."
                        if sub_type == "scout"
                        else f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly read-only. File mutations are prohibited."
                    )
                    return {"decision": "deny", "reason": reason}
                if _is_scratch_only_subagent(payload, state):
                    target = _extract_target_path(tool_name, args)
                    if not _is_allowed_research_scratch_write(target, payload, state):
                        sub_type = _get_subagent_type(payload, state)
                        role_word = "Council" if sub_type == "council" else "Research"
                        return {
                            "decision": "deny",
                            "reason": (
                                f"[ORCHESTRATOR GUARD] {role_word} subagents may only write markdown reports "
                                "to <artifactDirectoryPath>/scratch/*.md. Workspace mutations are prohibited."
                            ),
                        }

                allowed_files = _get_subagent_allowed_files(payload, state)
                target = _extract_target_path(tool_name, args)
                if allowed_files is not None and target and not _is_artifact_path(target, payload):
                    if not _is_target_in_allowed_files(target, allowed_files, payload.get("workspacePaths")):
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

            if tool_name == "run_command":
                cwd = _guard_cwd
                cmd = args.get("CommandLine") if isinstance(args.get("CommandLine"), str) else ""
                sub_cmds = _guard_sub_cmds if cmd.strip() == _guard_cmd else _split_shell_commands(cmd)

                if _is_strict_readonly_subagent(payload, state):
                    if cmd.strip() and not _is_readonly_subagent_allowlisted_command(
                        cmd, sub_cmds, cwd, payload, state, allow_scratch_writes=False
                    ):
                        sub_type = _get_subagent_type(payload, state) or "readonly"
                        reason = (
                            "[ORCHESTRATOR GUARD] Scout subagents (Model='flash_lite') are strictly "
                            "read-only. Shell file mutations are prohibited."
                            if sub_type == "scout"
                            else f"[ORCHESTRATOR GUARD] {sub_type.capitalize()} subagents are strictly read-only. Shell file mutations are prohibited."
                        )
                        return {"decision": "deny", "reason": reason}

                if _is_scratch_only_subagent(payload, state):
                    if cmd.strip() and not _is_readonly_subagent_allowlisted_command(
                        cmd, sub_cmds, cwd, payload, state, allow_scratch_writes=True
                    ):
                        sub_type = _get_subagent_type(payload, state)
                        role_word = "Council" if sub_type == "council" else "Research"
                        return {
                            "decision": "deny",
                            "reason": (
                                f"[ORCHESTRATOR GUARD] {role_word} subagents may only write markdown reports "
                                "to <artifactDirectoryPath>/scratch/*.md. Workspace mutations are prohibited."
                            ),
                        }

                allowed_files = _get_subagent_allowed_files(payload, state)
                if allowed_files is not None:
                    allowlist_denial = _validate_shell_allowlist(
                        cmd, sub_cmds, cwd, allowed_files, payload, allowlist_label="ALLOWED_FILES"
                    )
                    if allowlist_denial:
                        return allowlist_denial

                if _is_within_allowed_roots(cwd, roots) and _command_stays_in_workspace(cmd, cwd, roots):
                    budget_denial = _check_and_record_direct_budget(payload, tool_name, args, state)
                    if budget_denial:
                        return budget_denial

                    cleaned_cmd = _preprocess_shell_heredocs_and_comments(cmd)[0]
                    has_exec_wrapper = bool(SHELL_EXEC_WRAPPER_RE.search(cleaned_cmd)) or any(
                        bool(SHELL_EXEC_WRAPPER_RE.search(sc)) for sc in sub_cmds
                    )
                    write_targets, has_unverifiable_mut = (
                        (_guard_wts, _guard_has_unverifiable)
                        if cmd.strip() == _guard_cmd
                        else _extract_shell_write_targets(cmd, sub_cmds, cwd=cwd, payload=payload)
                    )
                    non_exempt_targets = [
                        _resolve_cmd_target(wt, cwd)
                        for wt in write_targets
                        if not _is_artifact_path(_resolve_cmd_target(wt, cwd), payload)
                        and not _is_tmp_path(_resolve_cmd_target(wt, cwd))
                    ]
                    has_other_mutating_shell = (
                        any(bool(MUTATING_SHELL_CMD_RE.search(sc)) for sc in sub_cmds) and not write_targets
                    )
                    if non_exempt_targets or has_exec_wrapper or has_unverifiable_mut or has_other_mutating_shell:
                        is_root = _is_root_orchestrator(payload, state)
                        sub_mode = _current_subagent_meta(payload, state).get("mode") if not is_root else None
                        receipt = _get_test_receipt(state, payload)
                        receipt["mutation_epoch"] = receipt.get("mutation_epoch", 0) + 1
                        should_dirty = bool(non_exempt_targets or has_exec_wrapper or has_unverifiable_mut) or any(
                            re.search(
                                r"\b(?:patch|sed\s+-[^\s]*i|perl\s+-[^\s]*i|git\s+(?:merge|rebase|cherry-pick|apply|am|checkout|reset|restore|rm|mv))\b",
                                sc,
                                re.I,
                            )
                            or re.search(
                                r"\b(?:python|pypy)(?:\d+(?:\.\d+)*)?\b[^\n]*" + _PYTHON_MUTATING_OPS_STR,
                                sc,
                                re.I,
                            )
                            for sc in sub_cmds
                        )
                        if should_dirty:
                            receipt["dirty"] = True
                            receipt["verified"] = False
                            if non_exempt_targets:
                                receipt["last_mutated_file"] = non_exempt_targets[0]
                        _save_state(payload, state)
                        if not is_root and sub_mode != "best_of_n" and should_dirty:
                            last_file = receipt.get("last_mutated_file")
                            deferred_actions.append(
                                lambda lf=last_file: _propagate_receipt_to_parent_brain(
                                    payload, dirty=True, verified=False, last_mutated_file=lf
                                )
                            )

                    overrides = [f"command({cmd})"] + [f"file({r}/**)" for r in roots]
                    result = {"decision": "allow", "permissionOverrides": overrides}
                else:
                    result = {"decision": "ask"}
            else:
                target_path = _extract_any_path_arg(tool_name, args)
                if target_path is not None:
                    if _is_within_allowed_roots(target_path, roots):
                        budget_denial = _check_and_record_direct_budget(payload, tool_name, args, state)
                        if budget_denial:
                            return budget_denial
                        if tool_name in MUTATION_TOOLS and not _is_artifact_path(target_path, payload) and _is_code_file(target_path):
                            receipt = _get_test_receipt(state, payload)
                            receipt["dirty"] = True
                            receipt["verified"] = False
                            receipt["mutation_epoch"] = receipt.get("mutation_epoch", 0) + 1
                            try:
                                receipt["last_mutated_file"] = str(Path(os.path.expanduser(target_path)).resolve())
                            except (OSError, ValueError):
                                receipt["last_mutated_file"] = target_path
                            _save_state(payload, state)
                            if not _is_root_orchestrator(payload, state) and _current_subagent_meta(payload, state).get("mode") != "best_of_n":
                                last_file = receipt["last_mutated_file"]
                                deferred_actions.append(
                                    lambda lf=last_file: _propagate_receipt_to_parent_brain(
                                        payload, dirty=True, verified=False, last_mutated_file=lf
                                    )
                                )
                        result = {"decision": "allow"}
                    else:
                        result = {"decision": "ask"}

    for action in deferred_actions:
        action()
    return result

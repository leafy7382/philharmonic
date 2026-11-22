"""Closed-world Root & read-only/scratch subagent run_command allowlists."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from guard.ast_guard import (
    _SHELL_WRAPPER_VERBS,
    _check_pytest_mutating_args,
    _check_test_runner_args,
    _pytest_targets_staging_or_tmp,
    _python_inline_code_is_mutating,
)
from guard.constants import (
    PROJECT_ROOT,
    _has_nul_or_surrogate_input,
    _is_test_command,
    _preprocess_shell_heredocs_and_comments,
    _split_shell_commands_with_delims,
)
from guard.identity import _allowed_roots, _is_within_allowed_roots
from guard.path_guards import (
    _has_shell_path_expansion,
    _is_allowed_research_scratch_write,
    _is_guard_plugin_cwd,
    _is_guard_plugin_target,
    _is_guard_state_target,
    _is_in_allowed_scratch_dir,
    _is_readonly_canonical_guard_invocation,
    _is_staging_or_tmp_path,
    _is_tmp_path,
    _resolve_cmd_target,
)
from guard.shell_targets import _extract_shell_write_targets
from guard.shell_tokens import (
    SAFE_READONLY_SHELL_VERBS,
    SAFE_XARGS_READONLY_CMDS,
    _advance_effective_cwd,
    _extract_shell_redir_targets,
    _has_unquoted_cmd_subst,
    _is_dangerous_env_assignment,
    _is_declared_eval_command,
    _is_safe_git_inspection_command,
    _split_shell_tokens_raw_with_meta,
    _strip_redir_tokens_with_meta,
    _strip_shell_quoted_literals,
    _unquoted_group_delta,
    _unquoted_subshell_opens_closes,
    _unwrap_shell_command_tokens,
    _unwrap_shell_command_tokens_with_raw,
)

_ROOT_DEV_META_VERBS = frozenset({
    "python", "python3", "py", "pypy", "node", "npm", "pnpm", "yarn", "go", "cargo", "rustc",
    "pytest", "py.test", "ruff", "mypy", "pylint", "pyright", "flake8", "black", "isort", "tsc", "playwright", "git",
})
_ROOT_READONLY_GIT_SUBCMDS = frozenset({
    "status", "diff", "log", "show", "rev-parse", "describe", "ls-files", "ls-tree", "shortlog",
    "blame", "grep", "rev-list", "cat-file", "for-each-ref", "merge-base", "name-rev",
    "check-ignore", "version", "help",
})
_BAD_GIT_OPTS = frozenset({
    "-c", "--config-env", "--exec-path", "--upload-pack", "--receive-pack", "--output",
    "--ext-diff", "--open-files-in-pager", "--no-index", "--exec",
})
_BAD_GIT_PREFIXES = (
    "--config-env=", "--exec-path=", "--upload-pack=", "--receive-pack=", "--output=",
    "--open-files-in-pager=", "--exec=",
)
_GIT_GLOBAL_VAL_OPTS = frozenset(
    {"-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--list-cmds"}
)
_GIT_GLOBAL_BOOL_OPTS = frozenset({
    "--no-pager", "-p", "--paginate", "-P", "--bare", "--literal-pathspecs", "--glob-pathspecs",
    "--noglob-pathspecs", "--icase-pathspecs", "--no-optional-locks", "--version", "-v",
    "--help", "-h", "--html-path", "--man-path", "--info-path",
})
_SIMPLE_PARAM_RE = re.compile(r"\$\{[#!]?[A-Za-z_][A-Za-z0-9_]*\}|\{\}")
_BAD_VERB_CHARS = set("$`{}*?[]<>()")
_VERSION_HELP_FLAGS = frozenset({"--version", "-V", "--help", "-h"})


def _has_unsafe_ansi_c_literal(text: str) -> bool:
    """Return True if text contains unquoted $'...' with << or \\n/\\r/\\x/\\0 or is unclosed."""
    in_sq = in_dq = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if escaped:
            escaped = False
        elif in_sq:
            if ch == "'":
                in_sq = False
        elif in_dq:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_dq = False
        elif ch == "\\":
            escaped = True
        elif ch == "'":
            in_sq = True
        elif ch == '"':
            in_dq = True
        elif ch == "$" and i + 1 < n and text[i + 1] == "'":
            j, body_chars, closed = i + 2, [], False
            while j < n:
                c2 = text[j]
                if c2 == "\\" and j + 1 < n:
                    body_chars.extend((c2, text[j + 1]))
                    j += 2
                    continue
                if c2 == "'":
                    closed = True
                    j += 1
                    break
                body_chars.append(c2)
                j += 1
            if not closed:
                return True
            body = "".join(body_chars)
            if "<<" in body or "\n" in body or "\r" in body or re.search(r"\\(?:n|r|x|0)", body):
                return True
            i = j
            continue
        i += 1
    return False


def _git_val_is_safe(val: str, cur_dir: str, payload: Dict[str, Any]) -> bool:
    return not (
        _has_shell_path_expansion(val)
        or _is_guard_plugin_target(val, cur_dir)
        or _is_guard_state_target(val, payload=payload)
        or _is_staging_or_tmp_path(_resolve_cmd_target(val, cur_dir), payload)
    )


def _is_bad_git_opt(tok: str) -> bool:
    return (
        tok in _BAD_GIT_OPTS
        or tok.startswith(_BAD_GIT_PREFIXES)
        or (tok.startswith("-c") and not tok.startswith("--") and len(tok) > 2)
    )


_GIT_COMMIT_BOOL_LONG = frozenset(
    {"--no-verify", "--signoff", "--amend", "--no-edit", "--allow-empty", "--quiet", "--verbose", "--all"}
)


def _parse_root_git_subcmd(
    arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any], *, require_allowed_roots: bool = False
) -> Optional[Tuple[str, List[str], str]]:
    """Parse git global options and return (subcmd, sub_args, eff_dir) or None if unsafe."""
    subcmd, sub_args, eff_dir, skip_opt = "", [], cur_dir, ""

    def _safe_val(opt_name: str, val: str) -> bool:
        nonlocal eff_dir
        if not _git_val_is_safe(val, eff_dir, payload):
            return False
        if require_allowed_roots and opt_name in {"-C", "--git-dir", "--work-tree"}:
            res = _resolve_cmd_target(val, eff_dir)
            if not _is_within_allowed_roots(res, _allowed_roots(payload)):
                return False
            if opt_name == "-C":
                eff_dir = res
        return True

    for idx, tok in enumerate(arg_tokens):
        low = tok.lower()
        if "ext::" in low or "fd::" in low:
            return None
        if skip_opt:
            opt_name, skip_opt = skip_opt, ""
            if not _safe_val(opt_name, tok):
                return None
            continue
        if _is_bad_git_opt(tok):
            return None
        if tok in _GIT_GLOBAL_VAL_OPTS:
            skip_opt = tok
        elif tok.startswith(("--git-dir=", "--work-tree=", "--namespace=", "--super-prefix=", "--list-cmds=")):
            if not _safe_val(*tok.split("=", 1)):
                return None
        elif tok.startswith("-C") and len(tok) > 2:
            if not _safe_val("-C", tok[2:]):
                return None
        elif tok in _GIT_GLOBAL_BOOL_OPTS:
            continue
        elif tok.startswith("-"):
            return None
        else:
            subcmd, sub_args = tok.lower(), arg_tokens[idx + 1 :]
            break
    if skip_opt:
        return None
    after_dd = False
    for t in sub_args:
        if "ext::" in t.lower() or "fd::" in t.lower():
            return None
        if not after_dd and t == "--":
            after_dd = True
        elif not after_dd and _is_bad_git_opt(t):
            return None
    return subcmd, sub_args, eff_dir


def _is_safe_git_path(
    tok: str, eff_dir: str, payload: Dict[str, Any], *, allow_scratch_or_tmp: bool = False
) -> bool:
    """Return True if tok is a safe non-expanded, non-state path for git add/commit."""
    if not tok or tok == "-" or tok.startswith(":") or _has_shell_path_expansion(tok):
        return False
    if _is_guard_state_target(tok, payload=payload):
        return False
    resolved = _resolve_cmd_target(tok, eff_dir)
    if _is_guard_state_target(resolved, payload=payload):
        return False
    try:
        fb_state = (PROJECT_ROOT / ".agents" / ".state").resolve()
        res_path = Path(resolved).expanduser().resolve()
    except (OSError, ValueError):
        return False
    if res_path == fb_state or fb_state in res_path.parents or res_path in fb_state.parents:
        return False
    if allow_scratch_or_tmp:
        return (
            _is_within_allowed_roots(resolved, [PROJECT_ROOT.resolve()])
            or _is_in_allowed_scratch_dir(resolved, payload)
            or _is_tmp_path(resolved)
        )
    return not _is_staging_or_tmp_path(resolved, payload) and _is_within_allowed_roots(
        resolved, [PROJECT_ROOT.resolve()]
    )


def _is_safe_git_stage_or_commit(arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any]) -> bool:
    """Closed-world check for non-interactive git add / git commit on Root."""
    parsed = _parse_root_git_subcmd(arg_tokens, cur_dir, payload, require_allowed_roots=True) if arg_tokens else None
    if parsed is None:
        return False
    subcmd, sub_args, eff_dir = parsed
    if subcmd == "add":
        after_dd = saw_update = saw_path = False
        for tok in sub_args:
            if not after_dd and tok == "--":
                after_dd = True
            elif not after_dd and tok.startswith("-"):
                if tok in {"-u", "--update"}:
                    saw_update = True
                elif tok in {"-v", "--verbose", "-n", "--dry-run"} or (
                    not tok.startswith("--") and len(tok) > 1 and all(c in "uvn" for c in tok[1:])
                ):
                    saw_update = saw_update or (not tok.startswith("--") and "u" in tok[1:])
                else:
                    return False
            else:
                if not _is_safe_git_path(tok, eff_dir, payload):
                    return False
                saw_path = True
        return saw_update or saw_path
    if subcmd == "commit":
        after_dd = has_msg = has_file = has_no_edit = False
        expect_val = ""
        for tok in sub_args:
            if expect_val:
                if expect_val == "m":
                    has_msg = True
                elif not _is_safe_git_path(tok, eff_dir, payload, allow_scratch_or_tmp=True):
                    return False
                else:
                    has_file = True
                expect_val = ""
            elif not after_dd and tok == "--":
                after_dd = True
            elif not after_dd and tok.startswith("-"):
                if tok in _GIT_COMMIT_BOOL_LONG:
                    has_no_edit = has_no_edit or tok == "--no-edit"
                elif tok == "--message":
                    expect_val = "m"
                elif tok.startswith("--message="):
                    has_msg = True
                elif tok == "--file":
                    expect_val = "F"
                elif tok.startswith("--file="):
                    if not _is_safe_git_path(tok.split("=", 1)[1], eff_dir, payload, allow_scratch_or_tmp=True):
                        return False
                    has_file = True
                elif not tok.startswith("--") and len(tok) > 1:
                    flags = tok[1:]
                    for k, ch in enumerate(flags):
                        if ch in "saqv":
                            continue
                        if ch == "m":
                            if k < len(flags) - 1:
                                has_msg = True
                            else:
                                expect_val = "m"
                            break
                        if ch == "F":
                            rest = flags[k + 1 :]
                            if rest:
                                if not _is_safe_git_path(rest, eff_dir, payload, allow_scratch_or_tmp=True):
                                    return False
                                has_file = True
                            else:
                                expect_val = "F"
                            break
                        return False
                else:
                    return False
            elif not _is_safe_git_path(tok, eff_dir, payload):
                return False
        return not expect_val and (has_msg or has_file or has_no_edit)
    return False


def _is_root_safe_git_cmd(
    arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any]
) -> bool:
    """Closed-world check for read-only git inspection subcommands on Root."""
    if not arg_tokens:
        return True
    parsed = _parse_root_git_subcmd(arg_tokens, cur_dir, payload)
    if parsed is None:
        return False
    subcmd, sub_args, _ = parsed
    if not subcmd:
        return all(t in {"--version", "-v", "--help", "-h"} for t in arg_tokens)
    if subcmd in {"status", "diff", "log"}:
        quoted_cmd = "git " + " ".join(f"'{a}'" if " " in a else a for a in arg_tokens)
        return _is_safe_git_inspection_command(quoted_cmd) or subcmd in _ROOT_READONLY_GIT_SUBCMDS
    if subcmd in _ROOT_READONLY_GIT_SUBCMDS:
        if subcmd == "grep":
            for t in sub_args:
                if t == "--":
                    break
                if t.startswith("-") and not t.startswith("--") and "O" in t[1:]:
                    return False
        return True
    if subcmd == "branch":
        val_f = {"--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format", "--abbrev", "--color"}
        bool_f = {"-a", "--all", "-r", "--remotes", "-v", "--verbose", "-vv", "-l", "--list", "--show-current", "--no-color", "--column", "--no-column", "--no-abbrev"}
        b_skip, saw_list = False, False
        for t in sub_args:
            if b_skip:
                b_skip = False
                continue
            if t == "--":
                return False
            if t in bool_f:
                saw_list = saw_list or t in {"-l", "--list"}
            elif t in val_f:
                b_skip = True
            elif t.startswith(tuple(f + "=" for f in val_f)) or t.startswith("--color="):
                continue
            elif t.startswith("-") and not t.startswith("--"):
                if any(ch in t[1:] for ch in "dDmMcCftu") or not all(ch in "arvl" for ch in t[1:]):
                    return False
                saw_list = saw_list or ("l" in t[1:])
            elif saw_list and not t.startswith("-"):
                continue
            else:
                return False
        return not b_skip
    if subcmd == "tag":
        val_f = {"-n", "--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format", "--color"}
        bool_f = {"-l", "--list", "-v", "--verify", "--no-color", "-i", "--ignore-case"}
        t_skip, saw_list = False, False
        for t in sub_args:
            if t_skip:
                t_skip = False
                continue
            if t == "--":
                break
            if t in bool_f:
                saw_list = True
            elif t in val_f:
                t_skip = True
            elif t.startswith(tuple(f + "=" for f in val_f)) or t.startswith("-n"):
                continue
            elif t.startswith("-") and not t.startswith("--"):
                if any(ch in t[1:] for ch in "dasufmF") or not all(ch in "lvi" for ch in t[1:]):
                    return False
                saw_list = True
            elif not t.startswith("-") and saw_list:
                continue
            else:
                return False
        return not t_skip
    if subcmd == "remote":
        return not sub_args or all(t in {"-v", "--verbose"} for t in sub_args) or sub_args[0] in {"show", "get-url"}
    if subcmd == "reflog":
        return not sub_args or sub_args[0] in {"show", "exists"} or (
            sub_args[0].startswith("-") and not any(x in sub_args for x in ("expire", "delete", "drop"))
        )
    return subcmd == "fsck" and "--lost-found" not in sub_args


def _is_safe_readonly_verb_invocation(
    v_name: str, v_args: List[str], v_uq: Optional[List[bool]] = None
) -> bool:
    if v_uq is not None and any(uq and tok in {"(", ")", "{", "}"} for tok, uq in zip(v_args, v_uq)):
        return False
    if v_name == "printf":
        for t in v_args:
            if t == "--":
                break
            if t == "-v" or (t.startswith("-") and not t.startswith("--") and "v" in t[1:]):
                return False
        return True
    if v_name == "sort":
        for t in v_args:
            if t == "--":
                break
            opt = t.split("=", 1)[0]
            if opt in {"-o", "--output", "--compress-program"} or (
                t.startswith("-") and not t.startswith("--") and "o" in t[1:]
            ):
                return False
        return True
    if v_name == "rg":
        return not any(t == "--pre" or t.startswith("--pre=") for t in v_args[: (v_args.index("--") if "--" in v_args else len(v_args))])
    return v_name in SAFE_READONLY_SHELL_VERBS


def _is_root_allowlisted_command(
    cmd: str,
    sub_cmds: List[str],
    cwd: str,
    payload: Dict[str, Any],
    state: Dict[str, Any],
    *,
    allow_scratch_writes: bool = True,
    require_doc_ext: bool = False,
    allow_git_commit: bool = True,
) -> bool:
    """Closed-world allowlist for Root orchestrator (and read-only/scratch subagent) run_command."""
    if not cmd or not cmd.strip():
        return True
    if _has_nul_or_surrogate_input(cmd):
        return False
    cleaned_cmd, h_subst, h_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    if h_subst or h_unclosed or _has_unquoted_cmd_subst(cleaned_cmd) or _has_unsafe_ansi_c_literal(cleaned_cmd):
        return False
    unquoted_cmd = _strip_shell_quoted_literals(cleaned_cmd)
    if "<<" in unquoted_cmd or "<(" in unquoted_cmd or ">(" in unquoted_cmd:
        return False
    if re.search(
        r"(?:^|[;&|\n({]\s*)(?:(?:if|then|elif|else|do|while|until|time|!)\s+)*(?:function\s+\w+|(?:coproc|case|select)\b)|\b\w+\s*\(\s*\)\s*\{",
        unquoted_cmd,
    ):
        return False
    if not sub_cmds:
        return False

    delim_parts = [(d, p) for d, p in _split_shell_commands_with_delims(cleaned_cmd) if p]
    use_delim_map = [p for _, p in delim_parts] == list(sub_cmds)
    pipe_group_depth = 0
    cur_dir = cwd or str(PROJECT_ROOT)
    dir_stack: List[str] = []
    cwd_unknown = False
    subshell_stack: List[Tuple[str, List[str], bool]] = []

    for sc_idx, sc in enumerate(sub_cmds):
        opens, closes = _unquoted_subshell_opens_closes(sc)
        for _ in range(opens):
            subshell_stack.append((cur_dir, list(dir_stack), cwd_unknown))

        sc_delim = delim_parts[sc_idx][0] if use_delim_map else None
        is_direct_pipe = sc_delim in ("|", "|&")
        is_piped_in = is_direct_pipe or pipe_group_depth > 0
        if is_piped_in:
            pipe_group_depth = max(0, pipe_group_depth + _unquoted_group_delta(sc))

        sc_cleaned = _preprocess_shell_heredocs_and_comments(sc)[0]
        sc_unquoted = _strip_shell_quoted_literals(sc_cleaned)
        if re.search(r"(?:&>>|&>|>>|>\||>&(?!\d|-)|>(?!&))\s*(?:$|[;|&()])", sc_unquoted):
            return False
        # Finding #3: reject unquoted { or } brace tokens outside simple ${VAR} expansions
        brace_check_str = _SIMPLE_PARAM_RE.sub("", sc_unquoted)
        if "{" in brace_check_str or "}" in brace_check_str:
            return False

        for rt in _extract_shell_redir_targets(sc):
            if rt == "/dev/null":
                continue
            if not allow_scratch_writes or cwd_unknown or _has_shell_path_expansion(rt):
                return False
            resolved_rt = _resolve_cmd_target(rt, cur_dir)
            if _is_guard_state_target(rt, payload=payload) or _is_guard_state_target(resolved_rt, payload=payload):
                return False
            if require_doc_ext:
                if not _is_allowed_research_scratch_write(resolved_rt, payload, state):
                    return False
            elif not _is_in_allowed_scratch_dir(resolved_rt, payload, state):
                return False

        raw_meta = _split_shell_tokens_raw_with_meta(sc)
        has_input_redir = any(is_redir and "<" in tok for tok, is_redir, _ in raw_meta)
        toks, unquoted_flags = _strip_redir_tokens_with_meta(raw_meta)
        arg_uq: List[bool] = []
        unwrap_meta: Dict[str, bool] = {}
        verb, raw_cmd_tok, arg_tokens, unwrap_unverifiable = _unwrap_shell_command_tokens_with_raw(
            toks, unquoted_flags, out_arg_flags=arg_uq, out_meta=unwrap_meta
        )
        if unwrap_unverifiable or unwrap_meta.get("opaque") or not verb:
            return False
        if any(ch in raw_cmd_tok for ch in _BAD_VERB_CHARS) or any(
            uq and ("{" in t or "}" in t) and t != "{}" and not _SIMPLE_PARAM_RE.fullmatch(t)
            for t, uq in zip(arg_tokens, arg_uq)
        ):
            return False

        is_py_verb = bool(re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)", verb))
        allowed_sc = False

        # 1. Canonical read-only guard CLI
        if Path(raw_cmd_tok).name.lower() == "orchestrator_guard.py":
            allowed_sc = _is_readonly_canonical_guard_invocation(
                raw_cmd_tok, arg_tokens, cur_dir,
                is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown,
            )
        elif is_py_verb and arg_tokens:
            py_after_dd = py_skip = False
            py_ok_flags = True
            cand_idx: Optional[int] = None
            for p_idx, p_tok in enumerate(arg_tokens):
                if py_skip:
                    py_skip = False
                    continue
                if not py_after_dd and p_tok == "--":
                    py_after_dd = True
                    continue
                if not py_after_dd and p_tok.startswith("-") and p_tok != "-":
                    if p_tok in {"-W", "-X"}:
                        py_skip = True
                        continue
                    if not p_tok.startswith("--") and all(ch in "BuOqsSEI" for ch in p_tok[1:]):
                        continue
                    py_ok_flags = False
                    break
                cand_idx = p_idx
                break
            if (
                py_ok_flags
                and not py_skip
                and cand_idx is not None
                and Path(arg_tokens[cand_idx]).name.lower() == "orchestrator_guard.py"
            ):
                allowed_sc = _is_readonly_canonical_guard_invocation(
                    arg_tokens[cand_idx], arg_tokens[cand_idx + 1 :], cur_dir,
                    is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown,
                )

        # 2. Safe scratch-directory subcommand (mkdir/tee/touch inside scratch/)
        if not allowed_sc and allow_scratch_writes and not cwd_unknown:
            allowed_sc = _is_safe_scratch_shell_subcmd(
                sc, cur_dir, payload, state, require_doc_ext=require_doc_ext
            )

        # 3. Safe git inspection (and safe non-interactive git add/commit on Root)
        if not allowed_sc and verb == "git" and not has_input_redir:
            allowed_sc = _is_root_safe_git_cmd(arg_tokens, cur_dir, payload) or (
                allow_git_commit
                and not cwd_unknown
                and _is_safe_git_stage_or_commit(arg_tokens, cur_dir, payload)
            )

        # 4. Verified test command or declared EVAL_CMD
        if not allowed_sc and _is_declared_eval_command(sc, state, payload):
            allowed_sc = True
        if not allowed_sc and not cwd_unknown and _is_test_command(sc):
            if not _is_guard_plugin_cwd(cur_dir) and not _is_staging_or_tmp_path(
                _resolve_cmd_target(".", cur_dir), payload
            ):
                allowed_sc = _check_test_runner_args(verb, arg_tokens, cur_dir, payload)

        # 5. Dev-tool metadata (--version/-V/--help/-h) & safe read-only linters/typecheckers
        if not allowed_sc and (verb in _ROOT_DEV_META_VERBS or is_py_verb):
            if (
                (
                    (arg_tokens and all(t in _VERSION_HELP_FLAGS for t in arg_tokens))
                    or (verb == "go" and arg_tokens == ["version"])
                )
                and not is_piped_in
                and not has_input_redir
            ):
                allowed_sc = True
        if not allowed_sc and verb in {"mypy", "pylint", "pyright", "flake8", "tsc", "playwright", "ruff", "black", "isort"}:
            if (
                not cwd_unknown
                and not is_piped_in
                and not has_input_redir
                and not _is_guard_plugin_cwd(cur_dir)
                and not _is_staging_or_tmp_path(_resolve_cmd_target(".", cur_dir), payload)
            ):
                pre_dd = arg_tokens[: arg_tokens.index("--")] if "--" in arg_tokens else arg_tokens
                bad_linter = (
                    any(
                        t.split("=", 1)[0] in {
                            "--init", "--load-plugins", "--init-hook", "--rcfile", "--output", "-o",
                            "--eval-args", "--sqlite-cache", "--install-types", "--cache-dir",
                            "--shadow-file", "--config-file", "--custom-typeshed", "--fix", "--write",
                            "--reporter",
                        }
                        for t in pre_dd
                    )
                    or (verb == "playwright" and any(t in {"code", "codegen", "create", "install"} for t in pre_dd))
                    or (verb == "tsc" and ("--noEmit" not in pre_dd or any(t.lower() == "false" for t in pre_dd)))
                )
                if not bad_linter:
                    sc_wts, sc_unv = _extract_shell_write_targets(sc, [sc], cwd=cur_dir, payload=payload)
                    allowed_sc = not sc_wts and not sc_unv

        # 6. Pure AST-verified non-mutating python -c
        if not allowed_sc and is_py_verb:
            if (
                not cwd_unknown
                and not is_piped_in
                and not has_input_redir
                and not _is_guard_plugin_cwd(cur_dir)
                and not _is_staging_or_tmp_path(_resolve_cmd_target(".", cur_dir), payload)
                and not any(re.search(r"(?:/tmp(?:/|\b)|\S*scratch(?:/|\b)|/brain(?:/|\b))", t, re.I) for t in arg_tokens)
            ):
                after_dd = skip_next = False
                inline_code, py_c_ok = "", True
                for p_idx, p_tok in enumerate(arg_tokens):
                    if skip_next:
                        skip_next = False
                        continue
                    if not after_dd and p_tok == "--":
                        after_dd = True
                        continue
                    if not after_dd and p_tok.startswith("-") and p_tok != "-":
                        if p_tok in {"-W", "-X"}:
                            skip_next = True
                            continue
                        if not p_tok.startswith("--") and len(p_tok) > 1:
                            flags = p_tok[1:]
                            for k, ch in enumerate(flags):
                                if ch == "c":
                                    inline_code = flags[k + 1 :] or (arg_tokens[p_idx + 1] if p_idx + 1 < len(arg_tokens) else "")
                                    if not inline_code:
                                        py_c_ok = False
                                    break
                                if ch in {"W", "X"}:
                                    if k == len(flags) - 1:
                                        skip_next = True
                                    break
                                if ch not in "BuOqsSEI":
                                    py_c_ok = False
                                    break
                            if inline_code or not py_c_ok:
                                break
                            continue
                        py_c_ok = False
                        break
                    py_c_ok = False
                    break
                if py_c_ok and not skip_next and inline_code and not _python_inline_code_is_mutating(inline_code):
                    allowed_sc = True

        # 7. Pure read-only inspection (SAFE_READONLY_SHELL_VERBS + printf/sort/rg)
        if not allowed_sc and (verb in SAFE_READONLY_SHELL_VERBS or verb in {"printf", "sort", "rg"}):
            allowed_sc = _is_safe_readonly_verb_invocation(verb, arg_tokens, arg_uq)

        # 8. Safe read-only find
        if not allowed_sc and verb == "find":
            find_ok, in_exec, exec_toks, f_skip, in_expr = True, False, [], False, False
            for f_tok in arg_tokens:
                if in_exec:
                    if f_tok in {";", "+"}:
                        in_exec = False
                        inner_v, inner_a, inner_unv = _unwrap_shell_command_tokens(exec_toks)
                        if inner_unv or not inner_v or inner_v in _SHELL_WRAPPER_VERBS:
                            find_ok = False
                            break
                        ok_exec = (
                            _is_root_safe_git_cmd(inner_a, cur_dir, payload)
                            if inner_v == "git"
                            else _is_safe_readonly_verb_invocation(inner_v, inner_a)
                        )
                        if not ok_exec:
                            find_ok = False
                            break
                        exec_toks = []
                    else:
                        exec_toks.append(f_tok)
                    continue
                if not in_expr:
                    if f_skip:
                        f_skip = False
                        continue
                    if f_tok in {"-H", "-L", "-P"} or f_tok.startswith("-O"):
                        continue
                    if f_tok == "-D":
                        f_skip = True
                        continue
                    if f_tok.startswith("-") or f_tok in {"(", "!", ","}:
                        in_expr = True
                    else:
                        continue
                if f_tok in {"-delete", "-fls", "-execdir", "-ok", "-okdir"} or f_tok.startswith("-fprint"):
                    find_ok = False
                    break
                if f_tok == "-exec":
                    in_exec, exec_toks = True, []
            allowed_sc = find_ok and not in_exec

        # 9. Safe read-only xargs
        if not allowed_sc and verb == "xargs":
            def _safe_xargs_file(af: str) -> bool:
                return bool(af) and not (
                    _has_shell_path_expansion(af)
                    or _is_guard_state_target(af, payload=payload)
                    or _is_guard_plugin_target(af, cur_dir)
                    or not _is_within_allowed_roots(_resolve_cmd_target(af, cur_dir), _allowed_roots(payload))
                )

            x_skip = x_after_dd = False
            x_bin, x_args, x_ok = "", [], True
            for x_idx, x_tok in enumerate(arg_tokens):
                if x_skip:
                    x_skip = False
                    continue
                if not x_after_dd and x_tok == "--":
                    x_after_dd = True
                    continue
                if not x_after_dd and x_tok.startswith("-") and x_tok != "-":
                    if x_tok in {"-a", "--arg-file"}:
                        if x_idx + 1 >= len(arg_tokens) or not _safe_xargs_file(arg_tokens[x_idx + 1]):
                            x_ok = False
                            break
                        x_skip = True
                        continue
                    if x_tok.startswith("--arg-file="):
                        if not _safe_xargs_file(x_tok.split("=", 1)[1]):
                            x_ok = False
                            break
                        continue
                    if x_tok in {"-I", "-L", "-n", "-P", "-d", "-E", "-s"}:
                        x_skip = True
                        continue
                    if x_tok.startswith(("--max-args=", "--max-procs=", "--delimiter=", "--eof=", "--max-chars=", "--replace=")):
                        continue
                    if not x_tok.startswith("--") and len(x_tok) > 1:
                        for k, ch in enumerate(x_tok[1:]):
                            if ch == "a":
                                rest = x_tok[1:][k + 1 :]
                                af = rest or (arg_tokens[x_idx + 1] if x_idx + 1 < len(arg_tokens) else "")
                                if not _safe_xargs_file(af):
                                    x_ok = False
                                if not rest:
                                    x_skip = True
                                break
                            if ch in "ILnPdEs":
                                if k == len(x_tok[1:]) - 1:
                                    x_skip = True
                                break
                        if not x_ok:
                            break
                    continue
                x_bin, x_args = x_tok, arg_tokens[x_idx + 1 :]
                break
            if x_ok and x_bin:
                x_name = Path(x_bin).name.lower()
                if x_name in SAFE_XARGS_READONLY_CMDS:
                    allowed_sc = (
                        _is_safe_readonly_verb_invocation(x_name, x_args)
                        if x_name in {"rg", "printf"}
                        else True
                    )

        # 10. Safe read-only archive listing (tar -tf, unzip -l) — Finding #4
        if not allowed_sc and verb == "tar":
            tar_has_list, tar_ok, tar_skip = False, True, False
            _TAR_BAD_LONG = {
                "--extract", "--get", "--create", "--append", "--update", "--catenate",
                "--concatenate", "--delete", "--remove-files", "--rsh-command", "--to-command",
                "--info-script", "--new-volume-script", "--checkpoint-action",
                "--use-compress-program", "--files-from", "--index-file", "--rmt-command", "--exclude-from",
            }
            for t_idx, t_tok in enumerate(arg_tokens):
                if t_tok.startswith("--") and t_tok.split("=", 1)[0] in _TAR_BAD_LONG:
                    tar_ok = False
                    break
                if tar_skip:
                    tar_skip = False
                    continue
                if t_tok == "--":
                    break
                if t_tok == "--list":
                    tar_has_list = True
                    continue
                if t_tok.startswith("--"):
                    if t_tok in {"--file", "--directory"}:
                        tar_skip = True
                    continue
                if t_tok.startswith("-") and t_tok != "-":
                    for k, ch in enumerate(t_tok[1:]):
                        if ch in "xcruAFITX":
                            tar_ok = False
                            break
                        if ch == "t":
                            tar_has_list = True
                        if ch in "fC":
                            if k == len(t_tok[1:]) - 1:
                                tar_skip = True
                            break
                    if not tar_ok:
                        break
                    continue
                if t_idx == 0:
                    if any(ch in "xcruAFITX" for ch in t_tok):
                        tar_ok = False
                        break
                    if "t" in t_tok:
                        tar_has_list = True
                    if "f" in t_tok:
                        tar_skip = True
            allowed_sc = tar_ok and tar_has_list and not tar_skip

        if not allowed_sc and verb == "unzip":
            uz_has_list, uz_ok = False, True
            for u_tok in arg_tokens:
                if u_tok == "--":
                    break
                if u_tok.startswith("--"):
                    if u_tok.split("=", 1)[0].lower() not in {"--help", "--version"}:
                        uz_ok = False
                        break
                elif u_tok.startswith("-") and u_tok != "-":
                    if any(ch in "doufnjx" for ch in u_tok[1:]):
                        uz_ok = False
                        break
                    if any(ch in "lZ" for ch in u_tok[1:]):
                        uz_has_list = True
            allowed_sc = uz_ok and uz_has_list

        # 11. Safe cd <workspace_subdir> and safe export VAR=val
        if not allowed_sc and verb in {"cd", "pushd", "popd"}:
            new_dir, is_cd_cmd, cd_unv = _advance_effective_cwd(sc, cur_dir, dir_stack, payload)
            if (
                is_cd_cmd
                and not cd_unv
                and not _is_guard_plugin_cwd(new_dir)
                and not _is_staging_or_tmp_path(new_dir, payload)
                and _is_within_allowed_roots(new_dir, _allowed_roots(payload))
            ):
                cur_dir = new_dir
                allowed_sc = True
        if not allowed_sc and verb == "export":
            if arg_tokens and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t) and not _is_dangerous_env_assignment(t)
                for t in arg_tokens
            ):
                allowed_sc = True

        if not allowed_sc:
            return False

        for _ in range(closes):
            if subshell_stack:
                cur_dir, dir_stack, cwd_unknown = subshell_stack.pop()

    return True


def _is_readonly_subagent_allowlisted_command(
    cmd: str,
    sub_cmds: List[str],
    cwd: str,
    payload: Dict[str, Any],
    state: Dict[str, Any],
    *,
    allow_scratch_writes: bool = False,
) -> bool:
    """Closed-world allowlist for strict read-only (scout/verifier/reviewer/judge) and scratch-only (research/council) subagents."""
    return _is_root_allowlisted_command(
        cmd,
        sub_cmds,
        cwd,
        payload,
        state,
        allow_scratch_writes=allow_scratch_writes,
        require_doc_ext=allow_scratch_writes,
        allow_git_commit=False,
    )


def _is_safe_scratch_shell_subcmd(
    sc: str,
    cwd: str,
    payload: Dict[str, Any],
    state: Dict[str, Any],
    *,
    require_doc_ext: bool = False,
) -> bool:
    """Return True if a mutating subcommand is solely tee/touch/mkdir targeting non-state scratch paths."""
    tokens, unquoted_flags = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(sc))
    verb, arg_tokens, unwrap_unverifiable = _unwrap_shell_command_tokens(tokens, unquoted_flags)
    if unwrap_unverifiable or not verb or verb not in {"tee", "touch", "mkdir"} or not arg_tokens:
        return False
    targets: List[str] = []
    after_dd = skip_next = False
    for tok in arg_tokens:
        if skip_next:
            skip_next = False
            continue
        if not after_dd and tok == "--":
            after_dd = True
            continue
        if not after_dd and tok.startswith("-"):
            if (verb == "mkdir" and tok in {"-m", "--mode"}) or (
                verb == "touch" and tok in {"-d", "--date", "-r", "--reference", "-t"}
            ):
                skip_next = True
            continue
        targets.append(tok)
    if not targets:
        return False
    for tgt in targets:
        if _is_guard_state_target(tgt, payload=payload):
            return False
        resolved_tgt = _resolve_cmd_target(tgt, cwd)
        if _is_guard_state_target(resolved_tgt, payload=payload):
            return False
        if verb in {"tee", "touch"} and require_doc_ext:
            if not _is_allowed_research_scratch_write(resolved_tgt, payload, state):
                return False
        elif not _is_in_allowed_scratch_dir(resolved_tgt, payload, state):
            return False
    return True

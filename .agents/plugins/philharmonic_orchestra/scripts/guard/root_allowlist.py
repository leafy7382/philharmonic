"""Closed-world Root & read-only/scratch subagent run_command allowlists."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from guard.ast_guard import _check_test_runner_args, _is_test_command
from guard.constants import PROJECT_ROOT
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
from guard.shell_targets import _LINTER_VERBS, _is_safe_linter_subcmd, _is_safe_readonly_git_tokens, _parse_git_tokens
from guard.shell_tokens import (
    SAFE_READONLY_SHELL_VERBS,
    _PY_VERB_RE,
    _advance_effective_cwd,
    _extract_shell_redir_targets,
    _has_nul_or_surrogate_input,
    _has_unquoted_cmd_subst,
    _is_declared_eval_command,
    _parse_python_cli,
    _preprocess_shell_heredocs_and_comments,
    _split_shell_commands_with_delims,
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
_SIMPLE_PARAM_RE = re.compile(r"\$\{[#!]?[A-Za-z_][A-Za-z0-9_]*\}|\{\}")
_BAD_VERB_CHARS = set("$`{}*?[]<>()")
_VERSION_HELP_FLAGS = frozenset({"--version", "-V", "--help", "-h"})
_GIT_COMMIT_BOOL_LONG = frozenset({"--no-verify", "--signoff", "--amend", "--no-edit", "--allow-empty", "--quiet", "--verbose", "--all"})
_GIT_PUSH_BOOL_LONG = frozenset({"--set-upstream", "--dry-run", "--verbose", "--quiet", "--tags", "--follow-tags", "--atomic", "--porcelain", "--progress", "--no-progress"})
_GIT_PUSH_REMOTE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_GIT_PUSH_REF_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]*$")
_CANONICAL_GIT_BINS = frozenset({"git", "/usr/bin/git", "/bin/git", "/usr/local/bin/git"})
_READONLY_VERB_BLOCKED_FLAGS: Dict[str, Tuple[frozenset[str], str]] = {
    "printf": (frozenset({"-v"}), "v"),
    "sort": (frozenset({"-o", "--output", "--compress-program"}), "o"),
    "rg": (frozenset({"--pre"}), ""),
}


def _has_unsafe_ansi_c_literal(text: str) -> bool:
    """Return True if text contains unquoted $'...' with << or \\n/\\r/\\x/\\0 or is unclosed."""
    in_sq = in_dq = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if escaped:
            escaped = False
        elif in_sq:
            in_sq = ch != "'"
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
                elif c2 == "'":
                    closed, j = True, j + 1
                    break
                else:
                    body_chars.append(c2)
                    j += 1
            body = "".join(body_chars)
            if not closed or "<<" in body or "\n" in body or "\r" in body or re.search(r"\\(?:n|r|x|0)", body):
                return True
            i = j
            continue
        i += 1
    return False


def _is_safe_git_path(tok: str, eff_dir: str, payload: Dict[str, Any], *, allow_scratch_or_tmp: bool = False) -> bool:
    """Return True if tok is a safe non-expanded, non-state path for git add/commit."""
    if not tok or tok == "-" or tok.startswith(":") or _has_shell_path_expansion(tok):
        return False
    resolved = _resolve_cmd_target(tok, eff_dir)
    if _is_guard_plugin_target(tok, eff_dir) or _is_guard_plugin_target(resolved, eff_dir):
        return False
    if _is_guard_state_target(tok, payload=payload, cwd=eff_dir) or _is_guard_state_target(resolved, payload=payload, cwd=eff_dir):
        return False
    try:
        fb_state = (PROJECT_ROOT / ".agents" / ".state").resolve()
        res_path = Path(resolved).expanduser().resolve()
    except (OSError, ValueError):
        return False
    if res_path == fb_state or fb_state in res_path.parents or res_path in fb_state.parents:
        return False
    if allow_scratch_or_tmp:
        return _is_within_allowed_roots(resolved, [PROJECT_ROOT.resolve()]) or _is_in_allowed_scratch_dir(resolved, payload) or _is_tmp_path(resolved)
    return not _is_staging_or_tmp_path(resolved, payload) and _is_within_allowed_roots(resolved, [PROJECT_ROOT.resolve()])


def _is_safe_git_stage_or_commit(arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any]) -> bool:
    """Closed-world check for non-interactive git add / commit / non-force push on Root."""
    parsed = _parse_git_tokens(arg_tokens, cur_dir, payload, require_allowed_roots=True, allow_git_dir=False) if arg_tokens else None
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
                elif tok in {"-v", "--verbose", "-n", "--dry-run"} or (not tok.startswith("--") and len(tok) > 1 and all(c in "uvn" for c in tok[1:])):
                    saw_update = saw_update or (not tok.startswith("--") and "u" in tok[1:])
                else:
                    return False
            elif not _is_safe_git_path(tok, eff_dir, payload):
                return False
            else:
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
                    for k, ch in enumerate(tok[1:]):
                        if ch in "saqv":
                            continue
                        if ch == "m":
                            has_msg, expect_val = (True, "") if k < len(tok[1:]) - 1 else (has_msg, "m")
                            break
                        if ch == "F":
                            rest = tok[1:][k + 1 :]
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
    if subcmd == "push":
        after_dd = False
        pos: List[str] = []
        for tok in sub_args:
            if not after_dd and tok == "--":
                after_dd = True
            elif not after_dd and tok.startswith("-"):
                if tok in _GIT_PUSH_BOOL_LONG or (not tok.startswith("--") and len(tok) > 1 and all(c in "unvq" for c in tok[1:])):
                    continue
                return False
            else:
                pos.append(tok)
        if not pos:
            return True
        if not _GIT_PUSH_REMOTE_RE.fullmatch(pos[0]) or ".." in pos[0]:
            return False
        try:
            eff_p = Path(eff_dir).resolve()
        except (OSError, ValueError):
            return False
        if (eff_p / pos[0]).exists() or (eff_p / f"{pos[0]}.git").exists():
            return False
        git_cfg = eff_p / ".git" / "config"
        if not git_cfg.is_file() and (eff_p == PROJECT_ROOT or PROJECT_ROOT in eff_p.parents):
            git_cfg = PROJECT_ROOT / ".git" / "config"
        if git_cfg.is_file():
            try:
                cfg_txt = git_cfg.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            if not re.search(rf'(?m)^\s*\[remote\s+"{re.escape(pos[0])}"\]', cfg_txt):
                return False
        for ref in pos[1:]:
            if ".." in ref or "//" in ref or "@{" in ref or _has_shell_path_expansion(ref):
                return False
            parts = ref.split(":")
            if len(parts) > 2 or any(not p or not _GIT_PUSH_REF_RE.fullmatch(p) for p in parts):
                return False
        return True
    return False


def _is_safe_readonly_verb_invocation(v_name: str, v_args: List[str], v_uq: Optional[List[bool]] = None) -> bool:
    """Validate safe read-only inspection verbs via declarative blocked-flag table."""
    if v_uq is not None and any(uq and tok in {"(", ")", "{", "}"} for tok, uq in zip(v_args, v_uq)):
        return False
    if spec := _READONLY_VERB_BLOCKED_FLAGS.get(v_name):
        bad_opts, bad_shorts = spec
        pre_dd = v_args[: v_args.index("--")] if "--" in v_args else v_args
        return not any(
            t.split("=", 1)[0] in bad_opts or (bad_shorts and t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in bad_shorts))
            for t in pre_dd
        )
    return v_name in SAFE_READONLY_SHELL_VERBS


def _is_safe_scratch_shell_subcmd(
    sc: str, cwd: str, payload: Dict[str, Any], state: Dict[str, Any], *, require_doc_ext: bool = False
) -> bool:
    """Return True if a mutating subcommand is solely tee/touch/mkdir targeting non-state scratch paths."""
    tokens, unquoted_flags = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(sc))
    verb, arg_tokens, unwrap_unv = _unwrap_shell_command_tokens(tokens, unquoted_flags)
    if unwrap_unv or verb not in {"tee", "touch", "mkdir"} or not arg_tokens:
        return False
    targets: List[str] = []
    after_dd = skip_next = False
    for tok in arg_tokens:
        if skip_next:
            skip_next = False
        elif not after_dd and tok == "--":
            after_dd = True
        elif not after_dd and tok.startswith("-"):
            if (verb == "mkdir" and tok in {"-m", "--mode"}) or (verb == "touch" and tok in {"-d", "--date", "-r", "--reference", "-t"}):
                skip_next = True
        else:
            targets.append(tok)
    if not targets:
        return False
    for tgt in targets:
        resolved_tgt = _resolve_cmd_target(tgt, cwd)
        if _is_guard_state_target(tgt, payload=payload, cwd=cwd) or _is_guard_state_target(resolved_tgt, payload=payload, cwd=cwd):
            return False
        if verb in {"tee", "touch"} and require_doc_ext:
            if not _is_allowed_research_scratch_write(resolved_tgt, payload, state):
                return False
        elif not _is_in_allowed_scratch_dir(resolved_tgt, payload, state):
            return False
    return True


def _is_root_allowlisted_command(
    cmd: str, sub_cmds: List[str], cwd: str, payload: Dict[str, Any], state: Dict[str, Any],
    *, allow_scratch_writes: bool = True, require_doc_ext: bool = False, allow_git_commit: bool = True,
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
    if "<<" in unquoted_cmd or "<(" in unquoted_cmd or ">(" in unquoted_cmd or not sub_cmds or re.search(
        r"(?:^|[;&|\n({]\s*)(?:(?:if|then|elif|else|do|while|until|time|!)\s+)*(?:function\s+\w+|(?:coproc|case|select)\b)|\b\w+\s*\(\s*\)\s*\{",
        unquoted_cmd,
    ):
        return False

    delim_parts = [(d, p) for d, p in _split_shell_commands_with_delims(cleaned_cmd) if p]
    use_delim_map = [p for _, p in delim_parts] == list(sub_cmds)
    pipe_group_depth, cur_dir, dir_stack, cwd_unknown, subshell_stack = 0, cwd or str(PROJECT_ROOT), [], False, []

    for sc_idx, sc in enumerate(sub_cmds):
        opens, closes = _unquoted_subshell_opens_closes(sc)
        subshell_stack.extend((cur_dir, list(dir_stack), cwd_unknown) for _ in range(opens))
        is_piped_in = ((delim_parts[sc_idx][0] if use_delim_map else None) in ("|", "|&")) or pipe_group_depth > 0
        if is_piped_in:
            pipe_group_depth = max(0, pipe_group_depth + _unquoted_group_delta(sc))

        sc_uq = _strip_shell_quoted_literals(_preprocess_shell_heredocs_and_comments(sc)[0])
        if re.search(r"(?:&>>|&>|>>|>\||>&(?!\d|-)|>(?!&))\s*(?:$|[;|&()])", sc_uq):
            return False
        brace_chk = _SIMPLE_PARAM_RE.sub("", sc_uq)
        if "{" in brace_chk or "}" in brace_chk:
            return False

        for rt in _extract_shell_redir_targets(sc):
            if rt == "/dev/null":
                continue
            if not allow_scratch_writes or cwd_unknown or _has_shell_path_expansion(rt):
                return False
            res_rt = _resolve_cmd_target(rt, cur_dir)
            if _is_guard_state_target(rt, payload=payload, cwd=cur_dir) or _is_guard_state_target(res_rt, payload=payload, cwd=cur_dir):
                return False
            if not (_is_allowed_research_scratch_write(res_rt, payload, state) if require_doc_ext else _is_in_allowed_scratch_dir(res_rt, payload, state)):
                return False

        raw_meta = _split_shell_tokens_raw_with_meta(sc)
        has_input_redir = any(is_r and "<" in tok for tok, is_r, _ in raw_meta)
        toks, uq_flags = _strip_redir_tokens_with_meta(raw_meta)
        arg_uq: List[bool] = []
        um: Dict[str, bool] = {}
        verb, raw_tok, arg_tokens, unwrap_unv = _unwrap_shell_command_tokens_with_raw(toks, uq_flags, out_arg_flags=arg_uq, out_meta=um)
        if unwrap_unv or um.get("opaque") or not verb or any(ch in raw_tok for ch in _BAD_VERB_CHARS) or any(
            uq and ("{" in t or "}" in t) and t != "{}" and not _SIMPLE_PARAM_RE.fullmatch(t) for t, uq in zip(arg_tokens, arg_uq)
        ):
            return False

        is_py_verb = bool(_PY_VERB_RE.match(verb))
        allowed_sc = False
        if Path(raw_tok).name.lower() == "orchestrator_guard.py":
            allowed_sc = _is_readonly_canonical_guard_invocation(raw_tok, arg_tokens, cur_dir, is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown)
        elif is_py_verb and arg_tokens:
            ok_py, mode, target, rest = _parse_python_cli(arg_tokens)
            if ok_py and mode == "script" and Path(target).name.lower() == "orchestrator_guard.py":
                allowed_sc = _is_readonly_canonical_guard_invocation(target, rest, cur_dir, is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown)

        if not allowed_sc and allow_scratch_writes and not cwd_unknown:
            allowed_sc = _is_safe_scratch_shell_subcmd(sc, cur_dir, payload, state, require_doc_ext=require_doc_ext)
        if not allowed_sc and verb == "git" and raw_tok in _CANONICAL_GIT_BINS and not has_input_redir:
            allowed_sc = _is_safe_readonly_git_tokens(arg_tokens, cur_dir, payload) or (
                allow_git_commit and not cwd_unknown and _is_safe_git_stage_or_commit(arg_tokens, cur_dir, payload)
            )
        if not allowed_sc and _is_declared_eval_command(sc, state, payload):
            allowed_sc = True
        if not allowed_sc and not cwd_unknown and _is_test_command(sc, cur_dir=cur_dir, payload=payload):
            if not _is_guard_plugin_cwd(cur_dir) and not _is_staging_or_tmp_path(_resolve_cmd_target(".", cur_dir), payload):
                allowed_sc = _check_test_runner_args(verb, arg_tokens, cur_dir, payload)
        if not allowed_sc and (verb in _ROOT_DEV_META_VERBS or is_py_verb) and raw_tok.lower() in {verb, f"/usr/bin/{verb}", f"/bin/{verb}"} and not is_piped_in and not has_input_redir:
            allowed_sc = bool(arg_tokens and all(t in _VERSION_HELP_FLAGS for t in arg_tokens)) or (verb == "go" and arg_tokens == ["version"])
        if not allowed_sc and verb in _LINTER_VERBS:
            allowed_sc = _is_safe_linter_subcmd(verb, arg_tokens, cur_dir, payload, is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown)
        if not allowed_sc and (verb in SAFE_READONLY_SHELL_VERBS or verb in _READONLY_VERB_BLOCKED_FLAGS):
            allowed_sc = _is_safe_readonly_verb_invocation(verb, arg_tokens, arg_uq)
        if not allowed_sc and verb in {"cd", "pushd", "popd"}:
            new_dir, is_cd_cmd, cd_unv = _advance_effective_cwd(sc, cur_dir, dir_stack, payload)
            if is_cd_cmd and not cd_unv and not _is_guard_plugin_cwd(new_dir) and not _is_staging_or_tmp_path(new_dir, payload) and _is_within_allowed_roots(new_dir, _allowed_roots(payload)):
                cur_dir, allowed_sc = new_dir, True
        if not allowed_sc:
            return False
        for _ in range(min(closes, len(subshell_stack))):
            cur_dir, dir_stack, cwd_unknown = subshell_stack.pop()
    return True


def _is_readonly_subagent_allowlisted_command(
    cmd: str, sub_cmds: List[str], cwd: str, payload: Dict[str, Any], state: Dict[str, Any], *, allow_scratch_writes: bool = False
) -> bool:
    """Closed-world allowlist for strict read-only (scout/verifier/reviewer/judge) and scratch-only (research/council) subagents."""
    return _is_root_allowlisted_command(
        cmd, sub_cmds, cwd, payload, state, allow_scratch_writes=allow_scratch_writes, require_doc_ext=allow_scratch_writes, allow_git_commit=False
    )

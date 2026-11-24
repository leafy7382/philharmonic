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
    FlagSpec, ParsedCommand, ParsedSubcmd, SAFE_READONLY_SHELL_VERBS, _PY_VERB_RE,
    _advance_effective_cwd, _is_canonical_cmd_path, _is_declared_eval_command, _parse_python_cli, _payload_has_raw_cr, parse_shell_command, walk_cli_flags,
)

_ROOT_DEV_META_VERBS = frozenset({
    "python", "python3", "py", "pypy", "node", "npm", "pnpm", "yarn", "go", "cargo", "rustc", "uv", "poetry",
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
    """Return True if text contains any unquoted ANSI-C $'...' literal."""
    return bool(text) and parse_shell_command(text).has_ansi_c


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
        w = walk_cli_flags(sub_args, FlagSpec())
        if any(f not in {"-u", "--update", "-v", "--verbose", "-n", "--dry-run", "--"} for f in w.flags_seen):
            return False
        if any(not _is_safe_git_path(tok, eff_dir, payload) for tok in w.positionals):
            return False
        return bool(w.positionals) or "-u" in w.flags_seen or "--update" in w.flags_seen
    if subcmd == "commit":
        w = walk_cli_flags(sub_args, FlagSpec(val_opts=frozenset({"-m", "--message", "-F", "--file"}), short_val_chars="mF"))
        if w.missing_value or any(f not in (_GIT_COMMIT_BOOL_LONG | {"-s", "-a", "-q", "-v", "-m", "--message", "-F", "--file", "--"}) for f in w.flags_seen):
            return False
        has_msg = any(opt in {"-m", "--message"} for opt, _ in w.opt_pairs)
        file_vals = [val for opt, val in w.opt_pairs if opt in {"-F", "--file"}]
        if any(not _is_safe_git_path(fv, eff_dir, payload, allow_scratch_or_tmp=True) for fv in file_vals) or any(not _is_safe_git_path(tok, eff_dir, payload) for tok in w.positionals):
            return False
        return has_msg or bool(file_vals) or "--no-edit" in w.flags_seen
    if subcmd == "push":
        w = walk_cli_flags(sub_args, FlagSpec())
        if any(f not in (_GIT_PUSH_BOOL_LONG | {"-u", "-n", "-v", "-q", "--"}) for f in w.flags_seen):
            return False
        pos = list(w.positionals)
        if not pos:
            return True
        if not _GIT_PUSH_REMOTE_RE.fullmatch(pos[0]) or ".." in pos[0]:
            return False
        try:
            eff_p = Path(eff_dir).resolve()
            if (eff_p / pos[0]).exists() or (eff_p / f"{pos[0]}.git").exists():
                return False
            git_cfg = eff_p / ".git" / "config"
            if not git_cfg.is_file() and (eff_p == PROJECT_ROOT or PROJECT_ROOT in eff_p.parents):
                git_cfg = PROJECT_ROOT / ".git" / "config"
            if git_cfg.is_file():
                cfg_txt = git_cfg.read_text(encoding="utf-8", errors="replace")
                if not re.search(rf'(?m)^\s*\[remote\s+"{re.escape(pos[0])}"\]', cfg_txt):
                    return False
        except (OSError, ValueError):
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
        return not walk_cli_flags(v_args, FlagSpec(bad_opts=spec[0], bad_short_chars=spec[1], stop_at_dd=False, gnu_abbrev=True)).bad_hit
    return v_name in SAFE_READONLY_SHELL_VERBS


def _is_safe_scratch_shell_subcmd(
    sc: str, cwd: str, payload: Dict[str, Any], state: Dict[str, Any], *, require_doc_ext: bool = False, parsed_sub: Optional[ParsedSubcmd] = None
) -> bool:
    """Return True if a mutating subcommand is solely tee/touch/mkdir targeting non-state scratch paths."""
    if parsed_sub is None:
        subs = parse_shell_command(sc).subcmds
        if not subs:
            return False
        parsed_sub = subs[0]
    verb, arg_tokens = parsed_sub.verb, parsed_sub.args
    if parsed_sub.unwrap_unv or verb not in {"tee", "touch", "mkdir"} or not arg_tokens:
        return False
    val_opts = frozenset({"-m", "--mode"}) if verb == "mkdir" else (frozenset({"-d", "--date", "-r", "--reference", "-t"}) if verb == "touch" else frozenset())
    targets = walk_cli_flags(arg_tokens, FlagSpec(val_opts=val_opts, short_val_chars=("m" if verb == "mkdir" else ("drt" if verb == "touch" else "")), gnu_abbrev=True)).positionals
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
    parsed: Optional[ParsedCommand] = None,
) -> bool:
    """Closed-world allowlist for Root orchestrator (and read-only/scratch subagent) run_command."""
    if not cmd or not cmd.strip():
        return True
    p = parsed if parsed is not None else parse_shell_command(cmd)
    if p.has_nul_surr or p.has_ansi_c or p.has_subshell or p.has_heredoc_subst or p.has_unclosed or p.has_cmd_subst or _payload_has_raw_cr(cmd, payload):
        return False
    if "<<" in p.unquoted or "<(" in p.unquoted or ">(" in p.unquoted or not sub_cmds or re.search(
        r"(?:^|[;&|\n({]\s*)(?:(?:if|then|elif|else|do|while|until|time|!)\s+)*(?:function\s+\w+|(?:coproc|case|select)\b)|\b\w+\s*\(\s*\)\s*\{",
        p.unquoted,
    ):
        return False

    use_pipe_meta = [s.raw for s in p.subcmds] == list(sub_cmds)
    iter_subs = p.subcmds if use_pipe_meta else tuple(s for sc in sub_cmds for s in parse_shell_command(sc).subcmds)
    cur_dir = cwd or str(PROJECT_ROOT)

    for sub in iter_subs:
        sc = sub.raw
        if sub.subshell_opens or sub.subshell_closes:
            return False
        is_piped_in = sub.is_piped_in if use_pipe_meta else False
        if re.search(r"(?:&>>|&>|>>|>\||>&(?!\d|-)|>(?!&))\s*(?:$|[;|&()])", sub.unquoted):
            return False
        brace_chk = _SIMPLE_PARAM_RE.sub("", sub.unquoted)
        if "{" in brace_chk or "}" in brace_chk:
            return False

        for rt in sub.redir_targets:
            if not allow_scratch_writes or _has_shell_path_expansion(rt):
                return False
            res_rt = _resolve_cmd_target(rt, cur_dir)
            if _is_guard_state_target(rt, payload=payload, cwd=cur_dir) or _is_guard_state_target(res_rt, payload=payload, cwd=cur_dir):
                return False
            if not (_is_allowed_research_scratch_write(res_rt, payload, state) if require_doc_ext else _is_in_allowed_scratch_dir(res_rt, payload, state)):
                return False

        has_input_redir = sub.has_in_redir
        verb, raw_tok, arg_tokens, arg_uq = sub.verb, sub.raw_verb, list(sub.args), list(sub.arg_uq)
        if sub.unwrap_unv or sub.opaque or not verb or any(ch in raw_tok for ch in _BAD_VERB_CHARS) or any(
            uq and ("{" in t or "}" in t) and t != "{}" and not _SIMPLE_PARAM_RE.fullmatch(t) for t, uq in zip(arg_tokens, arg_uq)
        ):
            return False

        is_py_verb, is_canon = bool(_PY_VERB_RE.match(verb)), _is_canonical_cmd_path(verb, raw_tok)
        allowed_sc = False
        if Path(raw_tok).name.lower() == "orchestrator_guard.py":
            allowed_sc = _is_readonly_canonical_guard_invocation(raw_tok, arg_tokens, cur_dir, is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=False)
        elif is_py_verb and is_canon and arg_tokens:
            ok_py, mode, target, rest = _parse_python_cli(arg_tokens)
            if ok_py and mode == "script" and Path(target).name.lower() == "orchestrator_guard.py":
                allowed_sc = _is_readonly_canonical_guard_invocation(target, rest, cur_dir, is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=False)

        if not allowed_sc and allow_scratch_writes and is_canon:
            allowed_sc = _is_safe_scratch_shell_subcmd(sc, cur_dir, payload, state, require_doc_ext=require_doc_ext, parsed_sub=sub)
        if not allowed_sc and verb == "git" and raw_tok in _CANONICAL_GIT_BINS and not has_input_redir:
            allowed_sc = _is_safe_readonly_git_tokens(arg_tokens, cur_dir, payload) or (
                allow_git_commit and _is_safe_git_stage_or_commit(arg_tokens, cur_dir, payload)
            )
        if not allowed_sc and _is_declared_eval_command(sc, state, payload, parsed_sub=sub):
            allowed_sc = True
        if not allowed_sc and is_canon and _is_test_command(sc, cur_dir=cur_dir, payload=payload, parsed_sub=sub):
            if not _is_guard_plugin_cwd(cur_dir) and not _is_staging_or_tmp_path(_resolve_cmd_target(".", cur_dir), payload):
                allowed_sc = _check_test_runner_args(verb, arg_tokens, cur_dir, payload)
        if not allowed_sc and (verb in _ROOT_DEV_META_VERBS or is_py_verb) and is_canon and not is_piped_in and not has_input_redir:
            allowed_sc = bool(arg_tokens and all(t in _VERSION_HELP_FLAGS for t in arg_tokens)) or (verb == "go" and arg_tokens == ["version"])
        if not allowed_sc and verb in _LINTER_VERBS and is_canon:
            allowed_sc = _is_safe_linter_subcmd(verb, arg_tokens, cur_dir, payload, is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=False)
        if not allowed_sc and is_canon and (verb in SAFE_READONLY_SHELL_VERBS or verb in _READONLY_VERB_BLOCKED_FLAGS):
            allowed_sc = _is_safe_readonly_verb_invocation(verb, arg_tokens, arg_uq)
        if not allowed_sc and is_canon and verb == "cd":
            new_dir, is_cd_cmd, cd_unv = _advance_effective_cwd(sc, cur_dir, None, payload, parsed_sub=sub)
            if is_cd_cmd and not cd_unv and not _is_guard_plugin_cwd(new_dir) and not _is_staging_or_tmp_path(new_dir, payload) and _is_within_allowed_roots(new_dir, _allowed_roots(payload)):
                cur_dir, allowed_sc = new_dir, True
        if not allowed_sc:
            return False
    return True


def _is_readonly_subagent_allowlisted_command(
    cmd: str, sub_cmds: List[str], cwd: str, payload: Dict[str, Any], state: Dict[str, Any],
    *, allow_scratch_writes: bool = False, parsed: Optional[ParsedCommand] = None,
) -> bool:
    """Closed-world allowlist for strict read-only (scout/verifier/reviewer/judge) and scratch-only (research/council) subagents."""
    return _is_root_allowlisted_command(
        cmd, sub_cmds, cwd, payload, state, allow_scratch_writes=allow_scratch_writes, require_doc_ext=allow_scratch_writes, allow_git_commit=False, parsed=parsed
    )

"""Shell write-target extraction and ALLOWED_FILES shell allowlist validation."""
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
    MUTATING_GIT_SUBCMD_RE,
    MUTATING_SHELL_CMD_RE,
    PROJECT_ROOT,
    SHELL_EXEC_WRAPPER_RE,
    _has_nul_or_surrogate_input,
    _is_test_command,
    _preprocess_shell_heredocs_and_comments,
    _split_shell_commands_with_delims,
)
from guard.identity import _allowed_roots, _is_within_allowed_roots
from guard.path_guards import (
    _guard_state_sentinel,
    _has_shell_path_expansion,
    _is_artifact_path,
    _is_guard_plugin_cwd,
    _is_guard_plugin_target,
    _is_guard_state_target,
    _is_readonly_canonical_guard_invocation,
    _is_staging_or_tmp_path,
    _is_target_in_allowed_files,
    _is_tmp_path,
    _resolve_cmd_target,
    _target_is_ancestor_of_protected_guard,
)
from guard.shell_tokens import (
    SAFE_READONLY_SHELL_VERBS,
    SAFE_XARGS_READONLY_CMDS,
    _OPAQUE_EXEC_VERBS,
    _UNVERIFIABLE_WRAPPERS,
    _advance_effective_cwd,
    _extract_shell_redir_targets,
    _has_unquoted_cmd_subst,
    _is_safe_git_inspection_command,
    _split_shell_tokens_raw_with_meta,
    _strip_redir_tokens_with_meta,
    _strip_shell_quoted_literals,
    _unquoted_group_delta,
    _unquoted_subshell_opens_closes,
    _unwrap_shell_command_tokens,
    _unwrap_shell_command_tokens_with_raw,
)

_SIMPLE_FILE_MUTATORS = frozenset(
    {"rm", "unlink", "shred", "touch", "truncate", "tee", "mkdir", "rmdir"}
)
_PERM_MUTATORS = frozenset({"chmod", "chown", "chgrp"})
_COPY_MOVE_MUTATORS = frozenset({"cp", "mv", "ln", "install"})
_NON_PY_INTERPS = re.compile(r"^(?:perl|node|ruby|php|lua)(?:\d+(?:\.\d+)*)?$")
_PY_VERBS = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)$")
_DEV_META_FLAGS = frozenset({"--version", "-V", "--help", "-h"})


def _extract_positional_args(
    args: List[str], val_flags: Tuple[str, ...] = (), short_val_chars: str = ""
) -> Tuple[List[str], List[str]]:
    """Return (positional_args, option_values) from arg tokens."""
    pos: List[str] = []
    opt_vals: List[str] = []
    after_dd = skip_next = False
    for idx, tok in enumerate(args):
        if skip_next:
            opt_vals.append(tok)
            skip_next = False
            continue
        if not after_dd and tok == "--":
            after_dd = True
            continue
        if not after_dd and tok.startswith("-") and tok != "-":
            opt = tok.split("=", 1)[0]
            if "=" in tok:
                if opt in val_flags:
                    opt_vals.append(tok.split("=", 1)[1])
                continue
            if opt in val_flags:
                skip_next = True
                continue
            if not tok.startswith("--") and short_val_chars:
                flags = tok[1:]
                for k, ch in enumerate(flags):
                    if ch in short_val_chars:
                        if k < len(flags) - 1:
                            opt_vals.append(flags[k + 1 :])
                        elif idx + 1 < len(args):
                            skip_next = True
                        break
            continue
        pos.append(tok)
    return pos, opt_vals


def _extract_shell_write_targets(
    cmd: str,
    sub_cmds: List[str],
    cwd: str = "",
    payload: Optional[Dict[str, Any]] = None,
    *,
    check_unclosed: bool = True,
    _depth: int = 0,
) -> Tuple[List[str], bool]:
    """Extract shell write targets and detect unverifiable mutating or nested shell execution commands."""
    guard_sentinel = _guard_state_sentinel()
    safe_payload: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    cleaned_cmd, h_cmd_subst, h_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    write_targets: List[str] = (
        list(_extract_shell_redir_targets(cleaned_cmd)) if not sub_cmds else []
    )
    unquoted_cmd = _strip_shell_quoted_literals(cleaned_cmd)
    has_unverifiable_mut = (
        _has_nul_or_surrogate_input(cmd)
        or h_cmd_subst
        or (check_unclosed and h_unclosed)
        or bool(SHELL_EXEC_WRAPPER_RE.search(cleaned_cmd))
        or _has_unquoted_cmd_subst(cleaned_cmd)
        or "<(" in unquoted_cmd
        or ">(" in unquoted_cmd
        or bool(
            re.search(
                r"(?:^|[;&|\n({]\s*)(?:(?:if|then|elif|else|do|while|until|coproc|time|!)\s+)*(?:function\b|case\b|coproc\b|[^\s;|&()<>'\"`]+?\s*\(\s*\))",
                unquoted_cmd,
            )
        )
    )

    initial_cwd = cwd or str(PROJECT_ROOT)
    effective_cwd = initial_cwd
    dir_stack: List[str] = []
    cwd_unknown = False
    subshell_stack: List[Tuple[str, List[str], bool]] = []
    pipe_group_depth = 0
    stdin_exec_redir = False

    delim_source = cleaned_cmd if cleaned_cmd else " ; ".join(sub_cmds)
    delim_parts = [(d, p) for d, p in _split_shell_commands_with_delims(delim_source) if p]
    use_delim_map = [p for _, p in delim_parts] == list(sub_cmds)

    def _is_protected(tgt: str) -> bool:
        res = _resolve_cmd_target(tgt, effective_cwd)
        return (
            _is_guard_plugin_target(tgt, effective_cwd)
            or _is_guard_plugin_target(res, effective_cwd)
            or _is_guard_state_target(tgt, payload=safe_payload)
            or _is_guard_state_target(res, payload=safe_payload)
            or _target_is_ancestor_of_protected_guard(tgt, effective_cwd, safe_payload)
        )

    def _flag_if_protected(candidates: List[str]) -> None:
        nonlocal has_unverifiable_mut
        for cand in candidates:
            if cand and _is_protected(cand):
                write_targets.append(guard_sentinel)
                has_unverifiable_mut = True

    for sc_idx, sc in enumerate(sub_cmds):
        opens, closes = _unquoted_subshell_opens_closes(sc)
        for _ in range(opens):
            subshell_stack.append((effective_cwd, list(dir_stack), cwd_unknown))

        sc_delim = delim_parts[sc_idx][0] if use_delim_map else None
        is_direct_pipe = sc_delim in ("|", "|&")
        is_piped_in = is_direct_pipe or pipe_group_depth > 0
        if is_piped_in:
            pipe_group_depth = max(0, pipe_group_depth + _unquoted_group_delta(sc))

        wt_start = len(write_targets)
        raw_meta = _split_shell_tokens_raw_with_meta(sc)
        sc_has_input_redir = any(is_redir and "<" in tok for tok, is_redir, _ in raw_meta)
        has_input_redir = sc_has_input_redir or stdin_exec_redir
        for rt in _extract_shell_redir_targets(sc):
            if rt != "/dev/null":
                write_targets.append(rt)

        tokens, unquoted_flags = _strip_redir_tokens_with_meta(raw_meta)
        arg_uq: List[bool] = []
        unwrap_meta: Dict[str, bool] = {}
        verb, raw_cmd_tok, arg_tokens, unwrap_unv = _unwrap_shell_command_tokens_with_raw(
            tokens, unquoted_flags, out_arg_flags=arg_uq, out_meta=unwrap_meta
        )
        if verb == "exec" and sc_has_input_redir:
            stdin_exec_redir = True
        if unwrap_unv or unwrap_meta.get("opaque"):
            has_unverifiable_mut = True

        new_cwd, is_cd_cmd, cd_unv = _advance_effective_cwd(
            sc, effective_cwd, dir_stack, safe_payload
        )
        if is_cd_cmd:
            effective_cwd = new_cwd
            if cd_unv:
                cwd_unknown = True
            if _is_guard_plugin_cwd(effective_cwd):
                write_targets.append(guard_sentinel)
                has_unverifiable_mut = True
        elif not verb:
            if sc.strip() and not tokens:
                pass
            elif sc.strip() and MUTATING_SHELL_CMD_RE.search(sc):
                has_unverifiable_mut = True
        elif any(ch in raw_cmd_tok for ch in "${}*?[]<>()"):
            has_unverifiable_mut = True
            _flag_if_protected([raw_cmd_tok, *arg_tokens])
        elif verb in _SHELL_WRAPPER_VERBS or verb in _UNVERIFIABLE_WRAPPERS or verb in _OPAQUE_EXEC_VERBS:
            if not (
                verb not in {"eval", "source", "."}
                and arg_tokens
                and all(t in _DEV_META_FLAGS for t in arg_tokens)
                and not is_piped_in
                and not has_input_redir
            ):
                has_unverifiable_mut = True
                _flag_if_protected(arg_tokens)
        elif Path(raw_cmd_tok).name.lower() == "orchestrator_guard.py":
            if not _is_readonly_canonical_guard_invocation(
                raw_cmd_tok, arg_tokens, effective_cwd,
                is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown,
            ):
                write_targets.append(guard_sentinel)
                has_unverifiable_mut = True
        elif verb in _SIMPLE_FILE_MUTATORS:
            if verb == "mkdir":
                pos, _ = _extract_positional_args(arg_tokens, ("-m", "--mode"), "m")
            elif verb == "touch":
                pos, _ = _extract_positional_args(arg_tokens, ("-d", "--date", "-r", "--reference", "-t"), "drt")
            elif verb == "truncate":
                pos, _ = _extract_positional_args(arg_tokens, ("-s", "--size", "-r", "--reference"), "sr")
            elif verb == "shred":
                pos, _ = _extract_positional_args(arg_tokens, ("-n", "--iterations", "-s", "--size", "--random-source"), "ns")
            else:
                pos, _ = _extract_positional_args(arg_tokens)
            write_targets.extend(pos)
            if verb in {"rm", "rmdir"}:
                for tgt in pos:
                    if _target_is_ancestor_of_protected_guard(tgt, effective_cwd, safe_payload):
                        write_targets.append(guard_sentinel)
                        has_unverifiable_mut = True
        elif verb in _PERM_MUTATORS:
            pos, _ = _extract_positional_args(arg_tokens, ("--reference",), "")
            targets = pos if any(t.startswith("--reference=") or t == "--reference" for t in arg_tokens) else pos[1:]
            write_targets.extend(targets)
            for tgt in targets:
                if _target_is_ancestor_of_protected_guard(tgt, effective_cwd, safe_payload):
                    write_targets.append(guard_sentinel)
                    has_unverifiable_mut = True
        elif verb in _COPY_MOVE_MUTATORS:
            pos, opt_vals = _extract_positional_args(
                arg_tokens, ("-t", "--target-directory", "-S", "--suffix", "-m", "--mode", "-o", "--owner", "-g", "--group"), "tSmog"
            )
            t_dir = None
            for idx_a, tok_a in enumerate(arg_tokens):
                if tok_a in {"-t", "--target-directory"} and idx_a + 1 < len(arg_tokens):
                    t_dir = arg_tokens[idx_a + 1]
                elif tok_a.startswith("--target-directory="):
                    t_dir = tok_a.split("=", 1)[1]
                elif tok_a.startswith("-t") and not tok_a.startswith("--") and len(tok_a) > 2:
                    t_dir = tok_a[2:]
            is_symlink = verb == "ln" and any(
                t in {"-s", "--symbolic"} or (t.startswith("-") and not t.startswith("--") and "s" in t[1:])
                for t in arg_tokens
            )
            is_hardlink = verb == "ln" and not is_symlink
            if verb == "cp" and any(
                t in {"-s", "--symbolic-link", "-l", "--link"}
                or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "sl"))
                for t in arg_tokens
            ):
                has_unverifiable_mut = True
            if is_symlink or is_hardlink:
                has_unverifiable_mut = True
            if t_dir:
                dests = [t_dir]
                srcs = pos
            elif len(pos) >= 2:
                dests = [pos[-1]]
                srcs = pos[:-1]
            else:
                dests = list(pos)
                srcs = []
            write_targets.extend(dests)
            if verb == "mv":
                write_targets.extend(srcs)
            is_recursive = verb == "cp" and any(
                t in {"-r", "-R", "-a", "--recursive", "--archive"}
                or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "rRa"))
                for t in arg_tokens
            )
            if verb == "mv" or is_recursive:
                for tgt in srcs + dests:
                    if _target_is_ancestor_of_protected_guard(tgt, effective_cwd, safe_payload):
                        write_targets.append(guard_sentinel)
                        has_unverifiable_mut = True
            ws_roots = _allowed_roots(safe_payload) if safe_payload else [PROJECT_ROOT]
            for src in srcs:
                res_src = _resolve_cmd_target(src, effective_cwd)
                if not _is_within_allowed_roots(res_src, ws_roots) or _is_protected(src):
                    has_unverifiable_mut = True
                    _flag_if_protected([src])
        elif verb == "dd":
            saw_of = False
            for tok in arg_tokens:
                if tok.startswith("of="):
                    saw_of = True
                    write_targets.append(tok[3:])
                elif tok.startswith("if="):
                    if _is_protected(tok[3:]):
                        write_targets.append(guard_sentinel)
                        has_unverifiable_mut = True
            if not saw_of:
                has_unverifiable_mut = True
        elif verb == "sed":
            in_place = any(
                t == "--in-place" or t.startswith("--in-place=")
                or (t.startswith("-") and not t.startswith("--") and "i" in t[1:])
                for t in arg_tokens
            )
            has_e_or_f = any(
                t in {"-e", "-f", "--expression", "--file"} or t.startswith(("--expression=", "--file="))
                or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "ef"))
                for t in arg_tokens
            )
            has_f = any(
                t in {"-f", "--file"} or t.startswith("--file=")
                or (t.startswith("-") and not t.startswith("--") and "f" in t[1:])
                for t in arg_tokens
            )
            pos, opt_vals = _extract_positional_args(
                arg_tokens, ("-e", "-f", "--expression", "--file", "-l", "--line-length"), "efl"
            )
            if in_place:
                files = pos if has_e_or_f else pos[1:]
                if not files or has_f:
                    has_unverifiable_mut = True
                write_targets.extend(files)
            elif has_f or (has_e_or_f and any("w" in ov or "e" in ov for ov in opt_vals)) or (
                not has_e_or_f and pos and ("w" in pos[0] or "e" in pos[0])
            ):
                has_unverifiable_mut = True
                _flag_if_protected(pos + opt_vals)
        elif verb == "sort":
            pos, opt_vals = _extract_positional_args(arg_tokens, ("-o", "--output", "--compress-program", "-t", "-k", "-S", "-T"), "otkST")
            for idx_a, tok_a in enumerate(arg_tokens):
                if tok_a == "--":
                    break
                if tok_a in {"-o", "--output"} and idx_a + 1 < len(arg_tokens):
                    write_targets.append(arg_tokens[idx_a + 1])
                elif tok_a.startswith("--output="):
                    write_targets.append(tok_a.split("=", 1)[1])
                elif tok_a.startswith("-o") and not tok_a.startswith("--") and len(tok_a) > 2:
                    write_targets.append(tok_a[2:])
                elif tok_a == "--compress-program" or tok_a.startswith("--compress-program="):
                    has_unverifiable_mut = True
            _ = pos, opt_vals
        elif verb == "printf":
            for t in arg_tokens:
                if t == "--":
                    break
                if t == "-v" or (t.startswith("-") and not t.startswith("--") and "v" in t[1:]) or ("{" in t or "}" in t):
                    has_unverifiable_mut = True
        elif verb == "rg":
            pre_dd = arg_tokens[: arg_tokens.index("--")] if "--" in arg_tokens else arg_tokens
            if any(t == "--pre" or t.startswith("--pre=") for t in pre_dd):
                has_unverifiable_mut = True
        elif verb == "git":
            if not _is_safe_git_inspection_command(sc):
                if MUTATING_GIT_SUBCMD_RE.match(sc) or MUTATING_SHELL_CMD_RE.search(sc) or has_input_redir:
                    has_unverifiable_mut = True
                    _flag_if_protected(arg_tokens)
        elif _PY_VERBS.match(verb):
            if arg_tokens and all(t in _DEV_META_FLAGS for t in arg_tokens) and not is_piped_in and not has_input_redir:
                pass
            else:
                after_dd = skip_next = False
                inline_code = ""
                script_tok = None
                mod_name = ""
                mod_args: List[str] = []
                is_mod = False
                for p_i, p_t in enumerate(arg_tokens):
                    if skip_next:
                        skip_next = False
                        continue
                    if not after_dd and p_t == "--":
                        after_dd = True
                        continue
                    if not after_dd and p_t.startswith("-") and p_t != "-":
                        if p_t in {"-W", "-X", "--check-hash-based-pycs"}:
                            skip_next = True
                            continue
                        if not p_t.startswith("--") and len(p_t) > 1:
                            flags = p_t[1:]
                            for k, ch in enumerate(flags):
                                if ch == "c":
                                    inline_code = flags[k + 1 :] or (arg_tokens[p_i + 1] if p_i + 1 < len(arg_tokens) else "")
                                    break
                                if ch == "m":
                                    is_mod = True
                                    rest = flags[k + 1 :]
                                    if rest:
                                        mod_name, mod_args = rest, arg_tokens[p_i + 1 :]
                                    elif p_i + 1 < len(arg_tokens):
                                        mod_name, mod_args = arg_tokens[p_i + 1], arg_tokens[p_i + 2 :]
                                    break
                                if ch in ("W", "X"):
                                    if k == len(flags) - 1:
                                        skip_next = True
                                    break
                            if inline_code or is_mod:
                                break
                        continue
                    script_tok = p_t
                    break
                if inline_code:
                    if _python_inline_code_is_mutating(inline_code):
                        has_unverifiable_mut = True
                elif is_mod and mod_name in {"pytest", "py.test"}:
                    pt_tgts, pt_mut = _check_pytest_mutating_args(mod_args)
                    write_targets.extend(pt_tgts)
                    if (
                        pt_mut
                        or _pytest_targets_staging_or_tmp(mod_args, effective_cwd, safe_payload)
                        or cwd_unknown
                        or _is_staging_or_tmp_path(_resolve_cmd_target(".", effective_cwd), safe_payload)
                        or not _is_test_command(sc)
                    ):
                        has_unverifiable_mut = True
                elif is_mod and mod_name == "unittest" and _is_test_command(sc):
                    if (
                        cwd_unknown
                        or _is_staging_or_tmp_path(_resolve_cmd_target(".", effective_cwd), safe_payload)
                        or any(
                            _is_staging_or_tmp_path(_resolve_cmd_target(a.split("=", 1)[-1], effective_cwd), safe_payload)
                            for a in mod_args
                            if a not in {"discover", "-v", "--verbose"}
                        )
                    ):
                        has_unverifiable_mut = True
                elif script_tok and Path(script_tok).name.lower() == "orchestrator_guard.py":
                    rest_args = arg_tokens[arg_tokens.index(script_tok) + 1 :]
                    if not _is_readonly_canonical_guard_invocation(
                        script_tok, rest_args, effective_cwd,
                        is_piped_in=is_piped_in, has_input_redir=has_input_redir, cwd_unknown=cwd_unknown,
                    ):
                        write_targets.append(guard_sentinel)
                        has_unverifiable_mut = True
                else:
                    has_unverifiable_mut = True
                    _flag_if_protected(arg_tokens)
        elif verb in {"pytest", "py.test"}:
            pt_tgts, pt_mut = _check_pytest_mutating_args(arg_tokens)
            write_targets.extend(pt_tgts)
            if pt_mut or _pytest_targets_staging_or_tmp(arg_tokens, effective_cwd, safe_payload) or not _is_test_command(sc):
                has_unverifiable_mut = True
        elif _is_test_command(sc):
            if (
                cwd_unknown
                or _is_staging_or_tmp_path(_resolve_cmd_target(".", effective_cwd), safe_payload)
                or not _check_test_runner_args(verb, arg_tokens, effective_cwd, safe_payload)
            ):
                has_unverifiable_mut = True
        elif verb in {"mypy", "pylint", "pyright", "flake8", "tsc", "playwright", "ruff", "black", "isort"}:
            pre_dd = arg_tokens[: arg_tokens.index("--")] if "--" in arg_tokens else arg_tokens
            bad_linter = (
                cwd_unknown
                or is_piped_in
                or has_input_redir
                or _is_guard_plugin_cwd(effective_cwd)
                or _is_staging_or_tmp_path(_resolve_cmd_target(".", effective_cwd), safe_payload)
                or any(
                    t.split("=", 1)[0] in {
                        "--init", "--load-plugins", "--init-hook", "--rcfile", "--output", "-o",
                        "--output-file", "--eval-args", "--fix", "--write", "-i", "--in-place",
                        "--sqlite-cache", "--install-types", "--cache-dir", "--shadow-file",
                        "--config-file", "--custom-typeshed", "--reporter",
                    }
                    or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "oi"))
                    for t in pre_dd
                )
                or (verb == "playwright" and any(t in {"code", "codegen", "create", "install"} for t in pre_dd))
                or (verb == "tsc" and ("--noEmit" not in pre_dd or any(t.lower() == "false" for t in pre_dd)))
                or any(
                    _is_staging_or_tmp_path(_resolve_cmd_target(t.split("=", 1)[-1], effective_cwd), safe_payload)
                    or _is_guard_plugin_target(t.split("=", 1)[-1], effective_cwd)
                    or _is_guard_state_target(t.split("=", 1)[-1], payload=safe_payload)
                    for t in arg_tokens
                    if t != "--" and (not t.startswith("-") or "=" in t)
                )
            )
            if bad_linter:
                has_unverifiable_mut = True
        elif verb in {"go", "cargo", "rustc"} and arg_tokens in (["version"], ["--version"], ["-V"], ["--help"], ["-h"]):
            pass
        elif verb in SAFE_READONLY_SHELL_VERBS:
            if verb == "cat" and has_input_redir and not sc_has_input_redir:
                pass
        elif arg_tokens and all(t in _DEV_META_FLAGS for t in arg_tokens) and not is_piped_in and not has_input_redir:
            pass
        elif verb == "find":
            in_exec = False
            exec_toks: List[str] = []
            find_bad = False
            for ft in arg_tokens:
                if in_exec:
                    if ft in {";", "+"}:
                        in_exec = False
                        iv, ia, iunv = _unwrap_shell_command_tokens(exec_toks)
                        if iunv or not iv or iv in _SHELL_WRAPPER_VERBS or iv not in SAFE_READONLY_SHELL_VERBS:
                            find_bad = True
                            break
                        exec_toks = []
                    else:
                        exec_toks.append(ft)
                elif ft in {"-delete", "-fls", "-execdir", "-ok", "-okdir"} or ft.startswith("-fprint"):
                    find_bad = True
                    break
                elif ft == "-exec":
                    in_exec, exec_toks = True, []
            if find_bad or in_exec:
                has_unverifiable_mut = True
                _flag_if_protected(arg_tokens)
        elif verb == "xargs":
            pos, opt_vals = _extract_positional_args(arg_tokens, ("-a", "--arg-file", "-I", "-L", "-n", "-P", "-d", "-E", "-s"), "aILnPdEs")
            x_name = Path(pos[0]).name.lower() if pos else ""
            if x_name not in SAFE_XARGS_READONLY_CMDS or x_name in {"rg", "printf"}:
                has_unverifiable_mut = True
                _flag_if_protected(pos + opt_vals)
        else:
            has_unverifiable_mut = True
            _flag_if_protected(arg_tokens)

        if len(write_targets) > wt_start:
            if cwd_unknown:
                has_unverifiable_mut = True
            elif effective_cwd != initial_cwd:
                for wi in range(wt_start, len(write_targets)):
                    write_targets[wi] = _resolve_cmd_target(write_targets[wi], effective_cwd)
        elif cwd_unknown and bool(MUTATING_SHELL_CMD_RE.search(sc)):
            has_unverifiable_mut = True

        for _ in range(closes):
            if subshell_stack:
                effective_cwd, dir_stack, cwd_unknown = subshell_stack.pop()

    if any(_has_shell_path_expansion(wt) for wt in write_targets):
        has_unverifiable_mut = True

    return write_targets, has_unverifiable_mut


def _validate_shell_allowlist(
    cmd: str,
    sub_cmds: List[str],
    cwd: str,
    allowed_files: List[str],
    payload: Dict[str, Any],
    allowlist_label: str = "ALLOWED_FILES",
) -> Optional[Dict[str, Any]]:
    """Validate shell write targets and mutating/wrapper commands against an allowlist."""
    write_targets, has_unverifiable_mut = _extract_shell_write_targets(
        cmd, sub_cmds, cwd=cwd, payload=payload
    )
    if has_unverifiable_mut:
        return {
            "decision": "deny",
            "reason": (
                f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: "
                f"mutating or nested shell command '{cmd.strip()[:60]}' cannot be verified against {allowlist_label} "
                f"({', '.join(allowed_files) if allowed_files else 'NONE'})."
            ),
        }
    for wt in write_targets:
        resolved_wt = _resolve_cmd_target(wt, cwd)
        if not _is_artifact_path(resolved_wt, payload) and not _is_tmp_path(resolved_wt):
            if not _is_target_in_allowed_files(
                resolved_wt, allowed_files, payload.get("workspacePaths")
            ):
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: "
                        f"'{wt}' is outside {allowlist_label} "
                        f"({', '.join(allowed_files) if allowed_files else 'NONE'})."
                    ),
                }
    return None

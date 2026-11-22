"""Shell write-target extraction, safe git/linter helpers, and ALLOWED_FILES shell validation."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from guard.ast_guard import _check_test_runner_args, _extract_pytest_target_check, _is_test_command
from guard.constants import PROJECT_ROOT
from guard.identity import _allowed_roots, _is_within_allowed_roots
from guard.path_guards import (
    _guard_state_sentinel, _has_shell_path_expansion, _is_artifact_path, _is_guard_plugin_cwd, _is_guard_plugin_target,
    _is_guard_state_target, _is_readonly_canonical_guard_invocation, _is_staging_or_tmp_path, _is_target_in_allowed_files,
    _is_tmp_path, _resolve_cmd_target, _target_is_ancestor_of_protected_guard,
)
from guard.shell_tokens import (
    SAFE_READONLY_SHELL_VERBS, SAFE_XARGS_READONLY_CMDS, _OPAQUE_EXEC_VERBS, _PY_VERB_RE, _SHELL_WRAPPER_VERBS,
    _UNVERIFIABLE_WRAPPERS, _advance_effective_cwd, _extract_shell_redir_targets, _has_nul_or_surrogate_input,
    _has_unquoted_cmd_subst, _parse_python_cli, _preprocess_shell_heredocs_and_comments, _split_shell_commands_with_delims,
    _split_shell_tokens_raw_with_meta, _strip_redir_tokens_with_meta, _strip_shell_quoted_literals, _unquoted_group_delta,
    _unquoted_subshell_opens_closes, _unwrap_shell_command_tokens, _unwrap_shell_command_tokens_with_raw,
)

# Table-driven simple file mutators: verb -> (val_flags, short_val_chars)
_SIMPLE_MUTATOR_SPECS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "mkdir": (("-m", "--mode"), "m"), "touch": (("-d", "--date", "-r", "--reference", "-t"), "drt"),
    "truncate": (("-s", "--size", "-r", "--reference"), "sr"), "shred": (("-n", "--iterations", "-s", "--size", "--random-source"), "ns"),
    "rm": ((), ""), "unlink": ((), ""), "rmdir": ((), ""), "tee": ((), ""),
}
_PERM_MUTATORS = frozenset({"chmod", "chown", "chgrp"})
_CHMOD_LONG_FLAGS = frozenset({"--recursive", "--changes", "--silent", "--quiet", "--verbose", "--preserve-root", "--no-preserve-root"})
_COPY_MOVE_MUTATORS = frozenset({"cp", "mv", "ln", "install"})
_LINTER_VERBS = frozenset({"mypy", "pylint", "pyright", "flake8", "tsc", "playwright", "ruff", "black", "isort"})
_BAD_LINTER_OPTS = frozenset({"--init", "--load-plugins", "--init-hook", "--rcfile", "--output", "-o", "--output-file", "--eval-args", "--fix", "--write", "-i", "--in-place", "--sqlite-cache", "--install-types", "--cache-dir", "--shadow-file", "--config-file", "--custom-typeshed", "--reporter"})
_DEV_META_FLAGS = frozenset({"--version", "-V", "--help", "-h"})
_SAFE_DEV_META_VERBS = frozenset({
    "python", "python3", "pytest", "node", "npm", "npx", "git", "cargo", "go", "rustc", "ruff", "flake8",
    "mypy", "eslint", "tsc", "pip", "pip3", "uv", "poetry", "make", "cmake", "gcc", "clang", "java", "javac", "ruby", "perl", "php",
})
_READONLY_GIT_SUBCMDS = frozenset({"status", "diff", "log", "show", "rev-parse", "describe", "ls-files", "ls-tree", "shortlog", "blame", "grep", "rev-list", "cat-file", "for-each-ref", "merge-base", "name-rev", "check-ignore", "version", "help"})
_BAD_GIT_OPTS = frozenset({"-c", "--config-env", "--exec-path", "--upload-pack", "--receive-pack", "--output", "--ext-diff", "--open-files-in-pager", "--no-index", "--exec"})
_BAD_GIT_PREFIXES = tuple(o + "=" for o in _BAD_GIT_OPTS if o.startswith("--"))
_GIT_GLOBAL_VAL_OPTS = frozenset({"-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--list-cmds"})
_GIT_GLOBAL_BOOL_OPTS = frozenset({"--no-pager", "-p", "--paginate", "-P", "--bare", "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs", "--no-optional-locks", "--version", "-v", "--help", "-h", "--html-path", "--man-path", "--info-path"})
_GIT_LIST_SUBCMD_SPECS: Dict[str, Tuple[frozenset[str], frozenset[str], str, str, bool]] = {
    "branch": (frozenset({"--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format", "--abbrev", "--color"}), frozenset({"-a", "--all", "-r", "--remotes", "-v", "--verbose", "-vv", "-l", "--list", "--show-current", "--no-color", "--column", "--no-column", "--no-abbrev"}), "l", "dDmMcCftu", False),
    "tag": (frozenset({"-n", "--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format", "--color"}), frozenset({"-l", "--list", "-v", "--verify", "--no-color", "-i", "--ignore-case"}), "lv", "dasufmF", True),
}


def _is_safe_dev_meta_cmd(verb: str, raw_tok: str, args: List[str], is_piped_in: bool, has_in_redir: bool) -> bool:
    """Return True only for known dev/runtime binaries invoked without custom paths and with meta flags."""
    if is_piped_in or has_in_redir or not args or verb not in _SAFE_DEV_META_VERBS or raw_tok.lower() not in {verb, f"/usr/bin/{verb}", f"/bin/{verb}"}:
        return False
    return all(t in _DEV_META_FLAGS for t in args) or (verb in {"go", "cargo", "rustc"} and args == ["version"])


def _extract_positional_args(
    args: List[str], val_flags: Tuple[str, ...] = (), short_val_chars: str = "", out_pairs: Optional[List[Tuple[str, str]]] = None
) -> Tuple[List[str], List[str]]:
    """Return (positional_args, option_values) from arg tokens."""
    pos, opt_vals, after_dd, skip_opt = [], [], False, ""
    for idx, tok in enumerate(args):
        if skip_opt:
            opt_vals.append(tok)
            if out_pairs is not None: out_pairs.append((skip_opt, tok))
            skip_opt = ""
        elif not after_dd and tok == "--": after_dd = True
        elif not after_dd and tok.startswith("-") and tok != "-":
            if tok.startswith("--"):
                opt, has_eq, val = tok.partition("=")
                if opt in val_flags:
                    if has_eq: opt_vals.append(val); (out_pairs.append((opt, val)) if out_pairs is not None else None)
                    else: skip_opt = opt
            elif tok in val_flags: skip_opt = tok
            elif short_val_chars:
                for k, ch in enumerate(tok[1:]):
                    if ch in short_val_chars:
                        if k < len(tok[1:]) - 1:
                            val = tok[1:][k + 1 :]; opt_vals.append(val)
                            if out_pairs is not None: out_pairs.append(("-" + ch, val))
                        elif idx + 1 < len(args): skip_opt = "-" + ch
                        break
        else: pos.append(tok)
    return pos, opt_vals


def _parse_git_tokens(arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any], *, require_allowed_roots: bool = False, allow_git_dir: bool = True) -> Optional[Tuple[str, List[str], str]]:
    """Parse git global options and return (subcmd, sub_args, eff_dir) or None if unsafe."""
    subcmd, sub_args, eff_dir, skip_opt = "", [], cur_dir, ""
    _bad_opt = lambda t: t in _BAD_GIT_OPTS or t.startswith(_BAD_GIT_PREFIXES) or (t.startswith("-c") and not t.startswith("--") and len(t) > 2)

    def _safe_val(opt_name: str, val: str) -> bool:
        nonlocal eff_dir
        if not allow_git_dir and opt_name in {"--git-dir", "--work-tree"}: return False
        if _has_shell_path_expansion(val) or _is_guard_plugin_target(val, eff_dir) or _is_guard_state_target(val, payload=payload, cwd=eff_dir) or _is_staging_or_tmp_path(_resolve_cmd_target(val, eff_dir), payload):
            return False
        if require_allowed_roots and opt_name in {"-C", "--git-dir", "--work-tree"}:
            res = _resolve_cmd_target(val, eff_dir)
            if not _is_within_allowed_roots(res, _allowed_roots(payload)): return False
            eff_dir = res if opt_name == "-C" else eff_dir
        return True

    for idx, tok in enumerate(arg_tokens):
        if "ext::" in tok.lower() or "fd::" in tok.lower(): return None
        if skip_opt:
            opt_name, skip_opt = skip_opt, ""
            if not _safe_val(opt_name, tok): return None
        elif _bad_opt(tok): return None
        elif tok in _GIT_GLOBAL_VAL_OPTS: skip_opt = tok
        elif tok.startswith(("--git-dir=", "--work-tree=", "--namespace=", "--super-prefix=", "--list-cmds=")) or (tok.startswith("-C") and len(tok) > 2):
            if not _safe_val(*(tok.split("=", 1) if tok.startswith("--") else ("-C", tok[2:]))): return None
        elif tok not in _GIT_GLOBAL_BOOL_OPTS or (not allow_git_dir and tok == "--bare"):
            if tok.startswith("-"): return None
            subcmd, sub_args = tok.lower(), arg_tokens[idx + 1 :]
            break
    if skip_opt or any("ext::" in t.lower() or "fd::" in t.lower() for t in sub_args) or any(_bad_opt(t) for t in sub_args[: (sub_args.index("--") if "--" in sub_args else len(sub_args))]):
        return None
    return subcmd, sub_args, eff_dir


def _is_safe_readonly_git_tokens(arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any]) -> bool:
    """Closed-world check for read-only git subcommands."""
    if not arg_tokens: return True
    if (parsed := _parse_git_tokens(arg_tokens, cur_dir, payload)) is None: return False
    subcmd, sub_args, _ = parsed
    if not subcmd: return all(t in {"--version", "-v", "--help", "-h"} for t in arg_tokens)
    if subcmd in _READONLY_GIT_SUBCMDS:
        return subcmd != "grep" or not any(t.startswith("-") and not t.startswith("--") and "O" in t[1:] for t in sub_args[: (sub_args.index("--") if "--" in sub_args else len(sub_args))])
    if spec := _GIT_LIST_SUBCMD_SPECS.get(subcmd):
        val_f, bool_f, allow_short, bad_short, allow_dd = spec
        skip = saw_list = False
        for t in sub_args:
            if skip: skip = False
            elif t == "--": return allow_dd
            elif t in bool_f: saw_list = saw_list or t in {"-l", "--list", "-v", "--verify"}
            elif t in val_f: skip = True
            elif t.startswith(tuple(f + "=" for f in val_f)) or t.startswith("--color=") or (subcmd == "tag" and t.startswith("-n")): continue
            elif t.startswith("-") and not t.startswith("--") and not any(c in t[1:] for c in bad_short) and all(c in (allow_short + ("arvi" if subcmd == "branch" else "i")) for c in t[1:]):
                saw_list = saw_list or any(c in t[1:] for c in allow_short)
            elif not (saw_list and not t.startswith("-")): return False
        return not skip
    if subcmd == "remote": return not sub_args or all(t in {"-v", "--verbose"} for t in sub_args) or sub_args[0] in {"show", "get-url"}
    if subcmd == "reflog": return not sub_args or sub_args[0] in {"show", "exists"} or (sub_args[0].startswith("-") and not any(x in sub_args for x in ("expire", "delete", "drop")))
    return subcmd == "fsck" and "--lost-found" not in sub_args


def _is_safe_linter_subcmd(verb: str, arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any], *, is_piped_in: bool, has_input_redir: bool, cwd_unknown: bool) -> bool:
    """Validate read-only linter/typechecker invocations."""
    if cwd_unknown or is_piped_in or has_input_redir or _is_guard_plugin_cwd(cur_dir) or _is_staging_or_tmp_path(_resolve_cmd_target(".", cur_dir), payload):
        return False
    pre_dd = arg_tokens[: arg_tokens.index("--")] if "--" in arg_tokens else arg_tokens
    if any(t.split("=", 1)[0] in _BAD_LINTER_OPTS or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "oi")) for t in pre_dd):
        return False
    if (verb == "playwright" and any(t in {"code", "codegen", "create", "install"} for t in pre_dd)) or (verb == "tsc" and ("--noEmit" not in pre_dd or any(t.lower() == "false" for t in pre_dd))):
        return False
    return not any(_is_staging_or_tmp_path(_resolve_cmd_target(t.split("=", 1)[-1], cur_dir), payload) or _is_guard_plugin_target(t.split("=", 1)[-1], cur_dir) or _is_guard_state_target(t.split("=", 1)[-1], payload=payload, cwd=cur_dir) for t in arg_tokens if t != "--" and (not t.startswith("-") or "=" in t))


def _sed_script_has_unsafe_cmd(script: str) -> bool:
    """Return True if a sed script uses e/w/W commands or s///e|w flags."""
    i, n = 0, len(script)
    def _skip(pos: int, d: str, regex: bool = False) -> int:
        while pos < n:
            if script[pos] == "\\" and pos + 1 < n: pos += 2
            elif regex and script[pos] == "[" and d not in "[]":
                j = pos + 1 + (pos + 1 < n and script[pos + 1] == "^")
                j += (j < n and script[j] == "]")
                while j < n and script[j] != "]":
                    if script[j] == "[" and j + 1 < n and script[j + 1] in ":.=":
                        end = script.find(script[j + 1] + "]", j + 2)
                        j = end + 2 if end >= 0 else n
                    else: j += 1
                pos = j + 1 if j < n else pos + 1
            elif script[pos] == d: return pos + 1
            else: pos += 1
        return -1
    while i < n:
        while i < n and script[i] in " \t\r\n;{}": i += 1
        if i >= n: break
        if script[i] == "#": nl = script.find("\n", i); i = n if nl < 0 else nl + 1; continue
        while i < n:
            if script[i] in " \t,0123456789$+~!": i += 1
            elif script[i] == "/" or (script[i] == "\\" and i + 1 < n):
                if (i := _skip(i + (1 if script[i] == "/" else 2), "/" if script[i] == "/" else script[i + 1], True)) < 0: return True
                while i < n and script[i] in "IM": i += 1
            else: break
        while i < n and script[i] in " \t!": i += 1
        if i >= n or script[i] in ";\n{}": continue
        cmd, i = script[i], i + 1
        if cmd in ("e", "w", "W"): return True
        if cmd in ("s", "y"):
            if i >= n or (p1 := _skip(i + 1, (d := script[i]), cmd == "s")) < 0 or (i := _skip(p1, d)) < 0: return True
            while cmd == "s" and i < n and script[i] not in ";\n}#":
                if script[i] in ("e", "w", "W"): return True
                i += 1
        elif cmd in ("a", "i", "c"):
            while i < n and script[i] != "\n": i += 2 if script[i] == "\\" and i + 1 < n else 1
        elif cmd in (":", "b", "t", "T", "r", "R"):
            while i < n and script[i] not in ";\n}": i += 1
    return False


def _extract_shell_write_targets(cmd: str, sub_cmds: List[str], cwd: str = "", payload: Optional[Dict[str, Any]] = None, *, check_unclosed: bool = True) -> Tuple[List[str], bool]:
    """Extract shell write targets and detect unverifiable mutating or nested shell execution commands."""
    guard_sentinel, safe_payload = _guard_state_sentinel(), (payload if isinstance(payload, dict) else {})
    cleaned_cmd, h_cmd_subst, h_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    write_targets: List[str] = list(_extract_shell_redir_targets(cleaned_cmd)) if not sub_cmds else []
    unquoted_cmd = _strip_shell_quoted_literals(cleaned_cmd)
    has_unv = (
        _has_nul_or_surrogate_input(cmd) or h_cmd_subst or (check_unclosed and h_unclosed) or _has_unquoted_cmd_subst(cleaned_cmd)
        or "<(" in unquoted_cmd or ">(" in unquoted_cmd
        or bool(re.search(r"(?:^|[;&|\n({]\s*)(?:(?:if|then|elif|else|do|while|until|coproc|time|!)\s+)*(?:function\b|case\b|coproc\b|[^\s;|&()<>'\"`]+?\s*\(\s*\))", unquoted_cmd))
    )
    initial_cwd = cwd or str(PROJECT_ROOT)
    eff_cwd, dir_stack, cwd_unknown, subshell_stack, pipe_depth, stdin_exec = initial_cwd, [], False, [], 0, False
    delim_parts = [(d, p) for d, p in _split_shell_commands_with_delims(cleaned_cmd or " ; ".join(sub_cmds)) if p]
    use_delim_map = [p for _, p in delim_parts] == list(sub_cmds)
    _is_prot = lambda t: _is_guard_plugin_target(t, eff_cwd) or _is_guard_state_target(t, payload=safe_payload, cwd=eff_cwd) or _target_is_ancestor_of_protected_guard(t, eff_cwd, safe_payload)
    _flag_prot = lambda cands: (write_targets.append(guard_sentinel), True)[1] if any(c and _is_prot(c) for c in cands) else False

    for sc_idx, sc in enumerate(sub_cmds):
        opens, closes = _unquoted_subshell_opens_closes(sc)
        subshell_stack.extend((eff_cwd, list(dir_stack), cwd_unknown) for _ in range(opens))
        is_piped_in = ((delim_parts[sc_idx][0] if use_delim_map else None) in ("|", "|&")) or pipe_depth > 0
        pipe_depth = max(0, pipe_depth + _unquoted_group_delta(sc)) if is_piped_in else pipe_depth
        wt_start, raw_meta = len(write_targets), _split_shell_tokens_raw_with_meta(sc)
        sc_has_in = any(is_r and "<" in tok for tok, is_r, _ in raw_meta)
        has_in_redir = sc_has_in or stdin_exec
        write_targets.extend(rt for rt in _extract_shell_redir_targets(sc) if rt != "/dev/null")
        tokens, uq_flags = _strip_redir_tokens_with_meta(raw_meta)
        um: Dict[str, bool] = {}
        verb, raw_tok, args, unwrap_unv = _unwrap_shell_command_tokens_with_raw(tokens, uq_flags, out_meta=um)
        stdin_exec, has_unv = stdin_exec or (verb == "exec" and sc_has_in), has_unv or unwrap_unv or bool(um.get("opaque"))

        new_cwd, is_cd, cd_unv = _advance_effective_cwd(sc, eff_cwd, dir_stack, safe_payload)
        if is_cd:
            eff_cwd, cwd_unknown = new_cwd, cwd_unknown or cd_unv
            if _is_guard_plugin_cwd(eff_cwd): has_unv = (write_targets.append(guard_sentinel), True)[1]
        elif not verb: has_unv = has_unv or (bool(sc.strip()) and bool(tokens))
        elif any(ch in raw_tok for ch in "${}*?[]<>()"): has_unv = _flag_prot([raw_tok, *args]) or True
        elif verb in (_SHELL_WRAPPER_VERBS | _UNVERIFIABLE_WRAPPERS | _OPAQUE_EXEC_VERBS):
            if not _is_safe_dev_meta_cmd(verb, raw_tok, args, is_piped_in, has_in_redir): has_unv = _flag_prot(args) or True
        elif Path(raw_tok).name.lower() == "orchestrator_guard.py":
            if not _is_readonly_canonical_guard_invocation(raw_tok, args, eff_cwd, is_piped_in=is_piped_in, has_input_redir=has_in_redir, cwd_unknown=cwd_unknown): has_unv = (write_targets.append(guard_sentinel), True)[1]
        elif spec := _SIMPLE_MUTATOR_SPECS.get(verb):
            pos, _ = _extract_positional_args(args, *spec)
            write_targets.extend(pos)
            if verb in {"rm", "rmdir"} and any(_target_is_ancestor_of_protected_guard(t, eff_cwd, safe_payload) for t in pos): has_unv = (write_targets.append(guard_sentinel), True)[1]
        elif verb in _PERM_MUTATORS:
            has_ref = after_dd = skip_ref = False; pos = []
            if verb == "chmod":
                for tok in args:
                    if skip_ref: has_ref, skip_ref = True, False
                    elif not after_dd and tok == "--": after_dd = True
                    elif not after_dd and (tok == "--reference" or tok.startswith("--reference=")): has_ref, skip_ref = True, tok == "--reference"
                    elif not after_dd and (tok in _CHMOD_LONG_FLAGS or (len(tok) >= 2 and tok[0] == "-" and all(c in "Rfvc" for c in tok[1:]))): continue
                    else: pos.append(tok)
            else:
                has_ref = any(t == "--reference" or t.startswith("--reference=") for t in args)
                pos, _ = _extract_positional_args(args, ("--reference", "--from"), "")
            targets = pos if has_ref else pos[1:]
            write_targets.extend(targets)
            if skip_ref or any(_target_is_ancestor_of_protected_guard(t, eff_cwd, safe_payload) for t in targets): has_unv = (write_targets.append(guard_sentinel), True)[1]
        elif verb in _COPY_MOVE_MUTATORS:
            cm_pairs: List[Tuple[str, str]] = []
            pos, _ = _extract_positional_args(args, ("-t", "--target-directory", "-S", "--suffix", "-m", "--mode", "-o", "--owner", "-g", "--group"), "tSmog", out_pairs=cm_pairs)
            t_dirs = [v for k, v in cm_pairs if k in {"-t", "--target-directory"}]
            has_unv = has_unv or verb == "ln" or (verb == "cp" and any(t in {"-s", "--symbolic-link", "-l", "--link"} or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "sl")) for t in args))
            dests, srcs = (t_dirs, pos) if t_dirs else (([pos[-1]], pos[:-1]) if len(pos) >= 2 else (list(pos), []))
            write_targets.extend(dests + (srcs if verb == "mv" else []))
            is_rec = verb == "cp" and any(t in {"-r", "-R", "-a", "--recursive", "--archive"} or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "rRa")) for t in args)
            if (verb == "mv" or is_rec) and any(_target_is_ancestor_of_protected_guard(t, eff_cwd, safe_payload) for t in srcs + dests): has_unv = (write_targets.append(guard_sentinel), True)[1]
            ws_roots = _allowed_roots(safe_payload) if safe_payload else [PROJECT_ROOT]
            for src in srcs:
                if not _is_within_allowed_roots(_resolve_cmd_target(src, eff_cwd), ws_roots) or _is_prot(src): has_unv = _flag_prot([src]) or True
        elif verb == "dd":
            saw_of = False
            for tok in args:
                if tok.startswith("of="): saw_of = True; write_targets.append(tok[3:])
                elif tok.startswith("if=") and _is_prot(tok[3:]): has_unv = (write_targets.append(guard_sentinel), True)[1]
            has_unv = has_unv or not saw_of
        elif verb == "sed":
            in_place = has_f = saw_ef = after_dd = False; skip_mode, scripts, pos = "", [], []
            for tok in args:
                if skip_mode: (scripts.append(tok) if skip_mode == "e" else None); skip_mode = ""
                elif not after_dd and tok == "--": after_dd = True
                elif not after_dd and tok.startswith("--"):
                    opt, has_eq, val = tok.partition("=")
                    if opt == "--in-place": in_place = True
                    elif opt == "--expression": saw_ef = True; (scripts.append(val) if has_eq else (skip_mode := "e"))
                    elif opt == "--file": saw_ef = has_f = True; skip_mode = "" if has_eq else "f"
                    elif opt == "--line-length" and not has_eq: skip_mode = "l"
                elif not after_dd and tok.startswith("-") and tok != "-":
                    for k, ch in enumerate(tok[1:]):
                        if ch == "i": in_place = True; break
                        if ch in "efl":
                            saw_ef, has_f = saw_ef or ch in "ef", has_f or ch == "f"
                            if rest := tok[1:][k + 1 :]: scripts.append(rest) if ch == "e" else None
                            else: skip_mode = ch
                            break
                else: pos.append(tok)
            files = pos if saw_ef else pos[1:]
            if not saw_ef and pos: scripts.append(pos[0])
            if in_place: has_unv = has_unv or not files; write_targets.extend(files)
            if has_f or skip_mode or any(_sed_script_has_unsafe_cmd(s) for s in scripts): has_unv = _flag_prot(files + scripts) or True
        elif verb == "sort":
            sort_pairs: List[Tuple[str, str]] = []
            _extract_positional_args(args, ("-o", "--output", "-k", "--key", "-t", "--field-separator", "-T", "--temporary-directory", "-S", "--buffer-size", "--compress-program", "--random-source", "--batch-size", "--parallel", "--files0-from"), "oktTS", out_pairs=sort_pairs)
            write_targets.extend(v for k, v in sort_pairs if k in {"-o", "--output"})
            pre_dd = args[: (args.index("--") if "--" in args else len(args))]
            if any(k == "--compress-program" for k, _ in sort_pairs) or any(t == "--compress-program" or t.startswith("--compress-program=") for t in pre_dd): has_unv = True
        elif verb in {"printf", "rg"}:
            pre = args[: (args.index("--") if "--" in args else len(args))]
            has_unv = has_unv or (verb == "printf" and any(t == "-v" or (t.startswith("-") and not t.startswith("--") and "v" in t[1:]) or "{" in t or "}" in t for t in pre)) or (verb == "rg" and any(t == "--pre" or t.startswith("--pre=") for t in pre))
        elif verb == "git":
            if has_in_redir or not _is_safe_readonly_git_tokens(args, eff_cwd, safe_payload): has_unv = _flag_prot(args) or True
        elif _PY_VERB_RE.match(verb):
            _, mode, target, mod_args = _parse_python_cli(args)
            if mode == "meta" and not is_piped_in and not has_in_redir and raw_tok.lower() in {verb, f"/usr/bin/{verb}", f"/bin/{verb}"}: pass
            elif mode == "m" and target in {"pytest", "py.test", "unittest"}:
                if target != "unittest":
                    pt_tgts, pt_mut = _extract_pytest_target_check(mod_args, eff_cwd, safe_payload)
                    write_targets.extend(pt_tgts); has_unv = has_unv or pt_mut
                has_unv = has_unv or cwd_unknown or _is_guard_plugin_cwd(eff_cwd) or _is_staging_or_tmp_path(_resolve_cmd_target(".", eff_cwd), safe_payload) or not _is_test_command(sc, cur_dir=eff_cwd, payload=safe_payload) or not _check_test_runner_args(verb, args, eff_cwd, safe_payload)
            elif mode == "script" and Path(target).name.lower() == "orchestrator_guard.py":
                if not _is_readonly_canonical_guard_invocation(target, mod_args, eff_cwd, is_piped_in=is_piped_in, has_input_redir=has_in_redir, cwd_unknown=cwd_unknown): has_unv = (write_targets.append(guard_sentinel), True)[1]
            else: has_unv = (_flag_prot(args) if mode != "c" else False) or True
        else:
            bad_cwd = cwd_unknown or _is_guard_plugin_cwd(eff_cwd) or _is_staging_or_tmp_path(_resolve_cmd_target(".", eff_cwd), safe_payload)
            if not bad_cwd and _is_safe_dev_meta_cmd(verb, raw_tok, args, is_piped_in, has_in_redir): pass
            elif verb in {"pytest", "py.test"} or _is_test_command(sc, cur_dir=eff_cwd, payload=safe_payload):
                if verb in {"pytest", "py.test"}:
                    pt_tgts, pt_mut = _extract_pytest_target_check(args, eff_cwd, safe_payload)
                    write_targets.extend(pt_tgts); has_unv = has_unv or bad_cwd or pt_mut or not _is_test_command(sc, cur_dir=eff_cwd, payload=safe_payload)
                else: has_unv = has_unv or bad_cwd or not _check_test_runner_args(verb, args, eff_cwd, safe_payload)
            elif verb in _LINTER_VERBS: has_unv = has_unv or not _is_safe_linter_subcmd(verb, args, eff_cwd, safe_payload, is_piped_in=is_piped_in, has_input_redir=has_in_redir, cwd_unknown=cwd_unknown)
            elif verb in SAFE_READONLY_SHELL_VERBS: pass
            elif verb == "find":
                in_exec, ex_t, bad = False, [], False
                for ft in args:
                    if in_exec and ft in {";", "+"}:
                        iv, _, iunv = _unwrap_shell_command_tokens(ex_t)
                        bad, in_exec, ex_t = (bad or iunv or not iv or iv in _SHELL_WRAPPER_VERBS or iv not in SAFE_READONLY_SHELL_VERBS), False, []
                    elif in_exec: ex_t.append(ft)
                    elif ft in {"-delete", "-fls", "-execdir", "-ok", "-okdir"} or ft.startswith("-fprint"): bad = True
                    elif ft == "-exec": in_exec, ex_t = True, []
                if bad or in_exec: has_unv = _flag_prot(args) or True
            elif verb == "xargs":
                pos, opt_vals = _extract_positional_args(args, ("-a", "--arg-file", "-I", "-L", "-n", "-P", "-d", "-E", "-s"), "aILnPdEs")
                if (Path(pos[0]).name.lower() if pos else "") not in (SAFE_XARGS_READONLY_CMDS - {"rg", "printf"}): has_unv = _flag_prot(pos + opt_vals) or True
            else: has_unv = _flag_prot(args) or True
        if len(write_targets) > wt_start:
            if cwd_unknown: has_unv = True
            elif eff_cwd != initial_cwd: write_targets[wt_start:] = [_resolve_cmd_target(w, eff_cwd) for w in write_targets[wt_start:]]
        for _ in range(min(closes, len(subshell_stack))): eff_cwd, dir_stack, cwd_unknown = subshell_stack.pop()

    return write_targets, has_unv or any(_has_shell_path_expansion(wt) for wt in write_targets)


def _is_mutating_or_wrapper_shell_subcmd(sc: str, cwd: str = "", payload: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if a shell subcommand uses a mutating verb or shell/exec wrapper (beyond pure redirs on readonly verbs)."""
    wts, unv = _extract_shell_write_targets(sc, [sc], cwd=cwd, payload=payload)
    if unv: return True
    if not wts: return False
    toks, uq = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(sc))
    return _unwrap_shell_command_tokens(toks, uq)[0] not in SAFE_READONLY_SHELL_VERBS


def _validate_shell_allowlist(cmd: str, sub_cmds: List[str], cwd: str, allowed_files: List[str], payload: Dict[str, Any], allowlist_label: str = "ALLOWED_FILES") -> Optional[Dict[str, Any]]:
    """Validate shell write targets and mutating/wrapper commands against an allowlist."""
    write_targets, has_unverifiable_mut = _extract_shell_write_targets(cmd, sub_cmds, cwd=cwd, payload=payload)
    if has_unverifiable_mut:
        return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: mutating or nested shell command '{cmd.strip()[:60]}' cannot be verified against {allowlist_label} ({', '.join(allowed_files) if allowed_files else 'NONE'})."}
    for wt in write_targets:
        resolved_wt = _resolve_cmd_target(wt, cwd)
        if not _is_artifact_path(resolved_wt, payload) and not _is_tmp_path(resolved_wt) and not _is_target_in_allowed_files(resolved_wt, allowed_files, payload.get("workspacePaths")):
            return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: '{wt}' is outside {allowlist_label} ({', '.join(allowed_files) if allowed_files else 'NONE'})."}
    return None

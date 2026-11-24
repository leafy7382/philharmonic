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
    FlagSpec, ParsedCommand, ParsedSubcmd, SAFE_READONLY_SHELL_VERBS, SAFE_XARGS_READONLY_CMDS,
    _OPAQUE_EXEC_VERBS, _PY_VERB_RE, _SHELL_WRAPPER_VERBS, _UNVERIFIABLE_WRAPPERS,
    _advance_effective_cwd, _is_canonical_cmd_path, _parse_python_cli, _payload_has_raw_cr, _unwrap_shell_command_tokens_with_raw, parse_shell_command, walk_cli_flags,
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
_BAD_LINTER_OPTS = frozenset({"--init", "--load-plugins", "--init-hook", "--rcfile", "--output", "-o", "--output-file", "--eval-args", "--fix", "--fix-only", "--unsafe-fixes", "--add-noqa", "--write", "-i", "--in-place", "--sqlite-cache", "--install-types", "--cache-dir", "--shadow-file", "--config-file", "--config", "-c", "--settings-path", "--settings-file", "--python-executable", "--custom-typeshed", "--custom-typeshed-dir", "--junit-xml", "--reporter"})
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
_GIT_LIST_REQ_VAL_OPTS = frozenset({"--sort", "--format"})
_GIT_LIST_OPT_VAL_OPTS = frozenset({"--contains", "--no-contains", "--merged", "--no-merged", "--points-at"})
_GIT_LIST_SUBCMD_SPECS: Dict[str, Tuple[frozenset[str], str, str, bool]] = {
    "branch": (frozenset({"-a", "--all", "-r", "--remotes", "-v", "--verbose", "-vv", "-l", "--list", "--show-current", "--color", "--no-color", "--column", "--no-column", "--abbrev", "--no-abbrev"}), "l", "dDmMcCftu", False),
    "tag": (frozenset({"-l", "--list", "-v", "--verify", "--color", "--no-color", "-i", "--ignore-case"}), "lv", "dasufmF", True),
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
    res = walk_cli_flags(args, FlagSpec(val_opts=frozenset(val_flags), short_val_chars=short_val_chars, gnu_abbrev=True))
    if out_pairs is not None: out_pairs.extend(res.opt_pairs)
    return list(res.positionals), [v for _, v in res.opt_pairs]


def _parse_git_tokens(arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any], *, require_allowed_roots: bool = False, allow_git_dir: bool = True) -> Optional[Tuple[str, List[str], str]]:
    """Parse git global options and return (subcmd, sub_args, eff_dir) or None if unsafe."""
    subcmd, sub_args, eff_dir, skip_opt = "", [], cur_dir, ""
    _bad_opt = lambda t: (
        t in _BAD_GIT_OPTS or t.startswith(_BAD_GIT_PREFIXES) or (t.startswith("-c") and not t.startswith("--") and len(t) > 2)
        or (t.startswith("--") and len(opt := t.partition("=")[0]) >= 4 and opt not in (_GIT_GLOBAL_BOOL_OPTS | _GIT_GLOBAL_VAL_OPTS) and any(b.startswith(opt) for b in _BAD_GIT_OPTS if b.startswith("--")))
    )

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
        bool_f, allow_short, bad_short, allow_dd = spec
        skip_req = skip_opt = saw_list = False
        eq_prefixes = tuple(f + "=" for f in (_GIT_LIST_REQ_VAL_OPTS | _GIT_LIST_OPT_VAL_OPTS)) + ("--color=", "--abbrev=")
        for t in sub_args:
            if skip_req: skip_req = False; continue
            if skip_opt and not t.startswith("-"): skip_opt = False; continue
            skip_opt = False
            if t == "--": return allow_dd and saw_list
            elif t in bool_f: saw_list = saw_list or t in {"-l", "--list", "-v", "--verify"}
            elif t in _GIT_LIST_REQ_VAL_OPTS: skip_req = True
            elif t in _GIT_LIST_OPT_VAL_OPTS: skip_opt = True
            elif t.startswith(eq_prefixes) or (subcmd == "tag" and (t == "-n" or (t.startswith("-n") and t[2:].isdigit()))): continue
            elif t.startswith("-") and not t.startswith("--") and not any(c in t[1:] for c in bad_short) and all(c in (allow_short + ("arvi" if subcmd == "branch" else "i")) for c in t[1:]):
                saw_list = saw_list or any(c in t[1:] for c in allow_short)
            elif not (saw_list and not t.startswith("-")): return False
        return not skip_req
    if subcmd == "remote": return not sub_args or all(t in {"-v", "--verbose"} for t in sub_args) or sub_args[0] in {"show", "get-url"}
    if subcmd == "reflog": return not sub_args or sub_args[0] in {"show", "exists"} or (sub_args[0].startswith("-") and not any(x in sub_args for x in ("expire", "delete", "drop")))
    return subcmd == "fsck" and not walk_cli_flags(sub_args, FlagSpec(bad_opts=frozenset({"--lost-found"}), stop_at_dd=False, gnu_abbrev=True)).bad_hit


def _is_safe_linter_subcmd(verb: str, arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any], *, is_piped_in: bool, has_input_redir: bool, cwd_unknown: bool) -> bool:
    """Validate read-only linter/typechecker invocations."""
    if cwd_unknown or is_piped_in or has_input_redir or _is_guard_plugin_cwd(cur_dir) or _is_staging_or_tmp_path(_resolve_cmd_target(".", cur_dir), payload): return False
    if walk_cli_flags(arg_tokens, FlagSpec(bad_opts=_BAD_LINTER_OPTS, bad_short_chars="oic", stop_at_dd=False, gnu_abbrev=True)).bad_hit: return False
    if verb == "mypy" and any(t.startswith("--") and t.partition("=")[0].endswith("-report") for t in arg_tokens): return False
    if verb == "pylint" and (any(":" in v for _, v in walk_cli_flags(arg_tokens, FlagSpec(val_opts=frozenset({"-f", "--output-format"}), short_val_chars="f", stop_at_dd=False, gnu_abbrev=True)).opt_pairs) or any(t.startswith("-") and ":" in t for t in arg_tokens)): return False
    pre_dd = arg_tokens[: arg_tokens.index("--")] if "--" in arg_tokens else arg_tokens
    _has_standalone = lambda allowed: any(t in allowed and (idx == 0 or not (pre_dd[idx - 1].startswith("-") and pre_dd[idx - 1] != "-" and "=" not in pre_dd[idx - 1])) for idx, t in enumerate(pre_dd))
    if (verb == "playwright" and any(t in {"code", "codegen", "create", "install"} for t in arg_tokens)) or (verb == "tsc" and (not _has_standalone(frozenset({"--noEmit"})) or any(t.lower() == "false" for t in arg_tokens))): return False
    if verb == "black" and not _has_standalone(frozenset({"--check", "--diff"})): return False
    if verb == "isort" and not _has_standalone(frozenset({"--check", "--diff", "--check-only"})): return False
    if verb == "ruff" and "format" in arg_tokens and not _has_standalone(frozenset({"--check", "--diff"})): return False
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
                    if script[j] == "\n": return -1
                    if script[j] == "[" and j + 1 < n and script[j + 1] in ":.=":
                        end = script.find(script[j + 1] + "]", j + 2)
                        if end < 0 or "\n" in script[j:end]: return -1
                        j = end + 2
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
            if "\\" in script[i:] and "\n" in script[i:]: return True
            while i < n and script[i] != "\n": i += 1
        elif cmd in (":", "b", "t", "T", "r", "R"):
            while i < n and script[i] not in ";\n}": i += 1
    return False


def _extract_shell_write_targets(cmd: str, sub_cmds: List[str], cwd: str = "", payload: Optional[Dict[str, Any]] = None, *, check_unclosed: bool = True, parsed: Optional[ParsedCommand] = None) -> Tuple[List[str], bool]:
    """Extract shell write targets and detect unverifiable mutating or nested shell execution commands."""
    guard_sentinel, safe_payload = _guard_state_sentinel(), (payload if isinstance(payload, dict) else {})
    p = parsed if parsed is not None else parse_shell_command(cmd)
    eff_p = p if (p.cleaned or not sub_cmds) else parse_shell_command(" ; ".join(sub_cmds))
    write_targets: List[str] = list(p.redir_targets) if not sub_cmds else []
    has_unv = (
        p.has_nul_surr or eff_p.has_nul_surr or eff_p.has_ansi_c or eff_p.has_subshell or _payload_has_raw_cr(cmd, safe_payload)
        or p.has_heredoc_subst or eff_p.has_heredoc_subst or (check_unclosed and (p.has_unclosed or eff_p.has_unclosed))
        or p.has_cmd_subst or eff_p.has_cmd_subst or "<(" in eff_p.unquoted or ">(" in eff_p.unquoted
        or bool(re.search(r"(?:^|[;&|\n({]\s*)(?:(?:if|then|elif|else|do|while|until|coproc|time|!)\s+)*(?:function\b|case\b|coproc\b|[^\s;|&()<>'\"`]+?\s*\(\s*\))", eff_p.unquoted))
    )
    initial_cwd = cwd or str(PROJECT_ROOT)
    eff_cwd, cwd_unknown, stdin_exec = initial_cwd, False, False
    use_pipe_meta = not sub_cmds or [s.raw for s in eff_p.subcmds] == list(sub_cmds)
    iter_subs = () if not sub_cmds else (eff_p.subcmds if use_pipe_meta else tuple(s for sc in sub_cmds for s in parse_shell_command(sc).subcmds))
    _is_prot = lambda t: _is_guard_plugin_target(t, eff_cwd) or _is_guard_state_target(t, payload=safe_payload, cwd=eff_cwd) or _target_is_ancestor_of_protected_guard(t, eff_cwd, safe_payload)
    _flag_prot = lambda cands: (write_targets.append(guard_sentinel), True)[1] if any(c and _is_prot(c) for c in cands) else False

    for sub in iter_subs:
        sc = sub.raw
        is_piped_in, wt_start, sc_has_in = (sub.is_piped_in if use_pipe_meta else False), len(write_targets), sub.has_in_redir
        has_in_redir = sc_has_in or stdin_exec
        write_targets.extend(sub.redir_targets)
        verb, raw_tok, args = sub.verb, sub.raw_verb, list(sub.args)
        stdin_exec, has_unv = stdin_exec or (verb == "exec" and sc_has_in), has_unv or sub.unwrap_unv or sub.opaque or bool(sub.subshell_opens or sub.subshell_closes)
        if verb and Path(raw_tok).name.lower() != "orchestrator_guard.py" and not _is_canonical_cmd_path(verb, raw_tok):
            has_unv = _flag_prot([raw_tok, *args]) or True

        new_cwd, is_cd, cd_unv = _advance_effective_cwd(sc, eff_cwd, None, safe_payload, parsed_sub=sub)
        if is_cd:
            eff_cwd, cwd_unknown = new_cwd, cwd_unknown or cd_unv
            if verb in {"pushd", "popd"}: has_unv = True
            if _is_guard_plugin_cwd(eff_cwd): has_unv = (write_targets.append(guard_sentinel), True)[1]
        elif not verb: has_unv = has_unv or (bool(sc.strip()) and bool(sub.tokens))
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
                    elif not after_dd and tok.startswith("--"):
                        opt, has_eq, _ = tok.partition("=")
                        if len(opt) >= 3 and "--reference".startswith(opt) and (len(opt) >= 5 or has_eq): has_ref, skip_ref = True, not has_eq
                        elif not (opt in _CHMOD_LONG_FLAGS or (len(opt) >= 3 and any(f.startswith(opt) for f in _CHMOD_LONG_FLAGS))): pos.append(tok)
                    elif not after_dd and len(tok) >= 2 and tok[0] == "-" and all(c in "Rfvc" for c in tok[1:]): continue
                    else: pos.append(tok)
            else:
                pw = walk_cli_flags(args, FlagSpec(val_opts=frozenset({"--reference", "--from"}), gnu_abbrev=True))
                has_ref, pos, skip_ref = ("--reference" in pw.flags_seen or any(k == "--reference" for k, _ in pw.opt_pairs)), list(pw.positionals), pw.missing_value
            targets = pos if has_ref else pos[1:]
            write_targets.extend(targets)
            if skip_ref or any(_target_is_ancestor_of_protected_guard(t, eff_cwd, safe_payload) for t in targets): has_unv = (write_targets.append(guard_sentinel), True)[1]
        elif verb in _COPY_MOVE_MUTATORS:
            cm_bad = frozenset({"-s", "--symbolic-link", "-l", "--link"}) if verb == "cp" else (frozenset({"--strip-program"}) if verb == "install" else frozenset())
            cm_w = walk_cli_flags(args, FlagSpec(val_opts=frozenset({"-t", "--target-directory", "-S", "--suffix", "-m", "--mode", "-o", "--owner", "-g", "--group"}), short_val_chars="tSmog", bad_opts=cm_bad, bad_short_chars=("sl" if verb == "cp" else ""), gnu_abbrev=True))
            pos, t_dirs = list(cm_w.positionals), [v for k, v in cm_w.opt_pairs if k in {"-t", "--target-directory"}]
            has_unv = has_unv or verb == "ln" or cm_w.bad_hit
            is_inst_dir = verb == "install" and (bool({"-d", "--directory"} & set(cm_w.flags_seen)) or any(f.startswith("--") and len(f) >= 3 and "--directory".startswith(f) for f in cm_w.flags_seen))
            dests, srcs = (list(pos), []) if is_inst_dir else ((t_dirs, pos) if t_dirs else (([pos[-1]], pos[:-1]) if len(pos) >= 2 else (list(pos), [])))
            write_targets.extend(dests + (srcs if verb == "mv" else []))
            is_rec = verb == "cp" and (bool({"-r", "-R", "-a", "--recursive", "--archive"} & set(cm_w.flags_seen)) or any(f.startswith("--") and len(f) >= 3 and ("--recursive".startswith(f) or "--archive".startswith(f)) for f in cm_w.flags_seen))
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
                    if len(opt) >= 3 and "--in-place".startswith(opt): in_place = True
                    elif len(opt) >= 3 and "--expression".startswith(opt): saw_ef = True; (scripts.append(val) if has_eq else (skip_mode := "e"))
                    elif len(opt) >= 4 and "--file".startswith(opt): saw_ef = has_f = True; skip_mode = "" if has_eq else "f"
                    elif len(opt) >= 3 and "--line-length".startswith(opt) and not has_eq: skip_mode = "l"
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
            sw = walk_cli_flags(args, FlagSpec(val_opts=frozenset({"-o", "--output", "-k", "--key", "-t", "--field-separator", "-T", "--temporary-directory", "-S", "--buffer-size", "--compress-program", "--random-source", "--batch-size", "--parallel", "--files0-from"}), short_val_chars="oktTS", bad_opts=frozenset({"--compress-program"}), stop_at_dd=False, gnu_abbrev=True))
            write_targets.extend(v for k, v in sw.opt_pairs if k in {"-o", "--output"})
            has_unv = has_unv or sw.bad_hit
        elif verb in {"printf", "rg"}:
            has_unv = has_unv or walk_cli_flags(args, FlagSpec(bad_opts=frozenset({"-v" if verb == "printf" else "--pre"}), bad_short_chars=("v" if verb == "printf" else ""), stop_at_dd=False, gnu_abbrev=True)).bad_hit or (verb == "printf" and any("{" in t or "}" in t for t in args))
        elif verb == "git":
            if has_in_redir or not _is_safe_readonly_git_tokens(args, eff_cwd, safe_payload): has_unv = _flag_prot(args) or True
        elif _PY_VERB_RE.match(verb):
            _, mode, target, mod_args = _parse_python_cli(args)
            if mode == "meta" and not is_piped_in and not has_in_redir: pass
            elif mode == "m" and target in {"pytest", "py.test", "unittest"}:
                if target != "unittest":
                    pt_tgts, pt_mut = _extract_pytest_target_check(mod_args, eff_cwd, safe_payload)
                    write_targets.extend(pt_tgts); has_unv = has_unv or pt_mut
                has_unv = has_unv or cwd_unknown or _is_guard_plugin_cwd(eff_cwd) or _is_staging_or_tmp_path(_resolve_cmd_target(".", eff_cwd), safe_payload) or not _is_test_command(sc, cur_dir=eff_cwd, payload=safe_payload, parsed_sub=sub) or not _check_test_runner_args(verb, args, eff_cwd, safe_payload)
            elif mode == "script" and Path(target).name.lower() == "orchestrator_guard.py":
                if not _is_readonly_canonical_guard_invocation(target, mod_args, eff_cwd, is_piped_in=is_piped_in, has_input_redir=has_in_redir, cwd_unknown=cwd_unknown): has_unv = (write_targets.append(guard_sentinel), True)[1]
            else: has_unv = (_flag_prot(args) if mode != "c" else False) or True
        else:
            bad_cwd = cwd_unknown or _is_guard_plugin_cwd(eff_cwd) or _is_staging_or_tmp_path(_resolve_cmd_target(".", eff_cwd), safe_payload)
            if not bad_cwd and _is_safe_dev_meta_cmd(verb, raw_tok, args, is_piped_in, has_in_redir): pass
            elif verb in {"pytest", "py.test"} or _is_test_command(sc, cur_dir=eff_cwd, payload=safe_payload, parsed_sub=sub):
                if verb in {"pytest", "py.test"}:
                    pt_tgts, pt_mut = _extract_pytest_target_check(args, eff_cwd, safe_payload)
                    write_targets.extend(pt_tgts); has_unv = has_unv or bad_cwd or pt_mut or not _is_test_command(sc, cur_dir=eff_cwd, payload=safe_payload, parsed_sub=sub)
                else: has_unv = has_unv or bad_cwd or not _check_test_runner_args(verb, args, eff_cwd, safe_payload)
            elif verb in _LINTER_VERBS: has_unv = has_unv or not _is_safe_linter_subcmd(verb, args, eff_cwd, safe_payload, is_piped_in=is_piped_in, has_input_redir=has_in_redir, cwd_unknown=cwd_unknown)
            elif verb in SAFE_READONLY_SHELL_VERBS: pass
            elif verb == "find":
                in_exec, ex_t, bad = False, [], False
                for ft in args:
                    if in_exec and ft in {";", "+"}:
                        iv, iraw, _, iunv = _unwrap_shell_command_tokens_with_raw(ex_t)
                        bad, in_exec, ex_t = (bad or iunv or not iv or not _is_canonical_cmd_path(iv, iraw) or iv in _SHELL_WRAPPER_VERBS or iv not in SAFE_READONLY_SHELL_VERBS), False, []
                    elif in_exec: ex_t.append(ft)
                    elif ft in {"-delete", "-fls", "-execdir", "-ok", "-okdir"} or ft.startswith("-fprint"): bad = True
                    elif ft == "-exec": in_exec, ex_t = True, []
                if bad or in_exec: has_unv = _flag_prot(args) or True
            elif verb == "xargs":
                xw = walk_cli_flags(args, FlagSpec(val_opts=frozenset({"-a", "--arg-file", "-I", "-L", "-n", "--max-args", "-P", "--max-procs", "-d", "--delimiter", "-E", "-s", "--max-chars", "--process-slot-var"}), short_val_chars="aILnPdEs", stop_at_first_pos=True, gnu_abbrev=True))
                rem_x = args[xw.rest_index :]
                xv = Path(rem_x[0]).name.lower() if rem_x else ""
                if not rem_x or not _is_canonical_cmd_path(xv, rem_x[0]) or xv not in (SAFE_XARGS_READONLY_CMDS - {"rg", "printf"}): has_unv = _flag_prot(rem_x + [v for _, v in xw.opt_pairs]) or True
            else: has_unv = _flag_prot(args) or True
        if len(write_targets) > wt_start:
            if cwd_unknown: has_unv = True
            elif eff_cwd != initial_cwd: write_targets[wt_start:] = [_resolve_cmd_target(w, eff_cwd) for w in write_targets[wt_start:]]

    return write_targets, has_unv or any(_has_shell_path_expansion(wt) for wt in write_targets)


def _is_mutating_or_wrapper_shell_subcmd(sc: str, cwd: str = "", payload: Optional[Dict[str, Any]] = None, *, parsed_sub: Optional[ParsedSubcmd] = None) -> bool:
    """Return True if a shell subcommand uses a mutating verb or shell/exec wrapper (beyond pure redirs on readonly verbs)."""
    if parsed_sub is not None:
        p = ParsedCommand(parsed_sub.raw, parsed_sub.raw, parsed_sub.unquoted, False, False, bool(parsed_sub.subshell_opens or parsed_sub.subshell_closes), False, False, False, False, False, parsed_sub.redir_targets, (parsed_sub,))
        wts, unv = _extract_shell_write_targets(parsed_sub.raw, [parsed_sub.raw], cwd=cwd, payload=payload, parsed=p)
        return True if unv else (bool(wts) and parsed_sub.verb not in SAFE_READONLY_SHELL_VERBS)
    p = parse_shell_command(sc)
    wts, unv = _extract_shell_write_targets(sc, [s.raw for s in p.subcmds] or [sc], cwd=cwd, payload=payload, parsed=p)
    return True if unv else (bool(wts) and (not p.subcmds or p.subcmds[0].verb not in SAFE_READONLY_SHELL_VERBS))


def _validate_shell_allowlist(cmd: str, sub_cmds: List[str], cwd: str, allowed_files: List[str], payload: Dict[str, Any], allowlist_label: str = "ALLOWED_FILES", *, parsed: Optional[ParsedCommand] = None) -> Optional[Dict[str, Any]]:
    """Validate shell write targets and mutating/wrapper commands against an allowlist."""
    write_targets, has_unverifiable_mut = _extract_shell_write_targets(cmd, sub_cmds, cwd=cwd, payload=payload, parsed=parsed)
    if has_unverifiable_mut:
        return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: mutating or nested shell command '{cmd.strip()[:60]}' cannot be verified against {allowlist_label} ({', '.join(allowed_files) if allowed_files else 'NONE'})."}
    for wt in write_targets:
        resolved_wt = _resolve_cmd_target(wt, cwd)
        if not _is_artifact_path(resolved_wt, payload) and not _is_tmp_path(resolved_wt) and not _is_target_in_allowed_files(resolved_wt, allowed_files, payload.get("workspacePaths")):
            return {"decision": "deny", "reason": f"[ORCHESTRATOR GUARD] File allowlist violation in run_command: '{wt}' is outside {allowlist_label} ({', '.join(allowed_files) if allowed_files else 'NONE'})."}
    return None

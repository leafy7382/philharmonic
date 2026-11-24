"""Shared test-runner argument validation and test command recognition."""
from __future__ import annotations

import posixpath
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import TRIVIAL_TEST_FLAG_RE
from guard.path_guards import _is_staging_or_tmp_path, _resolve_cmd_target
from guard.shell_tokens import (
    FlagSpec, ParsedCommand, ParsedSubcmd, _OPAQUE_EXEC_VERBS, _PY_VERB_RE, _SHELL_WRAPPER_VERBS,
    _fold_shell_line_continuations, _has_unquoted_cmd_subst, _is_canonical_cmd_path, _is_dangerous_env_assignment,
    _parse_python_cli, _payload_has_raw_cr, parse_shell_command, walk_cli_flags,
)

_PYTEST_PATH_OUTPUT_FLAGS = frozenset({"--junitxml", "--junit-xml", "--basetemp", "--resultlog", "--result-log", "--html", "--cov-report", "--log-file"})
_PYTEST_STAGING_VAL_FLAGS = frozenset({"--rootdir", "--confcutdir", "--basetemp", "--junitxml", "--junit-xml", "-c"})
_PYTEST_OTHER_MUT_FLAGS = frozenset({"-o", "--override-ini", "-c", "-p", "--config-file", "--rootdir", "--confcutdir"})
_PYTEST_SKIP_VAL_FLAGS = frozenset({
    "-k", "-m", "-p", "-o", "--override-ini", "-W", "--maxfail", "--tb", "--capture", "--show-capture",
    "--durations", "--durations-min", "--log-level", "--log-format", "--log-date-format", "--import-mode",
    "--color", "--code-highlight", "--deselect", "--ignore", "--ignore-glob", "--resultlog", "--result-log",
    "--html", "--cov-report", "--log-file",
})
_JEST_BAD = frozenset({
    "-u", "--updateSnapshot", "--update-snapshot", "--init", "--clearCache", "--clear-cache", "--passWithNoTests",
    "--pass-with-no-tests", "--coverage", "--coverageDirectory", "--reporter", "--reporters", "-R", "--config",
    "-c", "--setupFiles", "--setupFilesAfterEnv", "--testEnvironment", "--resolver", "--moduleNameMapper",
    "--transform", "--globalSetup", "--globalTeardown",
})
_NPM_BAD = frozenset({"--", "-c", "-r", "--config", "--userconfig", "--globalconfig", "--require", "--prefix", "-C"})
_GO_BAD = frozenset({
    "-c", "--c", "-C", "--C", "-n", "--n", "-exec", "--exec", "-list", "--list", "-toolexec", "--toolexec",
    "-overlay", "--overlay", "-modfile", "--modfile", "-coverprofile", "--coverprofile", "-cpuprofile",
    "--cpuprofile", "-memprofile", "--memprofile", "-blockprofile", "--blockprofile", "-mutexprofile",
    "--mutexprofile", "-trace", "--trace", "-outputdir", "--outputdir", "-o", "--o",
})
# Shared test-runner rules: (req_subcmd, bad_opts, bad_short_chars, bad_words, stop_at_dd, check_staging)
_TEST_RUNNER_RULES: Dict[str, Tuple[Optional[str], frozenset[str], str, frozenset[str], bool, bool]] = {
    "jest": (None, _JEST_BAD, "vucR", frozenset(), True, True),
    "vitest": (None, _JEST_BAD, "vucR", frozenset({"init", "list", "bench"}), True, True),
    "npm": ("test", _NPM_BAD, "crC", frozenset(), False, True),
    "pnpm": ("test", _NPM_BAD, "crC", frozenset(), False, True),
    "yarn": ("test", _NPM_BAD, "crC", frozenset(), False, True),
    "tox": (None, frozenset({"-", "--", "-c", "--conf", "--root", "--installpkg", "--force-dep"}), "c", frozenset(), False, False),
    "nox": (None, frozenset({"-", "--", "-f", "--noxfile", "--installpkg", "--force-dep"}), "f", frozenset(), False, False),
    "go": ("test", _GO_BAD, "", frozenset(), True, True),
    "cargo": ("test", frozenset({"--target-dir", "--manifest-path", "--config", "--list"}), "", frozenset(), True, False),
}


_PYTEST_FLAG_SPEC = FlagSpec(
    val_opts=_PYTEST_PATH_OUTPUT_FLAGS | _PYTEST_STAGING_VAL_FLAGS | _PYTEST_OTHER_MUT_FLAGS | _PYTEST_SKIP_VAL_FLAGS,
    short_val_chars="ocpkmW", bad_opts=_PYTEST_PATH_OUTPUT_FLAGS | _PYTEST_OTHER_MUT_FLAGS, bad_short_chars="ocp", stop_at_dd=False, gnu_abbrev=True,
)
_UNITTEST_FLAG_SPEC = FlagSpec(
    val_opts=frozenset({"-s", "--start-directory", "-t", "--top-level-directory", "-p", "--pattern", "-k"}),
    short_val_chars="stpk", stop_at_dd=False, gnu_abbrev=True,
)


def _is_staging_test_path(val: str, cur_dir: str, safe_pl: Dict[str, Any]) -> bool:
    if not val: return False
    res = _resolve_cmd_target(val, cur_dir)
    if _is_staging_or_tmp_path(res, safe_pl): return True
    norm = posixpath.normpath(res.replace("\\", "/"))
    if norm in {"/tmp", "/dev/shm"} or norm.startswith(("/tmp/", "/private/tmp/", "/var/tmp/", "/dev/shm/")): return True
    parts = norm.split("/")
    return (val.replace("\\", "/").startswith("/") and any(any(c in p for c in "*?[") for p in parts[1:3])) or any(c in (parts[1] if len(parts) > 1 else "") for c in "*?[")


def _check_unittest_args(u_args: List[str], cur_dir: str, safe_pl: Dict[str, Any]) -> bool:
    uw = walk_cli_flags(u_args, _UNITTEST_FLAG_SPEC)
    cands: List[str] = [v for _, v in uw.opt_pairs] + [p for p in uw.positionals if p != "discover"]
    for tok in u_args:
        if tok in {"discover", "--", "-v", "--verbose"}: continue
        if "=" in tok: cands.append(tok.split("=", 1)[1])
        elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 2 and tok[1] in "stp": cands.append(tok[2:])
    return not any(_is_staging_test_path(c, cur_dir, safe_pl) for c in cands)


def _extract_pytest_target_check(pytest_args: List[str], cur_dir: str = "", payload: Optional[Dict[str, Any]] = None) -> Tuple[List[str], bool]:
    """Check pytest args for mutating/config/staging flags and extract any output paths."""
    safe_pl = payload if isinstance(payload, dict) else {}
    w = walk_cli_flags(pytest_args, _PYTEST_FLAG_SPEC)
    targets = [out_v for opt, val in w.opt_pairs if opt in _PYTEST_PATH_OUTPUT_FLAGS and (out_v := (val.split(":", 1)[1] if opt == "--cov-report" and ":" in val else val))]
    mut = w.bad_hit or any(opt in _PYTEST_STAGING_VAL_FLAGS and _is_staging_test_path(val, cur_dir, safe_pl) for opt, val in w.opt_pairs) or any(_is_staging_test_path(tok.split("::", 1)[0], cur_dir, safe_pl) for tok in w.positionals)
    return targets, mut


def _check_test_runner_args(verb: str, arg_tokens: List[str], cur_dir: str = "", payload: Optional[Dict[str, Any]] = None) -> bool:
    """Validate test runner flags against mutation/arbitrary-exec/staging flags."""
    safe_pl = payload if isinstance(payload, dict) else {}
    if verb in {"pytest", "py.test"}:
        return not _extract_pytest_target_check(arg_tokens, cur_dir, safe_pl)[1]
    if verb == "unittest":
        return _check_unittest_args(arg_tokens, cur_dir, safe_pl)
    if _PY_VERB_RE.match(verb):
        ok_py, mode, mod, mod_args = _parse_python_cli(arg_tokens)
        if not ok_py or mode != "m":
            return False
        if mod in {"pytest", "py.test"}:
            return not _extract_pytest_target_check(mod_args, cur_dir, safe_pl)[1]
        return mod == "unittest" and _check_unittest_args(mod_args, cur_dir, safe_pl)
    if (rule := _TEST_RUNNER_RULES.get(verb)) is None:
        return True
    req_sub, bad_opts, bad_short, bad_words, _, chk_staging = rule
    if req_sub is not None:
        if not arg_tokens or arg_tokens[0] != req_sub:
            return False
        arg_tokens = arg_tokens[1:]
    if walk_cli_flags(arg_tokens, FlagSpec(bad_opts=bad_opts, bad_short_chars=bad_short, stop_at_dd=False, gnu_abbrev=True)).bad_hit:
        return False
    if any(t in bad_words for t in arg_tokens):
        return False
    if chk_staging and any((val := (t.split("=", 1)[1] if "=" in t else ("" if t.startswith("-") else t))) and _is_staging_test_path(val, cur_dir, safe_pl) for t in arg_tokens if t != "--"):
        return False
    return True


def _is_test_command(
    cmd: str, *, cur_dir: str = "", payload: Optional[Dict[str, Any]] = None,
    parsed: Optional[ParsedCommand] = None, parsed_sub: Optional[ParsedSubcmd] = None,
) -> bool:
    """Return True if any sub-command invokes a recognized test runner without masking chaining or unsafe flags."""
    if parsed_sub is not None:
        if parsed_sub.subshell_opens or parsed_sub.subshell_closes or (("$(" in parsed_sub.raw or "`" in parsed_sub.raw or "<(" in parsed_sub.unquoted or ">(" in parsed_sub.unquoted) and _has_unquoted_cmd_subst(parsed_sub.raw)):
            return False
        subs: Tuple[ParsedSubcmd, ...] = (parsed_sub,)
        check_next = False
    else:
        if not cmd:
            return False
        p = parsed if parsed is not None else parse_shell_command(cmd)
        if p.has_nul_surr or p.has_ansi_c or p.has_subshell or p.has_cmd_subst or p.has_unclosed or p.has_masking or _payload_has_raw_cr(cmd, payload) or not p.subcmds:
            return False
        subs, check_next = p.subcmds, True
    found_test = False
    for sub in subs:
        if any(_is_dangerous_env_assignment(f"{k}{'+=' if app else '='}{v}") for k, app, v in sub.env_assigns):
            return False
        verb, args_t = sub.verb, list(sub.args)
        if sub.unwrap_unv or verb in (_SHELL_WRAPPER_VERBS | _OPAQUE_EXEC_VERBS | {"xargs", "pushd", "popd"}) or (verb == "find" and any(t in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for t in args_t)):
            return False
        if verb in {"export", "declare", "typeset", "local", "readonly"}:
            if any(_is_dangerous_env_assignment(t if "=" in t else f"{t}=") for t in args_t if t != "--" and not t.startswith("-")):
                return False
            continue
        is_runner = (
            verb in {"pytest", "py.test", "unittest", "jest", "vitest", "tox", "nox"}
            or (verb in {"npm", "pnpm", "yarn", "go", "cargo"} and bool(args_t) and args_t[0] == "test")
            or (_PY_VERB_RE.match(verb) and _parse_python_cli(args_t)[:3] in {(True, "m", "pytest"), (True, "m", "py.test"), (True, "m", "unittest")})
        )
        if not is_runner:
            continue
        if not _is_canonical_cmd_path(verb, sub.raw_verb) or TRIVIAL_TEST_FLAG_RE.search(_fold_shell_line_continuations(sub.raw)) or TRIVIAL_TEST_FLAG_RE.search(" ".join([verb, *args_t])) or not _check_test_runner_args(verb, args_t, cur_dir, payload):
            return False
        if check_next and sub.next_delim in ("|", "|&", ";", "\n", "||", "&"):
            return False
        found_test = True
    return found_test

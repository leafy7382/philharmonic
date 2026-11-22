"""Shared test-runner argument validation and test command recognition."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from guard.constants import TRIVIAL_TEST_FLAG_RE
from guard.path_guards import _is_staging_or_tmp_path, _resolve_cmd_target
from guard.shell_tokens import (
    _OPAQUE_EXEC_VERBS, _PY_VERB_RE, _SHELL_WRAPPER_VERBS,
    _fold_shell_line_continuations, _has_shell_masking_or_chaining,
    _is_dangerous_env_assignment, _parse_leading_env_assignments,
    _parse_python_cli, _split_shell_commands_with_delims,
    _split_shell_tokens_raw_with_meta, _strip_redir_tokens_with_meta,
    _unwrap_shell_command_tokens,
)

_PYTEST_PATH_OUTPUT_FLAGS = frozenset({"--junitxml", "--junit-xml", "--basetemp", "--resultlog", "--result-log", "--html", "--cov-report", "--log-file"})
_PYTEST_STAGING_VAL_FLAGS = frozenset({"--rootdir", "--confcutdir", "--basetemp", "--junitxml", "--junit-xml", "-c"})
_PYTEST_OTHER_MUT_FLAGS = frozenset({"-o", "--override-ini", "-c", "-p", "--config-file", "--rootdir", "--confcutdir"})
_PYTEST_SKIP_VAL_FLAGS = frozenset({
    "-k", "-m", "-p", "-o", "--override-ini", "-W", "--maxfail", "--tb", "--capture", "--show-capture",
    "--durations", "--durations-min", "--log-level", "--log-format", "--log-date-format", "--import-mode",
    "--color", "--code-highlight", "--deselect", "--ignore", "--ignore-glob", "--resultlog", "--result-log",
    "--html", "--cov", "--cov-report", "--log-file",
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
    "tox": (None, frozenset({"-", "--", "--installpkg", "--force-dep"}), "", frozenset(), False, False),
    "nox": (None, frozenset({"-", "--", "--installpkg", "--force-dep"}), "", frozenset(), False, False),
    "go": ("test", _GO_BAD, "", frozenset(), True, True),
    "cargo": ("test", frozenset({"--target-dir", "--manifest-path", "--list"}), "", frozenset(), True, False),
}


def _extract_pytest_target_check(pytest_args: List[str], cur_dir: str = "", payload: Optional[Dict[str, Any]] = None) -> Tuple[List[str], bool]:
    """Check pytest args for mutating/config/staging flags and extract any output paths."""
    targets, mut, after_dd, skip_next = [], False, False, False
    safe_pl = payload if isinstance(payload, dict) else {}
    _staged = lambda v: bool(v) and _is_staging_or_tmp_path(_resolve_cmd_target(v, cur_dir), safe_pl)
    for idx, tok in enumerate(pytest_args):
        if skip_next:
            skip_next = False
        elif not after_dd and tok == "--":
            after_dd = True
        elif not after_dd and tok.startswith("-") and tok != "-":
            opt, has_eq, eq_val = tok.partition("=")
            val = eq_val if has_eq else (pytest_args[idx + 1] if idx + 1 < len(pytest_args) else "")
            if opt in _PYTEST_PATH_OUTPUT_FLAGS:
                mut = True
                out_v = val.split(":", 1)[1] if opt == "--cov-report" and ":" in val else val
                if out_v:
                    targets.append(out_v)
            if (opt in _PYTEST_STAGING_VAL_FLAGS and _staged(val)) or opt in _PYTEST_OTHER_MUT_FLAGS:
                mut = True
            if not has_eq and opt in _PYTEST_SKIP_VAL_FLAGS:
                skip_next = True
            elif not tok.startswith("--"):
                for k, ch in enumerate(tok[1:]):
                    if ch in "ocp" or (ch == "c" and _staged(tok[1:][k + 1 :] or (pytest_args[idx + 1] if idx + 1 < len(pytest_args) else ""))):
                        mut = True
                    if ch in "ocpkmW":
                        skip_next = k == len(tok[1:]) - 1
                        break
        elif _staged(tok.split("::", 1)[0]):
            mut = True
    return targets, mut


def _check_test_runner_args(verb: str, arg_tokens: List[str], cur_dir: str = "", payload: Optional[Dict[str, Any]] = None) -> bool:
    """Validate test runner flags against mutation/arbitrary-exec/staging flags."""
    safe_pl = payload if isinstance(payload, dict) else {}
    if verb in {"pytest", "py.test"}:
        return not _extract_pytest_target_check(arg_tokens, cur_dir, safe_pl)[1]
    if _PY_VERB_RE.match(verb):
        ok_py, mode, mod, mod_args = _parse_python_cli(arg_tokens)
        if not ok_py or mode != "m":
            return False
        if mod in {"pytest", "py.test"}:
            return not _extract_pytest_target_check(mod_args, cur_dir, safe_pl)[1]
        return mod == "unittest" and not any(_is_staging_or_tmp_path(_resolve_cmd_target(a.split("=", 1)[-1], cur_dir), safe_pl) for a in mod_args if a not in {"discover", "-v", "--verbose"})
    if (rule := _TEST_RUNNER_RULES.get(verb)) is None:
        return True
    req_sub, bad_opts, bad_short, bad_words, stop_dd, chk_staging = rule
    if req_sub is not None:
        if not arg_tokens or arg_tokens[0] != req_sub:
            return False
        arg_tokens = arg_tokens[1:]
    for t in arg_tokens:
        if stop_dd and t == "--":
            break
        if t in bad_words or t.split("=", 1)[0] in bad_opts or (bad_short and t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in bad_short)):
            return False
        if chk_staging and (val := (t.split("=", 1)[1] if "=" in t else ("" if t.startswith("-") else t))) and _is_staging_or_tmp_path(_resolve_cmd_target(val, cur_dir), safe_pl):
            return False
    return True


def _is_test_command(cmd: str, *, cur_dir: str = "", payload: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if any sub-command invokes a recognized test runner without masking chaining or unsafe flags."""
    if not cmd or _has_shell_masking_or_chaining(cmd) or not (parts := _split_shell_commands_with_delims(cmd)):
        return False
    found_test = False
    for idx, (_, subcmd) in enumerate(parts):
        if not subcmd:
            continue
        folded = _fold_shell_line_continuations(subcmd)
        if any(_is_dangerous_env_assignment(f"{k}{'+=' if app else '='}{v}") for k, app, v in _parse_leading_env_assignments(folded)[0]):
            return False
        toks, uq = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(folded))
        verb, args_t, unwrap_unv = _unwrap_shell_command_tokens(toks, uq)
        if unwrap_unv or verb in (_SHELL_WRAPPER_VERBS | _OPAQUE_EXEC_VERBS | {"xargs"}) or (verb == "find" and any(t in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for t in args_t)):
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
        if TRIVIAL_TEST_FLAG_RE.search(folded) or TRIVIAL_TEST_FLAG_RE.search(" ".join([verb, *args_t])) or not _check_test_runner_args(verb, args_t, cur_dir, payload):
            return False
        if (parts[idx + 1][0] if idx + 1 < len(parts) else None) in ("|", "|&", ";", "\n", "||", "&"):
            return False
        found_test = True
    return found_test

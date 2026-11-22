"""Python AST inline inspection and pytest argument checks."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

from guard.constants import PROJECT_ROOT
from guard.path_guards import _is_staging_or_tmp_path, _resolve_cmd_target

_PY_MUTATING_MODULES = frozenset(
    {
        "os", "posix", "nt", "pathlib", "shutil", "subprocess", "io", "codecs",
        "ctypes", "importlib", "builtins", "tempfile", "glob", "sys", "runpy",
        "pty", "fcntl", "mmap", "socket", "http", "urllib", "sqlite3", "shelve",
        "dbm", "zipfile", "tarfile", "gzip", "bz2", "lzma", "pickle", "marshal",
        "code", "pdb", "webbrowser", "logging", "guard", "orchestrator_guard",
        "state_store", "identity", "policies", "verification", "constants", "cli",
        "pip", "setuptools", "pkg_resources", "wheel", "pytest", "_pytest",
        "pluggy", "uv", "poetry", "tox", "nox", "ruff", "black", "isort",
        "pylint", "mypy", "pyright", "yaml", "requests", "httpx",
    }
)
_PY_MUTATING_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
        "delattr", "globals", "locals", "vars", "breakpoint", "input", "memoryview",
    }
)
_PY_MUTATING_ATTRS = (
    frozenset(
        {
            "write", "writelines", "write_text", "write_bytes", "truncate",
            "unlink", "remove", "removedirs", "rmdir", "rename", "replace",
            "rmtree", "mkdir", "makedirs", "chmod", "chown", "touch", "symlink",
            "symlink_to", "hardlink_to", "link", "system", "popen", "open",
            "fdopen", "copy", "copyfile", "copy2", "copytree", "move", "extract",
            "extractall", "scanner", "decoder", "encoder", "codecs", "JSONDecoder",
            "JSONEncoder", "load", "dump", "modules", "attrgetter", "itemgetter",
            "methodcaller", "run_docstring_examples", "posix_spawn", "posix_spawnp",
        }
    )
    | _PY_MUTATING_MODULES
)
_PY_FRAME_ATTRS = frozenset(
    {
        "f_builtins", "f_globals", "f_locals", "f_back", "f_code",
        "gi_frame", "cr_frame", "ag_frame", "tb_frame", "func_globals", "co_consts",
    }
)
_SAFE_PURE_STDLIB_MODULES = frozenset({"math", "cmath", "json"})
_SAFE_JSON_IMPORTS = frozenset({"dumps", "loads", "JSONDecodeError"})
_SAFE_PYTHON_MODULES = frozenset({"unittest", "pytest"})
_SAFE_WORKFLOW_STDLIB_MODULES = frozenset(
    {
        "json", "re", "math", "cmath", "collections", "itertools", "functools",
        "dataclasses", "typing", "datetime", "string", "textwrap", "hashlib",
        "base64", "statistics", "copy", "enum", "decimal", "fractions",
        "bisect", "heapq", "time", "uuid", "csv", "difflib", "pprint",
    }
)
_SHELL_WRAPPER_VERBS = frozenset(
    {
        "sh", "bash", "dash", "zsh", "ksh", "ash", "csh", "tcsh",
        "fish", "mksh", "rbash", "eval", "source", ".",
    }
)


def _unwrap_matching_quotes(text: str) -> str:
    """Strip one pair of matching outer single/double quotes if present."""
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s


def _is_trivial_python_inline(code: str) -> bool:
    """Return True if inline python -c code only performs pass/print/exit/trivial asserts without real test logic."""
    inner = _unwrap_matching_quotes(code)
    if not inner:
        return True
    try:
        tree = ast.parse(inner)
    except (SyntaxError, ValueError, TypeError):
        return False
    except (RecursionError, MemoryError):
        return True
    if not tree.body:
        return True

    def _is_trivial_stmt(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Import):
            return all(alias.name == "sys" for alias in stmt.names)
        if isinstance(stmt, ast.ImportFrom):
            return stmt.module == "sys"
        if isinstance(stmt, ast.Assert):
            test = stmt.test
            if isinstance(test, ast.Constant):
                return bool(test.value) is True
            if (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.left, ast.Constant)
                and isinstance(test.comparators[0], ast.Constant)
            ):
                return test.left.value == test.comparators[0].value
            return False
        if isinstance(stmt, ast.Expr):
            val = stmt.value
            if isinstance(val, ast.Constant):
                return True
            if isinstance(val, ast.Call):
                func = val.func
                if isinstance(func, ast.Name) and func.id in {"print", "exit", "quit"}:
                    return True
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "sys"
                    and func.attr in {"exit", "quit"}
                ):
                    return True
            return False
        return False

    return all(_is_trivial_stmt(stmt) for stmt in tree.body)


def _python_inline_code_is_mutating(code: str) -> bool:
    """Inspect inline Python -c code via AST; return True on mutating/dangerous constructs or syntax error."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return True
    stdlib_mods = getattr(sys, "stdlib_module_names", frozenset())

    def _is_blocked_root_mod(mod_name: str) -> bool:
        if not mod_name:
            return False
        if mod_name.startswith("_") or mod_name in _PY_MUTATING_MODULES:
            return True
        if mod_name in stdlib_mods:
            return mod_name not in _SAFE_PURE_STDLIB_MODULES
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError, AttributeError, OSError):
            return False
        if spec is None:
            return False
        proj_root = PROJECT_ROOT.resolve()
        candidates: List[str] = []
        origin = getattr(spec, "origin", None)
        if origin and origin not in {"built-in", "frozen", "namespace"}:
            candidates.append(str(origin))
        for loc in getattr(spec, "submodule_search_locations", None) or []:
            candidates.append(str(loc))
        for cand in candidates:
            try:
                resolved_cand = Path(cand).resolve()
                if resolved_cand != proj_root and proj_root not in resolved_cand.parents:
                    return True
            except (OSError, ValueError):
                return True
        return False

    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    alias_name = alias.name or ""
                    if alias_name.startswith("json.") or _is_blocked_root_mod(alias_name.split(".", 1)[0]):
                        return True
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("json."):
                    return True
                if mod == "json" and any((a.name or "") not in _SAFE_JSON_IMPORTS for a in node.names):
                    return True
                if _is_blocked_root_mod(mod.split(".", 1)[0]):
                    return True
                if not node.module:
                    for alias in node.names:
                        an = alias.name or ""
                        if an.startswith("json.") or _is_blocked_root_mod(an.split(".", 1)[0]):
                            return True
            elif isinstance(node, ast.Name):
                if node.id in _PY_MUTATING_NAMES or node.id.startswith("_"):
                    return True
            elif isinstance(node, ast.Attribute):
                if node.attr in _PY_MUTATING_ATTRS or node.attr.startswith("_") or node.attr in _PY_FRAME_ATTRS:
                    return True
    except (RecursionError, MemoryError):
        return True
    return False


_PYTEST_STAGING_VAL_FLAGS = frozenset({"--rootdir", "--confcutdir", "--basetemp", "--junitxml", "--junit-xml", "-c"})
_PYTEST_SKIP_VAL_FLAGS = frozenset(
    {
        "-k", "-m", "-p", "-o", "--override-ini", "-W", "--maxfail", "--tb",
        "--capture", "--show-capture", "--durations", "--durations-min",
        "--log-level", "--log-format", "--log-date-format", "--import-mode",
        "--color", "--code-highlight", "--deselect", "--ignore", "--ignore-glob",
        "--resultlog", "--result-log", "--html", "--cov", "--cov-report", "--log-file",
    }
)
_PYTEST_PATH_OUTPUT_FLAGS = frozenset(
    {"--junitxml", "--junit-xml", "--basetemp", "--resultlog", "--result-log", "--html", "--cov-report", "--log-file"}
)
_PYTEST_OTHER_MUT_FLAGS = frozenset({"-o", "--override-ini", "-c", "-p"})


def _pytest_targets_staging_or_tmp(
    pt_args: List[str], effective_cwd: str, payload: Dict[str, Any]
) -> bool:
    """Return True if pytest args target /tmp, /brain/, or scratch/ via flags or positional paths."""
    after_dd = skip_next = False
    for idx, tok in enumerate(pt_args):
        if skip_next:
            skip_next = False
            continue
        if not after_dd and tok == "--":
            after_dd = True
            continue
        if not after_dd and tok.startswith("-") and tok != "-":
            if tok in _PYTEST_STAGING_VAL_FLAGS:
                if idx + 1 < len(pt_args) and _is_staging_or_tmp_path(
                    _resolve_cmd_target(pt_args[idx + 1], effective_cwd), payload
                ):
                    return True
                skip_next = True
                continue
            if tok.startswith(tuple(f + "=" for f in _PYTEST_STAGING_VAL_FLAGS if f.startswith("--"))):
                val = tok.split("=", 1)[1]
                if val and _is_staging_or_tmp_path(_resolve_cmd_target(val, effective_cwd), payload):
                    return True
                continue
            if tok in _PYTEST_SKIP_VAL_FLAGS:
                skip_next = True
                continue
            if not tok.startswith("--") and len(tok) > 1:
                flags = tok[1:]
                for k, ch in enumerate(flags):
                    if ch == "c":
                        val = flags[k + 1 :] or (pt_args[idx + 1] if idx + 1 < len(pt_args) else "")
                        if val and _is_staging_or_tmp_path(_resolve_cmd_target(val, effective_cwd), payload):
                            return True
                        if not flags[k + 1 :]:
                            skip_next = True
                        break
                    if ch in ("k", "m", "p", "o", "W"):
                        if k == len(flags) - 1:
                            skip_next = True
                        break
            continue
        pos_val = tok.split("::", 1)[0]
        if pos_val and _is_staging_or_tmp_path(_resolve_cmd_target(pos_val, effective_cwd), payload):
            return True
    return False


def _check_pytest_mutating_args(pytest_args: List[str]) -> Tuple[List[str], bool]:
    """Check pytest CLI args for file-writing or config/plugin-loading flags and extract any output paths."""
    targets: List[str] = []
    mut = after_dd = skip_next = False
    for idx, tok in enumerate(pytest_args):
        if skip_next:
            skip_next = False
            continue
        if not after_dd and tok == "--":
            after_dd = True
            continue
        if not after_dd and tok.startswith("-") and tok != "-":
            if tok in _PYTEST_PATH_OUTPUT_FLAGS:
                mut = True
                if idx + 1 < len(pytest_args):
                    targets.append(pytest_args[idx + 1])
                    skip_next = True
            elif tok.startswith(tuple(f + "=" for f in _PYTEST_PATH_OUTPUT_FLAGS)):
                mut = True
                val = tok.split("=", 1)[1]
                if tok.startswith("--cov-report=") and ":" in val:
                    val = val.split(":", 1)[1]
                if val:
                    targets.append(val)
            elif tok in _PYTEST_OTHER_MUT_FLAGS:
                mut = True
                if idx + 1 < len(pytest_args):
                    skip_next = True
            elif tok.startswith("--override-ini="):
                mut = True
            elif not tok.startswith("--") and any(ch in tok[1:] for ch in ("o", "c", "p")):
                mut = True
                flags = tok[1:]
                for k, ch in enumerate(flags):
                    if ch in ("o", "c", "p"):
                        if k == len(flags) - 1 and idx + 1 < len(pytest_args):
                            skip_next = True
                        break
    return targets, mut


def _check_test_runner_args(verb: str, arg_tokens: List[str], cur_dir: str, payload: Dict[str, Any]) -> bool:
    """Validate test runner flags against mutation/arbitrary-exec/staging flags (Finding #5)."""
    if verb in {"pytest", "py.test"}:
        _, pt_mut = _check_pytest_mutating_args(arg_tokens)
        return (not pt_mut) and (not _pytest_targets_staging_or_tmp(arg_tokens, cur_dir, payload))
    if re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)", verb):
        for p_i, p_t in enumerate(arg_tokens):
            if p_t == "-m" and p_i + 1 < len(arg_tokens) and arg_tokens[p_i + 1] in {"pytest", "py.test"}:
                pt_args = arg_tokens[p_i + 2 :]
                _, pt_mut = _check_pytest_mutating_args(pt_args)
                return (not pt_mut) and (not _pytest_targets_staging_or_tmp(pt_args, cur_dir, payload))
            if p_t in {"-mpytest", "-mpy.test"}:
                pt_args = arg_tokens[p_i + 1 :]
                _, pt_mut = _check_pytest_mutating_args(pt_args)
                return (not pt_mut) and (not _pytest_targets_staging_or_tmp(pt_args, cur_dir, payload))
        return True
    if verb in {"tox", "nox"}:
        return not any(
            t in {"-", "--"} or t.split("=", 1)[0] in {"--installpkg", "--force-dep"}
            for t in arg_tokens
        )
    if verb in {"jest", "vitest"}:
        bad_jest = {
            "-u", "--updateSnapshot", "--update-snapshot", "--init",
            "--clearCache", "--clear-cache", "--passWithNoTests",
            "--pass-with-no-tests", "--coverage", "--coverageDirectory", "--reporter",
            "--reporters", "-R", "--config", "-c", "--setupFiles",
            "--setupFilesAfterEnv", "--testEnvironment", "--resolver",
            "--moduleNameMapper", "--transform", "--globalSetup", "--globalTeardown",
        }
        for t in arg_tokens:
            if t == "--":
                break
            if verb == "vitest" and t == "init":
                return False
            opt = t.split("=", 1)[0]
            if opt in bad_jest or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "ucR")):
                return False
            val = t.split("=", 1)[1] if "=" in t else ("" if t.startswith("-") else t)
            if val and _is_staging_or_tmp_path(_resolve_cmd_target(val, cur_dir), payload):
                return False
        return True
    if verb in {"npm", "pnpm", "yarn"} and arg_tokens and arg_tokens[0] == "test":
        for t in arg_tokens[1:]:
            if t == "--" or t.split("=", 1)[0] in {
                "-c", "-r", "--config", "--userconfig", "--globalconfig", "--require", "--prefix", "-C",
            }:
                return False
            if t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "crC"):
                return False
            if _is_staging_or_tmp_path(_resolve_cmd_target(t.split("=", 1)[-1], cur_dir), payload):
                return False
        return True
    if verb == "playwright":
        return not any(
            t.split("=", 1)[0] in {"--reporter", "--output", "-o"}
            or (
                (not t.startswith("-") or "=" in t)
                and _is_staging_or_tmp_path(_resolve_cmd_target(t.split("=", 1)[-1], cur_dir), payload)
            )
            for t in arg_tokens
        )
    if verb == "go" and arg_tokens and arg_tokens[0] == "test":
        bad_go = {
            "-coverprofile", "-cpuprofile", "-memprofile", "-blockprofile",
            "-mutexprofile", "-trace", "-outputdir", "-o",
            "--coverprofile", "--cpuprofile", "--memprofile", "--blockprofile",
            "--mutexprofile", "--trace", "--outputdir", "--o",
        }
        for t in arg_tokens[1:]:
            if t == "--":
                break
            if t.split("=", 1)[0] in bad_go:
                return False
        return True
    if verb == "cargo" and arg_tokens and arg_tokens[0] == "test":
        for t in arg_tokens[1:]:
            if t == "--":
                break
            if t.split("=", 1)[0] in {"--target-dir", "--manifest-path", "--list"}:
                return False
        return True
    return True

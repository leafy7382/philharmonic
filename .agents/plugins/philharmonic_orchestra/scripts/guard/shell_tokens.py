"""Shell lexing, redirection extraction, wrapper unwrapping, and eval inspection."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
from typing import Any, Dict, Iterator, List, Optional, Tuple

from guard.ast_guard import _SHELL_WRAPPER_VERBS, _is_trivial_python_inline
from guard.constants import (
    MUTATING_SHELL_CMD_RE,
    PROJECT_ROOT,
    SAFE_GIT_INSPECTION_RE,
    SHELL_EXEC_WRAPPER_RE,
    TEST_CMD_START_RE,
    TRIVIAL_EVAL_CMD_RE,
    TRIVIAL_TEST_FLAG_RE,
    _fold_shell_line_continuations,
    _has_nul_or_surrogate_input,
    _has_shell_masking_or_chaining,
    _is_git_only_command,
    _is_test_command,
    _parse_leading_env_assignments,
    _preprocess_shell_heredocs_and_comments,
    _split_shell_commands,
    _split_shell_commands_with_delims,
)
from guard.path_guards import (
    _is_guard_plugin_cwd,
    _is_staging_or_tmp_path,
    _resolve_cmd_target,
    _sanitize_decoded_shell_token,
)

SAFE_XARGS_READONLY_CMDS = frozenset(
    {
        "grep", "egrep", "fgrep", "rg", "wc", "cat", "head", "tail", "ls",
        "stat", "file", "echo", "printf", "basename", "dirname", "md5sum",
        "sha256sum", "sha1sum", "cksum", "du", "realpath", "readlink",
    }
)
SAFE_READONLY_SHELL_VERBS = frozenset(
    {
        "ls", "dir", "vdir", "pwd", "echo", "cat", "tac", "nl", "head", "tail",
        "wc", "grep", "egrep", "fgrep", "zgrep", "stat", "file", "basename",
        "dirname", "realpath", "readlink", "md5sum", "sha1sum", "sha256sum",
        "sha512sum", "cksum", "sum", "du", "df", "id", "whoami", "uname",
        "date", "which", "type", "test", "[", "[[", "true", "false", ":",
        "cut", "tr", "paste", "comm", "diff", "cmp", "rev", "fold", "fmt",
        "expand", "unexpand", "column", "strings", "od", "hexdump", "jq",
        "sleep", "printenv", "hostname", "uptime", "nproc", "getconf", "seq", "expr",
    }
)

_REDIR_OP_RE = re.compile(r"(?:[0-9]*&>>|[0-9]*&>|[0-9]*>>|[0-9]*>\||[0-9]*>&|[0-9]*>)")
_TOKEN_REDIR_OP_RE = re.compile(r"(?:&>>|&>|>>|>\||>&|<<<|<<-|<<|<&|<(?!\()|>(?!\())")
_ANSI_C_SIMPLE_ESCAPES = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n",
    "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?",
}

_LEADING_GROUPING_WORDS = frozenset(
    {"(", "{", "!", "if", "while", "until", "then", "do", "else", "elif", "coproc"}
)
_TRAILING_GROUPING_TOKENS = frozenset({")", "}"})
_STANDALONE_END_WORDS = frozenset({"fi", "done", "esac"})
_DANGEROUS_ENV_VARS = frozenset(
    {
        "GCONV_PATH", "HOSTALIASES", "LOCPATH", "NLSPATH", "ENV", "BASH_ENV",
        "CDPATH", "IFS", "PROMPT_COMMAND", "PS4", "FPATH", "ZDOTDIR", "SHELL",
        "PATH", "GLOBIGNORE", "GIT_EXTERNAL_DIFF", "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR",
        "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_ASKPASS", "SSH_ASKPASS",
        "GIT_PROXY_COMMAND", "GIT_UPLOAD_PACK", "GIT_RECEIVE_PACK", "GIT_DIR",
        "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "EDITOR", "VISUAL",
        "PYTHONSTARTUP", "PYTHONHOME", "PYTHONEXECUTABLE", "PYTHONUSERBASE",
        "PYTHONINSPECT", "PYTHONWARNINGS", "PERL5LIB", "PERLLIB", "PERL5OPT",
        "RUBYLIB", "RUBYOPT", "NODE_OPTIONS", "NODE_PATH", "AWKPATH", "AWKLIBPATH",
        "LUA_PATH", "LUA_CPATH", "HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
        "XDG_CACHE_HOME", "PYTHONBREAKPOINT", "PYTHONCOERCECLOCALE", "PYTHONDEVMODE",
        "PYTHONMALLOC", "GIT_CONFIG_NOSYSTEM", "GIT_ATTR_SOURCE", "GIT_GRAFT_FILE",
        "GIT_NOTES_REF", "GIT_REPLACE_REF_BASE", "HISTFILE", "LESSOPEN", "LESSCLOSE",
        "MANPAGER", "BROWSER", "TMPDIR", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
        "GOFLAGS", "GOENV", "CC", "CXX", "LD", "AR", "AS", "RUSTC", "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_HOME",
        "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER",
        "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "CARGO_TARGET_DIR",
    }
)
_DANGEROUS_ENV_PREFIXES = (
    "LD_", "DYLD_", "GIT_TRACE", "LUA_INIT", "NPM_CONFIG_", "YARN_", "PNPM_",
    "CARGO_TARGET_", "CARGO_REGISTRIES_", "UV_", "PIP_", "POETRY_", "PIPENV_", "PYTEST_",
)
_SAFE_GIT_EDITOR_VALS = frozenset({"", "true", "/bin/true", ":", "false", "/bin/false"})
_SAFE_PAGER_VALS = frozenset({"", "cat", "/bin/cat", "true", "/bin/true", ":"})
_UNVERIFIABLE_WRAPPERS = frozenset(
    {
        "flock", "script", "stdbuf", "parallel", "busybox", "strace", "ltrace",
        "chrt", "ionice", "taskset", "unshare", "nsenter", "chroot", "watch",
        "prlimit", "setarch", "linux32", "linux64", "numactl", "setsid",
        "ed", "ex", "uvx",
    }
)
_OPAQUE_EXEC_VERBS = _UNVERIFIABLE_WRAPPERS | {
    "bun", "deno", "expect", "tclsh", "wish", "rscript", "julia", "pwsh", "sqlite3",
}
_SAFE_NPX_RUNNERS = frozenset(
    {"jest", "vitest", "mocha", "ava", "tap", "playwright", "tsc", "pyright", "mypy"}
)


def _decode_ansi_c_escape(text: str, idx: int) -> Tuple[str, int]:
    """Decode one ANSI-C $'...' backslash escape starting at idx."""
    n = len(text)
    if idx + 1 >= n:
        return "\\", idx + 1
    nxt = text[idx + 1]
    if nxt in _ANSI_C_SIMPLE_ESCAPES:
        return _ANSI_C_SIMPLE_ESCAPES[nxt], idx + 2
    if nxt == "c" and idx + 2 < n and text[idx + 2] != "'":
        if text[idx + 2] == "\\" and idx + 3 < n and text[idx + 3] in ("\\", "'"):
            ctrl_ch, step = text[idx + 3], 4
        else:
            ctrl_ch, step = text[idx + 2], 3
        return chr(0x7F if ctrl_ch == "?" else (ord(ctrl_ch) & 0x1F)), idx + step
    if nxt == "x":
        j = idx + 2
        while j < n and j < idx + 4 and text[j] in "0123456789abcdefABCDEF":
            j += 1
        return (chr(int(text[idx + 2 : j], 16) & 0xFF), j) if j > idx + 2 else ("\\x", idx + 2)
    if nxt in ("u", "U"):
        max_len = 4 if nxt == "u" else 8
        j = idx + 2
        while j < n and j < idx + 2 + max_len and text[j] in "0123456789abcdefABCDEF":
            j += 1
        if j > idx + 2:
            cp = int(text[idx + 2 : j], 16)
            return ("\ufffd" if 0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF else chr(cp)), j
        return "\\" + nxt, idx + 2
    if nxt in "01234567":
        j = idx + 1
        while j < n and j < idx + 4 and text[j] in "01234567":
            j += 1
        return chr(int(text[idx + 1 : j], 8) & 0xFF), j
    return nxt, idx + 2


def _iter_unquoted_shell_indices(text: str) -> Iterator[int]:
    """Yield indices in text that lie outside single, ANSI-C, double quotes, and backslash escapes."""
    in_sq = in_ansi = in_dq = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_ansi:
            if ch == "\\":
                _, i = _decode_ansi_c_escape(text, i)
                continue
            if ch == "'":
                in_ansi = False
            i += 1
            continue
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_sq:
            escaped = True
            i += 1
            continue
        if not in_sq and not in_dq and ch == "$" and i + 1 < n and text[i + 1] == "'":
            in_ansi = True
            i += 2
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        if not in_sq and not in_dq:
            yield i
        i += 1


def _read_shell_word_from(cmd: str, start: int) -> Tuple[str, int, bool, bool]:
    """Read one dequoted shell word starting at start, returning (text, end_idx, saw_quote_or_esc, trimmed_rbrace)."""
    tok_chars: List[str] = []
    t_sq = t_ansi = t_trunc = t_dq = t_esc = saw_qe = saw_lbrace = last_rbrace = False
    j, n = start, len(cmd)
    while j < n:
        tch = cmd[j]
        if t_esc:
            tok_chars.append(tch)
            saw_lbrace = saw_lbrace or (tch == "{")
            last_rbrace = False
            t_esc = False
            j += 1
            continue
        if t_ansi:
            if tch == "\\":
                dec, j = _decode_ansi_c_escape(cmd, j)
                if "\x00" in dec:
                    t_trunc = True
                elif not t_trunc:
                    tok_chars.append(dec)
                    saw_lbrace = saw_lbrace or ("{" in dec)
                    last_rbrace = False
                continue
            if tch == "'":
                t_ansi = t_trunc = False
            elif tch == "\x00":
                t_trunc = True
            elif not t_trunc:
                tok_chars.append(tch)
                saw_lbrace = saw_lbrace or (tch == "{")
                last_rbrace = False
            j += 1
            continue
        if t_sq:
            if tch == "'":
                t_sq = False
            else:
                tok_chars.append(tch)
                saw_lbrace = saw_lbrace or (tch == "{")
                last_rbrace = False
            j += 1
            continue
        if t_dq:
            if tch == "\\" and j + 1 < n and cmd[j + 1] in ('"', "\\", "$", "`", "\n"):
                if cmd[j + 1] != "\n":
                    tok_chars.append(cmd[j + 1])
                    last_rbrace = False
                j += 2
                continue
            if tch == '"':
                t_dq = False
            else:
                tok_chars.append(tch)
                saw_lbrace = saw_lbrace or (tch == "{")
                last_rbrace = False
            j += 1
            continue
        if tch == "\\":
            t_esc = saw_qe = True
            j += 1
            continue
        if tch == "$" and j + 1 < n and cmd[j + 1] == "'":
            t_ansi = saw_qe = True
            t_trunc = False
            j += 2
            continue
        if tch == "'":
            t_sq = saw_qe = True
            j += 1
            continue
        if tch == '"':
            t_dq = saw_qe = True
            j += 1
            continue
        if tch in " \t\n;|><()" or (tch == "&" and tok_chars):
            break
        saw_lbrace = saw_lbrace or (tch == "{")
        last_rbrace = tch == "}"
        tok_chars.append(tch)
        j += 1
    sink = _sanitize_decoded_shell_token("".join(tok_chars)).strip()
    if last_rbrace and not saw_lbrace and sink.endswith("}"):
        sink = sink[:-1].strip()
    return sink, j, saw_qe, last_rbrace


def _extract_shell_redir_targets(cmd: str) -> List[str]:
    """Extract write targets from unquoted shell redirections (>, >>, >|, &>, N>)."""
    cmd = _fold_shell_line_continuations(_preprocess_shell_heredocs_and_comments(cmd)[0])
    if not cmd:
        return []
    targets: List[str] = []
    n = len(cmd)
    skip_until = 0
    for i in _iter_unquoted_shell_indices(cmd):
        if i < skip_until:
            continue
        ch = cmd[i]
        if ch == "<" or (ch.isdigit() and i > 0 and cmd[i - 1].isalnum()):
            continue
        if ch in (">", "&") or ch.isdigit():
            m = _REDIR_OP_RE.match(cmd, i)
            if not m or (m.group(0).startswith("&") and i > 0 and cmd[i - 1] in (">", "<")):
                continue
            j = m.end()
            while j < n and cmd[j] in " \t":
                j += 1
            sink, j, saw_qe, _ = _read_shell_word_from(cmd, j)
            skip_until = j
            op_str = m.group(0)
            is_fd_dup = op_str.endswith(">&") and (not saw_qe) and (
                sink.isdigit() or sink == "-" or (sink.endswith("-") and sink[:-1].isdigit())
            )
            if (
                not is_fd_dup
                and sink
                and sink != "/dev/null"
                and not (not saw_qe and sink in {"&1", "&2"})
            ):
                targets.append(sink)
    return targets


def _split_shell_tokens_raw_with_meta(text: str) -> List[Tuple[str, bool, bool]]:
    """Lex a shell string into (dequoted_token, is_unquoted_redir_op, is_unquoted_token) tuples."""
    text = _fold_shell_line_continuations(_preprocess_shell_heredocs_and_comments(text)[0])
    if not text:
        return []
    tokens: List[Tuple[str, bool, bool]] = []
    current: List[str] = []
    has_token = saw_quote = in_sq = in_ansi = ansi_trunc = in_dq = False
    n, i = len(text), 0

    def _flush() -> None:
        nonlocal current, has_token, saw_quote
        if has_token:
            tokens.append((_sanitize_decoded_shell_token("".join(current)), False, not saw_quote))
            current = []
            has_token = saw_quote = False

    while i < n:
        ch = text[i]
        if in_ansi:
            if ch == "\\":
                dec, i = _decode_ansi_c_escape(text, i)
                if "\x00" in dec:
                    ansi_trunc = True
                elif not ansi_trunc:
                    current.append(dec)
                continue
            if ch == "'":
                in_ansi = ansi_trunc = False
            elif ch == "\x00":
                ansi_trunc = True
            elif not ansi_trunc:
                current.append(ch)
            i += 1
            continue
        if in_sq:
            if ch == "'":
                in_sq = False
            else:
                current.append(ch)
            i += 1
            continue
        if in_dq:
            if ch == "\\" and i + 1 < n and text[i + 1] in ('"', "\\", "$", "`", "\n"):
                if text[i + 1] != "\n":
                    current.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_dq = False
            else:
                current.append(ch)
            i += 1
            continue
        if ch in " \t\r\n":
            _flush()
            i += 1
            continue
        if ch == "\\":
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt == "\n":
                    i += 2
                    continue
                if nxt in ("(", ")", "{", "}", "!"):
                    saw_quote = True
                current.append(nxt)
                has_token = True
                i += 2
            else:
                current.append(ch)
                has_token = True
                i += 1
            continue
        if ch == "$" and i + 1 < n and text[i + 1] == "'":
            in_ansi = has_token = saw_quote = True
            ansi_trunc = False
            i += 2
            continue
        if ch == "'":
            in_sq = has_token = saw_quote = True
            i += 1
            continue
        if ch == '"':
            in_dq = has_token = saw_quote = True
            i += 1
            continue
        if ch in ("$", "<", ">") and i + 1 < n and text[i + 1] == "(":
            current.append(ch + "(")
            has_token = True
            i += 2
            continue
        if ch in ("(", ")"):
            _flush()
            tokens.append((ch, False, True))
            i += 1
            continue
        if not has_token and ch in ("{", "}") and (i + 1 >= n or text[i + 1] in " \t\r\n()"):
            tokens.append((ch, False, True))
            i += 1
            continue
        if not has_token and ch == "!" and (i + 1 >= n or text[i + 1] in " \t\r\n({"):
            tokens.append(("!", False, True))
            i += 1
            continue
        if (ch in (">", "<") and not (i + 1 < n and text[i + 1] == "(")) or (
            ch == "&" and i + 1 < n and text[i + 1] == ">"
        ):
            m_op = _TOKEN_REDIR_OP_RE.match(text, i)
            if m_op:
                op = m_op.group(0)
                if has_token and not saw_quote and current and "".join(current).isdigit() and not op.startswith("&"):
                    op = "".join(current) + op
                    current = []
                    has_token = saw_quote = False
                else:
                    _flush()
                tokens.append((op, True, True))
                i = m_op.end()
                continue
        current.append(ch)
        has_token = True
        i += 1
    _flush()
    return tokens


def _split_shell_tokens_raw(text: str) -> List[Tuple[str, bool]]:
    """Lex a shell string into (dequoted_token, is_unquoted_redir_op) tuples with ANSI-C $'...' support."""
    return [(tok, is_redir) for tok, is_redir, _ in _split_shell_tokens_raw_with_meta(text)]


def _split_shell_tokens(text: str) -> List[str]:
    """Tokenize a shell command string with ANSI-C $'...', quote concatenation, and backslash support."""
    return [tok for tok, _ in _split_shell_tokens_raw(text)]


def _tokenize_shell_tail(text: str) -> List[str]:
    """Tokenize a shell command tail safely, handling ANSI-C $'...' and interior quotes."""
    return _split_shell_tokens(text)


def _is_safe_git_inspection_command(cmd: str) -> bool:
    """Return True iff cmd is exclusively safe read-only git status/diff/log inspection."""
    if not cmd or not _is_git_only_command(cmd) or _extract_shell_redir_targets(cmd):
        return False
    sub_cmds = _split_shell_commands(cmd)
    if not sub_cmds:
        return False
    bad_opts = {
        "-c", "--output", "--ext-diff", "--exec-path", "--config-env",
        "--no-index", "--upload-pack", "--receive-pack", "--open-files-in-pager",
    }
    bad_prefixes = (
        "--output=", "--exec-path=", "--config-env=", "--upload-pack=",
        "--receive-pack=", "--open-files-in-pager=",
    )
    for sc in sub_cmds:
        if not SAFE_GIT_INSPECTION_RE.match(sc):
            return False
        for tok in _split_shell_tokens(sc):
            if tok == "--":
                break
            if tok in bad_opts or tok.startswith(bad_prefixes) or (
                tok.startswith("-c") and len(tok) > 2 and not tok.startswith("--")
            ):
                return False
    return True


def _strip_redir_tokens_with_meta(
    raw_tokens: List[Tuple[str, bool, bool]],
) -> Tuple[List[str], List[bool]]:
    """Remove unquoted redirection operators and their operand while preserving unquoted flags."""
    cleaned: List[str] = []
    unquoted: List[bool] = []
    skip_next = False
    for tok, is_redir, is_unquoted in raw_tokens:
        if skip_next:
            skip_next = False
            continue
        if is_redir:
            skip_next = True
            continue
        cleaned.append(tok)
        unquoted.append(is_unquoted)
    return cleaned, unquoted


def _is_dangerous_env_assignment(tok: str) -> bool:
    """Return True if a KEY=VAL or KEY+=VAL assignment injects preload/shell/git/interpreter code-exec risk."""
    name, eq, val = tok.partition("=")
    if not eq:
        return False
    is_append = name.endswith("+")
    if is_append:
        name = name[:-1]
    if name.startswith("BASH_FUNC_"):
        return True
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return False
    upper_name = name.upper()
    if is_append and (
        name in _DANGEROUS_ENV_VARS
        or upper_name in _DANGEROUS_ENV_VARS
        or name in {"PYTHONPATH", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_PAGER", "PAGER"}
        or upper_name.startswith(_DANGEROUS_ENV_PREFIXES)
    ):
        return True
    if name in {"GIT_EDITOR", "GIT_SEQUENCE_EDITOR"}:
        return val not in _SAFE_GIT_EDITOR_VALS
    if name in {"GIT_PAGER", "PAGER"}:
        return val not in _SAFE_PAGER_VALS
    if name == "PYTHONPATH":
        return any(seg not in {"", "."} for seg in val.split(":"))
    if upper_name.startswith(_DANGEROUS_ENV_PREFIXES):
        return True
    return name in _DANGEROUS_ENV_VARS or upper_name in _DANGEROUS_ENV_VARS


def _consume_short_flag_cluster(
    tok: str, value_chars: str, remaining: List[str], pop_fn: Any
) -> bool:
    """Pop a short-option token and its attached or following value arg if any flag char is in value_chars."""
    pop_fn()
    flags = tok[1:]
    for k, ch in enumerate(flags):
        if ch in value_chars:
            if k == len(flags) - 1 and remaining:
                pop_fn()
            return True
    return False


# Safe prefix-wrapper flag tables: (long_val_flags, short_val_chars, consume_trailing_positional_count)
_SAFE_WRAPPER_SPECS: Dict[str, Tuple[frozenset[str], str, int]] = {
    "nohup": (frozenset(), "", 0),
    "command": (frozenset(), "", 0),
    "builtin": (frozenset(), "", 0),
    "exec": (frozenset({"-a"}), "a", 0),
    "nice": (frozenset({"-n", "--adjustment"}), "n", 0),
    "timeout": (frozenset({"-k", "--kill-after", "-s", "--signal"}), "sk", 1),
    "time": (frozenset({"-o", "--output", "-f", "--format"}), "of", 0),
}
_RUNNER_GLOBAL_VAL_FLAGS = frozenset(
    {
        "--directory", "-C", "--project", "--config-file", "--python", "-p",
        "--cache-dir", "--index", "--default-index", "--index-url",
        "--extra-index-url", "--find-links",
    }
)
_RUNNER_RUN_VAL_FLAGS = _RUNNER_GLOBAL_VAL_FLAGS | frozenset(
    {"--with", "--with-editable", "--with-requirements", "--from", "--script", "-s", "--module", "-m", "--gui-script", "--env-file"}
)


def _unwrap_shell_command_tokens_with_raw(
    tokens: List[str],
    unquoted_flags: Optional[List[bool]] = None,
    out_arg_flags: Optional[List[bool]] = None,
    out_meta: Optional[Dict[str, bool]] = None,
) -> Tuple[str, str, List[str], bool]:
    """Strip grouping tokens, leading KEY=VAL env assignments, and safe prefix wrappers."""
    remaining = list(tokens)
    flags = list(unquoted_flags) if unquoted_flags is not None else [True] * len(remaining)
    unwrap_unverifiable = saw_opaque = False

    def _finish(v: str, raw_t: str, args_t: List[str], unv: bool) -> Tuple[str, str, List[str], bool]:
        if out_meta is not None:
            out_meta["opaque"] = saw_opaque
        return v, raw_t, args_t, unv

    def _pop_front() -> str:
        if flags:
            flags.pop(0)
        return remaining.pop(0)

    def _pop_back() -> str:
        if flags:
            flags.pop()
        return remaining.pop()

    while remaining:
        stripped_group = False
        while remaining and flags[0] and remaining[0] in _LEADING_GROUPING_WORDS:
            _pop_front()
            stripped_group = True
        while remaining and flags[-1] and (
            remaining[-1] in _TRAILING_GROUPING_TOKENS
            or (len(remaining) == 1 and remaining[-1] in _STANDALONE_END_WORDS)
        ):
            _pop_back()
            stripped_group = True
        if stripped_group:
            continue
        while remaining and re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", remaining[0]):
            if _is_dangerous_env_assignment(remaining[0]):
                unwrap_unverifiable = True
            _pop_front()
        if not remaining:
            break
        if flags[0] and remaining[0] in _LEADING_GROUPING_WORDS:
            continue

        head = Path(remaining[0]).name.lower()
        if head in _UNVERIFIABLE_WRAPPERS:
            unwrap_unverifiable = saw_opaque = True
            break

        if head == "env":
            _pop_front()
            while remaining:
                t = remaining[0]
                if t == "--":
                    _pop_front()
                    break
                if (
                    t in {"-S", "--split-string", "-C", "--chdir"}
                    or t.startswith(("--split-string=", "--chdir="))
                    or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "SC"))
                ):
                    unwrap_unverifiable = saw_opaque = True
                    break
                if t in {"-u", "--unset", "-a", "--argv0"}:
                    _pop_front()
                    if remaining:
                        _pop_front()
                    continue
                if t.startswith(("--unset=", "--argv0=")):
                    _pop_front()
                    continue
                if t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "ua"):
                    _consume_short_flag_cluster(t, "ua", remaining, _pop_front)
                    continue
                if t.startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", t):
                    if _is_dangerous_env_assignment(t):
                        unwrap_unverifiable = True
                    _pop_front()
                    continue
                break
            if saw_opaque:
                break
            continue

        if head in _SAFE_WRAPPER_SPECS:
            val_longs, val_shorts, trail_pos = _SAFE_WRAPPER_SPECS[head]
            _pop_front()
            while remaining:
                t = remaining[0]
                if t == "--":
                    _pop_front()
                    break
                if head == "time" and (
                    t in {"-o", "--output"}
                    or t.startswith("--output=")
                    or (t.startswith("-") and not t.startswith("--") and "o" in t[1:])
                ):
                    unwrap_unverifiable = saw_opaque = True
                if t in val_longs:
                    _pop_front()
                    if remaining:
                        _pop_front()
                    continue
                if t.startswith("-") and not t.startswith("--") and len(t) > 1 and val_shorts:
                    _consume_short_flag_cluster(t, val_shorts, remaining, _pop_front)
                    continue
                if t.startswith("-"):
                    _pop_front()
                    continue
                break
            for _ in range(trail_pos):
                if remaining:
                    _pop_front()
            continue

        if head in {"uv", "poetry", "pipenv"}:
            raw_head = _pop_front()
            while remaining:
                t = remaining[0]
                if t == "--":
                    break
                if t in _RUNNER_GLOBAL_VAL_FLAGS or t.startswith(
                    tuple(f + "=" for f in _RUNNER_GLOBAL_VAL_FLAGS if f.startswith("--"))
                ):
                    if t.split("=", 1)[0] != "--cache-dir":
                        unwrap_unverifiable = True
                    _pop_front()
                    if t in _RUNNER_GLOBAL_VAL_FLAGS and remaining:
                        _pop_front()
                    continue
                if t.startswith("-") and not t.startswith("--") and len(t) > 1:
                    if any(c in t[1:] for c in "Cp"):
                        unwrap_unverifiable = True
                    _consume_short_flag_cluster(t, "Cp", remaining, _pop_front)
                    continue
                if t.startswith("-"):
                    _pop_front()
                    continue
                break
            if remaining and remaining[0] == "run":
                _pop_front()
                while remaining:
                    t = remaining[0]
                    if t == "--":
                        _pop_front()
                        break
                    if t in _RUNNER_RUN_VAL_FLAGS or t.startswith(
                        tuple(f + "=" for f in _RUNNER_RUN_VAL_FLAGS if f.startswith("--"))
                    ):
                        if t.split("=", 1)[0] != "--cache-dir":
                            unwrap_unverifiable = True
                        _pop_front()
                        if t in _RUNNER_RUN_VAL_FLAGS and remaining:
                            _pop_front()
                        continue
                    if t.startswith("-") and not t.startswith("--") and len(t) > 1:
                        if any(c in t[1:] for c in "Csmp"):
                            unwrap_unverifiable = True
                        _consume_short_flag_cluster(t, "Csmp", remaining, _pop_front)
                        continue
                    if t.startswith("-"):
                        _pop_front()
                        continue
                    break
                continue
            if not (remaining and all(t in {"-V", "--version", "-h", "--help"} for t in remaining)):
                unwrap_unverifiable = True
            if out_arg_flags is not None:
                out_arg_flags.extend(flags)
            return _finish(head, raw_head, remaining, unwrap_unverifiable)

        if head in {"npx", "bunx"}:
            _pop_front()
            while remaining:
                t = remaining[0]
                if t == "--":
                    _pop_front()
                    break
                if t in {"-y", "--yes", "-q", "--quiet", "--no", "--no-install"} or (
                    t.startswith("-") and not t.startswith("--") and len(t) > 1 and set(t[1:]) <= {"y", "q"}
                ):
                    _pop_front()
                    continue
                if t in {"-c", "--call", "-p", "--package"} or t.startswith(("--call=", "--package=")):
                    unwrap_unverifiable = True
                    _pop_front()
                    if t in {"-c", "--call", "-p", "--package"} and remaining:
                        _pop_front()
                    continue
                if t.startswith("-") and not t.startswith("--") and len(t) > 1 and any(c in t[1:] for c in "cp"):
                    unwrap_unverifiable = True
                    _consume_short_flag_cluster(t, "cp", remaining, _pop_front)
                    continue
                if t.startswith("-"):
                    unwrap_unverifiable = True
                    _pop_front()
                    continue
                break
            if not remaining or Path(remaining[0]).name.lower() not in _SAFE_NPX_RUNNERS:
                unwrap_unverifiable = True
            continue
        break

    if not remaining or not remaining[0]:
        return _finish("", "", [], unwrap_unverifiable)
    raw_cmd_tok = remaining[0]
    if out_arg_flags is not None:
        out_arg_flags.extend(flags[1:])
    return _finish(Path(raw_cmd_tok).name.lower(), raw_cmd_tok, remaining[1:], unwrap_unverifiable)


def _unwrap_shell_command_tokens(
    tokens: List[str], unquoted_flags: Optional[List[bool]] = None
) -> Tuple[str, List[str], bool]:
    """Strip grouping tokens, leading KEY=VAL env assignments, and safe prefix wrappers."""
    verb, _, arg_tokens, unwrap_unverifiable = _unwrap_shell_command_tokens_with_raw(
        tokens, unquoted_flags
    )
    return verb, arg_tokens, unwrap_unverifiable


def _has_unquoted_cmd_subst(cmd: str) -> bool:
    """Return True if cmd contains unescaped $(, backtick, <(, or >( outside single/ANSI-C quotes."""
    cleaned, has_h_subst, _ = _preprocess_shell_heredocs_and_comments(cmd)
    if has_h_subst:
        return True
    if not cleaned:
        return False
    in_sq = in_ansi = in_dq = escaped = False
    i, n = 0, len(cleaned)
    while i < n:
        ch = cleaned[i]
        if escaped:
            escaped = False
        elif ch == "\\" and not in_sq:
            escaped = True
        elif not in_sq and not in_dq and not in_ansi and ch == "$" and i + 1 < n and cleaned[i + 1] == "'":
            in_ansi = True
            i += 2
            continue
        elif ch == "'" and not in_dq:
            if in_ansi:
                in_ansi = False
            else:
                in_sq = not in_sq
        elif ch == '"' and not in_sq and not in_ansi:
            in_dq = not in_dq
        elif not in_sq and not in_ansi:
            if ch == "`" or (ch == "$" and i + 1 < n and cleaned[i + 1] == "("):
                return True
            if not in_dq and ch in ("<", ">") and i + 1 < n and cleaned[i + 1] == "(":
                return True
        i += 1
    return False


def _has_unquoted_or_operator(cmd: str) -> bool:
    """Return True if cmd contains an unquoted shell '||' operator."""
    cmd = _preprocess_shell_heredocs_and_comments(cmd)[0]
    if not cmd:
        return False
    return any(cmd[i : i + 2] == "||" for i in _iter_unquoted_shell_indices(cmd))


def _strip_leading_env_assignments(cmd: str) -> str:
    if not cmd:
        return ""
    return _parse_leading_env_assignments(_fold_shell_line_continuations(cmd))[1]


def _strip_shell_quoted_literals(cmd: str) -> str:
    """Replace single, ANSI-C, and double-quoted shell literal contents while keeping unquoted syntax."""
    cmd = _preprocess_shell_heredocs_and_comments(cmd)[0]
    if not cmd:
        return ""
    out: List[str] = []
    in_sq = in_ansi = in_dq = escaped = False
    i, n = 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if escaped:
            if not (in_dq or in_ansi):
                out.append(ch)
            escaped = False
        elif ch == "\\" and not in_sq:
            if not (in_dq or in_ansi):
                out.append(ch)
            escaped = True
        elif not in_sq and not in_dq and not in_ansi and ch == "$" and i + 1 < n and cmd[i + 1] == "'":
            in_ansi = True
            out.append("''")
            i += 2
            continue
        elif ch == "'" and not in_dq:
            if in_ansi:
                in_ansi = False
            else:
                in_sq = not in_sq
                if in_sq:
                    out.append("''")
        elif ch == '"' and not in_sq and not in_ansi:
            in_dq = not in_dq
            if in_dq:
                out.append('""')
        elif not (in_sq or in_dq or in_ansi):
            out.append(ch)
        i += 1
    return "".join(out)


def _unquoted_group_delta(sc: str) -> int:
    """Return net change in unquoted ( / { vs ) / } grouping depth across a subcommand."""
    delta = 0
    for i in _iter_unquoted_shell_indices(sc):
        if sc[i] in ("(", "{"):
            delta += 1
        elif sc[i] in (")", "}"):
            delta -= 1
    return delta


def _unquoted_subshell_opens_closes(sc: str) -> Tuple[int, int]:
    """Count unquoted '(' and ')' subshell delimiters in a subcommand."""
    opens = closes = 0
    for i in _iter_unquoted_shell_indices(sc):
        if sc[i] == "(":
            opens += 1
        elif sc[i] == ")":
            closes += 1
    return opens, closes


def _advance_effective_cwd(
    sc: str, current_cwd: str, dir_stack: List[str], payload: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool, bool]:
    """Advance effective cwd for cd/pushd/popd subcommands, returning (new_cwd, is_cd_cmd, cd_unverifiable)."""
    tokens, unquoted_flags = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(sc))
    verb, args, unwrap_unv = _unwrap_shell_command_tokens(tokens, unquoted_flags)
    if verb not in {"cd", "pushd", "popd"}:
        return current_cwd, False, False
    if unwrap_unv:
        return current_cwd, True, True
    if verb == "popd":
        if any(t != "--" for t in args) or not dir_stack:
            return current_cwd, True, True
        return dir_stack.pop(), True, False
    after_dd = False
    pos_args: List[str] = []
    for tok in args:
        if not after_dd and tok == "--":
            after_dd = True
            continue
        if not after_dd and ((tok.startswith("-") and tok != "-") or (verb == "pushd" and tok.startswith("+"))):
            if verb == "cd" and re.fullmatch(r"-[LPe@]+", tok):
                continue
            return current_cwd, True, True
        pos_args.append(tok)
    if len(pos_args) != 1:
        return current_cwd, True, True
    arg = pos_args[0]
    if not arg or arg in {"-", "~"} or any(ch in arg for ch in "$`*?[]~{}"):
        return current_cwd, True, True
    if verb == "pushd":
        dir_stack.append(current_cwd)
    return _resolve_cmd_target(arg, current_cwd), True, False


def _cmd_enters_guard_plugin_cwd(sub_cmds: List[str], cwd: str) -> bool:
    """Return True if initial cwd or any subcommand's effective cwd enters the guard plugin directory."""
    effective_cwd = cwd or str(PROJECT_ROOT)
    if _is_guard_plugin_cwd(effective_cwd):
        return True
    dir_stack: List[str] = []
    subshell_stack: List[Tuple[str, List[str]]] = []
    for sc in sub_cmds:
        opens, closes = _unquoted_subshell_opens_closes(sc)
        for _ in range(opens):
            subshell_stack.append((effective_cwd, list(dir_stack)))
        new_cwd, is_cd_cmd, cd_unv = _advance_effective_cwd(sc, effective_cwd, dir_stack, None)
        if is_cd_cmd:
            effective_cwd = new_cwd
            if cd_unv or _is_guard_plugin_cwd(effective_cwd):
                return True
        for _ in range(closes):
            if subshell_stack:
                effective_cwd, dir_stack = subshell_stack.pop()
    return False


def _extract_inline_sh(verb: str, arg_tokens: List[str]) -> Optional[str]:
    """Extract inline script from sh/bash/zsh/csh/fish -c or eval for trivial-eval checks."""
    if verb == "eval":
        args = arg_tokens[1:] if (arg_tokens and arg_tokens[0] == "--") else arg_tokens
        return " ".join(args) if args else None
    after_dd = skip_next = saw_c = False
    for idx, tok in enumerate(arg_tokens):
        if skip_next:
            skip_next = False
            continue
        if not after_dd and tok == "--":
            after_dd = True
            continue
        if not after_dd and (tok.startswith("-") and tok != "-" or tok.startswith("+") and len(tok) > 1):
            if tok in {"-o", "+o", "-O", "+O", "--rcfile", "--init-file", "-p", "--profile", "-P", "--features"}:
                skip_next = True
            elif tok in {"--command", "--init-command"}:
                return arg_tokens[idx + 1] if idx + 1 < len(arg_tokens) else None
            elif tok.startswith(("--command=", "--init-command=")):
                return tok.split("=", 1)[1]
            elif not tok.startswith("--") and "c" in tok[1:].lower():
                saw_c = True
            continue
        return tok if saw_c else None
    return None


def _unwrap_trivial_eval_prefix(cmd: str) -> str:
    """Strip leading env assignments and benign prefix wrappers."""
    cur = _strip_leading_env_assignments(cmd)
    for _ in range(8):
        if not cur:
            return cur
        m_first = re.match(r"^(\S+)(?:\s+([\s\S]*))?$", cur)
        if not m_first:
            break
        head = os.path.basename(m_first.group(1)).lower()
        tail = (m_first.group(2) or "").strip()
        if head not in {"command", "builtin", "nohup", "env", "nice", "time", "timeout"}:
            break
        try:
            toks = shlex.split(tail) if head == "env" else tail.split()
        except ValueError:
            toks = tail.split()
        val_flags = {
            "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
            "nice": {"-n", "--adjustment"},
            "time": {"-o", "--output", "-f", "--format"},
            "timeout": {"-k", "--kill-after", "-s", "--signal"},
        }.get(head, set())
        idx = 0
        while idx < len(toks):
            tok = toks[idx]
            if tok == "--":
                idx += 1
                break
            if tok in val_flags:
                idx += 2
                continue
            if tok.startswith("-") or (head == "env" and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", tok)):
                idx += 1
                continue
            break
        if head == "timeout" and idx < len(toks):
            idx += 1
        cur = _strip_leading_env_assignments(" ".join(toks[idx:]))
    return cur


def _is_trivial_eval_subcmd(subcmd: str) -> bool:
    """Return True if a single shell sub-command is a trivial no-op, cd setup, or trivial shell/python -c wrapper."""
    stripped = _unwrap_trivial_eval_prefix(subcmd)
    if not stripped or TRIVIAL_EVAL_CMD_RE.match(stripped) or re.match(r"^cd\s+\S+$", stripped):
        return True
    unwrapped = _strip_leading_env_assignments(stripped)
    if TEST_CMD_START_RE.match(unwrapped) and not _is_test_command(unwrapped):
        return True
    toks = _tokenize_shell_tail(unwrapped)
    if toks:
        verb = Path(toks[0]).name.lower()
        args_t = toks[1:]
        if verb in (_SHELL_WRAPPER_VERBS - {"source", "."}):
            inner_sh = _extract_inline_sh(verb, args_t)
            if inner_sh:
                return _is_trivial_eval_command(inner_sh)
        elif re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", verb):
            after_dd = skip_next = False
            for idx, tok in enumerate(args_t):
                if skip_next:
                    skip_next = False
                    continue
                if not after_dd and tok == "--":
                    after_dd = True
                    continue
                if not after_dd and tok.startswith("-") and tok != "-":
                    if tok in {"-W", "-X", "--check-hash-based-pycs"}:
                        skip_next = True
                        continue
                    if not tok.startswith("--") and len(tok) > 1:
                        flags = tok[1:]
                        for k, ch in enumerate(flags):
                            if ch == "c":
                                code = flags[k + 1 :] or (args_t[idx + 1] if idx + 1 < len(args_t) else "")
                                if code:
                                    return _is_trivial_python_inline(code)
                                break
                            if ch in ("W", "X"):
                                if k == len(flags) - 1:
                                    skip_next = True
                                break
                    continue
                break
    return bool(TRIVIAL_TEST_FLAG_RE.search(stripped))


def _is_trivial_eval_command(eval_cmd: str) -> bool:
    """Return True if eval_cmd is empty, masks failure with '||'/masking, or runs only trivial no-ops."""
    if not eval_cmd or not eval_cmd.strip():
        return True
    if _has_unquoted_or_operator(eval_cmd) or _has_shell_masking_or_chaining(eval_cmd):
        return True
    parts = _split_shell_commands_with_delims(eval_cmd)
    if not parts:
        return True
    for idx, (_, part) in enumerate(parts):
        if not part or _is_trivial_eval_subcmd(part):
            continue
        next_delim = parts[idx + 1][0] if idx + 1 < len(parts) else None
        return next_delim in ("|", "|&", ";", "\n", "||", "&")
    return True


def _is_unsafe_eval_command(cmd_str: str) -> bool:
    """Return True if an EVAL_CMD invokes mutating shell verbs, shell/exec wrappers, or awk/sed (Finding #1)."""
    if not cmd_str:
        return False
    return bool(
        MUTATING_SHELL_CMD_RE.search(cmd_str)
        or SHELL_EXEC_WRAPPER_RE.search(cmd_str)
        or re.search(
            r"\b(?:[gmn]?awk|sed|perl|node|ruby|php|lua|(?:ba|da|z|k|a|c|tc|mk|rba)?sh|fish|eval|source)\b",
            cmd_str,
            re.I,
        )
    )


def _is_declared_eval_command(cmd: str, state: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Return True if cmd invokes a non-trivial, non-mutating declared EVAL_CMD without shell-chaining bypasses."""
    if (
        not cmd
        or "$(" in cmd
        or "`" in cmd
        or _has_unquoted_or_operator(cmd)
        or _has_shell_masking_or_chaining(cmd)
        or SHELL_EXEC_WRAPPER_RE.search(cmd)
        or _is_unsafe_eval_command(cmd)
    ):
        return False

    declared_raw: List[str] = []
    cid = str(payload.get("conversationId") or "").strip().lower()
    raw_subs = state.get("subagents")
    sub_entry = raw_subs.get(cid) if isinstance(raw_subs, dict) else None
    if isinstance(sub_entry, dict) and sub_entry.get("eval_cmd"):
        declared_raw.append(str(sub_entry["eval_cmd"]))
    self_meta = state.get("self_subagent_meta")
    if isinstance(self_meta, dict) and self_meta.get("eval_cmd"):
        declared_raw.append(str(self_meta["eval_cmd"]))
    if state.get("active_eval_cmd"):
        declared_raw.append(str(state["active_eval_cmd"]))
    for cohort_entry in (state.get("cohorts") or {}).values():
        if isinstance(cohort_entry, dict) and cohort_entry.get("eval_cmd"):
            declared_raw.append(str(cohort_entry["eval_cmd"]))

    declared = [
        d
        for raw_d in declared_raw
        if isinstance(raw_d, str) and raw_d.strip()
        for d in (_strip_leading_env_assignments(raw_d),)
        if d and not _is_trivial_eval_command(d) and not _is_unsafe_eval_command(d)
    ]
    if not declared:
        return False

    parts = _split_shell_commands_with_delims(cmd)
    if not parts:
        return False
    matched_eval = False
    cur_eval_dir = PROJECT_ROOT.resolve()
    for idx, (_, sc) in enumerate(parts):
        assignments, remainder = _parse_leading_env_assignments(_fold_shell_line_continuations(sc))
        if any(_is_dangerous_env_assignment(f"{k}{'+=' if is_app else '='}{v}") for k, is_app, v in assignments):
            return False
        stripped = remainder.strip()
        if not stripped:
            continue
        matched_decl = next(
            (decl for decl in declared if stripped == decl or stripped.startswith(decl + " ")),
            None,
        )
        if matched_decl is not None:
            if _is_trivial_eval_subcmd(stripped) or TRIVIAL_TEST_FLAG_RE.search(stripped[len(matched_decl) :]):
                return False
            next_delim = parts[idx + 1][0] if idx + 1 < len(parts) else None
            if next_delim in ("|", "|&", ";", "\n", "||", "&"):
                return False
            matched_eval = True
            continue
        cd_toks = _tokenize_shell_tail(stripped)
        if len(cd_toks) == 2 and cd_toks[0] == "cd":
            cd_arg = cd_toks[1]
            if (
                not cd_arg
                or cd_arg.startswith(("-", "~"))
                or any(ch in cd_arg for ch in "$`*?[]{}")
                or re.search(r"(?:^|/)(?:tmp|scratch|brain)(?:/|$)", cd_arg, re.I)
            ):
                return False
            try:
                next_dir = (cur_eval_dir / cd_arg).resolve()
            except (OSError, ValueError):
                return False
            proj_res = PROJECT_ROOT.resolve()
            if (next_dir != proj_res and proj_res not in next_dir.parents) or _is_staging_or_tmp_path(str(next_dir), {}):
                return False
            cur_eval_dir = next_dir
            continue
        return False
    return matched_eval

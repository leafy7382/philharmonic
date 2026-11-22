"""Shell lexing, redirection extraction, wrapper unwrapping, and eval inspection."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from guard.constants import PROJECT_ROOT, TEST_CMD_START_RE, TRIVIAL_EVAL_CMD_RE, TRIVIAL_TEST_FLAG_RE
from guard.path_guards import _is_guard_plugin_cwd, _is_staging_or_tmp_path, _resolve_cmd_target, _sanitize_decoded_shell_token

SAFE_XARGS_READONLY_CMDS = frozenset({"grep", "egrep", "fgrep", "rg", "wc", "cat", "head", "tail", "ls", "stat", "file", "echo", "printf", "basename", "dirname", "md5sum", "sha256sum", "sha1sum", "cksum", "du", "realpath", "readlink"})
SAFE_READONLY_SHELL_VERBS = frozenset({"ls", "dir", "vdir", "pwd", "echo", "cat", "tac", "nl", "head", "tail", "wc", "grep", "egrep", "fgrep", "zgrep", "stat", "file", "basename", "dirname", "realpath", "readlink", "md5sum", "sha1sum", "sha256sum", "sha512sum", "cksum", "sum", "du", "df", "id", "whoami", "uname", "date", "which", "type", "true", "false", ":", "cut", "tr", "paste", "comm", "diff", "cmp", "rev", "fold", "fmt", "expand", "unexpand", "column", "strings", "od", "hexdump", "jq", "sleep", "printenv", "hostname", "uptime", "nproc", "getconf", "seq", "expr"})
_SHELL_WRAPPER_VERBS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "ash", "csh", "tcsh", "fish", "mksh", "rbash", "eval", "source", "."})
_UNVERIFIABLE_WRAPPERS = frozenset({"flock", "script", "stdbuf", "parallel", "busybox", "strace", "ltrace", "chrt", "ionice", "taskset", "unshare", "nsenter", "chroot", "watch", "prlimit", "setarch", "linux32", "linux64", "numactl", "setsid", "ed", "ex", "uvx"})
_OPAQUE_EXEC_VERBS = _UNVERIFIABLE_WRAPPERS | {"bun", "deno", "expect", "tclsh", "wish", "rscript", "julia", "pwsh", "sqlite3"}
_SAFE_NPX_RUNNERS = frozenset({"jest", "vitest", "mocha", "ava", "tap", "playwright", "tsc", "pyright", "mypy"})
_DANGEROUS_ENV_VARS = frozenset({"GCONV_PATH", "HOSTALIASES", "LOCPATH", "NLSPATH", "ENV", "BASH_ENV", "CDPATH", "IFS", "PROMPT_COMMAND", "PS4", "FPATH", "ZDOTDIR", "SHELL", "PATH", "GLOBIGNORE", "GIT_EXTERNAL_DIFF", "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_GLOBAL", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_PROXY_COMMAND", "GIT_UPLOAD_PACK", "GIT_RECEIVE_PACK", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_SSL_NO_VERIFY", "GIT_SSL_CAINFO", "GIT_SSL_CAPATH", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "GIT_ALLOW_PROTOCOL", "GNUPGHOME", "SSLKEYLOGFILE", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "all_proxy", "EDITOR", "VISUAL", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONEXECUTABLE", "PYTHONUSERBASE", "PYTHONINSPECT", "PYTHONWARNINGS", "PYTHONBREAKPOINT", "PYTHONCOERCECLOCALE", "PYTHONDEVMODE", "PYTHONMALLOC", "PERL5LIB", "PERLLIB", "PERL5OPT", "RUBYLIB", "RUBYOPT", "NODE_OPTIONS", "NODE_PATH", "AWKPATH", "AWKLIBPATH", "LUA_PATH", "LUA_CPATH", "HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "GIT_CONFIG_NOSYSTEM", "GIT_ATTR_SOURCE", "GIT_GRAFT_FILE", "GIT_NOTES_REF", "GIT_REPLACE_REF_BASE", "HISTFILE", "LESSOPEN", "LESSCLOSE", "MANPAGER", "BROWSER", "TMPDIR", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "GOFLAGS", "GOENV", "CC", "CXX", "LD", "AR", "AS", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_HOME", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "CARGO_TARGET_DIR"})
_DANGEROUS_ENV_PREFIXES = ("LD_", "DYLD_", "GIT_TRACE", "LUA_INIT", "NPM_CONFIG_", "YARN_", "PNPM_", "CARGO_TARGET_", "CARGO_REGISTRIES_", "UV_", "PIP_", "POETRY_", "PIPENV_", "PYTEST_")
_SAFE_GIT_EDITOR_VALS = frozenset({"", "true", "/bin/true", "/usr/bin/true", ":", "false", "/bin/false"})
_SAFE_PAGER_VALS = frozenset({"", "cat", "/bin/cat", "/usr/bin/cat", "true", "/bin/true", "/usr/bin/true", ":"})
_ANSI_C_ESCAPES = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}
_TOKEN_REDIR_OP_RE = re.compile(r"(?:&>>|&>|>>|>\||>&|<<<|<<-|<<|<&|<(?!\()|>(?!\())")
_SHELL_MASKING_TAIL_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*(?:(?:(?:/(?:usr/)?bin/)?(?:true|echo|printf|cat|pwd|ls|sleep)|exit(?:\s+0)?)\b|:(?=$|\s))", re.I)
_LEADING_GROUPING_WORDS = frozenset({"(", "{", "!", "if", "while", "until", "then", "do", "else", "elif", "coproc"})
_TRAILING_GROUPING_TOKENS = frozenset({")", "}"})
_STANDALONE_END_WORDS = frozenset({"fi", "done", "esac"})
_PY_VERB_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)$")
_MUTATING_EVAL_VERBS = frozenset({"rm", "unlink", "shred", "mv", "cp", "tee", "touch", "truncate", "dd", "patch", "install", "ln", "chmod", "chown", "chgrp", "mkdir", "rmdir", "rsync", "split", "csplit", "link", "mktemp", "mkfifo", "mknod", "wget", "dos2unix", "unix2dos", "fallocate", "sponge", "tar", "unzip", "xargs"})
_INTERPRETER_EVAL_RE = re.compile(r"^(?:[gmn]?awk|sed|(?:perl|node|ruby|php|lua)(?:\d+(?:\.\d+)*)?)$")


def _decode_ansi_c_escape(text: str, idx: int, *, nul_on_surrogate: bool = False) -> Tuple[str, int]:
    """Decode one ANSI-C $'...' backslash escape starting at idx."""
    n = len(text)
    if idx + 1 >= n:
        return "\\", idx + 1
    nxt = text[idx + 1]
    if nxt in _ANSI_C_ESCAPES:
        return _ANSI_C_ESCAPES[nxt], idx + 2
    if nxt == "c" and idx + 2 < n and text[idx + 2] != "'":
        cch, step = (text[idx + 3], 4) if (text[idx + 2] == "\\" and idx + 3 < n and text[idx + 3] in ("\\", "'")) else (text[idx + 2], 3)
        return ("\x00" if cch in ("@", " ", "\x00") else chr(0x7F if cch == "?" else (ord(cch) & 0x1F))), idx + step
    if nxt in ("x", "u", "U"):
        j, limit = idx + 2, idx + (4 if nxt == "x" else (6 if nxt == "u" else 10))
        while j < n and j < limit and text[j] in "0123456789abcdefABCDEF":
            j += 1
        if j <= idx + 2:
            return "\\" + nxt, idx + 2
        cp = int(text[idx + 2 : j], 16)
        return (chr(cp & 0xFF), j) if nxt == "x" else (("\x00" if nul_on_surrogate else "\ufffd", j) if (0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF) else (chr(cp), j))
    if nxt in "01234567":
        j = idx + 1
        while j < n and j < idx + 4 and text[j] in "01234567":
            j += 1
        return chr(int(text[idx + 1 : j], 8) & 0xFF), j
    return nxt, idx + 2


def _read_quoted_span(text: str, i: int, *, nul_on_surrogate: bool = False, stop_on_nul: bool = False) -> Tuple[str, int, bool]:
    """Read a $'...', '...', or "..." span starting at text[i]; return (decoded, next_idx, closed)."""
    n, chars = len(text), []
    if text[i : i + 2] == "$'":
        j, trunc = i + 2, False
        while j < n and text[j] != "'":
            dec, j = _decode_ansi_c_escape(text, j, nul_on_surrogate=nul_on_surrogate) if text[j] == "\\" else (text[j], j + 1)
            if stop_on_nul and "\x00" in dec:
                trunc = True
            elif not trunc:
                chars.append(dec)
        return "".join(chars), (j + 1 if j < n else n), j < n
    q, j = text[i], i + 1
    while j < n and text[j] != q:
        if q == '"' and text[j] == "\\" and j + 1 < n and text[j + 1] in ('"', "\\", "$", "`", "\n", "\r"):
            step = 3 if (text[j + 1] == "\r" and j + 2 < n and text[j + 2] == "\n") else 2
            if step == 2 and text[j + 1] != "\n":
                chars.append(text[j + 1])
            j += step
        else:
            chars.append(text[j])
            j += 1
    return "".join(chars), (j + 1 if j < n else n), j < n


def _iter_shell_spans(text: str) -> Iterator[Tuple[str, int, int, bool]]:
    """Yield (kind, start, end, closed) across shell text; kind in {'esc','ansi','sq','dq','raw'}."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            yield "esc", i, min(n, i + 2), i + 1 < n
            i += 2
        elif text[i : i + 2] == "$'" or ch in ("'", '"'):
            _, end, closed = _read_quoted_span(text, i)
            yield ("ansi" if text[i] == "$" else ("sq" if ch == "'" else "dq")), i, end, closed
            i = end
        else:
            j = i + 1
            while j < n and (text[j] not in "\\'\"$" or (text[j] == "$" and (j + 1 >= n or text[j + 1] != "'"))):
                j += 1
            yield "raw", i, j, True
            i = j


def _fold_shell_line_continuations(text: str) -> str:
    """Fold unquoted and double-quoted shell line continuations while preserving '...' and $'...' verbatim."""
    if not text or ("\\\n" not in text and "\\\r\n" not in text):
        return text
    joined = "".join((re.sub(r"(?<!\\)((?:\\\\)*)\\\r?\n", r"\1", text[s:e]) if k == "dq" else text[s:e]) for k, s, e, _ in _iter_shell_spans(text) if not (k == "esc" and s + 1 < len(text) and (text[s + 1] == "\n" or (text[s + 1] == "\r" and s + 2 < len(text) and text[s + 2] == "\n"))))
    return re.sub(r"(?<!\\)((?:\\\\)*)\\\r\n", r"\1", joined) if "\\\r\n" in text else joined


def _iter_unquoted_shell_indices(text: str) -> Iterator[int]:
    for kind, s, e, _ in _iter_shell_spans(text):
        if kind == "raw":
            yield from range(s, e)


def _has_nul_or_surrogate_input(text: str) -> bool:
    """Return True if text contains raw NUL, lone UTF-16 surrogates, or ANSI-C NUL/surrogate escapes."""
    if not text:
        return False
    if not isinstance(text, str) or "\x00" in text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return True
    folded = _fold_shell_line_continuations(text)
    return any("\x00" in _read_quoted_span(folded, s, nul_on_surrogate=True)[0] for kind, s, _, _ in _iter_shell_spans(folded) if kind == "ansi")


def _preprocess_shell_heredocs_and_comments(cmd: str) -> Tuple[str, bool, bool]:
    """Strip unquoted shell comments and heredoc bodies, returning (cleaned_cmd, has_heredoc_cmd_subst, has_unclosed_syntax)."""
    if not cmd:
        return "", False, False
    out, pending, has_h_subst, has_unclosed, in_dq, i, n = [], [], False, False, False, 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if ch == "\\":
            if i + 1 < n and (cmd[i + 1] == "\n" or (cmd[i + 1] == "\r" and i + 2 < n and cmd[i + 2] == "\n")):
                i += 3 if cmd[i + 1] == "\r" else 2; continue
            has_unclosed = has_unclosed or (i + 1 >= n); out.append(cmd[i : i + 2]); i += 2
        elif not in_dq and (cmd[i : i + 2] == "$'" or ch == "'"):
            _, end, closed = _read_quoted_span(cmd, i)
            has_unclosed = has_unclosed or not closed; out.append(cmd[i:end]); i = end
        elif ch == '"':
            in_dq, _ = not in_dq, out.append(ch); i += 1
        elif not in_dq and ch == "#" and (not out or out[-1] in " \t\n;|&()"):
            while i < n and cmd[i] != "\n": i += 1
        elif not in_dq and cmd[i : i + 2] == "<<" and (i + 2 >= n or cmd[i + 2] != "<"):
            out.append("<<"); i += 2
            strip_tabs = i < n and cmd[i] == "-"
            if strip_tabs: out.append("-"); i += 1
            while i < n and cmd[i] in " \t": out.append(cmd[i]); i += 1
            dparts, was_q, bad_d = [], False, False
            while i < n and cmd[i] not in " \t\r\n;|&()<>":
                if cmd[i] == "\\" and i + 1 < n:
                    was_q, _, i = True, (dparts.append(cmd[i + 1]), out.append(cmd[i : i + 2])), i + 2
                elif cmd[i : i + 2] == "$'" or cmd[i] in ("'", '"'):
                    dec, end, closed = _read_quoted_span(cmd, i)
                    was_q, bad_d, _, i = True, bad_d or not closed, (dparts.append(dec), out.append(cmd[i:end])), end
                else:
                    dparts.append(cmd[i]); out.append(cmd[i]); i += 1
            delim = "".join(dparts)
            if bad_d or not delim or (not was_q and ("$" in delim or "`" in delim)): has_unclosed = True
            else: pending.append((delim, strip_tabs, was_q))
        elif not in_dq and ch == "\n":
            out.append("\n"); i += 1
            for delim, strip_tabs, was_q in pending:
                matched, body_lines = False, []
                while i < n:
                    le = cmd.find("\n", i)
                    raw_line, i = (cmd[i:], n) if le == -1 else (cmd[i:le], le + 1)
                    if (raw_line.lstrip("\t") if strip_tabs else raw_line) == delim: matched = True; break
                    body_lines.append(raw_line)
                if not was_q and body_lines and re.search(r"(?<!\\)(?:\\\\)*(?:`|\$\()", re.sub(r"(?<!\\)((?:\\\\)*)\\\r?\n", r"\1", "\n".join(body_lines))):
                    has_h_subst = True
                has_unclosed = has_unclosed or not matched
            pending.clear()
        else:
            out.append(ch); i += 1
    return "".join(out), has_h_subst, has_unclosed or in_dq or bool(pending)


def _split_shell_commands_with_delims(cmd: str) -> List[Tuple[Optional[str], str]]:
    """Split a shell command string on unquoted &&, ||, ;, |, |&, &, and newline delimiters, tracking preceding delimiter."""
    if not cmd or not (cleaned := _preprocess_shell_heredocs_and_comments(cmd)[0]):
        return []
    parts, prec, last_start, skip_until, n = [], None, 0, 0, len(cleaned)
    for i in _iter_unquoted_shell_indices(cleaned):
        if i < skip_until:
            continue
        ch = cleaned[i]
        delim, step = ((";;&", 3) if cleaned[i : i + 3] == ";;&" else ((cleaned[i : i + 2], 2) if cleaned[i : i + 2] in ("&&", "||", "|&", ";;", ";&") else (("&", 1) if ch == "&" and not ((i > 0 and cleaned[i - 1] in (">", "<")) or (i + 1 < n and cleaned[i + 1] == ">")) else ((ch, 1) if ch in (";", "|", "\n") else (None, 1)))))
        if delim is not None:
            part = cleaned[last_start:i].strip()
            if part or prec == "&":
                parts.append((prec, part))
            if not (ch == "\n" and part == "" and prec in (None, "|", "|&", "&&", "||")):
                prec = delim
            last_start = skip_until = i + step
    tail = cleaned[last_start:].strip()
    if tail or prec == "&":
        parts.append((prec, tail))
    return parts


def _split_shell_commands(cmd: str) -> List[str]:
    return [part for _, part in _split_shell_commands_with_delims(cmd) if part]


def _has_shell_masking_or_chaining(cmd: str) -> bool:
    return bool(cmd) and any(delim in ("||", "&") or (delim in ("|", "|&", ";", "\n") and part and _SHELL_MASKING_TAIL_RE.match(part)) for delim, part in _split_shell_commands_with_delims(cmd))


def _split_shell_tokens_raw_with_meta(text: str) -> List[Tuple[str, bool, bool]]:
    """Lex a shell string into (dequoted_token, is_unquoted_redir_op, is_unquoted_token) tuples."""
    if not (text := _fold_shell_line_continuations(_preprocess_shell_heredocs_and_comments(text)[0])):
        return []
    tokens, current, has_token, saw_quote, i, n = [], [], False, False, 0, len(text)

    def _flush() -> None:
        nonlocal current, has_token, saw_quote
        if has_token:
            tokens.append((_sanitize_decoded_shell_token("".join(current)), False, not saw_quote))
            current, has_token, saw_quote = [], False, False

    while i < n:
        ch = text[i]
        if ch in " \t\r\n": _flush()
        elif ch == "\\" and i + 1 < n:
            if text[i + 1] != "\n":
                saw_quote, has_token = saw_quote or text[i + 1] in ("(", ")", "{", "}", "!"), True
                current.append(text[i + 1])
            i += 2; continue
        elif text[i : i + 2] == "$'" or ch in ("'", '"'):
            dec, i, _ = _read_quoted_span(text, i, stop_on_nul=True)
            current.append(dec); has_token = saw_quote = True; continue
        elif ch in ("$", "<", ">") and i + 1 < n and text[i + 1] == "(":
            current.append(ch + "("); has_token, i = True, i + 2; continue
        elif ch in ("(", ")"):
            _flush(); tokens.append((ch, False, True))
        elif not has_token and ((ch in ("{", "}") and (i + 1 >= n or text[i + 1] in " \t\r\n()")) or (ch == "!" and (i + 1 >= n or text[i + 1] in " \t\r\n({"))):
            tokens.append((ch, False, True))
        elif ((ch in (">", "<") and not (i + 1 < n and text[i + 1] == "(")) or (ch == "&" and i + 1 < n and text[i + 1] == ">")) and (m_op := _TOKEN_REDIR_OP_RE.match(text, i)):
            op = m_op.group(0)
            if has_token and not saw_quote and current and "".join(current).isdigit() and not op.startswith("&"):
                op, current, has_token, saw_quote = "".join(current) + op, [], False, False
            else:
                _flush()
            tokens.append((op, True, True)); i = m_op.end(); continue
        else:
            current.append(ch); has_token = True
        i += 1
    _flush()
    return tokens


def _split_shell_tokens_raw(text: str) -> List[Tuple[str, bool]]:
    return [(tok, is_redir) for tok, is_redir, _ in _split_shell_tokens_raw_with_meta(text)]


def _split_shell_tokens(text: str) -> List[str]:
    return [tok for tok, _, _ in _split_shell_tokens_raw_with_meta(text)]


def _tokenize_shell_tail(text: str) -> List[str]:
    return _split_shell_tokens(text)


def _extract_shell_redir_targets(cmd: str) -> List[str]:
    """Extract write targets from unquoted shell redirections (>, >>, >|, &>, N>)."""
    raw, targets = _split_shell_tokens_raw_with_meta(cmd), []
    for idx, (op, is_redir, _) in enumerate(raw):
        if not is_redir or ">" not in op or idx + 1 >= len(raw):
            continue
        sink, sink_redir, sink_uq = raw[idx + 1]
        if sink_redir:
            continue
        sink = sink[:-1].strip() if sink_uq and sink.endswith("}") and "{" not in sink else sink
        is_fd_dup = op.endswith(">&") and sink_uq and (sink.isdigit() or sink == "-" or (sink.endswith("-") and sink[:-1].isdigit()))
        if not is_fd_dup and sink and sink != "/dev/null" and not (sink_uq and sink in {"&1", "&2"}):
            targets.append(sink)
    return targets


def _strip_redir_tokens_with_meta(raw_tokens: List[Tuple[str, bool, bool]]) -> Tuple[List[str], List[bool]]:
    cleaned, unquoted, skip_next = [], [], False
    for tok, is_redir, is_unquoted in raw_tokens:
        if skip_next:
            skip_next = False
        elif is_redir:
            skip_next = True
        else:
            cleaned.append(tok); unquoted.append(is_unquoted)
    return cleaned, unquoted


def _parse_leading_env_assignments(cmd: str) -> Tuple[List[Tuple[str, bool, str]], str]:
    """Parse leading shell KEY=VAL / KEY+=VAL assignments in a quote-aware manner."""
    assignments, i, n = [], 0, len(cmd)
    while i < n:
        while i < n and cmd[i] in " \t\r\n":
            i += 1
        if not (m := re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\+?=)", cmd[i:])):
            break
        key, is_append, val_chars, i = m.group(1), m.group(2) == "+=", [], i + m.end()
        while i < n and cmd[i] not in " \t\r\n;|&<>()":
            if cmd[i : i + 2] == "$'" or cmd[i] in ("'", '"'):
                dec, i, _ = _read_quoted_span(cmd, i)
                val_chars.append(dec)
            elif cmd[i] == "\\" and i + 1 < n:
                if cmd[i + 1] in ("\n", "\r"):
                    i += 3 if (cmd[i + 1] == "\r" and i + 2 < n and cmd[i + 2] == "\n") else 2
                else:
                    val_chars.append(cmd[i + 1]); i += 2
            else:
                val_chars.append(cmd[i]); i += 1
        assignments.append((key, is_append, "".join(val_chars)))
    return assignments, cmd[i:].strip()


def _strip_leading_env_assignments(cmd: str) -> str:
    return _parse_leading_env_assignments(_fold_shell_line_continuations(cmd))[1] if cmd else ""


def _is_dangerous_env_assignment(tok: str) -> bool:
    """Return True if a KEY=VAL or KEY+=VAL assignment injects preload/shell/git/interpreter code-exec risk."""
    name, eq, val = tok.partition("=")
    if not eq:
        return False
    is_append = name.endswith("+")
    name = name[:-1] if is_append else name
    if name.startswith("BASH_FUNC_"):
        return True
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return False
    upper = name.upper()
    if is_append and (name in _DANGEROUS_ENV_VARS or upper in _DANGEROUS_ENV_VARS or upper.startswith(_DANGEROUS_ENV_PREFIXES) or name in {"PYTHONPATH", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_PAGER", "PAGER"}):
        return True
    if name in {"GIT_EDITOR", "GIT_SEQUENCE_EDITOR"}:
        return val not in _SAFE_GIT_EDITOR_VALS
    if name in {"GIT_PAGER", "PAGER"}:
        return val not in _SAFE_PAGER_VALS
    return any(seg not in {"", "."} for seg in val.split(":")) if name == "PYTHONPATH" else (upper.startswith(_DANGEROUS_ENV_PREFIXES) or name in _DANGEROUS_ENV_VARS or upper in _DANGEROUS_ENV_VARS)


def _consume_short_flag_cluster(tok: str, value_chars: str, remaining: List[str], pop_fn: Callable[[], str]) -> bool:
    pop_fn()
    for k, ch in enumerate(tok[1:]):
        if ch in value_chars:
            if k == len(tok[1:]) - 1 and remaining:
                pop_fn()
            return True
    return False


_SAFE_WRAPPER_SPECS: Dict[str, Tuple[frozenset[str], str, int]] = {"nohup": (frozenset(), "", 0), "command": (frozenset(), "", 0), "builtin": (frozenset(), "", 0), "exec": (frozenset({"-a"}), "a", 0), "nice": (frozenset({"-n", "--adjustment"}), "n", 0), "timeout": (frozenset({"-k", "--kill-after", "-s", "--signal"}), "sk", 1), "time": (frozenset({"-o", "--output", "-f", "--format"}), "of", 0)}
_RUNNER_GLOBAL_VAL_FLAGS = frozenset({"--directory", "-C", "--project", "--config-file", "--python", "-p", "--cache-dir", "--index", "--default-index", "--index-url", "--extra-index-url", "--find-links"})
_RUNNER_RUN_VAL_FLAGS = _RUNNER_GLOBAL_VAL_FLAGS | frozenset({"--with", "--with-editable", "--with-requirements", "--from", "--script", "-s", "--module", "-m", "--gui-script", "--env-file"})


def _unwrap_shell_command_tokens_with_raw(
    tokens: List[str], unquoted_flags: Optional[List[bool]] = None, out_arg_flags: Optional[List[bool]] = None, out_meta: Optional[Dict[str, bool]] = None,
) -> Tuple[str, str, List[str], bool]:
    """Strip grouping tokens, leading KEY=VAL env assignments, and safe prefix wrappers."""
    rem, flags, unv, opaque = list(tokens), (list(unquoted_flags) if unquoted_flags is not None else [True] * len(tokens)), False, False
    _pop = lambda front=True: (flags.pop(0 if front else -1) if flags else None, rem.pop(0 if front else -1))[1]

    def _consume_flags(val_flags: frozenset[str], short_vals: str, *, stop_dd: bool = True, mark_unv_on: Any = None) -> None:
        nonlocal unv
        eq_p = tuple(f + "=" for f in val_flags if f.startswith("--"))
        while rem:
            t = rem[0]
            if t == "--":
                if stop_dd: _pop()
                break
            if t in val_flags or (eq_p and t.startswith(eq_p)):
                unv = unv or bool(mark_unv_on and mark_unv_on(t)); _pop()
                if t in val_flags and rem: _pop()
            elif t.startswith("-") and not t.startswith("--") and len(t) > 1 and short_vals:
                unv = unv or bool(mark_unv_on and any(c in t[1:] for c in short_vals) and mark_unv_on(t))
                _consume_short_flag_cluster(t, short_vals, rem, _pop)
            elif t.startswith("-"):
                unv = unv or bool(mark_unv_on and mark_unv_on(t)); _pop()
            else: break

    while rem:
        if (flags[0] and rem[0] in _LEADING_GROUPING_WORDS) or (flags[-1] and (rem[-1] in _TRAILING_GROUPING_TOKENS or (len(rem) == 1 and rem[-1] in _STANDALONE_END_WORDS))):
            while rem and flags[0] and rem[0] in _LEADING_GROUPING_WORDS: _pop()
            while rem and flags[-1] and (rem[-1] in _TRAILING_GROUPING_TOKENS or (len(rem) == 1 and rem[-1] in _STANDALONE_END_WORDS)): _pop(False)
            continue
        while rem and re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", rem[0]): unv = unv or _is_dangerous_env_assignment(_pop())
        if not rem or (flags[0] and rem[0] in _LEADING_GROUPING_WORDS): continue
        head = Path(rem[0]).name.lower()
        if head in _UNVERIFIABLE_WRAPPERS: unv = opaque = True; break
        if head == "env":
            _pop()
            while rem and rem[0] != "--":
                t = rem[0]
                if t in {"-S", "--split-string", "-C", "--chdir"} or t.startswith(("--split-string=", "--chdir=")) or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "SC")):
                    unv = opaque = True; break
                if t in {"-u", "--unset", "-a", "--argv0"} or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "ua")):
                    _consume_short_flag_cluster(t, "ua", rem, _pop) if not t.startswith("--") and t not in {"-u", "-a"} else (_pop(), _pop() if rem else None)
                elif t.startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", t): unv = unv or _is_dangerous_env_assignment(_pop())
                else: break
            if opaque: break
            if rem and rem[0] == "--": _pop()
        elif spec := _SAFE_WRAPPER_SPECS.get(head):
            _pop()
            if head == "time" and any(t in {"-o", "--output"} or t.startswith("--output=") or (t.startswith("-") and not t.startswith("--") and "o" in t[1:]) for t in rem[: (rem.index("--") if "--" in rem else len(rem))]):
                unv = opaque = True
            _consume_flags(spec[0], spec[1])
            for _ in range(spec[2]):
                if rem: _pop()
        elif head in {"uv", "poetry", "pipenv"}:
            raw_head = _pop()
            _consume_flags(_RUNNER_GLOBAL_VAL_FLAGS, "Cp", stop_dd=False, mark_unv_on=lambda t: t.split("=", 1)[0] != "--cache-dir" and (t in _RUNNER_GLOBAL_VAL_FLAGS or "=" in t or any(c in t[1:] for c in "Cp")))
            if rem and rem[0] == "run":
                _pop(); _consume_flags(_RUNNER_RUN_VAL_FLAGS, "Csmp", mark_unv_on=lambda t: t.split("=", 1)[0] != "--cache-dir" and (t in _RUNNER_RUN_VAL_FLAGS or "=" in t or any(c in t[1:] for c in "Csmp")))
                continue
            unv = unv or not (rem and all(t in {"-V", "--version", "-h", "--help"} for t in rem))
            if out_arg_flags is not None: out_arg_flags.extend(flags)
            if out_meta is not None: out_meta["opaque"] = opaque
            return head, raw_head, rem, unv
        elif head in {"npx", "bunx"}:
            _pop()
            _consume_flags(frozenset({"-c", "--call", "-p", "--package"}), "cp", mark_unv_on=lambda t: not (t in {"-y", "--yes", "-q", "--quiet", "--no", "--no-install"} or (t.startswith("-") and not t.startswith("--") and len(t) > 1 and set(t[1:]) <= {"y", "q"})))
            unv = unv or not rem or Path(rem[0]).name.lower() not in _SAFE_NPX_RUNNERS
        else: break
    if out_meta is not None: out_meta["opaque"] = opaque
    if not rem or not rem[0]: return "", "", [], unv
    if out_arg_flags is not None: out_arg_flags.extend(flags[1:])
    return Path(rem[0]).name.lower(), rem[0], rem[1:], unv


def _unwrap_shell_command_tokens(tokens: List[str], unquoted_flags: Optional[List[bool]] = None) -> Tuple[str, List[str], bool]:
    verb, _, arg_tokens, unv = _unwrap_shell_command_tokens_with_raw(tokens, unquoted_flags)
    return verb, arg_tokens, unv


def _parse_python_cli(arg_tokens: List[str]) -> Tuple[bool, str, str, List[str]]:
    """Parse Python interpreter CLI flags, returning (ok_flags, mode, target_or_code, rest_args)."""
    after_dd = skip_next = False
    for idx, tok in enumerate(arg_tokens):
        if skip_next:
            skip_next = False
        elif not after_dd and tok == "--":
            after_dd = True
        elif not after_dd and tok.startswith("-") and tok != "-":
            if tok in {"-W", "-X", "--check-hash-based-pycs"}:
                skip_next = True
            elif tok.startswith("--") and tok not in {"--version", "--help"}:
                return False, "none", "", []
            elif not tok.startswith("--"):
                for k, ch in enumerate(tok[1:]):
                    if ch in ("c", "m"):
                        rest = tok[1:][k + 1 :]
                        val = rest or (arg_tokens[idx + 1] if idx + 1 < len(arg_tokens) else "")
                        return bool(val), ch, val, arg_tokens[idx + (1 if rest else 2) :]
                    if ch in "WX":
                        skip_next = k == len(tok[1:]) - 1; break
                    if ch not in "BuOqsSEIvhV":
                        return False, "none", "", []
        else:
            return not skip_next, "script", tok, arg_tokens[idx + 1 :]
    return (False, "none", "", []) if skip_next else ((True, "meta", "", []) if (arg_tokens and all(t in {"--version", "-V", "--help", "-h"} for t in arg_tokens)) else (True, "none", "", []))


def _has_unquoted_cmd_subst(cmd: str) -> bool:
    cleaned, has_h_subst, _ = _preprocess_shell_heredocs_and_comments(cmd)
    return has_h_subst or (bool(cleaned) and any(("`" in cleaned[s:e] or "$(" in cleaned[s:e] or (k == "raw" and ("<(" in cleaned[s:e] or ">(" in cleaned[s:e]))) for k, s, e, _ in _iter_shell_spans(cleaned) if k in ("raw", "dq")))


def _has_unquoted_or_operator(cmd: str) -> bool:
    cleaned = _preprocess_shell_heredocs_and_comments(cmd)[0]
    return bool(cleaned) and any(cleaned[i : i + 2] == "||" for i in _iter_unquoted_shell_indices(cleaned))


def _strip_shell_quoted_literals(cmd: str) -> str:
    cleaned = _preprocess_shell_heredocs_and_comments(cmd)[0]
    return "".join('""' if k == "dq" else ("''" if k in ("sq", "ansi") else cleaned[s:e]) for k, s, e, _ in _iter_shell_spans(cleaned)) if cleaned else ""


def _unquoted_group_delta(sc: str) -> int:
    return sum(1 if sc[i] in ("(", "{") else (-1 if sc[i] in (")", "}") else 0) for i in _iter_unquoted_shell_indices(sc))


def _unquoted_subshell_opens_closes(sc: str) -> Tuple[int, int]:
    idxs = list(_iter_unquoted_shell_indices(sc))
    return sum(sc[i] == "(" for i in idxs), sum(sc[i] == ")" for i in idxs)


def _advance_effective_cwd(sc: str, current_cwd: str, dir_stack: List[str], payload: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, bool]:
    tokens, uq = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(sc))
    verb, args, unv = _unwrap_shell_command_tokens(tokens, uq)
    if verb not in {"cd", "pushd", "popd"}:
        return current_cwd, False, False
    if unv:
        return current_cwd, True, True
    if verb == "popd":
        return (current_cwd, True, True) if (any(t != "--" for t in args) or not dir_stack) else (dir_stack.pop(), True, False)
    after_dd, pos = False, []
    for tok in args:
        if not after_dd and tok == "--": after_dd = True
        elif not after_dd and ((tok.startswith("-") and tok != "-") or (verb == "pushd" and tok.startswith("+"))):
            if not (verb == "cd" and re.fullmatch(r"-[LPe@]+", tok)):
                return current_cwd, True, True
        else:
            pos.append(tok)
    if len(pos) != 1 or not pos[0] or pos[0] in {"-", "~"} or any(ch in pos[0] for ch in "$`*?[]~{}"):
        return current_cwd, True, True
    if verb == "pushd": dir_stack.append(current_cwd)
    return _resolve_cmd_target(pos[0], current_cwd), True, False


def _cmd_enters_guard_plugin_cwd(sub_cmds: List[str], cwd: str) -> bool:
    eff, dstack, sstack = cwd or str(PROJECT_ROOT), [], []
    if _is_guard_plugin_cwd(eff):
        return True
    for sc in sub_cmds:
        opens, closes = _unquoted_subshell_opens_closes(sc)
        sstack.extend((eff, list(dstack)) for _ in range(opens))
        eff, is_cd, cd_unv = _advance_effective_cwd(sc, eff, dstack, None)
        if is_cd and (cd_unv or _is_guard_plugin_cwd(eff)):
            return True
        for _ in range(min(closes, len(sstack))):
            eff, dstack = sstack.pop()
    return False


def _parse_git_subcmd_tokens(sc: str, *, allow_config_flags: bool = True) -> Optional[Tuple[str, List[str]]]:
    assignments, rem = _parse_leading_env_assignments(_fold_shell_line_continuations(sc))
    if any(is_app or not ((k in {"GIT_EDITOR", "GIT_SEQUENCE_EDITOR"} and v in _SAFE_GIT_EDITOR_VALS) or (k in {"GIT_PAGER", "PAGER"} and v in _SAFE_PAGER_VALS) or (k == "GIT_TERMINAL_PROMPT" and v in {"0", "1"}) or (k in {"LC_ALL", "LANG"} and v in {"C", "C.UTF-8", "en_US.UTF-8"})) for k, is_app, v in assignments):
        return None
    toks, _ = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(rem))
    if not toks or Path(toks[0]).name.lower() != "git":
        return None
    val_opts = {"-C", "--git-dir", "--work-tree"} | ({"-c", "--namespace", "--super-prefix", "--attr-source"} if allow_config_flags else set())
    bool_opts = {"--no-pager", "--paginate", "-p", "-P", "--bare", "--no-replace-objects", "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs", "--no-optional-locks"}
    skip_next = False
    for idx, tok in enumerate(toks[1:], start=1):
        if skip_next: skip_next = False
        elif tok in val_opts: skip_next = True
        elif not (tok.startswith(tuple(o + "=" for o in val_opts if o.startswith("--"))) or (tok.startswith("-C") and len(tok) > 2) or (allow_config_flags and tok.startswith("-c") and len(tok) > 2 and not tok.startswith("--")) or tok in bool_opts):
            return None if tok.startswith("-") else (tok.lower(), toks[idx + 1 :])
    return ("", []) if not skip_next else None


def _is_git_only_command(cmd: str) -> bool:
    if not cmd or "$(" in cmd or "`" in cmd:
        return False
    cleaned, has_h_subst, has_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    unquoted = _strip_shell_quoted_literals(cleaned)
    if has_h_subst or has_unclosed or _extract_shell_redir_targets(cleaned) or "<(" in unquoted or ">(" in unquoted:
        return False
    sub_cmds = _split_shell_commands(cleaned)
    return bool(sub_cmds) and all(_parse_git_subcmd_tokens(sc, allow_config_flags=True) is not None for sc in sub_cmds)


def _is_safe_git_inspection_command(cmd: str) -> bool:
    if not cmd or not _is_git_only_command(cmd) or _extract_shell_redir_targets(cmd):
        return False
    bad_opts = {"-c", "--output", "--ext-diff", "--exec-path", "--config-env", "--no-index", "--upload-pack", "--receive-pack", "--open-files-in-pager"}
    bad_prefixes = tuple(o + "=" for o in bad_opts if o.startswith("--"))
    for sc in (sub_cmds := _split_shell_commands(cmd)):
        if (parsed := _parse_git_subcmd_tokens(sc, allow_config_flags=False)) is None or parsed[0] not in {"status", "diff", "log"}:
            return False
        if any("ext::" in t.lower() or t in bad_opts or t.startswith(bad_prefixes) or (t.startswith("-c") and len(t) > 2 and not t.startswith("--")) for t in parsed[1][: (parsed[1].index("--") if "--" in parsed[1] else len(parsed[1]))]):
            return False
    return bool(sub_cmds)


def _extract_inline_sh(verb: str, arg_tokens: List[str]) -> Optional[str]:
    if verb == "eval":
        args = arg_tokens[1:] if (arg_tokens and arg_tokens[0] == "--") else arg_tokens
        return " ".join(args) if args else None
    after_dd = skip_next = saw_c = False
    for idx, tok in enumerate(arg_tokens):
        if skip_next: skip_next = False
        elif not after_dd and tok == "--": after_dd = True
        elif not after_dd and ((tok.startswith("-") and tok != "-") or (tok.startswith("+") and len(tok) > 1)):
            if tok in {"-o", "+o", "-O", "+O", "--rcfile", "--init-file", "-p", "--profile", "-P", "--features"}: skip_next = True
            elif tok in {"--command", "--init-command"} or tok.startswith(("--command=", "--init-command=")):
                return tok.split("=", 1)[1] if "=" in tok else (arg_tokens[idx + 1] if idx + 1 < len(arg_tokens) else None)
            elif not tok.startswith("--") and "c" in tok[1:].lower(): saw_c = True
        else:
            return tok if saw_c else None
    return None


def _is_trivial_eval_subcmd(subcmd: str) -> bool:
    toks, uq = _strip_redir_tokens_with_meta(_split_shell_tokens_raw_with_meta(subcmd))
    verb, args_t, _ = _unwrap_shell_command_tokens(toks, uq)
    stripped = _strip_leading_env_assignments(_fold_shell_line_continuations(subcmd))
    if not verb or TRIVIAL_EVAL_CMD_RE.match(" ".join([verb, *args_t])) or (verb == "cd" and len(args_t) == 1):
        return True
    from guard.ast_guard import _is_test_command

    if (TEST_CMD_START_RE.match(stripped) and not _is_test_command(stripped)) or (_PY_VERB_RE.match(verb) and _parse_python_cli(args_t)[1] == "c"):
        return True
    if verb in (_SHELL_WRAPPER_VERBS - {"source", "."}) and (inner_sh := _extract_inline_sh(verb, args_t)):
        return _is_trivial_eval_command(inner_sh)
    return bool(TRIVIAL_TEST_FLAG_RE.search(stripped))


def _is_trivial_eval_command(eval_cmd: str) -> bool:
    if not eval_cmd or not eval_cmd.strip() or _has_unquoted_or_operator(eval_cmd) or _has_shell_masking_or_chaining(eval_cmd):
        return True
    parts = _split_shell_commands_with_delims(eval_cmd)
    for idx, (_, part) in enumerate(parts):
        if part and not _is_trivial_eval_subcmd(part):
            return (parts[idx + 1][0] if idx + 1 < len(parts) else None) in ("|", "|&", ";", "\n", "||", "&")
    return True


_UNSAFE_EVAL_FLAG_OPTS: Dict[str, Tuple[frozenset[str], str]] = {
    "find": (frozenset({"-delete", "-fls", "-exec", "-execdir", "-ok", "-okdir"}), ""), "rg": (frozenset({"--pre"}), ""),
    "sort": (frozenset({"-o", "--output", "--compress-program"}), "o"), "iconv": (frozenset({"-o", "--output"}), "o"),
    "curl": (frozenset({"-o", "--output", "-O", "--remote-name", "--remote-name-all", "-K", "--config", "-D", "--dump-header", "-c", "--cookie-jar"}), "oOKDc"),
}


def _is_unsafe_eval_command(cmd_str: str) -> bool:
    if not cmd_str:
        return False
    if _has_nul_or_surrogate_input(cmd_str):
        return True
    cleaned, h_subst, h_unclosed = _preprocess_shell_heredocs_and_comments(cmd_str)
    if h_subst or h_unclosed or _has_unquoted_cmd_subst(cleaned) or _extract_shell_redir_targets(cleaned):
        return True
    for sc in _split_shell_commands(cleaned):
        raw_meta = _split_shell_tokens_raw_with_meta(sc)
        has_in = any(is_r and "<" in t for t, is_r, _ in raw_meta)
        toks, uq = _strip_redir_tokens_with_meta(raw_meta)
        um: Dict[str, bool] = {}
        verb, _, args_t, unv = _unwrap_shell_command_tokens_with_raw(toks, uq, out_meta=um)
        if unv or um.get("opaque"):
            return True
        if verb and (
            verb in (_SHELL_WRAPPER_VERBS | _UNVERIFIABLE_WRAPPERS | _OPAQUE_EXEC_VERBS | _MUTATING_EVAL_VERBS)
            or _INTERPRETER_EVAL_RE.match(verb)
            or (verb == "git" and not _is_safe_git_inspection_command(sc))
            or (_PY_VERB_RE.match(verb) and (has_in or not (p := _parse_python_cli(args_t))[0] or p[1] == "c"))
            or ((spec := _UNSAFE_EVAL_FLAG_OPTS.get(verb)) and any(t.split("=", 1)[0] in spec[0] or (verb == "find" and t.startswith("-fprint")) or (spec[1] and t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in spec[1])) for t in args_t))
        ):
            return True
    return False


def _is_declared_eval_command(cmd: str, state: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Return True if cmd invokes a non-trivial, non-mutating declared EVAL_CMD without shell-chaining bypasses."""
    if not cmd or "$(" in cmd or "`" in cmd or _has_unquoted_or_operator(cmd) or _has_shell_masking_or_chaining(cmd) or _is_unsafe_eval_command(cmd):
        return False
    cid = str(payload.get("conversationId") or "").strip().lower()
    raw_subs = state.get("subagents")
    sub_entry = raw_subs.get(cid) if isinstance(raw_subs, dict) else None
    declared = [
        d for raw_d in (
            sub_entry.get("eval_cmd") if isinstance(sub_entry, dict) else None,
            (state.get("self_subagent_meta") or {}).get("eval_cmd") if isinstance(state.get("self_subagent_meta"), dict) else None,
            state.get("active_eval_cmd"), *(c.get("eval_cmd") for c in (state.get("cohorts") or {}).values() if isinstance(c, dict)),
        ) if isinstance(raw_d, str) and raw_d.strip() for d in (_strip_leading_env_assignments(raw_d),)
        if d and not _is_trivial_eval_command(d) and not _is_unsafe_eval_command(d)
    ]
    if not declared or not (parts := _split_shell_commands_with_delims(cmd)):
        return False
    matched_eval, cur_eval_dir, dir_stack = False, str(PROJECT_ROOT.resolve()), []
    for idx, (_, sc) in enumerate(parts):
        assignments, remainder = _parse_leading_env_assignments(_fold_shell_line_continuations(sc))
        if any(_is_dangerous_env_assignment(f"{k}{'+=' if is_app else '='}{v}") for k, is_app, v in assignments):
            return False
        if not (stripped := remainder.strip()):
            continue
        if (matched_decl := next((d for d in declared if stripped == d or stripped.startswith(d + " ")), None)) is not None:
            if _is_trivial_eval_subcmd(stripped) or TRIVIAL_TEST_FLAG_RE.search(stripped[len(matched_decl) :]) or ((parts[idx + 1][0] if idx + 1 < len(parts) else None) in ("|", "|&", ";", "\n", "||", "&")):
                return False
            matched_eval = True; continue
        next_dir, is_cd, cd_unv = _advance_effective_cwd(stripped, cur_eval_dir, dir_stack)
        if is_cd and not cd_unv and ((nd := Path(next_dir).resolve()) == (pr := PROJECT_ROOT.resolve()) or pr in nd.parents) and not _is_staging_or_tmp_path(str(nd), {}):
            cur_eval_dir = str(nd); continue
        return False
    return matched_eval

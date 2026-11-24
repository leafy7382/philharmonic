"""Shell lexing, redirection extraction, wrapper unwrapping, and eval inspection."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Any, Dict, FrozenSet, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from guard.constants import PROJECT_ROOT, TEST_CMD_START_RE, TRIVIAL_EVAL_CMD_RE, TRIVIAL_TEST_FLAG_RE
from guard.path_guards import _is_guard_plugin_cwd, _is_staging_or_tmp_path, _resolve_cmd_target, _sanitize_decoded_shell_token

SAFE_XARGS_READONLY_CMDS = frozenset({"grep", "egrep", "fgrep", "rg", "wc", "cat", "head", "tail", "ls", "stat", "file", "echo", "printf", "basename", "dirname", "md5sum", "sha256sum", "sha1sum", "cksum", "du", "realpath", "readlink"})
SAFE_READONLY_SHELL_VERBS = frozenset({"ls", "dir", "vdir", "pwd", "echo", "cat", "tac", "nl", "head", "tail", "wc", "grep", "egrep", "fgrep", "zgrep", "stat", "file", "basename", "dirname", "realpath", "readlink", "md5sum", "sha1sum", "sha256sum", "sha512sum", "cksum", "sum", "du", "df", "id", "whoami", "uname", "date", "which", "type", "true", "false", ":", "cut", "tr", "paste", "comm", "diff", "cmp", "rev", "fold", "fmt", "expand", "unexpand", "column", "strings", "od", "hexdump", "jq", "sleep", "printenv", "hostname", "uptime", "nproc", "getconf", "seq", "expr"})
_SHELL_WRAPPER_VERBS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "ash", "csh", "tcsh", "fish", "mksh", "rbash", "eval", "source", "."})
_UNVERIFIABLE_WRAPPERS = frozenset({"flock", "script", "stdbuf", "parallel", "busybox", "strace", "ltrace", "chrt", "ionice", "taskset", "unshare", "nsenter", "chroot", "watch", "prlimit", "setarch", "linux32", "linux64", "numactl", "setsid", "ed", "ex", "uvx"})
_OPAQUE_EXEC_VERBS = _UNVERIFIABLE_WRAPPERS | {"bun", "deno", "expect", "tclsh", "wish", "rscript", "julia", "pwsh", "sqlite3"}
_SAFE_NPX_RUNNERS = frozenset({"jest", "vitest", "mocha", "ava", "tap", "playwright", "tsc", "pyright", "mypy"})
_DANGEROUS_ENV_VARS = frozenset({"GCONV_PATH", "HOSTALIASES", "LOCPATH", "NLSPATH", "ENV", "BASH_ENV", "CDPATH", "IFS", "PROMPT_COMMAND", "PS4", "FPATH", "ZDOTDIR", "SHELL", "PATH", "GLOBIGNORE", "GIT_EXTERNAL_DIFF", "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_GLOBAL", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_PROXY_COMMAND", "GIT_UPLOAD_PACK", "GIT_RECEIVE_PACK", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_SSL_NO_VERIFY", "GIT_SSL_CAINFO", "GIT_SSL_CAPATH", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "GIT_ALLOW_PROTOCOL", "GNUPGHOME", "SSLKEYLOGFILE", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "all_proxy", "EDITOR", "VISUAL", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONEXECUTABLE", "PYTHONUSERBASE", "PYTHONINSPECT", "PYTHONWARNINGS", "PYTHONBREAKPOINT", "PYTHONCOERCECLOCALE", "PYTHONDEVMODE", "PYTHONMALLOC", "PERL5LIB", "PERLLIB", "PERL5OPT", "RUBYLIB", "RUBYOPT", "NODE_OPTIONS", "NODE_PATH", "AWKPATH", "AWKLIBPATH", "LUA_PATH", "LUA_CPATH", "HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "GIT_CONFIG_NOSYSTEM", "GIT_ATTR_SOURCE", "GIT_GRAFT_FILE", "GIT_NOTES_REF", "GIT_REPLACE_REF_BASE", "HISTFILE", "LESSOPEN", "LESSCLOSE", "MANPAGER", "BROWSER", "TMPDIR", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "GOFLAGS", "GOENV", "CC", "CXX", "LD", "AR", "AS", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_HOME", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "CARGO_TARGET_DIR", "RIPGREP_CONFIG_PATH", "PYLINTRC", "MYPY_CACHE_DIR", "RUFF_CACHE_DIR", "TOX_OVERRIDE", "TOX_CONF", "TOX_WORK_DIR"})
_DANGEROUS_ENV_PREFIXES = ("LD_", "DYLD_", "GIT_TRACE", "LUA_INIT", "NPM_CONFIG_", "YARN_", "PNPM_", "CARGO_TARGET_", "CARGO_REGISTRIES_", "UV_", "PIP_", "POETRY_", "PIPENV_", "PYTEST_", "TOX_")
_SAFE_GIT_EDITOR_VALS = frozenset({"", "true", "/bin/true", "/usr/bin/true", ":", "false", "/bin/false"})
_SAFE_PAGER_VALS = frozenset({"", "cat", "/bin/cat", "/usr/bin/cat", "true", "/bin/true", "/usr/bin/true", ":"})
_TOKEN_REDIR_OP_RE = re.compile(r"(?:&>>|&>|>>|>\||>&|<<<|<<-|<<|<>|<&|<(?!\()|>(?!\())")
_SHELL_MASKING_TAIL_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*(?:(?:(?:/(?:usr/)?bin/)?(?:true|echo|printf|cat|pwd|ls|sleep)|exit(?:\s+0)?)\b|:(?=$|\s))", re.I)
_LEADING_GROUPING_WORDS = frozenset({"(", "{", "!", "if", "while", "until", "then", "do", "else", "elif", "coproc"})
_TRAILING_GROUPING_TOKENS = frozenset({")", "}"})
_STANDALONE_END_WORDS = frozenset({"fi", "done", "esac"})
_PY_VERB_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)$")
_MUTATING_EVAL_VERBS = frozenset({"rm", "unlink", "shred", "mv", "cp", "tee", "touch", "truncate", "dd", "patch", "install", "ln", "chmod", "chown", "chgrp", "mkdir", "rmdir", "rsync", "split", "csplit", "link", "mktemp", "mkfifo", "mknod", "wget", "dos2unix", "unix2dos", "fallocate", "sponge", "tar", "unzip", "xargs"})
_INTERPRETER_EVAL_RE = re.compile(r"^(?:[gmn]?awk|sed|(?:perl|node|ruby|php|lua)(?:\d+(?:\.\d+)*)?)$")


def _read_quoted_span(text: str, i: int) -> Tuple[str, int, bool]:
    """Read a $'...', '...', or "..." span starting at text[i]; return (decoded, next_idx, closed)."""
    n = len(text)
    if text[i : i + 2] == "$'":
        j = i + 2
        while j < n and text[j] != "'":
            j += 2 if (text[j] == "\\" and j + 1 < n) else 1
        return text[i + 2 : j], (j + 1 if j < n else n), j < n
    q, j, chars = text[i], i + 1, []
    while j < n and text[j] != q:
        if q == '"' and text[j] == "\\" and j + 1 < n and text[j + 1] in ('"', "\\", "$", "`", "\n"):
            if text[j + 1] != "\n": chars.append(text[j + 1])
            j += 2
        else: chars.append(text[j]); j += 1
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
    """Fold unquoted and double-quoted shell \\\n line continuations while preserving '...' and $'...' verbatim."""
    if not text or "\\\n" not in text: return text
    return "".join(
        (re.sub(r"(?<!\\)((?:\\\\)*)\\\n", r"\1", text[s:e]) if k == "dq" else text[s:e])
        for k, s, e, _ in _iter_shell_spans(text)
        if not (k == "esc" and s + 1 < len(text) and text[s + 1] == "\n")
    )


def _iter_unquoted_shell_indices(text: str) -> Iterator[int]:
    for kind, s, e, _ in _iter_shell_spans(text):
        if kind == "raw":
            yield from range(s, e)


_ANSI_NUL_SURR_RE = re.compile(
    r"(?<!\\)(?:\\\\)*\\(?:000|0{1,3}(?![0-7])|x00|x0(?![0-9a-fA-F])|u0000|u0{1,3}(?![0-9a-fA-F])|U00000000|U0{1,7}(?![0-9a-fA-F])|u[dD][89a-fA-F][0-9a-fA-F]{2}|U0{0,4}[dD][89a-fA-F][0-9a-fA-F]{2}|c[@ ])"
)


def _has_nul_or_surrogate_input(text: str) -> bool:
    """Return True if text contains raw NUL, lone UTF-16 surrogates, or ANSI-C NUL/surrogate escapes."""
    if not text: return False
    if not isinstance(text, str) or "\x00" in text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in text): return True
    if "$" not in text or "'" not in text: return False
    folded = _fold_shell_line_continuations(text)
    return any(bool(_ANSI_NUL_SURR_RE.search(folded[s + 2 : e])) for kind, s, e, _ in _iter_shell_spans(folded) if kind == "ansi")


def _payload_has_raw_cr(cmd: str, payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict) or not isinstance(tc := payload.get("toolCall"), dict) or not isinstance(ar := tc.get("args"), dict): return False
    return isinstance(rc := ar.get("CommandLine"), str) and "\r" in rc and (not cmd or rc.strip() == cmd.strip())


def _preprocess_shell_heredocs_and_comments(cmd: str) -> Tuple[str, bool, bool]:
    """Strip unquoted shell comments and heredoc bodies, returning (cleaned_cmd, has_heredoc_cmd_subst, has_unclosed_syntax)."""
    if not cmd: return "", False, False
    out, pending, has_h_subst, has_unclosed, in_dq, i, n = [], [], False, ("\r" in cmd), False, 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if ch == "\\":
            if i + 1 < n and cmd[i + 1] == "\n":
                i += 2; continue
            has_unclosed = has_unclosed or (i + 1 >= n); out.append(cmd[i : i + 2]); i += 2
        elif not in_dq and (cmd[i : i + 2] == "$'" or ch == "'"):
            _, end, closed = _read_quoted_span(cmd, i)
            has_unclosed = has_unclosed or not closed; out.append(cmd[i:end]); i = end
        elif ch == '"':
            in_dq, _ = not in_dq, out.append(ch); i += 1
        elif not in_dq and ch == "#" and (not out or out[-1] in " \t\n;|&()"):
            while i < n and cmd[i] != "\n": i += 1
        elif not in_dq and cmd[i : i + 3] == "<<<":
            out.append("<<<"); i += 3
        elif not in_dq and cmd[i : i + 2] == "<<":
            out.append("<<"); i += 2
            strip_tabs = i < n and cmd[i] == "-"
            if strip_tabs: out.append("-"); i += 1
            while i < n and cmd[i] in " \t": out.append(cmd[i]); i += 1
            dparts, was_q, bad_d = [], False, False
            while i < n and cmd[i] not in " \t\n;|&()<>":
                if cmd[i] == "\\" and i + 1 < n:
                    was_q, _, i = True, (dparts.append(cmd[i + 1]), out.append(cmd[i : i + 2])), i + 2
                elif cmd[i : i + 2] == "$'" or cmd[i] in ("'", '"'):
                    dec, end, closed = _read_quoted_span(cmd, i)
                    was_q, bad_d, _, i = True, bad_d or not closed, (dparts.append(dec), out.append(cmd[i:end])), end
                else:
                    dparts.append(cmd[i]); out.append(cmd[i]); i += 1
            delim = "".join(dparts)
            if bad_d or not delim or (not was_q and ("$" in delim or "`" in delim)): has_unclosed = True
            else:
                if not was_q: has_h_subst = True
                pending.append((delim, strip_tabs))
        elif not in_dq and ch == "\n":
            out.append("\n"); i += 1
            for delim, strip_tabs in pending:
                matched = False
                while i < n:
                    le = cmd.find("\n", i)
                    raw_line, i = (cmd[i:], n) if le == -1 else (cmd[i:le], le + 1)
                    if (raw_line.lstrip("\t") if strip_tabs else raw_line) == delim: matched = True; break
                has_unclosed = has_unclosed or not matched
            pending.clear()
        else:
            out.append(ch); i += 1
    return "".join(out), has_h_subst, has_unclosed or in_dq or bool(pending)


def _split_shell_commands_with_delims(cmd: str) -> List[Tuple[Optional[str], str]]:
    """Split a shell command string on unquoted &&, ||, ;, |, |&, &, and newline delimiters, tracking preceding delimiter."""
    if not cmd: return []
    parsed, parts = parse_shell_command(cmd), []
    for idx, s in enumerate(parsed.subcmds):
        parts.append((s.prec_delim, s.raw))
        if s.next_delim == "&" and (idx + 1 == len(parsed.subcmds) or parsed.subcmds[idx + 1].prec_delim != "&"): parts.append(("&", ""))
    if not parsed.subcmds and parsed.has_masking: parts.append(("&", ""))
    return parts


def _split_shell_commands(cmd: str) -> List[str]:
    return [s.raw for s in parse_shell_command(cmd).subcmds] if cmd else []


def _has_shell_masking_or_chaining(cmd: str) -> bool:
    return bool(cmd) and parse_shell_command(cmd).has_masking


def _split_shell_tokens_raw_with_meta(text: str, *, _preprocessed: bool = False) -> List[Tuple[str, bool, bool]]:
    """Lex a shell string into (dequoted_token, is_unquoted_redir_op, is_unquoted_token) tuples."""
    if not _preprocessed: text = _fold_shell_line_continuations(_preprocess_shell_heredocs_and_comments(text)[0])
    if not text: return []
    tokens, current, has_token, saw_quote, i, n = [], [], False, False, 0, len(text)

    def _flush() -> None:
        nonlocal current, has_token, saw_quote
        if has_token:
            tokens.append((_sanitize_decoded_shell_token("".join(current)), False, not saw_quote))
            current, has_token, saw_quote = [], False, False

    while i < n:
        ch = text[i]
        if ch in " \t\n": _flush()
        elif ch == "\\" and i + 1 < n:
            if text[i + 1] != "\n":
                saw_quote = has_token = True
                current.append(text[i + 1])
            i += 2; continue
        elif text[i : i + 2] == "$'" or ch in ("'", '"'):
            dec, i, _ = _read_quoted_span(text, i)
            current.append(dec); has_token = saw_quote = True; continue
        elif ch in ("$", "<", ">") and i + 1 < n and text[i + 1] == "(":
            current.append(ch + "("); has_token, i = True, i + 2; continue
        elif ch in ("(", ")"):
            _flush(); tokens.append((ch, False, True))
        elif not has_token and ((ch in ("{", "}") and (i + 1 >= n or text[i + 1] in " \t\n()")) or (ch == "!" and (i + 1 >= n or text[i + 1] in " \t\n({"))):
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


def _extract_redir_targets_from_raw(raw: List[Tuple[str, bool, bool]]) -> Tuple[str, ...]:
    targets: List[str] = []
    for idx, (op, is_redir, _) in enumerate(raw):
        if not is_redir or ">" not in op or idx + 1 >= len(raw): continue
        sink, sink_redir, sink_uq = raw[idx + 1]
        if sink_redir: continue
        is_fd_dup = op.endswith(">&") and sink_uq and (sink.isdigit() or sink == "-" or (sink.endswith("-") and sink[:-1].isdigit()))
        if not is_fd_dup and sink and sink != "/dev/null" and not (sink_uq and sink in {"&1", "&2"}): targets.append(sink)
    return tuple(targets)


def _extract_shell_redir_targets(cmd: str) -> List[str]:
    """Extract write targets from unquoted shell redirections (>, >>, >|, &>, N>)."""
    return list(parse_shell_command(cmd).redir_targets) if cmd else []


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
        while i < n and cmd[i] in " \t\n":
            i += 1
        if not (m := re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\+?=)", cmd[i:])):
            break
        key, is_append, val_chars, i = m.group(1), m.group(2) == "+=", [], i + m.end()
        while i < n and cmd[i] not in " \t\n;|&<>()":
            if cmd[i : i + 2] == "$'" or cmd[i] in ("'", '"'):
                dec, i, _ = _read_quoted_span(cmd, i)
                val_chars.append(dec)
            elif cmd[i] == "\\" and i + 1 < n:
                if cmd[i + 1] == "\n": i += 2
                else: val_chars.append(cmd[i + 1]); i += 2
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


class FlagSpec(NamedTuple):
    val_opts: FrozenSet[str] = frozenset()
    short_val_chars: str = ""
    bad_opts: FrozenSet[str] = frozenset()
    bad_short_chars: str = ""
    stop_at_dd: bool = True
    stop_at_first_pos: bool = False
    gnu_abbrev: bool = False


class FlagWalkResult(NamedTuple):
    positionals: Tuple[str, ...]
    opt_pairs: Tuple[Tuple[str, str], ...]
    flags_seen: Tuple[str, ...]
    bad_hit: bool
    missing_value: bool
    rest_index: int


def walk_cli_flags(args: Sequence[str], spec: FlagSpec) -> FlagWalkResult:
    """Declarative CLI flag walker supporting --opt=val, --opt val, short clusters, GNU abbreviations, --, and bad-flag detection."""
    pos, opt_pairs, flags_seen = [], [], []
    bad_hit, after_dd, skip_opt, rest_index = False, False, "", len(args)
    for i, tok in enumerate(args):
        if skip_opt: opt_pairs.append((skip_opt, tok)); skip_opt = ""
        elif not after_dd and tok == "--":
            if spec.stop_at_dd:
                flags_seen.append("--")
                if spec.stop_at_first_pos: rest_index = i + 1; break
                after_dd = True
            else:
                if tok in spec.bad_opts: bad_hit = True
                if spec.stop_at_first_pos: rest_index = i; break
                flags_seen.append(tok)
        elif not after_dd and tok.startswith("--"):
            opt, has_eq, val = tok.partition("=")
            matched_val: Optional[str] = opt if opt in spec.val_opts else None
            if matched_val is None and spec.gnu_abbrev and len(opt) >= 3:
                vm = [v for v in spec.val_opts if v.startswith("--") and v.startswith(opt)]
                if len(vm) == 1: matched_val = vm[0]
            flags_seen.append(matched_val or opt)
            if opt in spec.bad_opts or (spec.gnu_abbrev and len(opt) >= 3 and any(b.startswith(opt) for b in spec.bad_opts if b.startswith("--"))):
                bad_hit = True
            if matched_val is not None:
                if has_eq: opt_pairs.append((matched_val, val))
                else: skip_opt = matched_val
        elif not after_dd and tok.startswith("-") and tok != "-":
            opt, has_eq, val = tok.partition("=")
            is_bad = tok in spec.bad_opts or opt in spec.bad_opts
            if is_bad: bad_hit = True
            if tok in spec.val_opts:
                flags_seen.append(tok); bad_hit = bad_hit or (len(tok) == 2 and tok[1] in spec.bad_short_chars); skip_opt = tok
            elif has_eq and len(opt) > 2 and opt in spec.val_opts: flags_seen.append(opt); opt_pairs.append((opt, val))
            elif is_bad and len(opt) > 2: flags_seen.append(opt)
            else:
                for k, ch in enumerate(tok[1:]):
                    flags_seen.append("-" + ch)
                    if ch in spec.bad_short_chars: bad_hit = True
                    if ch in spec.short_val_chars:
                        if k < len(tok[1:]) - 1: opt_pairs.append(("-" + ch, tok[1:][k + 1 :]))
                        else: skip_opt = "-" + ch
                        break
        else:
            if not after_dd and tok == "-" and tok in spec.bad_opts: bad_hit = True
            if spec.stop_at_first_pos: rest_index = i; break
            pos.append(tok)
    return FlagWalkResult(tuple(pos), tuple(opt_pairs), tuple(flags_seen), bad_hit, bool(skip_opt), rest_index)


def _is_canonical_cmd_path(verb: str, raw_tok: str) -> bool:
    """Return True if raw_tok is bare or in /bin, /usr/bin, /usr/local/bin (or sys.executable for python)."""
    if "/" not in raw_tok: return True
    low = raw_tok.lower()
    if low in {f"/bin/{verb}", f"/usr/bin/{verb}", f"/usr/local/bin/{verb}"}: return True
    return bool(_PY_VERB_RE.match(verb)) and bool(sys.executable) and low == sys.executable.lower()


_SAFE_WRAPPER_SPECS: Dict[str, Tuple[frozenset[str], str, int]] = {"nohup": (frozenset(), "", 0), "command": (frozenset(), "", 0), "builtin": (frozenset(), "", 0), "exec": (frozenset({"-a"}), "a", 0), "nice": (frozenset({"-n", "--adjustment"}), "n", 0), "timeout": (frozenset({"-k", "--kill-after", "-s", "--signal"}), "sk", 1), "time": (frozenset({"-o", "--output", "-f", "--format"}), "of", 0)}
_RUNNER_GLOBAL_VAL_FLAGS = frozenset({"--directory", "-C", "--project", "--config-file", "--python", "-p", "--cache-dir", "--index", "--default-index", "--index-url", "--extra-index-url", "--find-links"})
_RUNNER_RUN_VAL_FLAGS = _RUNNER_GLOBAL_VAL_FLAGS | frozenset({"-w", "--with", "--with-editable", "--with-requirements", "--from", "--script", "-s", "--module", "-m", "--gui-script", "--env-file"})
_ENV_FLAG_SPEC = FlagSpec(val_opts=frozenset({"-u", "--unset", "-a", "--argv0"}), short_val_chars="ua", bad_opts=frozenset({"-S", "--split-string", "-C", "--chdir"}), bad_short_chars="SC", stop_at_dd=False, stop_at_first_pos=True, gnu_abbrev=True)
_SAFE_NPX_FLAGS = frozenset({"-y", "--yes", "-q", "--quiet", "--no", "--no-install", "--"})


def _unwrap_shell_command_tokens_with_raw(
    tokens: List[str], unquoted_flags: Optional[List[bool]] = None, out_arg_flags: Optional[List[bool]] = None, out_meta: Optional[Dict[str, bool]] = None,
) -> Tuple[str, str, List[str], bool]:
    """Strip grouping tokens, leading KEY=VAL env assignments, and safe prefix wrappers."""
    rem, flags, unv, opaque = list(tokens), (list(unquoted_flags) if unquoted_flags is not None else [True] * len(tokens)), False, False
    _pop = lambda front=True: (flags.pop(0 if front else -1) if flags else None, rem.pop(0 if front else -1))[1]

    def _consume_flags(val_flags: frozenset[str], short_vals: str, *, stop_dd: bool = True, bad_opts: frozenset[str] = frozenset(), bad_shorts: str = "", gnu_abbrev: bool = False) -> FlagWalkResult:
        nonlocal unv
        res = walk_cli_flags(rem, FlagSpec(val_opts=val_flags, short_val_chars=short_vals, bad_opts=bad_opts, bad_short_chars=bad_shorts, stop_at_dd=stop_dd, stop_at_first_pos=True, gnu_abbrev=gnu_abbrev))
        if res.bad_hit: unv = True
        del rem[: res.rest_index], flags[: res.rest_index]
        return res

    while rem:
        if (flags[0] and rem[0] in _LEADING_GROUPING_WORDS) or (flags[-1] and (rem[-1] in _TRAILING_GROUPING_TOKENS or (len(rem) == 1 and rem[-1] in _STANDALONE_END_WORDS))):
            while rem and flags[0] and rem[0] in _LEADING_GROUPING_WORDS: _pop()
            while rem and flags[-1] and (rem[-1] in _TRAILING_GROUPING_TOKENS or (len(rem) == 1 and rem[-1] in _STANDALONE_END_WORDS)): _pop(False)
            continue
        while rem and re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", rem[0]): unv = unv or _is_dangerous_env_assignment(_pop())
        if not rem or (flags[0] and rem[0] in _LEADING_GROUPING_WORDS): continue
        head = Path(rem[0]).name.lower()
        if head in _UNVERIFIABLE_WRAPPERS: unv = opaque = True; break
        if (head == "env" or head in _SAFE_WRAPPER_SPECS or head in {"uv", "poetry", "pipenv", "npx", "bunx"}) and not _is_canonical_cmd_path(head, rem[0]):
            unv = opaque = True; break
        if head == "env":
            _pop()
            while rem and rem[0] != "--":
                t = rem[0]
                if t == "-": _pop()
                elif t.startswith("-"):
                    ew = walk_cli_flags(rem, _ENV_FLAG_SPEC)
                    if ew.bad_hit: unv = opaque = True; break
                    del rem[: ew.rest_index], flags[: ew.rest_index]
                elif re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", t): unv = unv or _is_dangerous_env_assignment(_pop())
                else: break
            if opaque: break
            if rem and rem[0] == "--": _pop()
        elif spec := _SAFE_WRAPPER_SPECS.get(head):
            _pop()
            res = _consume_flags(spec[0], spec[1], bad_opts=(frozenset({"-o", "--output"}) if head == "time" else frozenset()), bad_shorts=("o" if head == "time" else ""), gnu_abbrev=True)
            if head == "time" and res.bad_hit: opaque = True
            for _ in range(spec[2]):
                if rem: _pop()
        elif head in {"uv", "poetry", "pipenv"}:
            raw_head = _pop()
            if rem and all(t in {"-V", "--version", "-h", "--help"} for t in rem):
                if out_arg_flags is not None: out_arg_flags.extend(flags)
                if out_meta is not None: out_meta["opaque"] = opaque
                return head, raw_head, rem, unv
            _consume_flags(_RUNNER_GLOBAL_VAL_FLAGS, "Cp", stop_dd=False, bad_opts=_RUNNER_GLOBAL_VAL_FLAGS - {"--cache-dir"}, bad_shorts="Cp", gnu_abbrev=True)
            if rem and rem[0] == "run":
                _pop(); _consume_flags(_RUNNER_RUN_VAL_FLAGS, "Csmpw", bad_opts=_RUNNER_RUN_VAL_FLAGS - {"--cache-dir"}, bad_shorts="Csmpw", gnu_abbrev=True)
                continue
            unv = True
            if out_arg_flags is not None: out_arg_flags.extend(flags)
            if out_meta is not None: out_meta["opaque"] = opaque
            return head, raw_head, rem, unv
        elif head in {"npx", "bunx"}:
            _pop()
            res = _consume_flags(frozenset({"-c", "--call", "-p", "--package"}), "cp")
            unv = unv or any(f not in _SAFE_NPX_FLAGS for f in res.flags_seen) or not rem or Path(rem[0]).name.lower() not in _SAFE_NPX_RUNNERS
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


ParsedSubcmd = NamedTuple("ParsedSubcmd", [
    ("raw", str), ("prec_delim", Optional[str]), ("next_delim", Optional[str]), ("unquoted", str),
    ("group_delta", int), ("subshell_opens", int), ("subshell_closes", int), ("is_piped_in", bool),
    ("raw_tokens", Tuple[Tuple[str, bool, bool], ...]), ("redir_targets", Tuple[str, ...]), ("has_in_redir", bool),
    ("tokens", Tuple[str, ...]), ("token_uq", Tuple[bool, ...]), ("env_assigns", Tuple[Tuple[str, bool, str], ...]),
    ("stripped_env", str), ("verb", str), ("raw_verb", str), ("args", Tuple[str, ...]),
    ("arg_uq", Tuple[bool, ...]), ("unwrap_unv", bool), ("opaque", bool),
])

ParsedCommand = NamedTuple("ParsedCommand", [
    ("raw", str), ("cleaned", str), ("unquoted", str), ("has_nul_surr", bool), ("has_ansi_c", bool), ("has_subshell", bool),
    ("has_heredoc_subst", bool), ("has_unclosed", bool), ("has_cmd_subst", bool), ("has_or", bool), ("has_masking", bool),
    ("redir_targets", Tuple[str, ...]), ("subcmds", Tuple[ParsedSubcmd, ...]),
])


def _first_subcmd(sc: str, parsed_sub: Optional[ParsedSubcmd]) -> Optional[ParsedSubcmd]:
    if parsed_sub is not None: return parsed_sub
    subs = parse_shell_command(sc).subcmds
    return subs[0] if subs else None


@lru_cache(maxsize=64)
def parse_shell_command(cmd: str) -> ParsedCommand:
    """Single-pass cached shell parser producing immutable ParsedCommand / ParsedSubcmd IR."""
    if not cmd: return ParsedCommand(cmd if isinstance(cmd, str) else "", "", "", False, False, False, False, False, False, False, False, (), ())
    has_nul_surr = _has_nul_or_surrogate_input(cmd)
    cleaned, has_h_subst, has_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    if not cleaned: return ParsedCommand(cmd, "", "", has_nul_surr, False, False, has_h_subst, has_unclosed, has_h_subst, False, False, (), ())

    spans = list(_iter_shell_spans(cleaned))
    has_ansi_c = any(k == "ansi" for k, _, _, _ in spans)
    unquoted = "".join('""' if k == "dq" else ("''" if k in ("sq", "ansi") else cleaned[s:e]) for k, s, e, _ in spans)
    has_cmd_subst = has_h_subst or any(("`" in seg or "$(" in seg or (k == "raw" and ("<(" in seg or ">(" in seg))) for k, s, e, _ in spans if k in ("raw", "dq") for seg in (cleaned[s:e],))
    has_or = any("||" in cleaned[s:e] for k, s, e, _ in spans if k == "raw")

    raw_parts, prec, last_start, skip_until, n = [], None, 0, 0, len(cleaned)
    for k, s, e, _ in spans:
        if k != "raw": continue
        for i in range(s, e):
            if i < skip_until: continue
            ch = cleaned[i]
            delim, step = ((";;&", 3) if cleaned[i : i + 3] == ";;&" else ((cleaned[i : i + 2], 2) if cleaned[i : i + 2] in ("&&", "||", "|&", ";;", ";&") else (("&", 1) if ch == "&" and not ((i > 0 and cleaned[i - 1] in (">", "<")) or (i + 1 < n and cleaned[i + 1] == ">")) else ((ch, 1) if ch in (";", "|", "\n") else (None, 1)))))
            if delim is not None:
                part = cleaned[last_start:i].strip()
                if part or prec == "&": raw_parts.append((prec, part))
                if not (ch == "\n" and part == "" and prec in (None, "|", "|&", "&&", "||")): prec = delim
                last_start = skip_until = i + step
    tail = cleaned[last_start:].strip()
    if tail or prec == "&": raw_parts.append((prec, tail))

    has_masking = any(d in ("||", "&") or (d in ("|", "|&", ";", "\n") and p and _SHELL_MASKING_TAIL_RE.match(p)) for d, p in raw_parts)
    subcmds, all_redirs, pipe_depth, subst_depth = [], [], 0, 0
    for idx, (p_delim, sc) in enumerate(raw_parts):
        if not sc: continue
        next_delim = raw_parts[idx + 1][0] if idx + 1 < len(raw_parts) else None
        sc_spans = list(_iter_shell_spans(sc))
        sc_uq = "".join('""' if k == "dq" else ("''" if k in ("sq", "ansi") else sc[s:e]) for k, s, e, _ in sc_spans)
        g_delta = opens = closes = 0
        for k, s, e, _ in sc_spans:
            if k == "raw":
                seg = sc[s:e]; j, sn = 0, len(seg)
                while j < sn:
                    c2 = seg[j]
                    if c2 in ("$", "<", ">") and j + 1 < sn and seg[j + 1] == "(": subst_depth += 1; j += 2; continue
                    if c2 == "(": opens += 1; g_delta += 1
                    elif c2 == ")":
                        if subst_depth > 0: subst_depth -= 1
                        else: closes += 1; g_delta -= 1
                    elif c2 == "{": g_delta += 1
                    elif c2 == "}": g_delta -= 1
                    j += 1
        is_piped_in = (p_delim in ("|", "|&")) or pipe_depth > 0
        if is_piped_in: pipe_depth = max(0, pipe_depth + g_delta)
        folded = _fold_shell_line_continuations(sc)
        raw_toks = _split_shell_tokens_raw_with_meta(folded, _preprocessed=True)
        sc_redirs = _extract_redir_targets_from_raw(raw_toks)
        all_redirs.extend(sc_redirs)
        has_in = any(is_r and "<" in tok for tok, is_r, _ in raw_toks)
        toks, uq = _strip_redir_tokens_with_meta(raw_toks)
        env_assigns, stripped_env = _parse_leading_env_assignments(folded)
        arg_uq: List[bool] = []; meta: Dict[str, bool] = {}
        verb, raw_verb, args_list, unwrap_unv = _unwrap_shell_command_tokens_with_raw(toks, uq, out_arg_flags=arg_uq, out_meta=meta)
        subcmds.append(ParsedSubcmd(sc, p_delim, next_delim, sc_uq, g_delta, opens, closes, is_piped_in, tuple(raw_toks), sc_redirs, has_in, tuple(toks), tuple(uq), tuple(env_assigns), stripped_env, verb, raw_verb, tuple(args_list), tuple(arg_uq), unwrap_unv, bool(meta.get("opaque"))))
    has_subshell = any(s.subshell_opens or s.subshell_closes for s in subcmds)
    return ParsedCommand(cmd, cleaned, unquoted, has_nul_surr, has_ansi_c, has_subshell, has_h_subst, has_unclosed, has_cmd_subst, has_or, has_masking, tuple(all_redirs), tuple(subcmds))


def _has_unquoted_cmd_subst(cmd: str) -> bool:
    return bool(cmd) and parse_shell_command(cmd).has_cmd_subst


def _has_unquoted_or_operator(cmd: str) -> bool:
    return bool(cmd) and parse_shell_command(cmd).has_or


def _strip_shell_quoted_literals(cmd: str) -> str:
    return parse_shell_command(cmd).unquoted if cmd else ""


def _advance_effective_cwd(sc: str, current_cwd: str, dir_stack: Optional[List[str]] = None, payload: Optional[Dict[str, Any]] = None, *, parsed_sub: Optional[ParsedSubcmd] = None) -> Tuple[str, bool, bool]:
    if (sub := _first_subcmd(sc, parsed_sub)) is None or sub.verb not in {"cd", "pushd", "popd"}:
        return current_cwd, False, False
    if sub.unwrap_unv or sub.verb in {"pushd", "popd"} or not _is_canonical_cmd_path(sub.verb, sub.raw_verb): return current_cwd, True, True
    after_dd, pos = False, []
    for tok in sub.args:
        if not after_dd and tok == "--": after_dd = True
        elif not after_dd and tok.startswith("-") and tok != "-":
            if not re.fullmatch(r"-[LPe@]+", tok): return current_cwd, True, True
        else: pos.append(tok)
    if len(pos) != 1 or not pos[0] or pos[0] in {"-", "~"} or any(ch in pos[0] for ch in "$`*?[]~{}"): return current_cwd, True, True
    return _resolve_cmd_target(pos[0], current_cwd), True, False


def _cmd_enters_guard_plugin_cwd(sub_cmds: List[str], cwd: str, *, parsed: Optional[ParsedCommand] = None) -> bool:
    eff = cwd or str(PROJECT_ROOT)
    if _is_guard_plugin_cwd(eff): return True
    if parsed is not None and parsed.has_subshell: return True
    subs = parsed.subcmds if parsed is not None else tuple(s for sc in sub_cmds for s in parse_shell_command(sc).subcmds)
    for sub in subs:
        if sub.subshell_opens or sub.subshell_closes: return True
        eff, is_cd, cd_unv = _advance_effective_cwd(sub.raw, eff, None, None, parsed_sub=sub)
        if is_cd and (cd_unv or _is_guard_plugin_cwd(eff)): return True
    return False


def _parse_git_subcmd_tokens(sc: str, *, allow_config_flags: bool = True, parsed_sub: Optional[ParsedSubcmd] = None) -> Optional[Tuple[str, List[str]]]:
    if (sub := _first_subcmd(sc, parsed_sub)) is None: return None
    if any(is_app or not ((k in {"GIT_EDITOR", "GIT_SEQUENCE_EDITOR"} and v in _SAFE_GIT_EDITOR_VALS) or (k in {"GIT_PAGER", "PAGER"} and v in _SAFE_PAGER_VALS) or (k == "GIT_TERMINAL_PROMPT" and v in {"0", "1"}) or (k in {"LC_ALL", "LANG"} and v in {"C", "C.UTF-8", "en_US.UTF-8"})) for k, is_app, v in sub.env_assigns):
        return None
    toks = list(sub.tokens[len(sub.env_assigns) :])
    if not toks or Path(toks[0]).name.lower() != "git" or not _is_canonical_cmd_path("git", toks[0]): return None
    val_opts = {"-C", "--git-dir", "--work-tree"} | ({"-c", "--namespace", "--super-prefix", "--attr-source"} if allow_config_flags else set())
    bool_opts = {"--no-pager", "--paginate", "-p", "-P", "--bare", "--no-replace-objects", "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs", "--no-optional-locks"}
    skip_next = False
    for idx, tok in enumerate(toks[1:], start=1):
        if skip_next: skip_next = False
        elif tok in val_opts: skip_next = True
        elif not (tok.startswith(tuple(o + "=" for o in val_opts if o.startswith("--"))) or (tok.startswith("-C") and len(tok) > 2) or (allow_config_flags and tok.startswith("-c") and len(tok) > 2 and not tok.startswith("--")) or tok in bool_opts):
            return None if tok.startswith("-") else (tok.lower(), toks[idx + 1 :])
    return ("", []) if not skip_next else None


def _is_git_only_command(cmd: str, *, parsed: Optional[ParsedCommand] = None, parsed_sub: Optional[ParsedSubcmd] = None) -> bool:
    if parsed_sub is not None:
        return not ("$(" in parsed_sub.raw or "`" in parsed_sub.raw or parsed_sub.redir_targets or "<(" in parsed_sub.unquoted or ">(" in parsed_sub.unquoted) and _parse_git_subcmd_tokens(parsed_sub.raw, allow_config_flags=True, parsed_sub=parsed_sub) is not None
    if not cmd or "$(" in cmd or "`" in cmd: return False
    p = parsed if parsed is not None else parse_shell_command(cmd)
    return bool(p.subcmds) and not (p.has_ansi_c or p.has_subshell or p.has_heredoc_subst or p.has_unclosed or p.redir_targets or "<(" in p.unquoted or ">(" in p.unquoted) and all(_parse_git_subcmd_tokens(s.raw, allow_config_flags=True, parsed_sub=s) is not None for s in p.subcmds)


def _is_safe_git_inspection_command(cmd: str, *, parsed: Optional[ParsedCommand] = None, parsed_sub: Optional[ParsedSubcmd] = None) -> bool:
    p = None if parsed_sub is not None else (parsed if parsed is not None else (parse_shell_command(cmd) if cmd else None))
    subs = (parsed_sub,) if parsed_sub is not None else (p.subcmds if p is not None else ())
    if not subs or (parsed_sub.redir_targets if parsed_sub is not None else p.redir_targets) or not _is_git_only_command(cmd, parsed=p, parsed_sub=parsed_sub): return False
    bad_opts = frozenset({"-c", "--output", "--ext-diff", "--exec-path", "--config-env", "--no-index", "--upload-pack", "--receive-pack", "--open-files-in-pager"})
    for s in subs:
        if (git_p := _parse_git_subcmd_tokens(s.raw, allow_config_flags=False, parsed_sub=s)) is None or git_p[0] not in {"status", "diff", "log"}: return False
        pre = git_p[1][: (git_p[1].index("--") if "--" in git_p[1] else len(git_p[1]))]
        if any("ext::" in t.lower() or (t.startswith("-c") and not t.startswith("--")) for t in pre) or walk_cli_flags(pre, FlagSpec(bad_opts=bad_opts, stop_at_dd=False, gnu_abbrev=True)).bad_hit: return False
    return True


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
        else: return tok if saw_c else None
    return None


def _is_trivial_eval_subcmd(subcmd: str, *, parsed_sub: Optional[ParsedSubcmd] = None) -> bool:
    if (sub := _first_subcmd(subcmd, parsed_sub)) is None: return True
    verb, args_t, stripped = sub.verb, list(sub.args), sub.stripped_env
    if not verb or TRIVIAL_EVAL_CMD_RE.match(" ".join([verb, *args_t])) or (verb == "cd" and len(args_t) == 1): return True
    from guard.ast_guard import _is_test_command
    if (TEST_CMD_START_RE.match(stripped) and not _is_test_command(stripped, parsed_sub=(sub if not sub.env_assigns else None))) or (_PY_VERB_RE.match(verb) and _parse_python_cli(args_t)[1] == "c"): return True
    if verb in (_SHELL_WRAPPER_VERBS - {"source", "."}) and (inner_sh := _extract_inline_sh(verb, args_t)): return _is_trivial_eval_command(inner_sh)
    return bool(TRIVIAL_TEST_FLAG_RE.search(stripped))


def _is_trivial_eval_command(eval_cmd: str, *, parsed: Optional[ParsedCommand] = None) -> bool:
    if not eval_cmd or not eval_cmd.strip(): return True
    p = parsed if parsed is not None else parse_shell_command(eval_cmd)
    if p.has_or or p.has_masking: return True
    for s in p.subcmds:
        if not _is_trivial_eval_subcmd(s.raw, parsed_sub=s): return s.next_delim in ("|", "|&", ";", "\n", "||", "&")
    return True


_UNSAFE_EVAL_FLAG_OPTS: Dict[str, Tuple[frozenset[str], str]] = {
    "find": (frozenset({"-delete", "-fls", "-exec", "-execdir", "-ok", "-okdir"}), ""), "rg": (frozenset({"--pre"}), ""),
    "sort": (frozenset({"-o", "--output", "--compress-program"}), "o"), "iconv": (frozenset({"-o", "--output"}), "o"),
    "curl": (frozenset({"-o", "--output", "-O", "--remote-name", "--remote-name-all", "-K", "--config", "-D", "--dump-header", "-c", "--cookie-jar"}), "oOKDc"),
}


def _is_unsafe_eval_command(cmd_str: str, *, parsed: Optional[ParsedCommand] = None, parsed_sub: Optional[ParsedSubcmd] = None) -> bool:
    if parsed_sub is not None:
        if parsed_sub.redir_targets or parsed_sub.subshell_opens or parsed_sub.subshell_closes: return True
        subs: Tuple[ParsedSubcmd, ...] = (parsed_sub,)
    else:
        if not cmd_str: return False
        p = parsed if parsed is not None else parse_shell_command(cmd_str)
        if p.has_nul_surr or p.has_ansi_c or p.has_subshell or p.has_heredoc_subst or p.has_unclosed or p.has_cmd_subst or p.redir_targets: return True
        subs = p.subcmds
    for s in subs:
        if s.unwrap_unv or s.opaque: return True
        args_t = list(s.args)
        if s.verb and (
            s.verb in (_SHELL_WRAPPER_VERBS | _UNVERIFIABLE_WRAPPERS | _OPAQUE_EXEC_VERBS | _MUTATING_EVAL_VERBS | {"pushd", "popd"}) or _INTERPRETER_EVAL_RE.match(s.verb)
            or (s.verb in (SAFE_READONLY_SHELL_VERBS | set(_UNSAFE_EVAL_FLAG_OPTS) | {"git"}) and not _is_canonical_cmd_path(s.verb, s.raw_verb))
            or (s.verb == "git" and not _is_safe_git_inspection_command(s.raw, parsed_sub=s))
            or (_PY_VERB_RE.match(s.verb) and (not _is_canonical_cmd_path(s.verb, s.raw_verb) or s.has_in_redir or not (py := _parse_python_cli(args_t))[0] or py[1] == "c"))
            or ((spec := _UNSAFE_EVAL_FLAG_OPTS.get(s.verb)) and (walk_cli_flags(args_t, FlagSpec(bad_opts=spec[0], bad_short_chars=spec[1], stop_at_dd=False, gnu_abbrev=True)).bad_hit or (s.verb == "find" and any(t.startswith("-fprint") for t in args_t))))
        ): return True
    return False


def _is_declared_eval_command(cmd: str, state: Dict[str, Any], payload: Dict[str, Any], *, parsed: Optional[ParsedCommand] = None, parsed_sub: Optional[ParsedSubcmd] = None) -> bool:
    """Return True if cmd invokes a non-trivial, non-mutating declared EVAL_CMD without shell-chaining bypasses."""
    if parsed_sub is not None:
        if "$(" in parsed_sub.raw or "`" in parsed_sub.raw or _is_unsafe_eval_command(parsed_sub.raw, parsed_sub=parsed_sub): return False
        subs, check_next = (parsed_sub,), False
    else:
        if not cmd or "$(" in cmd or "`" in cmd: return False
        p = parsed if parsed is not None else parse_shell_command(cmd)
        if p.has_or or p.has_masking or _is_unsafe_eval_command(cmd, parsed=p) or not p.subcmds: return False
        subs, check_next = p.subcmds, True
    cid, raw_subs = str(payload.get("conversationId") or "").strip().lower(), state.get("subagents")
    sub_entry = raw_subs.get(cid) if isinstance(raw_subs, dict) else None
    declared = [
        d for raw_d in (
            sub_entry.get("eval_cmd") if isinstance(sub_entry, dict) else None,
            (state.get("self_subagent_meta") or {}).get("eval_cmd") if isinstance(state.get("self_subagent_meta"), dict) else None,
            state.get("active_eval_cmd"), *(c.get("eval_cmd") for c in (state.get("cohorts") or {}).values() if isinstance(c, dict)),
        ) if isinstance(raw_d, str) and raw_d.strip() for d in (_strip_leading_env_assignments(raw_d),)
        if d and not _is_trivial_eval_command(d) and not _is_unsafe_eval_command(d)
    ]
    if not declared: return False
    matched_eval, cur_eval_dir = False, str(PROJECT_ROOT.resolve())
    for s in subs:
        if any(_is_dangerous_env_assignment(f"{k}{'+=' if is_app else '='}{v}") for k, is_app, v in s.env_assigns): return False
        if not (stripped := s.stripped_env): continue
        sub_for_stripped = s if not s.env_assigns else None
        if (matched_decl := next((d for d in declared if stripped == d or stripped.startswith(d + " ")), None)) is not None:
            if _is_trivial_eval_subcmd(stripped, parsed_sub=sub_for_stripped) or TRIVIAL_TEST_FLAG_RE.search(stripped[len(matched_decl) :]) or (check_next and s.next_delim in ("|", "|&", ";", "\n", "||", "&")): return False
            matched_eval = True; continue
        next_dir, is_cd, cd_unv = _advance_effective_cwd(stripped, cur_eval_dir, None, parsed_sub=sub_for_stripped)
        if is_cd and not cd_unv and ((nd := Path(next_dir).resolve()) == (pr := PROJECT_ROOT.resolve()) or pr in nd.parents) and not _is_staging_or_tmp_path(str(nd), {}):
            cur_eval_dir = str(nd); continue
        return False
    return matched_eval

"""Subagent identity tracking, transcript pairing, role resolution, and context reminder injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import posixpath
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ALLOWED_SYSTEM_PATH_PREFIXES,
    ISOLATED_WORKSPACES,
    MAX_CONSECUTIVE_FAILURES,
    PENDING_SPAWN_TTL_SECONDS,
    PROJECT_ROOT,
    SCRATCH_ONLY_SUBAGENT_TYPES,
    SLASH_COMMAND_RE,
    STRICT_READONLY_SUBAGENT_TYPES,
    SUBAGENT_SPAWN_RE,
    UUID_RE,
    _fold_shell_line_continuations,
    _has_nul_or_surrogate_input,
    _split_shell_commands,
)
from guard.state_store import (
    _current_invocation_key,
    _has_other_locked_path,
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)


def _allowed_roots(payload: Dict[str, Any]) -> List[Path]:
    roots: List[Path] = [PROJECT_ROOT]
    raw_wps = payload.get("workspacePaths")
    if isinstance(raw_wps, (list, tuple)):
        for wp in raw_wps:
            if not isinstance(wp, str) or not wp:
                continue
            try:
                roots.append(Path(wp).resolve())
            except (OSError, ValueError):
                pass
    artifact_dir = payload.get("artifactDirectoryPath")
    if isinstance(artifact_dir, str) and artifact_dir:
        try:
            resolved_artifact = Path(artifact_dir).resolve()
            roots.append(resolved_artifact)
            if resolved_artifact.parent.name == "brain":
                roots.append(resolved_artifact.parent)
                worktrees_dir = resolved_artifact.parent.parent / "worktrees"
                roots.append(worktrees_dir)
        except (OSError, ValueError):
            pass
    return roots


def _is_within_allowed_roots(target: Optional[str], roots: List[Path]) -> bool:
    if (
        not isinstance(target, str)
        or not target
        or "\x00" in target
        or any(0xD800 <= ord(ch) <= 0xDFFF for ch in target)
    ):
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except (OSError, ValueError, TypeError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


_WORKSPACE_PATH_PREFIX_STR = (
    r"(?:^|[\s=\"'><:,{};|&(!+?]|(?:^|[^A-Za-z0-9._/])-|(?:\$\{[^}\s]*[-])|(?:(?:^|\s)-[A-Za-z]+))"
)
_ANSI_C_ESCAPES = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n",
    "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?",
}
_WS_UNSAFE_WRAPPERS = frozenset(
    {
        "stdbuf", "setsid", "chrt", "ionice", "taskset", "flock", "script",
        "parallel", "busybox", "strace", "ltrace", "unshare", "nsenter",
        "chroot", "watch", "prlimit", "setarch", "linux32", "linux64", "numactl",
    }
)
_WS_SAFE_WRAPPER_FLAGS: Dict[str, Tuple[frozenset[str], str, int]] = {
    "builtin": (frozenset(), "", 0),
    "nohup": (frozenset(), "", 0),
    "command": (frozenset(), "", 0),
    "time": (frozenset({"-f", "--format", "-o", "--output"}), "of", 0),
    "nice": (frozenset({"-n", "--adjustment"}), "n", 0),
    "exec": (frozenset({"-a"}), "a", 0),
    "timeout": (frozenset({"-k", "--kill-after", "-s", "--signal"}), "ks", 1),
}
_WS_SHELL_INTERP_VERBS = frozenset(
    {"sh", "bash", "dash", "zsh", "ksh", "ash", "csh", "tcsh", "fish", "mksh", "rbash", "eval", "source", "."}
)


def _safe_ansi_char(cp: int) -> str:
    """Convert a decoded ANSI-C code point to a safe non-NUL, non-surrogate character."""
    if cp == 0:
        return " "
    return "\ufffd" if (0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF) else chr(cp)


def _decode_ansi_c_in_cmd(command_line: str) -> str:
    """Replace ANSI-C $'...' literals outside single/double quotes with decoded '...' literals."""
    command_line = _fold_shell_line_continuations(command_line)
    if "$'" not in command_line:
        return command_line
    out: List[str] = []
    in_sq = in_dq = escaped = False
    i, n = 0, len(command_line)
    while i < n:
        ch = command_line[i]
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\" and not in_sq:
            escaped = True
            out.append(ch)
        elif not in_sq and not in_dq and ch == "$" and i + 1 < n and command_line[i + 1] == "'":
            i += 2
            decoded: List[str] = []
            while i < n and command_line[i] != "'":
                sch = command_line[i]
                if sch == "\\" and i + 1 < n:
                    nxt = command_line[i + 1]
                    if nxt in _ANSI_C_ESCAPES:
                        decoded.append(_ANSI_C_ESCAPES[nxt])
                        i += 2
                    elif nxt == "c" and i + 2 < n and command_line[i + 2] != "'":
                        esc_c = command_line[i + 2] == "\\" and i + 3 < n and command_line[i + 3] in ("\\", "'")
                        ctrl_ch = command_line[i + 3] if esc_c else command_line[i + 2]
                        decoded.append(_safe_ansi_char(0x7F if ctrl_ch == "?" else (ord(ctrl_ch) & 0x1F)))
                        i += 4 if esc_c else 3
                    elif nxt in ("x", "u", "U"):
                        mlen = 2 if nxt == "x" else (4 if nxt == "u" else 8)
                        j = i + 2
                        while j < n and j < i + 2 + mlen and command_line[j] in "0123456789abcdefABCDEF":
                            j += 1
                        if j > i + 2:
                            decoded.append(_safe_ansi_char(int(command_line[i + 2 : j], 16)))
                            i = j
                        else:
                            decoded.append(nxt)
                            i += 2
                    elif nxt in "01234567":
                        j = i + 1
                        while j < n and j < i + 4 and command_line[j] in "01234567":
                            j += 1
                        decoded.append(_safe_ansi_char(int(command_line[i + 1 : j], 8) & 0xFF))
                        i = j
                    else:
                        decoded.append(nxt)
                        i += 2
                else:
                    decoded.append(sch)
                    i += 1
            if i < n and command_line[i] == "'":
                i += 1
            out.append("'" + "".join(decoded).replace("'", "'\\''") + "'")
            continue
        else:
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            out.append(ch)
        i += 1
    return "".join(out)


def _extract_dequoted_shell_words(text: str) -> List[Tuple[str, bool]]:
    """Split shell text into (dequoted_word, was_quoted) pairs, folding line continuations."""
    if not text:
        return []
    words: List[Tuple[str, bool]] = []
    buf: List[str] = []
    has_tok = saw_quote = in_sq = in_dq = False
    i, n = 0, len(text)

    def _flush() -> None:
        nonlocal buf, has_tok, saw_quote
        if has_tok:
            words.append(("".join(buf), saw_quote))
            buf = []
            has_tok = saw_quote = False

    while i < n:
        ch = text[i]
        if in_sq:
            if ch == "'":
                in_sq = False
            else:
                buf.append(ch)
        elif in_dq:
            if ch == "\\" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "\n" or (nxt == "\r" and i + 2 < n and text[i + 2] == "\n"):
                    i += 3 if nxt == "\r" else 2
                    continue
                if nxt in ('"', "\\", "$", "`"):
                    buf.append(nxt)
                else:
                    buf.extend((ch, nxt))
                i += 2
                continue
            if ch == '"':
                in_dq = False
            else:
                buf.append(ch)
        elif ch in " \t\r\n;|&<>()":
            _flush()
        elif ch == "'":
            in_sq = has_tok = saw_quote = True
        elif ch == '"':
            in_dq = has_tok = saw_quote = True
        elif ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "\n" or (nxt == "\r" and i + 2 < n and text[i + 2] == "\n"):
                i += 3 if nxt == "\r" else 2
                continue
            has_tok = True
            buf.append(nxt)
            i += 2
            continue
        else:
            has_tok = True
            buf.append(ch)
        i += 1
    _flush()
    return words


def _mask_active_subst_for_ws(text: str) -> Tuple[str, str]:
    """Return (scan_text, active_subst_text) with single-quoted and escaped chars masked."""
    scan_chars: List[str] = []
    subst_chars: List[str] = []
    in_sq = in_dq = False
    idx, cmd_len = 0, len(text)
    while idx < cmd_len:
        ch = text[idx]
        if in_sq:
            if ch == "'":
                in_sq = False
            scan_chars.append(ch)
            subst_chars.append(" ")
        elif in_dq:
            if ch == "\\" and idx + 1 < cmd_len:
                nxt = text[idx + 1]
                if nxt == "\n" or (nxt == "\r" and idx + 2 < cmd_len and text[idx + 2] == "\n"):
                    idx += 3 if nxt == "\r" else 2
                    continue
                if nxt in ('"', "\\", "$", "`"):
                    scan_chars.append(nxt)
                    subst_chars.append(" ")
                else:
                    scan_chars.extend((ch, nxt))
                    subst_chars.extend((ch, nxt))
                idx += 2
                continue
            if ch == '"':
                in_dq = False
            scan_chars.append(ch)
            subst_chars.append(ch)
        elif ch == "'":
            in_sq = True
            scan_chars.append(ch)
            subst_chars.append(" ")
        elif ch == '"':
            in_dq = True
            scan_chars.append(ch)
            subst_chars.append(ch)
        elif ch == "\\" and idx + 1 < cmd_len:
            nxt = text[idx + 1]
            if nxt == "\n" or (nxt == "\r" and idx + 2 < cmd_len and text[idx + 2] == "\n"):
                idx += 3 if nxt == "\r" else 2
                continue
            scan_chars.append(nxt)
            subst_chars.append(" " if nxt in ("$", "`", "'", '"', "\\") else nxt)
            idx += 2
            continue
        else:
            scan_chars.append(ch)
            subst_chars.append(ch)
        idx += 1
    return "".join(scan_chars), "".join(subst_chars)


def _command_stays_in_workspace(
    command_line: str, cwd: str, roots: List[Path], *, _depth: int = 0
) -> bool:
    """Return True unless the command targets paths outside allowed roots."""
    if _has_nul_or_surrogate_input(command_line):
        return False
    decoded_cmd = _decode_ansi_c_in_cmd(command_line)
    scan_cmd, active_subst = _mask_active_subst_for_ws(decoded_cmd)
    if "$(" in active_subst or "`" in active_subst:
        return False
    if re.search(
        r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)(?:[^\s;|&'\"<>]*)/",
        active_subst,
    ) or re.search(r"\$\{(?:(?:HOME|OLDPWD)[^}A-Za-z0-9_][^}]*|![^}]+)\}", active_subst):
        return False
    try:
        base_dir = Path(os.path.expanduser(cwd)).resolve() if cwd else PROJECT_ROOT
    except (OSError, ValueError):
        return False

    def _check_path_matches(
        text_to_scan: str,
        *,
        is_word: bool = False,
        word_was_quoted: bool = False,
        cmd_verb: str = "",
        cur_base_dir: Optional[Path] = None,
    ) -> bool:
        active_base = cur_base_dir or base_dir
        for match in re.finditer(rf"{_WORKSPACE_PATH_PREFIX_STR}(/[A-Za-z0-9._/\-]*)", text_to_scan):
            abs_path = match.group(1)
            start_idx, end_idx = match.span(1)
            if abs_path.startswith("//"):
                scheme = re.search(r"([A-Za-z][A-Za-z0-9+.-]*):$", text_to_scan[:start_idx])
                if scheme and scheme.group(1).lower() != "file":
                    continue
            norm_abs = re.sub(r"^/+", "/", abs_path)
            if is_word:
                if (
                    word_was_quoted
                    and cmd_verb in {"awk", "gawk", "mawk", "nawk", "sed"}
                    and start_idx == 1
                    and end_idx == len(text_to_scan)
                    and re.fullmatch(r"/[^/]+/[dgip]?", norm_abs)
                ):
                    continue
            elif (
                start_idx > 0
                and text_to_scan[start_idx - 1] in ("'", '"')
                and end_idx < len(text_to_scan)
                and text_to_scan[end_idx] in ("'", '"', " ")
                and re.fullmatch(r"/[^/]+/[dgip]?", norm_abs)
            ):
                prefix_scs = _split_shell_commands(text_to_scan[:start_idx])
                cur_pref = prefix_scs[-1] if prefix_scs else text_to_scan[:start_idx]
                if re.search(r"(?:^|[;&|\n({])\s*(?:[gmn]?awk|sed)\b", cur_pref):
                    continue
            norm_abs = posixpath.normpath(norm_abs)
            if norm_abs == "/tmp" or norm_abs.startswith(ALLOWED_SYSTEM_PATH_PREFIXES):
                continue
            if not _is_within_allowed_roots(norm_abs, roots):
                return False

        home_val = os.environ.get("HOME") or str(Path.home())
        oldpwd_val = os.environ.get("OLDPWD") or ""
        for match in re.finditer(
            rf"{_WORKSPACE_PATH_PREFIX_STR}(\$(?:\{{HOME\}}|HOME|\{{OLDPWD\}}|OLDPWD)[A-Za-z0-9._/\-]*)",
            text_to_scan,
        ):
            exp = (
                match.group(1)
                .replace("${HOME}", home_val).replace("$HOME", home_val)
                .replace("${OLDPWD}", oldpwd_val).replace("$OLDPWD", oldpwd_val)
            )
            if not _is_within_allowed_roots(exp, roots):
                return False
        for match in re.finditer(rf"{_WORKSPACE_PATH_PREFIX_STR}(~[A-Za-z0-9._/\-+]*)", text_to_scan):
            raw_tilde = match.group(1)
            if raw_tilde.startswith(("~-", "~+")):
                return False
            try:
                tilde_path = os.path.expanduser(raw_tilde)
            except (OSError, ValueError):
                return False
            if tilde_path == raw_tilde or not _is_within_allowed_roots(tilde_path, roots):
                return False
        for match in re.finditer(rf"{_WORKSPACE_PATH_PREFIX_STR}([A-Za-z0-9._\-/]*\.\.[A-Za-z0-9._\-/]*)", text_to_scan):
            try:
                cand = (active_base / match.group(1)).resolve()
            except (OSError, ValueError):
                return False
            if not _is_within_allowed_roots(str(cand), roots):
                return False
        for match in re.finditer(rf"{_WORKSPACE_PATH_PREFIX_STR}([A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*)", text_to_scan):
            tok = match.group(1)
            if tok.startswith(("-", "$")):
                continue
            try:
                cand = active_base / tok
                chk, has_sym, cur = cand, False, active_base
                for part in Path(tok).parts:
                    cur = cur / part
                    if cur.is_symlink():
                        has_sym, chk = True, cur
                        break
                if cand.is_symlink() or cand.exists() or has_sym:
                    res = (cand if (cand.is_symlink() or cand.exists()) else chk).resolve()
                    if not _is_within_allowed_roots(str(res), roots):
                        return False
            except (OSError, ValueError):
                return False
        return True

    if not _check_path_matches(scan_cmd, is_word=False):
        return False

    def _count_parens(text: str) -> Tuple[int, int]:
        opens = closes = 0
        in_s = in_d = esc = False
        for c in text:
            if esc:
                esc = False
            elif in_s:
                in_s = c != "'"
            elif c == "\\":
                esc = True
            elif in_d:
                in_d = c != '"'
            elif c == "'":
                in_s = True
            elif c == '"':
                in_d = True
            elif c == "(":
                opens += 1
            elif c == ")":
                closes += 1
        return opens, closes

    def _short_flag_step(tok: str, arg_flags: str) -> int:
        for c_idx, ch in enumerate(tok[1:]):
            if ch in arg_flags:
                return 2 if c_idx == len(tok[1:]) - 1 else 1
        return 1

    cur_base_dir = base_dir
    dir_stack: List[Path] = []
    subshell_stack: List[Tuple[Path, List[Path]]] = []
    subcmds = _split_shell_commands(decoded_cmd) or [decoded_cmd]
    for subcmd in subcmds:
        opens, closes = _count_parens(subcmd)
        for _ in range(opens):
            subshell_stack.append((cur_base_dir, list(dir_stack)))
        words = _extract_dequoted_shell_words(subcmd)
        cmd_verb, verb_idx, w_idx = "", -1, 0
        while w_idx < len(words):
            w, _ = words[w_idx]
            if w in {"{", "!", "(", "if", "then", "elif", "else", "while", "until", "do"} or re.match(
                r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", w
            ):
                w_idx += 1
                continue
            base_w = posixpath.basename(w).lower()
            if base_w in _WS_UNSAFE_WRAPPERS:
                return False
            if base_w in _WS_SAFE_WRAPPER_FLAGS:
                val_longs, val_shorts, trail_pos = _WS_SAFE_WRAPPER_FLAGS[base_w]
                w_idx += 1
                while w_idx < len(words):
                    tok = words[w_idx][0]
                    if tok == "--":
                        w_idx += 1
                        break
                    if tok in val_longs:
                        w_idx += 2
                    elif tok.startswith(tuple(f + "=" for f in val_longs if f.startswith("--"))):
                        w_idx += 1
                    elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 1 and val_shorts:
                        w_idx += _short_flag_step(tok, val_shorts)
                    elif tok.startswith("-"):
                        w_idx += 1
                    else:
                        break
                w_idx += trail_pos
                continue
            if base_w == "env":
                w_idx += 1
                while w_idx < len(words):
                    tok = words[w_idx][0]
                    if tok == "--":
                        w_idx += 1
                        break
                    if (
                        tok in {"-C", "--chdir", "-S", "--split-string"}
                        or tok.startswith(("--chdir=", "--split-string="))
                        or (tok.startswith("-") and not tok.startswith("--") and any(c in tok[1:] for c in "CS"))
                    ):
                        return False
                    if tok in {"-u", "--unset", "-a", "--argv0"}:
                        w_idx += 2
                    elif tok.startswith(("--unset=", "--argv0=")):
                        w_idx += 1
                    elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
                        w_idx += _short_flag_step(tok, "ua")
                    elif tok.startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_%-]*\+?=", tok):
                        w_idx += 1
                    else:
                        break
                continue
            cmd_verb, verb_idx = base_w, w_idx
            break

        _, sub_active = _mask_active_subst_for_ws(subcmd)
        # Medium #1: fail closed on coproc and unquoted brace expansion
        if cmd_verb == "coproc" or re.search(r"(?:^|[;&|\n({]\s*)coproc\b", sub_active):
            return False
        if re.search(r"(?<!\$)\{[^{}]*(?:,|\.\.)[^{}]*\}", sub_active):
            return False

        if cmd_verb in _WS_SHELL_INTERP_VERBS:
            sh_args = [w for w, _ in words[verb_idx + 1 :]]
            if not (cmd_verb not in {"eval", "source", "."} and sh_args and all(t in {"--version", "--help", "-h"} for t in sh_args)):
                return False
        if cmd_verb in {"read", "mapfile", "readarray", "declare", "typeset"}:
            return False
        if cmd_verb == "printf" and verb_idx >= 0:
            for tok, was_q in words[verb_idx + 1 :]:
                if tok == "--":
                    break
                if (
                    tok == "-v"
                    or (tok.startswith("-") and not tok.startswith("--") and "v" in tok[1:])
                    or (not was_q and ("{" in tok or "}" in tok))
                ):
                    return False

        if re.search(
            r"[^\s;|&'\"<>()=]\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)"
            r"|\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)[^\s;|&'\"<>()]",
            sub_active,
        ):
            return False
        if cmd_verb != "echo" and re.search(
            r"\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*|\d+|[*@#?$!-])", sub_active
        ):
            return False
        for word, was_quoted in words:
            if not _check_path_matches(
                f" {word}", is_word=True, word_was_quoted=was_quoted, cmd_verb=cmd_verb, cur_base_dir=cur_base_dir
            ):
                return False
        if cmd_verb in {"cd", "pushd", "popd"} and verb_idx >= 0:
            cd_args = [w for w, _ in words[verb_idx + 1 :]]
            if cmd_verb == "popd":
                if cd_args or not dir_stack:
                    return False
                cur_base_dir = dir_stack.pop()
            else:
                after_dd = False
                pos_cd: List[str] = []
                for tok in cd_args:
                    if not after_dd and tok == "--":
                        after_dd = True
                        continue
                    if not after_dd and ((tok.startswith("-") and tok != "-") or (cmd_verb == "pushd" and tok.startswith("+"))):
                        if cmd_verb == "cd" and re.fullmatch(r"-[LPe@]+", tok):
                            continue
                        return False
                    pos_cd.append(tok)
                if len(pos_cd) != 1:
                    return False
                cd_arg = pos_cd[0]
                if not cd_arg or cd_arg in {"-", "~"} or any(ch in cd_arg for ch in "$`*?[]~{}"):
                    return False
                try:
                    new_base = (cur_base_dir / cd_arg).resolve()
                except (OSError, ValueError):
                    return False
                norm_new = posixpath.normpath(str(new_base))
                if not (
                    _is_within_allowed_roots(str(new_base), roots)
                    or norm_new == "/tmp"
                    or norm_new.startswith(ALLOWED_SYSTEM_PATH_PREFIXES)
                ):
                    return False
                if cmd_verb == "pushd":
                    dir_stack.append(cur_base_dir)
                cur_base_dir = new_base
        for _ in range(closes):
            if subshell_stack:
                cur_base_dir, dir_stack = subshell_stack.pop()
    return True


def _is_parent_orchestrator(cid: str, state: Dict[str, Any]) -> bool:
    """Return True if cid is recorded as the root/parent orchestrator."""
    subagents = state.get("subagents")
    if not cid or (isinstance(subagents, dict) and cid in subagents):
        return False
    if state.get("root_conversation_id") == cid:
        return True
    prefix = f'["{cid}"'
    recorded = state.get("recorded_spawn_calls")
    if not isinstance(recorded, list):
        return False
    return any(
        isinstance(call, str) and call.startswith(prefix) for call in recorded
    )


def _normalize_subagent_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(meta)
    if "type" in normalized and "type_name" not in normalized:
        normalized["type_name"] = normalized["type"]
    elif "type_name" in normalized and "type" not in normalized:
        normalized["type"] = normalized["type_name"]
    if meta.get("candidate_id") is not None:
        cand_val = str(meta["candidate_id"]).strip()
        if cand_val:
            normalized["candidate_id"] = cand_val
        else:
            normalized.pop("candidate_id", None)
    for field_name in ("council_id", "council_role"):
        if meta.get(field_name) is not None:
            field_val = str(meta[field_name]).strip()
            if field_val:
                normalized[field_name] = field_val
            else:
                normalized.pop(field_name, None)
    if "active_worktree" not in normalized:
        ws = str(normalized.get("workspace") or "").strip().lower()
        if ws in ISOLATED_WORKSPACES or normalized.get("active_branch"):
            normalized["active_worktree"] = bool(
                normalized.get("active_branch") or ws in ISOLATED_WORKSPACES
            )
    return normalized


def _workspaces_compatible(
    caller_payload: Dict[str, Any], sib_data: Dict[str, Any]
) -> bool:
    """Check whether caller payload and sibling state belong to compatible workspaces."""
    raw_caller_wps = caller_payload.get("workspacePaths")
    caller_wps = raw_caller_wps if isinstance(raw_caller_wps, (list, tuple)) else []
    raw_sib_roots = sib_data.get("workspace_roots")
    sib_roots = raw_sib_roots if isinstance(raw_sib_roots, (list, tuple)) else []
    if not caller_wps or not sib_roots:
        return True

    caller_resolved = set()
    for wp in caller_wps:
        if not isinstance(wp, str) or not wp:
            continue
        try:
            caller_resolved.add(str(Path(wp).resolve()))
        except (OSError, ValueError):
            pass

    sib_resolved = set()
    for wp in sib_roots:
        if not isinstance(wp, str) or not wp:
            continue
        try:
            sib_resolved.add(str(Path(wp).resolve()))
        except (OSError, ValueError):
            pass

    if not caller_resolved or not sib_resolved:
        return True

    if caller_resolved & sib_resolved:
        return True

    for c_path in caller_resolved:
        c_parts = Path(c_path).parts
        for s_path in sib_resolved:
            s_parts = Path(s_path).parts
            if "worktrees" in c_parts and "worktrees" in s_parts:
                c_idx = c_parts.index("worktrees")
                s_idx = s_parts.index("worktrees")
                if c_parts[: c_idx + 1] == s_parts[: s_idx + 1]:
                    return True
            if "worktrees" in c_parts and str(Path(s_path)) in c_path:
                return True
            if "worktrees" in s_parts and str(Path(c_path)) in s_path:
                return True

    return False


def _read_sibling_snapshot(sibling_state_path: Path) -> Optional[Dict[str, Any]]:
    """Read sibling state safely without acquiring LOCK_SH when another state lock is already held."""
    if _has_other_locked_path(sibling_state_path):
        try:
            data = json.loads(sibling_state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    try:
        return _load_state(sibling_state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _is_non_subagent_user_session(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True when payload/state clearly identifies an independent root user session."""
    if payload.get("isSubagent") is True:
        return False
    if payload.get("isSubagent") is False:
        return True
    if state.get("session_evaluated") is True and state.get("active") is False:
        return True
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    raw_wps = payload.get("workspacePaths")
    wps_iter = raw_wps if isinstance(raw_wps, (list, tuple)) else ()
    if (
        state.get("session_evaluated") is True
        and not state.get("self_subagent_meta")
        and not state.get("is_subagent_session")
        and payload.get("isSubagent") is not True
        and not (cid and isinstance(subagents, dict) and cid in subagents)
        and not any(
            isinstance(wp, str) and "/worktrees/" in wp.replace("\\", "/")
            for wp in wps_iter
        )
    ):
        return True
    if (
        "lastUserInput" in payload
        and payload.get("lastUserInput") is not None
        and not _is_slash_command_triggered(payload)
    ):
        return True
    return False


def _is_known_subagent(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    *,
    persist_self_meta: bool = True,
) -> bool:
    """Return True if this invocation belongs to a spawned worker subagent."""
    if state.get("self_subagent_meta"):
        return True
    cid = str(payload.get("conversationId") or "").strip().lower()
    if _is_parent_orchestrator(cid, state):
        return False
    local_subagents = state.get("subagents")
    if cid and isinstance(local_subagents, dict) and isinstance(local_subagents.get(cid), dict):
        return True
    raw_tc = payload.get("toolCall")
    tool_call = raw_tc if isinstance(raw_tc, dict) else {}
    raw_name = tool_call.get("name")
    tool_name = raw_name if isinstance(raw_name, str) else ""
    can_claim_pending = (
        not _is_non_subagent_user_session(payload, state)
        and tool_name not in {"exit_plan_mode", "enter_plan_mode"}
    )
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                for sibling_state_path in resolved_art.parent.glob(
                    "*/orchestrator_state.json"
                ):
                    if sibling_state_path.parent == resolved_art:
                        continue
                    sib_data = _read_sibling_snapshot(sibling_state_path)
                    if not isinstance(sib_data, dict):
                        continue

                    if not _workspaces_compatible(payload, sib_data):
                        continue

                    sibling_subagents = sib_data.get("subagents")
                    if (
                        cid
                        and isinstance(sibling_subagents, dict)
                        and isinstance(sibling_subagents.get(cid), dict)
                    ):
                        meta = _normalize_subagent_meta(sibling_subagents[cid])
                        if "parent_conversation_id" not in meta:
                            meta["parent_conversation_id"] = (
                                sib_data.get("root_conversation_id")
                                or sibling_state_path.parent.name
                            )
                        state["self_subagent_meta"] = meta
                        if persist_self_meta:
                            _save_state(payload, state)
                        return True

                    if not can_claim_pending:
                        continue

                    pending_list = sib_data.get("pending_spawns")
                    if not isinstance(pending_list, list):
                        pending_list = []
                    has_active_pending = any(
                        isinstance(p, dict) and p.get("status", "active") == "active"
                        for p in pending_list
                    )
                    if (
                        has_active_pending
                        and cid != sib_data.get("root_conversation_id")
                        and not _has_other_locked_path(sibling_state_path)
                    ):
                        claimed_meta: Optional[Dict[str, Any]] = None
                        with state_transaction(sibling_state_path):
                            fresh_sib_data = _load_state(sibling_state_path)
                            if _workspaces_compatible(payload, fresh_sib_data) and (
                                not _is_parent_orchestrator(cid, fresh_sib_data)
                            ):
                                fresh_subagents = fresh_sib_data.get("subagents")
                                if (
                                    cid
                                    and isinstance(fresh_subagents, dict)
                                    and isinstance(fresh_subagents.get(cid), dict)
                                ):
                                    meta = _normalize_subagent_meta(fresh_subagents[cid])
                                    if "parent_conversation_id" not in meta:
                                        meta["parent_conversation_id"] = (
                                            fresh_sib_data.get("root_conversation_id")
                                            or sibling_state_path.parent.name
                                        )
                                    claimed_meta = meta
                                else:
                                    raw_pending = fresh_sib_data.get("pending_spawns")
                                    fresh_pending = raw_pending if isinstance(raw_pending, list) else []
                                    active_idx = next(
                                        (
                                            idx
                                            for idx, p in enumerate(fresh_pending)
                                            if isinstance(p, dict)
                                            and p.get("status", "active") == "active"
                                        ),
                                        None,
                                    )
                                    if active_idx is not None:
                                        try:
                                            age = time.time() - sibling_state_path.stat().st_mtime
                                            if age < PENDING_SPAWN_TTL_SECONDS:
                                                parent_cid = (
                                                    fresh_sib_data.get("root_conversation_id")
                                                    or sibling_state_path.parent.name
                                                )
                                                if cid:
                                                    claimed = _normalize_subagent_meta(
                                                        fresh_pending.pop(active_idx)
                                                    )
                                                    claimed.setdefault("status", "active")
                                                    claimed["parent_conversation_id"] = parent_cid
                                                    fresh_sib_data["pending_spawns"] = fresh_pending
                                                    fresh_sib_data.setdefault("subagents", {})[cid] = claimed
                                                    _save_state(sibling_state_path, fresh_sib_data)
                                                else:
                                                    claimed = _normalize_subagent_meta(
                                                        fresh_pending[active_idx]
                                                    )
                                                    claimed.setdefault("status", "active")
                                                    claimed["parent_conversation_id"] = parent_cid
                                                claimed_meta = claimed
                                        except (OSError, ValueError):
                                            pass
                        if claimed_meta is not None:
                            state["self_subagent_meta"] = claimed_meta
                            if persist_self_meta:
                                _save_state(payload, state)
                            return True
        except (OSError, ValueError):
            pass
    if payload.get("isSubagent") is True or state.get("is_subagent_session") is True:
        return True
    raw_wps = payload.get("workspacePaths")
    if isinstance(raw_wps, (list, tuple)):
        for wp in raw_wps:
            if isinstance(wp, str) and "/worktrees/" in wp.replace("\\", "/"):
                return True
    return False


def _preclaim_subagent_meta(payload: Dict[str, Any]) -> None:
    """Resolve and persist sibling subagent metadata before acquiring state_transaction(payload)."""
    if _is_slash_command_triggered(payload):
        return
    raw_tc = payload.get("toolCall")
    tool_call = raw_tc if isinstance(raw_tc, dict) else {}
    raw_name = tool_call.get("name")
    if (raw_name if isinstance(raw_name, str) else "") in {
        "exit_plan_mode",
        "enter_plan_mode",
    }:
        return
    pre_state = _load_state(payload)
    if pre_state.get("self_subagent_meta"):
        return
    if _is_non_subagent_user_session(payload, pre_state):
        return
    cid = str(payload.get("conversationId") or "").strip().lower()
    if _is_parent_orchestrator(cid, pre_state):
        return
    pre_subagents = pre_state.get("subagents")
    if cid and isinstance(pre_subagents, dict) and cid in pre_subagents:
        return
    _is_known_subagent(payload, pre_state, persist_self_meta=False)
    claimed = pre_state.get("self_subagent_meta")
    if claimed is not None:
        with state_transaction(payload):
            own_state = _load_state(payload)
            if not own_state.get("self_subagent_meta"):
                own_state["self_subagent_meta"] = claimed
            own_state["is_subagent_session"] = True
            _save_state(payload, own_state)


def _sync_subagents_from_transcript(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> bool:
    """Scan transcript.jsonl incrementally to pair pending_spawns with conversationIds."""
    transcript_path = payload.get("transcriptPath")
    try:
        if (
            not isinstance(transcript_path, str)
            or not transcript_path
            or not os.path.exists(transcript_path)
        ):
            return False
    except (OSError, ValueError):
        return False

    raw_pending = state.get("pending_spawns")
    pending: List[Dict[str, Any]] = raw_pending if isinstance(raw_pending, list) else []
    raw_subagents = state.get("subagents")
    if not isinstance(raw_subagents, dict):
        raw_subagents = {}
        state["subagents"] = raw_subagents
    subagents: Dict[str, Dict[str, Any]] = raw_subagents
    parent_cid = str(payload.get("conversationId") or "").strip().lower()
    changed = False

    try:
        file_size = os.path.getsize(transcript_path)
        offset = int(state.get("transcript_offset", 0))
        if offset < 0 or offset > file_size:
            offset = 0

        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            raw_new_offset = fh.tell()

        if not chunk:
            return changed

        if chunk.endswith(b"\n"):
            process_chunk = chunk
            new_offset = raw_new_offset
        else:
            last_nl = chunk.rfind(b"\n")
            if last_nl != -1:
                process_chunk = chunk[: last_nl + 1]
                new_offset = offset + last_nl + 1
            else:
                stripped = chunk.strip()
                if stripped.startswith(b"{") and not stripped.endswith(b"}"):
                    return changed
                process_chunk = chunk
                new_offset = raw_new_offset

        if new_offset != offset:
            state["transcript_offset"] = new_offset
            changed = True

        text = process_chunk.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            if (
                "docs/hooks.md" in raw_line
                or "docs/" in raw_line
                or "view_file" in raw_line
                or "read_file" in raw_line
                or "File Path: `file://" in raw_line
            ):
                continue
            if not (
                "Created the following subagent" in raw_line
                or "invoke_subagent" in raw_line
            ):
                continue
            matches = list(SUBAGENT_SPAWN_RE.finditer(raw_line))
            if not matches and "Created the following subagent" in raw_line:
                matches = list(UUID_RE.finditer(raw_line))
            for match in matches:
                cid = match.group(1).lower()
                if cid == parent_cid or cid in subagents:
                    continue
                active_idx = next(
                    (
                        idx
                        for idx, p in enumerate(pending)
                        if isinstance(p, dict)
                        and p.get("status", "active") == "active"
                    ),
                    None,
                )
                if active_idx is not None:
                    spawn_meta = pending.pop(active_idx)
                    if isinstance(spawn_meta, dict):
                        cand_id = spawn_meta.get("candidate_id")
                        c_id = spawn_meta.get("council_id")
                        c_role = spawn_meta.get("council_role")
                        spawn_meta = _normalize_subagent_meta(spawn_meta)
                        if cand_id is not None and "candidate_id" not in spawn_meta:
                            spawn_meta["candidate_id"] = str(cand_id)
                        if c_id is not None and "council_id" not in spawn_meta:
                            spawn_meta["council_id"] = str(c_id)
                        if c_role is not None and "council_role" not in spawn_meta:
                            spawn_meta["council_role"] = str(c_role)
                        spawn_meta.setdefault("status", "active")
                    subagents[cid] = spawn_meta
                    changed = True
    except (OSError, ValueError):
        return changed

    return changed


def _is_slash_command_triggered(payload: Dict[str, Any]) -> bool:
    if payload.get("isSubagent") is True:
        return False
    user_input = payload.get("lastUserInput")
    if isinstance(user_input, str):
        if (
            re.search(
                r"(?m)^\s*(?:ALLOWED_FILES|SPEC_FILE|REVIEW_LENS|ROLE|WORK_DIR|TASK|OBJECTIVE|FILES|CONSTRAINTS|ACCEPTANCE_CRITERIA):\s*\S",
                user_input,
            )
            and not SLASH_COMMAND_RE.search(user_input)
        ):
            return False
        if SLASH_COMMAND_RE.search(user_input) or "skills/philharmonic/SKILL.md" in user_input:
            return True

    raw_tc = payload.get("toolCall")
    tool_call = raw_tc if isinstance(raw_tc, dict) else {}
    if tool_call:
        raw_name = tool_call.get("name")
        tool_name = raw_name.lower() if isinstance(raw_name, str) else ""
        raw_args = tool_call.get("args")
        args = raw_args if isinstance(raw_args, dict) else {}
        if tool_name in {"read_file", "view_file"}:
            for key in ("Path", "FilePath", "file_path", "path", "AbsolutePath"):
                val = args.get(key)
                if isinstance(val, str) and val.replace("\\", "/").endswith("skills/philharmonic/SKILL.md"):
                    return True
        elif tool_name == "lookup_skill":
            skill_name = str(args.get("skill_name") or args.get("SkillName") or "").strip().lower()
            if skill_name == "philharmonic":
                return True
        elif tool_name == "lookup_skills":
            skill_names = args.get("skill_names") or args.get("SkillNames") or []
            if isinstance(skill_names, list) and any(
                str(s).strip().lower() == "philharmonic" for s in skill_names
            ):
                return True
    return False


def _bind_root_session_if_unset(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Bind root_conversation_id on state when caller is a non-subagent root session and it is unset."""
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    if (
        cid
        and not state.get("root_conversation_id")
        and payload.get("isSubagent") is not True
        and not state.get("self_subagent_meta")
        and not state.get("is_subagent_session")
        and not (isinstance(subagents, dict) and cid in subagents)
    ):
        state["root_conversation_id"] = cid
        return True
    return False


def _ensure_session_active(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    state_path: Optional[Path] = None,
) -> bool:
    if _is_slash_command_triggered(payload):
        state["active"] = True
        state["session_evaluated"] = True
        _bind_root_session_if_unset(payload, state)
        if state_path is not None:
            _save_state(state_path, state)
        return True

    if state.get("active") is True:
        return True

    if _is_non_subagent_user_session(payload, state):
        if not state.get("session_evaluated"):
            state["session_evaluated"] = True
            state["active"] = False
            if state_path is not None:
                _save_state(state_path, state)
        return False

    if _is_known_subagent(payload, state):
        state["active"] = True
        state["session_evaluated"] = True
        cid = str(payload.get("conversationId") or "").strip().lower()
        subagents = state.get("subagents")
        if not (cid and isinstance(subagents, dict) and cid in subagents) and not state.get(
            "root_conversation_id"
        ):
            state["is_subagent_session"] = True
        if state_path is not None:
            _save_state(state_path, state)
        return True

    if "lastUserInput" in payload and payload.get("lastUserInput") is not None:
        if not state.get("session_evaluated"):
            state["session_evaluated"] = True
            state["active"] = False
            if state_path is not None:
                _save_state(state_path, state)
        return False

    if state.get("session_evaluated") is True:
        return bool(state.get("active", False))

    return True


def handle_pre_invocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    _preclaim_subagent_meta(payload)
    with state_transaction(payload):
        state_path = _state_file_path(payload)
        state = _load_state(payload)
        if not _ensure_session_active(payload, state, state_path):
            return {}
        cid = str(payload.get("conversationId") or "").strip().lower()

        if not _is_known_subagent(payload, state):
            changed = False
            if cid and not state.get("root_conversation_id"):
                state["root_conversation_id"] = cid
                changed = True

            user_input = str(payload.get("lastUserInput") or "")
            if state.get("plan_execute_phase"):
                norm_input = user_input.strip()
                should_reset = (
                    "[plan review]" in user_input.lower()
                    or (bool(norm_input) and bool(state.get("plan_execute_turn_finished")))
                    or (
                        bool(norm_input)
                        and bool(state.get("plan_execute_consumed"))
                        and state.get("plan_execute_user_input") is not None
                        and norm_input != state.get("plan_execute_user_input")
                    )
                )
                if should_reset:
                    state["plan_execute_phase"] = False
                    state["plan_execute_consumed"] = False
                    state["plan_execute_user_input"] = None
                    state["plan_execute_turn_finished"] = False
                    state["plan_allowed_files"] = None
                    state["active_eval_cmd"] = None
                    changed = True
                else:
                    if not state.get("plan_execute_consumed"):
                        state["plan_execute_consumed"] = True
                        changed = True
                    if norm_input and not state.get("plan_execute_user_input"):
                        state["plan_execute_user_input"] = norm_input
                        changed = True

            wps: List[str] = []
            raw_wps = payload.get("workspacePaths")
            if isinstance(raw_wps, (list, tuple)):
                for wp in raw_wps:
                    if not isinstance(wp, str) or not wp:
                        continue
                    try:
                        wps.append(str(Path(wp).resolve()))
                    except (OSError, ValueError):
                        pass
            if wps and state.get("workspace_roots") != wps:
                state["workspace_roots"] = wps
                changed = True

            inv_num = payload.get("invocationNum")
            budget_map = state.get("direct_budget")
            if inv_num is not None and inv_num != 0 and inv_num != "0":
                prev_inv = state.get("current_invocation_num")
                if prev_inv != inv_num:
                    state["current_invocation_num"] = inv_num
                    changed = True
                if isinstance(budget_map, dict):
                    if str(inv_num) in budget_map:
                        budget_map.pop(str(inv_num), None)
                        changed = True
                    if "0" in budget_map:
                        budget_map.pop("0", None)
                        changed = True
                if (
                    inv_num == 1
                    or inv_num == "1"
                    or (prev_inv is not None and str(prev_inv) != str(inv_num))
                ):
                    from guard.verification import _get_circuit_breaker

                    cb = _get_circuit_breaker(state, payload)
                    if cb.get("failure_streak") or cb.get("tripped") or cb.get("replan_count"):
                        cb.update(
                            {
                                "failure_streak": 0,
                                "tripped": False,
                                "seen_failures": [],
                                "last_error": "",
                                "replan_count": 0,
                            }
                        )
                        changed = True
            else:
                if isinstance(budget_map, dict) and "0" in budget_map:
                    budget_map.pop("0", None)
                    changed = True
            if changed:
                _save_state(payload, state)
            return {}

        dedup_key = f"{cid}:{_current_invocation_key(payload, state)}"
        reminded: List[str] = state.setdefault("isolation_reminded_invocations", [])
        if dedup_key in reminded:
            return {}
        reminded.append(dedup_key)
        _save_state(payload, state)
        return {
            "injectSteps": [
                {
                    "ephemeralMessage": (
                        "[ORCHESTRATOR GUARD] Subagent context isolation active: ignore root orchestrator "
                        "delegation rules in SKILL.md. Execute only your assigned TASK and RULES, and return "
                        "strictly the 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                        "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                        "file:// .log URI when BLOCKED or FAILED; status may be COMPLETED, CHECKPOINT, "
                        "NEEDS_REPLAN, BLOCKED, or FAILED; scouts may use compact 2-header STATUS + NOTES)."
                    )
                }
            ]
        }


def _get_subagent_type(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
    if not _is_known_subagent(payload, state):
        return ""
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict):
        t = meta.get("type_name") or meta.get("type")
        if t is not None:
            return str(t).strip().lower()
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    if not isinstance(subagents, dict):
        subagents = {}
        state["subagents"] = subagents
    sub_info = subagents.get(cid)
    if isinstance(sub_info, dict):
        t = sub_info.get("type_name") or sub_info.get("type")
        if t is not None:
            return str(t).strip().lower()
    return ""


def _is_research_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a research subagent."""
    return _get_subagent_type(payload, state) in {"research", "researcher"}


def _is_scratch_only_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is restricted to scratch-only writes (research, researcher, council)."""
    return _get_subagent_type(payload, state) in SCRATCH_ONLY_SUBAGENT_TYPES


def _is_scout_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a scout subagent."""
    return _get_subagent_type(payload, state) == "scout"


def _is_strict_readonly_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a strictly read-only subagent (scout, verifier, reviewer, judge, or council judge)."""
    if _get_subagent_type(payload, state) in STRICT_READONLY_SUBAGENT_TYPES:
        return True
    if not _is_known_subagent(payload, state):
        return False
    meta = state.get("self_subagent_meta")
    if not isinstance(meta, dict):
        cid = str(payload.get("conversationId") or "").strip().lower()
        subagents = state.get("subagents")
        if not isinstance(subagents, dict):
            subagents = {}
            state["subagents"] = subagents
        sub_info = subagents.get(cid)
        if isinstance(sub_info, dict):
            meta = sub_info
    if isinstance(meta, dict) and meta.get("council_role") in {"judge", "arbiter"}:
        return True
    return False


def _mark_subagent_status_in_parent_brain(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    child_cid: str,
    status: str,
    *,
    update_local: bool = True,
) -> None:
    """Update a subagent's status in its own state (when update_local) and parent brain state if present."""
    norm_cid = (child_cid or "").strip().lower()
    if not norm_cid:
        return
    if update_local:
        local_changed = False
        subagents = state.get("subagents")
        if (
            isinstance(subagents, dict)
            and norm_cid in subagents
            and isinstance(subagents[norm_cid], dict)
            and subagents[norm_cid].get("status") != status
        ):
            subagents[norm_cid]["status"] = status
            local_changed = True
        self_meta = state.get("self_subagent_meta")
        if isinstance(self_meta, dict) and self_meta.get("status") != status:
            self_meta["status"] = status
            local_changed = True
        if local_changed:
            _save_state(payload, state)

    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir:
        return
    try:
        resolved_art = Path(artifact_dir).resolve()
        if resolved_art.parent.name != "brain":
            return
        meta = state.get("self_subagent_meta") or {}
        parent_cid = str(meta.get("parent_conversation_id") or "").strip().lower()
        for sibling_state_path in resolved_art.parent.glob("*/orchestrator_state.json"):
            if sibling_state_path.parent == resolved_art:
                continue
            if _has_other_locked_path(sibling_state_path):
                continue
            with state_transaction(sibling_state_path):
                sib_data = _load_state(sibling_state_path)
                sib_subagents = sib_data.get("subagents") or {}
                root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
                dir_name = sibling_state_path.parent.name.lower()
                matches_subagent = norm_cid in sib_subagents
                matches_parent = bool(
                    parent_cid and (root_cid == parent_cid or dir_name == parent_cid)
                )
                if (
                    (matches_subagent or matches_parent)
                    and _workspaces_compatible(payload, sib_data)
                    and norm_cid in sib_subagents
                    and isinstance(sib_subagents[norm_cid], dict)
                    and sib_subagents[norm_cid].get("status") != status
                ):
                    sib_subagents[norm_cid]["status"] = status
                    _save_state(sibling_state_path, sib_data)
    except (OSError, ValueError):
        pass


def _apply_cohort_outcome(
    target_state: Dict[str, Any],
    child_cid: str,
    cohort_id: str,
    outcome: str,
    error_summary: str = "",
    *,
    create_if_missing: bool = False,
    cb: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update a cohort's succeeded/failed tracking and parent circuit breaker in target_state."""
    cohorts = target_state.setdefault("cohorts", {})
    cohort_entry = cohorts.get(cohort_id)
    if not isinstance(cohort_entry, dict):
        if not create_if_missing:
            return False
        cohort_entry = {
            "total_spawned": 0,
            "succeeded": [],
            "failed": [],
            "created_at": time.time(),
        }
        cohorts[cohort_id] = cohort_entry
    succeeded = cohort_entry.setdefault("succeeded", [])
    failed = cohort_entry.setdefault("failed", [])
    cb_target = (
        cb
        if cb is not None
        else target_state.setdefault(
            "circuit_breaker",
            {
                "failure_streak": 0,
                "tripped": False,
                "seen_failures": [],
                "last_error": "",
                "replan_count": 0,
            },
        )
    )
    norm_cid = (child_cid or "").strip().lower()
    if outcome == "succeeded":
        key = norm_cid or f"ok:{len(succeeded)}"
        if key not in succeeded:
            succeeded.append(key)
        if key in failed:
            failed.remove(key)
        cb_target["failure_streak"] = 0
        cb_target["tripped"] = False
    else:
        key = norm_cid or f"fail:{len(failed)}"
        newly_failed = key not in succeeded and key not in failed
        if newly_failed:
            failed.append(key)
        total_spawned = int(cohort_entry.get("total_spawned", 0) or 0)
        if total_spawned >= 2:
            if newly_failed and len(failed) >= total_spawned and len(succeeded) == 0:
                cb_target["failure_streak"] = max(
                    MAX_CONSECUTIVE_FAILURES,
                    int(cb_target.get("failure_streak", 0) or 0) + 1,
                )
                cb_target["tripped"] = True
                cb_target["last_error"] = (
                    error_summary or "All Best-of-N cohort candidates failed"
                )
        elif newly_failed:
            cb_target["failure_streak"] = int(cb_target.get("failure_streak", 0) or 0) + 1
            cb_target["last_error"] = error_summary or "Best-of-N candidate failed"
            if cb_target["failure_streak"] >= MAX_CONSECUTIVE_FAILURES:
                cb_target["tripped"] = True
    return True


def _record_cohort_outcome_in_parent_brain(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    child_cid: str,
    cohort_id: str,
    outcome: str,
    error_summary: str = "",
    *,
    update_local: bool = True,
) -> None:
    """Record an async Best-of-N subagent completion outcome in parent brain cohorts."""
    norm_cid = (child_cid or "").strip().lower()
    conv_cid = str(payload.get("conversationId") or "").strip().lower()
    if not norm_cid or not cohort_id:
        return
    if update_local and _apply_cohort_outcome(
        state, norm_cid, cohort_id, outcome, error_summary
    ):
        _save_state(payload, state)

    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir:
        return
    try:
        resolved_art = Path(artifact_dir).resolve()
        if resolved_art.parent.name != "brain":
            return
        meta = state.get("self_subagent_meta") or {}
        parent_cid = str(meta.get("parent_conversation_id") or "").strip().lower()
        for sibling_state_path in resolved_art.parent.glob("*/orchestrator_state.json"):
            if sibling_state_path.parent == resolved_art:
                continue
            if _has_other_locked_path(sibling_state_path):
                continue
            with state_transaction(sibling_state_path):
                sib_data = _load_state(sibling_state_path)
                sib_subagents = sib_data.get("subagents") or {}
                root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
                dir_name = sibling_state_path.parent.name.lower()
                matches_subagent = bool(
                    norm_cid in sib_subagents or (conv_cid and conv_cid in sib_subagents)
                )
                matches_parent = bool(
                    parent_cid and (root_cid == parent_cid or dir_name == parent_cid)
                )
                if (matches_subagent or matches_parent) and _workspaces_compatible(
                    payload, sib_data
                ):
                    sib_meta = sib_subagents.get(conv_cid) if conv_cid else None
                    effective_cid = str(
                        (sib_meta.get("candidate_id") if isinstance(sib_meta, dict) else None)
                        or norm_cid
                    ).strip().lower()
                    if _apply_cohort_outcome(
                        sib_data, effective_cid, cohort_id, outcome, error_summary
                    ):
                        _save_state(sibling_state_path, sib_data)
    except (OSError, ValueError):
        pass


def _is_root_orchestrator(
    payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> bool:
    """Return True when the caller is the root orchestrator."""
    if state is None:
        state = _load_state(payload)
    return not _is_known_subagent(payload, state)


def register_subagent(
    payload: Dict[str, Any],
    conversation_id: str,
    type_name: str,
    model: str,
    workspace: str = "inherit",
    role: str = "",
    allowed_files: Optional[List[str]] = None,
    status: Optional[str] = None,
    spawned_mutation_epoch: Optional[int] = None,
    review_lens: Optional[str] = None,
    active_worktree: Optional[bool] = None,
    mode: Optional[str] = None,
    cohort: Optional[str] = None,
    eval_cmd: Optional[str] = None,
    candidate_id: Optional[str] = None,
    council_id: Optional[str] = None,
    council_role: Optional[str] = None,
) -> None:
    """Helper for explicitly registering a subagent conversationId -> tier mapping."""
    with state_transaction(payload):
        state = _load_state(payload)
        cid = conversation_id.strip().lower()
        norm_ws = workspace.strip().lower()
        record: Dict[str, Any] = {
            "role": role or type_name,
            "type": type_name,
            "model": model,
            "workspace": workspace,
            "active_branch": norm_ws == "branch",
            "active_worktree": (
                active_worktree
                if active_worktree is not None
                else (norm_ws in ISOLATED_WORKSPACES)
            ),
        }
        if allowed_files is not None:
            record["allowed_files"] = list(allowed_files)
        if status is not None:
            record["status"] = status
        if spawned_mutation_epoch is not None:
            record["spawned_mutation_epoch"] = spawned_mutation_epoch
        if review_lens is not None:
            record["review_lens"] = review_lens
        if mode is not None:
            record["mode"] = mode
        if cohort is not None:
            record["cohort"] = cohort
        if eval_cmd is not None:
            record["eval_cmd"] = eval_cmd
        if candidate_id is not None:
            record["candidate_id"] = candidate_id
        if council_id is not None:
            record["council_id"] = council_id
        if council_role is not None:
            record["council_role"] = council_role
        state.setdefault("subagents", {})[cid] = _normalize_subagent_meta(record)
        _save_state(payload, state)

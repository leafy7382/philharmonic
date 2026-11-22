"""Path allowlists, artifact/scratch/tmp checks, and guard/plugin/state protection."""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ALLOWED_FILES_HEADER_RE,
    DIR_LIST_TOOLS,
    MUTATION_TOOLS,
    PATH_ARG_KEYS,
    PROJECT_ROOT,
    READ_TOOLS,
    SCRIPT_DIR,
    SEARCH_TOOLS,
)
from guard.identity import (
    _is_known_subagent,
    _normalize_subagent_meta,
    _read_sibling_snapshot,
    _workspaces_compatible,
)
from guard.state_store import _save_state

_STATE_NAMES = frozenset({"orchestrator_state.json", "orchestrator_state.lock"})
_GLOB_CHARS = "*?[{}]"
_POSIX_GLOB_CLASSES = {
    "alpha": "A-Za-z",
    "alnum": "0-9A-Za-z",
    "digit": "0-9",
    "lower": "a-z",
    "upper": "A-Z",
    "xdigit": "0-9A-Fa-f",
    "word": "0-9A-Za-z_",
    "punct": "._",
    "space": " \t",
    "blank": " \t",
    "print": "\x20-\x7e",
    "graph": "\x22-\x7e!",
}


def _has_nul_or_surr(text: str) -> bool:
    return "\x00" in text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def _guard_state_sentinel() -> str:
    """Return the fallback orchestrator_state.json sentinel path string."""
    return str(PROJECT_ROOT / ".agents" / ".state" / "orchestrator_state.json")


def _sanitize_decoded_shell_token(tok: str) -> str:
    """Truncate at first NUL byte and replace any lone UTF-16 surrogates."""
    if "\x00" in tok:
        tok = tok.split("\x00", 1)[0]
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in tok):
        tok = "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in tok)
    return tok


def _has_shell_path_expansion(target: str) -> bool:
    """Return True if target contains shell variable, command, tilde, glob, or brace expansion."""
    if not target:
        return False
    if any(ch in target for ch in ("$", "`", "*", "?", "[")) or target.startswith("~"):
        return True
    return bool(re.search(r"\{[^}]*(?:,|\.\.)[^}]*\}", target))


def _parse_allowed_files_header(prompt_text: str) -> Optional[List[str]]:
    """Parse ALLOWED_FILES: header from a subagent prompt, returning None if omitted."""
    if not prompt_text:
        return None
    match = ALLOWED_FILES_HEADER_RE.search(prompt_text)
    if not match:
        return None
    tokens = [
        tok.strip().strip("`'\"")
        for tok in match.group(1).strip().split(",")
        if tok.strip().strip("`'\"")
    ]
    if len(tokens) == 1 and tokens[0].upper() == "NONE":
        return []
    return tokens


def _get_subagent_allowed_files(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Optional[List[str]]:
    """Return the per-subagent ALLOWED_FILES list if configured, or None if unrestricted."""
    if not _is_known_subagent(payload, state):
        return None
    cid = str(payload.get("conversationId") or "").strip().lower()
    subagents = state.get("subagents")
    sub_info = subagents.get(cid) if isinstance(subagents, dict) else None
    if isinstance(sub_info, dict) and isinstance(sub_info.get("allowed_files"), list):
        return list(sub_info["allowed_files"])
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir and cid:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                for sibling_state_path in resolved_art.parent.glob("*/orchestrator_state.json"):
                    if sibling_state_path.parent == resolved_art:
                        continue
                    sib_data = _read_sibling_snapshot(sibling_state_path)
                    if not isinstance(sib_data, dict) or not _workspaces_compatible(payload, sib_data):
                        continue
                    sib_subagents = sib_data.get("subagents") or {}
                    sib_entry = sib_subagents.get(cid) if isinstance(sib_subagents, dict) else None
                    if isinstance(sib_entry, dict) and isinstance(sib_entry.get("allowed_files"), list):
                        updated_meta = _normalize_subagent_meta(sib_entry)
                        updated_meta["parent_conversation_id"] = (
                            sib_data.get("root_conversation_id") or sibling_state_path.parent.name
                        )
                        if state.get("self_subagent_meta") != updated_meta:
                            state["self_subagent_meta"] = updated_meta
                            _save_state(payload, state)
                        return list(sib_entry["allowed_files"])
        except (OSError, ValueError):
            pass
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict) and isinstance(meta.get("allowed_files"), list):
        return list(meta["allowed_files"])
    return None


def _is_target_in_allowed_files(
    target: str, allowed_files: List[str], workspace_paths: Optional[List[str]] = None
) -> bool:
    """Return True if target matches any file, directory, or allowlist entry in allowed_files."""
    if not allowed_files or not target or _has_nul_or_surr(target) or _has_shell_path_expansion(target):
        return False
    try:
        raw_target = Path(os.path.expanduser(target))
        target_path = (raw_target if raw_target.is_absolute() else (PROJECT_ROOT / raw_target)).resolve()
    except (OSError, ValueError):
        return False

    search_roots = [PROJECT_ROOT]
    if isinstance(workspace_paths, (list, tuple)):
        for wp in workspace_paths:
            if isinstance(wp, str) and wp:
                try:
                    search_roots.append(Path(wp).resolve())
                except (OSError, ValueError):
                    continue

    for raw_entry in allowed_files:
        entry = str(raw_entry).strip().strip("`'\"")
        if not entry or entry.upper() == "NONE":
            continue
        try:
            raw_ep = Path(os.path.expanduser(entry))
            entry_paths = (
                [raw_ep.resolve()]
                if raw_ep.is_absolute()
                else [(root / raw_ep).resolve() for root in search_roots]
            )
        except (OSError, ValueError):
            continue
        for entry_path in entry_paths:
            if target_path == entry_path:
                return True
            try:
                target_path.relative_to(entry_path)
                return True
            except ValueError:
                pass
    return False


def _normalize_allowlist_entry(entry: str) -> Optional[Path]:
    """Normalize an ALLOWED_FILES entry relative to PROJECT_ROOT, stripping glob tails."""
    cleaned = str(entry or "").strip().strip("`'\"")
    if not cleaned or cleaned.upper() == "NONE":
        return None
    cleaned = cleaned.replace("\\", "/")
    if any(ch in cleaned for ch in ("*", "?", "[")):
        non_glob: List[str] = []
        for part in Path(cleaned).parts:
            if any(ch in part for ch in ("*", "?", "[")):
                break
            non_glob.append(part)
        cleaned = str(Path(*non_glob)) if non_glob else "."
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    if not cleaned:
        return None
    try:
        raw_path = Path(os.path.expanduser(cleaned))
        return (raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)).resolve()
    except (OSError, ValueError):
        return None


def _find_allowed_files_overlap(files_a: List[str], files_b: List[str]) -> Optional[Tuple[str, str]]:
    """Return the first conflicting pair (entry_a, entry_b) if two allowlists overlap."""
    if not files_a or not files_b:
        return None
    norm_a = [(item, p) for item in files_a if (p := _normalize_allowlist_entry(item)) is not None]
    norm_b = [(item, p) for item in files_b if (p := _normalize_allowlist_entry(item)) is not None]
    for raw_a, path_a in norm_a:
        for raw_b, path_b in norm_b:
            if path_a == path_b:
                return (raw_a, raw_b)
            for p1, p2 in ((path_a, path_b), (path_b, path_a)):
                try:
                    p1.relative_to(p2)
                    return (raw_a, raw_b)
                except ValueError:
                    pass
    return None


def _is_artifact_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside the conversation artifact or brain directory (never worktrees or state files)."""
    if not payload or not isinstance(payload, dict):
        return False
    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir or not target or _has_nul_or_surr(target):
        return False
    try:
        if Path(target).name in _STATE_NAMES:
            return False
        resolved_target = Path(os.path.expanduser(target)).resolve()
        resolved_artifact = Path(artifact_dir).resolve()
    except (OSError, ValueError):
        return False
    if resolved_target.name in _STATE_NAMES:
        return False
    exempt_roots = [resolved_artifact]
    if resolved_artifact.parent.name == "brain":
        exempt_roots.append(resolved_artifact.parent)
    return any(
        resolved_target == root or root in resolved_target.parents for root in exempt_roots
    )


def _is_worktree_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside a subagent worktree directory."""
    if not target or _has_nul_or_surr(target):
        return False
    try:
        resolved_target = Path(os.path.expanduser(target)).resolve()
    except (OSError, ValueError):
        return False
    artifact_dir = (payload or {}).get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_artifact = Path(artifact_dir).resolve()
            if resolved_artifact.parent.name == "brain":
                wt_root = resolved_artifact.parent.parent / "worktrees"
                if resolved_target == wt_root or wt_root in resolved_target.parents:
                    return True
        except (OSError, ValueError):
            pass
    norm = str(resolved_target).replace("\\", "/")
    return "/worktrees/" in norm or norm.endswith("/worktrees")


def _is_artifact_or_worktree_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside the conversation artifact/brain directory or a subagent worktree."""
    return _is_artifact_path(target, payload) or _is_worktree_path(target, payload)


def _is_tmp_path(target: str) -> bool:
    """Return True only when resolved target path is inside /tmp."""
    if not target or _has_nul_or_surr(target) or _has_shell_path_expansion(target):
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
        tmp_root = Path("/tmp").resolve()
        return resolved == tmp_root or tmp_root in resolved.parents
    except (OSError, ValueError):
        return False


def _is_staging_or_tmp_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside artifactDir, /brain/, /tmp/, or a scratch/ directory."""
    if not target:
        return False
    if _is_artifact_path(target, payload) or _is_tmp_path(target):
        return True
    parts = [p.lower() for p in target.replace("\\", "/").rstrip("/").split("/") if p]
    return "brain" in parts or "scratch" in parts


def _is_in_allowed_scratch_dir(
    target: str, payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> bool:
    """Return True if target is inside own or parent brain's scratch/ directory (never state files)."""
    if not target or _has_shell_path_expansion(target) or not _is_artifact_path(target, payload):
        return False
    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir:
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
        resolved_artifact = Path(artifact_dir).resolve()
    except (OSError, ValueError):
        return False
    if "scratch" not in resolved.parts:
        return False
    own_scratch = (resolved_artifact / "scratch").resolve()
    if resolved == own_scratch or own_scratch in resolved.parents:
        return True
    if resolved_artifact.parent.name == "brain":
        raw_meta = (state or {}).get("self_subagent_meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        parent_cid = str(meta.get("parent_conversation_id") or "").strip().lower()
        cid = str(payload.get("conversationId") or "").strip().lower()
        try:
            for sibling_dir in resolved_artifact.parent.iterdir():
                if sibling_dir == resolved_artifact or not sibling_dir.is_dir():
                    continue
                sib_scratch = (sibling_dir / "scratch").resolve()
                if resolved != sib_scratch and sib_scratch not in resolved.parents:
                    continue
                sib_state_file = sibling_dir / "orchestrator_state.json"
                sib_data = _read_sibling_snapshot(sib_state_file) if sib_state_file.exists() else None
                if isinstance(sib_data, dict):
                    root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
                    sib_subs = sib_data.get("subagents")
                    if (
                        parent_cid
                        and (root_cid == parent_cid or sibling_dir.name.lower() == parent_cid)
                    ) or (cid and isinstance(sib_subs, dict) and cid in sib_subs):
                        return True
        except (OSError, ValueError):
            pass
    return False


def _is_allowed_research_scratch_write(
    target: str, payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> bool:
    """Return True if target is a .md or .txt file inside own or parent brain's scratch/ directory."""
    if not _is_in_allowed_scratch_dir(target, payload, state):
        return False
    try:
        return Path(os.path.expanduser(target)).resolve().suffix.lower() in {".md", ".txt"}
    except (OSError, ValueError):
        return False


def _extract_any_path_arg(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Return the first present path argument value from PATH_ARG_KEYS, or None."""
    for key in PATH_ARG_KEYS:
        if key in args and args.get(key) is not None:
            return str(args.get(key))
    return None


def _extract_target_path(tool_name: str, args: Dict[str, Any]) -> str:
    """Extract the target file or directory path for native Antigravity tools."""
    if tool_name in (SEARCH_TOOLS | READ_TOOLS | DIR_LIST_TOOLS | MUTATION_TOOLS):
        return _extract_any_path_arg(tool_name, args) or ""
    return ""


def _resolve_cmd_target(target: str, cwd: str) -> str:
    """Resolve a command target path relative to cwd if non-absolute."""
    if not target:
        return target
    target = _sanitize_decoded_shell_token(target)
    if not target:
        return target
    cwd = _sanitize_decoded_shell_token(cwd) if cwd else cwd
    try:
        raw = Path(os.path.expanduser(target))
        base = Path(os.path.expanduser(cwd)) if cwd else PROJECT_ROOT
        return str((raw if raw.is_absolute() else (base / raw)).resolve())
    except (OSError, ValueError):
        return os.path.normpath(str(target))


def _is_canonical_guard_script(tok: str, cwd: str) -> bool:
    """Return True only if tok resolves to the canonical orchestrator_guard.py script."""
    if not tok or tok == "-" or _has_nul_or_surr(tok):
        return False
    resolved = _resolve_cmd_target(tok, cwd)
    candidates = (
        PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra" / "scripts" / "orchestrator_guard.py",
        SCRIPT_DIR / "orchestrator_guard.py",
    )
    for cand in candidates:
        try:
            canon = str(cand.resolve())
        except (OSError, ValueError):
            canon = os.path.normpath(str(cand))
        if resolved == canon:
            return True
    return False


def _is_readonly_canonical_guard_invocation(
    tok: str,
    following_args: List[str],
    cwd: str,
    is_piped_in: bool = False,
    has_input_redir: bool = False,
    cwd_unknown: bool = False,
) -> bool:
    """Return True iff tok is the canonical guard script invoked with exactly one read-only subcommand and no stdin input."""
    if _has_nul_or_surr(tok):
        return False
    try:
        is_abs = Path(os.path.expanduser(tok)).is_absolute()
    except (OSError, ValueError):
        return False
    if cwd_unknown and not is_abs:
        return False
    return (
        _is_canonical_guard_script(tok, cwd)
        and not is_piped_in
        and not has_input_redir
        and len(following_args) == 1
        and following_args[0] in {"doctor", "status", "-h", "--help"}
    )


def _is_guard_plugin_cwd(cwd_str: str) -> bool:
    """Return True if cwd_str resolves inside the philharmonic_orchestra plugin directory."""
    try:
        resolved = Path(_resolve_cmd_target(".", cwd_str)).resolve()
    except (OSError, ValueError):
        return False
    for root in (
        (PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra").resolve(),
        SCRIPT_DIR.parent.resolve(),
    ):
        if resolved == root or root in resolved.parents:
            return True
    return False


def _safe_fnmatchcase(name: str, pat: str) -> bool:
    """Match literal name against bash/glob pattern safely, handling [^...] and POSIX [:...:], failing closed on unknown POSIX classes."""
    translated: List[str] = []
    i, n = 0, len(pat)
    while i < n:
        ch = pat[i]
        if ch == "\\" and i + 1 < n and pat[i + 1] == "[":
            translated.append(pat[i : i + 2])
            i += 2
            continue
        if ch == "[":
            j = i + 1
            neg = j < n and pat[j] in ("!", "^")
            if neg:
                j += 1
            class_body: List[str] = []
            has_posix_dash = False
            if j < n and pat[j] == "]":
                class_body.append("]")
                j += 1
            closed = False
            while j < n:
                if pat[j] == "[" and j + 1 < n and pat[j + 1] == ":":
                    end_posix = pat.find(":]", j + 2)
                    if end_posix != -1:
                        cls_name = pat[j + 2 : end_posix]
                        if cls_name not in _POSIX_GLOB_CLASSES:
                            return True
                        if cls_name == "punct":
                            has_posix_dash = True
                        class_body.append(_POSIX_GLOB_CLASSES[cls_name])
                        j = end_posix + 2
                        continue
                if pat[j] == "]":
                    closed = True
                    j += 1
                    break
                class_body.append(pat[j])
                j += 1
            if closed:
                if has_posix_dash:
                    class_body.append("-")
                translated.append("[" + ("!" if neg else "") + "".join(class_body) + "]")
                i = j
                continue
        translated.append(ch)
        i += 1
    try:
        return fnmatch.fnmatchcase(name, "".join(translated))
    except re.error:
        return name == pat


def _expand_simple_braces(seg: str, limit: int = 16) -> Optional[List[str]]:
    """Expand non-nested {a,b} brace alternatives in a path segment up to limit, returning None on ranges/nesting/overflow."""
    if "{" not in seg and "}" not in seg:
        return [seg]
    results = [""]
    i, n = 0, len(seg)
    while i < n:
        lbrace = seg.find("{", i)
        rbrace = seg.find("}", i)
        if rbrace != -1 and (lbrace == -1 or rbrace < lbrace):
            return None
        if lbrace == -1:
            results = [p + seg[i:] for p in results]
            break
        if lbrace > i:
            results = [p + seg[i:lbrace] for p in results]
        close = seg.find("}", lbrace + 1)
        if close == -1:
            return None
        body = seg[lbrace + 1 : close]
        if "{" in body or (".." in body and "," not in body):
            return None
        if "," not in body:
            results = [p + "{" + body + "}" for p in results]
        else:
            alts = body.split(",")
            if len(results) * len(alts) > limit:
                return None
            results = [p + alt for p in results for alt in alts]
        i = close + 1
    return results


def _segment_matches_literal(literal_seg: str, user_seg: str) -> bool:
    """Return True if user_seg (with optional globs or simple braces) can match literal_seg."""
    if "{" in user_seg or "}" in user_seg:
        expanded = _expand_simple_braces(user_seg)
        return True if expanded is None else any(_safe_fnmatchcase(literal_seg, alt) for alt in expanded)
    return _safe_fnmatchcase(literal_seg, user_seg)


def _path_matches_protected_segments(
    user_parts: Tuple[str, ...], protected_parts: Tuple[str, ...], allow_descendants: bool
) -> bool:
    """Return True if user_parts (possibly containing globs, braces, or **) can match protected_parts."""
    if not user_parts or not protected_parts:
        return False
    for idx, u_seg in enumerate(user_parts):
        if u_seg == "**":
            return idx <= len(protected_parts)
        if idx >= len(protected_parts):
            return allow_descendants
        if not _segment_matches_literal(protected_parts[idx], u_seg):
            return False
    return len(user_parts) == len(protected_parts)


def _expand_target_braces(norm_target: str) -> Optional[List[str]]:
    """Expand {a,b} braces in norm_target if present, returning None on unexpanded/complex braces."""
    if "{" not in norm_target and "}" not in norm_target:
        return [norm_target]
    return _expand_simple_braces(norm_target)


def _resolve_with_fallback(path_str: str, base: Path) -> Tuple[Path, Path]:
    """Return (resolved_or_norm, norm_path) for path_str joined with base, raising ValueError on bad paths."""
    expanded = Path(os.path.expanduser(path_str))
    cand = expanded if expanded.is_absolute() else (base / expanded)
    norm_cand = Path(os.path.normpath(str(cand)))
    try:
        return cand.resolve(), norm_cand
    except OSError:
        return norm_cand, norm_cand


def _guard_roots(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute resolved protected guard/plugin/state/artifact Path anchors."""
    proj_root = PROJECT_ROOT.resolve()
    agents_dir = (proj_root / ".agents").resolve()
    plugins_dir = (agents_dir / "plugins").resolve()
    plugin_dir = (plugins_dir / "philharmonic_orchestra").resolve()
    hooks_json = (agents_dir / "hooks.json").resolve()
    fallback_state = (agents_dir / ".state").resolve()
    script_dir = SCRIPT_DIR.resolve()
    script_plugin_dir = script_dir.parent.resolve()

    resolved_art: Optional[Path] = None
    brain_dir: Optional[Path] = None
    art_raw = (payload or {}).get("artifactDirectoryPath")
    if art_raw:
        try:
            resolved_art = Path(os.path.expanduser(str(art_raw))).resolve()
            if resolved_art.parent.name == "brain":
                brain_dir = resolved_art.parent
        except (OSError, ValueError):
            resolved_art = None
    return {
        "proj_root": proj_root,
        "agents_dir": agents_dir,
        "plugins_dir": plugins_dir,
        "plugin_dir": plugin_dir,
        "hooks_json": hooks_json,
        "fallback_state": fallback_state,
        "script_dir": script_dir,
        "script_plugin_dir": script_plugin_dir,
        "resolved_art": resolved_art,
        "brain_dir": brain_dir,
    }


def _is_guard_plugin_target(target: str, cwd: str = "") -> bool:
    """Return True if target overlaps the philharmonic_orchestra plugin dir, SCRIPT_DIR, .agents/plugins, or .agents/hooks.json."""
    if not target:
        return False
    if _has_nul_or_surr(target):
        return True
    try:
        r = _guard_roots()
        base = Path(os.path.expanduser(cwd)).resolve() if cwd else r["proj_root"]
        resolved, _ = _resolve_with_fallback(target, base)
    except (OSError, ValueError):
        return True

    desc_roots = (r["plugin_dir"], r["script_dir"], r["script_plugin_dir"], r["hooks_json"])
    exact_roots = (r["plugins_dir"], r["agents_dir"])
    if any(resolved == rt or rt in resolved.parents for rt in desc_roots) or resolved in exact_roots:
        return True

    norm_target = str(target).replace("\\", "/")
    if any(ch in norm_target for ch in _GLOB_CHARS):
        expanded = _expand_target_braces(norm_target)
        if expanded is None:
            return True
        prot_targets = (
            (r["plugin_dir"].parts, True),
            (r["script_dir"].parts, True),
            (r["script_plugin_dir"].parts, True),
            (r["hooks_json"].parts, False),
            (r["plugins_dir"].parts, False),
            (r["agents_dir"].parts, False),
        )
        for alt in expanded:
            try:
                alt_res, alt_norm = _resolve_with_fallback(alt, base)
            except (OSError, ValueError):
                return True
            if any(alt_res == rt or rt in alt_res.parents for rt in desc_roots) or alt_res in exact_roots:
                return True
            for cand_path in (alt_res, alt_norm):
                if any(_path_matches_protected_segments(cand_path.parts, pp, desc) for pp, desc in prot_targets):
                    return True
    return False


def _matches_state_glob_parent(parent: Path, name_pat: str, r: Dict[str, Any]) -> bool:
    """Return True if parent + glob filename pattern can match any protected state/agents/brain entry."""
    if not any(ch in name_pat for ch in _GLOB_CHARS):
        return False
    try:
        res_parent = parent.resolve()
    except ValueError:
        return True
    except OSError:
        res_parent = parent
    if res_parent == r["fallback_state"] or r["fallback_state"] in res_parent.parents:
        return True
    if res_parent == r["plugins_dir"] and _segment_matches_literal("philharmonic_orchestra", name_pat):
        return True
    if res_parent == r["agents_dir"] and any(
        _segment_matches_literal(seg, name_pat) for seg in (".state", "plugins", "hooks.json")
    ):
        return True
    if res_parent == r["proj_root"] and _segment_matches_literal(".agents", name_pat):
        return True
    if any(_segment_matches_literal(st_name, name_pat) for st_name in _STATE_NAMES):
        if "orchestrator_state" in name_pat:
            return True
        res_art, brain_dir = r["resolved_art"], r["brain_dir"]
        if res_art is not None:
            roots = [res_art, brain_dir] if brain_dir is not None else [res_art]
            if any(res_parent == rt or rt in res_parent.parents for rt in roots):
                return True
        elif res_parent.name == "brain" or res_parent.parent.name == "brain":
            return True
    return False


def _is_guard_state_target(target: str, payload: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if target points to or globs orchestrator_state.json / orchestrator_state.lock or targets artifactDirectoryPath."""
    if not target:
        return False
    if _has_nul_or_surr(target):
        return True
    try:
        expanded = Path(os.path.expanduser(target))
        raw_name = expanded.name.lower()
    except ValueError:
        return True
    except OSError:
        return False
    if raw_name in _STATE_NAMES:
        return True

    try:
        resolved_target = expanded.resolve()
    except ValueError:
        return True
    except OSError:
        resolved_target = Path(os.path.normpath(str(expanded)))
    resolved_name = resolved_target.name.lower()
    if resolved_name in _STATE_NAMES:
        return True

    r = _guard_roots(payload)
    fb, agents_dir, plugins_dir, plugin_dir, hooks_json = (
        r["fallback_state"],
        r["agents_dir"],
        r["plugins_dir"],
        r["plugin_dir"],
        r["hooks_json"],
    )
    res_art, brain_dir = r["resolved_art"], r["brain_dir"]
    if (
        resolved_target == fb
        or fb in resolved_target.parents
        or resolved_target in {agents_dir, plugins_dir, plugin_dir}
        or (res_art is not None and resolved_target == res_art)
        or (brain_dir is not None and resolved_target == brain_dir)
    ):
        return True

    norm_target = str(target).replace("\\", "/")
    if any(ch in norm_target for ch in "*?[") and ".agents/.state" in norm_target:
        return True

    if any(ch in norm_target for ch in _GLOB_CHARS):
        expanded_alts = _expand_target_braces(norm_target)
        if expanded_alts is None:
            return True
        state_prot: List[Tuple[Tuple[str, ...], bool]] = [
            (fb.parts, True),
            (agents_dir.parts, False),
            (plugins_dir.parts, False),
            (plugin_dir.parts, True),
            (hooks_json.parts, False),
        ]
        if res_art is not None:
            state_prot.extend(
                [
                    (res_art.parts, False),
                    ((res_art / "orchestrator_state.json").parts, False),
                    ((res_art / "orchestrator_state.lock").parts, False),
                ]
            )
        if brain_dir is not None:
            state_prot.append((brain_dir.parts, False))

        for alt in expanded_alts:
            try:
                alt_exp = Path(os.path.expanduser(alt))
                alt_name = alt_exp.name.lower()
            except ValueError:
                return True
            except OSError:
                continue
            if alt_name in _STATE_NAMES:
                return True
            try:
                alt_res = alt_exp.resolve()
            except ValueError:
                return True
            except OSError:
                alt_res = Path(os.path.normpath(str(alt_exp)))
            alt_res_name = alt_res.name.lower()
            if alt_res_name in _STATE_NAMES:
                return True
            if (
                alt_res == fb
                or fb in alt_res.parents
                or alt_res in {agents_dir, plugins_dir, plugin_dir, hooks_json}
                or (res_art is not None and alt_res == res_art)
                or (brain_dir is not None and alt_res == brain_dir)
            ):
                return True
            cand_paths = [alt_res, Path(os.path.normpath(str(alt_exp)))]
            if not alt_exp.is_absolute():
                cand_paths.append(Path(os.path.normpath(str(r["proj_root"] / alt_exp))))
            for cand in cand_paths:
                if any(_path_matches_protected_segments(cand.parts, pp, desc) for pp, desc in state_prot):
                    return True
                if (
                    brain_dir is not None
                    and len(cand.parts) == len(brain_dir.parts) + 2
                    and _path_matches_protected_segments(cand.parts[:-2], brain_dir.parts, False)
                    and any(_segment_matches_literal(st, cand.parts[-1].lower()) for st in _STATE_NAMES)
                ):
                    return True
            if any(_matches_state_glob_parent(alt_res.parent, cn, r) for cn in (alt_name, alt_res_name)):
                return True

    return any(_matches_state_glob_parent(resolved_target.parent, cn, r) for cn in (raw_name, resolved_name))


def _target_is_ancestor_of_protected_guard(
    target: str, cwd: str, payload: Optional[Dict[str, Any]] = None
) -> bool:
    """Return True if target (including brace/glob expansions) is an ancestor of or overlaps protected guard/state/artifact paths."""
    if not target or not isinstance(target, str):
        return False
    if _has_nul_or_surr(target):
        return True
    try:
        r = _guard_roots(payload)
    except (OSError, ValueError):
        return True

    protected_roots: List[Path] = [
        r["agents_dir"],
        r["fallback_state"],
        r["plugin_dir"],
        r["hooks_json"],
        r["script_dir"],
        r["script_plugin_dir"],
    ]
    if r["resolved_art"] is not None:
        protected_roots.append(r["resolved_art"])
    if r["brain_dir"] is not None:
        protected_roots.append(r["brain_dir"])

    def _is_ancestor_or_eq(p: Path) -> bool:
        return any(p == rt or p in rt.parents for rt in protected_roots)

    if _has_shell_path_expansion(target) or "{" in target or "}" in target:
        if "$" in target or "`" in target or target.startswith("~"):
            return True
        expanded = _expand_target_braces(target)
        if expanded is None:
            return True
        for alt in expanded:
            resolved_alt = _resolve_cmd_target(alt, cwd)
            if (
                _is_guard_plugin_target(alt, cwd)
                or _is_guard_plugin_target(resolved_alt, cwd)
                or _is_guard_state_target(alt, payload=payload)
                or _is_guard_state_target(resolved_alt, payload=payload)
            ):
                return True
            if any(ch in alt for ch in ("*", "?", "[")):
                parts = Path(alt.replace("\\", "/")).parts
                non_glob: List[str] = []
                first_glob: Optional[str] = None
                for part in parts:
                    if any(ch in part for ch in ("*", "?", "[")):
                        first_glob = part
                        break
                    non_glob.append(part)
                prefix_str = str(Path(*non_glob)) if non_glob else "."
                try:
                    prefix_path = Path(_resolve_cmd_target(prefix_str, cwd)).resolve()
                except (OSError, ValueError):
                    return True
                for rt in protected_roots:
                    if prefix_path == rt:
                        return True
                    if prefix_path in rt.parents:
                        if first_glob in (None, "**"):
                            return True
                        next_seg = rt.parts[len(prefix_path.parts)]
                        if _segment_matches_literal(next_seg, first_glob):
                            return True
            else:
                try:
                    if _is_ancestor_or_eq(Path(resolved_alt).resolve()):
                        return True
                except (OSError, ValueError):
                    return True
        return False

    try:
        resolved_path = Path(_resolve_cmd_target(target, cwd)).resolve()
    except (OSError, ValueError):
        return True
    if (
        _is_guard_plugin_target(target, cwd)
        or _is_guard_plugin_target(str(resolved_path), cwd)
        or _is_guard_state_target(target, payload=payload)
        or _is_guard_state_target(str(resolved_path), payload=payload)
    ):
        return True
    return _is_ancestor_or_eq(resolved_path)

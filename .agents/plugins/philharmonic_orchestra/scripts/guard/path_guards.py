"""Path allowlists, artifact/scratch/tmp checks, and guard/plugin/state protection."""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from guard.constants import (
    ALLOWED_FILES_HEADER_RE, DIR_LIST_TOOLS, MUTATION_TOOLS, PATH_ARG_KEYS,
    PROJECT_ROOT, READ_TOOLS, SCRIPT_DIR, SEARCH_TOOLS,
)
from guard.identity import _is_known_subagent, _normalize_subagent_meta, _read_sibling_snapshot, _workspaces_compatible
from guard.state_store import _save_state

_STATE_NAMES = frozenset({"orchestrator_state.json", "orchestrator_state.lock"})
_GLOB_CHARS = "*?[{}]"


def _has_nul_or_surr(text: str) -> bool:
    return "\x00" in text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def _guard_state_sentinel() -> str:
    """Return the fallback orchestrator_state.json sentinel path string."""
    return str(PROJECT_ROOT / ".agents" / ".state" / "orchestrator_state.json")


def _sanitize_decoded_shell_token(tok: str) -> str:
    """Truncate at first NUL byte and replace any lone UTF-16 surrogates."""
    tok = tok.split("\x00", 1)[0] if "\x00" in tok else tok
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in tok):
        tok = "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in tok)
    return tok


def _has_shell_path_expansion(target: str) -> bool:
    """Return True if target contains shell variable, command, tilde, glob, or brace expansion."""
    return bool(target) and (
        any(ch in target for ch in ("$", "`", "*", "?", "["))
        or target.startswith("~")
        or bool(re.search(r"\{[^}]*(?:,|\.\.)[^}]*\}", target))
    )


def _parse_allowed_files_header(prompt_text: str) -> Optional[List[str]]:
    """Parse ALLOWED_FILES: header from a subagent prompt, returning None if omitted."""
    if not (match := ALLOWED_FILES_HEADER_RE.search(prompt_text) if prompt_text else None):
        return None
    tokens = [tok.strip().strip("`'\"") for tok in match.group(1).strip().split(",") if tok.strip().strip("`'\"")]
    return [] if (len(tokens) == 1 and tokens[0].upper() == "NONE") else tokens


def _sibling_brain_entries(payload: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path], List[Tuple[Path, Dict[str, Any]]]]:
    """Return (resolved_art, brain_dir, [(sib_dir, sib_data), ...]) for sibling brain dirs."""
    r = _guard_roots(payload)
    resolved_art, brain_dir = r["resolved_art"], r["brain_dir"]
    if resolved_art is None or brain_dir is None:
        return resolved_art, brain_dir, []
    entries: List[Tuple[Path, Dict[str, Any]]] = []
    try:
        for sib_dir in brain_dir.iterdir():
            if sib_dir != resolved_art and sib_dir.is_dir() and (sib_state := sib_dir / "orchestrator_state.json").exists():
                if isinstance(sib_data := _read_sibling_snapshot(sib_state), dict):
                    entries.append((sib_dir, sib_data))
    except (OSError, ValueError):
        pass
    return resolved_art, brain_dir, entries


def _get_subagent_allowed_files(payload: Dict[str, Any], state: Dict[str, Any]) -> Optional[List[str]]:
    """Return the per-subagent ALLOWED_FILES list if configured, or None if unrestricted."""
    if not _is_known_subagent(payload, state):
        return None
    cid = str(payload.get("conversationId") or "").strip().lower()
    sub_info = state["subagents"].get(cid) if isinstance(state.get("subagents"), dict) else None
    if isinstance(sub_info, dict) and isinstance(sub_info.get("allowed_files"), list):
        return list(sub_info["allowed_files"])
    if cid:
        for sib_dir, sib_data in _sibling_brain_entries(payload)[2]:
            sib_subs = sib_data.get("subagents") or {}
            sib_entry = sib_subs.get(cid) if _workspaces_compatible(payload, sib_data) and isinstance(sib_subs, dict) else None
            if isinstance(sib_entry, dict) and isinstance(sib_entry.get("allowed_files"), list):
                updated_meta = _normalize_subagent_meta(sib_entry)
                updated_meta["parent_conversation_id"] = sib_data.get("root_conversation_id") or sib_dir.name
                if state.get("self_subagent_meta") != updated_meta:
                    state["self_subagent_meta"] = updated_meta
                    _save_state(payload, state)
                return list(sib_entry["allowed_files"])
    meta = state.get("self_subagent_meta")
    return list(meta["allowed_files"]) if isinstance(meta, dict) and isinstance(meta.get("allowed_files"), list) else None


def _is_target_in_allowed_files(target: str, allowed_files: List[str], workspace_paths: Optional[List[str]] = None) -> bool:
    """Return True if target matches any file, directory, or simple glob in allowed_files."""
    if not allowed_files or not target or _has_nul_or_surr(target) or _has_shell_path_expansion(target):
        return False
    try:
        raw_target = Path(os.path.expanduser(target))
        target_path = (raw_target if raw_target.is_absolute() else (PROJECT_ROOT / raw_target)).resolve()
    except (OSError, ValueError):
        return False
    search_roots = [PROJECT_ROOT]
    for wp in (workspace_paths if isinstance(workspace_paths, (list, tuple)) else ()):
        if isinstance(wp, str) and wp:
            try:
                search_roots.append(Path(wp).resolve())
            except (OSError, ValueError):
                pass
    for raw_entry in allowed_files:
        if not (entry := str(raw_entry).strip().strip("`'\"")) or entry.upper() == "NONE":
            continue
        try:
            raw_ep = Path(os.path.expanduser(entry))
            entry_paths = [raw_ep.resolve()] if raw_ep.is_absolute() else [(rt / raw_ep).resolve() for rt in search_roots]
        except (OSError, ValueError):
            continue
        is_glob = any(ch in entry for ch in ("*", "?", "["))
        if any(target_path == ep or ep in target_path.parents or (is_glob and _safe_fnmatchcase(str(target_path), str(ep))) for ep in entry_paths):
            return True
    return False


def _normalize_allowlist_entry(entry: str) -> Optional[Path]:
    """Normalize an ALLOWED_FILES entry relative to PROJECT_ROOT, stripping glob tails."""
    if not (cleaned := str(entry or "").strip().strip("`'\"")) or cleaned.upper() == "NONE":
        return None
    cleaned = cleaned.replace("\\", "/")
    if any(ch in cleaned for ch in ("*", "?", "[")):
        parts = Path(cleaned).parts
        stop = next((i for i, p in enumerate(parts) if any(c in p for c in ("*", "?", "["))), len(parts))
        cleaned = str(Path(*parts[:stop])) if stop else "."
    if not (cleaned := cleaned.rstrip("/") if len(cleaned) > 1 else cleaned):
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
    return next(((ra, rb) for ra, pa in norm_a for rb, pb in norm_b if pa == pb or pb in pa.parents or pa in pb.parents), None)


def _is_artifact_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside the conversation artifact or brain directory (never worktrees or state files)."""
    if not payload or not isinstance(payload, dict) or not payload.get("artifactDirectoryPath") or not target or _has_nul_or_surr(target):
        return False
    try:
        if Path(target).name.lower() in _STATE_NAMES:
            return False
        resolved_target = Path(os.path.expanduser(target)).resolve()
    except (OSError, ValueError):
        return False
    if resolved_target.name.lower() in _STATE_NAMES:
        return False
    r = _guard_roots(payload)
    return any(resolved_target == rt or rt in resolved_target.parents for rt in (r["resolved_art"], r["brain_dir"]) if rt is not None)


def _is_worktree_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside a subagent worktree directory."""
    if not target or _has_nul_or_surr(target):
        return False
    try:
        resolved_target = Path(os.path.expanduser(target)).resolve()
    except (OSError, ValueError):
        return False
    if (brain_dir := _guard_roots(payload)["brain_dir"]) is not None:
        wt_root = brain_dir.parent / "worktrees"
        if resolved_target == wt_root or wt_root in resolved_target.parents:
            return True
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
        resolved, tmp_root = Path(os.path.expanduser(target)).resolve(), Path("/tmp").resolve()
        return resolved == tmp_root or tmp_root in resolved.parents
    except (OSError, ValueError):
        return False


def _is_staging_or_tmp_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside artifactDir, /brain/, /tmp/, or a scratch/ directory."""
    return bool(target) and (
        _is_artifact_path(target, payload) or _is_tmp_path(target)
        or any(p in {"brain", "scratch"} for p in target.lower().replace("\\", "/").rstrip("/").split("/") if p)
    )


def _is_in_allowed_scratch_dir(target: str, payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if target is inside own or parent brain's scratch/ directory (never state files)."""
    if not target or _has_shell_path_expansion(target) or not _is_artifact_path(target, payload):
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except (OSError, ValueError):
        return False
    if "scratch" not in resolved.parts:
        return False
    resolved_art, brain_dir, siblings = _sibling_brain_entries(payload)
    if resolved_art is None:
        return False
    own_scratch = (resolved_art / "scratch").resolve()
    if resolved == own_scratch or own_scratch in resolved.parents:
        return True
    if brain_dir is None:
        return False
    meta = (state or {}).get("self_subagent_meta")
    parent_cid = str((meta if isinstance(meta, dict) else {}).get("parent_conversation_id") or "").strip().lower()
    cid = str(payload.get("conversationId") or "").strip().lower()
    for sib_dir, sib_data in siblings:
        sib_scratch = (sib_dir / "scratch").resolve()
        if resolved != sib_scratch and sib_scratch not in resolved.parents:
            continue
        root_cid = str(sib_data.get("root_conversation_id") or "").strip().lower()
        sib_subs = sib_data.get("subagents")
        if (parent_cid and (root_cid == parent_cid or sib_dir.name.lower() == parent_cid)) or (cid and isinstance(sib_subs, dict) and cid in sib_subs):
            return True
    return False


def _is_allowed_research_scratch_write(target: str, payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if target is a .md or .txt file inside own or parent brain's scratch/ directory."""
    if not _is_in_allowed_scratch_dir(target, payload, state):
        return False
    try:
        return Path(os.path.expanduser(target)).resolve().suffix.lower() in {".md", ".txt"}
    except (OSError, ValueError):
        return False


def _extract_any_path_arg(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Return the first present path argument value from PATH_ARG_KEYS, or None."""
    return next((str(args.get(k)) for k in PATH_ARG_KEYS if k in args and args.get(k) is not None), None)


def _extract_target_path(tool_name: str, args: Dict[str, Any]) -> str:
    """Extract the target file or directory path for native Antigravity tools."""
    return (_extract_any_path_arg(tool_name, args) or "") if tool_name in (SEARCH_TOOLS | READ_TOOLS | DIR_LIST_TOOLS | MUTATION_TOOLS) else ""


def _resolve_cmd_target(target: str, cwd: str) -> str:
    """Resolve a command target path relative to cwd if non-absolute."""
    if not (target := _sanitize_decoded_shell_token(target) if target else target):
        return target
    cwd = _sanitize_decoded_shell_token(cwd) if cwd else cwd
    try:
        raw, base = Path(os.path.expanduser(target)), (Path(os.path.expanduser(cwd)) if cwd else PROJECT_ROOT)
        return str((raw if raw.is_absolute() else (base / raw)).resolve())
    except (OSError, ValueError):
        return os.path.normpath(str(target))


def _is_canonical_guard_script(tok: str, cwd: str) -> bool:
    """Return True only if tok resolves to the canonical orchestrator_guard.py script."""
    if not tok or tok == "-" or _has_nul_or_surr(tok):
        return False
    r = _guard_roots()
    return _resolve_cmd_target(tok, cwd) in {
        str(r["plugin_dir"] / "scripts" / "orchestrator_guard.py"),
        str(r["script_dir"] / "orchestrator_guard.py"),
    }


def _is_readonly_canonical_guard_invocation(
    tok: str, following_args: List[str], cwd: str, is_piped_in: bool = False, has_input_redir: bool = False, cwd_unknown: bool = False
) -> bool:
    """Return True iff tok is the canonical guard script invoked with exactly one read-only subcommand and no stdin input."""
    if _has_nul_or_surr(tok):
        return False
    try:
        is_abs = Path(os.path.expanduser(tok)).is_absolute()
    except (OSError, ValueError):
        return False
    return (
        not (cwd_unknown and not is_abs) and _is_canonical_guard_script(tok, cwd)
        and not is_piped_in and not has_input_redir and following_args in (["doctor"], ["status"], ["-h"], ["--help"])
    )


def _is_guard_plugin_cwd(cwd_str: str) -> bool:
    """Return True if cwd_str resolves inside the philharmonic_orchestra plugin directory."""
    try:
        resolved = Path(_resolve_cmd_target(".", cwd_str)).resolve()
    except (OSError, ValueError):
        return False
    r = _guard_roots()
    return any(resolved == rt or rt in resolved.parents for rt in (r["plugin_dir"], r["script_plugin_dir"]))


def _safe_fnmatchcase(name: str, pat: str) -> bool:
    """Match literal name against glob pattern via fnmatchcase, failing closed on POSIX classes or braces."""
    return False if ("[:" in pat or "{" in pat or "}" in pat) else fnmatch.fnmatchcase(name, pat)


def _segment_matches_literal(literal_seg: str, user_seg: str) -> bool:
    """Fail-closed segment matcher for protected targets (True on POSIX classes or braces)."""
    return True if ("[:" in user_seg or "{" in user_seg or "}" in user_seg) else fnmatch.fnmatchcase(literal_seg, user_seg)


def _path_matches_protected_segments(user_parts: Tuple[str, ...], protected_parts: Tuple[str, ...], allow_descendants: bool) -> bool:
    """Return True if user_parts (possibly containing globs or **) can match protected_parts."""
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
    proj_root, script_dir = PROJECT_ROOT.resolve(), SCRIPT_DIR.resolve()
    agents_dir = (proj_root / ".agents").resolve()
    plugins_dir = (agents_dir / "plugins").resolve()
    resolved_art = brain_dir = None
    if art_raw := (payload or {}).get("artifactDirectoryPath"):
        try:
            resolved_art = Path(os.path.expanduser(str(art_raw))).resolve()
            brain_dir = resolved_art.parent if resolved_art.parent.name == "brain" else None
        except (OSError, ValueError):
            resolved_art = None
    return {
        "proj_root": proj_root, "agents_dir": agents_dir, "plugins_dir": plugins_dir,
        "plugin_dir": (plugins_dir / "philharmonic_orchestra").resolve(),
        "hooks_json": (agents_dir / "hooks.json").resolve(), "fallback_state": (agents_dir / ".state").resolve(),
        "script_dir": script_dir, "script_plugin_dir": script_dir.parent.resolve(),
        "resolved_art": resolved_art, "brain_dir": brain_dir,
    }


def _has_fail_closed_glob_syntax(target: str) -> bool:
    return _has_nul_or_surr(target) or "[:" in target or "{" in target or "}" in target


def _is_guard_plugin_target(target: str, cwd: Optional[str] = None) -> bool:
    """Return True if target overlaps the philharmonic_orchestra plugin dir, SCRIPT_DIR, .agents/plugins, or .agents/hooks.json."""
    if not target:
        return False
    if _has_fail_closed_glob_syntax(target):
        return True
    try:
        r = _guard_roots()
        resolved, norm_cand = _resolve_with_fallback(target, Path(os.path.expanduser(cwd)).resolve() if cwd else r["proj_root"])
    except (OSError, ValueError):
        return True
    desc_roots = (r["plugin_dir"], r["script_dir"], r["script_plugin_dir"], r["hooks_json"])
    exact_roots = (r["plugins_dir"], r["agents_dir"])
    if any(resolved == rt or rt in resolved.parents for rt in desc_roots) or resolved in exact_roots:
        return True
    if any(ch in str(target).replace("\\", "/") for ch in "*?["):
        prot = (*((rt.parts, True) for rt in desc_roots[:3]), (r["hooks_json"].parts, False), *((rt.parts, False) for rt in exact_roots))
        return any(_path_matches_protected_segments(cand.parts, pp, desc) for cand in (resolved, norm_cand) for pp, desc in prot)
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
    if (
        res_parent == r["fallback_state"] or r["fallback_state"] in res_parent.parents
        or (res_parent == r["plugins_dir"] and _segment_matches_literal("philharmonic_orchestra", name_pat))
        or (res_parent == r["agents_dir"] and any(_segment_matches_literal(s, name_pat) for s in (".state", "plugins", "hooks.json")))
        or (res_parent == r["proj_root"] and _segment_matches_literal(".agents", name_pat))
    ):
        return True
    if any(_segment_matches_literal(st, name_pat) for st in _STATE_NAMES):
        if "orchestrator_state" in name_pat:
            return True
        if (res_art := r["resolved_art"]) is not None:
            return any(res_parent == rt or rt in res_parent.parents for rt in (res_art, r["brain_dir"]) if rt is not None)
        return res_parent.name == "brain" or res_parent.parent.name == "brain"
    return False


def _is_guard_state_target(target: str, payload: Optional[Dict[str, Any]] = None, cwd: Optional[str] = None) -> bool:
    """Return True if target points to or globs orchestrator_state.json/.lock or targets artifactDirectoryPath."""
    if not target:
        return False
    if _has_fail_closed_glob_syntax(target):
        return True
    try:
        expanded = Path(os.path.expanduser(target))
        if (raw_name := expanded.name.lower()) in _STATE_NAMES:
            return True
    except ValueError:
        return True
    except OSError:
        return False
    r = _guard_roots(payload)
    try:
        base = Path(os.path.expanduser(cwd)).resolve() if cwd else None
        resolved_target, norm_cand = _resolve_with_fallback(target, base) if base is not None else (expanded.resolve(), Path(os.path.normpath(str(expanded))))
    except OSError:
        resolved_target = norm_cand = Path(os.path.normpath(str(expanded)))
    except ValueError:
        return True
    if (resolved_name := resolved_target.name.lower()) in _STATE_NAMES:
        return True
    fb, agents_dir, plugins_dir, plugin_dir = r["fallback_state"], r["agents_dir"], r["plugins_dir"], r["plugin_dir"]
    res_art, brain_dir = r["resolved_art"], r["brain_dir"]
    cand_paths = [resolved_target, norm_cand]
    if base is None and not expanded.is_absolute():
        cand_paths.append(Path(os.path.normpath(str(r["proj_root"] / expanded))))
    if any(p == fb or fb in p.parents or p in {agents_dir, plugins_dir, plugin_dir} or p == res_art or p == brain_dir for p in cand_paths if p is not None):
        return True
    if any(ch in (norm_target := str(target).replace("\\", "/")) for ch in "*?["):
        if ".agents/.state" in norm_target:
            return True
        state_prot = [(fb.parts, True), (agents_dir.parts, False), (plugins_dir.parts, False), (plugin_dir.parts, True), (r["hooks_json"].parts, False)]
        if res_art is not None:
            state_prot.extend([(res_art.parts, False), *[((res_art / s).parts, False) for s in _STATE_NAMES]])
        if brain_dir is not None:
            state_prot.append((brain_dir.parts, False))
        for cand in cand_paths:
            if any(_path_matches_protected_segments(cand.parts, pp, desc) for pp, desc in state_prot) or (
                brain_dir is not None and len(cand.parts) == len(brain_dir.parts) + 2
                and _path_matches_protected_segments(cand.parts[:-2], brain_dir.parts, False)
                and any(_segment_matches_literal(st, cand.parts[-1].lower()) for st in _STATE_NAMES)
            ):
                return True
    return any(_matches_state_glob_parent(resolved_target.parent, cn, r) for cn in (raw_name, resolved_name))


def _target_is_ancestor_of_protected_guard(target: str, cwd: str, payload: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if target is an ancestor of or overlaps protected guard/state/artifact paths."""
    if not target or not isinstance(target, str):
        return False
    if _has_fail_closed_glob_syntax(target) or "$" in target or "`" in target or target.startswith("~"):
        return True
    try:
        r = _guard_roots(payload)
    except (OSError, ValueError):
        return True
    protected_roots = [rt for rt in (r["agents_dir"], r["fallback_state"], r["plugin_dir"], r["hooks_json"], r["script_dir"], r["script_plugin_dir"], r["resolved_art"], r["brain_dir"]) if rt is not None]
    resolved_target = _resolve_cmd_target(target, cwd)
    if _is_guard_plugin_target(target, cwd) or _is_guard_plugin_target(resolved_target, cwd) or _is_guard_state_target(target, payload=payload, cwd=cwd) or _is_guard_state_target(resolved_target, payload=payload, cwd=cwd):
        return True
    if any(ch in target for ch in ("*", "?", "[")):
        parts = Path(target.replace("\\", "/")).parts
        stop = next((i for i, p in enumerate(parts) if any(c in p for c in "*?[")), len(parts))
        first_glob = parts[stop] if stop < len(parts) else None
        try:
            prefix_path = Path(_resolve_cmd_target(str(Path(*parts[:stop])) if stop else ".", cwd)).resolve()
        except (OSError, ValueError):
            return True
        return any(prefix_path == rt or (prefix_path in rt.parents and (first_glob in (None, "**") or _segment_matches_literal(rt.parts[len(prefix_path.parts)], first_glob))) for rt in protected_roots)
    try:
        resolved_path = Path(resolved_target).resolve()
    except (OSError, ValueError):
        return True
    return any(resolved_path == rt or resolved_path in rt.parents for rt in protected_roots)

"""State persistence, JSON transactions, and per-process flock management."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Dict

from guard.constants import (
    MAX_STATE_INVOCATION_HISTORY,
    MAX_STATE_LIST_ENTRIES,
    PENDING_SPAWN_TTL_SECONDS,
    PROJECT_ROOT,
)

_LOCK_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK_REGISTRY_LOCK = threading.RLock()


def _state_file_path(payload: Dict[str, Any]) -> Path:
    artifact_dir = payload.get("artifactDirectoryPath")
    base = Path(artifact_dir) if artifact_dir else (PROJECT_ROOT / ".agents" / ".state")
    base.mkdir(parents=True, exist_ok=True)
    return base / "orchestrator_state.json"


def _resolve_state_path(payload_or_path: Any) -> Path:
    if isinstance(payload_or_path, Path):
        return payload_or_path
    return Path(payload_or_path) if isinstance(payload_or_path, str) else _state_file_path(payload_or_path)


def _current_invocation_key(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Return invocation key from payload (if non-zero) or persisted state from PreInvocation."""
    inv = payload.get("invocationNum")
    return str(inv) if (inv is not None and inv != 0 and inv != "0") else str(state.get("current_invocation_num", 0))


def _default_state() -> Dict[str, Any]:
    return {
        "active": False,
        "session_evaluated": False,
        "isolation_reminded_invocations": [],
        "recorded_spawn_calls": [],
        "pending_spawns": [],
        "subagents": {},
        "stop_reminded_executions": [],
        "direct_budget": {},
        "test_receipt": {"mutation_epoch": 0},
        "circuit_breaker": {"failure_streak": 0, "tripped": False, "seen_failures": [], "last_error": "", "replan_count": 0},
        "plan_execute_phase": False,
        "plan_execute_consumed": False,
        "plan_execute_user_input": None,
        "plan_execute_turn_finished": False,
        "plan_allowed_files": None,
        "active_eval_cmd": None,
        "cohorts": {},
        "councils": {},
        "swarm_workers": {},
        "current_invocation_num": 0,
    }


def _prune_dict(bucket: Any, limit: int, key_fn: Callable[[Any], Any]) -> None:
    if isinstance(bucket, dict) and len(bucket) > limit:
        for old_key in sorted(bucket.keys(), key=key_fn)[:-limit]:
            bucket.pop(old_key, None)


def _prune_state(state: Dict[str, Any]) -> None:
    """Bound per-invocation dictionaries and deduplication lists so state files cannot grow without limit."""
    _prune_dict(state.get("direct_budget"), MAX_STATE_INVOCATION_HISTORY, lambda k: int(k) if str(k).isdigit() else -1)
    for list_key in ("isolation_reminded_invocations", "recorded_spawn_calls", "stop_reminded_executions"):
        if isinstance(items := state.get(list_key), list) and len(items) > MAX_STATE_LIST_ENTRIES:
            state[list_key] = items[-MAX_STATE_LIST_ENTRIES:]

    if isinstance(pending := state.get("pending_spawns"), list):
        cutoff = time.time() - (PENDING_SPAWN_TTL_SECONDS * 5)
        filtered = []
        for p in pending:
            if isinstance(p, dict) and "spawned_at" in p:
                try:
                    if float(p["spawned_at"]) < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            filtered.append(p)
        state["pending_spawns"] = filtered[-MAX_STATE_LIST_ENTRIES:]

    cb = state.get("circuit_breaker")
    if isinstance(cb, dict) and isinstance(cb.get("seen_failures"), list) and len(cb["seen_failures"]) > MAX_STATE_LIST_ENTRIES:
        cb["seen_failures"] = cb["seen_failures"][-MAX_STATE_LIST_ENTRIES:]

    for group_key, inner_keys in (("cohorts", ("succeeded", "failed")), ("councils", ("proposers", "judges", "synthesizers", "models"))):
        if isinstance(group := state.get(group_key), dict):
            for entry in group.values():
                if isinstance(entry, dict):
                    for k in inner_keys:
                        if isinstance(items := entry.get(k), list) and len(items) > MAX_STATE_LIST_ENTRIES:
                            entry[k] = items[-MAX_STATE_LIST_ENTRIES:]
            _prune_dict(group, MAX_STATE_INVOCATION_HISTORY, lambda k, g=group: float(g[k].get("created_at", 0) or 0) if isinstance(g.get(k), dict) else 0.0)

    if isinstance(sw := state.get("swarm_workers"), dict) and len(sw) > MAX_STATE_LIST_ENTRIES:
        killed = sorted((k for k, v in sw.items() if isinstance(v, dict) and v.get("status") == "killed"), key=lambda k: float(sw[k].get("spawned_at", 0) or 0))
        for old_key in killed[: len(sw) - MAX_STATE_LIST_ENTRIES]:
            sw.pop(old_key, None)


def _safe_close_lock(fd: int, unlock: bool = False) -> None:
    if fd < 0:
        return
    for op in ((lambda: fcntl.flock(fd, fcntl.LOCK_UN),) if unlock else ()) + (lambda: os.close(fd),):
        try:
            op()
        except OSError:
            pass


@contextlib.contextmanager
def state_transaction(payload_or_path: Any):
    """Re-entrant cross-process lock manager for state file read-modify-write transactions."""
    lock_path = _resolve_state_path(payload_or_path).with_suffix(".lock")
    canon_key = str(lock_path.resolve())
    with _LOCK_REGISTRY_LOCK:
        if (entry := _LOCK_REGISTRY.get(canon_key)) is not None:
            entry["depth"] += 1
        else:
            fd = -1
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                pass
            _LOCK_REGISTRY[canon_key] = {"fd": fd, "depth": 1}
    try:
        yield
    finally:
        with _LOCK_REGISTRY_LOCK:
            if (entry := _LOCK_REGISTRY.get(canon_key)) is not None:
                entry["depth"] -= 1
                if entry["depth"] <= 0:
                    _safe_close_lock(entry.get("fd", -1), unlock=True)
                    _LOCK_REGISTRY.pop(canon_key, None)


def _has_other_locked_path(payload_or_path: Any) -> bool:
    """Return True if a state_transaction lock is currently held for a different state file."""
    canon_key = str(_resolve_state_path(payload_or_path).with_suffix(".lock").resolve())
    with _LOCK_REGISTRY_LOCK:
        return any(k != canon_key for k in _LOCK_REGISTRY)


def _normalize_loaded_state(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _default_state()
    defaults = _default_state()
    for k, v in defaults.items():
        data.setdefault(k, v)
    for k in ("subagents", "direct_budget", "cohorts", "councils", "swarm_workers"):
        if not isinstance(data.get(k), dict):
            data[k] = {}
    for k in ("pending_spawns", "recorded_spawn_calls", "isolation_reminded_invocations", "stop_reminded_executions"):
        if not isinstance(data.get(k), list):
            data[k] = []
    if not isinstance(cb := data.get("circuit_breaker"), dict):
        data["circuit_breaker"] = defaults["circuit_breaker"]
    else:
        for int_key in ("failure_streak", "replan_count"):
            if not isinstance(val := cb.get(int_key), int) or isinstance(val, bool):
                cb[int_key] = 0
        if not isinstance(cb.get("seen_failures"), list):
            cb["seen_failures"] = []
    if not isinstance(tr := data.get("test_receipt"), dict):
        data["test_receipt"] = {"dirty": False, "verified": False, "last_test_cmd": "", "mutation_epoch": 0}
    else:
        for int_key in ("mutation_epoch", "audited_epoch"):
            if (int_key in tr or int_key == "mutation_epoch") and (not isinstance(val := tr.get(int_key), int) or isinstance(val, bool)):
                tr[int_key] = 0
    return data


def _load_state(payload_or_path: Any) -> Dict[str, Any]:
    path = _resolve_state_path(payload_or_path)
    if not path.exists():
        return _default_state()
    lock_path = path.with_suffix(".lock")
    with _LOCK_REGISTRY_LOCK:
        already_locked = str(lock_path.resolve()) in _LOCK_REGISTRY
    try:
        is_non_empty = path.stat().st_size > 0
    except OSError:
        is_non_empty = False

    def _read_locked() -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return _normalize_loaded_state(json.load(fh))
        except json.JSONDecodeError:
            if is_non_empty:
                corrupt = _default_state()
                corrupt["active"] = corrupt["session_evaluated"] = True
                return corrupt
            return _default_state()
        except OSError:
            return _default_state()

    if already_locked:
        return _read_locked()
    fd = -1
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_SH)
        return _read_locked()
    except OSError:
        return _default_state()
    finally:
        _safe_close_lock(fd, unlock=True)


def _save_state(payload_or_path: Any, state: Dict[str, Any]) -> None:
    path = _resolve_state_path(payload_or_path)
    _prune_state(state)
    with state_transaction(path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".orch_state_", suffix=".tmp", dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(state, fh, indent=2, sort_keys=True)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, path)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        except OSError:
            pass

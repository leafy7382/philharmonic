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
from typing import Any, Dict, List, Optional

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
    if artifact_dir:
        base = Path(artifact_dir)
    else:
        base = PROJECT_ROOT / ".agents" / ".state"
    base.mkdir(parents=True, exist_ok=True)
    return base / "orchestrator_state.json"


def _resolve_state_path(payload_or_path: Any) -> Path:
    if isinstance(payload_or_path, Path):
        return payload_or_path
    if isinstance(payload_or_path, str):
        return Path(payload_or_path)
    return _state_file_path(payload_or_path)


def _current_invocation_key(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Return invocation key from payload (if non-zero) or persisted state from PreInvocation."""
    inv = payload.get("invocationNum")
    if inv is not None and inv != 0 and inv != "0":
        return str(inv)
    return str(state.get("current_invocation_num", 0))


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
        "test_receipt": {},
        "circuit_breaker": {
            "failure_streak": 0,
            "tripped": False,
            "seen_failures": [],
            "last_error": "",
            "replan_count": 0,
        },
        "plan_execute_phase": False,
        "plan_execute_consumed": False,
        "plan_allowed_files": None,
        "current_invocation_num": 0,
    }


def _prune_state(state: Dict[str, Any]) -> None:
    """Bound per-invocation dictionaries and deduplication lists so state files cannot grow without limit."""
    bucket = state.get("direct_budget")
    if isinstance(bucket, dict) and len(bucket) > MAX_STATE_INVOCATION_HISTORY:
        sorted_keys = sorted(
            bucket.keys(),
            key=lambda k: int(k) if str(k).isdigit() else -1,
        )
        for old_key in sorted_keys[:-MAX_STATE_INVOCATION_HISTORY]:
            bucket.pop(old_key, None)

    for list_key in (
        "isolation_reminded_invocations",
        "recorded_spawn_calls",
        "stop_reminded_executions",
    ):
        items = state.get(list_key)
        if isinstance(items, list) and len(items) > MAX_STATE_LIST_ENTRIES:
            state[list_key] = items[-MAX_STATE_LIST_ENTRIES:]

    pending = state.get("pending_spawns")
    if isinstance(pending, list):
        now = time.time()
        max_age = PENDING_SPAWN_TTL_SECONDS * 5
        filtered_pending = []
        for p in pending:
            if isinstance(p, dict) and "spawned_at" in p:
                try:
                    if (now - float(p["spawned_at"])) > max_age:
                        continue
                except (ValueError, TypeError):
                    pass
            filtered_pending.append(p)
        if len(filtered_pending) > MAX_STATE_LIST_ENTRIES:
            filtered_pending = filtered_pending[-MAX_STATE_LIST_ENTRIES:]
        state["pending_spawns"] = filtered_pending

    cb = state.get("circuit_breaker")
    if isinstance(cb, dict):
        seen_failures = cb.get("seen_failures")
        if isinstance(seen_failures, list) and len(seen_failures) > MAX_STATE_LIST_ENTRIES:
            cb["seen_failures"] = seen_failures[-MAX_STATE_LIST_ENTRIES:]


@contextlib.contextmanager
def state_transaction(payload_or_path: Any):
    """Re-entrant cross-process lock manager for state file read-modify-write transactions."""
    path = _resolve_state_path(payload_or_path)
    lock_path = path.with_suffix(".lock")
    canon_key = str(lock_path.resolve())

    with _LOCK_REGISTRY_LOCK:
        entry = _LOCK_REGISTRY.get(canon_key)
        if entry is not None:
            entry["depth"] += 1
            is_outer = False
        else:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
            except OSError:
                fd = -1
            if fd >= 0:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except OSError:
                    pass
            _LOCK_REGISTRY[canon_key] = {"fd": fd, "depth": 1}
            is_outer = True

    try:
        yield
    finally:
        with _LOCK_REGISTRY_LOCK:
            entry = _LOCK_REGISTRY.get(canon_key)
            if entry is not None:
                entry["depth"] -= 1
                if entry["depth"] <= 0:
                    fd = entry.get("fd", -1)
                    if fd >= 0:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                        except OSError:
                            pass
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                    _LOCK_REGISTRY.pop(canon_key, None)


def _load_state(payload_or_path: Any) -> Dict[str, Any]:
    path = _resolve_state_path(payload_or_path)
    if not path.exists():
        return _default_state()

    lock_path = path.with_suffix(".lock")
    canon_key = str(lock_path.resolve())

    with _LOCK_REGISTRY_LOCK:
        already_locked = canon_key in _LOCK_REGISTRY

    is_non_empty = False
    try:
        is_non_empty = path.stat().st_size > 0
    except OSError:
        pass

    def _read_and_parse() -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                defaults = _default_state()
                for k, v in defaults.items():
                    data.setdefault(k, v)
                data["isolation_reminded_invocations"] = (
                    data.get("isolation_reminded_invocations") or []
                )
                if isinstance(data.get("circuit_breaker"), dict):
                    data["circuit_breaker"].setdefault("replan_count", 0)
                return data
        return _default_state()

    if already_locked:
        try:
            return _read_and_parse()
        except json.JSONDecodeError:
            if is_non_empty:
                corrupt = _default_state()
                corrupt["active"] = True
                corrupt["session_evaluated"] = True
                return corrupt
            return _default_state()
        except OSError:
            return _default_state()

    fd = -1
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            return _read_and_parse()
        except json.JSONDecodeError:
            if is_non_empty:
                corrupt = _default_state()
                corrupt["active"] = True
                corrupt["session_evaluated"] = True
                return corrupt
            return _default_state()
        except OSError:
            return _default_state()
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    except OSError:
        return _default_state()
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _save_state(payload_or_path: Any, state: Dict[str, Any]) -> None:
    path = _resolve_state_path(payload_or_path)
    _prune_state(state)
    with state_transaction(path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".orch_state_", suffix=".tmp", dir=str(path.parent)
            )
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

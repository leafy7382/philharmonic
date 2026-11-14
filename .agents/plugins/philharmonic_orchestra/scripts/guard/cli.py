"""CLI diagnostics and entrypoint dispatcher for Philharmonic guard."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import sys
from typing import Any, Dict, List

from guard.constants import PROJECT_ROOT
from guard.identity import handle_pre_invocation
from guard.policies import handle_pre_tool
from guard.state_store import (
    _current_invocation_key,
    _default_state,
    _load_state,
    _save_state,
    _state_file_path,
    state_transaction,
)
from guard.verification import (
    _get_circuit_breaker,
    _get_test_receipt,
    handle_post_tool,
    handle_stop,
)


def _cli_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = _load_state(payload)
    inv_key = _current_invocation_key(payload, state)
    return {
        "projectRoot": str(PROJECT_ROOT),
        "stateFile": str(_state_file_path(payload)),
        "active": bool(state.get("active", False)),
        "planExecutePhase": bool(state.get("plan_execute_phase", False)),
        "rootConversationId": state.get("root_conversation_id"),
        "currentInvocation": inv_key,
        "activeSubagents": state.get("subagents", {}),
        "pendingSpawns": state.get("pending_spawns", []),
        "directBudget": state.get("direct_budget", {}).get(inv_key, {}),
        "testReceipt": _get_test_receipt(state, payload),
        "circuitBreaker": _get_circuit_breaker(state, payload),
    }


def _cli_toggle_active(payload: Dict[str, Any], active: bool) -> Dict[str, Any]:
    with state_transaction(payload):
        state = _load_state(payload)
        state_path = _state_file_path(payload)
        state["active"] = active
        state["session_evaluated"] = True
        _save_state(state_path, state)
        return {
            "status": "activated" if active else "deactivated",
            "active": active,
            "state_path": str(state_path),
        }


def _cli_activate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _cli_toggle_active(payload, True)


def _cli_deactivate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _cli_toggle_active(payload, False)


def _cli_reset(payload: Dict[str, Any]) -> Dict[str, Any]:
    with state_transaction(payload):
        fresh = _default_state()
        _save_state(payload, fresh)
        return {"status": "reset", "stateFile": str(_state_file_path(payload))}


def _cli_doctor() -> Dict[str, Any]:
    plugin_hooks = (
        PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra" / "hooks.json"
    )
    workspace_hooks = PROJECT_ROOT / ".agents" / "hooks.json"
    skill_file = (
        PROJECT_ROOT
        / ".agents"
        / "plugins"
        / "philharmonic_orchestra"
        / "skills"
        / "philharmonic"
        / "SKILL.md"
    )
    issues: List[str] = []
    if workspace_hooks.exists():
        try:
            wh_data = json.loads(workspace_hooks.read_text(encoding="utf-8"))
            if wh_data.get("philharmonic_orchestra-guard", {}).get("enabled", True) is not False:
                issues.append(
                    "Duplicate .agents/hooks.json is enabled alongside plugin hooks.json."
                )
        except (OSError, json.JSONDecodeError):
            issues.append("Unreadable .agents/hooks.json.")
    if not skill_file.exists():
        issues.append(
            "Missing .agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md."
        )
    else:
        try:
            skill_content = skill_file.read_text(encoding="utf-8")
            if "name: philharmonic" not in skill_content:
                issues.append(
                    "Skill SKILL.md does not contain 'name: philharmonic'."
                )
        except OSError:
            issues.append("Unable to read SKILL.md.")
    return {
        "healthy": len(issues) == 0 and plugin_hooks.exists() and skill_file.exists(),
        "pluginHooksPresent": plugin_hooks.exists(),
        "issues": issues,
    }


CLI_MODES = {"status", "reset", "activate", "deactivate"}


def _read_stdin_payload(is_cli_mode: bool) -> Dict[str, Any]:
    if sys.stdin is None:
        return {}
    try:
        if sys.stdin.isatty():
            return {}
    except (AttributeError, OSError, ValueError):
        pass

    if is_cli_mode:
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            fd = -1
        if isinstance(fd, int) and fd >= 0 and os.name != "nt":
            try:
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    return {}
            except (OSError, ValueError):
                pass

    try:
        parsed = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre-tool"
    if mode == "doctor":
        json.dump(_cli_doctor(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    payload = _read_stdin_payload(is_cli_mode=(mode in CLI_MODES))

    if mode == "status":
        json.dump(_cli_status(payload), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if mode == "reset":
        json.dump(_cli_reset(payload), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if mode == "activate":
        json.dump(_cli_activate(payload), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if mode == "deactivate":
        json.dump(_cli_deactivate(payload), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if mode == "pre-invocation":
        result = handle_pre_invocation(payload)
    elif mode == "post-tool":
        result = handle_post_tool(payload)
    elif mode == "stop":
        result = handle_stop(payload)
    else:
        result = handle_pre_tool(payload)

    json.dump(result, sys.stdout)

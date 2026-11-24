"""CLI diagnostics and entrypoint dispatcher for Philharmonic guard."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List


def _cli_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    from guard.constants import PROJECT_ROOT
    from guard.model_tiers import get_model_tier_map
    from guard.state_store import _current_invocation_key, _get_circuit_breaker, _get_test_receipt, _load_state, _state_file_path
    state = _load_state(payload)
    inv_key = _current_invocation_key(payload, state)
    subagents_summary: Dict[str, Any] = {}
    for cid, meta in (state.get("subagents") or {}).items():
        if isinstance(meta, dict):
            entry = dict(meta)
            entry.setdefault("status", "unknown")
            if entry.get("review_lens") and "reviewLens" not in entry:
                entry["reviewLens"] = entry["review_lens"]
            subagents_summary[cid] = entry
        else:
            subagents_summary[cid] = meta
    receipt = dict(_get_test_receipt(state, payload))
    receipt.setdefault("mutationEpoch", receipt.get("mutation_epoch", 0))
    return {
        "projectRoot": str(PROJECT_ROOT),
        "stateFile": str(_state_file_path(payload)),
        "active": bool(state.get("active", False)),
        "planExecutePhase": bool(state.get("plan_execute_phase", False)),
        "rootConversationId": state.get("root_conversation_id"),
        "currentInvocation": inv_key,
        "modelTiers": get_model_tier_map(),
        "activeSubagents": subagents_summary,
        "pendingSpawns": state.get("pending_spawns", []),
        "cohorts": state.get("cohorts", {}),
        "councils": state.get("councils", {}),
        "swarmWorkers": state.get("swarm_workers", {}),
        "activeEvalCmd": state.get("active_eval_cmd"),
        "directBudget": state.get("direct_budget", {}).get(inv_key, {}),
        "testReceipt": receipt,
        "circuitBreaker": _get_circuit_breaker(state, payload),
    }


def _cli_toggle_active(payload: Dict[str, Any], active: bool) -> Dict[str, Any]:
    from guard.state_store import _load_state, _save_state, _state_file_path, state_transaction

    with state_transaction(payload):
        state = _load_state(payload)
        state_path = _state_file_path(payload)
        state["active"] = active
        state["session_evaluated"] = True
        _save_state(state_path, state)
        return {"status": "activated" if active else "deactivated", "active": active, "state_path": str(state_path)}


def _cli_activate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _cli_toggle_active(payload, True)


def _cli_deactivate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _cli_toggle_active(payload, False)


def _cli_reset(payload: Dict[str, Any]) -> Dict[str, Any]:
    from guard.state_store import _default_state, _save_state, _state_file_path, state_transaction

    with state_transaction(payload):
        _save_state(payload, _default_state())
        return {"status": "reset", "stateFile": str(_state_file_path(payload))}


def _cli_doctor() -> Dict[str, Any]:
    from guard.constants import PROJECT_ROOT
    from guard.model_tiers import get_model_tier_map, validate_model_config

    plugin_dir = PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra"
    plugin_hooks, workspace_hooks = plugin_dir / "hooks.json", PROJECT_ROOT / ".agents" / "hooks.json"
    skill_file = plugin_dir / "skills" / "philharmonic" / "SKILL.md"
    issues: List[str] = []
    if workspace_hooks.exists():
        try:
            wh_data = json.loads(workspace_hooks.read_text(encoding="utf-8"))
            if wh_data.get("philharmonic_orchestra-guard", {}).get("enabled", True) is not False:
                issues.append("Duplicate .agents/hooks.json is enabled alongside plugin hooks.json.")
        except (OSError, json.JSONDecodeError):
            issues.append("Unreadable .agents/hooks.json.")
    if not skill_file.exists():
        issues.append("Missing .agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md.")
    else:
        try:
            if "name: philharmonic" not in skill_file.read_text(encoding="utf-8"):
                issues.append("Skill SKILL.md does not contain 'name: philharmonic'.")
        except OSError:
            issues.append("Unable to read SKILL.md.")
    issues.extend(validate_model_config())
    return {
        "healthy": not issues and plugin_hooks.exists() and skill_file.exists(),
        "pluginHooksPresent": plugin_hooks.exists(),
        "modelTiers": get_model_tier_map(),
        "issues": issues,
    }


CLI_MODES = {"status", "reset", "activate", "deactivate"}
_CLI_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "status": _cli_status,
    "reset": _cli_reset,
    "activate": _cli_activate,
    "deactivate": _cli_deactivate,
}


def _dispatch_hook(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "pre-invocation":
        from guard.identity import handle_pre_invocation
        return handle_pre_invocation(payload)
    if mode == "post-tool":
        from guard.verification import handle_post_tool
        return handle_post_tool(payload)
    if mode == "stop":
        from guard.verification import handle_stop
        return handle_stop(payload)
    from guard.policies import handle_pre_tool
    return handle_pre_tool(payload)


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
            import select

            fd = sys.stdin.fileno()
            if isinstance(fd, int) and fd >= 0 and os.name != "nt" and not select.select([fd], [], [], 0.05)[0]:
                return {}
        except (AttributeError, OSError, ValueError):
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
    if mode in _CLI_HANDLERS:
        json.dump(_CLI_HANDLERS[mode](payload), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    json.dump(_dispatch_hook(mode, payload), sys.stdout)

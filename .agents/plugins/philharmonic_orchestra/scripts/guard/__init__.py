"""Deterministic enforcement and workspace permission hook package for Philharmonic."""
from __future__ import annotations

from typing import Any

__all__ = (
    "CLI_MODES",
    "PROJECT_ROOT",
    "SCRIPT_DIR",
    "audit_diff_text",
    "handle_post_tool",
    "handle_pre_invocation",
    "handle_pre_tool",
    "handle_stop",
    "main",
    "register_subagent",
    "state_transaction",
    "validate_subagent_report",
)

_EXPORT_MAP = {
    "CLI_MODES": "cli",
    "PROJECT_ROOT": "constants",
    "SCRIPT_DIR": "constants",
    "audit_diff_text": "verification",
    "handle_post_tool": "verification",
    "handle_pre_invocation": "identity",
    "handle_pre_tool": "policies",
    "handle_stop": "verification",
    "main": "cli",
    "register_subagent": "identity",
    "state_transaction": "state_store",
    "validate_subagent_report": "verification",
}

_SUBMODULES = (
    "cli",
    "verification",
    "policies",
    "subagent_policy",
    "identity",
    "root_allowlist",
    "shell_targets",
    "ast_guard",
    "shell_tokens",
    "path_guards",
    "state_store",
    "constants",
    "model_tiers",
)


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports and legacy internal attribute lookups across guard submodules."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module 'guard' has no attribute {name!r}")
    import importlib

    if name in _EXPORT_MAP:
        mod = importlib.import_module(f"guard.{_EXPORT_MAP[name]}")
        val = getattr(mod, name)
        globals()[name] = val
        return val
    for mod_name in _SUBMODULES:
        mod = importlib.import_module(f"guard.{mod_name}")
        if hasattr(mod, name):
            val = getattr(mod, name)
            globals()[name] = val
            return val
    raise AttributeError(f"module 'guard' has no attribute {name!r}")

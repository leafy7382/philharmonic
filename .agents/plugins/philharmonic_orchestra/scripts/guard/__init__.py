"""Deterministic enforcement and workspace permission hook package for Philharmonic."""
from __future__ import annotations

import importlib
from typing import Any

from guard.cli import CLI_MODES, main
from guard.constants import PROJECT_ROOT, SCRIPT_DIR
from guard.identity import handle_pre_invocation, register_subagent
from guard.policies import handle_pre_tool
from guard.state_store import state_transaction
from guard.verification import (
    audit_diff_text,
    handle_post_tool,
    handle_stop,
    validate_subagent_report,
)

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
    """Delegate legacy internal attribute lookups across guard submodules."""
    for mod_name in _SUBMODULES:
        mod = importlib.import_module(f"guard.{mod_name}")
        if hasattr(mod, name):
            val = getattr(mod, name)
            globals()[name] = val
            return val
    raise AttributeError(f"module 'guard' has no attribute {name!r}")

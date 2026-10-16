#!/usr/bin/env python3
"""Workspace wrapper delegating to the pro-orchestrator plugin guard script."""

from pathlib import Path
import sys

PLUGIN_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / "plugins"
    / "pro-orchestrator"
    / "scripts"
)
if str(PLUGIN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SCRIPT_DIR))

from orchestrator_guard import main  # noqa: E402

if __name__ == "__main__":
    main()

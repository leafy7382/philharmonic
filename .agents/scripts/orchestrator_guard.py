#!/usr/bin/env python3
"""Workspace wrapper delegating to the pro-orchestrator plugin guard script."""

import json
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


def _run() -> None:
    try:
        from orchestrator_guard import main  # noqa: E402

        main()
    except Exception:
        mode = sys.argv[1] if len(sys.argv) > 1 else "pre-tool"
        if mode in {"pre-invocation", "post-tool"}:
            json.dump({}, sys.stdout)
        else:
            json.dump({"decision": "allow"}, sys.stdout)


if __name__ == "__main__":
    _run()

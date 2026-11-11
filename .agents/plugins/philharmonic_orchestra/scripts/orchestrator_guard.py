#!/usr/bin/env python3
"""Deterministic enforcement and workspace permission hook entrypoint for Philharmonic."""
from __future__ import annotations

import json
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import guard as _guard_pkg
from guard import __all__ as _guard_all, main

for _sym in _guard_all:
    globals()[_sym] = getattr(_guard_pkg, _sym)


def _safe_entrypoint() -> None:
    try:
        main()
    except Exception as exc:
        mode = sys.argv[1].lower() if len(sys.argv) > 1 else "pre-tool"
        if mode in ("pre-tool", "stop"):
            fallback = {"decision": "allow", "reason": f"Guard fail-safe: {exc}"}
        else:
            fallback = {}
        sys.stdout.write(json.dumps(fallback) + "\n")
        sys.exit(0)


if __name__ == "__main__":
    _safe_entrypoint()

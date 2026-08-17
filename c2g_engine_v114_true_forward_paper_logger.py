"""Compatibility entrypoint for the hardened V1.14 forward-paper logger."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(root / "scripts" / "run_forward.py"), "--profile", "v114"],
            cwd=root,
        )
    )

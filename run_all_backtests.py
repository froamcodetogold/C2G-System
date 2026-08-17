"""One-command entrypoint for the canonical C2G backtest and report suite."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(root / "scripts" / "run_backtests.py")],
            cwd=root,
        )
    )

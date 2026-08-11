"""Local lint helper: ruff safe subset (E4/E7/E9/F) over src/scripts/tests.

Requires `pip install ruff` (it is pinned in requirements.txt). Exit code 0
when clean; the CI workflow runs the same command.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cmd = [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"]
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""Standalone runner for live feed — avoids types.py shadow issue."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhairav.live_feed import main
main()

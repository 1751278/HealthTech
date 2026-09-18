"""Ensure the repo root (which holds liveOCR.py) is importable regardless of
how pytest is invoked or what pytest's rootdir/import-mode inserts by default.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
